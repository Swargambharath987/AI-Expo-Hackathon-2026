"""
API behaviour that matters on the public deployment: per-session/per-tab
conversations, rate limits, per-user profiles, and errors that don't leak
internals. The LLM is replaced by a fake agent — no API key or network needed.
"""

import time

import pytest
from fastapi.testclient import TestClient

import app as appmod
import agents.react_agent as react_agent
from agents.tools import fill_form


@pytest.fixture(autouse=True)
def fake_agent(monkeypatch):
    def fake_chat(self, message, profile=None, llm=None):
        self._profile = profile or {}
        self.last_form = None
        if message == "boom":
            raise RuntimeError("secret internal detail sk-ant-xyz")
        self._history.append(message)
        used = f"{llm['base_url']}|{llm['model']}" if llm else "server"
        yield f"history={self._history} rank={self._profile.get('rank', '')} llm={used}"

    monkeypatch.setattr(react_agent.ReActAgent, "chat", fake_chat)
    monkeypatch.setattr(appmod, "RATE_PER_IP_MINUTE", 100)
    appmod._agents.clear()
    appmod._rate_hits.clear()
    appmod._key_status.update(ok=None, checked=0.0)
    yield


def chat(client, message, tab="travel", **extra):
    return client.post("/api/chat", json={"message": message, "tab": tab, **extra})


def test_each_browser_gets_its_own_conversation():
    a, b = TestClient(appmod.app), TestClient(appmod.app)
    chat(a, "from-a")
    reply = chat(b, "from-b").json()["response"]
    assert "from-b" in reply and "from-a" not in reply


def test_tabs_have_separate_conversations():
    c = TestClient(appmod.app)
    chat(c, "travel-msg", tab="travel")
    reply = chat(c, "leave-msg", tab="leave").json()["response"]
    assert "leave-msg" in reply and "travel-msg" not in reply


def test_clear_history_only_clears_requested_tab():
    c = TestClient(appmod.app)
    chat(c, "t1", tab="travel")
    chat(c, "l1", tab="leave")
    c.delete("/api/chat/history?tab=travel")
    assert "t1" not in chat(c, "t2", tab="travel").json()["response"]
    assert "l1" in chat(c, "l2", tab="leave").json()["response"]


def test_session_cookie_is_httponly():
    r = TestClient(appmod.app).get("/api/health")
    cookie = r.headers["set-cookie"]
    assert "dl_session=" in cookie and "HttpOnly" in cookie


def test_per_ip_rate_limit_returns_429(monkeypatch):
    monkeypatch.setattr(appmod, "RATE_PER_IP_MINUTE", 3)
    c = TestClient(appmod.app)
    codes = [chat(c, f"m{i}").status_code for i in range(4)]
    assert codes == [200, 200, 200, 429]
    r = chat(c, "again")
    assert int(r.headers["retry-after"]) > 0


def test_global_daily_cap_applies_across_ips(monkeypatch):
    monkeypatch.setattr(appmod, "RATE_GLOBAL_DAY", 2)
    c = TestClient(appmod.app)
    assert chat(c, "1", headers={"x-forwarded-for": "1.1.1.1"}).status_code == 200
    assert chat(c, "2", headers={"x-forwarded-for": "2.2.2.2"}).status_code == 200
    assert chat(c, "3", headers={"x-forwarded-for": "3.3.3.3"}).status_code == 429


def test_spoofed_forwarded_for_prefix_is_ignored(monkeypatch):
    # Only the last X-Forwarded-For entry (added by Cloud Run) identifies the client
    monkeypatch.setattr(appmod, "RATE_PER_IP_MINUTE", 1)
    c = TestClient(appmod.app)
    assert chat(c, "1", headers={"x-forwarded-for": "9.9.9.1, 5.5.5.5"}).status_code == 200
    assert chat(c, "2", headers={"x-forwarded-for": "9.9.9.2, 5.5.5.5"}).status_code == 429


def test_profile_comes_from_each_request():
    a, b = TestClient(appmod.app), TestClient(appmod.app)
    assert "rank=SGT" in chat(a, "hi", profile={"rank": "SGT"}).json()["response"]
    assert "rank=CPT" in chat(b, "hi", profile={"rank": "CPT"}).json()["response"]


def test_bad_profile_fields_are_dropped_not_rejected():
    r = chat(TestClient(appmod.app), "hi", profile={"ssn_last4": "12ab", "rank": "SGT"})
    assert r.status_code == 200


def test_errors_do_not_leak_internals():
    body = chat(TestClient(appmod.app), "boom").json()
    assert body["error"] == "internal_error"
    assert "secret" not in body["response"] and "sk-ant" not in str(body)


def test_profile_endpoints_removed():
    c = TestClient(appmod.app)
    assert c.post("/api/profile", json={"rank": "SGT"}).status_code in (404, 405)


def test_fill_form_uses_the_requesting_users_profile(monkeypatch, tmp_path):
    import agents.tools as tools
    monkeypatch.setattr(tools, "OUTPUT_DIR", tmp_path)
    r = fill_form("DA_31", {"leave_type": "Ordinary"},
                  profile={"name_last_first": "DOE, JANE", "rank": "SSG"})
    assert r["success"]
    assert r["filled_fields"]["Name (Last, First, MI)"] == "DOE, JANE"
    assert r["filled_fields"]["Rank"] == "SSG"


OWN_KEY = {"provider": "openai", "api_key": "sk-test-" + "x" * 30, "model": "gpt-test"}


def test_own_key_is_routed_to_the_fixed_provider_url():
    reply = chat(TestClient(appmod.app), "hi", llm=OWN_KEY).json()["response"]
    assert "llm=https://api.openai.com/v1|gpt-test" in reply


def test_own_key_defaults_model_and_cannot_set_base_url():
    r = chat(TestClient(appmod.app), "hi", llm={"provider": "anthropic", "api_key": "k" * 30,
                                                 "base_url": "https://evil.example"}).json()
    assert "llm=https://api.anthropic.com/v1|claude-sonnet-5-5" in r["response"]


def test_unknown_provider_or_malformed_key_rejected():
    c = TestClient(appmod.app)
    assert chat(c, "hi", llm={**OWN_KEY, "provider": "evil"}).status_code == 422
    assert chat(c, "hi", llm={**OWN_KEY, "api_key": "short"}).status_code == 422


def test_server_key_used_without_own_key():
    assert "llm=server" in chat(TestClient(appmod.app), "hi").json()["response"]


def test_own_key_requests_skip_global_daily_cap(monkeypatch):
    monkeypatch.setattr(appmod, "RATE_GLOBAL_DAY", 1)
    c = TestClient(appmod.app)
    assert chat(c, "1").status_code == 200
    assert chat(c, "2").status_code == 429
    assert chat(c, "3", llm=OWN_KEY).status_code == 200


def test_health_reports_rejected_server_key(monkeypatch):
    monkeypatch.setattr(appmod, "LLM_BASE_URL", "https://api.anthropic.com/v1")
    monkeypatch.setattr(appmod, "_probe_llm_key", lambda url, key: (False, []))
    assert TestClient(appmod.app).get("/api/health").json()["llm_ready"] is False


def test_health_reports_accepted_server_key(monkeypatch):
    monkeypatch.setattr(appmod, "LLM_BASE_URL", "https://api.anthropic.com/v1")
    monkeypatch.setattr(appmod, "_probe_llm_key", lambda url, key: (True, ["m"]))
    assert TestClient(appmod.app).get("/api/health").json()["llm_ready"] is True


def test_key_check_endpoint_returns_models(monkeypatch):
    seen = {}
    def probe(url, key):
        seen["url"] = url
        return True, ["gpt-a", "gpt-b"]
    monkeypatch.setattr(appmod, "_probe_llm_key", probe)
    r = TestClient(appmod.app).post("/api/llm/check", json={"provider": "openai", "api_key": OWN_KEY["api_key"]}).json()
    assert r == {"valid": True, "models": ["gpt-a", "gpt-b"], "default_model": "gpt-4.1"}
    assert seen["url"] == "https://api.openai.com/v1"


def test_forms_only_downloadable_by_the_session_that_made_them(monkeypatch, tmp_path):
    import agents.tools as tools
    monkeypatch.setattr(tools, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(appmod, "OUTPUT_DIR", tmp_path)

    def fake_chat(self, message, profile=None, llm=None):
        self.last_form = fill_form("DA_31", {}, profile={"name_last_first": "DOE, JANE"})
        yield "\n⚙ Filling DA_31...\n  → done\nYour form is ready."
    monkeypatch.setattr(react_agent.ReActAgent, "chat", fake_chat)

    owner, other = TestClient(appmod.app), TestClient(appmod.app)
    url = chat(owner, "fill").json()["form_output"]["pdf_url"]
    assert owner.get(url).status_code == 200
    assert other.get(url).status_code == 404


def test_form_filenames_are_unguessable(monkeypatch, tmp_path):
    import agents.tools as tools
    monkeypatch.setattr(tools, "OUTPUT_DIR", tmp_path)
    a = fill_form("DA_31", {})["pdf_path"]
    b = fill_form("DA_31", {})["pdf_path"]
    assert a != b and len(a.rsplit("_", 1)[-1]) >= len("0123456789abcdef.pdf")


def test_old_generated_forms_are_pruned(monkeypatch, tmp_path):
    import os
    import agents.tools as tools
    monkeypatch.setattr(tools, "OUTPUT_DIR", tmp_path)
    old = tmp_path / "DA_31_old.pdf"
    old.write_text("x")
    os.utime(old, (0, 0))
    fill_form("DA_31", {})
    assert not old.exists()


def test_miles_hint_ignores_minutes():
    agent = react_agent.ReActAgent.__new__(react_agent.ReActAgent)
    assert "one_way_miles" not in agent._enrich_message("It's a 45 minute drive, 3 days")
    assert "one_way_miles=370" in agent._enrich_message("driving 370 miles for 5 days")


def test_history_window_starts_with_user_turn():
    agent = react_agent.ReActAgent.__new__(react_agent.ReActAgent)
    agent._profile = {}
    agent._history = [{"role": "assistant", "content": "orphan"}] + [
        {"role": r, "content": str(i)} for i, r in enumerate(["user", "assistant"] * 4)
    ][1:]
    roles = [m["role"] for m in agent._build_messages()[1:]]
    assert roles[0] == "user"


def test_openrouter_key_goes_to_openrouter():
    key = {"provider": "openrouter", "api_key": "sk-or-v1-" + "x" * 30}
    reply = chat(TestClient(appmod.app), "hi", llm=key).json()["response"]
    assert "llm=https://openrouter.ai/api/v1|anthropic/claude-sonnet-5.5" in reply


def test_counseling_form_puts_user_as_counselor_not_soldier(monkeypatch, tmp_path):
    import agents.tools as tools
    monkeypatch.setattr(tools, "OUTPUT_DIR", tmp_path)
    r = fill_form("DA_4856", {"soldier_name": "SMITH, JOHN", "soldier_rank": "PFC"},
                  profile={"name_last_first": "RIVERA, MARIA J", "rank": "SSG", "ssn_last4": "1234", "unit": "1-503 IN"})
    f = r["filled_fields"]
    assert f["Name of Soldier"] == "SMITH, JOHN" and f["Rank"] == "PFC"
    assert f["Counselor Name"] == "RIVERA, MARIA J" and f["Counselor Rank/Title"] == "SSG"
    assert "1234" not in r["summary"]          # counselor's SSN never on the soldier's form


def test_form_for_someone_else_never_carries_users_identity(monkeypatch, tmp_path):
    import agents.tools as tools
    monkeypatch.setattr(tools, "OUTPUT_DIR", tmp_path)
    r = fill_form("DD_1610", {"subject_name": "RIVERA, MARIA", "subject_rank": "SPC", "destination": "Fort Moore"},
                  profile={"name_last_first": "CARTER, JAMES R", "rank": "SSG", "ssn_last4": "4321",
                           "dod_id": "1234567890", "unit": "1-503 IN"})
    f = r["filled_fields"]
    assert f["Traveler Name"] == "RIVERA, MARIA" and f["Grade/Rank"] == "SPC"
    assert "CARTER" not in r["summary"] and "4321" not in r["summary"]
    assert f["Organization"] == "1-503 IN"          # shared unit still useful


def test_unknown_form_keys_are_reported_not_silently_dropped(monkeypatch, tmp_path):
    import agents.tools as tools
    monkeypatch.setattr(tools, "OUTPUT_DIR", tmp_path)
    r = fill_form("DD_1610", {"destination": "Fort Moore", "made_up_key": "x"})
    assert r["ignored_fields"] == ["made_up_key"]


def test_form_values_nobody_gave_are_left_blank(monkeypatch, tmp_path):
    import agents.tools as tools
    monkeypatch.setattr(tools, "OUTPUT_DIR", tmp_path)
    context = {"user_text": "Counsel PFC Daniel Ortiz, he was late on 2 Nov.", "tool_results": []}
    r = fill_form("DA_4856", {"soldier_name": "Ortiz, Daniel", "soldier_rank": "PFC",
                              "leader_position": "Squad Leader"},
                  profile={"name_last_first": "CARTER, JAMES R", "rank": "SSG"}, context=context)
    f = r["filled_fields"]
    assert f["Name of Soldier"] == "Ortiz, Daniel" and f["Counselor Name"] == "CARTER, JAMES R"
    assert "Counselor Position" in r["missing_fields"]
    assert r["unverified_fields"] == {"leader_position": "Squad Leader"}


def test_form_values_from_user_profile_or_renamed_post_are_kept(monkeypatch, tmp_path):
    import agents.tools as tools
    monkeypatch.setattr(tools, "OUTPUT_DIR", tmp_path)
    context = {"user_text": "I'm the platoon sergeant. Send SPC Rivera, B Company, from Fort Bragg.",
               "tool_results": []}
    r = fill_form("DD_1610", {"subject_name": "Rivera", "subject_rank": "SPC",
                              "unit": "B Co", "installation": "Fort Liberty"},
                  profile={"name_last_first": "CARTER, JAMES R", "rank": "SSG"}, context=context)
    assert "unverified_fields" not in r
    assert r["filled_fields"]["Permanent Duty Station"] == "Fort Liberty"


def test_dd1610_cost_boxes_come_from_the_cost_tool(monkeypatch, tmp_path):
    import agents.tools as tools
    monkeypatch.setattr(tools, "OUTPUT_DIR", tmp_path)
    estimate = {"success": True, "grand_total": 1222.0,
                "lodging": {"total": 416}, "meals": {"total": 288}, "mileage": {"total": 518.0}}
    context = {"user_text": "TDY to Fort Moore", "tool_results": [{"name": "calculate_travel_cost", "result": estimate}]}
    r = fill_form("DD_1610", {"destination": "Fort Moore"}, context=context)
    f = r["filled_fields"]
    assert f["Total Estimated Cost"] == "$1,222.00" and f["Per Diem Estimate"] == "$704.00"


def test_form_check_is_off_without_conversation_context(monkeypatch, tmp_path):
    import agents.tools as tools
    monkeypatch.setattr(tools, "OUTPUT_DIR", tmp_path)
    r = fill_form("DA_4856", {"soldier_name": "Ortiz", "leader_position": "Squad Leader"})
    assert r["filled_fields"]["Counselor Position"] == "Squad Leader"


def test_agent_gives_fill_form_the_whole_conversation():
    agent = react_agent.ReActAgent()
    agent._history = [{"role": "user", "content": "Counsel PFC Ortiz"},
                      {"role": "assistant", "content": "He is a Squad Leader"},   # not a source
                      {"role": "user", "content": "late twice"}]
    agent._remember_tool_result("calculate_travel_cost", '{"success": true, "grand_total": 5}')
    agent._remember_tool_result("search_regulations", '{"results": [{"text": "squad leader"}]}')
    ctx = agent._form_context()
    assert "Ortiz" in ctx["user_text"] and "late twice" in ctx["user_text"]
    assert "Squad Leader" not in ctx["user_text"]
    assert [r["name"] for r in ctx["tool_results"]] == ["calculate_travel_cost"]


def test_trip_details_without_a_box_go_to_remarks(monkeypatch, tmp_path):
    import agents.tools as tools
    monkeypatch.setattr(tools, "OUTPUT_DIR", tmp_path)
    r = fill_form("DA_4187", {"action_requested": "Reassignment", "destination": "Fort Moore, GA"})
    assert r["filled_fields"]["Remarks"] == "Destination: Fort Moore, GA"
    assert "ignored_fields" not in r
