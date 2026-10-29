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
    def fake_chat(self, message, profile=None):
        self._profile = profile or {}
        self.last_form = None
        if message == "boom":
            raise RuntimeError("secret internal detail sk-ant-xyz")
        self._history.append(message)
        yield f"history={self._history} rank={self._profile.get('rank', '')}"

    monkeypatch.setattr(react_agent.ReActAgent, "chat", fake_chat)
    monkeypatch.setattr(appmod, "RATE_PER_IP_MINUTE", 100)
    appmod._agents.clear()
    appmod._rate_hits.clear()
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
