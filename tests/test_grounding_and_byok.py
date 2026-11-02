"""
Own-key provider registry, the custom-endpoint SSRF guard, and the runtime
grounding guardrails (dollar figures / citations checked against this turn's
tool results). No network: DNS is monkeypatched and the LLM is a fake client.
"""

import json
import socket
from types import SimpleNamespace

import httpcore
import httpx
import pytest
from fastapi.testclient import TestClient

import app as appmod
import agents.react_agent as ra

KEY = "sk-test-" + "x" * 30


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    monkeypatch.setattr(appmod, "RATE_PER_IP_MINUTE", 1000)
    monkeypatch.delenv("ALLOW_PRIVATE_LLM_URLS", raising=False)
    appmod._agents.clear()
    appmod._rate_hits.clear()
    yield


def fake_dns(monkeypatch, *addrs):
    """Make every hostname resolve to the given addresses."""
    def getaddrinfo(host, port, *a, **kw):
        out = []
        for addr in addrs:
            fam = socket.AF_INET6 if ":" in addr else socket.AF_INET
            sa = (addr, port, 0, 0) if fam == socket.AF_INET6 else (addr, port)
            out.append((fam, socket.SOCK_STREAM, 6, "", sa))
        return out
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)


@pytest.fixture
def echo_agent(monkeypatch):
    """Replace the agent turn with one that echoes the LLM config it got."""
    def fake_chat(self, message, profile=None, llm=None):
        self.last_form = None
        used = f"{llm['provider']}|{llm['base_url']}|{llm['model']}" if llm else "server"
        yield f"llm={used}"
    monkeypatch.setattr(ra.ReActAgent, "chat", fake_chat)


def chat(client, message="hi", **extra):
    return client.post("/api/chat", json={"message": message, "tab": "travel", **extra})


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

def test_providers_endpoint_shape():
    providers = TestClient(appmod.app).get("/api/llm/providers").json()
    ids = [p["id"] for p in providers]
    assert ids[-1] == "custom"
    assert {"anthropic", "openai", "openrouter", "groq", "mistral", "gemini",
            "deepseek", "xai", "together"} <= set(ids)
    for p in providers:
        assert set(p) == {"id", "label", "key_placeholder", "default_model", "needs_base_url"}
        assert p["needs_base_url"] is (p["id"] == "custom")
    by_id = {p["id"]: p for p in providers}
    assert by_id["anthropic"]["default_model"] == "claude-sonnet-5-5"
    assert by_id["openai"]["default_model"] == "gpt-4.1"
    assert by_id["openrouter"]["default_model"] == "anthropic/claude-sonnet-5.5"


def test_new_preset_routes_to_its_fixed_url(echo_agent):
    r = chat(TestClient(appmod.app), llm={"provider": "groq", "api_key": KEY, "model": "some-model",
                                          "base_url": "https://evil.example"}).json()
    assert r["response"] == "llm=groq|https://api.groq.com/openai/v1|some-model"


def test_preset_without_default_model_needs_a_model(echo_agent):
    assert chat(TestClient(appmod.app), llm={"provider": "groq", "api_key": KEY}).status_code == 422


# ---------------------------------------------------------------------------
# Custom endpoint — SSRF guard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "http://api.example.com/v1",              # not https
    "https://user:pw@api.example.com/v1",     # userinfo
    "https:///v1",                            # no hostname
    "https://api.example.com/v1?x=1",         # query
    "ftp://api.example.com",
    "",
])
def test_custom_url_syntax_rejected(url):
    with pytest.raises(ra.UnsafeURLError):
        ra.check_custom_base_url(url, resolve=False)


@pytest.mark.parametrize("addr", [
    "169.254.169.254",          # cloud metadata
    "127.0.0.1", "10.0.0.5", "192.168.1.1", "172.16.0.1", "100.64.0.1", "0.0.0.0",
    "::1", "fe80::1", "fd00::1", "::ffff:169.254.169.254",
])
def test_custom_url_non_public_address_rejected(monkeypatch, addr):
    fake_dns(monkeypatch, addr)
    with pytest.raises(ra.UnsafeURLError):
        ra.check_custom_base_url("https://llm.example.com/v1")


def test_custom_url_rejected_if_any_address_is_private(monkeypatch):
    fake_dns(monkeypatch, "93.184.215.14", "10.0.0.1")
    with pytest.raises(ra.UnsafeURLError):
        ra.check_custom_base_url("https://llm.example.com/v1")


def test_custom_url_public_accepted(monkeypatch):
    fake_dns(monkeypatch, "93.184.215.14", "2606:2800:21f:cb07:6820:80da:af6b:8b2c")
    assert ra.check_custom_base_url("https://llm.example.com/v1/") == "https://llm.example.com/v1"


def test_allow_private_env_relaxes_for_local_dev(monkeypatch):
    fake_dns(monkeypatch, "127.0.0.1")
    with pytest.raises(ra.UnsafeURLError):
        ra.check_custom_base_url("http://localhost:11434/v1")
    monkeypatch.setenv("ALLOW_PRIVATE_LLM_URLS", "1")
    assert ra.check_custom_base_url("http://localhost:11434/v1") == "http://localhost:11434/v1"


def test_chat_with_custom_public_endpoint(monkeypatch, echo_agent):
    fake_dns(monkeypatch, "93.184.215.14")
    r = chat(TestClient(appmod.app), llm={"provider": "custom", "api_key": KEY, "model": "m1",
                                          "base_url": "https://llm.example.com/v1"})
    assert r.status_code == 200
    assert r.json()["response"] == "llm=custom|https://llm.example.com/v1|m1"


def test_chat_with_custom_metadata_endpoint_rejected(monkeypatch, echo_agent):
    fake_dns(monkeypatch, "169.254.169.254")
    c = TestClient(appmod.app)
    r = chat(c, llm={"provider": "custom", "api_key": KEY, "model": "m1",
                     "base_url": "https://metadata.google.internal/v1"})
    assert r.status_code == 422 and "not allowed" in r.json()["detail"]
    # missing base_url / missing model for custom
    assert chat(c, llm={"provider": "custom", "api_key": KEY, "model": "m1"}).status_code == 422
    fake_dns(monkeypatch, "93.184.215.14")
    assert chat(c, llm={"provider": "custom", "api_key": KEY,
                        "base_url": "https://llm.example.com/v1"}).status_code == 422


def test_key_check_custom_rejected_before_probe(monkeypatch):
    fake_dns(monkeypatch, "10.1.2.3")
    called = []
    monkeypatch.setattr(appmod, "_probe_llm_key", lambda *a, **kw: called.append(a) or (True, []))
    r = TestClient(appmod.app).post("/api/llm/check", json={
        "provider": "custom", "api_key": KEY, "base_url": "https://internal.example.com/v1"})
    assert r.status_code == 422 and not called


def test_key_check_custom_uses_guarded_probe(monkeypatch):
    fake_dns(monkeypatch, "93.184.215.14")
    seen = {}
    def probe(url, key, custom=False):
        seen.update(url=url, custom=custom)
        return True, ["m-a"]
    monkeypatch.setattr(appmod, "_probe_llm_key", probe)
    r = TestClient(appmod.app).post("/api/llm/check", json={
        "provider": "custom", "api_key": KEY, "base_url": "https://llm.example.com/v1/"}).json()
    assert r == {"valid": True, "models": ["m-a"], "default_model": ""}
    assert seen == {"url": "https://llm.example.com/v1", "custom": True}


def test_custom_probe_does_not_follow_redirects(monkeypatch):
    hits = []
    def handler(request):
        hits.append(str(request.url))
        return httpx.Response(302, headers={"Location": "http://169.254.169.254/computeMetadata/v1/"})
    monkeypatch.setattr(ra, "public_only_transport", lambda: httpx.MockTransport(handler))
    assert appmod._probe_llm_key("https://llm.example.com/v1", KEY, True) == (None, [])
    assert hits == ["https://llm.example.com/v1/models"]


def test_custom_probe_lists_models_incl_bare_list(monkeypatch):
    def handler(request):
        assert request.headers["authorization"] == f"Bearer {KEY}"
        return httpx.Response(200, json=[{"id": "chat-model"}, {"id": "text-embed-x"}])
    monkeypatch.setattr(ra, "public_only_transport", lambda: httpx.MockTransport(handler))
    assert appmod._probe_llm_key("https://llm.example.com/v1", KEY, True) == (True, ["chat-model"])


def test_transport_rechecks_address_at_connect_time(monkeypatch):
    # DNS rebinding: the hostname now resolves to the metadata server
    fake_dns(monkeypatch, "169.254.169.254")
    with ra.llm_http_client(custom=True, timeout=2) as c:
        with pytest.raises(httpx.ConnectError):
            c.get("https://llm.example.com/v1/models")


def test_transport_connects_to_the_validated_ip(monkeypatch):
    fake_dns(monkeypatch, "93.184.215.14")
    seen = []
    def connect_tcp(self, host, port, **kw):
        seen.append((host, port))
        raise httpcore.ConnectError("stop here")
    monkeypatch.setattr(httpcore.SyncBackend, "connect_tcp", connect_tcp)
    with ra.llm_http_client(custom=True, timeout=2) as c:
        with pytest.raises(httpx.ConnectError):
            c.get("https://llm.example.com/v1/models")
    assert seen == [("93.184.215.14", 443)]


def test_agent_llm_clients_never_follow_redirects():
    agent = ra.ReActAgent()
    agent._use_llm({"provider": "custom", "base_url": "https://llm.example.com/v1",
                    "api_key": KEY, "model": "m"})
    http = agent._client._client
    assert http.follow_redirects is False and http.trust_env is False
    assert isinstance(http._transport._pool._network_backend, ra._PublicOnlyBackend)
    assert agent._gen_params == {"temperature": 0.1, "max_tokens": 512}
    agent._use_llm({"provider": "openai", "base_url": "https://api.openai.com/v1",
                    "api_key": KEY, "model": "m"})
    assert agent._client._client.follow_redirects is False
    assert agent._gen_params == {"max_completion_tokens": 2048}
    # Claude 4.7+ rejects temperature < 1; Claude 5 thinking needs room
    agent._use_llm({"provider": "anthropic", "base_url": "https://api.anthropic.com/v1",
                    "api_key": KEY, "model": "claude-sonnet-5-5"})
    assert agent._gen_params == {"max_tokens": 4096}


# ---------------------------------------------------------------------------
# Grounding — extraction and verification
# ---------------------------------------------------------------------------

TDY_RESULT = json.dumps({
    "success": True, "destination": "Columbus, GA", "travel_days": 5, "travel_mode": "POV",
    "lodging": {"nights": 4, "rate": 104, "total": 416.0},
    "meals": {"days": 5, "daily_rate": 64, "first_last_rate": 48.0, "total": 288.0},
    "mileage": {"mode": "POV", "one_way_miles": 370, "rate_per_mile": 0.7, "total": 518.0},
    "grand_total": 1222.0,
    "summary": "Lodging: $416.00 | Meals: $288.00 | Mileage: $518.00 | TOTAL: $1222.00",
})
TDY_TOOLS = [{"name": "calculate_travel_cost", "args": {}, "result": TDY_RESULT}]

TDY_ANSWER = """**TDY Cost -- Fort Moore, GA (5 days, POV)**
- Lodging: $416 (4 nights x $104)
- Meals: $288 (3 full days x $64, 2 travel days x $48)
- Mileage: $518 (370 mi x $0.70 x 2)
- **Total: $1,222**
Your DD Form 1610 (Request and Authorization for TDY Travel) is ready for download."""


def search_result(text, section="", source="JTR.pdf"):
    return {"name": "search_regulations", "args": {"query": "q"},
            "result": json.dumps({"success": True, "results": [
                {"text": text, "section": section, "source": source, "domain": "travel", "score": 0.8}]})}


def test_dollar_extraction():
    vals = [v for _, v in ra.extract_dollar_amounts(TDY_ANSWER)]
    assert vals == [416, 104, 288, 64, 48, 518, 0.70, 1222]
    assert ra.extract_dollar_amounts("costs $ 1,234.56.") == [("$1,234.56", 1234.56)]


def test_citation_extraction_on_realistic_answers():
    def kinds(text):
        return [(w, k) for w, k, _ in ra.extract_citations(text)]
    assert kinds("Yes, the GTC is mandatory ... -- JTR 010204") == [("010204", "section")]
    assert kinds("You accrue 2.5 days per month. -- AR 600-8-10, para 3-1") == [("para 3-1", "para")]
    assert kinds("See JTR Table 2-10 and Chapter 3.") == [("Table 2-10", "table"), ("Chapter 3", "chapter")]
    # dollar figures, dates, regulation numbers and form numbers are not citations
    assert kinds(TDY_ANSWER) == []
    assert kinds("AR 623-3 / AFI 36-2406 / DA Form 4856, due 2026-07-10") == []


def test_correct_tdy_answer_verifies():
    v = ra.verify_answer(TDY_ANSWER, TDY_TOOLS)
    assert v == {"checked": True, "numbers_ok": True, "unverified_numbers": [],
                 "citations_ok": True, "unverified_citations": []}


def test_wrong_figure_is_flagged():
    v = ra.verify_answer(TDY_ANSWER.replace("$1,222", "$1,300"), TDY_TOOLS)
    assert v["numbers_ok"] is False and v["unverified_numbers"] == ["$1,300"]


def test_whole_dollar_rounding_allowed_but_not_wrong_cents():
    tools = [{"name": "get_per_diem", "args": {}, "result": json.dumps({"success": True, "lodging": 415.75})}]
    assert ra.verify_answer("Lodging is $416/night.", tools)["numbers_ok"]
    assert not ra.verify_answer("Lodging is $415.70/night.", tools)["numbers_ok"]


def test_numbers_without_tool_calls_are_unverified():
    v = ra.verify_answer("That trip costs about $900.", [])
    assert v["numbers_ok"] is False and v["unverified_numbers"] == ["$900"]


def test_user_supplied_figures_count_as_known():
    assert ra.verify_answer("Your $150 hotel exceeds the rate.", [], "I paid $150 a night")["numbers_ok"]


def test_nothing_to_check_returns_none():
    assert ra.verify_answer("Hi! How can I help with travel, leave or forms?", []) is None


def test_citations_checked_against_search_results():
    tools = [search_result("010204. Government Travel Charge Card. The GTC is mandatory ...",
                           section="Chapter 1 Part B")]
    assert ra.verify_answer("Yes, the GTC is mandatory. -- JTR 010204", tools)["citations_ok"]
    v = ra.verify_answer("Yes, the GTC is mandatory. -- JTR 020210", tools)
    assert v["citations_ok"] is False and v["unverified_citations"] == ["020210"]
    assert ra.verify_answer("-- JTR, Chapter 1", tools)["citations_ok"]


def test_para_citation_matching_is_exact():
    tools = [search_result("3-10. Leave accrual. Soldiers accrue 2.5 days ...")]
    assert not ra.verify_answer("-- AR 600-8-10, para 3-1", tools)["citations_ok"]
    tools = [search_result("3–1. Accrual of leave. Soldiers accrue 2.5 days ...")]
    assert ra.verify_answer("-- AR 600-8-10, para 3-1", tools)["citations_ok"]
    # citations with no search this turn are unverified
    assert not ra.verify_answer("-- AR 600-8-10, para 3-1", TDY_TOOLS)["citations_ok"]


# ---------------------------------------------------------------------------
# Grounding — corrective retry inside the agent loop
# ---------------------------------------------------------------------------

def _resp(content="", tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="tool_calls" if tool_calls else "stop",
        message=SimpleNamespace(content=content, tool_calls=tool_calls))])


class FakeLLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.calls.append(kw)
        return self.responses.pop(0)


def run_agent(monkeypatch, fake, message="TDY to Fort Moore 5 days POV"):
    monkeypatch.setattr(ra, "execute_tool", lambda name, args, profile=None, context=None: TDY_RESULT)
    agent = ra.ReActAgent()
    def use_fake(llm):
        agent._client, agent._model, agent._custom = fake, "fake", False
        agent._base_url, agent._gen_params = "https://fake", {"max_tokens": 512}
    monkeypatch.setattr(agent, "_use_llm", use_fake)
    out = "".join(agent.chat(message))
    return agent, out


TOOL_CALL = [SimpleNamespace(id="t1", function=SimpleNamespace(
    name="calculate_travel_cost", arguments='{"destination_city": "Columbus"}'))]


def test_wrong_figure_triggers_one_corrective_retry(monkeypatch):
    fake = FakeLLM(_resp(tool_calls=TOOL_CALL),
                   _resp(TDY_ANSWER.replace("$1,222", "$1,300")),
                   _resp(TDY_ANSWER))
    agent, out = run_agent(monkeypatch, fake)
    assert len(fake.calls) == 3
    retry = fake.calls[2]
    assert "tools" not in retry and "$1,300" in retry["messages"][-1]["content"]
    assert "$1,222" in out and "$1,300" not in out
    assert agent.last_verification["numbers_ok"] is True
    assert agent.last_tool_results[0]["name"] == "calculate_travel_cost"
    assert agent.get_history()[-1]["content"] == TDY_ANSWER   # history holds the corrected answer


def test_still_wrong_after_retry_is_marked_unverified(monkeypatch):
    wrong = TDY_ANSWER.replace("$1,222", "$1,300")
    fake = FakeLLM(_resp(tool_calls=TOOL_CALL), _resp(wrong), _resp(wrong))
    agent, out = run_agent(monkeypatch, fake)
    assert len(fake.calls) == 3          # only ONE retry
    assert "$1,300" in out
    assert agent.last_verification["numbers_ok"] is False
    assert agent.last_verification["unverified_numbers"] == ["$1,300"]


def test_grounded_answer_needs_no_retry(monkeypatch):
    fake = FakeLLM(_resp(tool_calls=TOOL_CALL), _resp(TDY_ANSWER))
    agent, _ = run_agent(monkeypatch, fake)
    assert len(fake.calls) == 2 and agent.last_verification["numbers_ok"] is True


def test_greeting_has_no_verification_and_tool_log_resets(monkeypatch):
    agent, _ = run_agent(monkeypatch, FakeLLM(_resp(tool_calls=TOOL_CALL), _resp(TDY_ANSWER)))
    fake2 = FakeLLM(_resp("Hello! How can I help?"))
    agent._use_llm = lambda llm: setattr(agent, "_client", fake2)
    "".join(agent.chat("hi"))
    assert agent.last_verification is None and agent.last_tool_results == []


# ---------------------------------------------------------------------------
# /api/chat exposes the verification report
# ---------------------------------------------------------------------------

def test_chat_response_includes_verification(monkeypatch):
    def fake_chat(self, message, profile=None, llm=None):
        self.last_form = None
        self.last_verification = ra.verify_answer("Total: $1,300 -- JTR 020210", TDY_TOOLS)
        yield "Total: $1,300 -- JTR 020210"
    monkeypatch.setattr(ra.ReActAgent, "chat", fake_chat)
    v = chat(TestClient(appmod.app)).json()["verification"]
    assert v == {"checked": True, "numbers_ok": False, "unverified_numbers": ["$1,300"],
                 "citations_ok": False, "unverified_citations": ["020210"]}


def test_chat_response_verification_null_when_nothing_to_check(echo_agent):
    body = chat(TestClient(appmod.app)).json()
    assert "verification" in body and body["verification"] is None


def test_billing_refusal_is_explained_and_flips_health(monkeypatch):
    import openai
    from fastapi.testclient import TestClient
    import app as appmod
    import agents.react_agent as ra

    err = openai.BadRequestError(
        "Error code: 400 - Your credit balance is too low to access the Anthropic API.",
        response=type("R", (), {"status_code": 400, "headers": {}, "request": None})(), body=None)

    def boom(self):
        raise err
        yield  # pragma: no cover
    monkeypatch.setattr(ra.ReActAgent, "_react_loop", boom)
    monkeypatch.setattr(appmod, "LLM_BASE_URL", "https://api.anthropic.com/v1")
    monkeypatch.setattr(appmod, "_probe_llm_key", lambda url, key: (True, ["m"]))
    appmod._key_status.update(ok=None, checked=0.0)
    ra.SERVER_LLM_ISSUE.update(reason=None, at=0.0)
    c = TestClient(appmod.app)
    body = c.post("/api/chat", json={"message": "hi", "tab": "travel"}).json()
    assert "out of credit" in body["response"] and "own API key" in body["response"]
    assert c.get("/api/health").json()["llm_ready"] is False
    ra.SERVER_LLM_ISSUE.update(reason=None, at=0.0)
