"""Provider-chain tests (no network)."""
import json

import pytest

from apex_signal.config import load_settings
from apex_signal.llm import BaseLLM, FallbackLLM, LLMError
from apex_signal.schemas import AngleRationale
from apex_signal.search import FallbackSearch, SearchError, SearchResult

OK = {"hook_summary": "h", "hypothesis": "x", "product_relevance": "y", "assumptions": []}


class Down(BaseLLM):
    name, model = "gemini", "m"
    def complete(self, *a):
        raise LLMError("429 RESOURCE_EXHAUSTED")


class Up(BaseLLM):
    name, model = "groq", "m"
    def complete(self, *a):
        return json.dumps(OK)


def test_llm_falls_back_and_records_it():
    llm = FallbackLLM([Down(), Up()])
    assert llm.call_json("t", "s", "u", AngleRationale).hypothesis == "x"
    assert "fell back to groq" in llm.events[0]


def test_llm_all_down_raises():
    with pytest.raises(LLMError):
        FallbackLLM([Down()]).call_json("t", "s", "u", AngleRationale)


def test_search_falls_back():
    class Bad:
        name = "tavily"
        def search(self, *a, **k):
            raise SearchError("432 credits exhausted")
    class Good:
        name = "duckduckgo"
        def search(self, *a, **k):
            return [SearchResult(url="https://x.com", title="t", snippet="s")]
    assert FallbackSearch([Bad(), Good()]).search("q")[0].url == "https://x.com"


def test_free_defaults(monkeypatch):
    for k in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "TAVILY_API_KEY", "SERPER_API_KEY", "LLM_PROVIDER", "GOOGLE_API_KEY"]:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("GROQ_API_KEY", "q")
    s = load_settings()
    assert s.llm_chain == ["gemini", "groq"]
    assert s.search_chain == ["duckduckgo"]
    assert s.missing_credentials() == []
    monkeypatch.delenv("GEMINI_API_KEY"); monkeypatch.delenv("GROQ_API_KEY")
    assert load_settings().missing_credentials()


def _fake_client(ids, fail_models=(), reject_json=False):
    from types import SimpleNamespace as NS
    calls = []
    def create(**kw):
        calls.append(kw)
        if kw["model"] in fail_models:
            raise Exception("Error code: 404 model not found")
        if reject_json and "response_format" in kw:
            raise Exception("Error code: 400 response_format json_object not supported")
        return NS(choices=[NS(message=NS(content=json.dumps(OK)))])
    return NS(chat=NS(completions=NS(create=create)),
              models=NS(list=lambda: NS(data=[NS(id=i) for i in ids]))), calls


def test_openrouter_discovery_only_picks_free_models():
    from apex_signal.llm import OpenAICompatLLM
    llm = OpenAICompatLLM("openrouter", "k", "gone/model:free")
    llm.client, _ = _fake_client(["anthropic/claude-x", "openai/gpt-oss-120b:free"], fail_models={"gone/model:free"})
    llm.call_json("t", "s", "u", AngleRationale)
    assert llm.model == "openai/gpt-oss-120b:free"
    llm2 = OpenAICompatLLM("openrouter", "k", "gone/model:free")
    llm2.client, _ = _fake_client(["anthropic/claude-x"], fail_models={"gone/model:free"})
    with pytest.raises(LLMError):
        llm2.call_json("t", "s", "u", AngleRationale)


def test_json_mode_rejected_falls_back_to_plain():
    from apex_signal.llm import OpenAICompatLLM
    llm = OpenAICompatLLM("openrouter", "k", "x:free")
    llm.client, calls = _fake_client([], reject_json=True)
    assert llm.call_json("t", "s", "u", AngleRationale).hypothesis == "x"
    assert "response_format" not in calls[-1]


def test_ipo_is_indirect_and_depends_on_ownership():
    from apex_signal import policy
    assert policy.RELEVANCE["ipo_or_listing"] == "INDIRECT"
    assert "ownership_status" in policy.SIGNAL_DEPENDS_ON["ipo_or_listing"]
