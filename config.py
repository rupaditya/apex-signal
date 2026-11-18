"""Runtime configuration. All credentials come from environment variables
(or Streamlit secrets, which app.py copies into the environment).

Zero-cost default: Gemini -> Groq -> OpenRouter free models (LLM),
Tavily -> Serper -> DuckDuckGo via `ddgs` (search). All free, no card."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

LLM_ORDER = ["groq", "gemini", "openrouter", "anthropic", "openai"]  # Groq first: Gemini free quota proved too small in live runs


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True)
class Settings:
    llm_keys: dict          # provider -> api key (only providers with a key)
    llm_models: dict        # provider -> model id
    llm_chain: list         # providers in the order they will be tried
    tavily_api_key: str
    serper_api_key: str
    search_chain: list      # e.g. ["tavily", "duckduckgo"]
    db_path: str
    http_timeout: float
    max_sources_for_extraction: int
    fetch_pages: bool = True

    # back-compat helpers used by the UI
    @property
    def llm_provider(self) -> str:
        return self.llm_chain[0] if self.llm_chain else "none"

    @property
    def search_provider(self) -> str:
        return self.search_chain[0] if self.search_chain else "none"

    def missing_credentials(self) -> list[str]:
        if not self.llm_chain:
            return ["GEMINI_API_KEY or GROQ_API_KEY (both free)"]
        return []


def load_settings() -> Settings:
    keys = {
        "gemini": _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY"),
        "groq": _env("GROQ_API_KEY"),
        "openrouter": _env("OPENROUTER_API_KEY"),
        "anthropic": _env("ANTHROPIC_API_KEY"),
        "openai": _env("OPENAI_API_KEY"),
    }
    keys = {k: v for k, v in keys.items() if v}
    models = {
        "gemini": _env("GEMINI_MODEL", "gemini-3-flash-preview"),
        "groq": _env("GROQ_MODEL", "openai/gpt-oss-120b"),
        "openrouter": _env("OPENROUTER_MODEL", "openrouter/free"),
        "anthropic": _env("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
        "openai": _env("OPENAI_MODEL", "gpt-4o-mini"),
    }
    preferred = _env("LLM_PROVIDER").lower()
    order = ([preferred] if preferred in keys else []) + [p for p in LLM_ORDER if p in keys and p != preferred]

    tavily, serper = _env("TAVILY_API_KEY"), _env("SERPER_API_KEY")
    search_chain = []
    if _env("SEARCH_PROVIDER").lower() == "serper" and serper:
        search_chain.append("serper")
    if tavily:
        search_chain.append("tavily")
    if serper and "serper" not in search_chain:
        search_chain.append("serper")
    if _env("DISABLE_DDG_FALLBACK") != "1":
        search_chain.append("duckduckgo")  # keyless, always available as last resort

    return Settings(
        llm_keys=keys, llm_models=models, llm_chain=order,
        tavily_api_key=tavily, serper_api_key=serper, search_chain=search_chain,
        db_path=_env("APEX_DB_PATH", "apex_signal.db"),
        http_timeout=float(_env("HTTP_TIMEOUT", "30")),
        max_sources_for_extraction=int(_env("MAX_SOURCES", "14")),
        fetch_pages=_env("APEX_FETCH_PAGES", "1") != "0",
    )
