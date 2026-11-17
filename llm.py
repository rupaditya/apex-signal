"""Thin LLM client: asks for JSON, validates against a pydantic model,
retries once with the validation error, then raises LLMError."""
from __future__ import annotations

import json
import re
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


def _extract_json(text: str) -> str:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        return fence.group(1).strip()
    start = min([i for i in (text.find("{"), text.find("[")) if i != -1], default=-1)
    if start == -1:
        return text
    end = max(text.rfind("}"), text.rfind("]"))
    return text[start:end + 1]


class BaseLLM:
    name = "base"
    model = ""

    def complete(self, system: str, user: str, task: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def call_json(self, task: str, system: str, user: str, model_cls: Type[T]) -> T:
        schema = json.dumps(model_cls.model_json_schema())
        sys_prompt = (f"{system}\n\nRespond with ONLY a JSON object that validates against this JSON schema "
                      f"(no prose, no markdown):\n{schema}")
        last_err = ""
        prompt = user
        for attempt in range(2):
            raw = self.complete(sys_prompt, prompt, task)
            try:
                return model_cls.model_validate_json(_extract_json(raw))
            except (ValidationError, ValueError) as e:
                last_err = str(e)[:800]
                prompt = (f"{user}\n\nYour previous reply was invalid: {last_err}\n"
                          f"Return corrected JSON only.")
        raise LLMError(f"Model returned invalid JSON for task '{task}' after retry: {last_err}")


class AnthropicLLM(BaseLLM):
    name = "anthropic"

    def __init__(self, api_key: str, model: str, timeout: float = 60):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=2)
        self.model = model

    def complete(self, system: str, user: str, task: str) -> str:
        try:
            msg = self.client.messages.create(
                model=self.model, max_tokens=4000,
                system=system, messages=[{"role": "user", "content": user}],
            )
        except Exception as e:  # SDK raises several error types
            raise LLMError(f"Anthropic API error ({task}): {e}") from e
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


class OpenAICompatLLM(BaseLLM):
    """OpenAI Chat Completions API – also used for Gemini and Groq via their
    OpenAI-compatible endpoints (both have free tiers, no card required)."""
    BASE_URLS = {
        "openai": None,
        "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "groq": "https://api.groq.com/openai/v1",
        "openrouter": "https://openrouter.ai/api/v1",
    }

    def __init__(self, provider: str, api_key: str, model: str, timeout: float = 60):
        from openai import OpenAI
        self.name = provider
        self.model = model
        self.notes: list[str] = []
        # SDK retries 429/5xx with exponential backoff – important on free-tier rate limits
        self.client = OpenAI(api_key=api_key, base_url=self.BASE_URLS.get(provider), timeout=timeout, max_retries=2)

    # preference order used when the configured model id no longer exists (providers retire ids often)
    PREFERENCES = {
        "gemini": [r"gemini-3[\w.-]*-flash(-preview)?$", r"gemini-2\.5-flash$", r"gemini-[\w.-]*flash(-preview)?(-\d+)?$",
                   r"gemini-[\w.-]*flash-lite"],
        "groq": [r"openai/gpt-oss-120b", r"openai/gpt-oss-20b", r"qwen", r"llama-4", r"llama"],
        "openai": [r"gpt-4o-mini", r"gpt-4\.1-mini", r"gpt-"],
        # free-only: never auto-select a paid OpenRouter model
        "openrouter": [r"^openrouter/free$", r"gpt-oss-120b:free$", r"llama-3\.3-70b[\w.-]*:free$",
                       r"qwen[\w./-]*:free$", r"deepseek[\w./-]*:free$", r"gemma[\w./-]*:free$", r":free$"],
    }
    EXCLUDE = re.compile(r"(image|tts|audio|live|embed|whisper|guard|vision|veo|imagen|lyria|computer|native|robotics|learnlm|aqa|thinking-exp)", re.I)

    def _discover_model(self) -> str | None:
        try:
            ids = [m.id.replace("models/", "") for m in self.client.models.list().data]
        except Exception:
            return None
        ids = [i for i in ids if not self.EXCLUDE.search(i)]
        for pat in self.PREFERENCES.get(self.name, []):
            hits = sorted([i for i in ids if re.search(pat, i)], reverse=True)
            if hits:
                return hits[0]
        if self.name == "openrouter":
            return None
        return ids[0] if ids else None

    def _create(self, system, user):
        kwargs = dict(model=self.model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        if not getattr(self, "_no_json_mode", False):
            kwargs["response_format"] = {"type": "json_object"}
        if self.name != "gemini":  # Google recommends default temperature for Gemini 3
            kwargs["temperature"] = 0
        try:
            return self.client.chat.completions.create(**kwargs)
        except Exception as e:
            # some free models reject JSON mode; the prompt already demands JSON and we validate it
            if "response_format" in kwargs and ("response_format" in str(e) or "json" in str(e).lower()) and "400" in str(e):
                self._no_json_mode = True
                kwargs.pop("response_format")
                return self.client.chat.completions.create(**kwargs)
            raise

    def complete(self, system: str, user: str, task: str) -> str:
        try:
            resp = self._create(system, user)
        except Exception as e:
            msg = str(e)
            if ("404" in msg or "not found" in msg.lower() or "does not exist" in msg.lower()) and not getattr(self, "_rediscovered", False):
                self._rediscovered = True
                new = self._discover_model()
                if new and new != self.model:
                    old, self.model = self.model, new
                    self.notes.append(f"{self.name}: model '{old}' unavailable, switched to '{new}'")
                    try:
                        resp = self._create(system, user)
                    except Exception as e2:
                        raise LLMError(f"{self.name} API error ({task}) with {new}: {e2}") from e2
                    return resp.choices[0].message.content or ""
            raise LLMError(f"{self.name} API error ({task}) [model {self.model}]: {msg[:300]}") from e
        return resp.choices[0].message.content or ""


class FallbackLLM(BaseLLM):
    """Tries each provider in order; moves to the next on API errors or
    invalid JSON (e.g. a free-tier daily quota being exhausted)."""

    def __init__(self, providers: list[BaseLLM]):
        if not providers:
            raise LLMError("No LLM provider configured (set GEMINI_API_KEY or GROQ_API_KEY – both free).")
        self.providers = providers
        self.name = " → ".join(p.name for p in providers)
        self.model = providers[0].model
        self.events: list[str] = []

    COOLDOWN_S = 600  # a rate-limited free tier is skipped for 10 min instead of being retried on every call

    def call_json(self, task, system, user, model_cls):
        import time
        errors = []
        now = time.time()
        if not hasattr(self, "_cooldown"):
            self._cooldown = {}
        active = [p for p in self.providers if self._cooldown.get(p.name, 0) <= now] or self.providers
        for p in active:
            try:
                out = p.call_json(task, system, user, model_cls)
                for n in getattr(p, "notes", []):
                    if n not in self.events:
                        self.events.append(n)
                if errors:
                    self.events.append(f"{task}: fell back to {p.name} after: {errors[-1][:160]}")
                return out
            except LLMError as e:
                errors.append(str(e))
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "rate limit" in str(e).lower():
                    self._cooldown[p.name] = time.time() + self.COOLDOWN_S
                    note = f"{p.name} rate-limited (429): skipped for {self.COOLDOWN_S // 60} min"
                    if note not in self.events:
                        self.events.append(note)
        raise LLMError(" | ".join(errors))


def build_llm(settings) -> BaseLLM:
    providers: list[BaseLLM] = []
    for name in settings.llm_chain:
        key, model = settings.llm_keys[name], settings.llm_models[name]
        if name == "anthropic":
            providers.append(AnthropicLLM(key, model, settings.http_timeout * 2))
        else:
            providers.append(OpenAICompatLLM(name, key, model, settings.http_timeout * 2))
    return FallbackLLM(providers)
