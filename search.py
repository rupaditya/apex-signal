"""Live web-search integrations (Tavily primary, Serper alternative).

Both return a list of SearchResult. Failures raise SearchError so the
pipeline can record them instead of silently returning nothing."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from urllib.parse import urlparse

import requests


class SearchError(RuntimeError):
    pass


@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str
    content: str = ""                  # longer page text when the provider returns it
    published_date: str | None = None
    query: str = ""
    provider: str = ""

    @property
    def domain(self) -> str:
        return domain_of(self.url)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["domain"] = self.domain
        return d


def domain_of(url: str) -> str:
    try:
        host = urlparse(url if "://" in url else f"https://{url}").netloc.lower()
    except ValueError:
        return ""
    host = host.split("@")[-1].split(":")[0]
    return host[4:] if host.startswith("www.") else host


class TavilySearch:
    name = "tavily"
    URL = "https://api.tavily.com/search"

    def __init__(self, api_key: str, timeout: float = 30):
        self.api_key = api_key
        self.timeout = timeout

    def search(self, query: str, max_results: int = 5, news: bool = False,
               include_domains: list[str] | None = None) -> list[SearchResult]:
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",   # 1 credit/query on the free plan (advanced = 2)
            "include_raw_content": False,
            "topic": "news" if news else "general",
        }
        if news:
            payload["days"] = 730
        if include_domains:
            payload["include_domains"] = include_domains
        try:
            r = requests.post(self.URL, json=payload, timeout=self.timeout,
                              headers={"Authorization": f"Bearer {self.api_key}"})
        except requests.RequestException as e:
            raise SearchError(f"Tavily request failed: {e}") from e
        if r.status_code != 200:
            raise SearchError(f"Tavily HTTP {r.status_code}: {r.text[:200]}")
        try:
            data = r.json()
        except ValueError as e:
            raise SearchError("Tavily returned non-JSON") from e
        out = []
        for item in data.get("results", []) or []:
            if not item.get("url"):
                continue
            content = (item.get("content") or "").strip()
            out.append(SearchResult(
                url=item["url"], title=(item.get("title") or "").strip(),
                snippet=content[:400], content=content[:4000],
                published_date=item.get("published_date"), query=query, provider=self.name,
            ))
        return out


class SerperSearch:
    name = "serper"

    def __init__(self, api_key: str, timeout: float = 30):
        self.api_key = api_key
        self.timeout = timeout

    def search(self, query: str, max_results: int = 5, news: bool = False,
               include_domains: list[str] | None = None) -> list[SearchResult]:
        if include_domains:
            query = f"{query} " + " OR ".join(f"site:{d}" for d in include_domains)
        endpoint = "https://google.serper.dev/news" if news else "https://google.serper.dev/search"
        try:
            r = requests.post(endpoint, json={"q": query, "num": max_results}, timeout=self.timeout,
                              headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"})
        except requests.RequestException as e:
            raise SearchError(f"Serper request failed: {e}") from e
        if r.status_code != 200:
            raise SearchError(f"Serper HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        items = data.get("news" if news else "organic", []) or []
        out = []
        for item in items[:max_results]:
            link = item.get("link")
            if not link:
                continue
            snip = (item.get("snippet") or "").strip()
            out.append(SearchResult(url=link, title=item.get("title", ""), snippet=snip, content=snip,
                                    published_date=item.get("date"), query=query, provider=self.name))
        return out


class DuckDuckGoSearch:
    """Keyless search via the `ddgs` package. Free, no account; used as the
    last-resort fallback. Snippets only, and it can be rate-limited."""
    name = "duckduckgo"

    def search(self, query: str, max_results: int = 5, news: bool = False,
               include_domains: list[str] | None = None) -> list[SearchResult]:
        try:
            from ddgs import DDGS
        except ImportError as e:
            raise SearchError("ddgs package not installed") from e
        if include_domains:
            query = f"{query} " + " OR ".join(f"site:{d}" for d in include_domains)
        try:
            with DDGS() as d:
                items = list(d.news(query, max_results=max_results) if news else d.text(query, max_results=max_results))
        except Exception as e:
            if "No results" in str(e):
                return []
            raise SearchError(f"DuckDuckGo search failed: {e}") from e
        out = []
        for it in items or []:
            link = it.get("url") or it.get("href")
            if not link:
                continue
            body = (it.get("body") or "").strip()
            out.append(SearchResult(url=link, title=it.get("title", ""), snippet=body[:400], content=body,
                                    published_date=it.get("date"), query=query, provider=self.name))
        return out


class FallbackSearch:
    """Tries each search provider in order; moves on when one errors
    (e.g. Tavily free credits used up)."""

    def __init__(self, providers: list):
        self.providers = providers
        self.name = " → ".join(p.name for p in providers)

    def search(self, query, max_results=5, news=False, include_domains=None):
        errors = []
        for p in self.providers:
            try:
                return p.search(query, max_results=max_results, news=news, include_domains=include_domains)
            except SearchError as e:
                errors.append(str(e))
        raise SearchError(" | ".join(errors) or "no search provider configured")


def build_search_client(settings):
    providers = []
    for name in settings.search_chain:
        if name == "tavily":
            providers.append(TavilySearch(settings.tavily_api_key, settings.http_timeout))
        elif name == "serper":
            providers.append(SerperSearch(settings.serper_api_key, settings.http_timeout))
        elif name == "duckduckgo":
            providers.append(DuckDuckGoSearch())
    return FallbackSearch(providers)
