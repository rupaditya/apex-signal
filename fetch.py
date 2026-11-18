"""Read the actual source pages (free, no API) instead of relying only on
search snippets. Gives the extractor more real text to quote from and
recovers publication dates from page metadata, so fewer items are 'undated'."""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import requests

from .search import domain_of

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36 ApexSignalResearchBot/1.0")
# sites that block bots or need a login – don't waste time on them
SKIP_DOMAINS = {"linkedin.com", "facebook.com", "instagram.com", "x.com", "twitter.com", "youtube.com",
                "zoominfo.com", "glassdoor.com", "glassdoor.co.in", "tiktok.com"}
MAX_BYTES = 1_500_000
MAX_TEXT = 6000

DATE_META = [
    ("property", "article:published_time"), ("property", "og:published_time"),
    ("name", "article:published_time"), ("itemprop", "datePublished"), ("name", "pubdate"),
    ("name", "publish-date"), ("name", "date"), ("name", "DC.date.issued"), ("name", "dc.date"),
    ("property", "article:modified_time"),
]


def _published_from_html(soup) -> Optional[str]:
    for attr, val in DATE_META:
        tag = soup.find("meta", attrs={attr: val})
        if tag and tag.get("content"):
            return tag["content"].strip()
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except (ValueError, TypeError):
            continue
        items = data if isinstance(data, list) else data.get("@graph", [data]) if isinstance(data, dict) else []
        for it in items:
            if isinstance(it, dict) and it.get("datePublished"):
                return str(it["datePublished"])
    t = soup.find("time", attrs={"datetime": True})
    if t:
        return t["datetime"].strip()
    return None


def fetch_page(url: str, timeout: float = 8) -> dict:
    """Returns {ok, text, published_date, error}. Never raises."""
    d = domain_of(url)
    if any(d == s or d.endswith("." + s) for s in SKIP_DOMAINS):
        return {"ok": False, "text": "", "published_date": None, "error": "skipped (site blocks automated reading)"}
    if url.lower().split("?")[0].endswith((".pdf", ".doc", ".docx", ".xls", ".xlsx", ".zip")):
        return {"ok": False, "text": "", "published_date": None, "error": "skipped (not an HTML page)"}
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": UA, "Accept": "text/html"}, stream=True)
        if r.status_code != 200:
            return {"ok": False, "text": "", "published_date": None, "error": f"HTTP {r.status_code}"}
        if "html" not in r.headers.get("Content-Type", "html").lower():
            return {"ok": False, "text": "", "published_date": None, "error": "not HTML"}
        raw = r.raw.read(MAX_BYTES, decode_content=True)
        html = raw.decode(r.encoding or "utf-8", errors="replace")
    except Exception as e:  # network errors, timeouts, bad encodings
        return {"ok": False, "text": "", "published_date": None, "error": type(e).__name__}
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        published = _published_from_html(soup)
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg"]):
            tag.decompose()
        root = soup.find("article") or soup.find("main") or soup.body or soup
        text = re.sub(r"\s+", " ", root.get_text(" ", strip=True))
    except Exception as e:
        return {"ok": False, "text": "", "published_date": None, "error": f"parse error: {type(e).__name__}"}
    if len(text) < 200:
        return {"ok": False, "text": "", "published_date": published, "error": "too little text (likely JS-rendered)"}
    return {"ok": True, "text": text[:MAX_TEXT], "published_date": published, "error": None}


def fetch_many(urls: list[str], cache=None, workers: int = 6) -> dict[str, dict]:
    def one(u):
        if cache:
            hit = cache.cache_get("page", u, max_age_hours=72)
            if hit is not None:
                hit["cached"] = True
                return u, hit
        res = fetch_page(u)
        if cache and (res["ok"] or res["error"] and "HTTP 4" in res["error"]):
            cache.cache_put("page", u, res)
        res["cached"] = False
        return u, res

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return dict(ex.map(one, urls))


# ---------------------------------------------------------------- passage selection
SIGNAL_WORDS = re.compile(
    r"\b(invoice|payable|accounts|finance|financial|cfo|controller|comptroller|erp|sap|oracle|netsuite|workday|"
    r"dynamics|tally|ledger|procure|procurement|supplier|vendor|automation|automate|transformation|shared services|"
    r"hiring|hires|appoint|appointed|joins|acquisition|acquire|acquires|merger|expansion|expand|plant|facility|"
    r"office|warehouse|ipo|listing|drhp|raise|raised|funding|series|revenue|growth|employees|headcount|audit|"
    r"go-live|went live|implementation|migrat)\w*", re.I)


def select_passages(text: str, must: list[str], max_chars: int = 1500) -> str:
    """Keep the sentences that matter (mention the company/person or a finance signal), in original order.
    Cuts tokens ~3x versus sending whole pages, which keeps free-tier LLMs under their per-minute limits."""
    if len(text) <= max_chars:
        return text
    sents = re.split(r"(?<=[.!?])\s+", text)
    must_l = [m.lower() for m in must if m]
    scored = []
    for i, s in enumerate(sents):
        sl = s.lower()
        score = 3 * sum(m in sl for m in must_l) + 2 * min(len(SIGNAL_WORDS.findall(s)), 2)
        if score:
            scored.append((score, i, s))
    keep, used = set(), 0
    for score, i, s in sorted(scored, key=lambda x: (-x[0], x[1])):
        if used + len(s) > max_chars:
            continue
        keep.add(i)
        used += len(s) + 1
    if not keep:
        return text[:max_chars]
    return " … ".join(sents[i] for i in sorted(keep))
