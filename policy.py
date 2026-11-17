"""Explicit decision rules. No confidence scores: every evidence item gets
categorical labels (grounded / tier / recency / relevance / entity / conflict)
and every exclusion is recorded with a reason the reviewer can read."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

from dateutil import parser as dateparser

from .search import domain_of

# ---------------------------------------------------------------- sources
TIER_A_DOMAINS = {
    "sec.gov", "reuters.com", "bloomberg.com", "wsj.com", "ft.com", "cnbc.com", "apnews.com",
    "businesswire.com", "prnewswire.com", "globenewswire.com", "nytimes.com", "economictimes.indiatimes.com",
    "livemint.com", "business-standard.com", "thehindubusinessline.com", "moneycontrol.com",
    "techcrunch.com", "cfo.com", "cfodive.com", "theguardian.com", "bbc.co.uk", "bbc.com",
    "accesswire.com", "newswire.ca", "bizjournals.com", "fortune.com",
}
TIER_C_DOMAINS = {  # data brokers, scraped profiles, user-generated content
    "zoominfo.com", "rocketreach.co", "signalhire.com", "contactout.com", "apollo.io", "lusha.com",
    "leadiq.com", "datanyze.com", "owler.com", "craft.co", "reddit.com", "quora.com", "medium.com",
    "facebook.com", "x.com", "twitter.com", "instagram.com", "tiktok.com", "youtube.com",
    "wikiwand.com", "peoplefinders.com", "spokeo.com", "clay.com", "6sense.com", "cbinsights.com",
    "theorg.com", "golden.com", "crustdata.com", "leadgenius.com",
}


def source_tier(url: str, company_domain: Optional[str]) -> str:
    d = domain_of(url)
    if company_domain and (d == company_domain or d.endswith("." + company_domain)):
        return "A"  # first-party
    if d.endswith(".gov") or d.endswith(".gov.in") or d.endswith(".gov.uk"):
        return "A"
    if any(d == x or d.endswith("." + x) for x in TIER_A_DOMAINS):
        return "A"
    if any(d == x or d.endswith("." + x) for x in TIER_C_DOMAINS):
        return "C"
    return "B"  # identifiable publisher, job board, LinkedIn, trade press, etc.


# ---------------------------------------------------------------- recency
FRESH_DAYS, AGING_DAYS = 365, 730


def parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        dt = dateparser.parse(str(s), default=datetime(2000, 1, 1), fuzzy=True)
    except (ValueError, OverflowError, TypeError):
        return None
    if dt.year < 1990:
        return None
    return dt.date()


def recency_label(d: Optional[date], today: date) -> str:
    if d is None:
        return "undated"
    age = (today - d).days
    if age < -31:
        return "undated"  # future date: treat as unreliable
    if age <= FRESH_DAYS:
        return "fresh"
    if age <= AGING_DAYS:
        return "aging"
    return "stale"


# ---------------------------------------------------------------- relevance
RELEVANCE = {
    "ap_finance_hiring": "DIRECT",
    "erp_or_finance_system_change": "DIRECT",
    "finance_transformation": "DIRECT",
    "stated_ap_pain": "DIRECT",
    "finance_leadership_change": "INDIRECT",
    "ipo_or_listing": "INDIRECT",
    "acquisition_or_merger": "INDIRECT",
    "expansion": "INDIRECT",
    "rapid_growth": "INDIRECT",
    "prospect_statement": "INDIRECT",
    "funding": "NONE",
    "product_or_marketing": "NONE",
    "award_or_pr": "NONE",
    "other": "NONE",
}
RELEVANCE_WHY = {
    "DIRECT": "Signal is about AP / finance operations itself.",
    "INDIRECT": "Signal plausibly increases AP workload or creates a buying window, but the link is a hypothesis.",
    "NONE": "Credible company news with no defensible link to invoice processing.",
}
# Facts a hook of this type silently depends on. If they conflict, the hook is unsafe.
SIGNAL_DEPENDS_ON = {
    "finance_leadership_change": {"cfo_name"},
    "erp_or_finance_system_change": {"erp_system"},
    "acquisition_or_merger": {"ownership_status"},
    "ipo_or_listing": {"ownership_status"},
    "rapid_growth": {"employee_count"},
}


# ---------------------------------------------------------------- text utils
def norm(s: str) -> str:
    s = s.lower().replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"[^a-z0-9$%.' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


CORP_SUFFIX = re.compile(r"\b(inc|incorporated|llc|ltd|limited|plc|corp|corporation|co|company|gmbh|pvt|private|group|holdings|sa|ag|bv)\b\.?", re.I)


def company_core(name: str) -> str:
    return re.sub(r"\s+", " ", CORP_SUFFIX.sub("", norm(name))).strip(" .")


def is_grounded(quote: str, content: str) -> bool:
    """Quote must actually appear in the retrieved source text (allowing for
    whitespace/punctuation/ellipsis differences)."""
    q, c = norm(quote), norm(content)
    if len(q) < 12 or not c:
        return False
    if q in c:
        return True
    words = q.split()
    if len(words) < 4:
        return False
    shingles = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
    hit = sum(1 for sh in shingles if sh in c)
    return hit / len(shingles) >= 0.8


def mentions_company(text: str, company_name: str, company_domain: Optional[str]) -> bool:
    t = norm(text)
    core = company_core(company_name)
    if core and core in t:
        return True
    if company_domain and company_domain.split(".")[0] in t.replace(" ", ""):
        return True
    return False


# ---------------------------------------------------------------- roles
ROLE_FAMILIES = {
    "cfo": ["cfo", "chief financial", "chief finance"],
    "controller": ["controller", "comptroller", "chief accounting", "head of accounting", "accounting director"],
    "finance_leader": ["vp finance", "vice president finance", "vice president of finance", "finance director",
                       "director of finance", "head of finance", "finance head", "svp finance", "fp&a"],
    "ap": ["accounts payable", "ap manager", "ap lead", "payables", "procure to pay", "p2p", "ap director"],
}
FINANCE_FAMILIES = set(ROLE_FAMILIES)


def role_families(title: str) -> set[str]:
    t = " " + norm(title).replace(",", " ") + " "
    t = t.replace(" of ", " of ").replace("v.p.", "vp")
    fams = set()
    for fam, kws in ROLE_FAMILIES.items():
        if any(f" {kw} " in t or t.strip().startswith(kw) or f" {kw}" in t for kw in kws):
            fams.add(fam)
    return fams


def roles_compatible(given: str, found: str) -> bool:
    g, f = role_families(given), role_families(found)
    if g and f:
        return bool(g & f) or ({"finance_leader", "cfo"} >= (g | f))
    return norm(given) in norm(found) or norm(found) in norm(given)


def in_buyer_persona(role: str) -> bool:
    return bool(role_families(role) & FINANCE_FAMILIES)


# ---------------------------------------------------------------- identity
@dataclass
class IdentityDecision:
    company_status: str          # PROVIDED | RESOLVED | UNRESOLVED | AMBIGUOUS
    company_domain: Optional[str]
    prospect_status: str         # CONFIRMED | ROLE_MISMATCH | NOT_FOUND | AMBIGUOUS | CONFLICTING
    persona_fit: bool
    reasons: list[str] = field(default_factory=list)
    blocking: bool = False

    def to_dict(self):
        return self.__dict__.copy()


def employer_matches(employer: str, company_name: str, company_domain: Optional[str]) -> bool:
    e, core = company_core(employer), company_core(company_name)
    if core and (core in e or (e and e in core)):
        return True
    return bool(company_domain and company_domain.split(".")[0] in e.replace(" ", ""))


def decide_identity(extraction, sources: dict, company_name: str, prospect_name: str, role: str,
                    website: Optional[str]) -> IdentityDecision:
    reasons: list[str] = []
    # --- company
    if website:
        company_domain = domain_of(website)
        company_status = "PROVIDED"
        reasons.append(f"Company website provided by rep: {company_domain}.")
    else:
        cands = [c for c in extraction.company_candidates if c.domain]
        domains = {}
        for c in cands:
            d = domain_of(c.domain)
            if source_tier("https://" + d, None) == "C" or d in TIER_A_DOMAINS:
                continue  # a publisher/broker is not a candidate company
            root = d.split(".")[0]  # acme.com / acme.co.uk -> same organisation
            if not any(k.split(".")[0] == root for k in domains):
                domains[d] = c
        if len(domains) == 1:
            company_domain = next(iter(domains))
            company_status = "RESOLVED"
            reasons.append(f"Single matching organisation found: {company_domain}.")
        elif len(domains) > 1:
            company_domain = None
            company_status = "AMBIGUOUS"
            listed = "; ".join(f"{c.name} ({d}) – {c.description}" for d, c in list(domains.items())[:4])
            reasons.append(f"{len(domains)} different organisations match '{company_name}': {listed}.")
        else:
            company_domain = None
            company_status = "UNRESOLVED"
            reasons.append("No organisation with an identifiable website found for this name.")

    # --- prospect (only grounded matches count)
    pname = norm(prospect_name)
    grounded = []
    for m in extraction.person_matches:
        src = sources.get(m.source_id)
        if not src or norm(m.name) != pname and pname not in norm(m.name):
            continue
        if not is_grounded(m.quote, src["content"] + " " + src["title"]):
            continue
        if source_tier(src["url"], company_domain) == "C":
            continue  # data brokers don't establish identity
        grounded.append(m)

    at_target_current = [m for m in grounded if m.is_current and employer_matches(m.employer, company_name, company_domain)]
    elsewhere_current = [m for m in grounded if m.is_current and not employer_matches(m.employer, company_name, company_domain)]
    former_at_target = [m for m in grounded if not m.is_current and employer_matches(m.employer, company_name, company_domain)]
    other_employers = {company_core(m.employer) for m in elsewhere_current}

    if at_target_current and (former_at_target or elsewhere_current) and former_at_target:
        prospect_status = "CONFLICTING"
        reasons.append(f"Sources disagree on whether {prospect_name} still works at {company_name} "
                       f"(listed as current in {len(at_target_current)} source(s), former in {len(former_at_target)}).")
    elif at_target_current:
        if any(roles_compatible(role, m.title) for m in at_target_current):
            prospect_status = "CONFIRMED"
            m = next(m for m in at_target_current if roles_compatible(role, m.title))
            reasons.append(f"{prospect_name} confirmed as '{m.title}' at {m.employer} ({m.source_id}).")
        else:
            prospect_status = "ROLE_MISMATCH"
            reasons.append(f"{prospect_name} found at {company_name} but titled "
                           f"'{at_target_current[0].title}', not '{role}'. Email will not assert the role.")
        if elsewhere_current:
            reasons.append(f"Note: same name also appears at {', '.join(sorted(other_employers))} (treated as a different person).")
    elif former_at_target:
        prospect_status = "CONFLICTING"
        reasons.append(f"Sources describe {prospect_name}'s role at {company_name} as former/past.")
    elif elsewhere_current:
        prospect_status = "AMBIGUOUS"
        reasons.append(f"'{prospect_name}' found only at other organisation(s): {', '.join(sorted(other_employers))}. "
                       f"Cannot confirm this is the intended person at {company_name}.")
    else:
        prospect_status = "NOT_FOUND"
        reasons.append(f"No public source ties {prospect_name} to {company_name}; email will be company-level only.")

    persona_fit = in_buyer_persona(role)
    if not persona_fit:
        reasons.append(f"Role '{role}' is outside the AP-automation buyer persona (CFO/Controller/Finance/AP).")

    blocking = company_status == "AMBIGUOUS" or prospect_status in ("AMBIGUOUS", "CONFLICTING")
    return IdentityDecision(company_status, company_domain, prospect_status, persona_fit, reasons, blocking)


# ---------------------------------------------------------------- evidence
@dataclass
class EvidenceItem:
    id: str
    source_id: str
    url: str
    domain: str
    title: str
    published_date: Optional[str]
    quote: str
    observation: str
    signal_type: str
    subject: str
    tier: str
    recency: str
    relevance: str
    grounded: bool
    entity_ok: bool
    conflicted_keys: list[str] = field(default_factory=list)
    eligible: bool = False
    exclusions: list[str] = field(default_factory=list)

    def to_dict(self):
        return self.__dict__.copy()


def _norm_fact(key: str, value: str):
    v = norm(value)
    if key == "employee_count":
        nums = re.findall(r"\d[\d,\.]*", value.replace(",", ""))
        if nums:
            try:
                n = float(nums[0])
                if re.search(r"\b(k|thousand)\b", v):
                    n *= 1000
                return n
            except ValueError:
                pass
    if key in ("prospect_current_employer",):
        return company_core(value)
    return v


def detect_conflicts(facts, sources: dict, company_domain: Optional[str]) -> dict[str, list[dict]]:
    """Two credible (tier A/B), grounded sources disagree on the same fact key."""
    by_key: dict[str, list[dict]] = {}
    for f in facts:
        src = sources.get(f.source_id)
        if not src or not is_grounded(f.quote, src["content"] + " " + src["title"]):
            continue
        if source_tier(src["url"], company_domain) == "C":
            continue
        by_key.setdefault(f.key, []).append({"value": f.value, "norm": _norm_fact(f.key, f.value),
                                             "source_id": f.source_id, "url": src["url"]})
    conflicts = {}
    for key, items in by_key.items():
        vals = [i["norm"] for i in items]
        if key == "employee_count":
            nums = [v for v in vals if isinstance(v, float)]
            if len(nums) >= 2 and max(nums) > 1.5 * min(nums):
                conflicts[key] = items
        else:
            distinct = {v for v in vals if v}
            # values where one contains the other ("acme" vs "acme corp") are not conflicts
            reduced = [v for v in distinct if not any(v != o and (v in o or o in v) for o in distinct)]
            if len(reduced) >= 2 and len({i["source_id"] for i in items}) >= 2:
                conflicts[key] = items
    return conflicts


def assess_evidence(observations, sources: dict, company_name: str, company_domain: Optional[str],
                    identity: IdentityDecision, conflicts: dict, today: date) -> list[EvidenceItem]:
    items = []
    for i, o in enumerate(observations, 1):
        src = sources.get(o.source_id)
        if not src:
            continue
        pub = src.get("published_date") or o.event_date
        rec = recency_label(parse_date(pub), today)
        tier = source_tier(src["url"], company_domain)
        grounded = is_grounded(o.quote, src["content"] + " " + src["title"])
        entity_ok = o.subject in ("company", "prospect") and mentions_company(
            src["content"] + " " + src["title"], company_name, company_domain)
        if o.subject == "prospect" and identity.prospect_status != "CONFIRMED":
            entity_ok = False
        relevance = RELEVANCE.get(o.signal_type, "NONE")
        conflicted = sorted(SIGNAL_DEPENDS_ON.get(o.signal_type, set()) & set(conflicts))
        ex = []
        if not grounded:
            ex.append("Quote not found in retrieved source text (possible model hallucination).")
        if not entity_ok:
            ex.append("Not verifiably about this company/prospect.")
        if tier == "C":
            ex.append("Low-reliability source (data broker / user-generated).")
        if rec == "stale":
            ex.append(f"Stale (> {AGING_DAYS // 365} years old).")
        if relevance == "NONE":
            ex.append(f"No product relevance ({o.signal_type}).")
        if conflicted:
            ex.append(f"Depends on conflicting facts: {', '.join(conflicted)}.")
        items.append(EvidenceItem(
            id=f"E{i}", source_id=o.source_id, url=src["url"], domain=domain_of(src["url"]), title=src["title"],
            published_date=pub, quote=o.quote, observation=o.observation, signal_type=o.signal_type,
            subject=o.subject, tier=tier, recency=rec, relevance=relevance, grounded=grounded,
            entity_ok=entity_ok, conflicted_keys=conflicted, eligible=not ex, exclusions=ex,
        ))
    return items


# ---------------------------------------------------------------- angle selection
OUTCOMES = {
    "DRAFT_READY": "Draft ready for review",
    "DRAFT_WITH_CAVEATS": "Draft ready – reviewer must check caveats",
    "NEEDS_INPUT_AMBIGUOUS_IDENTITY": "Needs rep input – identity ambiguous",
    "ABSTAIN_CONFLICTING_EVIDENCE": "Abstained – conflicting / unreliable evidence",
    "ABSTAIN_NO_CREDIBLE_EVIDENCE": "Abstained – no credible evidence",
    "ABSTAIN_LOW_RELEVANCE": "Abstained – signals found but not relevant to AP automation",
}
_REC_ORDER = {"fresh": 0, "aging": 1, "undated": 2, "stale": 3}


def rank_key(e: EvidenceItem):
    d = parse_date(e.published_date)
    return (0 if e.relevance == "DIRECT" else 1, 0 if e.tier == "A" else 1, _REC_ORDER[e.recency],
            0 if e.subject == "prospect" else 1, -(d.toordinal() if d else 0))


@dataclass
class AngleDecision:
    outcome: str
    reason: str
    primary: Optional[EvidenceItem] = None
    supporting: list[EvidenceItem] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    def to_dict(self):
        return {"outcome": self.outcome, "label": OUTCOMES[self.outcome], "reason": self.reason,
                "primary": self.primary.id if self.primary else None,
                "supporting": [s.id for s in self.supporting], "caveats": self.caveats}


def select_angle(items: list[EvidenceItem], identity: IdentityDecision, conflicts: dict,
                 search_failures: int = 0) -> AngleDecision:
    if identity.blocking:
        return AngleDecision("NEEDS_INPUT_AMBIGUOUS_IDENTITY", " ".join(identity.reasons))
    if "prospect_current_employer" in conflicts:
        vals = sorted({c["value"] for c in conflicts["prospect_current_employer"]})
        return AngleDecision("ABSTAIN_CONFLICTING_EVIDENCE",
                             f"Credible sources disagree on the prospect's current employer: {', '.join(vals)}.")

    credible = [e for e in items if e.grounded and e.entity_ok and e.tier in ("A", "B")]
    eligible = sorted([e for e in items if e.eligible], key=rank_key)

    if not eligible:
        relevant_but_blocked = [e for e in credible if e.relevance != "NONE"]
        if relevant_but_blocked and all(e.conflicted_keys or e.recency == "stale" for e in relevant_but_blocked):
            if any(e.conflicted_keys for e in relevant_but_blocked):
                keys = sorted({k for e in relevant_but_blocked for k in e.conflicted_keys})
                return AngleDecision("ABSTAIN_CONFLICTING_EVIDENCE",
                                     f"Relevant signals exist but depend on facts sources disagree on ({', '.join(keys)}).")
            return AngleDecision("ABSTAIN_NO_CREDIBLE_EVIDENCE",
                                 "Only stale (> 2 years) relevant signals found; nothing current to anchor outreach.")
        if credible and all(e.relevance == "NONE" for e in credible):
            kinds = sorted({e.signal_type for e in credible})
            return AngleDecision("ABSTAIN_LOW_RELEVANCE",
                                 f"Found {len(credible)} credible signal(s) ({', '.join(kinds)}) but none connects "
                                 "defensibly to invoice processing. Recommend nurture rather than a cold pitch.")
        low_rel = [e for e in items if e.grounded and e.entity_ok and e.tier == "C" and e.relevance != "NONE"]
        if low_rel:
            return AngleDecision("ABSTAIN_CONFLICTING_EVIDENCE",
                                 f"{len(low_rel)} relevant claim(s) found only on low-reliability sources "
                                 "(data brokers / user-generated); not safe to cite.")
        hallucinated = [e for e in items if not e.grounded]
        msg = "No credible, company-specific public evidence was retrieved."
        if hallucinated:
            msg += f" {len(hallucinated)} extracted claim(s) were rejected because they did not appear in the source text."
        if search_failures:
            msg += f" ({search_failures} search query/queries failed.)"
        return AngleDecision("ABSTAIN_NO_CREDIBLE_EVIDENCE", msg)

    primary = eligible[0]
    supporting = [e for e in eligible[1:] if e.source_id != primary.source_id][:2]
    caveats = []
    if primary.relevance == "INDIRECT":
        caveats.append("Hook is INDIRECT: the link to AP workload is a hypothesis and must be phrased as a question.")
        if primary.tier != "A" and not supporting:
            caveats.append("Single non-primary source for an indirect signal; verify before sending.")
    if primary.recency == "undated":
        caveats.append("Primary source has no publication date; confirm it is current.")
    if primary.recency == "aging":
        caveats.append("Primary signal is 1–2 years old.")
    if identity.prospect_status != "CONFIRMED":
        caveats.append(f"Prospect identity status: {identity.prospect_status}. Email avoids personal claims.")
    if not identity.persona_fit:
        caveats.append("Prospect's role is outside the finance buyer persona; consider redirecting to finance.")
    if conflicts:
        caveats.append(f"Conflicting facts excluded from the email: {', '.join(sorted(conflicts))}.")
    if search_failures:
        caveats.append(f"{search_failures} search query/queries failed; research may be incomplete.")
    outcome = "DRAFT_WITH_CAVEATS" if caveats else "DRAFT_READY"
    return AngleDecision(outcome, f"Strongest eligible signal: {primary.id} ({primary.signal_type}, "
                                  f"{primary.relevance}, tier {primary.tier}, {primary.recency}).",
                         primary, supporting, caveats)


# ---------------------------------------------------------------- draft checks
BANNED_PHRASES = ["hope this finds you well", "i hope you're doing well", "synergy", "game-changer",
                  "revolutionize", "touch base", "circle back", "guaranteed", "100%"]
MAX_BODY_WORDS = 150


def deterministic_draft_checks(draft, allowed_ids: set[str], evidence_text: str, conflicts: dict) -> list[str]:
    problems = []
    words = len(draft.body.split())
    if words > MAX_BODY_WORDS:
        problems.append(f"Body is {words} words (limit {MAX_BODY_WORDS}).")
    low = draft.body.lower()
    for p in BANNED_PHRASES:
        if p in low:
            problems.append(f"Contains banned phrase: '{p}'.")
    for c in draft.claims:
        if c.kind == "fact":
            if not c.evidence_ids:
                problems.append(f"Factual sentence has no evidence citation: \"{c.sentence[:80]}\"")
            bad = [i for i in c.evidence_ids if i not in allowed_ids]
            if bad:
                problems.append(f"Cites evidence not selected for use ({', '.join(bad)}): \"{c.sentence[:80]}\"")
    ev = norm(evidence_text)
    body_wo_asks = re.sub(r"\b\d{1,2}\s?-?\s?(minute|min|mins)\b", " ", draft.body, flags=re.I)
    ev_flat = ev.replace(",", "")
    for num in set(re.findall(r"(?<![\w])[$€£₹]?\d[\d,\.]*%?", body_wo_asks)):
        n = norm(num).rstrip(".").replace(",", "")
        if n and not re.search(rf"(?<![\d.]){re.escape(n)}(?![\d])", ev_flat):
            problems.append(f"Number '{num}' does not appear in the selected evidence.")
    for key, vals in conflicts.items():
        for v in vals:
            if len(v["value"]) > 3 and v["value"].lower() in low:
                problems.append(f"Mentions a conflicted fact value ({key}: {v['value']}).")
    if "http" in low:
        problems.append("Contains a URL.")
    return problems
