"""The 8-stage workflow. Each stage is persisted (RUNNING -> DONE/FAILED/SKIPPED)
before and after it executes, and an on_event callback lets the UI render
real progress as the backend reports it."""
from __future__ import annotations

import json
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from typing import Callable, Optional

from . import policy
from .db import RunStore
from .llm import BaseLLM, LLMError
from .schemas import (AngleRationale, EmailDraft, ExtractionResult, IdentityExtraction,
                      ValidationVerdict)
from .search import SearchError, SearchResult, domain_of

SELLER = ("Apex (fictional) sells accounts-payable automation for mid-market finance teams: invoice capture "
          "and processing, 2/3-way PO matching, approval routing, and exception management.")

STAGES = [
    (1, "Identify company & prospect"),
    (2, "Research public sources"),
    (3, "Extract evidence"),
    (4, "Evaluate credibility & relevance"),
    (5, "Select outreach angle"),
    (6, "Draft email"),
    (7, "Validate draft against evidence"),
    (8, "Queue for human review"),
]

Event = Callable[[int, str, str, str, object], None]


@dataclass
class ProspectInput:
    prospect_name: str
    company_name: str
    prospect_role: str
    company_website: Optional[str] = None


class StageFailed(Exception):
    pass


class Pipeline:
    def __init__(self, store: RunStore, search, llm: BaseLLM, today: Optional[date] = None,
                 max_sources: int = 14, on_event: Optional[Event] = None):
        self.store, self.search, self.llm = store, search, llm
        self.today = today or date.today()
        self.max_sources = max_sources
        self.on_event = on_event or (lambda *a: None)

    # ------------------------------------------------------------ helpers
    def _start(self, run_id, no):
        name = dict(STAGES)[no]
        self.store.stage_started(run_id, no, name)
        self.on_event(no, name, "RUNNING", "", None)

    def _finish(self, run_id, no, status, summary, output=None):
        name = dict(STAGES)[no]
        self.store.stage_finished(run_id, no, status, summary, output)
        self.on_event(no, name, status, summary, output)

    def _skip_rest(self, run_id, from_no, why):
        for no, _ in STAGES:
            if no >= from_no:
                self._start(run_id, no)
                self._finish(run_id, no, "SKIPPED", why)

    def _run_queries(self, queries: list[dict]) -> tuple[list[SearchResult], list[dict]]:
        results, log = [], []

        def one(q):
            try:
                res = self.search.search(q["q"], max_results=q.get("n", 5), news=q.get("news", False),
                                         include_domains=q.get("domains"))
                return q, res, None
            except SearchError as e:
                return q, [], str(e)
            except Exception as e:  # never let one query kill the run
                return q, [], f"Unexpected search error: {e}"

        with ThreadPoolExecutor(max_workers=4) as ex:
            for q, res, err in ex.map(one, queries):
                log.append({"query": q["q"], "news": q.get("news", False), "results": len(res), "error": err})
                results.extend(res)
        return results, log

    @staticmethod
    def _dedupe(results: list[SearchResult]) -> list[SearchResult]:
        seen, out = set(), []
        for r in results:
            key = r.url.split("#")[0].rstrip("/")
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        return out

    @staticmethod
    def _sources_block(sources: dict) -> str:
        parts = []
        for sid, s in sources.items():
            parts.append(f"[{sid}] {s['title']}\nURL: {s['url']}\nPublished: {s.get('published_date') or 'unknown'}\n"
                         f"TEXT: {s['content'][:3500]}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------ main
    def run(self, inp: ProspectInput) -> str:
        providers = f"search={getattr(self.search, 'name', '?')}; llm={self.llm.name}:{self.llm.model}"
        run_id = self.store.create_run(inp.prospect_name, inp.company_name, inp.prospect_role,
                                       inp.company_website, providers)
        try:
            self._execute(run_id, inp)
        except StageFailed as e:
            self.store.update_run(run_id, status="FAILED", error=str(e), review_status="N/A")
        except Exception as e:  # last-resort guard; keep the trace for debugging
            self.store.update_run(run_id, status="FAILED", review_status="N/A",
                                  error=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")
        fb = getattr(self.llm, "events", None)
        if fb:  # record provider fallbacks (e.g. Gemini quota hit -> Groq) for traceability
            self.store.update_run(run_id, providers=providers + " | fallbacks: " + "; ".join(fb))
            fb.clear()
        return run_id

    def _fail(self, run_id, no, msg, output=None):
        self._finish(run_id, no, "FAILED", msg, output)
        for n, _ in STAGES:
            if n > no:
                self.store.stage_started(run_id, n, dict(STAGES)[n])
                self.store.stage_finished(run_id, n, "SKIPPED", "Not run: earlier stage failed.", None)
                self.on_event(n, dict(STAGES)[n], "SKIPPED", "Not run: earlier stage failed.", None)
        raise StageFailed(f"Stage {no} failed: {msg}")

    def _execute(self, run_id: str, inp: ProspectInput):
        company, person, role = inp.company_name.strip(), inp.prospect_name.strip(), inp.prospect_role.strip()
        site_domain = domain_of(inp.company_website) if inp.company_website else None
        sources: dict[str, dict] = {}

        def register(results):
            for r in results:
                if any(s["url"] == r.url for s in sources.values()):
                    continue
                sid = f"S{len(sources) + 1}"
                sources[sid] = {"id": sid, "url": r.url, "title": r.title, "content": r.content or r.snippet,
                                "published_date": r.published_date, "query": r.query, "domain": r.domain}

        # ---------------- Stage 1: identify
        self._start(run_id, 1)
        id_queries = [
            {"q": f'"{person}" {company}', "n": 6},
            {"q": f'"{person}" {role} {company}', "n": 5},
        ]
        if site_domain:
            id_queries.append({"q": f"{company} about", "n": 3, "domains": [site_domain]})
        else:
            id_queries.append({"q": f"{company} company official website", "n": 6})
        id_results, id_log = self._run_queries(id_queries)
        if all(l["error"] for l in id_log):
            self._fail(run_id, 1, "All identity searches failed: " + id_log[0]["error"], {"queries": id_log})
        register(self._dedupe(id_results))
        if sources:
            try:
                ident_x = self.llm.call_json(
                    "identity",
                    "You identify organisations and people in web search results. Only report what the "
                    "text shows. Quotes must be copied verbatim from the given source text.",
                    f"Company name given by sales rep: {company}\nWebsite given: {inp.company_website or 'none'}\n"
                    f"Prospect: {person}, role given: {role}\n\nSOURCES:\n{self._sources_block(sources)}\n\n"
                    "1) company_candidates: distinct real organisations whose name matches the company name, with their own "
                    "website domain. Do NOT list news sites, directories, job boards or social networks, and do not list "
                    "the same organisation twice for regional sites or subsidiaries.\n"
                    f"2) person_matches: every source that mentions a person named {person} with employer and title.",
                    IdentityExtraction)
            except LLMError as e:
                self._fail(run_id, 1, str(e), {"queries": id_log})
        else:
            ident_x = IdentityExtraction()
        identity = policy.decide_identity(ident_x, sources, company, person, role, inp.company_website)
        self._finish(run_id, 1, "DONE",
                     f"Company: {identity.company_status} · Prospect: {identity.prospect_status}",
                     {"queries": id_log, "extraction": ident_x.model_dump(), "decision": identity.to_dict()})

        if identity.blocking:
            angle = policy.select_angle([], identity, {})
            self._skip_rest(run_id, 2, "Not run: identity must be resolved by the rep first.")
            self.store.update_run(run_id, status="COMPLETED", outcome=angle.outcome, outcome_reason=angle.reason,
                                  review_status="N/A")
            return

        # ---------------- Stage 2: research
        self._start(run_id, 2)
        cq = f'"{company}"'
        queries = [
            {"q": f"{company} news", "news": True, "n": 6},
            {"q": f"{cq} accounts payable OR finance operations job", "n": 5},
            {"q": f"{cq} ERP OR SAP OR NetSuite OR Oracle OR Workday finance implementation", "n": 5},
            {"q": f"{cq} appoints CFO OR controller OR \"finance director\"", "news": True, "n": 5},
            {"q": f"{cq} acquisition OR acquires OR expansion OR new facility", "news": True, "n": 5},
            {"q": f'"{person}" {company} interview OR podcast OR finance', "n": 4},
        ]
        if identity.company_domain:
            queries.append({"q": f"{company} press release news", "n": 5, "domains": [identity.company_domain]})
        res, log = self._run_queries(queries)
        failures = sum(1 for l in log if l["error"])
        if failures == len(log):
            self._fail(run_id, 2, "All research searches failed (search provider unavailable).", {"queries": log})
        before = len(sources)
        register(self._dedupe(res))
        self._finish(run_id, 2, "DONE",
                     f"{len(queries)} queries · {len(sources) - before} new sources · {failures} failed",
                     {"queries": log, "sources": [{k: v for k, v in s.items() if k != "content"} | {"chars": len(s["content"])}
                                                  for s in sources.values()]})

        # ---------------- Stage 3: extract
        self._start(run_id, 3)
        ordered = sorted(sources.values(), key=lambda s: (policy.source_tier(s["url"], identity.company_domain) == "C",
                                                          not policy.mentions_company(s["content"] + s["title"], company, identity.company_domain)))
        picked = {s["id"]: s for s in ordered[: self.max_sources]}
        observations, facts, errors = [], [], []
        if picked:
            ids = list(picked)
            for i in range(0, len(ids), 7):
                batch = {k: picked[k] for k in ids[i:i + 7]}
                try:
                    x = self.llm.call_json(
                        "extract",
                        "You are a B2B research analyst. Extract ONLY what the source text states. Every quote must "
                        "be copied verbatim from that source's TEXT (no paraphrase, no ellipsis joining). If a source "
                        "is about a different organisation or person, mark subject other_entity. Do not infer. "
                        "Classify carefully: erp_or_finance_system_change ONLY when the company is changing its OWN "
                        "internal accounting/ERP systems – a product or service the company SELLS (e.g. its platform "
                        "integrating with customers' ERPs) is product_or_marketing. ap_finance_hiring ONLY for jobs "
                        "inside the company's own finance/AP team. IPO filings/listing plans are ipo_or_listing. Plans stated inside a funding announcement (e.g. 'will use the funds to expand') are part of funding, not a separate expansion signal.",
                        f"Target company: {company} ({identity.company_domain or 'domain unknown'})\n"
                        f"Prospect: {person} ({role})\n\nSOURCES:\n{self._sources_block(batch)}\n\n"
                        "Extract (a) observations useful for understanding this company's finance/AP situation or "
                        "any notable recent company event, one per distinct event, and (b) facts for the listed keys "
                        "when explicitly stated.",
                        ExtractionResult)
                    observations += x.observations
                    facts += x.facts
                except LLMError as e:
                    errors.append(str(e))
            if errors and not observations and len(errors) == (len(ids) + 6) // 7:
                self._fail(run_id, 3, "Extraction failed: " + errors[0])
        self._finish(run_id, 3, "DONE",
                     f"{len(observations)} observations · {len(facts)} facts from {len(picked)} sources"
                     + (f" · {len(errors)} batch error(s)" if errors else ""),
                     {"observations": [o.model_dump() for o in observations], "facts": [f.model_dump() for f in facts],
                      "errors": errors})

        # ---------------- Stage 4: evaluate
        self._start(run_id, 4)
        conflicts = policy.detect_conflicts(facts, sources, identity.company_domain)
        items = policy.assess_evidence(observations, sources, company, identity.company_domain, identity,
                                       conflicts, self.today)
        n_ok = sum(e.eligible for e in items)
        self._finish(run_id, 4, "DONE",
                     f"{n_ok}/{len(items)} evidence items eligible · {len(conflicts)} conflicting fact(s)",
                     {"evidence": [e.to_dict() for e in items], "conflicts": conflicts,
                      "sources": {k: {kk: vv for kk, vv in v.items() if kk != "content"} for k, v in sources.items()}})

        # ---------------- Stage 5: angle
        self._start(run_id, 5)
        angle = policy.select_angle(items, identity, conflicts, search_failures=failures)
        rationale = None
        if angle.primary:
            ev = [angle.primary] + angle.supporting
            try:
                rationale = self.llm.call_json(
                    "rationale",
                    "You are a careful sales strategist. Separate what is observed from what is hypothesised. "
                    "Never upgrade a hypothesis into a fact.",
                    f"Seller: {SELLER}\nProspect: {person}, {role} at {company}\n"
                    f"Selected evidence:\n" + "\n".join(f"{e.id} [{e.signal_type}, {e.relevance}] {e.observation} "
                                                         f"(quote: \"{e.quote}\")" for e in ev)
                    + "\n\nExplain the hook, the business implication we suspect, why AP automation is relevant, "
                      "and list the assumptions the evidence does not prove.",
                    AngleRationale)
            except LLMError as e:
                self._fail(run_id, 5, str(e), angle.to_dict())
        out5 = angle.to_dict() | {"rationale": rationale.model_dump() if rationale else None}
        self._finish(run_id, 5, "DONE", policy.OUTCOMES[angle.outcome], out5)

        if not angle.primary:
            self._skip_rest(run_id, 6, "Not run: process abstained (see stage 5).")
            self.store.update_run(run_id, status="COMPLETED", outcome=angle.outcome, outcome_reason=angle.reason,
                                  review_status="N/A")
            return

        # ---------------- Stage 6: draft
        self._start(run_id, 6)
        ev = [angle.primary] + angle.supporting
        allowed_ids = {e.id for e in ev}
        evidence_text = "\n".join(f"{e.quote} {e.observation} {e.published_date or ''}" for e in ev)
        # numbers/claims are checked against the full text of the cited sources, not just the quote
        cited_text = evidence_text + "\n" + "\n".join(sources[e.source_id]["content"] for e in ev)
        first_name = person.split()[0]
        personal_ok = identity.prospect_status == "CONFIRMED"
        draft_prompt = (
            f"Seller: {SELLER}\n"
            f"Recipient: {first_name} ({'verified ' + role if personal_ok else 'role NOT verified – do not mention their title or personal history'}) at {company}\n"
            f"Hook: {rationale.hook_summary}\nHypothesis (NOT verified – phrase as a question): {rationale.hypothesis}\n"
            f"Relevance: {rationale.product_relevance}\n"
            f"ALLOWED EVIDENCE (the only facts you may state):\n"
            + "\n".join(f"{e.id}: {e.observation} | quote: \"{e.quote}\" | date: {e.published_date or 'unknown'}" for e in ev)
            + f"\nDO NOT mention: {', '.join(conflicts) if conflicts else 'n/a'}; any number, customer, metric or "
              "result not in the allowed evidence; any claim about the seller's results.\n\n"
              "Write a plain-text cold email: 60–120 words. "
              f"Start with 'Hi {first_name},' on its own line{'' if personal_ok else ' only if that looks like a real first name, otherwise Hi there,'}. "
              "Subject: under 8 words, about THEIR signal, never a product pitch (bad: 'AP Automation for X'). "
              f"Open with the PRIMARY evidence {angle.primary.id} (supporting evidence may only add detail after it); "
              "subject must also be about the primary signal. No flattery, no 'I hope this finds you well'. Then one question linking "
              "it to AP workload. Do NOT assume their current process is manual, slow or broken – ask, don't assert. "
              "One sentence on what Apex does, then a low-friction ask. Sign off as '[Your name]'. "
              "In `claims`, list every sentence with its kind and the evidence ids it relies on.")
        try:
            draft = self.llm.call_json("draft", "You write short, specific, honest B2B emails that sound human.",
                                       draft_prompt, EmailDraft)
        except LLMError as e:
            self._fail(run_id, 6, str(e))
        self._finish(run_id, 6, "DONE", f"Draft v1: {len(draft.body.split())} words",
                     {"draft": draft.model_dump()})

        # ---------------- Stage 7: validate (+ one repair)
        self._start(run_id, 7)
        attempts = []
        for attempt in range(2):
            problems = policy.deterministic_draft_checks(draft, allowed_ids, cited_text, conflicts)
            try:
                verdict = self.llm.call_json(
                    "validate",
                    "You are a strict fact-checker. Judge each sentence ONLY against the evidence provided.",
                    f"EVIDENCE:\n{cited_text[:6000]}\nSELLER DESCRIPTION (claims about the seller's capabilities are "
                    f"allowed): {SELLER}\n\nEMAIL:\nSubject: {draft.subject}\n{draft.body}\n\n"
                    "For each sentence: supported (stated in evidence), framed_as_hypothesis (clearly a question or "
                    "hedged), neutral (greeting, sign-off, ask, seller description), unsupported (asserts something "
                    "about the prospect/company not in evidence).",
                    ValidationVerdict)
                unsupported = [s for s in verdict.sentences if s.verdict == "unsupported"]
            except LLMError as e:
                verdict, unsupported = None, []
                problems.append(f"LLM fact-check unavailable: {e}")
            for s in unsupported:
                problems.append(f"Unsupported: \"{s.sentence[:100]}\" – {s.reason}")
            attempts.append({"draft": draft.model_dump(), "problems": problems,
                             "verdict": verdict.model_dump() if verdict else None})
            if not problems or attempt == 1:
                break
            try:  # one repair pass
                draft = self.llm.call_json(
                    "repair", "You fix B2B emails so every factual statement is supported by evidence.",
                    draft_prompt + "\n\nYOUR PREVIOUS DRAFT FAILED THESE CHECKS – fix all of them:\n- "
                    + "\n- ".join(problems) + f"\n\nPrevious draft:\n{draft.body}", EmailDraft)
            except LLMError as e:
                problems.append(f"Repair failed: {e}")
                break
        passed = not attempts[-1]["problems"]
        self._finish(run_id, 7, "DONE" if passed else "WARNING",
                     "Passed" + (" after 1 repair" if len(attempts) > 1 else "") if passed
                     else f"{len(attempts[-1]['problems'])} issue(s) remain – flagged for reviewer",
                     {"attempts": attempts, "passed": passed})

        # ---------------- Stage 8: review queue
        self._start(run_id, 8)
        outcome = angle.outcome
        caveats = list(angle.caveats)
        if not passed:
            outcome = "DRAFT_WITH_CAVEATS"
            caveats.append("Validation issues remain; see stage 7.")
        reason = angle.reason + (" Caveats: " + " ".join(caveats) if caveats else "")
        self.store.update_run(run_id, status="COMPLETED", outcome=outcome, outcome_reason=reason,
                              draft_subject=draft.subject, draft_body=draft.body,
                              edited_subject=draft.subject, edited_body=draft.body, review_status="PENDING_REVIEW")
        self._finish(run_id, 8, "DONE", "Awaiting human review – nothing is sent automatically.",
                     {"caveats": caveats, "validation_passed": passed})
