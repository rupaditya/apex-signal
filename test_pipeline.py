"""End-to-end pipeline tests on LABELLED FIXTURES (see tests/fixtures.py).
They verify the decision policy, persistence and failure handling — not live research."""
import copy

import pytest

from apex_signal.db import RunStore
from apex_signal.pipeline import Pipeline, ProspectInput
from tests import fixtures as fx


def run(tmp_path, pages, llm_replies, inp, fail_search=False):
    store = RunStore(str(tmp_path / "t.db"))
    events = []
    p = Pipeline(store, fx.FixtureSearch(pages, fail=fail_search), fx.FixtureLLM(copy.deepcopy(llm_replies)),
                 today=fx.TODAY, on_event=lambda *a: events.append(a[:3]))
    rid = p.run(inp)
    return store.get_run(rid), store.get_stages(rid), events


HAPPY_IN = ProspectInput("Priya Shah", "Northwind Freight", "CFO", "https://northwindfreight.example")


def test_happy_path_produces_validated_draft(tmp_path):
    r, stages, events = run(tmp_path, fx.HAPPY_PAGES, fx.HAPPY_LLM, HAPPY_IN)
    assert r["status"] == "COMPLETED", r["error"]
    assert r["outcome"] == "DRAFT_READY", r["outcome_reason"]
    assert r["review_status"] == "PENDING_REVIEW"
    assert "NetSuite" in r["draft_body"]
    assert [s["status"] for s in stages] == ["DONE"] * 8
    ev = stages[3]["output"]["evidence"]
    # ERP change (DIRECT, tier A, fresh) beats the AP job posting (DIRECT, tier B)
    assert stages[4]["output"]["primary"] == "E1"
    # "40 warehouses" observation is credible but irrelevant -> excluded with a reason
    other = next(e for e in ev if e["signal_type"] == "other")
    assert not other["eligible"] and any("relevance" in x for x in other["exclusions"])
    # events were emitted RUNNING then DONE for each stage, in order
    assert events[0] == (1, "Identify company & prospect", "RUNNING")
    assert events[-1][0] == 8 and events[-1][2] == "DONE"


def test_no_credible_information(tmp_path):
    llm = {"identity": [{"company_candidates": [], "person_matches": []}]}
    r, stages, _ = run(tmp_path, [], llm, ProspectInput("Zed Quorn", "Qwxyz Holdings", "CFO"))
    assert r["outcome"] == "ABSTAIN_NO_CREDIBLE_EVIDENCE"
    assert r["draft_body"] is None
    assert stages[5]["status"] == "SKIPPED" and stages[6]["status"] == "SKIPPED"


def test_signal_without_product_relevance_abstains(tmp_path):
    r, stages, _ = run(tmp_path, fx.LOWREL_PAGES, fx.LOWREL_LLM,
                       ProspectInput("Arjun Mehta", "Brightvale Foods", "CFO"))
    assert r["outcome"] == "ABSTAIN_LOW_RELEVANCE", r["outcome_reason"]
    assert "funding" in r["outcome_reason"]
    assert r["draft_body"] is None


def test_ambiguous_company_needs_input_and_skips_research(tmp_path):
    r, stages, _ = run(tmp_path, fx.AMBIG_PAGES, fx.AMBIG_LLM, ProspectInput("Sam Lee", "Apex Solutions", "Controller"))
    assert r["outcome"] == "NEEDS_INPUT_AMBIGUOUS_IDENTITY"
    assert "2 different organisations" in r["outcome_reason"]
    assert all(s["status"] == "SKIPPED" for s in stages[1:])


def test_conflicting_facts_block_dependent_hook(tmp_path):
    r, stages, _ = run(tmp_path, fx.CONFLICT_PAGES, fx.CONFLICT_LLM, ProspectInput("Dana Ruiz", "Coastal Brands", "Controller"))
    assert r["outcome"] == "ABSTAIN_CONFLICTING_EVIDENCE", r["outcome_reason"]
    assert "cfo_name" in stages[3]["output"]["conflicts"]


def test_ambiguous_person_found_only_elsewhere(tmp_path):
    pages = [{"match": ["priya shah"], "url": "https://www.linkedin.com/in/ps-other", "title": "Priya Shah - CFO - Contoso Bank",
              "content": "Priya Shah, CFO at Contoso Bank, Singapore."}] + fx.HAPPY_PAGES[1:]
    llm = copy.deepcopy(fx.HAPPY_LLM)
    llm["identity"] = [{"company_candidates": [], "person_matches": [
        {"name": "Priya Shah", "employer": "Contoso Bank", "title": "CFO", "is_current": True, "source_id": "S1",
         "quote": "Priya Shah, CFO at Contoso Bank, Singapore."}]}]
    r, _, _ = run(tmp_path, pages, llm, HAPPY_IN)
    assert r["outcome"] == "NEEDS_INPUT_AMBIGUOUS_IDENTITY"
    assert "Contoso" in r["outcome_reason"] or "contoso" in r["outcome_reason"]


def test_hallucinated_quote_is_rejected(tmp_path):
    llm = copy.deepcopy(fx.HAPPY_LLM)
    llm["extract"][0]["observations"] = [
        {"source_id": "S3", "quote": "Northwind Freight reported a backlog of 12,000 unpaid supplier invoices",
         "observation": "Invoice backlog.", "signal_type": "stated_ap_pain", "subject": "company"}]
    r, stages, _ = run(tmp_path, fx.HAPPY_PAGES, llm, HAPPY_IN)
    ev = stages[3]["output"]["evidence"][0]
    assert ev["grounded"] is False and not ev["eligible"]
    assert r["outcome"] == "ABSTAIN_NO_CREDIBLE_EVIDENCE"


def test_unsupported_number_triggers_repair(tmp_path):
    llm = copy.deepcopy(fx.HAPPY_LLM)
    bad = copy.deepcopy(llm["draft"][0])
    bad["body"] = bad["body"].replace("Worth a 20-minute look", "Teams like yours cut costs 63%. Worth a look")
    llm["draft"] = [bad]
    llm["repair"] = [fx.HAPPY_LLM["draft"][0]]
    r, stages, _ = run(tmp_path, fx.HAPPY_PAGES, llm, HAPPY_IN)
    attempts = stages[6]["output"]["attempts"]
    assert len(attempts) == 2
    assert any("63%" in p for p in attempts[0]["problems"])
    assert attempts[1]["problems"] == []
    assert "63%" not in r["draft_body"]


def test_search_outage_fails_run_cleanly(tmp_path):
    r, stages, _ = run(tmp_path, fx.HAPPY_PAGES, fx.HAPPY_LLM, HAPPY_IN, fail_search=True)
    assert r["status"] == "FAILED"
    assert "searches failed" in r["error"]
    assert stages[0]["status"] == "FAILED"
    assert all(s["status"] == "SKIPPED" for s in stages[1:])


def test_invalid_model_json_retries_then_fails(tmp_path):
    llm = copy.deepcopy(fx.HAPPY_LLM)
    llm["identity"] = ["not json at all", "{still: broken"]
    r, stages, _ = run(tmp_path, fx.HAPPY_PAGES, llm, HAPPY_IN)
    assert r["status"] == "FAILED"
    assert "invalid JSON" in r["error"]


def test_invalid_model_json_recovers_on_retry(tmp_path):
    llm = copy.deepcopy(fx.HAPPY_LLM)
    llm["identity"] = ["oops", fx.HAPPY_LLM["identity"][0]]
    r, _, _ = run(tmp_path, fx.HAPPY_PAGES, llm, HAPPY_IN)
    assert r["status"] == "COMPLETED" and r["outcome"] == "DRAFT_READY"


def test_unverified_prospect_gets_caveats(tmp_path):
    llm = copy.deepcopy(fx.HAPPY_LLM)
    llm["identity"][0]["person_matches"] = []
    r, _, _ = run(tmp_path, fx.HAPPY_PAGES, llm, HAPPY_IN)
    assert r["outcome"] == "DRAFT_WITH_CAVEATS"
    assert "NOT_FOUND" in r["outcome_reason"]


def test_relevant_claim_only_on_data_broker_is_not_used(tmp_path):
    pages = fx.LOWREL_PAGES[:1] + [{"match": ["news"], "url": "https://www.zoominfo.com/c/brightvale/1",
        "title": "Brightvale Foods company profile", "date": "2026-09-01",
        "content": "Brightvale Foods is migrating its finance stack to SAP S/4HANA this year."}]
    llm = copy.deepcopy(fx.LOWREL_LLM)
    llm["extract"] = [{"observations": [{"source_id": "S2", "quote": "Brightvale Foods is migrating its finance stack to SAP S/4HANA",
        "observation": "Migrating to SAP.", "signal_type": "erp_or_finance_system_change", "subject": "company"}], "facts": []}]
    r, stages, _ = run(tmp_path, pages, llm, ProspectInput("Arjun Mehta", "Brightvale Foods", "CFO"))
    assert r["outcome"] == "ABSTAIN_CONFLICTING_EVIDENCE"
    assert "low-reliability" in r["outcome_reason"]
