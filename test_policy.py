from datetime import date

from apex_signal import policy


def test_source_tiers():
    assert policy.source_tier("https://www.reuters.com/x", None) == "A"
    assert policy.source_tier("https://acme.com/news", "acme.com") == "A"
    assert policy.source_tier("https://www.zoominfo.com/p/x", None) == "C"
    assert policy.source_tier("https://someregionalpaper.com/x", None) == "B"


def test_recency():
    t = date(2026, 11, 17)
    assert policy.recency_label(date(2026, 6, 1), t) == "fresh"
    assert policy.recency_label(date(2025, 6, 1), t) == "aging"
    assert policy.recency_label(date(2023, 1, 1), t) == "stale"
    assert policy.recency_label(None, t) == "undated"


def test_grounding():
    content = "Acme Corp today announced it will migrate to SAP S/4HANA by 2027, the company said."
    assert policy.is_grounded("will migrate to SAP S/4HANA by 2027", content)
    assert policy.is_grounded("Acme Corp today announced it will migrate to SAP S/4HANA", content)
    assert not policy.is_grounded("Acme has a backlog of 5,000 invoices", content)


def test_roles():
    assert policy.roles_compatible("CFO", "Chief Financial Officer")
    assert policy.roles_compatible("Controller", "Corporate Controller")
    assert not policy.roles_compatible("CFO", "Head of Marketing")
    assert policy.in_buyer_persona("Accounts Payable Manager")
    assert not policy.in_buyer_persona("VP Sales")


def test_company_core():
    assert policy.company_core("Acme Holdings Pvt. Ltd.") == "acme"


def test_number_check_uses_whole_numbers_and_allows_meeting_length():
    from apex_signal.schemas import EmailDraft
    d = EmailDraft(subject="x", body="You have 20 sites. Worth a 15-minute call?", claims=[])
    probs = policy.deterministic_draft_checks(d, set(), "published 2026-09-14, 40 warehouses", {})
    assert any("'20'" in p for p in probs)
    assert not any("15" in p for p in probs)
