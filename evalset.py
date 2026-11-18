"""Golden evaluation set: real, public prospects with the outcome the decision policy SHOULD reach.

These are inputs + expectations only — every evaluation run does live research. An outcome is a pass
if it is in the `expect` set (some cases legitimately allow two outcomes, e.g. a draft that may or may
not need caveats depending on what is published that day)."""

DRAFT = {"DRAFT_READY", "DRAFT_WITH_CAVEATS"}

EVAL_CASES = [
    {"id": "happy-cfo", "prospect_name": "Mark Fleming", "company_name": "Beca", "prospect_role": "CFO",
     "company_website": "https://www.beca.com", "expect": DRAFT,
     "tests": "Happy path: verified CFO, fresh public finance-transformation signal (Oracle Fusion, Oct 2026)."},
    {"id": "happy-no-site", "prospect_name": "Mark Fleming", "company_name": "Beca", "prospect_role": "CFO",
     "company_website": "", "expect": DRAFT | {"NEEDS_INPUT_AMBIGUOUS_IDENTITY"},
     "tests": "Same prospect without a website: a namesake 'Mark Fleming' is CFO of Ventia — must not confuse them."},
    {"id": "wrong-role", "prospect_name": "Mark Fleming", "company_name": "Beca", "prospect_role": "Head of Marketing",
     "company_website": "https://www.beca.com", "expect": {"DRAFT_WITH_CAVEATS", "ABSTAIN_LOW_RELEVANCE"},
     "tests": "Rep entered the wrong role: must flag ROLE_MISMATCH / persona and never assert the wrong title."},
    {"id": "non-buyer", "prospect_name": "Bhavik Chheda", "company_name": "JSW One Platforms",
     "prospect_role": "Business Head", "company_website": "", "expect": {"DRAFT_WITH_CAVEATS", "ABSTAIN_LOW_RELEVANCE"},
     "tests": "Real person, wrong buyer (sales leader): draft only on a DIRECT signal, with a redirect-to-finance caveat."},
    {"id": "ambiguous-company", "prospect_name": "Sam Lee", "company_name": "Apex Solutions", "prospect_role": "Controller",
     "company_website": "", "expect": {"NEEDS_INPUT_AMBIGUOUS_IDENTITY"},
     "tests": "Several companies share the name and the person can't be placed: stop and ask, don't guess."},
    {"id": "low-relevance", "prospect_name": "Varun Garg", "company_name": "Revspot", "prospect_role": "Co-founder",
     "company_website": "", "expect": {"ABSTAIN_LOW_RELEVANCE"},
     "tests": "Real company, real news (Series A, product) — but nothing that justifies an AP pitch to a founder."},
    {"id": "no-evidence", "prospect_name": "Rohan Vexler", "company_name": "Quillmarsh Ledgerworks", "prospect_role": "CFO",
     "company_website": "", "expect": {"ABSTAIN_NO_CREDIBLE_EVIDENCE", "NEEDS_INPUT_AMBIGUOUS_IDENTITY"},
     "tests": "Invented prospect: nothing credible exists, so nothing may be drafted."},
]
