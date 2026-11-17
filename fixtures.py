"""TEST FIXTURES ONLY — synthetic search results and canned model replies.

These are NOT live research and are never used by the app. They let the
automated tests exercise the decision policy deterministically, without
network access or API keys. Company names/domains here are fictional."""
from __future__ import annotations

import json
from datetime import date

from apex_signal.llm import BaseLLM
from apex_signal.search import SearchError, SearchResult

TODAY = date(2026, 11, 17)


class FixtureSearch:
    name = "fixture-search"

    def __init__(self, pages: list[dict], fail: bool = False):
        self.pages, self.fail, self.calls = pages, fail, []

    def search(self, query, max_results=5, news=False, include_domains=None):
        self.calls.append(query)
        if self.fail:
            raise SearchError("fixture: provider down (HTTP 503)")
        q = query.lower()
        out = []
        for p in self.pages:
            if any(k in q for k in p["match"]):
                out.append(SearchResult(url=p["url"], title=p["title"], snippet=p["content"][:300],
                                        content=p["content"], published_date=p.get("date"), query=query,
                                        provider=self.name))
        return out[:max_results]


class FixtureLLM(BaseLLM):
    name = "fixture-llm"
    model = "canned"

    def __init__(self, replies: dict[str, list]):
        # replies[task] is a list consumed in order; last one repeats
        self.replies = {k: list(v) for k, v in replies.items()}
        self.calls = []

    def complete(self, system, user, task):
        self.calls.append(task)
        seq = self.replies.get(task)
        if not seq:
            raise AssertionError(f"no fixture reply for task {task}")
        r = seq.pop(0) if len(seq) > 1 else seq[0]
        return r if isinstance(r, str) else json.dumps(r)


# ---------------------------------------------------------------- happy path
NW = "northwindfreight.example"
HAPPY_PAGES = [
    {"match": ["priya shah"], "url": "https://www.linkedin.com/in/priya-shah-nw", "title": "Priya Shah - CFO - Northwind Freight",
     "content": "Priya Shah. Chief Financial Officer at Northwind Freight. Mumbai, India. Leads finance, treasury and accounting."},
    {"match": ["about"], "url": f"https://{NW}/about", "title": "About Northwind Freight",
     "content": "Northwind Freight is a mid-market logistics provider operating 40 warehouses across India."},
    {"match": ["erp", "news"], "url": "https://www.businesswire.com/news/northwind-netsuite", "title": "Northwind Freight selects NetSuite",
     "date": "2026-09-14",
     "content": "Northwind Freight today announced it has selected Oracle NetSuite to replace its legacy accounting system, "
                "with go-live planned for the first quarter of 2027 across all 40 warehouses."},
    {"match": ["accounts payable"], "url": "https://jobs.lever.co/northwindfreight/ap-specialist", "title": "Accounts Payable Specialist - Northwind Freight",
     "date": "2026-10-30",
     "content": "Northwind Freight is hiring an Accounts Payable Specialist to process high volumes of vendor invoices "
                "and resolve PO matching exceptions."},
]
HAPPY_LLM = {
    "identity": [{"company_candidates": [{"name": "Northwind Freight", "domain": NW, "description": "Logistics, India", "source_ids": ["S2"]}],
                  "person_matches": [{"name": "Priya Shah", "employer": "Northwind Freight", "title": "Chief Financial Officer",
                                      "is_current": True, "source_id": "S1", "quote": "Priya Shah. Chief Financial Officer at Northwind Freight."}]}],
    "extract": [{"observations": [
        {"source_id": "S3", "quote": "has selected Oracle NetSuite to replace its legacy accounting system",
         "observation": "Northwind Freight selected NetSuite to replace its legacy accounting system, go-live Q1 2027.",
         "signal_type": "erp_or_finance_system_change", "subject": "company"},
        {"source_id": "S4", "quote": "hiring an Accounts Payable Specialist to process high volumes of vendor invoices",
         "observation": "Northwind is hiring an AP Specialist to process high invoice volumes.",
         "signal_type": "ap_finance_hiring", "subject": "company"},
        {"source_id": "S2", "quote": "operating 40 warehouses across India",
         "observation": "Northwind operates 40 warehouses in India.", "signal_type": "other", "subject": "company"},
    ], "facts": [{"source_id": "S3", "key": "erp_system", "value": "Oracle NetSuite", "quote": "has selected Oracle NetSuite"}]}],
    "rationale": [{"hook_summary": "NetSuite migration announced Sept 2026",
                   "hypothesis": "AP workflows will be redesigned during the migration",
                   "product_relevance": "Migration is a natural moment to automate invoice capture and PO matching",
                   "assumptions": ["AP is currently manual", "Prospect owns the AP process decision"]}],
    "draft": [{"subject": "NetSuite go-live and AP",
               "body": "Hi Priya,\n\nSaw that Northwind Freight selected NetSuite to replace its legacy accounting system, with go-live planned across all 40 warehouses. "
                       "Are you also rethinking how vendor invoices get captured and matched to POs before cutover?\n\n"
                       "Apex automates invoice processing, PO matching and approval routing for mid-market finance teams.\n\n"
                       "Worth a 20-minute look ahead of the migration?\n\n[Your name]",
               "claims": [{"sentence": "Saw that Northwind Freight selected NetSuite...", "kind": "fact", "evidence_ids": ["E1"]},
                          {"sentence": "Are you also rethinking...", "kind": "question", "evidence_ids": []},
                          {"sentence": "Apex automates...", "kind": "offer", "evidence_ids": []}]}],
    "validate": [{"sentences": [{"sentence": "Saw that Northwind...", "verdict": "supported"},
                                {"sentence": "Are you also rethinking...", "verdict": "framed_as_hypothesis"}]}],
}

# ---------------------------------------------------------------- low relevance
LOWREL_PAGES = [
    {"match": ["arjun mehta"], "url": "https://www.linkedin.com/in/arjun-mehta-bv", "title": "Arjun Mehta - CFO - Brightvale Foods",
     "content": "Arjun Mehta, CFO at Brightvale Foods. Finance leader."},
    {"match": ["news", "about"], "url": "https://techcrunch.com/2026/10/02/brightvale-series-b", "title": "Brightvale Foods raises $40M Series B",
     "date": "2026-10-02",
     "content": "Brightvale Foods has raised a $40M Series B led by Peak Partners to expand its plant-based snack range."},
    {"match": ["news"], "url": "https://brightvale.example/blog/new-flavours", "title": "New flavours launch",
     "date": "2026-08-01", "content": "Brightvale Foods launches three new flavours of its protein crisps this summer."},
]
LOWREL_LLM = {
    "identity": [{"company_candidates": [{"name": "Brightvale Foods", "domain": "brightvale.example", "description": "Snacks"}],
                  "person_matches": [{"name": "Arjun Mehta", "employer": "Brightvale Foods", "title": "CFO", "is_current": True,
                                      "source_id": "S1", "quote": "Arjun Mehta, CFO at Brightvale Foods."}]}],
    "extract": [{"observations": [
        {"source_id": "S2", "quote": "Brightvale Foods has raised a $40M Series B led by Peak Partners",
         "observation": "Raised $40M Series B.", "signal_type": "funding", "subject": "company"},
        {"source_id": "S3", "quote": "launches three new flavours of its protein crisps",
         "observation": "Launched new flavours.", "signal_type": "product_or_marketing", "subject": "company"},
    ], "facts": []}],
}

# ---------------------------------------------------------------- ambiguous company
AMBIG_PAGES = [
    {"match": ["official website", "sam lee"], "url": "https://www.apexsolutions-us.example/", "title": "Apex Solutions | HVAC Services Texas",
     "content": "Apex Solutions provides HVAC services in Texas."},
    {"match": ["official website", "sam lee"], "url": "https://apexsolutions.example.in/", "title": "Apex Solutions Pvt Ltd - IT consulting",
     "content": "Apex Solutions Pvt Ltd is an IT consulting firm in Pune."},
]
AMBIG_LLM = {
    "identity": [{"company_candidates": [
        {"name": "Apex Solutions", "domain": "apexsolutions-us.example", "description": "HVAC services, Texas"},
        {"name": "Apex Solutions Pvt Ltd", "domain": "apexsolutions.example.in", "description": "IT consulting, Pune"}],
        "person_matches": []}],
}

# ---------------------------------------------------------------- conflicting
CONFLICT_PAGES = [
    {"match": ["dana ruiz"], "url": "https://www.linkedin.com/in/dana-ruiz", "title": "Dana Ruiz - Controller - Coastal Brands",
     "content": "Dana Ruiz. Controller at Coastal Brands."},
    {"match": ["cfo", "news"], "url": "https://www.prnewswire.com/coastal-cfo", "title": "Coastal Brands names Mark Ito CFO",
     "date": "2026-07-01", "content": "Coastal Brands today named Mark Ito as Chief Financial Officer, effective immediately."},
    {"match": ["cfo", "news"], "url": "https://www.retaildive.com/coastal-cfo-exit", "title": "Coastal Brands CFO search continues",
     "date": "2026-07-20", "content": "Coastal Brands said interim CFO Lena Park will remain in the role while the search continues."},
]
CONFLICT_LLM = {
    "identity": [{"company_candidates": [{"name": "Coastal Brands", "domain": "coastalbrands.example"}],
                  "person_matches": [{"name": "Dana Ruiz", "employer": "Coastal Brands", "title": "Controller", "is_current": True,
                                      "source_id": "S1", "quote": "Dana Ruiz. Controller at Coastal Brands."}]}],
    "extract": [{"observations": [
        {"source_id": "S2", "quote": "Coastal Brands today named Mark Ito as Chief Financial Officer",
         "observation": "Named Mark Ito CFO.", "signal_type": "finance_leadership_change", "subject": "company"},
        {"source_id": "S3", "quote": "interim CFO Lena Park will remain in the role while the search continues",
         "observation": "Interim CFO Lena Park remains; search continues.", "signal_type": "finance_leadership_change", "subject": "company"},
    ], "facts": [
        {"source_id": "S2", "key": "cfo_name", "value": "Mark Ito", "quote": "named Mark Ito as Chief Financial Officer"},
        {"source_id": "S3", "key": "cfo_name", "value": "Lena Park", "quote": "interim CFO Lena Park will remain in the role"},
    ]}],
}
