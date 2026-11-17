"""Structured contracts for every LLM call. The LLM only extracts and
writes; all decisions are made by deterministic rules in policy.py."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

SignalType = Literal[
    "ap_finance_hiring",            # job postings for AP / finance-ops roles
    "erp_or_finance_system_change",  # the company changing ITS OWN internal ERP / finance systems (not products it sells)
    "ipo_or_listing",               # IPO filing / listing preparation (public-company controls)
    "finance_transformation",       # shared services, automation, close/AP process initiatives
    "stated_ap_pain",               # public mention of invoice backlog, late supplier payments, payables control issues
    "finance_leadership_change",    # new CFO / controller / VP finance
    "acquisition_or_merger",
    "expansion",                    # new entities, sites, geographies
    "rapid_growth",                 # headcount / revenue / volume growth
    "prospect_statement",           # interview, talk, post by the prospect on a finance topic
    "funding",
    "product_or_marketing",
    "award_or_pr",
    "other",
]

FactKey = Literal[
    "prospect_current_employer", "prospect_current_title", "company_headquarters",
    "employee_count", "cfo_name", "ownership_status", "erp_system",
]


# ---- Stage 1 -------------------------------------------------------------
class CompanyCandidate(BaseModel):
    name: str
    domain: Optional[str] = Field(None, description="Primary website domain, e.g. acme.com")
    description: str = Field("", description="One line: industry and location")
    source_ids: list[str] = Field(default_factory=list)


class PersonMatch(BaseModel):
    name: str
    employer: str
    title: str
    is_current: bool = Field(True, description="False if the source says this role is past/former")
    source_id: str
    quote: str = Field(..., description="Exact text from the source that shows the name/employer/title")


class IdentityExtraction(BaseModel):
    company_candidates: list[CompanyCandidate] = Field(
        default_factory=list,
        description="DISTINCT real-world organisations that plausibly match the company name. "
                    "List separate entries only if they are clearly different organisations.")
    person_matches: list[PersonMatch] = Field(default_factory=list)


# ---- Stage 3 -------------------------------------------------------------
class Observation(BaseModel):
    source_id: str
    quote: str = Field(..., description="Verbatim sentence or phrase copied from the source text")
    observation: str = Field(..., description="Neutral one-sentence statement of what the source says")
    signal_type: SignalType
    subject: Literal["prospect", "company", "other_entity", "unclear"]
    event_date: Optional[str] = Field(None, description="Date of the event if stated in the text (YYYY-MM or YYYY-MM-DD)")


class Fact(BaseModel):
    source_id: str
    key: FactKey
    value: str
    quote: str


class ExtractionResult(BaseModel):
    observations: list[Observation] = Field(default_factory=list)
    facts: list[Fact] = Field(default_factory=list)


# ---- Stage 5 -------------------------------------------------------------
class AngleRationale(BaseModel):
    hook_summary: str = Field(..., description="One line: the observed signal the email will open with")
    hypothesis: str = Field(..., description="The business implication we SUSPECT (not verified)")
    product_relevance: str = Field(..., description="How AP automation relates to that implication")
    assumptions: list[str] = Field(default_factory=list,
                                   description="Things we are assuming that the evidence does NOT show")


# ---- Stage 6 -------------------------------------------------------------
class EmailClaim(BaseModel):
    sentence: str
    kind: Literal["fact", "hypothesis", "question", "offer", "greeting_or_signoff"]
    evidence_ids: list[str] = Field(default_factory=list)


class EmailDraft(BaseModel):
    subject: str
    body: str
    claims: list[EmailClaim]


# ---- Stage 7 -------------------------------------------------------------
class SentenceVerdict(BaseModel):
    sentence: str
    verdict: Literal["supported", "framed_as_hypothesis", "neutral", "unsupported"]
    reason: str = ""


class ValidationVerdict(BaseModel):
    sentences: list[SentenceVerdict]
