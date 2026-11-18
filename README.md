# Apex Signal — evidence-first prospect research & outreach drafts

Zamp AI Solutions Analyst case study · **PS-3 Personalised Outreach**

A rep enters a prospect (name, company, role, optional website). Apex Signal runs live web searches,
extracts quoted evidence with an LLM, applies **explicit, deterministic rules** to decide whether there is a
credible *and commercially relevant* reason to contact this person, and then either drafts a short email
for human review or **abstains with a stated reason**. Nothing is ever sent.

The seller is a fictional AP-automation vendor ("Apex": invoice processing, PO matching, approval routing,
exception management). Buyers are CFOs, Finance Directors, Controllers and AP leaders at mid-market companies.

---

### What it does for an SDR

- **One prospect** (New run) with a live 8-stage view, or **a whole list** (Batch: paste/upload CSV, export results).
- Every outcome explains itself: the hook it chose, the **other hooks it considered and why they lost**, and the
  evidence it excluded (with reasons).
- When it can't tell who someone is, it **stops and asks**, then resumes with one click ("Use beca.com" / "I've
  confirmed this person").
- An **Evaluation** page runs a labelled set of real prospects live and scores the judgement (outcome accuracy,
  drafts passing validation, correct abstentions, speed).

## 1. Cost: ₹0

Every component runs on a free tier that needs **no credit card**:

| Need | Provider | Free allowance | Get a key |
|---|---|---|---|
| LLM (backup) | Google Gemini (`gemini-3-flash-preview`) | Free tier, rate-limited per minute/day | aistudio.google.com/apikey |
| LLM (primary) | Groq (`openai/gpt-oss-120b`) | Free tier, rate-limited | console.groq.com/keys |
| Web search (primary) | Tavily | 1,000 credits/month (basic search = 1 credit) | app.tavily.com |
| LLM (last resort) | OpenRouter free models (`openrouter/free`) | ~50 requests/day, free models only | openrouter.ai/keys |
| Web search (backup) | Serper (Google results) | One-time free query allowance | serper.dev |
| Web search (last resort) | DuckDuckGo via `ddgs` | No key, no account | – |
| Hosting | Streamlit Community Cloud | Free public apps | share.streamlit.io |
| Code | GitHub | Free | github.com |

A run uses about 10 searches and 6–8 LLM calls, so the free tiers cover roughly 100 runs a month. Groq is tried first (Gemini's free quota proved
too small in live stress tests); on a rate limit the app first tries the same provider's other free models
(separate quotas), then the next provider, and skips a rate-limited provider for 10 minutes instead of retrying it
on every call; if Tavily credits run out, it continues on Serper, then
DuckDuckGo. If a provider retires a model id, the app lists that provider's models and switches to the closest
available one (OpenRouter: free models only, so nothing can ever be billed). Fallbacks are
recorded on the run. Note: Google may use free-tier Gemini API traffic to improve its products — only public web
data and prospect names go through it here.

## 2. Setup (local)

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.10+
pip install -r requirements-dev.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # paste your free keys
streamlit run app.py
```

Tests need no keys or network (they use labelled fixtures): `pytest -q`

One live run from the terminal (confirms your keys work):

```bash
export GEMINI_API_KEY=... GROQ_API_KEY=... TAVILY_API_KEY=...
python scripts/live_run.py "Jane Doe" "Acme Logistics" "CFO" https://acme.com
```

## 3. Environment variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GEMINI_API_KEY` | at least one LLM key | – | Primary LLM (free) |
| `GROQ_API_KEY` | recommended | – | Backup LLM (free), used automatically on errors/quota |
| `TAVILY_API_KEY` | recommended | – | Live web + news search with page excerpts and dates (free 1,000/month) |
| `OPENROUTER_API_KEY` | optional | – | Last-resort LLM, free models only |
| `SERPER_API_KEY` | optional | – | Backup search (free one-time allowance) |
| `GEMINI_MODEL` / `GROQ_MODEL` | no | `gemini-3-flash-preview` / `openai/gpt-oss-120b` | Model ids |
| `LLM_PROVIDER` | no | first available | Force which LLM is tried first |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` | no | – | Paid alternatives, supported but not needed |
| `DISABLE_DDG_FALLBACK` | no | – | Set to `1` to turn off the keyless search fallback |
| `APEX_DB_PATH` | no | `apex_signal.db` | SQLite file for run history |
| `MAX_SOURCES` | no | `14` | Sources passed to extraction per run |

Keys are read from environment variables or Streamlit secrets and never displayed.

## 4. Deployment (Streamlit Community Cloud, free, ~10 minutes)

1. Create a free GitHub account/repo and upload this folder's contents (drag-and-drop on github.com works).
2. share.streamlit.io → sign in with GitHub → **Create app** → pick the repo, branch `main`, file `app.py`.
3. **Advanced settings → Secrets**: paste your three keys in the format of `.streamlit/secrets.toml.example`.
4. Deploy. The resulting `*.streamlit.app` URL is your "live and runnable" link.

The SQLite history lives on the app's disk: it survives normal use but resets when the app reboots or redeploys.
Free Streamlit apps sleep after inactivity — open the link a few minutes before your demo to wake it.

## 5. Architecture

```
app.py                     Streamlit UI: form, live stage view, decision/evidence/draft tabs, run history
apex_signal/
  pipeline.py              8-stage workflow; persists each stage RUNNING -> DONE/FAILED/SKIPPED/WARNING
  policy.py                ALL decision rules (identity, source tiers, recency, relevance, conflicts, angle, draft checks)
  schemas.py               Pydantic contracts for every LLM call
  llm.py                   Gemini / Groq (OpenAI-compatible) + Anthropic/OpenAI; JSON-only, schema-validated,
                           one corrective retry, automatic provider fallback
  search.py                Tavily -> Serper -> DuckDuckGo fallback; 24h SQLite cache (CachedSearch)
  fetch.py                 Reads the real source pages (free) + publish dates from page metadata; selects the
                           sentences that matter (company/person/finance signals) so prompts stay ~3x smaller
  evalset.py               Labelled real prospects with expected outcomes (used by the Evaluation page)
  db.py                    SQLite: runs + per-stage JSON outputs
scripts/live_run.py        CLI live run
tests/                     pytest suite; tests/fixtures.py = LABELLED synthetic data, never used by the app
```

**Division of labour:** the LLM *reads and writes* (extract quotes, classify signal type, explain, draft,
fact-check). **Rules decide** (is it grounded, credible, current, about the right entity, relevant, conflicted;
which hook wins; draft or abstain). This keeps every decision explainable and testable.

### The 8 stages

| # | Stage | What actually happens |
|---|---|---|
| 1 | Identify | 3 searches (person+company, person+role+company, company site). LLM lists company candidates & person mentions with verbatim quotes. Rules decide company status (PROVIDED / RESOLVED / AMBIGUOUS / UNRESOLVED) and prospect status (CONFIRMED / ROLE_MISMATCH / NOT_FOUND / AMBIGUOUS / CONFLICTING). Blocking ambiguity stops the run here. |
| 2 | Research | 6–7 parallel searches targeted at AP-relevant signals (news, AP/finance jobs, ERP changes, finance leadership, M&A/expansion, prospect interviews, first-party press). Then the top ~14 result pages are **actually read** (free HTTP fetch, 72h cache) to get full text and publish dates. Searches are cached for 24h, so reruns cost nothing. |
| 3 | Extract | Per source, only the relevant sentences are kept (passage selection); the LLM extracts observations (verbatim quote, neutral statement, signal type, subject) and key facts (CFO name, ERP, employer, headcount, …), usually in one call. |
| 4 | Evaluate | Each observation gets categorical labels: grounded? tier A/B/C? fresh/aging/stale/undated? about target? relevance DIRECT/INDIRECT/NONE? depends on a conflicting fact? Every exclusion is recorded with a reason. |
| 5 | Select angle | Rule-based outcome (see policy). If drafting, the top eligible item is chosen by an ordered rule (not a score); the LLM writes hook / hypothesis / product relevance / assumptions. |
| 6 | Draft | LLM writes 60–120 words using only the selected evidence; returns each sentence with its kind and cited evidence ids. Unverified prospects get no personal claims. |
| 7 | Validate | Deterministic checks (citations; numbers must appear in cited sources or the seller description; no 9-word copy of a source; sign-off must be the `[Your name]` placeholder; length; banned phrases; no conflicted values; no URLs) + LLM sentence-level fact-check. One automatic repair pass; remaining issues are flagged, not hidden. |
| 8 | Review | Draft saved as PENDING_REVIEW. Reviewer edits, saves, approves *for manual send* or rejects. Original draft is kept alongside edits. |

## 6. Decision policy (explicit rules, no confidence scores)

**Source tier** — A: company's own domain, government/SEC, wire services, major business press. C: data
brokers, scraped-profile sites, user-generated/social. B: everything else identifiable (trade press, job boards,
LinkedIn). Tier C can never support a hook or establish identity.

**Grounding** — an extracted quote must actually appear in the retrieved text (normalised match, or ≥80% of its
3-word shingles). Anything else is treated as a model hallucination and excluded.

**Recency** — fresh ≤ 12 months, aging 12–24, stale > 24 (excluded), undated (allowed, flagged).

**Relevance (credible ≠ relevant)**

| DIRECT (can open the email) | INDIRECT (only as a question; flagged) | NONE (never a hook) |
|---|---|---|
| AP / finance-ops hiring (own team) · change to the company's OWN ERP/finance systems · finance transformation / shared services · publicly stated AP pain | New CFO/controller · IPO / listing preparation · M&A · expansion · rapid growth · prospect's own finance commentary | Funding · product/marketing · awards/PR · other |

**Conflicts** — two grounded tier A/B sources giving different values for the same fact key (e.g. two different
CFO names; headcounts differing > 50%). Hooks that depend on a conflicted fact are excluded, and conflicted values
are banned from the draft. Conflicting prospect employment blocks the run.

**Rules added after live stress testing (each one came from a real failure)**

- *Funding is funding.* An "expansion" or "growth" claim that comes from a funding announcement ("will use the fresh
  capital to expand…") is reclassified as funding in code — a prompt instruction alone was ignored by the model.
- *Don't stack guesses.* An INDIRECT hook sent to someone outside the finance persona (or whose identity isn't
  verified) is two unverified assumptions in a row → abstain and recommend redirecting to finance.
- *Career history isn't a conflict.* An earlier, different role at the same company (Group Finance Director → CFO) is
  history; only a source calling the *same* role "former" counts as a conflict.
- *Never put the source's words in our mouth.* Drafts that copy 9+ consecutive words from a source fail validation
  (this caught a draft that pasted the prospect's own quote — "has helped *us*…" — as if the seller said it).
- *No invented sender.* The sign-off must be `[Your name]`; a model once signed as "Alex".

**Outcome rules (first match wins)**

1. Company ambiguous (≥ 2 distinct organisations, no website) or prospect ambiguous/conflicting → **NEEDS_INPUT_AMBIGUOUS_IDENTITY** (research skipped; rep asked for website/LinkedIn).
2. Sources disagree on prospect's current employer → **ABSTAIN_CONFLICTING_EVIDENCE**.
3. No eligible evidence, and relevant items were all conflicted → **ABSTAIN_CONFLICTING_EVIDENCE**; all stale → **ABSTAIN_NO_CREDIBLE_EVIDENCE**.
4. No eligible evidence, credible items exist but all are relevance NONE → **ABSTAIN_LOW_RELEVANCE** (e.g. "raised a Series B" alone does not justify pitching invoice automation; recommend nurture).
5. Relevant claims only on tier-C sources → **ABSTAIN_CONFLICTING_EVIDENCE** (unreliable).
6. Nothing credible at all (incl. rejected hallucinations) → **ABSTAIN_NO_CREDIBLE_EVIDENCE**.
7. Otherwise draft. Ranking of eligible items: DIRECT before INDIRECT → tier A before B → fresh → aging → undated → about the prospect before the company → newest. **DRAFT_READY** if no caveats; **DRAFT_WITH_CAVEATS** if the hook is INDIRECT, undated/aging, the prospect is unverified or out of persona, conflicts were excluded, searches partially failed, or validation issues remain.

## 7. Edge cases (all four implemented; none hard-coded)

| Edge case | How it is detected | Behaviour |
|---|---|---|
| No credible information | No grounded, on-target, tier A/B observation | Abstain, explain what was searched/rejected, no draft |
| Conflicting / unreliable information | Fact-key disagreement across credible sources; or relevant claims only on tier C | Abstain or exclude the affected hook; conflicts listed in Evidence tab |
| Real signal, insufficient product relevance | Credible evidence exists but all relevance NONE | Abstain with the signals found and a nurture recommendation |
| Ambiguous prospect identity | Multiple organisations match the name; or the person is found only at other employers; or current vs former disagreement | Stop after stage 1 and ask the rep for the website/LinkedIn |

Also handled: role mismatch (found at company with a different title → no role claims, caveat), non-finance
persona (caveat), hallucinated quotes, invalid model JSON (one corrective retry, then a clean failure),
LLM/API errors, search outages, unsupported numbers in drafts (automatic repair).

## 8. Demo script (5 minutes)

1. **Happy path** (New run): Mark Fleming · Beca · CFO · https://www.beca.com — verified CFO, fresh public signal
   about Beca's Oracle finance transformation. Walk the Decision tab (four boxes + other hooks considered), the
   Evidence tab (exclusions with reasons), edit and approve the draft.
2. **Ambiguity** (New run): Sam Lee · Apex Solutions · Controller, no website → stops at stage 1, lists the matching
   companies; click one to resume (linked re-run).
3. **The SDR's real job** (Batch): click *Fill example rows* → *Run batch*; show the progress table, outcome tiles and
   CSV export. Includes a correct low-relevance abstention (Revspot: funding news only, founder not a finance buyer).
4. **Proof** (Evaluation): open the last evaluation scorecard — outcome accuracy across the labelled real prospects.

Repeated runs within 24h use cached searches, so rehearsals don't burn free quota and the live demo is fast.

## 9. Known limitations

- Research quality is bounded by the search API's snippets (Tavily returns relevant page excerpts, not full pages; DuckDuckGo returns only short snippets, so the keyless fallback finds less). Paywalled or JS-only pages are not read.
- Free tiers are rate-limited; a burst of demo runs can hit Gemini's per-minute limit (the SDK retries with backoff, then Groq takes over). Free-tier models are smaller than paid frontier models, so extraction is occasionally less complete — the grounding check stops this from turning into invented facts.
- LinkedIn content is only seen if the search provider indexes it; there is no LinkedIn API integration.
- Source-tier lists are hand-curated and incomplete; unknown publishers default to tier B.
- Company disambiguation relies on the LLM listing candidates; same-root domains (acme.com / acme.co.uk) are merged and treated as one organisation.
- Role matching uses keyword families (CFO / controller / finance leader / AP); unusual titles may be marked ROLE_MISMATCH (safe failure: no role claims).
- The LLM fact-check is itself a model judgement; the deterministic checks are the hard guarantees.
- Runs execute synchronously in the Streamlit session (~40–90 s each; batches run sequentially); a closed tab mid-run leaves the run in RUNNING.
- Page reading skips sites that block bots (LinkedIn, Glassdoor) and JS-only pages; those fall back to search snippets.
- SQLite on Streamlit Cloud is ephemeral; use a persistent volume or Postgres for production.
- No email sending, CRM sync or multi-user auth by design.

## 10. Assumptions

- Mid-market buyers; India/US/UK sources treated alike. "Recent" = last 24 months.
- A funding round, product launch or award is credible but *not* a reason to pitch AP automation on its own.
- Hypotheses (e.g. "AP is manual") are never stated as facts; they appear as questions and are listed as assumptions.
- Where identity is uncertain, the safe action is to ask the rep rather than guess.
