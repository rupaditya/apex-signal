"""Apex Signal – Streamlit UI. Run with: streamlit run app.py"""
from __future__ import annotations

import os
import io
import time
from datetime import date, datetime

import pandas as pd
import streamlit as st

# Streamlit Cloud secrets -> env (never displayed)
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str) and not os.environ.get(_k):
            os.environ[_k] = _v
except Exception:
    pass

import sys, types  # flat repo layout: treat the repo root as the apex_signal package
if not os.path.isdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "apex_signal")):
    _pkg = types.ModuleType("apex_signal")
    _pkg.__path__ = [os.path.dirname(os.path.abspath(__file__))]
    sys.modules.setdefault("apex_signal", _pkg)

from apex_signal.config import load_settings
from apex_signal.db import RunStore
from apex_signal.llm import build_llm
from apex_signal.evalset import EVAL_CASES
from apex_signal.pipeline import STAGES, Pipeline, ProspectInput
from apex_signal.policy import OUTCOMES
from apex_signal.search import build_search_client

st.set_page_config(page_title="Apex Signal", page_icon="◆", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root { --bg:#f3f2ee; --card:#ffffff; --ink:#1c1f26; --muted:#6b7080; --line:#e7e5df;
        --lime:#cfd76f; --lime-soft:#eef1c9; --pink:#efb8e6; --pink-soft:#f8e1f4; --orange:#f2693c; }
html, body, [class*="css"], .stMarkdown, .stTextInput, .stTextArea, button { font-family:'Inter',sans-serif !important; }
.stApp { background:var(--bg); }
.block-container { padding-top:3rem; max-width:1240px; }
h1,h2,h3,h4 { color:var(--ink); letter-spacing:-0.02em; font-weight:700; }
/* sidebar */
section[data-testid="stSidebar"] { background:#ffffff; border-right:1px solid var(--line); }
section[data-testid="stSidebar"] [role="radiogroup"] label { width:100%; padding:9px 14px; border-radius:999px; margin-bottom:4px; }
section[data-testid="stSidebar"] [role="radiogroup"] label[data-selected="true"] { background:var(--ink); }
section[data-testid="stSidebar"] [role="radiogroup"] label[data-selected="true"] p { color:#fff !important; font-weight:600; }
section[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child:not([data-testid="stMarkdownContainer"]) { display:none !important; }
.brand { display:flex; align-items:center; gap:10px; font-size:1.25rem; font-weight:700; color:var(--ink); margin:4px 0 2px; }
.brand .logo { width:30px; height:30px; border-radius:9px; background:var(--orange); display:flex; align-items:center;
               justify-content:center; color:#fff; font-size:15px; }
.intg { display:flex; justify-content:space-between; align-items:center; background:var(--bg); border-radius:12px;
        padding:8px 12px; margin-bottom:6px; font-size:0.85rem; }
.dot { width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:6px; }
/* form, inputs, buttons */
[data-testid="stForm"] { background:var(--card); border:none; border-radius:20px; padding:22px 24px; box-shadow:0 1px 2px rgba(0,0,0,.04); }
.stTextInput input, .stTextArea textarea { border-radius:12px !important; background:#faf9f6 !important; }
.stButton button, .stFormSubmitButton button { border-radius:999px !important; padding:0.45rem 1.2rem !important; font-weight:600 !important; }
.stButton button[kind="primary"], .stFormSubmitButton button[kind="primaryFormSubmit"] { background:var(--ink) !important; border-color:var(--ink) !important; color:#fff !important; }
/* tabs as pills */
.stTabs [data-baseweb="tab-list"] { gap:6px; border-bottom:none; }
[data-testid="stTab"] { background:#fff; border-radius:999px !important; padding:6px 18px !important; height:auto; margin-right:6px; }
[data-testid="stTab"][aria-selected="true"] { background:var(--ink) !important; }
.react-aria-SelectionIndicator { display:none !important; }
.stTabs [aria-selected="true"] p { color:#fff !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }
/* alerts, tables, expanders */
[data-testid="stAlert"] { border-radius:16px; border:none; }
[data-testid="stDataFrame"] { background:#fff; border-radius:16px; padding:6px; }
[data-testid="stExpander"] details { background:#fff; border-radius:14px; border:1px solid var(--line); }
[data-testid="stMetric"] { display:none; }
/* custom components */
.hello { font-size:2rem; font-weight:700; color:var(--ink); letter-spacing:-0.03em; margin-bottom:0; }
.sub { color:var(--muted); font-size:0.88rem; margin-bottom:14px; }
.badge { display:inline-block; padding:3px 11px; border-radius:999px; font-size:0.75rem; font-weight:600; vertical-align:middle; }
.b-green { background:var(--lime-soft); color:#4f5a10; } .b-amber { background:#fdebd3; color:#8a4b00; }
.b-red { background:#fde0d8; color:#a3300e; } .b-grey { background:#eceae4; color:#4a4f5c; }
.b-blue { background:var(--pink-soft); color:#7d2a6f; }
.stage { display:flex; gap:12px; align-items:center; padding:11px 16px; border-radius:16px; margin-bottom:7px; background:#fff; }
.stage.run { background:var(--pink-soft); } .stage.ok { background:#fff; } .stage.fail { background:#fde0d8; }
.stage.skip { background:transparent; border:1px dashed #d8d5cc; } .stage.warn { background:#fdebd3; }
.stage .ic { width:26px; height:26px; border-radius:50%; display:flex; align-items:center; justify-content:center;
             font-weight:700; font-size:0.8rem; flex-shrink:0; }
.stage .nm { font-weight:600; min-width:270px; color:var(--ink); } .stage .sm { color:var(--muted); font-size:0.88rem; }
.stage .tm { margin-left:auto; font-size:0.78rem; color:var(--ink); white-space:nowrap; border:1px solid var(--line);
             border-radius:999px; padding:2px 10px; background:#fff; }
.card { background:#fff; border-radius:18px; padding:16px 18px; margin-bottom:10px; }
.card h4 { margin:0 0 8px 0; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; color:var(--muted); }
.card.lime { background:var(--lime); } .card.pink { background:var(--pink); }
.quote { border-left:3px solid var(--lime); padding-left:10px; color:#3c4350; font-style:italic; margin:6px 0; }
.small { font-size:0.85rem; color:var(--muted); }
.tiles { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:6px 0 16px; }
.tile { border-radius:20px; padding:16px 18px; background:#fff; min-height:104px; }
.tile .lbl { font-size:0.85rem; font-weight:600; color:var(--ink); } .tile .num { font-size:2.3rem; font-weight:700; color:var(--ink); letter-spacing:-0.03em; }
.tile.lime { background:var(--lime); } .tile.pink { background:var(--pink); } .tile.dark { background:var(--ink); }
.tile.dark .lbl, .tile.dark .num { color:#fff; }
.rcards { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-bottom:16px; }
.rcard { background:#fff; border-radius:20px; padding:14px 16px; }
.rcard.lime { background:var(--lime); } .rcard.pink { background:var(--pink); }
.rcard .src { font-size:0.72rem; color:var(--muted); } .rcard .ttl { font-weight:600; color:var(--ink); margin:2px 0 10px; }
.rcard .row { display:flex; justify-content:space-between; align-items:flex-end; }
.rcard .big { font-size:1.7rem; font-weight:700; color:var(--ink); }
.chip { display:inline-block; border-radius:999px; padding:2px 10px; font-size:0.75rem; background:rgba(255,255,255,.7); border:1px solid rgba(0,0,0,.08); }
.sec { font-size:1.25rem; font-weight:700; color:var(--ink); margin:8px 0 8px; }
</style>
""", unsafe_allow_html=True)

settings = load_settings()


@st.cache_resource
def get_store(path: str) -> RunStore:
    return RunStore(path)


store = get_store(settings.db_path)

OUTCOME_STYLE = {
    "DRAFT_READY": "b-green", "DRAFT_WITH_CAVEATS": "b-amber", "NEEDS_INPUT_AMBIGUOUS_IDENTITY": "b-blue",
    "ABSTAIN_CONFLICTING_EVIDENCE": "b-red", "ABSTAIN_NO_CREDIBLE_EVIDENCE": "b-grey",
    "ABSTAIN_LOW_RELEVANCE": "b-grey",
}
SHORT = {"DRAFT_READY": "Draft ready", "DRAFT_WITH_CAVEATS": "Draft + caveats",
         "NEEDS_INPUT_AMBIGUOUS_IDENTITY": "Needs input", "ABSTAIN_CONFLICTING_EVIDENCE": "Conflicting",
         "ABSTAIN_NO_CREDIBLE_EVIDENCE": "No evidence", "ABSTAIN_LOW_RELEVANCE": "Low relevance"}
STATUS_ICON = {"RUNNING": ("●", "#7d2a6f", "#fff", "run"), "DONE": ("✓", "#fff", "#1c1f26", "ok"),
               "FAILED": ("✕", "#fff", "#f2693c", "fail"), "SKIPPED": ("–", "#8a8f9c", "#eceae4", "skip"),
               "WARNING": ("!", "#1c1f26", "#cfd76f", "warn"), "PENDING": ("○", "#8a8f9c", "#f3f2ee", "ok")}


def badge(text, cls):
    return f'<span class="badge {cls}">{text}</span>'


def outcome_badge(run):
    if run["status"] == "FAILED":
        return badge("Run failed", "b-red")
    if run["status"] == "RUNNING":
        return badge("Running", "b-blue")
    o = run.get("outcome") or ""
    return badge(OUTCOMES.get(o, o), OUTCOME_STYLE.get(o, "b-grey"))


def stage_row(no, name, status, summary="", elapsed=None):
    ic, fg, bg, cls = STATUS_ICON.get(status, STATUS_ICON["PENDING"])
    tm = f'<div class="tm">⏱ {elapsed:.1f}s</div>' if elapsed is not None else ""
    return (f'<div class="stage {cls}"><div class="ic" style="color:{fg};background:{bg}">{ic}</div>'
            f'<div class="nm">{no}. {name}</div><div class="sm">{summary or status.title()}</div>{tm}</div>')


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown('<div class="brand"><div class="logo">◆</div>Apex Signal</div>', unsafe_allow_html=True)
    st.caption("Evidence-first prospect research & outreach drafts for AP automation.")
    view = st.radio("View", ["New run", "Batch", "Run history", "Evaluation", "Tutorial"], label_visibility="collapsed", key="view")
    st.divider()
    missing = settings.missing_credentials()
    search_ok = bool(settings.search_chain)
    llm_ok = bool(settings.llm_chain)
    llm_label = " → ".join(f"{p}" for p in settings.llm_chain) or "none"
    llm_model = settings.llm_models.get(settings.llm_provider, "")

    def _intg(label, value, ok):
        col = "#8fa315" if ok else "#f2693c"
        return (f'<div class="intg"><span><span class="dot" style="background:{col}"></span>{label}</span>'
                f'<code>{value}</code></div>')
    st.markdown('<div class="small" style="margin-bottom:6px;font-weight:600">INTEGRATIONS</div>'
                + _intg("Search", " → ".join(settings.search_chain), search_ok)
                + _intg("LLM", llm_label, llm_ok)
                + (f'<div class="small" style="margin:-2px 0 6px 4px">primary model: {llm_model}</div>' if llm_model else ""),
                unsafe_allow_html=True)
    if "duckduckgo" == settings.search_provider:
        st.caption("No TAVILY_API_KEY: using keyless DuckDuckGo search (snippets only). A free Tavily key gives better research.")
    if missing:
        st.error("Missing credentials: " + ", ".join(missing) + ". Add one as an environment variable or Streamlit secret.")
    st.divider()
    st.caption("Nothing is ever sent. Approved drafts are marked for a human to send manually.")


# ------------------------------------------------------------------ run detail
def render_run(run_id: str):
    run = store.get_run(run_id)
    if not run:
        st.warning("Run not found.")
        return
    stages = {s["stage_no"]: s for s in store.get_stages(run_id)}
    s1 = (stages.get(1) or {}).get("output") or {}
    s4 = (stages.get(4) or {}).get("output") or {}
    s5 = (stages.get(5) or {}).get("output") or {}
    s7 = (stages.get(7) or {}).get("output") or {}
    ident = s1.get("decision") or {}

    st.markdown(f'<div class="sec" style="font-size:1.5rem">{run["prospect_name"]} · {run["prospect_role"]} · '
                f'{run["company_name"]} &nbsp; {outcome_badge(run)}</div>', unsafe_allow_html=True)
    st.caption(f"Run `{run['id']}` · {run['created_at']} UTC · {run['providers']}")
    if run["status"] == "FAILED":
        st.error(f"**Run failed.** {run['error'].splitlines()[0] if run['error'] else ''}")
    elif run.get("outcome_reason"):
        box = st.success if run["outcome"] == "DRAFT_READY" else st.warning if run["outcome"] == "DRAFT_WITH_CAVEATS" else st.info
        box(run["outcome_reason"])

    if run.get("outcome") == "NEEDS_INPUT_AMBIGUOUS_IDENTITY":
        render_resolution(run, s1)
    if run.get("parent_run_id"):
        st.caption(f"Re-run of `{run['parent_run_id']}` with the rep's answer.")

    tabs = st.tabs(["Decision", "Evidence & sources", "Draft & review", "Stage log"])

    # --- Decision
    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="card"><h4>Identity</h4>'
                        f'Company: <b>{ident.get("company_status", "—")}</b> ({ident.get("company_domain") or "no domain"})<br>'
                        f'Prospect: <b>{ident.get("prospect_status", "—")}</b> · Buyer persona fit: '
                        f'<b>{"yes" if ident.get("persona_fit") else "no"}</b><br><span class="small">'
                        + "<br>".join(ident.get("reasons", [])) + '</span></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="card"><h4>Decision rule outcome</h4>'
                        f'{OUTCOMES.get(run.get("outcome") or "", run.get("outcome") or "—")}<br>'
                        f'<span class="small">{s5.get("reason", "")}</span></div>', unsafe_allow_html=True)
        evidence = {e["id"]: e for e in s4.get("evidence", [])}
        rat = s5.get("rationale") or {}
        if s5.get("primary"):
            used = [evidence[i] for i in [s5["primary"]] + s5.get("supporting", []) if i in evidence]
            st.markdown("##### 1 · Verified observations (used)")
            for e in used:
                st.markdown(f'<div class="card"><b>{e["id"]}</b> {badge(e["signal_type"], "b-grey")} '
                            f'{badge(e["relevance"], "b-green" if e["relevance"] == "DIRECT" else "b-amber")} '
                            f'{badge("tier " + e["tier"], "b-grey")} {badge(e["recency"], "b-grey")}<br>'
                            f'{e["observation"]}<div class="quote">“{e["quote"]}”</div>'
                            f'<span class="small"><a href="{e["url"]}" target="_blank">{e["domain"]}</a> · '
                            f'{e.get("published_date") or "date unknown"}</span></div>', unsafe_allow_html=True)
            st.markdown("##### 2 · Hypothesised business implication (not verified)")
            st.markdown(f'<div class="card pink">{rat.get("hypothesis", "—")}</div>', unsafe_allow_html=True)
            st.markdown("##### 3 · Relevance to the seller's product")
            st.markdown(f'<div class="card lime">{rat.get("product_relevance", "—")}</div>', unsafe_allow_html=True)
        alts = s5.get("alternatives") or []
        if alts:
            with st.expander(f"Other hooks considered ({len(alts)}) – and why they ranked lower"):
                st.dataframe(pd.DataFrame([{"id": a["id"], "signal": a["signal_type"], "relevance": a["relevance"],
                                            "tier": a["tier"], "recency": a["recency"], "observation": a["observation"],
                                            "ranked lower because": a["why_lower"]} for a in alts]),
                             hide_index=True, width="stretch")
        st.markdown("##### 4 · Unsupported assumptions & evidence limitations")
        lims = list(rat.get("assumptions", [])) + list(s5.get("caveats", []))
        excluded = [e for e in s4.get("evidence", []) if not e["eligible"]]
        if excluded:
            lims.append(f"{len(excluded)} extracted item(s) excluded by policy – see Evidence tab.")
        st.markdown('<div class="card">' + ("<br>".join(f"• {x}" for x in lims) or "None recorded.") + "</div>",
                    unsafe_allow_html=True)

    # --- Evidence
    with tabs[1]:
        ev = s4.get("evidence", [])
        if ev:
            df = pd.DataFrame([{
                "id": e["id"], "eligible": "✅" if e["eligible"] else "—", "signal": e["signal_type"],
                "relevance": e["relevance"], "tier": e["tier"], "recency": e["recency"],
                "grounded": "yes" if e["grounded"] else "NO", "about target": "yes" if e["entity_ok"] else "no",
                "observation": e["observation"], "excluded because": " ".join(e["exclusions"]),
                "source": e["url"]} for e in ev])
            st.dataframe(df, hide_index=True, width="stretch",
                         column_config={"source": st.column_config.LinkColumn("source")})
        else:
            st.info("No evidence was extracted for this run.")
        if s4.get("conflicts"):
            st.markdown("**Conflicting facts detected**")
            for k, vals in s4["conflicts"].items():
                st.markdown(f"- `{k}`: " + " vs ".join(f'“{v["value"]}” ([{v["source_id"]}]({v["url"]}))' for v in vals))
        srcs = s4.get("sources") or {}
        if srcs:
            with st.expander(f"All retrieved sources ({len(srcs)})"):
                for sid, s in srcs.items():
                    read = {"read": " · 📄 page read"}.get(s.get("fetch"), f" · snippet only ({s['fetch']})" if s.get("fetch") else "")
                    dated = f" (date from {s['date_from']})" if s.get("date_from") else ""
                    st.markdown(f"**{sid}** [{s['title'] or s['url']}]({s['url']}) · {s.get('published_date') or 'undated'}{dated}"
                                f"{read} · query: _{s.get('query', '')}_")

    # --- Draft
    with tabs[2]:
        if not run.get("draft_body"):
            st.info("No draft was generated. The process abstained or failed – see the Decision tab for why.")
        else:
            st.markdown(f"Review status: {badge(run['review_status'], {'APPROVED': 'b-green', 'REJECTED': 'b-red'}.get(run['review_status'], 'b-amber'))}",
                        unsafe_allow_html=True)
            last = (s7.get("attempts") or [{}])[-1]
            if s7.get("passed"):
                st.success("Validation passed: every factual sentence traces to cited evidence.")
            else:
                st.warning("Validation issues remain:\n\n" + "\n".join(f"- {p}" for p in last.get("problems", [])))
            subj = st.text_input("Subject", value=run["edited_subject"] or run["draft_subject"], key=f"subj_{run_id}")
            body = st.text_area("Body", value=run["edited_body"] or run["draft_body"], height=260, key=f"body_{run_id}")
            note = st.text_input("Reviewer note (optional)", value=run.get("reviewer_note") or "", key=f"note_{run_id}")
            st.caption(f"{len(body.split())} words")
            b1, b2, b3, _ = st.columns([1, 1.4, 1, 3])
            if b1.button("Save edits", key=f"save_{run_id}"):
                store.update_run(run_id, edited_subject=subj, edited_body=body, reviewer_note=note)
                st.toast("Edits saved")
                st.rerun()
            if b2.button("Approve for manual send", type="primary", key=f"appr_{run_id}"):
                store.update_run(run_id, edited_subject=subj, edited_body=body, reviewer_note=note, review_status="APPROVED")
                st.rerun()
            if b3.button("Reject", key=f"rej_{run_id}"):
                store.update_run(run_id, reviewer_note=note, review_status="REJECTED")
                st.rerun()
            if run["edited_body"] and run["edited_body"] != run["draft_body"]:
                with st.expander("Original generated draft"):
                    st.text(f"Subject: {run['draft_subject']}\n\n{run['draft_body']}")
            with st.expander("Sentence-level fact check"):
                v = (last.get("verdict") or {}).get("sentences", [])
                if v:
                    st.dataframe(pd.DataFrame(v), hide_index=True, width="stretch")

    # --- Stage log
    with tabs[3]:
        rows = "".join(stage_row(no, name, (stages.get(no) or {}).get("status", "PENDING"),
                                 (stages.get(no) or {}).get("summary", "")) for no, name in STAGES)
        st.markdown(rows, unsafe_allow_html=True)
        for no, name in STAGES:
            s = stages.get(no)
            if s and s.get("output") is not None:
                with st.expander(f"Stage {no} output · {s['started_at']} → {s['finished_at']}"):
                    st.json(s["output"], expanded=False)
        if run.get("error"):
            with st.expander("Error detail"):
                st.code(run["error"])


# ------------------------------------------------------------------ shared runners
def queue_rerun(run, website=None, rep_confirmed=False):
    """Button callback: queue a re-run with the rep's answer and jump to the live view."""
    st.session_state["pending"] = dict(prospect_name=run["prospect_name"], company_name=run["company_name"],
                                       prospect_role=run["prospect_role"],
                                       company_website=website or run.get("company_website"),
                                       rep_confirmed=rep_confirmed, parent_run_id=run["id"])
    st.session_state["view"] = "New run"


def render_resolution(run, s1):
    cands = [c for c in (s1.get("extraction") or {}).get("company_candidates", []) if c.get("domain")]
    st.markdown('<div class="card pink"><h4>Resolve and continue</h4>The process stopped instead of guessing. '
                'Pick the right organisation, or confirm the person yourself, and it re-runs with your answer '
                '(the new run is linked to this one).</div>', unsafe_allow_html=True)
    cols = st.columns(min(max(len(cands), 1), 3) + 1)
    for i, c in enumerate(cands[:3]):
        cols[i].button(f"Use {c['domain']}", key=f"res_{run['id']}_{i}", help=c.get("description", ""),
                       on_click=queue_rerun, args=(run,), kwargs={"website": "https://" + c["domain"]})
    cols[-1].button("I've confirmed this person – continue", key=f"conf_{run['id']}",
                    on_click=queue_rerun, args=(run,), kwargs={"rep_confirmed": True})


def make_pipeline(on_event=None):
    return Pipeline(store, build_search_client(settings), build_llm(settings),
                    max_sources=settings.max_sources_for_extraction, on_event=on_event,
                    fetch_pages=settings.fetch_pages)


def run_with_live_view(inp: ProspectInput):
    st.markdown('<div class="sec">Live execution</div>', unsafe_allow_html=True)
    slots = {no: st.empty() for no, _ in STAGES}
    started = {}
    for no, name in STAGES:
        slots[no].markdown(stage_row(no, name, "PENDING", "Waiting"), unsafe_allow_html=True)

    def on_event(no, name, status, summary, output):
        if status == "RUNNING":
            started[no] = time.time()
            slots[no].markdown(stage_row(no, name, "RUNNING", "Running…"), unsafe_allow_html=True)
        else:
            el = time.time() - started.get(no, time.time())
            slots[no].markdown(stage_row(no, name, status, summary, el), unsafe_allow_html=True)

    try:
        return make_pipeline(on_event).run(inp)
    except Exception as e:
        st.error(f"Could not start run: {e}")
        return None


def duration_s(r):
    try:
        return (datetime.fromisoformat(r["updated_at"]) - datetime.fromisoformat(r["created_at"])).total_seconds()
    except Exception:
        return None


def runs_table(runs):
    return pd.DataFrame([{
        "run": r["id"], "prospect": r["prospect_name"], "company": r["company_name"], "role": r["prospect_role"],
        "status": r["status"], "outcome": OUTCOMES.get(r["outcome"] or "", r["outcome"] or (r["error"] or "")[:60]),
        "review": r["review_status"], "seconds": round(duration_s(r) or 0),
        "subject": r["edited_subject"] or ""} for r in runs])


def export_csv(runs) -> bytes:
    df = pd.DataFrame([{
        "prospect_name": r["prospect_name"], "company_name": r["company_name"], "prospect_role": r["prospect_role"],
        "company_website": r["company_website"] or "", "outcome": r["outcome"] or r["status"],
        "reason": r["outcome_reason"] or (r["error"] or ""), "review_status": r["review_status"],
        "subject": r["edited_subject"] or "", "body": r["edited_body"] or "", "run_id": r["id"]} for r in runs])
    return df.to_csv(index=False).encode()


def parse_prospects(text: str) -> tuple[list[ProspectInput], list[str]]:
    """Accepts CSV with or without a header: name, company, role[, website]."""
    import csv
    rows, errors = [], []
    lines = [l for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return [], ["No rows found."]
    reader = list(csv.reader(lines))
    head = [h.strip().lower() for h in reader[0]]
    if any(h in ("name", "prospect", "prospect_name", "company", "company_name") for h in head):
        idx = {k: next((i for i, h in enumerate(head) if any(a == h for a in al)), None) for k, al in {
            "name": ("name", "prospect", "prospect_name", "full name"), "company": ("company", "company_name", "account"),
            "role": ("role", "title", "prospect_role", "job title"), "site": ("website", "company_website", "domain", "url")}.items()}
        body = reader[1:]
    else:
        idx, body = {"name": 0, "company": 1, "role": 2, "site": 3}, reader
    for n, r in enumerate(body, 1):
        get = lambda k: (r[idx[k]].strip() if idx.get(k) is not None and idx[k] < len(r) else "")
        name, company, role = get("name"), get("company"), get("role")
        if not (name and company and role):
            errors.append(f"Row {n}: needs name, company and role – skipped ({', '.join(r)[:60]}).")
            continue
        site = get("site")
        rows.append(ProspectInput(name, company, role, (site if site.startswith("http") else f"https://{site}") if site else None))
    return rows, errors


def run_batch(inputs: list[ProspectInput], batch_name: str):
    bid = store.create_batch(batch_name)
    prog = st.progress(0.0, text=f"0 / {len(inputs)} done")
    table = st.empty()
    rows = [{"#": i + 1, "prospect": p.prospect_name, "company": p.company_name, "status": "queued", "outcome": ""}
            for i, p in enumerate(inputs)]
    table.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    for i, p in enumerate(inputs):
        rows[i]["status"] = "running…"
        table.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        def on_event(no, name, status, summary, output, i=i):
            if status == "RUNNING":
                rows[i]["status"] = f"stage {no}/8 · {name}"
                table.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        p.batch_id = bid
        try:
            rid = make_pipeline(on_event).run(p)
            r = store.get_run(rid)
            rows[i]["status"] = r["status"].lower()
            rows[i]["outcome"] = OUTCOMES.get(r["outcome"] or "", r["outcome"] or (r["error"] or "")[:50])
        except Exception as e:  # one bad row never stops the batch
            rows[i]["status"], rows[i]["outcome"] = "error", str(e)[:60]
        prog.progress((i + 1) / len(inputs), text=f"{i + 1} / {len(inputs)} done")
        table.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    return bid


def outcome_tiles(runs):
    n_draft = sum(1 for r in runs if r["draft_body"])
    n_abst = sum(1 for r in runs if (r["outcome"] or "").startswith(("ABSTAIN", "NEEDS")))
    n_fail = sum(1 for r in runs if r["status"] == "FAILED")
    n_appr = sum(1 for r in runs if r["review_status"] == "APPROVED")
    st.markdown(f'''<div class="tiles">
      <div class="tile dark"><div class="lbl">Runs</div><div class="num">{len(runs)}</div></div>
      <div class="tile lime"><div class="lbl">Drafts for review</div><div class="num">{n_draft}</div></div>
      <div class="tile pink"><div class="lbl">Abstained / needs input</div><div class="num">{n_abst}</div></div>
      <div class="tile"><div class="lbl">Approved · failed</div><div class="num">{n_appr} · {n_fail}</div></div>
    </div>''', unsafe_allow_html=True)


def header(title, sub):
    st.markdown(f'<div class="hello">{title}</div><div class="sub">{date.today():%A, %B %d, %Y} · {sub}</div>',
                unsafe_allow_html=True)


# ------------------------------------------------------------------ new run
if view == "New run":
    header("New prospect run", "Researches live public sources, applies explicit evidence rules, then drafts for "
                               "review or abstains with a reason.")
    pending = st.session_state.pop("pending", None)
    with st.form("prospect"):
        c1, c2 = st.columns(2)
        pn = c1.text_input("Prospect name *", placeholder="e.g. Jane Doe")
        role = c2.text_input("Prospect role *", placeholder="e.g. CFO, Controller, AP Manager")
        cn = c1.text_input("Company name *", placeholder="e.g. Acme Logistics")
        site = c2.text_input("Company website (optional, resolves ambiguity)", placeholder="https://")
        conf = st.checkbox("I've already confirmed this is the right person (e.g. checked their LinkedIn)",
                           help="Lets the run continue if public sources can't place the person. The email still "
                                "makes no personal claims.")
        go = st.form_submit_button("Run research", type="primary", disabled=bool(missing))

    if pending:
        st.info(f"Re-running **{pending['prospect_name']} @ {pending['company_name']}** with your answer "
                f"({'website ' + pending['company_website'] if pending.get('company_website') else 'identity confirmed by rep'}).")
        rid = run_with_live_view(ProspectInput(**pending))
        if rid:
            st.session_state["last_run"] = rid
    elif go:
        if not (pn.strip() and cn.strip() and role.strip()):
            st.error("Prospect name, company name and role are required.")
        else:
            rid = run_with_live_view(ProspectInput(pn.strip(), cn.strip(), role.strip(), site.strip() or None,
                                                   rep_confirmed=conf))
            if rid:
                st.session_state["last_run"] = rid
    if st.session_state.get("last_run"):
        st.divider()
        render_run(st.session_state["last_run"])

# ------------------------------------------------------------------ batch
elif view == "Batch":
    header("Batch research", "An SDR's real job: a list of prospects. Each row runs the full 8-stage process; "
                             "drafts land in the review queue, abstentions explain themselves.")
    st.markdown('<div class="card"><h4>Input</h4>Paste CSV rows <code>name, company, role, website(optional)</code> '
                '(a header row is fine), or upload a CSV exported from your CRM/spreadsheet. Runs are sequential '
                '(~1 min each) to stay inside free API limits; repeated searches are served from a 24h cache.</div>',
                unsafe_allow_html=True)
    example = ("name,company,role,website\nMark Fleming,Beca,CFO,beca.com\nSam Lee,Apex Solutions,Controller,\n"
               "Varun Garg,Revspot,Co-founder,")
    up = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
    if up is not None and st.session_state.get("_uploaded") != up.file_id:
        st.session_state["batch_text"] = up.getvalue().decode("utf-8", "replace")
        st.session_state["_uploaded"] = up.file_id
    st.button("Fill example rows", on_click=lambda: st.session_state.update(batch_text=example))
    txt = st.text_area("Prospects", key="batch_text", height=150, placeholder=example)
    inputs, errs = parse_prospects(txt) if txt.strip() else ([], [])
    for e in errs:
        st.warning(e)
    c1, c2 = st.columns([1, 3])
    max_rows = 25
    c2.caption(f"{len(inputs)} valid row(s)" + (f" – only the first {max_rows} will run" if len(inputs) > max_rows else ""))
    if c1.button("Run batch", type="primary", disabled=not inputs or bool(missing)):
        bid = run_batch(inputs[:max_rows], f"Batch {datetime.now():%d %b %H:%M}")
        st.session_state["last_batch"] = bid
    batches = store.list_batches()
    if batches:
        st.markdown('<div class="sec">Batches</div>', unsafe_allow_html=True)
        labels = {b["id"]: f'{b["name"]} · {len(b["run_ids"])} run(s)' for b in batches}
        default = list(labels).index(st.session_state["last_batch"]) if st.session_state.get("last_batch") in labels else 0
        bsel = st.selectbox("Batch", list(labels), index=default, format_func=lambda k: labels[k])
        bruns = [r for r in (store.get_run(i) for i in next(b for b in batches if b["id"] == bsel)["run_ids"]) if r]
        if bruns:
            outcome_tiles(bruns)
            st.dataframe(runs_table(bruns), hide_index=True, width="stretch")
            st.download_button("Download results (CSV)", export_csv(bruns), file_name=f"apex_signal_{bsel}.csv",
                               mime="text/csv")
            pick = st.selectbox("Open a run from this batch", [r["id"] for r in bruns],
                                format_func=lambda k: next(f'{r["prospect_name"]} @ {r["company_name"]}' for r in bruns if r["id"] == k))
            st.divider()
            render_run(pick)

# ------------------------------------------------------------------ evaluation
elif view == "Evaluation":
    header("Evaluation", "Does the judgement hold up? A labelled set of real prospects, each with the outcome the "
                         "policy should reach. Every evaluation run is live research.")
    st.dataframe(pd.DataFrame([{"case": c["id"], "prospect": c["prospect_name"], "company": c["company_name"],
                                "role": c["prospect_role"], "website": c["company_website"] or "—",
                                "expected": " or ".join(sorted(SHORT.get(o, o) for o in c["expect"])),
                                "what it tests": c["tests"]} for c in EVAL_CASES]), hide_index=True, width="stretch")
    if st.button(f"Run evaluation ({len(EVAL_CASES)} live runs, ~{len(EVAL_CASES)} min)", type="primary", disabled=bool(missing)):
        bid = run_batch([ProspectInput(c["prospect_name"], c["company_name"], c["prospect_role"], c["company_website"] or None)
                         for c in EVAL_CASES], f"Evaluation {datetime.now():%d %b %H:%M}")
        st.session_state["last_eval"] = bid
    evals = [b for b in store.list_batches() if b["name"].startswith("Evaluation")]
    if evals:
        b = next((x for x in evals if x["id"] == st.session_state.get("last_eval")), evals[0])
        runs = [store.get_run(i) for i in b["run_ids"]]
        rows, passed = [], 0
        for c, r in zip(EVAL_CASES, runs):
            if not r:
                continue
            got = r["outcome"] or r["status"]
            ok = got in c["expect"]
            passed += ok
            s7 = next((x["output"] for x in store.get_stages(r["id"]) if x["stage_no"] == 7 and x["output"]), {}) or {}
            rows.append({"case": c["id"], "result": "✅ pass" if ok else "❌ fail",
                         "expected": " or ".join(sorted(SHORT.get(o, o) for o in c["expect"])),
                         "actual": SHORT.get(got, got), "draft validated": {True: "yes", False: "issues flagged"}.get(s7.get("passed"), "—"),
                         "seconds": round(duration_s(r) or 0), "run": r["id"]})
        n = len(rows)
        drafts = [x for x in rows if x["draft validated"] != "—"]
        st.markdown(f'''<div class="tiles">
          <div class="tile dark"><div class="lbl">Outcome accuracy</div><div class="num">{passed}/{n}</div></div>
          <div class="tile lime"><div class="lbl">Drafts passing validation</div><div class="num">{sum(x["draft validated"] == "yes" for x in drafts)}/{len(drafts)}</div></div>
          <div class="tile pink"><div class="lbl">Correct abstentions</div><div class="num">{sum(1 for c, x in zip(EVAL_CASES, rows) if x["result"].startswith("✅") and not (c["expect"] & {"DRAFT_READY", "DRAFT_WITH_CAVEATS"}))}</div></div>
          <div class="tile"><div class="lbl">Median seconds / run</div><div class="num">{int(pd.Series([x["seconds"] for x in rows]).median()) if rows else 0}</div></div>
        </div>''', unsafe_allow_html=True)
        st.caption(f'{b["name"]} · live results change as the web changes; failures are shown, not hidden.')
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if rows:
            pick = st.selectbox("Open an evaluation run", [x["run"] for x in rows],
                                format_func=lambda k: next(f'{x["case"]} – {x["actual"]}' for x in rows if x["run"] == k))
            st.divider()
            render_run(pick)

# ------------------------------------------------------------------ tutorial
elif view == "Tutorial":
    header("Tutorial", "What Apex Signal does, how it decides, and how to use every page")

    def card(title, body, cls=""):
        st.markdown(f'<div class="card {cls}"><h4>{title}</h4>{body}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec">1 · The problem and the use case</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        card("The problem", "An SDR working a list of 200 prospects can't spend 20 minutes researching each one, so "
             "they send templates – and templates get deleted. The signal that would earn a reply (a finance hire, "
             "an ERP migration, an IPO filing) is usually public; finding it and judging whether it matters is the work.")
    with c2:
        card("What Apex Signal does", "Give it a prospect (or a list). It researches live public sources, decides with "
             "<b>explicit rules</b> whether there is a credible <i>and</i> relevant reason to contact this person, and "
             "either drafts a short email for human review or <b>abstains with a stated reason</b>. Nothing is ever "
             "sent automatically.", "lime")
    card("Who it's for", "Seller: <b>Apex</b> (fictional) – accounts-payable automation: invoice capture, 2/3-way PO "
         "matching, approval routing, exception management. Buyers: CFOs, Finance Directors, Controllers and AP "
         "leaders at mid-market companies. The goal is not polished emails – it is a <b>defensible reason to write</b>, "
         "and the discipline to not write when there isn't one.")

    st.markdown('<div class="sec">2 · How to navigate</div>', unsafe_allow_html=True)
    nav = [
        ("New run", "Research one prospect. Enter name, role, company and (optional) website, press <b>Run research</b> "
         "and watch the 8 stages execute live. Tick <i>I've already confirmed this person</i> if you've checked "
         "their LinkedIn and public sources can't place them."),
        ("Batch", "Research a list. Paste CSV rows <code>name, company, role, website</code> or upload a CSV "
         "(<b>Fill example rows</b> loads a demo). Rows run one after another with a live status table; download "
         "all outcomes and drafts as CSV."),
        ("Run history", "Every run ever made: status, outcome, review state and duration. Open any run to see its "
         "full decision, evidence and draft. Export everything as CSV."),
        ("Evaluation", "A labelled set of real prospects with the outcome the rules <i>should</i> reach. "
         "<b>Run evaluation</b> researches all of them live and scores the judgement: outcome accuracy, drafts "
         "passing validation, correct abstentions, speed."),
        ("Tutorial", "This page."),
    ]
    st.markdown('<div class="card">' + "".join(f"<p><b>{n}</b> – {d}</p>" for n, d in nav) + "</div>",
                unsafe_allow_html=True)
    card("Reading a run", "<b>Decision</b> tab: identity check, the rule outcome, and four separated boxes – "
         "① verified observations (with quote, source, date), ② the hypothesised business implication (pink: not "
         "verified), ③ relevance to the product (lime), ④ unsupported assumptions and limitations – plus <b>Other "
         "hooks considered</b> and why they ranked lower. <b>Evidence &amp; sources</b>: every extracted item with "
         "its labels and the written reason it was excluded; every source with its query and whether the page was "
         "read. <b>Draft &amp; review</b>: validation result, editable subject/body, <i>Save edits</i>, <i>Approve for "
         "manual send</i>, <i>Reject</i>, and the sentence-level fact check. <b>Stage log</b>: raw output of every "
         "stage with timestamps.")
    card("When a run stops to ask", "If the company name matches several organisations, or the person can't be "
         "placed at the company, the run stops after stage 1 instead of guessing. A <b>Resolve and continue</b> "
         "panel offers one-click buttons (<i>Use example.com</i> / <i>I've confirmed this person</i>); the re-run is "
         "linked to the original.", "pink")

    st.markdown('<div class="sec">3 · What happens in a run (the 8 stages)</div>', unsafe_allow_html=True)
    stages_txt = [
        ("Identify", "Searches person + company; the AI lists matching organisations and mentions of the person with "
         "verbatim quotes; <b>rules</b> decide: company PROVIDED / RESOLVED / AMBIGUOUS, prospect CONFIRMED / "
         "ROLE_MISMATCH / NOT_FOUND / AMBIGUOUS / CONFLICTING. Ambiguity stops the run here."),
        ("Research", "6–7 targeted searches (news, finance/AP jobs, ERP changes, finance leadership, M&amp;A/expansion, "
         "prospect interviews, the company's own press pages), then the top pages are actually read to get full "
         "text and publish dates. Searches are cached for 24h."),
        ("Extract", "Only the sentences about this company/person/finance signals are kept; the AI extracts "
         "observations (verbatim quote + neutral statement + signal type) and key facts (CFO, ERP, headcount…)."),
        ("Evaluate", "Each observation gets labels: quote really in the source? source tier A/B/C? fresh / aging / "
         "stale / undated? about this company? relevance DIRECT / INDIRECT / NONE? depends on a conflicting fact?"),
        ("Select angle", "Rules pick the outcome and the strongest hook (ordered rule, not a score). The AI then "
         "explains hook, hypothesis, product relevance and assumptions."),
        ("Draft", "60–120 words, only from the selected evidence, each sentence tagged with its evidence ids. "
         "Unverified prospects get no personal claims."),
        ("Validate", "Rule checks first (citations, numbers, no copying sources, <code>[Your name]</code> sign-off, "
         "length, banned phrases, conflicted values), then an AI sentence-level fact check; up to two automatic "
         "repairs; anything left is flagged, never hidden."),
        ("Review", "Saved as PENDING_REVIEW. A human edits, approves for manual send, or rejects."),
    ]
    st.markdown('<div class="card">' + "".join(f"<p><b>{i}. {n}</b> – {d}</p>" for i, (n, d) in enumerate(stages_txt, 1))
                + "</div>", unsafe_allow_html=True)

    st.markdown('<div class="sec">4 · Methodology: how it decides</div>', unsafe_allow_html=True)
    card("Principle", "<b>The AI reads and writes; rules decide.</b> No confidence scores – every evidence item gets "
         "categorical labels and every exclusion has a written reason, so any decision can be explained to a "
         "sales manager and is covered by automated tests. A credible signal is not automatically a relevant one.", "lime")
    st.table(pd.DataFrame([
        {"relevance": "DIRECT – can open the email", "signals": "Finance/AP hiring in the company's own team · change "
         "to its OWN ERP/finance system · finance transformation / shared services · publicly stated AP pain"},
        {"relevance": "INDIRECT – only as a question, always flagged", "signals": "New CFO/controller · IPO / listing "
         "preparation · M&A · expansion · rapid growth · the prospect's own finance commentary"},
        {"relevance": "NONE – never a hook", "signals": "Funding · product launches / marketing · awards / PR"},
    ]).set_index("relevance"))
    c1, c2 = st.columns(2)
    with c1:
        card("Source & time rules", "Tier A: company's own site, government/SEC filings, wire services, major business "
             "press. Tier C (never a hook, never proves identity): data brokers, scraped profiles, social/UGC. "
             "Everything else: tier B.<br>Fresh ≤ 12 months · aging 12–24 · stale &gt; 24 (excluded) · undated "
             "(allowed, flagged).<br>A quote must actually appear in the retrieved text, or it is treated as a "
             "hallucination and dropped.")
    with c2:
        card("Rules learned from live stress tests", "• Expansion plans inside funding news count as <b>funding</b>.<br>"
             "• INDIRECT hook + non-finance or unverified recipient = two guesses stacked → <b>abstain</b>.<br>"
             "• An earlier, different role at the same company is career history, not a conflict.<br>"
             "• Drafts may not copy 9+ words from a source.<br>• Sign-off must be <code>[Your name]</code>.<br>"
             "• Facts are compared by meaning (Oracle ERP = Oracle Fusion).", "pink")
    st.table(pd.DataFrame([{"outcome": OUTCOMES[k], "when": w} for k, w in [
        ("NEEDS_INPUT_AMBIGUOUS_IDENTITY", "Several organisations match, or the person can't be placed / sources disagree on their employment."),
        ("ABSTAIN_CONFLICTING_EVIDENCE", "Relevant signals depend on facts credible sources disagree on, or exist only on unreliable sources."),
        ("ABSTAIN_LOW_RELEVANCE", "Credible news exists but none links to AP – or an indirect hook to a non-buyer."),
        ("ABSTAIN_NO_CREDIBLE_EVIDENCE", "Nothing credible, current and about this company was found."),
        ("DRAFT_WITH_CAVEATS", "A usable hook, but something the reviewer must check (indirect link, undated source, unverified prospect…)."),
        ("DRAFT_READY", "A DIRECT, credible, current hook about a verified finance buyer.")]]).set_index("outcome"))

    st.markdown('<div class="sec">5 · Try it</div>', unsafe_allow_html=True)
    st.table(pd.DataFrame([
        {"try": "Happy path", "name": "Mark Fleming", "role": "CFO", "company": "Beca", "website": "https://www.beca.com",
         "what you'll see": "A verified CFO and a fresh finance-systems / AP signal → validated draft"},
        {"try": "Ambiguous identity", "name": "Sam Lee", "role": "Controller", "company": "Apex Solutions", "website": "",
         "what you'll see": "Stops at stage 1, lists the matching companies, one-click resolve"},
        {"try": "Credible but irrelevant", "name": "Varun Garg", "role": "Co-founder", "company": "Revspot", "website": "",
         "what you'll see": "Abstains: funding/product news, founder isn't an AP buyer"},
        {"try": "Nothing to find", "name": "Rohan Vexler", "role": "CFO", "company": "Quillmarsh Ledgerworks", "website": "",
         "what you'll see": "No credible evidence → no draft"},
    ]).set_index("try"))

    st.markdown('<div class="sec">6 · Stack, cost and limits</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        card("Built on free tiers (₹0)", "Search: Tavily → Serper → DuckDuckGo. AI: Groq (gpt-oss) → Gemini, with "
             "automatic fallback to another free model or provider on rate limits. Python · Streamlit · SQLite. "
             "Every run, stage output and review action is persisted.")
    with c2:
        card("Known limits", "Can't read LinkedIn or paywalled/JS-only pages · ~40–90 s per run on free tiers · "
             "public web results change daily · history resets when the hosted app restarts · no CRM sync or "
             "sending by design.")

# ------------------------------------------------------------------ history
else:
    header("Run history", "every run, its stages, decision and review state")
    runs = store.list_runs()
    if not runs:
        st.info("No runs yet.")
    else:
        outcome_tiles(runs)
        cards = []
        for r in runs[:3]:
            stg = store.get_stages(r["id"])
            done = sum(1 for x in stg if x["status"] in ("DONE", "WARNING"))
            o = r["outcome"] or ""
            cls = "lime" if o.startswith("DRAFT") else "pink" if o.startswith(("ABSTAIN", "NEEDS")) else ""
            label = "Run failed" if r["status"] == "FAILED" else SHORT.get(o, r["status"].title())
            cards.append(f'<div class="rcard {cls}"><div class="src">{r["company_name"]} · {r["prospect_role"]}</div>'
                         f'<div class="ttl">{r["prospect_name"]}</div><div class="row">'
                         f'<span class="chip">{label}</span><span class="big">{done}/8</span></div></div>')
        st.markdown('<div class="sec">Recent runs</div><div class="rcards">' + "".join(cards) + "</div>",
                    unsafe_allow_html=True)
        st.markdown('<div class="sec">All runs</div>', unsafe_allow_html=True)
        st.dataframe(runs_table(runs), hide_index=True, width="stretch")
        st.download_button("Download all runs (CSV)", export_csv(runs), file_name="apex_signal_runs.csv", mime="text/csv")
        labels = {r["id"]: f'{r["created_at"][:16].replace("T", " ")} · {r["prospect_name"]} @ {r["company_name"]}' for r in runs}
        sel = st.selectbox("Open run", list(labels), format_func=lambda k: labels[k])
        if sel:
            st.divider()
            render_run(sel)
