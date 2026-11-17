"""Apex Signal – Streamlit UI. Run with: streamlit run app.py"""
from __future__ import annotations

import os
import time
from datetime import date

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
    view = st.radio("View", ["New run", "Run history"], label_visibility="collapsed", key="view")
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
                    st.markdown(f"**{sid}** [{s['title'] or s['url']}]({s['url']}) · {s.get('published_date') or 'undated'} "
                                f"· query: _{s.get('query', '')}_")

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


# ------------------------------------------------------------------ new run
if view == "New run":
    st.markdown(f'<div class="hello">New prospect run</div><div class="sub">{date.today():%A, %B %d, %Y} · '
                'Researches live public sources, applies explicit evidence rules, then drafts for review or abstains with a reason.</div>',
                unsafe_allow_html=True)
    with st.form("prospect"):
        c1, c2 = st.columns(2)
        pn = c1.text_input("Prospect name *", placeholder="e.g. Jane Doe")
        role = c2.text_input("Prospect role *", placeholder="e.g. CFO, Controller, AP Manager")
        cn = c1.text_input("Company name *", placeholder="e.g. Acme Logistics")
        site = c2.text_input("Company website (optional, resolves ambiguity)", placeholder="https://")
        go = st.form_submit_button("Run research", type="primary", disabled=bool(missing))

    if go:
        if not (pn.strip() and cn.strip() and role.strip()):
            st.error("Prospect name, company name and role are required.")
        else:
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
                pipe = Pipeline(store, build_search_client(settings), build_llm(settings),
                                max_sources=settings.max_sources_for_extraction, on_event=on_event)
                rid = pipe.run(ProspectInput(pn.strip(), cn.strip(), role.strip(), site.strip() or None))
                st.session_state["last_run"] = rid
            except Exception as e:
                st.error(f"Could not start run: {e}")
    if st.session_state.get("last_run"):
        st.divider()
        render_run(st.session_state["last_run"])

# ------------------------------------------------------------------ history
else:
    st.markdown(f'<div class="hello">Run history</div><div class="sub">{date.today():%A, %B %d, %Y} · every run, its stages, decision and review state</div>',
                unsafe_allow_html=True)
    runs = store.list_runs()
    if not runs:
        st.info("No runs yet.")
    else:
        n_draft = sum(1 for r in runs if r["draft_body"])
        n_abst = sum(1 for r in runs if (r["outcome"] or "").startswith(("ABSTAIN", "NEEDS")))
        n_appr = sum(1 for r in runs if r["review_status"] == "APPROVED")
        st.markdown(f'''<div class="tiles">
          <div class="tile dark"><div class="lbl">Total runs</div><div class="num">{len(runs)}</div></div>
          <div class="tile lime"><div class="lbl">Drafts produced</div><div class="num">{n_draft}</div></div>
          <div class="tile pink"><div class="lbl">Abstained / needs input</div><div class="num">{n_abst}</div></div>
          <div class="tile"><div class="lbl">Approved for manual send</div><div class="num">{n_appr}</div></div>
        </div>''', unsafe_allow_html=True)
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
        df = pd.DataFrame([{
            "run": r["id"], "created (UTC)": r["created_at"].replace("T", " ")[:16], "prospect": r["prospect_name"],
            "company": r["company_name"], "role": r["prospect_role"], "status": r["status"],
            "outcome": OUTCOMES.get(r["outcome"] or "", r["outcome"] or (r["error"] or "")[:60]),
            "review": r["review_status"], "subject": r["edited_subject"] or ""} for r in runs])
        st.dataframe(df, hide_index=True, width="stretch")
        labels = {r["id"]: f'{r["created_at"][:16].replace("T", " ")} · {r["prospect_name"]} @ {r["company_name"]}' for r in runs}
        sel = st.selectbox("Open run", list(labels), format_func=lambda k: labels[k])
        if sel:
            st.divider()
            render_run(sel)
