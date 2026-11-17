"""Run one LIVE pipeline from the terminal (real search + real LLM) and print the result.

Usage: python scripts/live_run.py "Prospect Name" "Company" "Role" [website]"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from apex_signal.config import load_settings  # noqa: E402
from apex_signal.db import RunStore  # noqa: E402
from apex_signal.llm import build_llm  # noqa: E402
from apex_signal.pipeline import Pipeline, ProspectInput  # noqa: E402
from apex_signal.search import build_search_client  # noqa: E402

if len(sys.argv) < 4:
    sys.exit(__doc__)
s = load_settings()
if s.missing_credentials():
    sys.exit("Missing credentials: " + ", ".join(s.missing_credentials()))
store = RunStore(s.db_path)
pipe = Pipeline(store, build_search_client(s), build_llm(s), max_sources=s.max_sources_for_extraction,
                on_event=lambda no, name, status, summary, _: print(f"[{status:8}] {no}. {name}  {summary or ''}"))
rid = pipe.run(ProspectInput(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None))
run = store.get_run(rid)
print(json.dumps({k: run[k] for k in ("id", "status", "providers", "outcome", "outcome_reason", "draft_subject", "draft_body", "error")}, indent=2))
