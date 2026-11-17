"""Smoke test of the Streamlit UI using FIXTURE clients (not live research)."""
import copy

import pytest
from streamlit.testing.v1 import AppTest

from tests import fixtures as fx


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-dummy")
    monkeypatch.setenv("TAVILY_API_KEY", "test-dummy")
    monkeypatch.setenv("APEX_DB_PATH", str(tmp_path / "ui.db"))
    import apex_signal.llm as llm_mod
    import apex_signal.search as search_mod
    monkeypatch.setattr(search_mod, "build_search_client", lambda s: fx.FixtureSearch(fx.HAPPY_PAGES))
    monkeypatch.setattr(llm_mod, "build_llm", lambda s: fx.FixtureLLM(copy.deepcopy(fx.HAPPY_LLM)))
    at = AppTest.from_file("../app.py", default_timeout=30)
    at.run()
    return at


def test_form_run_and_review(app):
    at = app
    assert not at.exception
    def field(prefix):
        return next(t for t in at.text_input if t.label.startswith(prefix))
    field("Prospect name").input("Priya Shah")
    field("Prospect role").input("CFO")
    field("Company name").input("Northwind Freight")
    field("Company website").input("https://northwindfreight.example")
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    assert any("Strongest eligible signal" in s.value for s in at.success)
    assert any("NetSuite" in t.value for t in at.text_area)
    # approve
    approve = next(b for b in at.button if b.label == "Approve for manual send")
    approve.click().run()
    assert not at.exception
    # history view lists the run
    at.radio[0].set_value("Run history").run()
    assert not at.exception
    assert at.dataframe[0].value.iloc[0]["review"] == "APPROVED"
