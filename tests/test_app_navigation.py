"""Smoke tests for the client-facing Streamlit navigation."""
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP = Path(__file__).parents[1] / "app" / "streamlit_app.py"


def test_briefing_explore_and_evidence_routes_render():
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    assert not app.exception
    assert app.segmented_control[0].value == "Briefing"
    assert app.title[0].value == "Where might manufacturing employment contract severely?"

    app.segmented_control[0].set_value("Explore").run()
    assert not app.exception
    assert app.segmented_control[1].value == "A single case"
    assert app.title[0].value == "Case explorer"

    app.segmented_control[0].set_value("Evidence").run()
    app.segmented_control[1].set_value("Methods & limitations").run()
    assert not app.exception
    assert app.title[0].value == "Methods & limitations"
