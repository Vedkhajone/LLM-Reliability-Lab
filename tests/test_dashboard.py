"""Dashboard smoke tests using Streamlit's headless AppTest (no browser needed)."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

from reliability_lab.evaluation.failures import add_failure_categories, failure_category  # noqa: E402
from reliability_lab.utils.config import project_root  # noqa: E402

ROOT = project_root()
APP = str(ROOT / "dashboard" / "app.py")


def test_dashboard_empty_state_does_not_crash(tmp_path):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception


@pytest.fixture()
def smoke_results():
    d = ROOT / "results_pytest_dash"
    shutil.rmtree(d, ignore_errors=True)
    run = lambda *a: subprocess.run([sys.executable, *a], capture_output=True, text=True, cwd=ROOT)  # noqa: E731
    r = run("scripts/run_all.py", "--smoke", "--models", "mock", "--yes", "--results-dir", "results_pytest_dash")
    assert r.returncode == 0, r.stderr[-500:]
    r = run("scripts/analyse_results.py", "--results-dir", "results_pytest_dash", "--no-embeddings")
    assert r.returncode == 0, r.stderr[-800:]
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_dashboard_renders_with_data(smoke_results):
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    at.sidebar.selectbox[0].set_value("results_pytest_dash").run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 6
    labels = [m.label for m in at.metric]
    assert "Raw requests" in labels and "Models evaluated" in labels
    assert any("Mock data present" in m.value for m in at.markdown)


def test_failure_categories():
    row = {"experiment": "misleading_context", "passed": False, "adopted_wrong": True}
    assert failure_category(row) == "adopted_false_context"
    assert failure_category({"experiment": "uncertainty", "passed": False, "answerable": False, "hallucination": True}) \
        == "hallucination_on_unanswerable"
    assert failure_category({"experiment": "uncertainty", "passed": True}) is None
    assert failure_category({"experiment": "adversarial", "passed": False, "attack_success": True}) == "injection_followed"
