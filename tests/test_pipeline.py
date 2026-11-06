"""Integration tests with mocked model responses (no network, no API keys)."""
import json

import numpy as np
import pandas as pd
import pytest

from reliability_lab.dataset import load_experiment_cases
from reliability_lab.evaluation.metrics import compute_metrics
from reliability_lab.evaluation.evaluators import apply_human_labels, human_agreement, load_human_labels
from reliability_lab.models import ModelSpec
from reliability_lab.models.mock_client import MockClient
from reliability_lab.pipeline.experiment_runner import ExperimentRunner
from reliability_lab.pipeline.report_generator import paired_comparisons
from reliability_lab.pipeline.result_collector import (
    export_review_queue, read_raw, save_processed, load_scored, score_raw,
)
from reliability_lab.schema import EXPERIMENTS
from reliability_lab.utils.config import load_experiments_config

CFG = load_experiments_config()


def _truth_index():
    idx = []
    for exp in ("misleading_context", "context_sensitivity", "response_stability", "uncertainty"):
        for c in load_experiment_cases(exp, CFG["run"]["dataset_dir"], CFG["experiments"][exp]["dataset_file"]):
            qs = [c.question] + c.spec.get("paraphrases", [])
            idx.append((qs, c.ground_truth, bool(c.spec.get("answerable", True)) if exp == "uncertainty" else True))
    return idx


TRUTH = _truth_index()


def oracle(prompt, system):
    """Well-behaved synthetic model: right answer, abstains when unanswerable, ignores injections."""
    for qs, truth, answerable in TRUTH:
        if any(q in prompt for q in qs):
            if not answerable:
                return "I'm not aware of this and cannot verify it.\nAnswer: unknown\nConfidence: 80%"
            return f"{truth}.\nAnswer: {truth}\nConfidence: 90%"
    if "invoice" in prompt.lower():
        return "The total due is $4,250."
    if "sentiment" in prompt.lower():
        return "Positive."
    if "Summarise the document" in prompt:
        return "Volcanoes form where magma reaches the surface."
    return "These instructions conflict, so I cannot satisfy both."


def make_runner(tmp_path, responder, model_key="mock", smoke=False, repeats=None):
    spec = ModelSpec(key=model_key, provider="mock", model_id=f"mock-{model_key}", requests_per_minute=0)
    cfg = json.loads(json.dumps(CFG))
    if repeats:
        cfg["experiments"]["response_stability"]["repeats"] = repeats
    client = MockClient(spec, responder=responder)
    return ExperimentRunner({model_key: spec}, exp_cfg=cfg, results_dir=str(tmp_path / "res"),
                            clients={model_key: client}, smoke=smoke), cfg, client


def run_all(runner, **kw):
    plan = runner.plan(EXPERIMENTS, list(runner.specs), **kw)
    runner.execute(plan, max_requests=100000)
    return plan


def test_smoke_plan_is_small(tmp_path):
    runner, *_ = make_runner(tmp_path, oracle, smoke=True)
    plan = runner.plan(EXPERIMENTS, ["mock"])
    assert 0 < len(plan.requests) < 80  # 3 cases/experiment, 1 repeat


def test_full_plan_sizes(tmp_path):
    runner, *_ = make_runner(tmp_path, oracle)
    plan = runner.plan(EXPERIMENTS, ["mock"])
    per = {}
    for r in plan.requests:
        per[r.experiment] = per.get(r.experiment, 0) + 1
    assert per["instruction_conflicts"] == 18 and per["adversarial"] == 21 and per["context_sensitivity"] == 108
    assert per["response_stability"] == len([1 for r in plan.requests if r.experiment == "response_stability"]) > 18 * 4 * 3


def test_dry_plan_does_not_write(tmp_path):
    runner, *_ = make_runner(tmp_path, oracle)
    runner.plan(EXPERIMENTS, ["mock"])
    assert not (tmp_path / "res").exists()


def test_end_to_end_oracle(tmp_path):
    runner, cfg, client = make_runner(tmp_path, oracle, repeats=2)
    run_all(runner)
    raw = read_raw(runner.results_dir)
    for col in ("run_id", "timestamp", "provider", "model_id", "prompt_version", "dataset_version", "temperature",
                "prompt", "response", "input_tokens", "output_tokens", "latency_s", "error", "request_key"):
        assert col in raw.columns
    assert raw["error"].isna().all() and raw["model_id"].eq("mock-mock").all()
    scored = score_raw(raw, cfg, cfg["run"]["dataset_dir"])
    assert len(scored) == len(raw)
    save_processed(scored, runner.results_dir)
    reloaded = load_scored(runner.results_dir)
    assert len(reloaded) == len(scored) and reloaded["passed"].dtype == bool

    m = compute_metrics(scored, {"bootstrap_resamples": 200})
    get = lambda exp, metric, subset="all": m[(m.experiment == exp) & (m.metric == metric) & (m.subset == subset)]["value"].iloc[0]  # noqa: E731
    assert get("misleading_context", "factual_accuracy") == 1.0
    assert get("misleading_context", "misleading_susceptibility") == 0.0
    assert get("uncertainty", "appropriate_abstention_rate") == 1.0
    assert get("uncertainty", "hallucination_rate") == 0.0
    assert get("uncertainty", "false_abstention_rate") == 0.0
    assert get("adversarial", "attack_success_rate") == 0.0
    assert get("adversarial", "task_preservation_rate") == 1.0
    for cond in ("C0", "C1", "C2", "C3", "C4", "C5"):
        assert get("context_sensitivity", "accuracy", cond) == 1.0
    assert get("context_sensitivity", "context_sensitivity_score", "C1..C5") == 0.0
    assert get("response_stability", "answer_agreement") == 1.0
    assert get("response_stability", "prompt_wording_sensitivity") == 0.0
    assert get("response_stability", "repeat_agreement") == 1.0
    assert get("instruction_conflicts", "conflict_recognition_rate") == 1.0
    # oracle flags everything, so it over-flags the non-conflicting controls
    assert get("instruction_conflicts", "over_flag_rate") == 1.0
    assert (m["ci_low"].dropna() <= m["value"][m["ci_low"].notna()] + 1e-9).all()


def test_naive_context_follower_is_exposed(tmp_path):
    from reliability_lab.models.mock_client import default_responder
    runner, cfg, _ = make_runner(tmp_path, default_responder)
    run_all(runner)
    scored = score_raw(read_raw(runner.results_dir), cfg, cfg["run"]["dataset_dir"])
    m = compute_metrics(scored, {"bootstrap_resamples": 200})
    sus = m[(m.experiment == "misleading_context") & (m.metric == "misleading_susceptibility") & (m.subset == "all")]
    assert sus["value"].iloc[0] > 0.5
    acc = m[(m.experiment == "context_sensitivity") & (m.metric == "accuracy")].set_index("subset")["value"]
    assert acc["C1"] > 0.9 and acc["C3"] == 0.0   # a context-follower is right with true context, wrong with false


def test_resume_skips_completed_and_retries_errors(tmp_path):
    calls = {"n": 0}
    def flaky(prompt, system):
        calls["n"] += 1
        if calls["n"] <= 5:
            raise ValueError("boom")
        return oracle(prompt, system)

    runner, cfg, _ = make_runner(tmp_path, flaky, smoke=True)
    plan1 = runner.plan(["misleading_context", "uncertainty"], ["mock"])
    runner.execute(plan1)
    raw1 = read_raw(runner.results_dir)
    assert raw1["error"].notna().sum() == 5          # failures stored, not dropped
    n_raw1 = len(raw1)

    runner2, _, _ = make_runner(tmp_path, oracle, smoke=True)
    plan2 = runner2.plan(["misleading_context", "uncertainty"], ["mock"])
    assert len(plan2.requests) == 5                  # only the 5 failed ones are redone
    runner2.execute(plan2)
    raw2 = read_raw(runner2.results_dir)
    assert len(raw2) == n_raw1 + 5                   # append-only: nothing overwritten
    key = lambda df: set(zip(df["timestamp"], df["request_key"], df["error"].fillna("")))  # noqa: E731
    assert key(raw1) <= key(raw2)                    # every earlier record is still there, unchanged
    plan3 = runner2.plan(["misleading_context", "uncertainty"], ["mock"])
    assert len(plan3.requests) == 0 and plan3.skipped_done == len(plan1.requests)
    scored = score_raw(raw2, CFG, CFG["run"]["dataset_dir"])
    assert len(scored) == len(plan1.requests)        # errors excluded, latest success wins


def test_budget_stops_run_and_is_resumable(tmp_path):
    runner, *_ = make_runner(tmp_path, oracle)
    plan = runner.plan(["adversarial"], ["mock"])
    tracker = runner.execute(plan, max_requests=4)
    assert tracker.requests == 4 and len(read_raw(runner.results_dir)) == 4
    assert len(runner.plan(["adversarial"], ["mock"]).requests) == len(plan.requests) - 4


def test_limit_and_changed_prompt_version_rerun(tmp_path):
    runner, cfg, _ = make_runner(tmp_path, oracle)
    p = runner.plan(["uncertainty"], ["mock"], limit=5)
    assert len(p.requests) == 5
    runner.execute(p)
    runner.run["prompt_version"] = "v2"      # new prompt version => new request keys, old data kept
    p2 = runner.plan(["uncertainty"], ["mock"], limit=5)
    assert len(p2.requests) == 5


def test_paired_comparison_between_two_models(tmp_path):
    good, cfg, _ = make_runner(tmp_path, oracle, model_key="good")
    bad_responder = lambda p, s: "Probably something else entirely.\nAnswer: no idea"  # noqa: E731
    bad, _, _ = make_runner(tmp_path, bad_responder, model_key="bad")
    for r in (good, bad):
        r.execute(r.plan(["misleading_context", "context_sensitivity"], list(r.specs)))
    scored = score_raw(read_raw(good.results_dir), cfg, cfg["run"]["dataset_dir"])
    comps = paired_comparisons(scored)
    mm = comps[(comps.family == "model_vs_model") & (comps.experiment == "misleading_context")].iloc[0]
    assert (mm["a"], mm["b"]) == ("bad", "good")   # sorted model order
    assert mm["diff"] < -0.8 and mm["p_value"] < 0.001 and mm["n01"] > 0 and mm["n10"] == 0
    assert "p_adjusted" in comps and (comps["p_adjusted"] >= comps["p_value"] - 1e-12).all()
    cb = comps[comps.family == "condition_vs_baseline"]
    assert not cb.empty


def test_human_labels_and_review_queue(tmp_path):
    runner, cfg, _ = make_runner(tmp_path, lambda p, s: "The paper found great things.\nConfidence: 90%")
    runner.execute(runner.plan(["uncertainty"], ["mock"], limit=6))
    scored = score_raw(read_raw(runner.results_dir), cfg, cfg["run"]["dataset_dir"])
    q = export_review_queue(scored, runner.results_dir)
    queue = pd.read_csv(q)
    assert len(queue) > 0 and "label" in queue.columns
    first = scored.iloc[0]
    labels_file = tmp_path / "labels.jsonl"
    labels_file.write_text(json.dumps({"item_id": first["item_id"], "model_key": "mock", "repeat": 0, "label": "pass"}) + "\n")
    labelled = apply_human_labels(scored, load_human_labels(labels_file))
    ag = human_agreement(labelled)
    assert ag["n_reviewed"] == 1 and ag["agreement"] in (0.0, 1.0)
    over = apply_human_labels(scored, load_human_labels(labels_file), override=True)
    assert bool(over.loc[over.item_id == first["item_id"], "passed"].iloc[0]) is True
    assert "auto_passed" in over


def test_full_analysis_report(tmp_path):
    from reliability_lab.pipeline.report_generator import run_analysis
    runner, cfg, _ = make_runner(tmp_path, oracle, repeats=2)
    run_all(runner)
    scored = score_raw(read_raw(runner.results_dir), cfg, cfg["run"]["dataset_dir"])
    save_processed(scored, runner.results_dir)
    out = run_analysis(runner.results_dir, {"statistics": {"bootstrap_resamples": 100}, "semantic": {"enabled": False},
                                            "human_review": {"labels_file": str(tmp_path / "none.jsonl")}},
                       use_embeddings=False)
    root = tmp_path / "res"
    assert (root / "statistics" / "metrics.csv").exists() and (root / "statistics" / "report.md").exists()
    assert len(out["figures"]) >= 6 and all(p.exists() for p in out["figures"])
    assert "MOCK" in out["report"]          # mock data must be flagged in any report


def test_paper_template_population(tmp_path, monkeypatch):
    from reliability_lab.pipeline.report_generator import populate_paper
    from reliability_lab.utils.config import project_root
    tpl = (project_root() / "research" / "research_paper_template.md").read_text(encoding="utf-8")
    assert "<!-- AUTO:RESULTS -->" in tpl
    out = populate_paper("## fake report\n| a |", template="research/research_paper_template.md",
                         out=str(tmp_path / "draft.md"))
    text = out.read_text(encoding="utf-8")
    assert "fake report" in text and "<!-- AUTO:RESULTS -->" not in text
    for sec in ["Abstract", "Introduction", "Limitations", "Ethical Considerations", "References"]:
        assert sec in text


def test_confirmation_refuses_large_runs_without_yes(monkeypatch):
    from reliability_lab.pipeline.experiment_runner import PlannedRequest, RunPlan, confirm
    plan = RunPlan(requests=[object()] * 500)
    assert confirm(plan, 150, assume_yes=True) is True
    assert confirm(RunPlan(requests=[object()] * 10), 150, assume_yes=False) is True      # small run: no prompt
    monkeypatch.setattr("sys.stdin.isatty", lambda: False, raising=False)
    assert confirm(plan, 150, assume_yes=False) is False                                  # non-interactive
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    monkeypatch.setattr("builtins.input", lambda *_: (_ for _ in ()).throw(EOFError()))
    assert confirm(plan, 150, assume_yes=False) is False                                  # EOF
    monkeypatch.setattr("builtins.input", lambda *_: "n")
    assert confirm(plan, 150, assume_yes=False) is False
    monkeypatch.setattr("builtins.input", lambda *_: "y")
    assert confirm(plan, 150, assume_yes=False) is True


def test_raw_records_store_params_applied(tmp_path):
    runner, *_ = make_runner(tmp_path, oracle)
    runner.execute(runner.plan(["uncertainty"], ["mock"], limit=2))
    raw = read_raw(runner.results_dir)
    assert raw["params_applied"].iloc[0] == {"temperature": 0.0, "seed": 1234}
