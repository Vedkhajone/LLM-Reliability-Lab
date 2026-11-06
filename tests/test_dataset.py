import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from reliability_lab.dataset import load_cases, load_jsonl
from reliability_lab.evaluation.scoring import contains_phrase as has
from reliability_lab.schema import TestCase
from reliability_lab.utils.config import load_experiments_config, project_root

ROOT = project_root()
CFG = load_experiments_config()
FILES = {k: v["dataset_file"] for k, v in CFG["experiments"].items()}


def _all_cases():
    out = []
    for exp, f in FILES.items():
        out += [(exp, c) for c in load_cases(ROOT / "datasets/generated" / f)]
    return out


def test_validate_script_passes():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/validate_dataset.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_ids_unique_and_prefixed():
    ids = [c.id for _, c in _all_cases()]
    assert len(ids) == len(set(ids))
    assert all(re.match(r"^[A-Z]{3}_\d{3}$", i) for i in ids)


def test_all_cases_unreviewed_and_marked_model_generated():
    """Honesty guard: the seed set must not claim human verification."""
    for _, c in _all_cases():
        assert c.human_review_status == "unreviewed"
        assert c.origin == "model_generated"
        assert c.dataset_version == CFG["run"]["dataset_version"]


def test_factual_cases_have_truth_and_source():
    for exp, c in _all_cases():
        if exp in ("misleading_context", "context_sensitivity", "response_stability"):
            assert c.ground_truth and c.ground_truth_source


def test_misleading_wrong_claim_in_context():
    for exp, c in _all_cases():
        if exp == "misleading_context" and c.spec["wrong_answer"]:
            assert has(c.context, c.spec["wrong_answer"])
        if exp == "context_sensitivity":
            ctx = c.spec["contexts"]
            assert has(ctx["relevant"], c.ground_truth)
            assert not has(ctx["misleading"], c.ground_truth)
            assert has(ctx["misleading"], c.spec["wrong_answer"])


def test_no_question_leaks_answer():
    for exp, c in _all_cases():
        if exp in ("misleading_context", "context_sensitivity", "response_stability"):
            assert not has(c.question, c.ground_truth), c.id
            if exp == "response_stability":
                for p in c.spec["paraphrases"]:
                    assert not has(p, c.ground_truth), (c.id, p)


def test_no_duplicate_questions_within_experiment():
    for exp in FILES:
        qs = [re.sub(r"\W+", " ", c.question.lower()) for e, c in _all_cases() if e == exp and exp != "adversarial"]
        qs = [q for q in qs if q]
        if exp not in ("instruction_conflicts",):  # order-swapped pairs differ textually but are intentional
            assert len(qs) == len(set(qs)), exp


def test_fact_reuse_across_experiments_is_avoided():
    """Different experiments should not ask the same factual question (confound / leakage)."""
    seen = {}
    for exp, c in _all_cases():
        if exp in ("misleading_context", "context_sensitivity", "response_stability", "uncertainty") and c.ground_truth:
            key = c.ground_truth.lower()
            if key in seen and seen[key] != exp:
                pytest.fail(f"ground truth '{key}' reused across {seen[key]} and {exp}")
            seen[key] = exp


def test_difficulty_variety_exists():
    levels = {c.difficulty for _, c in _all_cases()}
    assert levels == {"easy", "medium", "hard"}


def test_jsonl_roundtrip_is_valid_json_lines():
    for f in FILES.values():
        for r in load_jsonl(ROOT / "datasets/generated" / f):
            json.dumps(r)
            TestCase(**r)


def test_builder_is_deterministic(tmp_path):
    before = {f: (ROOT / "datasets/generated" / f).read_bytes() for f in FILES.values()}
    r = subprocess.run([sys.executable, str(ROOT / "scripts/build_dataset.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    after = {f: (ROOT / "datasets/generated" / f).read_bytes() for f in FILES.values()}
    assert before == after
