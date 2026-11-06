"""Validate dataset files: schema, uniqueness, duplicates, spec completeness, balance.

Exit code 0 = valid, 1 = errors (warnings never fail the run).
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import _common  # noqa: F401
from reliability_lab.dataset import load_jsonl
from reliability_lab.schema import EXPERIMENT_CATEGORY, TestCase
from reliability_lab.utils.config import load_experiments_config, project_root

REQUIRED_SPEC = {
    "instruction_conflicts": ["conflict_type", "acceptable"],
    "misleading_context": ["context_type", "provenance"],
    "uncertainty": ["subtype", "answerable"],
    "adversarial": ["attack_type", "task_id", "task_checks"],
    "context_sensitivity": ["contexts", "wrong_answer"],
    "response_stability": ["paraphrases"],
}


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", (s or "").lower()).strip()


def validate(dataset_dir: str, files: dict[str, str]) -> tuple[list[str], list[str], dict]:
    errors, warnings, stats = [], [], {}
    all_ids: Counter = Counter()
    for exp, fname in files.items():
        p = Path(dataset_dir) if Path(dataset_dir).is_absolute() else project_root() / dataset_dir
        p = p / fname
        if not p.exists():
            errors.append(f"{exp}: missing file {p}")
            continue
        rows = load_jsonl(p)
        cases = []
        for i, r in enumerate(rows, 1):
            try:
                cases.append(TestCase(**r))
            except Exception as e:  # noqa: BLE001
                errors.append(f"{fname}:{i}: schema error: {str(e).splitlines()[0]}")
        for c in cases:
            all_ids[c.id] += 1
            if c.category != EXPERIMENT_CATEGORY[exp]:
                errors.append(f"{c.id}: category {c.category} != {EXPERIMENT_CATEGORY[exp]}")
            for k in REQUIRED_SPEC[exp]:
                if k not in c.spec:
                    errors.append(f"{c.id}: spec missing '{k}'")
            if exp in ("misleading_context", "context_sensitivity", "response_stability") and not c.ground_truth:
                errors.append(f"{c.id}: ground_truth required")
            if c.ground_truth and not c.ground_truth_source:
                warnings.append(f"{c.id}: no ground_truth_source")
            if exp == "adversarial" and c.spec.get("canary") and c.spec["canary"] not in (c.context or ""):
                errors.append(f"{c.id}: canary not present in context")
        qs = Counter(_norm(c.question) for c in cases if exp not in ("adversarial",))
        for q, n in qs.items():
            if n > 1 and exp not in ("instruction_conflicts",):
                warnings.append(f"{exp}: duplicate question text x{n}: {q[:60]}")
        stats[exp] = {
            "n": len(cases),
            "difficulty": dict(Counter(c.difficulty for c in cases)),
            "review": dict(Counter(c.human_review_status for c in cases)),
            "origin": dict(Counter(c.origin for c in cases)),
        }
    errors += [f"duplicate id {i} x{n}" for i, n in all_ids.items() if n > 1]
    return errors, warnings, stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir")
    a = ap.parse_args(argv)
    cfg = load_experiments_config()
    files = {k: v["dataset_file"] for k, v in cfg["experiments"].items()}
    errors, warnings, stats = validate(a.dataset_dir or cfg["run"]["dataset_dir"], files)
    total = sum(s["n"] for s in stats.values())
    print(f"Cases per experiment: { {k: v['n'] for k, v in stats.items()} } (total {total})")
    for k, v in stats.items():
        print(f"  {k}: difficulty={v['difficulty']} review={v['review']}")
    for w in warnings:
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    print(f"{len(errors)} error(s), {len(warnings)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
