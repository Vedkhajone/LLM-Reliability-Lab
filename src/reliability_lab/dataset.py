"""Dataset loading, validation and versioning helpers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from .schema import EXPERIMENT_CATEGORY, TestCase
from .utils.config import project_root


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else project_root() / p


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(_resolve(path), "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i}: invalid JSON: {e}") from e
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=False) + "\n")


def load_cases(path: str | Path) -> list[TestCase]:
    return [TestCase(**r) for r in load_jsonl(path)]


def load_experiment_cases(
    experiment: str,
    dataset_dir: str | Path,
    dataset_file: str,
    verified_only: bool = False,
    include_dirs: tuple[str, ...] = (),
) -> list[TestCase]:
    """Load cases for an experiment from the generated dir (and optional extra dirs).

    `verified_only=True` keeps cases whose human_review_status == "verified"; the
    generated seed set is "unreviewed" until a human has checked it.
    """
    cases: list[TestCase] = []
    for d in (dataset_dir, *include_dirs):
        p = _resolve(d) / dataset_file
        if p.exists():
            cases.extend(load_cases(p))
    cat = EXPERIMENT_CATEGORY[experiment]
    seen: dict[str, TestCase] = {}
    for c in cases:  # later dirs (human-verified) override generated cases with the same id
        seen[c.id] = c
    cases = [c for c in seen.values() if c.category == cat]
    if verified_only:
        cases = [c for c in cases if c.human_review_status == "verified"]
    return cases


def dataset_fingerprint(paths: Iterable[str | Path]) -> str:
    """Short content hash for provenance (stored in dataset card / reports)."""
    h = hashlib.sha256()
    for p in sorted(str(_resolve(x)) for x in paths):
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:12]
