"""Raw-result storage (append-only), loading, scoring and processed-result export."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..dataset import load_experiment_cases
from ..evaluation.evaluators import LLMJudge, apply_human_labels, load_human_labels
from ..experiments import get_experiment
from ..utils.config import project_root
from ..utils.logging import get_logger

log = get_logger(__name__)


def resolve_dir(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else project_root() / p


def raw_path(results_dir: str | Path, experiment: str, model_key: str) -> Path:
    return resolve_dir(results_dir) / "raw" / f"{experiment}__{model_key}.jsonl"


def append_raw(results_dir: str | Path, record: dict[str, Any]) -> None:
    """Append one raw record. Raw files are never rewritten or truncated."""
    p = raw_path(results_dir, record["experiment"], record["model_key"])
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def read_raw(results_dir: str | Path, experiments: Iterable[str] | None = None,
             models: Iterable[str] | None = None) -> pd.DataFrame:
    rows: list[dict] = []
    for p in sorted((resolve_dir(results_dir) / "raw").glob("*__*.jsonl")):
        exp, model = p.stem.split("__", 1)
        if (experiments and exp not in experiments) or (models and model not in models):
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def completed_keys(results_dir: str | Path, experiment: str, model_key: str) -> set[str]:
    """Keys of successfully completed requests (errors are retried on resume)."""
    p = raw_path(results_dir, experiment, model_key)
    done: set[str] = set()
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if not r.get("error"):
                    done.add(r["request_key"])
    return done


def score_raw(
    raw: pd.DataFrame,
    exp_cfg: dict[str, Any],
    dataset_dir: str | Path,
    judge: LLMJudge | None = None,
    verified_only: bool = False,
    include_dirs: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Rebuild items from the dataset, join raw responses and apply each experiment's rubric.

    When several raw records share a request_key (e.g. retry after an error), the latest
    successful one wins.
    """
    if raw.empty:
        return pd.DataFrame()
    out_rows: list[dict] = []
    ok = raw[raw["error"].isna()] if "error" in raw else raw
    ok = ok.sort_values("timestamp").drop_duplicates("request_key", keep="last")
    for exp_name, g in ok.groupby("experiment"):
        cfg = {**(exp_cfg.get("experiments", {}).get(exp_name, {}))}
        exp = get_experiment(exp_name, cfg)
        cases = load_experiment_cases(exp_name, dataset_dir, cfg["dataset_file"], verified_only=verified_only,
                                     include_dirs=include_dirs)
        items = {it.item_id: it for it in exp.build_items(cases)}
        for rec in g.to_dict("records"):
            item = items.get(rec["item_id"])
            if item is None:
                log.warning("item %s not in current dataset; skipped", rec["item_id"])
                continue
            row = exp.score(item, rec["response"] or "")
            if judge and judge.applies(exp_name, rec):
                row.update(judge.judge(exp_name, item.prompt, rec["response"] or ""))
            base = {k: rec.get(k) for k in (
                "run_id", "timestamp", "experiment", "model_key", "provider", "model_id", "prompt_version",
                "dataset_version", "case_id", "item_id", "condition", "repeat", "response", "input_tokens",
                "output_tokens", "latency_s", "cost_usd", "error", "request_key", "temperature")}
            out_rows.append({**base, "evaluation_method": "deterministic" + ("+judge" if "judge_error" in row or any(k.startswith("judge_") for k in row) else ""), **row})
    return pd.DataFrame(out_rows)


def _json_default(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if o is pd.NA or o is pd.NaT:
        return None
    return str(o)


def _clean(v):
    if v is pd.NA or (isinstance(v, float) and v != v):
        return None
    return v


def save_processed(scored: pd.DataFrame, results_dir: str | Path) -> Path:
    d = resolve_dir(results_dir) / "processed"
    d.mkdir(parents=True, exist_ok=True)
    scored.to_csv(d / "scored.csv", index=False)
    with open(d / "scored.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in scored.to_dict("records"):
            f.write(json.dumps({k: _clean(v) for k, v in r.items()}, ensure_ascii=False, default=_json_default) + "\n")
    return d / "scored.csv"


def load_scored(results_dir: str | Path) -> pd.DataFrame:
    """Load processed scores (from JSONL so that booleans/None round-trip exactly)."""
    p = resolve_dir(results_dir) / "processed" / "scored.jsonl"
    if not p.exists():
        return pd.DataFrame()
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return pd.DataFrame(rows)


def export_review_queue(scored: pd.DataFrame, results_dir: str | Path, max_rows: int = 200) -> Path:
    """CSV of rows flagged `needs_review`, with a blank `label` column for human annotators.

    Fill `label` with pass/fail, save as results/human_labels.jsonl (see README) and re-run analysis.
    """
    q = scored[scored.get("needs_review", False) == True].copy()  # noqa: E712
    q = q.head(max_rows)[["item_id", "model_key", "repeat", "experiment", "response", "passed"]]
    q["label"] = ""
    p = resolve_dir(results_dir) / "processed" / "review_queue.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    q.to_csv(p, index=False)
    return p
