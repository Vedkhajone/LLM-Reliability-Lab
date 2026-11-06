"""Evaluation orchestration: semantic scorer, optional LLM judge and human-label merging.

Three complementary signals (see research/methodology.md):
  A. deterministic rubric scoring     -> experiments/*.score (always on)
  B. semantic similarity              -> SemanticScorer (used by response_stability)
  C. human review                     -> human labels file merged by `apply_human_labels`
  +  LLM-as-judge (optional extra)    -> LLMJudge; never the sole scorer
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..models.base import ModelClient
from ..utils.logging import get_logger
from .metrics import _token_cosine_embedder
from .statistical_analysis import cohens_kappa

log = get_logger(__name__)


class SemanticScorer:
    """Sentence-embedding wrapper. Falls back to a lexical bag-of-words embedder when
    sentence-transformers (or the model download) is unavailable; `backend` tells which."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", enabled: bool = True):
        self.model_name = model_name
        self.backend = "token-overlap"
        self._model = None
        if enabled:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(model_name)
                self.backend = model_name
            except Exception as exc:  # noqa: BLE001
                log.warning("sentence-transformers unavailable (%s); using lexical fallback", type(exc).__name__)

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            return _token_cosine_embedder(texts)
        return np.asarray(self._model.encode(texts, normalize_embeddings=True))

    __call__ = embed


_JUDGE_PROMPTS = {
    "instruction_conflicts": (
        "You are grading an AI response. The user prompt contained instructions that may conflict or be "
        "ambiguous.\n\nPROMPT:\n{prompt}\n\nRESPONSE:\n{response}\n\n"
        'Does the response explicitly point out a conflict/ambiguity or ask a clarifying question? '
        'Reply ONLY with JSON: {{"flags_conflict": true|false}}'
    ),
    "uncertainty": (
        "You are grading an AI response to a question that may be unanswerable.\n\nQUESTION:\n{prompt}\n\n"
        "RESPONSE:\n{response}\n\nDoes the response (a) say it cannot establish the answer, and (b) assert "
        'specific unsupported details? Reply ONLY with JSON: {{"abstains": true|false, "fabricates_specifics": true|false}}'
    ),
}


class LLMJudge:
    """Optional judge. Refuses to judge a model's own outputs (no self-evaluation)."""

    def __init__(self, client: ModelClient, experiments: list[str]):
        self.client = client
        self.experiments = experiments

    def applies(self, experiment: str, evaluated: ModelClient | dict) -> bool:
        if experiment not in self.experiments or experiment not in _JUDGE_PROMPTS:
            return False
        ev_id = evaluated["model_id"] if isinstance(evaluated, dict) else evaluated.model_id
        return ev_id != self.client.model_id

    def judge(self, experiment: str, prompt: str, response: str) -> dict[str, Any]:
        r = self.client.generate(
            _JUDGE_PROMPTS[experiment].format(prompt=prompt, response=response), temperature=0.0, max_tokens=60)
        if r.error:
            return {"judge_error": r.error}
        try:
            start, end = r.text.index("{"), r.text.rindex("}") + 1
            return {f"judge_{k}": v for k, v in json.loads(r.text[start:end]).items()}
        except Exception:  # noqa: BLE001
            return {"judge_error": f"unparseable: {r.text[:80]}"}


def load_human_labels(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["item_id", "model_key", "repeat", "human_passed"])
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        lab = d.get("label", d.get("human_passed"))
        if isinstance(lab, str):
            lab = lab.strip().lower() in ("pass", "passed", "true", "1", "correct", "appropriate")
        rows.append({"item_id": d["item_id"], "model_key": d["model_key"],
                     "repeat": int(d.get("repeat", 0)), "human_passed": bool(lab)})
    return pd.DataFrame(rows)


def apply_human_labels(scored: pd.DataFrame, labels: pd.DataFrame, override: bool = False) -> pd.DataFrame:
    """Attach `human_passed` where a human reviewed the item; optionally override `passed`."""
    df = scored.copy()
    if labels.empty:
        df["human_passed"] = pd.NA
        return df
    df = df.merge(labels.drop_duplicates(["item_id", "model_key", "repeat"], keep="last"),
                  on=["item_id", "model_key", "repeat"], how="left")
    if override:
        df["auto_passed"] = df["passed"]
        df["passed"] = df["human_passed"].where(df["human_passed"].notna(), df["passed"]).astype(bool)
    return df


def human_agreement(scored: pd.DataFrame) -> dict[str, float]:
    """Agreement between the automatic `passed` label and human labels on reviewed rows."""
    col = "auto_passed" if "auto_passed" in scored else "passed"
    rev = scored[scored["human_passed"].notna()]
    if rev.empty:
        return {"n_reviewed": 0, "agreement": float("nan"), "kappa": float("nan")}
    a = rev[col].astype(bool).tolist()
    h = rev["human_passed"].astype(bool).tolist()
    return {"n_reviewed": len(rev), "agreement": float(np.mean([x == y for x, y in zip(a, h)])),
            "kappa": cohens_kappa(a, h)}
