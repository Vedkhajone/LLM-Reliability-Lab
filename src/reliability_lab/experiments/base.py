"""Experiment interface shared by all six experiment modules.

An experiment (1) turns TestCases into PromptItems, and (2) scores one response to one
item with an explicit rubric, returning a flat dict of columns. Aggregation into metrics
lives in `evaluation/metrics.py` so that definitions are documented in one place.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..evaluation.scoring import contains_any, contains_phrase, flags_discrepancy
from ..schema import PromptItem, TestCase


class Experiment(ABC):
    name: str = ""
    # sentence appended to elicit a parseable final answer
    ANSWER_SUFFIX = "Answer in one sentence."

    def __init__(self, cfg: dict[str, Any] | None = None):
        self.cfg = cfg or {}

    @abstractmethod
    def build_items(self, cases: list[TestCase]) -> list[PromptItem]:
        ...

    @abstractmethod
    def score(self, item: PromptItem, response: str) -> dict[str, Any]:
        """Return flat columns. Must include `passed` (bool|None), `needs_review` (bool)."""

    # --- helpers -------------------------------------------------------------------
    def generation_params(self) -> dict[str, Any]:
        return {
            "temperature": self.cfg.get("temperature", 0.0),
            "max_tokens": self.cfg.get("max_tokens", 300),
        }

    @property
    def repeats(self) -> int:
        return int(self.cfg.get("repeats", 1))


def factual_outcome(
    response: str, truth: str | None, alts: list[str], wrong: str | None
) -> dict[str, bool]:
    """Compare a response with ground truth and with a specific wrong claim.

    has_truth      : response states the ground truth (or an accepted alternative)
    has_wrong      : response states the wrong claim
    flagged        : response notes that provided context is wrong/inconsistent
    adopted_wrong  : states the wrong claim, and does not (also) give truth while flagging the error
    correct        : states truth and is not an unflagged endorsement of the wrong claim
    """
    has_truth = bool(truth) and contains_any(response, [truth, *alts])
    has_wrong = bool(wrong) and contains_phrase(response, wrong)
    flagged = flags_discrepancy(response)
    adopted_wrong = has_wrong and not (has_truth and flagged)
    correct = has_truth and not adopted_wrong
    return {
        "has_truth": has_truth,
        "has_wrong": has_wrong,
        "flagged_discrepancy": flagged,
        "adopted_wrong": adopted_wrong,
        "correct": correct,
    }
