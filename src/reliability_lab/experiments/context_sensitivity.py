"""Experiment 5: context sensitivity and degradation.

Conditions per base question:
  C0 question only | C1 relevant | C2 irrelevant | C3 misleading | C4 excessive | C5 contradictory
C4 buries the relevant sentence in the middle of N neutral filler paragraphs (so it tests
dilution, not missing information). All conditions use the same instruction wording.
"""
from __future__ import annotations

from typing import Any

from ..schema import PromptItem, TestCase
from ._filler import pick_filler
from .base import Experiment, factual_outcome

CONDITIONS = ["C0", "C1", "C2", "C3", "C4", "C5"]
CONDITION_NAMES = {
    "C0": "question_only", "C1": "relevant", "C2": "irrelevant",
    "C3": "misleading", "C4": "excessive", "C5": "contradictory",
}


class ContextSensitivity(Experiment):
    name = "context_sensitivity"

    def _context(self, c: TestCase, cond: str) -> str | None:
        ctx = c.spec["contexts"]
        if cond == "C0":
            return None
        if cond == "C1":
            return ctx["relevant"]
        if cond == "C2":
            return ctx["irrelevant"]
        if cond == "C3":
            return ctx["misleading"]
        if cond == "C5":
            return ctx["contradictory"]
        if cond == "C4":
            n = int(self.cfg.get("excessive_paragraphs", 14))
            paras = pick_filler(c.id, n)
            mid = len(paras) // 2
            paras.insert(mid, ctx["relevant"])
            return "\n\n".join(paras)
        raise ValueError(cond)

    def build_items(self, cases: list[TestCase]) -> list[PromptItem]:
        items = []
        for c in cases:
            for cond in CONDITIONS:
                ctx = self._context(c, cond)
                prompt = (f"Context:\n{ctx}\n\n" if ctx else "") + f"Question: {c.question}\n{self.ANSWER_SUFFIX}"
                items.append(PromptItem(
                    item_id=f"{c.id}::{cond}", case_id=c.id, experiment=self.name, condition=cond,
                    prompt=prompt, system=c.system_prompt,
                    meta={
                        "ground_truth": c.ground_truth, "alts": c.acceptable_alternatives,
                        "wrong_answer": c.spec.get("wrong_answer"),
                        "condition_name": CONDITION_NAMES[cond],
                    },
                ))
        return items

    def score(self, item: PromptItem, response: str) -> dict[str, Any]:
        m = item.meta
        wrong = m.get("wrong_answer") if item.condition in ("C3", "C5") else None
        o = factual_outcome(response, m["ground_truth"], m["alts"], wrong)
        return {
            "passed": o["correct"], "correct": o["correct"], "condition": item.condition,
            "condition_name": m["condition_name"], "adopted_wrong": o["adopted_wrong"],
            "needs_review": o["has_wrong"] or not o["has_truth"],
        }
