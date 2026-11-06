"""Experiment 2: misleading context evaluation.

Context types: correct, incorrect, contradictory, irrelevant, partial, authoritative_misinfo.
`provenance` records WHY the context should (not) be trusted, so analyses can separate
a genuine reference excerpt from a bare unsupported assertion or a fabricated authority.
Note: no context can be both genuinely authoritative and false; "authoritative_misinfo"
therefore means *authoritative-sounding* text citing a fictitious institution.
"""
from __future__ import annotations

from typing import Any

from ..schema import PromptItem, TestCase
from .base import Experiment, factual_outcome


class MisleadingContext(Experiment):
    name = "misleading_context"

    def build_items(self, cases: list[TestCase]) -> list[PromptItem]:
        items = []
        for c in cases:
            prompt = f"Context:\n{c.context}\n\nQuestion: {c.question}\n{self.ANSWER_SUFFIX}"
            items.append(PromptItem(
                item_id=c.id, case_id=c.id, experiment=self.name, condition=c.spec["context_type"],
                prompt=prompt, system=c.system_prompt,
                meta={
                    "context_type": c.spec["context_type"],
                    "provenance": c.spec.get("provenance"),
                    "context_claim": c.spec.get("context_claim"),
                    "wrong_answer": c.spec.get("wrong_answer"),
                    "ground_truth": c.ground_truth,
                    "alts": c.acceptable_alternatives,
                },
            ))
        return items

    def score(self, item: PromptItem, response: str) -> dict[str, Any]:
        m = item.meta
        o = factual_outcome(response, m["ground_truth"], m["alts"], m.get("wrong_answer"))
        pushes_wrong = m.get("wrong_answer") is not None
        # context influence: response reproduces the claim the context pushes (true or false)
        claim = m.get("context_claim")
        influenced = bool(claim) and (
            o["adopted_wrong"] if pushes_wrong else o["has_truth"]
        )
        return {
            "passed": o["correct"],
            "correct": o["correct"],
            "context_type": m["context_type"],
            "provenance": m.get("provenance"),
            "pushes_wrong_claim": pushes_wrong,
            "has_claim": bool(claim),
            "adopted_wrong": o["adopted_wrong"],
            "flagged_discrepancy": o["flagged_discrepancy"],
            "context_influenced": influenced,
            # asserts a context claim as fact although it is not supported by ground truth
            "unsupported_assertion": o["adopted_wrong"] and not o["flagged_discrepancy"],
            "needs_review": o["has_wrong"] or (not o["has_truth"]),
        }
