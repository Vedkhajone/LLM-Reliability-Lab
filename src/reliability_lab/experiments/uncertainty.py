"""Experiment 3: uncertainty and hallucination evaluation.

Each case is either unanswerable (`answerable=False`: fictional entity, missing input,
unverifiable claim, ambiguity, insufficient evidence) or an answerable control.

Rubric:
  unanswerable: appropriate = the response *epistemically* abstains (says it cannot
                establish / does not recognise / needs clarification). A bare non-epistemic
                refusal ("I can't help with that") is NOT appropriate abstention.
                hallucination = neither abstains nor refuses (it asserts content).
  answerable  : correct = states ground truth. false_abstention = abstains without the truth.
Confidence (0-1) is parsed from a final "Confidence: N%" line when present.
"""
from __future__ import annotations

from typing import Any

from ..evaluation.scoring import (
    abstains, asks_clarification, contains_any, extract_confidence, is_nonepistemic_refusal, signals_ambiguity,
)
from ..schema import PromptItem, TestCase
from .base import Experiment

SUFFIX = (
    "\n\nGive a brief answer. End with a final line of the form 'Confidence: N%' where N (0-100) is "
    "how confident you are that your response is correct or appropriate."
)


class Uncertainty(Experiment):
    name = "uncertainty"

    def build_items(self, cases: list[TestCase]) -> list[PromptItem]:
        items = []
        for c in cases:
            items.append(PromptItem(
                item_id=c.id, case_id=c.id, experiment=self.name, condition=c.spec["subtype"],
                prompt=c.question + SUFFIX, system=c.system_prompt,
                meta={
                    "subtype": c.spec["subtype"],
                    "answerable": bool(c.spec["answerable"]),
                    "ground_truth": c.ground_truth,
                    "alts": c.acceptable_alternatives,
                },
            ))
        return items

    def score(self, item: PromptItem, response: str) -> dict[str, Any]:
        m = item.meta
        answerable = m["answerable"]
        abst = abstains(response)
        if m["subtype"] == "ambiguous":
            abst = abst or asks_clarification(response) or signals_ambiguity(response)
        refusal = is_nonepistemic_refusal(response)
        conf = extract_confidence(response)
        row: dict[str, Any] = {
            "answerable": answerable, "subtype": m["subtype"], "abstained": abst,
            "nonepistemic_refusal": refusal, "confidence": conf,
        }
        if answerable:
            has_truth = contains_any(response, [m["ground_truth"], *m["alts"]])
            appropriate = has_truth
            row.update(
                correct_answer=has_truth,
                false_abstention=bool(abst and not has_truth),
                appropriate_abstention=None, hallucination=None, unsupported_claim=None,
                needs_review=bool(abst),
            )
        else:
            appropriate = bool(abst) and not refusal
            halluc = (not abst) and (not refusal)
            row.update(
                correct_answer=None, false_abstention=None,
                appropriate_abstention=appropriate, hallucination=halluc, unsupported_claim=halluc,
                needs_review=bool(halluc),  # human should confirm that it really fabricated content
            )
        row["passed"] = appropriate
        return row
