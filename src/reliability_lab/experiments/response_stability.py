"""Experiment 6: response stability.

Variants per case: original, two semantically equivalent paraphrases, a structured-format
rewrite and (when the case has `context_items`) reversed context order. The runner repeats
every item `repeats` times at the configured temperature, so both prompt sensitivity and
sampling variance are observed. Factual agreement uses the canonicalised final
"Answer:" line; stylistic similarity is measured separately from the full response text.
"""
from __future__ import annotations

from typing import Any

from ..evaluation.scoring import contains_any, extract_answer, normalize, word_count
from ..schema import PromptItem, TestCase
from .base import Experiment

SUFFIX = "End your reply with a final line of the form 'Answer: <short answer>'."


class ResponseStability(Experiment):
    name = "response_stability"

    @staticmethod
    def _with_context(items: list[str], q: str) -> str:
        return "Context:\n" + "\n".join(items) + f"\n\nQuestion: {q}"

    def _variants(self, c: TestCase) -> dict[str, str]:
        items = c.spec.get("context_items")
        mk = (lambda q, its=items: self._with_context(its, q)) if items else (lambda q, its=None: f"Question: {q}")
        v = {
            "original": f"{mk(c.question)}\n{SUFFIX}",
            "paraphrase_1": f"{mk(c.spec['paraphrases'][0])}\n{SUFFIX}",
            "paraphrase_2": f"{mk(c.spec['paraphrases'][1])}\n{SUFFIX}",
            "format_structured": (
                (f"### Background\n" + "\n".join(f"- {s}" for s in items) + "\n\n" if items else "")
                + f"### Task\n{c.question}\n\n### Output format\nBrief explanation, then the final line "
                "'Answer: <short answer>'."
            ),
        }
        if items and len(items) > 1:
            v["context_reversed"] = f"{self._with_context(list(reversed(items)), c.question)}\n{SUFFIX}"
        return v

    def build_items(self, cases: list[TestCase]) -> list[PromptItem]:
        out = []
        for c in cases:
            for name, prompt in self._variants(c).items():
                out.append(PromptItem(
                    item_id=f"{c.id}::{name}", case_id=c.id, experiment=self.name, condition=name,
                    prompt=prompt, system=c.system_prompt,
                    meta={"ground_truth": c.ground_truth, "alts": c.acceptable_alternatives},
                ))
        return out

    def score(self, item: PromptItem, response: str) -> dict[str, Any]:
        m = item.meta
        final = extract_answer(response)
        basis = final if final is not None else response
        correct = contains_any(basis, [m["ground_truth"], *m["alts"]])
        # canonical form: correct answers collapse to the ground truth; others keep their text
        canon = normalize(m["ground_truth"]) if correct else normalize(basis)[:80]
        return {
            "passed": correct, "correct": correct, "variant": item.condition,
            "has_answer_line": final is not None, "final_answer": canon,
            "n_words": word_count(response), "needs_review": final is None,
        }
