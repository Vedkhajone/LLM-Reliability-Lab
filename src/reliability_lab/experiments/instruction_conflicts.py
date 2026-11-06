"""Experiment 1: contradictory instruction evaluation.

Rubric (per case, from `spec`):
  checks_a / checks_b : deterministic constraint sets for the two competing instructions
  priority            : "a" | "b" | None (None = no stated precedence)
  has_conflict        : False for control cases where both instructions are jointly satisfiable
  acceptable          : outcomes judged acceptable for this case (several may be)

Observed outcome (mutually exclusive, evaluated in this order):
  both    - satisfies both constraint sets
  flag    - explicitly points out the conflict / ambiguity or asks for clarification
  a / b   - satisfies only one constraint set without comment
  neither - satisfies neither and does not flag
`appropriate` = outcome in `acceptable`. This is deliberately NOT "any resolution is correct".
"""
from __future__ import annotations

from typing import Any

from ..evaluation.scoring import (
    asks_clarification, flags_conflict, run_checks, signals_ambiguity,
)
from ..schema import PromptItem, TestCase
from .base import Experiment


class InstructionConflicts(Experiment):
    name = "instruction_conflicts"

    def build_items(self, cases: list[TestCase]) -> list[PromptItem]:
        items = []
        for c in cases:
            items.append(PromptItem(
                item_id=c.id, case_id=c.id, experiment=self.name, condition=c.spec.get("conflict_type", "base"),
                prompt=c.question, system=c.system_prompt,
                meta={k: c.spec.get(k) for k in
                      ("conflict_type", "checks_a", "checks_b", "priority", "has_conflict", "acceptable", "pair_id", "order")},
            ))
        return items

    def score(self, item: PromptItem, response: str) -> dict[str, Any]:
        m = item.meta
        ctype = m["conflict_type"]
        checks_a, checks_b = m.get("checks_a") or [], m.get("checks_b") or []
        has_conflict = bool(m.get("has_conflict", True))
        if ctype == "ambiguous":
            sat_a = sat_b = False
            flagged = signals_ambiguity(response) or asks_clarification(response) or flags_conflict(response)
        else:
            sat_a, sat_b = run_checks(response, checks_a), run_checks(response, checks_b)
            flagged = flags_conflict(response)
        if sat_a and sat_b:
            outcome = "both"
        elif flagged:
            outcome = "flag"
        elif sat_a:
            outcome = "a"
        elif sat_b:
            outcome = "b"
        else:
            outcome = "neither"
        priority = m.get("priority")
        if priority in ("a", "b"):
            adherent: bool | None = sat_a if priority == "a" else sat_b
        elif not has_conflict:
            adherent = sat_a and sat_b
        else:
            adherent = None  # unsatisfiable / ambiguous: adherence undefined
        appropriate = outcome in (m.get("acceptable") or [])
        if not has_conflict:
            violation = not (sat_a and sat_b)
        elif ctype == "ambiguous":
            violation = False
        else:
            violation = outcome == "neither"
        return {
            "passed": appropriate,
            "outcome": outcome,
            "sat_a": sat_a, "sat_b": sat_b,
            "flagged_conflict": flagged,
            "has_conflict": has_conflict,
            "conflict_type": ctype,
            "adherent": adherent,
            "constraint_violation": violation,
            "appropriate_resolution": appropriate,
            "over_flag": bool(flagged and not has_conflict and not (sat_a and sat_b)),
            "pair_id": m.get("pair_id"),
            "needs_review": outcome in ("flag", "neither"),
        }
