"""Shared data schemas: dataset test cases, prompt items and stored records."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Category = Literal[
    "instruction_conflict",
    "misleading_context",
    "uncertainty",
    "adversarial",
    "context_sensitivity",
    "response_stability",
]
Difficulty = Literal["easy", "medium", "hard"]
ReviewStatus = Literal["unreviewed", "verified", "rejected", "needs_revision"]
Origin = Literal["author_written", "model_generated", "adapted_from_public", "human_verified"]

# experiment name -> dataset category
EXPERIMENT_CATEGORY: dict[str, str] = {
    "instruction_conflicts": "instruction_conflict",
    "misleading_context": "misleading_context",
    "uncertainty": "uncertainty",
    "adversarial": "adversarial",
    "context_sensitivity": "context_sensitivity",
    "response_stability": "response_stability",
}
EXPERIMENTS = list(EXPERIMENT_CATEGORY)


class TestCase(BaseModel):
    """One evaluation case. Experiment-specific scoring parameters live in `spec`."""

    __test__ = False  # not a pytest class

    id: str
    category: Category
    question: str
    context: str | None = None
    system_prompt: str | None = None
    ground_truth: str | None = None
    expected_behavior: str
    difficulty: Difficulty
    evaluation_criteria: list[str] = Field(default_factory=list)
    acceptable_alternatives: list[str] = Field(default_factory=list)
    ground_truth_source: str | None = None
    origin: Origin = "author_written"
    human_review_status: ReviewStatus = "unreviewed"
    dataset_version: str = "0.1.0"
    tags: list[str] = Field(default_factory=list)
    spec: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _id_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("id must be non-empty")
        return v


@dataclass
class PromptItem:
    """A concrete prompt to send to a model, derived from a TestCase."""

    item_id: str  # unique per (case, condition/variant)
    case_id: str
    experiment: str
    condition: str  # e.g. "C3", "attack:quoted", "paraphrase_1", "base"
    prompt: str
    system: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResponse:
    text: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_s: float | None = None
    finish_reason: str | None = None
    error: str | None = None
    attempts: int = 1
    # sampling params actually sent (None value = not applied/unsupported); stored with every raw record
    params_applied: dict[str, Any] | None = None
