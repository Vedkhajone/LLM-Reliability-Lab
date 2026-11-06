from __future__ import annotations

from typing import Any

from .adversarial import Adversarial
from .base import Experiment
from .context_sensitivity import ContextSensitivity
from .instruction_conflicts import InstructionConflicts
from .misleading_context import MisleadingContext
from .response_stability import ResponseStability
from .uncertainty import Uncertainty

REGISTRY: dict[str, type[Experiment]] = {
    "instruction_conflicts": InstructionConflicts,
    "misleading_context": MisleadingContext,
    "uncertainty": Uncertainty,
    "adversarial": Adversarial,
    "context_sensitivity": ContextSensitivity,
    "response_stability": ResponseStability,
}


def get_experiment(name: str, cfg: dict[str, Any] | None = None) -> Experiment:
    if name not in REGISTRY:
        raise KeyError(f"Unknown experiment '{name}'. Choose from {sorted(REGISTRY)}")
    return REGISTRY[name](cfg)


__all__ = ["REGISTRY", "get_experiment", "Experiment"]
