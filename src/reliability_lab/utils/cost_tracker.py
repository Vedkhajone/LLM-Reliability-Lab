"""Token and cost accounting plus request/cost budget enforcement."""
from __future__ import annotations

from dataclasses import dataclass, field


class BudgetExceeded(RuntimeError):
    pass


def estimate_cost(
    input_tokens: int | None,
    output_tokens: int | None,
    price_in: float | None,
    price_out: float | None,
) -> float | None:
    """USD cost given per-million-token prices; None when prices/tokens are unknown."""
    if price_in is None or price_out is None or input_tokens is None or output_tokens is None:
        return None
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000.0


def approx_tokens(text: str) -> int:
    """Crude token estimate (~4 characters per token). Used only for dry-run planning."""
    return max(1, len(text) // 4)


@dataclass
class CostTracker:
    max_requests: int | None = None
    max_cost_usd: float | None = None
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    unpriced_requests: int = 0
    by_model: dict[str, dict[str, float]] = field(default_factory=dict)

    def check_before_request(self) -> None:
        if self.max_requests is not None and self.requests >= self.max_requests:
            raise BudgetExceeded(f"request budget reached ({self.max_requests})")
        if self.max_cost_usd is not None and self.cost_usd >= self.max_cost_usd:
            raise BudgetExceeded(f"cost budget reached (${self.max_cost_usd:.2f})")

    def record(
        self,
        model_key: str,
        input_tokens: int | None,
        output_tokens: int | None,
        cost: float | None,
    ) -> None:
        self.requests += 1
        self.input_tokens += input_tokens or 0
        self.output_tokens += output_tokens or 0
        m = self.by_model.setdefault(
            model_key, {"requests": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
        )
        m["requests"] += 1
        m["input_tokens"] += input_tokens or 0
        m["output_tokens"] += output_tokens or 0
        if cost is None:
            self.unpriced_requests += 1
        else:
            self.cost_usd += cost
            m["cost_usd"] += cost

    def summary(self) -> dict:
        return {
            "requests": self.requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "unpriced_requests": self.unpriced_requests,
            "by_model": self.by_model,
        }
