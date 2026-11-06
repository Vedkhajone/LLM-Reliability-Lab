"""Common model interface with retry, rate limiting, timeout and error logging.

Adapters only implement `_call()`; everything else (retry/backoff, rate limiting,
latency measurement, token accounting, error capture) is shared so that all providers
are treated identically by the experiment runner.
"""
from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..schema import ModelResponse
from ..utils.cost_tracker import estimate_cost
from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ModelSpec:
    key: str
    provider: str
    model_id: str
    enabled: bool = True
    temperature: float = 0.0
    max_tokens: int = 512
    timeout_s: float = 60.0
    max_retries: int = 4
    requests_per_minute: float = 30.0
    api_key_env: str | None = None
    price_in: float | None = None
    price_out: float | None = None
    verified: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class RetryableError(Exception):
    """Raised by adapters for transient failures (rate limit, timeout, 5xx)."""


def is_retryable(exc: Exception) -> bool:
    if isinstance(exc, RetryableError):
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status, int) and (status in (408, 409, 425, 429) or status >= 500):
        return True
    name = type(exc).__name__.lower()
    return any(t in name for t in ("timeout", "ratelimit", "connection", "overloaded", "unavailable"))


class ModelClient(ABC):
    """Provider-agnostic client. Use `generate()`; adapters implement `_call()`."""

    def __init__(self, spec: ModelSpec, sleep=time.sleep, clock=time.monotonic):
        self.spec = spec
        self._sleep = sleep
        self._clock = clock
        self._last_request_t: float | None = None

    # ---- to implement -------------------------------------------------------------
    @abstractmethod
    def _call(
        self,
        prompt: str,
        system: str | None,
        temperature: float,
        max_tokens: int,
        seed: int | None,
    ) -> ModelResponse:
        """One raw API call. Raise on failure; do not retry here."""

    # ---- shared -------------------------------------------------------------------
    @property
    def key(self) -> str:
        return self.spec.key

    @property
    def provider(self) -> str:
        return self.spec.provider

    @property
    def model_id(self) -> str:
        return self.spec.model_id

    def _rate_limit(self) -> None:
        rpm = self.spec.requests_per_minute
        if not rpm or rpm <= 0:
            return
        min_interval = 60.0 / rpm
        now = self._clock()
        if self._last_request_t is not None:
            wait = min_interval - (now - self._last_request_t)
            if wait > 0:
                self._sleep(wait)
        self._last_request_t = self._clock()

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> ModelResponse:
        temp = self.spec.temperature if temperature is None else temperature
        mtok = self.spec.max_tokens if max_tokens is None else max_tokens
        attempts = 0
        last_err: str | None = None
        while attempts <= self.spec.max_retries:
            attempts += 1
            self._rate_limit()
            t0 = self._clock()
            try:
                resp = self._call(prompt, system, temp, mtok, seed)
                resp.latency_s = self._clock() - t0
                resp.attempts = attempts
                return resp
            except Exception as exc:  # noqa: BLE001 - provider SDKs raise many types
                last_err = f"{type(exc).__name__}: {exc}"
                retry = is_retryable(exc) and attempts <= self.spec.max_retries
                log.warning(
                    "[%s] API error (attempt %d/%d, retry=%s): %s",
                    self.key, attempts, self.spec.max_retries + 1, retry, last_err[:300],
                )
                if not retry:
                    break
                self._sleep(min(60.0, (2 ** attempts) + random.uniform(0, 1)))
        return ModelResponse(error=last_err or "unknown error", attempts=attempts)

    def cost(self, resp: ModelResponse) -> float | None:
        return estimate_cost(resp.input_tokens, resp.output_tokens, self.spec.price_in, self.spec.price_out)

    def describe(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "provider": self.provider,
            "model_id": self.model_id,
            "temperature": self.spec.temperature,
            "max_tokens": self.spec.max_tokens,
        }
