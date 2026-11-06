"""Model registry: build clients from configs/models.yaml."""
from __future__ import annotations

import os
from typing import Any

from ..utils.config import load_models_config
from ..utils.logging import get_logger
from .base import ModelClient, ModelSpec, RetryableError, is_retryable

log = get_logger(__name__)

__all__ = ["ModelClient", "ModelSpec", "RetryableError", "is_retryable", "load_specs", "build_client"]

_PROVIDER_DEFAULT_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def load_specs(config: dict[str, Any] | None = None) -> dict[str, ModelSpec]:
    cfg = config or load_models_config()
    d = cfg.get("defaults", {})
    specs: dict[str, ModelSpec] = {}
    for key, m in (cfg.get("models") or {}).items():
        known = {
            "enabled", "provider", "model_id", "temperature", "max_tokens", "timeout_s", "max_retries",
            "requests_per_minute", "api_key_env", "price_per_mtok_input", "price_per_mtok_output", "verified",
        }
        merged = {**d, **m}
        extra = {k: v for k, v in merged.items() if k not in known}
        specs[key] = ModelSpec(
            key=key,
            provider=merged["provider"],
            model_id=str(merged["model_id"]),
            enabled=bool(merged.get("enabled", True)),
            temperature=float(merged.get("temperature", 0.0)),
            max_tokens=int(merged.get("max_tokens", 512)),
            timeout_s=float(merged.get("timeout_s", 60)),
            max_retries=int(merged.get("max_retries", 4)),
            requests_per_minute=float(merged.get("requests_per_minute", 30)),
            api_key_env=merged.get("api_key_env") or _PROVIDER_DEFAULT_ENV.get(merged["provider"]),
            price_in=merged.get("price_per_mtok_input"),
            price_out=merged.get("price_per_mtok_output"),
            verified=bool(merged.get("verified", False)),
            extra=extra,
        )
    return specs


def has_credentials(spec: ModelSpec) -> bool:
    if spec.provider in ("mock", "local"):
        return True
    return bool(spec.api_key_env and os.environ.get(spec.api_key_env))


def build_client(spec: ModelSpec, **kw) -> ModelClient:
    p = spec.provider
    if p == "anthropic":
        from .anthropic_client import AnthropicClient
        return AnthropicClient(spec, **kw)
    if p == "openai":
        from .openai_client import OpenAIClient
        return OpenAIClient(spec, **kw)
    if p == "gemini":
        from .gemini_client import GeminiClient
        return GeminiClient(spec, **kw)
    if p == "local":
        from .local_model import LocalModelClient
        return LocalModelClient(spec, **kw)
    if p == "mock":
        from .mock_client import MockClient
        return MockClient(spec, **kw)
    raise ValueError(f"Unknown provider: {p}")
