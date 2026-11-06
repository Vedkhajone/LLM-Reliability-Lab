"""Anthropic Claude adapter (Messages API).

Sampling parameters are applied only if the installed SDK / API accepts them: recent SDK
versions may not expose `temperature` on `messages.create`. What was actually applied is
returned in `params_applied` and stored with every response, so reproducibility metadata never
claims a temperature that was not sent.
"""
from __future__ import annotations

import inspect
import os

from ..schema import ModelResponse
from ..utils.logging import get_logger
from .base import ModelClient, ModelSpec

log = get_logger(__name__)


class AnthropicClient(ModelClient):
    def __init__(self, spec: ModelSpec, **kw):
        super().__init__(spec, **kw)
        import anthropic  # lazy: only needed when this model is used

        key = os.environ.get(spec.api_key_env or "ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(f"Missing API key env var {spec.api_key_env or 'ANTHROPIC_API_KEY'}")
        # max_retries=0: retry logic lives in ModelClient so behaviour is identical across providers
        self._client = anthropic.Anthropic(api_key=key, timeout=spec.timeout_s, max_retries=0)
        self._send_temperature = self._sdk_accepts("temperature")

    def _sdk_accepts(self, name: str) -> bool:
        try:
            return name in inspect.signature(self._client.messages.create).parameters
        except (TypeError, ValueError):
            return True

    def _create(self, kwargs):
        return self._client.messages.create(**kwargs)

    def _call(self, prompt, system, temperature, max_tokens, seed):
        kwargs = dict(
            model=self.spec.model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            kwargs["system"] = system
        if self._send_temperature:
            kwargs["temperature"] = temperature
        try:
            msg = self._create(kwargs)
        except Exception as exc:  # noqa: BLE001
            m = str(exc).lower()
            if self._send_temperature and "temperature" in m and any(
                t in m for t in ("deprecated", "unsupported", "not supported", "invalid", "not allowed")
            ):
                log.warning("[%s] API rejects temperature; retrying without it", self.key)
                self._send_temperature = False
                kwargs.pop("temperature", None)
                msg = self._create(kwargs)
            else:
                raise
        text = "".join(getattr(b, "text", "") for b in msg.content)
        usage = getattr(msg, "usage", None)
        return ModelResponse(
            text=text,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            finish_reason=getattr(msg, "stop_reason", None),
            params_applied={"temperature": temperature if self._send_temperature else None, "seed": None},
        )
