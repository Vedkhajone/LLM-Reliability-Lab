"""OpenAI adapter (Chat Completions API).

Some newer OpenAI models reject `max_tokens` / non-default `temperature`. The adapter
falls back gracefully: it first tries `max_completion_tokens`, and drops `temperature`
if the API rejects it (recorded in the response `finish_reason` notes via logs).
"""
from __future__ import annotations

import os

from ..schema import ModelResponse
from ..utils.logging import get_logger
from .base import ModelClient, ModelSpec

log = get_logger(__name__)


class OpenAIClient(ModelClient):
    def __init__(self, spec: ModelSpec, **kw):
        super().__init__(spec, **kw)
        import openai

        key = os.environ.get(spec.api_key_env or "OPENAI_API_KEY")
        if not key:
            raise RuntimeError(f"Missing API key env var {spec.api_key_env or 'OPENAI_API_KEY'}")
        self._client = openai.OpenAI(api_key=key, timeout=spec.timeout_s, max_retries=0)
        self._send_temperature = True

    def _call(self, prompt, system, temperature, max_tokens, seed):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs = dict(model=self.spec.model_id, messages=messages, max_completion_tokens=max_tokens)
        if self._send_temperature:
            kwargs["temperature"] = temperature
        if seed is not None:
            kwargs["seed"] = seed
        try:
            r = self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if self._send_temperature and "temperature" in msg and "unsupported" in msg:
                log.warning("[%s] model rejects temperature; retrying without it", self.key)
                self._send_temperature = False
                kwargs.pop("temperature", None)
                r = self._client.chat.completions.create(**kwargs)
            else:
                raise
        choice = r.choices[0]
        usage = getattr(r, "usage", None)
        return ModelResponse(
            text=choice.message.content or "",
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            finish_reason=choice.finish_reason,
            params_applied={"temperature": temperature if self._send_temperature else None, "seed": seed},
        )
