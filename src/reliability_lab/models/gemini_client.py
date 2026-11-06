"""Google Gemini adapter (google-genai SDK)."""
from __future__ import annotations

import os

from ..schema import ModelResponse
from .base import ModelClient, ModelSpec


class GeminiClient(ModelClient):
    def __init__(self, spec: ModelSpec, **kw):
        super().__init__(spec, **kw)
        from google import genai
        from google.genai import types

        key = os.environ.get(spec.api_key_env or "GEMINI_API_KEY")
        if not key:
            raise RuntimeError(f"Missing API key env var {spec.api_key_env or 'GEMINI_API_KEY'}")
        self._types = types
        self._client = genai.Client(
            api_key=key, http_options=types.HttpOptions(timeout=int(spec.timeout_s * 1000))
        )

    def _call(self, prompt, system, temperature, max_tokens, seed):
        cfg = dict(temperature=temperature, max_output_tokens=max_tokens)
        if system:
            cfg["system_instruction"] = system
        if seed is not None:
            cfg["seed"] = seed
        r = self._client.models.generate_content(
            model=self.spec.model_id,
            contents=prompt,
            config=self._types.GenerateContentConfig(**cfg),
        )
        usage = getattr(r, "usage_metadata", None)
        finish = None
        try:
            finish = str(r.candidates[0].finish_reason)
        except Exception:  # noqa: BLE001
            pass
        return ModelResponse(
            text=r.text or "",
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
            finish_reason=finish,
            params_applied={"temperature": temperature, "seed": seed},
        )
