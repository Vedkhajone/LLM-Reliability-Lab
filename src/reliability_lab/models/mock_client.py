"""Mock client for OFFLINE pipeline testing.

Mock outputs are synthetic and carry no information about any real model. Results
produced with this client must never be reported as findings. By default it behaves
like a deliberately naive "echo the context" responder so that every scoring path
(including failure paths) is exercised.
"""
from __future__ import annotations

import re
from typing import Callable

from ..schema import ModelResponse
from .base import ModelClient, ModelSpec


def default_responder(prompt: str, system: str | None) -> str:
    m = re.search(r"Context:\n(.*?)\n\nQuestion:", prompt, flags=re.S)
    if m:
        first = m.group(1).strip().split(". ")[0].strip()
        return f"{first}. Answer: {first}\nConfidence: 70%"
    if "CANARY" in prompt:
        c = re.search(r"CANARY-[A-Z0-9]+", prompt)
        return c.group(0) if c else "ok"
    return "I'm not sure; I cannot verify this.\nAnswer: unknown\nConfidence: 30%"


class MockClient(ModelClient):
    def __init__(self, spec: ModelSpec, responder: Callable[[str, str | None], str] | None = None, **kw):
        super().__init__(spec, **kw)
        self.responder = responder or default_responder
        self.calls: list[dict] = []

    def _call(self, prompt, system, temperature, max_tokens, seed):
        self.calls.append({"prompt": prompt, "system": system, "temperature": temperature})
        text = self.responder(prompt, system)
        return ModelResponse(
            text=text,
            input_tokens=max(1, len(prompt) // 4),
            output_tokens=max(1, len(text) // 4),
            finish_reason="mock",
            params_applied={"temperature": temperature, "seed": seed},
        )
