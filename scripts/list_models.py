"""Check configured model identifiers against each provider's live model list.

Needs the relevant API keys. Prints OK / NOT FOUND per configured model so that you
never run an experiment against a stale or mistyped model id.
"""
import os
import sys

import _common  # noqa: F401
from reliability_lab.models import load_specs


def _anthropic(key):
    import anthropic
    return {m.id for m in anthropic.Anthropic(api_key=key).models.list(limit=1000)}


def _openai(key):
    import openai
    return {m.id for m in openai.OpenAI(api_key=key).models.list()}


def _gemini(key):
    from google import genai
    return {m.name.split("/", 1)[-1] for m in genai.Client(api_key=key).models.list()}


LISTERS = {"anthropic": _anthropic, "openai": _openai, "gemini": _gemini}


def main() -> int:
    rc = 0
    cache: dict[str, set[str]] = {}
    for k, s in load_specs().items():
        if s.provider not in LISTERS:
            print(f"{k:12s} {s.provider:10s} {s.model_id:30s} (not checked)")
            continue
        key = os.environ.get(s.api_key_env or "")
        if not key:
            print(f"{k:12s} {s.provider:10s} {s.model_id:30s} SKIPPED (no {s.api_key_env})")
            continue
        try:
            ids = cache.setdefault(s.provider, LISTERS[s.provider](key))
        except Exception as exc:  # noqa: BLE001
            print(f"{k:12s} {s.provider:10s} list failed: {type(exc).__name__}: {exc}")
            rc = 1
            continue
        ok = s.model_id in ids
        print(f"{k:12s} {s.provider:10s} {s.model_id:30s} {'OK' if ok else 'NOT FOUND'}")
        if not ok:
            rc = 1
            close = sorted(i for i in ids if i.split("-")[0] == s.model_id.split("-")[0])[:8]
            print(f"             similar ids: {close}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
