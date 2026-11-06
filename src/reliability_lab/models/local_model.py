"""Open-weight instruction-tuned model via Hugging Face Transformers (local inference)."""
from __future__ import annotations

from ..schema import ModelResponse
from .base import ModelClient, ModelSpec


class LocalModelClient(ModelClient):
    def __init__(self, spec: ModelSpec, **kw):
        super().__init__(spec, **kw)
        if spec.model_id.startswith("REPLACE"):
            raise RuntimeError("Set `model_id` for the open_weight model in configs/models.yaml")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        device = spec.extra.get("device", "auto")
        self._tok = AutoTokenizer.from_pretrained(spec.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            spec.model_id, device_map="auto" if device == "auto" else None
        )
        if device not in ("auto", None):
            self._model.to(device)

    def _call(self, prompt, system, temperature, max_tokens, seed):
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        try:
            ids = self._tok.apply_chat_template(
                msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
            )
        except Exception:  # noqa: BLE001 - models without a system role
            merged = (system + "\n\n" if system else "") + prompt
            ids = self._tok.apply_chat_template(
                [{"role": "user", "content": merged}],
                add_generation_prompt=True, return_tensors="pt", return_dict=True,
            )
        ids = {k: v.to(self._model.device) for k, v in ids.items()}
        if seed is not None:
            self._torch.manual_seed(seed)
        do_sample = temperature > 0
        gen_kwargs = dict(max_new_tokens=max_tokens, do_sample=do_sample)
        if do_sample:
            gen_kwargs["temperature"] = temperature
        out = self._model.generate(**ids, **gen_kwargs, pad_token_id=self._tok.eos_token_id)
        n_in = ids["input_ids"].shape[1]
        text = self._tok.decode(out[0][n_in:], skip_special_tokens=True)
        return ModelResponse(
            text=text, input_tokens=int(n_in), output_tokens=int(out.shape[1] - n_in),
            params_applied={"temperature": temperature if do_sample else 0.0, "seed": seed},
        )
