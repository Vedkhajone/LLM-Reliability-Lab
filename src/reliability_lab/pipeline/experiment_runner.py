"""Resumable, budget-limited experiment execution.

Safety rails against accidental spend:
  * dry-run mode (plans and estimates, sends nothing)
  * confirmation prompt above `confirm_above_requests` (skip with --yes)
  * hard caps: max requests and max USD (the latter only when prices are configured)
  * per-model rate limiting and retry (in ModelClient)
Resumability: each completed request is appended to results/raw/<experiment>__<model>.jsonl
with a deterministic `request_key`; re-running skips keys already completed without error.
Raw files are append-only and never overwritten.
"""
from __future__ import annotations

import hashlib
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..dataset import load_experiment_cases
from ..experiments import get_experiment
from ..models import build_client, has_credentials
from ..models.base import ModelClient, ModelSpec
from ..schema import PromptItem
from ..utils.config import load_experiments_config
from ..utils.cost_tracker import BudgetExceeded, CostTracker, approx_tokens
from ..utils.logging import get_logger
from .result_collector import append_raw, completed_keys

log = get_logger(__name__)


def request_key(experiment: str, model_id: str, item: PromptItem, repeat: int, params: dict, prompt_version: str,
                dataset_version: str) -> str:
    h = hashlib.sha256()
    for part in (experiment, model_id, item.item_id, str(repeat), prompt_version, dataset_version,
                 item.system or "", item.prompt, repr(sorted(params.items()))):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:24]


@dataclass
class PlannedRequest:
    experiment: str
    model_key: str
    item: PromptItem
    repeat: int
    params: dict[str, Any]
    key: str


@dataclass
class RunPlan:
    requests: list[PlannedRequest] = field(default_factory=list)
    skipped_done: int = 0
    est_input_tokens: int = 0
    est_cost_usd: float | None = None

    def per_model(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.requests:
            out[r.model_key] = out.get(r.model_key, 0) + 1
        return out


class ExperimentRunner:
    def __init__(
        self,
        specs: dict[str, ModelSpec],
        exp_cfg: dict[str, Any] | None = None,
        results_dir: str | None = None,
        clients: dict[str, ModelClient] | None = None,
        smoke: bool = False,
    ):
        self.cfg = exp_cfg or load_experiments_config()
        run = self.cfg["run"]
        self.run = run
        self.smoke = smoke
        self.results_dir = results_dir or (run["results_dir"] + ("_smoke" if smoke else ""))
        self.specs = specs
        self._clients: dict[str, ModelClient] = dict(clients or {})
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]

    # ---------------------------------------------------------------- planning ----
    def _experiment_cfg(self, name: str) -> dict[str, Any]:
        cfg = dict(self.cfg["experiments"][name])
        if self.smoke:
            cfg["repeats"] = self.run["smoke"]["repeats"]
        return cfg

    def plan(self, experiments: list[str], model_keys: list[str], limit: int | None = None,
             verified_only: bool = False) -> RunPlan:
        plan = RunPlan()
        total_cost, priced = 0.0, True
        for name in experiments:
            ecfg = self._experiment_cfg(name)
            exp = get_experiment(name, ecfg)
            cases = load_experiment_cases(name, self.run["dataset_dir"], ecfg["dataset_file"], verified_only=verified_only,
                                         include_dirs=tuple(self.run.get("extra_dataset_dirs", [])))
            n_cases = limit if limit is not None else (self.run["smoke"]["cases_per_experiment"] if self.smoke else None)
            if n_cases is not None:
                cases = cases[:n_cases]
            items = exp.build_items(cases)
            params = exp.generation_params()
            for mk in model_keys:
                spec = self.specs[mk]
                done = completed_keys(self.results_dir, name, mk)
                for it in items:
                    for rep in range(exp.repeats):
                        k = request_key(name, spec.model_id, it, rep, params, self.run["prompt_version"],
                                        self.run["dataset_version"])
                        if k in done:
                            plan.skipped_done += 1
                            continue
                        plan.requests.append(PlannedRequest(name, mk, it, rep, params, k))
                        tin = approx_tokens((it.system or "") + it.prompt)
                        plan.est_input_tokens += tin
                        if spec.price_in is None or spec.price_out is None:
                            priced = False
                        else:
                            total_cost += (tin * spec.price_in + params["max_tokens"] * 0.5 * spec.price_out) / 1e6
        plan.est_cost_usd = total_cost if priced and plan.requests else None
        return plan

    # --------------------------------------------------------------- execution ----
    def client(self, key: str) -> ModelClient:
        if key not in self._clients:
            self._clients[key] = build_client(self.specs[key])
        return self._clients[key]

    def execute(
        self,
        plan: RunPlan,
        max_requests: int | None = None,
        max_cost_usd: float | None = None,
    ) -> CostTracker:
        tracker = CostTracker(
            max_requests=max_requests if max_requests is not None else self.run.get("default_max_requests"),
            max_cost_usd=max_cost_usd if max_cost_usd is not None else self.run.get("default_max_cost_usd"),
        )
        n = len(plan.requests)
        for i, pr in enumerate(plan.requests, 1):
            try:
                tracker.check_before_request()
            except BudgetExceeded as exc:
                log.error("Stopping: %s. Re-run to resume.", exc)
                break
            cl = self.client(pr.model_key)
            seed = int(self.run.get("seed", 0)) + pr.repeat
            resp = cl.generate(
                pr.item.prompt, system=pr.item.system,
                temperature=pr.params["temperature"], max_tokens=pr.params["max_tokens"], seed=seed,
            )
            cost = cl.cost(resp)
            tracker.record(pr.model_key, resp.input_tokens, resp.output_tokens, cost)
            append_raw(self.results_dir, {
                "run_id": self.run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request_key": pr.key,
                "experiment": pr.experiment,
                "model_key": pr.model_key,
                "provider": cl.provider,
                "model_id": cl.model_id,
                "prompt_version": self.run["prompt_version"],
                "dataset_version": self.run["dataset_version"],
                "case_id": pr.item.case_id,
                "item_id": pr.item.item_id,
                "condition": pr.item.condition,
                "repeat": pr.repeat,
                "temperature": pr.params["temperature"],
                "max_tokens": pr.params["max_tokens"],
                "seed": seed,
                "system": pr.item.system,
                "prompt": pr.item.prompt,
                "response": resp.text,
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "latency_s": resp.latency_s,
                "cost_usd": cost,
                "finish_reason": resp.finish_reason,
                "params_applied": resp.params_applied,
                "attempts": resp.attempts,
                "error": resp.error,
            })
            if i % 10 == 0 or i == n:
                log.info("progress %d/%d | requests=%d cost=$%.4f", i, n, tracker.requests, tracker.cost_usd)
        return tracker


def resolve_models(specs: dict[str, ModelSpec], requested: list[str] | None, mock_ok: bool = True) -> list[str]:
    """Pick model keys: explicit request, or all enabled models with credentials."""
    if requested:
        unknown = [m for m in requested if m not in specs]
        if unknown:
            raise SystemExit(f"Unknown model key(s): {unknown}. Known: {sorted(specs)}")
        missing = [m for m in requested if not has_credentials(specs[m])]
        if missing:
            raise SystemExit(f"Missing API key env var for: {missing} (see .env.example)")
        return requested
    chosen = []
    for k, s in specs.items():
        if not s.enabled:
            continue
        if not has_credentials(s):
            log.warning("Skipping '%s': env var %s not set", k, s.api_key_env)
            continue
        chosen.append(k)
    return chosen


def confirm(plan: RunPlan, threshold: int, assume_yes: bool) -> bool:
    if len(plan.requests) <= threshold or assume_yes:
        return True
    if not sys.stdin.isatty():
        print(f"Refusing to run {len(plan.requests)} requests non-interactively without --yes.", file=sys.stderr)
        return False
    try:
        ans = input(f"About to send {len(plan.requests)} API requests. Continue? [y/N] ").strip().lower()
    except EOFError:  # stdin closed / not really interactive: refuse rather than spend
        print("No interactive input available; refusing without --yes.", file=sys.stderr)
        return False
    return ans in ("y", "yes")
