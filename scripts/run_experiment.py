"""Run one or more experiments against one or more models.

Examples
  python scripts/run_experiment.py --experiment misleading_context --smoke --models mock
  python scripts/run_experiment.py --experiment uncertainty --models claude openai --dry-run
  python scripts/run_experiment.py --experiment adversarial --models claude --limit 5
"""
import argparse
import sys

import _common  # noqa: F401
from reliability_lab.models import load_specs
from reliability_lab.pipeline.experiment_runner import ExperimentRunner, confirm, resolve_models
from reliability_lab.schema import EXPERIMENTS
from reliability_lab.utils.logging import setup_logging


def build_parser(all_experiments: bool = False) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    if not all_experiments:
        p.add_argument("--experiment", "-e", nargs="+", required=True, choices=EXPERIMENTS + ["all"])
    p.add_argument("--models", "-m", nargs="+", help="model keys from configs/models.yaml (default: all enabled with keys)")
    p.add_argument("--limit", type=int, help="max test cases per experiment")
    p.add_argument("--smoke", action="store_true", help="tiny run (few cases, 1 repeat), separate results dir")
    p.add_argument("--dry-run", action="store_true", help="plan and estimate only; send nothing")
    p.add_argument("--yes", "-y", action="store_true", help="skip the large-run confirmation")
    p.add_argument("--max-requests", type=int)
    p.add_argument("--max-cost", type=float, help="USD cap (needs prices in models.yaml)")
    p.add_argument("--results-dir")
    p.add_argument("--verified-only", action="store_true", help="only human-verified cases")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv=None, all_experiments: bool = False) -> int:
    args = build_parser(all_experiments).parse_args(argv)
    setup_logging(args.log_level)
    experiments = EXPERIMENTS if all_experiments or "all" in args.experiment else args.experiment
    specs = load_specs()
    models = resolve_models(specs, args.models)
    if not models:
        print("No runnable models: set API keys in .env (see .env.example) or pass --models mock.", file=sys.stderr)
        return 2
    if any(specs[m].provider == "mock" for m in models):
        print("NOTE: mock client selected; outputs are synthetic and for pipeline testing only.")
    runner = ExperimentRunner(specs, results_dir=args.results_dir, smoke=args.smoke)
    plan = runner.plan(experiments, models, limit=args.limit, verified_only=args.verified_only)
    cost = "n/a (prices not configured)" if plan.est_cost_usd is None else f"~${plan.est_cost_usd:.2f}"
    print(f"Plan: {len(plan.requests)} requests to send ({plan.skipped_done} already done) | "
          f"per model: {plan.per_model()} | est. input tokens: {plan.est_input_tokens:,} | est. cost: {cost}")
    print(f"Results dir: {runner.results_dir}")
    if args.dry_run:
        print("Dry run: nothing sent.")
        return 0
    if not plan.requests:
        print("Nothing to do.")
        return 0
    if not confirm(plan, runner.run["confirm_above_requests"], args.yes):
        print("Aborted.")
        return 1
    tracker = runner.execute(plan, max_requests=args.max_requests, max_cost_usd=args.max_cost)
    print("Done:", tracker.summary())
    print(f"Next: python scripts/analyse_results.py --results-dir {runner.results_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
