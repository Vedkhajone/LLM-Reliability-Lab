"""Score raw responses, compute metrics + statistics, write figures and a report.

  python scripts/analyse_results.py                       # results/
  python scripts/analyse_results.py --results-dir results_smoke --no-embeddings
"""
import argparse
import sys

import _common  # noqa: F401
from reliability_lab.models import build_client, load_specs
from reliability_lab.evaluation.evaluators import LLMJudge
from reliability_lab.pipeline.report_generator import run_analysis
from reliability_lab.pipeline.result_collector import export_review_queue, read_raw, save_processed, score_raw
from reliability_lab.utils.config import load_evaluation_config, load_experiments_config
from reliability_lab.utils.logging import setup_logging


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--no-embeddings", action="store_true", help="use lexical fallback instead of sentence-transformers")
    ap.add_argument("--judge", action="store_true", help="enable the optional LLM judge (see configs/evaluation.yaml)")
    ap.add_argument("--human-override", action="store_true", help="human labels override automatic `passed`")
    ap.add_argument("--verified-only", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args(argv)
    setup_logging(a.log_level)

    exp_cfg, eval_cfg = load_experiments_config(), load_evaluation_config()
    results_dir = a.results_dir or exp_cfg["run"]["results_dir"]
    raw = read_raw(results_dir)
    if raw.empty:
        print(f"No raw results in {results_dir}/raw. Run an experiment first.", file=sys.stderr)
        return 2
    judge = None
    if a.judge or eval_cfg["judge"]["enabled"]:
        spec = load_specs()[eval_cfg["judge"]["model_key"]]
        judge = LLMJudge(build_client(spec), eval_cfg["judge"]["experiments"])
    scored = score_raw(raw, exp_cfg, exp_cfg["run"]["dataset_dir"], judge=judge, verified_only=a.verified_only,
                      include_dirs=tuple(exp_cfg["run"].get("extra_dataset_dirs", [])))
    if scored.empty:
        print("No scorable rows (all errors?).", file=sys.stderr)
        return 2
    save_processed(scored, results_dir)
    q = export_review_queue(scored, results_dir)
    out = run_analysis(results_dir, eval_cfg, use_embeddings=not a.no_embeddings, override_with_human=a.human_override)
    print(out["report"][:4000])
    print(f"\nWrote: {results_dir}/processed/scored.*, statistics/metrics.csv, statistics/comparisons.csv, "
          f"statistics/report.md, figures/*.png\nHuman review queue: {q}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
