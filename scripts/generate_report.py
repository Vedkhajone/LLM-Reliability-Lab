"""Populate research/paper_draft.md from the template and latest analysis report.

Only the AUTO:RESULTS marker is filled. All interpretation must be written by you after
looking at the data; nothing here invents findings.
"""
import argparse
import sys

import _common  # noqa: F401
from reliability_lab.pipeline.report_generator import populate_paper
from reliability_lab.utils.config import project_root


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results")
    a = ap.parse_args(argv)
    rp = project_root() / a.results_dir / "statistics" / "report.md"
    if not rp.exists():
        print(f"{rp} not found. Run scripts/analyse_results.py first.", file=sys.stderr)
        return 2
    out = populate_paper(rp.read_text(encoding="utf-8"))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
