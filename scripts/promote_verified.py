"""Move human-verified cases from datasets/generated to datasets/processed.

Workflow: edit a case in datasets/generated/<file>.jsonl, check ground truth + rubric, set
"human_review_status": "verified" and "reviewer": <your name> in `tags`/notes, then run this
script. Verified cases are copied (origin -> "human_verified") into datasets/processed/ and
removed from generated/. Use --dry-run to preview.
"""
import argparse
import sys
from pathlib import Path

import _common  # noqa: F401
from reliability_lab.dataset import load_jsonl, write_jsonl
from reliability_lab.utils.config import project_root


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    gen, proc = project_root() / "datasets/generated", project_root() / "datasets/processed"
    moved = 0
    for f in sorted(gen.glob("*.jsonl")):
        rows = load_jsonl(f)
        keep = [r for r in rows if r.get("human_review_status") != "verified"]
        ver = [dict(r, origin="human_verified") for r in rows if r.get("human_review_status") == "verified"]
        if not ver:
            continue
        moved += len(ver)
        print(f"{f.name}: promoting {len(ver)} case(s)")
        if a.dry_run:
            continue
        target = proc / f.name
        existing = load_jsonl(target) if target.exists() else []
        ids = {r["id"] for r in ver}
        write_jsonl(target, [r for r in existing if r["id"] not in ids] + ver)
        write_jsonl(f, keep)
    print(f"{moved} case(s) {'would be ' if a.dry_run else ''}promoted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
