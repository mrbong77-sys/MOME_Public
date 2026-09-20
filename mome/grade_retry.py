#!/usr/bin/env python3
"""Grade the retry cells.

`run_retry.py` generates and packs, and stops there: grading plays no part in
the gate's decision, so it is deliberately not on the campaign's execution
path.  Evaluation, though, has to ask whether the answer that was served is
correct, which is what this script is for.  It runs the same grader the
campaigns used, unchanged -- no grading rule is invented here.

    python3 grade_retry.py            # only what is missing
    python3 grade_retry.py --force    # regrade everything

Grading happens strictly after the gate has decided.  The gate never reads a
correct answer.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import paths
CONFIGS_DIR = paths.RETRY_CONFIGS
DATA_DIR = paths.RETRY
GRADER = paths.CODE / "grade.py"


def _covers(cell_dir: Path, grades: Path) -> bool:
    """Does the grades file cover every error-free record in the cell?"""
    sys.path.insert(0, str(paths.CODE))
    import ollama_http as oh
    recs = {oh.record_key(r) for r in oh.read_jsonl(cell_dir / "records.jsonl")
            if r.get("error") is None}
    done = {(g["item_id"], g.get("path_index"))
            for g in (json.loads(l) for l in
                      grades.read_text(encoding="utf-8").splitlines() if l.strip())}
    missing = recs - done
    if missing:
        print(f"  {cell_dir.name}: {len(missing)} ungraded pairs across "
              f"{len({i for i, _ in missing})} items -- regrading", flush=True)
    return not missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only")
    args = ap.parse_args()

    cfgs = sorted(CONFIGS_DIR.glob("*_retry_sc*.json"))
    if args.only:
        cfgs = [c for c in cfgs if c.stem == args.only]
        if not cfgs:
            print(f"ERROR: --only {args.only} is not a cell name", file=sys.stderr)
            return 4
    missing, graded, skipped = [], 0, 0
    for path in cfgs:
        cfg = json.loads(path.read_text(encoding="utf-8"))
        cell_dir = paths.ROOT / cfg["out_dir"]
        records = cell_dir / "records.jsonl"
        grades = cell_dir / "grades.jsonl"
        if not cell_dir.exists() or not (records.exists() or list(cell_dir.glob("records*.jsonl.gz"))):
            missing.append(path.stem)
            continue
        #: Skipping a cell merely because a grades file exists leaves pairs
        #: bought later ungraded forever.  One cell did exactly that: after a
        #: band correction it bought 20 more pairs, and because grades.jsonl
        #: was already there those 20 stayed ungraded (506 records, 486
        #: grades).  So the question is not "does it exist" but "does it cover
        #: every record".
        if grades.exists() and not args.force and _covers(cell_dir, grades):
            skipped += 1
            continue
        cmd = [sys.executable, str(GRADER), "--records", str(records),
               "--benchmark", *cfg["benchmark_path"]]
        print("+ " + " ".join(cmd[1:]), flush=True)
        code = subprocess.call(cmd, cwd=paths.ROOT)
        if code != 0:
            print(f"ERROR: grading {path.stem} exited with code {code}", file=sys.stderr)
            return code
        graded += 1

    print(f"\ngraded {graded} cells, skipped {skipped} already covered", end="")
    if missing:
        print(f", {len(missing)} cells with no records: " + ", ".join(missing))
        print("Generation is unfinished for those. Run run_retry.py to completion "
              "and try again.")
    else:
        print(f" -- all {len(cfgs)} cells are graded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
