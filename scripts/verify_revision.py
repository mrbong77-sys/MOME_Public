#!/usr/bin/env python3
"""Recompute every number of the round-1 revision and check it against results/ssot.json.

    python3 scripts/verify_revision.py              # all analyses, about 15 minutes on one core
    python3 scripts/verify_revision.py --fast       # skip the four re-fitting analyses

The revision added analyses that the manuscript reads from the `review_r1` block of
results/ssot.json (Sections 4.4 and 4.7-4.10, Tables 14-18, Figure 1). Each analysis has a
script in scripts/revision/ that works from the per-item records in results/revision/ and the
files of the main evaluation, with no GPU, no model and none of the generation records.

This runs every script into a temporary directory, so nothing published is overwritten, and
compares each recomputed number with the published one. Two further checks come first:

1. The MOME score of every item of the four further-family backbones is recomputed from its
   26 features with the frozen predictor (model/frozen_predictor_newfamily.json), and so is
   its stage-1 decision against the frozen tau_high.
2. The published copies in results/revision/*.json are the ones in ssot.json.

A mismatch is a failure, and the exit code is non-zero.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mome"))
import paths  # noqa: E402
import gate   # noqa: E402

REVISION = paths.RESULTS / "revision"
SCRIPTS = ROOT / "scripts" / "revision"
#: Analysis -> script. Those marked True re-fit the predictor (scikit-learn) and are slow.
ANALYSES = {
    "output_stats_full": False, "xai_stratum_logprob": False, "serve_precision_heldout": False,
    "budget_arm": False, "latency_per_item": False, "stage2_samples": False, "new_family": False,
    "problem_disjoint": True, "family_out": True, "feature_ablation": True, "sensitivity": True,
}
TOL = 1e-6


class Audit:
    def __init__(self) -> None:
        self.checked = self.failed = 0

    def check(self, name: str, mine, published, tol: float = TOL) -> bool:
        self.checked += 1
        if isinstance(mine, (int, float)) and isinstance(published, (int, float)) \
                and not isinstance(mine, bool):
            ok = abs(float(mine) - float(published)) <= tol
        else:
            ok = mine == published
        if not ok:
            self.failed += 1
            print(f"XX  {name}: {mine!r} vs {published!r}   MISMATCH")
        return ok


def leaves(x, p=""):
    """(path, value) of every number, flag and string in a JSON value."""
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{p}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{p}[{i}]")
    else:
        yield p, x


def lookup(x, path: str):
    for part in path.strip("/").replace("[", "/[").split("/"):
        if not part:
            continue
        if part.startswith("["):
            x = x[int(part[1:-1])]
        else:
            x = x[part]
    return x


def compare(a: Audit, name: str, mine: dict, published: dict) -> None:
    n0, f0 = a.checked, a.failed
    for path, v in leaves(mine):
        if isinstance(v, str):
            continue                      # labels and notes, not numbers
        try:
            pub = lookup(published, path)
        except (KeyError, IndexError, TypeError):
            a.check(f"{name}{path} published", "present", "missing")
            continue
        a.check(f"{name}{path}", v, pub)
    print(f"{'ok' if a.failed == f0 else 'XX'}  {name:<26} {a.checked - n0:6d} numbers, "
          f"{a.failed - f0} mismatches")


def check_frozen_scores(a: Audit) -> None:
    model = json.loads((paths.MODEL / "frozen_predictor_newfamily.json").read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in (REVISION / "newfamily_items.jsonl").read_text(encoding="utf-8").splitlines()]
    n0, f0 = a.checked, a.failed
    worst, flips = 0.0, 0
    for r in rows:
        s = gate.c_safe(r["features"], model)
        worst = max(worst, abs(s - r["c_safe"]))
        flips += (s >= model["tau_high"]) != r["stage1_serve"]
    a.check("further families: largest c_safe difference", worst, 0.0, tol=1e-9)
    a.check("further families: stage-1 decisions that differ", flips, 0)
    a.check("further families: items", len(rows), 3600)
    print(f"{'ok' if a.failed == f0 else 'XX'}  {'frozen predictor':<26} {a.checked - n0:6d} checks, "
          f"{a.failed - f0} mismatches ({len(rows)} items, largest |c_safe| difference {worst:.1e})")


def main() -> int:
    fast = "--fast" in sys.argv
    ss = json.loads(paths.SSOT.read_text(encoding="utf-8"))["review_r1"]
    a = Audit()
    print("## 1. Scores of the further-family backbones\n")
    check_frozen_scores(a)
    print("\n## 2. Published copies against ssot.json\n")
    for name in ANALYSES:
        pub = json.loads((REVISION / f"{name}.json").read_text(encoding="utf-8"))
        compare(a, f"{name} (copy)", ss[name], pub)
    print("\n## 3. Every analysis recomputed from the per-item records\n")
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "MOME_REVISION_OUT": tmp}
        for name, slow in ANALYSES.items():
            if slow and fast:
                print(f"--  {name:<26} skipped (--fast)")
                continue
            t0 = time.time()
            r = subprocess.run([sys.executable, str(SCRIPTS / f"{name}.py")], env=env,
                               capture_output=True, text=True)
            if r.returncode != 0:
                a.check(f"{name} ran", r.returncode, 0)
                print(r.stderr[-2000:])
                continue
            mine = json.loads((Path(tmp) / f"{name}.json").read_text(encoding="utf-8"))
            compare(a, name, mine, ss[name])
            print(f"    ({time.time() - t0:.0f} s)")
    print(f"\n{a.checked} checks, {a.failed} mismatches")
    return 1 if a.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
