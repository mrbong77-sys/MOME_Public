#!/usr/bin/env python3
"""Draw every figure in the paper, from the published results.

    python3 scripts/figures/make_all.py

No model is called and no generation happens: each figure reads a committed
result file.  A figure whose input is absent is skipped and named at the end
rather than failing the run, so a partial checkout still produces what it can.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "mome"))
import paths                                    # noqa: E402

FIGURES = [
    (["fig1_indistinguishable.py"], "Figure 1, the indistinguishable strata"),
    (["fig2_architecture.py"], "Figure 2, the two-stage architecture"),
    (["fig4_risk_coverage.py"], "Figure 4, risk-coverage"),
    (["fig5_layer_sweep.py"], "Figure 5, the layer sweep"),
    (["fig6_quadrant.py", "--backbone", "gemma4-e4b"], "Figure 6, the quadrant"),
    (["fig7_forest.py"], "Figure 7, per-cell correct yield"),
]


def main() -> int:
    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    made, skipped = [], []
    for argv, label in FIGURES:
        script = HERE / argv[0]
        proc = subprocess.run([sys.executable, str(script), *argv[1:]],
                              capture_output=True, text=True)
        if proc.returncode == 0:
            made.append(label)
            print(f"ok   {label}")
        else:
            why = (proc.stderr or proc.stdout).strip().splitlines()
            skipped.append(label)
            print(f"skip {label}: {why[-1] if why else 'no reason given'}")

    files = sorted(p.name for p in paths.FIGURES.glob("*.pdf"))
    print(f"\ndrawn {len(made)}, skipped {len(skipped)}, "
          f"{len(files)} PDF files in {paths.FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
