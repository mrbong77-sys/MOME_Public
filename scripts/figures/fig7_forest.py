#!/usr/bin/env python3
"""Figure 7 -- per-cell difference in correct yield, with intervals.

    python3 scripts/figures/fig7_forest.py

Reads the per-cell intervals from the published results. No generation.

One pooled average hides the heterogeneity across the 15 cells, and the
question that matters is whether the result is even or carried by a few cells.
Each cell's interval comes from resampling items within that cell only, with
the cell sizes held fixed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "mome"))
import paths                                    # noqa: E402
SRC = paths.SSOT
OUT = paths.FIGURES / "fig7-forest"

BENCH = {"gsm8k": "GSM8K (en)", "hrm8k-gsm8k-ko": "HRM8K (ko)",
         "math500": "MATH-500 (en)"}


def label(cell: str) -> str:
    bench, bb = cell.rsplit("_", 1)
    return f"{BENCH.get(bench, bench)}  ×  {bb}"


def main() -> int:
    if not SRC.exists():
        print(f"missing: {SRC} -- run scripts/export_ssot.py first.")
        return 1
    d = json.loads(SRC.read_text(encoding="utf-8"))
    rows = d.get("gate", {}).get("cells_ci")
    if not rows:
        print("ssot.json has no gate.cells_ci -- rerun export_ssot.py.")
        return 1
    unc = d["gate"].get("uncertainty") or {}

    #: Largest gain at the top, so the eye reads the ordering downward.
    rows = sorted(rows, key=lambda r: r["delta"])
    S.setup()
    fig, ax = plt.subplots(figsize=(S.FULL_W, 0.30 * len(rows) + 1.5))

    ax.axvline(0.0, color=S.INK, lw=0.8, zorder=1)
    for i, r in enumerate(rows):
        lo, hi, dv = r["lo"] * 100, r["hi"] * 100, r["delta"] * 100
        #: Grey when the interval covers zero, colored when it does not. The
        #: marker shape changes too, so color is not the only cue: a filled
        #: circle against an open one.
        crosses = lo <= 0 <= hi
        col = S.MUTED if crosses else (S.S1 if dv > 0 else S.S2)
        ax.plot([lo, hi], [i, i], color=col, lw=1.4, solid_capstyle="butt", zorder=2)
        ax.plot([dv], [i], marker="o", ms=4.2, color=col, zorder=3,
                markerfacecolor=(S.SURFACE if crosses else col))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label(r["cell"]) for r in rows])
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_xlabel("correct yield, gate − ungated (percentage points)")
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)

    if unc:
        lo, hi = unc["ci"]["delta"]
        pt = unc["point"]["delta"]
        ax.axvspan(lo * 100, hi * 100, color=S.S1, alpha=0.10, zorder=0)
        ax.annotate(f"all cells: {pt * 100:+.1f} p  [{lo * 100:+.1f}, {hi * 100:+.1f}]",
                    xy=(0.5, 1.02), xycoords="axes fraction", ha="center",
                    va="bottom", fontsize=8, color=S.INK2)

    fig.tight_layout()
    S.save(fig, OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
