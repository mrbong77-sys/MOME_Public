#!/usr/bin/env python3
"""Figure 6 -- where the model is confident out loud and wrong inside.

    python3 scripts/figures/fig6_quadrant.py --backbone gemma4-e4b

Reads the per-layer summary for one backbone.  No generation.

The horizontal axis is the confidence the model shows outwardly, its mean
token log-probability; the vertical axis is the probability of correctness
that the internal probe assigns.  Lower left is where inside and outside agree
that the model does not know.  Lower RIGHT is the interesting quadrant: the
internal representation says wrong while the output sounds certain.
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


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="gemma4-e4b")
    args = ap.parse_args()
    src = paths.LAYERS / f"p5-s1-layers-{args.backbone}.json"
    if not src.exists():
        raise SystemExit(f"{src.name} is missing.\n"
                         "  This figure needs the hidden-state readout, which is "
                         "published with the data.")
    d = json.loads(src.read_text(encoding="utf-8"))
    q = d.get("quadrant")
    if not q:
        raise SystemExit("no quadrant data for this backbone: too few items "
                         "carry log-probabilities.")
    pts = q["points"]
    lp_med = q["logprob_median"]
    out = paths.FIGURES / f"fig6-quadrant-{args.backbone}"

    cell = [p for p in pts if p["p_correct"] < 0.5 and p["mean_logprob"] >= lp_med]
    wrong_in_cell = sum(1 for p in cell if not p["correct"])
    declined = sum(1 for p in cell if p["declined"])
    base_wrong = sum(1 for p in pts if not p["correct"]) / len(pts)

    S.setup()
    fig, ax = plt.subplots(figsize=(S.FULL_W * 0.60, 2.9))

    #: Lay the interesting quadrant down first; that is where the eye goes.
    xhi = max(p["mean_logprob"] for p in pts)
    ax.axvspan(lp_med, xhi + 0.02, ymin=0, ymax=0.5, color=S.S2, alpha=0.07,
               linewidth=0, zorder=1)

    for correct, color, mk, lab in ((True, S.S1, "o", "actually correct"),
                                    (False, S.S2, "^", "actually wrong")):
        v = [p for p in pts if p["correct"] is correct]
        ax.scatter([p["mean_logprob"] for p in v], [p["p_correct"] for p in v],
                   s=7, marker=mk, facecolor=color, edgecolor="none",
                   alpha=0.45, label=f"{lab} (n={len(v)})", zorder=3)

    ax.axhline(0.5, color=S.INK2, linewidth=0.8, zorder=2)
    ax.axvline(lp_med, color=S.INK2, linewidth=0.8, zorder=2)

    #: Just inside that quadrant, below the 0.5 line, away from the dense cloud.
    note = (f"internally wrong,\noutwardly confident\n"
            f"{len(cell)} items, {wrong_in_cell} wrong ({wrong_in_cell / len(cell):.0%})\n"
            f"gate declined {declined}" if cell else "no items in this quadrant")
    ax.text(0.985, 0.475, note, transform=ax.transAxes, ha="right", va="top",
            fontsize=7, color=S.INK, zorder=4, linespacing=1.4)
    ax.text(0.015, 0.03, f"base error rate {base_wrong:.0%}", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=7, color=S.MUTED, zorder=4)

    ax.set_xlabel("outward confidence   (mean log-probability)")
    ax.set_ylabel("internal probe   $P(\\mathrm{correct})$")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", handlelength=1.2, fontsize=7.5, borderaxespad=0.2,
              scatterpoints=1, markerscale=1.6)
    ax.set_title(f"{args.backbone}", loc="left", fontsize=8, color=S.INK2)
    S.save(fig, out)


if __name__ == "__main__":
    main()
