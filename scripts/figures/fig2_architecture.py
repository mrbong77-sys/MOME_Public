#!/usr/bin/env python3
"""Figure 2 -- the two-stage gate, with the shares that actually flow through it.

    python3 scripts/figures/fig2_architecture.py

The band shares are read from the committed first-stage decisions; no
generation.  Putting the measured shares on the diagram is deliberate: how much
flows where explains the cost and the conclusion at the same time.

Two rules the layout keeps.

1. **Leave the edges empty.**  A box flush against the axis boundary gets its
   border clipped even with a tight bounding box, so everything is drawn inside
   a margin on all four sides.
2. **Separate final states from intermediate ones, by color and by shape.**
   Verification is not a final action, it is a waypoint.  Only serving (green)
   and declining (red) end the path, and the diagram has to show at a glance
   that verification is the box between them.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "mome"))
import paths                                    # noqa: E402
DEC = paths.DECISIONS
OUT = paths.FIGURES / "fig2-architecture"

#: One font size (FS), only the first line bold. Every box has the same height
#: and line weight, on a fixed column/row grid. Color carries meaning only:
#: processing (grey), served (green), declined (red).
PROC_BG, PROC_EDGE = "#f2f2f2", "#4d4d4d"
SERVED_BG, SERVED_EDGE = "#e6f2ea", "#2e7d4f"
DECL_BG, DECL_EDGE = "#fbe9e4", "#b5452a"
FS = 7.5
LW = 0.8
H = 0.25                                  # height of every box
PITCH = 0.072                             # line pitch inside a box
COLS = {"in": (0.00, 0.10), "s1": (0.145, 0.245), "route": (0.46, 0.24), "out": (0.795, 0.205)}
ROWS = {"top": 0.78, "mid": 0.47, "bot": 0.16}   # row centres


def box(ax, col, row, lines, bg, edge):
    x, w = COLS[col]
    y = ROWS[row] - H / 2
    ax.add_patch(FancyBboxPatch(
        (x, y), w, H, boxstyle="round,pad=0,rounding_size=0.012",
        linewidth=LW, edgecolor=edge, facecolor=bg, zorder=3))
    n = len(lines)
    for i, ln in enumerate(lines):
        #: Same pitch in every box, so lines align whatever the line count.
        ax.text(x + w / 2, ROWS[row] + PITCH * ((n - 1) / 2 - i), ln, ha="center", va="center",
                fontsize=FS, fontweight=("bold" if i == 0 else "normal"),
                color=S.INK, zorder=4)
    return x, y, w


def arrow(ax, p, q, color=PROC_EDGE):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle="-|>", mutation_scale=8, linewidth=LW, color=color,
        zorder=2, shrinkA=0, shrinkB=0))


def main() -> None:
    if not DEC.exists():
        raise SystemExit(f"{DEC.name} is missing. This figure puts the measured "
                         "band shares on the diagram and cannot be drawn without it.")
    c = collections.Counter(json.loads(l)["decision"]
                            for l in DEC.read_text(encoding="utf-8").splitlines())
    n = sum(c.values())
    serve = c["SERVE"] / n
    stage2 = (c["RETRY"] + c["DECLINE"]) / n

    S.setup()
    fig, ax = plt.subplots(figsize=(S.FULL_W, 2.55))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.10, 1.02)
    ax.axis("off")

    # ---- Column heads -------------------------------------------------------------
    for col, label in (("in", "Input"), ("s1", "Stage 1"), ("route", "Route"),
                       ("out", "Final state")):
        x, w = COLS[col]
        ax.text(x + w / 2, 0.98, label, ha="center", va="center", fontsize=FS,
                color=S.INK2, fontweight="bold")

    # ---- Boxes ---------------------------------------------------------------
    box(ax, "in", "mid", ["Item $x$"], "#ffffff", PROC_EDGE)
    box(ax, "s1", "mid", ["Score", "answer + $\\leq$2 probes",
                          r"fingerprint $\rightarrow$ $c_{\mathrm{safe}}$"], PROC_BG, PROC_EDGE)
    box(ax, "route", "top", [f"Serve ({serve:.1%})", "no samples drawn"],
        SERVED_BG, SERVED_EDGE)
    box(ax, "route", "bot", [f"Verify ({stage2:.1%})", "2 samples, 3 votes",
                             r"serve if share $\geq 0.6$"], PROC_BG, PROC_EDGE)
    box(ax, "out", "top", ["Served", "greedy answer"], SERVED_BG, SERVED_EDGE)
    box(ax, "out", "mid", ["Served", "majority answer"], SERVED_BG, SERVED_EDGE)
    box(ax, "out", "bot", ["Declined", "no majority"], DECL_BG, DECL_EDGE)

    # ---- Arrows -------------------------------------------------------------
    xr = lambda col: COLS[col][0] + COLS[col][1]          # right edge
    xl = lambda col: COLS[col][0]                          # left edge
    arrow(ax, (xr("in"), ROWS["mid"]), (xl("s1"), ROWS["mid"]))
    arrow(ax, (xr("s1"), ROWS["mid"] + 0.06), (xl("route"), ROWS["top"]), SERVED_EDGE)
    arrow(ax, (xr("s1"), ROWS["mid"] - 0.06), (xl("route"), ROWS["bot"]))
    arrow(ax, (xr("route"), ROWS["top"]), (xl("out"), ROWS["top"]), SERVED_EDGE)
    arrow(ax, (xr("route"), ROWS["bot"] + 0.06), (xl("out"), ROWS["mid"]), SERVED_EDGE)
    arrow(ax, (xr("route"), ROWS["bot"]), (xl("out"), ROWS["bot"]), DECL_EDGE)
    #: Stage-1 branch conditions sit outside the arrows, same font size; the
    #: stage-2 conditions are written inside the final-state boxes.
    ax.text(0.425, 0.70, r"$c_{\mathrm{safe}} \geq \tau_{\mathrm{high}}$", ha="right",
            va="center", fontsize=FS, color=SERVED_EDGE)
    ax.text(0.425, 0.25, "otherwise", ha="right", va="center", fontsize=FS, color=S.INK2)

    # ---- Cost ---------------------------------------------------------------
    ax.text(0.5, -0.06,
            r"Generations per item $= 1 + p + 2 \times$ (share routed to verify),"
            r"   $p$ = mean probes per item",
            ha="center", va="center", fontsize=FS, color=S.INK2)

    S.save(fig, OUT)


if __name__ == "__main__":
    main()
