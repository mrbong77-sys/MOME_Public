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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "mome"))
import paths                                    # noqa: E402
DEC = paths.DECISIONS
OUT = paths.FIGURES / "fig2-architecture"

#: Pale fills with a matching border. Border weight carries the distinction
#: too, so color is never the only cue.
SERVED_BG, SERVED_EDGE = "#eaf5ef", "#2f8a5f"
DECL_BG, DECL_EDGE = "#fdeee8", "#c2552b"
PROC_BG, PROC_EDGE = "#eef3fa", "#3f6ea8"
LANE = "#f6f6f4"
FS = 7.4


def box(ax, x, y, w, h, lines, bg, edge, lw=1.0, bold_first=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.028",
        linewidth=lw, edgecolor=edge, facecolor=bg, zorder=3))
    n = len(lines)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h * (n - i - 0.5) / n, ln, ha="center", va="center",
                fontsize=FS + (0.6 if (bold_first and i == 0) else 0),
                fontweight=("bold" if (bold_first and i == 0) else "normal"),
                color=S.INK, zorder=4)
    return (x, y, w, h)


def elbow(ax, p, q, color=S.INK2, lw=1.0, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle="-|>", mutation_scale=9, linewidth=lw, color=color,
        zorder=2, shrinkA=2, shrinkB=3,
        connectionstyle=f"arc3,rad={rad}"))


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
    fig, ax = plt.subplots(figsize=(S.FULL_W, 2.75))
    #: Leave the edges empty; this is where an earlier version clipped.
    ax.set_xlim(-0.030, 1.050)
    ax.set_ylim(-0.11, 0.99)
    ax.axis("off")

    #: No vertical bands. SERVE is a first-stage outcome, and a band would
    #: make it look like it sits inside the second stage -- the diagram would
    #: then teach the structure wrongly. The labels go above their boxes.

    # ---- Input ---------------------------------------------------------------
    b_in = box(ax, 0.000, 0.425, 0.140, 0.150, ["item $x$"], "#ffffff", S.INK2, lw=0.9)

    # ---- First stage -----------------------------------------------------------
    b_s1 = box(ax, 0.185, 0.335, 0.225, 0.330,
               ["greedy answer", "+ 2 probes", r"$\rightarrow$ fingerprint",
                r"$\rightarrow$ $c_{\mathrm{safe}}$"], PROC_BG, PROC_EDGE, lw=1.1)
    elbow(ax, (b_in[0] + b_in[2], 0.500), (0.185, 0.500))
    #: A stage label on this column would be wrong: the third column holds
    #: both a first-stage outcome (SERVE) and the second stage (VERIFY). The
    #: split between fixed and conditional cost is already stated by the cost
    #: expression below, so nothing is said here.

    # ---- Branches --------------------------------------------------------------
    b_serve1 = box(ax, 0.470, 0.700, 0.245, 0.210,
                   [f"SERVE  ·  {serve:.1%}", "no samples bought"],
                   SERVED_BG, SERVED_EDGE, lw=1.1, bold_first=True)
    b_ver = box(ax, 0.470, 0.075, 0.245, 0.235,
                [f"VERIFY  ·  {stage2:.1%}", "+2 samples", r"3 votes, share $\geq 0.6$"],
                PROC_BG, PROC_EDGE, lw=1.1, bold_first=True)

    elbow(ax, (0.412, 0.612), (0.468, 0.792), rad=-0.16)
    elbow(ax, (0.412, 0.392), (0.468, 0.212), rad=0.16)
    #: Branch conditions sit OUTSIDE the arrows; inside they always collide
    #: with the lines.
    ax.text(0.404, 0.760, r"$c_{\mathrm{safe}} \geq \tau_{\mathrm{high}}$",
            ha="right", va="center", fontsize=FS - 0.4, color=SERVED_EDGE, zorder=4)
    ax.text(0.404, 0.245, "otherwise", ha="right", va="center",
            fontsize=FS - 0.4, color=S.INK2, zorder=4)

    # ---- Final states ------------------------------------------------------------
    b_out1 = box(ax, 0.762, 0.700, 0.265, 0.210, ["ANSWER SERVED"],
                 SERVED_BG, SERVED_EDGE, lw=1.3, bold_first=True)
    b_out2 = box(ax, 0.762, 0.340, 0.265, 0.195, ["ANSWER SERVED", "majority held"],
                 SERVED_BG, SERVED_EDGE, lw=1.3, bold_first=True)
    b_out3 = box(ax, 0.762, 0.030, 0.265, 0.210,
                 ["DECLINE", "with a reason", "the device can show"],
                 DECL_BG, DECL_EDGE, lw=1.3, bold_first=True)

    elbow(ax, (0.715, 0.805), (0.762, 0.805), color=SERVED_EDGE)
    elbow(ax, (0.715, 0.255), (0.762, 0.418), color=SERVED_EDGE, rad=-0.16)
    elbow(ax, (0.715, 0.148), (0.762, 0.132), color=DECL_EDGE, rad=0.10)

    # ---- Cost ------------------------------------------------------------------
    ax.text(0.5, -0.085,
            r"cost per item  $=$  1 answer  $+$  $p$ probes (fixed)  $+$  "
            r"2 $\times$ (verify share)",
            ha="center", va="bottom", fontsize=FS + 0.3, color=S.INK2, zorder=4)

    S.save(fig, OUT)


if __name__ == "__main__":
    main()
