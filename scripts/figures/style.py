"""Shared style for the paper's figures.

The journal is a print medium, so there is no interaction layer and no series
is distinguished by color alone: every series also carries a line style or a
hatch, and where there are two, each is labelled directly in addition to the
legend.  Only the first three slots of the validated palette are used, which is
the range that passes an all-pairs contrast check.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

#: Light-mode values from the validated palette; first three slots only.
S1 = "#2a78d6"   # blue
S2 = "#eb6834"   # orange
S3 = "#1baf7a"   # teal
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8c8b86"
GRID = "#e3e2de"
SURFACE = "#ffffff"

#: The journal's template is single-column; the text block is about 160 mm.
FULL_W = 6.3       # inch
HALF_W = 3.1


def setup() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.6,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix("." + ext))
    print(f"written: {path.with_suffix('.pdf')}")
