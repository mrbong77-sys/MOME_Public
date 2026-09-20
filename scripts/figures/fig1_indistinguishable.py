#!/usr/bin/env python3
"""Figure 1 -- the model's own confidence does not separate right from wrong.

    python3 scripts/figures/fig1_indistinguishable.py

Reads the committed stratum records and nothing else.  No generation.

(a) Empirical cumulative distributions of mean token log-probability for the
    two strata.  The more they overlap, the less they are distinguishable, and
    the area between the curves is exactly how far the AUROC sits from 0.5 --
    so (a) and (b) show the same thing two ways and are read together.  Unlike
    a density estimate this needs no bandwidth choice, and it treats two strata
    of very different size (1338 against 99) fairly.
(b) AUROC of five output statistics between the same two strata. 0.5 is no
    discrimination.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "mome"))
import paths                                    # noqa: E402
ROWS = paths.STRATA_ROWS
OUT = paths.FIGURES / "fig1-indistinguishable"
#: The two strata, taken within the items the first stage served.
A, B = "S1_CORRECT", "S1_CONFIDENT_WRONG"
XLO = -0.20
METRICS = [("mean_logprob", "mean\nlog-prob"), ("min_logprob", "min\nlog-prob"),
           ("ans_logprob", "answer-span\nlog-prob"), ("ans_entropy", "answer-span\nentropy"),
           ("n_tokens", "generated\ntokens")]


def auroc(pos, neg) -> float:
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks, i = {}, 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    rp = sum(ranks[k] for k, (_, y) in enumerate(allv) if y == 1)
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def ecdf(ax, v, color, ls, label):
    x = np.sort(np.asarray(v, dtype=float))
    y = np.arange(1, len(x) + 1) / len(x)
    ax.step(np.concatenate([[XLO], x]), np.concatenate([[y[0] if x[0] < XLO else 0.0], y]),
            where="post", color=color, linestyle=ls, label=label)
    return x, y


def main() -> None:
    rows = [json.loads(l) for l in ROWS.read_text(encoding="utf-8").splitlines()]
    a = [r for r in rows if r["stratum"] == A]
    b = [r for r in rows if r["stratum"] == B]
    S.setup()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(S.FULL_W, 2.5),
                                   gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.42})

    # (a) Empirical cumulative distributions --------------------------------
    va = [r["mean_logprob"] for r in a if r.get("mean_logprob") is not None]
    vb = [r["mean_logprob"] for r in b if r.get("mean_logprob") is not None]
    ecdf(ax1, va, S.S1, "-", f"served & correct (n={len(va)})")
    ecdf(ax1, vb, S.S2, "--", f"confidently wrong (n={len(vb)})")
    below = sum(1 for v in vb if v < XLO)
    ax1.set_xlim(XLO, 0)
    ax1.set_ylim(0, 1.02)
    ax1.set_xlabel("mean log-probability")
    ax1.set_ylabel("cumulative fraction")
    ax1.grid(axis="y", zorder=0)
    ax1.set_axisbelow(True)
    ax1.legend(loc="upper left", handlelength=1.6, fontsize=7.5, borderaxespad=0.2)
    ax1.set_title("(a) the two strata overlap", loc="left", color=S.INK)
    if below:
        ax1.text(0.03, 0.62, f"{below} confidently-wrong items\nfall below the axis",
                 transform=ax1.transAxes, ha="left", va="top", fontsize=7, color=S.MUTED)

    # (b) AUROC bars ---------------------------------------------------------
    vals = []
    for key, _ in METRICS:
        pa = [r[key] for r in a if r.get(key) is not None]
        pb = [r[key] for r in b if r.get(key) is not None]
        vals.append(auroc(pa, pb))
    #: What is being measured is distance from 0.5, so the bars are anchored
    #: there: length is discriminative power, direction says which stratum
    #: scores higher. The quantity is bipolar, hence two colors about a
    #: neutral baseline.
    ypos = np.arange(len(METRICS))[::-1]
    ax2.barh(ypos, [v - 0.5 for v in vals], left=0.5, height=0.58, zorder=3,
             color=[S.S1 if v >= 0.5 else S.S2 for v in vals])
    ax2.axvline(0.5, color=S.INK2, linewidth=0.9, zorder=4)
    for y, v in zip(ypos, vals):
        off = 0.006 if v >= 0.5 else -0.006
        ax2.text(v + off, y, f"{v:.3f}", va="center", fontsize=8, color=S.INK,
                 ha="left" if v >= 0.5 else "right")
    ax2.set_yticks(ypos, [n for _, n in METRICS])
    ax2.set_xlim(0.34, 0.68)
    ax2.set_xticks([0.4, 0.5, 0.6])
    ax2.set_xlabel("AUROC  (0.5 = no discrimination)")
    ax2.grid(axis="x", zorder=0)
    ax2.set_axisbelow(True)
    ax2.set_title("(b) no indicator departs from 0.5", loc="left", color=S.INK)

    S.save(fig, OUT)


if __name__ == "__main__":
    main()
