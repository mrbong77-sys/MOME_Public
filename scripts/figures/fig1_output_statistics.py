#!/usr/bin/env python3
"""Figure 1 (revised after review round 1) -- output statistics of the runtime against the
MOME score on all greedy answers.

    python3 scripts/figures/fig1_output_statistics.py

Reads committed records only: results/s0_rows.jsonl (the five statistics and the stratum
of every greedy answer), results/stage1_decisions.jsonl (label) and
results/revision/output_stats_full.json (AUROC with 95% intervals on all items).

(a) Empirical cumulative distributions of mean token log-probability for all correct and
    all wrong greedy answers.
(b) AUROC with 95% interval on all greedy answers, for the five statistics and the MOME
    score. Open markers: the same statistics on the answers that stage 1 served
    immediately. That stratum was selected by the MOME score, which contains mean token
    log-probability, so its AUROCs are range-restricted (Reviewer 1, comment 2).

The previous Figure 1 (fig1_indistinguishable.py) showed only the stage-1 stratum.
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
from fig1_indistinguishable import auroc, ecdf  # noqa: E402

ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "mome"))
import paths                                    # noqa: E402
ROWS = paths.STRATA_ROWS
DEC = paths.DECISIONS
FULL = paths.RESULTS / "revision" / "output_stats_full.json"
OUT = paths.FIGURES / "fig1-output-statistics"
XLO = -0.20
REVERSED = {"ans_entropy", "n_tokens"}
METRICS = [("c_safe", "MOME score"), ("mean_logprob", "mean\nlog-prob"),
           ("min_logprob", "min\nlog-prob"), ("ans_logprob", "answer-span\nlog-prob"),
           ("ans_entropy", "answer-span\nentropy (rev.)"), ("n_tokens", "generated\ntokens (rev.)")]


def main() -> None:
    rows = [json.loads(l) for l in ROWS.read_text(encoding="utf-8").splitlines()]
    lab = {}
    for l in DEC.read_text(encoding="utf-8").splitlines():
        d = json.loads(l)
        lab[(d["cell"], d["item_id"])] = int(d["label"])
    full = json.loads(FULL.read_text(encoding="utf-8"))["auroc"]
    S.setup()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(S.FULL_W, 3.0),
                                   gridspec_kw={"width_ratios": [1.0, 1.15], "wspace": 0.55})

    ok = [r["mean_logprob"] for r in rows if lab[(r["cell"], r["item"])] == 1]
    no = [r["mean_logprob"] for r in rows if lab[(r["cell"], r["item"])] == 0]
    ecdf(ax1, ok, S.S1, "-", f"correct (n={len(ok)})")
    ecdf(ax1, no, S.S2, "--", f"wrong (n={len(no)})")
    ax1.set_xlim(XLO, 0)
    ax1.set_ylim(0, 1.02)
    ax1.set_xlabel("mean log-probability")
    ax1.set_ylabel("cumulative fraction")
    ax1.grid(axis="y", zorder=0)
    ax1.set_axisbelow(True)
    ax1.legend(loc="upper left", handlelength=1.6, fontsize=7.5, borderaxespad=0.2)
    ax1.set_title("(a) all greedy answers", loc="left", color=S.INK)

    a = [r for r in rows if r["stratum"] == "S1_CORRECT"]
    b = [r for r in rows if r["stratum"] == "S1_CONFIDENT_WRONG"]
    ypos = np.arange(len(METRICS))[::-1]
    for y, (key, _) in zip(ypos, METRICS):
        v = full[key]
        col = S.INK if key == "c_safe" else S.S1
        ax2.errorbar(v["point"], y, xerr=[[v["point"] - v["ci95"][0]], [v["ci95"][1] - v["point"]]],
                     fmt="o", color=col, ms=4, capsize=2, zorder=4)
        ax2.text(v["ci95"][1] + 0.01, y, f"{v['point']:.3f}", va="center", fontsize=7.5, color=S.INK)
        pa = [r[key] for r in a if r.get(key) is not None]
        pb = [r[key] for r in b if r.get(key) is not None]
        s1 = auroc(pa, pb)
        if key in REVERSED:
            s1 = 1 - s1
        ax2.plot(s1, y, marker="o", mfc="none", mec=S.S2, ms=4.5, ls="none", zorder=3)
    ax2.axvline(0.5, color=S.INK2, linewidth=0.9, zorder=2)
    ax2.set_yticks(ypos, [n for _, n in METRICS])
    ax2.set_xlim(0.42, 0.97)
    ax2.set_xlabel("AUROC  (0.5 = no discrimination)")
    ax2.grid(axis="x", zorder=0)
    ax2.set_axisbelow(True)
    ax2.plot([], [], marker="o", color=S.S1, ls="none", ms=4, label="all items, 95% CI")
    ax2.plot([], [], marker="o", mfc="none", mec=S.S2, ls="none", ms=4.5, label="stage-1 served only")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.35, -0.22), ncol=2, fontsize=7,
               frameon=False, handletextpad=0.3, columnspacing=1.0)
    ax2.set_title("(b) AUROC by signal", loc="left", color=S.INK)
    S.save(fig, OUT)


if __name__ == "__main__":
    main()
