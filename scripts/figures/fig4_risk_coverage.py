#!/usr/bin/env python3
"""Figure 4 -- risk-coverage: the fingerprint against the model's own confidence.

    python3 scripts/figures/fig4_risk_coverage.py

Reads the `curves` block of the published results and nothing else. No
generation.

The curves give selective accuracy when the same items are served at the same
rate.  Because coverage is matched, the figure isolates the discriminative
power of the signal: no policy and no extra generation enters into it.
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
OUT = paths.FIGURES / "fig4-risk-coverage"
#: Coverage of the adopted design, marking where the gate actually sits.
OP = 0.812


def main() -> None:
    if not SRC.exists():
        raise SystemExit("results/ssot.json is missing. Generate it first:\n"
                         "  python3 scripts/export_ssot.py")
    d = json.loads(SRC.read_text(encoding="utf-8"))["curves"]
    cs, ml = d["c_safe"], d["mean_logprob"]
    x = [r["coverage"] for r in cs]
    y1 = [r["acc"] for r in cs]
    y2 = [r["acc"] for r in ml]

    S.setup()
    fig, ax = plt.subplots(figsize=(S.FULL_W * 0.62, 2.7))
    ax.fill_between(x, y2, y1, color=S.S1, alpha=0.10, linewidth=0, zorder=1)
    ax.plot(x, y1, color=S.S1, linestyle="-", zorder=3,
            label="metamorphic fingerprint $c_{safe}$")
    ax.plot(x, y2, color=S.S2, linestyle="--", zorder=3,
            label="model's own confidence (mean log-prob)")

    #: Mark the operating point, so that what the figure shows connects to
    #: where the gate is used. The space above both curves is empty, so the
    #: annotation goes there.
    ax.axvline(OP, color=S.MUTED, linewidth=0.8, linestyle=":", zorder=2)
    ax.text(OP - 0.015, 0.955, "gate operates here\n(81.2% coverage)",
            ha="right", va="top", fontsize=7, color=S.INK2)

    #: Two series, so each is labelled directly as well as in the legend.
    #: The curves meet at 100% coverage, so the labels go where they separate.
    def at(t):
        return min(range(len(x)), key=lambda j: abs(x[j] - t))
    i1, i2 = at(0.33), at(0.55)
    ax.text(x[i1], y1[i1] + 0.010, "$c_{safe}$", color=S.S1,
            ha="center", va="bottom", fontsize=8)
    ax.text(x[i2], y2[i2] - 0.012, "log-prob", color=S.S2,
            ha="center", va="top", fontsize=8)

    ax.set_xlim(0.08, 1.02)
    ax.set_ylim(0.655, 0.97)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["20%", "40%", "60%", "80%", "100%"])
    ax.set_xlabel("coverage (fraction of items answered)")
    ax.set_ylabel("selective accuracy")
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", handlelength=1.8, fontsize=7.5, borderaxespad=0.2)
    S.save(fig, OUT)


if __name__ == "__main__":
    main()
