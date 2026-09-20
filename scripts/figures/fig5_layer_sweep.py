#!/usr/bin/env python3
"""Figure 5 -- the internal probe across all layers, and what perturbation adds.

    python3 scripts/figures/fig5_layer_sweep.py

Reads the small per-layer summary the readout left behind; the hidden states
themselves are not needed.  No generation.

Top: the probe's cross-validated AUROC at each layer.  Bottom: at the same
layers, the difference between the combined signal and the probe alone, with
95% intervals.  Where the lower band sits above zero, perturbation is adding
information the internal representation does not carry, and showing every
layer is what makes that more than a coincidence at one or two of them.

The backbones have different depths (43 layers against 33), so the horizontal
axis is relative depth: 0 at the input, 1 at the output.
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
OUT = paths.FIGURES / "fig5-layer-sweep"
SERIES = [("gemma4-e4b", S.S1, "-", "o"), ("qwen3.5-9b", S.S2, "--", "s")]


def load(bb: str) -> dict | None:
    p = paths.LAYERS / f"p5-s1-layers-{bb}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", default=",".join(b for b, *_ in SERIES),
                    help="comma-separated backbone names; change only to test the plumbing")
    args = ap.parse_args()
    names = [b.strip() for b in args.backbones.split(",") if b.strip()]
    marks = [(S.S1, "-", "o"), (S.S2, "--", "s"), (S.S3, "-.", "^")]
    data = [(bb, *marks[i % len(marks)], load(bb)) for i, bb in enumerate(names)]
    have = [d for d in data if d[4]]
    if not have:
        raise SystemExit("no per-layer summary found in results/layers/.\n"
                         "  This figure needs the hidden-state readout, which is "
                         "published with the data.")

    S.setup()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(S.FULL_W * 0.72, 3.9), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1], "hspace": 0.30})

    for bb, color, ls, mk, d in have:
        n = max(x["layer"] for x in d["layers"]) or 1
        xs = [x["layer"] / n for x in d["layers"]]
        ys = [x["cv_auroc"] for x in d["layers"]]
        ax1.plot(xs, ys, color=color, linestyle=ls, marker=mk, markersize=3,
                 markeredgewidth=0, label=bb, zorder=3)
        if d.get("gate_auroc") is not None:
            ax1.axhline(d["gate_auroc"], color=color, linestyle=":", linewidth=0.9, zorder=2)
            ax1.text(1.005, d["gate_auroc"], "  external $c_{safe}$", color=color,
                     fontsize=6.5, va="center", ha="left", transform=ax1.get_yaxis_transform())
        j = d.get("judge", {}).get("layer")
        if j is not None:
            ax1.plot([j / n], [d["median_auroc"]], marker="o", markersize=6.5,
                     markerfacecolor="none", markeredgecolor=color,
                     markeredgewidth=1.2, zorder=4)

        sw = d.get("sweep") or []
        xs2 = [x["layer"] / n for x in sw]
        mid = [x["mix_minus_probe"] for x in sw]
        lo = [x["lo"] for x in sw]
        hi = [x["hi"] for x in sw]
        ax2.fill_between(xs2, lo, hi, color=color, alpha=0.13, linewidth=0, zorder=2)
        ax2.plot(xs2, mid, color=color, linestyle=ls, zorder=3, label=bb)

    ax1.axhline(0.5, color=S.MUTED, linewidth=0.8, zorder=1)
    ax1.set_ylabel("probe CV AUROC")
    ax1.grid(axis="y", zorder=0)
    ax1.set_axisbelow(True)
    ax1.legend(loc="lower left", ncol=2, handlelength=2.0, fontsize=7.5,
               borderaxespad=0.2)
    ax1.set_title("(a) hidden-state probe, layer by layer   "
                  "(open circle = median layer, used for the verdict)",
                  loc="left", fontsize=7.5, color=S.INK)

    ax2.axhline(0.0, color=S.INK2, linewidth=0.9, zorder=1)
    ax2.set_ylabel("combined $-$ probe")
    ax2.set_xlabel("relative depth   (0 = input, 1 = output)")
    ax2.grid(axis="y", zorder=0)
    ax2.set_axisbelow(True)
    ax2.set_title("(b) what the external perturbation signal adds  "
                  "(band = 95% paired interval)", loc="left", fontsize=7.5, color=S.INK)
    ax2.set_xlim(-0.02, 1.02)
    S.save(fig, OUT)


if __name__ == "__main__":
    main()
