#!/usr/bin/env python3
"""Choose the two policy thresholds, and size the bands they produce.

Split discipline, inherited from the predictor's own: a threshold is chosen
on the training side only.  Within each backbone-out fold, an inner
leave-one-backbone-out CV over the four training backbones fixes the
thresholds, and the held-out backbone is then read exactly once with them.

The two objectives were registered before the data was seen:

  tau_high = the lowest threshold whose serve precision on inner validation
             still meets SERVE_PRECISION_TARGET -- that is, serve as many
             items immediately as the precision contract allows
  tau_low  = the highest threshold below which inner-validation accuracy is
             at or under DECLINE_ACC_CEILING -- below that, an answer is not
             worth serving and the item is declined outright

Reports the serve / retry / decline shares per held-out backbone, the serve
precision, the accuracy of the decline band, and the size of the middle band,
which is what the second stage has to pay for.

    python3 thresholds.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import paths
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_s import BACKBONES, feature_matrix  # noqa: E402

P2_OUT = paths.MODEL
OUT = paths.REPORTS

#: Registered constants, fixed before the data was seen.
#: The serve-precision floor comes from the precision band an earlier gate
#: measured (roughly 94-96%).  It was not chosen on this data.
SERVE_PRECISION_TARGET = 0.95
#: At or below this accuracy an answer is not worth serving, so decline.
DECLINE_ACC_CEILING = 0.20
K_SAMPLES = 4                   # second-stage sample paths
SC_TEMPERATURE = 0.7            # sampling temperature for those paths


def fit(train, names):
    X = feature_matrix(train, names); y = np.array([r["label"] for r in train])
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(X), y)
    return sc, clf


def score(sc, clf, rows, names):
    return clf.predict_proba(sc.transform(feature_matrix(rows, names)))[:, 1]


def pick_tau_high(scores, labels):
    """Lowest threshold that still meets the precision target, which is the
    one that serves the most.  None when no threshold does."""
    order = np.argsort(-scores)
    ys = labels[order]
    cum = np.cumsum(ys)
    best = None
    for k in range(1, len(ys) + 1):
        if cum[k - 1] / k >= SERVE_PRECISION_TARGET:
            best = float(scores[order][k - 1])
    return best


def pick_tau_low(scores, labels):
    """Highest threshold under which accuracy stays at or below the ceiling."""
    order = np.argsort(scores)          # ascending: lowest scores first
    ys = labels[order]
    cum = np.cumsum(ys)
    best = 0.0
    for k in range(1, len(ys) + 1):
        if cum[k - 1] / k <= DECLINE_ACC_CEILING:
            best = float(scores[order][k - 1])
    return best


def main():
    rows = [json.loads(l) for l in paths.FEATURES.read_text(encoding="utf-8").splitlines()]
    names = sorted(rows[0]["features"].keys())

    lines = ["# Policy thresholds and bands (chosen on the training side only)", "",
             f"Registered constants: serve precision target >= {SERVE_PRECISION_TARGET:.2f}, "
             f"decline-band accuracy ceiling <= {DECLINE_ACC_CEILING:.2f}, "
             f"second stage k={K_SAMPLES} at temperature {SC_TEMPERATURE} "
             f"(an operating point validated earlier)", "",
             "| held-out backbone | tau_high | tau_low | serve | retry | decline "
             "| serve precision | decline-band accuracy |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    totals = {"serve": 0, "retry": 0, "decline": 0, "n": 0, "serve_ok": 0}
    per_bb = {}
    for held in BACKBONES:
        train = [r for r in rows if r["backbone"] != held]
        test = [r for r in rows if r["backbone"] == held]
        # Inner CV: leave one out among the four training backbones to collect
        # threshold candidates.
        inner_high, inner_low = [], []
        inner_bbs = sorted({r["backbone"] for r in train})
        for inner in inner_bbs:
            itr = [r for r in train if r["backbone"] != inner]
            iva = [r for r in train if r["backbone"] == inner]
            sc, clf = fit(itr, names)
            s = score(sc, clf, iva, names); y = np.array([r["label"] for r in iva])
            th = pick_tau_high(s, y); tl = pick_tau_low(s, y)
            if th is not None:
                inner_high.append(th)
            inner_low.append(tl)
        tau_high = float(np.median(inner_high)) if inner_high else 0.95
        # Keep the bands from degenerating: tau_low <= tau_high.  Maximizing
        # the two objectives independently lets them cross and the retry band
        # vanishes; the first parameterization did exactly that, landing at
        # tau_low 0.663 above tau_high 0.604, a band of 0.2%.  Serve precision
        # is the honesty contract and takes priority, so the decline band is
        # only ever taken below it.
        tau_low = min(float(np.median(inner_low)), tau_high)

        # Apply to the held-out backbone, read exactly once.
        sc, clf = fit(train, names)
        s = score(sc, clf, test, names); y = np.array([r["label"] for r in test])
        serve = s >= tau_high
        decline = s < tau_low
        retry = ~serve & ~decline
        sp = y[serve].mean() if serve.sum() else float("nan")
        da = y[decline].mean() if decline.sum() else float("nan")
        lines.append(f"| {held} | {tau_high:.3f} | {tau_low:.3f} | {serve.mean():.1%} | "
                     f"{retry.mean():.1%} | {decline.mean():.1%} | {sp:.3f} | {da:.3f} |")
        per_bb[held] = {"tau_high": round(tau_high, 4), "tau_low": round(tau_low, 4),
                        "serve_frac": round(float(serve.mean()), 4),
                        "retry_frac": round(float(retry.mean()), 4),
                        "decline_frac": round(float(decline.mean()), 4),
                        "serve_precision": None if serve.sum() == 0 else round(float(sp), 4),
                        "decline_band_acc": None if decline.sum() == 0 else round(float(da), 4)}
        totals["serve"] += int(serve.sum()); totals["retry"] += int(retry.sum())
        totals["decline"] += int(decline.sum()); totals["n"] += len(y)
        totals["serve_ok"] += int(y[serve].sum())

    n = totals["n"]
    lines += ["",
              f"**Total** -- serve {totals['serve']/n:.1%} (precision {totals['serve_ok']/max(totals['serve'],1):.3f}), "
              f"retry {totals['retry']/n:.1%}, decline {totals['decline']/n:.1%}, items {n}",
              "",
              "## Generation cost (only the retry band generates anew)",
              ""]
    retry_frac = totals["retry"] / n
    for label, items in (("300 items per cell", 300), ("500 items per cell", 500)):
        gen = retry_frac * items * 3 * 5 * K_SAMPLES   # 3 benchmarks x 5 backbones x k
        extra_stage1 = 0 if items == 300 else (items - 300) * 3 * 5 * 3  # 1 greedy + ~2 probes
        lines.append(f"- **{label}**: second stage at k={K_SAMPLES} ~ {gen:,.0f} generations"
                     + (f" plus ~{extra_stage1:,.0f} for the wider first stage "
                        f"(greedy and probes, needs a new probe manifest)" if extra_stage1 else
                        " (no extra first-stage generation; existing records are reused)"))
    lines += ["",
              "Note: the retry seed is 1. Seed sensitivity is re-checked on one",
              "backbone with three seeds. The first stage is greedy, so it "
              "carries no seed dependence at all."]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "p3-thresholds.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (paths.THRESHOLDS).write_text(
        json.dumps({"constants": {"serve_precision_target": SERVE_PRECISION_TARGET,
                                  "decline_acc_ceiling": DECLINE_ACC_CEILING,
                                  "k_samples": K_SAMPLES, "sc_temperature": SC_TEMPERATURE},
                    "per_backbone": per_bb}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
