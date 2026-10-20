#!/usr/bin/env python3
"""Reviewer 3, major comment 4: sensitivity of the adopted gate to its thresholds and its
vote rule (no generation).

The adopted gate has two decision parameters: the stage-1 threshold tau_high (per backbone,
chosen backbone-out under a serve-precision target of 0.95) and the stage-2 vote rule
(serve the plurality answer when its share is >= 0.6, over greedy + k samples). tau_low is
not used by the adopted two-band gate.

1. Serve-precision target. tau_high is re-derived with the registered procedure for targets
   0.90 ... 0.98; every other step is unchanged.
2. tau_high shifted by a fixed amount on every backbone.
3. Vote threshold x number of samples k.

Lowering tau_high moves items from stage 2 to stage 1, which the records cover. Raising it
moves items that were served at stage 1 into stage 2; these items have no stage-2 samples.
They are counted (`n_unobserved`) and reported under two assumptions: declined (lower bound
on yield) and served with the greedy answer.

    python3 scripts/revision/sensitivity.py   # writes results/revision/sensitivity.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

OUT = C.OUT_DIR / "sensitivity.json"
TARGETS = [0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98]
SHIFTS = [-0.10, -0.05, -0.02, 0.0, 0.01, 0.02, 0.03]
#: Items moved into stage 2 by a stricter threshold have no stage-2 samples. Three readings:
#: "decline" (lower bound), "greedy" (served with the greedy answer; upper reading) and
#: "nearest" (expected outcome of the 50 observed stage-2 items closest to the serve band,
#: per backbone and greedy label).
VOTES = [0.34, 0.5, 0.6, 0.7, 0.8, 1.0]


def fit(rows, names):
    X = C.feature_matrix(rows, names); y = np.array([r["label"] for r in rows])
    sc = StandardScaler().fit(X)
    return sc, LogisticRegression(max_iter=2000).fit(sc.transform(X), y)


def tau_for_targets(rows, names):
    """Registered P3.1 rule (inner leave-one-backbone-out, median) for several targets."""
    out = {t: {} for t in TARGETS}
    for held in C.BACKBONES:
        train = [r for r in rows if r["backbone"] != held]
        per_t = {t: [] for t in TARGETS}
        for inner in sorted({r["backbone"] for r in train}):
            itr = [r for r in train if r["backbone"] != inner]
            iva = [r for r in train if r["backbone"] == inner]
            sc, clf = fit(itr, names)
            s = clf.predict_proba(sc.transform(C.feature_matrix(iva, names)))[:, 1]
            y = np.array([r["label"] for r in iva])
            order = np.argsort(-s); ys = y[order]; prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
            for t in TARGETS:
                ok = np.where(prec >= t)[0]
                if len(ok):
                    per_t[t].append(float(s[order][ok.max()]))
        for t in TARGETS:
            out[t][held] = float(np.median(per_t[t])) if per_t[t] else 0.95
    return out


def row(g, cells):
    s = C.summarize(g)
    s["delta_yield_ci_pp"] = C.delta_yield_ci(g, cells)
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in s.items()}


def main() -> None:
    items = C.load_items()
    keys = sorted(items)
    its = [items[k] for k in keys]
    cells = [k[0] for k in keys]
    s = np.array([it["c_safe"] for it in its])
    reg = C.registered_tau_high()
    # the recorded decision and the rounded score must agree with the registered tau
    mism = sum((sc >= reg[it["backbone"]]) != (it["decision"] == "SERVE") for sc, it in zip(s, its))
    res = {"n_items": len(its), "decision_mismatch_vs_record": int(mism), "registered_tau": reg}

    imp = C.nearest_band_rates(its)
    res["nearest_band_rates"] = {bb: {str(l): v for l, v in d.items()} for bb, d in imp.items()}
    rows = [r for r in C.load_dataset() if (r["cell"], r["item_id"]) in items]
    names = sorted(rows[0]["features"].keys())
    taus = tau_for_targets(rows, names)
    res["target"] = {}
    for t in TARGETS:
        res["target"][f"{t:.2f}"] = {"tau": taus[t],
                                     **{a: row(C.run_gate(its, s, taus[t], unobserved=a, impute=imp), cells)
                                        for a in ("decline", "greedy", "nearest")}}
        r = res["target"][f"{t:.2f}"]["nearest"]
        print(f"target {t:.2f}: cov {r['coverage']:.3f} acc {r['selective_acc']:.3f} "
              f"Δ {r['delta_yield_pp']:+.2f} gens {r['gens_per_item']:.2f} unobs {r['n_unobserved']}",
              flush=True)
    res["shift"] = {}
    for d in SHIFTS:
        tau = {bb: v + d for bb, v in reg.items()}
        res["shift"][f"{d:+.2f}"] = {a: row(C.run_gate(its, s, tau, unobserved=a, impute=imp), cells)
                                     for a in ("decline", "greedy", "nearest")}
        r = res["shift"][f"{d:+.2f}"]["nearest"]
        print(f"shift {d:+.2f}: cov {r['coverage']:.3f} acc {r['selective_acc']:.3f} "
              f"Δ {r['delta_yield_pp']:+.2f} gens {r['gens_per_item']:.2f} unobs {r['n_unobserved']}",
              flush=True)
    res["vote"] = {}
    for k in (1, 2, 3, 4):
        for v in VOTES:
            r = row(C.run_gate(its, s, reg, k=k, vote_serve=v), cells)
            res["vote"][f"k={k}/share>={v:.2f}"] = r
            print(f"k={k} share>={v:.2f}: cov {r['coverage']:.3f} acc {r['selective_acc']:.3f} "
                  f"Δ {r['delta_yield_pp']:+.2f} gens {r['gens_per_item']:.2f}", flush=True)
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("written", OUT)


if __name__ == "__main__":
    main()
