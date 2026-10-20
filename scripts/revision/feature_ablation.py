#!/usr/bin/env python3
"""Reviewer 3, major comment 2: which components of the 26-feature fingerprint carry the
MOME score (no generation).

Every variant is fitted exactly as the registered predictor (standardize -> logistic L2,
C = 1, backbone-out: the model applied to a backbone is fitted on the other four), and it
scores all 4,497 greedy answers. Two read-outs:

* AUROC on all items (the set on which the full score reaches 0.863),
* selective accuracy when the score alone selects the adopted gate's coverage (81.2%) of
  all items by global ranking (the matched-coverage comparison of the paper, 81.9% for the
  full score vs 76.4% for mean token log-probability).

Variants: the full model; each feature group alone; the full model without each group.
Intervals: 2,000 bootstrap replicates resampled within cells, paired with the full model
(fitted models held fixed).

    python3 scripts/revision/feature_ablation.py   # writes results/revision/feature_ablation.json
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

OUT = C.OUT_DIR / "feature_ablation.json"
COVERAGE = 0.812

#: Groups follow the three groups of the manuscript (Section 3.3), with the third split into
#: what is measured on the original answer alone and what compares it with the probes.
GROUPS = {
    "must_change": ["flip_n", "flip_change_viol_any", "flip_changed_frac", "flip_dir_ok_frac",
                    "scale_n", "scale_change_viol_any", "scale_changed_frac",
                    "scale_dir_ok_frac", "change_confirmed", "spurious_consistency"],
    "invariant": ["hold_n", "hold_violation_any", "hold_violation_frac", "hold_instability"],
    "counts_truncation": ["n_probes", "n_unanswered", "frac_unanswered", "orig_truncated",
                          "nc_truncated"],
    "probe_vs_original_text": ["max_text_sim", "mean_text_sim", "min_len_ratio", "max_len_ratio"],
    "original_answer_stats": ["nc_mean_logprob", "nc_token_count", "nc_hidden_token"],
}
#: Union of the two probe-answer groups: everything that needs an edited problem's answer.
COMBOS = {"probe_answers": GROUPS["must_change"] + GROUPS["invariant"]}


def backbone_out_scores(rows, names):
    s = np.zeros(len(rows))
    coefs = {}
    for held in C.BACKBONES:
        tr = [i for i, r in enumerate(rows) if r["backbone"] != held]
        te = [i for i, r in enumerate(rows) if r["backbone"] == held]
        X = C.feature_matrix([rows[i] for i in tr], names)
        y = np.array([rows[i]["label"] for i in tr])
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(X), y)
        s[te] = clf.predict_proba(sc.transform(C.feature_matrix([rows[i] for i in te], names)))[:, 1]
        coefs[held] = dict(zip(names, map(float, clf.coef_[0])))
    return s, coefs


def acc_at_coverage(s, y, cov=COVERAGE):
    k = round(cov * len(s))
    order = np.argsort(-s, kind="mergesort")
    return float(y[order[:k]].mean())


def main() -> None:
    rows = C.load_dataset()
    assert len(rows) == 4497, len(rows)
    allnames = sorted(rows[0]["features"].keys())
    grouped = sorted(sum(GROUPS.values(), []))
    assert grouped == allnames, set(allnames) ^ set(grouped)
    y = np.array([r["label"] for r in rows])
    cells = [r["cell"] for r in rows]
    boots = list(C.cell_resamples(cells))

    full, coefs = backbone_out_scores(rows, allnames)
    variants = {"full": allnames}
    for g, fs in {**GROUPS, **COMBOS}.items():
        variants[f"only:{g}"] = fs
        variants[f"without:{g}"] = [f for f in allnames if f not in fs]
    res = {"n_items": len(rows), "coverage": COVERAGE, "groups": {**GROUPS, **COMBOS},
           "variants": {}}
    for name, fs in variants.items():
        s = full if name == "full" else backbone_out_scores(rows, sorted(fs))[0]
        a = C.auroc(s, y); acc = acc_at_coverage(s, y)
        d_a = [C.auroc(s[i], y[i]) - C.auroc(full[i], y[i]) for i in boots]
        d_c = [acc_at_coverage(s[i], y[i]) - acc_at_coverage(full[i], y[i]) for i in boots]
        res["variants"][name] = {
            "n_features": len(fs), "auroc": a,
            "auroc_ci": C.ci([C.auroc(s[i], y[i]) for i in boots]),
            "auroc_minus_full_ci": C.ci(d_a), "acc_at_coverage": acc,
            "acc_at_coverage_minus_full_ci": C.ci(d_c)}
        print(f"{name:34}{len(fs):>3}  AUROC {a:.3f}  acc@{COVERAGE:.1%} {acc:.3f}", flush=True)

    # standardized coefficients of the five backbone-out fits (mean and range)
    res["std_coef"] = {f: {"mean": float(np.mean([coefs[b][f] for b in C.BACKBONES])),
                           "min": float(np.min([coefs[b][f] for b in C.BACKBONES])),
                           "max": float(np.max([coefs[b][f] for b in C.BACKBONES]))}
                       for f in allnames}
    # how often each kind of violation occurred among correct and wrong greedy answers
    mc = [r for r in rows if r["features"]["flip_n"] + r["features"]["scale_n"] > 0]
    res["violation_rates"] = {
        "must_change": {"n_items_with_probe": len(mc),
                        "wrong": float(np.mean([r["features"]["spurious_consistency"] for r in mc if r["label"] == 0])),
                        "correct": float(np.mean([r["features"]["spurious_consistency"] for r in mc if r["label"] == 1]))},
        "invariant": {"wrong": float(np.mean([r["features"]["hold_violation_any"] for r in rows if r["label"] == 0])),
                      "correct": float(np.mean([r["features"]["hold_violation_any"] for r in rows if r["label"] == 1]))}}
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("written", OUT)


if __name__ == "__main__":
    main()
