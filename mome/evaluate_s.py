#!/usr/bin/env python3
"""Fit and score the symbolic predictor against the free baselines.

Splits, all pre-registered:

  backbone-out      5 folds -- train without one backbone, score on it, so
                    the predictor meets a model it never saw
  distribution-out  2 folds -- prose benchmarks against the competition set,
                    with translation pairs kept on the same side so that no
                    problem leaks across the split in its other language
  cell-out          train without the single cell whose signal runs the other
                    way, and score on it

Baselines: mean token log-probability, and a free composite of truncation and
hidden-token count.  Both are sign-flipped so that higher always means "more
likely correct", which puts them on the same axis as the predictor.

The metric is AUROC of c_safe, the probability an answer is correct.
Detecting wrong answers is the symmetric problem (1 - AUROC), so everything
is reported on the one axis: the probability a correct answer is ranked above
a wrong one.  Intervals are item bootstraps.

    python3 evaluate_s.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import paths
OUT = paths.RESULTS
DATA = paths.FEATURES

BACKBONES = ["gemma4-e2b", "gemma4-e4b", "qwen3.5-9b", "solar-10.7b", "qwen2-math-1.5b"]
PROSE = {"gsm8k", "hrm8k-gsm8k-ko"}


def load() -> list[dict]:
    return [json.loads(l) for l in DATA.read_text(encoding="utf-8").splitlines()]


def feature_matrix(rows: list[dict], names: list[str]) -> np.ndarray:
    X = np.zeros((len(rows), len(names)))
    for i, r in enumerate(rows):
        for j, n in enumerate(names):
            v = r["features"].get(n)
            X[i, j] = 0.0 if v is None else float(v)
    return X


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """P(score of a correct answer > score of a wrong one), labels 1 = correct.

    Wrong-answer detection is the symmetric problem, so this one number covers
    both readings.
    """
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    # Ties take the average rank.
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts); starts = csum - counts
    avg = (starts + csum + 1) / 2.0
    ranks = avg[inv]
    r_pos = ranks[labels == 1].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def boot_ci(scores, labels, n=1000, seed=0):
    rng = random.Random(seed)
    idx = list(range(len(scores)))
    vals = []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        a = auroc(scores[s], labels[s])
        if a == a:
            vals.append(a)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


def train_eval(train, test, names):
    Xtr = feature_matrix(train, names); ytr = np.array([r["label"] for r in train])
    Xte = feature_matrix(test, names); yte = np.array([r["label"] for r in test])
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight=None)
    clf.fit(sc.transform(Xtr), ytr)
    p = clf.predict_proba(sc.transform(Xte))[:, 1]  # P(correct)
    return p, yte


def baseline_logprob(test):
    yte = np.array([r["label"] for r in test])
    lp = np.array([r["features"].get("nc_mean_logprob") if r["features"].get("nc_mean_logprob") is not None
                   else -10.0 for r in test])
    return lp, yte  # higher means more likely correct


def baseline_nocost(test):
    """Free composite: truncation and hidden tokens, negated so that the axis
    matches the others.  Needs no probe."""
    yte = np.array([r["label"] for r in test])
    s = np.array([-(2.0 * r["features"]["nc_truncated"] + 0.02 * r["features"]["nc_hidden_token"])
                  for r in test])
    return s, yte


def report_split(name, folds, rows, names):
    print(f"\n## {name}")
    print(f"{'fold':28}{'n':>5}{'symbolic':>10}{'logprob':>9}{'free':>8}")
    agg = {"s": [], "lp": [], "nc": []}
    for fold_name, train, test in folds:
        if not train or not test:
            continue
        ps, y = train_eval(train, test, names)
        lp, _ = baseline_logprob(test)
        nc, _ = baseline_nocost(test)
        a_s, a_lp, a_nc = auroc(ps, y), auroc(lp, y), auroc(nc, y)
        agg["s"].append(a_s); agg["lp"].append(a_lp); agg["nc"].append(a_nc)
        print(f"{fold_name:28}{len(test):>5}{a_s:>8.3f}{a_lp:>9.3f}{a_nc:>8.3f}")
    import statistics
    print(f"{'mean':28}{'':>5}{statistics.mean(agg['s']):>10.3f}"
          f"{statistics.mean(agg['lp']):>9.3f}{statistics.mean(agg['nc']):>8.3f}")
    return agg


def main():
    rows = load()
    names = sorted(rows[0]["features"].keys())

    # backbone-out, 5 folds
    bo = [(f"−{b}", [r for r in rows if r["backbone"] != b],
           [r for r in rows if r["backbone"] == b]) for b in BACKBONES]
    report_split("backbone-out (unseen model, 5 folds)", bo, rows, names)

    # distribution-out, 2 folds, translation pairs kept on the same side
    do = [
        ("prose→math500", [r for r in rows if r["benchmark"] in PROSE],
         [r for r in rows if r["benchmark"] == "math500"]),
        ("math500→prose", [r for r in rows if r["benchmark"] == "math500"],
         [r for r in rows if r["benchmark"] in PROSE]),
    ]
    report_split("distribution-out (translation pairs kept together, 2 folds)", do, rows, names)

    # cell-out: qwen2-math × ko
    co = [("−(qwen2-math×ko)",
           [r for r in rows if r["cell"] != "hrm8k-gsm8k-ko_qwen2-math-1.5b"],
           [r for r in rows if r["cell"] == "hrm8k-gsm8k-ko_qwen2-math-1.5b"])]
    report_split("cell-out (the cell whose signal runs the other way)", co, rows, names)

    # Registered decision: on every backbone-out fold, does the symbolic
    # predictor beat mean log-probability, with a bootstrap interval?
    print("\n## Does the symbolic predictor beat mean log-probability on every "
          "backbone? (bootstrap 95%)")
    verdict = {}
    for b in BACKBONES:
        train = [r for r in rows if r["backbone"] != b]
        test = [r for r in rows if r["backbone"] == b]
        ps, y = train_eval(train, test, names)
        lp, _ = baseline_logprob(test)
        a_s = auroc(ps, y); a_lp = auroc(lp, y)
        lo, hi = boot_ci(ps, y)
        # Bootstrap the difference, not the two AUROCs separately.
        rng = random.Random(1); idx = list(range(len(y))); diffs = []
        for _ in range(1000):
            s = np.array([rng.choice(idx) for _ in idx])
            diffs.append(auroc(ps[s], y[s]) - auroc(lp[s], y[s]))
        diffs.sort(); dlo, dhi = diffs[25], diffs[975]
        win = dlo > 0
        verdict[b] = {"auroc_s": round(a_s, 3), "auroc_logprob": round(a_lp, 3),
                      "s_ci": [round(lo, 3), round(hi, 3)],
                      "diff_ci": [round(dlo, 3), round(dhi, 3)], "s_wins": bool(win)}
        print(f"  {b:16} S={a_s:.3f} [{lo:.3f},{hi:.3f}]  logprob={a_lp:.3f}  "
              f"d95%=[{dlo:+.3f},{dhi:+.3f}]  {'wins' if win else 'not established'}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "s_verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    print(f"\nwritten: {OUT/'s_verdict.json'}")


if __name__ == "__main__":
    main()
