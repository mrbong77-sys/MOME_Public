#!/usr/bin/env python3
"""Reviewer 3, major comment 1: calibration and test on disjoint problems (no generation).

The registered evaluation is backbone-out: the predictor and the threshold applied to a
backbone were fitted on the other four backbones, which answered the same problems. Here the
fitting data additionally exclude every problem that is scored:

* problem-disjoint (5-fold): problems are split into five folds; the English and Korean
  GSM8K versions of one problem stay in the same fold. For held-out backbone B and fold f,
  the predictor is fitted on the other four backbones' items whose problem is not in f, and
  it scores B's items in f. Every scored item is then new in both backbone and problem.
* benchmark-disjoint: the predictor for B is fitted on the other backbones' items of the
  other problem family only (GSM8K + HRM8K-ko vs MATH-500), so the scored benchmark is
  unseen.

Thresholds: (a) the registered tau_high of B, unchanged; (b) tau_high re-derived with the
registered procedure (inner leave-one-backbone-out on the fitting rows, serve precision
>= 0.95, median over inner folds), so that no scored problem enters the threshold either.

The gate is the adopted one (two bands, k = 2, vote share >= 0.6, answer equivalence).
Stage-2 samples exist only for items the registered gate sent to stage 2; an item that a
re-fitted score sends to stage 2 but that was served at stage 1 has no vote. Such items are
counted and reported under three readings (declined; served with the greedy answer; the
expected outcome of the observed stage-2 items nearest the serve band, common.py).

    python3 scripts/revision/problem_disjoint.py   # writes results/revision/problem_disjoint.json
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

OUT = C.OUT_DIR / "problem_disjoint.json"
N_FOLDS = 5
SERVE_PRECISION_TARGET = 0.95


def fit(rows, names):
    X = C.feature_matrix(rows, names); y = np.array([r["label"] for r in rows])
    sc = StandardScaler().fit(X)
    return sc, LogisticRegression(max_iter=2000).fit(sc.transform(X), y)


def score(model, rows, names):
    sc, clf = model
    return clf.predict_proba(sc.transform(C.feature_matrix(rows, names)))[:, 1]


def pick_tau_high(scores, labels):
    order = np.argsort(-scores); ys = labels[order]; cum = np.cumsum(ys)
    best = None
    for k in range(1, len(ys) + 1):
        if cum[k - 1] / k >= SERVE_PRECISION_TARGET:
            best = float(scores[order][k - 1])
    return best


def inner_tau(train, names):
    """The registered P3.1 rule, applied to the fitting rows only."""
    ths = []
    for inner in sorted({r["backbone"] for r in train}):
        itr = [r for r in train if r["backbone"] != inner]
        iva = [r for r in train if r["backbone"] == inner]
        th = pick_tau_high(score(fit(itr, names), iva, names), np.array([r["label"] for r in iva]))
        if th is not None:
            ths.append(th)
    return float(np.median(ths)) if ths else 0.95


def folds_of(rows):
    groups = sorted({C.problem_group(r) for r in rows})
    rng = np.random.default_rng(C.SEED)
    perm = rng.permutation(len(groups))
    return {groups[j]: int(i % N_FOLDS) for i, j in enumerate(perm)}


def main() -> None:
    rows = C.load_dataset()
    names = sorted(rows[0]["features"].keys())
    items = C.load_items()
    key = lambda r: (r["cell"], r["item_id"])          # noqa: E731
    rows = [r for r in rows if key(r) in items]
    assert len(rows) == 4497, len(rows)
    fold = folds_of(rows)
    reg_tau = C.registered_tau_high()

    schemes = {}
    # --- reference: backbone-out refitted on the same features (problems shared). On the
    #     registered features it reproduces the registered scores; every separation below
    #     is compared with this refit.
    s_bo = {}; tau_bo = {}
    for held in C.BACKBONES:
        train = [r for r in rows if r["backbone"] != held]
        test = [r for r in rows if r["backbone"] == held]
        m = fit(train, names)
        t = inner_tau(train, names)
        tau_bo[held] = t
        for r, s in zip(test, score(m, test, names)):
            s_bo[key(r)] = (float(s), t)
        print("backbone-out refit", held, round(t, 4), flush=True)
    schemes["backbone_out_refit"] = (s_bo, tau_bo)

    # --- problem-disjoint, 5 folds
    s_pd = {}; tau_pd = {}
    for held in C.BACKBONES:
        taus = []
        for f in range(N_FOLDS):
            train = [r for r in rows if r["backbone"] != held and fold[C.problem_group(r)] != f]
            test = [r for r in rows if r["backbone"] == held and fold[C.problem_group(r)] == f]
            m = fit(train, names)
            t = inner_tau(train, names)
            taus.append(t)
            for r, s in zip(test, score(m, test, names)):
                s_pd[key(r)] = (float(s), t)
        tau_pd[held] = taus
        print("problem-disjoint", held, [round(t, 4) for t in taus], flush=True)
    schemes["problem_disjoint_5fold"] = (s_pd, tau_pd)

    # --- benchmark-disjoint (problem family held out)
    fam = lambda r: "math" if r["benchmark"] == "math500" else "gsm"   # noqa: E731
    s_bd = {}; tau_bd = {}
    for held in C.BACKBONES:
        taus = {}
        for fm in ("gsm", "math"):
            train = [r for r in rows if r["backbone"] != held and fam(r) != fm]
            test = [r for r in rows if r["backbone"] == held and fam(r) == fm]
            m = fit(train, names)
            t = inner_tau(train, names)
            taus[fm] = t
            for r, s in zip(test, score(m, test, names)):
                s_bd[key(r)] = (float(s), t)
        tau_bd[held] = taus
        print("benchmark-disjoint", held, {k: round(v, 4) for k, v in taus.items()}, flush=True)
    schemes["benchmark_disjoint"] = (s_bd, tau_bd)

    keys = sorted(items)
    its = [items[k] for k in keys]
    cells = [k[0] for k in keys]
    labels = np.array([it["label"] for it in its])
    orig = np.array([it["c_safe"] for it in its])
    reg = C.run_gate(its, orig, reg_tau)
    result = {"n_items": len(keys), "n_folds": N_FOLDS, "seed": C.SEED,
              "registered": {"auroc": C.auroc(orig, labels), **C.summarize(reg),
                             "delta_yield_ci_pp": C.delta_yield_ci(reg, cells)},
              "schemes": {}}
    boots = list(C.cell_resamples(cells))
    imp = C.nearest_band_rates(its)
    for name, (sc_map, taus) in schemes.items():
        s = np.array([sc_map[k][0] for k in keys])
        t_item = np.array([sc_map[k][1] for k in keys])
        a_new, a_orig = C.auroc(s, labels), C.auroc(orig, labels)
        diffs = [C.auroc(s[i], labels[i]) - C.auroc(orig[i], labels[i]) for i in boots]
        if name != "backbone_out_refit":
            ref = np.array([schemes["backbone_out_refit"][0][k][0] for k in keys])
            res_ref = {"auroc_minus_refit": a_new - C.auroc(ref, labels),
                       "auroc_minus_refit_ci": C.ci([C.auroc(s[i], labels[i]) - C.auroc(ref[i], labels[i])
                                                     for i in boots])}
        else:
            res_ref = {}
        res = {"auroc": a_new, "auroc_ci": C.ci([C.auroc(s[i], labels[i]) for i in boots]),
               "auroc_minus_registered": a_new - a_orig, "auroc_minus_registered_ci": C.ci(diffs),
               "per_backbone_auroc": {}, "tau_rederived": taus, "gate": {}, **res_ref}
        for bb in C.BACKBONES:
            m = np.array([it["backbone"] == bb for it in its])
            res["per_backbone_auroc"][bb] = {"new": C.auroc(s[m], labels[m]),
                                             "registered": C.auroc(orig[m], labels[m])}
        # (a) registered tau ; (b) re-derived tau (per item, from its own fit)
        for tname, tau_map in (("registered_tau", reg_tau), ("rederived_tau", None)):
            for assume in ("decline", "greedy", "nearest"):
                if tau_map is None:
                    # per-item threshold: encode by shifting the score
                    shifted = s - t_item
                    g = C.run_gate(its, shifted, {bb: 0.0 for bb in C.BACKBONES}, unobserved=assume,
                                   impute=imp)
                else:
                    g = C.run_gate(its, s, tau_map, unobserved=assume, impute=imp)
                summ = C.summarize(g)
                summ["delta_yield_ci_pp"] = C.delta_yield_ci(g, cells)
                stage1 = ~g["stage2"]
                summ["stage1_serve_frac"] = float(stage1.mean())
                summ["stage1_serve_precision"] = float(labels[stage1].mean()) if stage1.any() else None
                res["gate"][f"{tname}/unobserved={assume}"] = summ
        result["schemes"][name] = res
        print(name, json.dumps({k: res[k] for k in ("auroc", "auroc_ci", "auroc_minus_registered",
                                                     "auroc_minus_refit", "auroc_minus_refit_ci") if k in res}),
              flush=True)
        for gk, gv in res["gate"].items():
            print("  ", gk, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in gv.items()
                            if k in ("coverage", "selective_acc", "yield", "delta_yield_pp",
                                     "gens_per_item", "n_unobserved", "stage1_serve_frac",
                                     "stage1_serve_precision", "delta_yield_ci_pp")}, flush=True)
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("written", OUT)


if __name__ == "__main__":
    main()
