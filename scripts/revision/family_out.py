#!/usr/bin/env python3
"""Reviewer 3, major comment 3 (what existing records can answer): transfer to a held-out
model family (no generation).

The five backbones come from three families: Gemma 4 (gemma4-e2b, gemma4-e4b), Qwen
(qwen2-math-1.5b, qwen3.5-9b) and SOLAR (solar-10.7b, Llama architecture). The registered
backbone-out fit scores gemma4-e2b with a predictor that saw gemma4-e4b, and likewise for the
two Qwen models. Here the predictor applied to a backbone is fitted without any backbone of
its family; the threshold is the registered one, or re-derived with the registered
procedure on the fitting backbones only. Gate, imputation and read-outs as in
problem_disjoint.py.

    python3 scripts/revision/family_out.py   # writes results/revision/family_out.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from problem_disjoint import fit, inner_tau, score  # noqa: E402

OUT = C.OUT_DIR / "family_out.json"
FAMILY = {"gemma4-e2b": "gemma4", "gemma4-e4b": "gemma4", "qwen2-math-1.5b": "qwen",
          "qwen3.5-9b": "qwen", "solar-10.7b": "solar"}


def main() -> None:
    rows = C.load_dataset()
    names = sorted(rows[0]["features"].keys())
    items = C.load_items()
    key = lambda r: (r["cell"], r["item_id"])          # noqa: E731
    rows = [r for r in rows if key(r) in items]
    sc_map, ref_map, taus = {}, {}, {}
    for fam in sorted(set(FAMILY.values())):
        train = [r for r in rows if FAMILY[r["backbone"]] != fam]
        test = [r for r in rows if FAMILY[r["backbone"]] == fam]
        m = fit(train, names)
        t = inner_tau(train, names)
        taus[fam] = t
        for r, s in zip(test, score(m, test, names)):
            sc_map[key(r)] = (float(s), t)
        print("family-out", fam, len(train), len(test), round(t, 4), flush=True)
    for held in C.BACKBONES:                     # like-for-like reference (same features)
        train = [r for r in rows if r["backbone"] != held]
        test = [r for r in rows if r["backbone"] == held]
        for r, s in zip(test, score(fit(train, names), test, names)):
            ref_map[key(r)] = float(s)

    keys = sorted(items)
    its = [items[k] for k in keys]
    cells = [k[0] for k in keys]
    y = np.array([it["label"] for it in its])
    s = np.array([sc_map[k][0] for k in keys]); t_item = np.array([sc_map[k][1] for k in keys])
    ref = np.array([ref_map[k] for k in keys])
    boots = list(C.cell_resamples(cells))
    imp = C.nearest_band_rates(its)
    res = {"families": FAMILY, "tau_rederived": taus,
           "auroc": C.auroc(s, y), "auroc_ci": C.ci([C.auroc(s[i], y[i]) for i in boots]),
           "auroc_refit_backbone_out": C.auroc(ref, y),
           "auroc_minus_refit_ci": C.ci([C.auroc(s[i], y[i]) - C.auroc(ref[i], y[i]) for i in boots]),
           "per_family": {}, "gate": {}}
    for fam in sorted(set(FAMILY.values())):
        msk = np.array([FAMILY[it["backbone"]] == fam for it in its])
        res["per_family"][fam] = {"n": int(msk.sum()), "auroc": C.auroc(s[msk], y[msk]),
                                  "auroc_refit_backbone_out": C.auroc(ref[msk], y[msk])}
    reg = C.registered_tau_high()
    for tname in ("registered_tau", "rederived_tau"):
        for a in ("decline", "greedy", "nearest"):
            if tname == "registered_tau":
                g = C.run_gate(its, s, reg, unobserved=a, impute=imp)
            else:
                g = C.run_gate(its, s - t_item, {bb: 0.0 for bb in C.BACKBONES},
                               unobserved=a, impute=imp)
            sm = C.summarize(g); sm["delta_yield_ci_pp"] = C.delta_yield_ci(g, cells)
            st1 = ~g["stage2"]
            sm["stage1_serve_frac"] = float(st1.mean())
            sm["stage1_serve_precision"] = float(y[st1].mean()) if st1.any() else None
            res["gate"][f"{tname}/unobserved={a}"] = sm
            print(tname, a, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in sm.items()
                             if k in ("coverage", "selective_acc", "delta_yield_pp", "n_unobserved",
                                      "stage1_serve_precision", "delta_yield_ci_pp")}, flush=True)
    print(json.dumps({k: res[k] for k in ("auroc", "auroc_ci", "auroc_refit_backbone_out",
                                          "auroc_minus_refit_ci", "per_family")}, indent=1))
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
