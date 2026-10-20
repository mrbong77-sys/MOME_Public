#!/usr/bin/env python3
"""Backbones of further families, and the stage-2 temperature (Section 4.9, Tables 16 and 17).

    python3 scripts/revision/new_family.py     # writes results/revision/new_family.json

Four backbones of further families (llama3.2-3b, phi4-mini, exaone3.5-7.8b, granite3.3-8b)
answered the same 900 problems per backbone. Their answers were scored by a predictor and a
tau_high fitted once on the five backbones of the main evaluation and then frozen
(model/frozen_predictor_newfamily.json); nothing from the new cells reached the fit. The gate
is the one of the main evaluation: two bands, vote share >= 0.6 over greedy + k samples under
answer equivalence, k = 2 and k = 4. Stage-2 samples were drawn at temperature 0.7 (as in the
main evaluation) and at 0.3. The five backbones of the main evaluation were read again with
their stage-2 samples drawn at 0.3, with their registered per-backbone tau_high.

Inputs:
  results/revision/newfamily_items.jsonl        per item: the 26 features, c_safe under the
                                                 frozen predictor, label, mean log-probability,
                                                 and the ballots of k = 1..4 at 0.7 and 0.3
  results/revision/registered_temperature.jsonl  per item of the main evaluation: the ballots
                                                 of k = 2 and k = 4 at 0.7 and 0.3

A ballot is [vote share of the plurality answer, whether that answer is correct], counted on
Linux with the registered answer equivalence. Read-outs: AUROC of the MOME score and of mean
token log-probability with their item-paired difference, coverage, selective accuracy and
correct yield of the gate; 2,000 bootstrap replicates resampled within cells, seed 0.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

SRC_NEW = C.REVISION / "newfamily_items.jsonl"
SRC_REG = C.REVISION / "registered_temperature.jsonl"
FROZEN = C.paths.MODEL / "frozen_predictor_newfamily.json"
OUT = C.OUT_DIR / "new_family.json"

SEC23 = ["llama3.2-3b", "phi4-mini"]          # the first two further-family backbones
EXT = ["exaone3.5-7.8b", "granite3.3-8b"]     # the two added after them
KS = (2, 4)
PRIMARY_K = 2
YIELD_MARGIN_PP = -2.0                        # criterion 2: CI lower bound of the yield difference


def ci(values) -> list[float | None]:
    v = np.asarray([x for x in values if x == x], float)
    if not len(v):
        return [None, None]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def _f(x):
    return None if x is None or x != x else float(x)


def evaluate(items: list[dict], tau_high: float) -> dict:
    """All read-outs for one set of items (one backbone, or pooled)."""
    cells = [it["cell"] for it in items]
    y = np.array([it["label"] for it in items])
    s = np.array([it["c_safe"] for it in items])
    lp = np.array([it["mean_logprob"] if it["mean_logprob"] is not None else -np.inf for it in items])
    boots = list(C.cell_resamples(cells))
    a_m, a_l = C.auroc(s, y), C.auroc(lp, y)
    bm = [C.auroc(s[i], y[i]) for i in boots]
    bl = [C.auroc(lp[i], y[i]) for i in boots]
    res = {"n": len(items), "n_cells": len(set(cells)), "n_correct": int(y.sum()),
           "auroc_mome": _f(a_m), "auroc_mome_ci": ci(bm),
           "auroc_mean_logprob": _f(a_l), "auroc_mean_logprob_ci": ci(bl),
           "auroc_diff": _f(a_m - a_l), "auroc_diff_ci": ci([m - l for m, l in zip(bm, bl)]),
           "ungated_acc": float(y.mean()) if len(y) else None,
           "stage1_serve_frac": float(np.mean(s >= tau_high)) if len(s) else None,
           "stage1_serve_precision": (float(y[s >= tau_high].mean()) if (s >= tau_high).any() else None),
           "no_samples": int(sum(not it["has_samples"] for it in items)),
           "gate": {}}
    tau = {it["backbone"]: tau_high for it in items}
    for k in KS:
        g = C.run_gate(items, s, tau, k=k)
        sm = C.summarize(g)
        served, correct = g["served"], g["correct"]
        lab = g["labels"].astype(float)
        d_sel, d_yield = [], []
        for idx in boots:
            a = served[idx].sum()
            d_sel.append(correct[idx].sum() / a - lab[idx].mean() if a else float("nan"))
            d_yield.append((correct[idx] - lab[idx]).mean() * 100)
        sel = sm["selective_acc"]
        res["gate"][f"k={k}"] = {
            "coverage": _f(sm["coverage"]), "selective_acc": _f(sel),
            "correct_yield": _f(sm["yield"]), "ungated_acc": _f(sm["ungated_acc"]),
            "delta_yield_pp": _f(sm["delta_yield_pp"]), "delta_yield_ci_pp": ci(d_yield),
            "sel_minus_ungated_pp": _f((sel - sm["ungated_acc"]) * 100) if sel == sel else None,
            "sel_minus_ungated_ci_pp": [None if v is None else v * 100 for v in ci(d_sel)],
            "gens_per_item": _f(sm["gens_per_item"]), "stage2_frac": _f(sm["stage2_frac"]),
            "n_unobserved": sm["n_unobserved"],
            "served": int(served.sum()), "correct": int(correct.sum())}
    res["adoption"] = adoption(res)
    return res


def adoption(res: dict) -> dict:
    """The registered rule between k = 2 and k = 4: among variants whose correct yield is at
    least the ungated accuracy, the highest selective accuracy; ties go to the lower cost."""
    g = res["gate"]
    ok = {k: v for k, v in g.items() if v["correct"] >= res["n_correct"]}
    if not ok:
        return {"pick": None, "eligible": [], "note": "no variant reaches the ungated yield"}
    pick = max(ok.items(), key=lambda kv: (kv[1]["selective_acc"] or 0.0, -kv[1]["gens_per_item"]))
    return {"pick": pick[0], "eligible": sorted(ok)}


def criteria(per_bb: dict, pooled: dict) -> dict:
    """The four criteria registered before generation for the further-family backbones."""
    out = {}
    c1 = {}
    for bb, r in per_bb.items():
        lo = r["auroc_diff_ci"][0]
        c1[bb] = {"auroc_diff": r["auroc_diff"], "ci": r["auroc_diff_ci"],
                  "met": bool(lo is not None and lo > 0)}
    out["1_discrimination_per_backbone"] = {"rule": "item-paired AUROC difference (MOME - mean "
                                                    "log-probability), 95% CI above 0, each new backbone",
                                            "per_backbone": c1,
                                            "met": bool(c1) and all(v["met"] for v in c1.values())}
    g = pooled["gate"][f"k={PRIMARY_K}"]
    lo = g["delta_yield_ci_pp"][0]
    out["2_yield_preserved_pooled"] = {
        "rule": f"pooled, k={PRIMARY_K}: delta correct yield point >= 0 and 95% CI lower bound "
                f">= {YIELD_MARGIN_PP} points",
        "delta_yield_pp": g["delta_yield_pp"], "ci_pp": g["delta_yield_ci_pp"],
        "met": bool(g["delta_yield_pp"] is not None and g["delta_yield_pp"] >= 0
                    and lo is not None and lo >= YIELD_MARGIN_PP)}
    lo = g["sel_minus_ungated_ci_pp"][0]
    out["3_selective_accuracy_pooled"] = {
        "rule": f"pooled, k={PRIMARY_K}: selective accuracy - ungated accuracy, 95% CI above 0",
        "diff_pp": g["sel_minus_ungated_pp"], "ci_pp": g["sel_minus_ungated_ci_pp"],
        "met": bool(lo is not None and lo > 0)}
    out["4_k_report_only"] = {
        "rule": "report k=2 and k=4 and the pick of the registered adoption rule",
        "k2": {k: pooled["gate"]["k=2"][k] for k in ("correct_yield", "selective_acc", "gens_per_item")},
        "k4": {k: pooled["gate"]["k=4"][k] for k in ("correct_yield", "selective_acc", "gens_per_item")},
        "rule_pick": pooled["adoption"]["pick"], "met": None}
    return out


def crit_at(pooled: dict, k: str) -> dict:
    """Criteria 2 and 3 read at one k (both k are reported side by side)."""
    g = pooled["gate"][k]
    lo_y, lo_s = g["delta_yield_ci_pp"][0], g["sel_minus_ungated_ci_pp"][0]
    return {"2_yield_preserved_pooled": {"delta_yield_pp": g["delta_yield_pp"], "ci_pp": g["delta_yield_ci_pp"],
                                         "met": bool(g["delta_yield_pp"] >= 0 and lo_y >= YIELD_MARGIN_PP)},
            "3_selective_accuracy_pooled": {"diff_pp": g["sel_minus_ungated_pp"], "ci_pp": g["sel_minus_ungated_ci_pp"],
                                            "met": bool(lo_s > 0)}}


def block(items: list[dict], backbones: list[str], tau: float) -> dict:
    per = {bb: evaluate([it for it in items if it["backbone"] == bb], tau) for bb in backbones}
    pooled = evaluate(items, tau)
    cells: dict = {}
    for it in items:
        cells.setdefault(it["cell"], []).append(it)
    return {"per_backbone": per, "pooled": pooled, "criteria": criteria(per, pooled),
            "per_cell": {c: evaluate(v, tau) for c, v in sorted(cells.items())},
            "criteria_by_k": {k: crit_at(pooled, k) for k in ("k=2", "k=4")}}


def registered(rows: list[dict], t: str, k: int, tau: dict) -> dict:
    """The five backbones of the main evaluation at one temperature and k."""
    items = []
    for r in rows:
        b = r[t][f"k={k}"]
        items.append({**{x: r[x] for x in ("cell", "item_id", "backbone", "label", "c_safe", "probe_gens")},
                      "has_samples": b["has_samples"], "n_samples": b["n_samples"],
                      "vote": {str(k): b["vote"]} if b["vote"] is not None else {}})

    def evaluate_reg(its: list[dict]) -> dict:
        s = np.array([it["c_safe"] for it in its])
        g = C.run_gate(its, s, tau, k=k)
        sm = C.summarize(g)
        served, correct, lab = g["served"], g["correct"], g["labels"].astype(float)
        dy, ds = [], []
        for idx in C.cell_resamples([it["cell"] for it in its]):
            a = served[idx].sum()
            dy.append((correct[idx] - lab[idx]).mean() * 100)
            ds.append((correct[idx].sum() / a - lab[idx].mean()) * 100 if a else float("nan"))
        q = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]  # noqa: E731
        return {"n": len(its), "ungated_acc": sm["ungated_acc"], "coverage": sm["coverage"],
                "selective_acc": sm["selective_acc"], "correct_yield": sm["yield"],
                "delta_yield_pp": sm["delta_yield_pp"], "delta_yield_ci_pp": q(dy),
                "sel_minus_ungated_pp": (sm["selective_acc"] - sm["ungated_acc"]) * 100,
                "sel_minus_ungated_ci_pp": q(ds), "n_unobserved": sm["n_unobserved"]}

    bbs = sorted({it["backbone"] for it in items})
    return {"per_backbone": {bb: evaluate_reg([it for it in items if it["backbone"] == bb]) for bb in bbs},
            "pooled": evaluate_reg(items)}


def main() -> int:
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    tau = frozen["tau_high"]
    rows = [json.loads(l) for l in SRC_NEW.read_text(encoding="utf-8").splitlines() if l.strip()]
    res = {}
    for t in ("T0.7", "T0.3"):
        items = [{**{k: r[k] for k in ("cell", "item_id", "backbone", "bench", "label", "c_safe",
                                        "mean_logprob", "probe_gens")}, **r[t]} for r in rows]
        sec23 = [it for it in items if it["backbone"] in SEC23]
        ext = [it for it in items if it["backbone"] in EXT]
        res[t] = {"sec23": block(sec23, SEC23, tau), "ext": block(ext, EXT, tau),
                  "all4": block(sec23 + ext, SEC23 + EXT, tau)}
    reg_rows = [json.loads(l) for l in SRC_REG.read_text(encoding="utf-8").splitlines() if l.strip()]
    reg_tau = C.registered_tau_high()
    out = {"tau_high_frozen": tau, "tau_high_registered": reg_tau,
           "k4_at_0.3": {"new_family": True, "registered": True}, "results": res,
           "sec26": {t: registered(reg_rows, t, 2, reg_tau) for t in ("T0.7", "T0.3")},
           "sec26_k4": {t: registered(reg_rows, t, 4, reg_tau) for t in ("T0.7", "T0.3")}}
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float) + "\n", encoding="utf-8")
    for t in ("T0.7", "T0.3"):
        for k in ("k=2", "k=4"):
            g = res[t]["all4"]["pooled"]["gate"][k]
            r = out["sec26" if k == "k=2" else "sec26_k4"][t]["pooled"]
            print(f"{t} {k}  further families: yield {g['delta_yield_pp']:+.2f} pp, sel. acc "
                  f"{g['selective_acc'] * 100:.1f}%   main evaluation: yield {r['delta_yield_pp']:+.2f} pp, "
                  f"sel. acc {r['selective_acc'] * 100:.1f}%")
    print("written", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
