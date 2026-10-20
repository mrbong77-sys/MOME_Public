"""Shared helpers for the round-1 review analyses (scripts/revision/).

Everything here is arithmetic on the published per-item records; nothing generates.
Outputs go to results/revision/, or to the directory named by MOME_REVISION_OUT
(scripts/verify_revision.py uses that to recompute without overwriting).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "mome"))
import paths  # noqa: E402
from evaluate_s import BACKBONES, feature_matrix  # noqa: E402,F401

REVISION = paths.RESULTS / "revision"
OUT_DIR = Path(os.environ.get("MOME_REVISION_OUT") or REVISION)
DATASET = paths.FEATURES
THRESHOLDS = paths.THRESHOLDS
ITEMS = REVISION / "items.jsonl"
VOTE_SERVE = 0.6
K = 2                      # stage-2 samples of the headline operating point
N_BOOT, SEED = 2000, 0


def load_dataset() -> list[dict]:
    return [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines()]


def load_items() -> dict:
    """(cell, item_id) -> per-item row of item_table.py."""
    out = {}
    for l in ITEMS.read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        out[(r["cell"], r["item_id"])] = r
    return out


def registered_tau_high() -> dict:
    t = json.loads(THRESHOLDS.read_text(encoding="utf-8"))["per_backbone"]
    return {bb: v["tau_high"] for bb, v in t.items()}


def problem_group(row: dict) -> str:
    """The English and Korean GSM8K cells hold the same problems with the same index."""
    idx = row["item_id"].rsplit("-", 1)[1]
    return ("math-" if row["benchmark"] == "math500" else "gsm-") + idx


def auroc(scores, labels) -> float:
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # average ranks over ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cell_resamples(cells: list[str], n_boot: int = N_BOOT, seed: int = SEED):
    """Index arrays for bootstrap replicates resampled within each cell."""
    rng = np.random.default_rng(seed)
    groups: dict = {}
    for i, c in enumerate(cells):
        groups.setdefault(c, []).append(i)
    groups = [np.array(v) for _, v in sorted(groups.items())]
    for _ in range(n_boot):
        yield np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])


def ci(values) -> list[float]:
    v = np.asarray(values, float)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def nearest_band_rates(items: list[dict], k: int = K, vote_serve: float = VOTE_SERVE,
                       n_near: int = 50) -> dict:
    """Imputation for items without stage-2 samples: for each backbone and greedy label, the
    rate of (served, served-and-correct) among the n_near observed stage-2 items with the
    highest MOME score, i.e. the items closest to the serve band."""
    out: dict = {}
    for bb in {it["backbone"] for it in items}:
        out[bb] = {}
        for lab in (0, 1):
            obs = sorted((it for it in items if it["backbone"] == bb and it["has_samples"]
                          and it["label"] == lab), key=lambda it: -it["c_safe"])[:n_near]
            sv = [it["vote"][str(k)][0] >= vote_serve for it in obs]
            ok = [s and it["vote"][str(k)][1] for s, it in zip(sv, obs)]
            out[bb][lab] = (float(np.mean(sv)) if obs else 0.0, float(np.mean(ok)) if obs else 0.0)
    return out


def run_gate(items: list[dict], scores, tau: dict, k: int = K,
             vote_serve: float = VOTE_SERVE, unobserved: str = "decline",
             impute: dict | None = None) -> dict:
    """Two-band gate with answer-equivalence voting (the adopted variant).

    items[i] carries backbone, label, probe_gens, has_samples and vote; scores[i] is the
    MOME score used for item i. An item below tau without stage-2 samples cannot be voted:
    `unobserved` says what to assume for it -- "decline" (lower bound on yield), "greedy"
    (served with its greedy answer, i.e. the vote agreed with the greedy answer) or
    "nearest" (expected outcome from `impute`, see nearest_band_rates).
    Returns per-item served/correct arrays and the summary.
    """
    n = len(items)
    served = np.zeros(n); correct = np.zeros(n)
    gens = np.zeros(n); unobs = np.zeros(n, bool); stage2 = np.zeros(n, bool)
    for i, (it, s) in enumerate(zip(items, scores)):
        gens[i] = it["probe_gens"] + 1.0
        if s >= tau[it["backbone"]]:
            served[i] = True; correct[i] = bool(it["label"]); continue
        stage2[i] = True
        if k == 0:
            continue
        if not it["has_samples"]:
            unobs[i] = True
            gens[i] += k
            if unobserved == "greedy":
                served[i] = True; correct[i] = bool(it["label"])
            elif unobserved == "nearest":
                served[i], correct[i] = impute[it["backbone"]][it["label"]]
            continue
        gens[i] += min(k, it["n_samples"])
        share, ok = it["vote"][str(k)]
        if share >= vote_serve:
            served[i] = True; correct[i] = bool(ok)
    labels = np.array([it["label"] for it in items], bool)
    return {"served": served, "correct": correct, "gens": gens, "unobserved": unobs,
            "stage2": stage2, "labels": labels}


def summarize(g: dict) -> dict:
    n = len(g["served"])
    a = float(g["served"].sum()); c = float(g["correct"].sum())
    van = float(g["labels"].mean())
    return {"n": n, "coverage": a / n, "selective_acc": c / a if a else float("nan"),
            "yield": c / n, "delta_yield_pp": (c / n - van) * 100,
            "gens_per_item": float(g["gens"].mean()),
            "stage2_frac": float(g["stage2"].mean()),
            "n_unobserved": int(g["unobserved"].sum()), "ungated_acc": van}


def delta_yield_ci(g: dict, cells: list[str]) -> list[float]:
    d = g["correct"] - g["labels"].astype(float)
    return ci([d[idx].mean() * 100 for idx in cell_resamples(cells)])
