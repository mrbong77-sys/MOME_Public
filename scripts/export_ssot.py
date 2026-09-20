#!/usr/bin/env python3
"""Recompute every number the manuscript uses, from the per-item records.

    python3 scripts/export_ssot.py        # -> results/ssot.json

This file is the manuscript's single source of truth.  Every table and every
figure in the paper is generated from it, so there is nowhere for a number to
be copied by hand -- not "check what you typed", but "never type it".

The slow part is calling the registered readout (`evaluate_gate`) as it
stands, because that runs the symbolic comparisons again.  Requires the
generation records; see REPRODUCE.md.
"""

from __future__ import annotations

import collections
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mome"))
import paths                                        # noqa: E402
import ollama_http as oh                            # noqa: E402
from build_dataset import mean_logprob              # noqa: E402
from greedy_path import by_item as greedy_by_item   # noqa: E402
import evaluate_gate as EG                          # noqa: E402

OUT = paths.SSOT
DEC = paths.DECISIONS
MAN = paths.MANIFESTS
AOUT = paths.REPORTS
P5OUT = paths.RESULTS
K_FINAL = 2
GRID = [i / 100 for i in range(10, 101, 5)]
#: Coverage of the two operating points the text uses. k=2 is the adopted
#: design; k=4 is the comparison row.
COVERAGES = {"k2": 0.812, "k4": 0.790}


def auroc(pos, neg):
    if not pos or not neg:
        return None
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks, i = {}, 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    rp = sum(ranks[k] for k, (_, y) in enumerate(allv) if y == 1)
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def paired_diff(a, b, y, n_boot=2000, seed=0):
    import random
    rng = random.Random(seed)
    n = len(y)
    base = auroc([x for x, t in zip(a, y) if t], [x for x, t in zip(a, y) if not t]) - \
        auroc([x for x, t in zip(b, y) if t], [x for x, t in zip(b, y) if not t])
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        if len({y[i] for i in idx}) < 2:
            continue
        aa = auroc([a[i] for i in idx if y[i]], [a[i] for i in idx if not y[i]])
        bb = auroc([b[i] for i in idx if y[i]], [b[i] for i in idx if not y[i]])
        vals.append(aa - bb)
    vals.sort()
    return [base, vals[int(0.025 * len(vals))], vals[min(len(vals) - 1, int(0.975 * len(vals)))]]


def probes_per_item():
    out = {}
    for f in sorted(MAN.glob("*_manifest.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        out[f.stem.replace("_manifest", "")] = m["n_probes"] / len({p["item_id"] for p in m["probes"]})
    return out


def strata_block() -> dict:
    rows = [json.loads(l) for l in (P5OUT / "s0_rows.jsonl").read_text(encoding="utf-8").splitlines()]
    a = [r for r in rows if r["stratum"] == "S1_CORRECT"]
    b = [r for r in rows if r["stratum"] == "S1_CONFIDENT_WRONG"]
    keys = ["mean_logprob", "min_logprob", "ans_logprob", "ans_entropy", "n_tokens"]
    au = {k: auroc([r[k] for r in a if r.get(k) is not None],
                   [r[k] for r in b if r.get(k) is not None]) for k in keys}
    sizes = collections.Counter(r["stratum"] for r in rows)
    return {"n_items": len(rows), "n_A": len(a), "n_B": len(b),
            "auroc": au, "auroc_min": min(au.values()), "auroc_max": max(au.values()),
            "sizes": dict(sizes)}


def bands_and_cost(dec) -> dict:
    p = probes_per_item()
    by = collections.defaultdict(collections.Counter)
    for (cell, _), (_, _, d) in dec.items():
        by[cell][d] += 1
    cells, tot, wsum = [], collections.Counter(), 0.0
    for cell in sorted(by):
        c = by[cell]
        n = sum(c.values())
        serve = c["SERVE"] / n
        pp = p[cell.rsplit("_", 1)[0]]
        g = 1 + pp + K_FINAL * (1 - serve)
        wsum += g * n
        tot.update(c)
        cells.append({"cell": cell, "n": n, "serve": serve, "stage2": 1 - serve,
                      "probes": pp, "gens": g})
    N = sum(tot.values())
    return {"cells": cells, "serve": tot["SERVE"] / N, "stage2": 1 - tot["SERVE"] / N,
            "gens_mean": wsum / N, "probes_min": min(p.values()), "probes_max": max(p.values()),
            "gens_min": min(c["gens"] for c in cells), "gens_max": max(c["gens"] for c in cells),
            "cell_min": min(cells, key=lambda c: c["gens"])["cell"],
            "cell_max": max(cells, key=lambda c: c["gens"])["cell"]}


def curves(dec, lp) -> dict:
    def curve(scored):
        rows = sorted(scored, key=lambda t: -t[0])
        n = len(rows)
        out = []
        for cov in GRID:
            k = max(1, round(cov * n))
            c = sum(y for _, y in rows[:k])
            out.append({"coverage": k / n, "acc": c / k, "yield": c / n})
        return out

    keys = [k for k in dec if k in lp]
    cs = [(dec[k][0], dec[k][1]) for k in keys]
    ml = [(lp[k], dec[k][1]) for k in keys]
    base = sum(y for _, y in cs) / len(cs)
    at = {}
    rows = sorted(ml, key=lambda t: -t[0])
    for name, cov in COVERAGES.items():
        k = round(cov * len(rows))
        c = sum(y for _, y in rows[:k])
        at[name] = {"coverage": k / len(rows), "acc": c / k, "yield": c / len(rows)}
    return {"n": len(keys), "base_acc": base, "c_safe": curve(cs),
            "mean_logprob": curve(ml), "logprob_at": at}


def crosslingual() -> dict:
    import random
    src = paths.FEATURES
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines()]
    by = {}
    for r in rows:
        v = r["features"].get("hold_violation_frac")
        if v is not None:
            by.setdefault((r["backbone"], r["benchmark"]), {})[r["item_id"].split("-")[-1]] = float(v)
    out, allp = [], []
    for bb in sorted({b for b, _ in by}):
        en, ko = by.get((bb, "gsm8k")), by.get((bb, "hrm8k-gsm8k-ko"))
        if not en or not ko:
            continue
        keys = sorted(set(en) & set(ko), key=int)
        pairs = [(en[k], ko[k]) for k in keys]
        allp += pairs
        out.append({"backbone": bb, "n": len(pairs),
                    "en": sum(x for x, _ in pairs) / len(pairs),
                    "ko": sum(y for _, y in pairs) / len(pairs), **_ci(pairs)})
    out.sort(key=lambda r: -r["diff"])
    return {"rows": out, "n": len(allp),
            "en": sum(x for x, _ in allp) / len(allp),
            "ko": sum(y for _, y in allp) / len(allp), **_ci(allp)}


def _ci(pairs, n_boot=2000, seed=0):
    import random
    rng = random.Random(seed)
    d = [b - a for a, b in pairs]
    n = len(d)
    vals = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return {"diff": sum(d) / n, "lo": vals[int(0.025 * n_boot)], "hi": vals[int(0.975 * n_boot)]}


def _trim(bundles, k):
    out = []
    for b in bundles:
        c = dict(b)
        if b["has_samples"]:
            c["paths"] = b["paths"][:1 + k]
            c["has_samples"] = len(c["paths"]) > 1
        c["k_samples"] = k
        out.append(c)
    return out


def per_item_outcomes(bundles, k=K_FINAL, two_band=True, eq=True) -> list:
    """Each item's final state. The confusion matrix and the stratified
    bootstrap are both built from this.

    Follows exactly the same branches as `EG.run_variant`, but returns items
    instead of totals.  One item's decision does not depend on which other
    items are in the sample -- the thresholds were fixed backbone-out, before
    any of this -- which is what makes resampling this list legitimate.
    """
    out = []
    for b in _trim(bundles, k):
        band = b["decision"]
        rec = {"cell": b["cell"], "label": int(b["label"])}
        if band == EG.gate.SERVE:
            rec.update(state="serve_stage1", served=1, correct=int(b["label"]))
        elif band == EG.gate.DECLINE and not two_band:
            rec.update(state="decline_band", served=0, correct=0)
        elif not b["has_samples"]:
            rec.update(state="decline_no_ballot", served=0, correct=0)
        else:
            _, share, ok = EG.vote_of(b, eq)
            if share >= EG.gate.VOTE_SERVE:
                rec.update(state="serve_stage2", served=1, correct=int(bool(ok)))
            else:
                rec.update(state="decline_vote", served=0, correct=0)
        out.append(rec)
    return out


def _rates(recs) -> dict:
    n = len(recs)
    served = sum(r["served"] for r in recs)
    correct = sum(r["correct"] for r in recs)
    van = sum(r["label"] for r in recs)
    return {"coverage": served / n if n else 0.0,
            "acc": correct / served if served else float("nan"),
            "yield": correct / n if n else 0.0,
            "vanilla": van / n if n else 0.0,
            "delta": (correct - van) / n if n else 0.0}


def stratified_ci(recs, n_boot=2000, seed=0) -> dict:
    """Resample items within each cell, because the design fixes 300 items
    per cell.

    Pooling items from different backbones and different benchmarks into one
    urn lets the cell sizes wobble from replicate to replicate, and the
    heterogeneity between cells then leaks into the interval width.  Holding
    each cell's n fixed makes the interval answer the question actually being
    asked: what if the same 15 cells had been drawn again?
    """
    import random
    rng = random.Random(seed)
    by_cell = collections.defaultdict(list)
    for r in recs:
        by_cell[r["cell"]].append(r)
    cells = sorted(by_cell)
    keys = ("coverage", "acc", "yield", "vanilla", "delta")
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        samp = []
        for c in cells:
            pool = by_cell[c]
            samp.extend(pool[rng.randrange(len(pool))] for _ in range(len(pool)))
        r = _rates(samp)
        for k in keys:
            draws[k].append(r[k])
    point = _rates(recs)
    out = {"n_boot": n_boot, "n_cells": len(cells), "point": point, "ci": {}}
    for k in keys:
        d = sorted(draws[k])
        out["ci"][k] = [d[int(0.025 * n_boot)], d[int(0.975 * n_boot)]]
    return out


def per_cell_ci(recs, n_boot=2000, seed=0) -> list:
    """Per-cell difference in correct yield with its interval: the forest
    plot's rows."""
    import random
    by_cell = collections.defaultdict(list)
    for r in recs:
        by_cell[r["cell"]].append(r)
    rows = []
    for i, c in enumerate(sorted(by_cell)):
        pool = by_cell[c]
        rng = random.Random(seed + i)
        d = sorted((sum(x["correct"] - x["label"] for x in
                        (pool[rng.randrange(len(pool))] for _ in range(len(pool))))
                    / len(pool)) for _ in range(n_boot))
        p = _rates(pool)
        rows.append({"cell": c, "n": len(pool), "vanilla": p["vanilla"],
                     "yield": p["yield"], "delta": p["delta"],
                     "lo": d[int(0.025 * n_boot)], "hi": d[int(0.975 * n_boot)]})
    return rows


def confusion(recs) -> dict:
    """Final state against correctness. A served row scores the answer that
    was emitted; a declined row scores the greedy answer that was withheld.
    What was filtered out, and what was thrown away with it."""
    order = ["serve_stage1", "serve_stage2", "decline_vote", "decline_no_ballot"]
    out = {}
    for st_ in order:
        rows = [r for r in recs if r["state"] == st_]
        judged = [r["correct"] if r["served"] else r["label"] for r in rows]
        out[st_] = {"n": len(rows), "correct": sum(judged),
                    "wrong": len(rows) - sum(judged)}
    out["total"] = {"n": len(recs),
                    "correct": sum(v["correct"] for k, v in out.items() if k in order),
                    "wrong": sum(v["wrong"] for k, v in out.items() if k in order)}
    #: Split what the second stage did into its two directions: wrong greedy
    #: answers it rescued, and correct ones it lost. Whether the gain comes
    #: from declining or from rescuing can only be answered by reading these
    #: two counts alongside the declined row.
    s2 = [r for r in recs if r["state"] == "serve_stage2"]
    out["stage2_rescued"] = sum(1 for r in s2 if r["label"] == 0 and r["correct"] == 1)
    out["stage2_lost"] = sum(1 for r in s2 if r["label"] == 1 and r["correct"] == 0)
    out["stage2_vanilla_correct"] = sum(r["label"] for r in s2)
    return out


def gate_results() -> dict:
    """Call the registered readout as it stands and recompute the k sweep,
    the variants and the per-backbone rows."""
    cells, missing = EG.load_cells("", 4)
    allb = [b for c in cells.values() for b in c.values()]
    van = sum(b["label"] for b in allb) / len(allb)

    def trim(bundles, k):
        return _trim(bundles, k)

    def row(bundles, k=K_FINAL, two_band=True, eq=True):
        r = EG.run_variant(trim(bundles, k), two_band=two_band, equivalence=eq)
        v = sum(b["label"] for b in bundles) / len(bundles)
        return {"n": len(bundles), "vanilla": v, "coverage": r["coverage"],
                "acc": r["selective_acc"], "yield": r["yield"],
                "delta": r["yield"] - v, "gens": r["gens_per_item"]}

    ksweep = [{"k": k, "votes": k + 1, **row(allb, k)} for k in (0, 1, 2, 3, 4)]
    variants = []
    for name, (tb, eq) in EG.VARIANTS.items():
        variants.append({"name": name, **row(allb, 4, tb, eq)})

    by_bb = collections.defaultdict(list)
    by_cell = collections.defaultdict(list)
    for b in allb:
        by_bb[b["cell"].rsplit("_", 1)[-1]].append(b)
        by_cell[b["cell"]].append(b)
    #: The registered baseline, computable only where samples were bought:
    #: the immediate-serve band bought none.
    sc = EG.baseline_sc_always(allb, True)
    #: Truncation: items whose greedy generation hit the token limit.
    trunc = sum(1 for b in allb if b["paths"][0].get("truncated"))
    recs = per_item_outcomes(allb, K_FINAL)
    return {"missing": missing, "n": len(allb), "vanilla": van,
            "uncertainty": stratified_ci(recs),
            "cells_ci": per_cell_ci(recs),
            "confusion": confusion(recs),
            "sc_always": sc, "truncated": trunc, "truncated_frac": trunc / len(allb),
            "overall": row(allb),
            "ksweep": ksweep, "variants": variants,
            "backbones": [{"backbone": k, **row(v)} for k, v in sorted(by_bb.items())],
            "cells": [{"cell": k, **row(v)} for k, v in sorted(by_cell.items())],
            "stage1_only": row(allb, 0)}


def xai() -> dict:
    out = {}
    for f in sorted(paths.LAYERS.glob("p5-s1-layers-*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        bb = d["backbone"]
        sw = d.get("sweep") or []
        out[bb] = {"median_layer": d.get("median_layer"), "median_auroc": d.get("median_auroc"),
                   "best_layer": d.get("best_layer"), "best_auroc": d.get("best_auroc"),
                   "n_layers": len(d.get("layers") or []),
                   "gate_auroc": d.get("gate_auroc"), "judge": d.get("judge"),
                   "sweep_mix_gt_probe": sum(1 for x in sw if (x["lo"] or -1) > 0),
                   "sweep_n": len(sw)}
        q = d.get("quadrant")
        if q:
            pts, med = q["points"], q["logprob_median"]
            cell = [p for p in pts if p["p_correct"] < 0.5 and p["mean_logprob"] >= med]
            out[bb]["quadrant"] = {
                "n_total": len(pts), "n_cell": len(cell),
                "n_wrong": sum(1 for p in cell if not p["correct"]),
                "n_declined": sum(1 for p in cell if p["declined"]),
                "base_err": sum(1 for p in pts if not p["correct"]) / len(pts)}
        out[bb]["source"] = "layers-json"
    return out


def coverage_table() -> dict:
    """Rule coverage, counted directly from the committed manifests."""
    out = {}
    for f in sorted(MAN.glob("*_manifest.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        bench = f.stem.replace("_manifest", "")
        by = collections.defaultdict(list)
        for p in m["probes"]:
            by[p["item_id"]].append(p)
        items = len(by)
        chg = {i for i, ps in by.items() if any(p["expectation"] == "MUST_CHANGE" for p in ps)}
        high = {i for i, ps in by.items()
                if any(p["expectation"] == "MUST_CHANGE" and p.get("confidence") == "high" for p in ps)}
        out[bench] = {"items": items, "high": len(high), "any_change": len(chg),
                      "no_change": items - len(chg)}
    return out


def parsing() -> dict:
    """How often no answer could be extracted from a probe response.

    Both voting and grading require an answer to be read, so the first
    question anyone asks is how many were unreadable.  The fingerprint already
    counted this per item; here it is only summed per benchmark, and nothing
    is judged anew.
    """
    src = paths.FEATURES
    tot = collections.defaultdict(lambda: {"probes": 0, "unanswered": 0, "items": 0,
                                           "items_affected": 0})
    allb = {"probes": 0, "unanswered": 0, "items": 0, "items_affected": 0}
    for line in src.open(encoding="utf-8"):
        r = json.loads(line)
        f = r["features"]
        for d in (tot[r["benchmark"]], allb):
            d["probes"] += f["n_probes"]
            d["unanswered"] += f["n_unanswered"]
            d["items"] += 1
            d["items_affected"] += 1 if f["n_unanswered"] else 0
    for d in list(tot.values()) + [allb]:
        d["frac"] = d["unanswered"] / d["probes"] if d["probes"] else 0.0
        d["frac_items"] = d["items_affected"] / d["items"] if d["items"] else 0.0
    return {"all": allb, "by_benchmark": dict(tot)}


def main() -> None:
    if not DEC.exists():
        raise SystemExit(
            f"{DEC.relative_to(ROOT)} is missing. Every number in the paper "
            f"rests on it.\n"
            "  It is published with the results; regenerate it with:\n"
            "  python3 mome/select_retry.py")
    dec = {}
    for line in DEC.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        dec[(d["cell"], d["item_id"])] = (float(d["c_safe"]), int(d["label"]), d["decision"])

    lp = {}
    for cell in sorted({c for c, _ in dec}):
        vdir = EG.vanilla_dir(cell)
        recs = greedy_by_item([r for r in oh.read_jsonl(vdir / "records.jsonl")
                               if r.get("error") is None], vdir.name)
        for item, rec in recs.items():
            if (cell, item) in dec:
                lp[(cell, item)] = mean_logprob(rec) or -99.0

    th = json.loads(paths.THRESHOLDS.read_text(encoding="utf-8"))
    data = {
        "k_final": K_FINAL,
        "strata": strata_block(),
        "bands": bands_and_cost(dec),
        "curves": curves({k: (v[0], v[1]) for k, v in dec.items()}, lp),
        "crosslingual": crosslingual(),
        "mr_coverage": coverage_table(),
        "parsing": parsing(),
        "thresholds": th,
        "gate": gate_results(),
        "xai": xai(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"written: {OUT}  ({OUT.stat().st_size // 1024} KB)")

    #: Check the file against the per-item records the moment it is written.
    #: This is the only place that stops a mismatch from reaching the paper.
    print()
    import verify_results
    raise SystemExit(verify_results.main())


if __name__ == "__main__":
    from quiet_mp import drop_spawn_bootstrap_noise
    with drop_spawn_bootstrap_noise():
        main()
