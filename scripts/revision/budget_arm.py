#!/usr/bin/env python3
"""Reviewer 1, comment 1 (context): the registered independent check of k = 2.

The registration lowered k from 4 to 2 after the k sweep and fixed, before the data were seen,
an independent check on a separate generation run: the MATH-500 cells regenerated with a
4096-token limit. Criterion: on math500 x gemma4-e2b and math500 x qwen3.5-9b, the correct yield
of the k = 2 two-band gate is at least the ungated accuracy of the same run. This script reads
that run (same frozen predictor, registered thresholds, k = 2, two bands, equivalence voting)
next to the 2048-token cells of the main evaluation.

Input: results/revision/budget_b4096_items.jsonl, one row per item of the 4096-token run with
its stage-1 decision and its stage-2 ballot (vote share and whether the winning answer is
right, counted with the registered answer equivalence); results/revision/items.jsonl for the
2048-token cells.

    python3 scripts/revision/budget_arm.py   # writes results/revision/budget_arm.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import gate  # noqa: E402

SRC = C.REVISION / "budget_b4096_items.jsonl"
OUT = C.OUT_DIR / "budget_arm.json"
CHECK_CELLS = ("math500_gemma4-e2b", "math500_qwen3.5-9b")


def run_variant(rows: list[dict]) -> dict:
    """evaluate_gate.run_variant with two bands and equivalence voting, on per-item ballots."""
    n = answered = correct = 0
    gens = 0.0
    for r in rows:
        n += 1
        gens += r["probe_gens"]
        if r["decision"] == gate.SERVE:
            answered += 1; correct += r["label"]; gens += 1
            continue
        if not r["has_samples"]:
            gens += 1
            continue
        gens += r["n_paths"]
        if r["share"] >= gate.VOTE_SERVE:
            answered += 1
            correct += bool(r["ok"])
    return {"coverage": answered / n if n else 0.0,
            "selective_acc": correct / answered if answered else float("nan"),
            "yield": correct / n if n else 0.0,
            "gens_per_item": gens / n if n else 0.0}


def main() -> None:
    rows: dict = {}
    for line in SRC.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        rows.setdefault(r["cell"], []).append(r)
    items = C.load_items()
    res = {"cells": {}, "criterion_cells": list(CHECK_CELLS)}
    allb = []
    for base, b in sorted(rows.items()):
        allb += b
        r = run_variant(b)
        van = sum(x["label"] for x in b) / len(b)
        its = [it for (c, _), it in items.items() if c == base]
        g = C.summarize(C.run_gate(its, [it["c_safe"] for it in its], C.registered_tau_high()))
        res["cells"][base] = {
            "b4096": {"n": len(b), "ungated_acc": van, "yield": r["yield"],
                      "delta_pp": (r["yield"] - van) * 100, "coverage": r["coverage"],
                      "selective_acc": r["selective_acc"], "gens_per_item": r["gens_per_item"],
                      "truncated_frac": sum(x["truncated"] for x in b) / len(b)},
            "b2048": {"n": g["n"], "ungated_acc": g["ungated_acc"], "yield": g["yield"],
                      "delta_pp": g["delta_yield_pp"]}}
        print(base, json.dumps(res["cells"][base]), flush=True)
    r = run_variant(allb)
    van = sum(x["label"] for x in allb) / len(allb)
    res["pooled_b4096"] = {"n": len(allb), "ungated_acc": van, "yield": r["yield"],
                           "delta_pp": (r["yield"] - van) * 100, "coverage": r["coverage"],
                           "selective_acc": r["selective_acc"]}
    res["criterion_met"] = all(res["cells"][c]["b4096"]["delta_pp"] >= 0 for c in CHECK_CELLS)
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res["pooled_b4096"]), "criterion met:", res["criterion_met"])


if __name__ == "__main__":
    main()
