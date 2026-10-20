#!/usr/bin/env python3
"""Reviewer 1, minor comment 1: serve-band precision on the held-out backbone (Table 4).

The registered thresholds of each backbone were selected on the other four backbones with the
constraint serve-band precision >= 0.95. This script reports, per held-out backbone, the
ungated accuracy, the stage-1 serve share and precision at the registered tau_high, and the
largest serve share at which any threshold on that backbone would reach 0.95 (an oracle
read-out with the backbone's own labels, for diagnosis only). Arithmetic on results/revision/items.jsonl.

    python3 scripts/revision/serve_precision_heldout.py   # writes results/revision/serve_precision_heldout.json
"""
from __future__ import annotations

import json
from collections import defaultdict

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

TARGET = 0.95
OUT = C.OUT_DIR / "serve_precision_heldout.json"


def main() -> None:
    items = [json.loads(l) for l in C.ITEMS.read_text(encoding="utf-8").splitlines()]
    tau = C.registered_tau_high()
    by = defaultdict(list)
    for r in items:
        by[r["backbone"]].append(r)
    out = {"target": TARGET, "per_backbone": {}}
    for bb, rs in sorted(by.items()):
        served = [r for r in rs if r["c_safe"] >= tau[bb]]
        n_ok = sum(r["label"] for r in served)
        ranked = sorted(rs, key=lambda r: -r["c_safe"])
        best, c = 0, 0
        for i, r in enumerate(ranked, 1):
            c += r["label"]
            if c / i >= TARGET:
                best = i
        out["per_backbone"][bb] = {
            "n": len(rs), "ungated_acc": sum(r["label"] for r in rs) / len(rs),
            "tau_high": tau[bb], "served": len(served), "serve_share": len(served) / len(rs),
            "serve_precision": n_ok / len(served) if served else None, "served_wrong": len(served) - n_ok,
            "max_share_at_target": best / len(rs)}
    pb = out["per_backbone"].values()
    s = sum(v["served"] for v in pb)
    w = sum(v["served_wrong"] for v in pb)
    out["pooled"] = {"served": s, "served_wrong": w, "serve_precision": (s - w) / s,
                     "n_below_target": sum(v["serve_precision"] < TARGET for v in pb)}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for bb, v in out["per_backbone"].items():
        print(f"{bb:16s} acc {v['ungated_acc']:.3f} serve {v['serve_share']:.3f} prec {v['serve_precision']:.3f} "
              f"wrong {v['served_wrong']} max@0.95 {v['max_share_at_target']:.3f}")
    print("pooled", out["pooled"])


if __name__ == "__main__":
    main()
