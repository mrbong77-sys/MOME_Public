#!/usr/bin/env python3
"""Reviewer 2, comment 2: total wall-clock time per item, generations included.

For every evaluated item, the time of each generation the gate buys was read from the records
of the evaluation runs: the greedy answer, the probe generations and, for items sent to stage 2,
the first two seeded samples (k = 2). The time of one generation is the runtime's own
`total_duration - load_duration` (model loading excluded; it happens once per session, not per
item). results/revision/latency_items.jsonl keeps these seconds per item; this script sums them.

Two schedules are reported:
* sequential -- one request at a time, the default of a single-GPU local runtime;
* parallel   -- the greedy answer and the probes are independent requests, and so are the
  stage-2 samples; with enough parallel slots stage 1 takes as long as its slowest request
  and stage 2 likewise. This is a lower bound for the same hardware.

    python3 scripts/revision/latency_per_item.py   # writes results/revision/latency_per_item.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

SRC = C.REVISION / "latency_items.jsonl"
OUT = C.OUT_DIR / "latency_per_item.json"
OVERHEAD = C.paths.OVERHEAD


def pct(v, q):
    return float(np.percentile(np.asarray(v, float), q))


def main() -> None:
    per_item = []
    for line in SRC.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        g, pr, sm = r["greedy"], r["probes"], r["samples"]
        per_item.append({"backbone": r["backbone"], "bench": r["bench"], "stage2": r["stage2"],
                         "seq": g + sum(pr) + sum(sm),
                         "par": max([g] + pr) + (max(sm) if sm else 0.0),
                         "greedy": g, "n_gen": 1 + len(pr) + len(sm)})

    res = {"n_items": len(per_item), "unit": "seconds", "overhead_source": str(OVERHEAD.name),
           "pooled": {}, "per_backbone": {}, "per_benchmark": {}}

    def block(rows):
        seq = [r["seq"] for r in rows]; par = [r["par"] for r in rows]
        gre = [r["greedy"] for r in rows]
        return {"n": len(rows), "gens_mean": float(np.mean([r["n_gen"] for r in rows])),
                "greedy_only_median": pct(gre, 50), "greedy_only_mean": float(np.mean(gre)),
                "seq_median": pct(seq, 50), "seq_mean": float(np.mean(seq)), "seq_p95": pct(seq, 95),
                "par_median": pct(par, 50), "par_mean": float(np.mean(par)), "par_p95": pct(par, 95),
                "seq_over_greedy_mean": float(np.mean(seq) / np.mean(gre))}

    res["pooled"] = block(per_item)
    for bb in C.BACKBONES:
        res["per_backbone"][bb] = block([r for r in per_item if r["backbone"] == bb])
    for b in sorted({r["bench"] for r in per_item}):
        res["per_benchmark"][b] = block([r for r in per_item if r["bench"] == b])
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for k, v in [("pooled", res["pooled"])] + list(res["per_backbone"].items()) + list(res["per_benchmark"].items()):
        print(f"{k:18} n={v['n']:5} gens={v['gens_mean']:.2f} greedy med {v['greedy_only_median']:.2f}s "
              f"seq med {v['seq_median']:.1f}s mean {v['seq_mean']:.1f}s p95 {v['seq_p95']:.1f}s | "
              f"par med {v['par_median']:.1f}s mean {v['par_mean']:.1f}s | x{v['seq_over_greedy_mean']:.2f}")
    print("written", OUT)


if __name__ == "__main__":
    main()
