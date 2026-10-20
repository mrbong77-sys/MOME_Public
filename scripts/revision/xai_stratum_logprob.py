#!/usr/bin/env python3
"""Reviewer 1, comment 2 (Table 15): mean token log-probability on the same items and with
the same outcome label as the linear-probe comparison (no generation, no hidden states).

The committed read-out of the hidden-state study (results/layers/p5-s1-layers-<bb>.json) keeps,
for every item of its stratum (897 on gemma4-e4b, 899 on qwen3.5-9b), the mean token
log-probability of the served greedy answer, the probe's out-of-fold probability at the
median layer and the outcome label (correctness of the answer the gate would deliver).
This script computes the AUROC of mean token log-probability on exactly those items, with
a 95% item-bootstrap interval, and the item-paired difference to the linear probe.

    python3 scripts/revision/xai_stratum_logprob.py   # writes results/revision/xai_stratum_logprob.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

SRC = C.paths.LAYERS
OUT = C.OUT_DIR / "xai_stratum_logprob.json"


def main() -> None:
    res = {}
    rng = np.random.default_rng(C.SEED)
    for bb in ("gemma4-e4b", "qwen3.5-9b"):
        d = json.loads((SRC / f"p5-s1-layers-{bb}.json").read_text(encoding="utf-8"))
        pts = d["quadrant"]["points"]
        lp = np.array([p["mean_logprob"] for p in pts], float)
        pr = np.array([p["p_correct"] for p in pts], float)
        y = np.array([int(p["correct"]) for p in pts])
        a_lp, a_pr = C.auroc(lp, y), C.auroc(pr, y)
        bl, bd = [], []
        for _ in range(C.N_BOOT):
            i = rng.integers(0, len(y), len(y))
            bl.append(C.auroc(lp[i], y[i])); bd.append(C.auroc(pr[i], y[i]) - C.auroc(lp[i], y[i]))
        res[bb] = {"n": len(y), "n_correct": int(y.sum()),
                   "mean_logprob_auroc": a_lp, "mean_logprob_auroc_ci": C.ci(bl),
                   "probe_auroc_recomputed": a_pr, "probe_auroc_recorded": d["judge"]["probe"],
                   "probe_minus_mean_logprob": a_pr - a_lp, "probe_minus_mean_logprob_ci": C.ci(bd),
                   "mome_auroc_recorded": d["gate_auroc"]}
        print(bb, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[bb].items()}))
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
