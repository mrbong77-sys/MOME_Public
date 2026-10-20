#!/usr/bin/env python3
"""Reviewer 1, comment 2: AUROC of the five runtime output statistics on all 4,497 stage-1
greedy answers (the set on which the MOME score reaches 0.863), with 95% intervals.

Reads only committed files: results/stage1_decisions.jsonl (MOME score c_safe and
the correctness label of the greedy answer) and results/s0_rows.jsonl (the five
output statistics of the same greedy answers). No generation.

Interval: 2,000 bootstrap replicates, items resampled within each cell (cell sizes held
fixed), as for every other interval in the paper. The difference to the MOME score is
item-paired (same resample for both scores).

    python3 scripts/revision/output_stats_full.py      # writes results/revision/output_stats_full.json
"""
from __future__ import annotations

import collections
import json
import random
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

DEC = C.paths.DECISIONS
ROWS = C.paths.STRATA_ROWS
OUT = C.OUT_DIR / "output_stats_full.json"
STATS = ["mean_logprob", "min_logprob", "ans_logprob", "ans_entropy", "n_tokens"]
REVERSED = {"ans_entropy", "n_tokens"}      # larger = more uncertain; ranked in reverse
N_BOOT, SEED = 2000, 0


def auroc(scores, labels):
    """P(score of a correct answer > score of a wrong one), ties count one half."""
    pairs = sorted(zip(scores, labels))
    n1 = sum(labels)
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg = (i + 1 + j) / 2
        rank_sum += avg * sum(y for _, y in pairs[i:j])
        i = j
    return (rank_sum - n1 * (n1 + 1) / 2) / (n1 * n0)


def main() -> None:
    dec = {}
    for line in DEC.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        dec[(d["cell"], d["item_id"])] = (float(d["c_safe"]), int(d["label"]), d["decision"])
    rows = {}
    for line in ROWS.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        rows[(r["cell"], r["item"])] = r
    keys = sorted(k for k in dec if k in rows)
    missing = {s: sum(rows[k].get(s) is None for k in keys) for s in STATS}

    def score(k, s):
        if s == "c_safe":
            return dec[k][0]
        v = rows[k].get(s)
        if v is None:
            return float("-inf")          # no value: ranked as least confident
        return -v if s in REVERSED else v

    names = ["c_safe"] + STATS
    y = [dec[k][1] for k in keys]
    point = {s: auroc([score(k, s) for k in keys], y) for s in names}

    by_cell = collections.defaultdict(list)
    for i, k in enumerate(keys):
        by_cell[k[0]].append(i)
    rng = random.Random(SEED)
    boots = {s: [] for s in names}
    diffs = {s: [] for s in STATS}
    sc = {s: [score(k, s) for k in keys] for s in names}
    for _ in range(N_BOOT):
        idx = [rng.choice(ix) for ix in by_cell.values() for _ in ix]
        yb = [y[i] for i in idx]
        a = {s: auroc([sc[s][i] for i in idx], yb) for s in names}
        for s in names:
            boots[s].append(a[s])
        for s in STATS:
            diffs[s].append(a["c_safe"] - a[s])

    def ci(v):
        v = sorted(v)
        return [v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]]

    out = {"n": len(keys), "n_correct": sum(y), "n_wrong": len(y) - sum(y),
           "missing_values": missing, "reversed": sorted(REVERSED),
           "auroc": {s: {"point": point[s], "ci95": ci(boots[s])} for s in names},
           "c_safe_minus_stat": {s: {"point": point["c_safe"] - point[s], "ci95": ci(diffs[s])}
                                 for s in STATS},
           "per_backbone": {}}
    bbs = sorted({k[0].split("_", 1)[1] for k in keys})
    for bb in bbs:
        ks = [i for i, k in enumerate(keys) if k[0].split("_", 1)[1] == bb]
        out["per_backbone"][bb] = {"n": len(ks), **{s: auroc([sc[s][i] for i in ks], [y[i] for i in ks])
                                                     for s in names}}
    #: The stratum of the original Figure 1: answers served at stage 1 (decision SERVE).
    #: It was selected by c_safe itself, so every AUROC on it is range-restricted.
    s1 = [i for i, k in enumerate(keys) if dec[k][2] == "SERVE"]
    out["stage1_served"] = {"n": len(s1), "n_wrong": sum(1 - y[i] for i in s1),
                            **{s: auroc([sc[s][i] for i in s1], [y[i] for i in s1]) for s in names}}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for s in names:
        a = out["auroc"][s]
        print(f"{s:14s} {a['point']:.3f} [{a['ci95'][0]:.3f}, {a['ci95'][1]:.3f}]")
    print("n", out["n"], "correct", out["n_correct"], "missing", missing)


if __name__ == "__main__":
    main()
