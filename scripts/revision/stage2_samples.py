#!/usr/bin/env python3
"""Accuracy of the stage-2 samples against the greedy answer, per backbone and temperature.

    python3 scripts/revision/stage2_samples.py     # writes results/revision/stage2_samples.json

For every backbone (the five of the main evaluation and the four of further families), on the
items that its gate sends to stage 2, this reports the accuracy of the greedy answer and of the
first two stage-2 samples (seeds 0 and 1) at temperature 0.7 and 0.3. Correctness comes from
the grades only, so the numbers do not depend on the answer-equivalence check of the vote.

Input: results/revision/stage2_samples_items.jsonl, one row per stage-2 item with the
correctness of each sample. The main evaluation sends an item to stage 2 by its registered
decision; a further-family backbone by the frozen predictor and tau_high
(model/frozen_predictor_newfamily.json).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

SRC = C.REVISION / "stage2_samples_items.jsonl"
OUT = C.OUT_DIR / "stage2_samples.json"


def main() -> int:
    by: dict = {}
    for line in SRC.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        by.setdefault(r["backbone"], []).append(r)
    res = {}
    for bb, items in sorted(by.items()):
        row = {"family": items[0]["family"], "n_stage2": len(items),
               "greedy_acc": sum(r["label"] for r in items) / len(items)}
        for t in ("T0.7", "T0.3"):
            s = [ok for r in items for ok in r[t].values()]
            row[f"sample_acc_{t}"] = sum(s) / len(s) if s else None
            row[f"n_samples_{t}"] = len(s)
        res[bb] = row
        print(bb, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    OUT.write_text(json.dumps({"what": "greedy vs stage-2 sample accuracy (seeds 0-1) on the stage-2 items",
                               "backbones": res}, indent=1) + "\n", encoding="utf-8")
    print("written", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
