#!/usr/bin/env python3
"""Recompute every published number and check it against results/ssot.json.

    python3 scripts/verify_results.py

This needs no GPU, no model, no network and none of the generation records.
It works from the four small files published in `results/` and `model/`, and
it recomputes rather than re-reads: a number only passes if it can be derived
again from the per-item records.

What it checks, in order:

1. **The predictor reproduces its own scores.** `c_safe` is recomputed for all
   items from the 26 fingerprint features and the frozen backbone-out
   coefficients, and compared with the score recorded at decision time.
2. **The thresholds reproduce the bands.** Each recomputed score is put
   through the registered thresholds and compared with the recorded
   first-stage decision.
3. **The headline rates follow from the items.** Coverage, selective accuracy,
   correct yield and the difference against the ungated model are recomputed
   by counting per-item outcomes.
4. **The confusion matrix adds up**, and the difference in correct yield
   decomposes exactly into rescued minus lost minus withheld-but-correct.
5. **The AUROC panel** is recomputed from the stratum records.
6. **The stratified bootstrap** is rerun with the published seed, and its
   intervals are checked to reproduce and to cover their point estimates.
7. **The file is internally consistent** -- per-cell rows sum to the total,
   served counts agree with coverage, and so on.
8. **The README's results table** matches the file it summarizes. A README is
   prose and its numbers were typed, which is exactly where a number drifts.

Every check prints the recomputed value beside the published one.  A
mismatch is a failure, and the exit code is non-zero.
"""

from __future__ import annotations

import collections
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mome"))
import paths                                        # noqa: E402
import gate                                         # noqa: E402

TOL = 5e-4


class Audit:
    def __init__(self) -> None:
        self.checked = self.failed = self.skipped = 0

    def check(self, name: str, mine, published, tol: float = TOL) -> None:
        if mine is None or published is None:
            self.skipped += 1
            print(f"--  {name:<52} not available")
            return
        self.checked += 1
        if isinstance(mine, float) or isinstance(published, float):
            ok = abs(float(mine) - float(published)) <= tol
            shown = f"{float(mine):.6g} vs {float(published):.6g}"
        else:
            ok = mine == published
            shown = f"{mine} vs {published}"
        if ok:
            print(f"ok  {name:<52} {shown}")
        else:
            self.failed += 1
            print(f"XX  {name:<52} {shown}   MISMATCH")

    def report(self) -> int:
        print(f"\n{self.checked} checks, {self.failed} mismatches, "
              f"{self.skipped} skipped")
        return 1 if self.failed else 0


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def auroc(pos: list[float], neg: list[float]) -> float | None:
    """P(a positive scores above a negative), ties counted as half."""
    if not pos or not neg:
        return None
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks: dict[float, float] = {}
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    rsum = sum(ranks[k] for k, (_, lab) in enumerate(allv) if lab == 1)
    n1, n0 = len(pos), len(neg)
    return (rsum - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def backbone_of(cell: str) -> str:
    return cell.split("_", 1)[1]


# ---- 1 and 2: the predictor and the thresholds --------------------------------

def check_predictor(a: Audit) -> None:
    print("\n## 1-2. The frozen predictor and the registered thresholds\n")
    feats = {(r["cell"], r["item_id"]): r["features"] for r in jsonl(paths.FEATURES)}
    models = json.loads(paths.MODEL_BACKBONE_OUT.read_text(encoding="utf-8"))
    taus = json.loads(paths.THRESHOLDS.read_text(encoding="utf-8"))["per_backbone"]
    decisions = jsonl(paths.DECISIONS)

    score_max = 0.0
    band_wrong = 0
    for d in decisions:
        f = feats.get((d["cell"], d["item_id"]))
        if f is None:
            continue
        bb = backbone_of(d["cell"])
        again = gate.stage1(f, models[bb], taus[bb]["tau_high"], taus[bb]["tau_low"])
        score_max = max(score_max, abs(again["c_safe"] - d["c_safe"]))
        band_wrong += again["decision"] != d["decision"]

    a.check("items scored", len(decisions), len(feats))
    a.check("largest c_safe difference on recomputation", score_max, 0.0, tol=1e-9)
    a.check("first-stage decisions that differ", band_wrong, 0)


# ---- 3 and 4: the rates, from the items ---------------------------------------

def check_rates(a: Audit, ss: dict) -> dict:
    print("\n## 3-4. Headline rates, recomputed by counting items\n")
    rows = jsonl(paths.PER_ITEM)
    g, o = ss["gate"], ss["gate"]["overall"]
    n = len(rows)
    served = sum(r["served"] for r in rows)
    correct = sum(r["correct"] for r in rows)
    ungated = sum(r["label"] for r in rows)

    a.check("items", n, g["n"])
    a.check("ungated accuracy", ungated / n, g["vanilla"])
    a.check("coverage", served / n, o["coverage"])
    a.check("selective accuracy", correct / served, o["acc"])
    a.check("correct yield", correct / n, o["yield"])
    a.check("correct yield - ungated", correct / n - ungated / n, o["delta"])
    a.check("truncated items", sum(r["truncated"] for r in rows), g["truncated"])

    print()
    cm = g["confusion"]
    by_state = collections.Counter(r["state"] for r in rows)
    for state in ("serve_stage1", "serve_stage2", "decline_vote", "decline_no_ballot"):
        a.check(f"confusion row {state}", by_state.get(state, 0), cm[state]["n"])
        got = sum(r["correct"] if r["served"] else r["label"]
                  for r in rows if r["state"] == state)
        a.check(f"  of those, correct", got, cm[state]["correct"])
    a.check("confusion rows sum to n", sum(cm[s]["n"] for s in
            ("serve_stage1", "serve_stage2", "decline_vote", "decline_no_ballot")), n)

    print()
    rescued = sum(1 for r in rows
                  if r["state"] == "serve_stage2" and r["label"] == 0 and r["correct"] == 1)
    lost = sum(1 for r in rows
               if r["state"] == "serve_stage2" and r["label"] == 1 and r["correct"] == 0)
    withheld = sum(r["label"] for r in rows if not r["served"])
    a.check("second stage rescued", rescued, cm.get("stage2_rescued"))
    a.check("second stage lost", lost, cm.get("stage2_lost"))
    a.check("declined but would have been correct", withheld,
            cm["decline_vote"]["correct"] + cm["decline_no_ballot"]["correct"])
    a.check("rescued - lost - withheld == correct - ungated",
            rescued - lost - withheld, correct - ungated)
    return {"rows": rows}


# ---- 5: the AUROC panel -------------------------------------------------------

def check_strata(a: Audit, ss: dict) -> None:
    print("\n## 5. Output statistics between the two served strata\n")
    if not paths.STRATA_ROWS.exists():
        a.check("stratum records", None, None)
        return
    rows = jsonl(paths.STRATA_ROWS)
    st = ss["strata"]
    A = [r for r in rows if r["stratum"] == "S1_CORRECT"]
    B = [r for r in rows if r["stratum"] == "S1_CONFIDENT_WRONG"]
    a.check("stratum records", len(rows), st["n_items"])
    a.check("served and correct", len(A), st["n_A"])
    a.check("served and wrong", len(B), st["n_B"])
    for key, published in st["auroc"].items():
        mine = auroc([r[key] for r in A if r.get(key) is not None],
                     [r[key] for r in B if r.get(key) is not None])
        a.check(f"AUROC {key}", mine, published)


# ---- 6: the stratified bootstrap ----------------------------------------------

def _rates(rows: list[dict]) -> tuple[float, float, float, float]:
    n = len(rows)
    served = sum(r["served"] for r in rows)
    correct = sum(r["correct"] for r in rows)
    ungated = sum(r["label"] for r in rows)
    return (served / n, correct / served if served else float("nan"),
            correct / n, correct / n - ungated / n)


def check_bootstrap(a: Audit, ss: dict, rows: list[dict],
                    n_boot: int = 2000, seed: int = 0) -> None:
    print(f"\n## 6. Stratified bootstrap, {n_boot} replicates, seed {seed}\n")
    unc = ss["gate"].get("uncertainty")
    if not unc:
        a.check("uncertainty block", None, None)
        return
    by_cell: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_cell[r["cell"]].append(r)

    rng = random.Random(seed)
    draws: dict[str, list[float]] = {k: [] for k in ("coverage", "acc", "yield", "delta")}
    for _ in range(n_boot):
        sample: list[dict] = []
        for cell_rows in by_cell.values():
            m = len(cell_rows)
            sample.extend(cell_rows[rng.randrange(m)] for _ in range(m))
        cov, acc, yld, delta = _rates(sample)
        draws["coverage"].append(cov); draws["acc"].append(acc)
        draws["yield"].append(yld); draws["delta"].append(delta)

    for key in ("coverage", "acc", "yield", "delta"):
        vals = sorted(draws[key])
        lo, hi = vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot)]
        plo, phi = unc["ci"][key]
        a.check(f"{key} interval, lower", lo, plo, tol=3e-3)
        a.check(f"{key} interval, upper", hi, phi, tol=3e-3)
        point = unc["point"][key]
        inside = plo - 1e-12 <= point <= phi + 1e-12
        a.check(f"{key} interval covers its point estimate", inside, True)


# ---- 7: the file against itself -----------------------------------------------

def check_internal(a: Audit, ss: dict) -> None:
    print("\n## 7. The published file against itself\n")
    g = ss["gate"]
    cells = g["cells"]
    a.check("per-cell rows", len(cells), len(g["cells_ci"]))
    a.check("per-cell items sum to the total", sum(c["n"] for c in cells), g["n"])
    a.check("per-backbone items sum to the total",
            sum(b["n"] for b in g["backbones"]), g["n"])
    k2 = next((r for r in g["ksweep"] if r["k"] == 2), None)
    if k2:
        for key in ("coverage", "acc", "yield"):
            a.check(f"k=2 sweep row matches the headline {key}",
                    k2[key], g["overall"][key])
    cov = g["confusion"]
    served = cov["serve_stage1"]["n"] + cov["serve_stage2"]["n"]
    a.check("confusion served / n == coverage", served / g["n"], g["overall"]["coverage"])
    right = cov["serve_stage1"]["correct"] + cov["serve_stage2"]["correct"]
    a.check("confusion correct / n == correct yield", right / g["n"], g["overall"]["yield"])
    down = sum(1 for c in g["cells_ci"] if c["hi"] < 0)
    up = sum(1 for c in g["cells_ci"] if c["lo"] > 0)
    a.check("cells whose interval is below zero", down, g.get("cells_down", down))
    a.check("cells whose interval is above zero", up, g.get("cells_up", up))


# ---- 8: the README against the results ----------------------------------------

def check_readme(a: Audit, ss: dict) -> None:
    """The README quotes a results table. Check it, rather than trusting it.

    Every number in the paper is generated from ssot.json, but a README is
    prose and its table was typed. Typed numbers drift -- two of these were
    wrong on first publication -- so the same discipline applies here: the
    figures are read back out of the file and compared.
    """
    print("\n## 8. The README's results table against ssot.json\n")
    readme = paths.ROOT / "README.md"
    if not readme.exists():
        a.check("README.md", None, None)
        return
    rows = {}
    for line in readme.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or "---" in line:
            continue
        cells = [c.strip().strip("*").strip() for c in line.strip("|").split("|")]
        if len(cells) != 5 or not cells[1].endswith("%"):
            continue
        rows[cells[0].lower()] = cells[1:]

    def pct(x: float) -> str:
        return f"{x * 100:.1f}%"

    g, o = ss["gate"], ss["gate"]["overall"]
    lp = ss["curves"]["logprob_at"]["k2"]
    want = {
        "ungated model": ["100.0%", pct(g["vanilla"]), pct(g["vanilla"]), "1.00"],
        "log-probability gate, matched coverage":
            [pct(lp["coverage"]), pct(lp["acc"]), pct(lp["yield"]), "1.00"],
        "mome": [pct(o["coverage"]), pct(o["acc"]), pct(o["yield"]),
                 f"{o['gens']:.2f}"],
    }
    for name, expected in want.items():
        got = rows.get(name)
        if got is None:
            a.check(f"README row '{name}'", None, None)
            continue
        a.check(f"README row '{name}'", got, expected)


def main() -> int:
    if not paths.SSOT.exists():
        print(f"missing: {paths.SSOT}")
        return 2
    ss = json.loads(paths.SSOT.read_text(encoding="utf-8"))
    a = Audit()
    check_predictor(a)
    state = check_rates(a, ss)
    check_strata(a, ss)
    check_bootstrap(a, ss, state["rows"])
    check_internal(a, ss)
    check_readme(a, ss)
    return a.report()


if __name__ == "__main__":
    raise SystemExit(main())
