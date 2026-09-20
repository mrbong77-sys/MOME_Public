#!/usr/bin/env python3
"""Measure how far the rule pack reaches, before any generation is spent.

Runs the perturbation engine over the 300 items of each of the three
benchmarks and counts how many valid probes each item yields.  No model is
called.  The point is to expose gaps in the rule pack before the probe
campaign runs, not to evaluate anything.

Inputs are read-only: the benchmark item text (not committed; rebuild it
locally from the public sources), the item lists, and one generation cell per
benchmark whose recorded gold answers the rebuilt items are checked against.

Before measuring anything it compares every rebuilt item's answer with that
recorded gold answer and stops on a single mismatch.  Coverage measured over
different items than the ones that were actually run says nothing.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perturb import MUST_CHANGE, MUST_HOLD, perturb, select_operating_set  # noqa: E402

import paths
BENCH_DIR = paths.BENCHMARKS
ITEMS_DIR = paths.ITEMS
CELLS_DIR = paths.UNGATED
OUT_DIR = paths.REPORTS

#: (benchmark, item file, language, one cell whose gold answers to check against)
BENCHES = [
    ("gsm8k", "gsm8k-test.jsonl", "en", "gsm8k_qwen3.5-9b_smoking_cb_vanilla"),
    ("hrm8k-gsm8k-ko", "hrm8k-gsm8k-ko.jsonl", "ko", "hrm8k-gsm8k-ko_solar-10.7b_smoking_cb_vanilla"),
    ("math500", "math500.jsonl", "en", "math500_solar-10.7b_smoking_cb_vanilla"),
]


def _close(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (ValueError, TypeError):
        return str(a).strip() == str(b).strip()


def verify_reconstruction(bench: dict, cell: str) -> tuple[int, int]:
    grades_path = CELLS_DIR / cell / "grades.jsonl"
    n = ok = 0
    for line in grades_path.read_text(encoding="utf-8").splitlines():
        g = json.loads(line)
        row = bench.get(g["item_id"])
        n += 1
        if row and _close(row["answer"], g["gold"]):
            ok += 1
    return ok, n


def survey(name: str, bench_file: str, lang: str, check_cell: str) -> dict:
    bench = {
        r["id"]: r
        for r in map(json.loads, (BENCH_DIR / bench_file).read_text(encoding="utf-8").splitlines())
    }
    ok, n = verify_reconstruction(bench, check_cell)
    if ok != n:
        raise SystemExit(f"{name}: rebuilt items disagree with the recorded "
                         f"gold on {n - ok}/{n} items -- stopping")

    ids = (ITEMS_DIR / f"{name}_smoking_n300.txt").read_text().split()
    rule_freq: Counter = Counter()
    per_item = []
    for item_id in ids:
        probes = perturb(bench[item_id]["question"], lang=lang)
        rule_freq.update(p.rule for p in probes)
        change = [p for p in probes if p.expectation == MUST_CHANGE]
        sel = select_operating_set(probes, k_change=1)
        sel_change = [p for p in sel if p.expectation == MUST_CHANGE]
        per_item.append({
            "item_id": item_id,
            "n_probes": len(probes),
            "n_change_high": sum(p.confidence == "high" for p in change),
            "n_change_medium": sum(p.confidence == "medium" for p in change),
            "n_hold": sum(p.expectation == MUST_HOLD for p in probes),
            "operating_change_conf": sel_change[0].confidence if sel_change else None,
        })

    n_items = len(per_item)
    high = sum(1 for r in per_item if r["n_change_high"] > 0)
    any_change = sum(1 for r in per_item if r["n_change_high"] + r["n_change_medium"] > 0)
    return {
        "benchmark": name,
        "language": lang,
        "n_items": n_items,
        "gold_check": f"{ok}/{n}",
        "items_with_high_change": high,
        "items_with_any_change": any_change,
        "items_hold_only": n_items - any_change,
        "mean_probes_per_item": round(sum(r["n_probes"] for r in per_item) / n_items, 2),
        "operating_set": dict(Counter(r["operating_change_conf"] or "none" for r in per_item)),
        "rule_frequency": dict(rule_freq.most_common()),
        "uncovered_items": [r["item_id"] for r in per_item
                            if r["n_change_high"] + r["n_change_medium"] == 0][:20],
        "per_item": per_item,
    }


def render(results: list[dict]) -> str:
    lines = [
        "# Perturbation coverage",
        "",
        "Produced by `coverage_rehearsal.py`. The perturbation engine was applied",
        "to the 300 items of each of the three benchmarks, to measure the rule",
        "pack's reach before the probe campaign was run. Every rebuilt item's "
        "answer matched the gold answer recorded in a generation cell.",
        "",
        "| benchmark | items | gold check | >=1 high-conf MUST_CHANGE "
        "| >=1 MUST_CHANGE (any) | hold probe only | mean probes/item |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['benchmark']} ({r['language']}) | {r['n_items']} | {r['gold_check']} "
            f"| {r['items_with_high_change']} ({r['items_with_high_change']/r['n_items']:.0%}) "
            f"| {r['items_with_any_change']} ({r['items_with_any_change']/r['n_items']:.0%}) "
            f"| {r['items_hold_only']} | {r['mean_probes_per_item']} |"
        )
    lines += ["", "## What the operating set is made of "
              "(one MUST_CHANGE and one MUST_HOLD per item)", "",
              "| benchmark | high-conf flip | medium-conf scaling | no MUST_CHANGE |",
              "|---|---:|---:|---:|"]
    for r in results:
        op = r["operating_set"]
        lines.append(f"| {r['benchmark']} | {op.get('high', 0)} | {op.get('medium', 0)} | {op.get('none', 0)} |")
    lines += ["", "## How often each rule fired", ""]
    for r in results:
        lines.append(f"### {r['benchmark']}")
        for rule, cnt in r["rule_frequency"].items():
            lines.append(f"- `{rule}`: {cnt}")
        if r["uncovered_items"]:
            lines.append(f"- items with no MUST_CHANGE probe (up to 20): "
                         f"{', '.join(r['uncovered_items'])}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    results = [survey(*spec) for spec in BENCHES]
    report = render(results)
    print(report)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "ccf-coverage.md").write_text(report, encoding="utf-8")
    slim = [{k: v for k, v in r.items() if k != "per_item"} for r in results]
    (OUT_DIR / "ccf-coverage.json").write_text(
        json.dumps({"summary": slim, "per_item": {r["benchmark"]: r["per_item"] for r in results}},
                   ensure_ascii=False, indent=1),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
