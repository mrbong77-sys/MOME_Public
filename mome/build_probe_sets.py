#!/usr/bin/env python3
"""Generate the probe item sets and the campaign configurations for them.

    python3 build_probe_sets.py                 # generate and check manifests
    python3 build_probe_sets.py --write-configs # rewrite the configs too

What it writes, per benchmark:

- `<bench>_probes.jsonl` -- the probe items themselves.  NOT committed: they
  are derivatives of the benchmark text and follow the same redistribution
  policy as the item files.  This script regenerates them deterministically.
- `<bench>_probe_ids.txt` -- the list of probe ids (committed).
- `<bench>_manifest.json` -- per-probe metadata plus the SHA-256 of each
  edited problem (committed).  That pins the probe set exactly without
  redistributing any benchmark text.

Rules it follows:

- Inputs are read-only; everything written lands under this script's own
  output directories.
- Before doing anything it checks every reconstructed item's answer against
  the gold answers recorded in a generation cell, and stops on a mismatch.
- When a committed manifest is already present it runs in check mode: if a
  regenerated probe's hash differs from the manifest, it stops.  The probe
  set is a pre-registered object, so changing it has to be a deliberate act
  -- delete the manifest, regenerate, and commit the new one.

Probe id convention: `<item_id>__<rule>`, e.g. `gsm8k-0__en.scale.x2`.  The
operating set is one MUST_CHANGE probe and one MUST_HOLD probe per item.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coverage_rehearsal import BENCHES, verify_reconstruction  # noqa: E402
from perturb import perturb, select_operating_set  # noqa: E402

import paths
BENCH_DIR = paths.BENCHMARKS
ITEMS_DIR = paths.ITEMS
PROBES_DIR = paths.MANIFESTS
CONFIGS_DIR = paths.PROBE_CONFIGS

#: The backbones, at the same operating point as the ungated campaigns.
BACKBONES = [
    ("gemma4-e2b", "gemma4:e2b"),
    ("solar-10.7b", "solar:10.7b"),
    ("qwen3.5-9b", "qwen3.5:9b"),
    ("qwen2-math-1.5b", "qwen2-math:1.5b"),
    ("gemma4-e4b", "gemma4:e4b"),  # added as an extension to the first four
]

#: Per-benchmark routing fields, copied verbatim from the ungated cells'
#: campaign metadata so that the two runs share an operating point.
BENCH_FIELDS = {
    "gsm8k": {"bench_file": "gsm8k-test.jsonl", "route": "numeric",
              "language": "en", "answer_kind": "numeric"},
    "hrm8k-gsm8k-ko": {"bench_file": "hrm8k-gsm8k-ko.jsonl", "route": "numeric",
                       "language": "ko", "answer_kind": "numeric"},
    "math500": {"bench_file": "math500.jsonl", "route": "numeric",
                "language": "en", "answer_kind": "expression"},
}

GENERATOR_FILES = ["mome/perturb.py", "mome/build_probe_sets.py"]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generator_fingerprint() -> dict:
    return {p: _sha((paths.ROOT / p).read_text(encoding="utf-8")) for p in GENERATOR_FILES}


def build_bench(name: str, lang: str, check_cell: str) -> tuple[list[dict], list[dict]]:
    """(probe rows, manifest entries)."""
    bench_path = BENCH_DIR / BENCH_FIELDS[name]["bench_file"]
    if not bench_path.exists():
        raise SystemExit(
            f"{bench_path} is missing. Item files are not committed; rebuild "
            f"them from the public benchmark sources first."
        )
    bench = {r["id"]: r for r in map(json.loads, bench_path.read_text(encoding="utf-8").splitlines())}
    ok, n = verify_reconstruction(bench, check_cell)
    if ok != n:
        raise SystemExit(f"{name}: rebuilt items disagree with the recorded gold "
                         f"answers on {n - ok}/{n} items -- stopping")

    ids = (ITEMS_DIR / f"{name}_smoking_n300.txt").read_text().split()
    rows, manifest = [], []
    for item_id in ids:
        for p in select_operating_set(perturb(bench[item_id]["question"], lang=lang), k_change=1):
            probe_id = f"{item_id}__{p.rule}"
            rows.append({"id": probe_id, "question": p.text, "answer": ""})
            manifest.append({
                "probe_id": probe_id, "item_id": item_id, "rule": p.rule,
                "expectation": p.expectation, "confidence": p.confidence,
                "direction": p.direction, "scale": p.scale, "note": p.note,
                "question_sha256": _sha(p.text),
            })
    return rows, manifest


def write_or_verify(name: str, rows: list[dict], manifest: list[dict]) -> str:
    PROBES_DIR.mkdir(parents=True, exist_ok=True)
    man_path = PROBES_DIR / f"{name}_manifest.json"
    payload = {
        "benchmark": name,
        "n_probes": len(manifest),
        "operating_set": "one MUST_CHANGE (highest confidence first) "
                         "and one MUST_HOLD probe per item",
        "generator_fingerprint": generator_fingerprint(),
        "probes": manifest,
    }
    if man_path.exists():
        committed = json.loads(man_path.read_text(encoding="utf-8"))
        theirs = {p["probe_id"]: p["question_sha256"] for p in committed["probes"]}
        ours = {p["probe_id"]: p["question_sha256"] for p in manifest}
        if theirs != ours:
            added = sorted(set(ours) - set(theirs))[:5]
            gone = sorted(set(theirs) - set(ours))[:5]
            changed = sorted(k for k in ours.keys() & theirs.keys() if ours[k] != theirs[k])[:5]
            raise SystemExit(
                f"{name}: regenerated probes differ from the committed manifest "
                f"-- stopping.\n"
                f"  added {added}\n  gone {gone}\n  changed {changed}\n"
                f"If the perturbation engine changed on purpose, update the "
                f"manifest deliberately: delete it, rerun, and commit."
            )
        status = "manifest check passed"
    else:
        man_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        status = "manifest written"

    with (PROBES_DIR / f"{name}_probes.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (PROBES_DIR / f"{name}_probe_ids.txt").write_text(
        "".join(r["id"] + "\n" for r in rows), encoding="utf-8")
    return status


def write_configs() -> int:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for bench, fields in BENCH_FIELDS.items():
        for backbone, model in BACKBONES:
            cell = f"{bench}_{backbone}_ccf_probes_v1"
            cfg = {
                "name": cell,
                "backbone": backbone,
                "model": model,
                "benchmark": bench,
                "benchmark_path": [f"data/manifests/{bench}_probes.jsonl"],
                "items_file": f"data/manifests/{bench}_probe_ids.txt",
                "route": fields["route"],
                "language": fields["language"],
                "answer_kind": fields["answer_kind"],
                "endpoint": "/api/generate",
                "options": {"num_predict": 2048, "temperature": 0.0, "top_p": 1.0},
                "seed": None,
                "logprobs": True,
                "top_logprobs": 10,
                "think": "auto",
                "host": "http://localhost:11434",
                "timeout_seconds": 900,
                "keep_alive": None,
                "out_dir": f"data/probes/{cell}",
                "purpose": (
                    "One cell of the probe campaign, at the same operating "
                    "point as the ungated cell it pairs with (num_predict "
                    "2048, temperature 0.0, top_p 1.0, top_logprobs 10, think "
                    "auto): one greedy generation for each edited version of "
                    "the same 300 items, one MUST_CHANGE and one MUST_HOLD "
                    "per item. Nothing is graded here -- reading a probe "
                    "needs no correct answer. The probe set is pinned by the "
                    "committed manifest, whose hashes build_probe_sets.py "
                    "checks before the run starts."
                ),
                "probe_note": (
                    "This cell pairs with the ungated cell "
                    f"{bench}_{backbone}_{'ext' if backbone == 'gemma4-e4b' else 'smoking'}_cb_vanilla. The fingerprint is "
                    "assembled by joining that cell's original responses with "
                    "this cell's probe responses on the item_id inside each "
                    "probe_id."
                ),
            }
            (CONFIGS_DIR / f"{cell}.json").write_text(
                json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
            n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-configs", action="store_true",
                        help="also (re)write the campaign configuration files")
    args = parser.parse_args()

    total = 0
    for name, _bench_file, lang, check_cell in BENCHES:
        rows, manifest = build_bench(name, lang, check_cell)
        status = write_or_verify(name, rows, manifest)
        total += len(rows)
        print(f"{name}: {len(rows)} probes -- {status}")
    print(f"total {total} probes (generation budget: {total} per backbone, "
          f"{4 * total} across four)")
    if args.write_configs:
        print(f"wrote {write_configs()} campaign configs -> {CONFIGS_DIR}")


if __name__ == "__main__":
    main()
