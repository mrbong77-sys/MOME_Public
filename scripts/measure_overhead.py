#!/usr/bin/env python3
"""What the monitor itself costs: no generation, no runtime, any machine.

    python3 scripts/measure_overhead.py            # measure on committed records
    python3 scripts/measure_overhead.py --n 2000   # more repetitions

What is measured and what is not is worth stating first.  This is a routing
wrapper: it never opens the model, loads no second network, and needs no tensor
runtime.  What it adds on the host is therefore a rule engine and a logistic
score over 26 numbers, and that is exactly what this script times:

  1. probe generation   perturb + select_operating_set   (rules, no model call)
  2. probe observation  fingerprint.observe              (normalize, compare)
  3. score and decide   c_safe + the first-stage bands   (pure-Python logistic)

Everything else in an item's wall-clock is the backbone emitting tokens.  That
is a property of the model's size, its quantization and the host, not of this
method, so it is not measured here and is not reported as this method's cost.
It is also why generations per item is the cost unit throughout: it removes the
host and leaves the policy.

The result is written to results/overhead.json.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mome"))
import paths                 # noqa: E402
import perturb as P          # noqa: E402
import fingerprint as F      # noqa: E402
import gate                  # noqa: E402

OUT = paths.OVERHEAD
DATASET = paths.FEATURES
BENCH = paths.BENCHMARKS
MODEL_BO = paths.MODEL_BACKBONE_OUT

LANG = {"gsm8k": "en", "hrm8k-gsm8k-ko": "ko", "math500": "en"}
KIND = {"gsm8k": "numeric", "hrm8k-gsm8k-ko": "numeric", "math500": "expression"}
#: Benchmark file names (only the test split is kept for the word problems).
FILE = {"gsm8k": "gsm8k-test.jsonl", "hrm8k-gsm8k-ko": "hrm8k-gsm8k-ko.jsonl",
        "math500": "math500.jsonl"}


def pct(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))]


def ms(xs) -> dict:
    return {"p50_ms": round(pct(xs, 0.50) * 1e3, 4),
            "p95_ms": round(pct(xs, 0.95) * 1e3, 4),
            "mean_ms": round(st.mean(xs) * 1e3, 4), "n": len(xs)}


def questions(bench: str, n: int) -> list[str]:
    src = BENCH / FILE[bench]
    out = []
    for line in src.open(encoding="utf-8"):
        r = json.loads(line)
        out.append(r.get("question") or r.get("problem"))
        if len(out) >= n:
            break
    return out


def time_probe_generation(n_per_bench: int) -> dict:
    out = {}
    for bench, lang in LANG.items():
        qs = questions(bench, n_per_bench)
        ts = []
        for q in qs:
            t0 = time.perf_counter()
            P.select_operating_set(P.perturb(q, lang))
            ts.append(time.perf_counter() - t0)
        out[bench] = ms(ts)
    return out


def time_observation(n: int) -> dict:
    """Cost of one observation: answer normalization, plus the symbolic
    comparison when the answer is an expression."""
    pairs = {"numeric": [("42", "42"), ("42", "84"), ("1,200", "1200")],
             "expression": [("x^2+2x+1", "(x+1)^2"), ("\\frac{1}{2}", "0.5"),
                            ("2\\pi", "2\\pi")]}
    out = {}
    for kind, ps in pairs.items():
        ts = []
        for i in range(n):
            a, b = ps[i % len(ps)]
            t0 = time.perf_counter()
            F.answers_equal(a, b, kind)
            ts.append(time.perf_counter() - t0)
        out[kind] = ms(ts)
    return out


def time_scoring(n: int) -> dict:
    models = json.loads(MODEL_BO.read_text(encoding="utf-8"))
    rows = []
    for line in DATASET.open(encoding="utf-8"):
        rows.append(json.loads(line))
        if len(rows) >= n:
            break
    ts, tau = [], (0.70, 0.3125)
    for r in rows:
        m = models[r["backbone"]]
        t0 = time.perf_counter()
        gate.stage1(r["features"], m, tau[0], tau[1])
        ts.append(time.perf_counter() - t0)
    return ms(ts)


def footprint() -> dict:
    """Peak Python allocation with everything the decision needs loaded, and
    the size of the coefficient file on disk."""
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    models = json.loads(MODEL_BO.read_text(encoding="utf-8"))
    m = models["gemma4-e4b"]
    gate.stage1({k: 0.0 for k in m["feature_names"]}, m, 0.70, 0.3125)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return {"predictor_bytes_on_disk": MODEL_BO.stat().st_size,
            "one_predictor_bytes_on_disk": len(json.dumps(m).encode("utf-8")),
            "python_peak_bytes_loading_and_scoring": peak - base,
            "n_features": len(m["feature_names"]),
            "third_party_at_deployment": []}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="items / repetitions")
    a = ap.parse_args()

    print("What the monitor itself costs -- zero model calls")
    gen = time_probe_generation(min(a.n, 300))
    print("  probe generation   " + ", ".join(f"{k} p50 {v['p50_ms']:.2f} ms" for k, v in gen.items()))
    obs = time_observation(a.n)
    print("  probe observation  " + ", ".join(f"{k} p50 {v['p50_ms']:.3f} ms" for k, v in obs.items()))
    sc = time_scoring(a.n)
    print(f"  score and decide   p50 {sc['p50_ms']:.3f} ms")
    fp = footprint()
    print(f"  predictor          {fp['one_predictor_bytes_on_disk'] / 1024:.1f} KB, "
          f"{fp['n_features']} features, no third-party library at deployment")

    data = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.version.split()[0],
        "note": ("what MOME itself costs on the host: the rule engine, the observation "
                 "and the logistic score. The generation time of the backbone is not "
                 "measured here and is not a property of this method."),
        "probe_generation": gen,
        "observation": obs,
        "score_and_decision": sc,
        "footprint": fp,
    }
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
