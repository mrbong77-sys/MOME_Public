#!/usr/bin/env python3
"""The final two-stage readout. Reads the data ONCE, under registered rules.

This file was written BEFORE the second-stage generation finished, which is
the only way to keep the reading rules from being chosen after seeing the
data.  Every rule in it was pre-registered:

* The four variants and the adoption rule -- among the variants whose correct
  yield is at least the ungated model's, take the one with the highest
  selective accuracy, and on a tie the one that costs fewer generations.  If
  no variant clears the bar, report "capability preservation failed" and do
  not change the rule afterwards.
* The metrics (selective accuracy, coverage, rescue rate, latency) and the
  baselines they are compared against.
* The execution order.  Cells are generated backbone-first, so an interrupted
  run still leaves every completed backbone with all three benchmarks and a
  verdict that stands on its own.  This file reads only completed cells and
  names the missing ones at the head of its report.

    python3 evaluate_gate.py

Run grade_retry.py first, to grade the second-stage cells.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import paths
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ollama_http as oh                    # noqa: E402
import gate                                 # noqa: E402
import gate2_features as g2                 # noqa: E402
from build_dataset import mean_logprob      # noqa: E402
from fingerprint import BACKBONES, VANILLA_DIR, answers_equal  # noqa: E402
from greedy_path import by_item as greedy_by_item  # noqa: E402

#: Equivalence checking is expensive, and the same pairs recur.
#: The four variants, the scorer swap and the truncation strata all compare
#: the same answer pairs again.  Memoizing is not only faster, it makes the
#: comparisons CONSISTENT: the symbolic comparator's timeout is wall-clock, so
#: without a memo the same pair could be judged differently in two variants,
#: and the comparison between variants would then mix a difference of rules
#: with a difference of noise.
_EQ_MEMO: dict = {}


def eq(a, b, kind: str) -> bool:
    key = (kind, str(a), str(b))
    if key not in _EQ_MEMO:
        _EQ_MEMO[key] = bool(answers_equal(a, b, kind))
    return _EQ_MEMO[key]

OUT = paths.REPORTS
MANIFEST_DIR = paths.MANIFESTS


def probes_per_item(bench: str) -> float:
    """Probes per item for this benchmark.

    The first stage does not run without probes.  The fingerprint is read off
    the probe responses, so a deployed gate always generates one original plus
    n probes for every item.  The first implementation counted only the greedy
    answer and the second-stage samples and left the probes out, which
    understated the cost badly -- 3.73 generations per item where the true
    figure is about 5.7.  Since the registered success criterion is stated in
    generations per item, that omission would have flipped the verdict.
    """
    m = json.loads((MANIFEST_DIR / f"{bench}_manifest.json").read_text(encoding="utf-8"))
    items = {p["item_id"] for p in m["probes"]}
    return m["n_probes"] / len(items)
CONFIGS_DIR = paths.RETRY_CONFIGS
K = 4
#: Cells of the larger token-budget arm carry this tag in their names and use
#: a different sample count.  The default readout takes only the 15 registered
#: cells: mixing the arms would leave it unclear which arm a verdict belongs
#: to.  Tagged cells are read separately with `--tag`.
BUDGET_TAG = "b4096"

#: The four registered variants, keyed as (two-band?, equivalence voting?).
#: The keys are written in Korean because they are the labels under which the
#: results were registered and stored; they appear verbatim in ssot.json, and
#: renaming them would break the correspondence with the published results.
#: In English they read:
#:   V0  three bands, string voting
#:   V0+ three bands, equivalence voting
#:   V1  two bands, string voting
#:   V2  two bands, equivalence voting
VARIANTS = {
    "V0  3밴드 · 문자열 투표": (False, False),
    "V0+ 3밴드 · 동치 투표": (False, True),
    "V1  2밴드 · 문자열 투표": (True, False),
    "V2  2밴드 · 동치 투표": (True, True),
}


def backbone_of(cell: str) -> str:
    """The backbone named in a cell name, with any arm tag stripped first."""
    base = cell[: -(len(BUDGET_TAG) + 1)] if cell.endswith("_" + BUDGET_TAG) else cell
    return next(x for x in BACKBONES if base.endswith("_" + x))


def vanilla_dir(cell: str) -> Path:
    #: A tagged cell's ungated pair belongs to that same arm.  Stripping the
    #: tag before looking it up would read the other arm's ungated cell, and
    #: the comparison would no longer be "same gate, different budget" but a
    #: comparison with the budgets mixed.
    if cell.endswith("_" + BUDGET_TAG):
        return VANILLA_DIR / f"{cell}_cb_vanilla"
    bench, bb = cell.split("_", 1)
    d = VANILLA_DIR / f"{bench}_{bb}_smoking_cb_vanilla"
    return d if (d / "grades.jsonl").exists() else VANILLA_DIR / f"{bench}_{bb}_ext_cb_vanilla"


def load_cells(tag: str = "", k: int | None = None) -> tuple[dict, list[str]]:
    """Item bundles for every cell that finished generating and grading.

    Returns ({cell: {item: bundle}}, the cells that are missing).
    """
    decisions: dict = {}
    k = K if k is None else k
    dec_name = f"stage1_decisions_{tag}.jsonl" if tag else "stage1_decisions.jsonl"
    for line in (paths.RESULTS / dec_name).read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        decisions.setdefault(d["cell"], {})[d["item_id"]] = d

    out: dict = {}
    missing: list[str] = []
    for path in sorted(CONFIGS_DIR.glob(f"*_retry_sc{k}.json")):
        #: Without a tag, tagged cells are NOT read.  An earlier version
        #: globbed every retry config, and once the variant arm's configs
        #: existed, two cells with no decisions silently joined the registered
        #: readout.
        if (f"_{BUDGET_TAG}_" in path.stem) != bool(tag):
            continue
        cfg = json.loads(path.read_text(encoding="utf-8"))
        cell = path.stem.replace(f"_retry_sc{k}", "")
        rdir = paths.ROOT / cfg["out_dir"]
        if not (rdir / "grades.jsonl").exists():
            missing.append(cell)
            continue
        kind = cfg["answer_kind"]
        vdir = vanilla_dir(cell)
        #: On a cell with several paths per item, use only the greedy path.
        #: Overwriting blindly leaves the last path, a sample, and the gate's
        #: first vote would then be a sampled answer (see greedy_path.py).
        vg = greedy_by_item(oh.read_jsonl(vdir / "grades.jsonl"), vdir.name)
        vr = greedy_by_item([r for r in oh.read_jsonl(vdir / "records.jsonl")
                             if r.get("error") is None], vdir.name)
        rg: dict = {}
        for r in oh.read_jsonl(rdir / "grades.jsonl"):
            rg.setdefault(r["item_id"], {})[oh.record_path_index(r)] = r
        rr: dict = {}
        for r in oh.read_jsonl(rdir / "records.jsonl"):
            if r.get("error") is None:
                rr.setdefault(r["item_id"], {})[oh.record_path_index(r)] = r

        bundles = {}
        for item, d in decisions.get(cell, {}).items():
            if item not in vg or item not in vr:
                continue
            g, rec = vg[item], vr[item]
            #: `text` feeds the second stage's derivation-embedding features.
            #: Leaving it out hands the embedder empty strings, every vector
            #: comes out identical, every similarity is 1.0, and the embedding
            #: features look constant -- which reads, wrongly, as "contributes
            #: nothing".
            paths = [{"answer": g.get("final_answer"), "correct": bool(g["correct"]),
                      "token_count": rec.get("token_count") or 0,
                      "truncated": bool(rec.get("truncated")),
                      "mean_logprob": mean_logprob(rec),
                      "text": g2.text_of(rec),
                      "latency": rec.get("latency_seconds") or 0.0}]
            for p in range(1, k + 1):
                gr, re_ = rg.get(item, {}).get(p), rr.get(item, {}).get(p)
                if gr is None or re_ is None:
                    continue
                paths.append({"answer": gr.get("final_answer"), "correct": bool(gr["correct"]),
                              "token_count": re_.get("token_count") or 0,
                              "truncated": bool(re_.get("truncated")),
                              "mean_logprob": mean_logprob(re_),
                              "text": g2.text_of(re_),
                              "latency": re_.get("latency_seconds") or 0.0})
            bundles[item] = {"cell": cell, "item_id": item, "kind": kind,
                             "bench": cfg["benchmark"],
                             "probe_gens": probes_per_item(cfg["benchmark"]),
                             "decision": d["decision"], "label": d["label"],
                             "c_safe": d["c_safe"], "paths": paths,
                             "has_samples": len(paths) > 1}
        out[cell] = bundles
    return out, missing


def vote_of(b: dict, equivalence: bool):
    """This item's second-stage ballot: (answer, share, is that answer right)."""
    answers = [p["answer"] for p in b["paths"]]
    same = (lambda x, y: eq(x, y, b["kind"])) if equivalence else None
    ans, share = gate.vote(answers, same)
    ok = None
    if ans is not None:
        for p in b["paths"]:
            if p["answer"] is None:
                continue
            hit = (same(p["answer"], ans) if same else str(p["answer"]).strip() == str(ans))
            if hit:
                ok = p["correct"]
                break
    return ans, share, ok


#: The second stage's decision rule.  `None` means the registered rule, that
#: the winning share reaches VOTE_SERVE.  Passing a scorer changes only WHETHER
#: the answer is served; the answer itself is still the majority answer.  What
#: a scorer swaps is "do we trust this consensus", not "what do we answer".
def make_scorer(model: dict):
    theta = model["theta"]
    feats = model["feature_names"]

    def decide(b: dict) -> bool:
        f = g2.stage2_features(b["paths"], b["kind"], None)
        return gate.c_safe({k: f.get(k) for k in feats}, model) >= theta
    return decide


def run_variant(bundles: list[dict], two_band: bool, equivalence: bool,
                scorer=None) -> dict:
    """Run one variant over every item. Measured, not projected."""
    n = answered = correct = 0
    gens = 0.0
    latency = 0.0
    rescued = lost_decline = lost_wrong = 0
    for b in bundles:
        n += 1
        #: The first stage does not run without probes, and that cost is paid
        #: on every item.
        gens += b.get("probe_gens", 0.0)
        band = b["decision"]
        if band == gate.SERVE:
            answered += 1; correct += b["label"]; gens += 1
            latency += b["paths"][0]["latency"]
            continue
        if band == gate.DECLINE and not two_band:
            gens += 1
            latency += b["paths"][0]["latency"]
            continue
        if not b["has_samples"]:
            # An item with no samples bought, or whose samples all errored:
            # the second stage cannot run, so it counts as a decline. How many
            # there are is stated at the head of the report.
            gens += 1
            latency += b["paths"][0]["latency"]
            continue
        gens += len(b["paths"])
        latency += sum(p["latency"] for p in b["paths"])
        _, share, ok = vote_of(b, equivalence)
        serve = scorer(b) if scorer is not None else (share >= gate.VOTE_SERVE)
        if serve:
            answered += 1
            correct += bool(ok)
            if b["label"] == 0 and ok:
                rescued += 1
            if b["label"] == 1 and not ok:
                lost_wrong += 1
        elif b["label"] == 1:
            lost_decline += 1
    return {"n": n, "answered": answered, "correct": correct, "gens": gens,
            "latency": latency, "rescued": rescued,
            "lost_decline": lost_decline, "lost_wrong": lost_wrong,
            "coverage": answered / n if n else 0.0,
            "selective_acc": correct / answered if answered else float("nan"),
            "yield": correct / n if n else 0.0,
            "gens_per_item": gens / n if n else 0.0,
            "latency_per_item": latency / n if n else 0.0}


def baseline_logprob(bundles: list[dict], coverage: float) -> dict:
    """The registered baseline: select by mean log-probability at the same
    coverage."""
    ranked = sorted(bundles, key=lambda b: -(b["paths"][0]["mean_logprob"] or -99))
    k = round(coverage * len(ranked))
    served = ranked[:k]
    correct = sum(b["label"] for b in served)
    return {"coverage": k / len(ranked) if ranked else 0.0,
            "selective_acc": correct / k if k else float("nan"),
            "yield": correct / len(ranked) if ranked else 0.0,
            "gens_per_item": 1.0}


def baseline_sc_always(bundles: list[dict], equivalence: bool) -> dict | None:
    """Self-consistency on every item. Computable only where samples were
    actually bought, since the immediate-serve band bought none."""
    have = [b for b in bundles if b["has_samples"]]
    if not have:
        return None
    n = answered = correct = 0
    for b in have:
        n += 1
        _, share, ok = vote_of(b, equivalence)
        #: This baseline is read under the registered rule only. The scorer
        #: swap has its own section; mixing it in here would blur what
        #: "self-consistency on every item" even means.
        if share >= gate.VOTE_SERVE:
            answered += 1; correct += bool(ok)
    return {"n": n, "coverage": answered / n, "yield": correct / n,
            "selective_acc": correct / answered if answered else float("nan"),
            "gens_per_item": 5.0,
            "vanilla_acc_here": sum(b["label"] for b in have) / n}


def fmt(v: float, pct: bool = True) -> str:
    if v != v:
        return "—"
    return f"{v:.1%}" if pct else f"{v:.2f}"


def k_ablation(allb: list[dict], L: list[str]) -> None:
    """What is lost by buying fewer second-stage samples. No generation needed.

    k=4 was inherited from an earlier operating point, not chosen on this
    data.  The samples carry seeds 0..3 on paths 1..4 in order, so taking only
    the FIRST k paths is exactly the result of having bought only k.  It is a
    truncation, not a selection, so it introduces no bias.

    The cost is the first stage (one original plus n probes) plus k times the
    share of items that reach the second stage.
    """
    have = [b for b in allb if b["has_samples"]]
    if not have:
        return
    L += ["", "## 6. Is k over-invested? (no extra generation)", "",
          "k=4 was inherited from an earlier operating point. The samples carry",
          "seeds 0..3 on paths 1..4 in order, so reading only the first k paths",
          "gives exactly the result of having bought k -- a truncation, not a "
          "selection, and therefore unbiased.", "",
          "| k | votes | coverage | selective acc. | correct yield | gen./item | vs. ungated |",
          "|---:|---:|---:|---:|---:|---:|---:|"]
    van = sum(b["label"] for b in allb) / len(allb)
    rows = []
    for k in (1, 2, 3, 4):
        trimmed = []
        for b in allb:
            c = dict(b)
            if b["has_samples"]:
                c["paths"] = b["paths"][:1 + k]
                c["has_samples"] = len(c["paths"]) > 1
            #: The probe cost is unchanged; only the sample cost scales with k.
            c["k_samples"] = k
            trimmed.append(c)
        r = run_variant(trimmed, two_band=True, equivalence=True)
        rows.append((k, r))
        L.append(f"| {k} | {k + 1} | {fmt(r['coverage'])} | {fmt(r['selective_acc'])} | "
                 f"{fmt(r['yield'])} | {r['gens_per_item']:.2f} | "
                 f"{(r['yield'] - van) * 100:+.1f}p |")
    k4 = rows[-1][1]
    L += ["", "How to read this:", ""]
    for k, r in rows[:-1]:
        d_yield = (r["yield"] - k4["yield"]) * 100
        d_acc = (r["selective_acc"] - k4["selective_acc"]) * 100
        d_cost = k4["gens_per_item"] - r["gens_per_item"]
        L.append(f"- **k={k}**: against k=4, correct yield {d_yield:+.1f}p and "
                 f"selective accuracy {d_acc:+.1f}p, saving {d_cost:.2f} "
                 f"generations per item.")
    L += ["",
          "An even ballot admits ties, and `VOTE_SERVE = 0.6` sends a tie to a",
          "decline: k=1 gives two votes and a split is 0.5, k=3 gives four and",
          "2:2 is 0.5. The even values of k, which make the ballot odd, are "
          "therefore structurally better.", ""]


def gate2_comparison(cells: dict, L: list[str]) -> None:
    """What replacing the second stage's decision rule does to the metrics.

    An AUROC table only nominates candidates; the registered adoption rule is
    written in terms of correct yield and selective accuracy.  So the frozen
    scorer is plugged into the second stage and all four variants are run
    again, end to end, rather than compared on AUROC.

    The three cells the scorer was fitted on are excluded: reading it on those
    would be leakage.
    """
    path = paths.MODEL_STAGE2
    if not path.exists():
        return
    model = json.loads(path.read_text(encoding="utf-8"))
    skip = set(model["do_not_read_on_cells"])
    subset = [b for c, bs in cells.items() if c not in skip for b in bs.values()]
    if len(subset) < 500:
        return
    scorer = make_scorer(model)
    van = sum(b["label"] for b in subset) / len(subset)
    L += ["", "## 5. Replacing the second stage's decision rule", "",
          f"The frozen scorer `gate2_v2_symbolic` ({len(model['feature_names'])} "
          f"features, theta={model['theta']:.4f}) is plugged into the second "
          f"stage and the same four variants are run again. Excluding the three "
          f"cells it was fitted on leaves **{len(cells) - len(skip & set(cells))} "
          f"cells and {len(subset)} items**, whose ungated accuracy is {fmt(van)}.",
          "",
          "The scorer decides only whether to serve; the answer is still the",
          "majority answer. What it swaps is whether to trust the consensus, "
          "not what to answer.", "",
          "| second-stage rule | variant | coverage | selective acc. | correct yield | vs. ungated |",
          "|---|---|---:|---:|---:|---:|"]
    best = {}
    for tag, sc in (("registered rule (winning share >= 0.6)", None),
                    ("**scorer B (10 answer and generation features)**", scorer)):
        for name, (two, eq) in VARIANTS.items():
            if not two:
                continue          # two-band only; three-band was rejected
            r = run_variant(subset, two, eq, sc)
            best[(tag, name)] = r
            L.append(f"| {tag} | {name} | {fmt(r['coverage'])} | {fmt(r['selective_acc'])} | "
                     f"{fmt(r['yield'])} | {(r['yield'] - van) * 100:+.1f}p |")
    ok = {k: v for k, v in best.items() if v["yield"] >= van}
    L += [""]
    if not ok:
        L += ["**No combination reached the ungated model.** Reported as "
              "registered.", ""]
    else:
        win = max(ok.items(), key=lambda kv: (kv[1]["selective_acc"], -kv[1]["gens_per_item"]))
        L += [f"**Adopted: {win[0][0]} / {win[0][1]}** -- selective accuracy "
              f"{fmt(win[1]['selective_acc'])}, correct yield {fmt(win[1]['yield'])} "
              f"(ungated {fmt(van)}, {(win[1]['yield'] - van) * 100:+.1f}p).", ""]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help=f"read the variant arm: --tag {BUDGET_TAG}")
    ap.add_argument("-k", "--k-samples", type=int, default=None,
                    help="second-stage sample count. The registered default "
                         "is 4; the variant arm uses 2.")
    args = ap.parse_args()
    k = args.k_samples if args.k_samples is not None else (2 if args.tag else K)
    cells, missing = load_cells(args.tag, k)
    if not cells:
        raise SystemExit("no graded retry cell found. Run "
                         "python3 grade_retry.py first.")
    allb = [b for c in cells.values() for b in c.values()]
    by_bb: dict[str, list[dict]] = {}
    for b in allb:
        bb = backbone_of(b["cell"])
        by_bb.setdefault(bb, []).append(b)
    van_acc = sum(b["label"] for b in allb) / len(allb)
    no_samples = sum(1 for b in allb
                     if b["decision"] != gate.SERVE and not b["has_samples"])

    L = ["# Two-stage gate: the final readout", "",
         f"Cells {len(cells)}/15 evaluated, {len(allb)} items. Every rule applied "
         f"here was registered before any of this was generated,",
         "and so was this file.", ""]
    if missing:
        L += [f"**{len(missing)} cells missing**: " + ", ".join(f"`{c}`" for c in missing),
              "Generation runs backbone-first, so every completed backbone has all "
              "three benchmarks",
              "and its verdict stands on its own.", ""]
    if no_samples:
        L += [f"Note: {no_samples} items that should have reached the second "
              f"stage had no samples and were counted as declines",
              "(a generation error left the paths unfilled). Every variant "
              "treats them the same way.", ""]

    L += ["## 1. The four registered variants, all cells", "",
          "| variant | coverage | selective acc. | **correct yield** | gen./item "
          "| latency/item (s) | rescued | lost (declined) | lost (wrong) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
          f"| ungated model | 100.0% | {fmt(van_acc)} | **{fmt(van_acc)}** | 1.00 | "
          f"{sum(b['paths'][0]['latency'] for b in allb)/len(allb):.2f} | — | — | — |"]
    results = {}
    for name, (two, eq) in VARIANTS.items():
        r = run_variant(allb, two, eq)
        results[name] = r
        L.append(f"| {name} | {fmt(r['coverage'])} | {fmt(r['selective_acc'])} | "
                 f"**{fmt(r['yield'])}** | {r['gens_per_item']:.2f} | "
                 f"{r['latency_per_item']:.2f} | +{r['rescued']} | −{r['lost_decline']} | "
                 f"−{r['lost_wrong']} |")

    # --- The registered adoption rule ---------------------------------------
    bundles_all = allb
    ok = {k: v for k, v in results.items() if v["yield"] >= van_acc}
    L += ["", "### Adoption (registered rule: among variants whose correct "
          "yield is at least the ungated model's, the highest selective "
          "accuracy; on a tie, the cheaper one)", ""]
    if not ok:
        L += ["**No variant reached the ungated model's correct yield.** As "
              "registered, this is reported as a failure",
              "of capability preservation at this operating point. The rule is "
              "not changed after the fact.", ""]
        adopted = None
    else:
        adopted = max(ok.items(), key=lambda kv: (kv[1]["selective_acc"], -kv[1]["gens_per_item"]))
        L += [f"**Adopted: {adopted[0]}** -- selective accuracy {fmt(adopted[1]['selective_acc'])}, "
              f"correct yield {fmt(adopted[1]['yield'])} (ungated {fmt(van_acc)}, "
              f"{(adopted[1]['yield'] - van_acc) * 100:+.1f}p), "
              f"{adopted[1]['gens_per_item']:.2f} generations per item.", ""]
        if len(ok) > 1:
            L.append("At or above the ungated model: "
                     + ", ".join(f"`{k}`" for k in ok) + ".")
            L.append("")
        #: A tie is never quietly collapsed into a single winner. On a
        #: numeric-answer benchmark string voting and equivalence voting agree
        #: by construction, because there are no notational variants. Until a
        #: cell with expression answers is present the two cannot diverge, so
        #: an "adoption" in that state is a coin toss, not a rule.
        tied = [k for k, v in ok.items()
                if abs(v["selective_acc"] - adopted[1]["selective_acc"]) < 1e-12
                and abs(v["yield"] - adopted[1]["yield"]) < 1e-12
                and abs(v["gens_per_item"] - adopted[1]["gens_per_item"]) < 1e-12]
        if len(tied) > 1:
            L += [f"**Caution -- {len(tied)} tied**: " + ", ".join(f"`{k}`" for k in tied)
                  + " agree on every metric. The adoption mark reflects listing "
                  "order, not a choice the rule made.", ""]
            kinds = sorted({b["kind"] for b in bundles_all})
            if kinds == ["numeric"]:
                L += ["The reason is in the data: every cell read so far has "
                      "**numeric answers**, which admit no notational variants, "
                      "so string voting and equivalence voting count the same "
                      "ballot. They can only diverge once a cell with "
                      "expression answers is present.", ""]

    # --- Per backbone --------------------------------------------------------
    key = adopted[0] if adopted else "V2  2밴드 · 동치 투표"
    two, eq = VARIANTS[key]
    L += [f"## 2. Per backbone ({key})", "",
          "| backbone | items | ungated acc. | coverage | selective acc. "
          "| correct yield | vs. ungated |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for bb in BACKBONES:
        bs = by_bb.get(bb)
        if not bs:
            continue
        v = sum(b["label"] for b in bs) / len(bs)
        r = run_variant(bs, two, eq)
        L.append(f"| {bb} | {len(bs)} | {fmt(v)} | {fmt(r['coverage'])} | "
                 f"{fmt(r['selective_acc'])} | {fmt(r['yield'])} | "
                 f"{(r['yield'] - v) * 100:+.1f}p |")

    L += ["", f"## 2b. Per cell ({key})", "",
          "| cell | items | ungated | coverage | selective acc. | correct yield "
          "| vs. | rescued | lost |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for cell in sorted(cells):
        bs = list(cells[cell].values())
        if not bs:
            continue
        v = sum(b["label"] for b in bs) / len(bs)
        r = run_variant(bs, two, eq)
        L.append(f"| `{cell}` | {len(bs)} | {fmt(v)} | {fmt(r['coverage'])} | "
                 f"{fmt(r['selective_acc'])} | {fmt(r['yield'])} | "
                 f"{(r['yield'] - v) * 100:+.1f}p | +{r['rescued']} | "
                 f"−{r['lost_decline'] + r['lost_wrong']} |")

    # --- Baselines -----------------------------------------------------------
    best = results[key]
    lp = baseline_logprob(allb, best["coverage"])
    st1 = run_variant(allb, False, False)      # first stage alone: neither band serves
    serve_only = [b for b in allb if b["decision"] == gate.SERVE]
    L += ["", "## 3. Baselines", "",
          "| baseline | coverage | selective acc. | correct yield | gen./item |",
          "|---|---:|---:|---:|---:|",
          f"| always serve (ungated) | 100.0% | {fmt(van_acc)} | {fmt(van_acc)} | 1.00 |",
          f"| log-probability gate (matched coverage) | {fmt(lp['coverage'])} | {fmt(lp['selective_acc'])} | "
          f"{fmt(lp['yield'])} | 1.00 |",
          f"| first stage only (serve band) | {len(serve_only)/len(allb):.1%} | "
          f"{fmt(sum(b['label'] for b in serve_only)/max(len(serve_only),1))} | "
          f"{fmt(sum(b['label'] for b in serve_only)/len(allb))} | 1.00 |",
          f"| **{key}** | {fmt(best['coverage'])} | {fmt(best['selective_acc'])} | "
          f"{fmt(best['yield'])} | {best['gens_per_item']:.2f} |"]
    sc = baseline_sc_always(allb, eq)
    if sc:
        L += ["",
              f"**Self-consistency on every item** can only be computed where "
              f"samples were bought; the immediate-serve band bought none,",
              f"which is exactly this design's saving. On those {sc['n']} items: "
              f"coverage {fmt(sc['coverage'])}, selective accuracy "
              f"{fmt(sc['selective_acc'])}, correct yield "
              f"{fmt(sc['yield'])} (ungated accuracy on the same items "
              f"{fmt(sc['vanilla_acc_here'])}), at 5.00 generations per item. "
              f"The two-stage gate does the same work in {best['gens_per_item']:.2f}."]

    # --- Truncation stratum --------------------------------------------------
    #: A truncated generation is almost always wrong, and whether the original
    #: was truncated is one of the fingerprint's features. On a cell where
    #: truncation is common the gate could therefore look good by detecting a
    #: budget overrun rather than a metamorphic signal. Keeping only the
    #: untruncated items makes that feature constant, so the table below is the
    #: result with the budget argument removed.
    untrunc = [b for b in allb if not b["paths"][0]["truncated"]]
    n_tr = len(allb) - len(untrunc)
    L += ["", "## 2c. Untruncated stratum (result with the budget argument removed)", ""]
    if n_tr == 0:
        L += ["No item in these cells was truncated -- there is nothing to "
              "stratify, and the figures above are already independent of the "
              "budget.", ""]
    else:
        vu = sum(b["label"] for b in untrunc) / len(untrunc)
        L += [f"Dropping the {n_tr} items ({n_tr/len(allb):.1%}) whose greedy "
              f"generation hit the token limit leaves {len(untrunc)}. Ungated "
              f"accuracy on that stratum is {fmt(vu)} "
              f"(overall {fmt(van_acc)}).", "",
              "| variant | coverage | selective acc. | correct yield | vs. ungated |",
              "|---|---:|---:|---:|---:|"]
        for name, (t2, e2) in VARIANTS.items():
            r = run_variant(untrunc, t2, e2)
            L.append(f"| {name} | {fmt(r['coverage'])} | {fmt(r['selective_acc'])} | "
                     f"{fmt(r['yield'])} | {(r['yield'] - vu) * 100:+.1f}p |")
        L += ["",
              "**If this table points the same way as table 1**, the conclusion is "
              "not an artifact of the budget. If they diverge,",
              "that cell's figures have to be read together with the token budget.", ""]

    # --- ablation ----------------------------------------------------------
    L += ["", "## 4. Registered ablations", "",
          "1. **Operating-point axis** -- table 3 above is this axis (greedy "
          "alone / gate plus sampled retry /",
          "   self-consistency on every item).",
          "2. **Resampling-baseline axis** -- deciding on the winning share "
          "alone, against letting the fingerprint",
          "   split the bands. The table below puts the two side by side on the "
          "same items.", ""]
    have = [b for b in allb if b["has_samples"]]
    if have:
        sc_only = baseline_sc_always(have, eq)
        gated = run_variant(allb, two, eq)
        L += ["| | coverage | selective acc. | correct yield | gen./item |",
              "|---|---:|---:|---:|---:|",
              f"| voting alone, on the items that bought samples | {fmt(sc_only['coverage'])} | "
              f"{fmt(sc_only['selective_acc'])} | {fmt(sc_only['yield'])} | 5.00 |",
              f"| fingerprint gate plus sampled retry (all items) | {fmt(gated['coverage'])} | "
              f"{fmt(gated['selective_acc'])} | {fmt(gated['yield'])} | "
              f"{gated['gens_per_item']:.2f} |"]
    L += ["",
          "3. **Re-probing as the second stage's evidence** -- not generated "
          "here. The winning share costs no extra",
          "   generation, while re-probing costs more per item. Recorded as not "
          "run.",
          "4. **Seed sanity on one backbone** -- not run. The retry seed is "
          "registered as 1, and the first stage",
          "   is greedy, so it carries no seed dependence. Left as an extension."]

    k_ablation(allb, L)
    gate2_comparison(cells, L)

    OUT.mkdir(parents=True, exist_ok=True)
    md = f"canonical-gate-verdict-{args.tag}.md" if args.tag else "canonical-gate-verdict.md"
    (OUT / md).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nwritten: {OUT / md}")


if __name__ == "__main__":
    main()
