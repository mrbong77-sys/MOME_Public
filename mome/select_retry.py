#!/usr/bin/env python3
"""Select the items the second stage will sample, and write its configs.

Applies the first stage -- the frozen backbone-out predictor together with
that backbone's registered thresholds -- to every item of all 15 cells, then
writes a sampling campaign for the items it did not serve immediately.  The
default scope is RETRY plus DECLINE, the wider of the two: buying samples for
the decline band as well lets both the two-band and the three-band variant be
read off the *same* data afterwards, where the narrow scope would allow only
one of them.

Thresholds are taken exactly as the threshold-selection step chose them on
the training side.  Nothing is re-chosen here.

    python3 select_retry.py            # write item lists and configs
    python3 select_retry.py --stats    # summary only

Outputs: the retry item ids per cell and the sampling config per cell (both
committed, since they are pre-registered), and the first-stage decision for
every item.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import paths
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fingerprint import BENCHES, VANILLA_DIR  # noqa: E402
from gate import DECLINE, RETRY, stage1  # noqa: E402

BACKBONE_OUT_MODELS = paths.MODEL_BACKBONE_OUT


def load_backbone_out_models() -> dict:
    if not BACKBONE_OUT_MODELS.exists():
        raise SystemExit("run export_backbone_out.py first, to freeze the "
                         "backbone-out predictors used for evaluation.")
    return json.loads(BACKBONE_OUT_MODELS.read_text(encoding="utf-8"))

P2_OUT = paths.MODEL
ITEMS_DIR = paths.RETRY_ITEMS
CONFIGS_DIR = paths.RETRY_CONFIGS
OUT = paths.RESULTS

#: The default sample count stays at 4: the committed arm was generated with
#: it, and changing the default would make that arm's configs impossible to
#: regenerate.  The adopted design is k=2, so a NEW arm is generated with
#: `-k 2`.  Reading k=2 needs no new generation at all -- the seeds are
#: attached to the paths in order, so taking the first two paths IS k=2.
K_SAMPLES, SC_TEMPERATURE, SEED_BASE = 4, 0.7, 0
#: The retry config is inherited from the configuration the cell's ungated
#: campaign actually ran.  Retyping it by hand would let one mismatched
#: prompt template, token budget or thinking setting slip through unnoticed,
#: and then the second stage's ballot -- the existing greedy vote plus the new
#: samples -- would not be a ballot cast under one condition.  Only the
#: sampling parameters and the names and paths are changed.
INHERIT = ("backbone", "model", "benchmark", "benchmark_path", "route", "language",
           "answer_kind", "prompt_template_id", "endpoint", "options", "seed",
           "logprobs", "top_logprobs", "think", "host", "timeout_seconds", "keep_alive",
           "anchor_item_check")


def vanilla_config(bench: str, backbone: str, tag: str = "") -> dict:
    """The configuration the cell's ungated campaign recorded as having run."""
    if tag:
        cell = VANILLA_DIR / f"{bench}_{backbone}_{tag}_cb_vanilla"
    else:
        cell = VANILLA_DIR / f"{bench}_{backbone}_smoking_cb_vanilla"
        if not (cell / "campaign_meta.json").exists():
            cell = VANILLA_DIR / f"{bench}_{backbone}_ext_cb_vanilla"
    return json.loads((cell / "campaign_meta.json").read_text(encoding="utf-8"))["config"]


def main() -> None:
    global K_SAMPLES
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--scope", choices=["retry", "retry+decline"], default="retry+decline",
                    help="which bands buy samples. The default includes the "
                         "decline band: on the stronger backbones the correct "
                         "answers a first-stage decline throws away outweigh "
                         "what the second stage loses. Buying the wider scope "
                         "keeps the narrow variant readable afterwards as a "
                         "subset; the reverse is not possible.")
    ap.add_argument("--tag", default="",
                    help="tag for a variant arm, e.g. b4096. Reads "
                         "dataset_<tag>.jsonl and writes "
                         "stage1_decisions_<tag>.jsonl, so the main arm's "
                         "decisions are never overwritten.")
    ap.add_argument("-k", "--k-samples", type=int, default=K_SAMPLES,
                    help="how many self-consistency samples the second stage "
                         "buys. The default 4 regenerates the committed arm; "
                         "the adopted design is 2. Choose an EVEN k so that "
                         "the ballot is odd: one greedy vote plus k samples.")
    args = ap.parse_args()
    K_SAMPLES = args.k_samples
    if K_SAMPLES % 2:
        raise SystemExit(f"k={K_SAMPLES} gives {K_SAMPLES + 1} votes, an even "
                         "ballot. A tie then falls below VOTE_SERVE=0.6 and "
                         "becomes a decline, which is a structural loss. Use "
                         "an even k.")
    wanted = {RETRY} if args.scope == "retry" else {RETRY, DECLINE}

    #: Selection and evaluation score with the predictor fitted WITHOUT this
    #: backbone.  The shipped predictor is fitted on everything and carries a
    #: calibration layer, so its scores sit on a different scale, while the
    #: thresholds were chosen on the backbone-out scale.  Mixing the two
    #: shifts the bands badly: serve drops from 31.8% to 17.2%.
    models = load_backbone_out_models()
    taus = json.loads(paths.THRESHOLDS.read_text(encoding="utf-8"))["per_backbone"]
    ds = (paths.RESULTS / f"fingerprints_{args.tag}.jsonl") if args.tag else paths.FEATURES
    if not ds.exists():
        raise SystemExit(f"{ds} is missing. Run "
                         f"python3 mome/build_dataset.py"
                         f"{' --tag ' + args.tag if args.tag else ''} first.")
    rows = [json.loads(l) for l in ds.read_text(encoding="utf-8").splitlines()]

    by_cell: dict[str, list[str]] = {}
    decisions = []
    for r in rows:
        t = taus[r["backbone"]]
        d = stage1(r["features"], models[r["backbone"]], t["tau_high"], t["tau_low"])
        decisions.append({"cell": r["cell"], "item_id": r["item_id"], "label": r["label"],
                          **d})
        if d["decision"] in wanted:
            by_cell.setdefault(r["cell"], []).append(r["item_id"])

    OUT.mkdir(parents=True, exist_ok=True)
    dec_name = f"stage1_decisions_{args.tag}.jsonl" if args.tag else "stage1_decisions.jsonl"
    (OUT / dec_name).write_text(
        "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in decisions), encoding="utf-8")

    total = 0
    if not args.stats:
        ITEMS_DIR.mkdir(parents=True, exist_ok=True)
        CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"scope: {args.scope}\n")
    print(f"{'cell':32}{'retry':>7}{'all':>7}{'share':>8}")
    for cell in sorted({r["cell"] for r in rows}):
        items = by_cell.get(cell, [])
        n_all = sum(1 for r in rows if r["cell"] == cell)
        total += len(items)
        print(f"{cell:32}{len(items):>7}{n_all:>7}{len(items)/n_all:>8.1%}")
        if args.stats:
            continue
        bench = next(b for b in BENCHES if cell.startswith(b + "_"))
        backbone = cell[len(bench) + 1:]
        if args.tag and backbone.endswith("_" + args.tag):
            backbone = backbone[: -(len(args.tag) + 1)]
        (ITEMS_DIR / f"{cell}.txt").write_text("".join(i + "\n" for i in items), encoding="utf-8")
        van = vanilla_config(bench, backbone, args.tag)
        name = f"{cell}_retry_sc{K_SAMPLES}"
        cfg = {"name": name}
        cfg.update({k: van[k] for k in INHERIT if k in van})
        cfg["items_file"] = f"mome/retry_items/{cell}.txt"
        cfg["out_dir"] = f"mome/data/retry/{name}"
        cfg["pool"] = {"n_samples": K_SAMPLES, "temperature": SC_TEMPERATURE, "top_p": 1.0,
                       "seed_base": SEED_BASE, "include_greedy": False,
                       "sample_template": "same"}
        cfg["inherited_from"] = {
            "cell": f"{bench}_{backbone}_(smoking|ext)_cb_vanilla -> its campaign_meta.json",
            "keys": list(INHERIT),
            "why": ("The second stage's ballot is the cell's existing greedy "
                    "record plus the samples bought here. Both sides must come "
                    "from the same prompt template, token budget, thinking "
                    "setting and endpoint for the ballot to be cast under one "
                    "condition, so the operating point is inherited from the "
                    "configuration that cell actually ran rather than retyped. "
                    "Only the sampling block, the item list and the output "
                    "path differ."),
        }
        cfg["purpose"] = (
            "Second-stage retry path. For every item the first stage did not "
            f"serve immediately (scope {args.scope}), generate "
            f"{K_SAMPLES} self-consistency samples (temperature {SC_TEMPERATURE}, "
            f"seeds {SEED_BASE}..{SEED_BASE + K_SAMPLES - 1}, one seed set). The "
            "greedy path is reused from the ungated cell and is not bought "
            "again (include_greedy=false, so path numbering starts at 1 and "
            "path 0 is the ungated cell's record). The ballot is the greedy "
            "answer plus the samples; the samples-only ballot is recorded as "
            "an ablation. The operating point, the thresholds and the decision "
            "rules were all registered before any of this was generated.")
        #: The item gate is inherited from the ungated cell, which admits
        #: only ids drawn from the committed benchmark order, plus one line
        #: saying why this cell is a subset of those 300 items. One backbone's
        #: ungated cell ran through a runner that has no item gate and so
        #: carries no such block; the same gate is therefore built here, from
        #: the same item source.
        if "anchor_item_check" not in cfg:
            cfg["anchor_item_check"] = {
                "scope": "benchmark_order",
                "order_file": f"data/items/{bench}.txt",
                "registered_in": "data/items/smoking_n300_provenance.json",
                "why": ("This cell's items are a subset of the 300 drawn with "
                        "a fixed seed from the committed 500-item benchmark "
                        "order, and those 300 are exactly the list this "
                        f"backbone's ungated cell used ({bench}). With no "
                        "separate anchor cell to check against, the gate is "
                        "narrowed to the committed benchmark order rather than "
                        "switched off."),
            }
        else:
            cfg["anchor_item_check"] = dict(cfg["anchor_item_check"])
        cfg["anchor_item_check"]["why"] += (
                " This retry cell runs only the subset of those 300 that the "
                f"first stage did not serve immediately (scope {args.scope}). "
                "Selection used the frozen backbone-out predictor and the "
                "thresholds chosen on the training side, and no correct answer "
                "was read.")
        (CONFIGS_DIR / f"{name}.json").write_text(
            json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\nretry items {total}, new generations {total * K_SAMPLES:,} "
          f"({K_SAMPLES} samples per item, greedy reused)")
    if not args.stats:
        print(f"written: {ITEMS_DIR}/, {CONFIGS_DIR}/, {OUT / dec_name}")


if __name__ == "__main__":
    main()
