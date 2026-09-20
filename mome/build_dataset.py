#!/usr/bin/env python3
"""Assemble the per-item feature matrix the score model is fitted on.

Joins each item's fingerprint with the free signals already present in its
ungated record, and lays all 15 cells out as one flat table.  Each row holds:

  cell, benchmark, backbone, item_id, label (was the ungated answer correct)
  symbolic features: the answer-level observations aggregated over the probes
  free signals: mean_logprob, truncated, hidden_token
  text features: excerpts of the original and probe responses, written to a
  separate file for an optional embedding channel

    python3 build_dataset.py         # write the dataset
    python3 build_dataset.py --stats # summary only

The ungated records are read-only.  The only label is whether the original
answer was correct; probe answers are never graded.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import paths
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ollama_http as oh  # noqa: E402
from greedy_path import greedy_rows  # noqa: E402
from fingerprint import BACKBONES, BENCHES, VANILLA_DIR  # noqa: E402

FP_DIR = paths.FINGERPRINTS
OUT_DIR = paths.RESULTS

#: Rule group -> feature-name prefix.  Individual rules are too sparse to
#: carry a column of their own, so they are pooled by group.
RULE_GROUPS = {
    "flip": "flip",       # comparison, extremum and sign flips (high conf.)
    "scale": "scale",     # numeric and word-number scaling (medium conf.)
    "hold": "hold",       # redundant clause, answer must hold
}


def rule_group(rule: str) -> str:
    for key, name in RULE_GROUPS.items():
        if key in rule:
            return name
    return "other"


def mean_logprob(rec: dict) -> float | None:
    entries = rec.get("logprobs")
    if not entries:
        return None
    vals = [e.get("logprob") for e in entries
            if isinstance(e, dict) and e.get("logprob") is not None]
    return sum(vals) / len(vals) if vals else None


def hidden_tokens(rec: dict) -> int:
    """Tokens counted by the runtime but present in neither the text nor the
    log-probabilities: the model's hidden reasoning tokens."""
    ec = (rec.get("ollama") or {}).get("eval_count")
    n = rec.get("n_logprob_tokens")
    if ec is None or n is None:
        return 0
    reason = rec.get("done_reason")
    expected = 0 if reason == "length" else 1
    return max(0, (ec - n) - expected)


def vanilla_signals(cell_dir: Path) -> dict[str, dict]:
    """item_id -> {mean_logprob, truncated, hidden_token, token_count}.

    When a cell holds several paths per item, only the greedy one is used
    (see greedy_path.py).  On a single-path cell nothing changes.
    """
    out = {}
    rows = [r for r in oh.read_jsonl(cell_dir / "records.jsonl") if not r.get("error")]
    for rec in greedy_rows(rows, cell_dir.name):
        if rec.get("error"):
            continue
        out[rec["item_id"]] = {
            "mean_logprob": mean_logprob(rec),
            "truncated": bool(rec.get("truncated")) or rec.get("done_reason") == "length",
            "hidden_token": hidden_tokens(rec),
            "token_count": rec.get("token_count") or 0,
            "orig_text": (rec.get("text_visible") or rec.get("response_text") or ""),
        }
    return out


def symbolic_features(fp: dict) -> dict:
    """One item's fingerprint -> its symbolic features.  Neither the label nor
    any correct answer is read."""
    probes = fp["probes"]
    s = fp["summary"]
    feats = {
        "n_probes": s["n_probes"],
        "n_unanswered": s["n_unanswered"],
        "frac_unanswered": s["n_unanswered"] / max(s["n_probes"], 1),
        "spurious_consistency": int(s["spurious_consistency"]),
        "hold_instability": int(s["hold_instability"]),
        "change_confirmed": int(s["change_confirmed"]),
        "max_text_sim": s["max_text_sim"] if s["max_text_sim"] is not None else 0.0,
        "orig_truncated": int(fp["orig_truncated"]),
    }
    # Violations and changes, aggregated per rule group.
    for grp in ("flip", "scale", "hold"):
        gp = [p for p in probes if rule_group(p["rule"]) == grp]
        feats[f"{grp}_n"] = len(gp)
        if grp == "hold":
            feats["hold_violation_any"] = int(any(p.get("hold_violation") is True for p in gp))
            feats["hold_violation_frac"] = (
                sum(1 for p in gp if p.get("hold_violation") is True) / max(len(gp), 1))
        else:
            feats[f"{grp}_change_viol_any"] = int(any(p.get("change_violation") is True for p in gp))
            feats[f"{grp}_changed_frac"] = (
                sum(1 for p in gp if p.get("changed") is True) / max(len(gp), 1))
            feats[f"{grp}_dir_ok_frac"] = (
                sum(1 for p in gp if p.get("dir_ok") is True) / max(len(gp), 1))
    # Text similarity and length ratio: how much the derivation was copied.
    sims = [p["text_sim"] for p in probes if p.get("text_sim") is not None]
    lens = [p["len_ratio"] for p in probes if p.get("len_ratio") is not None]
    feats["mean_text_sim"] = sum(sims) / len(sims) if sims else 0.0
    feats["min_len_ratio"] = min(lens) if lens else 1.0
    feats["max_len_ratio"] = max(lens) if lens else 1.0
    return feats


def build(tag: str = "") -> tuple[list[dict], list[dict]]:
    """`tag` selects a variant arm; empty is the main 15-cell configuration."""
    rows, texts = [], []
    suffix = f"_{tag}" if tag else ""
    for backbone in BACKBONES:
        for bench in BENCHES:
            fp_path = FP_DIR / f"{bench}_{backbone}{suffix}.jsonl"
            if not fp_path.exists():
                print(f"  skipped, no fingerprint: {bench} x {backbone}", file=sys.stderr)
                continue
            if tag:
                vcell = VANILLA_DIR / f"{bench}_{backbone}_{tag}_cb_vanilla"
            else:
                vcell = VANILLA_DIR / f"{bench}_{backbone}_smoking_cb_vanilla"
                if not vcell.exists():
                    vcell = VANILLA_DIR / f"{bench}_{backbone}_ext_cb_vanilla"
            sig = vanilla_signals(vcell)
            fp_rows = [json.loads(l) for l in fp_path.read_text(encoding="utf-8").splitlines()][1:]
            for fp in fp_rows:
                if fp["label_correct"] is None:
                    continue
                iid = fp["item_id"]
                s = sig.get(iid, {})
                feats = symbolic_features(fp)
                feats.update({
                    "nc_mean_logprob": s.get("mean_logprob"),
                    "nc_truncated": int(s.get("truncated", False)),
                    "nc_hidden_token": s.get("hidden_token", 0),
                    "nc_token_count": s.get("token_count", 0),
                })
                #: The tag goes into the cell name so the arms do not mix, but
                #: `backbone` is left alone: the frozen predictor and the
                #: registered thresholds are looked up by backbone name, and
                #: the same gate under a different budget is the point of the
                #: variant arm.
                rows.append({
                    "cell": f"{bench}_{backbone}{suffix}", "benchmark": bench,
                    "backbone": backbone,
                    "item_id": iid, "label": int(fp["label_correct"]),  # 1 = correct
                    "features": feats,
                })
                texts.append({
                    "cell": f"{bench}_{backbone}{suffix}", "item_id": iid,
                    "orig_tail": s.get("orig_text", "")[-700:],
                    "probes": [{"rule": p["rule"], "expectation": p["expectation"],
                                "note": p.get("note", ""),
                                "probe_tail": (p.get("probe_answer") or "")}
                               for p in fp["probes"]],
                })
    return rows, texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--tag", default="",
                    help="tag for a variant arm, e.g. b4096. Output goes to "
                         "dataset_<tag>.jsonl and never overwrites the main "
                         "dataset.")
    args = ap.parse_args()
    rows, texts = build(args.tag)
    ds_name = f"fingerprints_{args.tag}.jsonl" if args.tag else "fingerprints.jsonl"
    tx_name = f"text_{args.tag}.jsonl.gz" if args.tag else "text.jsonl.gz"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.stats:
        (OUT_DIR / ds_name).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        with gzip.open(OUT_DIR / tx_name, "wt", encoding="utf-8") as f:
            for t in texts:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        print(f"written: {OUT_DIR / ds_name} ({len(rows)} rows), {OUT_DIR / tx_name}")
    from collections import Counter
    cells = Counter(r["cell"] for r in rows)
    print(f"cells {len(cells)}, items {len(rows)}, features {len(rows[0]['features'])} each")
    pos = sum(r["label"] for r in rows)
    print(f"correct {pos} / wrong {len(rows)-pos} (error rate {1-pos/len(rows):.1%})")


if __name__ == "__main__":
    main()
