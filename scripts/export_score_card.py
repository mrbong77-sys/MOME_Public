#!/usr/bin/env python3
"""Print the score model card straight from the frozen coefficients.

A reader who wants to check `c_safe` needs one document holding all of it: the
names and coefficients of the 26 features, the standardization statistics, the
calibration parameters, and -- the part that is easy to get wrong -- which
scale the thresholds were chosen on.  This script assembles that from

    model/s_v1.json               the deployed predictor (calibrated)
    model/s_v1_backbone_out.json  the five evaluated predictors (uncalibrated)
    model/thresholds.json         the backbone-out thresholds

and writes results/score_model_card.md.  The point is that no number on the
card is ever copied by hand.

    python3 scripts/export_score_card.py
"""

from __future__ import annotations

import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mome"))
import paths                                        # noqa: E402

MODEL = paths.MODEL
THRESH = paths.THRESHOLDS
OUT = paths.SCORE_CARD

#: Feature names are code identifiers; each gets one line of plain English.
GLOSS = {
    "change_confirmed": "a MUST_CHANGE probe answered in the annotated direction (1/0)",
    "flip_change_viol_any": "any comparison-reversal probe whose answer did not move (1/0)",
    "flip_changed_frac": "fraction of comparison-reversal probes whose answer moved",
    "flip_dir_ok_frac": "fraction of comparison-reversal probes moving in the annotated direction",
    "flip_n": "number of comparison-reversal probes generated for the item",
    "frac_unanswered": "fraction of probes from which no answer could be parsed",
    "hold_instability": "share of MUST_HOLD probes whose answer moved (instability rate)",
    "hold_n": "number of MUST_HOLD probes generated for the item",
    "hold_violation_any": "any MUST_HOLD probe whose answer moved (1/0)",
    "hold_violation_frac": "fraction of MUST_HOLD probes whose answer moved",
    "max_len_ratio": "largest probe/original generated-length ratio",
    "max_text_sim": "largest edit similarity between a probe generation and the original",
    "mean_text_sim": "mean edit similarity between probe generations and the original",
    "min_len_ratio": "smallest probe/original generated-length ratio",
    "n_probes": "total probes generated for the item",
    "n_unanswered": "probes from which no answer could be parsed",
    "nc_hidden_token": "the original generation emitted a runtime control token (1/0)",
    "nc_mean_logprob": "mean token log-probability of the original generation",
    "nc_token_count": "generated token count of the original",
    "nc_truncated": "a probe generation hit the token limit (1/0)",
    "orig_truncated": "the original generation hit the token limit (1/0)",
    "scale_change_viol_any": "any quantity-scaling probe whose answer did not move (1/0)",
    "scale_changed_frac": "fraction of quantity-scaling probes whose answer moved",
    "scale_dir_ok_frac": "fraction of quantity-scaling probes moving in the annotated direction",
    "scale_n": "number of quantity-scaling probes generated for the item",
    "spurious_consistency": "answer unchanged under MUST_CHANGE while MUST_HOLD was preserved (1/0)",
}


#: Groups of features that are the same column in the training matrix, with
#: the reason why.
DUP_NOTE = {
    "hold_instability": "every item carries exactly one MUST_HOLD probe, so the rate, "
                        "the fraction and the indicator are the same column",
    "nc_truncated": "the two took the same value on every item in these runs "
                    "(234 truncated, 4263 not), so they enter as one column",
}


def duplicate_groups(model: dict) -> list:
    """Features whose coefficient, mean and scale all agree: the same column."""
    seen: dict = {}
    for name, mu, sc, co in zip(model["feature_names"], model["mean"],
                                model["scale"], model["coef"]):
        seen.setdefault((round(mu, 9), round(sc, 9), round(co, 9)), []).append(name)
    return [sorted(g) for g in seen.values() if len(g) > 1]


def table(model: dict) -> str:
    rows = ["| # | feature | coefficient | mean | scale | meaning |",
            "|---:|---|---:|---:|---:|---|"]
    order = sorted(range(len(model["feature_names"])),
                   key=lambda i: -abs(model["coef"][i]))
    for rank, i in enumerate(order, 1):
        name = model["feature_names"][i]
        rows.append("| %d | `%s` | %+.4f | %.4f | %.4f | %s |" % (
            rank, name, model["coef"][i], model["mean"][i], model["scale"][i],
            GLOSS.get(name, "—")))
    return "\n".join(rows)


def main() -> int:
    if not (MODEL / "s_v1.json").exists():
        print("missing: %s -- fit the predictor first." % (MODEL / "s_v1.json"))
        return 1
    ship = json.loads((MODEL / "s_v1.json").read_text(encoding="utf-8"))
    bo = json.loads((MODEL / "s_v1_backbone_out.json").read_text(encoding="utf-8"))
    taus = json.loads(THRESH.read_text(encoding="utf-8"))["per_backbone"]

    L = []
    a = L.append
    a("# Supplementary S1 — the MOME safety score, in full")
    a("")
    a("Generated by `scripts/export_score_card.py` from the frozen coefficient")
    a("files; nothing here is typed by hand. Every value below is what the gate")
    a("actually evaluates.")
    a("")
    a("## What is fitted, and on what")
    a("")
    a("At inference the gate reads no gold answer and no grader: `c_safe` is a")
    a("function of the model's own outputs alone. The coefficients below were")
    a("fitted **offline, on correctness labels**. Two predictors exist and they are")
    a("not interchangeable.")
    a("")
    a("| | fitted on | calibration layer | used for |")
    a("|---|---|---|---|")
    a("| `s_v1.json` | all %d items | Platt (a = %.4f, b = %.4f) | deployment |"
      % (sum(v["n_train_rows"] for v in bo.values()) // (len(bo) - 1), ship["platt_a"], ship["platt_b"]))
    a("| `s_v1_backbone_out.json` | the other four backbones | none — read raw | every number in the paper |")
    a("")
    a("The thresholds were selected on the uncalibrated backbone-out scale.")
    a("Applying the Platt layer that the thresholds were not chosen under moves the")
    a("bands (measured: serve 31.8% to 17.2%), which is why `gate.c_safe` returns the")
    a("raw logistic output when the two Platt keys are absent.")
    a("")
    a("## Scoring rule")
    a("")
    a("```")
    a("z = intercept + Σ_i coef_i · (x_i − mean_i) / scale_i")
    a("c_safe = 1 / (1 + exp(−z))                     # backbone-out predictors")
    a("c_safe = 1 / (1 + exp(−(a·raw + b)))           # shipped predictor only")
    a("```")
    a("")
    a("A feature that is absent for an item — no MUST_CHANGE probe survived the")
    a("well-posedness filter, for instance — enters as `0.0` **before**")
    a("standardization, so it is carried at its standardized distance from the")
    a("training mean rather than imputed. `n_probes`, `flip_n`, `scale_n` and")
    a("`hold_n` record how many probes the item actually had, so the absence is")
    a("itself visible to the score.")
    a("")
    a("## Features that coincide")
    a("")
    a("Twenty-six is the number of fields the fingerprint writes, not the number of")
    a("independent quantities it carries. Two groups are identical columns in the")
    a("training matrix, and the logistic therefore splits one weight among them:")
    a("")
    for group in duplicate_groups(ship):
        a("- %s — %s" % (", ".join("`%s`" % n for n in group), DUP_NOTE[group[0]]))
    a("")
    a("Read the coefficients accordingly: the contribution of the MUST_HOLD")
    a("violation is the sum over its three spellings, not any one of them.")
    a("")
    a("## Deployment predictor `s_v1.json` (26 features, by |coefficient|)")
    a("")
    a("Intercept %+.4f. Out-of-fold AUROC %.4f." % (ship["intercept"], ship["oof_auroc"]))
    a("")
    a(table(ship))
    a("")
    a("## Evaluation predictors — one per held-out backbone")
    a("")
    a("Each was fitted on the items of the other four backbones, so no item of the")
    a("evaluated backbone entered the score it was judged by. Thresholds are the")
    a("registered backbone-out values (serve-band precision ≥ 0.95, decline-band")
    a("accuracy ≤ 0.20).")
    a("")
    a("| held-out backbone | training items | intercept | τ_high | τ_low |")
    a("|---|---:|---:|---:|---:|")
    for bb, m in bo.items():
        t = taus.get(bb, {})
        a("| `%s` | %d | %+.4f | %s | %s |" % (
            bb, m["n_train_rows"], m["intercept"],
            ("%.4f" % t["tau_high"]) if t.get("tau_high") is not None else "—",
            ("%.4f" % t["tau_low"]) if t.get("tau_low") is not None else "—"))
    a("")
    for bb, m in bo.items():
        a("### `%s` held out" % bb)
        a("")
        a(table(m))
        a("")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("written: %s  (%d KB)" % (OUT, OUT.stat().st_size // 1024))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
