#!/usr/bin/env python3
"""MOME two-stage gate -- the decision module (standard library only, portable).

The frozen predictor `model/s_v1.json` (standardize -> logistic ->
Platt, no neural network) turns a counterfactual fingerprint into c_safe, and
the pre-registered three-band policy turns c_safe into a decision.

    c_safe >= tau_high         -> SERVE    (cleared at stage 1, low-cost path)
    tau_low <= c_safe < tau_high -> RETRY  (self-consistency SC@4 -> stage 2)
    c_safe < tau_low           -> DECLINE  (abstain immediately)

Stage 2, after the retry: serve when the most common answer holds at least
VOTE_SERVE of the vote, otherwise decline.  The vote is **the existing greedy
answer plus the 4 newly sampled ones = 5 votes**, the same shape as the legacy
pool5 policy (only 4 generations are bought).  The samples-only vote is
recorded alongside it as an ablation.

No gold answer and no grader is read anywhere in this file -- the gate is
label-free.  Every string this module emits is English and meant to be shown to
a person: a terminal safety device has to be able to say why it refused.

    python3 mome/gate.py --selftest
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import paths

MODEL_PATH = paths.MODEL_DEPLOYED

SERVE, RETRY, DECLINE = "SERVE", "RETRY", "DECLINE"

#: Stage 2: serve when the most common of the 5 votes holds at least this
#: share.  3/5 = 0.6 is the smallest working majority, and it is the same axis
#: the legacy pool5 majority vote used.
VOTE_SERVE = 0.6


def load_model(path: Path = MODEL_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def c_safe(features: dict, model: dict) -> float:
    """Fingerprint features -> calibrated probability that the answer is safe.

    Pure Python on purpose: the deployed gate must run without numpy, sklearn
    or torch.

    `platt_a` / `platt_b` are optional.  The shipped predictor carries them and
    is calibrated; the held-out (backbone-out) predictors the study scores with
    do not, and their thresholds were chosen on that same uncalibrated scale.
    Applying a calibration layer a threshold was not chosen under moves the
    bands, so a model without the two keys is used raw.
    """
    z = model["intercept"]
    for name, mean, scale, coef in zip(model["feature_names"], model["mean"],
                                       model["scale"], model["coef"]):
        v = features.get(name)
        v = 0.0 if v is None else float(v)
        z += coef * ((v - mean) / (scale if scale else 1.0))
    raw = 1.0 / (1.0 + math.exp(-z))                      # logistic output
    if model.get("platt_a") is None or model.get("platt_b") is None:
        return raw
    zz = model["platt_a"] * raw + model["platt_b"]        # Platt calibration
    return 1.0 / (1.0 + math.exp(-zz))


def stage1(features: dict, model: dict, tau_high: float, tau_low: float) -> dict:
    """Stage 1 gate.  Returns {decision, c_safe, reason}."""
    p = c_safe(features, model)
    if p >= tau_high:
        d = SERVE
    elif p < tau_low:
        d = DECLINE
    else:
        d = RETRY
    return {"decision": d, "c_safe": round(p, 4), "reason": reason_text(features, p, d)}


def vote(answers: list, same=None) -> tuple[object, float]:
    """The most common answer and its share.  Empty answers (None/'') do not vote.

    `same(a, b)` decides when two answers are the same answer.  The default is
    exact string equality, which keeps this module free of dependencies.  That
    default splits a vote across notations of one value -- 0.5 / 1/2 /
    \\frac{1}{2} are three votes for the same number -- and on the committed
    MATH-500 pool cells that fragmentation throws away correct answers at a
    measurable rate (serve 59.3% -> 71.4% once the votes are merged, with the
    served answers' accuracy barely moving).  A caller that has a symbolic
    comparator available should pass it in.
    """
    valid = [a for a in answers if a is not None and str(a).strip() != ""]
    if not valid:
        return None, 0.0
    if same is None:
        counts: dict = {}
        for a in valid:
            key = str(a).strip()
            counts[key] = counts.get(key, 0) + 1
        best = max(counts.items(), key=lambda kv: kv[1])
        return best[0], best[1] / len(valid)
    clusters: list[list] = []
    for a in valid:
        for c in clusters:
            if same(a, c[0]):
                c.append(a)
                break
        else:
            clusters.append([a])
    best_c = max(clusters, key=len)
    return best_c[0], len(best_c) / len(valid)


def stage2(greedy_answer, sampled_answers: list, same=None) -> dict:
    """Stage 2 gate.

    The decision reads the greedy answer together with the samples (the legacy
    pool5 shape).  The samples-only vote is computed as well and reported, but
    it decides nothing -- it is the registered ablation.  `same` is the answer
    equivalence used for counting votes (see `vote`).
    """
    pool5_ans, pool5_share = vote([greedy_answer] + list(sampled_answers), same)
    s_ans, s_share = vote(list(sampled_answers), same)
    decision = SERVE if pool5_share >= VOTE_SERVE else DECLINE
    return {
        "decision": decision,
        "answer": pool5_ans if decision == SERVE else None,
        "vote_share": round(pool5_share, 4),
        "vote_share_samples_only": round(s_share, 4),
        "answer_samples_only": s_ans,
        "reason": (f"On retry the most common answer held {pool5_share:.0%} of 5 votes -- "
                   + ("a working majority, so this answer is served."
                      if decision == SERVE
                      else "agreement is too weak, so no answer is given.")),
    }


def reason_text(f: dict, p: float, decision: str) -> str:
    """The human-readable justification: which observation drove the decision."""
    bits = []
    if f.get("orig_truncated"):
        bits.append("the solution was cut off at the token limit")
    if f.get("hold_violation_any"):
        bits.append("the answer changed under a restatement that kept the meaning (unstable)")
    if f.get("flip_change_viol_any"):
        bits.append("the answer did not change when the comparison was reversed "
                    "(the condition was ignored)")
    if f.get("scale_change_viol_any"):
        bits.append("the answer did not change when a quantity was changed "
                    "(the condition was ignored)")
    if f.get("frac_unanswered", 0) > 0:
        bits.append("some of the perturbed versions got no answer at all")
    if not bits:
        bits.append("the perturbed versions were answered the way they should be")
    head = {SERVE: "Answering",
            RETRY: "Checking this once more",
            DECLINE: "Not answering"}[decision]
    return f"{head} (confidence {p:.2f}): " + ", ".join(bits) + "."


def _selftest() -> None:
    model = load_model()
    names = model["feature_names"]
    healthy = {n: 0.0 for n in names}
    healthy.update({"change_confirmed": 1, "flip_changed_frac": 1.0, "scale_changed_frac": 1.0,
                    "hold_n": 1, "flip_n": 1, "n_probes": 2, "max_text_sim": 0.2})
    broken = {n: 0.0 for n in names}
    broken.update({"spurious_consistency": 1, "hold_instability": 1, "hold_violation_any": 1,
                   "hold_violation_frac": 1.0, "scale_change_viol_any": 1, "orig_truncated": 1,
                   "hold_n": 1, "flip_n": 1, "n_probes": 2, "max_text_sim": 0.95})
    ph, pb = c_safe(healthy, model), c_safe(broken, model)
    print(f"healthy fingerprint c_safe={ph:.3f} / broken fingerprint c_safe={pb:.3f}")
    assert ph > pb, "the healthy fingerprint must score higher"
    th, tl = 0.911, 0.525
    for f, tag in ((healthy, "healthy"), (broken, "broken")):
        r = stage1(f, model, th, tl)
        print(f"  [{tag}] {r['decision']}  {r['reason']}")
    s2 = stage2("42", ["42", "42", "17", "42"])
    print(f"  [stage2 agreed] {s2['decision']} share={s2['vote_share']} -- {s2['reason']}")
    assert s2["decision"] == SERVE and s2["answer"] == "42"
    s2b = stage2("42", ["17", "9", "3", "51"])
    print(f"  [stage2 split]  {s2b['decision']} share={s2b['vote_share']} -- {s2b['reason']}")
    assert s2b["decision"] == DECLINE and s2b["answer"] is None
    print("selftest OK")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    else:
        ap.print_help()
