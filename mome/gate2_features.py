#!/usr/bin/env python3
"""Second-stage features: everything the retry paths expose without a label.

The deployed second stage votes on one number, the share of the winning
answer.  The retry records hold considerably more than that, and all of it is
readable without touching a correct answer.  What is extracted here:

**From the answers** -- the winning share under equivalence, the winning share
under string identity (an ablation), the number of answer clusters, the
normalized entropy of the cluster sizes, whether the greedy answer falls in
the majority cluster, and how many paths produced no answer at all.

**From the generations** -- how many paths were truncated, the coefficient of
variation of the token counts, and the mean and standard deviation of the
per-path mean log-probability.  Together these say whether the samples spread
out in length and in confidence.

**From the derivations (optional)** -- mean and minimum pairwise embedding
similarity among the sampled derivations, similarity inside the majority
cluster, and similarity between the majority and the minority clusters.

Embeddings are optional: with `embed=None` only the answer and generation
features are produced, which is the configuration that runs with no second
network present.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate                                  # noqa: E402
from fingerprint import answers_equal        # noqa: E402

#: Cap on derivation length fed to the embedder.  The tail is kept rather
#: than the head, because the conclusion sits at the end.
MAX_CHARS = 2000
#: Answer and generation features only -- the configuration that needs no
#: embedding model at the endpoint.
SYMBOLIC = ["share_eq", "share_str", "n_clusters", "cluster_entropy", "greedy_in_majority",
            "n_unanswered", "n_truncated", "len_cv", "logprob_mean", "logprob_sd"]
#: With derivation embeddings as well.
NEURAL = ["sim_mean", "sim_min", "sim_within_win", "sim_win_vs_lose"]
ALL = SYMBOLIC + NEURAL


def same_value(kind: str):
    """Equality test for answers.  Symbolic, never embedding-based: whether two
    answers are the same value is a symbolic question."""
    return lambda a, b: bool(answers_equal(a, b, kind))


def clusters(answers: list, kind: str) -> list[list[int]]:
    """Group answers by equal value.  Returns lists of original positions."""
    same = same_value(kind)
    out: list[list[int]] = []
    reps: list = []
    for pos, a in enumerate(answers):
        if a is None or str(a).strip() == "":
            continue
        for rep, idx in zip(reps, out):
            if same(a, rep):
                idx.append(pos)
                break
        else:
            reps.append(a)
            out.append([pos])
    return out


def _cv(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    if m == 0:
        return 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var) / abs(m)


def _sd(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def stage2_features(paths: list[dict], kind: str, vectors=None) -> dict:
    """Retry paths (the greedy one is `paths[0]`) -> second-stage features.

    Each `paths[i]` carries {answer, token_count, truncated, mean_logprob,
    text}.  Passing `vectors`, unit embeddings in the same order, adds the
    derivation features.
    """
    answers = [p.get("answer") for p in paths]
    cl = clusters(answers, kind)
    n_valid = sum(len(c) for c in cl)
    win = max(cl, key=len) if cl else []
    lose = [i for i in range(len(paths)) if i not in win]

    share_eq = len(win) / n_valid if n_valid else 0.0
    _, share_str = gate.vote(answers)
    entropy = 0.0
    if n_valid:
        for c in cl:
            p = len(c) / n_valid
            entropy -= p * math.log(p)
        if len(cl) > 1:
            entropy /= math.log(len(cl))          # 0 = one cluster, 1 = even spread

    feats = {
        "share_eq": round(share_eq, 4),
        "share_str": round(share_str, 4),
        "n_clusters": len(cl),
        "cluster_entropy": round(entropy, 4),
        "greedy_in_majority": int(0 in win),
        "n_unanswered": len(paths) - n_valid,
        "n_truncated": sum(1 for p in paths if p.get("truncated")),
        "len_cv": round(_cv([p.get("token_count") or 0 for p in paths]), 4),
        "logprob_mean": round(sum(p.get("mean_logprob") or 0.0 for p in paths) / len(paths), 4),
        "logprob_sd": round(_sd([p.get("mean_logprob") for p in paths]), 4),
    }
    if vectors is None:
        return feats

    n = len(vectors)
    sims = [float(vectors[i] @ vectors[j]) for i in range(n) for j in range(i + 1, n)]
    within = [float(vectors[i] @ vectors[j])
              for a, i in enumerate(win) for j in win[a + 1:]]
    across = [float(vectors[i] @ vectors[j]) for i in win for j in lose]
    feats.update({
        "sim_mean": round(sum(sims) / len(sims), 4) if sims else 0.0,
        "sim_min": round(min(sims), 4) if sims else 0.0,
        "sim_within_win": round(sum(within) / len(within), 4) if within else 0.0,
        "sim_win_vs_lose": round(sum(across) / len(across), 4) if across else 0.0,
    })
    return feats


def text_of(rec: dict) -> str:
    for key in ("text_visible", "response_text", "response"):
        v = rec.get(key)
        if isinstance(v, str) and v.strip():
            return v[-MAX_CHARS:]
    return ""
