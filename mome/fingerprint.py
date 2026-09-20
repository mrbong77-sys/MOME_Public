#!/usr/bin/env python3
"""Assemble the metamorphic fingerprint of each item.

Pairs the original answer with the probe answers by item_id, reads a handful
of label-free observation bits per probe, and folds them into one fingerprint
row per item.

    python3 fingerprint.py --all            # every cell that has probe records
    python3 mome/fingerprint.py --bench gsm8k --backbone gemma4-e2b

Inputs, all read-only: the ungated cell's records and grades, the probe
cell's records, and the probe manifest that states each probe's expected
response.  Output: one JSONL row per item.

Whether the original answer was correct travels along as the training label
only.  It enters no observation bit, and probe answers are never graded --
that is what makes the fingerprint label-free at inference.

## Observation bits, per probe

- answered      an answer could be extracted from the probe response
- changed       the probe answer differs from the original, after normalizing
- dir_ok        the change agrees with the rule's direction note (numeric only)
- mag_ok        the answer scaled as the factor note said (a bonus bit; no
                claim of linearity is made)
- hold_ok       a MUST_HOLD probe kept the answer
- change_violation  MUST_CHANGE, yet the answer did not move -- spurious
                consistency
- hold_violation    MUST_HOLD, yet the answer moved -- sensitivity to surface
                form
- text_sim      string similarity of the two derivations, a copy signal
- len_ratio     ratio of generated token counts

Answer normalization follows the same convention as the generation records:
numeric answers reuse `normalize_numeric` from the campaign code.  Expression
answers are compared as strings with whitespace and dollar signs stripped,
falling back to numeric comparison when both read as numbers; the resulting
file records in its metadata that this is weaker than full symbolic
equivalence.
"""

from __future__ import annotations

import argparse
import difflib
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
import ollama_http as oh  # noqa: E402  (record reader: .gz and split-part fallbacks)
from extract import normalize_numeric  # noqa: E402  (same normalization as generation)

VANILLA_DIR = paths.UNGATED
PROBE_DATA_DIR = paths.PROBES
MANIFEST_DIR = paths.MANIFESTS
OUT_DIR = paths.FINGERPRINTS

BENCHES = ["gsm8k", "hrm8k-gsm8k-ko", "math500"]
BACKBONES = ["gemma4-e2b", "solar-10.7b", "qwen3.5-9b", "qwen2-math-1.5b", "gemma4-e4b"]
SIM_CAP = 4000  # cap on text length fed to the similarity ratio (long-tail cost)


# ---- Answer comparison ------------------------------------------------------

def norm_answer(answer: str | None, kind: str) -> str | None:
    if answer is None:
        return None
    s = str(answer).strip()
    if not s:
        return None
    if kind == "numeric":
        return normalize_numeric(s)
    cleaned = s.replace(" ", "").replace("$", "").replace("\\!", "").replace("\\,", "")
    numeric = normalize_numeric(s)
    try:
        float(numeric)
        return numeric
    except ValueError:
        return cleaned


#: Import the symbolic comparator once and reuse it, falling back to string
#: comparison when it is absent.  It catches notational equivalence between
#: expressions ('ab+5b+2a+10' == '(a+5)(b+2)') and so removes hold-violation
#: false positives.  Same version as the grader used for the campaigns.
try:
    from math_verify import parse as _mv_parse, verify as _mv_verify
    _MV = True
except ImportError:  # pragma: no cover
    _MV = False


from quiet_mp import drop_spawn_bootstrap_noise  # noqa: E402
from greedy_path import greedy_rows  # noqa: E402


def _bare(s: str) -> str:
    """Strip enclosing braces and dollar signs: `{(b+2)(a+5)}` -> `(b+2)(a+5)`."""
    s = str(s).strip()
    changed = True
    while changed and len(s) > 1:
        changed = False
        if s[0] == "{" and s[-1] == "}":
            s = s[1:-1].strip(); changed = True
        elif s[0] == "$" and s[-1] == "$":
            s = s[1:-1].strip(); changed = True
    return s


def _mv_equal(a: str, b: str) -> bool | None:
    """Do the two answers denote the same value?  None when undecidable.

    Wrapping the operands is not optional.  Handed a bare string with no LaTeX
    anchor, the comparator's `parse` does not read an expression at all -- it
    picks out the last number:

        parse("(a + 5)(b + 2)")          -> [2]          (not the expression)
        parse("6r^2 + -4r + -24")        -> [-24]
        parse("2 \\cdot 3 \\cdot 4 \\cdot 5 \\cdot 1") -> [1]

    Comparing in that state calls two different expressions equivalent merely
    because their trailing numbers agree, and calls two equal expressions
    different when those numbers do not.  Wrapping in `$...$` sends the whole
    expression through the LaTeX extractor and both failures disappear:

        ("(a + 5)(b + 2)", "{(b + 2)(a + 5)}")   False bare -> True wrapped
        ("6r^2 + -4r + -24", "6r^2 - 4r - 24")   False -> True
        ("x^2+2x+1", "(x+1)^2")                  False -> True
        ("2·3·4·5·1", "2·3·4·6")                 False -> False  (still unequal)
        ("0.01171875", "0.03515625")             False -> False

    Strategies are tried in order, and the verdict of the first strategy that
    parses BOTH operands is returned as it stands.  Or-ing the strategies
    together would bring the trailing-number coincidence back as a false
    equivalence.
    """
    if not _MV:
        return None
    for wrap in (lambda s: f"${_bare(s)}$", lambda s: str(s)):
        try:
            pa, pb = _mv_parse(wrap(a)), _mv_parse(wrap(b))
        except Exception:  # noqa: BLE001 -- unparsable this way; try the next
            continue
        if not pa or not pb:
            continue
        try:
            return bool(_mv_verify(pa, pb))
        except Exception:  # noqa: BLE001
            continue
    return None


def answers_equal(a: str | None, b: str | None, kind: str) -> bool | None:
    na, nb = norm_answer(a, kind), norm_answer(b, kind)
    if na is None or nb is None:
        return None
    if na == nb:
        return True
    if kind == "expression":
        # Different as strings: ask the symbolic comparator, which absorbs
        # differences of notation.
        mv = _mv_equal(str(a), str(b))
        if mv is not None:
            return mv
    return False


def _as_float(normalized: str | None) -> float | None:
    if normalized is None:
        return None
    try:
        return float(normalized)
    except ValueError:
        return None


# ---- Probe observation ------------------------------------------------------

def observe(orig: dict, probe: dict, meta: dict, kind: str) -> dict:
    """Observation bits for one probe.

    `orig` and `probe` are generation record rows; `meta` is the probe's
    manifest entry, which carries its expected response.
    """
    orig_ans, probe_ans = orig.get("extracted_answer"), probe.get("extracted_answer")
    equal = answers_equal(orig_ans, probe_ans, kind)
    changed = None if equal is None else not equal

    expectation = meta["expectation"]
    obs = {
        "probe_id": meta["probe_id"],
        "rule": meta["rule"],
        "expectation": expectation,
        "confidence": meta["confidence"],
        "answered": probe_ans is not None and str(probe_ans).strip() != "",
        "probe_truncated": bool(probe.get("truncated")),
        "changed": changed,
        "dir_ok": None,
        "mag_ok": None,
        "hold_ok": None,
        "change_violation": None,
        "hold_violation": None,
        "probe_answer": probe_ans,
    }

    if expectation == "MUST_CHANGE":
        obs["change_violation"] = None if changed is None else not changed
        ov = _as_float(norm_answer(orig_ans, kind))
        pv = _as_float(norm_answer(probe_ans, kind))
        direction, scale = meta.get("direction"), meta.get("scale")
        if changed and ov is not None and pv is not None:
            if direction in ("increase", "decrease"):
                delta = pv - ov
                obs["dir_ok"] = delta > 0 if direction == "increase" else delta < 0
            if scale and ov != 0:
                obs["mag_ok"] = abs(pv - ov * scale) <= 1e-6 * max(1.0, abs(ov * scale))
    else:  # MUST_HOLD
        obs["hold_ok"] = equal
        obs["hold_violation"] = None if equal is None else not equal

    o_text = (orig.get("text_visible") or orig.get("response_text") or "")[:SIM_CAP]
    p_text = (probe.get("text_visible") or probe.get("response_text") or "")[:SIM_CAP]
    if o_text and p_text:
        obs["text_sim"] = round(difflib.SequenceMatcher(None, o_text, p_text).ratio(), 4)
    else:
        obs["text_sim"] = None
    o_tok, p_tok = orig.get("token_count") or 0, probe.get("token_count") or 0
    obs["len_ratio"] = round(p_tok / o_tok, 4) if o_tok else None
    return obs


def item_fingerprint(item_id: str, orig: dict, grade_row: dict | None,
                     probe_obs: list[dict]) -> dict:
    """One item's fingerprint.  `correct` is a label column only; no
    observation is computed from it."""
    change = [o for o in probe_obs if o["expectation"] == "MUST_CHANGE"]
    hold = [o for o in probe_obs if o["expectation"] == "MUST_HOLD"]
    return {
        "item_id": item_id,
        "orig_answer": orig.get("extracted_answer"),
        "orig_truncated": bool(orig.get("truncated")),
        "orig_token_count": orig.get("token_count"),
        "label_correct": None if grade_row is None else bool(grade_row["correct"]),
        "probes": probe_obs,
        "summary": {
            "n_probes": len(probe_obs),
            "n_change": len(change),
            "n_hold": len(hold),
            "n_unanswered": sum(1 for o in probe_obs if not o["answered"]),
            "spurious_consistency": any(o["change_violation"] is True for o in change),
            "hold_instability": any(o["hold_violation"] is True for o in hold),
            "change_confirmed": any(o["changed"] is True for o in change),
            "max_text_sim": max((o["text_sim"] for o in probe_obs
                                 if o["text_sim"] is not None), default=None),
        },
    }


# ---- Cell assembly ----------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    # Plain, .gz and split (records_p*.jsonl.gz + records_parts.json) layouts
    # all go through the runner's own reader.  A private loader here once
    # missed that cells over 90 MiB are committed split, and reported a cell
    # with full records as having none.
    return oh.read_jsonl(path.with_suffix("") if path.suffix == ".gz" else path)


def load_records(cell_dir: Path) -> dict[str, dict]:
    rows = load_jsonl(cell_dir / "records.jsonl")
    if not rows:
        raise FileNotFoundError(cell_dir / "records.jsonl")
    #: When a cell holds several paths per item, keep only the greedy one.
    #: Overwriting blindly would leave the last path, a sample, standing in as
    #: the fingerprint's "original answer" (see greedy_path.py).
    rows = greedy_rows([r for r in rows if not r.get("error")], cell_dir.name)
    return {rec["item_id"]: rec for rec in rows}


def cell_paths(bench: str, backbone: str, tag: str = "") -> tuple[Path, Path, str]:
    """(ungated cell, probe cell, fingerprint file name).

    `tag` selects a variant arm.  The empty tag is the main configuration: the
    15 committed cells were assembled through this path and their names must
    not move.  `tag="b4096"` is the larger token-budget arm, deliberately
    named differently so that both sit side by side.
    """
    if tag:
        return (VANILLA_DIR / f"{bench}_{backbone}_{tag}_cb_vanilla",
                PROBE_DATA_DIR / f"{bench}_{backbone}_ccf_probes_{tag}",
                f"{bench}_{backbone}_{tag}")
    vanilla = VANILLA_DIR / f"{bench}_{backbone}_smoking_cb_vanilla"
    if not vanilla.exists():  # the extension cells carry a different suffix
        vanilla = VANILLA_DIR / f"{bench}_{backbone}_ext_cb_vanilla"
    return (vanilla, PROBE_DATA_DIR / f"{bench}_{backbone}_ccf_probes_v1",
            f"{bench}_{backbone}")


def build_cell(bench: str, backbone: str, tag: str = "") -> dict | None:
    vanilla, probe_cell, out_name = cell_paths(bench, backbone, tag)
    if not oh.jsonl_sources(probe_cell / "records.jsonl"):
        return None  # probes not generated yet
    manifest = json.loads((MANIFEST_DIR / f"{bench}_manifest.json").read_text(encoding="utf-8"))
    kind = "expression" if bench == "math500" else "numeric"

    orig_records = load_records(vanilla)
    grades = {g["item_id"]: g
              for g in greedy_rows(load_jsonl(vanilla / "grades.jsonl"), vanilla.name)}
    probe_records = load_records(probe_cell)

    by_item: dict[str, list[dict]] = {}
    missing_probe = 0
    for meta in manifest["probes"]:
        rec = probe_records.get(meta["probe_id"])
        if rec is None:
            missing_probe += 1
            continue
        orig = orig_records.get(meta["item_id"])
        if orig is None:
            continue
        by_item.setdefault(meta["item_id"], []).append(observe(orig, rec, meta, kind))

    rows = [item_fingerprint(i, orig_records[i], grades.get(i), obs)
            for i, obs in sorted(by_item.items())]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{out_name}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({
            "_meta": {
                "benchmark": bench, "backbone": backbone, "answer_kind": kind,
                "vanilla_cell": vanilla.name, "probe_cell": probe_cell.name,
                "n_items": len(rows), "n_probes_manifest": manifest["n_probes"],
                "n_probes_missing": missing_probe,
                "orig_missing": len({p["item_id"] for p in manifest["probes"]}
                                    - set(orig_records)),
                "equality_note": ("expression comparison is string normalization, "
                                   "not full symbolic equivalence; under that limit "
                                   "`changed` can be overcounted, since a difference "
                                   "of notation reads as a change"),
            }}, ensure_ascii=False) + "\n")
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n = len(rows)
    sc = sum(r["summary"]["spurious_consistency"] for r in rows)
    hi = sum(r["summary"]["hold_instability"] for r in rows)
    wrong = [r for r in rows if r["label_correct"] is False]
    sc_wrong = sum(r["summary"]["spurious_consistency"] for r in wrong)
    return {"bench": bench, "backbone": backbone, "items": n,
            "spurious": sc, "hold_unstable": hi,
            "wrong": len(wrong), "spurious_among_wrong": sc_wrong,
            "out": str(out_path.relative_to(paths.ROOT) if out_path.is_relative_to(paths.ROOT) else out_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--bench", choices=BENCHES)
    parser.add_argument("--backbone", choices=BACKBONES)
    parser.add_argument("--tag", default="",
                        help="variant tag. 'b4096' is the larger token-budget "
                             "arm; the default (empty) is the main configuration.")
    args = parser.parse_args()
    combos = ([(b, k) for k in BACKBONES for b in BENCHES] if args.all
              else [(args.bench, args.backbone)])
    if not args.all and (not args.bench or not args.backbone):
        parser.error("pass --all, or both --bench and --backbone")

    built = 0
    #: On Windows the comparator's timeout spawns a child process per
    #: comparison, and some of those die during bootstrap with a traceback.
    #: Results are unaffected, but a log buried under them hides real errors.
    #: Only that block is filtered; no verdict, timeout or result is touched.
    with drop_spawn_bootstrap_noise():
        built = _build_all(combos, args)
    if built == 0:
        print("no cell assembled. Run the probe campaign (run_probes.py) first.")


def _build_all(combos, args) -> int:
    built = 0
    for bench, backbone in combos:
        result = build_cell(bench, backbone, args.tag)
        if result is None:
            print(f"{bench} x {backbone}: no probe records yet")
            continue
        built += 1
        print(f"{bench} × {backbone}{'/' + args.tag if args.tag else ''}: "
              f"items {result['items']} | "
              f"spurious consistency {result['spurious']} ("
              f"{result['spurious_among_wrong']} of {result['wrong']} wrong) | "
              f"hold violations {result['hold_unstable']} "
              f"-> {result['out']}")
    return built


if __name__ == "__main__":
    main()
