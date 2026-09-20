#!/usr/bin/env python3
"""What thinking setting did a committed campaign actually run under?

`num_predict` 2048 is the canonical budget because the deliverable must be a
realistically deployable adapter, and 2,048 tokens is a comfortable budget only
with thinking DISABLED.  That is not a stylistic preference: the I channel is
defined as proximity to the token budget, so if thinking were on, truncation
would be driven by the length of the model's *thinking*, and I would be
measuring the thinking budget instead of the difficulty of the derivation.
Under no-think, budget proximity means what we intend it to mean.

A setting that was used but not recorded is not provable after the fact, so this
module never guesses.  It reads what is on disk and answers with one of four
states, and `unknown` is a real answer, not a synonym for "off":

  `no_think`        `think: false` was sent and is recorded.  Proved.
  `thinking`        `think: true` was sent and is recorded.
  `not_applicable`  the option was not sent AND the recorded `/api/show`
                    capabilities do not list "thinking", so the model exposes no
                    thinking mode and there was nothing to disable.
  `unknown`         nothing bearing on it was recorded, or the option was not
                    sent to a model that DOES expose thinking (the server
                    default then applied and the record does not say what it
                    was).  Never read this as `no_think`.

Two independent places carry the evidence, and they are checked separately so a
disagreement is visible rather than averaged away:

  * `campaign_meta.json` -> `think_sent` (plus `model_capabilities`), written by
    `run_campaign.py` and `run_pool.py` before the first generation;
  * every `records.jsonl` row -> `request.think`, which `make_record` copies out
    of the payload that was actually sent (`run_campaign.make_record`).

`--scan-records` additionally reads the token stream.  `think: false` stops the
server RETURNING a `thinking` field, but it does not guarantee the model never
emits a thought block in band: a block that is emitted anyway still spends the
`num_predict` budget, which is precisely the contamination the setting exists to
prevent.  The scan reuses the committed `prompts.strip_think_blocks`, so it
recognises exactly the markers the rest of Phase 1 recognises, and it reports
both a closed block and one that ran off the end of the budget unclosed.

The same detector is what `run_campaign.make_record` calls live, so the runner
flags the condition on the record as `think_block_leaked` while the run is
happening instead of only after it.  `record_leak_state` / `count_leaks` read
that flag back, and they read the FLAG, never the stream: a campaign that
predates the field has no flag and is `unknown`, exactly as an absent
`think_sent` is.  Nothing anywhere excludes, drops, reweights or thresholds a
flagged row -- whether the I channel should is the owner's decision, to be taken
against canonical data, and it is not taken here.

READ ONLY.  Nothing here writes, and nothing here imports `mome`.

    python mome/think_check.py data/ungated/math500_gemma4-e2b_pool
    python mome/think_check.py --all
    python mome/think_check.py --all --scan-records --json

Exit code 0 = every campaign inspected reports `no_think` or `not_applicable`;
3 = at least one reports `thinking` or `unknown`, or meta and records disagree.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ollama_http as oh          # noqa: E402
import prompts                    # noqa: E402

NO_THINK = "no_think"
THINKING = "thinking"
NOT_APPLICABLE = "not_applicable"
UNKNOWN = "unknown"

#: the states that mean "the I channel measures derivation length, not thinking"
SETTLED = (NO_THINK, NOT_APPLICABLE)

#: where committed campaigns live, relative to the repository root
CAMPAIGN_ROOTS = ("data/phase1", "smoke")


# --- the tri-state, from one recorded flag -----------------------------------

def classify(flag, capabilities, flag_recorded: bool) -> tuple[str, str]:
    """(state, why) for one recorded `think` flag.

    `flag_recorded` is what separates "the option was deliberately not sent"
    (flag None, key present) from "this campaign predates the field" (key
    absent).  The second is `unknown` and must never be reported as `no_think`:
    we do not know it, and asserting it would be inventing a fact.
    """
    if not flag_recorded:
        return UNKNOWN, "no `think` was recorded at all; the setting is not provable from this campaign"
    if flag is True:
        return THINKING, "`think: true` was sent, so thinking tokens count against num_predict"
    if flag is False:
        return NO_THINK, "`think: false` was sent and recorded"
    # recorded as null: the option was not sent.  Whether that mattered depends
    # on whether the model had a thinking mode to leave switched on.
    if isinstance(capabilities, list):
        if "thinking" not in capabilities:
            return NOT_APPLICABLE, ("the option was not sent, and /api/show did not list a "
                                    "\"thinking\" capability, so there was no thinking mode to disable")
        return UNKNOWN, ("the option was not sent although /api/show DID list \"thinking\"; the "
                         "server default applied and nothing recorded what it was")
    return UNKNOWN, "the option was not sent and no /api/show capabilities were recorded"


def meta_status(meta: dict) -> tuple[str, str]:
    """The state `campaign_meta.json` proves, and why."""
    if not isinstance(meta, dict):
        return UNKNOWN, "campaign_meta.json is not an object"
    return classify(meta.get("think_sent"), meta.get("model_capabilities"),
                    "think_sent" in meta)


def record_status(rec: dict) -> tuple[str, str]:
    """The state one `records.jsonl` row proves, and why.

    A row carries no capabilities of its own, so a row whose `request` has no
    `think` key is `unknown` here even when the campaign's meta settles it; the
    campaign-level answer combines the two.
    """
    request = (rec or {}).get("request")
    if not isinstance(request, dict):
        return UNKNOWN, "the row has no `request` object"
    return classify(request.get("think"), None, "think" in request)


# --- the token stream: was a thought block emitted anyway? -------------------

def stream_text(rec: dict) -> str | None:
    """The generated tokens joined back into text, or None without logprobs."""
    lp = rec.get("logprobs")
    if not isinstance(lp, list) or not lp:
        return None
    return "".join(str(e.get("token") or "") for e in lp if isinstance(e, dict))


def thought_block_leaked(rec: dict) -> bool | None:
    """True when a thought block is present in the token stream anyway.

    None when the row carries no logprobs and the question cannot be answered.
    `strip_think_blocks` removes a closed block and also an unclosed one that
    runs to the end of the text, and otherwise returns the input stripped -- so
    an inequality here is exactly "a thought block was in the stream".
    """
    text = stream_text(rec)
    if text is None:
        return None
    return prompts.strip_think_blocks(text) != text.strip()


#: the field `run_campaign.make_record` writes with the answer above.  It is a
#: RECORD field, written live; nothing reads it to exclude, drop, reweight or
#: threshold anything, and no committed campaign is backfilled with it.  A
#: campaign that predates it simply does not have it, which is `unknown` -- the
#: same rule `think_sent` already lives under.
LEAK_FIELD = "think_block_leaked"

LEAKED = "leaked"
CLEAN = "clean"


def record_leak_state(rec: dict) -> str:
    """`leaked` / `clean` / `unknown` for one row, from the RECORDED flag only.

    This never re-derives the answer from the token stream: a row that does not
    carry the field is `unknown`, because the campaign that wrote it did not ask
    the question.  Absent is not false, and `null` (the runner looked and the
    row had no logprobs to look at) is not false either.
    """
    got = (rec or {}).get(LEAK_FIELD, "<absent>")
    if got is True:
        return LEAKED
    if got is False:
        return CLEAN
    return UNKNOWN


def count_leaks(records) -> dict[str, int]:
    """How many rows leaked, how many are clean, how many cannot say.

    A count, never a filter.  The caller prints it; deciding what a leaked row
    means for the I channel is the owner's call, to be made against canonical
    data, and no code in this repository makes it.
    """
    out = {LEAKED: 0, CLEAN: 0, UNKNOWN: 0}
    for rec in records:
        out[record_leak_state(rec)] += 1
    return out


def leak_note(counts: dict[str, int]) -> str:
    """One human line for a run's status table and the runner's stdout."""
    n, total = counts[LEAKED], sum(counts.values())
    unknown = counts[UNKNOWN]
    if not total:
        return "thought-block leak: no rows read"
    head = (f"thought-block leak: {n} of {total} row(s) carry a `<|channel>thought` block "
            f"despite `think: false`" if n else
            f"thought-block leak: none of {total} row(s)")
    if unknown:
        head += f"; {unknown} row(s) unknown (no flag recorded)"
    return head + ". FLAGGED ONLY -- nothing is excluded or reweighted."


# --- a whole campaign --------------------------------------------------------

def iter_records(records_path: str | Path):
    """Every record row, one at a time, from whichever layout is on disk.

    `oh.read_jsonl` would return the whole campaign as a list, and a 500-item
    pool at `top_logprobs` 10 is gigabytes of logprob objects once parsed -- so
    this streams instead, over exactly the files `oh.jsonl_sources` says the
    readers use (the plain file, the single `.gz`, or the per-path set).

    Nothing here needs `records_parts.json`: every answer this module gives is a
    COUNT over rows, and a count does not depend on the order the runner wrote
    them in.  Anything that does depend on that order must use `oh.read_jsonl`.
    """
    for src in oh.jsonl_sources(records_path):
        with oh.open_jsonl(src) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue          # a half-written trailing line: skipped, never fatal
                if isinstance(row, dict):
                    yield row


def campaign_status(campaign_dir: str | Path, scan_records: bool = False) -> dict:
    """Everything the committed files say about this campaign's think setting."""
    d = Path(campaign_dir)
    out: dict = {
        "campaign_dir": str(d), "meta": {"state": UNKNOWN, "why": "campaign_meta.json is missing",
                                         "think_sent": None, "model_capabilities": None},
        "records": {"state": UNKNOWN, "why": "no records were read", "counts": {}, "n_rows": 0},
        "scan": None, "state": UNKNOWN, "agree": None,
    }
    meta_path = d / "campaign_meta.json"
    meta = None
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as e:
            out["meta"] = {"state": UNKNOWN, "why": f"campaign_meta.json is unreadable ({e})",
                           "think_sent": None, "model_capabilities": None}
    if isinstance(meta, dict):
        state, why = meta_status(meta)
        out["meta"] = {"state": state, "why": why,
                       "think_sent": meta.get("think_sent") if "think_sent" in meta else "<absent>",
                       "model_capabilities": meta.get("model_capabilities"),
                       "num_predict": ((meta.get("config") or {}).get("options") or {}).get("num_predict"),
                       "config_think": (meta.get("config") or {}).get("think", "<absent>")}

    records_path = d / "records.jsonl"
    counts: dict[str, int] = {}
    n_rows = leaked = scanned = 0
    for rec in iter_records(records_path):
        st, _ = record_status(rec)
        counts[st] = counts.get(st, 0) + 1
        n_rows += 1
        if scan_records:
            got = thought_block_leaked(rec)
            if got is not None:
                scanned += 1
                leaked += 1 if got else 0
    if n_rows:
        states = set(counts)
        if len(states) == 1:
            only = states.pop()
            out["records"] = {"state": only, "n_rows": n_rows, "counts": counts,
                              "why": f"all {n_rows} row(s) recorded the same `request.think`"}
        else:
            out["records"] = {"state": UNKNOWN, "n_rows": n_rows, "counts": counts,
                              "why": f"rows disagree: {counts}"}
    else:
        out["records"]["why"] = f"no rows at {oh.jsonl_describe(records_path)}"
    if scan_records:
        out["scan"] = {"rows_with_logprobs": scanned, "rows_with_a_thought_block": leaked}

    m, r = out["meta"]["state"], out["records"]["state"]
    if m == r:
        out["state"], out["agree"] = m, True
    elif r == UNKNOWN:
        out["state"], out["agree"] = m, True     # meta settles it; rows carry no capabilities
    elif m == UNKNOWN:
        out["state"], out["agree"] = r, True
    else:
        out["state"], out["agree"] = UNKNOWN, False
    return out


def find_campaigns(root: Path) -> list[Path]:
    """Every directory under the known roots that has a campaign_meta.json."""
    found: list[Path] = []
    for rel in CAMPAIGN_ROOTS:
        base = root / rel
        if not base.is_dir():
            continue
        for meta in sorted(base.glob("*/campaign_meta.json")):
            found.append(meta.parent)
    return found


# --- CLI ---------------------------------------------------------------------

def render(status: dict) -> str:
    lines = [f"{status['campaign_dir']}",
             f"  overall           {status['state']}"
             + ("" if status["agree"] else "   <-- meta and records DISAGREE")]
    m = status["meta"]
    lines.append(f"  campaign_meta     {m['state']}  (think_sent={m['think_sent']!r}"
                 + (f", num_predict={m.get('num_predict')}" if "num_predict" in m else "") + ")")
    lines.append(f"                    {m['why']}")
    r = status["records"]
    lines.append(f"  records           {r['state']}  ({r['n_rows']} row(s); {r['counts'] or 'none'})")
    lines.append(f"                    {r['why']}")
    if status["scan"] is not None:
        s = status["scan"]
        n = s["rows_with_a_thought_block"]
        lines.append(f"  token stream      {n} of {s['rows_with_logprobs']} row(s) carry a thought "
                     f"block anyway" + ("" if n else "  (none)"))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    oh.force_utf8()
    ap = argparse.ArgumentParser(description="Report the recorded thinking setting of a campaign.")
    ap.add_argument("campaign", nargs="*", help="campaign directories to inspect")
    ap.add_argument("--all", action="store_true",
                    help=f"every campaign under {' and '.join(CAMPAIGN_ROOTS)}")
    ap.add_argument("--scan-records", action="store_true",
                    help="also read the token stream and count rows carrying a thought block")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    dirs = [Path(c) for c in args.campaign]
    if args.all:
        dirs += find_campaigns(oh.MOME_ROOT)
    if not dirs:
        ap.error("give at least one campaign directory, or --all")

    results = [campaign_status(d, scan_records=args.scan_records) for d in dirs]
    if args.json:
        print(json.dumps(results, indent=1, ensure_ascii=False))
    else:
        for res in results:
            print(render(res), flush=True)
        settled = sum(1 for r in results if r["state"] in SETTLED)
        print(f"\n{settled} of {len(results)} campaign(s) prove a settled thinking setting "
              f"({', '.join(SETTLED)}).", flush=True)
    bad = [r for r in results if r["state"] not in SETTLED or not r["agree"]]
    return 3 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
