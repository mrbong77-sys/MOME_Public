#!/usr/bin/env python3
r"""Phase 1 pool runner: N generation paths per item, with per-token logprobs.

`run_campaign.py` draws ONE greedy generation per item.  The
classifier needs a **pool** per item, because two of the things it is measured
against are defined over several paths and cannot be computed from one:

  * the baselines it has to beat - `vote_share` over the pool, and "how many
    candidates agree with the greedy answer" (the 2-generation agreement
    baseline);
  * the **I channel**, whose training target is inter-path disagreement.

This mirrors the numeric route of the finished MOME study: a greedy anchor plus
four temperature-sampled CoT candidates (`_pool5`, mome/solver/routed.py:83-115,
`K_SAMPLES = 4`, `SAMPLING_TEMPERATURE = 0.7`, `seed = seed_base + i`).  MOME
drew its four samples with the `cot` template while the anchor used `format`;
Phase 1 defaults instead to the SAME template on every path, so the only thing
that separates a sampled path from the greedy one is the sampling parameters.
A config that wants MOME's mixture sets `"sample_template": "cot"`.

Everything else is `run_campaign.py`'s: the prompt building, the request body,
the logprob parsing and the record shape are imported from it, never copied, so
a sampled request differs from a greedy one only in `options.temperature`,
`options.top_p` and `options.seed`.

**One optional key, and it is absent from every committed cell.**  A config may
carry `resolve_feedback` -- `{"file": ..., "field": "told" | "chan"}` -- and then
one committed paragraph is put BEFORE the template text for each item, which is
what makes an informed re-solve informed (`mome/resolve_feedback.py`).
With the key absent, `rf.prefix` returns `""` and every prompt this file builds is
byte-for-byte the prompt it has always built; no template is edited, no output
contract is moved and no parser changes.  `mome/test_resolve.py` asserts
the byte-identity on the committed templates rather than trusting this sentence.

    python mome\run_pool.py --config mome\configs\pool\gsm8k_gemma4-e2b_pool.json --dry-run
    python mome\run_pool.py --config mome\configs\pool\gsm8k_gemma4-e2b_pool.json --limit 3
    python mome\run_pool.py --config mome\configs\pool\gsm8k_gemma4-e2b_pool.json

Resume is per **(item, path)**, not per item: a run killed after three of four
samples on item 250 resumes at sample four, not at item 250 path 0.

A response that breaks the `eval_count - n_logprob_tokens` rule is NEVER stored
as a clean record: it is re-requested (`run_campaign.draw_generation`), and only
if it never comes back clean is the record written with `error` set, which keeps
it out of grading and makes the next run request it again.

Records carry the fields of `schema.md` section 1 plus `path_index`,
`path_kind`, `n_paths` and `sampling` (schema.md section 1.1).

Standard library only.  Never imports `mome`; never writes outside ``.

Exit codes: 0 finished (per-item errors are counted in the summary); 2 the
server could not be reached at start; 3 aborted because the server stopped
answering mid-run, or answered several generations in a row with a response
that broke the logprob rule on every attempt; 4 configuration / benchmark /
guard problem.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ollama_http as oh        # noqa: E402
import prompts                  # noqa: E402
import resolve_feedback as rf   # noqa: E402  the informed re-solve's one paragraph
import run_campaign as rc       # noqa: E402
import think_check as tc        # noqa: E402

SCHEMA_VERSION = rc.SCHEMA_VERSION
CONSECUTIVE_CONN_ABORT = rc.CONSECUTIVE_CONN_ABORT
# The non-final-response retry and its abort guard live in run_campaign.py, so
# both runners apply one implementation (see rc.draw_generation).
ANOMALY_ATTEMPTS = rc.ANOMALY_ATTEMPTS
CONSECUTIVE_ANOMALY_ABORT = rc.CONSECUTIVE_ANOMALY_ABORT

#: Pool defaults, reimplemented from mome/solver/routed.py:37 (`K_SAMPLES = 4`),
#: mome/solver/solve.py:38 (`SAMPLING_TEMPERATURE = 0.7`), and the defaults of
#: `GenerationParams` (mome/inference/base.py:22-27: `top_p = 1.0`), which the
#: native Ollama backend always puts in `options` (mome/inference/ollama.py:
#: 122-129).  `seed_base` is MOME's: `seed = seed_base + i` for the i-th sample
#: (mome/solver/routed.py:96-97), default 0 (mome/solver/solve.py:96).
POOL_DEFAULTS = {
    "n_samples": 4,
    "temperature": 0.7,
    "top_p": 1.0,
    "seed_base": 0,
    "include_greedy": True,
    "sample_template": "same",
}


class Guard(SystemExit):
    """A refusal to run: raised before anything is opened for writing."""


# --- configuration ----------------------------------------------------------

def load_pool_config(path: Path, args) -> dict:
    cfg = rc.load_config(path, args)
    pool = dict(POOL_DEFAULTS)
    pool.update(cfg.get("pool") or {})
    unknown = sorted(set(pool) - set(POOL_DEFAULTS))
    if unknown:
        raise Guard(f"config {path}: unknown key(s) in \"pool\": {', '.join(unknown)}")
    if int(pool["n_samples"]) < 0:
        raise Guard(f"config {path}: pool.n_samples must be >= 0")
    if float(pool["temperature"]) <= 0 and int(pool["n_samples"]) > 0:
        raise Guard(f"config {path}: pool.temperature is {pool['temperature']}; sampled paths at "
                    "temperature 0 would be copies of the greedy path, which measures nothing")
    if pool["sample_template"] not in ("same", "cot"):
        raise Guard(f"config {path}: pool.sample_template must be \"same\" or \"cot\"")
    if not pool["include_greedy"] and int(pool["n_samples"]) == 0:
        raise Guard(f"config {path}: the pool would be empty")
    cfg["pool"] = pool
    return cfg


def path_plan(cfg: dict, template_id: str) -> list[dict]:
    """The paths one item gets, in the order they are generated.

    Path 0 is the greedy anchor and sends exactly what `run_campaign.py` sends
    for this cell (no `options_override`, so the config's own `options` and
    `seed` go out untouched).  Paths 1..n are the sampled ones.
    """
    pool = cfg["pool"]
    plan: list[dict] = []
    if pool["include_greedy"]:
        opts = dict(cfg["options"])
        if cfg.get("seed") is not None:
            opts["seed"] = cfg["seed"]
        plan.append({
            "path_index": 0, "path_kind": "greedy", "template_id": template_id,
            "options_override": None,
            "sampling": {"temperature": opts.get("temperature"), "top_p": opts.get("top_p"),
                         "seed": opts.get("seed")},
        })
    if pool["sample_template"] == "cot":
        sample_template = "cot_" + cfg["language"]
    else:
        sample_template = template_id
    # With `include_greedy: false` the indices still start at 1: path 0 is then
    # the already-committed greedy campaign's record, and the two directories
    # join on (item_id, path_index) without renumbering anything.
    for i in range(int(pool["n_samples"])):
        override = {"temperature": float(pool["temperature"]), "top_p": float(pool["top_p"]),
                    "seed": int(pool["seed_base"]) + i}
        plan.append({
            "path_index": len(plan) if pool["include_greedy"] else i + 1,
            "path_kind": "sampled", "template_id": sample_template,
            "options_override": override, "sampling": dict(override),
        })
    for spec in plan:
        spec["n_paths"] = len(plan)
    return plan


# --- the informed re-solve's one paragraph ----------------------------------
#
# `resolve_feedback` is the ONLY thing this runner ever puts in front of a
# template, and it is put there for the arms that name it and for nothing else.
# With the optional config key `resolve_feedback` absent -- which is every
# committed cell, and both BLIND arms of the informed re-solve -- `rf.prefix`
# returns `""` and the prompt below is byte-for-byte the prompt this file has
# always built.  `mome/test_resolve.py` asserts that on the committed
# templates rather than trusting the sentence.
#
# The template itself is never edited and the output contract is never touched:
# the paragraph goes BEFORE `prompts.build_prompt`'s text, so
# `template_sha256` and every parser stay exactly as they are.  That is the
# first smoke round's lesson, paid for once: `tir3` moved the output contract
# and 37 of 60 generations answered with a PEP 526 annotation that runs and
# prints nothing, and the resulting 0.0 was a fact about our tooling.

def prompt_for(cfg: dict, spec: dict, item_id: str, question: str) -> str:
    """The prompt one (item, path) is sent: the arm's paragraph, then the template."""
    return rf.build_prompt(prompts, spec["template_id"], question,
                           cfg.get("resolve_feedback"), item_id)


def check_feedback_covers(cfg: dict, items: list[tuple[str, dict]]) -> str | None:
    """Every item of an informed arm has a committed wording, or the run refuses.

    Asked BEFORE the first request, because an item with no wording would be
    generated blind inside an arm whose whole content is that it is not blind,
    and nothing downstream could tell the two apart afterwards.
    """
    fb = cfg.get("resolve_feedback")
    if not fb:
        return None
    path = rf.resolve_path(fb)
    if not path.exists():
        raise Guard(f"config {cfg['name']}: resolve_feedback.file {path} is not there. Build it "
                    "with\n\n    python mome\\make_resolve_items.py\n")
    data = rf.load(path)
    missing = [i for i, _ in items if str(i) not in data["by_item"]]
    if missing:
        raise Guard(f"config {cfg['name']}: {len(missing)} item(s) have no feedback wording in "
                    f"{path} (first: {missing[:5]}); an item with no wording would run BLIND "
                    "inside an informed arm")
    return f"feedback: {fb['field']!r} wording for {len(items)} item(s) from {path}"


# --- guards -----------------------------------------------------------------

def campaign_name_of(out_dir: Path) -> str | None:
    """The campaign already living in `out_dir`, from its metadata or records."""
    meta = out_dir / "campaign_meta.json"
    if meta.exists():
        try:
            return ((json.load(open(meta, encoding="utf-8")) or {}).get("config") or {}).get("name")
        except ValueError:
            pass
    src = oh.jsonl_source(out_dir / "records.jsonl")
    if src is not None:
        with oh.open_jsonl(src) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    return json.loads(line).get("campaign")
                except ValueError:
                    return None
    return None


def guard_out_dir(cfg: dict, out_dir: Path, anchor_dir: Path | None) -> None:
    """Refuse to write into another campaign's directory.

    The greedy cell `data/ungated/gsm8k_gemma4-e2b/` is a real 500-item
    measurement that is already committed.  A pool run must never append to it
    or overwrite its `campaign_meta.json` / `env.json`, so the two ways that
    could happen are closed here, before any file is opened.
    """
    if anchor_dir is not None and out_dir.resolve() == anchor_dir.resolve():
        raise Guard(f"out_dir {out_dir} is the anchor campaign's own directory; a pool run must "
                    "write somewhere else (set \"out_dir\" to e.g. "
                    f"data/ungated/{cfg['name']})")
    existing = campaign_name_of(out_dir)
    if existing is not None and existing != cfg["name"]:
        raise Guard(f"out_dir {out_dir} already holds campaign {existing!r}, not {cfg['name']!r}; "
                    "writing here would mix two campaigns in one directory")


def anchor_item_ids(anchor_dir: Path) -> tuple[list[str], str] | None:
    """The item ids of an already-finished campaign, and where they were read.

    `grades.jsonl` is preferred because it is one small line per item; the
    records are scanned only when there are no grades, and then without parsing
    the logprob arrays (the committed cell's `records.jsonl.gz` is 33.7 MB
    packed and far larger in memory).
    """
    grades = anchor_dir / "grades.jsonl"
    if oh.jsonl_source(grades) is not None:
        ids = [str(r["item_id"]) for r in oh.read_jsonl(grades) if r.get("item_id") is not None]
        if ids:
            return ids, str(oh.jsonl_source(grades))
    src = oh.jsonl_source(anchor_dir / "records.jsonl")
    if src is None:
        return None
    seen: list[str] = []
    known: set[str] = set()
    with oh.open_jsonl(src) as f:
        for line in f:
            i = line.find('"item_id":')
            if i < 0:
                continue
            chunk = line[i + 10:i + 200]
            start = chunk.find('"')
            end = chunk.find('"', start + 1)
            if start < 0 or end < 0:
                continue
            iid = chunk[start + 1:end]
            if iid not in known:
                known.add(iid)
                seen.append(iid)
    return (seen, str(src)) if seen else None


#: The one scope name that narrows the item gate, and the keys the block that
#: asks for it must carry.  A config that names any other scope is refused
#: rather than defaulted: the point of the block is that the narrowing is
#: WRITTEN DOWN, and a typo that silently fell back to the strict gate -- or
#: worse, to none -- would defeat that.
def given_up_keys(existing: list[dict], give_up_after: int) -> set[tuple[str, int]]:
    """(item, path) pairs this run skips because they keep failing.

    `run_campaign.py` has the same rule keyed on the item; a pool's unit is the
    (item, path) pair, because one path of an item can fail deterministically
    while the others are fine.  A pair with an error-free record is never given
    up -- it is already done.  `give_up_after = 0` turns the rule off, which is
    the default and what every committed cell ran under.
    """
    if not give_up_after:
        return set()
    done = {oh.record_key(r) for r in existing if r.get("error") is None}
    errors: dict[tuple[str, int], int] = {}
    for r in existing:
        if r.get("error") is not None:
            key = oh.record_key(r)
            errors[key] = errors.get(key, 0) + 1
    return {k for k, c in errors.items() if c >= give_up_after and k not in done}


ANCHOR_ITEM_SCOPES = ("anchor_graded_ids", "benchmark_order")
ANCHOR_ITEM_CHECK_KEYS = ("scope", "order_file", "registered_in", "why")


def anchor_item_check_spec(cfg: dict) -> dict | None:
    """The config's `anchor_item_check` block, validated, or None if it has none.

    Absent is the default and is the STRICT gate: the ids must be a subset of
    what the anchor cell actually graded.  Every committed cell written before
    this key existed carries nothing here and keeps exactly that gate.
    """
    spec = cfg.get("anchor_item_check")
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise Guard("config: \"anchor_item_check\" must be an object carrying "
                    + ", ".join(ANCHOR_ITEM_CHECK_KEYS))
    missing = [k for k in ANCHOR_ITEM_CHECK_KEYS if not spec.get(k)]
    if missing:
        raise Guard(f"config: \"anchor_item_check\" is missing {', '.join(missing)}; a narrowed "
                    "gate has to say what it narrows to, which file the ids must come from, "
                    "where the narrowing is registered in writing, and why")
    if spec["scope"] not in ANCHOR_ITEM_SCOPES:
        raise Guard(f"config: \"anchor_item_check\".scope is {spec['scope']!r}; the only scopes "
                    f"are {', '.join(ANCHOR_ITEM_SCOPES)}")
    for key in ("order_file", "registered_in"):
        if not oh.resolve(spec[key]).exists():
            raise Guard(f"config: \"anchor_item_check\".{key} points at {spec[key]}, which is not "
                        "there; a narrowing whose registration cannot be read is not registered")
    return spec


def check_item_scope(items: list[tuple[str, dict]], spec: dict) -> str:
    """The narrowed gate: the ids must come from the committed benchmark order.

    This REPLACES the anchor's graded-id set with the committed order file as
    the set the run's ids must be drawn from.  It is not the gate switched off:
    a fresh draw of the benchmark, a reordered list or an id from anywhere else
    is refused here exactly as it would be refused against an anchor.  What it
    stops refusing is an id the anchor cell LOST -- which is a fact about that
    cell, and which a run that reuses none of that cell's data has no reason to
    inherit.
    """
    order = [ln.strip() for ln in open(oh.resolve(spec["order_file"]), encoding="utf-8")
             if ln.strip()]
    ours = [i for i, _ in items]
    extra = sorted(set(ours) - set(order))
    if extra:
        raise Guard(f"{len(extra)} item id(s) are not in {spec['order_file']} (first: "
                    f"{extra[:5]}); under the narrowed scope the ids must still come from the "
                    "committed benchmark order, never a fresh sample")
    note = (f"item scope NARROWED to {spec['scope']}: {len(ours)} of the {len(order)} ids of "
            f"{spec['order_file']}, registered in {spec['registered_in']}")
    if len(ours) < len(order):
        note += " (fewer than the order holds -- --limit, or a shorter list)"
    return note


def check_against_anchor(items: list[tuple[str, dict]], anchor_dir: Path) -> str:
    """Confirm the pool runs the anchor campaign's own 500 item ids."""
    found = anchor_item_ids(anchor_dir)
    if found is None:
        raise Guard(f"anchor_dir {anchor_dir} has no grades.jsonl and no records.jsonl(.gz); "
                    "point it at the finished greedy campaign, or remove it from the config")
    anchor_ids, source = found
    ours = [i for i, _ in items]
    if set(ours) - set(anchor_ids):
        extra = sorted(set(ours) - set(anchor_ids))[:5]
        raise Guard(f"{len(set(ours) - set(anchor_ids))} item id(s) are not in the anchor campaign "
                    f"{anchor_dir.name} (first: {extra}); the pool must reuse the anchor's items, "
                    "never a fresh sample of the benchmark")
    note = f"anchor {anchor_dir.name}: {len(anchor_ids)} item ids read from {source}"
    if len(ours) < len(anchor_ids):
        note += f"; this run covers {len(ours)} of them (--limit)"
    elif ours == anchor_ids:
        note += "; identical to this run's item list, same order"
    else:
        note += "; same set as this run's item list, different order"
    return note


# --- how big the committed file will be -------------------------------------

def mb(nbytes: float) -> str:
    if nbytes < 1024 ** 2:
        return f"{nbytes / 1024:.0f} kB"
    if nbytes < 1024 ** 3:
        return f"{nbytes / 1024 ** 2:.0f} MB"
    return f"{nbytes / 1024 ** 3:.2f} GB"



GITHUB_HARD_LIMIT = 100 * 1024 * 1024
GITHUB_WARN_LIMIT = 50 * 1024 * 1024
#: Measured on the committed greedy cell: records.jsonl is 173,496,694 bytes
#: raw and 33,680,386 packed, a ratio of 5.2.  Used only to warn about free
#: disk before a run, never reported as a size of anything.
RAW_TO_PACKED = 5.2


def size_projection(cfg: dict, anchor_dir: Path | None, n_items: int, n_paths: int) -> list[str]:
    """What this run will weigh on disk and per committed file.

    GitHub's 100 MB ceiling is per FILE, and `pack_records.py` splits a pool by
    `path_index`, so the number that matters is the size of ONE path file - one
    record per item - not the size of the whole campaign.  The figures are
    scaled from the anchor campaign's MEASURED packed size, never guessed, and
    printed before anything is generated: finding a size problem out after an
    hour of GPU time is the expensive way.
    """
    if anchor_dir is None:
        return []
    # The anchor may be packed either way: one `records.jsonl.gz` (a greedy
    # cell) or a per-path set (a pool).  Both are measured sizes of the same
    # thing - the packed bytes of that campaign's records - so both scale.
    base = anchor_dir / "records.jsonl"
    packed = [oh.gz_path(base)] if oh.gz_path(base).exists() else oh.part_paths(base)
    grades = anchor_dir / "grades.jsonl"
    if not packed or oh.jsonl_source(grades) is None:
        return []
    with oh.open_jsonl(oh.jsonl_source(grades)) as f:
        n_anchor = sum(1 for line in f if line.strip())
    if not n_anchor:
        return []
    anchor_bytes = sum(x.stat().st_size for x in packed)
    per_record = anchor_bytes / n_anchor
    anchor_cfg = campaign_config_of(anchor_dir) or {}
    anchor_k = anchor_cfg.get("top_logprobs")
    ours_k = cfg["top_logprobs"] if cfg["logprobs"] else 0
    where = (packed[0].name if len(packed) == 1
             else f"{len(packed)} per-path file(s) of {anchor_dir.name}")
    lines = [f"packed size: the anchor's {where} measure {anchor_bytes / 1e6:.1f} MB "
             f"over {n_anchor} records at top_logprobs {anchor_k}"]
    if anchor_k != ours_k:
        lines.append(f"  this run uses top_logprobs {ours_k}, so that figure does not scale directly; "
                     "the measured size-per-top_logprobs table is in mome/README.md")
        return lines
    # A record's size is set by how many tokens it carries, so a different
    # `num_predict` moves it and the anchor cannot say by how much.  Say so
    # instead of printing a number that quietly assumes the caps are equal.
    anchor_cap = (anchor_cfg.get("options") or {}).get("num_predict")
    ours_cap = (cfg.get("options") or {}).get("num_predict")
    if anchor_cap != ours_cap:
        lines.append(f"  the anchor ran at num_predict {anchor_cap} and this run at {ours_cap}, so the "
                     "figure below is a FLOOR, not an estimate: the records this run truncates less "
                     "will be longer and heavier")
    per_file = per_record * n_items
    total = per_file * n_paths
    lines.append(f"  estimate: {per_record / 1e3:.0f} kB/record x {n_items} items = about "
                 f"{mb(per_file)} per path file, {n_paths} of them, {mb(total)} packed in all "
                 f"(pack_records.py splits a pool by path_index)")
    lines.append(f"  the plain records.jsonl before packing is bigger: the anchor's packs "
                 f"{RAW_TO_PACKED:.1f}x, so budget about {mb(total * RAW_TO_PACKED)} raw + "
                 f"{mb(total)} packed = {mb(total * (1 + RAW_TO_PACKED))} free disk")
    if per_file > GITHUB_HARD_LIMIT:
        lines.append("  NOTE: one path file alone would be over GitHub's 100 MB hard per-file limit. "
                     "pack_records.py splits such a path further by item range "
                     "(records_p<path+100*shard>.jsonl.gz), so the push still works and "
                     "--top-logprobs stays at 10; watch the packed sizes at the end of the run.")
    elif per_file > 0.8 * GITHUB_HARD_LIMIT:
        lines.append("  NOTE: one path file is within 20% of GitHub's 100 MB hard per-file limit; "
                     "watch the packed sizes at the end of the run.")
    elif per_file > GITHUB_WARN_LIMIT:
        lines.append("  a path file is over GitHub's 50 MB warning line but well under the 100 MB "
                     "hard limit; the push works.")
    return lines


def campaign_config_of(d: Path) -> dict | None:
    meta = d / "campaign_meta.json"
    if not meta.exists():
        return None
    try:
        return (json.load(open(meta, encoding="utf-8")) or {}).get("config")
    except ValueError:
        return None


# --- progress ---------------------------------------------------------------

def hms(seconds: float) -> str:
    seconds = int(max(0, seconds))
    return f"{seconds // 3600:d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


class ETA:
    """A running estimate from this run's own completed requests.

    Deliberately measured, never assumed: the estimate only appears once
    `min_samples` requests of THIS run have finished, and it is printed as an
    estimate ("ETA"), not as a duration.  `written` is the raw bytes this run
    has appended to `records.jsonl` so far, so the disk footprint is visible
    while it grows instead of at the end.
    """

    def __init__(self, total: int, min_samples: int = 5):
        self.total = total
        self.min_samples = min_samples
        self.t0 = time.perf_counter()
        self.done = 0
        self.written = 0

    def tick(self, nbytes: int = 0) -> str:
        self.done += 1
        self.written += nbytes
        elapsed = time.perf_counter() - self.t0
        text = f"| {self.done}/{self.total} elapsed {hms(elapsed)} raw {mb(self.written)}"
        if self.done >= self.min_samples and self.done < self.total:
            per = elapsed / self.done
            text += (f" ETA {hms(per * (self.total - self.done))} (~{per:.2f}s/req, "
                     f"raw ~{mb(self.written / self.done * self.total)} total)")
        return text


# --- main -------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 1 pool runner: greedy + sampled paths per item.")
    ap.add_argument("--config", required=True, help="mome/configs/pool/<cell>.json")
    ap.add_argument("--limit", type=int, default=None, help="only the first N item ids (smoke runs)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the payloads, send nothing")
    ap.add_argument("--host", default=None, help="override host (default from config)")
    ap.add_argument("--out-dir", default=None, help="override output directory")
    ap.add_argument("--benchmark-path", nargs="+", default=None, help="override benchmark jsonl path(s)")
    ap.add_argument("--items-file", default=None)
    ap.add_argument("--anchor-dir", default=None, help="override the anchor campaign directory")
    ap.add_argument("--no-anchor-check", action="store_true",
                    help="run without cross-checking the item ids against the anchor campaign")
    ap.add_argument("--timeout", type=float, default=None, help="per-request timeout seconds")
    ap.add_argument("--top-logprobs", type=int, default=None)
    ap.add_argument("--endpoint", choices=["/api/generate", "/api/chat"], default=None)
    ap.add_argument("--no-env", action="store_true", help="skip env.json capture (not for real runs)")
    ap.add_argument("--give-up-after", type=int, default=0,
                    help="skip, for this run only, any (item, path) pair with "
                         "at least this many error rows and no error-free "
                         "record (0 disables). Same rule as run_campaign.py, "
                         "but the unit is the (item, path) pair rather than "
                         "the item: one generation that fails "
                         "deterministically would otherwise wedge the whole "
                         "cell forever, caught between the rule that a resume "
                         "retries errors first and the three-error abort.")
    args = ap.parse_args()

    try:
        cfg = load_pool_config(Path(args.config), args)
        items = rc.load_items(cfg, args.limit)
    except SystemExit as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 4

    # the route default unless the config names a template explicitly; the
    # optional key is what lets a cell run `tir2_en` / `tir3_en` without
    # touching the template every committed cell used (prompts.py)
    try:
        template_id = prompts.resolve_template_id(cfg["route"], cfg["language"],
                                                  cfg.get("prompt_template_id"))
    except ValueError as e:
        print(f"ERROR: config {args.config}: {e}", file=sys.stderr)
        return 4
    plan = path_plan(cfg, template_id)
    n_paths = len(plan)
    out_dir = oh.resolve(cfg["out_dir"])
    records_path = out_dir / "records.jsonl"
    anchor = cfg.get("anchor_dir") or None
    if args.anchor_dir:
        anchor = args.anchor_dir
    anchor_dir = oh.resolve(anchor) if anchor else None

    anchor_note = feedback_note = scope_note = None
    try:
        guard_out_dir(cfg, out_dir, anchor_dir)
        #: The item gate, in one of its two registered shapes.  With no
        #: `anchor_item_check` block this is byte-for-byte what it has always
        #: been: the ids must be a subset of what the anchor graded, and an arm
        #: with no anchor gets no item gate at all.  With the block the gate is
        #: NARROWED, not dropped -- the ids must come from the committed
        #: benchmark order instead -- and it then applies whether or not the arm
        #: has an anchor, which is what puts a gate on the two backbones that
        #: have no MATH-500 control cell to be gated against.
        scope = anchor_item_check_spec(cfg)
        if scope is not None and scope["scope"] == "benchmark_order":
            if not args.no_anchor_check:
                scope_note = check_item_scope(items, scope)
            if anchor_dir is not None and not args.no_anchor_check:
                #: the anchor is still READ and still reported; only the id set
                #: the run is held to has moved.
                anchor_note = (f"anchor {anchor_dir.name}: "
                               f"{len(anchor_item_ids(anchor_dir)[0])} item ids read, NOT used as "
                               "the item gate for this cell (see the scope line above)")
        elif anchor_dir is not None and not args.no_anchor_check:
            anchor_note = check_against_anchor(items, anchor_dir)
        feedback_note = check_feedback_covers(cfg, items)
    except SystemExit as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 4

    print(f"campaign {cfg['name']}: {len(items)} items x {n_paths} paths = {len(items) * n_paths} "
          f"generations  ->  {records_path}", flush=True)
    for spec in plan:
        s = spec["sampling"]
        print(f"  path {spec['path_index']} {spec['path_kind']:>7}  template {spec['template_id']}  "
              f"temperature={s.get('temperature')} top_p={s.get('top_p')} seed={s.get('seed')}", flush=True)
    if scope_note:
        print(f"  {scope_note}", flush=True)
    if anchor_note:
        print(f"  {anchor_note}", flush=True)
    if feedback_note:
        print(f"  {feedback_note}", flush=True)
    for line in size_projection(cfg, anchor_dir, len(items), n_paths):
        print(f"  {line}", flush=True)

    if args.dry_run:
        item_id, row = items[0]
        for spec in plan:
            prompt = prompt_for(cfg, spec, item_id, row["question"])
            payload = rc.build_payload(cfg, prompt, None, spec["options_override"])
            print(f"---- path {spec['path_index']} ({spec['path_kind']}) payload for {item_id} "
                  "(think decided from /api/show at run time) ----")
            print(json.dumps({k: v for k, v in payload.items() if k not in ("prompt", "messages")},
                             ensure_ascii=False, indent=1))
        print("---- path %d prompt (%s) ----" % (plan[0]["path_index"], items[0][0]))
        print(prompt_for(cfg, plan[0], item_id, row["question"]))
        return 0

    host = cfg["host"]
    timeout = float(cfg["timeout_seconds"])
    try:
        version = oh.server_version(host, timeout=min(timeout, 30))
    except oh.OllamaError as e:
        print(f"ERROR: cannot reach Ollama at {host} ({e}). Is `ollama serve` running?", file=sys.stderr)
        return 2
    print(f"Ollama {version} at {host}; model {cfg['model']}; endpoint {cfg['endpoint']}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    env = None if args.no_env else rc.capture_env(cfg, timeout)
    if env is not None:
        print("env.json: " + rc.write_run_meta(out_dir / "env.json", env, "captured_at"), flush=True)
    try:
        show_info = env["show"] if env and isinstance(env.get("show"), dict) and "error" not in env["show"] \
            else oh.show(host, cfg["model"], timeout=timeout)
    except oh.OllamaError:
        show_info = {}
    capabilities = show_info.get("capabilities") or []
    think_flag = rc.resolve_think(cfg, capabilities)
    digest = None
    tags = env.get("tags_entry") if env else None
    if tags is None:
        try:
            tags = oh.tags_entry(host, cfg["model"], timeout=timeout)
        except oh.OllamaError:
            tags = None
    if isinstance(tags, dict):
        digest = tags.get("digest")

    try:
        import check_templates
        template_check = check_templates.compare()
    except Exception as e:  # noqa: BLE001 - the check is informative, never blocking
        template_check = {"error": str(e)}
    templates_used = sorted({spec["template_id"] for spec in plan})
    meta = {"config": cfg, "config_path": str(Path(args.config)), "argv": sys.argv[1:],
            "prompt_template_id": template_id, "templates_used": templates_used,
            "prompt_template_text": {t: prompts.TEMPLATES[t][0][prompts.TEMPLATES[t][1]]
                                     for t in templates_used},
            "template_sha256": {t: prompts.template_sha256(t) for t in templates_used},
            "template_check": template_check, "think_sent": think_flag,
            "model_capabilities": capabilities, "n_items_requested": len(items),
            "pool": cfg["pool"], "n_paths": n_paths,
            "paths": [{k: v for k, v in spec.items() if k != "options_override"} for spec in plan],
            "anchor_dir": str(anchor_dir) if anchor_dir else None, "anchor_check": anchor_note,
            "anchor_item_check": cfg.get("anchor_item_check"), "item_scope_check": scope_note,
            "started_at": oh.now_iso(), "schema_version": SCHEMA_VERSION}
    #: An informed arm records the exact paragraph it sent, per item, in its own
    #: metadata -- added ONLY when the arm names one, so every committed cell's
    #: `campaign_meta.json` keeps the shape it has.  Without it the wording would
    #: be reconstructible only by re-running a builder script, and the prompt IS
    #: the experiment.
    if cfg.get("resolve_feedback"):
        fb_spec = cfg["resolve_feedback"]
        fb_data = rf.load(rf.resolve_path(fb_spec))
        meta["resolve_feedback"] = {
            "file": fb_spec["file"],
            "field": fb_spec["field"],
            "wording_module": "mome/resolve_feedback.py",
            "the_template_is_unchanged": "the paragraph goes BEFORE the template text; "
                                         "tir2b_en and the output contract are untouched",
            "sent": {str(i): fb_data["by_item"][str(i)][fb_spec["field"]] for i, _ in items},
        }
    print("campaign_meta.json: "
          + rc.write_run_meta(out_dir / "campaign_meta.json", meta, ("started_at", "argv"),
                              {"argv": sys.argv[1:]}), flush=True)

    # resume per (item, path): an (item, path) with an error-free record is
    # skipped; one that has only error records is retried with attempt + 1
    existing = oh.read_jsonl(records_path)
    done_keys = {oh.record_key(r) for r in existing if r.get("error") is None}
    attempts: dict[tuple[str, int], int] = {}
    for r in existing:
        key = oh.record_key(r)
        attempts[key] = max(attempts.get(key, 0), int(r.get("attempt") or 1))
    given_up = given_up_keys(existing, args.give_up_after)
    todo = [(item_id, row, spec) for item_id, row in items for spec in plan
            if (item_id, spec["path_index"]) not in done_keys
            and (item_id, spec["path_index"]) not in given_up]
    print(f"{len(items) * n_paths} generations; {len(done_keys)} already done; {len(todo)} to generate; "
          f"think={think_flag}; top_logprobs={cfg['top_logprobs'] if cfg['logprobs'] else 'off'}"
          + (f"; {len(given_up)} given up (--give-up-after {args.give_up_after}: "
             f"{sorted(given_up)[:5]}{'...' if len(given_up) > 5 else ''})" if given_up else ""),
          flush=True)

    conflict = oh.packed_conflict(records_path) if todo else None
    if conflict:
        print(f"ERROR: {conflict}", file=sys.stderr)
        return 4

    n_ok = n_err = n_nolp = n_prog = 0
    #: rows whose token stream carried a `<|channel>thought` block anyway.  A
    #: COUNT, printed live so a leak that eats a whole budget is visible while
    #: the run is happening -- never a filter: no generation is dropped,
    #: excluded or reweighted for it, here or anywhere downstream.
    n_leak = n_leak_empty = 0
    consecutive_conn = 0
    consecutive_anom = 0
    n_anom_resp = n_anom_retried = n_anom_failed = n_anom_stored = 0
    total_eval = 0
    latencies: list[float] = []
    anomalies: list[str] = []
    per_path_ok: dict[int, int] = {}
    greedy_answers: dict[str, str | None] = {}
    eta = ETA(len(todo))
    writer = open(records_path, "a", encoding="utf-8") if todo else contextlib.nullcontext(None)
    with writer as out:
        for item_id, row, spec in todo:
            prompt = prompt_for(cfg, spec, item_id, row["question"])
            payload = rc.build_payload(cfg, prompt, think_flag, spec["options_override"])
            key = (item_id, spec["path_index"])
            base_attempt = attempts.get(key, 0) + 1
            path_fields = {"path_index": spec["path_index"], "path_kind": spec["path_kind"],
                           "n_paths": n_paths, "sampling": spec["sampling"]}

            def build(resp, latency, attempt, error, _id=item_id, _prompt=prompt, _spec=spec,
                      _payload=payload, _fields=path_fields):
                return rc.make_record(cfg, _id, _prompt, _spec["template_id"], _payload, resp,
                                      latency, attempt, error, digest, version, _fields)

            # One request when the response is complete - the same bytes, the
            # same record, the same `attempt` as before; more only when the
            # server answers with a non-final object (rc.draw_generation).
            rec, anom, err_kind = rc.draw_generation(host, cfg["endpoint"], payload, timeout,
                                                     base_attempt, build)
            if err_kind is None:
                consecutive_conn = 0
            elif err_kind == "connection":
                consecutive_conn += 1
            latency = rec["latency_seconds"]
            for a, why in anom:
                n_anom_resp += 1
                anomalies.append(f"{item_id} p{spec['path_index']} attempt {a}: {why} (re-requested)")
                print(f"{item_id} p{spec['path_index']}: ANOMALY({why}) on attempt {a} - the response "
                      "is not stored; re-requesting", flush=True)
            if rc.is_anomaly_error(rec):
                n_anom_failed += 1
                consecutive_anom += 1
            elif rec["error"] is None:
                if anom:
                    n_anom_retried += 1
                consecutive_anom = 0
            error = rec["error"]
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            out.write(line)
            out.flush()
            progress = eta.tick(len(line.encode("utf-8")))
            if error:
                n_err += 1
                print(f"{item_id} p{spec['path_index']}: ERROR {error} {progress}", flush=True)
                if consecutive_conn >= CONSECUTIVE_CONN_ABORT:
                    print(f"ABORT: {consecutive_conn} consecutive connection failures - is the server up? "
                          "Re-run the same command to resume.", file=sys.stderr)
                    return 3
                if consecutive_anom >= CONSECUTIVE_ANOMALY_ABORT:
                    print(rc.anomaly_summary(n_anom_resp, n_anom_retried, n_anom_failed), flush=True)
                    print(f"ABORT: {consecutive_anom} generations in a row broke the eval_count/logprobs "
                          f"rule on all {ANOMALY_ATTEMPTS} attempts - the server is answering with "
                          "non-final responses. Re-run the same command to resume.", file=sys.stderr)
                    return 3
                continue
            n_ok += 1
            per_path_ok[spec["path_index"]] = per_path_ok.get(spec["path_index"], 0) + 1
            latencies.append(latency)
            total_eval += int(rec["token_count"] or 0)
            if rec["n_logprob_tokens"] == 0:
                n_nolp += 1
            if spec["path_index"] == 0:
                greedy_answers[item_id] = rec["extracted_answer"]
            # A response that breaks the rule is retried and never stored, so
            # this can only fire for a campaign run with logprobs off, where
            # `draw_generation` has no rule to apply.  It stays because a stored
            # record that breaks the rule must always be named.
            why = oh.logprob_anomaly(rec)
            if why:
                n_anom_stored += 1
                anomalies.append(f"{item_id} p{spec['path_index']}: {why} (STORED)")
            if cfg["route"] == "expression" and rec["program_text"]:
                n_prog += 1
            leaked = tc.record_leak_state(rec) == tc.LEAKED
            if leaked:
                n_leak += 1
                # the case that matters most: the block ran to the cap and the
                # generation came back with nothing visible, yet the row looks
                # like an ordinary length truncation
                if not (rec["text_visible"] or "").strip():
                    n_leak_empty += 1
            print(f"{item_id} p{spec['path_index']}/{n_paths - 1}: tokens={rec['token_count']} "
                  f"logprobs={rec['n_logprob_tokens']} {latency:.1f}s answer={rec['extracted_answer']!r}"
                  # the expression route's answer is the EXECUTED one, and execute.py
                  # can only run a path that emitted a program: whether one came out
                  # is the thing to watch live, not the text-extracted answer
                  + (f" program={'yes' if rec['program_text'] else 'no'}"
                     if cfg["route"] == "expression" else "")
                  + (" TRUNCATED" if rec["truncated"] else "")
                  + (" THINK-BLOCK-LEAK" if leaked else "")
                  + (f" ANOMALY({why})" if why else "")
                  + " " + progress, flush=True)

    mean_lat = sum(latencies) / len(latencies) if latencies else 0.0
    by_path = ", ".join(f"p{i}={per_path_ok[i]}" for i in sorted(per_path_ok)) or "none"
    print(f"done: {n_ok} generated ({by_path}), {n_err} errors, {len(done_keys)} skipped (resume); "
          f"eval tokens {total_eval}; mean wall latency {mean_lat:.2f}s; "
          f"total {hms(time.perf_counter() - eta.t0)}; raw written this run {mb(eta.written)}; "
          f"records -> {records_path}", flush=True)
    if records_path.exists():
        print(f"records.jsonl is now {mb(records_path.stat().st_size)} raw; "
              "finish_campaign.py packs it (a pool: one .gz per path)", flush=True)
    if cfg["route"] == "expression":
        print(f"expression route: {n_prog} of {n_ok} generations emitted a ```python block. "
              "Those are the ONLY generations execute.py can run, and the F channel is computed "
              "from what it produces; a path with no program can populate I but never F. "
              "Next: finish_campaign.py runs execute.py before grading.", flush=True)
    if n_nolp:
        print(f"WARNING: {n_nolp} of {n_ok} responses carried no logprobs (check Ollama version / model)",
              flush=True)
    if n_leak:
        print(f"THOUGHT-BLOCK LEAK: {n_leak} of {n_ok} generation(s) emitted a `<|channel>thought` "
              f"block although `think: false` was sent; {n_leak_empty} of those returned nothing "
              f"visible at all. Each is FLAGGED on its record as `{tc.LEAK_FIELD}` and NOTHING "
              "else: no generation is dropped, excluded or reweighted, and no threshold reads the "
              "flag. It is named because a block that eats the budget lands as an ordinary "
              "done_reason \"length\", and the I channel is proximity to that budget -- so such a "
              "row would otherwise report the wrong cause.", flush=True)
    elif n_ok:
        print(f"no thought-block leak: none of {n_ok} generation(s) carried a `<|channel>thought` "
              "block (`think: false` was obeyed in band)", flush=True)
    print(rc.anomaly_summary(n_anom_resp, n_anom_retried, n_anom_failed), flush=True)
    if anomalies:
        print(f"ANOMALIES: {len(anomalies)} response(s) break the eval_count/logprobs rule "
              "(stop -> difference 1, length -> 0; schema.md section 6):", flush=True)
        for line in anomalies[:10]:
            print(f"  {line}", flush=True)
        if len(anomalies) > 10:
            print(f"  ... and {len(anomalies) - 10} more", flush=True)
        if n_anom_stored:
            print(f"  {n_anom_stored} of them stored as a clean record anyway; finish_campaign.py "
                  "re-checks every record and will refuse to pack the campaign while those stand.",
                  flush=True)
        else:
            print("  none of them was stored as a clean record: a retry replaced it, or the record "
                  "was written with `error` set, which keeps it out of grading and makes the next "
                  "run re-request that (item, path).", flush=True)
    else:
        print(f"all {n_ok} records obey the eval_count/logprobs rule "
              "(stop -> difference 1, length -> 0)", flush=True)

    if anchor_dir is not None and greedy_answers:
        agreed = compare_greedy_to_anchor(greedy_answers, anchor_dir)
        if agreed is not None:
            n_same, n_cmp = agreed
            print(f"greedy path vs the committed anchor {anchor_dir.name}: {n_same}/{n_cmp} of the "
                  "path-0 answers generated in THIS run are string-identical to the anchor's "
                  "graded answer (not a grader comparison; ordinary GPU non-determinism can "
                  "differ at temperature 0)", flush=True)
    return 0


def compare_greedy_to_anchor(greedy: dict[str, str | None], anchor_dir: Path) -> tuple[int, int] | None:
    """How many path-0 answers match the anchor campaign's graded final answer."""
    grades = anchor_dir / "grades.jsonl"
    if oh.jsonl_source(grades) is None:
        return None
    theirs = {str(r["item_id"]): r.get("final_answer") for r in oh.read_jsonl(grades)
              if r.get("item_id") is not None and oh.record_path_index(r) == 0}
    pairs = [(a, theirs[i]) for i, a in greedy.items() if i in theirs]
    if not pairs:
        return None
    return sum(1 for a, b in pairs if str(a) == str(b)), len(pairs)


if __name__ == "__main__":
    raise SystemExit(main())
