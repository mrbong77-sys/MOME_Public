#!/usr/bin/env python3
"""Phase 1 campaign runner: greedy generation with per-token logprobs.

Plan v0.2, section 5.3.  For one (backbone, benchmark) cell this script sends
every Phase 0 item through the same prompt template MOME used (`format` for
numeric benchmarks, `tir` for MATH-500; mome/prompts.py), over the
NATIVE Ollama route with `"logprobs": true, "top_logprobs": K`, and appends
one raw record per item to `data/ungated/<name>/records.jsonl`
(schema: mome/schema.md).  Gold answers, execution and grading are
deliberately absent from the records; see execute.py and grade.py.

Standard library only.  Never imports `mome`; never writes outside ``.

Usage (Windows, from the repository root; `py` may be `python`):

    py mome\run_campaign.py --config mome\configs\gsm8k_gemma4-e2b.json --dry-run
    py mome\run_campaign.py --config mome\configs\gsm8k_gemma4-e2b.json --limit 5
    py mome\run_campaign.py --config mome\configs\gsm8k_gemma4-e2b.json

Re-running the same config resumes: items that already have an error-free
record are skipped; items whose only records are errors are retried.

A response that breaks the `eval_count - n_logprob_tokens` rule is NEVER stored
as a clean record: it is re-requested (`draw_generation` below), and only if it
never comes back clean is the record written with `error` set, which keeps it
out of grading and makes the next run request it again.

Exit codes: 0 finished (possibly with per-item errors, counted in the
summary); 2 the server could not be reached at start; 3 aborted because the
server stopped answering mid-run, or answered several items in a row with a
response that broke the logprob rule on every attempt; 4 configuration /
benchmark problem.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract  # noqa: E402
import ollama_http as oh  # noqa: E402
import prompts  # noqa: E402
import think_check  # noqa: E402

SCHEMA_VERSION = 1
CONSECUTIVE_CONN_ABORT = 3

# --- the non-final response, and the retry that answers it -------------------
#
# Ollama 0.33.3 sometimes answers `200 OK` with a single NON-FINAL JSON object:
# the accumulated text and logprobs are there, but `done` is false and
# `done_reason`, `eval_count` and the durations are absent.  The body is
# `Transfer-Encoding: chunked` with no `Content-Length`, so a short answer
# cannot be spotted from the headers, and it is one valid JSON value, so
# `json.loads` succeeds: `make_record` below reads every field with `.get()`
# and used to turn such a response into a record with `error: null` that looks
# complete.  Resume then counted the item as done for ever.
#
# Proved intermittent on math500-213 path 0 (mome/probe_request.py,
# the byte-identical request three times in a row: the first answer non-final,
# the second and third complete), so the fix is to re-request, not to fail.
# `oh.logprob_anomaly` is the single implementation of the rule that catches
# it - a missing `eval_count` fires it - and is used here and by
# finish_campaign.py, so the live check cannot drift from the packing check.
ANOMALY_ATTEMPTS = 3            # total requests per generation, retries included
CONSECUTIVE_ANOMALY_ABORT = 3   # generations in a row ending as an anomaly error row
#: `error` of a record that broke the rule on every attempt starts with this.
ANOMALY_ERROR_PREFIX = "anomaly: "


def draw_generation(host: str, endpoint: str, payload: dict, timeout: float,
                    base_attempt: int, build_rec):
    """One generation, re-requested while its response breaks the logprob rule.

    `build_rec(resp, latency, attempt, error)` builds the record; the two
    runners differ only in the pool fields they add, so both pass their own.
    Returns `(rec, anomalies, err_kind)`:

      * `rec` - the record to WRITE: the first response that obeys the rule, or,
        when every attempt broke it, the last one with `error` set to a message
        naming the anomaly (`ANOMALY_ERROR_PREFIX`).  An error row is excluded
        from grading and from finish_campaign.py's rule check, and a resumed run
        re-requests it;
      * `anomalies` - `(attempt, why)` for every attempt whose response broke the
        rule, so the caller can report the retries honestly;
      * `err_kind` - `OllamaError.kind` of the last attempt, or None when it got
        an HTTP response at all; this is what the callers' connection guard
        counts.

    A connection / timeout / HTTP failure is returned at once and is NOT retried
    here: that path is exactly what it was, one record with its error, and the
    existing per-run resume retries it.  `payload` is never touched, so every
    attempt sends the same bytes.
    """
    # only a request that asked for logprobs can be judged by the logprob rule
    check = bool(payload.get("logprobs"))
    anomalies: list[tuple[int, str]] = []
    rec: dict = {}
    why: str | None = None
    for i in range(ANOMALY_ATTEMPTS if check else 1):
        attempt = base_attempt + i
        t0 = time.perf_counter()
        resp, error, kind = None, None, None
        try:
            resp = oh.http_json("POST", host.rstrip("/") + endpoint, payload, timeout=timeout)
            if not isinstance(resp, dict):
                error = f"non-JSON response: {str(resp)[:200]}"
                resp = None
        except oh.OllamaError as e:
            error, kind = f"{e.kind}: {e}", e.kind
        latency = time.perf_counter() - t0
        rec = build_rec(resp, latency, attempt, error)
        if error is not None:
            return rec, anomalies, kind
        why = oh.logprob_anomaly(rec) if check else None
        if why is None:
            return rec, anomalies, None
        anomalies.append((attempt, why))
    rec["error"] = (f"{ANOMALY_ERROR_PREFIX}{why} - the response broke the eval_count/logprobs rule "
                    f"on all {len(anomalies)} attempt(s) and is not stored as a clean record")
    return rec, anomalies, None


def is_anomaly_error(rec: dict) -> bool:
    """True for a record `draw_generation` gave up on after every retry."""
    return str(rec.get("error") or "").startswith(ANOMALY_ERROR_PREFIX)


def anomaly_summary(n_responses: int, n_retried: int, n_failed: int) -> str:
    """The one line every run prints about the anomaly, abort included."""
    if not n_responses:
        return ("logprob anomaly: none - every generation obeyed the eval_count/logprobs rule on "
                "its first request")
    return (f"logprob anomaly: {n_responses} response(s) broke the eval_count/logprobs rule and "
            f"were re-requested instead of stored; {n_retried} generation(s) came back clean on a "
            f"retry, {n_failed} ended as an error row after {ANOMALY_ATTEMPTS} attempt(s)")


def load_config(path: Path, args) -> dict:
    cfg = json.load(open(path, encoding="utf-8"))
    required = ("name", "model", "benchmark", "benchmark_path", "items_file", "route", "language", "options")
    missing = [k for k in required if k not in cfg]
    if missing:
        raise SystemExit(f"config {path}: missing keys {missing}")
    if args.host:
        cfg["host"] = args.host
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    if args.benchmark_path:
        cfg["benchmark_path"] = args.benchmark_path
    if args.items_file:
        cfg["items_file"] = args.items_file
    if args.timeout is not None:
        cfg["timeout_seconds"] = args.timeout
    if args.top_logprobs is not None:
        cfg["top_logprobs"] = args.top_logprobs
    if args.endpoint:
        cfg["endpoint"] = args.endpoint
    cfg.setdefault("host", "http://localhost:11434")
    cfg.setdefault("endpoint", "/api/generate")
    cfg.setdefault("answer_kind", cfg["route"])
    cfg.setdefault("logprobs", True)
    cfg.setdefault("top_logprobs", 0)
    cfg.setdefault("think", "auto")
    cfg.setdefault("seed", None)
    cfg.setdefault("keep_alive", None)
    cfg.setdefault("timeout_seconds", 300)
    cfg.setdefault("out_dir", f"data/ungated/{cfg['name']}")
    if cfg["endpoint"] not in ("/api/generate", "/api/chat"):
        raise SystemExit("endpoint must be /api/generate or /api/chat (never /v1)")
    return cfg


def load_items(cfg: dict, limit: int | None) -> list[tuple[str, dict]]:
    ids = [ln.strip() for ln in open(oh.resolve(cfg["items_file"]), encoding="utf-8") if ln.strip()]
    if limit:
        ids = ids[:limit]
    paths = cfg["benchmark_path"]
    if isinstance(paths, str):
        paths = [paths]
    rows: dict[str, dict] = {}
    for p in paths:
        rp = oh.resolve(p)
        if not rp.exists():
            raise SystemExit(
                f"benchmark file not found: {rp}\n"
                "Prepare it first (read-only MOME script, writes to data/benchmarks/):\n"
                f"  python scripts/prepare_benchmarks.py {cfg['benchmark']} {p}")
        for row in oh.read_jsonl(rp):
            rows[str(row["id"])] = row
    missing = [i for i in ids if i not in rows]
    if missing:
        raise SystemExit(f"{len(missing)} item ids of {cfg['items_file']} are not in {paths} "
                         f"(first: {missing[:5]}); the benchmark file must be the same one Phase 0 used")
    return [(i, rows[i]) for i in ids]


def build_payload(cfg: dict, prompt: str, think_flag, options_override: dict | None = None) -> dict:
    """The request body for one generation.

    `options_override` is how `run_pool.py` draws a sampled path: it merges on
    top of the config's `options` (temperature / top_p / seed) and nothing else
    changes, so a sampled request differs from the greedy one only in those
    keys - same model, same prompt, same `num_predict`, same logprob settings,
    same `think`.
    """
    options = dict(cfg["options"])
    if cfg.get("seed") is not None:
        options["seed"] = cfg["seed"]
    if options_override:
        options.update(options_override)
    payload: dict = {"model": cfg["model"], "stream": False, "options": options,
                     "logprobs": bool(cfg["logprobs"])}
    if cfg["logprobs"] and cfg["top_logprobs"]:
        payload["top_logprobs"] = int(cfg["top_logprobs"])
    if cfg["endpoint"] == "/api/generate":
        payload["prompt"] = prompt
    else:
        payload["messages"] = [{"role": "user", "content": prompt}]
    if think_flag is not None:
        payload["think"] = think_flag
    if cfg.get("keep_alive") is not None:
        payload["keep_alive"] = cfg["keep_alive"]
    return payload


def resolve_think(cfg: dict, capabilities: list[str]) -> bool | None:
    """Mirror MOME: `think: false` only when /api/show lists the "thinking"
    capability (mome/inference/ollama.py:138-139, 157-164)."""
    t = cfg.get("think", "auto")
    if t == "auto":
        return False if "thinking" in (capabilities or []) else None
    if t in (True, False):
        return t
    return None


def capture_env(cfg: dict, timeout: float) -> dict:
    host = cfg["host"]
    env = {"captured_at": oh.now_iso(), "host": host, "model": cfg["model"],
           "platform": oh.platform_info(), "ollama_version": None, "show": None,
           "tags_entry": None, "nvidia_smi": None}
    env["ollama_version"] = oh.server_version(host, timeout=min(timeout, 30))
    try:
        env["show"] = oh.show(host, cfg["model"], timeout=timeout)
    except oh.OllamaError as e:
        env["show"] = {"error": str(e)}
    try:
        env["tags_entry"] = oh.tags_entry(host, cfg["model"], timeout=timeout)
    except oh.OllamaError as e:
        env["tags_entry"] = {"error": str(e)}
    env["nvidia_smi"] = oh.nvidia_smi_query("name,memory.total,driver_version")
    return env


#: A cell's `campaign_meta.json` and `env.json` are COMMITTED files.  Pointing a
#: runner at a cell that already exists is a RESUME, and a resume must not dirty
#: them: `started_at` / `captured_at` say when the CELL was started, not when a
#: process last looked at it, and `argv` is the command that started it.  The old
#: code re-stamped all three before the resume calculation and before a single
#: generation, so one read-only glance at a finished cell left two modified files
#: behind and blocked the next run.  Now the first write owns those fields, this
#: invocation is APPENDED under `resumes`, and a document that comes out
#: identical to the file on disk is not written at all.
RESUMES_FIELD = "resumes"


def resume_document(path: Path, new: dict, frozen: str | tuple[str, ...],
                    entry: dict | None = None) -> tuple[dict, str]:
    """The document that belongs at `path`, and a one-line note of what happened.

    `frozen` names the fields that belong to the first write and are never
    overwritten (`started_at` and `argv`; `captured_at`).  Everything else is
    refreshed to this invocation, because `config` has to describe what is being
    generated right now -- and whatever that changed is named in the appended
    `resumes` entry together with the values it replaced, so a changed
    configuration is never silently swallowed.

    Entries carry no clock of their own on purpose: two identical resumes then
    build one identical entry, the document stops growing, and the file stays
    byte-for-byte what it was.
    """
    frozen = (frozen,) if isinstance(frozen, str) else tuple(frozen)
    old = None
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                old = json.load(f)
        except ValueError:
            old = None
    if not isinstance(old, dict):
        return dict(new), "first write"

    merged = dict(new)
    for name in frozen:
        if name in old:
            merged[name] = old[name]
    history = list(old.get(RESUMES_FIELD) or [])
    changed = sorted(k for k in set(merged) | set(old)
                     if k != RESUMES_FIELD and k not in frozen and merged.get(k) != old.get(k))
    entry = dict(entry or {})
    if changed:
        entry["changed"] = changed
        entry["previous"] = {k: old.get(k) for k in changed}
    if entry:
        #: the first write is the baseline a first resume is measured against;
        #: after that it is the last entry, so re-running the same command twice
        #: appends once and re-running it a hundred times still appends once.
        prior = history[-1] if history else {k: old.get(k) for k in entry}
        if entry != prior:
            history.append(entry)
    if history:
        merged[RESUMES_FIELD] = history
    note = "resume; kept first-written " + ", ".join(frozen)
    if changed:
        note += f"; CHANGED {', '.join(changed)}"
    return merged, note


def write_run_meta(path: Path, new: dict, frozen: str | tuple[str, ...],
                   entry: dict | None = None) -> str:
    """Write a run's metadata file idempotently.  Returns a one-line note."""
    doc, note = resume_document(path, new, frozen, entry)
    text = json.dumps(doc, indent=1, ensure_ascii=False)
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                if f.read() == text:
                    return note + "; file unchanged, not written"
        except OSError:
            pass
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return note + "; written"


def make_record(cfg: dict, item_id: str, prompt: str, template_id: str, payload: dict,
                resp: dict | None, latency: float, attempt: int, error: str | None,
                digest, server_version, path: dict | None = None) -> dict:
    """One `records.jsonl` row (schema.md section 1).

    `path` carries the pool fields (`path_index`, `path_kind`, `n_paths`,
    `sampling`) for a pool campaign and is None for a greedy one, whose rows
    stay byte-for-byte the shape they had before pools existed.
    """
    request = {k: payload[k] for k in ("options", "logprobs", "top_logprobs", "think", "keep_alive", "stream")
               if k in payload}
    rec = {
        "schema_version": SCHEMA_VERSION, "campaign": cfg["name"], "item_id": item_id,
        "benchmark": cfg["benchmark"], "backbone": cfg.get("backbone"), "model_tag": cfg["model"],
        "model_digest": digest, "server_version": server_version, "endpoint": cfg["endpoint"],
        "route": cfg["route"], "language": cfg["language"], "prompt_template_id": template_id,
        "template_sha256": prompts.template_sha256(template_id), "prompt_sha256": prompts.prompt_sha256(prompt),
        "prompt_chars": len(prompt), "request": request,
        "response_text": None, "thinking": None, "text_visible": None, "extracted_answer": None,
        "program_text": None, "prose_answer": None, "logprobs": None, "n_logprob_tokens": 0,
        "done": None, "done_reason": None, "ollama": None, "token_count": None, "truncated": None,
        "latency_seconds": round(latency, 4), "attempt": attempt, "error": error, "created_at": oh.now_iso(),
    }
    if path:
        rec.update(path)
    if resp is None:
        return rec
    if cfg["endpoint"] == "/api/generate":
        text = resp.get("response") or ""
        thinking = resp.get("thinking")
    else:
        msg = resp.get("message") or {}
        text = msg.get("content") or ""
        thinking = msg.get("thinking")
    visible = prompts.strip_think_blocks(text)
    kind = cfg["answer_kind"]
    rec["response_text"] = text
    rec["thinking"] = thinking or None
    rec["text_visible"] = visible
    rec["extracted_answer"] = extract.extract_final_answer(visible, kind)
    if cfg["route"] == "expression":
        rec["program_text"] = prompts.extract_code_block(visible)
        # `prose_answer` is read by its LABEL only for the templates that put the
        # answer behind one (`prompts.PROSE_ANSWER_LABEL_TEMPLATES`).  Every other
        # template -- `tir2_*` and the four committed cells included -- keeps the
        # region scrape it was recorded with, so re-running this function over any
        # committed generation reproduces its row byte for byte.
        if prompts.wants_prose_answer_label(template_id):
            line = prompts.parse_prose_answer_line(visible)
            rec["prose_answer"] = (extract.extract_final_answer(line, kind)
                                   if line is not None else None)
        else:
            rec["prose_answer"] = extract.extract_final_answer(
                prompts.prose_outside_code(visible), kind)
    lp = resp.get("logprobs")
    rec["logprobs"] = lp if isinstance(lp, list) else None
    rec["n_logprob_tokens"] = len(lp) if isinstance(lp, list) else 0
    rec["done"] = resp.get("done")
    rec["done_reason"] = resp.get("done_reason")
    rec["ollama"] = {k: resp.get(k) for k in ("total_duration", "load_duration", "prompt_eval_count",
                                               "prompt_eval_duration", "eval_count", "eval_duration",
                                               "prompt_eval_cached_count")}
    rec["token_count"] = resp.get("eval_count")
    cap = (payload.get("options") or {}).get("num_predict")
    ec = resp.get("eval_count")
    rec["truncated"] = bool((cap and ec is not None and ec >= cap) or resp.get("done_reason") == "length")
    # `think: false` stops Ollama RETURNING a `thinking` field; it does not stop
    # the model emitting a thought block in band, and a block emitted anyway
    # still spends `num_predict`.  When it eats the whole budget the row lands
    # as an ordinary `done_reason: "length"` with nothing visible, so the I
    # channel -- which is defined as proximity to the budget -- would attribute
    # the truncation to the difficulty of the derivation instead of to a thought
    # block nobody asked for.  So the runner NAMES the condition on the record.
    #
    # It is a flag and only a flag.  Nothing excludes, drops, filters or
    # reweights a flagged row, and no threshold reads it: whether the I channel
    # should exclude them is a decision about the measurement, it belongs to the
    # owner, and it has to be taken against canonical data -- a rule fitted to
    # the ten rows that motivated this would be fitted to its own evidence.
    #
    # `think_check.thought_block_leaked` is the one detector, reused rather than
    # reimplemented, so this cannot drift from what `--scan-records` reports of
    # the committed campaigns.  True / False / None (no logprobs to read) are
    # all recorded as themselves; a campaign that predates the field carries no
    # key at all, and absence means UNKNOWN, never false.
    rec[think_check.LEAK_FIELD] = think_check.thought_block_leaked(rec)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 1 logprob campaign runner (native Ollama routes only).")
    ap.add_argument("--config", required=True, help="configs/<cell>.json")
    ap.add_argument("--limit", type=int, default=None, help="only the first N item ids (smoke runs)")
    ap.add_argument("--dry-run", action="store_true", help="print the first prompt and payload, send nothing")
    ap.add_argument("--host", default=None, help="override host (default from config)")
    ap.add_argument("--out-dir", default=None, help="override output directory")
    ap.add_argument("--benchmark-path", nargs="+", default=None, help="override benchmark jsonl path(s)")
    ap.add_argument("--items-file", default=None)
    ap.add_argument("--timeout", type=float, default=None, help="per-request timeout seconds")
    ap.add_argument("--top-logprobs", type=int, default=None)
    ap.add_argument("--endpoint", choices=["/api/generate", "/api/chat"], default=None)
    ap.add_argument("--no-env", action="store_true", help="skip env.json capture (not for real runs)")
    ap.add_argument("--give-up-after", type=int, default=0,
                    help="skip, for this run only, any item that has at least "
                         "this many error rows and no error-free record "
                         "(0 disables). An item that fails deterministically "
                         "otherwise wedges the whole cell forever, caught "
                         "between the rule that a resume retries errors first "
                         "and the rule that three consecutive errors abort: "
                         "the same three errors lead every resume and abort "
                         "it again.")
    args = ap.parse_args()

    try:
        cfg = load_config(Path(args.config), args)
        items = load_items(cfg, args.limit)
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
    out_dir = oh.resolve(cfg["out_dir"])
    records_path = out_dir / "records.jsonl"

    if args.dry_run:
        item_id, row = items[0]
        prompt = prompts.build_prompt(template_id, row["question"])
        payload = build_payload(cfg, prompt, None)
        print(f"campaign: {cfg['name']}  model: {cfg['model']}  endpoint: {cfg['endpoint']}  host: {cfg['host']}")
        print(f"items: {len(items)}  template: {template_id}  out: {records_path}")
        print("---- first prompt (%s) ----" % item_id)
        print(prompt)
        print("---- payload (think decided from /api/show at run time) ----")
        print(json.dumps({k: v for k, v in payload.items() if k not in ("prompt", "messages")},
                         ensure_ascii=False, indent=1))
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
    env = None if args.no_env else capture_env(cfg, timeout)
    if env is not None:
        print("env.json: " + write_run_meta(out_dir / "env.json", env, "captured_at"), flush=True)
    try:
        show_info = env["show"] if env and isinstance(env.get("show"), dict) and "error" not in env["show"] \
            else oh.show(host, cfg["model"], timeout=timeout)
    except oh.OllamaError:
        show_info = {}
    capabilities = show_info.get("capabilities") or []
    think_flag = resolve_think(cfg, capabilities)
    digest = None
    tags = env.get("tags_entry") if env else None
    if tags is None:
        try:
            tags = oh.tags_entry(host, cfg["model"], timeout=timeout)
        except oh.OllamaError:
            tags = None
    if isinstance(tags, dict):
        digest = tags.get("digest")

    # campaign_meta.json: config copy, template text/hash, identity check vs MOME source
    try:
        import check_templates
        template_check = check_templates.compare()
    except Exception as e:  # noqa: BLE001 - the check is informative, never blocking
        template_check = {"error": str(e)}
    table, lang = prompts.TEMPLATES[template_id]
    meta = {"config": cfg, "config_path": str(Path(args.config)), "argv": sys.argv[1:],
            "prompt_template_id": template_id, "prompt_template_text": table[lang],
            "template_sha256": prompts.template_sha256(template_id), "template_check": template_check,
            "think_sent": think_flag, "model_capabilities": capabilities, "n_items_requested": len(items),
            "started_at": oh.now_iso(), "schema_version": SCHEMA_VERSION}
    print("campaign_meta.json: "
          + write_run_meta(out_dir / "campaign_meta.json", meta, ("started_at", "argv"),
                           {"argv": sys.argv[1:]}), flush=True)

    # resume: skip ids with an error-free record; count attempts
    existing = oh.read_jsonl(records_path)
    done_ids = {r["item_id"] for r in existing if r.get("error") is None}
    attempts: dict[str, int] = {}
    error_rows: dict[str, int] = {}
    for r in existing:
        attempts[r["item_id"]] = max(attempts.get(r["item_id"], 0), int(r.get("attempt") or 1))
        if r.get("error") is not None:
            error_rows[r["item_id"]] = error_rows.get(r["item_id"], 0) + 1
    given_up: set[str] = set()
    if args.give_up_after:
        given_up = {i for i, c in error_rows.items()
                    if c >= args.give_up_after and i not in done_ids}
    todo = [(i, row) for i, row in items if i not in done_ids and i not in given_up]
    print(f"{len(items)} items; {len(done_ids)} already done; {len(todo)} to generate; "
          f"think={think_flag}; top_logprobs={cfg['top_logprobs'] if cfg['logprobs'] else 'off'}"
          + (f"; {len(given_up)} given up (--give-up-after {args.give_up_after}: "
             f"{sorted(given_up)[:5]}{'...' if len(given_up) > 5 else ''})" if given_up else ""),
          flush=True)

    conflict = oh.packed_conflict(records_path) if todo else None
    if conflict:
        print(f"ERROR: {conflict}", file=sys.stderr)
        return 4

    n_ok = n_err = n_nolp = 0
    #: generations whose token stream carried a `<|channel>thought` block anyway, and how
    #: many of those came back with nothing visible at all.  A COUNT, printed live so a leak
    #: that eats a whole budget is visible while the run happens -- never a filter: no
    #: generation is dropped, excluded or reweighted for it, here or anywhere downstream.
    #: `run_pool.py` counts the same thing the same way.
    n_leak = n_leak_empty = 0
    consecutive_conn = 0
    consecutive_anom = 0
    n_anom_resp = n_anom_retried = n_anom_failed = 0
    total_eval = 0
    latencies: list[float] = []
    # nothing to generate -> no file is opened, so a resumed run cannot leave an
    # empty records.jsonl shadowing a packed records.jsonl.gz
    writer = open(records_path, "a", encoding="utf-8") if todo else contextlib.nullcontext(None)
    with writer as out:
        for k, (item_id, row) in enumerate(todo, 1):
            prompt = prompts.build_prompt(template_id, row["question"])
            payload = build_payload(cfg, prompt, think_flag)
            base_attempt = attempts.get(item_id, 0) + 1

            def build(resp, latency, attempt, error, _id=item_id, _prompt=prompt, _payload=payload):
                return make_record(cfg, _id, _prompt, template_id, _payload, resp, latency,
                                   attempt, error, digest, version)

            # One request when the response is complete - the same bytes, the
            # same record, the same `attempt` as before; more only when the
            # server answers with a non-final object (see draw_generation).
            rec, anom, err_kind = draw_generation(host, cfg["endpoint"], payload, timeout,
                                                  base_attempt, build)
            if err_kind is None:
                consecutive_conn = 0
            elif err_kind == "connection":
                consecutive_conn += 1
            latency = rec["latency_seconds"]
            for a, why in anom:
                n_anom_resp += 1
                print(f"[{k}/{len(todo)}] {item_id}: ANOMALY({why}) on attempt {a} - the response is "
                      "not stored; re-requesting", flush=True)
            if is_anomaly_error(rec):
                n_anom_failed += 1
                consecutive_anom += 1
            elif rec["error"] is None:
                if anom:
                    n_anom_retried += 1
                consecutive_anom = 0
            error = rec["error"]
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            if error:
                n_err += 1
                print(f"[{k}/{len(todo)}] {item_id}: ERROR {error}", flush=True)
                if consecutive_conn >= CONSECUTIVE_CONN_ABORT:
                    print(f"ABORT: {consecutive_conn} consecutive connection failures - is the server up? "
                          "Re-run the same command to resume.", file=sys.stderr)
                    return 3
                if consecutive_anom >= CONSECUTIVE_ANOMALY_ABORT:
                    print(anomaly_summary(n_anom_resp, n_anom_retried, n_anom_failed), flush=True)
                    print(f"ABORT: {consecutive_anom} generations in a row broke the eval_count/logprobs "
                          f"rule on all {ANOMALY_ATTEMPTS} attempts - the server is answering with "
                          "non-final responses. Re-run the same command to resume.", file=sys.stderr)
                    return 3
                continue
            n_ok += 1
            latencies.append(latency)
            total_eval += int(rec["token_count"] or 0)
            if rec["n_logprob_tokens"] == 0:
                n_nolp += 1
            leaked = think_check.record_leak_state(rec) == think_check.LEAKED
            if leaked:
                n_leak += 1
                # the case that matters most: the block ran to the cap and the generation
                # came back with nothing visible, yet the row looks like an ordinary
                # length truncation
                if not (rec["text_visible"] or "").strip():
                    n_leak_empty += 1
            print(f"[{k}/{len(todo)}] {item_id}: tokens={rec['token_count']} logprobs={rec['n_logprob_tokens']} "
                  f"{latency:.1f}s answer={rec['extracted_answer']!r}"
                  + (f" program={'yes' if rec['program_text'] else 'no'}" if cfg["route"] == "expression" else "")
                  + (" TRUNCATED" if rec["truncated"] else "")
                  + (" THINK-BLOCK-LEAK" if leaked else ""), flush=True)

    mean_lat = sum(latencies) / len(latencies) if latencies else 0.0
    print(f"done: {n_ok} generated, {n_err} errors, {len(done_ids)} skipped (resume); "
          f"eval tokens {total_eval}; mean wall latency {mean_lat:.2f}s; records -> {records_path}", flush=True)
    print(anomaly_summary(n_anom_resp, n_anom_retried, n_anom_failed), flush=True)
    if n_nolp:
        print(f"WARNING: {n_nolp} of {n_ok} responses carried no logprobs (check Ollama version / model)",
              flush=True)
    if n_leak:
        print(f"THOUGHT-BLOCK LEAK: {n_leak} of {n_ok} generation(s) emitted a `<|channel>thought` "
              f"block although `think: false` was sent; {n_leak_empty} of those returned nothing "
              f"visible at all. Each is FLAGGED on its record as `{think_check.LEAK_FIELD}` and "
              "NOTHING else: no generation is dropped, excluded or reweighted, and no threshold "
              "reads the flag. It is named because a block that eats the budget lands as an "
              "ordinary done_reason \"length\", and the I channel is proximity to that budget -- "
              "so such a row would otherwise report the wrong cause.", flush=True)
    elif n_ok:
        print(f"no thought-block leak: none of {n_ok} generation(s) carried a `<|channel>thought` "
              "block (`think: false` was obeyed in band)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
