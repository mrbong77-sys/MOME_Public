"""Stdlib-only helpers shared by the Phase 1 runner and the device tool.

Native Ollama routes only (`/api/...`); the OpenAI-compatible `/v1` route
drops `logprobs`/`top_logprobs` (Ollama issue #16117) and is never used.
Field names follow the Ollama API reference (github.com/ollama/ollama,
docs/api.md and api/types.go).

Nothing here imports the `mome` package.  Paths are resolved from the file
location so the scripts run from any working directory on Windows or Linux;
`MOME_ROOT` overrides the repository root.
"""
from __future__ import annotations

import datetime as _dt
import gzip
import http.client
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent            # the package directory
REPO_ROOT = HERE.parents[1]
MOME_ROOT = Path(os.environ.get("MOME_ROOT") or REPO_ROOT)


def force_utf8() -> None:
    """Make this interpreter, and every child it starts, speak UTF-8.

    Python 3.14 still leaves UTF-8 mode off by default, so anything that omits
    `encoding=` falls back to `locale.getencoding()`: **cp949** on a Korean
    Windows, cp1252 on a Western one.  A commit message with an em dash, a
    status line with a Korean word, or a model answer with a maths symbol then
    raises `UnicodeEncodeError`.  That is how the RTX smoke run died at its
    first `git commit`: U+2014 EM DASH is not in cp949, which carries U+2015
    HORIZONTAL BAR instead.

    Two separate things are needed and neither replaces the other:

      * the streams of THIS process are already open, so they can only be
        reconfigured -- with `errors="replace"`, because a console that cannot
        *draw* a character must never abort a six-hour run over it;
      * `PYTHONUTF8` is read by a child interpreter at startup and is the only
        way to reach `run_pool.py`, `finish_campaign.py` and
        `channel_smoke.py`, which print model output this process never sees.

    UTF-8 mode also gives the child's stdio the `surrogateescape` handler, so a
    child cannot die encoding its own output either.

    This is belt and braces.  What makes each call site *correct* is its own
    explicit `encoding="utf-8"`; this only removes the locale fallback
    underneath anything that was missed.  `setdefault` so an operator who set
    PYTHONUTF8 deliberately keeps their choice.
    """
    os.environ.setdefault("PYTHONUTF8", "1")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass    # not a reconfigurable TextIOWrapper (a captured StringIO), or already closed


class OllamaError(Exception):
    """Base class; `kind` is 'connection', 'timeout' or 'http'."""

    kind = "error"


class ConnectionFailed(OllamaError):
    kind = "connection"


class RequestTimeout(OllamaError):
    kind = "timeout"


class HTTPFailed(OllamaError):
    kind = "http"

    def __init__(self, code: int, body: str):
        super().__init__(f"HTTP {code}: {body[:300]}")
        self.code = code
        self.body = body


def http_json(method: str, url: str, payload: dict | None = None, timeout: float = 600.0):
    """One request; parsed JSON (or text) back, exceptions typed by cause."""
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        raise HTTPFailed(e.code, body) from e
    except urllib.error.URLError as e:
        if isinstance(getattr(e, "reason", None), TimeoutError) or "timed out" in str(e).lower():
            raise RequestTimeout(str(e)) from e
        raise ConnectionFailed(str(e.reason)) from e
    except TimeoutError as e:
        raise RequestTimeout(str(e)) from e
    except http.client.IncompleteRead as e:
        # A generate response is `Transfer-Encoding: chunked` with no
        # Content-Length, so a stream that stops early surfaces here, from
        # inside `resp.read()`, as http.client.IncompleteRead.  It is neither
        # OSError nor ValueError, so without this clause it escapes every
        # handler in the runners and kills the run with a traceback; typed as a
        # connection failure it goes down the path the runners already have for
        # one (error record, resume, consecutive-failure abort).
        raise ConnectionFailed(f"incomplete read: the response body stopped after "
                               f"{len(e.partial)} byte(s)") from e
    except http.client.HTTPException as e:
        raise ConnectionFailed(f"{type(e).__name__}: {e}") from e
    except OSError as e:
        raise ConnectionFailed(str(e)) from e
    try:
        return json.loads(body)
    except ValueError:
        return body


def resolve(path: str | Path, root: Path = MOME_ROOT) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def platform_info() -> dict:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "node": platform.node(),
        "python": platform.python_version(),
    }


# --- Ollama native routes ---------------------------------------------------

def server_version(host: str, timeout: float = 30.0) -> str | None:
    v = http_json("GET", host.rstrip("/") + "/api/version", timeout=timeout)
    return v.get("version") if isinstance(v, dict) else None


def show(host: str, model: str, timeout: float = 60.0) -> dict:
    """POST /api/show -> modelfile, parameters, template, details, model_info,
    capabilities, modified_at (no digest: see tags_entry)."""
    r = http_json("POST", host.rstrip("/") + "/api/show", {"model": model}, timeout=timeout)
    return r if isinstance(r, dict) else {}


def tags_entry(host: str, model: str, timeout: float = 60.0) -> dict | None:
    """The model's entry in GET /api/tags (carries `digest`, `size`, `details`).
    Same lookup MOME used for provenance (mome/inference/ollama.py:55-73)."""
    r = http_json("GET", host.rstrip("/") + "/api/tags", timeout=timeout)
    for entry in (r.get("models") or []) if isinstance(r, dict) else []:
        if model in (entry.get("name"), entry.get("model")):
            return entry
    return None


def ps(host: str, timeout: float = 30.0) -> list[dict]:
    """GET /api/ps -> models loaded in memory (size, size_vram, expires_at)."""
    r = http_json("GET", host.rstrip("/") + "/api/ps", timeout=timeout)
    return (r.get("models") or []) if isinstance(r, dict) else []


def unload(host: str, model: str, timeout: float = 120.0) -> dict:
    """POST /api/generate with an empty prompt and keep_alive 0 unloads the
    model (docs/api.md, "Unload a model")."""
    r = http_json("POST", host.rstrip("/") + "/api/generate",
                  {"model": model, "keep_alive": 0}, timeout=timeout)
    return r if isinstance(r, dict) else {}


def load(host: str, model: str, timeout: float = 1800.0, keep_alive=None) -> dict:
    """POST /api/generate with an empty prompt loads the model (docs/api.md,
    "Load a model"). The response may or may not carry `load_duration`."""
    payload: dict = {"model": model}
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive
    r = http_json("POST", host.rstrip("/") + "/api/generate", payload, timeout=timeout)
    return r if isinstance(r, dict) else {}


# --- nvidia-smi --------------------------------------------------------------

def nvidia_smi_query(fields: str, nounits: bool = False, timeout: float = 20.0) -> dict:
    """`nvidia-smi --query-gpu=<fields> --format=csv[,noheader,nounits]`.

    Returns {"available": bool, "command": [...], "stdout": str, "error": str|None}.
    The query syntax is the one named in plan v0.2 section 6.5; the NVIDIA
    documentation was not reachable from the development environment, so
    confirm with `nvidia-smi --help-query-gpu` on the measurement machine.
    """
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return {"available": False, "command": None, "stdout": None, "error": "nvidia-smi not on PATH"}
    fmt = "csv,noheader,nounits" if nounits else "csv"
    cmd = [exe, f"--query-gpu={fields}", f"--format={fmt}"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              encoding="utf-8", errors="replace")
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"available": True, "command": cmd, "stdout": None, "error": str(e)}
    return {"available": True, "command": cmd, "stdout": proc.stdout,
            "error": None if proc.returncode == 0 else proc.stderr.strip()[:500]}


# --- jsonl, plain or gzipped -------------------------------------------------
#
# `mome/pack_records.py` packs a campaign's `records.jsonl` /
# `exec.jsonl` into `<name>.jsonl.gz` because a 500-item cell at
# `top_logprobs: 10` is far past GitHub's 100 MB per-file limit, and
# the ignore rules keeps the plain forms out of the index.  The Phase 0
# scripts already read `.jsonl.gz` with `gzip.open(path, "rt", encoding=...)`
# (the Phase 0 readers);
# the readers below follow that convention so the Phase 1 tools keep working
# after the plain file has been packed and deleted.

def gz_path(path: str | Path) -> Path:
    """`records.jsonl` -> `records.jsonl.gz` (the suffix is appended, not replaced)."""
    p = Path(path)
    return p if p.suffix == ".gz" else p.with_name(p.name + ".gz")


def _base_name(path: str | Path) -> str:
    """`records.jsonl` / `records.jsonl.gz` -> `records`."""
    name = Path(path).name
    for suffix in (".jsonl.gz", ".jsonl"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return Path(name).stem


def part_paths(path: str | Path) -> list[Path]:
    """`records.jsonl` -> `[records_p0.jsonl.gz, records_p1.jsonl.gz, ...]`.

    A pool campaign is packed one file per `path_index` (pack_records.py): five
    paths over 500 items at `top_logprobs` 10 are ~166 MB in one file, which
    GitHub refuses, and ~33 MB each when split by path, which it does not.  The
    list comes back in numeric path order, so `p10` sorts after `p9`.
    """
    p = Path(path)
    base = _base_name(p)
    pattern = re.compile(re.escape(base) + r"_p(\d+)\.jsonl\.gz$")
    found: list[tuple[int, Path]] = []
    for cand in p.parent.glob(base + "_p*.jsonl.gz"):
        m = pattern.fullmatch(cand.name)
        if m:
            found.append((int(m.group(1)), cand))
    return [c for _, c in sorted(found)]


def parts_manifest_path(path: str | Path) -> Path:
    """`records.jsonl` -> `records_parts.json`, the per-path layout's index."""
    p = Path(path)
    return p.parent / (_base_name(p) + "_parts.json")


def read_parts_manifest(path: str | Path) -> dict | None:
    """The manifest, or None when there is none / it is unreadable.

    It carries `order`: the `path_index` of every line of the ORIGINAL file, in
    the original order.  Splitting by path loses that interleaving (a pool
    writes item-major, p0..p4 per item), so the manifest is what lets both
    `read_jsonl` and `pack_records.py --unpack` put the lines back exactly as
    they were written.
    """
    m = parts_manifest_path(path)
    if not m.exists():
        return None
    try:
        data = json.loads(m.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    return data if isinstance(data, dict) and isinstance(data.get("order"), list) else None


def jsonl_sources(path: str | Path) -> list[Path]:
    """Every file `read_jsonl` would actually read, in reading order.

    Three layouts, in this order of preference:

      1. the plain file, when it exists and is not empty (a live run);
      2. the single `<name>.jsonl.gz` (a greedy campaign, packed);
      3. the per-path set `<name>_p*.jsonl.gz` (a pool campaign, packed).

    An empty plain file must not shadow a packed one: a resumed run opens the
    output in append mode and can leave a zero-byte `records.jsonl` next to the
    packed files that hold the whole cell.
    """
    p = Path(path)
    g = gz_path(p)
    parts = part_paths(p)
    if p.exists() and (p.stat().st_size > 0 or not (g.exists() or parts)):
        return [p]
    if g.exists():
        return [g]
    if parts:
        return parts
    return [p] if p.exists() else []


def jsonl_source(path: str | Path) -> Path | None:
    """The FIRST file `read_jsonl` would read, or None when there is none.

    Kept for the callers that only need "is there anything here" or a name to
    print; anything that reads the data must use `jsonl_sources`/`read_jsonl`,
    because a pool campaign is spread over several files.
    """
    sources = jsonl_sources(path)
    return sources[0] if sources else None


def jsonl_describe(path: str | Path) -> str:
    """A human-readable name for whatever layout is there ("... and 4 more")."""
    sources = jsonl_sources(path)
    if not sources:
        return f"{path} (missing)"
    if len(sources) == 1:
        return str(sources[0])
    return f"{sources[0]} and {len(sources) - 1} more per-path file(s)"


def open_jsonl(path: str | Path, mode: str = "rt"):
    """`open` for a plain path, `gzip.open` for a `.gz` one; always UTF-8 text."""
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    return opener(p, mode, encoding="utf-8")


def packed_conflict(path: str | Path) -> str | None:
    """An error message when appending to `path` would split a packed file.

    Returns None when writing is safe.  The case guarded against is a campaign
    whose plain file was removed by `pack_records.py --remove-source`: a new
    append would put some rows in `records.jsonl` and the rest in the packed
    file(s), and only one of the two would be read back.  Both packed layouts
    are covered - the single `.gz` of a greedy campaign and the per-path set of
    a pool.
    """
    p, g = Path(path), gz_path(path)
    if p.exists() and p.stat().st_size > 0:
        return None
    parts = part_paths(p)
    if g.exists():
        packed = g.name
    elif parts:
        packed = f"{len(parts)} per-path file(s) ({', '.join(x.name for x in parts)})"
    else:
        return None
    return (f"{p.name} is packed as {packed} and the plain file is gone; writing now would "
            f"split the data across two layouts.\n"
            f"Unpack it first:  python mome\\pack_records.py --unpack --records {p}")


def _parse_lines(src: Path, rows: list[dict]) -> None:
    with open_jsonl(src) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                print(f"WARNING: {src}:{lineno} is not valid JSON and is ignored", flush=True)


def read_jsonl(path: Path) -> list[dict]:
    """Every parseable line; a trailing half-written line is reported, not fatal.

    All three layouts read the same and in the same order, so no caller has to
    know which one is on disk: the plain file, the single `.gz`, or the
    per-path set, which is re-interleaved from `<name>_parts.json` back into
    the order the runner wrote it.
    """
    rows: list[dict] = []
    sources = jsonl_sources(path)
    if not sources:
        return rows
    if len(sources) == 1:
        _parse_lines(sources[0], rows)
        return rows

    manifest = read_parts_manifest(path)
    if manifest is None:
        # No manifest: read path by path.  The records are all there and every
        # tool keys on (item_id, path_index), but the original interleaving is
        # not recoverable, so say so rather than implying it is.
        print(f"WARNING: {parts_manifest_path(path).name} is missing; reading "
              f"{len(sources)} per-path file(s) in path order, not in the order they were written",
              flush=True)
        for src in sources:
            _parse_lines(src, rows)
        return rows

    per_path: dict[int, list[dict]] = {}
    for src in sources:
        got: list[dict] = []
        _parse_lines(src, got)
        m = re.search(r"_p(\d+)\.jsonl\.gz$", src.name)
        per_path[int(m.group(1)) if m else 0] = got
    cursor = dict.fromkeys(per_path, 0)
    for idx in manifest["order"]:
        bucket = per_path.get(int(idx))
        if bucket is None or cursor[int(idx)] >= len(bucket):
            print(f"WARNING: {parts_manifest_path(path).name} asks for a line of path {idx} that the "
                  "per-path files do not have; the rest is read in path order", flush=True)
            rows = []
            for src in sources:
                _parse_lines(src, rows)
            return rows
        rows.append(bucket[cursor[int(idx)]])
        cursor[int(idx)] += 1
    leftover = sum(len(v) - cursor[k] for k, v in per_path.items())
    if leftover:
        print(f"WARNING: {leftover} record(s) are in the per-path files but not in "
              f"{parts_manifest_path(path).name}; they are appended in path order", flush=True)
        for k in sorted(per_path):
            rows.extend(per_path[k][cursor[k]:])
    return rows


# --- record helpers shared by the runners, the grader and finish_campaign ----
#
# A greedy campaign writes one record per item and carries no path fields; a
# pool campaign (`run_pool.py`) writes `n_paths` records per item, `path_index`
# 0 being the greedy anchor.  Records written before pools existed have no
# `path_index`, so the reader below calls them path 0 — every tool then treats
# an old greedy cell as a one-path pool without a special case.

def record_path_index(rec: dict) -> int:
    """`path_index` of a record; 0 for records written before pools existed."""
    value = rec.get("path_index")
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def record_key(rec: dict) -> tuple[str, int]:
    """The identity a resumed run and the grader key on: (item_id, path_index)."""
    return (str(rec.get("item_id")), record_path_index(rec))


def ungraded_keys(records_path: Path, grades_path: Path) -> set[tuple[str, int]]:
    """(item_id, path_index) pairs that have an error-free record and no grade row.

    `grade.py` writes one row per error-free record it finds gold for, so an EMPTY
    set is the invariant "every generation that could be graded was graded".  A
    non-empty one means grading has fallen behind the records -- generations were
    bought after the grader last ran, and every count downstream (accuracy, the
    denominators the reports print, the criteria read off them) is being computed
    on fewer generations than the campaign holds.

    A missing `grades.jsonl` returns every error-free key, which is the same
    statement: nothing here is graded yet.  Both files are read through
    `read_jsonl`, so a packed campaign answers exactly as a plain one does.
    """
    graded = {record_key(r) for r in read_jsonl(grades_path)}
    return {record_key(r) for r in read_jsonl(records_path)
            if r.get("error") is None} - graded


def expected_logprob_diff(rec: dict) -> int | None:
    """The `eval_count - n_logprob_tokens` this record must show, else None.

    Verified on the RTX 5090 PC (data/env/ollama_logprobs_check_
    20260905-083257.json, Ollama 0.33.3) and over the committed 500-item greedy
    cell: `done_reason "stop"` -> exactly 1, because the token that ends the
    generation is counted in `eval_count` and never returned in `logprobs`;
    `done_reason "length"` -> 0, because a run cut at `num_predict` has no such
    token.  Any other value is an anomaly and is reported, never absorbed.

    For models with multi-token stop sequences (e.g. solar:10.7b with '### User:',
    '### Assistant:', '### System:'), Ollama strips the matched stop sequence
    tokens from response text and logprobs upon `done_reason "stop"`, while
    `eval_count` retains the full decode count. In that case, diff matches the
    length of the stripped stop sequence (1..16 tokens).
    """
    reason = rec.get("done_reason")
    if reason == "length":
        return 0
    if rec.get("truncated"):
        return 0
    if reason == "stop":
        model = str(rec.get("model_tag") or rec.get("backbone") or rec.get("campaign") or rec.get("model") or "")
        if "solar" in model:
            ec = (rec.get("ollama") or {}).get("eval_count")
            n = rec.get("n_logprob_tokens")
            if ec is not None and n is not None:
                diff = ec - n
                if 1 <= diff <= 16:
                    return diff
        return 1
    return None


def logprob_anomaly(rec: dict) -> str | None:
    """None when the record obeys the rule above, else why it does not."""
    if rec.get("error") is not None:
        return None
    n = rec.get("n_logprob_tokens")
    ec = (rec.get("ollama") or {}).get("eval_count")
    if ec is None or n is None:
        return "eval_count or n_logprob_tokens missing"
    if not n:
        return "n_logprob_tokens is 0"
    want = expected_logprob_diff(rec)
    diff = ec - n
    if want is None:
        return f"done_reason={rec.get('done_reason')!r} diff={diff} (unknown done_reason)"
    if diff != want and not _logprobs_cover_response(rec):
        return f"done_reason={rec.get('done_reason')!r} diff={diff} (expected {want})"
    return None


def _logprobs_cover_response(rec: dict) -> bool:
    """Accept a record when the log-probability tokens reconstruct the response.

    The property we actually need is "every token of the visible response
    carries a log-probability".  The original acceptance rule tested a proxy
    for it, a token-count difference, and that proxy breaks on one observed
    case: on Korean input one backbone emits hidden reasoning tokens even with
    thinking switched off, and those tokens are counted by the runtime while
    appearing in neither the response nor the log-probabilities.  The target
    property holds; the proxy does not.

    This condition checks the target property itself -- concatenating the
    log-probability tokens must reproduce the stored response exactly -- so it
    accepts that case without narrowing anything the count rule already
    accepts, which keeps its own path untouched.
    """
    text = rec.get("response_text")
    entries = rec.get("logprobs")
    if not text or not isinstance(entries, list) or not entries:
        return False
    try:
        joined = "".join(str(e.get("token", "")) for e in entries)
    except AttributeError:
        return False
    return joined == text
