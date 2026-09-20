#!/usr/bin/env python3
r"""Grader step: join records + exec + benchmark gold into grades.jsonl.

Kept out of the generation records on purpose (schema.md, section 3).  The
final answer per item follows the MOME `tir` / `format` rules:

  * numeric route: `extracted_answer`                      (method "format")
  * expression route, execution ok with an answer line:
        exec.answer_line                                   (method "tir", verified)
  * expression route otherwise: `prose_answer`             (method "tir-fallback")
    (mome/solver/pipeline.py:149-194)

Equivalence is the MOME grader chain re-implemented from the read-only
source `mome/grader/equivalence.py:38-58, 69-184` (numeric closeness first,
then math-verify if installed, then SymPy, then normalized strings; the
doubled-backslash retry of `equivalent`).  math-verify 0.9.0 is the version
the campaigns used.

On top of that chain, and only after it has already compared the answer
exactly as delivered and said no, comes the **predicted-answer notation
normalisation** (`NOTATION_RULES`): the `expression` route delivers what a
Python program PRINTED, and `x**5 - x**4`, `(1-12j)`, `Matrix([[a],[b]])`,
`137 1/2`, `1/tan(x)` are the MATH-500 gold answers `x^5 - x^4`, `1 - 12i`, a
`pmatrix`, `137\frac{1}{2}`, `\cot x` written in the other notation.  It
rewrites the PREDICTED answer only, never the gold, and never in a way that
changes a value; being a retry rather than a precondition, it cannot turn a
correct grade into a wrong one.  the regrade audit
enumerates every grade it moved on the committed cells, and
`mome/test_notation.py` is the adversarial test that it moves
nothing else.  **Every committed cell is graded on this one standard** --
`mome/regrade.py` re-grades all of them together, so no two numbers
in the reports come from two different graders.

**Without math-verify this script refuses to grade.**  The chain still runs
end to end when the package is missing, and it writes a `grades.jsonl` that
looks exactly like a real one -- same fields, same shape, an accuracy that
reads as a measurement.  It is not one: the fallback cannot read the LaTeX
shapes MOME lists at `mome/evalkit/extraction.py:398-405` (currency,
thousands separators, `\text{}` units), so it is a silently *stricter*
grader and the number comes out low.  `--allow-fallback` grades anyway for
development, marks every row `grader: "chain-fallback"`, and drops a
`GRADES_ARE_FALLBACK.txt` note beside the file so the numbers cannot be
mistaken for a measurement.

Usage:
    py mome\grade.py --records data\ungated\gsm8k_gemma4-e2b\records.jsonl \
        --benchmark data\benchmarks\gsm8k-test.jsonl

Exit codes: 0 graded, 8 math-verify is not importable in this interpreter.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract  # noqa: E402
import ollama_http as oh  # noqa: E402
import prompts  # noqa: E402  (wants_check_lines: which records carry a second derivation)

# reimplemented from mome/grader/equivalence.py:30,35
_MV_TIMEOUT = 5 if os.name == "posix" else None
_REL_TOL = Fraction(1, 10**9)

FALLBACK = "chain-fallback"
MARKER_NAME = "GRADES_ARE_FALLBACK.txt"
RC_NO_GRADER = 8


# --- which grader is actually live -----------------------------------------

_PROBE: tuple[str | None, str | None] | None = None


def probe_math_verify() -> tuple[str | None, str | None]:
    """`(version, None)` when the math-verify API is usable here, `(None, error)` when not.

    Two questions, deliberately kept apart, because conflating them is how a
    grader lies:

    1. **Can the API be imported and called?**  That, and only that, decides
       whether real grading happens.  `parse` and `verify` are imported by name,
       so a package that is present but broken is refused instead of being
       labelled as working while `_try_math_verify` swallows the same exception
       and grades through the fallback.  Every failure counts here, not only
       ImportError; MOME's `grader_backend`
       (mome/evalkit/extraction.py:407-429) catches ImportError alone.

    2. **Which version is it?**  A label, never a verdict.  The distribution
       name has a hyphen (`math-verify`), the module an underscore
       (`math_verify`), and the module exposes **no `__version__` attribute at
       all** -- reading one raises AttributeError.  So the version comes from
       `importlib.metadata.version("math-verify")`, falls back to a `__version__`
       if some future build grows one, and finally to `"unknown"`.  None of
       those three outcomes can downgrade the grader.
    """
    global _PROBE
    if _PROBE is None:
        try:
            from math_verify import parse, verify  # noqa: F401  the API grading actually calls
        except Exception as exc:  # noqa: BLE001
            _PROBE = (None, f"{type(exc).__name__}: {exc}")
        else:
            _PROBE = (_version_label(), None)
    return _PROBE


def _version_label() -> str:
    """Best-effort version string.  Cannot fail into `chain-fallback`."""
    try:
        from importlib.metadata import version as _version
        found = _version("math-verify")          # distribution name, hyphenated
        if found:
            return str(found)
    except Exception:  # noqa: BLE001
        pass
    try:
        import math_verify
        found = getattr(math_verify, "__version__", None)   # absent in 0.9.0
        if found:
            return str(found)
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def grader_name() -> str:
    version, _ = probe_math_verify()
    return f"math_verify-{version}" if version else FALLBACK


def refusal_text(allow_fallback: bool) -> str:
    """What to print when math-verify is not importable in *this* interpreter."""
    _, err = probe_math_verify()
    exe = sys.executable or "(unknown interpreter)"
    head = ("WARNING: grading WITHOUT math-verify (--allow-fallback)."
            if allow_fallback else
            "ERROR: math-verify is not importable, so this run will not grade.")
    tail = ("  The rows will say grader: \"chain-fallback\" and a "
            f"{MARKER_NAME} note goes next to grades.jsonl.\n"
            "  Numbers graded this way are for development only; do not report them."
            if allow_fallback else
            "  Grading anyway would write a normal-looking grades.jsonl whose accuracy is not\n"
            "  a measurement: the fallback chain cannot read currency, thousands separators,\n"
            "  percent signs or a trailing \\text{unit}, so it grades strictly low.\n"
            "  Use --allow-fallback only for development.")
    return (
        f"\n{head}\n"
        f"  running interpreter : {exe}\n"
        f"  python version      : {sys.version.splitlines()[0]}\n"
        f"  import error        : {err}\n"
        f"  fix                 : python -m pip install math-verify==0.9.0\n"
        f"                        Install it with THIS interpreter:\n"
        f'                        "{exe}" -m pip install math-verify==0.9.0\n'
        "\n"
        "  On Windows `py` and a conda `python` are usually DIFFERENT interpreters.  The `py`\n"
        "  launcher picks the newest registered Python (C:\\Python3xx\\), not the active conda\n"
        "  environment, so a `pip install` into conda base is invisible to `py` -- the package is\n"
        "  really installed and this interpreter really cannot see it.  Use `python`, NOT `py`,\n"
        "  for every script here:\n"
        "\n"
        "    python mome\\finish_campaign.py --dir \\data\\phase1\\<campaign> --regrade\n"
        "\n"
        "  Check which interpreter has the package with:\n"
        "    python -c \"import sys, importlib.metadata as m; "
        "print(sys.executable, m.version('math-verify'))\"\n"
        f"{tail}\n"
    )


def write_marker(out_path: Path, n_rows: int) -> Path:
    """Leave an unmistakable note beside a fallback-graded grades.jsonl."""
    marker = out_path.parent / MARKER_NAME
    marker.write_text(
        f"The file {out_path.name} in this directory was graded WITHOUT the "
        f"symbolic comparator.\nIts accuracy is not a measurement.\n"
        "\n"
        f"  grader        : {FALLBACK}\n"
        f"  rows graded   : {n_rows}\n"
        f"  graded at     : {oh.now_iso()}\n"
        f"  interpreter   : {sys.executable}\n"
        f"                  {sys.version.splitlines()[0]}\n"
        "\n"
        "Without math-verify, grade.py falls back to numeric comparison, then\n"
        "SymPy, then normalized string comparison. That is a development\n"
        "approximation, not the grader the campaigns used, so no accuracy\n"
        "produced here may be carried into any document as a measurement.\n"
        "\n"
        "To regrade, use the interpreter that actually has math-verify:\n"
        "\n"
        "    python -m pip install math-verify==0.9.0\n"
        f"    python finish_campaign.py --dir {out_path.parent} --regrade --no-git\n"
        "\n"
        "grade.py deletes this file itself once the real grader has run.\n",
        encoding="utf-8")
    return marker


def clear_marker(out_path: Path) -> Path | None:
    """Remove a stale fallback note after a real grading run.  Returns it if removed."""
    marker = out_path.parent / MARKER_NAME
    if marker.exists():
        marker.unlink()
        return marker
    return None


# --- equivalence ------------------------------------------------------------

def _as_number(text: str):
    try:
        return Fraction(text.strip().replace(",", ""))
    except (ValueError, ZeroDivisionError):
        return None


def _numerically_close(p: str, g: str):
    a, b = _as_number(p), _as_number(g)
    if a is None or b is None:
        return None
    if a == b:
        return True
    return abs(a - b) <= _REL_TOL * max(abs(a), abs(b))


def _parse_for_verify(text, parse):
    if "\\boxed" not in text:
        wrapped = parse("\\boxed{" + text + "}", parsing_timeout=_MV_TIMEOUT)
        if wrapped:
            return wrapped
    return parse(text, parsing_timeout=_MV_TIMEOUT)


def _try_math_verify(p: str, g: str):
    if probe_math_verify()[0] is None:
        return None
    try:
        from math_verify import parse, verify  # third-party, same package the campaigns used
    except Exception:  # noqa: BLE001
        return None
    try:
        return bool(verify(_parse_for_verify(g, parse), _parse_for_verify(p, parse), timeout_seconds=_MV_TIMEOUT))
    except Exception:  # noqa: BLE001
        return None


def _try_sympy(p: str, g: str):
    try:
        import sympy
        from sympy.parsing.sympy_parser import (implicit_multiplication_application, parse_expr,
                                                standard_transformations)
    except ImportError:
        return None
    tr = standard_transformations + (implicit_multiplication_application,)
    try:
        a = parse_expr(p, transformations=tr, evaluate=True)
        b = parse_expr(g, transformations=tr, evaluate=True)
        return bool(sympy.simplify(a - b) == 0)
    except Exception:  # noqa: BLE001
        return None


#: Units and money the *value* of a numeric answer does not depend on.  Copied
#: from MOME's `_LATEX_NOISE` (mome/evalkit/extraction.py:230) --
#: `\$|[$]|\text{...}|\times|\cdot|\div|\left|\right` -- plus `\mbox{}`, LaTeX
#: spacing and the percent sign.  MOME needs this set only inside *extraction*
#: because its grader reaches math-verify, which reads the shapes directly;
#: `_LATEX_NUMERIC` (mome/evalkit/extraction.py:398-405) names the very same
#: shapes -- "currency, thousands separators ... the fallback fallback (no
#: math-verify at all) cannot read any of them" -- and MOME's answer there is to
#: REFUSE the run.  We refuse too (see `refusal_text`), and for the development
#: fallback we close the gap instead of grading silently stricter than MOME.
_UNIT_NOISE = re.compile(
    r"\\text\{[^}]*\}"          # 36 \text{ hours}
    r"|\\mbox\{[^}]*\}"
    r"|\\\$|[$]"                # \$75.00
    r"|\\%|%"                   # 33\%
    r"|\\(?:times|cdot|div|left|right)\b"
    r"|\\[,!;:]"              # LaTeX thin spaces.  NOT a bare `\ `: `\\` is LaTeX's
)                               # row separator and eating it flattens a pmatrix


def _strip_units(text: str) -> str:
    r"""Currency, percent and a trailing `\text{unit}` off an otherwise numeric answer.

    Thousands separators need no rule here: `_as_number` already drops commas
    (mome/grader/equivalence.py:38-43).
    """
    return _UNIT_NOISE.sub("", text).strip()


def _equivalent(p: str, g: str) -> bool:
    if p == g:
        return True
    close = _numerically_close(p, g)
    if close is not None:
        return close
    mv = _try_math_verify(p, g)
    if mv:
        return True
    sp = _try_sympy(p, g)
    if sp is not None:
        return sp
    if mv is not None:
        return mv
    return extract.normalize_numeric(p) == extract.normalize_numeric(g)


def unit_strip_default() -> bool:
    """The unit retry is a fallback repair only; with math-verify live the chain
    is MOME's, unchanged."""
    return probe_math_verify()[0] is None


def _equivalent_chain(p: str, g: str, strip_units: bool) -> bool:
    r"""MOME's `equivalent` (mome/grader/equivalence.py:132-158) plus, in fallback
    mode only, a unit-stripping retry.  The answer exactly as delivered.

    Both retries follow MOME's rule for the doubled-backslash collapse: a second
    attempt, never a precondition, so "a repair that can only add agreement
    cannot destroy an answer that was already right".
    """
    p_s, g_s = p.strip(), g.strip()
    if _equivalent(p_s, g_s):
        return True
    collapsed = p.replace("\\\\", "\\").strip()
    if collapsed != p_s and _equivalent(collapsed, g_s):
        return True
    if strip_units:
        sp, sg = _strip_units(p_s), _strip_units(g_s)
        if sp and (sp, sg) != (p_s, g_s) and _equivalent(sp, sg):
            return True
    return False


# --- notation of the PREDICTED answer ---------------------------------------
#
# The `expression` route delivers whatever the sandbox PRINTED.  A Python or
# sympy `print` writes `x**5 - x**4`, `(1-12j)`, `Matrix([[-1/3], [2/3]])`,
# `137 1/2`, `1/tan(x)`; the MATH-500 gold for the same answers is written in
# LaTeX -- `x^5 - x^4`, `1 - 12i`, a `pmatrix`, `137\frac{1}{2}`, `\cot x`.
# a separate audit established mechanically, item by item, that
# these are the SAME answer written twice (see
# the audit notes).
#
# Three rules keep this from becoming a way to inflate an accuracy:
#
#   1. It rewrites the PREDICTED answer only.  The gold is never touched, so
#      no rule can ever move the target.
#   2. It is a RETRY, never a precondition -- the answer is compared exactly as
#      delivered first, and a rewrite is tried only after that returns False.
#      MOME's rule for the doubled-backslash collapse
#      (mome/grader/equivalence.py:132-158): "a repair that can only add
#      agreement cannot destroy an answer that was already right."  So this
#      cannot turn a correct grade into a wrong one -- not as a matter of
#      testing, as a matter of control flow.
#   3. Every rule is a NOTATION rewrite: it re-spells a value, it never
#      computes one.  A rewrite that could change the value is refused rather
#      than made more clever -- `_n_power_and_multiplication` drops an explicit
#      `*` only where LaTeX's implicit multiplication reads back the same
#      product, so `2*3` is left alone rather than becoming `23`.
#
# **What is deliberately NOT here.**  The same analysis found 5 items whose
# gold is a numeral in a base (`204_5`, `52_8`) and whose program printed the
# digits alone (`204`, `52`).  There is no rewrite of `204` that carries the
# base, so the only rule that could accept it would have to read the base off
# the GOLD -- which is rule 1 broken, and which would equally accept `204` for
# `204_9`.  Those 5 stay wrong.  Whether the base marker should be demanded of
# the prompt instead is a generation question, not a grading one.

def _unwrap_parens(s: str) -> str:
    """`(1-12i)` -> `1-12i`, but `(a)*(b)` unchanged: one BALANCED outer pair only."""
    t = s.strip()
    if not (t.startswith("(") and t.endswith(")")):
        return t
    depth = 0
    for i, ch in enumerate(t):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return t[1:-1].strip() if i == len(t) - 1 else t
    return t


#: An explicit `*` becomes LaTeX's implicit multiplication only where the two
#: readings are the same product.  The right-hand side must be a letter, an
#: opening paren or a backslash: `3*x` -> `3x` and `2*\sqrt{3}` -> `2\sqrt{3}`
#: are the same value, while `2*3` -> `23` and `x*2` -> `x2` are not, so those
#: are not matched at all.
_PY_MUL = re.compile(r"(?<=[0-9A-Za-z\)\}])\s*\*\s*(?=[A-Za-z\(\\])")


def _n_power_and_multiplication(s: str) -> str:
    r"""`x**5 - x**4` -> `x^5 - x^4`, `5*r**5` -> `5r^5`."""
    return _PY_MUL.sub("", s.replace("**", "^"))


def _n_complex_j(s: str) -> str:
    r"""Python prints the imaginary unit `j` and parenthesises: `(1-12j)` -> `1-12i`."""
    return _unwrap_parens(re.sub(r"(?<=[0-9])j\b", "i", s))


def _reads_back_as_vector(latex: str, n: int) -> bool:
    r"""Does the grader read this rewritten column vector back as an n-entry vector?

    math-verify cannot read every entry a Python program can print.  A `\pm`
    inside a `pmatrix` makes `parse` return the FIRST ENTRY instead of the
    matrix -- exactly the "silently different answer" hazard
    `mome/grader/equivalence.py:_parse_for_verify` was written about -- and a
    one-element answer compares equal to any gold that happens to be that
    element.  Measured here: `(0, \pm c)` rewritten as a `pmatrix` parses to
    `0` and would then have graded correct against an unrelated item whose gold
    is `0`.

    So a vector rewrite is offered only when the grader reads it back as the
    vector it was written as.  "Cannot check" counts as not verified: with no
    math-verify there is no reported number to protect, and refusing costs
    nothing.
    """
    try:
        from math_verify import parse
    except Exception:  # noqa: BLE001
        return False
    try:
        parsed = _parse_for_verify(latex, parse)
    except Exception:  # noqa: BLE001
        return False
    for obj in parsed:
        if hasattr(obj, "shape"):
            try:
                if len(obj) == n:
                    return True
            except TypeError:
                continue
    return False


def _n_matrix_to_pmatrix(s: str) -> str:
    r"""`Matrix([[-1/3], [2/3], [5/3]])` -> a `pmatrix` column vector."""
    m = re.fullmatch(r"Matrix\(\[\s*(.*?)\s*\]\)", s.strip(), re.S)
    if not m:
        return s
    rows = re.findall(r"\[([^\]]*)\]", m.group(1))
    if not rows:
        return s
    out = r"\begin{pmatrix} " + r" \\ ".join(r.strip() for r in rows) + r" \end{pmatrix}"
    return out if _reads_back_as_vector(out, len(rows)) else s


def _n_tuple_to_column_vector(s: str) -> str:
    r"""`(-18, -49, 96)` printed for the column vector the gold writes as a `pmatrix`.

    Entry for entry, in order, so this can only agree with a gold that carries
    the same values in the same places.
    """
    t = s.strip()
    m = re.fullmatch(r"\(\s*([^()]*?)\s*\)", t)
    if not m:
        return s
    parts = [p.strip() for p in m.group(1).split(",")]
    if len(parts) < 2 or not all(parts):
        return s
    out = r"\begin{pmatrix} " + r" \\ ".join(parts) + r" \end{pmatrix}"
    return out if _reads_back_as_vector(out, len(parts)) else s


def _n_sqrt(s: str) -> str:
    r"""`sqrt(3)` is `\sqrt{3}`.

    Only a parenthesised group with no nested parentheses, so the rewrite can
    never re-associate an expression it did not fully read.
    """
    return re.sub(r"\bsqrt\(([^()]*)\)", r"\\sqrt{\1}", s)


def _n_reciprocal_trig(s: str) -> str:
    r"""`1/tan(x)` -> `\cot x`; the whole answer or nothing."""
    m = re.fullmatch(r"1\s*/\s*(sin|cos|tan)\(([^()]*)\)", s.strip())
    if not m:
        return s
    return {"sin": r"\csc ", "cos": r"\sec ", "tan": r"\cot "}[m.group(1)] + m.group(2)


def _n_mixed_number(s: str) -> str:
    r"""`137 1/2` -> `137\frac{1}{2}`.

    Non-negative whole part only.  `-3 1/2` is written by people to mean
    -(3+1/2) and by LaTeX to mean -3+1/2, and no item in the analysis is
    negative, so the ambiguous case is left ungraded by this rule rather than
    guessed at.
    """
    m = re.fullmatch(r"(\d+)\s+(\d+)\s*/\s*(\d+)", s.strip())
    return f"{m.group(1)}\\frac{{{m.group(2)}}}{{{m.group(3)}}}" if m else s


def _n_inequalities_to_intervals(s: str) -> str:
    r"""`0 < x < 9 or 9 < x < 36` -> `(0,9) \cup (9,36)`; strict bounds only."""
    parts = [p.strip() for p in re.split(r"\bor\b", s)]
    out = []
    for p in parts:
        m = re.fullmatch(r"(-?[\d./]+)\s*<\s*[A-Za-z]\s*<\s*(-?[\d./]+)", p)
        if not m:
            return s
        out.append(f"({m.group(1)},{m.group(2)})")
    return s if len(out) < 2 else r" \cup ".join(out)


#: `(name, rewrite, what it means)`.  Every entry has at least one item behind
#: it in the regrade audit; nothing is here on speculation.
NOTATION_RULES = (
    ("python_power_and_multiplication", _n_power_and_multiplication,
     "`**` is `^`, and an explicit `*` is LaTeX's implicit multiplication"),
    ("python_complex_j", _n_complex_j,
     "Python writes the imaginary unit `j` and parenthesises the number"),
    ("sympy_matrix_to_pmatrix", _n_matrix_to_pmatrix,
     "`Matrix([[a], [b]])` is a column vector"),
    ("tuple_to_column_vector", _n_tuple_to_column_vector,
     "`(a, b, c)` printed for a column vector"),
    ("python_sqrt", _n_sqrt, "`sqrt(3)` is `\\sqrt{3}`"),
    ("reciprocal_trig", _n_reciprocal_trig, "`1/tan(x)` is `\\cot x`"),
    ("mixed_number", _n_mixed_number, "`137 1/2` is `137\\frac{1}{2}`"),
    ("inequalities_to_intervals", _n_inequalities_to_intervals,
     "`0 < x < 9 or 9 < x < 36` is `(0,9) \\cup (9,36)`"),
)

NOTATION_RULE_DOC = {name: doc for name, _fn, doc in NOTATION_RULES}
NOTATION_RULE_DOC["composed"] = "several of the rules above applied together"

#: What a `grades.jsonl` row records as the standard it was graded on.  Written
#: into every row so a file can never be read as if it came from the other one.
STANDARD = "predicted-answer notation normalisation v1"

#: One sentence, printed verbatim in every Phase 2 report, so that no reader has
#: to work out which grader produced a number or whether it can be set beside an
#: archived MOME table.  It says what standard is in force and what it is not
#: comparable with; it does not describe anything as a loss.
COMPARABILITY = (
    "**The standard these numbers are graded on.** Every accuracy here comes "
    "from `mome/grade.py`: math-verify 0.9.0 through MOME's "
    "equivalence chain (`mome/grader/equivalence.py`), with the delivered "
    "answer additionally re-spelt into the gold's notation by the named rules "
    "in `NOTATION_RULES` when — and only when — the chain has already rejected "
    "it as written. Every committed cell was graded together on this one "
    "standard by `mome/regrade.py`, and every grade that moved is "
    "listed in the regrade audit. The archived MOME tables "
    "were produced by MOME's own grader over MOME's own generations on a "
    "different Ollama build, so they are a different measurement and no row "
    "here is comparable with one of theirs.")


def notation_variants(text: str) -> list[tuple[str, str]]:
    """`[(rule name, rewritten predicted answer), ...]`, distinct and non-empty.

    The composition of all the rules is offered last, for an answer that is
    mis-spelt in more than one way at once.
    """
    base = (text or "").strip()
    if not base:
        return []
    out: list[tuple[str, str]] = []
    seen = {base}
    for name, fn, _doc in NOTATION_RULES:
        try:
            rewritten = fn(base)
        except Exception:  # noqa: BLE001  a rewrite must never break a grade
            continue
        if rewritten and rewritten.strip() and rewritten not in seen:
            seen.add(rewritten)
            out.append((name, rewritten))
    composed = base
    for _name, fn, _doc in NOTATION_RULES:
        try:
            composed = fn(composed)
        except Exception:  # noqa: BLE001
            pass
    if composed and composed.strip() and composed not in seen:
        out.append(("composed", composed))
    return out


def notation_match(p: str, g: str, strip_units: bool = False) -> tuple[str, str] | None:
    """`(rule, rewritten)` for the first rewrite the UNCHANGED chain accepts, else None.

    The comparison each rewrite goes through is `_equivalent_chain`, exactly the
    function that graded the campaigns before this existed.  So a rule can only
    ever hand the same grader a differently spelt string; it never relaxes what
    that grader considers equal.
    """
    gold = (g or "").strip()
    for name, rewritten in notation_variants(p):
        if _equivalent_chain(rewritten, gold, strip_units):
            return name, rewritten
    return None


def equivalent(p: str, g: str, strip_units: bool | None = None,
               notation: bool = True) -> bool:
    r"""The one standard every committed number is graded on.

    `_equivalent_chain` first -- the answer exactly as delivered, through MOME's
    chain -- and only on a False, the notation rewrites of the PREDICTED answer.
    `notation=False` asks for the chain alone, which is what the campaigns were
    graded with before the regrade audit; it exists so a test
    or an analysis can name the older behaviour, not so a report can choose
    between two standards.
    """
    if strip_units is None:
        strip_units = unit_strip_default()
    if _equivalent_chain(p, g, strip_units):
        return True
    return bool(notation and notation_match(p, g, strip_units))


def grade(pred, gold: str, strip_units: bool | None = None,
          notation: bool = True) -> bool:
    # mome/grader/equivalence.py:221-238 (NUMERIC and EXPRESSION both -> equivalent)
    if pred is None or not str(pred).strip():
        return False
    return equivalent(str(pred), str(gold), strip_units, notation)


def grade_explained(pred, gold: str, strip_units: bool | None = None) -> tuple[bool, str | None]:
    """`(correct, rule)` -- `rule` names the rewrite that made it accept, or None.

    None on a wrong answer and None on an answer the unmodified chain already
    accepted, so a non-null value in a `grades.jsonl` row is exactly the set of
    grades this normalisation is responsible for.
    """
    if pred is None or not str(pred).strip():
        return False, None
    if strip_units is None:
        strip_units = unit_strip_default()
    p, g = str(pred), str(gold)
    if _equivalent_chain(p, g, strip_units):
        return True, None
    hit = notation_match(p, g, strip_units)
    return (True, hit[0]) if hit else (False, None)


def derivation_agree(exec_row: dict | None, strip_units: bool) -> bool | None:
    """Do the program's two derivations of the same quantity agree?

    `TIR3_TEMPLATE` orders the program to compute its target a second time by a
    different method and print it as `CHECK: <value>` on the line before the
    answer.  `execute.py` parses those lines into `check_lines`; this compares
    each of them with the answer the program printed.

    The comparison is `equivalent` -- the SAME grader function that writes
    `program_vs_prose_agree` -- and never `==`.  Two derivations of one quantity
    print it in two notations by construction (`1/2` against `0.5`, `sqrt(2)`
    against `1.4142135623730951`, `Rational(3,4)` against `0.75`), so string
    equality would report a contradiction on every row where the second method
    did its job and the channel would fire on notation instead of on error.

    Returns None -- not False -- when the question has no answer: the program
    did not run clean, printed no answer line, or printed no check line at all.
    "The model did not give a second derivation" is the ABSENCE of evidence and
    must not be readable as disagreement.
    """
    if not exec_row or not exec_row.get("ok"):
        return None
    line = str(exec_row.get("answer_line") or "").strip()
    values = [str(c.get("value")).strip() for c in (exec_row.get("check_lines") or [])
              if isinstance(c, dict) and str(c.get("value") or "").strip()]
    if not line or not values:
        return None
    return all(equivalent(line, v, strip_units) for v in values)


# --- the step ---------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Grade Phase 1 records into grades.jsonl.")
    ap.add_argument("--records", required=True, help="path to records.jsonl (or records.jsonl.gz)")
    ap.add_argument("--benchmark", nargs="+", required=True, help="benchmark jsonl path(s) with id/answer")
    ap.add_argument("--exec", dest="exec_path", default=None, help="exec.jsonl (default next to records)")
    ap.add_argument("--out", default=None, help="grades.jsonl (default next to records)")
    ap.add_argument("--allow-fallback", action="store_true",
                    help="grade without math-verify (development only); marks the output")
    args = ap.parse_args()

    name = grader_name()
    if name == FALLBACK:
        print(refusal_text(args.allow_fallback), file=sys.stderr, flush=True)
        if not args.allow_fallback:
            return RC_NO_GRADER

    records_path = Path(args.records)
    exec_path = Path(args.exec_path) if args.exec_path else records_path.parent / "exec.jsonl"
    out_path = Path(args.out) if args.out else records_path.parent / "grades.jsonl"
    gold: dict[str, str] = {}
    for p in args.benchmark:
        for row in oh.read_jsonl(oh.resolve(p)):
            gold[str(row["id"])] = str(row["answer"])
    # Keyed on (item_id, path_index) for the same reason the records are: a pool
    # campaign has one PROGRAM per path, so an exec row keyed by item alone would
    # hand one path's execution to all five and every F-channel feature computed
    # from it would be wrong for four of them.  `record_key` reads a row without
    # `path_index` as path 0, so a greedy campaign's exec.jsonl -- old or new --
    # joins exactly as it did before.
    execs = {oh.record_key(r): r for r in oh.read_jsonl(exec_path)}
    # Keyed on (item_id, path_index), so a pool campaign gets one grade row per
    # PATH and a greedy campaign behaves exactly as before: its records carry no
    # `path_index` and `oh.record_key` calls them all path 0, which collapses to
    # the old one-row-per-item behaviour (a retry still overwrites its earlier
    # attempt, the last error-free record winning).
    latest: dict[tuple[str, int], dict] = {}
    # read_jsonl falls back to records.jsonl.gz when the plain file is gone
    for r in oh.read_jsonl(records_path):
        if r.get("error") is None:
            latest[oh.record_key(r)] = r
    strip_units = name == FALLBACK      # decided once, from the grader we RECORD
    n = n_correct = n_missing = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for (item_id, path_index), r in latest.items():
            if item_id not in gold:
                n_missing += 1
                continue
            g = gold[item_id]
            ex = execs.get((item_id, path_index))
            if r["route"] == "expression":
                if ex and ex.get("ok") and ex.get("answer_line"):
                    final, source, method, verified = ex["answer_line"], "exec", "tir", True
                else:
                    final, source, method, verified = r.get("prose_answer"), "prose", "tir-fallback", False
            else:
                final, source, method, verified = r.get("extracted_answer"), "text", "format", False
            correct, rule = grade_explained(final, g, strip_units)
            agree = None
            if r["route"] == "expression" and ex and ex.get("answer_line") and r.get("prose_answer"):
                agree = equivalent(str(ex["answer_line"]), str(r["prose_answer"]), strip_units)
            # `derivation_agree` is written ONLY for records drawn with a template
            # that ordered a second derivation.  A `tir_en` row therefore keeps
            # exactly the fields it has always had, and a re-grade of a committed
            # cell produces the same judgements in the same shape.
            second = prompts.wants_check_lines(r.get("prompt_template_id"))
            row = {"item_id": item_id, "campaign": r.get("campaign"), "gold": g,
                   "answer_kind": r.get("route"), "method": method, "final_answer": final,
                   "answer_source": source, "verified": verified, "correct": bool(correct),
                   "grader": name, "standard": STANDARD, "notation_rule": rule,
                   "program_vs_prose_agree": agree,
                   **({"derivation_agree": derivation_agree(ex, strip_units)} if second else {}),
                   "graded_at": oh.now_iso()}
            if r.get("path_index") is not None:   # pool campaign: one row per path
                row["path_index"] = path_index
                row["path_kind"] = r.get("path_kind")
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            n_correct += int(correct)
    unit = "generations" if any(r.get("path_index") is not None for r in latest.values()) else "items"
    print(f"graded {n} {unit} with {name} on the {STANDARD}: {n_correct} correct"
          + (f"; {n_missing} items had no gold in the benchmark file(s)" if n_missing else "")
          + f" -> {out_path}", flush=True)
    if name == FALLBACK:
        marker = write_marker(out_path, n)
        print(f"  NOT A MEASUREMENT: graded without math-verify; see {marker}", flush=True)
    else:
        gone = clear_marker(out_path)
        if gone:
            print(f"  removed the stale fallback note {gone.name}", flush=True)
    return 0


if __name__ == "__main__":
    #: On Windows the comparator's timeout spawns a child process per
    #: comparison, and some die during bootstrap with a traceback.  Grades are
    #: unaffected, but a log buried under them hides real errors, so that
    #: block is filtered out.  No grading rule and no timeout is touched.
    #: Grading must still work where this helper is absent, hence the try.
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis" / "ccf"))
        from quiet_mp import drop_spawn_bootstrap_noise
    except Exception:  # noqa: BLE001
        raise SystemExit(main())
    with drop_spawn_bootstrap_noise():
        code = main()
    raise SystemExit(code)
