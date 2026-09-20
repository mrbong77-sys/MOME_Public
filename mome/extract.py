"""Answer extraction, re-implemented from MOME so Phase 1 grades are comparable.

This is a line-for-line port of `mome/evalkit/extraction.py` (read-only
source; the runtime layer never imports the analysis layer).  The Phase 0 records' `predicted`
strings were produced by `extract_final_answer` there, so Phase 1 must read
answers with the same regexes and the same stage order.  `kind` is the
benchmark's answer kind: "numeric" (GSM8K, HRM8K-GSM8K-ko) or "expression"
(MATH-500).

Every regex and function below names the MOME source lines it was copied
from.  Behaviour was checked against the MOME function on stored Phase 0
completions (see mome/README.md).
"""
from __future__ import annotations

import math
import re

NUMERIC = "numeric"
EXPRESSION = "expression"

# reimplemented from mome/evalkit/extraction.py:19-80
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_ANSWER_PATTERNS = [
    re.compile(r"####\s*([\-\d.,/]+)"),
    re.compile(r"(?:final answer|answer|정답|답)\s*(?:is|:|은|는)?\s*([\-\d.,/]+)", re.IGNORECASE),
]
_LAST_NUMBER = re.compile(r"(-?\d[\d,]*(?:\.\d+)?(?:/\d+)?)(?!.*-?\d)")
_VERIFYING = re.compile(
    r"(?:check|verif\w*|confirm)\w*\s+(?:the|our|your|this)?\s*$", re.IGNORECASE
)
_MATH_SPAN = re.compile(
    r"\\\[(.+?)\\\]|\$\$(.+?)\$\$|(?<!\\)\$((?:[^$\\]|\\[^$])+?)(?<!\\)\$", re.DOTALL
)
_ASSIGNMENT = re.compile(r"^\s*[A-Za-z][A-Za-z0-9_]*(?:\([^)]*\))?\s*=\s*(.+)$", re.DOTALL)
_OPERATOR = re.compile(r"[+\-*/×÷]|\\(?:times|cdot|div)\b")
_COMPUTATION = re.compile(r"^(?P<left>[^=]+?)\s*=\s*(?P<right>[^=]+)$", re.DOTALL)
_ANSWER_CUE = re.compile(r"(?:final answer|answer|정답|답)\s*(?:is|:|은|는)?\s*", re.IGNORECASE)
_BOLD_NUMBER = re.compile(r"\*\*[^*]{0,14}?(-?\d[\d,]*(?:\.\d+)?)[^*]{0,18}?\*\*")
_CURRENCY_NUMBER = re.compile(r"\\?\$\s*(-?\d[\d,]*(?:\.\d+)?)")
_ESCAPED_CURRENCY = re.compile(r"\\\$\s*(-?\d[\d,]*(?:\.\d+)?)")
_ANY_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
_CONDITION = re.compile(r"(?<![<>=!])[<>](?![<>=])|\\(?:le|ge|leq|geq|neq|approx|ne)\b")
_STATED_VALUE = re.compile(
    r"(?:\b(?:is|are|was|were|be|has|have|costs?|equals)\b|=)\s*"
    r"[~\s]*\\?\$?\s*(-?\d[\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)


# reimplemented from mome/evalkit/extraction.py:83-107
def _boxed_balanced(text: str) -> str | None:
    key = "\\boxed{"
    start = text.rfind(key)
    if start < 0:
        return None
    depth, out = 1, []
    for char in text[start + len(key):]:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(out).strip()
        out.append(char)
    return None  # unterminated (truncated generation): not an answer


# reimplemented from mome/evalkit/extraction.py:110-162
def extract_final_answer(text: str, kind: str = NUMERIC) -> str | None:
    boxed = _boxed_balanced(text)
    if boxed:
        return boxed
    hashed = _ANSWER_PATTERNS[0].search(text)
    if hashed:
        return hashed.group(1).strip().rstrip(".,")
    stated = _answer_sentence(text, kind)
    if stated:
        return stated
    for pattern in _ANSWER_PATTERNS[1:]:
        for match in pattern.finditer(text):
            if not any(c.isdigit() for c in match.group(1)):
                continue
            if _VERIFYING.search(text[max(0, match.start() - 24): match.start()]):
                continue
            return match.group(1).strip().rstrip(".,")
    closing = _final_sentence(text, kind)
    if closing:
        return closing
    span = _last_math_span(text, kind)
    if span:
        return span
    if kind == EXPRESSION:
        return None
    match = _LAST_NUMBER.search(text)
    return match.group(1) if match else None


# reimplemented from mome/evalkit/extraction.py:167-188
_BOLD_TEXT = re.compile(r"\*\*\s*([^*\n]{1,30}?)\s*\*\*")


def _symbolic_in(sentence: str) -> str | None:
    spans = [next(g for g in m.groups() if g) for m in _MATH_SPAN.finditer(sentence)]
    if len(spans) == 1:
        candidate = _answer_side(spans[0].strip().rstrip("."))
        if candidate and "\n" not in candidate:
            return candidate
    if spans:
        return None
    emphasised = _BOLD_TEXT.findall(sentence)
    if len(emphasised) == 1 and not _ANY_NUMBER.fullmatch(emphasised[0]):
        return emphasised[0]
    return None


# reimplemented from mome/evalkit/extraction.py:191-223
def _answer_sentence(text: str, kind: str = NUMERIC) -> str | None:
    cues = list(_ANSWER_CUE.finditer(text))
    if not cues:
        return None
    tail = text[cues[-1].end():]
    sentence = tail.lstrip("*: \n")[:400].split("\n\n")[0]
    if kind == EXPRESSION:
        symbolic = _symbolic_in(sentence)
        if symbolic:
            return symbolic
        if _MATH_SPAN.search(sentence):
            return None
    for pattern in (_BOLD_NUMBER, _CURRENCY_NUMBER):
        found = pattern.findall(sentence)
        if len(set(found)) == 1:
            return found[0]
    numbers = _ANY_NUMBER.findall(sentence)
    if len(set(numbers)) == 1:
        return numbers[0]
    return None


# reimplemented from mome/evalkit/extraction.py:230-283
_LATEX_NOISE = re.compile(r"\\\$|[$]|\\text\{[^}]*\}|\\(?:times|cdot|div|left|right)\b")


def _final_sentence(text: str, kind: str = NUMERIC) -> str | None:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    if kind == EXPRESSION:
        symbolic = _symbolic_in(lines[-1])
        if symbolic:
            return symbolic
        if _MATH_SPAN.search(lines[-1]):
            return None
    currency = _ESCAPED_CURRENCY if _MATH_SPAN.search(lines[-1]) else _CURRENCY_NUMBER
    for pattern in (_BOLD_NUMBER, currency):
        found = pattern.findall(lines[-1])
        if len(set(found)) == 1:
            return found[0]
    sentence = _LATEX_NOISE.sub("", lines[-1]).strip()
    if not sentence:
        return None
    numbers = _ANY_NUMBER.findall(sentence)
    if len(set(numbers)) == 1:
        return numbers[0]
    stated = _STATED_VALUE.findall(lines[-1])
    if stated:
        return stated[-1]
    return None


# reimplemented from mome/evalkit/extraction.py:286-306
def _answer_side(candidate: str) -> str:
    assigned = _ASSIGNMENT.match(candidate)
    if assigned:
        return assigned.group(1).strip()
    computation = _COMPUTATION.match(candidate)
    if computation and _OPERATOR.search(computation.group("left")):
        return computation.group("right").strip()
    return candidate


# reimplemented from mome/evalkit/extraction.py:309-342
def _last_math_span(text: str, kind: str = NUMERIC) -> str | None:
    spans = [next(g for g in m.groups() if g) for m in _MATH_SPAN.finditer(text)]
    for raw in reversed(spans):
        span = raw.strip().rstrip(".")
        if not span or "\n" in span:
            continue
        if kind == EXPRESSION and _CONDITION.search(span):
            continue
        if "\\" in span or len(span) <= 24:
            side = _answer_side(span)
            if kind == NUMERIC and side and re.fullmatch(r"[A-Za-z]{1,3}", side):
                return _resolve_symbol(text, side)
            return side
    return None


# reimplemented from mome/evalkit/extraction.py:345-365
def _resolve_symbol(text: str, symbol: str) -> str | None:
    value = None
    for match in re.finditer(
        rf"(?<![A-Za-z0-9_]){re.escape(symbol)}\s*=\s*(-?\d[\d,]*(?:\.\d+)?)", text
    ):
        before = text[: match.start()].rstrip()
        if before.endswith(("+", "-", "*", "/", "=", "×", "÷")):
            continue
        rest = text[match.end():].lstrip()
        if rest[:1] in "+-*/×÷" or rest[:6].startswith(("\\times", "\\cdot", "\\div")):
            continue
        value = match.group(1)
    return value


# reimplemented from mome/evalkit/extraction.py:368-397
def normalize_numeric(answer: str) -> str:
    s = answer.strip().replace(",", "").replace(" ", "")
    if "/" in s:
        parts = s.split("/")
        if len(parts) == 2:
            try:
                value = float(parts[0]) / float(parts[1])
                return _format_float(value)
            except (ValueError, ZeroDivisionError):
                return s
    try:
        return _format_float(float(s))
    except ValueError:
        return s


def _format_float(value: float) -> str:
    if math.isinf(value) or math.isnan(value):
        return repr(value)
    if value == int(value):
        return str(int(value))
    return repr(value)
