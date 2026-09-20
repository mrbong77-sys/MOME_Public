#!/usr/bin/env python3
"""Rule-based metamorphic perturbation engine.

Takes one problem statement and returns a list of minimally edited probes.
No model call, deterministic, standard library only.  Every rule carries a
well-posedness filter: a perturbation that cannot be trusted is not made at
all.  Precision is preferred over coverage throughout, because a malformed
probe is worse than a missing one -- it puts noise into the fingerprint.

    from perturb import perturb
    probes = perturb("Janet has 16 eggs and eats at least 3.", lang="en")

Each probe carries the response the edit requires:

- MUST_CHANGE -- the answer has to differ.  The direction and scale notes are
  attached only where the rule can settle them.
- MUST_HOLD -- the answer has to stay the same.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MUST_CHANGE = "MUST_CHANGE"
MUST_HOLD = "MUST_HOLD"


@dataclass
class Perturbation:
    rule: str          # rule id, e.g. "en.flip.at_least"
    lang: str          # "en" | "ko" | "latex"
    expectation: str   # MUST_CHANGE | MUST_HOLD
    text: str          # the edited problem, in full
    confidence: str    # "high" | "medium" -- how sure the expectation is
    direction: str | None = None   # "increase" | "decrease" | None
    scale: float | None = None     # the rule's factor, NOT a claim about the answer
    note: str = ""
    meta: dict = field(default_factory=dict)


# ---- Substitution rule packs ------------------------------------------------
# (rule id, pattern, replacement).  A rule fires only when its pattern occurs
# exactly once in the problem.  With two or more occurrences the rule cannot
# tell which one carries the constraint being edited, so the probe is dropped.

_EN_FLIPS = [
    ("en.flip.at_least", r"\bat least\b", "at most"),
    ("en.flip.at_most", r"\bat most\b", "at least"),
    ("en.flip.more_than", r"\bmore than\b", "less than"),
    ("en.flip.less_than", r"\bless than\b", "more than"),
    ("en.flip.maximum", r"\bmaximum\b", "minimum"),
    ("en.flip.minimum", r"\bminimum\b", "maximum"),
    ("en.flip.greatest", r"\bgreatest\b", "least"),
    ("en.flip.smallest", r"\bsmallest\b", "largest"),
    # Added to lift high-confidence flip coverage, which the first pack left
    # at 6% of GSM8K.
    ("en.flip.fewer_than", r"\bfewer than\b", "more than"),
    ("en.flip.greater_than", r"\bgreater than\b", "less than"),
    ("en.flip.largest", r"\blargest\b", "smallest"),
    ("en.flip.highest", r"\bhighest\b", "lowest"),
    ("en.flip.lowest", r"\blowest\b", "highest"),
    ("en.flip.oldest", r"\boldest\b", "youngest"),
    ("en.flip.youngest", r"\byoungest\b", "oldest"),
    ("en.flip.increased", r"\bincreased\b", "decreased"),
    ("en.flip.decreased", r"\bdecreased\b", "increased"),
    ("en.flip.positive", r"\bpositive\b", "negative"),
    ("en.flip.negative", r"\bnegative\b", "positive"),
    # Bare 'even'/'odd' is ambiguous ("even more", "odd one out"), so these
    # two flip only when a numeric noun follows.
    ("en.flip.even", r"\beven\b(?=\s+(?:number|integer|digit))", "odd"),
    ("en.flip.odd", r"\bodd\b(?=\s+(?:number|integer|digit))", "even"),
    ("en.flip.twice_as", r"\btwice as\b", "half as"),
    ("en.flip.half_as", r"\bhalf as\b", "twice as"),
    # A GSM8K sample carried 'twice' in 31 problems and 'half' in 31, a clean
    # x2 <-> x1/2 pair.  Only the form followed by an article is flipped, which
    # keeps "twice a day" and similar adverbials intact.
    ("en.flip.twice_the", r"\btwice the\b", "half the"),
    ("en.flip.half_the", r"\bhalf the\b", "twice the"),
    ("en.flip.doubled", r"\bdoubled\b", "halved"),
    ("en.flip.halved", r"\bhalved\b", "doubled"),
]

#: The Korean pack, glossed for readers who do not read Korean. Each row
#: reverses a comparison, an extremum, a sign or a factor, exactly as the
#: English pack does:
#:   이상/이하        at least / at most
#:   초과/미만        greater than / less than
#:   최대/최소        maximum / minimum
#:   더 많/더 적      more / fewer
#:   가장 큰/가장 작은 largest / smallest
#:   가장 많/가장 적   most / fewest
#:   증가/감소        increase / decrease
#:   홀수/짝수        odd / even
#:   양수/음수        positive / negative
#:   두 배/절반       twice / half
#: The Korean edits do not adjust the particle they leave behind, which can
#: leave an ill-formed ending. That defect is bounded and reported in the
#: paper: it only depresses the score, so an affected item moves toward
#: verification rather than being served wrongly.
_KO_FLIPS = [
    ("ko.flip.이상", r"이상", "이하"),
    ("ko.flip.이하", r"이하", "이상"),
    ("ko.flip.초과", r"초과", "미만"),
    ("ko.flip.미만", r"미만", "초과"),
    ("ko.flip.최대", r"최대", "최소"),
    ("ko.flip.최소", r"최소", "최대"),
    ("ko.flip.더많", r"더 많", "더 적"),
    ("ko.flip.더적", r"더 적", "더 많"),
    # Second pack, same reason as the English one above.
    ("ko.flip.가장큰", r"가장 큰", "가장 작은"),
    ("ko.flip.가장작은", r"가장 작은", "가장 큰"),
    ("ko.flip.가장많", r"가장 많", "가장 적"),
    ("ko.flip.가장적", r"가장 적", "가장 많"),
    ("ko.flip.증가", r"증가", "감소"),
    ("ko.flip.감소", r"감소", "증가"),
    ("ko.flip.홀수", r"홀수", "짝수"),
    ("ko.flip.짝수", r"짝수", "홀수"),
    ("ko.flip.양수", r"양수", "음수"),
    ("ko.flip.음수", r"음수", "양수"),
    # An HRM8K sample carried "두 배" (twice) in 37 problems and "절반" (half)
    # in 28.  Doubling against halving is a flip whose answer must move.
    ("ko.flip.두배", r"두 배", "절반"),
    ("ko.flip.절반", r"절반", "두 배"),
]

#: Word-number scaling: a quantity spelled out ("four times as old") is
#: replaced by the word for twice that quantity.  'one' is left out as
#: degenerate.
_EN_WORD_DOUBLE = {
    "two": "four", "three": "six", "four": "eight", "five": "ten",
    "six": "twelve", "seven": "fourteen", "eight": "sixteen",
    "nine": "eighteen", "ten": "twenty",
}
_KO_MULT_DOUBLE = {"두 배": "네 배", "세 배": "여섯 배", "네 배": "여덟 배", "다섯 배": "열 배"}

_LATEX_FLIPS = [
    ("latex.flip.geq", r"\\geq?\b", r"\\le"),
    ("latex.flip.leq", r"\\leq?\b", r"\\ge"),
    ("latex.flip.max", r"\\max\b", r"\\min"),
    ("latex.flip.min", r"\\min\b", r"\\max"),
    ("latex.flip.gt", r"(?<![-<>=\\])>(?![=>])", "<"),
    ("latex.flip.lt", r"(?<![-<>=\\])<(?![=<])", ">"),
]

#: The floor MUST_HOLD probe: a redundant sentence that touches no condition.
#: It applies to every problem and cannot degenerate, which is why it is the
#: one probe guaranteed to exist.
_HOLD_SUFFIX = {
    "en": " (All quantities are exactly as stated.)",
    "ko": " (모든 수치는 문제에 제시된 그대로이다.)",
    "latex": " (All quantities are exactly as stated.)",
}

#: Numbers kept out of scaling: degenerate values, and values that read as
#: calendar years.
_SCALE_MIN, _SCALE_MAX = 2, 100_000
_YEAR_RANGE = range(1900, 2101)

#: Decimal quantities ($16.50) are matched too, to two decimal places, which
#: is where money and measurements sit.
_NUMBER = re.compile(r"(?<![\d.,\w])(\d{1,6}(?:\.\d{1,2})?)(?![\d.]|,\d)")


def _apply_flips(question: str, flips, lang: str) -> list[Perturbation]:
    out = []
    for rule, pattern, repl in flips:
        matches = list(re.finditer(pattern, question))
        if len(matches) != 1:  # absent, or ambiguous -- drop either way
            continue
        m = matches[0]
        out.append(Perturbation(
            rule=rule, lang=lang, expectation=MUST_CHANGE, confidence="high",
            text=question[: m.start()] + re.sub(pattern, repl, m.group(0)) + question[m.end():],
            note=f"{m.group(0)!r} -> {repl!r}",
            meta={"span": [m.start(), m.end()]},
        ))
    return out


def _scale_numbers(question: str, lang: str, factor: int = 2) -> list[Perturbation]:
    """Multiply one integer quantity by `factor`.

    The probe is made only when the number occurs exactly once, is neither
    degenerate nor a year, and does not collide with another number already in
    the problem.  The rule does *not* claim the factor carries through to the
    answer linearly, which is why its confidence is medium: observation asks
    only whether the answer moved, and a matching factor is read as a bonus
    bit rather than as the expectation.
    """
    values = [m.group(1) for m in _NUMBER.finditer(question)]
    out = []
    for m in _NUMBER.finditer(question):
        raw = m.group(1)
        value = float(raw)
        if values.count(raw) != 1:
            continue
        if not (_SCALE_MIN <= value <= _SCALE_MAX):
            continue
        if value.is_integer() and int(value) in _YEAR_RANGE:
            continue
        if "." in raw:  # keep the written precision so the edit reads naturally
            places = len(raw.split(".")[1])
            scaled = f"{value * factor:.{places}f}"
        else:
            scaled = str(int(value) * factor)
        if scaled in values:  # collides with another number -- roles could blur
            continue
        out.append(Perturbation(
            rule=f"{lang}.scale.x{factor}", lang=lang, expectation=MUST_CHANGE,
            confidence="medium", scale=float(factor), direction="increase",
            text=question[: m.start(1)] + scaled + question[m.end(1):],
            note=f"{raw} -> {scaled}",
            meta={"span": [m.start(1), m.end(1)], "original": value},
        ))
    return out


def _scale_word_numbers(question: str, lang: str) -> list[Perturbation]:
    """Double a quantity that is spelled out as a word.

    Made only when the source word occurs exactly once and the doubled word is
    not already present.  As with numeric scaling, no claim is made that the
    factor carries through to the answer, hence the medium confidence.
    """
    out = []
    table = _KO_MULT_DOUBLE if lang == "ko" else _EN_WORD_DOUBLE
    for word, doubled in table.items():
        pattern = word if lang == "ko" else rf"\b{word}\b"
        matches = list(re.finditer(pattern, question, flags=0 if lang == "ko" else re.IGNORECASE))
        if len(matches) != 1:
            continue
        if re.search(doubled if lang == "ko" else rf"\b{doubled}\b", question,
                     flags=0 if lang == "ko" else re.IGNORECASE):
            continue
        m = matches[0]
        out.append(Perturbation(
            rule=f"{lang}.scale.word_x2", lang=lang, expectation=MUST_CHANGE,
            confidence="medium", scale=2.0, direction="increase",
            text=question[: m.start()] + doubled + question[m.end():],
            note=f"{m.group(0)!r} -> {doubled!r}",
            meta={"span": [m.start(), m.end()]},
        ))
    return out


_MATH_SPAN = re.compile(r"\$[^$\n]{1,200}\$")


def _latex_probes(question: str) -> list[Perturbation]:
    """Inequality flips for LaTeX.

    When the problem carries backslash commands the flips apply to the whole
    text; when it does not, they apply only inside `$...$` spans.  That keeps
    two false positives out at once: the '$' of a dollar amount, and a stray
    inequality sign in prose.
    """
    probes = _apply_flips(question, _LATEX_FLIPS, "latex")
    if "\\" in question:
        return probes
    spans = [m.span() for m in _MATH_SPAN.finditer(question)]
    return [
        p for p in probes
        if any(s <= p.meta["span"][0] and p.meta["span"][1] <= e for s, e in spans)
    ]


def _hold_probe(question: str, lang: str) -> Perturbation:
    return Perturbation(
        rule=f"{lang}.hold.redundant_clause", lang=lang, expectation=MUST_HOLD,
        confidence="high", text=question.rstrip() + _HOLD_SUFFIX[lang],
    )


def _looks_latex(question: str) -> bool:
    # A bare '$' does not count: dollar amounts in GSM8K word problems trip it
    # on 95 of 300 sampled items.  A backslash command is required.
    return "\\" in question


def perturb(question: str, lang: str = "en", max_scale_probes: int = 2) -> list[Perturbation]:
    """All probes for one problem, ordered flips, then scalings, then the hold."""
    probes: list[Perturbation] = []
    if lang == "ko":
        probes += _apply_flips(question, _KO_FLIPS, "ko")
    else:
        probes += _apply_flips(question, _EN_FLIPS, "en")
    probes += _latex_probes(question)
    probes += _scale_numbers(question, lang)[:max_scale_probes]
    probes += _scale_word_numbers(question, lang)[:1]
    probes.append(_hold_probe(question, "latex" if _looks_latex(question) else lang))
    return probes


def select_operating_set(probes: list[Perturbation], k_change: int = 1) -> list[Perturbation]:
    """The deployed operating set: the top k MUST_CHANGE probes, high
    confidence first, plus one MUST_HOLD probe."""
    change = sorted(
        (p for p in probes if p.expectation == MUST_CHANGE),
        key=lambda p: (p.confidence != "high",),
    )[:k_change]
    hold = [p for p in probes if p.expectation == MUST_HOLD][:1]
    return change + hold


if __name__ == "__main__":
    import json
    import sys

    text = sys.stdin.read().strip()
    lang = sys.argv[1] if len(sys.argv) > 1 else "en"
    for p in perturb(text, lang):
        print(json.dumps(p.__dict__, ensure_ascii=False))
