"""Prompt templates and text helpers, re-implemented from MOME for Phase 1.

Phase 1 generations must be comparable with the Phase 0 campaign records, so
the two prompt templates the Phase 0 cells used (`format` for the numeric
benchmarks GSM8K / HRM8K-GSM8K-ko, `tir` for MATH-500) are copied here as
literal strings.  `` never imports the `mome` package (isolation rule,
README.md); `mome/check_templates.py` compares these literals
against the read-only MOME source text and records whether they are
byte-identical (plan v0.2, section 5.3).

Every literal below carries the MOME source it was copied from.
"""
from __future__ import annotations

import hashlib
import re

# reimplemented from mome/solver/pipeline.py:28-31 (FORMAT_TEMPLATE)
FORMAT_TEMPLATE = {
    "en": "{problem}\n\nGive the final answer as \\boxed{{}}.",
    "ko": "{problem}\n\n최종 답을 \\boxed{{}} 안에 넣어 답하세요.",
}

# reimplemented from mome/solver/pipeline.py:33-36 (COT_TEMPLATE).  MOME's
# pool-of-5 policy drew its four sampled candidates with this template while
# the greedy anchor used `format` (mome/solver/routed.py:92-99 -> cot_candidate).
# Phase 1 pools default to the greedy template for every path so that the only
# difference between the paths is the sampling parameters; a config that wants
# MOME's mixture sets `"sample_template": "cot"` (mome/run_pool.py).
COT_TEMPLATE = {
    "en": "{problem}\n\nSolve step by step. Put your final answer inside \\boxed{{}}.",
    "ko": "{problem}\n\n단계별로 풀이하고, 최종 답을 \\boxed{{}} 안에 넣어 답하세요.",
}

# reimplemented from mome/solver/pipeline.py:41-59 (TIR_TEMPLATE)
TIR_TEMPLATE = {
    "en": (
        "{problem}\n\n"
        "Write a Python program that solves this problem. Start the code with a "
        "comment '# target: ...' restating EXACTLY what the problem asks for "
        "(e.g. if it asks for a sum, the target is the sum, not the product), "
        "and print ONLY that final answer as the last line. "
        "You may use sympy, math, and fractions. "
        "Reply with a single ```python code block."
    ),
    "ko": (
        "{problem}\n\n"
        "이 문제를 푸는 파이썬 프로그램을 작성하세요. 코드 첫 줄에 '# 구하는 것: …' "
        "주석으로 문제가 최종적으로 요구하는 값이 무엇인지 정확히 적으세요 "
        "(예: 합을 물으면 곱이 아니라 합). 마지막 줄에는 그 최종 답 하나만 "
        "출력(print)하세요. sympy, math, fractions를 사용할 수 있습니다. "
        "```python 코드 블록 하나로만 답하세요."
    ),
}

# --- the channel templates (the runtime layer's own; NOT from MOME) ---------------------
#
# `tir_en` above ends "Reply with a single ```python code block.", so a `tir`
# generation contains exactly ONE derivation and the model never states an
# answer of its own.  Two consequences were measured on the committed cells and
# are the reason the two templates below exist (docs/
# channel_design_plan.md section 2):
#
#   * `program_vs_prose_agree` -- the grader's comparison of the executed answer
#     with the same generation's prose answer -- is computable on 14 of the 499
#     anchor records of `math500_gemma4-e2b_pool` and 51 of 500 on
#     `math500_gemma4-e4b_pool`, because `prose_answer` is almost always null;
#   * the bucket where the program ran clean, printed an answer and NO channel
#     fires (`X_ok_no_comparison`) holds 341 of 499 anchors on e2b, 40 of them
#     wrong -- 21.2% of that cell's delivered error, invisible to T, I and F.
#
# Neither template is copied from MOME and neither has a MOME counterpart:
# `check_templates.py` compares FORMAT/TIR/COT against `mome/solver/pipeline.py`
# and says nothing about these, which is correct -- there is nothing to compare
# them with.  They are Phase 1 originals and carry their own ids, so
# `prompt_template_id` and `template_sha256` separate a cell that used one of
# them from every committed cell.
#
# What is held fixed, in BOTH of them: the request that `tir_en` makes is
# repeated word for word -- write a Python program, open it with the
# '# target: ...' restatement, print ONLY the final answer as the LAST line, the
# same three libraries, one ```python code block.  Keeping the final answer on
# the last line is what leaves `execute.answer_line`, grading and every
# committed comparison untouched.  Neither template says anything about the
# problem, hints at an answer, or offers a method for solving it: what is added
# is a second, independently checkable result, not help.

#: (a) program + prose answer.  One line outside the code block carrying the
#: answer the model reached ITSELF.  `\boxed{}` is asked for because that is the
#: first and most robust rule in `extract.extract_final_answer` and the only one
#: that survives a non-numeric MATH-500 answer (a bare `Answer: \frac{3}{4}`
#: line extracts to None; `Answer: \boxed{\frac{3}{4}}` extracts exactly).
#: Record fields: none new.  `prose_answer` (run_campaign.py:301) and
#: `program_vs_prose_agree` (grade.py:704-705) already exist and are simply
#: almost always null under `tir`.
TIR2_TEMPLATE = {
    "en": (
        "{problem}\n\n"
        "Write a Python program that solves this problem. Start the code with a "
        "comment '# target: ...' restating EXACTLY what the problem asks for "
        "(e.g. if it asks for a sum, the target is the sum, not the product), "
        "and print ONLY that final answer as the last line. "
        "You may use sympy, math, and fractions. "
        "Reply with a single ```python code block, and after the block, on a line "
        "of its own outside it, write\n"
        "Answer: \\boxed{{<value>}}\n"
        "where <value> is the answer you reached by working the problem out "
        "yourself. The program is not run before you write that line, so it must "
        "be your own result and not a copy of what the program prints."
    ),
    "ko": (
        "{problem}\n\n"
        "이 문제를 푸는 파이썬 프로그램을 작성하세요. 코드 첫 줄에 '# 구하는 것: …' "
        "주석으로 문제가 최종적으로 요구하는 값이 무엇인지 정확히 적으세요 "
        "(예: 합을 물으면 곱이 아니라 합). 마지막 줄에는 그 최종 답 하나만 "
        "출력(print)하세요. sympy, math, fractions를 사용할 수 있습니다. "
        "```python 코드 블록 하나로 답하고, 그 블록 바깥에 한 줄을 더 써서\n"
        "Answer: \\boxed{{<값>}}\n"
        "형식으로 답을 적으세요. <값>은 당신이 직접 풀어서 얻은 답입니다. 그 줄을 "
        "쓰는 시점에 프로그램은 실행되지 않으므로, 프로그램이 출력할 값을 옮겨 적은 "
        "것이 아니라 당신 자신의 계산 결과여야 합니다."
    ),
}

#: (b) two derivations in one generation.  The same quantity computed twice by
#: different means inside one program, the second printed on the line BEFORE the
#: last as `CHECK: <value>`.  Agreement is evidence for T, disagreement fires F,
#: and both rest on one generation.
#:
#: Two sentences of it are there to stop the evidence being destroyed rather
#: than to help the model: "it must not reuse the first computation's result"
#: (a second derivation copied from the first agrees by construction and
#: measures nothing -- that failure is exactly what the smoke run's refute
#: criterion looks for), and "do not change either one to make them match"
#: (a model that silently reconciles the two prints one value twice and the
#: disagreement never reaches stdout).
#:
#: Record fields, new: `check_lines` on the exec row (`execute.run_code`, via
#: `parse_check_lines` below) and `derivation_agree` on the grade row
#: (`grade.py`, computed with `grade.equivalent`, never string equality).
#: Both are written ONLY for records whose template is one of
#: `CHECK_LINE_TEMPLATES`, so every committed `tir_en` row keeps its exact shape.
TIR3_TEMPLATE = {
    "en": (
        "{problem}\n\n"
        "Write a Python program that solves this problem. Start the code with a "
        "comment '# target: ...' restating EXACTLY what the problem asks for "
        "(e.g. if it asks for a sum, the target is the sum, not the product), "
        "and print ONLY that final answer as the last line. "
        "You may use sympy, math, and fractions. "
        "In the same program, compute that same target a SECOND time by a "
        "DIFFERENT method -- if the first computation is symbolic make the second "
        "numeric, and the other way round -- and print that second result on the "
        "line immediately before the last, as\n"
        "CHECK: <value>\n"
        "Write the second computation independently: it must not reuse the first "
        "computation's result or any variable that holds it. Print both values "
        "exactly as they come out, even if they disagree; do not change either "
        "one to make them match. "
        "Reply with a single ```python code block."
    ),
    "ko": (
        "{problem}\n\n"
        "이 문제를 푸는 파이썬 프로그램을 작성하세요. 코드 첫 줄에 '# 구하는 것: …' "
        "주석으로 문제가 최종적으로 요구하는 값이 무엇인지 정확히 적으세요 "
        "(예: 합을 물으면 곱이 아니라 합). 마지막 줄에는 그 최종 답 하나만 "
        "출력(print)하세요. sympy, math, fractions를 사용할 수 있습니다. "
        "같은 프로그램 안에서 그 값을 서로 다른 방법으로 한 번 더 구하세요 — 첫 "
        "계산이 기호적(symbolic)이면 두 번째는 수치적(numeric)으로, 그 반대도 "
        "마찬가지입니다 — 그리고 그 두 번째 결과를 마지막 줄 바로 앞 줄에\n"
        "CHECK: <값>\n"
        "형식으로 출력하세요. 두 번째 계산은 독립적으로 작성해야 합니다: 첫 계산의 "
        "결과나 그 값을 담은 변수를 재사용하지 마세요. 두 값이 서로 다르더라도 나온 "
        "그대로 둘 다 출력하고, 서로 맞추려고 어느 쪽도 고치지 마세요. "
        "```python 코드 블록 하나로만 답하세요."
    ),
}

#: (b, second attempt) the SAME device as TIR3_TEMPLATE, with the output
#: contract rewritten.  `docs/tir3_diagnosis.md` counted, on the
#: committed `tir3` arm's own program text, what the first wording actually
#: bought: of 60 generations, **37** wrote `CHECK: <value>` as a bare Python
#: SOURCE line.  That is not a syntax error -- Python parses it as a PEP 526
#: annotation (`AnnAssign`), so the program runs and prints nothing -- and 11
#: more printed the label but LAST, where the parser reserved the answer line.
#: Only one put a printed `CHECK:` where the parser looked, and it crashed
#: first.  48 of 60 generations computed a second value; what failed was the
#: form in which we asked for it, so the design has never been tested.
#:
#: What changed, and nothing else changed:
#:
#:   * the ask is now the CALL, shown in its call form -- `print(f"CHECK:
#:     {second}")` -- instead of the OUTPUT form `CHECK: <value>`.  A model that
#:     copies what it is shown now copies a print statement.  The one added
#:     clause says the line is program code rather than output; it teaches
#:     nothing about the problem.
#:   * the call may stand anywhere in the program.  `parse_check_lines` is
#:     position-invariant as of the same diagnosis, so line order carries no
#:     meaning any more and asking for one would only be another way to fail.
#:
#: Every sentence of `tir_en` is repeated word for word, as in the other three
#: templates, and TIR3_TEMPLATE's two protective sentences are kept verbatim:
#: "Write the second computation independently ..." (a copy of the first result
#: agrees by construction and measures nothing) and "Print both values exactly
#: as they come out ..." (a model that reconciles them prints one value twice
#: and the disagreement never reaches stdout).  Neither was the failure.
#:
#: TIR3_TEMPLATE is NOT edited: its 60 generations are the evidence for the
#: diagnosis above, and `template_sha256` separates the two cells.
#:
#: Record fields: the same two `tir3_*` already write -- `check_lines` on the
#: exec row and `derivation_agree` on the grade row -- and no others.
TIR3B_TEMPLATE = {
    "en": (
        "{problem}\n\n"
        "Write a Python program that solves this problem. Start the code with a "
        "comment '# target: ...' restating EXACTLY what the problem asks for "
        "(e.g. if it asks for a sum, the target is the sum, not the product), "
        "and print ONLY that final answer as the last line. "
        "You may use sympy, math, and fractions. "
        "In the same program, compute that same target a SECOND time by a "
        "DIFFERENT method -- if the first computation is symbolic make the second "
        "numeric, and the other way round -- and report that second result by "
        "executing the call\n"
        "print(f\"CHECK: {{second}}\")\n"
        "where second is the name that holds it. That line is program code, not "
        "output, and it may stand anywhere in the program. "
        "Write the second computation independently: it must not reuse the first "
        "computation's result or any variable that holds it. Print both values "
        "exactly as they come out, even if they disagree; do not change either "
        "one to make them match. "
        "Reply with a single ```python code block."
    ),
    "ko": (
        "{problem}\n\n"
        "이 문제를 푸는 파이썬 프로그램을 작성하세요. 코드 첫 줄에 '# 구하는 것: …' "
        "주석으로 문제가 최종적으로 요구하는 값이 무엇인지 정확히 적으세요 "
        "(예: 합을 물으면 곱이 아니라 합). 마지막 줄에는 그 최종 답 하나만 "
        "출력(print)하세요. sympy, math, fractions를 사용할 수 있습니다. "
        "같은 프로그램 안에서 그 값을 서로 다른 방법으로 한 번 더 구하세요 — 첫 "
        "계산이 기호적(symbolic)이면 두 번째는 수치적(numeric)으로, 그 반대도 "
        "마찬가지입니다 — 그리고 그 두 번째 결과를 다음 호출을 실행해서 넘기세요\n"
        "print(f\"CHECK: {{second}}\")\n"
        "여기서 second는 그 값을 담은 이름입니다. 이 줄은 출력이 아니라 실행되는 "
        "프로그램 코드이며, 프로그램 안 어디에 두어도 됩니다. 두 번째 계산은 "
        "독립적으로 작성해야 합니다: 첫 계산의 결과나 그 값을 담은 변수를 "
        "재사용하지 마세요. 두 값이 서로 다르더라도 나온 그대로 둘 다 출력하고, "
        "서로 맞추려고 어느 쪽도 고치지 마세요. "
        "```python 코드 블록 하나로만 답하세요."
    ),
}

#: (b, third attempt) the SAME device again, with BOTH values put behind a
#: labelled call.  `docs/smoke_gate_analysis.md` counted, on the
#: committed `tir3b` arm's own rows, what the second wording bought and what it
#: cost.  It bought compliance: of the 48 generations that ran clean, **47**
#: printed a `CHECK:` line, and 58 of 60 programs contain the call.  The bare
#: source line that killed `tir3` is gone.
#:
#: What it cost was the ANSWER line.  Being shown `print(f"CHECK: {second}")`
#: taught the model to label its OTHER print too, and `parse_answer_line`
#: returns the last non-check line WITH its label: 13 of 60 answer lines came
#: back as `result1: 13`, `Result from symbolic derivation: 8`,
#: `Symbolic solutions found: [10]`.  Ten of those thirteen are graded wrong for
#: the label alone, and because `grade.derivation_agree` compares that same
#: string to the check value, 8 of the arm's 17 disagreements are the label
#: firing rather than the derivations.  The labelling is ours: it appears in
#: 12/60 and 13/60 of the two check-line arms and in 0-2/60 of the other three.
#:
#: What changed, and nothing else changed:
#:
#:   * the final answer is now reported by its own call, `print(f"ANSWER:
#:     {answer}")`, in the same call form the check line already uses.  A
#:     labelled answer line stops being a violation and becomes the contract,
#:     and `parse_labelled_answer_line` reads the answer by its LABEL, so no
#:     stray prefix and no third printed line can be mistaken for it.  That also
#:     removes the last place where position still carried meaning: `tir3b`'s
#:     `math500-69` printed the right answer FIRST, a second derivation after
#:     it, and the "last non-check line" rule picked the wrong one.
#:   * the sentence ordering the answer onto the last line is dropped, because
#:     the label now says which line it is.  Nothing else about the task, the
#:     problem or the method is added or removed.
#:
#: Record fields, new: `answer_label_present` on the exec row, beside the
#: `check_lines` this family already writes.  Both are written ONLY for records
#: whose template is in the matching frozenset, so every committed row of every
#: other template keeps its exact shape.
TIR3C_TEMPLATE = {
    "en": (
        "{problem}\n\n"
        "Write a Python program that solves this problem. Start the code with a "
        "comment '# target: ...' restating EXACTLY what the problem asks for "
        "(e.g. if it asks for a sum, the target is the sum, not the product). "
        "You may use sympy, math, and fractions. "
        "Report that final answer by executing the call\n"
        "print(f\"ANSWER: {{answer}}\")\n"
        "where answer is the name that holds it. "
        "In the same program, compute that same target a SECOND time by a "
        "DIFFERENT method -- if the first computation is symbolic make the second "
        "numeric, and the other way round -- and report that second result by "
        "executing the call\n"
        "print(f\"CHECK: {{second}}\")\n"
        "where second is the name that holds it. Both of those lines are program "
        "code, not output, and either may stand anywhere in the program. "
        "Write the second computation independently: it must not reuse the first "
        "computation's result or any variable that holds it. Print both values "
        "exactly as they come out, even if they disagree; do not change either "
        "one to make them match. "
        "Reply with a single ```python code block."
    ),
    "ko": (
        "{problem}\n\n"
        "\uc774 \ubb38\uc81c\ub97c \ud478\ub294 \ud30c\uc774\uc36c \ud504\ub85c\uadf8\ub7a8\uc744 \uc791\uc131\ud558\uc138\uc694. \ucf54\ub4dc \uccab \uc904\uc5d0 '# \uad6c\ud558\ub294 \uac83: \u2026' "
        "\uc8fc\uc11d\uc73c\ub85c \ubb38\uc81c\uac00 \ucd5c\uc885\uc801\uc73c\ub85c \uc694\uad6c\ud558\ub294 \uac12\uc774 \ubb34\uc5c7\uc778\uc9c0 \uc815\ud655\ud788 \uc801\uc73c\uc138\uc694 "
        "(\uc608: \ud569\uc744 \ubb3c\uc73c\uba74 \uacf1\uc774 \uc544\ub2c8\ub77c \ud569). sympy, math, fractions\ub97c \uc0ac\uc6a9\ud560 \uc218 \uc788\uc2b5\ub2c8\ub2e4. "
        "\uadf8 \ucd5c\uc885 \ub2f5\uc740 \ub2e4\uc74c \ud638\ucd9c\uc744 \uc2e4\ud589\ud574\uc11c \ub118\uae30\uc138\uc694\n"
        "print(f\"ANSWER: {{answer}}\")\n"
        "\uc5ec\uae30\uc11c answer\ub294 \uadf8 \uac12\uc744 \ub2f4\uc740 \uc774\ub984\uc785\ub2c8\ub2e4. "
        "\uac19\uc740 \ud504\ub85c\uadf8\ub7a8 \uc548\uc5d0\uc11c \uadf8 \uac12\uc744 \uc11c\ub85c \ub2e4\ub978 \ubc29\ubc95\uc73c\ub85c \ud55c \ubc88 \ub354 \uad6c\ud558\uc138\uc694 \u2014 \uccab "
        "\uacc4\uc0b0\uc774 \uae30\ud638\uc801(symbolic)\uc774\uba74 \ub450 \ubc88\uc9f8\ub294 \uc218\uce58\uc801(numeric)\uc73c\ub85c, \uadf8 \ubc18\ub300\ub3c4 "
        "\ub9c8\ucc2c\uac00\uc9c0\uc785\ub2c8\ub2e4 \u2014 \uadf8\ub9ac\uace0 \uadf8 \ub450 \ubc88\uc9f8 \uacb0\uacfc\ub294 \ub2e4\uc74c \ud638\ucd9c\uc744 \uc2e4\ud589\ud574\uc11c \ub118\uae30\uc138\uc694\n"
        "print(f\"CHECK: {{second}}\")\n"
        "\uc5ec\uae30\uc11c second\ub294 \uadf8 \uac12\uc744 \ub2f4\uc740 \uc774\ub984\uc785\ub2c8\ub2e4. \uc774 \ub450 \uc904\uc740 \ubaa8\ub450 \ucd9c\ub825\uc774 \uc544\ub2c8\ub77c "
        "\uc2e4\ud589\ub418\ub294 \ud504\ub85c\uadf8\ub7a8 \ucf54\ub4dc\uc774\uba70, \ud504\ub85c\uadf8\ub7a8 \uc548 \uc5b4\ub514\uc5d0 \ub450\uc5b4\ub3c4 \ub429\ub2c8\ub2e4. \ub450 \ubc88\uc9f8 \uacc4\uc0b0\uc740 "
        "\ub3c5\ub9bd\uc801\uc73c\ub85c \uc791\uc131\ud574\uc57c \ud569\ub2c8\ub2e4: \uccab \uacc4\uc0b0\uc758 \uacb0\uacfc\ub098 \uadf8 \uac12\uc744 \ub2f4\uc740 \ubcc0\uc218\ub97c "
        "\uc7ac\uc0ac\uc6a9\ud558\uc9c0 \ub9c8\uc138\uc694. \ub450 \uac12\uc774 \uc11c\ub85c \ub2e4\ub974\ub354\ub77c\ub3c4 \ub098\uc628 \uadf8\ub300\ub85c \ub458 \ub2e4 \ucd9c\ub825\ud558\uace0, "
        "\uc11c\ub85c \ub9de\ucd94\ub824\uace0 \uc5b4\ub290 \ucabd\ub3c4 \uace0\uce58\uc9c0 \ub9c8\uc138\uc694. "
        "```python \ucf54\ub4dc \ube14\ub85d \ud558\ub098\ub85c\ub9cc \ub2f5\ud558\uc138\uc694."
    ),
}

#: (c) the base-N output contract.  `tir_en` word for word, plus ONE sentence
#: about how a base-N answer is to be PRINTED.  It is not a third proposal from
#: the design document: it fixes an output contract of our own that turns a
#: right derivation into a wrong answer.
#:
#: `docs/error_set_characterisation.md` section 3.1 measured it.  On
#: `math500_gemma4-e4b_pool` five of the six `form_rejected` items are a missing
#: base marker (`52_8`/`40_9`/`2516_8`/`204_5`/`4343_6`), four of them sit in
#: that cell's seven-item confident-and-unanimous-wrong bucket, and the program
#: had the digits right every time.  The grader is NOT the place to fix this: a
#: rule that accepted `204` for `204_5` would accept it for `204_9` too, which
#: is why that normalisation was declined and why `NOTATION_RULES` has no
#: base-subscript entry.  So the instruction goes where the loss is made.
#:
#: What the sentence is allowed to do is bounded, and the test pins both halves:
#: it says how to PRINT an answer and nothing about how to REACH one.  It names
#: no problem type, no subject, no method, and the worked example is `1011_2`,
#: a base-2 numeral that occurs in none of MATH-500's 500 committed golds (and
#: none of the six base-N golds among them is base 2), so it cannot leak an
#: answer to any item this template is ever run on.
#:
#: Record fields: none new, on either the exec row or the grade row.  It is
#: `tir_en` plus a sentence, so `answer_line`, grading and every committed
#: comparison keep exactly the shape they have.
TIR1N_TEMPLATE = {
    "en": (
        "{problem}\n\n"
        "Write a Python program that solves this problem. Start the code with a "
        "comment '# target: ...' restating EXACTLY what the problem asks for "
        "(e.g. if it asks for a sum, the target is the sum, not the product), "
        "and print ONLY that final answer as the last line. "
        "You may use sympy, math, and fractions. "
        "If that final answer is a numeral in a base other than ten, print it "
        "with its base in the notation the problem itself uses, for example "
        "1011_2 rather than 1011. "
        "Reply with a single ```python code block."
    ),
    "ko": (
        "{problem}\n\n"
        "이 문제를 푸는 파이썬 프로그램을 작성하세요. 코드 첫 줄에 '# 구하는 것: …' "
        "주석으로 문제가 최종적으로 요구하는 값이 무엇인지 정확히 적으세요 "
        "(예: 합을 물으면 곱이 아니라 합). 마지막 줄에는 그 최종 답 하나만 "
        "출력(print)하세요. sympy, math, fractions를 사용할 수 있습니다. "
        "그 최종 답이 십진법이 아닌 다른 밑의 수라면, 문제가 쓰는 표기 그대로 밑을 "
        "함께 출력하세요 — 예를 들어 1011이 아니라 1011_2 입니다. "
        "```python 코드 블록 하나로만 답하세요."
    ),
}

#: (a, second attempt) the SAME device as TIR2_TEMPLATE -- a program plus an
#: independently stated prose answer -- with ONE sentence added that says where
#: the working goes.  It is not a new proposal: `tir2_en`'s answer contract is
#: kept WORD FOR WORD, and the added sentence is aimed at the one thing the
#: damage sample measured going wrong.
#:
#: What was measured, on the committed 40-item damage sample's own two arms
#: (`math500_gemma4-e2b_damage_tir` and `..._tir2`, both `num_predict: 2048`,
#: same items, greedy):
#:
#:   * `tir2` hit the cap on 10 of 40 generations, the control on 5, and the 5
#:     extra are `math500-10`, `-21`, `-170`, `-333`, `-401`.  No item truncated
#:     under the control and finished under `tir2`.
#:   * The cost is NOT the answer line.  Segmenting each generation's own token
#:     stream at its code fences: on the 30 items neither arm truncated, `tir2`
#:     spent 7,156 more tokens than the control in total, and the region AFTER
#:     the closing fence -- the `Answer:` line itself, all of it -- accounts for
#:     299 of them, 4.2%.  That line costs a median of 8 tokens (min 8, max 24).
#:   * The cost is prose BEFORE the code block.  The control opens `` ```python ``
#:     as its very first token on 39 of 40 generations (median tokens before the
#:     fence: 0).  `tir2`'s median is 324 and its mean 753; 9 of its 40
#:     generations never opened a fence at all inside the budget, and 7 of those
#:     never wrote a line of Python.  The programs themselves got SHORTER
#:     (paired median -32 tokens): the derivation moved out of the comments and
#:     into free-running LaTeX prose ahead of the program.
#:   * It is not degenerate repetition.  Nine of the ten truncated generations
#:     have a duplicate-line fraction at or below 0.062, and the five EXTRA
#:     truncations are the lowest of all (0.000 to 0.043).  The one outlier is
#:     `math500-239` at 0.154 -- an item the CONTROL truncated too -- and even
#:     there no line repeats twice in a row.  Nothing here is a loop.
#:
#: So the added sentence names the POSITION of the working, which is the thing
#: that was measured, and not its length, which was not.  No token or line
#: budget is asked for: the control's own generations show the reasoning fits
#: inside 2048 when it is written as program comments (35 of 40 complete), and a
#: number chosen here would be a number we invented about how much work a
#: problem deserves.  The sentence says where to write, never what to think --
#: it names no problem type, no subject and no method, exactly as
#: `TIR1N_TEMPLATE`'s added sentence says how to PRINT and not how to reach.
#:
#: The answer stays AFTER the program, and that was decided on the records
#: rather than on taste:
#:
#:   * Ordering cannot buy back the comparison.  `program_vs_prose_agree` needs
#:     BOTH halves, and all 10 truncated `tir2` generations produced no exec row
#:     at all -- an unclosed fence yields no program to run.  Moving the answer
#:     line to the front would rescue it on exactly the generations where the
#:     other half is gone anyway.
#:   * Putting the answer first would make the program a rationalisation of it.
#:     That is `TIR3_TEMPLATE`'s own lesson -- "a second derivation copied from
#:     the first agrees by construction and measures nothing" -- with the copy
#:     running the other way.  The committed arm shows the present order is not
#:     degenerate: 19 agree, 9 disagree, 12 uncomputable.
#:   * What answer-first would really buy is a fallback on truncation, and this
#:     template buys that instead by REMOVING the truncations and by reading the
#:     answer line by its label (below), so a truncated generation yields no
#:     prose answer rather than a scraped fragment.
#:
#: Record fields: none new.  `prose_answer` is the field `tir2_*` already fills;
#: what changes for THIS template alone is how it is read -- see
#: `PROSE_ANSWER_LABEL_TEMPLATES` and `parse_prose_answer_line`.  Every
#: committed row of every other template keeps exactly the shape it has.
#:
#: TIR2_TEMPLATE is NOT edited: its 40 damage-sample generations are the
#: evidence for the diagnosis above, and `template_sha256` separates the cells.
TIR2B_TEMPLATE = {
    "en": (
        "{problem}\n\n"
        "Write a Python program that solves this problem. Start the code with a "
        "comment '# target: ...' restating EXACTLY what the problem asks for "
        "(e.g. if it asks for a sum, the target is the sum, not the product), "
        "and print ONLY that final answer as the last line. "
        "You may use sympy, math, and fractions. "
        "Begin your reply with the code block itself: write nothing before it, "
        "and keep any working you want to show in comments inside the program. "
        "Reply with a single ```python code block, and after the block, on a line "
        "of its own outside it, write\n"
        "Answer: \\boxed{{<value>}}\n"
        "where <value> is the answer you reached by working the problem out "
        "yourself. The program is not run before you write that line, so it must "
        "be your own result and not a copy of what the program prints."
    ),
    "ko": (
        "{problem}\n\n"
        "\uc774 \ubb38\uc81c\ub97c \ud478\ub294 \ud30c\uc774\uc36c \ud504\ub85c\uadf8\ub7a8\uc744 \uc791\uc131\ud558\uc138\uc694. \ucf54\ub4dc \uccab \uc904\uc5d0 '# \uad6c\ud558\ub294 \uac83: \u2026' "
        "\uc8fc\uc11d\uc73c\ub85c \ubb38\uc81c\uac00 \ucd5c\uc885\uc801\uc73c\ub85c \uc694\uad6c\ud558\ub294 \uac12\uc774 \ubb34\uc5c7\uc778\uc9c0 \uc815\ud655\ud788 \uc801\uc73c\uc138\uc694 "
        "(\uc608: \ud569\uc744 \ubb3c\uc73c\uba74 \uacf1\uc774 \uc544\ub2c8\ub77c \ud569). \ub9c8\uc9c0\ub9c9 \uc904\uc5d0\ub294 \uadf8 \ucd5c\uc885 \ub2f5 \ud558\ub098\ub9cc "
        "\ucd9c\ub825(print)\ud558\uc138\uc694. sympy, math, fractions\ub97c \uc0ac\uc6a9\ud560 \uc218 \uc788\uc2b5\ub2c8\ub2e4. "
        "\ub2f5\uc740 \ucf54\ub4dc \ube14\ub85d\uc73c\ub85c \uc2dc\uc791\ud558\uc138\uc694: \uadf8 \uc55e\uc5d0\ub294 \uc544\ubb34\uac83\ub3c4 \uc4f0\uc9c0 \ub9d0\uace0, \ubcf4\uc774\uace0 \uc2f6\uc740 "
        "\ud480\uc774 \uacfc\uc815\uc740 \ud504\ub85c\uadf8\ub7a8 \uc548 \uc8fc\uc11d\uc5d0 \uc801\uc73c\uc138\uc694. "
        "```python \ucf54\ub4dc \ube14\ub85d \ud558\ub098\ub85c \ub2f5\ud558\uace0, \uadf8 \ube14\ub85d \ubc14\uae65\uc5d0 \ud55c \uc904\uc744 \ub354 \uc368\uc11c\n"
        "Answer: \\boxed{{<\uac12>}}\n"
        "\ud615\uc2dd\uc73c\ub85c \ub2f5\uc744 \uc801\uc73c\uc138\uc694. <\uac12>\uc740 \ub2f9\uc2e0\uc774 \uc9c1\uc811 \ud480\uc5b4\uc11c \uc5bb\uc740 \ub2f5\uc785\ub2c8\ub2e4. \uadf8 \uc904\uc744 "
        "\uc4f0\ub294 \uc2dc\uc810\uc5d0 \ud504\ub85c\uadf8\ub7a8\uc740 \uc2e4\ud589\ub418\uc9c0 \uc54a\uc73c\ubbc0\ub85c, \ud504\ub85c\uadf8\ub7a8\uc774 \ucd9c\ub825\ud560 \uac12\uc744 \uc62e\uaca8 \uc801\uc740 "
        "\uac83\uc774 \uc544\ub2c8\ub77c \ub2f9\uc2e0 \uc790\uc2e0\uc758 \uacc4\uc0b0 \uacb0\uacfc\uc5ec\uc57c \ud569\ub2c8\ub2e4."
    ),
}


#: template id -> (template dict, language). The Phase 0 greedy anchors were
#: `format` on numeric benchmarks and `tir` on MATH-500
#: (the Phase 0 code/load_records.py:54-67; mome/evalkit/matrix.py:246-260, 384-400).
TEMPLATES = {
    "format_en": (FORMAT_TEMPLATE, "en"),
    "format_ko": (FORMAT_TEMPLATE, "ko"),
    "tir_en": (TIR_TEMPLATE, "en"),
    "tir_ko": (TIR_TEMPLATE, "ko"),
    "cot_en": (COT_TEMPLATE, "en"),
    "cot_ko": (COT_TEMPLATE, "ko"),
    "tir2_en": (TIR2_TEMPLATE, "en"),
    "tir2_ko": (TIR2_TEMPLATE, "ko"),
    "tir3_en": (TIR3_TEMPLATE, "en"),
    "tir3_ko": (TIR3_TEMPLATE, "ko"),
    "tir3b_en": (TIR3B_TEMPLATE, "en"),
    "tir3b_ko": (TIR3B_TEMPLATE, "ko"),
    "tir3c_en": (TIR3C_TEMPLATE, "en"),
    "tir3c_ko": (TIR3C_TEMPLATE, "ko"),
    "tir1n_en": (TIR1N_TEMPLATE, "en"),
    "tir1n_ko": (TIR1N_TEMPLATE, "ko"),
    "tir2b_en": (TIR2B_TEMPLATE, "en"),
    "tir2b_ko": (TIR2B_TEMPLATE, "ko"),
}

#: Which route each template family belongs to.  A config that names a template
#: explicitly is checked against this, so an `expression` cell cannot be run on
#: `format_en` (the record would say `route: "expression"` and carry no program).
TEMPLATE_ROUTE = {"format": "numeric", "cot": "numeric",
                  "tir": "expression", "tir2": "expression", "tir3": "expression",
                  "tir3b": "expression", "tir3c": "expression",
                  "tir1n": "expression", "tir2b": "expression"}

#: The templates that order the program to print a second, independently
#: computed value as a labelled stdout line.  `execute.py` parses `check_lines`
#: and `grade.py` writes `derivation_agree` for these records and for NO other,
#: which is what keeps every committed `tir_en` exec and grade row exactly the
#: shape it already has.
CHECK_LINE_TEMPLATES = frozenset({"tir3_en", "tir3_ko", "tir3b_en", "tir3b_ko",
                                  "tir3c_en", "tir3c_ko"})

#: The templates that order the FINAL ANSWER onto its own labelled call too, so
#: the answer line is read by its label instead of by position
#: (`parse_labelled_answer_line`) and the exec row carries
#: `answer_label_present`.  A strict subset of `CHECK_LINE_TEMPLATES`: every
#: template here also orders a check line, and no template outside it changes
#: how its answer line is computed, which is what keeps every committed
#: `tir_en`, `tir2_en`, `tir1n_en`, `tir3_en` and `tir3b_en` exec row exactly
#: the shape it already has.
ANSWER_LABEL_TEMPLATES = frozenset({"tir3c_en", "tir3c_ko"})

#: The templates that ask for the answer in prose OUTSIDE the code block, so
#: `prose_answer` -- and with it `program_vs_prose_agree` -- is populated.
#: Recorded for the same reason: a report must be able to say which cells were
#: asked for a prose answer without re-reading the template text.
PROSE_ANSWER_TEMPLATES = frozenset({"tir2_en", "tir2_ko", "tir2b_en", "tir2b_ko"})

#: The templates whose prose answer is read by its LABEL instead of by running
#: `extract.extract_final_answer` over everything outside the code block.  A
#: strict subset of `PROSE_ANSWER_TEMPLATES`, and `tir2_*` is deliberately NOT
#: in it: those 40 committed generations keep the `prose_answer` they were
#: recorded with, byte for byte.
#:
#: Why the label is read instead of the region.  On the committed damage
#: sample, all 10 truncated `tir2` generations left an UNCLOSED fence, and
#: `prose_outside_code` strips `` ```...``` `` pairs only -- so the whole partial
#: program counted as prose and `extract_final_answer` fell through to its last
#: rules and returned a fragment of the derivation: `'5.91'`, `'P'`,
#: `'= \frac{|(k_2-k_1)(k_4-k_3)|}{'`, `'0,'`.  Nine of the ten graded wrong
#: on those fragments and one (`math500-333`) graded RIGHT on one, which is
#: worse.  A contract that was not kept must cost what it costs and must not be
#: rescued -- or convicted -- by a scrape.
PROSE_ANSWER_LABEL_TEMPLATES = frozenset({"tir2b_en", "tir2b_ko"})


def template_for(route: str, language: str) -> str:
    """Template id for a route ('numeric' -> format, 'expression' -> tir)."""
    base = "tir" if route == "expression" else "format"
    if language not in ("en", "ko"):
        raise ValueError(f"unknown language {language!r}")
    return f"{base}_{language}"


def resolve_template_id(route: str, language: str, override: str | None = None) -> str:
    """The template a campaign uses: the route default, or the config's own id.

    `override` is the optional config key `prompt_template_id`.  With it absent
    -- which is every committed config -- this is `template_for` and nothing
    about a committed cell changes.  With it present the id must exist, must be
    in this language, and must belong to this route, so a typo is a refusal
    before any generation is bought rather than a cell that silently ran on the
    wrong prompt.
    """
    if override is None:
        return template_for(route, language)
    if override not in TEMPLATES:
        raise ValueError(f"unknown prompt_template_id {override!r}; "
                         f"known ids: {', '.join(sorted(TEMPLATES))}")
    base, _, lang = override.rpartition("_")
    if lang != language:
        raise ValueError(f"prompt_template_id {override!r} is not language {language!r}")
    if TEMPLATE_ROUTE.get(base) != route:
        raise ValueError(f"prompt_template_id {override!r} belongs to route "
                         f"{TEMPLATE_ROUTE.get(base)!r}, not {route!r}")
    return override


def wants_check_lines(template_id: str | None) -> bool:
    """Does this template order a `CHECK: <value>` line before the answer line?"""
    return template_id in CHECK_LINE_TEMPLATES


def wants_answer_label(template_id: str | None) -> bool:
    """Does this template order the final answer onto its own `ANSWER:` call?"""
    return template_id in ANSWER_LABEL_TEMPLATES


def wants_prose_answer(template_id: str | None) -> bool:
    """Does this template ask for the answer in prose outside the code block?"""
    return template_id in PROSE_ANSWER_TEMPLATES


def wants_prose_answer_label(template_id: str | None) -> bool:
    """Is this template's prose answer read by its LABEL (`parse_prose_answer_line`)?"""
    return template_id in PROSE_ANSWER_LABEL_TEMPLATES


def build_prompt(template_id: str, problem: str) -> str:
    table, lang = TEMPLATES[template_id]
    # mome/evalkit/matrix.py:250 and mome/solver/pipeline.py:144 pass the raw
    # question through str.format(problem=...) without stripping.
    return table[lang].format(problem=problem)


def template_sha256(template_id: str) -> str:
    table, lang = TEMPLATES[template_id]
    return hashlib.sha256(table[lang].encode("utf-8")).hexdigest()


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


# reimplemented from mome/inference/probe.py:74,78-80 (extract_code_block)
_CODE_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code_block(text: str) -> str | None:
    match = _CODE_FENCE.search(text)
    return match.group(1).strip() if match else None


# reimplemented from mome/solver/pipeline.py:121-126 (_prose_outside_code)
_FENCED = re.compile(r"```.*?```", re.DOTALL)


def prose_outside_code(text: str) -> str:
    """The completion with fenced code removed - what the model actually SAID."""
    return _FENCED.sub(" ", text)


# --- the `Answer:` contract of TIR2B_TEMPLATE -------------------------------
#
# The parser lives beside the template that mandates the line, for the same
# reason `parse_check_lines` does: the shape asked for and the shape read back
# cannot drift apart.  It is used ONLY for records whose template is in
# `PROSE_ANSWER_LABEL_TEMPLATES`; `tir2_*`'s committed rows keep the region
# scrape they were recorded with.

#: At most this many characters of one answer value reach the record, the same
#: bound `MAX_CHECK_VALUE_CHARS` puts on a check value and for the same reason.
MAX_PROSE_ANSWER_CHARS = 400

#: `Answer: <value>`, matched case-insensitively and allowing `=`, exactly as
#: `_CHECK_LINE` and `_ANSWER_LINE` are matched: what is being read is whether
#: the model stated an answer of its own at all, and refusing `Answer = 4`
#: would count a kept contract as a refusal.
_PROSE_ANSWER_LINE = re.compile(r"^answer\s*[:=]\s*(.+?)$", re.IGNORECASE)


def _prose_region(text: str | None) -> str:
    """`text` with every code block removed -- an unclosed one included.

    `prose_outside_code` removes matched ```` ``` ```` pairs and nothing else,
    which is right for the records it already wrote and wrong for a truncated
    generation: the fence opens, the budget runs out, the pair never closes and
    the whole partial PROGRAM counts as prose.  Every one of the 10 truncated
    `tir2` generations on the committed damage sample is that case.  After the
    pairs are gone any surviving ```` ``` ```` is unpaired by construction, so it
    opens a block that runs to the end of the completion, and everything from it
    on is code.
    """
    stripped = _FENCED.sub(" ", text or "")
    cut = stripped.find("```")
    return stripped if cut < 0 else stripped[:cut]


def parse_prose_answer_line(text: str | None) -> str | None:
    """The value of the labelled `Answer:` line, WHEREVER it stands, or None.

    Position-independent, which is the standing lesson of this repository:
    `parse_check_lines` scanned `lines[:-1]` and threw away 11 kept contracts,
    and `parse_labelled_answer_line` exists because "the last non-check line"
    picked the middle print of three.  Here it means the template may put the
    line after the block (which `TIR2B_TEMPLATE` asks for) and a generation that
    puts it first, or after a blank line, is read the same way.

    The LAST such line wins, for the reason `parse_labelled_answer_line` gives:
    a model that re-states after a correction means the final one.

    Only the PROSE region is scanned (`_prose_region`), so a bare `Answer: 4`
    line inside the program is not accepted.  That line is not output at all --
    Python parses it as a PEP 526 annotation and it prints nothing, which is the
    failure that cost `tir3` 37 of its 60 generations -- and reading it here
    would report a contract as kept that the program never carried out.

    When there is no such line the contract was not kept and this returns None.
    Nothing is scraped in its place: on the committed damage sample the scrape
    returned `'5.91'`, `'P'` and `'0,'` from mid-derivation and graded one
    generation RIGHT on a fragment.
    """
    out: str | None = None
    for line in _prose_region(text).splitlines():
        m = _PROSE_ANSWER_LINE.match(line.strip())
        if m is not None:
            out = m.group(1).strip()[:MAX_PROSE_ANSWER_CHARS]
    return out


# --- the `CHECK:` contract of TIR3_TEMPLATE ---------------------------------
#
# The parser lives beside the template that mandates the line, so the shape
# asked for and the shape read back cannot drift apart.  `execute.py` calls it
# on the sandbox's stdout; nothing else in Phase 1 reads stdout by prefix.

#: At most this many check lines are kept from one program's stdout, and each
#: value is cut to this many characters.  A bound is needed because the field
#: goes into every exec row of a `tir3` cell and a program that prints in a loop
#: would otherwise put a megabyte of stdout into `exec.jsonl` twice over
#: (`stdout` already holds it, cut at 8,000 characters by `execute._cut`).
MAX_CHECK_LINES = 8
MAX_CHECK_VALUE_CHARS = 400

#: `CHECK: <value>`, as the check-line templates order it, matched
#: case-insensitively and allowing `=` for the same reason: what is being
#: measured is whether the model emitted a second value at all, and refusing
#: `check = 4` would count a complied-with instruction as a refusal.  The label
#: is kept as matched (upper-cased) so a report can tell `CHECK` from `CHECK2`.
_CHECK_LINE = re.compile(r"^(check[a-z0-9_]{0,15})\s*[:=]\s*(.+?)$", re.IGNORECASE)


def _stdout_lines(stdout: str | None) -> list[str]:
    return [ln.strip() for ln in (stdout or "").splitlines() if ln.strip()]


def parse_check_lines(stdout: str | None) -> list[dict]:
    """Every labelled `CHECK:` line of one execution, WHEREVER it stands.

    Position-invariant since `docs/tir3_diagnosis.md`.  The first
    version of this scanned `lines[:-1]` only, because the last non-empty line
    is the answer line (`execute.run_code`, MOME's `mome/solver/sandbox.py:48-52`
    convention) and the template asked for the check on the line before it.  The
    committed `tir3` arm shows what that cost: 11 of its 59 executed generations
    printed the label exactly as ordered but LAST, so the parser read the check
    line as the answer, returned no check line at all, and graded all 11 wrong
    (7 of them are right when their other line is read).  A contract that is
    kept but in the wrong order is compliance, not refusal, and the label -- not
    the line number -- is what says which line is which.

    `parse_answer_line` is the other half of the same decision and must be used
    with this one: with the label now recognised anywhere, the answer line is
    the last non-empty line that is NOT a check line, so a printed `CHECK:` can
    never be mistaken for the answer.
    """
    out: list[dict] = []
    for line in _stdout_lines(stdout):
        m = _CHECK_LINE.match(line)
        if m is None:
            continue
        out.append({"label": m.group(1).upper(),
                    "value": m.group(2).strip()[:MAX_CHECK_VALUE_CHARS]})
        if len(out) >= MAX_CHECK_LINES:
            break
    return out


def parse_answer_line(stdout: str | None) -> str | None:
    """The last non-empty stdout line that is not a check line, or None.

    Used ONLY for records whose template ordered a check line
    (`wants_check_lines`); every other record keeps `execute.run_code`'s plain
    "last non-empty line" and therefore its committed value, byte for byte.

    Note that this scans past MAX_CHECK_LINES: the cap bounds what is STORED in
    the exec row, and a program that printed twenty `CHECK:` lines still printed
    no answer on any of them.
    """
    for line in reversed(_stdout_lines(stdout)):
        if _CHECK_LINE.match(line) is None:
            return line
    return None


#: `ANSWER: <value>`, as `TIR3C_TEMPLATE` orders it, matched the same way
#: `_CHECK_LINE` is: case-insensitively and allowing `=`.  It cannot collide
#: with `_CHECK_LINE`, whose label must begin with `check`.
_ANSWER_LINE = re.compile(r"^answer\s*[:=]\s*(.+?)$", re.IGNORECASE)


def parse_answer_label_values(stdout: str | None) -> list[str]:
    """The value of every labelled `ANSWER:` line of one execution, in order."""
    out: list[str] = []
    for line in _stdout_lines(stdout):
        m = _ANSWER_LINE.match(line)
        if m is not None:
            out.append(m.group(1).strip()[:MAX_CHECK_VALUE_CHARS])
    return out


def parse_labelled_answer_line(stdout: str | None) -> str | None:
    """The answer read by its LABEL, falling back to `parse_answer_line`.

    Used ONLY for records whose template ordered the answer onto its own call
    (`wants_answer_label`); every other record keeps the answer line it already
    had, byte for byte.

    `docs/smoke_gate_analysis.md` section 3.2 is why this exists.
    Being shown `print(f"CHECK: {second}")` taught the committed `tir3b` arm's
    model to label its other print too, and `parse_answer_line` hands back the
    last non-check line WITH whatever prefix it carries: 13 of 60 answer lines
    came back as `result1: 13`, `Result from symbolic derivation: 8`,
    `Symbolic solutions found: [10]`.  Ten of the thirteen are graded wrong for
    the label alone, and `grade.derivation_agree` compares that same string to
    the check value, so 8 of that arm's 17 disagreements are the label firing
    rather than the derivations.  Reading the answer by its own label removes
    both at once.

    It also removes the last place position still mattered.  `tir3b`'s
    `math500-69` printed the right answer FIRST, a second derivation on the next
    line and the check line after that; "the last non-check line" picked the
    middle one.  With a label there is no middle.

    The LAST `ANSWER:` line wins, for the same reason the answer line was ever
    the last line: a program that prints in a loop or re-reports after a
    correction means the final one.  When the program printed no `ANSWER:` line
    at all the contract was not kept, and the fallback grades that generation
    exactly as `tir3b` would have -- an unkept contract must cost what it costs
    and must not be rescued by a parser.
    """
    values = parse_answer_label_values(stdout)
    if values:
        return values[-1]
    return parse_answer_line(stdout)


# reimplemented from mome/inference/openai_compat.py:30-47 (strip_think_blocks);
# the native backend applies it to every response (mome/inference/ollama.py:148)
_THINK_OPEN = r"(?:<think>|<\|think\|>|<\|channel>thought\n?|<thought>\n?)"
_THINK_CLOSE = r"(?:</think>|<\|/think\|>|<\|end_think\|>|<\|think_end\|>|<channel\|>|</thought>)"
_THINK_BLOCK = re.compile(_THINK_OPEN + r".*?" + _THINK_CLOSE + r"\s*", re.DOTALL)
_OPEN_THINK = re.compile(_THINK_OPEN + r".*\Z", re.DOTALL)


def strip_think_blocks(text: str) -> str:
    cleaned = _THINK_BLOCK.sub("", text)
    cleaned = _OPEN_THINK.sub("", cleaned)
    return cleaned.strip()
