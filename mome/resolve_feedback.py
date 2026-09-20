#!/usr/bin/env python3
"""Owns the one sentence a re-solve prompt can be prefixed with.

    import resolve_feedback as rf
    rf.prefix(cfg.get("resolve_feedback"), item_id)     # "" when unset

The pool runner imports this module, but it plays no part in the gate this
package is about: none of the configurations published here set
`resolve_feedback`, so `prefix()` returns the empty string throughout and no
prompt is altered.  It is kept because the runner imports it, and because an
explicit no-op is better than a missing import that fails on somebody else's
configuration.

The rule it enforces where it is used is that only a paragraph may be
prepended.  The body template and the output convention are never edited, so
the template hash does not move.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

#: The schema of the committed feedback file.  A file written by an older
#: version is refused rather than read: the prompt text is the experiment.
FEEDBACK_SCHEMA = 1

#: The two fields a config may name.  `told` is one constant sentence; `chan`
#: is the per-item, channel-specific one.  Nothing else is allowed -- a config
#: that names a third field is a config that invented a wording.
FEEDBACK_FIELDS = ("told", "chan")

#: The blunt version, identical on every item.  It asserts one thing and that
#: assertion is FALSE on the anchor-correct declined items; that is the property
#: being tested, and it is recorded in the pre-registration rather than fixed.
TOLD_SENTENCE = (
    "You answered this problem before and your previous answer was wrong. "
    "Solve it again from the beginning, and do not reuse anything from your earlier attempt."
)

#: The channel-specific version.  `{observation}` is filled from the anchor's own
#: sub-kind (`sharpen.subkind_of`), so every sentence is a statement the committed
#: execution evidence actually supports.
CHAN_TEMPLATE = (
    "You answered this problem before. This is what an automatic check of that attempt "
    "observed: {observation}. "
    "Solve it again from the beginning, and do not reuse anything from your earlier attempt."
)

#: sub-kind -> what the channel saw, in words the model can act on.  Every one of
#: the nine names `sharpen.F_SUBKINDS` can return that the committed decline rule
#: (`sel_F | sel_I_structural`) can select is here; a sub-kind with no sentence is
#: a refusal, never a silent fallback to the blunt wording.
#:
#: The two T-side names (`T_ok_agrees`, `X_ok_no_comparison`) are deliberately
#: ABSENT: the rule does not decline on them, so an item carrying one of them is
#: not in the declined set and asking for its sentence is a bug worth stopping on.
OBSERVATION = {
    "I_no_program":
        "your reply contained no Python program at all, so nothing was run",
    "I_not_executed":
        "your reply contained a Python program, but it was never run, so there is no result "
        "from it",
    "F_blocked":
        "your Python program was rejected by a static safety screen before it ran, so it "
        "produced no result",
    "F_timed_out":
        "your Python program ran past the time limit and was stopped before it produced a "
        "result",
    "F_raised":
        "your Python program was run and exited with an error, so it produced no result",
    "F_ok_no_answer_line":
        "your Python program ran to completion, but it printed nothing that could be read as "
        "the answer",
    "F_ok_disagrees":
        "your Python program ran to completion and printed an answer, but that printed answer "
        "and the answer you wrote after the code block did not match",
}

#: `F_raised` alone carries the exception class, when the stored one IS a class
#: name.  One committed row holds a SymPy message in that field instead of a
#: class (`"No algorithms are implemented to solve equation ..."`), so the value
#: is used only when it looks like a dotted Python identifier and is short.
#: Anything else falls back to the class-free sentence above, which is still true.
RAISED_WITH_TYPE = (
    "your Python program was run and raised {exc}, so it produced no result"
)
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
MAX_EXC_CHARS = 40


def usable_exception_type(value) -> str | None:
    """The stored `exception_type`, if it is a class name and not a message."""
    if not isinstance(value, str):
        return None
    v = value.strip()
    if not v or len(v) > MAX_EXC_CHARS or not _IDENT.match(v):
        return None
    return v


def observation(subkind: str, exception_type=None) -> str:
    """What the channel saw, for one item, in one clause.  Raises on an unknown kind."""
    if subkind not in OBSERVATION:
        raise KeyError(
            f"resolve_feedback: no observation sentence for sub-kind {subkind!r}. The committed "
            "decline rule selects only the F sub-kinds and the structural I; a name outside "
            "that set means the wording and the rule have drifted apart, and this file refuses "
            "to guess a sentence rather than send one that may be false.")
    if subkind == "F_raised":
        exc = usable_exception_type(exception_type)
        if exc:
            return RAISED_WITH_TYPE.format(exc=exc)
    return OBSERVATION[subkind]


def told() -> str:
    return TOLD_SENTENCE


def chan(subkind: str, exception_type=None) -> str:
    return CHAN_TEMPLATE.format(observation=observation(subkind, exception_type))


def build_entry(subkind: str, exception_type=None) -> dict:
    """One item's row of the committed feedback file: both wordings and their basis."""
    return {
        "subkind": subkind,
        "exception_type": exception_type,
        "exception_type_used": usable_exception_type(exception_type),
        "told": told(),
        "chan": chan(subkind, exception_type),
    }


# --- the committed file, and how a run reads it -----------------------------

def load(path: Path) -> dict:
    """The committed feedback file, schema-checked."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != FEEDBACK_SCHEMA:
        raise SystemExit(f"{path}: feedback schema {data.get('schema')!r}, expected "
                         f"{FEEDBACK_SCHEMA}; the prompt text IS the experiment and a file of "
                         "another shape is not read.")
    if not isinstance(data.get("by_item"), dict) or not data["by_item"]:
        raise SystemExit(f"{path}: `by_item` is missing or empty")
    return data


def resolve_path(spec: dict) -> Path:
    p = Path(spec["file"])
    return p if p.is_absolute() else REPO / p


_CACHE: dict[str, dict] = {}


def prefix(spec: dict | None, item_id: str) -> str:
    """The paragraph this arm puts BEFORE the unchanged template, or `""`.

    With `resolve_feedback` absent from the config -- which is every committed
    cell and both blind arms -- this returns the empty string and the prompt is
    byte-identical to the one `prompts.build_prompt` has always produced.
    """
    if not spec:
        return ""
    field = spec.get("field")
    if field not in FEEDBACK_FIELDS:
        raise SystemExit(f"config resolve_feedback.field is {field!r}; it must be one of "
                         f"{', '.join(FEEDBACK_FIELDS)}. The wordings are pre-registered in "
                         "mome/resolve_feedback.py and a run does not invent one.")
    key = str(resolve_path(spec))
    data = _CACHE.get(key)
    if data is None:
        data = _CACHE[key] = load(resolve_path(spec))
    row = data["by_item"].get(str(item_id))
    if row is None:
        raise SystemExit(f"{resolve_path(spec)}: no feedback row for item {item_id!r}. Every "
                         "item this arm runs must have one; sending an item through with no "
                         "feedback would make it a blind generation inside an informed arm.")
    text = row.get(field)
    if not text:
        raise SystemExit(f"{resolve_path(spec)}: item {item_id!r} has no {field!r} wording")
    return text + "\n\n"


def build_prompt(prompts_module, template_id: str, question: str,
                 spec: dict | None, item_id: str) -> str:
    """`prefix` + the unchanged template.  The ONE place the two are joined."""
    return prefix(spec, item_id) + prompts_module.build_prompt(template_id, question)
