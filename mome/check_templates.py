#!/usr/bin/env python3
"""Record whether the prompt templates match an external reference source.

The prompts used here were adapted from an earlier codebase, and it was
recorded at the time whether each rewritten template was byte-identical to its
original.  This script performs that comparison: it reads the reference
`solver/pipeline.py` as TEXT (never imported), pulls the template dict
literals out with `ast`, and compares them with `prompts.py`.  The result is
printed as JSON and is embedded by the campaign runners into each campaign's
metadata, so the record travels with the generation rather than living in a
note somewhere.

The reference source is not part of this package.  Point `MOME_ROOT` at it to
run the comparison; with no such source present, `compare()` reports
`present: false` and `all_identical: null`, which is what the campaign
metadata then carries.  Generation is unaffected either way.

Exit code 0 when every template matches, 1 when the source is absent or any
template differs -- and the difference is printed, never hidden.
"""
from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
MOME_ROOT = Path(os.environ.get("MOME_ROOT") or REPO_ROOT)
MOME_PIPELINE = MOME_ROOT / "mome" / "solver" / "pipeline.py"

sys.path.insert(0, str(HERE))
import prompts  # noqa: E402


def _dict_literal(tree: ast.Module, name: str) -> dict | None:
    """Evaluate the `NAME = {Language.EN: "...", Language.KO: "..."}` literal."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            if not isinstance(node.value, ast.Dict):
                return None
            out = {}
            for k, v in zip(node.value.keys, node.value.values):
                key = ast.unparse(k).split(".")[-1].lower()  # Language.EN -> "en"
                out[key] = ast.literal_eval(v)
            return out
    return None


def compare() -> dict:
    result = {"mome_source": str(MOME_PIPELINE), "present": MOME_PIPELINE.exists(),
              "templates": {}, "all_identical": None}
    if not MOME_PIPELINE.exists():
        result["all_identical"] = None
        return result
    tree = ast.parse(MOME_PIPELINE.read_text(encoding="utf-8"))
    ok = True
    prefixes = {"FORMAT_TEMPLATE": "format", "TIR_TEMPLATE": "tir", "COT_TEMPLATE": "cot"}
    for mome_name, ours in (("FORMAT_TEMPLATE", prompts.FORMAT_TEMPLATE),
                            ("TIR_TEMPLATE", prompts.TIR_TEMPLATE),
                            ("COT_TEMPLATE", prompts.COT_TEMPLATE)):
        theirs = _dict_literal(tree, mome_name)
        for lang, text in ours.items():
            same = theirs is not None and theirs.get(lang) == text
            ok = ok and same
            result["templates"][f"{mome_name}[{lang}]"] = {
                "identical": same,
                "runtime_sha256": prompts.template_sha256(prefixes[mome_name] + "_" + lang),
                "mome_text": None if same else (theirs or {}).get(lang),
            }
    result["all_identical"] = ok
    return result


def main() -> int:
    r = compare()
    print(json.dumps(r, indent=1, ensure_ascii=False))
    return 0 if r["all_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
