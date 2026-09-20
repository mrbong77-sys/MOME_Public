#!/usr/bin/env python3
"""Pick the greedy path, and only the greedy path, out of an ungated cell.

An ungated cell is normally read as one record per item, which invites
`{r["item_id"]: r for r in records}`.  That shortcut is wrong the moment a
cell was generated with several paths per item (one greedy plus sampled
ones): the dict comprehension keeps whichever record came last, which is a
sampled answer at temperature 0.7 rather than the greedy answer the gate is
supposed to be measured against.  We hit exactly that on one cell, where the
'ungated accuracy' read 73.7% against a true greedy accuracy of 74.7%.

The fix belongs on the reading side, which is what this module is.  On a cell
that carries a single path it changes nothing, so numbers computed before it
existed are unaffected.
"""

from __future__ import annotations


def greedy_rows(rows: list[dict], where: str = "") -> list[dict]:
    """Keep only the greedy path (`path_index == 0`) when several are present.

    Several paths with no greedy one among them is a hard stop: silently
    reading a sampled answer as the ungated answer is precisely what this
    function exists to prevent.
    """
    paths = {r.get("path_index") for r in rows}
    if len(paths) <= 1:
        return rows
    greedy = [r for r in rows if r.get("path_index") == 0]
    if not greedy:
        raise SystemExit(
            f"{where or 'ungated cell'}: paths are "
            f"{sorted(p for p in paths if p is not None)} "
            "but none is the greedy path (path_index 0). A sampled answer "
            "cannot stand in for the ungated one.")
    return greedy


def by_item(rows: list[dict], where: str = "") -> dict[str, dict]:
    """item_id -> the item's greedy record."""
    return {r["item_id"]: r for r in greedy_rows(rows, where)}
