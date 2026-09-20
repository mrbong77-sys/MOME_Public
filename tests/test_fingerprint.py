"""Unit tests for the fingerprint assembler, on synthetic records only.

    python3 tests/test_fingerprint.py -v           (from the repository root)
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fingerprint as fp  # noqa: E402


def rec(item_id, answer, text="derivation text", tokens=100, truncated=False, error=None):
    return {"item_id": item_id, "extracted_answer": answer, "text_visible": text,
            "token_count": tokens, "truncated": truncated, "error": error}


def meta(probe_id, rule="en.scale.x2", expectation="MUST_CHANGE",
         confidence="medium", direction="increase", scale=2.0):
    return {"probe_id": probe_id, "item_id": probe_id.split("__")[0], "rule": rule,
            "expectation": expectation, "confidence": confidence,
            "direction": direction, "scale": scale, "note": "", "question_sha256": "x"}


class TestAnswersEqual(unittest.TestCase):
    def test_numeric_normalisation(self):
        self.assertTrue(fp.answers_equal("18", "18.0", "numeric"))
        self.assertTrue(fp.answers_equal("1,200", "1200", "numeric"))
        self.assertTrue(fp.answers_equal("1/2", "0.5", "numeric"))
        self.assertFalse(fp.answers_equal("18", "36", "numeric"))

    def test_missing_answer_is_unknown(self):
        self.assertIsNone(fp.answers_equal(None, "18", "numeric"))
        self.assertIsNone(fp.answers_equal("18", "", "numeric"))

    def test_expression_string_and_numeric_paths(self):
        self.assertTrue(fp.answers_equal("\\frac{1}{2}", "\\frac{1}{2}", "expression"))
        self.assertTrue(fp.answers_equal("2.0", "2", "expression"))
        self.assertFalse(fp.answers_equal("\\frac{1}{2}", "\\frac{1}{3}", "expression"))


class TestObserve(unittest.TestCase):
    def test_spurious_consistency_flagged(self):
        o = fp.observe(rec("i", "18"), rec("i", "18"), meta("i__en.scale.x2"), "numeric")
        self.assertFalse(o["changed"])
        self.assertTrue(o["change_violation"])
        self.assertIsNone(o["dir_ok"])  # no change, so no direction either

    def test_direction_and_magnitude_bits(self):
        o = fp.observe(rec("i", "18"), rec("i", "36"), meta("i__en.scale.x2"), "numeric")
        self.assertTrue(o["changed"])
        self.assertFalse(o["change_violation"])
        self.assertTrue(o["dir_ok"])
        self.assertTrue(o["mag_ok"])
        o2 = fp.observe(rec("i", "18"), rec("i", "30"), meta("i__en.scale.x2"), "numeric")
        self.assertTrue(o2["dir_ok"])
        self.assertFalse(o2["mag_ok"])

    def test_hold_bits(self):
        m = meta("i__en.hold.redundant_clause", rule="en.hold.redundant_clause",
                 expectation="MUST_HOLD", confidence="high", direction=None, scale=None)
        ok = fp.observe(rec("i", "18"), rec("i", "18"), m, "numeric")
        self.assertTrue(ok["hold_ok"]) and self.assertFalse(ok["hold_violation"])
        bad = fp.observe(rec("i", "18"), rec("i", "21"), m, "numeric")
        self.assertFalse(bad["hold_ok"])
        self.assertTrue(bad["hold_violation"])

    def test_unanswered_probe(self):
        o = fp.observe(rec("i", "18"), rec("i", None), meta("i__en.scale.x2"), "numeric")
        self.assertFalse(o["answered"])
        self.assertIsNone(o["changed"])
        self.assertIsNone(o["change_violation"])  # no answer is not asserted to be a violation


class TestBuildCellEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.old = (fp.VANILLA_DIR, fp.PROBE_DATA_DIR, fp.MANIFEST_DIR, fp.OUT_DIR)
        fp.VANILLA_DIR = base / "vanilla"
        fp.PROBE_DATA_DIR = base / "probes_data"
        fp.MANIFEST_DIR = base / "manifests"
        fp.OUT_DIR = base / "fingerprints"

        vcell = fp.VANILLA_DIR / "gsm8k_gemma4-e2b_smoking_cb_vanilla"
        pcell = fp.PROBE_DATA_DIR / "gsm8k_gemma4-e2b_ccf_probes_v1"
        for d in (vcell, pcell, fp.MANIFEST_DIR):
            d.mkdir(parents=True)
        # Originals: a is correct (18), b is wrong (7). b answers 7 again on
        # the scaling probe, which is spurious consistency.
        (vcell / "records.jsonl").write_text(
            "\n".join(json.dumps(r) for r in [rec("gsm8k-a", "18"), rec("gsm8k-b", "7")]))
        (vcell / "grades.jsonl").write_text(
            "\n".join(json.dumps(g) for g in [
                {"item_id": "gsm8k-a", "correct": True}, {"item_id": "gsm8k-b", "correct": False}]))
        (pcell / "records.jsonl").write_text(
            "\n".join(json.dumps(r) for r in [
                rec("gsm8k-a__en.scale.x2", "36"), rec("gsm8k-a__en.hold.redundant_clause", "18"),
                rec("gsm8k-b__en.scale.x2", "7"), rec("gsm8k-b__en.hold.redundant_clause", "7"),
                rec("gsm8k-c__en.scale.x2", "1", error="boom"),  # error rows are excluded
            ]))
        probes = [meta("gsm8k-a__en.scale.x2"),
                  meta("gsm8k-a__en.hold.redundant_clause", rule="en.hold.redundant_clause",
                       expectation="MUST_HOLD", confidence="high", direction=None, scale=None),
                  meta("gsm8k-b__en.scale.x2"),
                  meta("gsm8k-b__en.hold.redundant_clause", rule="en.hold.redundant_clause",
                       expectation="MUST_HOLD", confidence="high", direction=None, scale=None),
                  meta("gsm8k-c__en.scale.x2")]
        (fp.MANIFEST_DIR / "gsm8k_manifest.json").write_text(
            json.dumps({"benchmark": "gsm8k", "n_probes": len(probes), "probes": probes}))

    def tearDown(self):
        fp.VANILLA_DIR, fp.PROBE_DATA_DIR, fp.MANIFEST_DIR, fp.OUT_DIR = self.old
        self.tmp.cleanup()

    def test_cell_assembly(self):
        result = fp.build_cell("gsm8k", "gemma4-e2b")
        self.assertEqual(result["items"], 2)
        self.assertEqual(result["spurious"], 1)            # only b is spuriously consistent
        self.assertEqual(result["spurious_among_wrong"], 1)
        self.assertEqual(result["hold_unstable"], 0)
        lines = [json.loads(l) for l in
                 (fp.OUT_DIR / "gsm8k_gemma4-e2b.jsonl").read_text().splitlines()]
        head, rows = lines[0]["_meta"], lines[1:]
        self.assertEqual(head["n_probes_missing"], 1)      # the error row's probe
        self.assertEqual(head["orig_missing"], 1)          # that item has no original
        b = next(r for r in rows if r["item_id"] == "gsm8k-b")
        self.assertFalse(b["label_correct"])
        self.assertTrue(b["summary"]["spurious_consistency"])
        self.assertFalse(b["summary"]["change_confirmed"])

    def test_waiting_when_no_probe_records(self):
        self.assertIsNone(fp.build_cell("hrm8k-gsm8k-ko", "gemma4-e2b"))


class ExpressionEquality(unittest.TestCase):
    """Equality of expression answers, and why the operands must be wrapped.

    Unwrapped, `parse("(a + 5)(b + 2)")` returns `[2]` rather than the
    expression: with no LaTeX anchor the comparator falls back to an extractor
    that picks the last number.  Comparing in that state calls two different
    expressions equal whenever their trailing numbers coincide, and two equal
    expressions different when those numbers differ.  Hold violations are the
    strongest single signal in this method, so that misjudgment corrupts the
    fingerprint directly -- on one competition-set cell it flipped 10.7% of
    the items.
    """

    EQUAL = [("(a + 5)(b + 2)", "{(b + 2)(a + 5)}"),      # braces and factor order
             ("6r^2 + -4r + -24", "6r^2 - 4r - 24"),      # '+ -' spelling
             ("x^2+2x+1", "(x+1)^2"),                     # expanded vs factored
             ("2 \\cdot 3 \\cdot 4 \\cdot 5 \\cdot 1", "120"),
             ("\\frac{1}{2}", "0.5"),
             ("\\boxed{42}", "42")]
    #: Includes pairs the unwrapped comparison called equal because their
    #: trailing numbers agree.
    NOT_EQUAL = [("(a+5)(b+2)", "(c+1)(d+2)"),
                 ("2 \\cdot 3 \\cdot 4 \\cdot 5 \\cdot 1", "2 \\cdot 3 \\cdot 4 \\cdot 6"),
                 ("0.01171875", "0.03515625"),
                 ("-3", "3")]

    def test_equal_pairs(self):
        for a, b in self.EQUAL:
            with self.subTest(a=a, b=b):
                self.assertIs(fp.answers_equal(a, b, "expression"), True)

    def test_unequal_pairs(self):
        for a, b in self.NOT_EQUAL:
            with self.subTest(a=a, b=b):
                self.assertIsNot(fp.answers_equal(a, b, "expression"), True)

    def test_symmetric(self):
        for a, b in self.EQUAL + self.NOT_EQUAL:
            with self.subTest(a=a, b=b):
                self.assertEqual(fp.answers_equal(a, b, "expression"),
                                 fp.answers_equal(b, a, "expression"))

    def test_numeric_route_is_untouched(self):
        """Numeric answers never reach the symbolic comparator, so the
        word-problem cells are untouched by this fix."""
        self.assertIs(fp.answers_equal("42", "42.0", "numeric"), True)
        self.assertIsNot(fp.answers_equal("42", "43", "numeric"), True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
