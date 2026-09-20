"""Regression tests for the evaluator's decision logic. No data files needed.

The evaluator was written before the second-stage generation finished, and
the rules inside it are pre-registered.  These tests pin them so that the code
cannot drift away from the registered policy unnoticed:

1. **Band handling** -- SERVE emits the greedy answer unchanged; DECLINE
   abstains under the three-band policy and goes to the second stage under
   the two-band one.
2. **An item with no samples counts as a decline** -- that is a generation
   error leaving the paths unfilled, and every variant must treat it the same
   way or the comparison between variants is not honest.
3. **Equivalence-aware voting merges notational variants into one vote** --
   an item that splits under string voting can reach consensus under it.
4. **Rescue and loss accounting** -- a first-stage wrong answer served
   correctly by the second stage is a rescue; a first-stage correct answer
   that is declined, or flipped to wrong, is a loss.
5. **Generation cost** -- serving or declining immediately costs one
   generation; going through the second stage costs five.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "" / "phase1"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_gate as eg  # noqa: E402
import gate  # noqa: E402


def bundle(decision, label, answers, correct=None, kind="numeric", latency=1.0):
    """answers[0] is the greedy path. `correct` gives each path's correctness;
    omitted, it is filled from the label."""
    if correct is None:
        correct = [bool(label)] + [False] * (len(answers) - 1)
    paths = [{"answer": a, "correct": c, "token_count": 10, "truncated": False,
              "mean_logprob": -0.5, "latency": latency}
             for a, c in zip(answers, correct)]
    return {"cell": "gsm8k_gemma4-e2b", "item_id": "x", "kind": kind,
            "decision": decision, "label": label, "c_safe": 0.5,
            "paths": paths, "has_samples": len(paths) > 1}


class Bands(unittest.TestCase):
    def test_serve_band_passes_greedy_through(self):
        b = bundle(gate.SERVE, 1, ["7"])
        r = eg.run_variant([b], two_band=True, equivalence=True)
        self.assertEqual((r["answered"], r["correct"], r["gens"]), (1, 1, 1))

    def test_decline_band_is_abstention_in_three_band(self):
        b = bundle(gate.DECLINE, 1, ["7", "7", "7", "7", "7"], [True] * 5)
        r = eg.run_variant([b], two_band=False, equivalence=True)
        self.assertEqual(r["answered"], 0)
        self.assertEqual(r["gens"], 1, "a declined item buys no samples under the three-band policy")

    def test_decline_band_reaches_gate2_in_two_band(self):
        b = bundle(gate.DECLINE, 1, ["7", "7", "7", "7", "7"], [True] * 5)
        r = eg.run_variant([b], two_band=True, equivalence=True)
        self.assertEqual((r["answered"], r["correct"], r["gens"]), (1, 1, 5))

    def test_missing_samples_counts_as_abstention(self):
        b = bundle(gate.RETRY, 1, ["7"])          # an item whose samples were not filled
        self.assertFalse(b["has_samples"])
        for two in (False, True):
            r = eg.run_variant([b], two_band=two, equivalence=True)
            with self.subTest(two_band=two):
                self.assertEqual(r["answered"], 0)
                self.assertEqual(r["gens"], 1)


class Voting(unittest.TestCase):
    def test_equivalence_merges_notations(self):
        b = bundle(gate.RETRY, 1, ["\\frac{1}{2}", "0.5", "1/2", "0.5", "9"],
                   [True, True, True, True, False], kind="expression")
        _, share_str, _ = eg.vote_of(b, equivalence=False)
        _, share_eq, ok = eg.vote_of(b, equivalence=True)
        self.assertLess(share_str, gate.VOTE_SERVE)
        self.assertGreaterEqual(share_eq, gate.VOTE_SERVE)
        self.assertTrue(ok)

    def test_split_vote_declines(self):
        b = bundle(gate.RETRY, 1, ["7", "1", "2", "3", "4"])
        _, share, _ = eg.vote_of(b, equivalence=False)
        self.assertLess(share, gate.VOTE_SERVE)

    def test_empty_answers_do_not_vote(self):
        b = bundle(gate.RETRY, 1, ["7", None, "", "7", "7"], [True, False, False, True, True])
        ans, share, ok = eg.vote_of(b, equivalence=False)
        self.assertEqual(ans, "7")
        self.assertEqual(share, 1.0, "an empty answer leaves the denominator too")
        self.assertTrue(ok)


class RescueAndLoss(unittest.TestCase):
    def test_rescue_counted(self):
        """First stage wrong, sample majority correct: a rescue."""
        b = bundle(gate.RETRY, 0, ["9", "7", "7", "7", "7"],
                   [False, True, True, True, True])
        r = eg.run_variant([b], two_band=True, equivalence=False)
        self.assertEqual((r["rescued"], r["answered"], r["correct"]), (1, 1, 1))

    def test_loss_by_abstention(self):
        """First stage correct, votes split, item declined: a loss."""
        b = bundle(gate.RETRY, 1, ["7", "1", "2", "3", "4"])
        r = eg.run_variant([b], two_band=True, equivalence=False)
        self.assertEqual((r["lost_decline"], r["answered"]), (1, 0))

    def test_loss_by_wrong_majority(self):
        """First stage correct, sample majority wrong: a loss."""
        b = bundle(gate.RETRY, 1, ["7", "9", "9", "9", "9"],
                   [True, False, False, False, False])
        r = eg.run_variant([b], two_band=True, equivalence=False)
        self.assertEqual((r["lost_wrong"], r["answered"], r["correct"]), (1, 1, 0))


class Aggregates(unittest.TestCase):
    def setUp(self):
        self.bs = [bundle(gate.SERVE, 1, ["7"]),
                   bundle(gate.RETRY, 0, ["9", "7", "7", "7", "7"],
                          [False, True, True, True, True]),
                   bundle(gate.DECLINE, 1, ["7", "1", "2", "3", "4"])]

    def test_three_band_costs_less_and_answers_less(self):
        three = eg.run_variant(self.bs, two_band=False, equivalence=False)
        two = eg.run_variant(self.bs, two_band=True, equivalence=False)
        self.assertLess(three["gens_per_item"], two["gens_per_item"])
        self.assertLessEqual(three["coverage"], two["coverage"])

    def test_yield_is_correct_over_all_items(self):
        r = eg.run_variant(self.bs, two_band=True, equivalence=False)
        self.assertEqual(r["correct"], 2)            # one served plus one rescued
        self.assertAlmostEqual(r["yield"], 2 / 3)
        self.assertAlmostEqual(r["selective_acc"], 1.0)

    def test_generation_cost(self):
        r = eg.run_variant(self.bs, two_band=True, equivalence=False)
        self.assertEqual(r["gens"], 1 + 5 + 5)


class AdoptionRule(unittest.TestCase):
    """The registered adoption rule applies the "at least the ungated model"
    filter first."""

    def test_variants_registered(self):
        self.assertEqual(len(eg.VARIANTS), 4)
        self.assertEqual(set(eg.VARIANTS.values()),
                         {(False, False), (False, True), (True, False), (True, True)})


if __name__ == "__main__":
    unittest.main()
