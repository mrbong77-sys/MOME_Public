"""Unit tests for the perturbation engine. No data files; inline examples only.

    python3 -m unittest tests.test_perturb -v      (from the repository root)
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perturb import MUST_CHANGE, MUST_HOLD, perturb, select_operating_set  # noqa: E402


class TestEnglishFlips(unittest.TestCase):
    def test_at_least_flips_once(self):
        q = "A club needs at least 5 members to register. How many more are needed?"
        flips = [p for p in perturb(q, "en") if p.rule == "en.flip.at_least"]
        self.assertEqual(len(flips), 1)
        self.assertIn("at most 5 members", flips[0].text)
        self.assertEqual(flips[0].expectation, MUST_CHANGE)
        self.assertEqual(flips[0].confidence, "high")

    def test_ambiguous_flip_is_dropped(self):
        q = "She scored more than 3 and more than 7."  # 'more than' twice -> dropped
        self.assertFalse([p for p in perturb(q, "en") if "more_than" in p.rule])


class TestKoreanFlips(unittest.TestCase):
    def test_isang_flips(self):
        # "A box holds at least 12 apples. What is the minimum number of
        # boxes needed?" -- carries both 이상 (at least) and 최소 (minimum).
        q = "한 상자에 사과가 12개 이상 들어 있다. 최소 몇 상자가 필요한가?"
        rules = {p.rule for p in perturb(q, "ko")}
        self.assertIn("ko.flip.이상", rules)
        self.assertIn("ko.flip.최소", rules)
        flip = next(p for p in perturb(q, "ko") if p.rule == "ko.flip.이상")
        self.assertIn("12개 이하", flip.text)


class TestLatexFlips(unittest.TestCase):
    def test_geq_flip(self):
        q = r"Find the smallest $x$ with $x^2 \ge 30$."
        flips = [p for p in perturb(q, "en") if p.rule == "latex.flip.geq"]
        self.assertEqual(len(flips), 1)
        self.assertIn(r"x^2 \le 30", flips[0].text)

    def test_multiple_inequalities_dropped(self):
        q = r"Solve $a > b$ and $c > d$."
        self.assertFalse([p for p in perturb(q, "en") if p.rule == "latex.flip.gt"])

    def test_arrow_not_flipped(self):
        q = r"Compute $f(x) \to 3$ where $x -> 0$ meets $y \ge 2$."
        gt = [p for p in perturb(q, "en") if p.rule == "latex.flip.gt"]
        self.assertFalse(gt)  # the '>' of '->' is not an inequality


class TestScale(unittest.TestCase):
    Q = "Janet's ducks lay 16 eggs per day. She eats 3 and bakes with 4."

    def test_unique_numbers_scaled(self):
        scaled = [p for p in perturb(self.Q, "en") if p.rule == "en.scale.x2"]
        self.assertTrue(scaled)
        for p in scaled:
            self.assertEqual(p.expectation, MUST_CHANGE)
            self.assertEqual(p.confidence, "medium")
            self.assertEqual(p.scale, 2.0)

    def test_repeated_number_not_scaled(self):
        q = "He bought 5 pens and 5 books, 7 in total boxes."
        notes = [p.note for p in perturb(q, "en") if "scale" in p.rule]
        self.assertFalse(any(n.startswith("5 ") for n in notes))

    def test_year_and_trivial_not_scaled(self):
        q = "Since 2021 she reads 1 book each week."
        self.assertFalse([p for p in perturb(q, "en") if "scale" in p.rule])

    def test_collision_with_existing_number_dropped(self):
        q = "There are 8 red and 16 blue marbles in the game."
        notes = [p.note for p in perturb(q, "en") if "scale" in p.rule]
        self.assertFalse(any(n == "8 -> 16" for n in notes))  # 16 is already present


class TestP06Expansions(unittest.TestCase):
    def test_decimal_scaled_with_places_kept(self):
        q = "One pair of shorts costs $16.50 and a hat costs $9.25."
        notes = [p.note for p in perturb(q, "en") if p.rule == "en.scale.x2"]
        self.assertIn("16.50 -> 33.00", notes)

    def test_word_number_doubled(self):
        q = "Brandon's iPhone is four times as old as Ben's iPhone."
        words = [p for p in perturb(q, "en") if p.rule == "en.scale.word_x2"]
        self.assertEqual(len(words), 1)
        self.assertIn("eight times as old", words[0].text)
        self.assertEqual(words[0].confidence, "medium")

    def test_ko_multiplier_doubled(self):
        # "Cheolsu has twice as many sweets as Younghee." -- 두 배 (twice)
        # doubles to 네 배 (four times).
        q = "철수는 영희보다 두 배 많은 사탕을 가지고 있다."
        words = [p for p in perturb(q, "ko") if p.rule == "ko.scale.word_x2"]
        self.assertEqual(len(words), 1)
        self.assertIn("네 배 많은", words[0].text)

    def test_even_needs_number_noun(self):
        with_noun = "Find the largest even number below 100."
        without = "She wants even more candies than before."
        self.assertTrue([p for p in perturb(with_noun, "en") if p.rule == "en.flip.even"])
        self.assertFalse([p for p in perturb(without, "en") if p.rule == "en.flip.even"])

    def test_positive_flip(self):
        q = "Find the smallest positive integer divisible by 14."
        flips = {p.rule for p in perturb(q, "en")}
        self.assertIn("en.flip.positive", flips)
        self.assertIn("en.flip.smallest", flips)

    def test_dollar_span_inequality_flipped_without_backslash(self):
        q = "Solve $x > 5$ for the smallest integer $x$."
        gt = [p for p in perturb(q, "en") if p.rule == "latex.flip.gt"]
        self.assertEqual(len(gt), 1)
        self.assertIn("$x < 5$", gt[0].text)

    def test_money_text_gets_no_latex_probe(self):
        q = "He pays $2 for eggs and $3 for milk every single day."
        self.assertFalse([p for p in perturb(q, "en") if p.lang == "latex" and p.expectation == MUST_CHANGE])


class TestHoldAndOperatingSet(unittest.TestCase):
    def test_hold_probe_always_present(self):
        for q, lang in [("Add 2 and 9.", "en"), ("2와 9를 더하라.", "ko")]:
            holds = [p for p in perturb(q, lang) if p.expectation == MUST_HOLD]
            self.assertEqual(len(holds), 1)
            self.assertTrue(holds[0].text.startswith(q))

    def test_operating_set_shape(self):
        q = "A club needs at least 5 members. 12 have joined. How many extra?"
        sel = select_operating_set(perturb(q, "en"), k_change=1)
        self.assertEqual(len(sel), 2)
        self.assertEqual(sel[0].expectation, MUST_CHANGE)
        self.assertEqual(sel[0].confidence, "high")  # flips come before scalings
        self.assertEqual(sel[1].expectation, MUST_HOLD)


if __name__ == "__main__":
    unittest.main(verbosity=2)
