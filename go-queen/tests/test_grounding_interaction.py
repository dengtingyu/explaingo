"""Synthetic checks for grounding_interaction.py (standard-library unittest).

Written without being executed by its author; results count only once run:
    python -m unittest implementation/test_grounding_interaction.py -v
Fixture label grids copy the first Latin and first reciprocal candidates of
project/outputs/grounding-conflict-audit/report.json; every score is synthetic.
"""
from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path
import random
import sys
import unittest

_PATH = Path(__file__).resolve().parents[1] / "src/goqueen/grounding_interaction.py"
_SPEC = importlib.util.spec_from_file_location("grounding_interaction_under_test", _PATH)
gi = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = gi
_SPEC.loader.exec_module(gi)

LATIN_LABELS = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
LATIN_BOARDS = ["dallas-ply025", "bluespark-ply025", "stv-ply025"]
LATIN_QUERIES = ["W:S6", "W:P8", "W:P7"]
PAIR_LABELS = [[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]]
PAIR_BOARDS = ["pkunzip-ply008", "bluespark-ply008"]
PAIR_QUERIES = ["B:D16", "B:D15", "B:R5", "B:D17", "B:Q16", "B:D4"]


def log_probs(logits, off_label_mass=0.0):
    top = max(logits)
    normalizer = top + math.log(math.fsum(math.exp(v - top) for v in logits))
    return [v - normalizer + math.log1p(-off_label_mass) for v in logits]


def grid(logit_fn, labels, off=lambda b, q: 0.0):
    return [[log_probs(logit_fn(b, q), off(b, q)) for q in range(len(labels[0]))]
            for b in range(len(labels))]


def latin(scores, **options):
    return gi.analyze_grid(scores, LATIN_LABELS, LATIN_BOARDS, LATIN_QUERIES, **options)


def pair(scores, **options):
    return gi.analyze_grid(scores, PAIR_LABELS, PAIR_BOARDS, PAIR_QUERIES, **options)


def additive_logits(rng, boards, queries):
    f = [[rng.uniform(-3, 3) for _ in range(3)] for _ in range(queries)]
    g = [[rng.uniform(-3, 3) for _ in range(3)] for _ in range(boards)]
    h = [[rng.uniform(-3, 3) for _ in range(queries)] for _ in range(boards)]
    return lambda b, q: [f[q][y] + g[b][y] + h[b][q] for y in range(3)]


def reader(labels, kappa):
    return lambda b, q: [kappa if y == labels[b][q] else 0.0 for y in range(3)]


def random_table(rng, labels, scale=5.0):
    table = [[[rng.uniform(-scale, scale) for _ in range(3)] for _ in labels[0]] for _ in labels]
    return lambda b, q: table[b][q]


class DecompositionTests(unittest.TestCase):
    def test_additive_offsets_have_zero_interaction(self):
        rng = random.Random(11)
        off = lambda b, q: 0.002 * (1 + b + 3 * q)
        for _ in range(20):
            for labels, analyze in ((LATIN_LABELS, latin), (PAIR_LABELS, pair)):
                result = analyze(grid(additive_logits(rng, len(labels), len(labels[0])), labels, off))
                d = result["decomposition"]
                self.assertLess(d["I_RMS"], 1e-12)
                self.assertLess(abs(d["S"]), 1e-12)
                self.assertAlmostEqual(d["D"], d["G"], delta=1e-12)
                for block in result["reciprocal"]:
                    for item in block["reversed_label_combinations"]:
                        self.assertLess(abs(item["R"]), 1e-12)

    def test_balanced_pure_interaction_has_zero_board_main_effect(self):
        kappa = 4.0
        result = latin(grid(reader(LATIN_LABELS, kappa), LATIN_LABELS))
        d = result["decomposition"]
        self.assertLess(d["G"], 1e-12)
        self.assertLess(d["T"], 1e-12)
        self.assertAlmostEqual(d["I_RMS"], kappa * math.sqrt(2.0) / 3.0, delta=1e-12)
        self.assertAlmostEqual(d["D"], kappa * math.sqrt(2.0) / 3.0, delta=1e-12)
        self.assertAlmostEqual(d["S"], kappa, delta=1e-12)
        self.assertAlmostEqual(d["S_cosine"], 1.0, delta=1e-12)
        self.assertEqual(result["summary"]["forced_choice_correct"], 9)
        reference = result["additive_reference"]
        self.assertEqual(reference["observed_forced_choice_correct_minus_ceiling"], 2)
        self.assertLess(reference["observed_mean_three_label_ce_minus_infimum"], 0.0)

    def test_wrong_sign_interaction_is_large_with_negative_alignment(self):
        kappa = 4.0
        anti = latin(grid(reader(LATIN_LABELS, -kappa), LATIN_LABELS))
        d = anti["decomposition"]
        self.assertAlmostEqual(d["I_RMS"], kappa * math.sqrt(2.0) / 3.0, delta=1e-12)
        self.assertAlmostEqual(d["S"], -kappa, delta=1e-12)
        self.assertAlmostEqual(d["S_cosine"], -1.0, delta=1e-12)
        self.assertLess(d["G"], 1e-12)
        self.assertEqual(anti["summary"]["forced_choice_correct"], 0)
        shifted_labels = [[(label + 1) % 3 for label in row] for row in LATIN_LABELS]
        shifted = latin(grid(reader(shifted_labels, kappa), LATIN_LABELS))
        d = shifted["decomposition"]
        self.assertAlmostEqual(d["S"], -kappa / 2.0, delta=1e-12)
        self.assertAlmostEqual(d["S_cosine"], -0.5, delta=1e-12)
        self.assertEqual(shifted["summary"]["forced_choice_correct"], 0)

    def test_d_squared_equals_g_squared_plus_i_squared(self):
        rng = random.Random(5)
        for labels, analyze in ((LATIN_LABELS, latin), (PAIR_LABELS, pair)):
            for _ in range(20):
                d = analyze(grid(random_table(rng, labels), labels))["decomposition"]
                self.assertLess(abs(d["D_squared_minus_G_squared_plus_I_squared"]), 1e-10)
                self.assertGreater(d["I_RMS"], 0.0)


class ReciprocalTests(unittest.TestCase):
    def test_reader_and_anti_reader_values(self):
        kappa = 3.0
        for sign in (1.0, -1.0):
            (block,) = pair(grid(reader(PAIR_LABELS, sign * kappa), PAIR_LABELS))["reciprocal"]
            self.assertEqual(block["R_count"], 3)
            for item in block["reversed_label_combinations"]:
                self.assertAlmostEqual(item["R"], 4 * sign * kappa, delta=1e-12)
            for item in block["changed_label_dd"]:
                self.assertAlmostEqual(item["dd"], 2 * sign * kappa, delta=1e-12)

    def test_matches_evaluator_formula(self):
        scores = grid(random_table(random.Random(9), PAIR_LABELS), PAIR_LABELS)
        (block,) = pair(scores)["reciprocal"]
        index = {q: i for i, q in enumerate(PAIR_QUERIES)}
        dd = {item["query_id"]: item["dd"] for item in block["changed_label_dd"]}
        for item in block["reversed_label_combinations"]:
            lo, hi = (gi.LABEL_ORDER.index(name) for name in item["labels"])
            c = lambda q: (scores[0][q][lo] - scores[0][q][hi]) - (scores[1][q][lo] - scores[1][q][hi])
            qf, qr = index[item["forward_query_id"]], index[item["reverse_query_id"]]
            self.assertAlmostEqual(item["R"], c(qf) - c(qr), delta=1e-12)
            self.assertAlmostEqual(item["R"], dd[item["forward_query_id"]] + dd[item["reverse_query_id"]],
                                   delta=1e-12)

    def test_latin_grid_has_no_reversed_label_combinations(self):
        result = latin(grid(reader(LATIN_LABELS, 2.0), LATIN_LABELS))
        self.assertEqual([block["R_count"] for block in result["reciprocal"]], [0, 0, 0])
        self.assertTrue(all(block["R_mean"] is None for block in result["reciprocal"]))


class AdditiveReferenceTests(unittest.TestCase):
    LATIN_F = [[1.0, 0.9, 0.9], [0.9, 1.0, 0.0], [0.9, 0.0, 1.0]]
    LATIN_G = [[0.0, 0.0, 0.0], [0.0, 0.2, -0.2], [0.0, -0.2, 0.2]]
    PAIR_F = [[1.0, 0.5, -10.0], [0.0, 1.0, -10.0], [1.0, -10.0, 0.5],
              [0.0, -10.0, 1.0], [-10.0, 1.0, 0.5], [-10.0, 0.0, 1.0]]
    PAIR_G = [[0.0, 0.0, 0.0], [0.0, 1.0, 2.0]]

    def test_accuracy_ceilings_are_reached_exactly_by_additive_scores(self):
        result = latin(grid(lambda b, q: [self.LATIN_F[q][y] + self.LATIN_G[b][y] for y in range(3)],
                            LATIN_LABELS))
        self.assertEqual(result["additive_reference"]["structure"], "latin_3x3")
        self.assertEqual(result["summary"]["forced_choice_correct"], 7)
        self.assertEqual(result["additive_reference"]["observed_forced_choice_correct_minus_ceiling"], 0)
        self.assertEqual(result["summary"]["forced_choice_ties"], 0)
        self.assertGreater(result["summary"]["min_margin_among_forced_correct"], 0.09)
        self.assertLess(result["decomposition"]["I_RMS"], 1e-12)
        result = pair(grid(lambda b, q: [self.PAIR_F[q][y] + self.PAIR_G[b][y] for y in range(3)],
                           PAIR_LABELS))
        self.assertEqual(result["additive_reference"]["structure"], "two_board_reciprocal_matched")
        self.assertEqual(result["summary"]["forced_choice_correct"], 9)
        self.assertEqual(result["additive_reference"]["accuracy_ceiling_cells"], 9)
        self.assertEqual(result["summary"]["forced_choice_ties"], 0)
        self.assertLess(result["decomposition"]["I_RMS"], 1e-12)

    def test_three_label_ce_infima(self):
        uniform = latin(grid(lambda b, q: [0.0, 0.0, 0.0], LATIN_LABELS))
        self.assertAlmostEqual(uniform["summary"]["mean_three_label_ce"], gi.LN3, delta=1e-15)
        rng = random.Random(3)
        for _ in range(200):
            self.assertGreaterEqual(latin(grid(additive_logits(rng, 3, 3), LATIN_LABELS))
                                    ["summary"]["mean_three_label_ce"], gi.LN3 - 1e-12)
            self.assertGreaterEqual(pair(grid(additive_logits(rng, 2, 6), PAIR_LABELS))
                                    ["summary"]["mean_three_label_ce"], gi.LN2 - 1e-12)
        # ln 2 is approached, not attained, as the unused label's probability vanishes.
        near = pair(grid(lambda b, q: [0.0 if y in (PAIR_LABELS[0][q], PAIR_LABELS[1][q]) else -10.0
                                       for y in range(3)], PAIR_LABELS))
        gap = near["additive_reference"]["observed_mean_three_label_ce_minus_infimum"]
        self.assertGreater(gap, 0.0)
        self.assertLess(gap, 3e-5)

    def test_full_vocabulary_ce_never_below_three_label_ce(self):
        rng = random.Random(21)
        result = pair(grid(additive_logits(rng, 2, 6), PAIR_LABELS, lambda b, q: 0.05 * (1 + b) / (1 + q)))
        for cell in result["cells"]:
            self.assertGreater(cell["full_vocabulary_ce"], cell["three_label_ce"])
            self.assertGreater(cell["off_label_mass"], 0.0)
        for cell in pair(grid(additive_logits(rng, 2, 6), PAIR_LABELS))["cells"]:
            self.assertAlmostEqual(cell["full_vocabulary_ce"], cell["three_label_ce"], delta=1e-12)
            self.assertAlmostEqual(cell["off_label_mass"], 0.0, delta=1e-12)

    def test_combined_reference_for_disjoint_grids(self):
        zeros = lambda b, q: [0.0, 0.0, 0.0]
        latin_result, pair_result = latin(grid(zeros, LATIN_LABELS)), pair(grid(zeros, PAIR_LABELS))
        combined = gi.combine_disjoint([latin_result, pair_result])
        self.assertEqual((combined["cells"], combined["accuracy_ceiling_cells"]), (21, 16))
        self.assertAlmostEqual(combined["three_label_ce_infimum"], (9 * gi.LN3 + 12 * gi.LN2) / 21, delta=1e-15)
        self.assertAlmostEqual(combined["three_label_ce_infimum"], 0.866917941, delta=5e-10)
        scores = grid(zeros, PAIR_LABELS)
        for boards, queries in ((["dallas-ply025", "bluespark-ply008"], PAIR_QUERIES),
                                (PAIR_BOARDS, ["W:S6"] + PAIR_QUERIES[1:])):
            with self.assertRaises(gi.GridError):
                gi.combine_disjoint([latin_result, gi.analyze_grid(scores, PAIR_LABELS, boards, queries)])
        other = gi.analyze_grid(grid(zeros, [[0, 1], [0, 1]]), [[0, 1], [0, 1]], ["a", "b"], ["q1", "q2"])
        self.assertIsNone(gi.combine_disjoint([latin_result, other])["accuracy_ceiling_cells"])

    def test_structure_detection(self):
        zeros = lambda b, q: [0.0, 0.0, 0.0]
        cases = (([[0, 0], [1, 1]], "other"), ([[0, 1], [1, 1]], "other"),
                 ([[0, 1], [1, 0]], "two_board_reciprocal_matched"))
        for labels, structure in cases:
            result = gi.analyze_grid(grid(zeros, labels), labels, ["a", "b"], ["q1", "q2"])
            self.assertEqual(result["additive_reference"]["structure"], structure)
        non_latin = [[0, 1, 2], [0, 1, 2], [1, 2, 0]]
        result = gi.analyze_grid(grid(zeros, non_latin), non_latin, LATIN_BOARDS, LATIN_QUERIES)
        self.assertEqual(result["additive_reference"]["structure"], "other")
        self.assertIsNone(result["additive_reference"]["three_label_ce_infimum"])


class NormalizationTests(unittest.TestCase):
    def test_per_cell_constants_change_only_cross_entropy(self):
        base = grid(random_table(random.Random(13), PAIR_LABELS), PAIR_LABELS)
        shift = lambda b, q: 0.1 * (1 + b + q)
        moved = [[[v - shift(b, q) for v in base[b][q]] for q in range(6)] for b in range(2)]
        a, s = pair(base), pair(moved)
        for key in ("G", "T", "D", "I_RMS", "S", "S_cosine"):
            self.assertAlmostEqual(a["decomposition"][key], s["decomposition"][key], delta=1e-12)
        for x, y in zip(a["reciprocal"][0]["reversed_label_combinations"],
                        s["reciprocal"][0]["reversed_label_combinations"]):
            self.assertAlmostEqual(x["R"], y["R"], delta=1e-12)
        for x, y in zip(a["cells"], s["cells"]):
            b, q = PAIR_BOARDS.index(x["board_id"]), PAIR_QUERIES.index(x["query_id"])
            self.assertEqual(x["forced_choice_label"], y["forced_choice_label"])
            self.assertAlmostEqual(x["margin"], y["margin"], delta=1e-12)
            self.assertAlmostEqual(x["three_label_ce"], y["three_label_ce"], delta=1e-12)
            self.assertAlmostEqual(y["full_vocabulary_ce"] - x["full_vocabulary_ce"], shift(b, q), delta=1e-12)

    def test_exact_ties_are_flagged(self):
        result = latin(grid(lambda b, q: [0.0, 0.0, 0.0], LATIN_LABELS))
        summary = result["summary"]
        self.assertEqual(summary["forced_choice_ties"], 9)
        self.assertEqual(summary["strict_margin_correct"], 0)
        self.assertEqual(summary["forced_choice_correct"], 3)  # first index (empty) on the three empty cells
        self.assertTrue(all(cell["forced_choice_label"] == 0 for cell in result["cells"]))


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.scores = grid(reader(LATIN_LABELS, 1.0), LATIN_LABELS)

    def rejects(self, scores=None, labels=None, boards=None, queries=None, **options):
        with self.assertRaises(gi.GridError):
            gi.analyze_grid(self.scores if scores is None else scores,
                            LATIN_LABELS if labels is None else labels,
                            LATIN_BOARDS if boards is None else boards,
                            LATIN_QUERIES if queries is None else queries, **options)

    def test_rejects_malformed_partial_or_nonfinite_grids(self):
        ragged = copy.deepcopy(self.scores); ragged[1] = ragged[1][:2]
        missing = copy.deepcopy(self.scores); missing[0][1] = None
        short = copy.deepcopy(self.scores); short[2][2] = short[2][2][:2]
        for bad in (ragged, missing, short, self.scores[:1]):
            self.rejects(scores=bad)
        for value in (float("nan"), float("inf"), "-1.0", True, None):
            bad = copy.deepcopy(self.scores); bad[0][0] = [value, -1.0, -2.0]
            self.rejects(scores=bad)
        for triple in ([0.0, 0.0, 0.0], [0.5, -3.0, -3.0]):
            bad = copy.deepcopy(self.scores); bad[0][0] = triple
            self.rejects(scores=bad)
        for labels in ([[0, 1, 2], [1, 2, 0], [2, 0, 3]], [[0, 1, 2], [1, 2, 0], [2, 0, True]],
                       [[0, 1, 2], [1, 2, 0]], [[0, 1, 2], [1, 2, 0], [2, 0]]):
            self.rejects(labels=labels)
        for boards in (["a", "a", "b"], ["a", "b"], ["a", "b", ""]):
            self.rejects(boards=boards)
        self.rejects(queries=["x", "y", 3])
        self.rejects(scores=[[row[0]] for row in self.scores], labels=[[row[0]] for row in LATIN_LABELS],
                     queries=["q"])
        for tolerance in (-1e-9, float("nan")):
            self.rejects(mass_tolerance=tolerance)

    def test_inputs_are_not_mutated_and_raw_values_are_echoed(self):
        scores, labels = copy.deepcopy(self.scores), copy.deepcopy(LATIN_LABELS)
        before = copy.deepcopy((scores, labels))
        result = gi.analyze_grid(scores, labels, LATIN_BOARDS, LATIN_QUERIES)
        self.assertEqual((scores, labels), before)
        for cell in result["cells"]:
            b, q = LATIN_BOARDS.index(cell["board_id"]), LATIN_QUERIES.index(cell["query_id"])
            self.assertEqual(cell["log_probs"], scores[b][q])


if __name__ == "__main__":
    unittest.main()
