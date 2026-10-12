"""Synthetic counterexample checks for goqueen.transfer_metrics (standard library only).

Written without being executed by its author; results count only once run.
No model, tensor, file or saved report is used. Integrated location:
tests/test_transfer_metrics.py beside src/goqueen/transfer_metrics.py.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
if (HERE / "transfer_metrics.py").is_file():  # candidate layout: module beside this test
    sys.path.insert(0, str(HERE))
    import transfer_metrics as tm  # noqa: E402
else:  # integrated layout: tests/ beside src/goqueen/
    sys.path.insert(0, str(HERE.parent / "src"))
    from goqueen import transfer_metrics as tm  # noqa: E402

TOKENS = [3194, 11453, 5782]
OFF_LABEL_TOKEN = 99
META = ("question_id", "position_id", "game_id", "split", "ply", "to_play",
        "point_index", "point", "label", "source_task_ids")


def question(i, label, game="g1", side="B"):
    return {"question_id": f"q{i}", "position_id": f"{game}-ply-008", "game_id": game,
            "split": "val", "ply": 8, "to_play": side, "point_index": i, "point": f"P{i}",
            "label": label, "answer": {"occupancy": tm.LABEL_NAMES[label]},
            "prompt": f"canonical prompt {i}", "source_task_ids": [f"q{i}"]}


def row(q, values, forced=None, token=None, prefix=74):
    """A score row shaped like grounding_eval.score_questions output."""
    values = [float(v) for v in values]
    forced = max(range(3), key=values.__getitem__) if forced is None else forced
    token = TOKENS[forced] if token is None else token
    vocab_label = TOKENS.index(token) if token in TOKENS else None
    gold = q["label"]
    return {**{key: copy.deepcopy(q[key]) for key in META}, "prefix_tokens": prefix,
            "label_token_ids": list(TOKENS), "label_log_probs": values, "off_label_mass": 0.01,
            "forced_choice_label": forced, "unconstrained_token_id": token,
            "unconstrained_label": vocab_label, "forced_choice_correct": forced == gold,
            "unconstrained_label_correct": vocab_label == gold, "gold_value_nll": -values[gold]}


def scores_for(label, margin=1.0):
    """Finite <=0 scores whose unique maximum is `label`."""
    values = [-2.0, -2.0, -2.0]
    values[label] = -2.0 + margin
    return values


def json_safe(test, result):
    json.dumps(result, allow_nan=False)  # raises on NaN/inf


class SummaryArithmeticTests(unittest.TestCase):
    def setUp(self):
        # Two games, both sides, two items per class.
        self.questions = [question(0, 0, "g1", "B"), question(1, 1, "g1", "W"), question(2, 2, "g1", "B"),
                          question(3, 0, "g2", "W"), question(4, 1, "g2", "B"), question(5, 2, "g2", "W")]
        predictions = [0, 1, 1, 2, 1, 0]  # correct: q0, q1, q4
        self.rows = [row(q, scores_for(p)) for q, p in zip(self.questions, predictions)]

    def test_counts_confusion_groups_and_macro_recall(self):
        result = tm.summarize_transfer(self.rows, self.questions)
        json_safe(self, result)
        self.assertTrue(result["coverage_complete"])
        self.assertEqual(result["coverage_scope"], "complete")
        self.assertEqual((result["completed_questions"], result["requested_questions"]), (6, 6))
        self.assertEqual(result["confusion_matrix"], [[1, 0, 1], [0, 2, 0], [1, 1, 0]])
        self.assertEqual(result["forced_choice_correct"], 3)
        self.assertEqual(result["forced_choice_accuracy"], 0.5)
        # Balanced list: macro recall (1/2 + 2/2 + 0/2)/3 equals accuracy 3/6.
        self.assertAlmostEqual(result["macro_recall"], 0.5, places=15)
        self.assertEqual(result["by_class"]["white"], {"requested": 2, "completed": 2, "correct": 0, "recall": 0.0})
        self.assertEqual(result["by_game"], {
            "g1": {"requested": 3, "completed": 3, "correct": 2, "accuracy": 2 / 3},
            "g2": {"requested": 3, "completed": 3, "correct": 1, "accuracy": 1 / 3}})
        self.assertEqual(result["by_side"]["B"], {"requested": 3, "completed": 3, "correct": 2, "accuracy": 2 / 3})
        self.assertEqual(result["by_side"]["W"], {"requested": 3, "completed": 3, "correct": 1, "accuracy": 1 / 3})
        self.assertEqual((result["exact_top_tie_count"], result["unique_max_correct"]), (0, 3))

    def test_occupancy_and_color_follow_the_confusion_matrix(self):
        result = tm.summarize_transfer(self.rows, self.questions)
        # Occupancy right: q0 empty->empty, q1 black->black, q2 white->black, q4 black->black.
        self.assertEqual(result["occupancy"]["correct"], 4)
        self.assertEqual((result["occupancy"]["completed"], result["occupancy"]["requested"]), (6, 6))
        # Colour on the four truly occupied items: q1 and q4 right; white->black and white->empty wrong.
        color = result["color_on_true_occupied"]
        self.assertEqual((color["correct"], color["completed"], color["requested"]), (2, 4, 4))
        self.assertEqual(color["rate"], 0.5)

    def test_cross_entropies_match_independent_formulas(self):
        result = tm.summarize_transfer(self.rows, self.questions)
        three = full = 0.0
        for r in self.rows:
            probabilities = [math.exp(v) for v in r["label_log_probs"]]
            three += -math.log(probabilities[r["label"]] / sum(probabilities))
            full += -r["label_log_probs"][r["label"]]
        self.assertAlmostEqual(result["three_label_ce"], three / 6, places=12)
        self.assertAlmostEqual(result["full_vocabulary_ce"], full / 6, places=12)

    def test_large_magnitude_scores_use_stable_logsumexp(self):
        questions = [question(0, 0)]
        result = tm.summarize_transfer([row(questions[0], [-1000.0, -1001.0, -1002.0])], questions)
        self.assertAlmostEqual(result["three_label_ce"], math.log(1 + math.exp(-1) + math.exp(-2)), places=12)
        self.assertEqual(result["full_vocabulary_ce"], 1000.0)
        json_safe(self, result)

    def test_stored_full_vocabulary_argmax_and_off_label_choices(self):
        rows = copy.deepcopy(self.rows)
        rows[2] = row(self.questions[2], scores_for(1), token=OFF_LABEL_TOKEN)  # off-label argmax
        result = tm.summarize_transfer(rows, self.questions)["full_vocabulary_argmax"]
        self.assertEqual((result["correct"], result["off_label_choices"], result["completed"]), (3, 1, 6))

    def test_inputs_are_not_mutated(self):
        rows, questions = copy.deepcopy(self.rows), copy.deepcopy(self.questions)
        tm.summarize_transfer(rows, questions)
        self.assertEqual((rows, questions), (self.rows, self.questions))


class TieTests(unittest.TestCase):
    def test_exact_tie_keeps_first_choice_and_denominator(self):
        questions = [question(0, 1), question(1, 0), question(2, 2)]
        rows = [row(questions[0], [-1.0, -1.0, -3.0]),   # tie empty/black, gold black: first max = empty, wrong
                row(questions[1], [-1.0, -1.0, -3.0]),   # same tie, gold empty: first max right but not unique
                row(questions[2], [-3.0, -2.0, -0.5])]   # unique correct
        self.assertEqual([r["forced_choice_label"] for r in rows], [0, 0, 2])
        result = tm.summarize_transfer(rows, questions)
        self.assertEqual(result["exact_top_tie_count"], 2)
        self.assertEqual(result["forced_choice_correct"], 2)
        self.assertEqual(result["forced_choice_accuracy"], 2 / 3)
        self.assertEqual(result["unique_max_correct"], 1)

    def test_near_tie_is_not_an_exact_tie(self):
        questions = [question(0, 0)]
        result = tm.summarize_transfer([row(questions[0], [-1.0, math.nextafter(-1.0, -2.0), -3.0])], questions)
        self.assertEqual((result["exact_top_tie_count"], result["unique_max_correct"]), (0, 1))

    def test_saved_choice_must_be_the_first_maximum(self):
        questions = [question(0, 1)]
        with self.assertRaises(ValueError):
            tm.summarize_transfer([row(questions[0], [-1.0, -1.0, -3.0], forced=1)], questions)


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.questions = [question(0, 0, "g1"), question(1, 1, "g1"), question(2, 2, "g2"), question(3, 0, "g2")]

    def test_zero_rows(self):
        result = tm.summarize_transfer([], self.questions)
        json_safe(self, result)
        self.assertFalse(result["coverage_complete"])
        self.assertEqual(result["coverage_scope"], "empty")
        self.assertEqual((result["completed_questions"], result["requested_questions"]), (0, 4))
        for key in ("forced_choice_accuracy", "macro_recall", "three_label_ce", "full_vocabulary_ce"):
            self.assertIsNone(result[key], key)
        self.assertEqual(result["confusion_matrix"], [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.assertEqual(result["by_game"]["g2"], {"requested": 2, "completed": 0, "correct": 0, "accuracy": None})
        self.assertEqual(result["color_on_true_occupied"]["requested"], 2)
        self.assertIsNone(result["color_on_true_occupied"]["rate"])

    def test_partial_prefix_keeps_requested_denominators_and_null_uncovered_class(self):
        rows = [row(self.questions[0], scores_for(0)), row(self.questions[1], scores_for(0))]
        result = tm.summarize_transfer(rows, self.questions)
        self.assertFalse(result["coverage_complete"])
        self.assertEqual(result["coverage_scope"], "partial_prefix")
        self.assertEqual((result["completed_questions"], result["requested_questions"]), (2, 4))
        self.assertEqual(result["forced_choice_accuracy"], 0.5)
        self.assertEqual(result["by_class"]["white"], {"requested": 1, "completed": 0, "correct": 0, "recall": None})
        self.assertIsNone(result["macro_recall"])  # an uncovered class is not counted as zero
        self.assertEqual(result["by_game"]["g2"], {"requested": 2, "completed": 0, "correct": 0, "accuracy": None})
        self.assertEqual(result["occupancy"]["requested"], 4)


class SummaryRejectionTests(unittest.TestCase):
    def setUp(self):
        self.questions = [question(0, 0), question(1, 1), question(2, 2)]
        self.rows = [row(q, scores_for(q["label"])) for q in self.questions]

    def assert_rejected(self, rows, questions=None):
        with self.assertRaises(ValueError):
            tm.summarize_transfer(rows, self.questions if questions is None else questions)

    def test_order_identity_and_count_violations(self):
        self.assert_rejected([self.rows[1], self.rows[0], self.rows[2]])           # reordered
        self.assert_rejected([self.rows[0], self.rows[0]])                         # duplicate
        self.assert_rejected(self.rows + [row(question(9, 0), scores_for(0))])     # extra
        self.assert_rejected([row(question(7, 0), scores_for(0))])                 # foreign ID
        self.assert_rejected([self.rows[1]])                                       # not a prefix
        self.assert_rejected(self.rows, [self.questions[0], self.questions[0]])    # duplicate expected
        self.assert_rejected([], [])                                               # empty expected list

    def test_metadata_and_essential_field_violations(self):
        for field, value in (("game_id", "other"), ("to_play", "W"), ("point_index", 5),
                             ("position_id", "other"), ("ply", 25), ("source_task_ids", ["x"])):
            bad = copy.deepcopy(self.rows)
            bad[0][field] = value
            with self.subTest(field=field):
                self.assert_rejected(bad)
        relabelled = copy.deepcopy(self.rows)
        relabelled[1] = row({**self.questions[1], "label": 2}, scores_for(1))
        self.assert_rejected(relabelled)
        for field in ("question_id", "label", "game_id", "to_play"):
            bad = copy.deepcopy(self.rows)
            del bad[0][field]
            with self.subTest(missing=field):
                self.assert_rejected(bad)
        bad = copy.deepcopy(self.rows)
        bad[0]["label"] = False
        self.assert_rejected(bad)

    def test_score_validity_violations(self):
        cases = {
            "nan": [float("nan"), -1.0, -2.0], "inf": [-float("inf"), -1.0, -2.0],
            "positive": [0.5, -1.0, -2.0], "two_scores": [-1.0, -2.0], "boolean": [True, -1.0, -2.0],
        }
        for name, values in cases.items():
            bad = copy.deepcopy(self.rows)
            bad[0]["label_log_probs"] = values
            with self.subTest(case=name):
                self.assert_rejected(bad)

    def test_stored_full_vocabulary_mapping_must_be_consistent(self):
        bad = copy.deepcopy(self.rows)
        bad[0]["unconstrained_label"] = None          # label token stored as off-label
        self.assert_rejected(bad)
        bad = copy.deepcopy(self.rows)
        bad[0]["unconstrained_token_id"] = TOKENS[1]  # label token whose score is not the maximum
        bad[0]["unconstrained_label"] = 1
        self.assert_rejected(bad)
        bad = copy.deepcopy(self.rows)
        bad[0]["label_token_ids"] = [1, 1, 2]
        self.assert_rejected(bad)


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.questions = [question(i, i % 3, "g1" if i < 3 else "g2") for i in range(6)]
        self.saved = [row(q, scores_for((q["label"] + 1) % 3)) for q in self.questions]
        for saved in self.saved:
            self.assertNotIn("prompt", saved)

    def test_shuffled_saved_rows_return_in_question_order_as_copies(self):
        shuffled = [self.saved[i] for i in (4, 0, 5, 2, 1, 3)]
        before = copy.deepcopy(shuffled)
        aligned = tm.align_saved_reference(self.questions, shuffled)
        self.assertEqual([r["question_id"] for r in aligned], [q["question_id"] for q in self.questions])
        self.assertEqual(aligned, self.saved)
        aligned[0]["label_log_probs"][0] = -99.0
        self.assertEqual(shuffled, before)
        summary = tm.summarize_transfer(tm.align_saved_reference(self.questions, shuffled), self.questions)
        self.assertEqual(summary["forced_choice_correct"], 0)

    def test_missing_extra_duplicate_and_mismatched_rows_rejected(self):
        cases = {"missing": self.saved[:-1],
                 "extra": self.saved + [row(question(9, 0), scores_for(0))],
                 "duplicate": self.saved[:-1] + [self.saved[0]]}
        mismatched = copy.deepcopy(self.saved)
        mismatched[3]["to_play"] = "W"
        cases["metadata"] = mismatched
        for name, saved in cases.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                tm.align_saved_reference(self.questions, saved)


class RestorationTests(unittest.TestCase):
    def setUp(self):
        self.questions = [question(i, i % 3) for i in range(4)]
        self.saved = [row(q, [-1.0, -2.0, -3.0]) for q in self.questions]
        self.current = copy.deepcopy(self.saved)

    def compare(self, current=None, saved=None, **kwargs):
        result = tm.compare_restoration(self.current if current is None else current,
                                        self.saved if saved is None else saved, **kwargs)
        json_safe(self, result)
        return result

    def test_identical_rows_pass(self):
        result = self.compare()
        self.assertTrue(result["passed"])
        self.assertTrue(result["coverage_complete"] and result["predictions_match"])
        self.assertEqual(result["max_abs_logp_difference"], 0.0)
        self.assertEqual(result["tolerance"], 1e-5)
        self.assertEqual(len(result["differences"]), 4)

    def test_tolerance_is_inclusive_and_never_relaxed(self):
        self.current[1]["label_log_probs"] = [-1.0 + 2 ** -17, -2.0, -3.0]  # 7.6e-6, exactly representable
        self.assertTrue(self.compare()["passed"])
        self.current[1]["label_log_probs"] = [-1.0 + 2 ** -16, -2.0, -3.0]  # 1.5e-5
        result = self.compare()
        self.assertFalse(result["passed"])
        self.assertEqual(result["max_abs_logp_difference"], 2 ** -16)
        self.assertTrue(result["predictions_match"])
        self.current[1]["label_log_probs"] = [-0.75, -2.0, -3.0]
        self.assertTrue(self.compare(tolerance=0.25)["passed"])                # difference exactly at tolerance
        self.current[1]["label_log_probs"] = [-0.5, -2.0, -3.0]
        self.assertFalse(self.compare(tolerance=0.25)["passed"])

    def test_partial_extra_reordered_or_duplicate_rows_fail(self):
        cases = {"prefix": self.current[:3], "extra": self.current + [row(question(9, 0), [-1.0, -2.0, -3.0])],
                 "reordered": [self.current[1], self.current[0]] + self.current[2:],
                 "duplicate": self.current[:3] + [self.current[0]]}
        for name, current in cases.items():
            with self.subTest(case=name):
                result = self.compare(current=current)
                self.assertFalse(result["passed"])
        self.assertFalse(self.compare(current=self.current[:3])["coverage_complete"])

    def test_changed_prefix_tokens_or_metadata_fail(self):
        self.current[2]["prefix_tokens"] = 75
        self.assertFalse(self.compare()["passed"])
        self.current = copy.deepcopy(self.saved)
        tokens = [3194, 11453, 777]
        self.current[2]["label_token_ids"] = tokens
        self.assertFalse(self.compare()["passed"])
        self.current = copy.deepcopy(self.saved)
        self.current[2]["game_id"] = "other"
        self.assertFalse(self.compare()["passed"])

    def test_changed_forced_or_full_vocabulary_choice_fails(self):
        self.current[0] = row(self.questions[0], [-2.0, -1.0, -3.0])
        result = self.compare()
        self.assertFalse(result["passed"] or result["predictions_match"])
        self.current = copy.deepcopy(self.saved)
        self.current[0] = row(self.questions[0], [-1.0, -2.0, -3.0], token=OFF_LABEL_TOKEN)
        result = self.compare()
        self.assertFalse(result["passed"] or result["predictions_match"])
        self.assertEqual(result["max_abs_logp_difference"], 0.0)

    def test_nonfinite_scores_fail_without_nonfinite_output(self):
        self.current[3]["label_log_probs"] = [float("nan"), -2.0, -3.0]
        result = self.compare()
        self.assertFalse(result["passed"])
        self.assertIsNone(result["differences"][3]["max_abs_logp_difference"])
        self.assertEqual(result["max_abs_logp_difference"], 0.0)  # remaining finite comparisons

    def test_malformed_inputs_and_tolerances_raise(self):
        for tolerance in (-1e-5, float("nan"), float("inf"), True, "1e-5"):
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                tm.compare_restoration(self.current, self.saved, tolerance=tolerance)
        bad = copy.deepcopy(self.current)
        del bad[0]["prefix_tokens"]
        with self.assertRaises(ValueError):
            tm.compare_restoration(bad, self.saved)
        bad = copy.deepcopy(self.current)
        bad[0]["forced_choice_label"] = 2  # contradicts its own scores
        with self.assertRaises(ValueError):
            tm.compare_restoration(bad, self.saved)
        with self.assertRaises(ValueError):
            tm.compare_restoration([], [])


if __name__ == "__main__":
    unittest.main()
