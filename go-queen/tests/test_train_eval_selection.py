"""Unit tests for train_eval_selection.

Run from the project root:
PYTHONPATH=src python -m unittest discover -s tests -p test_train_eval_selection.py
"""
import copy
import hashlib
import json
import math
import unittest

from goqueen import train_eval_selection as tes
from goqueen.transfer_metrics import summarize_transfer

TOKENS = [3194, 11453, 5782]


def question(qid, position, game, to_play, point_index, label, split="train"):
    return {"question_id": qid, "position_id": position, "game_id": game, "split": split,
            "to_play": to_play, "point_index": point_index, "point": f"P{point_index}",
            "label": label, "prompt": f"{to_play}:{point_index}"}


def row(q, log_probs):
    forced = max(range(3), key=log_probs.__getitem__)
    identity = {k: q[k] for k in ("question_id", "position_id", "game_id", "split", "to_play",
                                  "point_index", "point", "label")}
    return {**identity, "prefix_tokens": 74, "label_token_ids": list(TOKENS),
            "label_log_probs": list(log_probs), "forced_choice_label": forced,
            "unconstrained_token_id": TOKENS[forced], "unconstrained_label": forced}


def digest(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def steps_for(order, batch_size=2):
    return [{"step": i + 1, "question_ids": order[i * batch_size:(i + 1) * batch_size],
             "optimizer_step_started": True, "optimizer_step_returned": True}
            for i in range(len(order) // batch_size)]


POOL = [question("t1", "g1-p1", "g1", "B", 0, 0), question("t2", "g1-p1", "g1", "B", 1, 1),
        question("t3", "g1-p1", "g1", "B", 2, 2), question("t4", "g1-p2", "g1", "W", 0, 0),
        question("t5", "g1-p2", "g1", "W", 1, 1), question("t6", "g1-p2", "g1", "W", 2, 2),
        question("t7", "g2-p1", "g2", "B", 3, 0)]          # in the pool, never sampled
BY_ID = {q["question_id"]: q for q in POOL}
# Batch size 2, four steps: t2 at steps 1 and 3, t5 at steps 2 and 4, the rest once.
ORDER = ["t1", "t2", "t3", "t5", "t4", "t2", "t6", "t5"]
COMPARISON = [question("v1", "v-p1", "v", "B", 0, 1, split="val"),
              question("v2", "v-p1", "v", "B", 9, 2, split="val"),
              question("v3", "v-p1", "v", "B", 10, 0, split="val")]
# All exp-sums are below one; forced choices: t1 0 (right), t2 0 (wrong), t3 2 (right),
# t4 1 (wrong), t5 1 (right), t6 2 (right).
SCORES = {"t1": [-0.3, -2.0, -3.0], "t2": [-0.6, -1.2, -2.5], "t3": [-2.0, -1.6, -0.5],
          "t4": [-1.3, -0.6, -2.2], "t5": [-2.2, -0.35, -2.0], "t6": [-1.0, -2.4, -0.7]}


def build(order=ORDER, pool=POOL, comparison=COMPARISON, slots=None, sha=None):
    return tes.build_eval_manifest(order if slots is None else slots, order, pool, comparison,
                                   order_sha256=digest(order) if sha is None else sha, batch_size=2,
                                   source={"report_sha256": "r" * 64})


def rows_for(manifest, limit=None):
    questions = tes.questions_from_manifest(manifest)[:limit]
    return [row(q, SCORES[q["question_id"]]) for q in questions]


def singleton(qid, field):
    return summarize_transfer([row(BY_ID[qid], SCORES[qid])], [BY_ID[qid]])[field]


def relabel_first(manifest):
    question = manifest["entries"][0]["question"]
    question["label"] = (question["label"] + 1) % 3


class ObservedSlotsTest(unittest.TestCase):
    def test_flattens_saved_steps_in_order(self):
        self.assertEqual(tes.observed_slots(steps_for(ORDER), steps=4, batch_size=2), ORDER)

    def test_rejects_incomplete_or_malformed_steps(self):
        with self.assertRaises(ValueError):
            tes.observed_slots(steps_for(ORDER), steps=5, batch_size=2)
        broken = steps_for(ORDER)
        broken[1]["step"] = 3
        with self.assertRaises(ValueError):
            tes.observed_slots(broken, steps=4, batch_size=2)
        broken = steps_for(ORDER)
        broken[2]["optimizer_step_returned"] = False
        with self.assertRaises(ValueError):
            tes.observed_slots(broken, steps=4, batch_size=2)
        broken = steps_for(ORDER)
        broken[0]["question_ids"].append("t3")
        with self.assertRaises(ValueError):
            tes.observed_slots(broken, steps=4, batch_size=2)


class ManifestTest(unittest.TestCase):
    def test_unique_ids_once_with_exposure_record_in_hash_order(self):
        manifest = build()
        ids = [e["question"]["question_id"] for e in manifest["entries"]]
        self.assertEqual(sorted(ids), ["t1", "t2", "t3", "t4", "t5", "t6"])
        self.assertEqual(ids, sorted(ids, key=lambda q: hashlib.sha256(q.encode("utf-8")).hexdigest()))
        record = {e["question"]["question_id"]: (e["exposure_count"], e["first_step"], e["last_step"])
                  for e in manifest["entries"]}
        self.assertEqual(record["t2"], (2, 1, 3))
        self.assertEqual(record["t5"], (2, 2, 4))
        self.assertEqual(record["t6"], (1, 4, 4))
        self.assertEqual(manifest["slots"], 8)
        self.assertEqual(manifest["counts"]["by_class"],
                         {"empty": {"questions": 2, "slots": 2}, "black": {"questions": 2, "slots": 4},
                          "white": {"questions": 2, "slots": 2}})
        self.assertEqual(manifest["counts"]["by_class_and_exposure"],
                         {"black:2": 2, "empty:1": 2, "white:1": 2})
        self.assertEqual((manifest["counts"]["games"], manifest["counts"]["positions"]), (1, 2))
        self.assertEqual(manifest["separation"], {"game_id_overlap": 0, "position_id_overlap": 0,
                                                  "question_id_overlap": 0, "comparison_prompts_in_list": 1})
        self.assertTrue(tes.validate_eval_manifest(manifest))

    def test_rejects_provenance_failures(self):
        swapped = ORDER[:]
        swapped[0], swapped[1] = swapped[1], swapped[0]
        with self.assertRaises(ValueError):
            build(slots=swapped)                                   # observed slots differ from the order
        with self.assertRaises(ValueError):
            build(sha="0" * 64)                                    # order digest does not match its pin
        with self.assertRaises(ValueError):
            build(order=ORDER[:-1] + ["x9"])                       # not in the training pool
        with self.assertRaises(ValueError):
            build(comparison=COMPARISON + [question("v4", "v-p9", "g1", "W", 4, 0, split="val")])
        relabelled = copy.deepcopy(POOL)
        relabelled[0]["split"] = "val"
        with self.assertRaises(ValueError):
            build(pool=relabelled)                                 # not a training question

    def test_validation_detects_tampering(self):
        manifest = build()
        for mutate in (lambda m: m["entries"][0].__setitem__("exposure_count", 3),
                       lambda m: m["entries"].reverse(),
                       relabel_first,
                       lambda m: m["source"].__setitem__("report_sha256", "s" * 64)):
            changed = copy.deepcopy(manifest)
            mutate(changed)
            with self.assertRaises(ValueError):
                tes.validate_eval_manifest(changed)


class SummaryTest(unittest.TestCase):
    def test_exposure_weighting_equals_scoring_every_slot(self):
        manifest = build()
        result = tes.exposure_weighted_summary(rows_for(manifest), manifest)
        self.assertEqual(result["confusion_matrix"], [[1, 1, 0], [2, 2, 0], [0, 0, 2]])
        self.assertEqual((result["requested_slots"], result["completed_slots"]), (8, 8))
        self.assertTrue(result["coverage_complete"])
        self.assertEqual(result["forced_choice_accuracy"], 5 / 8)
        self.assertEqual(result["by_class"]["black"], {"completed_slots": 4, "correct_slots": 2, "recall": 0.5})
        self.assertAlmostEqual(result["macro_recall"], 2 / 3, places=12)
        weights = {qid: ORDER.count(qid) for qid in SCORES}
        for field in ("three_label_ce", "full_vocabulary_ce"):
            expected = math.fsum(weights[q] * singleton(q, field) for q in SCORES) / 8
            self.assertAlmostEqual(result[field], expected, places=12)

    def test_single_exposures_reduce_to_the_unique_summary(self):
        order = ["t1", "t2", "t3", "t4", "t5", "t6"]
        manifest = build(order=order)
        rows = rows_for(manifest)
        weighted = tes.exposure_weighted_summary(rows, manifest)
        unique = summarize_transfer(rows, tes.questions_from_manifest(manifest))
        self.assertEqual(weighted["confusion_matrix"], unique["confusion_matrix"])
        self.assertAlmostEqual(weighted["three_label_ce"], unique["three_label_ce"], places=12)
        self.assertAlmostEqual(weighted["full_vocabulary_ce"], unique["full_vocabulary_ce"], places=12)

    def test_partial_prefix_keeps_the_full_slot_denominator_visible(self):
        manifest = build()
        rows = rows_for(manifest, limit=3)
        result = tes.exposure_weighted_summary(rows, manifest)
        done = sum(e["exposure_count"] for e in manifest["entries"][:3])
        self.assertEqual((result["requested_slots"], result["completed_slots"]), (8, done))
        self.assertFalse(result["coverage_complete"])
        with self.assertRaises(ValueError):
            tes.exposure_weighted_summary(rows_for(manifest)[1:], manifest)   # not an ordered prefix

    def test_describe_gives_per_class_ce_from_class_subsets(self):
        manifest = build()
        questions = tes.questions_from_manifest(manifest)
        result = tes.describe(rows_for(manifest), questions)
        self.assertEqual(result["summary"], summarize_transfer(rows_for(manifest), questions))
        black = result["per_class"]["black"]
        self.assertEqual((black["completed_questions"], black["forced_choice_correct"]), (2, 1))
        self.assertEqual(black["predicted_counts"], [1, 1, 0])
        expected = (singleton("t2", "three_label_ce") + singleton("t5", "three_label_ce")) / 2
        self.assertAlmostEqual(black["three_label_ce"], expected, places=12)
        partial = tes.describe([], questions)["per_class"]["white"]
        self.assertEqual((partial["requested_questions"], partial["completed_questions"]), (2, 0))
        self.assertIsNone(partial["three_label_ce"])


class RestorationSubsetTest(unittest.TestCase):
    def saved(self, sides=("B", "W", "B")):
        rows = []
        for n, side in enumerate(sides):
            for k, label in enumerate((1, 2, 0)):
                rows.append(row(question(f"s{n}-q{k}", f"s{n}", "v", side, k, label, split="val"),
                                [-1.0, -1.1, -1.2]))
        return rows

    def test_first_two_positions_cover_each_label_and_both_sides(self):
        subset = tes.restoration_subset(self.saved())
        self.assertEqual([r["question_id"] for r in subset], [f"s{n}-q{k}" for n in (0, 1) for k in range(3)])

    def test_rejects_single_side_or_incomplete_positions(self):
        with self.assertRaises(ValueError):
            tes.restoration_subset(self.saved(sides=("B", "B", "W")))
        rows = self.saved()
        del rows[1]
        with self.assertRaises(ValueError):
            tes.restoration_subset(rows)


if __name__ == "__main__":
    unittest.main()
