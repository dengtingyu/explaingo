"""Unit tests for epoch_schedule. Authored in a source-only turn; not yet run.

Run from candidate/:  python -m unittest test_epoch_schedule
"""
import unittest

from goqueen import epoch_schedule as es

IDS10 = [f"q{i:02d}" for i in range(10)]
IDS64 = [f"game-{i // 8:02d}:p{i % 8}" for i in range(64)]


def flatten(batches):
    return [example for batch in batches for example in batch]


class ValidateIdsTest(unittest.TestCase):
    def test_canonical_order_is_independent_of_input_order(self):
        self.assertEqual(es.validate_ids(reversed(IDS10)), tuple(IDS10))

    def test_rejects_empty_nonstring_and_duplicates(self):
        for bad in ([], ["a", ""], ["a", 3], ["a", None]):
            with self.assertRaises(ValueError):
                es.validate_ids(bad)
        with self.assertRaisesRegex(ValueError, "2 duplicate"):
            es.validate_ids(["a", "b", "a", "c", "b"])


class PlanTest(unittest.TestCase):
    def test_counts_with_partial_final_step(self):
        plan = es.make_plan(10, 4, 5)
        self.assertEqual((plan.steps_per_epoch, plan.final_step_size, plan.total_steps,
                          plan.example_passes), (3, 2, 15, 50))

    def test_counts_when_divisible_or_single_step(self):
        self.assertEqual(es.make_plan(8, 4, 5).final_step_size, 4)
        self.assertEqual(es.make_plan(8, 4, 5).total_steps, 10)
        single = es.make_plan(3, 4, 5)
        self.assertEqual((single.steps_per_epoch, single.final_step_size, single.total_steps), (1, 3, 5))

    def test_rejects_invalid_counts(self):
        for args in ((0, 4, 5), (10, 0, 5), (10, 4, 0), (10.0, 4, 5), (True, 4, 5)):
            with self.assertRaises(ValueError):
                es.make_plan(*args)


class OrderTest(unittest.TestCase):
    def test_each_epoch_is_a_permutation_independent_of_input_order(self):
        for epoch in range(5):
            order = es.epoch_order(IDS64, 42, epoch)
            self.assertEqual(sorted(order), sorted(IDS64))
            self.assertEqual(order, es.epoch_order(list(reversed(IDS64)), 42, epoch))

    def test_reproducible_and_varies_with_epoch_seed_and_seed_type(self):
        first = es.epoch_order(IDS64, 42, 0)
        self.assertEqual(first, es.epoch_order(IDS64, 42, 0))
        self.assertNotEqual(first, es.epoch_order(IDS64, 42, 1))
        self.assertNotEqual(first, es.epoch_order(IDS64, 43, 0))
        self.assertNotEqual(first, es.epoch_order(IDS64, "42", 0))

    def test_rejects_bad_seed_or_epoch(self):
        with self.assertRaises(TypeError):
            es.epoch_order(IDS10, True, 0)
        with self.assertRaises(TypeError):
            es.epoch_order(IDS10, 4.2, 0)
        with self.assertRaises(ValueError):
            es.epoch_order(IDS10, 42, -1)


class StepsTest(unittest.TestCase):
    def test_epoch_steps_keep_final_partial_and_follow_order(self):
        batches = es.epoch_steps(IDS10, 7, 3, 4)
        self.assertEqual([len(batch) for batch in batches], [4, 4, 2])
        self.assertEqual(flatten(batches), es.epoch_order(IDS10, 7, 3))

    def test_full_traversal_uses_every_id_once_per_epoch(self):
        all_steps = list(es.steps(IDS10, 7, 4, 5))
        self.assertEqual(len(all_steps), es.make_plan(10, 4, 5).total_steps)
        self.assertEqual([s.global_step for s in all_steps], list(range(15)))
        self.assertEqual([s.global_step for s in all_steps if s.final_in_epoch], [2, 5, 8, 11, 14])
        for epoch in range(5):
            used = flatten(s.example_ids for s in all_steps if s.epoch == epoch)
            self.assertEqual(sorted(used), sorted(IDS10))
            self.assertEqual(used, es.epoch_order(IDS10, 7, epoch))

    def test_resume_yields_exactly_the_remainder(self):
        full = list(es.steps(IDS10, 7, 4, 5))
        for start in (0, 1, 2, 3, 7, 14, 15):
            self.assertEqual(list(es.steps(IDS10, 7, 4, 5, start_step=start)), full[start:])

    def test_invalid_start_is_rejected_before_iteration(self):
        with self.assertRaises(ValueError):
            es.steps(IDS10, 7, 4, 5, start_step=16)
        with self.assertRaises(ValueError):
            es.steps(IDS10 + ["q00"], 7, 4, 5)


class PositionTest(unittest.TestCase):
    def test_locate_boundaries(self):
        plan = es.make_plan(10, 4, 5)
        self.assertEqual(es.locate(plan, 0), es.Position(0, 0, 0, False))
        self.assertEqual(es.locate(plan, 2), es.Position(2, 0, 2, False))
        self.assertEqual(es.locate(plan, 3), es.Position(3, 1, 0, False))
        self.assertEqual(es.locate(plan, 14), es.Position(14, 4, 2, False))
        self.assertEqual(es.locate(plan, 15), es.Position(15, 5, 0, True))
        with self.assertRaises(ValueError):
            es.locate(plan, 16)


class FingerprintAndAuditTest(unittest.TestCase):
    def test_fingerprint_sensitivity_and_input_order_independence(self):
        base = es.fingerprint(IDS10, 7, 4, 5)
        self.assertEqual(base, es.fingerprint(list(reversed(IDS10)), 7, 4, 5))
        for other in (es.fingerprint(IDS10, 8, 4, 5), es.fingerprint(IDS10, 7, 3, 5),
                      es.fingerprint(IDS10, 7, 4, 4), es.fingerprint(IDS10[:-1], 7, 4, 5)):
            self.assertNotEqual(base, other)

    def test_audit_reports_progress_and_completion(self):
        executed = [s.example_ids for s in es.steps(IDS10, 7, 4, 5)]
        partial = es.audit(executed[:4], IDS10, 7, 4, 5)
        self.assertEqual((partial["steps_completed"], partial["epochs_completed"],
                          partial["next_epoch"], partial["next_step_in_epoch"], partial["complete"]),
                         (4, 1, 1, 1, False))
        done = es.audit(executed, IDS10, 7, 4, 5)
        self.assertTrue(done["complete"])
        self.assertEqual(done["epochs_completed"], 5)

    def test_audit_detects_divergence_and_overrun(self):
        executed = [s.example_ids for s in es.steps(IDS10, 7, 4, 5)]
        swapped = executed[:]
        swapped[3], swapped[4] = swapped[4], swapped[3]
        with self.assertRaisesRegex(ValueError, "step 3"):
            es.audit(swapped, IDS10, 7, 4, 5)
        with self.assertRaises(ValueError):
            es.audit(executed + [executed[0]], IDS10, 7, 4, 5)
        repeated_epoch = executed[:3] + executed[:3]
        with self.assertRaises(ValueError):
            es.audit(repeated_epoch, IDS10, 7, 4, 5)


if __name__ == "__main__":
    unittest.main()
