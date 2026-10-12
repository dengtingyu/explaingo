"""Pure entry checks; no model imports, forward, optimizer or training run."""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_aps_warm_qk as entry


class EntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(entry.MANIFEST.read_text())
        cls.fit = json.loads(entry.FIT_REPORT.read_text())

    def test_existing_schedule_preserves_every_slot_and_validation_order(self):
        restore, val, saved, schedule, separation = entry.fixed_schedule(self.data, self.fit)
        self.assertEqual([q["question_id"] for q in schedule], self.data["train_order"])
        self.assertEqual(len(restore), 21)
        self.assertEqual([q["question_id"] for q in restore], [q["question_id"] for q in saved])
        self.assertEqual(val, [q for q in self.data["original_occupancy_questions"] if q["split"] == "val"])
        self.assertEqual(separation["new_training"], {"game_id_overlap": 0, "position_id_overlap": 0, "prompt_overlap": 52})

    def test_missing_foreign_or_duplicate_pool_identity_is_rejected(self):
        for change in ("short", "foreign", "duplicate"):
            data = copy.deepcopy(self.data)
            if change == "short":
                data["train_order"].pop()
            elif change == "foreign":
                data["train_order"][0] = "foreign-id"
            else:
                data["training_pool_questions"].append(data["training_pool_questions"][0])
            with self.subTest(change=change), self.assertRaises(ValueError):
                entry.fixed_schedule(data, self.fit)

    def test_full_schedule_is_not_silently_replaced_by_repeated_first_batch(self):
        data = copy.deepcopy(self.data)
        data["train_order"] = data["train_order"][:4] * 1000
        with self.assertRaises(ValueError):
            entry.fixed_schedule(data, self.fit)

    def test_reserve_uses_measured_cost_and_rejects_nonfinite(self):
        self.assertEqual(entry.closing_reserve(10), 300)
        self.assertEqual(entry.closing_reserve(200), 460)
        for value in (-1, math.nan, math.inf, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                entry.closing_reserve(value)

    def test_next_step_preserves_reserve_plus_slowest_observed_step(self):
        self.assertTrue(entry.can_start_step(1498, 1800, 300, 1))
        self.assertFalse(entry.can_start_step(1499, 1800, 300, 1))
        self.assertFalse(entry.can_start_step(1498, 1800, 300, 3))
        self.assertFalse(entry.can_start_step(math.nan, 1800, 300, 1))

    def test_invalid_budget_stops_before_any_parent_read_or_output_write(self):
        for seconds in (0, 1801):
            with patch.object(Path, "read_text") as read, self.assertRaises(ValueError):
                entry.main(["--seconds", str(seconds), "--outer-start", "1"])
            read.assert_not_called()

    def test_foreign_parent_and_existing_run_fail_before_heavy_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "outputs/aps-warm-qk/report.json"
            output.parent.mkdir(parents=True)
            parent = {"launcher_pid": -1, "outer_start_monotonic": 1,
                      "total_soft_cap_seconds": 1800, "runner_sha256": "0" * 64}
            (output.parent / "process-start.json").write_text(json.dumps(parent))
            with patch.object(entry, "ROOT", root), self.assertRaises(ValueError):
                entry.main(["--output", str(output), "--outer-start", "1"])
            self.assertFalse((output.parent / "run-start.json").exists())
            parent.update(launcher_pid=entry.os.getppid(), runner_sha256=entry.file_sha256(entry.__file__))
            (output.parent / "process-start.json").write_text(json.dumps(parent))
            retained = output.parent / "run-start.json"
            retained.write_text("retained")
            with patch.object(entry, "ROOT", root), self.assertRaises(ValueError):
                entry.main(["--output", str(output), "--outer-start", "1"])
            self.assertEqual(retained.read_text(), "retained")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
