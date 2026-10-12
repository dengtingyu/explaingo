"""External APS transfer outcomes with synthetic records and a mocked child only."""
from __future__ import annotations

import copy
from contextlib import ExitStack
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_assisted_point_transfer as launch


def completed_report():
    validation = [{"question_id": f"v{i}"} for i in range(54)]
    restoration = [{"question_id": f"t{i}"} for i in range(21)]
    return {
        "stage": "aps_original54_transfer", "status": "complete", "valid_complete": True,
        "budget_exceeded": False, "validation_questions": validation,
        "validation_rows": [{**q, "label_log_probs": [-0.2, -2.2, -3.2]} for q in validation],
        "restoration_questions": restoration,
        "restoration_rows": [{**q, "label_log_probs": [-0.1, -2.4, -4.1]} for q in restoration],
        "base_sha256_before": "a" * 64, "base_sha256_after": "a" * 64,
        "adapter_sha256_before": "b" * 64, "adapter_sha256_after": "b" * 64,
        "source_hashes_unchanged": True, "runtime_binding_cleared": True,
        "all_parameters_frozen_and_gradients_absent": True,
        "optimizer_constructed": False, "optimizer_updates": 0, "backward_calls": 0,
        "restoration": {"passed": True, "coverage_complete": True, "predictions_match": True,
                        "max_abs_logp_difference": 0.0, "tolerance": 1e-5},
        "summary": {"coverage_complete": True},
    }


class FullOutcomeTests(unittest.TestCase):
    def accepts(self, report=None, *, exit_code=0, elapsed=80.0, cap=300):
        return launch.effective_complete(completed_report() if report is None else report,
                                         exit_code, elapsed, cap)

    def test_complete_counts_and_runtime_contracts_accept_without_a_performance_threshold(self):
        report = completed_report()
        report["summary"]["correct"] = 0  # Descriptive success is not an accuracy gate.
        self.assertTrue(self.accepts(report))
        self.assertTrue(self.accepts(report, elapsed=300))

    def test_child_nonzero_signal_missing_and_bool_exit_never_accept(self):
        for code in (1, -15, -9, None, False):
            with self.subTest(code=code):
                self.assertFalse(self.accepts(exit_code=code))

    def test_external_wall_and_budget_must_be_finite_within_declared_bounds(self):
        for elapsed in (-1, 300.000001, math.nan, math.inf, False, "80"):
            with self.subTest(elapsed=elapsed):
                self.assertFalse(self.accepts(elapsed=elapsed))
        for cap in (0, 15, 301, math.inf, math.nan, True, "300", None):
            with self.subTest(cap=cap):
                self.assertFalse(self.accepts(cap=cap))

    def test_status_and_scope_flags_override_full_counts(self):
        for key, value in (("stage", "a0_assisted_point"), ("status", "stopped"),
                           ("valid_complete", False), ("budget_exceeded", True),
                           ("source_hashes_unchanged", False), ("runtime_binding_cleared", False),
                           ("all_parameters_frozen_and_gradients_absent", False),
                           ("optimizer_constructed", True), ("optimizer_updates", 1),
                           ("backward_calls", 1)):
            report = completed_report()
            report[key] = value
            with self.subTest(key=key):
                self.assertFalse(self.accepts(report))

    def test_exact_54_and_21_unique_ordered_id_coverage_is_required(self):
        for group in ("validation", "restoration"):
            for mutation in ("missing", "extra", "duplicate", "foreign", "reordered"):
                report = completed_report()
                rows = report[group + "_rows"]
                if mutation == "missing":
                    rows.pop()
                elif mutation == "extra":
                    rows.append(copy.deepcopy(rows[0]))
                elif mutation == "duplicate":
                    rows[-1]["question_id"] = rows[0]["question_id"]
                elif mutation == "foreign":
                    rows[-1]["question_id"] = "foreign"
                else:
                    rows.reverse()
                with self.subTest(group=group, mutation=mutation):
                    self.assertFalse(self.accepts(report))
            report = completed_report()
            report[group + "_questions"][-1]["question_id"] = report[group + "_questions"][0]["question_id"]
            self.assertFalse(self.accepts(report))

    def test_each_score_row_must_have_three_finite_numeric_log_probabilities(self):
        for group in ("validation", "restoration"):
            for values in ([], [-1, -2], [-1, -2, -3, -4], [math.nan, -1, -2],
                           [math.inf, -1, -2], [False, -1, -2], ["0", -1, -2]):
                report = completed_report()
                report[group + "_rows"][0]["label_log_probs"] = values
                with self.subTest(group=group, values=values):
                    self.assertFalse(self.accepts(report))

    def test_base_and_adapter_digests_must_be_measured_valid_and_unchanged(self):
        for group in ("base", "adapter"):
            for digest in (None, "", "a" * 63, "z" * 64):
                report = completed_report()
                report[group + "_sha256_before"] = report[group + "_sha256_after"] = digest
                with self.subTest(group=group, digest=digest):
                    self.assertFalse(self.accepts(report))
            report = completed_report()
            report[group + "_sha256_after"] = "c" * 64
            self.assertFalse(self.accepts(report))

    def test_restoration_mismatch_stops_even_with_54_complete_validation_rows(self):
        for key, value in (("passed", False), ("coverage_complete", False),
                           ("predictions_match", False), ("max_abs_logp_difference", 2e-5),
                           ("max_abs_logp_difference", math.nan), ("tolerance", 1e-4)):
            report = completed_report()
            report["restoration"][key] = value
            with self.subTest(key=key, value=value):
                self.assertFalse(self.accepts(report))

    def test_malformed_or_incomplete_reports_do_not_receive_acceptance(self):
        for value in (None, [], "complete", {}):
            self.assertFalse(launch.effective_complete(value, 0, 80, 300))
        for key, value in (("summary", None), ("summary", []), ("summary", {}),
                           ("restoration", None), ("restoration", []),
                           ("validation_rows", None), ("restoration_questions", "bad")):
            report = completed_report()
            report[key] = value
            with self.subTest(key=key, value=value):
                self.assertFalse(self.accepts(report))


class ProcessLifecycleTests(unittest.TestCase):
    """Real temporary records, fake clock and mocked Popen; never starts a process."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="aps-transfer-launch-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        (self.root / "scripts").mkdir()
        (self.root / "outputs").mkdir()
        self.runner = self.root / "scripts/eval_assisted_point_transfer.py"
        self.runner.write_text("# Synthetic runner source; never executed.\n")
        self.launcher = self.root / "scripts/run_assisted_point_transfer.py"
        self.launcher.write_text("# Synthetic launcher source; never executed.\n")
        self.output = self.root / "outputs/assisted-point-transfer"
        self.child = mock.Mock(pid=1234)
        self.child.wait.return_value = 0

    def run_fake(self, *, report=None, raw=None, missing=False, mutate=None, spawn_error=None,
                 clock=None, seconds=300):
        report = completed_report() if report is None else report

        def spawn(command, **kwargs):
            self.assertEqual(command, [sys.executable, "-B", str(self.runner), "--seconds", str(seconds),
                                       "--origin-start", "100.0"])
            self.assertEqual(kwargs["cwd"], self.root)
            self.assertEqual(Path(kwargs["stdout"].name), self.output / "process-stdout.log")
            self.assertEqual(Path(kwargs["stderr"].name), self.output / "process-stderr.log")
            start = json.loads((self.output / "process-start.json").read_text())
            self.assertEqual(start["origin_monotonic"], 100.0)
            self.assertEqual(start["seconds"], seconds)
            self.assertFalse(start["automatic_retry"])
            self.assertEqual((self.output / "launcher-at-run.py").read_bytes(), self.launcher.read_bytes())
            if spawn_error:
                raise spawn_error
            if not missing:
                (self.output / "report.json").write_text(raw if raw is not None else json.dumps(report))
            if mutate:
                mutate()
            return self.child

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(launch, "ROOT", self.root))
            stack.enter_context(mock.patch.object(launch, "OUTPUT", self.output))
            stack.enter_context(mock.patch.object(launch, "__file__", str(self.launcher)))
            stack.enter_context(mock.patch.object(launch, "PROGRAM_STARTED", 100.0))
            stack.enter_context(mock.patch.object(launch, "relative", side_effect=lambda p: Path(p).resolve().relative_to(self.root).as_posix()))
            stack.enter_context(mock.patch.object(launch.time, "perf_counter", side_effect=clock)
                                if clock is not None else mock.patch.object(launch.time, "perf_counter", return_value=180.0))
            popen = stack.enter_context(mock.patch.object(launch.subprocess, "Popen", side_effect=spawn))
            stdout = stack.enter_context(mock.patch("sys.stdout", new_callable=io.StringIO))
            code = launch.main(["--seconds", str(seconds)])
        result = json.loads((self.output / "process-completion.json").read_text())
        return code, result, popen, [json.loads(line) for line in stdout.getvalue().splitlines()]

    def test_one_successful_mock_child_records_shared_origin_exit_and_source(self):
        code, result, popen, printed = self.run_fake()
        self.assertEqual(code, 0)
        self.assertTrue(result["valid_complete"])
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["seconds_to_child_exit"], 80.0)
        self.assertTrue(result["source_unchanged"])
        self.assertFalse(result["terminated_after_grace"])
        popen.assert_called_once()
        self.child.wait.assert_called_once_with(timeout=250.0)
        self.child.terminate.assert_not_called()
        self.child.kill.assert_not_called()
        self.assertEqual(printed[0]["child_exit_code"], 0)
        self.assertEqual(printed[-1]["seconds_through_completion_record"], 80.0)

    def test_existing_directory_blocks_a_second_attempt_before_source_reads_or_child(self):
        self.output.mkdir()
        retained = self.output / "retained"
        retained.write_bytes(b"original")
        with mock.patch.object(launch, "OUTPUT", self.output), \
                mock.patch.object(launch, "file_sha256") as hashing, \
                mock.patch.object(launch.subprocess, "Popen") as popen:
            with self.assertRaises(FileExistsError):
                launch.main([])
        hashing.assert_not_called()
        popen.assert_not_called()
        self.assertEqual(retained.read_bytes(), b"original")
        self.assertEqual(list(self.output.iterdir()), [retained])

    def test_invalid_budget_stops_before_directory_creation(self):
        for seconds in (0, 15, 301):
            with mock.patch.object(launch, "OUTPUT", self.output), \
                    mock.patch.object(launch.subprocess, "Popen") as popen:
                with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                    launch.main(["--seconds", str(seconds)])
            popen.assert_not_called()
            self.assertFalse(self.output.exists())

    def test_successful_exit_with_partial_report_is_not_complete(self):
        report = completed_report()
        report["validation_rows"].pop()
        code, result, popen, _ = self.run_fake(report=report)
        self.assertEqual(code, 0)  # Process success remains distinct from measurement completeness.
        self.assertFalse(result["valid_complete"])
        popen.assert_called_once()

    def test_restore_stop_retains_zero_validation_coverage_without_retry(self):
        report = completed_report()
        report.update(status="stopped", valid_complete=False, validation_rows=[])
        report["restoration"]["passed"] = False
        code, result, popen, _ = self.run_fake(report=report)
        self.assertEqual(code, 0)
        self.assertFalse(result["valid_complete"])
        self.assertFalse(result["automatic_retry"])
        popen.assert_called_once()

    def test_missing_report_keeps_real_child_outcome_and_fails(self):
        code, result, popen, printed = self.run_fake(missing=True)
        self.assertEqual(code, 1)
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["report_parse_error"], "FileNotFoundError")
        self.assertFalse(result["valid_complete"])
        self.assertEqual(printed[0]["child_exit_code"], 0)
        popen.assert_called_once()

    def test_truncated_json_cannot_promote_an_exit_zero_to_valid_measurement(self):
        code, result, _, _ = self.run_fake(raw='{"status": "complete"')
        self.assertEqual(code, 1)
        self.assertEqual(result["report_parse_error"], "JSONDecodeError")
        self.assertFalse(result["valid_complete"])

    def test_launch_error_is_recorded_without_retry(self):
        code, result, popen, _ = self.run_fake(spawn_error=OSError("synthetic spawn error"))
        self.assertEqual(code, 1)
        self.assertEqual(result["launch_error"], "OSError")
        self.assertIsNone(result["exit_code"])
        self.assertFalse(result["valid_complete"])
        self.child.wait.assert_not_called()
        popen.assert_called_once()

    def test_natural_signal_exit_is_preserved(self):
        self.child.wait.return_value = -15
        code, result, _, _ = self.run_fake()
        self.assertEqual(code, 1)
        self.assertEqual(result["terminated_by_signal"], 15)
        self.assertFalse(result["valid_complete"])
        self.assertFalse(result["terminated_after_grace"])

    def test_native_timeout_terminates_once_without_retry(self):
        self.child.wait.side_effect = [subprocess.TimeoutExpired("fake", 330), -15]
        code, result, popen, _ = self.run_fake()
        self.assertEqual(code, 1)
        self.assertTrue(result["terminated_after_grace"])
        self.assertFalse(result["valid_complete"])
        self.child.terminate.assert_called_once_with()
        self.child.kill.assert_not_called()
        self.assertEqual(self.child.wait.call_args_list, [mock.call(timeout=250.0), mock.call(timeout=5)])
        popen.assert_called_once()

    def test_unresponsive_termination_is_killed_and_reaped(self):
        self.child.wait.side_effect = [subprocess.TimeoutExpired("fake", 330),
                                       subprocess.TimeoutExpired("fake", 5), -9]
        code, result, _, _ = self.run_fake()
        self.assertEqual(code, 1)
        self.assertEqual(result["terminated_by_signal"], 9)
        self.assertTrue(result["terminated_after_grace"])
        self.child.terminate.assert_called_once_with()
        self.child.kill.assert_called_once_with()
        self.assertEqual(self.child.wait.call_args_list[-1], mock.call())

    def test_runner_source_change_invalidates_otherwise_complete_report(self):
        code, result, _, _ = self.run_fake(mutate=lambda: self.runner.write_text("# changed\n"))
        self.assertEqual(code, 0)
        self.assertFalse(result["source_unchanged"])
        self.assertFalse(result["valid_complete"])

    def test_launcher_source_change_invalidates_otherwise_complete_report(self):
        code, result, _, _ = self.run_fake(mutate=lambda: self.launcher.write_text("# changed\n"))
        self.assertEqual(code, 0)
        self.assertFalse(result["source_unchanged"])
        self.assertFalse(result["valid_complete"])

    def test_source_and_report_hash_time_belongs_to_cap(self):
        code, result, _, printed = self.run_fake(clock=[101.0, 399.0, 400.01, 400.02, 400.03])
        self.assertEqual(code, 0)
        self.assertEqual(result["seconds_to_child_exit"], 299.0)
        self.assertGreater(result["seconds_before_completion_record"], 300)
        self.assertTrue(result["budget_exceeded"])
        self.assertFalse(result["valid_complete"])
        self.assertGreater(printed[-1]["seconds_through_completion_record"], 300)

    def test_completion_write_crossing_cap_durably_revokes_prewrite_success(self):
        code, result, _, printed = self.run_fake(clock=[101.0, 399.0, 399.5, 400.01, 400.02])
        self.assertEqual(code, 0)
        self.assertEqual(result["seconds_before_completion_record"], 299.5)
        self.assertGreater(result["seconds_after_initial_completion_record"], 300)
        self.assertFalse(result["valid_complete"])
        self.assertTrue(result["budget_exceeded"])
        self.assertFalse(printed[-1]["valid_complete"])


if __name__ == "__main__":
    unittest.main()
