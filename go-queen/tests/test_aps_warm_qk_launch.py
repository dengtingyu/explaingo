"""APS-warm Q/K launcher contracts using saved JSON and a mocked child only."""
from __future__ import annotations

from contextlib import ExitStack
import copy
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
import run_aps_warm_qk as launch


def completed_report(source_sha="a" * 64, seconds=1800):
    questions = [{"question_id": f"v{i}"} for i in range(54)]
    return {"stage": "aps_warm_qk", "status": "complete", "run_completed": True,
            "updates_completed": 1000, "sample_slots": 4000,
            "source_script_sha256": source_sha, "total_seconds": 30.0, "max_seconds": seconds,
            "validation_questions": questions,
            "evaluations": [{"name": name, "coverage_complete": True,
                             "rows": copy.deepcopy(questions)} for name in ("initial", "final")],
            "endpoint_conditions": {"frozen": True, "restore": True, "finite": True, "saved": True}}


class EndpointTests(unittest.TestCase):
    def accepts(self, report=None, **changes):
        options = dict(exit_code=0, wall_seconds=80.0, cap_seconds=1800, source_sha256="a" * 64)
        options.update(changes)
        return launch.effective_complete(completed_report() if report is None else report, **options)

    def test_complete_protocol_accepts_without_an_accuracy_gate(self):
        report = completed_report()
        report["evaluations"][1]["correct"] = 0
        self.assertTrue(self.accepts(report))
        self.assertTrue(self.accepts(report, wall_seconds=1800))

    def test_external_exit_source_and_total_budget_are_required(self):
        for option, values in {
            "exit_code": (1, -15, None, False),
            "wall_seconds": (-1, 1800.001, math.nan, math.inf, False, "80"),
            "cap_seconds": (0, 1801, 1800.0, True, "1800"),
            "source_sha256": ("b" * 64, "z" * 64, "", None),
        }.items():
            for value in values:
                with self.subTest(option=option, value=value):
                    self.assertFalse(self.accepts(**{option: value}))

    def test_incomplete_or_nonfinite_child_contract_never_accepts(self):
        for key, values in {
            "stage": ("a0_assisted_point",), "status": ("stopped", "error"),
            "run_completed": (False, 1), "updates_completed": (999, 1001, 1000.0),
            "sample_slots": (3996, 4004, 4000.0), "source_script_sha256": ("b" * 64, None),
            "total_seconds": (-1, 81, math.nan, math.inf, True),
            "max_seconds": (1799, 1801, math.nan, math.inf, True),
            "endpoint_conditions": ({}, None, [], {"finite": False}, {"saved": 1}),
        }.items():
            for value in values:
                report = completed_report()
                report[key] = value
                with self.subTest(key=key, value=value):
                    self.assertFalse(self.accepts(report))

    def test_two_exact_ordered_unique_54_question_evaluations_are_required(self):
        for index in (0, 1):
            for change in ("missing", "extra", "duplicate", "foreign", "reordered", "coverage"):
                report = completed_report()
                evaluation = report["evaluations"][index]
                rows = evaluation["rows"]
                if change == "missing":
                    rows.pop()
                elif change == "extra":
                    rows.append(copy.deepcopy(rows[0]))
                elif change == "duplicate":
                    rows[-1] = copy.deepcopy(rows[0])
                elif change == "foreign":
                    rows[-1]["question_id"] = "other"
                elif change == "reordered":
                    rows.reverse()
                else:
                    evaluation["coverage_complete"] = False
                with self.subTest(index=index, change=change):
                    self.assertFalse(self.accepts(report))
        report = completed_report()
        report["validation_questions"][-1] = copy.deepcopy(report["validation_questions"][0])
        self.assertFalse(self.accepts(report))

    def test_malformed_reports_and_duplicate_evaluation_names_fail_closed(self):
        for report in (None, [], "complete", {}):
            self.assertFalse(launch.effective_complete(report, exit_code=0, wall_seconds=80,
                                                       cap_seconds=1800, source_sha256="a" * 64))
        for key, value in (("validation_questions", [None] * 54),
                           ("validation_questions", [{"question_id": []}] * 54),
                           ("evaluations", [None, None]), ("evaluations", [])):
            report = completed_report()
            report[key] = value
            self.assertFalse(self.accepts(report))
        for change in ("duplicate", "reverse", "extra", "missing_rows"):
            report = completed_report()
            if change == "duplicate":
                report["evaluations"][1]["name"] = "initial"
            elif change == "reverse":
                report["evaluations"].reverse()
            elif change == "extra":
                report["evaluations"].append(copy.deepcopy(report["evaluations"][0]))
            else:
                report["evaluations"][0]["rows"] = None
            with self.subTest(change=change):
                self.assertFalse(self.accepts(report))


class LifecycleTests(unittest.TestCase):
    """Actual temporary files, synthetic report and mocked Popen; no child executes."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="aps-warm-qk-launch-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        (self.root / "scripts").mkdir()
        (self.root / "outputs").mkdir()
        self.runner = self.root / "scripts/train_aps_warm_qk.py"
        self.runner.write_text("# Synthetic runner, never executed.\n")
        self.launcher = self.root / "scripts/run_aps_warm_qk.py"
        self.launcher.write_text("# Synthetic launcher, never executed.\n")
        self.output = self.root / "outputs/aps-warm-qk/report.json"
        self.child = mock.Mock(pid=1234)
        self.child.wait.return_value = 0

    def run_fake(self, *, report=None, raw=None, missing=False, mutate=None, spawn_error=None, clock=None):
        report = completed_report(launch.file_sha256(self.runner)) if report is None else report

        def spawn(command, **kwargs):
            self.assertEqual(command, [sys.executable, "-B", str(self.runner), "--seconds", "1800",
                                      "--output", str(self.output), "--outer-start", "100.0"])
            self.assertEqual(kwargs["cwd"], self.root)
            self.assertEqual(Path(kwargs["stdout"].name), self.output.parent / "process-stdout.log")
            self.assertEqual(Path(kwargs["stderr"].name), self.output.parent / "process-stderr.log")
            start = json.loads((self.output.parent / "process-start.json").read_text())
            self.assertEqual(start["outer_start_monotonic"], 100.0)
            self.assertEqual(start["total_soft_cap_seconds"], 1800)
            self.assertEqual(start["stage"], "aps_warm_qk")
            self.assertIsInstance(start["launcher_pid"], int)
            self.assertFalse(start["automatic_retry"])
            self.assertEqual((self.output.parent / "launcher-at-run.py").read_bytes(), self.launcher.read_bytes())
            if spawn_error:
                raise spawn_error
            if not missing:
                self.output.write_text(raw if raw is not None else json.dumps(report))
            if mutate:
                mutate()
            return self.child

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(launch, "ROOT", self.root))
            stack.enter_context(mock.patch.object(launch, "__file__", str(self.launcher)))
            stack.enter_context(mock.patch.object(launch, "PROGRAM_STARTED", 100.0))
            stack.enter_context(mock.patch.object(launch.time, "perf_counter", side_effect=clock)
                                if clock is not None else mock.patch.object(launch.time, "perf_counter", return_value=180.0))
            popen = stack.enter_context(mock.patch.object(launch.subprocess, "Popen", side_effect=spawn))
            stdout = stack.enter_context(mock.patch("sys.stdout", new_callable=io.StringIO))
            code = launch.main(["--seconds", "1800", "--output", str(self.output)])
        result = json.loads((self.output.parent / "process-completion.json").read_text())
        return code, result, popen, [json.loads(line) for line in stdout.getvalue().splitlines()]

    def test_success_records_one_child_shared_origin_and_external_source_evidence(self):
        code, result, popen, printed = self.run_fake()
        self.assertEqual(code, 0)
        self.assertTrue(result["valid_complete"])
        self.assertTrue(result["source_unchanged"])
        self.assertEqual(result["hash_errors"], {})
        self.assertEqual(result["seconds_to_child_exit"], 80)
        self.assertEqual(printed[0]["child_exit_code"], 0)
        self.assertEqual(printed[-1]["seconds_through_completion_record"], 80)
        popen.assert_called_once()
        self.child.wait.assert_called_once_with(timeout=1750.0)
        self.child.terminate.assert_not_called()
        self.child.kill.assert_not_called()

    def test_existing_parent_preserves_unknown_history_and_prevents_any_source_read_or_child(self):
        self.output.parent.mkdir()
        retained = self.output.parent / "unrecognized-history"
        retained.write_bytes(b"retain")
        with mock.patch.object(launch, "ROOT", self.root), \
                mock.patch.object(launch, "file_sha256") as hashing, \
                mock.patch.object(launch.subprocess, "Popen") as popen:
            with self.assertRaises(FileExistsError):
                launch.main(["--output", str(self.output)])
        hashing.assert_not_called()
        popen.assert_not_called()
        self.assertEqual(retained.read_bytes(), b"retain")
        self.assertEqual(list(self.output.parent.iterdir()), [retained])

    def test_invalid_budget_path_and_reserved_output_names_stop_before_child(self):
        cases = [["--seconds", "0"], ["--seconds", "1801"],
                 ["--output", str(self.root / "outside/report.json")],
                 ["--output", str(self.root / "outputs/report.json")],
                 ["--output", str(self.output.parent / "PROCESS-START.json")],
                 ["--output", str(self.output.parent / "run-start.json")],
                 ["--output", str(self.output.parent / "not-json.txt")]]
        for args in cases:
            with mock.patch.object(launch, "ROOT", self.root), \
                    mock.patch.object(launch.subprocess, "Popen") as popen:
                with self.subTest(args=args), self.assertRaises(ValueError):
                    launch.main(args)
            popen.assert_not_called()
            self.assertFalse(self.output.parent.exists())

    def test_clean_partial_stop_does_not_become_complete_or_retry(self):
        report = completed_report(launch.file_sha256(self.runner))
        report.update(status="stopped", run_completed=False, updates_completed=900, sample_slots=3600)
        code, result, popen, _ = self.run_fake(report=report)
        self.assertEqual(code, 0)
        self.assertFalse(result["valid_complete"])
        self.assertFalse(result["automatic_retry"])
        popen.assert_called_once()

    def test_missing_report_retains_real_exit_evidence(self):
        code, result, popen, printed = self.run_fake(missing=True)
        self.assertEqual(code, 1)
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["report_parse_error"], "FileNotFoundError")
        self.assertEqual(result["hash_errors"]["report"], "FileNotFoundError")
        self.assertFalse(result["valid_complete"])
        self.assertEqual(printed[0]["child_exit_code"], 0)
        popen.assert_called_once()

    def test_truncated_report_is_not_accepted(self):
        code, result, _, _ = self.run_fake(raw='{"status":"complete"')
        self.assertEqual(code, 1)
        self.assertEqual(result["report_parse_error"], "JSONDecodeError")
        self.assertFalse(result["valid_complete"])

    def test_spawn_error_is_recorded_without_retry(self):
        code, result, popen, _ = self.run_fake(spawn_error=OSError("synthetic"))
        self.assertEqual(code, 1)
        self.assertEqual(result["launch_error"], "OSError")
        self.assertIsNone(result["exit_code"])
        self.assertFalse(result["valid_complete"])
        self.child.wait.assert_not_called()
        popen.assert_called_once()

    def test_signal_exit_cannot_accept_a_complete_report(self):
        self.child.wait.return_value = -15
        code, result, _, _ = self.run_fake()
        self.assertEqual(code, 1)
        self.assertEqual(result["terminated_by_signal"], 15)
        self.assertFalse(result["valid_complete"])
        self.assertFalse(result["terminated_after_native_return_grace"])

    def test_timeout_terminates_kills_and_reaps_without_retry(self):
        self.child.wait.side_effect = [subprocess.TimeoutExpired("fake", 1830),
                                       subprocess.TimeoutExpired("fake", 5), -9]
        code, result, popen, _ = self.run_fake()
        self.assertEqual(code, 1)
        self.assertTrue(result["terminated_after_native_return_grace"])
        self.assertEqual(result["terminated_by_signal"], 9)
        self.assertFalse(result["valid_complete"])
        self.child.terminate.assert_called_once_with()
        self.child.kill.assert_called_once_with()
        self.assertEqual(self.child.wait.call_args_list,
                         [mock.call(timeout=1750.0), mock.call(timeout=5), mock.call()])
        popen.assert_called_once()

    def test_changed_runner_cannot_receive_valid_completion(self):
        code, result, _, _ = self.run_fake(mutate=lambda: self.runner.write_text("# changed\n"))
        self.assertEqual(code, 0)
        self.assertFalse(result["source_unchanged"])
        self.assertFalse(result["valid_complete"])

    def test_removed_launcher_preserves_completion_as_invalid_instead_of_losing_child_outcome(self):
        code, result, _, _ = self.run_fake(mutate=self.launcher.unlink)
        self.assertEqual(code, 0)
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["hash_errors"]["launcher"], "FileNotFoundError")
        self.assertFalse(result["valid_complete"])

    def test_source_and_report_hash_time_is_inside_budget(self):
        code, result, _, printed = self.run_fake(clock=[101.0, 1899.0, 1900.01, 1900.02, 1900.03])
        self.assertEqual(code, 0)
        self.assertEqual(result["seconds_to_child_exit"], 1799)
        self.assertGreater(result["seconds_before_completion_record"], 1800)
        self.assertTrue(result["budget_exceeded"])
        self.assertFalse(result["valid_complete"])
        self.assertGreater(printed[-1]["seconds_through_completion_record"], 1800)

    def test_completion_write_crossing_budget_durably_revokes_success(self):
        code, result, _, printed = self.run_fake(clock=[101.0, 1899.0, 1899.5, 1900.01, 1900.02])
        self.assertEqual(code, 0)
        self.assertEqual(result["seconds_before_completion_record"], 1799.5)
        self.assertGreater(result["seconds_after_initial_completion_record"], 1800)
        self.assertTrue(result["budget_exceeded"])
        self.assertFalse(result["valid_complete"])
        self.assertFalse(printed[-1]["valid_complete"])


if __name__ == "__main__":
    unittest.main()
