#!/usr/bin/env python3
"""One fixed-final warm-QK training-item evaluation with retained OS outcome.

The fixed 1,500-second total is soft and includes finalization. A bounded
native-call return grace never extends the valid-result budget. No retry,
resume, alternate output, training or performance threshold is provided.
Use only for this fixed evaluation; complete coverage and scientific accuracy are separate.
"""
from __future__ import annotations

import time
PROGRAM_STARTED = time.perf_counter()

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/aps-warm-qk-train-eval"
STAGE = "aps_warm_qk_train_eval"
SECONDS, RESERVE_SECONDS, NATIVE_GRACE_SECONDS = 1500, 60, 30
TRAIN_COUNT, RESTORE_COUNT = 2702, 6
FORWARD_COUNT, TOLERANCE = TRAIN_COUNT + RESTORE_COUNT, 1e-5


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _real(value):
    return type(value) in (int, float) and math.isfinite(value)


def _digest(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def effective_complete(report, exit_code, elapsed, cap):
    """Execution validity only; accuracy does not decide completion."""
    if (not isinstance(report, dict) or type(cap) is not int or cap != SECONDS
            or type(exit_code) is not int or exit_code != 0
            or not _real(elapsed) or not 0 <= elapsed <= cap):
        return False
    try:
        groups = []
        for prefix, count in (("training", TRAIN_COUNT), ("restoration", RESTORE_COUNT)):
            questions, rows = report[prefix + "_questions"], report[prefix + "_rows"]
            if not isinstance(questions, list) or not isinstance(rows, list):
                return False
            ids = [q["question_id"] for q in questions]
            row_ids = [row["question_id"] for row in rows]
            if (len(ids) != count or not all(isinstance(i, str) and i for i in ids)
                    or len(set(ids)) != count or ids != row_ids):
                return False
            for row in rows:
                scores = row["label_log_probs"]
                if (not isinstance(scores, list) or len(scores) != 3
                        or not all(_real(v) and v <= 0 for v in scores)
                        or not _real(row["off_label_mass"]) or not 0 <= row["off_label_mass"] <= 1):
                    return False
            groups.append(set(ids))
        if groups[0] & groups[1]:
            return False
        restore = report["restoration"]
        summary_outer = report["summary"]
        if not isinstance(restore, dict) or not isinstance(summary_outer, dict):
            return False
        summary = summary_outer.get("summary")
        if not isinstance(summary, dict):
            return False
        measured = all(_digest(report.get(key)) for key in (
            "base_sha256_before", "base_sha256_after", "adapter_sha256_before", "adapter_sha256_after"))
        return (
            report.get("stage") == STAGE and report.get("status") == "complete"
            and report.get("valid_complete") is True and report.get("budget_exceeded") is False
            and measured and report["base_sha256_before"] == report["base_sha256_after"]
            and report["adapter_sha256_before"] == report["adapter_sha256_after"]
            and all(report.get(key) is True for key in (
                "source_hashes_unchanged", "features_unchanged", "manifest_rebuilt_equal",
                "runtime_binding_cleared", "all_parameters_frozen_and_gradients_absent"))
            and report.get("optimizer_constructed") is False
            and all(type(report.get(key)) is int and report[key] == 0
                    for key in ("optimizer_updates", "backward_calls"))
            and all(type(report.get(key)) is int and report[key] == FORWARD_COUNT
                    for key in ("forward_attempts", "forward_returns"))
            and all(restore.get(key) is True for key in ("passed", "coverage_complete", "predictions_match"))
            and all(type(restore.get(key)) is int and restore[key] == RESTORE_COUNT
                    for key in ("current_questions", "saved_questions"))
            and _real(restore.get("max_abs_logp_difference"))
            and 0 <= restore["max_abs_logp_difference"] <= TOLERANCE
            and _real(restore.get("tolerance")) and restore["tolerance"] == TOLERANCE
            and summary.get("coverage_complete") is True
            and all(type(summary.get(key)) is int and summary[key] == TRAIN_COUNT
                    for key in ("completed_questions", "requested_questions")))
    except (KeyError, TypeError, ValueError):
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=SECONDS)
    args = parser.parse_args(argv)
    if args.seconds != SECONDS:
        raise ValueError("This single evaluation requires exactly 1500 seconds")
    # Creating the entire directory exclusively also rejects partial old attempts.
    OUTPUT.mkdir(parents=False, exist_ok=False)
    runner, launcher = ROOT / "scripts/eval_aps_warm_qk_train.py", Path(__file__).resolve()
    runner_sha, launcher_sha = file_sha256(runner), file_sha256(launcher)
    start = {
        "stage": STAGE, "launcher_pid": os.getpid(), "origin_monotonic": PROGRAM_STARTED,
        "seconds": args.seconds, "reserve_seconds": RESERVE_SECONDS,
        "runner": runner.relative_to(ROOT).as_posix(), "runner_sha256": runner_sha,
        "launcher": launcher.relative_to(ROOT).as_posix(), "launcher_sha256": launcher_sha,
        "native_return_grace_seconds": NATIVE_GRACE_SECONDS, "automatic_retry": False,
    }
    with (OUTPUT / "process-start.json").open("x") as stream:
        stream.write(json.dumps(start, indent=2) + "\n")
    with (OUTPUT / "launcher-at-run.py").open("xb") as stream:
        stream.write(launcher.read_bytes())
    code, pid, launch_error, terminated = None, None, None, False
    with (OUTPUT / "process-stdout.log").open("x") as out, (OUTPUT / "process-stderr.log").open("x") as err:
        try:
            child = subprocess.Popen(
                [sys.executable, "-B", str(runner), "--seconds", str(args.seconds),
                 "--origin-start", repr(PROGRAM_STARTED)], cwd=ROOT, stdout=out, stderr=err)
            pid = child.pid
            try:
                remaining = args.seconds + NATIVE_GRACE_SECONDS - (time.perf_counter() - PROGRAM_STARTED)
                code = child.wait(timeout=max(.001, remaining))
            except subprocess.TimeoutExpired:
                terminated = True
                child.terminate()
                try:
                    code = child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    code = child.wait()
        except OSError as exc:
            launch_error = type(exc).__name__
    child_wall = time.perf_counter() - PROGRAM_STARTED
    # Retain the real terminal outcome before report parsing or final file work.
    print(json.dumps({"child_exit_code": code, "child_pid": pid,
                      "seconds_to_child_exit": child_wall}), flush=True)
    report_path = OUTPUT / "report.json"
    report, parse_error = {}, None
    try:
        report = json.loads(report_path.read_text())
    except (ValueError, OSError) as exc:
        parse_error = type(exc).__name__
    digests, hash_errors = {}, {}
    for name, path in (("runner", runner), ("launcher", launcher),
                       ("launcher_copy", OUTPUT / "launcher-at-run.py"), ("report", report_path)):
        try:
            digests[name] = file_sha256(path)
        except OSError as exc:
            digests[name], hash_errors[name] = None, type(exc).__name__
    source_same = (digests["runner"] == runner_sha and digests["launcher"] == launcher_sha
                   and digests["launcher_copy"] == launcher_sha)
    report_source_same = isinstance(report, dict) and report.get("source_script_sha256") == runner_sha
    wall = time.perf_counter() - PROGRAM_STARTED
    result = {
        "stage": STAGE, "exit_code": code, "child_pid": pid,
        "terminated_by_signal": -code if type(code) is int and code < 0 else None,
        "launch_error": launch_error, "report_parse_error": parse_error,
        "seconds_to_child_exit": child_wall, "seconds_before_completion_record": wall,
        "timing_scope": "Origin precedes launcher imports; completion duration includes child exit, log closure, report parsing and source/report hashing. Printed terminal wall includes the completion write.",
        "soft_total_seconds": args.seconds, "reserve_seconds": RESERVE_SECONDS,
        "budget_exceeded": wall > args.seconds, "terminated_after_grace": terminated,
        "automatic_retry": False, "runner_sha256": runner_sha, "launcher_sha256": launcher_sha,
        "launcher_at_run_copy_sha256": digests["launcher_copy"], "source_unchanged": source_same,
        "report_source_matches": report_source_same, "report_sha256": digests["report"],
        "hash_errors": hash_errors,
        "valid_complete": (not parse_error and not launch_error and not hash_errors and not terminated
                           and source_same and report_source_same
                           and effective_complete(report, code, wall, args.seconds)),
    }
    with (OUTPUT / "process-completion.json").open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    after_write = time.perf_counter() - PROGRAM_STARTED
    if after_write > args.seconds:
        result.update(valid_complete=False, budget_exceeded=True,
                      seconds_after_initial_completion_record=after_write)
        (OUTPUT / "process-completion.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({**result, "seconds_through_completion_record": time.perf_counter() - PROGRAM_STARTED}), flush=True)
    # Process success and complete measurement are intentionally separate.
    return 0 if code == 0 and not parse_error and not launch_error and not hash_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
