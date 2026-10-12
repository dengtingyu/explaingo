#!/usr/bin/env python3
"""Run one frozen APS original54 evaluation, retaining OS outcome and source.

The output directory is exclusive. A retained start prevents repeat/resume,
including after interruptions. The 300-second total is soft; a native call
gets a bounded return grace, with actual overruns recorded, never hidden.
"""
from __future__ import annotations

import time
PROGRAM_STARTED = time.perf_counter()

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_assisted_point_transfer import OUTPUT, TOLERANCE, restore_allows_validation
from train_grounding_microfit import file_sha256, relative, require


def effective_complete(report, exit_code, elapsed, cap):
    if (not isinstance(report, dict) or type(cap) not in (float, int)
            or not math.isfinite(cap) or not 15 < cap <= 300):
        return False
    questions, rows = report.get("validation_questions", []), report.get("validation_rows", [])
    train, restored = report.get("restoration_questions", []), report.get("restoration_rows", [])
    try:
        ids, row_ids = [q["question_id"] for q in questions], [q["question_id"] for q in rows]
        train_ids, restored_ids = [q["question_id"] for q in train], [q["question_id"] for q in restored]
        scores_finite = all(len(r["label_log_probs"]) == 3
                            and all(type(x) in (int, float) and math.isfinite(x) for x in r["label_log_probs"])
                            for r in rows + restored)
        measured = all(isinstance(report.get(k), str) and len(report[k]) == 64
                       and all(x in "0123456789abcdef" for x in report[k])
                       for k in ("base_sha256_before", "base_sha256_after", "adapter_sha256_before", "adapter_sha256_after"))
        return (type(exit_code) is int and exit_code == 0 and type(elapsed) in (float, int)
                and math.isfinite(elapsed) and 0 <= elapsed <= cap <= 300
                and report.get("stage") == "aps_original54_transfer"
                and report.get("status") == "complete" and report.get("valid_complete") is True
                and report.get("budget_exceeded") is False
                and len(ids) == len(set(ids)) == len(row_ids) == 54 and ids == row_ids
                and len(train_ids) == len(set(train_ids)) == len(restored_ids) == 21 and train_ids == restored_ids
                and scores_finite and measured
                and report["base_sha256_before"] == report["base_sha256_after"]
                and report["adapter_sha256_before"] == report["adapter_sha256_after"]
                and report.get("source_hashes_unchanged") is True
                and report.get("runtime_binding_cleared") is True
                and report.get("all_parameters_frozen_and_gradients_absent") is True
                and report.get("optimizer_constructed") is False
                and report.get("optimizer_updates") == 0 and report.get("backward_calls") == 0
                and restore_allows_validation(report.get("restoration") or {})
                and isinstance(report.get("summary"), dict)
                and report["summary"].get("coverage_complete") is True)
    except (KeyError, TypeError, ValueError):
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=300)
    args = parser.parse_args(argv)
    require(15 < args.seconds <= 300, "Budget must be 16..300 seconds")
    # No alternate output argument: this declared evaluation can start only once.
    OUTPUT.mkdir(parents=False, exist_ok=False)
    runner, launcher = ROOT / "scripts/eval_assisted_point_transfer.py", Path(__file__).resolve()
    runner_sha, launcher_sha = file_sha256(runner), file_sha256(launcher)
    start = {"stage": "aps_original54_transfer", "launcher_pid": os.getpid(),
             "origin_monotonic": PROGRAM_STARTED, "seconds": args.seconds,
             "runner": relative(runner), "runner_sha256": runner_sha,
             "launcher": relative(launcher), "launcher_sha256": launcher_sha,
             "native_return_grace_seconds": 30, "automatic_retry": False}
    (OUTPUT / "process-start.json").write_text(json.dumps(start, indent=2) + "\n")
    with (OUTPUT / "launcher-at-run.py").open("xb") as f:
        f.write(launcher.read_bytes())
    code, pid, launch_error, killed = None, None, None, False
    with (OUTPUT / "process-stdout.log").open("x") as out, (OUTPUT / "process-stderr.log").open("x") as err:
        try:
            child = subprocess.Popen([sys.executable, "-B", str(runner), "--seconds", str(args.seconds),
                                      "--origin-start", repr(PROGRAM_STARTED)],
                                     cwd=ROOT, stdout=out, stderr=err)
            pid = child.pid
            try:
                code = child.wait(timeout=max(.001, args.seconds + 30 - (time.perf_counter() - PROGRAM_STARTED)))
            except subprocess.TimeoutExpired:
                killed = True
                child.terminate()
                try:
                    code = child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    code = child.wait()
        except OSError as exc:
            launch_error = type(exc).__name__
    child_wall = time.perf_counter() - PROGRAM_STARTED
    # Emit terminal evidence before further file reads so a filesystem failure
    # cannot be mistaken for an unknown child outcome.
    print(json.dumps({"child_exit_code": code, "child_pid": pid, "seconds_to_child_exit": child_wall}), flush=True)
    report_path = OUTPUT / "report.json"
    report, error = {}, None
    try:
        report = json.loads(report_path.read_text())
    except (ValueError, OSError) as exc:
        error = type(exc).__name__
    source_same = file_sha256(runner) == runner_sha and file_sha256(launcher) == launcher_sha
    report_sha = file_sha256(report_path) if report_path.exists() else None
    wall = time.perf_counter() - PROGRAM_STARTED
    result = {"stage": "aps_original54_transfer", "exit_code": code, "child_pid": pid,
              "terminated_by_signal": -code if isinstance(code, int) and code < 0 else None,
              "launch_error": launch_error, "report_parse_error": error,
              "seconds_to_child_exit": child_wall, "seconds_before_completion_record": wall,
              "timing_scope": "Origin precedes launcher imports; second duration includes child report saving, parsing and source/report hashing, and excludes this final completion-record write. Printed terminal wall also includes that write.",
              "soft_total_seconds": args.seconds, "budget_exceeded": wall > args.seconds,
              "terminated_after_grace": killed, "automatic_retry": False,
              "runner_sha256": runner_sha, "launcher_sha256": launcher_sha,
              "source_unchanged": source_same, "report_sha256": report_sha,
              "valid_complete": (not error and not launch_error and not killed and source_same
                                   and effective_complete(report, code, wall, args.seconds))}
    with (OUTPUT / "process-completion.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    after_write = time.perf_counter() - PROGRAM_STARTED
    if after_write > args.seconds:
        # A completion write that crosses the cap cannot leave a valid endpoint.
        result.update(valid_complete=False, budget_exceeded=True,
                      seconds_after_initial_completion_record=after_write)
        (OUTPUT / "process-completion.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({**result, "seconds_through_completion_record": time.perf_counter() - PROGRAM_STARTED}), flush=True)
    return 0 if code == 0 and not error and not launch_error else 1


if __name__ == "__main__":
    raise SystemExit(main())
