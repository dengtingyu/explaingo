#!/usr/bin/env python3
"""Exclusive cloud process launcher for one fixed five-epoch run or explicit resume.

The parent never loads a tensor/model. It retains every attempt and its actual
exit status; no automatic restart, precision fallback or data change is made.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/train_qwen38_expanded.py"
STAGE = "qwen38_expanded_five_epoch"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def inside(relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Output must be project-relative")
    result = (ROOT / path).resolve()
    if not result.is_relative_to(ROOT):
        raise ValueError("Output path escapes project")
    return result


def microbatch_calls(examples, microbatch, completed_updates):
    """Expected top-level training calls for a persisted schedule prefix."""
    steps_per_epoch = math.ceil(examples / 16)
    epoch, within = divmod(completed_updates, steps_per_epoch)
    full_steps, remainder = divmod(examples, 16)
    per_epoch = full_steps * (16 // microbatch) + (math.ceil(remainder / microbatch) if remainder else 0)
    return epoch * per_epoch + within * (16 // microbatch)


def completed_contract(report, args):
    if not isinstance(report, dict) or report.get("stage") != STAGE or report.get("status") != "complete":
        return False
    total_updates = math.ceil(args.expected_train / 16) * 5
    restored = report.get("restored_or_initial_updates")
    if type(restored) is not int or not 0 <= restored <= total_updates:
        return False
    expected_calls = microbatch_calls(args.expected_train, args.microbatch, total_updates) - microbatch_calls(
        args.expected_train, args.microbatch, restored)
    evaluations = report.get("evaluations")
    return (report.get("valid_complete") is True and report.get("completed_updates") == total_updates
            and report.get("example_passes") == args.expected_train * 5 and report.get("epochs_completed") == 5
            and report.get("training_forward_attempts") == report.get("training_forward_returns") == expected_calls
            and report.get("backward_attempts") == report.get("backward_returns") == expected_calls
            and report.get("optimizer_step_attempts") == report.get("optimizer_step_returns") == total_updates - restored
            and report.get("evaluation_forward_attempts") == report.get("evaluation_forward_returns")
            and report.get("all_base_gradients_absent") is True
            and report.get("final_checkpoint_persistent_ack_verified") is True
            and report.get("checkpoint_publish_pending") is False
            and report.get("last_durable_updates") == total_updates
            and report.get("base_version_counters_unchanged") is True and report.get("cache_files_unchanged") is True
            and isinstance(evaluations, list) and len(evaluations) == 6
            and all(isinstance(item, dict) for item in evaluations)
            and [item.get("name") for item in evaluations] ==
                ["initial", "epoch-1", "epoch-2", "epoch-3", "epoch-4", "epoch-5"])


def latest_receipted_checkpoint(output, config, transport, state):
    """An older remote receipt cannot authorize discarding a newer local save."""
    identities = json.loads((output / "RUN.json").read_text())
    record = state.resolve_checkpoint(output / "checkpoints", identities)
    receipt = json.loads((output / "acks" / (record.path.name + ".json")).read_text())
    return record, transport.validate_ack(receipt, output, record, config)


def validate_final_checkpoint_reference(ack, record):
    expected = {"checkpoint_name": record.path.name,
                "checkpoint_manifest_sha256": record.manifest_sha256,
                "completed_updates": record.metadata["cursor"]["completed_updates"]}
    if not isinstance(ack, dict) or any(ack.get(key) != value for key, value in expected.items()):
        raise ValueError("Final artifact receipt does not cover the newest local checkpoint")
    return ack


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", required=True, choices=("bf16-two-gpu", "nf4-single-gpu"))
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--asset-record", required=True)
    parser.add_argument("--asset-hashes", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--output", default="outputs/qwen38-expanded")
    parser.add_argument("--expected-train", type=int, default=126561)
    parser.add_argument("--microbatch", type=int, choices=(1, 16), required=True)
    parser.add_argument("--cache-capacity", type=int, default=1404)
    parser.add_argument("--legacy-manifest", required=True)
    parser.add_argument("--persistence-config", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Linux" or not ROOT.is_relative_to(Path("/workspace")):
        parser.error("Cloud Linux /workspace project required")
    if args.expected_train < 100000 or args.cache_capacity < 1:
        parser.error("At least 100000 distinct training questions required")
    sys.path.insert(0, str(ROOT / "src"))
    from goqueen import cloud_checkpoint_transport as transport
    from goqueen import cloud_training_state as state
    config_path = Path(args.persistence_config).resolve()
    if not config_path.is_relative_to(Path("/workspace")):
        raise ValueError("Private persistence configuration must be on cloud /workspace")
    persistence_config = transport.load_config(config_path)
    persistence_config_sha = sha(config_path)
    receiver_path = ROOT / "scripts/receive_cloud_checkpoint.py"
    publisher_path = ROOT / "src/goqueen/cloud_checkpoint_transport.py"
    receiver_sha, publisher_sha = sha(receiver_path), sha(publisher_path)
    if transport.public_identity(persistence_config)["receiver_sha256"] != receiver_sha:
        raise ValueError("Persistent receiver does not match the pinned first-party receiver")
    output = inside(args.output)
    if args.resume:
        if not output.is_dir() or not (output / "RUN.json").is_file() or not (output / "checkpoints").is_dir():
            raise ValueError("Explicit resume requires an existing run identity and committed checkpoint root")
    else:
        output.mkdir(parents=True, exist_ok=False)
    with (output / "active.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        attempts = output / "attempts"
        attempts.mkdir(exist_ok=True)
        prior = [int(path.name) for path in attempts.iterdir() if path.is_dir() and path.name.isdigit()]
        attempt = attempts / f"{max(prior, default=-1) + 1:06d}"
        attempt.mkdir(exist_ok=False)
        outer = time.monotonic()
        launcher_sha, runner_sha = sha(__file__), sha(RUNNER)
        (attempt / "launcher-at-run.py").write_bytes(Path(__file__).read_bytes())
        (attempt / "runner-at-run.py").write_bytes(RUNNER.read_bytes())
        if sha(attempt / "launcher-at-run.py") != launcher_sha or sha(attempt / "runner-at-run.py") != runner_sha:
            raise ValueError("Source changed during launcher snapshot capture")
        visible_arguments = dict(vars(args), persistence_config="<private cloud configuration>")
        write_json(attempt / "process-start.json", {"stage": STAGE, "launcher_pid": os.getpid(),
                   "outer_start_monotonic": outer, "started_unix": time.time(), "arguments": visible_arguments,
                   "launcher_sha256": launcher_sha, "runner_sha256": runner_sha,
                   "persistence": transport.public_identity(persistence_config),
                   "publisher_sha256": publisher_sha, "receiver_sha256": receiver_sha,
                   "automatic_resume": False, "wall_cap_seconds": None})
        command = [sys.executable, "-B", str(RUNNER), *sys.argv[1:],
                   "--attempt", str(attempt.relative_to(ROOT)), "--outer-start", str(outer)]
        environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        child_pid, code, launch_error, child = None, None, None, None
        termination = {"signal": None, "deadline": None, "killed_after_grace": False}
        def terminate_child(signum, frame):
            if termination["signal"] is None:
                termination["signal"] = signal.Signals(signum).name
                termination["deadline"] = time.monotonic() + 60
            if child is not None and child.poll() is None:
                try:
                    os.killpg(child.pid, signum)
                except ProcessLookupError:
                    pass
        previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        for sig in previous_handlers:
            signal.signal(sig, terminate_child)
        with (attempt / "stdout.log").open("x") as stdout, (attempt / "stderr.log").open("x") as stderr:
            try:
                if termination["signal"] is not None:
                    raise OSError("Stop signal received before child launch")
                child = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=environment,
                            start_new_session=True, pass_fds=(lock.fileno(),))
                child_pid = child.pid
                print(json.dumps({"stage": STAGE, "attempt": str(attempt.relative_to(ROOT)),
                                  "child_pid": child_pid, "resume": args.resume}), flush=True)
                if termination["signal"] is not None:
                    terminate_child(getattr(signal, termination["signal"]), None)
                while True:
                    try:
                        timeout = 60 if termination["deadline"] is None else max(0.1, termination["deadline"] - time.monotonic())
                        code = child.wait(timeout=timeout)
                        break
                    except subprocess.TimeoutExpired:
                        if termination["deadline"] is not None and time.monotonic() >= termination["deadline"]:
                            os.killpg(child.pid, signal.SIGKILL)
                            termination["killed_after_grace"] = True
                            code = child.wait()
                            break
            except OSError as error:
                launch_error = str(error).replace(str(ROOT), "PROJECT")
            finally:
                for sig, previous in previous_handlers.items():
                    signal.signal(sig, previous)
        child_wall = time.monotonic() - outer
        report_path = attempt / "report.json"
        try:
            report = json.loads(report_path.read_text())
        except (OSError, ValueError):
            report = {}
        good = code == 0 and launch_error is None and termination["signal"] is None and completed_contract(report, args)
        # No tensor load: authenticate the exact committed manifest named by the child.
        verification_error = None
        if good:
            try:
                checkpoint = (output / report["final_checkpoint"]).resolve()
                good = (checkpoint.is_relative_to(output / "checkpoints")
                        and sha(checkpoint / "MANIFEST.json") == report["final_checkpoint_manifest_sha256"]
                        and (checkpoint / "READY").read_text().strip() == report["final_checkpoint_manifest_sha256"])
                if good:
                    record, ack = latest_receipted_checkpoint(output, persistence_config, transport, state)
                    good = (record.path == checkpoint and ack["unit_sha256"] == report["final_checkpoint_durable_unit_sha256"])
            except (OSError, ValueError, TypeError, KeyError, transport.TransportError) as error:
                good = False
                verification_error = str(error).replace(str(ROOT), "PROJECT")
        completion = {"stage": STAGE, "child_pid": child_pid, "exit_code": code,
                      "signal": signal.Signals(-code).name if type(code) is int and code < 0 else None,
                      "launch_error": launch_error, "external_child_wall_seconds": child_wall,
                      "completion_verification_error": verification_error,
                      "parent_stop_signal": termination["signal"],
                      "child_killed_after_stop_grace": termination["killed_after_grace"],
                      "effective_complete": good, "report_sha256": sha(report_path) if report_path.is_file() else None,
                      "runner_source_unchanged": sha(RUNNER) == runner_sha,
                      "launcher_source_unchanged": sha(__file__) == launcher_sha,
                      "publisher_source_unchanged": sha(publisher_path) == publisher_sha,
                      "receiver_source_unchanged": sha(receiver_path) == receiver_sha,
                      "private_persistence_config_unchanged": sha(config_path) == persistence_config_sha,
                      "manual_resume_only": True}
        if not all(completion[key] for key in ("runner_source_unchanged", "launcher_source_unchanged",
                "publisher_source_unchanged", "receiver_source_unchanged", "private_persistence_config_unchanged")):
            completion["effective_complete"] = False
        write_json(attempt / "process-completion.json", completion)
        # Export only after the actual parent completion and closed child logs
        # exist. Never rewrite that completion after it has been published.
        # The final receipt is separate, avoiding a report/ACK hash cycle.
        final_ack, export_error, latest_local_ack_verified = None, None, False
        final_artifact_coverage, all_ended_attempts_acknowledged = None, False
        try:
            if not all(completion[key] for key in ("publisher_source_unchanged", "receiver_source_unchanged",
                                                   "private_persistence_config_unchanged")):
                raise ValueError("Persistence source/config changed; final export refused")
            latest_record, _ = latest_receipted_checkpoint(output, persistence_config, transport, state)
            latest_local_ack_verified = True
            received = transport.publish_final_artifacts(output, persistence_config)
            final_artifact_coverage = transport.validate_final_artifacts(output, persistence_config)
            if (final_artifact_coverage.get("all_ended_attempts_acknowledged") is not True
                    or final_artifact_coverage.get("latest_final_ack") != received):
                raise ValueError("Final receipts do not cover every ended attempt")
            final_ack = validate_final_checkpoint_reference(received, latest_record)
            all_ended_attempts_acknowledged = True
        except Exception as error:
            export_error = str(error).replace(str(ROOT), "PROJECT").replace(str(config_path), "PRIVATE_CONFIG")
        durable_completion = {"stage": STAGE, "effective_complete": completion["effective_complete"],
                              "final_artifacts_acknowledged": final_ack is not None,
                              "final_ack": final_ack, "export_error": export_error,
                              "latest_local_checkpoint_ack_verified": latest_local_ack_verified,
                              "all_ended_attempts_acknowledged": all_ended_attempts_acknowledged,
                              "final_artifact_coverage": final_artifact_coverage,
                              "shutdown_ready": (latest_local_ack_verified and final_ack is not None
                                                 and all_ended_attempts_acknowledged),
                              "process_completion_sha256": sha(attempt / "process-completion.json"),
                              "terminal_wall_seconds": time.monotonic() - outer}
        write_json(attempt / "durability-completion.json", durable_completion)
        print(json.dumps({**completion, **durable_completion}), flush=True)
        return 0 if completion["effective_complete"] and final_ack is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
