#!/usr/bin/env python3
"""One frozen APS evaluation: 21 restoration cells, then the existing original54.

No optimizer, training, generation, expanded set, retry or automatic follow-on.
Invoke through run_assisted_point_transfer.py to retain the external outcome.
"""
from __future__ import annotations

import time
PROGRAM_STARTED = time.perf_counter()

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_grounding_microfit import file_sha256, content_sha256, require, relative, parameter_digest, LAYERS

OUTPUT = ROOT / "outputs/assisted-point-transfer"
MANIFEST = ROOT / "outputs/grounding-data/manifest.json"
FIT_REPORT = ROOT / "outputs/grounding-assisted-point/report.json"
CHECKPOINT = ROOT / "outputs/grounding-assisted-point/adapter.safetensors"
BASELINE = ROOT / "outputs/grounding-baseline/original-val-calibrated-report.json"
MANIFEST_SHA = "2c3deae2c0bde1cdd52b105889f0558e295662eb0dc222ffdbf2acbe07bd6279"
FIT_SHA = "2eb4bab5e4e86ff33e586bae94ee010a4cb005a84c0d76ada40903e94d9e75a4"
CHECKPOINT_SHA = "1532a7420e946c53983a6a6b3ce95fe9bf7dd7dd35e5ef1d1e4a02bc97bbad79"
BASELINE_SHA = "ace7d6db92f25cefaf5d8b63b1507827b7e3701002d088d013ec1e4e2e398c72"
TOLERANCE = 1e-5
RESERVE_SECONDS = 15


class StopEvaluation(Exception):
    pass


def select_questions(data, fit):
    """Preserve the declared lists; do not choose examples by observed scores."""
    train = fit["fixed_questions"]
    val = [q for q in data["original_occupancy_questions"] if q["split"] == "val"]
    require(len(train) == 21 and len(val) == 54, "Expected 21 restoration and 54 validation cells")
    for group in (train, val):
        require(len({q["question_id"] for q in group}) == len(group), "Duplicate question identity")
    require(Counter(q["label"] for q in val) == {0: 18, 1: 18, 2: 18}, "Validation class counts changed")
    require(Counter(q["to_play"] for q in val) == {"B": 27, "W": 27}, "Validation side counts changed")
    require(sorted(Counter(q["game_id"] for q in val).values()) == [18] * 3,
            "Validation game coverage changed")
    require(len({q["position_id"] for q in val}) == 18, "Validation position coverage changed")
    separation = {}
    for key in ("game_id", "position_id", "question_id", "prompt"):
        separation[key + "_overlap"] = len({q[key] for q in train} & {q[key] for q in val})
        require(separation[key + "_overlap"] == 0, "Training and validation overlap: " + key)
    separation["validation_cells_with_training_point"] = sum(
        q["point_index"] in {t["point_index"] for t in train} for q in val)
    require(separation["validation_cells_with_training_point"] == 1, "Coordinate overlap changed")
    require(content_sha256(train) == fit["source"]["fixed_questions_sha256"], "Training list changed")
    final = [e for e in fit["evaluations"] if e["name"] == "final"]
    require(len(final) == 1 and final[0]["updates_completed"] == 180
            and final[0]["coverage_complete"] and len(final[0]["rows"]) == 21,
            "Saved APS final reference incomplete")
    return train, val, final[0]["rows"], separation


def restore_allows_validation(comparison):
    if not isinstance(comparison, dict):
        return False
    return (comparison.get("passed") is True and comparison.get("coverage_complete") is True
            and comparison.get("predictions_match") is True
            and type(comparison.get("max_abs_logp_difference")) in (int, float)
            and math.isfinite(comparison["max_abs_logp_difference"])
            and 0 <= comparison["max_abs_logp_difference"] <= TOLERANCE
            and comparison.get("tolerance") == TOLERANCE)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--origin-start", type=float, required=True)
    args = parser.parse_args(argv)
    require(RESERVE_SECONDS < args.seconds <= 300 and math.isfinite(args.origin_start)
            and 0 < args.origin_start <= PROGRAM_STARTED, "Invalid total budget origin")
    origin, deadline = args.origin_start, args.origin_start + args.seconds
    process_start = json.loads((OUTPUT / "process-start.json").read_text())
    require(process_start["launcher_pid"] == os.getppid()
            and process_start["origin_monotonic"] == origin
            and process_start["seconds"] == args.seconds,
            "Use the single-run launcher")
    path, temporary = OUTPUT / "report.json", OUTPUT / "report.tmp"
    require(not any(p.exists() for p in (path, temporary, OUTPUT / "run-start.json")), "Existing run retained")
    source = Path(__file__).read_bytes()
    with (OUTPUT / "run-start.json").open("x") as f:
        json.dump({"stage": "aps_original54_transfer", "automatic_retry": False,
                   "runner_sha256": file_sha256(__file__), "seconds": args.seconds}, f)
    with (OUTPUT / "runner-at-run.py").open("xb") as f:
        f.write(source)
    report = {"stage": "aps_original54_transfer", "status": "running", "valid_complete": False,
              "scope": "Descriptive frozen-checkpoint assisted prediction on previously inspected validation games; not blind, causal, natural-selection or generated-answer evaluation.",
              "runtime": {"device": "cpu", "threads": 4, "dtype": "float32", "sdpa": "math",
                          "batch_size": 1, "soft_total_seconds": args.seconds, "reserve_seconds": RESERVE_SECONDS},
              "optimizer_constructed": False, "optimizer_updates": 0, "backward_calls": 0,
              "restoration_rows": [], "validation_rows": [], "restoration": None,
              "validation_started": False, "automatic_retry": False, "error": None,
              "metric_contract": "First argmax empty/black/white; exact top ties separate; correct/54, confusion, game and side counts; three-label CE distinct from full-vocabulary CE; no performance bands."}
    phase, model, monitor = "source_validation", None, None

    def record():
        report["seconds_before_report_write"] = time.perf_counter() - origin
        report["budget_exceeded"] = report["seconds_before_report_write"] > args.seconds
        if report["budget_exceeded"]:
            report["valid_complete"] = False
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    def expired(_signum, _frame):
        raise StopEvaluation("soft_deadline")

    def time_available():
        if time.perf_counter() >= deadline - RESERVE_SECONDS:
            raise StopEvaluation("evaluation_deadline_with_save_reserve")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - RESERVE_SECONDS - time.perf_counter()))
    record()
    try:
        pins = {relative(p): file_sha256(p) for p in (MANIFEST, FIT_REPORT, CHECKPOINT, BASELINE)}
        require(pins[relative(MANIFEST)] == MANIFEST_SHA and pins[relative(FIT_REPORT)] == FIT_SHA
                and pins[relative(CHECKPOINT)] == CHECKPOINT_SHA
                and pins[relative(BASELINE)] == BASELINE_SHA, "Pinned APS inputs changed")
        data, fit, baseline = [json.loads(p.read_text()) for p in (MANIFEST, FIT_REPORT, BASELINE)]
        train, val, saved_final, separation = select_questions(data, fit)
        report.update(input_sha256=pins, restoration_questions=train, validation_questions=val,
                      validation_questions_sha256=content_sha256(val), separation=separation)
        phase = "imports"
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        sys.path.insert(0, str(ROOT / "src"))
        from goqueen.assisted_point import AssistedPointGoQueen
        from goqueen.assisted_point_eval import score_questions
        from goqueen.checkpoints import load_adapter_checkpoint
        from goqueen.grounding_data import point_to_gtp, occupancy_prompt
        from goqueen.grounding_eval import empirical_prior_baselines
        from goqueen.transfer_metrics import summarize_transfer, compare_restoration, align_saved_reference
        from train_grounding_assisted_point import RoutingMonitor, binding_clear, imported_first_party

        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        old_rows = align_saved_reference(val, baseline["rows"])
        priors = empirical_prior_baselines(data, val)
        # Historical priors saved summaries, not individual rows. Recompute only
        # existing categorical arithmetic from the pinned training pool.
        for name in ("train_majority", "coordinate_side_empirical_prior"):
            require(priors[name]["summary"] == baseline["train_only_baselines"][name]["summary"],
                    "Historical categorical reference changed")
        report["references"] = {
            "scope": "Historical references, different supervision/checkpoint; not matched causal arms.",
            "old_natural": summarize_transfer(old_rows, val),
            "categorical_priors": {name: priors[name]["summary"] for name in
                                   ("train_majority", "coordinate_side_empirical_prior")},
            "prior_training_questions": priors["training_questions"],
            "prior_probability_space": "categorical only; no full-vocabulary CE",
            "reference_alignment_complete": True}
        for q in train + val:
            require(q["point"] == point_to_gtp(q["point_index"])
                    and q["prompt"] == occupancy_prompt(q["to_play"], q["point"]),
                    "Canonical coordinate or prompt mismatch")
        positions = {p["position_id"]: p for p in data["positions"]}
        features, feature_pins = {}, {}
        phase = "load_features"
        for pid in dict.fromkeys(q["position_id"] for q in train + val):
            time_available()
            position = positions[pid]
            feature_path = (ROOT / position["source_feature_file"]).resolve()
            digest = file_sha256(feature_path)
            require(digest == position["feature_sha256"], "Feature bytes changed")
            with np.load(feature_path, allow_pickle=False) as saved:
                layers = [saved[f"trunk_after_block{i}"] for i in range(11)]
                require(all(x.shape == (361, 768) and x.dtype == np.float32 and np.isfinite(x).all()
                            for x in layers), "Invalid expert cache")
                features[pid] = torch.from_numpy(np.stack(layers))
            for q in train + val:
                if q["position_id"] == pid:
                    require(q["game_id"] == position["game_id"] and q["to_play"] == position["to_play"]
                            and q["label"] == position["absolute_board_labels"][q["point_index"]],
                            "Question disagrees with pinned board metadata")
            feature_pins[pid] = {"file": relative(feature_path), "sha256": digest}
        report["feature_sources"] = feature_pins
        report["implementation_sha256"] = imported_first_party()
        copies = OUTPUT / "source-at-run"
        for name, digest in report["implementation_sha256"].items():
            target = copies / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as f:
                f.write((ROOT / name).read_bytes())
            require(file_sha256(target) == digest, "Runtime source copy mismatch")
        record()  # Selection, metrics and provenance are retained before forward.
        time_available()
        phase = "load_frozen_model"
        model_path = ROOT / fit["local_model"]["directory"]
        require(file_sha256(model_path / "config.json") == fit["local_model"]["config_sha256"], "Base config changed")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        language = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                                                       dtype=torch.float32, attn_implementation="sdpa")
        language.gradient_checkpointing_disable()
        torch.manual_seed(42)
        model = AssistedPointGoQueen(language, LAYERS, 768, 8)
        # Disk loader uses requires_grad to identify adapters: load before freezing.
        adapter_parameters = [p for p in model.parameters() if p.requires_grad]
        require(len(adapter_parameters) == 101 and sum(p.numel() for p in adapter_parameters) == 40438795,
                "Unexpected adapter layout")
        metadata = load_adapter_checkpoint(model, CHECKPOINT)
        require(metadata == fit["checkpoint_metadata"] and metadata["engineering_only"] == "false"
                and metadata["updates"] == "180", "Not the retained fitted APS checkpoint")
        model.requires_grad_(False)
        model.eval()
        require(all(not p.requires_grad and p.dtype == torch.float32 and p.device.type == "cpu"
                    for p in model.parameters()), "Model must be frozen CPU FP32")
        report["checkpoint_metadata"] = metadata
        report["base_sha256_before"] = parameter_digest(model.base_parameters)
        require(report["base_sha256_before"] == fit["frozen_base_sha256_after"], "Base parameters changed")
        report["adapter_sha256_before"] = parameter_digest(adapter_parameters)
        monitor = RoutingMonitor(model)

        def score_list(questions, key):
            for q in questions:
                time_available()
                with monitor.expect([q["point_index"]], "evaluation"):
                    rows = score_questions(model, tokenizer, [q], features, batch_size=1,
                                           deadline=deadline - RESERVE_SECONDS)
                require(len(rows) == 1, "A requested forward did not return one score")
                require(all(math.isfinite(x) for x in rows[0]["label_log_probs"])
                        and math.isfinite(rows[0]["off_label_mass"]), "Non-finite score")
                report[key].extend(rows)
                require(binding_clear(model), "Temporary APS binding not cleared")
                record()

        phase = "restoration"
        score_list(train, "restoration_rows")
        report["restoration"] = compare_restoration(report["restoration_rows"], saved_final, TOLERANCE)
        record()
        if not restore_allows_validation(report["restoration"]):
            raise StopEvaluation("restoration_guard_failed")
        phase = "validation"
        report["validation_started"] = True
        score_list(val, "validation_rows")
        report["summary"] = summarize_transfer(report["validation_rows"], val)
        phase = "final_checks"
        report["base_sha256_after"] = parameter_digest(model.base_parameters)
        report["adapter_sha256_after"] = parameter_digest(adapter_parameters)
        report["all_parameters_frozen_and_gradients_absent"] = all(
            not p.requires_grad and p.grad is None for p in model.parameters())
        report["runtime_binding_cleared"] = binding_clear(model)
        require(report["base_sha256_after"] == report["base_sha256_before"]
                and report["adapter_sha256_after"] == report["adapter_sha256_before"]
                and report["all_parameters_frozen_and_gradients_absent"], "Frozen state changed")
        after = {name: file_sha256(ROOT / name) for name in report["implementation_sha256"]}
        require(after == report["implementation_sha256"], "Imported source changed during evaluation")
        require(all(file_sha256(ROOT / name) == digest for name, digest in pins.items()), "Pinned input changed")
        require(all(file_sha256(ROOT / f["file"]) == f["sha256"] for f in feature_pins.values()), "Feature source changed")
        report["source_hashes_unchanged"] = True
        require(report["summary"]["coverage_complete"] and len(report["validation_rows"]) == 54,
                "Incomplete validation coverage")
        report["status"], report["valid_complete"] = "complete", True
    except StopEvaluation as exc:
        report["status"], report["stop_reason"] = "stopped", str(exc)
    except Exception as exc:
        report["status"] = "error"
        report["error"] = {"phase": phase, "type": type(exc).__name__,
                           "message": str(exc).replace(str(ROOT), ".")}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if monitor is not None:
            report["routing_observation"] = {"adapter_calls": monitor.adapter_calls,
                                              "routed_samples": monitor.routed_samples,
                                              "forward_calls": monitor.counts}
            monitor.close()
        report["last_phase"] = phase
        report["coverage"] = {"restoration": len(report["restoration_rows"]),
                              "validation": len(report["validation_rows"]),
                              "requested_restoration": 21, "requested_validation": 54}
        record()
        if time.perf_counter() > deadline:
            report["valid_complete"] = False
            report["budget_exceeded"] = True
            record()
    print(json.dumps({"status": report["status"], "valid_complete": report["valid_complete"],
                      "coverage": report["coverage"], "seconds": time.perf_counter() - origin}), flush=True)
    return 1 if report["status"] == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
