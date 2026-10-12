#!/usr/bin/env python3
"""Future single APS-warm Q/K run; invoke only through run_aps_warm_qk.py.

The frozen APS solution supplies all 101 adapter tensors. Only Q/K and query
normalization train after conversion to original natural attention. This file
does not run when imported; the implementation stage uses a separate zero-step
engineering driver. A valid endpoint additionally requires the launcher's
actual OS outcome and total wall time.
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
from eval_assisted_point_transfer import (MANIFEST, FIT_REPORT, CHECKPOINT, MANIFEST_SHA,
    FIT_SHA, CHECKPOINT_SHA, TOLERANCE, select_questions, restore_allows_validation)

STAGE, MAX_UPDATES, BATCH_SIZE = "aps_warm_qk", 1000, 4
SUFFIXES = (".query_norm.weight", ".query_norm.bias", ".query.weight", ".key.weight")


class StopRun(Exception):
    pass


def fixed_schedule(data, fit):
    """Use the existing 4,000 slots verbatim, without selecting by scores."""
    restoration, validation, saved, aps_separation = select_questions(data, fit)
    pool = data["training_pool_questions"]
    require(len({q["question_id"] for q in pool}) == len(pool), "Duplicate training IDs")
    lookup = {q["question_id"]: q for q in pool}
    order = data["train_order"]
    require(len(order) == 4000 and all(qid in lookup for qid in order), "Training order changed")
    selected = [lookup[qid] for qid in order]
    require(len(set(order)) == 2702 and Counter(q["label"] for q in selected) == {0: 1334, 1: 1333, 2: 1333},
            "Training identity or class counts changed")
    require(len({q["game_id"] for q in selected}) == 5
            and len({q["position_id"] for q in selected}) == 30
            and len({q["point_index"] for q in selected}) == 360, "Training coverage changed")
    separation = {key + "_overlap": len({q[key] for q in selected} & {q[key] for q in validation})
                  for key in ("game_id", "position_id", "prompt")}
    require(separation["game_id_overlap"] == separation["position_id_overlap"] == 0
            and separation["prompt_overlap"] == 52, "Declared separation changed")
    require(all(q["prompt"] in {t["prompt"] for t in selected} for q in validation),
            "All validation prompts must retain their declared training overlap")
    return restoration, validation, saved, selected, {"new_training": separation, "aps_training": aps_separation}


def closing_reserve(initial_seconds):
    require(type(initial_seconds) in (float, int) and math.isfinite(initial_seconds)
            and initial_seconds >= 0, "Invalid measured initial evaluation time")
    return max(300.0, 2 * initial_seconds + 60.0)


def can_start_step(now, deadline, reserve, slowest):
    return all(math.isfinite(x) for x in (now, deadline, reserve, slowest)) and now + reserve + slowest < deadline


def binding_clear(model):
    return (not model._bound and all(layer.expert_features is None
            and getattr(layer.adapter, "point_mask", None) is None
            and getattr(layer.adapter, "point_indices", None) is None for layer in model.injected_layers))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/aps-warm-qk/report.json")
    parser.add_argument("--outer-start", type=float, required=True)
    args = parser.parse_args(argv)
    require(1 <= args.seconds <= 1800 and math.isfinite(args.outer_start)
            and 0 < args.outer_start <= PROGRAM_STARTED, "Invalid total budget")
    output = args.output.resolve()
    require(output.parent.parent == (ROOT / "outputs").resolve(), "Use a new output directory")
    parent = json.loads((output.parent / "process-start.json").read_text())
    require(parent["launcher_pid"] == os.getppid()
            and parent["outer_start_monotonic"] == args.outer_start
            and parent["total_soft_cap_seconds"] == args.seconds
            and parent["runner_sha256"] == file_sha256(__file__), "Use the pinned single-run launcher")
    artifacts = [output, output.with_suffix(".tmp")] + [output.parent / n for n in
                 ("run-start.json", "runner-at-run.py", "adapter.safetensors", "adapter.partial.safetensors")]
    require(len({p.name.casefold() for p in artifacts}) == len(artifacts)
            and not any(p.exists() for p in artifacts), "Preserve existing run artifacts")
    origin, deadline = args.outer_start, args.outer_start + args.seconds
    source_sha = file_sha256(__file__)
    with (output.parent / "run-start.json").open("x") as stream:
        json.dump({"stage": STAGE, "source_sha256": source_sha, "max_seconds": args.seconds,
                   "automatic_retry": False}, stream)
    with (output.parent / "runner-at-run.py").open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    report = {"stage": STAGE, "status": "running", "run_completed": False,
              "source_script_sha256": source_sha, "max_seconds": args.seconds,
              "updates_completed": 0, "sample_slots": 0, "step_attempts": [],
              "restoration_rows": [], "evaluations": [], "endpoint_conditions": {},
              "automatic_retry": False, "optimizer_constructed": False,
              "scope": "Previously viewed game/position holdout with complete prompt overlap; finite recipe comparison, not causal addressing or generalization proof.",
              "runtime": {"device": "cpu", "threads": 4, "dtype": "float32", "sdpa": "math", "batch_size": 4}}
    phase, model, monitor = "sources", None, None

    def record():
        report["total_seconds"] = time.perf_counter() - origin
        report["budget_exceeded"] = report["total_seconds"] > args.seconds
        if report["budget_exceeded"]:
            report["run_completed"] = False
            report["endpoint_conditions"]["within_child_budget"] = False
        temp = output.with_suffix(".tmp")
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temp.replace(output)

    def time_available(reserve=30):
        if time.perf_counter() >= deadline - reserve:
            raise StopRun("soft_budget_reserve")

    def alarm(_signum, _frame):
        raise StopRun("soft_deadline")

    previous = signal.signal(signal.SIGALRM, alarm)
    try:
        signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - 30 - time.perf_counter()))
        record()
        pins = {relative(p): file_sha256(p) for p in (MANIFEST, FIT_REPORT, CHECKPOINT)}
        require([pins[relative(p)] for p in (MANIFEST, FIT_REPORT, CHECKPOINT)]
                == [MANIFEST_SHA, FIT_SHA, CHECKPOINT_SHA], "Pinned inputs changed")
        data, fit = json.loads(MANIFEST.read_text()), json.loads(FIT_REPORT.read_text())
        restore, val, saved, schedule, separation = fixed_schedule(data, fit)
        report.update(input_sha256=pins, restoration_questions=restore, validation_questions=val,
                      train_order=[q["question_id"] for q in schedule], train_order_sha256=content_sha256(data["train_order"]),
                      separation=separation)
        phase = "imports"
        import numpy as np
        import torch
        from torch.nn.attention import sdpa_kernel, SDPBackend
        from transformers import AutoModelForCausalLM, AutoTokenizer
        sys.path.insert(0, str(ROOT / "src"))
        from goqueen.assisted_point import AssistedPointGoQueen
        from goqueen.assisted_point_eval import score_questions as aps_score
        from goqueen.checkpoints import load_adapter_checkpoint
        from goqueen.grounding_data import point_to_gtp, occupancy_prompt, _absolute_labels
        from goqueen.grounding_eval import score_questions, encode_prefix, _forward_logits
        from goqueen.transfer_metrics import summarize_transfer, compare_restoration
        from goqueen.warm_qk import adapter_parameters, convert_to_natural, freeze_except_qk, save_full_checkpoint
        from train_grounding_assisted_point import RoutingMonitor, imported_first_party, optimizer_contract
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        import transformers
        report["versions"] = {"python": sys.version.split()[0], "torch": torch.__version__,
                              "transformers": transformers.__version__, "numpy": np.__version__}
        positions = {p["position_id"]: p for p in data["positions"]}
        unique = list({q["question_id"]: q for q in restore + val + schedule}.values())
        for q in unique:
            require(q["point"] == point_to_gtp(q["point_index"])
                    and q["prompt"] == occupancy_prompt(q["to_play"], q["point"]), "Noncanonical input")
        phase = "features"
        features, feature_pins = {}, {}
        for pid in dict.fromkeys(q["position_id"] for q in unique):
            time_available()
            position = positions[pid]
            path = ROOT / position["source_feature_file"]
            relative(path)
            require(file_sha256(path) == position["feature_sha256"], "Feature source changed")
            require(_absolute_labels(path, position["to_play"]) == position["absolute_board_labels"],
                    "Saved factual board differs from input planes")
            with np.load(path, allow_pickle=False) as cache:
                arrays = [cache[f"trunk_after_block{i}"] for i in range(11)]
                require(all(a.shape == (361, 768) and a.dtype == np.float32 and np.isfinite(a).all()
                            for a in arrays), "Invalid expert cache")
                features[pid] = torch.from_numpy(np.stack(arrays))
            for q in unique:
                if q["position_id"] == pid:
                    require(q["label"] == position["absolute_board_labels"][q["point_index"]]
                            and q["to_play"] == position["to_play"] and q["game_id"] == position["game_id"],
                            "Factual metadata disagreement")
            feature_pins[relative(path)] = position["feature_sha256"]
        report["feature_sha256"] = feature_pins
        implementation = imported_first_party()
        implementation[relative(ROOT / "scripts/run_aps_warm_qk.py")] = parent["launcher_sha256"]
        for name, digest in implementation.items():
            target = output.parent / "source-at-run" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write((ROOT / name).read_bytes())
            require(file_sha256(target) == digest, "Source changed before copying")
        report["implementation_sha256"] = implementation
        record()
        phase = "load_aps"
        time_available()
        model_path = ROOT / fit["local_model"]["directory"]
        require(file_sha256(model_path / "config.json") == fit["local_model"]["config_sha256"], "Base config changed")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        language = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                    dtype=torch.float32, attn_implementation="sdpa")
        language.gradient_checkpointing_disable()
        torch.manual_seed(42)
        model = AssistedPointGoQueen(language, LAYERS, 768, 8)
        named = adapter_parameters(model)
        require(len(named) == 101 and sum(p.numel() for _, p in named) == 40438795, "Adapter layout changed")
        metadata = load_adapter_checkpoint(model, CHECKPOINT)
        require(metadata == fit["checkpoint_metadata"] and metadata["engineering_only"] == "false"
                and metadata["updates"] == "180", "Not the fitted APS checkpoint")
        model.requires_grad_(False)
        model.eval()
        require(all(p.dtype == torch.float32 and p.device.type == "cpu" for p in model.parameters()), "Require CPU FP32")
        report["base_sha256_before"] = parameter_digest(model.base_parameters)
        require(report["base_sha256_before"] == fit["frozen_base_sha256_after"], "Base bytes differ")
        report["adapter_sha256_before_conversion"] = parameter_digest(p for _, p in named)
        monitor = RoutingMonitor(model)
        phase = "aps_restoration"
        for q in restore:
            time_available()
            with monitor.expect([q["point_index"]], "evaluation"):
                rows = aps_score(model, tokenizer, [q], features, batch_size=1, deadline=deadline - 30)
            require(len(rows) == 1, "Incomplete restoration forward")
            summarize_transfer(rows, [q])  # Reject non-finite/malformed data before retained JSON.
            report["restoration_rows"].extend(rows)
            record()
        report["restoration"] = compare_restoration(report["restoration_rows"], saved, TOLERANCE)
        require(restore_allows_validation(report["restoration"]), "Restoration guard failed; do not retry")
        report["aps_routing_calls"] = monitor.adapter_calls
        monitor.close()
        monitor = None
        phase = "natural_conversion"
        rng = torch.get_rng_state().clone()
        convert_to_natural(model)
        after_named = adapter_parameters(model)
        require(all(n == m and p is q for (n, p), (m, q) in zip(named, after_named))
                and len(after_named) == 101 and torch.equal(rng, torch.get_rng_state())
                and parameter_digest(p for _, p in named) == report["adapter_sha256_before_conversion"],
                "Conversion altered parameters or RNG")
        trainable = freeze_except_qk(model, named)
        require(len(trainable) == 44 and sum(p.numel() for _, p in trainable) == 20207616
                and all(n.endswith(SUFFIXES) for n, _ in trainable), "Trainable subset changed")
        frozen = [(n, p) for n, p in named if not p.requires_grad]
        report["frozen_adapter_sha256_before"] = parameter_digest(p for _, p in frozen)
        before_each = {n: parameter_digest([p]) for n, p in trainable}
        report["trainable_names"] = [n for n, _ in trainable]
        report["trainable_sha256_before"] = before_each
        report["conversion_verified"] = True

        def evaluate(name):
            evaluation = {"name": name, "updates_completed": report["updates_completed"],
                          "coverage_complete": False, "rows": []}
            report["evaluations"].append(evaluation)
            began = time.perf_counter()
            for q in val:
                time_available()
                rows = score_questions(model, tokenizer, [q], features, batch_size=1, deadline=deadline - 30)
                require(len(rows) == 1 and binding_clear(model), "Incomplete or uncleared natural scoring")
                summarize_transfer(rows, [q])
                evaluation["rows"].extend(rows)
                record()
            evaluation.update(summarize_transfer(evaluation["rows"], val))
            evaluation["seconds"] = time.perf_counter() - began
            record()
            return evaluation["seconds"]

        phase = "natural_initial"
        reserve = closing_reserve(evaluate("initial"))
        report["closing_reserve_seconds"] = reserve
        encoded = {q["prompt"]: encode_prefix(tokenizer, q["prompt"]) for q in unique}
        optimizer = torch.optim.AdamW([p for _, p in trainable], lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0)
        report["optimizer_contract"] = optimizer_contract(optimizer, trainable)
        report["optimizer_constructed"] = True
        phase, slowest = "training", 0.0
        for step in range(MAX_UPDATES):
            if not can_start_step(time.perf_counter(), deadline, reserve, slowest):
                report["stop_reason"] = "insufficient_budget_for_next_step_and_closing_reserve"
                break
            selected = schedule[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]
            entry = {"step": step + 1, "question_ids": [q["question_id"] for q in selected],
                     "optimizer_step_started": False, "optimizer_step_returned": False}
            report["step_attempts"].append(entry)
            record()
            began = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            prefixes = [encoded[q["prompt"]][0] for q in selected]
            targets = torch.tensor([encoded[q["prompt"]][1][q["label"]] for q in selected], dtype=torch.long)
            with sdpa_kernel(SDPBackend.MATH):
                logits = _forward_logits(model, tokenizer, prefixes, selected, features)
                loss = torch.nn.functional.cross_entropy(logits, targets)
                require(bool(torch.isfinite(loss)), "Non-finite full-vocabulary loss")
                loss.backward()
            require(binding_clear(model), "Temporary natural binding not cleared")
            require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for _, p in trainable)
                    and all(p.grad is None for _, p in frozen)
                    and all(p.grad is None for p in model.base_parameters), "Invalid gradient membership or finiteness")
            entry["full_vocabulary_ce"] = float(loss.detach())
            norms = {n: float(p.grad.double().norm()) for n, p in trainable}
            require(all(math.isfinite(v) for v in norms.values()), "Non-finite gradient norm")
            entry["gradient_norms"] = norms
            entry["optimizer_step_started"] = True
            optimizer.step()
            entry["optimizer_step_returned"] = True
            report["updates_completed"] += 1
            report["sample_slots"] += BATCH_SIZE
            require(all(bool(torch.isfinite(p).all()) for _, p in trainable), "Non-finite updated parameter")
            entry["seconds"] = time.perf_counter() - began
            slowest = max(slowest, entry["seconds"])
            del loss, logits
            record()
        phase = "natural_final"
        evaluate("final")
        phase = "save_and_verify"
        time_available()
        checkpoint_metadata = {"variant": "m", "training_recipe": STAGE, "engineering_only": "false",
            "updates": str(report["updates_completed"]), "source_aps_sha256": CHECKPOINT_SHA,
            "trainable_names": json.dumps(report["trainable_names"], separators=(",", ":")),
            "train_order_sha256": report["train_order_sha256"], "manifest_sha256": MANIFEST_SHA,
            "runner_sha256": source_sha}
        report["checkpoint"] = save_full_checkpoint(model, named, output.parent / "adapter.safetensors", checkpoint_metadata)
        report["checkpoint_metadata"] = checkpoint_metadata
        report["base_sha256_after"] = parameter_digest(model.base_parameters)
        report["frozen_adapter_sha256_after"] = parameter_digest(p for _, p in frozen)
        report["trainable_sha256_after"] = {n: parameter_digest([p]) for n, p in trainable}
        report["trainable_bytes_changed"] = {n: report["trainable_sha256_after"][n] != before_each[n] for n in before_each}
        report["all_adapter_finite"] = all(bool(torch.isfinite(p).all()) for _, p in named)
        report["endpoint_conditions"] = {
            "restoration": restore_allows_validation(report["restoration"]),
            "conversion": report["conversion_verified"], "updates": report["updates_completed"] == 1000,
            "slots": report["sample_slots"] == 4000,
            "initial_final_coverage": len(report["evaluations"]) == 2 and all(e["coverage_complete"] for e in report["evaluations"]),
            "base_frozen": report["base_sha256_before"] == report["base_sha256_after"],
            "other_adapters_frozen": report["frozen_adapter_sha256_before"] == report["frozen_adapter_sha256_after"],
            "frozen_gradients_absent": all(p.grad is None for _, p in frozen) and all(p.grad is None for p in model.base_parameters),
            "trainable_membership": [n for n, p in named if p.requires_grad] == report["trainable_names"],
            "checkpoint": report["checkpoint"]["loader_verified"] is True and report["checkpoint"]["count"] == 101,
            "finite": report["all_adapter_finite"], "binding_clear": binding_clear(model),
            "input_sources": all(file_sha256(ROOT / n) == h for n, h in {**pins, **feature_pins}.items()),
            "implementation": all(file_sha256(ROOT / n) == h for n, h in implementation.items()),
            "within_child_budget": time.perf_counter() <= deadline}
        report["run_completed"] = all(report["endpoint_conditions"].values())
        report["status"] = "complete" if report["run_completed"] else "stopped"
    except StopRun as exc:
        report.update(status="stopped", stop_reason=str(exc), run_completed=False)
    except Exception as exc:
        report.update(status="error", run_completed=False,
                      error={"phase": phase, "type": type(exc).__name__, "message": str(exc).replace(str(ROOT), ".")})
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if monitor is not None:
            monitor.close()
        report["last_phase"] = phase
        record()
        if time.perf_counter() > deadline:
            report["run_completed"] = False
            report["endpoint_conditions"]["within_child_budget"] = False
            record()
    print(json.dumps({"status": report["status"], "run_completed": report["run_completed"],
                      "updates_completed": report["updates_completed"], "seconds": time.perf_counter() - origin}), flush=True)
    return 1 if report["status"] == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
