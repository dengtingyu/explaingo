#!/usr/bin/env python3
"""One capped, fresh-adapter A0 value-token fit on the fixed 9+12 cells.

There is no generation, held-out evaluation, automatic retry or next arm.
The 300-second budget starts before heavy imports. SIGALRM is a soft deadline:
native operations can delay delivery, and any overrun is reported honestly.
"""
from __future__ import annotations

import time

PROGRAM_STARTED = time.perf_counter()

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
SEED, BATCH_SIZE, MAX_UPDATES = 42, 4, 180
LABEL_NAMES = ("empty", "black", "white")
LAYERS = [0, 3, 5, 8, 11, 14, 16, 19, 22, 24, 27]
SETS = (
    ("first_latin_nine", [[0, 1, 2], [1, 2, 0], [2, 0, 1]],
     [264, 223, 242], "cfa1560059fe8749cc0b326a85bf7758a4bbd3c0ccaf6f1339bf132fbaf0a4d3"),
    ("first_reciprocal_twelve", [[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]],
     [60, 79, 282, 41, 72, 288], "c3733feaf9f426daf9de686440799c5922eadcbda5765e1c1de3eba535131672"),
)


class BudgetExpired(TimeoutError):
    pass


class NonFiniteError(RuntimeError):
    pass


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def content_sha256(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fixed_cells(data, audit):
    """Validate exact membership and ordering without model-dependent selection."""
    pool = data["training_pool_questions"]
    require(content_sha256(pool) == audit["source"]["training_pool_content_sha256"]
            == data["hashes"]["question_lists"]["training_pool_questions"],
            "Training-pool content changed")
    require(content_sha256(data["positions"]) == data["hashes"]["positions"],
            "Position content changed")
    by_id = {q["question_id"]: q for q in pool}
    require(len(by_id) == len(pool), "Duplicate training-pool identity")
    questions, grids = [], []
    for name, expected_labels, points, expected_sha in SETS:
        candidate = audit[name]
        members = candidate["members"]
        require(content_sha256(members) == candidate["members_sha256"] == expected_sha,
                "Fixed candidate membership changed")
        require(candidate["label_matrix_boards_by_coordinates"] == expected_labels
                and candidate["point_indices"] == points, "Fixed grid changed")
        boards = candidate["position_ids"]
        width = len(points)
        require(len(members) == len(boards) * width, "Incomplete fixed grid")
        prompt_ids = [members[q]["prompt"] for q in range(width)]
        require(len(set(prompt_ids)) == width, "Duplicate query identity")
        ids = []
        for b, board in enumerate(boards):
            row = []
            for q, point in enumerate(points):
                member = members[b * width + q]
                require(by_id.get(member["question_id"]) == member and member["split"] == "train",
                        "Candidate is not an exact training-pool member")
                require(member["position_id"] == board and member["point_index"] == point
                        and member["label"] == expected_labels[b][q]
                        and member["prompt"] == prompt_ids[q]
                        and member["to_play"] == candidate["to_play"]
                        and member["ply"] == candidate["ply"]
                        and member["answer"] == {"occupancy": LABEL_NAMES[member["label"]]},
                        "Candidate label, order, prompt or answer changed")
                row.append(member["question_id"])
            ids.append(row)
        for group, prompt, point in zip(candidate["same_text_groups"], prompt_ids, points):
            require(group["point_index"] == point and group["prompt"] == prompt
                    and group["prompt_sha256"] == content_sha256(prompt),
                    "Prompt identity hash changed")
        grids.append({"name": name, "position_ids": list(boards), "query_ids": prompt_ids,
                      "point_indices": list(points), "labels": expected_labels,
                      "question_ids": ids, "members_sha256": expected_sha})
        questions.extend(members)
    require(len(questions) == 21 and len({q["question_id"] for q in questions}) == 21,
            "Expected 21 unique cells")
    require(Counter(q["label"] for q in questions) == {0: 7, 1: 7, 2: 7},
            "Combined class counts changed")
    require(not set(grids[0]["position_ids"]) & set(grids[1]["position_ids"])
            and not set(grids[0]["query_ids"]) & set(grids[1]["query_ids"]),
            "The two grids must have disjoint board and exact-prompt identities")
    return questions, grids


def parameter_digest(parameters):
    digest = hashlib.sha256()
    for parameter in parameters:
        array = parameter.detach().cpu().contiguous().numpy()
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/grounding-microfit/report.json")
    args = parser.parse_args()
    require(1 <= args.seconds <= 300, "Total budget must be 1..300 seconds")
    output = args.output.resolve()
    relative(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = output.parent / "adapter.safetensors"
    checkpoint_temp = output.parent / "adapter.partial.safetensors"
    temporary = output.with_suffix(".tmp")
    start_record = output.parent / "run-start.json"
    require(not any(p.exists() for p in (output, checkpoint, checkpoint_temp, temporary, start_record)),
            "Preserve the existing run; this script never resumes or overwrites it")
    # An exclusive, retained start record also prevents concurrent or crash retries.
    with start_record.open("x", encoding="utf-8") as stream:
        json.dump({"stage": "a0_value_token_microfit", "script_sha256": file_sha256(__file__),
                   "total_budget_seconds": args.seconds, "automatic_retry": False}, stream)
    deadline = PROGRAM_STARTED + args.seconds
    report = {
        "status": "running", "stage": "a0_value_token_microfit", "run_completed": False,
        "bounded_run_finalized": False, "stop_reason": None, "formal_grounding_acceptance": "not_assessed",
        "scope": "Training fit on 21 previously inspected cells only. Non-additive scores can reflect board-by-prompt memorization; neither generalization, geographical addressing nor playing strength is tested.",
        "device": "cpu", "threads": 4, "dtype": "float32", "sdpa_backend": "math",
        "wall_time_cap_seconds": args.seconds,
        "deadline_policy": "300 seconds includes imports, source validation, loading, evaluations, updates, optional checkpoint and reporting. Soft SIGALRM can be delayed by a native operation; actual elapsed time and any overrun are recorded.",
        "script": relative(__file__), "script_sha256": file_sha256(__file__),
        "config": {"seed": SEED, "fresh_adapters": True, "coordinate_interface": "A0 existing text",
                   "batch_size": BATCH_SIZE, "max_updates": MAX_UPDATES,
                   "optimizer": "AdamW", "learning_rate": 0.001, "warmup_updates": 0,
                   "gradient_clipping": None, "weight_decay": 0.0,
                   "loss": "Exactly one full-vocabulary causal value-token CE per question; the prefix ends before the occupancy value. All other tokens receive no supervision.",
                   "evaluation_batch_size": 1, "optional_evaluation_updates": [60, 120],
                   "minimum_final_eval_and_report_reserve_seconds": 30.0,
                   "injection_layers": LAYERS},
        "updates_completed": 0, "update_attempts": [], "evaluations": [],
        "actual_completed_sampling_counts": {}, "actual_attempted_sampling_counts": {},
        "checkpoint_status": "not_saved", "frozen_base_final_hash_status": "not_measured",
        "additive_reference": {"latin_accuracy_ceiling": 7, "reciprocal_accuracy_ceiling": 9,
                               "combined_accuracy_ceiling": 16, "cells": 21,
                               "three_label_ce_infimum": (9 * math.log(3) + 12 * math.log(2)) / 21,
                               "interpretation": "Mathematical reference only; margins and measured update-0 rounding floor remain necessary. No automatic criterion or next-arm decision."},
    }
    model = None
    phase = "imports"
    completed_counts, attempted_counts = Counter(), Counter()

    def record():
        report["total_seconds"] = time.perf_counter() - PROGRAM_STARTED
        report["budget_exceeded"] = report["total_seconds"] > args.seconds
        report["actual_completed_sampling_counts"] = dict(completed_counts)
        report["actual_attempted_sampling_counts"] = dict(attempted_counts)
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(output)

    def expired(_signum, _frame):
        raise BudgetExpired("All-inclusive soft wall-time deadline reached")

    signal.signal(signal.SIGALRM, expired)
    record()
    signal.setitimer(signal.ITIMER_REAL, max(0.001, deadline - time.perf_counter()))
    try:
        import numpy as np
        import torch
        from safetensors.torch import save_file
        from torch.nn import functional as F
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        sys.path.insert(0, str(ROOT / "src"))
        from goqueen.fusion import GoQueen
        from goqueen.grounding_data import occupancy_prompt, point_to_gtp
        from goqueen.grounding_eval import encode_prefix, _forward_logits, score_questions, summarize_rows
        from goqueen.grounding_interaction import analyze_grid, combine_disjoint

        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        phase = "source_validation"
        manifest_path = ROOT / "outputs/grounding-data/manifest.json"
        audit_path = ROOT / "outputs/grounding-conflict-audit/report.json"
        cfg_path = ROOT / "configs/mixed-train.json"
        data, audit, cfg = [json.loads(p.read_text()) for p in (manifest_path, audit_path, cfg_path)]
        require(audit["status"] == "complete" and file_sha256(manifest_path) == audit["source"]["manifest_sha256"],
                "Source manifest SHA256 changed")
        require(cfg["injection_layers"] == LAYERS and cfg["expert_width"] == 768 and cfg["adapter_heads"] == 8,
                "Fusion architecture configuration changed")
        questions, grids = fixed_cells(data, audit)
        positions = {p["position_id"]: p for p in data["positions"]}
        features, source_features = {}, {}
        for question in questions:
            require(question["point"] == point_to_gtp(question["point_index"])
                    and question["prompt"] == occupancy_prompt(question["to_play"], question["point"]),
                    "Canonical prompt or point index changed")
        for position_id in dict.fromkeys(q["position_id"] for q in questions):
            position = positions[position_id]
            path = ROOT / position["source_feature_file"]
            path_name, digest = relative(path), file_sha256(path)
            expected = audit["checked_source_features"][position_id]
            require(digest == position["feature_sha256"] == expected["sha256"]
                    and path_name == expected["file"] and position["to_play"] == expected["to_play"],
                    "Source feature identity changed")
            with np.load(path, allow_pickle=False) as saved:
                spatial = saved["input_spatial"]
                require(spatial.shape == (1, 22, 19, 19), "Unexpected input plane shape")
                own, other = spatial[0, 1].reshape(-1), spatial[0, 2].reshape(-1)
                require(np.isfinite(own).all() and np.isfinite(other).all()
                        and np.isin(own, [0, 1]).all() and np.isin(other, [0, 1]).all()
                        and not np.any((own != 0) & (other != 0)), "Invalid occupancy input planes")
                own_label = 1 if position["to_play"] == "B" else 2
                labels = np.where(own != 0, own_label, np.where(other != 0, 3 - own_label, 0)).astype(int).tolist()
                require(labels == position["absolute_board_labels"], "Source plane labels changed")
                layers = [saved[f"trunk_after_block{i}"] for i in range(11)]
                require(all(layer.shape == (361, 768) and layer.dtype == np.float32
                            and np.isfinite(layer).all() for layer in layers), "Invalid cached expert features")
                features[position_id] = torch.from_numpy(np.stack(layers))
            require(all(labels[q["point_index"]] == q["label"] for q in questions if q["position_id"] == position_id),
                    "Member labels disagree with source planes")
            source_features[position_id] = {"file": path_name, "sha256": digest,
                                           "board_id": "sha256:" + digest, "absolute_labels_checked": 361}
        for grid in grids:
            grid["board_ids"] = [source_features[p]["board_id"] for p in grid["position_ids"]]
            require(len(set(grid["board_ids"])) == len(grid["board_ids"]), "Repeated source feature identity")
        require(not set(grids[0]["board_ids"]) & set(grids[1]["board_ids"]), "Grid feature identities overlap")
        ids = [q["question_id"] for q in questions]
        sampler = random.Random(SEED)
        order = [sampler.randrange(len(questions)) for _ in range(MAX_UPDATES * BATCH_SIZE)]
        planned_ids = [ids[i] for i in order]
        completed_counts.update(dict.fromkeys(ids, 0))
        attempted_counts.update(dict.fromkeys(ids, 0))
        report.update(
            source={"manifest": relative(manifest_path), "manifest_sha256": file_sha256(manifest_path),
                    "conflict_audit": relative(audit_path), "conflict_audit_sha256": file_sha256(audit_path),
                    "fusion_config": relative(cfg_path), "fusion_config_sha256": file_sha256(cfg_path),
                    "fixed_questions_sha256": content_sha256(questions), "source_features": source_features},
            implementation_sha256={name: file_sha256(ROOT / name) for name in (
                "src/goqueen/fusion.py", "src/goqueen/grounding_data.py", "src/goqueen/grounding_eval.py", "src/goqueen/grounding_interaction.py", "src/goqueen/smoke_data.py")},
            fixed_questions=questions, fixed_grids=grids,
            sampling={"method": "Python Random(42).randrange(21), 720 independent slots with replacement; four successive slots per update",
                      "seed": SEED, "slot_indices": order, "question_ids": planned_ids,
                      "slot_indices_sha256": content_sha256(order), "question_ids_sha256": content_sha256(planned_ids),
                      "planned_counts": {q: planned_ids.count(q) for q in ids}})
        record()
        phase = "cached_model_loading"
        model_path = ROOT / "models/Qwen3-0.6B"
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        encoded = [encode_prefix(tokenizer, q["prompt"]) for q in questions]
        label_ids = encoded[0][1]
        require(all(item[1] == label_ids for item in encoded), "Label token identities changed")
        report["label_token_ids"] = list(label_ids)
        report["prefix_tokens"] = {q["question_id"]: len(item[0]) for q, item in zip(questions, encoded)}
        report["prefix_token_ids_sha256"] = content_sha256([item[0].tolist() for item in encoded])
        language = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                                                       dtype=torch.float32, attn_implementation="sdpa")
        language.gradient_checkpointing_disable()
        # Re-seed exactly at adapter creation, independently of loader internals.
        torch.manual_seed(SEED)
        report["config"]["adapter_seed_applied"] = "immediately before GoQueen construction"
        report["config_sha256"] = content_sha256(report["config"])
        report["local_model"] = {"directory": relative(model_path), "local_files_only": True,
                                 "config_sha256": file_sha256(model_path / "config.json")}
        model = GoQueen(language, LAYERS, 768, 8)
        named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        parameters = [p for _, p in named]
        require(all(not p.requires_grad and p.device.type == "cpu" and p.dtype == torch.float32
                    for p in model.base_parameters), "Base LM must be frozen CPU FP32")
        require(not {id(p) for p in parameters} & {id(p) for p in model.base_parameters}, "Optimizer contains base parameters")
        report["architecture"] = model.architecture()
        report["initial_adapter_parameter_sha256"] = parameter_digest(parameters)
        report["frozen_base_sha256_before"] = parameter_digest(model.base_parameters)
        require(all(value == 0.0 for value in model.gate_values()), "Fresh gates must start at zero")
        optimizer = torch.optim.AdamW(parameters, lr=0.001, weight_decay=0.0)
        report["loaded_seconds"] = time.perf_counter() - PROGRAM_STARTED

        def evaluate(name):
            nonlocal phase
            phase = "evaluation_" + name
            tick = time.perf_counter()
            evaluation = {"name": name, "updates_completed": report["updates_completed"],
                          "requested_questions": 21, "completed_questions": 0,
                          "coverage_complete": False, "batch_size": 1, "rows": [], "grids": []}
            report["evaluations"].append(evaluation)
            record()
            for question in questions:
                if time.perf_counter() >= deadline - 4.0:
                    break
                rows = score_questions(model, tokenizer, [question], features, batch_size=1, deadline=deadline - 4.0)
                if not rows:
                    break
                row = rows[0]
                if not all(math.isfinite(value) for value in row["label_log_probs"]):
                    raise NonFiniteError("Non-finite evaluation label log probability")
                evaluation["rows"].append(row)
                evaluation["completed_questions"] = len(evaluation["rows"])
                record()
            evaluation["seconds"] = time.perf_counter() - tick
            evaluation["coverage_complete"] = len(evaluation["rows"]) == 21
            evaluation["summary"] = summarize_rows(evaluation["rows"])
            lookup = {row["question_id"]: row for row in evaluation["rows"]}
            for grid in grids:
                grid_ids = [q for row in grid["question_ids"] for q in row]
                completed = sum(q in lookup for q in grid_ids)
                payload = {**grid, "completed_cells": completed, "coverage_complete": completed == len(grid_ids)}
                # Partial grids are explicitly absent, never padded with guessed scores.
                payload["scores"] = ([[lookup[q]["label_log_probs"] for q in row] for row in grid["question_ids"]]
                                     if payload["coverage_complete"] else None)
                payload["analysis"] = (analyze_grid(payload["scores"], grid["labels"], grid["board_ids"], grid["query_ids"])
                                       if payload["coverage_complete"] else None)
                evaluation["grids"].append(payload)
            evaluation["combined_analysis"] = (combine_disjoint([g["analysis"] for g in evaluation["grids"]])
                                               if evaluation["coverage_complete"] else None)
            record()
            return evaluation

        def gradient_norms():
            groups = {"gates": 0.0, "layer0_adapter": 0.0, "other_adapters": 0.0, "row_column_codes": 0.0}
            norms = []
            for name, parameter in named:
                require(parameter.grad is not None, "Missing adapter gradient")
                norm = float(torch.linalg.vector_norm(parameter.grad.detach(), dtype=torch.float64))
                if not math.isfinite(norm):
                    raise NonFiniteError("Non-finite adapter gradient norm")
                group = ("gates" if name.endswith(".alpha") else "row_column_codes" if name in ("rows.weight", "columns.weight")
                         else "layer0_adapter" if name.startswith("language_model.model.layers.0.adapter.") else "other_adapters")
                groups[group] = math.hypot(groups[group], norm)
                norms.append(norm)
            require(all(p.grad is None for p in model.base_parameters), "Frozen base received a gradient")
            total = math.hypot(*norms)
            if not math.isfinite(total):
                raise NonFiniteError("Non-finite global gradient norm")
            return {"global": total, **groups}

        initial = evaluate("initial")
        if not initial["coverage_complete"]:
            report.update(status="incomplete_budget", stop_reason="initial_evaluation_incomplete")
            return
        floor = {}
        for grid in initial["grids"]:
            scores = grid["scores"]
            floor[grid["name"]] = max(abs(scores[b][q][y] - scores[0][q][y])
                                           for b in range(len(scores)) for q in range(len(scores[0])) for y in range(3))
        report["initial_zero_gate_floor"] = {"max_abs_raw_label_logp_board_difference": floor,
                                              "board_independent_exactly": all(v == 0 for v in floor.values()),
                                              "scope": "Same prompt, single-cell batch; empirical update-0 floor, not a universal error bound."}
        evaluation_estimate = max(3.0, initial["seconds"] * 1.5)
        step_seconds = []
        with sdpa_kernel(SDPBackend.MATH):
            for step in range(MAX_UPDATES):
                reserve = max(30.0, evaluation_estimate + 10.0)
                step_estimate = max(1.3, max(step_seconds[-5:], default=0.0) * 1.35)
                remaining = deadline - time.perf_counter()
                if remaining < reserve + step_estimate:
                    report["stop_reason"] = "time_reserved_for_final_evaluation_and_report"
                    break
                phase = "optimization"
                tick = time.perf_counter()
                selected_indices = order[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]
                selected = [questions[i] for i in selected_indices]
                selected_ids = [q["question_id"] for q in selected]
                attempt = {"update": step + 1, "question_ids": selected_ids,
                           "optimizer_step_started": False, "optimizer_step_returned": False}
                report["update_attempts"].append(attempt)
                attempted_counts.update(selected_ids)
                record()
                model.train()
                optimizer.zero_grad(set_to_none=True)
                logits = _forward_logits(model, tokenizer, [encoded[i][0] for i in selected_indices], selected, features)
                if not bool(torch.isfinite(logits).all()):
                    raise NonFiniteError("Non-finite training logits")
                targets = torch.tensor([label_ids[q["label"]] for q in selected], dtype=torch.long)
                loss = F.cross_entropy(logits, targets)
                if not bool(torch.isfinite(loss)):
                    raise NonFiniteError("Non-finite value-token loss")
                attempt["full_vocabulary_value_ce"] = float(loss.detach())
                loss.backward()
                attempt["gradient_norms_before_step"] = gradient_norms()
                attempt["optimizer_step_started"] = True
                optimizer.step()
                attempt["optimizer_step_returned"] = True
                report["updates_completed"] = step + 1
                completed_counts.update(selected_ids)
                if not all(bool(torch.isfinite(p).all()) for p in parameters):
                    raise NonFiniteError("Non-finite adapter parameter after optimizer step")
                attempt["gates_after_step"] = model.gate_values()
                attempt["seconds"] = time.perf_counter() - tick
                step_seconds.append(attempt["seconds"])
                del logits, targets, loss
                record()
                if step + 1 in (60, 120):
                    if deadline - time.perf_counter() > reserve + evaluation_estimate + step_estimate + 3.0:
                        milestone = evaluate(f"update_{step + 1}")
                        if not milestone["coverage_complete"]:
                            report["stop_reason"] = "milestone_evaluation_incomplete_at_budget"
                            break
                        evaluation_estimate = max(evaluation_estimate, milestone["seconds"] * 1.5)
                    else:
                        report.setdefault("skipped_optional_evaluations", []).append(
                            {"update": step + 1, "reason": "preserve_final_evaluation_reserve"})
            else:
                report["stop_reason"] = "max_updates_reached"
        final = evaluate("final")
        report["bounded_run_finalized"] = final["coverage_complete"]
        report["run_completed"] = final["coverage_complete"] and report["updates_completed"] == MAX_UPDATES
        report["status"] = "complete" if report["run_completed"] else "incomplete_budget"
        if not final["coverage_complete"]:
            report["stop_reason"] = "final_evaluation_incomplete_at_budget"
        report["coverage"] = {"initial_questions": initial["completed_questions"], "final_questions": final["completed_questions"],
                              "planned_questions_per_evaluation": 21, "completed_updates": report["updates_completed"],
                              "planned_updates": MAX_UPDATES, "completed_training_slots": sum(completed_counts.values()),
                              "planned_training_slots": MAX_UPDATES * BATCH_SIZE}
        report["gates"] = model.gate_values()
        require(all(p.grad is None and not p.requires_grad for p in model.base_parameters), "Base freeze contract changed")
        report["frozen_base_gradients_absent"] = True
        phase = "optional_hash_and_checkpoint"
        if deadline - time.perf_counter() > 8.0:
            report["frozen_base_sha256_after"] = parameter_digest(model.base_parameters)
            report["frozen_base_final_hash_status"] = "measured"
            require(report["frozen_base_sha256_before"] == report["frozen_base_sha256_after"], "Frozen base weights changed")
        if final["coverage_complete"] and deadline - time.perf_counter() > 6.0:
            report["checkpoint_status"] = "saving"
            record()
            save_file(model.adapter_state_dict(), str(checkpoint_temp),
                      metadata={"stage": "a0_value_token_microfit", "updates": str(report["updates_completed"]),
                                "fixed_questions_sha256": report["source"]["fixed_questions_sha256"]})
            checkpoint_temp.replace(checkpoint)
            report["checkpoint_status"] = "saved"
            report["checkpoint"] = relative(checkpoint)
            report["checkpoint_sha256"] = file_sha256(checkpoint)
        elif not final["coverage_complete"]:
            report["checkpoint_status"] = "not_saved_incomplete_final_evaluation"
        else:
            report["checkpoint_status"] = "not_saved_to_preserve_report_budget"
    except BudgetExpired:
        report.update(status="incomplete_at_cap", run_completed=False, bounded_run_finalized=False,
                      stop_reason="soft_deadline", interrupted_phase=phase,
                      limitation="Native work may finish before the signal is delivered. An interrupted optimizer step is not assumed complete; recorded start/return flags distinguish it.")
    except NonFiniteError as exc:
        report.update(status="stopped_nonfinite", run_completed=False, bounded_run_finalized=False,
                      stop_reason="nonfinite", interrupted_phase=phase, limitation=str(exc))
    except Exception as exc:
        # Report only exception type; exceptions can contain private filesystem paths.
        report.update(status="failed", run_completed=False, bounded_run_finalized=False,
                      stop_reason="exception", interrupted_phase=phase, exception_type=type(exc).__name__)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        report["coverage"] = {"requested_questions_per_evaluation": 21,
                              "evaluations": [{"name": e["name"], "updates_completed": e["updates_completed"],
                                               "completed_questions": e["completed_questions"],
                                               "coverage_complete": e["coverage_complete"]} for e in report["evaluations"]],
                              "completed_updates": report["updates_completed"], "planned_updates": MAX_UPDATES,
                              "completed_training_slots": sum(completed_counts.values()),
                              "attempted_training_slots": sum(attempted_counts.values()),
                              "planned_training_slots": MAX_UPDATES * BATCH_SIZE}
        if report["checkpoint_status"] == "saving":
            report["checkpoint_status"] = "save_interrupted_or_hash_not_completed"
        record()
        print(json.dumps({"status": report["status"], "run_completed": report["run_completed"],
                          "updates_completed": report["updates_completed"], "stop_reason": report["stop_reason"],
                          "total_seconds": report["total_seconds"]}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
