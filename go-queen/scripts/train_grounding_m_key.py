#!/usr/bin/env python3
"""One opt-in M-Key factual fit; no automatic retry or follow-on run.

Uses the original M labels, inputs, sampler and optimizer. The only direct
formula change is a stronger live position copy in the key branch. The soft
300-second TOTAL includes loading, checks, saving and report delivery.
"""
from __future__ import annotations

import time
PROGRAM_STARTED = time.perf_counter()

import argparse
from collections import Counter
import json
import hashlib
import math
from pathlib import Path
import random
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_grounding_microfit import (SEED, BATCH_SIZE, MAX_UPDATES, LABEL_NAMES, LAYERS,
    BudgetExpired, NonFiniteError, file_sha256, content_sha256, relative, require,
    fixed_cells, parameter_digest)

TOLERANCE = 1e-5
ADDITIVE_CE_BOUND = 0.8669179411777301
REFERENCE_REPORT = ROOT / "outputs/grounding-microfit/report.json"
REFERENCE_REPORT_SHA256 = "23a0c4b4be9e5594bb06207f9acdc45b5ff93f1f0b64b3d0da4d786a815e7289"


class ReproductionMismatch(ValueError):
    pass


def expected_parameter_signature():
    """Original GoQueen registration order; verified against the original class."""
    suffixes = [("alpha", []), ("query_norm.weight", [1024]), ("query_norm.bias", [1024]),
                ("expert_norm.weight", [768]), ("expert_norm.bias", [768]),
                ("query.weight", [1024, 1024]), ("key.weight", [1024, 768]),
                ("value.weight", [1024, 768]), ("output.weight", [1024, 1024])]
    rows = [{"name": f"language_model.model.layers.{layer}.adapter.{name}",
             "shape": shape, "dtype": "torch.float32"}
            for layer in LAYERS for name, shape in suffixes]
    return rows + [{"name": name, "shape": [19, 768], "dtype": "torch.float32"}
                   for name in ("rows.weight", "columns.weight")]


def factual_indicators(rows):
    """Complete 21-cell factual metrics with explicitly guarded references."""
    require(len(rows) == 21 and len({r["question_id"] for r in rows}) == 21,
            "Indicators require all 21 unique factual cells")
    margins, ce = [], []
    for row in rows:
        values, gold = row["label_log_probs"], row["label"]
        require(len(values) == 3 and all(math.isfinite(v) for v in values)
                and type(gold) is int and 0 <= gold < 3, "Invalid factual scores")
        margins.append(values[gold] - max(v for y, v in enumerate(values) if y != gold))
        top = max(values)
        ce.append(top + math.log(math.fsum(math.exp(v-top) for v in values)) - values[gold])
    mean_ce = math.fsum(ce) / 21
    resolved = sum(v > TOLERANCE for v in margins)
    wrong = sum(v < -TOLERANCE for v in margins)
    accuracy_crossing = resolved > 16
    ce_crossing = mean_ce < ADDITIVE_CE_BOUND - TOLERANCE
    full_fit = resolved == 21
    return {"correct_above_margin_guard": resolved, "wrong_above_margin_guard": wrong,
            "unresolved": 21-resolved-wrong, "margins": margins, "full_fit": full_fit,
            "accuracy_crossing": accuracy_crossing, "ce_crossing": ce_crossing,
            "mean_three_label_ce": mean_ce,
            "category": "full_fit" if full_fit else "beyond_additive" if accuracy_crossing or ce_crossing
                        else "not_beyond_additive",
            "scope": "One factual training grid, not generalization or a failure-cause test."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/grounding-m-key/report.json")
    args = parser.parse_args()
    require(1 <= args.seconds <= 300, "Total budget must be 1..300 seconds")
    output = args.output.resolve()
    relative(output)
    require(output.is_relative_to(ROOT / "outputs") and not output.is_relative_to(REFERENCE_REPORT.parent)
            and not output.is_relative_to(ROOT / "outputs/grounding-prompt-control"),
            "M-Key output must be a new outputs directory outside the original M/P runs")
    require(output.suffix == ".json" and output.name.casefold() != "run-start.json",
            "Report output must have a .json suffix and must not use the retained run-start.json name")
    checkpoint = output.parent / "adapter.safetensors"
    checkpoint_temp = output.parent / "adapter.partial.safetensors"
    temporary = output.with_suffix(".tmp")
    start_record = output.parent / "run-start.json"
    script_copy = output.parent / "runner-at-run.py"
    artifact_paths = (output, checkpoint, checkpoint_temp, temporary, start_record, script_copy)
    require(len({p.resolve().as_posix().casefold() for p in artifact_paths}) == len(artifact_paths),
            "Report, temporary and retained artifact paths must all be distinct")
    require(not any(p.exists() for p in artifact_paths),
            "Preserve the existing run; this script never resumes or overwrites it")
    output.parent.mkdir(parents=True, exist_ok=True)
    # An exclusive, retained start record also prevents concurrent or crash retries.
    with start_record.open("x", encoding="utf-8") as stream:
        json.dump({"stage": "a0_m_key", "script_sha256": file_sha256(__file__),
                   "total_budget_seconds": args.seconds, "automatic_retry": False}, stream)
    with script_copy.open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    deadline = PROGRAM_STARTED + args.seconds
    report = {
        "status": "running", "stage": "a0_m_key", "run_completed": False,
        "bounded_run_finalized": False, "stop_reason": None, "formal_grounding_acceptance": "not_assessed",
        "scope": "One factual M-Key trajectory on the unchanged 21 M inputs. Shared position gradients can change values and other training dynamics. Any improvement is a total intervention contrast, not an exclusive addressing or failure-cause explanation; memorization and generalization are not distinguished.",
        "device": "cpu", "threads": 4, "dtype": "float32", "sdpa_backend": "math",
        "wall_time_cap_seconds": args.seconds,
        "deadline_policy": "300 seconds includes imports, source validation, loading, evaluations, updates, optional checkpoint and reporting. Soft SIGALRM can be delayed by a native operation; actual elapsed time and any overrun are recorded.",
        "script": relative(__file__), "script_sha256": file_sha256(__file__),
        "script_at_run_copy": relative(script_copy), "script_at_run_copy_sha256": file_sha256(script_copy),
        "config": {"seed": SEED, "fresh_adapters": True, "coordinate_interface": "A0 existing text",
                   "batch_size": BATCH_SIZE, "max_updates": MAX_UPDATES,
                   "optimizer": "AdamW", "learning_rate": 0.001, "betas": [0.9, 0.999], "eps": 1e-8,
                   "warmup_updates": 0,
                   "gradient_clipping": None, "weight_decay": 0.0,
                   "loss": "One full-vocabulary causal value-token CE against original factual labels; no gold value enters the prefix.",
                   "evaluation_batch_size": 1, "optional_evaluation_updates": [60, 120],
                   "minimum_final_eval_and_report_reserve_seconds": 30.0,
                   "injection_layers": LAYERS},
        "updates_completed": 0, "update_attempts": [], "evaluations": [],
        "actual_completed_sampling_counts": {}, "actual_attempted_sampling_counts": {},
        "checkpoint_status": "not_saved", "frozen_base_final_hash_status": "not_measured",
        "references": {"additive_correct_upper": 16, "additive_three_label_ce_lower": ADDITIVE_CE_BOUND,
                       "uniform_three_label_ce": math.log(3), "tolerance": TOLERANCE},
        "validity": {"pre_update": False, "endpoint": False, "checks": []},
        "final_m_key_endpoint": {"category": None, "indicators": None,
                                   "reason": "No valid complete final endpoint has been measured."},
    }
    model = None
    phase = "imports"
    completed_counts, attempted_counts = Counter(), Counter()
    factual_mapping = {}

    def compare(name, actual, expected):
        matched = actual == expected
        report["validity"]["checks"].append({"name": name, "matched": matched,
                                            "actual": actual, "expected": expected})
        if not matched:
            raise ReproductionMismatch(name)

    def record():
        report["total_seconds"] = time.perf_counter() - PROGRAM_STARTED
        report["budget_exceeded"] = report["total_seconds"] > args.seconds
        report["actual_completed_sampling_counts"] = dict(completed_counts)
        report["actual_attempted_sampling_counts"] = dict(attempted_counts)
        report["actual_completed_factual_label_counts"] = {name: sum(count for qid, count in completed_counts.items() if factual_mapping.get(qid) == y)
                                                         for y, name in enumerate(LABEL_NAMES)}
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
        from goqueen.m_key import MKeyGoQueen, POSITION_SCALE
        from goqueen.grounding_data import occupancy_prompt, point_to_gtp
        from goqueen.grounding_eval import encode_prefix, _forward_logits, score_questions, summarize_rows
        from goqueen.grounding_interaction import analyze_grid, combine_disjoint

        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        phase = "source_validation"
        compare("original_M_report_file_sha256", file_sha256(REFERENCE_REPORT), REFERENCE_REPORT_SHA256)
        reference = json.loads(REFERENCE_REPORT.read_text())
        require(reference["run_completed"] and reference["updates_completed"] == MAX_UPDATES,
                "Original M reference must be complete")
        compare("original_M_runner_source_sha256", file_sha256(ROOT / reference["script"]), reference["script_sha256"])
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
        compare("factual_questions_sha256", content_sha256(questions), reference["source"]["fixed_questions_sha256"])
        compare("factual_grids_sha256", content_sha256(grids), content_sha256(reference["fixed_grids"]))
        factual_mapping = {q["question_id"]: q["label"] for q in questions}
        ids = [q["question_id"] for q in questions]
        sampler = random.Random(SEED)
        order = [sampler.randrange(len(questions)) for _ in range(MAX_UPDATES * BATCH_SIZE)]
        planned_ids = [ids[i] for i in order]
        compare("ordered_sampling_slot_indices_sha256", content_sha256(order), reference["sampling"]["slot_indices_sha256"])
        compare("ordered_sampling_question_ids_sha256", content_sha256(planned_ids), reference["sampling"]["question_ids_sha256"])
        compare("exact_sampling_slot_indices", order == reference["sampling"]["slot_indices"], True)
        compare("exact_sampling_question_ids", planned_ids == reference["sampling"]["question_ids"], True)
        planned_factual_counts = Counter(factual_mapping[qid] for qid in planned_ids)
        compare("planned_factual_training_label_counts", dict(planned_factual_counts), {0: 223, 1: 241, 2: 256})
        completed_counts.update(dict.fromkeys(ids, 0))
        attempted_counts.update(dict.fromkeys(ids, 0))
        report.update(
            source={"manifest": relative(manifest_path), "manifest_sha256": file_sha256(manifest_path),
                    "conflict_audit": relative(audit_path), "conflict_audit_sha256": file_sha256(audit_path),
                    "fusion_config": relative(cfg_path), "fusion_config_sha256": file_sha256(cfg_path),
                    "fixed_questions_sha256": content_sha256(questions), "source_features": source_features},
            implementation_sha256={name: file_sha256(ROOT / name) for name in (
                "scripts/train_grounding_microfit.py", "src/goqueen/fusion.py", "src/goqueen/grounding_data.py", "src/goqueen/grounding_eval.py", "src/goqueen/grounding_interaction.py", "src/goqueen/smoke_data.py", "src/goqueen/m_key.py")},
            fixed_questions=questions, fixed_grids=grids,
            sampling={"method": "Python Random(42).randrange(21), 720 independent slots with replacement; four successive slots per update",
                      "seed": SEED, "slot_indices": order, "question_ids": planned_ids,
                      "slot_indices_sha256": content_sha256(order), "question_ids_sha256": content_sha256(planned_ids),
                      "planned_counts": {q: planned_ids.count(q) for q in ids},
                      "planned_factual_label_counts": {name: planned_factual_counts[i] for i, name in enumerate(LABEL_NAMES)}})
        for key, expected in reference["source"].items():
            compare("matched_source:" + key, report["source"][key], expected)
        for name, expected in reference["implementation_sha256"].items():
            compare("matched_implementation:" + name, report["implementation_sha256"][name], expected)
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
        compare("label_token_ids", list(label_ids), reference["label_token_ids"])
        compare("causal_prefix_token_ids_sha256", report["prefix_token_ids_sha256"], reference["prefix_token_ids_sha256"])
        compare("causal_prefix_lengths", report["prefix_tokens"], reference["prefix_tokens"])
        language = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                                                       dtype=torch.float32, attn_implementation="sdpa")
        language.gradient_checkpointing_disable()
        # Re-seed exactly at adapter creation, independently of loader internals.
        torch.manual_seed(SEED)
        report["config"]["adapter_seed_applied"] = "immediately before GoQueen construction"
        report["config_sha256"] = content_sha256(report["config"])
        report["local_model"] = {"directory": relative(model_path), "local_files_only": True,
                                 "config_sha256": file_sha256(model_path / "config.json")}
        compare("local_model_identity", report["local_model"], reference["local_model"])
        model = MKeyGoQueen(language, LAYERS, 768, 8)
        report["position_scale"] = POSITION_SCALE
        report["post_construction_cpu_rng_sha256"] = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
        report["rng_reference_limitation"] = "Original M construction RNG state was not saved; its value is not reconstructed."
        named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        parameters = [p for _, p in named]
        signature = [{"name": name, "shape": list(p.shape), "dtype": str(p.dtype)} for name,p in named]
        report["ordered_adapter_parameter_signature"] = signature
        compare("ordered_parameter_names_shapes_dtypes", signature, expected_parameter_signature())
        compare("adapter_tensor_count", len(named), 101)
        compare("adapter_parameter_count", sum(p.numel() for p in parameters), 40438795)
        require(all(not p.requires_grad and p.device.type == "cpu" and p.dtype == torch.float32
                    for p in model.base_parameters), "Base LM must be frozen CPU FP32")
        require(not {id(p) for p in parameters} & {id(p) for p in model.base_parameters}, "Optimizer contains base parameters")
        report["architecture"] = model.architecture()
        report["initial_adapter_parameter_sha256"] = parameter_digest(parameters)
        report["frozen_base_sha256_before"] = parameter_digest(model.base_parameters)
        compare("initial_adapter_parameter_sha256", report["initial_adapter_parameter_sha256"], reference["initial_adapter_parameter_sha256"])
        compare("initial_frozen_base_parameter_sha256", report["frozen_base_sha256_before"], reference["frozen_base_sha256_before"])
        require(all(value == 0.0 for value in model.gate_values()), "Fresh gates must start at zero")
        optimizer = torch.optim.AdamW(parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
        report["actual_optimizer_configuration"] = [{key: list(group[key]) if key == "betas" else group[key]
                                                     for key in ("lr", "betas", "eps", "weight_decay", "amsgrad", "maximize", "foreach", "capturable", "differentiable", "fused")}
                                                    for group in optimizer.param_groups]
        for key in ("seed", "fresh_adapters", "coordinate_interface", "batch_size", "max_updates", "optimizer",
                    "learning_rate", "warmup_updates", "gradient_clipping", "weight_decay", "evaluation_batch_size",
                    "optional_evaluation_updates", "minimum_final_eval_and_report_reserve_seconds", "injection_layers", "adapter_seed_applied"):
            compare("matched_regime:" + key, report["config"][key], reference["config"][key])
        report["loaded_seconds"] = time.perf_counter() - PROGRAM_STARTED

        def evaluate(name):
            nonlocal phase
            phase = "evaluation_" + name
            tick = time.perf_counter()
            evaluation = {"name": name, "updates_completed": report["updates_completed"],
                          "requested_questions": 21, "completed_questions": 0,
                          "coverage_complete": False, "batch_size": 1, "rows": [], "grids": [], "combined_analysis": None,
                          "descriptive_indicators": None}
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
            if evaluation["coverage_complete"]:
                evaluation["combined_analysis"] = combine_disjoint([g["analysis"] for g in evaluation["grids"]])
                evaluation["descriptive_indicators"] = factual_indicators(evaluation["rows"])
                evaluation["descriptive_indicators"].pop("category")
            # Descriptive scale telemetry uses the same feature tensors without
            # another language-model forward; it is not a probe or new branch.
            with torch.no_grad():
                position = model.rows(model.row_indices) + model.columns(model.column_indices)
                scaled_rms = float((POSITION_SCALE * position).square().mean().sqrt())
                normalized = []
                for i, layer in enumerate(model.injected_layers):
                    values = [float(layer.adapter.expert_norm(f[i] + position).square().mean().sqrt())
                              for f in features.values()]
                    normalized.append({"layer": model.layer_indices[i], "per_board_rms": values})
                evaluation["position_scale_descriptive"] = {"scaled_position_rms": scaled_rms,
                    "normalized_expert_per_layer": normalized,
                    "interpretation": "Realized scale only, not attenuation or functional contribution evidence."}
                if not math.isfinite(scaled_rms) or not all(math.isfinite(v) for row in normalized for v in row["per_board_rms"]):
                    raise NonFiniteError("Non-finite position scale telemetry")
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
            floor[grid["name"]] = max(max(scores[b][q][y] for b in range(len(scores)))
                                     - min(scores[b][q][y] for b in range(len(scores)))
                                     for q in range(len(scores[0])) for y in range(3))
        report["initial_zero_gate_floor"] = {"max_abs_raw_label_logp_board_difference": floor,
                                              "board_independent_exactly": all(v == 0 for v in floor.values()),
                                              "within_declared_tolerance": all(v <= TOLERANCE for v in floor.values()),
                                              "tolerance": TOLERANCE,
                                              "scope": "Maximum same-prompt pairwise board difference, single-cell batches; measured floor, not a universal error bound."}
        old_initials = [e for e in reference["evaluations"] if e["name"] == "initial"]
        require(len(old_initials) == 1 and old_initials[0]["coverage_complete"]
                and old_initials[0]["updates_completed"] == 0, "Original initial evaluation is incomplete")
        old_lookup = {row["question_id"]: row for row in old_initials[0]["rows"]}
        compare("initial_raw_score_question_id_coverage", sorted(row["question_id"] for row in initial["rows"]), sorted(old_lookup))
        delta = 0.0
        same_choices = True
        for row in initial["rows"]:
            old = old_lookup[row["question_id"]]
            compare("initial_factual_identity:" + row["question_id"],
                    {key: row[key] for key in ("label", "position_id", "point_index", "prefix_tokens", "label_token_ids")},
                    {key: old[key] for key in ("label", "position_id", "point_index", "prefix_tokens", "label_token_ids")})
            require(len(old["label_log_probs"]) == 3 and all(math.isfinite(v) for v in old["label_log_probs"]),
                    "Original initial label log probabilities are invalid")
            delta = max(delta, *(abs(a - b) for a, b in zip(row["label_log_probs"], old["label_log_probs"])))
            same_choices = same_choices and row["forced_choice_label"] == old["forced_choice_label"] \
                           and row["unconstrained_token_id"] == old["unconstrained_token_id"]
        report["initial_reproduction_comparison"] = {"tolerance_nats": TOLERANCE,
                                                    "max_abs_raw_label_logp_difference": delta,
                                                    "same_forced_and_full_vocabulary_choices": same_choices,
                                                    "reference_report": relative(REFERENCE_REPORT),
                                                    "reference_report_sha256": REFERENCE_REPORT_SHA256}
        compare("initial_raw_scores_within_declared_tolerance", delta <= TOLERANCE, True)
        compare("initial_choices_match_reference", same_choices, True)
        compare("initial_board_floor_within_declared_tolerance", all(v <= TOLERANCE for v in floor.values()), True)
        report["validity"]["pre_update"] = True
        record()
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
                selected_factual_labels = [factual_mapping[qid] for qid in selected_ids]
                attempt = {"update": step + 1, "question_ids": selected_ids,
                           "factual_target_labels": selected_factual_labels,
                           "optimizer_step_started": False, "optimizer_step_returned": False}
                report["update_attempts"].append(attempt)
                attempted_counts.update(selected_ids)
                record()
                model.train()
                optimizer.zero_grad(set_to_none=True)
                logits = _forward_logits(model, tokenizer, [encoded[i][0] for i in selected_indices], selected, features)
                if not bool(torch.isfinite(logits).all()):
                    raise NonFiniteError("Non-finite training logits")
                targets = torch.tensor([label_ids[label] for label in selected_factual_labels], dtype=torch.long)
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
        phase = "final_source_and_manifest_validity"
        compare("factual_questions_sha256_after", content_sha256(questions), reference["source"]["fixed_questions_sha256"])
        compare("original_M_report_unchanged", file_sha256(REFERENCE_REPORT), REFERENCE_REPORT_SHA256)
        for key, path in (("manifest_sha256", manifest_path), ("conflict_audit_sha256", audit_path), ("fusion_config_sha256", cfg_path)):
            compare("source_file_unchanged:" + key, file_sha256(path), report["source"][key])
        for position_id, source in source_features.items():
            compare("feature_file_unchanged:" + position_id, file_sha256(ROOT / source["file"]), source["sha256"])
        for name, expected in report["implementation_sha256"].items():
            compare("implementation_file_unchanged:" + name, file_sha256(ROOT / name), expected)
        compare("runner_at_run_copy", file_sha256(script_copy), report["script_sha256"])
        compare("runner_source_unchanged", file_sha256(__file__), report["script_sha256"])
        report["source_validity_after"] = True
        phase = "final_frozen_base_hash"
        if deadline - time.perf_counter() > 8.0:
            report["frozen_base_sha256_after"] = parameter_digest(model.base_parameters)
            report["frozen_base_final_hash_status"] = "measured"
            compare("final_frozen_base_parameter_sha256", report["frozen_base_sha256_after"], reference["frozen_base_sha256_after"])
            compare("frozen_base_before_after_equality", report["frozen_base_sha256_after"], report["frozen_base_sha256_before"])
        else:
            report["frozen_base_final_hash_status"] = "not_measured_to_preserve_report_budget"
        phase = "optional_checkpoint"
        if final["coverage_complete"] and deadline - time.perf_counter() > 6.0:
            report["checkpoint_status"] = "saving"
            record()
            save_file(model.adapter_state_dict(), str(checkpoint_temp),
                      metadata={"stage": "a0_m_key", "variant": "m_key", "position_scale": repr(POSITION_SCALE),
                                "updates": str(report["updates_completed"]),
                                "fixed_questions_sha256": report["source"]["fixed_questions_sha256"]})
            checkpoint_temp.replace(checkpoint)
            report["checkpoint_status"] = "saved"
            report["checkpoint"] = relative(checkpoint)
            report["checkpoint_sha256"] = file_sha256(checkpoint)
        elif not final["coverage_complete"]:
            report["checkpoint_status"] = "not_saved_incomplete_final_evaluation"
        else:
            report["checkpoint_status"] = "not_saved_to_preserve_report_budget"
        report["validity"]["endpoint"] = (report["run_completed"] and final["coverage_complete"]
                                            and report["validity"]["pre_update"]
                                            and report.get("source_validity_after", False)
                                            and report["frozen_base_final_hash_status"] == "measured"
                                            and report.get("frozen_base_gradients_absent", False)
                                            and all(check["matched"] for check in report["validity"]["checks"]))
        report["final_m_key_endpoint"]["indicators"] = final["descriptive_indicators"]
        report["final_m_key_endpoint"]["interpretation"] = "This one M-Key trajectory is a total intervention contrast. Shared position gradients change value trajectories as well as address geometry. Neither success nor finite failure exclusively identifies the original M failure cause, architecture addressability or generalization."
        if report["run_completed"] and report["validity"]["endpoint"] and final["coverage_complete"]:
            report["final_m_key_endpoint"].update(factual_indicators(final["rows"]))
            report["final_m_key_endpoint"]["reason"] = "Valid matched reproduction, 180 completed updates and all 21 final cells."
        else:
            report["final_m_key_endpoint"]["reason"] = "No endpoint category: incomplete planned update/final coverage or missing final validity. Measured factual metrics and indicators remain descriptive."
    except BudgetExpired:
        report.update(status="incomplete_at_cap", run_completed=False, bounded_run_finalized=False,
                      stop_reason="soft_deadline", interrupted_phase=phase,
                      limitation="Native work may finish before the signal is delivered. An interrupted optimizer step is not assumed complete; recorded start/return flags distinguish it.")
    except (NonFiniteError, FloatingPointError) as exc:
        report.update(status="stopped_nonfinite", run_completed=False, bounded_run_finalized=False,
                      stop_reason="nonfinite", interrupted_phase=phase, limitation=str(exc))
    except ReproductionMismatch as exc:
        report.update(status="invalid_reproduction", run_completed=False, bounded_run_finalized=False,
                      stop_reason="reproduction_mismatch", interrupted_phase=phase, mismatched_check=str(exc))
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
        if report["status"] in ("incomplete_at_cap", "stopped_nonfinite", "invalid_reproduction", "failed"):
            report["validity"]["endpoint"] = False
            report["final_m_key_endpoint"]["category"] = None
        if time.perf_counter() > deadline:
            report["validity"]["endpoint"] = False
            report["final_m_key_endpoint"]["category"] = None
            report["final_m_key_endpoint"]["reason"] = "All-inclusive soft budget exceeded; any measured scores remain descriptive only."
        record()
        # A final JSON write itself belongs to the budget. If it crosses the
        # soft cap, durably replace any endpoint category with an invalid one.
        if time.perf_counter() > deadline:
            report["validity"]["endpoint"] = False
            report["final_m_key_endpoint"]["category"] = None
            report["final_m_key_endpoint"]["reason"] = "All-inclusive soft budget exceeded during final report delivery; measured scores are descriptive only."
            record()
        print(json.dumps({"status": report["status"], "run_completed": report["run_completed"],
                          "updates_completed": report["updates_completed"], "stop_reason": report["stop_reason"],
                          "total_seconds": report["total_seconds"]}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
