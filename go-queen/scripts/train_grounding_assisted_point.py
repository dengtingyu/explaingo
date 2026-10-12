#!/usr/bin/env python3
"""One bounded M-APS factual fit on the fixed 21 cells; no automatic retry or follow-on run.

Everything except routing matches the original M trajectory: fresh seed42
adapters, the Random(42) 720-slot sampler, batch 4, at most 180 AdamW updates,
one full-vocabulary value-token CE per cell, and batch-1 scoring at 0/60/120/180.
Each adapter may attend only to the queried point, bound per actual forward
from the canonical coordinate; labels feed only the loss and the scores.
The soft 300-second TOTAL covers imports, loading, checks, training,
evaluation, hashing, saving and report delivery. The external launcher records
the OS exit, any signal and the actual wall time.
"""
from __future__ import annotations

import time
PROGRAM_STARTED = time.perf_counter()

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import random
import re
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_grounding_microfit import (SEED, BATCH_SIZE, MAX_UPDATES, LABEL_NAMES, LAYERS,
    BudgetExpired, NonFiniteError, file_sha256, content_sha256, relative, require,
    fixed_cells, parameter_digest)
from train_grounding_m_key import (TOLERANCE, ADDITIVE_CE_BOUND, REFERENCE_REPORT, REFERENCE_REPORT_SHA256,
    ReproductionMismatch, expected_parameter_signature, factual_indicators)

STAGE = "a0_assisted_point"
VARIANT = "m_assisted_point"
ROUTING_CONTRACT = "single_point_bool_mask_v1"
DEFAULT_OUTPUT = ROOT / "outputs/grounding-assisted-point/report.json"
PROTECTED_DIRECTORIES = ("grounding-microfit", "grounding-prompt-control", "grounding-m-key",
                         "assisted-point-engineering", "assisted-point-unit-check")
# Child artifacts plus the external launcher's process records share the output directory.
RESERVED_NAMES = ("run-start.json", "runner-at-run.py", "adapter.safetensors", "adapter.partial.safetensors",
                  "process-start.json", "process-completion.json")
INACTIVE_SUFFIXES = (".query_norm.weight", ".query_norm.bias", ".query.weight", ".key.weight")
INACTIVE_TENSORS, INACTIVE_VALUES = 44, 20207616
REQUIRED_IMPLEMENTATION = (
    "scripts/train_grounding_assisted_point.py", "scripts/train_grounding_microfit.py",
    "scripts/train_grounding_m_key.py", "src/goqueen/fusion.py", "src/goqueen/assisted_point.py",
    "src/goqueen/assisted_point_eval.py", "src/goqueen/checkpoints.py", "src/goqueen/m_key.py",
    "src/goqueen/grounding_data.py", "src/goqueen/grounding_eval.py",
    "src/goqueen/grounding_interaction.py", "src/goqueen/smoke_data.py")
EXPECTED_OPTIMIZER = {"lr": 0.001, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0,
                      "amsgrad": False, "maximize": False}


class ContractViolation(ValueError):
    pass


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    require(1 <= args.seconds <= 300, "Total budget must be 1..300 seconds")
    return args


def artifact_paths(output):
    """Validate a new dedicated report path; return every retained child artifact path."""
    output = Path(output).resolve()
    outputs = (ROOT / "outputs").resolve()
    require(output.is_relative_to(outputs) and output.parent != outputs,
            "APS report must sit in a dedicated directory under outputs/")
    require(not any(output.is_relative_to(outputs / name) for name in PROTECTED_DIRECTORIES),
            "APS output must not enter the original M, P, M-Key or APS engineering directories")
    require(output.suffix == ".json" and output.name.casefold() not in RESERVED_NAMES,
            "Report must be a .json file that does not alias a retained child or launcher artifact")
    parent = output.parent
    paths = {"report": output, "temporary": output.with_suffix(".tmp"),
             "start": parent / "run-start.json", "runner_copy": parent / "runner-at-run.py",
             "checkpoint": parent / "adapter.safetensors",
             "checkpoint_temp": parent / "adapter.partial.safetensors"}
    require(len({p.as_posix().casefold() for p in paths.values()}) == len(paths),
            "Report, temporary and retained artifact paths must all be distinct")
    require(not any(p.exists() for p in paths.values()),
            "Preserve the existing run; this script never resumes or overwrites it")
    return paths


def routed_metadata(question):
    """Label-free copy for forward routing: coordinate fields stay, answers never enter."""
    return {key: value for key, value in question.items() if key not in ("label", "answer")}


def source_features_digest(source_features):
    """The APS loader's source_features_sha256, defined exactly as in the engineering check."""
    payload = sorted(({"file": entry["file"], "sha256": entry["sha256"]}
                      for entry in source_features.values()), key=lambda item: item["file"])
    return content_sha256(payload), payload


def checkpoint_metadata(*, updates, manifest_sha256, questions_sha256, source_features_sha256):
    """Fit metadata satisfying goqueen.checkpoints' APS contract; never engineering-only weights."""
    require(type(updates) is int and 0 <= updates <= MAX_UPDATES, "Invalid completed update count")
    for digest in (manifest_sha256, questions_sha256, source_features_sha256):
        require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
                "Checkpoint digests must be lowercase SHA-256 hex")
    return {"variant": VARIANT, "routing_contract": ROUTING_CONTRACT, "engineering_only": "false",
            "stage": STAGE, "updates": str(updates), "source_manifest_sha256": manifest_sha256,
            "questions_sha256": questions_sha256, "source_features_sha256": source_features_sha256,
            "scope": "Adapters from one bounded factual APS trajectory; not engineering weights"}


def imported_first_party(modules=None):
    """Hashes of this runner and every imported project file under src/ or scripts/."""
    root = ROOT.resolve()
    paths = {Path(__file__).resolve()}
    for module in tuple((sys.modules if modules is None else modules).values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix == ".py" and (path.is_relative_to(root / "src") or path.is_relative_to(root / "scripts")):
            paths.add(path)
    return {relative(path): file_sha256(path) for path in sorted(paths)}


def inactive_parameters(named):
    """The 44 tensors expected to receive exactly-zero gradients under single-point routing."""
    return [(name, parameter) for name, parameter in named if name.endswith(INACTIVE_SUFFIXES)]


def require_exact_zero_gradients(inactive):
    """Exact zeros only (-0.0 counts as zero); a tolerance would hide a broken mask."""
    for name, parameter in inactive:
        if parameter.grad is None or not bool(parameter.grad.eq(0).all()):
            raise ContractViolation("Inactive APS tensor received a missing or nonzero gradient: " + name)


def optimizer_contract(optimizer, named):
    """One group holding exactly the 101 named tensors in order, fresh state, the fixed settings."""
    groups = optimizer.param_groups
    if len(groups) != 1 or [id(p) for p in groups[0]["params"]] != [id(p) for _, p in named]:
        raise ContractViolation("Optimizer must hold exactly the named adapter tensors in order")
    if len(optimizer.state):
        raise ContractViolation("Optimizer state must be fresh before the first update")
    group = groups[0]
    settings = {key: list(group[key]) if key == "betas" else group.get(key)
                for key in ("lr", "betas", "eps", "weight_decay", "amsgrad", "maximize",
                            "foreach", "capturable", "differentiable", "fused")}
    if any(settings[key] != value for key, value in EXPECTED_OPTIMIZER.items()):
        raise ContractViolation("Optimizer settings differ from the fixed AdamW configuration")
    return settings


def binding_clear(model):
    return (not model._bound and all(layer.expert_features is None and layer.adapter.point_mask is None
                                     and layer.adapter.point_indices is None
                                     for layer in model.injected_layers))


class RoutingMonitor:
    """Check every actual adapter call's bound mask against the expected canonical points.

    Forward pre-hooks only read the plain runtime attributes set by
    AssistedPointGoQueen.bind_expert; they register no tensor and change nothing.
    """

    def __init__(self, model):
        self.model = model
        self.adapters = [layer.adapter for layer in model.injected_layers]
        self.expected, self.mask, self.calls = None, None, 0
        self.counts = {"training": 0, "evaluation": 0}
        self.adapter_calls = self.routed_samples = 0
        self.handles = [adapter.register_forward_pre_hook(self._check) for adapter in self.adapters]

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _check(self, module, args):
        import torch
        if self.expected is None:
            raise ContractViolation("An APS adapter ran outside a checked routing expectation")
        batch, expected = args[0].shape[0], self.expected
        mask, points = module.point_mask, module.point_indices
        if mask is None or points is None or batch != len(expected):
            raise ContractViolation("APS adapter binding does not match the expected actual batch")
        if mask.dtype != torch.bool or tuple(mask.shape) != (batch, 1, 1, 361):
            raise ContractViolation("APS mask is not a [B,1,1,361] bool tensor")
        rows = mask.reshape(batch, 361)
        if rows.sum(dim=1).tolist() != [1] * batch or rows.nonzero()[:, 1].tolist() != expected:
            raise ContractViolation("APS mask does not allow exactly each canonical point")
        if points.tolist() != expected:
            raise ContractViolation("APS bound point indices differ from the canonical points")
        if self.mask is None:
            self.mask = mask
        elif mask is not self.mask:
            raise ContractViolation("APS adapters in one forward did not share one mask")
        self.calls += 1

    @contextmanager
    def expect(self, points, kind):
        if self.expected is not None or kind not in self.counts:
            raise ContractViolation("Invalid or nested APS routing expectation")
        self.expected, self.mask, self.calls = [int(p) for p in points], None, 0
        try:
            yield
        finally:
            expected, calls = self.expected, self.calls
            self.expected, self.mask = None, None
        # Reached only when the routed forward returned normally.
        if calls != len(self.adapters):
            raise ContractViolation(f"Expected {len(self.adapters)} routed adapter calls, observed {calls}")
        if not binding_clear(self.model):
            raise ContractViolation("APS binding state survived the routed forward")
        self.counts[kind] += 1
        self.adapter_calls += calls
        self.routed_samples += len(expected)


def endpoint_conditions(report):
    """Child-side conditions for a valid complete endpoint; the launcher adds OS exit and wall time."""
    validity = report.get("validity") or {}
    contract = report.get("assisted_point_contract") or {}
    finals = [e for e in report.get("evaluations") or [] if isinstance(e, dict) and e.get("name") == "final"]
    final = finals[0] if len(finals) == 1 else {}
    planned = [q.get("question_id") for q in report.get("fixed_questions") or [] if isinstance(q, dict)]
    scored = [r.get("question_id") for r in final.get("rows") or [] if isinstance(r, dict)]
    attempts = report.get("update_attempts") or []
    checks = validity.get("checks") or []
    before, after = report.get("frozen_base_sha256_before"), report.get("frozen_base_sha256_after")
    return {
        "pre_update_reproduction": validity.get("pre_update") is True,
        "all_recorded_checks_matched": bool(checks) and all(c.get("matched") is True for c in checks),
        "run_completed": report.get("run_completed") is True,
        "updates_180_returned": (report.get("updates_completed") == MAX_UPDATES and len(attempts) == MAX_UPDATES
                                 and all(a.get("optimizer_step_returned") is True for a in attempts)),
        "final_21_exact": (final.get("coverage_complete") is True and final.get("updates_completed") == MAX_UPDATES
                           and len(planned) == len(set(planned)) == 21
                           and len(scored) == len(set(scored)) == 21 and set(scored) == set(planned)),
        "finite_final_parameters": report.get("final_parameters_finite") is True,
        "optimizer_contract": contract.get("optimizer_contract_verified") is True,
        "routing_verified": contract.get("routing_verified") is True,
        "inactive_zero_gradients_every_update": contract.get("inactive_zero_gradient_updates") == MAX_UPDATES,
        "inactive_44_bytes_unchanged": contract.get("inactive_44_bytes_unchanged") is True,
        "binding_clear_at_end": contract.get("binding_clear_at_end") is True,
        "source_validity_after": report.get("source_validity_after") is True,
        "frozen_base_hash_measured_equal": (report.get("frozen_base_final_hash_status") == "measured"
                                            and isinstance(before, str)
                                            and re.fullmatch(r"[0-9a-f]{64}", before) is not None
                                            and before == after),
        "frozen_base_gradients_absent": report.get("frozen_base_gradients_absent") is True,
        "checkpoint_saved_and_loader_verified": report.get("checkpoint_status") == "saved_and_loader_verified",
    }


def main(argv=None):
    args = parse_arguments(argv)
    paths = artifact_paths(args.output)
    output, temporary = paths["report"], paths["temporary"]
    output.parent.mkdir(parents=True, exist_ok=True)
    # An exclusive, retained start record also prevents concurrent or crash retries.
    with paths["start"].open("x", encoding="utf-8") as stream:
        json.dump({"stage": STAGE, "script_sha256": file_sha256(__file__),
                   "total_budget_seconds": args.seconds, "automatic_retry": False}, stream)
    with paths["runner_copy"].open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    deadline = PROGRAM_STARTED + args.seconds
    report = {
        "status": "running", "stage": STAGE, "variant": VARIANT, "routing_contract": ROUTING_CONTRACT,
        "run_completed": False, "bounded_run_finalized": False, "stop_reason": None,
        "formal_grounding_acceptance": "not_assessed",
        "scope": "One factual APS trajectory on the unchanged 21 M inputs with externally routed single-point attention. A valid result concerns only this assisted-selection configuration on these 21 training facts; it establishes neither natural coordinate addressing, a unique cause of the earlier M failure, generic board reading nor generalization.",
        "device": "cpu", "threads": 4, "dtype": "float32", "sdpa_backend": "math",
        "wall_time_cap_seconds": args.seconds,
        "deadline_policy": "300 seconds includes imports, source validation, loading, evaluations, updates, hashing, checkpoint and reporting. Soft SIGALRM can be delayed by a native operation; actual elapsed time and any overrun are recorded.",
        "script": relative(__file__), "script_sha256": file_sha256(__file__),
        "script_at_run_copy": relative(paths["runner_copy"]),
        "script_at_run_copy_sha256": file_sha256(paths["runner_copy"]),
        "config": {"seed": SEED, "fresh_adapters": True, "coordinate_interface": "A0 existing text",
                   "batch_size": BATCH_SIZE, "max_updates": MAX_UPDATES,
                   "optimizer": "AdamW", "learning_rate": 0.001, "betas": [0.9, 0.999], "eps": 1e-8,
                   "warmup_updates": 0, "gradient_clipping": None, "weight_decay": 0.0,
                   "loss": "One full-vocabulary causal value-token CE against original factual labels; no gold value enters the prefix.",
                   "evaluation_batch_size": 1, "optional_evaluation_updates": [60, 120],
                   "minimum_final_eval_and_report_reserve_seconds": 30.0,
                   "injection_layers": LAYERS,
                   "routing": "Canonical point_index of each actual sample, bound per forward; labels never routed"},
        "updates_completed": 0, "update_attempts": [], "evaluations": [],
        "actual_completed_sampling_counts": {}, "actual_attempted_sampling_counts": {},
        "checkpoint_status": "not_saved", "frozen_base_final_hash_status": "not_measured",
        "references": {"additive_correct_upper": 16, "additive_three_label_ce_lower": ADDITIVE_CE_BOUND,
                       "uniform_three_label_ce": math.log(3), "tolerance": TOLERANCE},
        "validity": {"pre_update": False, "endpoint": False, "checks": [], "endpoint_conditions": None},
        "assisted_point_contract": {
            "optimizer_contract_verified": False, "routing_verified": False,
            "routed_forwards": {"training": 0, "evaluation": 0}, "adapter_calls_checked": 0,
            "routed_samples_checked": 0, "mixed_length_training_forwards": 0,
            "inactive_zero_gradient_updates": 0, "inactive_44_bytes_unchanged": None,
            "binding_clear_at_end": None, "violation": None,
            "scope": "Bound masks and point indices are checked on every actual adapter call; byte invariance of the 44 tensors does not by itself show that the routed point was correct."},
        "final_assisted_point_endpoint": {"valid": False, "category": None, "indicators": None,
                                          "reason": "No valid complete final endpoint has been measured."},
    }
    model = monitor = None
    phase = "imports"
    completed_counts, attempted_counts = Counter(), Counter()
    factual_mapping = {}
    contract = report["assisted_point_contract"]

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
        from goqueen.assisted_point import (AssistedPointGoQueen, VARIANT as CORE_VARIANT,
                                            ROUTING_CONTRACT as CORE_CONTRACT)
        from goqueen.assisted_point_eval import _forward_logits, score_questions
        from goqueen.checkpoints import APS_ROUTING_CONTRACT, load_adapter_checkpoint
        from goqueen.grounding_data import occupancy_prompt, point_to_gtp
        from goqueen.grounding_eval import encode_prefix, summarize_rows
        from goqueen.grounding_interaction import analyze_grid, combine_disjoint

        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        phase = "source_validation"
        compare("aps_variant_and_routing_contract", [CORE_VARIANT, CORE_CONTRACT, APS_ROUTING_CONTRACT],
                [VARIANT, ROUTING_CONTRACT, ROUTING_CONTRACT])
        pins = imported_first_party()
        report["implementation_sha256"] = pins
        compare("required_implementation_pinned", sorted(set(REQUIRED_IMPLEMENTATION) - set(pins)), [])
        compare("original_M_report_file_sha256", file_sha256(REFERENCE_REPORT), REFERENCE_REPORT_SHA256)
        reference = json.loads(REFERENCE_REPORT.read_text())
        require(reference["run_completed"] and reference["updates_completed"] == MAX_UPDATES,
                "Original M reference must be complete")
        compare("original_M_runner_source_sha256", file_sha256(ROOT / reference["script"]), reference["script_sha256"])
        for name, expected in reference["implementation_sha256"].items():
            compare("matched_implementation:" + name, pins.get(name), expected)
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
            require(type(question["point_index"]) is int and 0 <= question["point_index"] < 361
                    and question["point"] == point_to_gtp(question["point_index"])
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
                # APS masks cannot shield a non-finite unselected token, so every board token must be finite.
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
        features_digest, features_payload = source_features_digest(source_features)
        report.update(
            source={"manifest": relative(manifest_path), "manifest_sha256": file_sha256(manifest_path),
                    "conflict_audit": relative(audit_path), "conflict_audit_sha256": file_sha256(audit_path),
                    "fusion_config": relative(cfg_path), "fusion_config_sha256": file_sha256(cfg_path),
                    "fixed_questions_sha256": content_sha256(questions), "source_features": source_features},
            source_features_digest={"sha256": features_digest, "payload": features_payload},
            fixed_questions=questions, fixed_grids=grids,
            sampling={"method": "Python Random(42).randrange(21), 720 independent slots with replacement; four successive slots per update",
                      "seed": SEED, "slot_indices": order, "question_ids": planned_ids,
                      "slot_indices_sha256": content_sha256(order), "question_ids_sha256": content_sha256(planned_ids),
                      "planned_counts": {q: planned_ids.count(q) for q in ids},
                      "planned_factual_label_counts": {name: planned_factual_counts[i] for i, name in enumerate(LABEL_NAMES)}})
        for key, expected in reference["source"].items():
            compare("matched_source:" + key, report["source"][key], expected)
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
        compare("base_attention_dropout", float(getattr(language.config, "attention_dropout", 0.0)), 0.0)
        # Re-seed exactly at adapter creation, independently of loader internals.
        torch.manual_seed(SEED)
        report["config"]["adapter_seed_applied"] = "immediately before GoQueen construction"
        report["config_sha256"] = content_sha256(report["config"])
        report["local_model"] = {"directory": relative(model_path), "local_files_only": True,
                                 "config_sha256": file_sha256(model_path / "config.json")}
        compare("local_model_identity", report["local_model"], reference["local_model"])
        model = AssistedPointGoQueen(language, LAYERS, 768, 8)
        report["post_construction_cpu_rng_sha256"] = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
        report["rng_reference_limitation"] = "Original M construction RNG state was not saved; its value is not reconstructed."
        named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        parameters = [p for _, p in named]
        signature = [{"name": name, "shape": list(p.shape), "dtype": str(p.dtype)} for name, p in named]
        report["ordered_adapter_parameter_signature"] = signature
        compare("ordered_parameter_names_shapes_dtypes", signature, expected_parameter_signature())
        compare("adapter_tensor_count", len(named), 101)
        compare("adapter_parameter_count", sum(p.numel() for p in parameters), 40438795)
        require(all(not p.requires_grad and p.device.type == "cpu" and p.dtype == torch.float32
                    for p in model.base_parameters), "Base LM must be frozen CPU FP32")
        require(not {id(p) for p in parameters} & {id(p) for p in model.base_parameters}, "Optimizer contains base parameters")
        report["architecture"] = model.architecture()
        compare("architecture_variant_contract", [report["architecture"]["variant"], report["architecture"]["routing_contract"]],
                [VARIANT, ROUTING_CONTRACT])
        report["initial_adapter_parameter_sha256"] = parameter_digest(parameters)
        report["frozen_base_sha256_before"] = parameter_digest(model.base_parameters)
        # Equality with M's stored fresh digest also shows that no earlier checkpoint was loaded.
        compare("initial_adapter_parameter_sha256", report["initial_adapter_parameter_sha256"], reference["initial_adapter_parameter_sha256"])
        compare("initial_frozen_base_parameter_sha256", report["frozen_base_sha256_before"], reference["frozen_base_sha256_before"])
        require(all(value == 0.0 for value in model.gate_values()), "Fresh gates must start at zero")
        inactive = inactive_parameters(named)
        compare("inactive_tensor_count_and_values", [len(inactive), sum(p.numel() for _, p in inactive)],
                [INACTIVE_TENSORS, INACTIVE_VALUES])
        report["inactive_44_tensor_sha256_before"] = parameter_digest(p for _, p in inactive)
        optimizer = torch.optim.AdamW(parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
        report["actual_optimizer_configuration"] = optimizer_contract(optimizer, named)
        contract["optimizer_contract_verified"] = True
        for key in ("seed", "fresh_adapters", "coordinate_interface", "batch_size", "max_updates", "optimizer",
                    "learning_rate", "warmup_updates", "gradient_clipping", "weight_decay", "evaluation_batch_size",
                    "optional_evaluation_updates", "minimum_final_eval_and_report_reserve_seconds", "injection_layers", "adapter_seed_applied"):
            compare("matched_regime:" + key, report["config"][key], reference["config"][key])
        monitor = RoutingMonitor(model)
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
                # One actual batch-1 forward per question; the outer check above is the budget guard.
                with monitor.expect([question["point_index"]], "evaluation"):
                    rows = score_questions(model, tokenizer, [question], features, batch_size=1)
                require(len(rows) == 1 and rows[0]["question_id"] == question["question_id"],
                        "APS scorer returned an unexpected row")
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
                routed = [routed_metadata(q) for q in selected]
                points = [q["point_index"] for q in selected]
                prefixes = [encoded[i][0] for i in selected_indices]
                attempt = {"update": step + 1, "question_ids": selected_ids,
                           "factual_target_labels": selected_factual_labels, "point_indices": points,
                           "prefix_lengths": [len(prefix) for prefix in prefixes],
                           "optimizer_step_started": False, "optimizer_step_returned": False}
                report["update_attempts"].append(attempt)
                attempted_counts.update(selected_ids)
                record()
                model.train()
                optimizer.zero_grad(set_to_none=True)
                with monitor.expect(points, "training"):
                    logits = _forward_logits(model, tokenizer, prefixes, routed, features)
                attempt["routing_verified"] = True
                if len(set(attempt["prefix_lengths"])) > 1:
                    contract["mixed_length_training_forwards"] += 1
                if not bool(torch.isfinite(logits).all()):
                    raise NonFiniteError("Non-finite training logits")
                targets = torch.tensor([label_ids[label] for label in selected_factual_labels], dtype=torch.long)
                loss = F.cross_entropy(logits, targets)
                if not bool(torch.isfinite(loss)):
                    raise NonFiniteError("Non-finite value-token loss")
                attempt["full_vocabulary_value_ce"] = float(loss.detach())
                loss.backward()
                attempt["gradient_norms_before_step"] = gradient_norms()
                require_exact_zero_gradients(inactive)
                attempt["inactive_44_gradients_exact_zero"] = True
                contract["inactive_zero_gradient_updates"] += 1
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
        report["gates"] = model.gate_values()
        require(all(p.grad is None and not p.requires_grad for p in model.base_parameters), "Base freeze contract changed")
        report["frozen_base_gradients_absent"] = True

        phase = "final_contract_and_source_validity"
        if not all(bool(torch.isfinite(p).all()) for p in parameters):
            raise NonFiniteError("Non-finite final adapter parameter")
        report["final_parameters_finite"] = True
        report["inactive_44_tensor_sha256_after"] = parameter_digest(p for _, p in inactive)
        contract["inactive_44_bytes_unchanged"] = (report["inactive_44_tensor_sha256_after"]
                                                   == report["inactive_44_tensor_sha256_before"])
        if not contract["inactive_44_bytes_unchanged"]:
            raise ContractViolation("Inactive Q/K/query_norm tensor bytes changed during optimization")
        # Descriptive only: Adam step counters advance; exactly-zero gradients keep both moments zero.
        moments = [optimizer.state.get(p, {}) for _, p in inactive]
        contract["inactive_adam_moments_all_zero"] = all(
            all(bool(state[key].eq(0).all()) for key in ("exp_avg", "exp_avg_sq") if key in state) for state in moments)
        if [id(p) for p in optimizer.param_groups[0]["params"]] != [id(p) for p in parameters]:
            raise ContractViolation("Optimizer membership changed during the run")
        contract["binding_clear_at_end"] = binding_clear(model)
        if not contract["binding_clear_at_end"]:
            raise ContractViolation("APS binding state survived the run")
        contract.update(routed_forwards=dict(monitor.counts), adapter_calls_checked=monitor.adapter_calls,
                        routed_samples_checked=monitor.routed_samples)
        scored_rows = sum(e["completed_questions"] for e in report["evaluations"])
        contract["routing_verified"] = (monitor.counts["training"] == len(report["update_attempts"])
                                        and monitor.counts["evaluation"] == scored_rows
                                        and monitor.adapter_calls == len(LAYERS) * (monitor.counts["training"] + scored_rows))
        if not contract["routing_verified"]:
            raise ContractViolation("Not every actual APS forward was routing-checked")
        compare("factual_questions_sha256_after", content_sha256(questions), reference["source"]["fixed_questions_sha256"])
        compare("original_M_report_unchanged", file_sha256(REFERENCE_REPORT), REFERENCE_REPORT_SHA256)
        for key, path in (("manifest_sha256", manifest_path), ("conflict_audit_sha256", audit_path), ("fusion_config_sha256", cfg_path)):
            compare("source_file_unchanged:" + key, file_sha256(path), report["source"][key])
        for position_id, source in source_features.items():
            compare("feature_file_unchanged:" + position_id, file_sha256(ROOT / source["file"]), source["sha256"])
        compare("imported_implementation_set_and_bytes_unchanged", imported_first_party(), pins)
        compare("runner_at_run_copy", file_sha256(paths["runner_copy"]), report["script_sha256"])
        compare("runner_source_unchanged", file_sha256(__file__), report["script_sha256"])
        report["source_validity_after"] = True
        record()

        phase = "final_frozen_base_hash"
        if deadline - time.perf_counter() > 8.0:
            report["frozen_base_sha256_after"] = parameter_digest(model.base_parameters)
            report["frozen_base_final_hash_status"] = "measured"
            compare("final_frozen_base_parameter_sha256", report["frozen_base_sha256_after"], reference["frozen_base_sha256_after"])
            compare("frozen_base_before_after_equality", report["frozen_base_sha256_after"], report["frozen_base_sha256_before"])
        else:
            report["frozen_base_final_hash_status"] = "not_measured_to_preserve_report_budget"
        phase = "checkpoint"
        if final["coverage_complete"] and deadline - time.perf_counter() > 6.0:
            metadata = checkpoint_metadata(updates=report["updates_completed"],
                                           manifest_sha256=report["source"]["manifest_sha256"],
                                           questions_sha256=content_sha256(questions),
                                           source_features_sha256=features_digest)
            trained_digest = parameter_digest(parameters)
            report["checkpoint_status"] = "saving"
            report["checkpoint_metadata"] = metadata
            record()
            save_file(model.adapter_state_dict(), str(paths["checkpoint_temp"]), metadata=metadata)
            paths["checkpoint_temp"].replace(paths["checkpoint"])
            report["checkpoint_status"] = "saved_unverified"
            report["checkpoint"] = relative(paths["checkpoint"])
            report["checkpoint_sha256"] = file_sha256(paths["checkpoint"])
            # The actual APS loader preflights metadata, names, shapes and finiteness, then copies the
            # same values back; unchanged bytes show the saved file restores this trajectory's weights.
            loaded = load_adapter_checkpoint(model, paths["checkpoint"])
            if loaded != metadata or parameter_digest(parameters) != trained_digest:
                raise ContractViolation("Saved APS checkpoint does not restore the trained adapter bytes")
            report["checkpoint_status"] = "saved_and_loader_verified"
        elif not final["coverage_complete"]:
            report["checkpoint_status"] = "not_saved_incomplete_final_evaluation"
        else:
            report["checkpoint_status"] = "not_saved_to_preserve_report_budget"

        conditions = endpoint_conditions(report)
        report["validity"]["endpoint_conditions"] = conditions
        report["validity"]["endpoint"] = all(conditions.values())
        endpoint = report["final_assisted_point_endpoint"]
        endpoint["valid"] = report["validity"]["endpoint"]
        endpoint["indicators"] = final["descriptive_indicators"]
        endpoint["interpretation"] = "This one trajectory measures fitting of the 21 training facts after external single-point selection. Selection, removal of attention dilution and changed context are bundled. Neither success nor finite failure establishes natural coordinate addressing, a unique cause of the earlier M failure, generic reading or generalization."
        if endpoint["valid"]:
            endpoint.update(factual_indicators(final["rows"]))
            endpoint["reason"] = "Valid matched reproduction and contracts, 180 returned updates, all 21 final cells, measured equal base hash and a loader-verified checkpoint."
        else:
            failed = sorted(name for name, passed in conditions.items() if not passed)
            endpoint["reason"] = "No endpoint category; unmet conditions: " + ", ".join(failed) + ". Measured metrics remain descriptive."
    except BudgetExpired:
        report.update(status="incomplete_at_cap", run_completed=False, bounded_run_finalized=False,
                      stop_reason="soft_deadline", interrupted_phase=phase,
                      limitation="Native work may finish before the signal is delivered. An interrupted optimizer step is not assumed complete; recorded start/return flags distinguish it.")
    except (NonFiniteError, FloatingPointError) as exc:
        report.update(status="stopped_nonfinite", run_completed=False, bounded_run_finalized=False,
                      stop_reason="nonfinite", interrupted_phase=phase, limitation=str(exc))
    except ContractViolation as exc:
        contract["violation"] = str(exc)
        report.update(status="invalid_contract", run_completed=False, bounded_run_finalized=False,
                      stop_reason="contract_violation", interrupted_phase=phase)
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
        if monitor is not None:
            monitor.close()
        report["coverage"] = {"requested_questions_per_evaluation": 21,
                              "evaluations": [{"name": e["name"], "updates_completed": e["updates_completed"],
                                               "completed_questions": e["completed_questions"],
                                               "coverage_complete": e["coverage_complete"]} for e in report["evaluations"]],
                              "completed_updates": report["updates_completed"], "planned_updates": MAX_UPDATES,
                              "completed_training_slots": sum(completed_counts.values()),
                              "attempted_training_slots": sum(attempted_counts.values()),
                              "planned_training_slots": MAX_UPDATES * BATCH_SIZE}
        if report["checkpoint_status"] in ("saving", "saved_unverified"):
            report["checkpoint_status"] = "save_or_loader_verification_not_completed"
        endpoint = report["final_assisted_point_endpoint"]
        if report["status"] != "complete" or time.perf_counter() > deadline:
            report["validity"]["endpoint"] = False
            endpoint.update(valid=False, category=None)
        if time.perf_counter() > deadline:
            endpoint["reason"] = "All-inclusive soft budget exceeded; any measured scores remain descriptive only."
        record()
        # A final JSON write itself belongs to the budget. If it crosses the
        # soft cap, durably replace any endpoint category with an invalid one.
        if time.perf_counter() > deadline:
            report["validity"]["endpoint"] = False
            endpoint.update(valid=False, category=None,
                            reason="All-inclusive soft budget exceeded during final report delivery; measured scores are descriptive only.")
            record()
        print(json.dumps({"status": report["status"], "run_completed": report["run_completed"],
                          "updates_completed": report["updates_completed"], "stop_reason": report["stop_reason"],
                          "endpoint_valid": report["validity"]["endpoint"],
                          "total_seconds": report["total_seconds"]}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
