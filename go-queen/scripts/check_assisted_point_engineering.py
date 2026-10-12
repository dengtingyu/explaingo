#!/usr/bin/env python3
"""One bounded M-APS engineering check: two real backwards, zero updates.

This is not a fitting run. The artificial gate=.01 checkpoint is retained only
to verify the APS disk contract and restoration. A native operation may delay
the 240-second soft alarm; elapsed time and incomplete coverage remain explicit.
"""
from __future__ import annotations

import time
PROGRAM_STARTED = time.perf_counter()

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from train_grounding_microfit import (LAYERS, SEED, content_sha256, file_sha256,
                                     fixed_cells, parameter_digest, relative)

REFERENCE_SHA256 = "23a0c4b4be9e5594bb06207f9acdc45b5ff93f1f0b64b3d0da4d786a815e7289"
ROUTING_CONTRACT = "single_point_bool_mask_v1"


def expected_signature():
    suffixes = [("alpha", []), ("query_norm.weight", [1024]), ("query_norm.bias", [1024]),
                ("expert_norm.weight", [768]), ("expert_norm.bias", [768]),
                ("query.weight", [1024, 1024]), ("key.weight", [1024, 768]),
                ("value.weight", [1024, 768]), ("output.weight", [1024, 1024])]
    return [{"name": f"language_model.model.layers.{layer}.adapter.{suffix}",
             "shape": shape, "dtype": "torch.float32"}
            for layer in LAYERS for suffix, shape in suffixes] + [
                {"name": name, "shape": [19, 768], "dtype": "torch.float32"}
                for name in ("rows.weight", "columns.weight")]


def imported_first_party():
    """Capture the actual imported project files, including this producer."""
    paths = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix == ".py" and (path.is_relative_to(ROOT / "src")
                                     or path.is_relative_to(ROOT / "scripts")):
            paths.add(path)
    return {relative(path): file_sha256(path) for path in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=240)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/assisted-point-engineering/report.json")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 240:
        raise ValueError("Engineering total soft budget must be 1..240 seconds")
    output = args.output.resolve()
    reserved_names = {"run-start.json", "adapter.safetensors", "script-at-run.py"}
    if (not output.is_relative_to(ROOT / "outputs") or output.suffix != ".json"
            or output.name.casefold() in reserved_names
            or output.parent == ROOT / "outputs"):
        raise ValueError("Use a new dedicated engineering outputs directory")
    # Exclusive directory creation protects every artifact, including crash runs.
    output.parent.mkdir(parents=True, exist_ok=False)
    source_copy = output.parent / "script-at-run.py"
    with source_copy.open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    start_record = output.parent / "run-start.json"
    with start_record.open("x", encoding="utf-8") as stream:
        json.dump({"stage": "m_assisted_point_engineering", "optimizer_updates": 0,
                   "seconds": args.seconds, "automatic_retry": False,
                   "script_sha256": file_sha256(__file__)}, stream)
    deadline = PROGRAM_STARTED + args.seconds
    report = {
        "status": "running", "stage": "m_assisted_point_engineering",
        "engineering_completed": False, "optimizer_updates": 0,
        "optimizer_constructed": False, "backward_passes": 0,
        "new_fit_experiment": False, "formal_grounding_acceptance": "not_assessed",
        "scope": "One real local Qwen3-0.6B APS integration check. No fitting, optimizer step, or generalization test.",
        "device": "cpu", "threads": 4, "dtype": "float32", "sdpa": "math",
        "soft_cap_seconds": args.seconds,
        "deadline_policy": "Includes imports, loading, checks, hashes, saving and report delivery. SIGALRM is soft; native delays and actual overruns are recorded.",
        "script": relative(__file__), "script_sha256": file_sha256(__file__),
        "script_at_run_copy": relative(source_copy),
        "script_at_run_copy_sha256": file_sha256(source_copy),
        "initial_rows": [], "initial_requested_questions": 21,
        "initial_completed_questions": 0, "checks": [],
        "checkpoint_status": "not_saved", "frozen_base_final_hash_status": "not_measured",
    }
    model = None
    phase = "imports"
    first_feature = None

    def check(name, condition, **facts):
        report["checks"].append({"name": name, "passed": bool(condition), **facts})
        if not condition:
            raise AssertionError(name)

    def record():
        report["seconds"] = time.perf_counter() - PROGRAM_STARTED
        report["within_soft_cap"] = report["seconds"] <= args.seconds
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n", encoding="utf-8")

    def expired(_signum, _frame):
        raise TimeoutError("APS engineering soft deadline reached")

    previous_handler = signal.signal(signal.SIGALRM, expired)
    record()
    signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - time.perf_counter()))
    try:
        import numpy as np
        import torch
        from torch import nn
        from torch.nn import functional as F
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from safetensors.torch import save_file
        from goqueen.fusion import GoQueen
        from goqueen.assisted_point import AssistedPointGoQueen
        from goqueen.checkpoints import load_adapter_checkpoint
        from goqueen.grounding_data import occupancy_prompt, point_to_gtp
        from goqueen.grounding_eval import encode_prefix, _left_pad
        from goqueen.assisted_point_eval import _forward_logits, score_questions

        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        source_pins = imported_first_party()
        report["implementation_sha256"] = source_pins
        report["implementation_at_run_copies"] = {}
        for name, digest in source_pins.items():
            copy = output.parent / "source-at-run" / name
            copy.parent.mkdir(parents=True, exist_ok=True)
            with copy.open("xb") as stream:
                stream.write((ROOT / name).read_bytes())
            check("imported_source_copy:" + name, file_sha256(copy) == digest)
            report["implementation_at_run_copies"][name] = {
                "file": relative(copy), "sha256": digest}

        phase = "source_validation"
        reference_path = ROOT / "outputs/grounding-microfit/report.json"
        check("original_M_report", file_sha256(reference_path) == REFERENCE_SHA256)
        reference = json.loads(reference_path.read_text())
        check("original_M_reference_complete", reference["run_completed"]
              and reference["updates_completed"] == 180)
        check("original_M_runner_source", file_sha256(ROOT / reference["script"])
              == reference["script_sha256"])
        check("original_M_implementation_sources", all(file_sha256(ROOT / name) == digest
              for name, digest in reference["implementation_sha256"].items()))
        report["original_M_implementation_sha256"] = reference["implementation_sha256"]
        manifest_path = ROOT / "outputs/grounding-data/manifest.json"
        audit_path = ROOT / "outputs/grounding-conflict-audit/report.json"
        cfg_path = ROOT / "configs/mixed-train.json"
        data, audit, cfg = [json.loads(path.read_text()) for path in
                            (manifest_path, audit_path, cfg_path)]
        check("original_manifest", audit["status"] == "complete"
              and file_sha256(manifest_path) == audit["source"]["manifest_sha256"]
              == reference["source"]["manifest_sha256"])
        check("fusion_configuration", cfg["injection_layers"] == LAYERS
              and cfg["expert_width"] == 768 and cfg["adapter_heads"] == 8)
        questions, grids = fixed_cells(data, audit)
        check("exact_fixed21_cells", questions == reference["fixed_questions"]
              and content_sha256(questions) == reference["source"]["fixed_questions_sha256"])
        positions = {p["position_id"]: p for p in data["positions"]}
        features, source_features = {}, {}
        for q in questions:
            check("canonical_route:" + q["question_id"],
                  type(q["point_index"]) is int and 0 <= q["point_index"] < 361
                  and q["point"] == point_to_gtp(q["point_index"])
                  and q["prompt"] == occupancy_prompt(q["to_play"], q["point"]))
        for position_id in dict.fromkeys(q["position_id"] for q in questions):
            position = positions[position_id]
            path = ROOT / position["source_feature_file"]
            digest, name = file_sha256(path), relative(path)
            expected = audit["checked_source_features"][position_id]
            check("source_feature:" + position_id,
                  digest == position["feature_sha256"] == expected["sha256"]
                  and name == expected["file"] and position["to_play"] == expected["to_play"])
            with np.load(path, allow_pickle=False) as saved:
                spatial = saved["input_spatial"]
                check("source_planes:" + position_id, spatial.shape == (1, 22, 19, 19))
                own, other = spatial[0, 1].reshape(-1), spatial[0, 2].reshape(-1)
                check("source_plane_values:" + position_id,
                      np.isfinite(own).all() and np.isfinite(other).all()
                      and np.isin(own, [0, 1]).all() and np.isin(other, [0, 1]).all()
                      and not np.any((own != 0) & (other != 0)))
                own_label = 1 if position["to_play"] == "B" else 2
                labels = np.where(own != 0, own_label,
                                  np.where(other != 0, 3 - own_label, 0)).astype(int).tolist()
                check("source_absolute_labels:" + position_id,
                      labels == position["absolute_board_labels"]
                      and all(labels[q["point_index"]] == q["label"]
                              for q in questions if q["position_id"] == position_id))
                arrays = [saved[f"trunk_after_block{i}"] for i in range(11)]
                check("feature_shape_dtype_finite:" + position_id,
                      all(a.shape == (361, 768) and a.dtype == np.float32
                          and np.isfinite(a).all() for a in arrays))
                features[position_id] = torch.from_numpy(np.stack(arrays))
            source_features[position_id] = {"file": name, "sha256": digest,
                                           "board_id": "sha256:" + digest,
                                           "absolute_labels_checked": 361}
        check("exact_source_feature_records", source_features == reference["source"]["source_features"])
        feature_list = sorted(({"file": entry["file"], "sha256": entry["sha256"]}
                               for entry in source_features.values()), key=lambda item: item["file"])
        checkpoint_metadata = {
            "variant": "m_assisted_point", "routing_contract": ROUTING_CONTRACT,
            "updates": "0", "engineering_only": "true",
            "source_manifest_sha256": file_sha256(manifest_path),
            "questions_sha256": content_sha256(questions),
            "source_features_sha256": content_sha256(feature_list),
            "scope": "Artificial gates=.01; engineering restoration only; no optimizer updates"}
        report["source"] = {"reference": relative(reference_path), "reference_sha256": REFERENCE_SHA256,
                            "manifest": relative(manifest_path), "manifest_sha256": file_sha256(manifest_path),
                            "conflict_audit": relative(audit_path), "conflict_audit_sha256": file_sha256(audit_path),
                            "fusion_config": relative(cfg_path), "fusion_config_sha256": file_sha256(cfg_path),
                            "fixed_questions_sha256": content_sha256(questions),
                            "source_features": source_features,
                            "source_features_digest_payload": feature_list}
        report["checkpoint_metadata"] = checkpoint_metadata
        features_before = parameter_digest(features.values())
        report["expert_feature_tensor_sha256_before"] = features_before
        record()

        phase = "constructor_and_loading"
        class SkeletonDecoder(nn.Module):
            attention_type = "full_attention"

        class SkeletonLanguage(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(model_type="qwen3", hidden_size=1024, num_hidden_layers=28)
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([SkeletonDecoder() for _ in range(28)])

        torch.manual_seed(SEED)
        original = GoQueen(SkeletonLanguage(), LAYERS, 768, 8)
        original_named = [(n, p) for n, p in original.named_parameters() if p.requires_grad]
        original_signature = [{"name": n, "shape": list(p.shape), "dtype": str(p.dtype)}
                              for n, p in original_named]
        original_digest = parameter_digest(p for _, p in original_named)
        original_rng = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
        check("original_signature", original_signature == expected_signature())
        check("original_constructor_digest", original_digest == reference["initial_adapter_parameter_sha256"])
        del original_named, original
        tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/Qwen3-0.6B", local_files_only=True)
        encoded = [encode_prefix(tokenizer, q["prompt"]) for q in questions]
        label_ids = encoded[0][1]
        check("prefixes_and_label_tokens", all(ids == label_ids for _, ids in encoded)
              and list(label_ids) == reference["label_token_ids"]
              and content_sha256([prefix.tolist() for prefix, _ in encoded])
              == reference["prefix_token_ids_sha256"])
        language = AutoModelForCausalLM.from_pretrained(ROOT / "models/Qwen3-0.6B",
                    local_files_only=True, dtype=torch.float32, attn_implementation="sdpa")
        language.gradient_checkpointing_disable()
        torch.manual_seed(SEED)
        model = AssistedPointGoQueen(language, LAYERS, 768, 8)
        named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        signature = [{"name": name, "shape": list(p.shape), "dtype": str(p.dtype)} for name, p in named]
        digest = parameter_digest(p for _, p in named)
        rng = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
        report.update(ordered_adapter_parameter_signature=signature,
                      initial_adapter_parameter_sha256=digest,
                      post_construction_cpu_rng_sha256=rng,
                      rng_scope="Original and APS constructors compared now; historical M RNG was not saved.")
        check("exact_names_shapes_order", signature == original_signature)
        check("exact_initial_adapter_digest", digest == original_digest)
        check("constructor_rng_equivalence", rng == original_rng)
        check("tensor_and_parameter_count", len(named) == 101 and sum(p.numel() for _, p in named) == 40438795)
        check("all101_registered_trainable", {id(p) for _, p in named}
              == {id(p) for p in model.trainable_parameters()})
        check("base_frozen_CPU_FP32", all(not p.requires_grad and p.device.type == "cpu"
              and p.dtype == torch.float32 for p in model.base_parameters))
        check("fresh_zero_gates", all(value == 0.0 for value in model.gate_values()))
        before = parameter_digest(model.base_parameters)
        report["frozen_base_sha256_before"] = before
        check("base_matches_original_M", before == reference["frozen_base_sha256_before"])
        inactive = [(name, p) for name, p in named if name.endswith(
                    (".query_norm.weight", ".query_norm.bias", ".query.weight", ".key.weight"))]
        check("inactive_QK_querynorm_44", len(inactive) == 44)
        inactive_before = parameter_digest(p for _, p in inactive)
        report["inactive_44_tensor_sha256_before"] = inactive_before
        report["architecture"] = model.architecture()

        @contextmanager
        def capture_adapter_sdpa(points, records):
            saved_sdpa = F.scaled_dot_product_attention

            def observed(query, key, value, *positional, **kwargs):
                if not (query.ndim == key.ndim == value.ndim == 4
                        and key.shape[-2] == 361 and key.shape[1] == 8):
                    return saved_sdpa(query, key, value, *positional, **kwargs)
                mask = kwargs.get("attn_mask", positional[0] if positional else None)
                dropout = kwargs.get("dropout_p", positional[1] if len(positional) > 1 else 0.0)
                causal = kwargs.get("is_causal", positional[2] if len(positional) > 2 else False)
                if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
                        or query.shape[0] != len(points) or dropout != 0.0 or causal):
                    raise AssertionError("APS SDPA mask/batch/dropout/causal contract")
                expanded = torch.broadcast_to(mask, (*query.shape[:-1], 361))
                indices = torch.tensor(points, device=query.device, dtype=torch.long)
                right_points = expanded.gather(-1, indices[:, None, None, None].expand(
                    query.shape[0], query.shape[1], query.shape[2], 1))
                if not bool(expanded.sum(-1).eq(1).all()) or not bool(right_points.all()):
                    raise AssertionError("APS SDPA must allow exactly each canonical point")
                if not all(bool(torch.isfinite(x).all()) for x in (query, key, value)):
                    raise AssertionError("APS Q/K/V finite contract")
                attended = saved_sdpa(query, key, value, *positional, **kwargs)
                selected_v = value[torch.arange(len(points), device=value.device), :, indices, :]
                expected = selected_v.unsqueeze(2).expand_as(attended)
                difference = float((attended.detach() - expected.detach()).abs().max())
                if difference != 0.0:
                    raise AssertionError("Singleton APS SDPA differs from selected V reference")
                records.append({"batch_size": len(points), "points": list(points),
                                "heads": query.shape[1], "query_tokens": query.shape[2],
                                "mask_shape": list(mask.shape), "selected_value_max_abs_difference": difference})
                return attended

            F.scaled_dot_product_attention = observed
            try:
                yield
            finally:
                F.scaled_dot_product_attention = saved_sdpa

        def binding_is_clear():
            return (not model._bound and all(layer.expert_features is None
                    and hasattr(layer.adapter, "point_indices")
                    and layer.adapter.point_indices is None
                    and layer.adapter.point_mask is None for layer in model.injected_layers))

        model.eval()
        with sdpa_kernel(SDPBackend.MATH):
            phase = "initial21_scoring_and_mask"
            old = next(e for e in reference["evaluations"] if e["name"] == "initial")["rows"]
            old_by_id = {row["question_id"]: row for row in old}
            captured = []
            for question in questions:
                if time.perf_counter() >= deadline - 5.0:
                    raise TimeoutError("Insufficient engineering time for initial21 coverage")
                calls = []
                with capture_adapter_sdpa([question["point_index"]], calls):
                    rows = score_questions(model, tokenizer, [question], features,
                                           batch_size=1, deadline=deadline - 5.0)
                check("initial_row:" + question["question_id"], len(rows) == 1
                      and rows[0]["question_id"] == question["question_id"] and len(calls) == 11)
                report["initial_rows"].extend(rows)
                report["initial_completed_questions"] = len(report["initial_rows"])
                captured.extend(calls)
                record()
            rows = report["initial_rows"]
            delta = max(abs(value - old_by_id[row["question_id"]]["label_log_probs"][i])
                        for row in rows for i, value in enumerate(row["label_log_probs"]))
            choices = all(row["forced_choice_label"] == old_by_id[row["question_id"]]["forced_choice_label"]
                          and row["unconstrained_token_id"] == old_by_id[row["question_id"]]["unconstrained_token_id"]
                          for row in rows)
            check("all21_initial_scores_match", len(rows) == 21 and len({r["question_id"] for r in rows}) == 21
                  and math.isfinite(delta) and delta <= 1e-5 and choices,
                  max_abs_logp_difference=delta, tolerance=1e-5)
            check("actual_all21_singleton_mask_and_value_reference", len(captured) == 231,
                  calls=len(captured), per_layer_calls=21, all_selected_value_differences_zero=True)
            report["initial_sdpa_capture"] = captured

            # Labels are removed from forward metadata. Only the external CE
            # targets below read them; the helper must validate canonical prompts.
            selected = [questions[0], questions[-1]]
            routed = [{key: value for key, value in q.items() if key not in ("label", "answer")}
                      for q in selected]
            prefixes = [encoded[0][0], encoded[-1][0]]
            point_list = [q["point_index"] for q in selected]
            point_tensor = torch.tensor(point_list, dtype=torch.long)
            targets = torch.tensor([label_ids[q["label"]] for q in selected], dtype=torch.long)
            report["real_backward_route_question_ids"] = [q["question_id"] for q in selected]
            report["real_backward_route_points"] = point_list
            check("no_gold_fields_in_forward_metadata", all("label" not in q and "answer" not in q for q in routed))
            phase = "zero_gate_backward"
            model.zero_grad(set_to_none=True)
            calls = []
            with capture_adapter_sdpa(point_list, calls):
                logits = _forward_logits(model, tokenizer, prefixes, routed, features)
            check("zero_gate_real_batch_mask", len(calls) == 11)
            F.cross_entropy(logits, targets).backward()
            report["backward_passes"] += 1
            check("zero_gate_gradients_finite", all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                                                    for _, p in named))
            check("zero_gate_gate_gradient_present", any(bool(p.grad.ne(0).any()) for n, p in named if n.endswith(".alpha")))
            check("zero_gate_other_gradients_zero", all(bool(p.grad.eq(0).all()) for n, p in named if not n.endswith(".alpha")))
            check("zero_gate_binding_cleanup", binding_is_clear())

            phase = "manual_nonzero_gate_backward"
            model.zero_grad(set_to_none=True)
            with torch.no_grad():
                for layer in model.injected_layers:
                    layer.adapter.alpha.fill_(.01)
            first_feature = features[selected[0]["position_id"]]
            first_feature.requires_grad_(True)
            calls = []
            with capture_adapter_sdpa(point_list, calls):
                logits = _forward_logits(model, tokenizer, prefixes, routed, features)
            check("nonzero_gate_real_batch_mask", len(calls) == 11)
            check("nonzero_gate_forward_finite", bool(torch.isfinite(logits).all()))
            expected_logits = logits.detach().clone()
            F.cross_entropy(logits, targets).backward()
            report["backward_passes"] += 1
            check("nonzero_gate_gradients_finite", all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                                                       for _, p in named))
            check("nonzero_gate_QK_querynorm_exact_zero", all(bool(p.grad.eq(0).all()) for _, p in inactive), tensors=44)
            groups = {"value": [p for n, p in named if n.endswith(".value.weight")],
                      "output": [p for n, p in named if n.endswith(".output.weight")],
                      "gate": [p for n, p in named if n.endswith(".alpha")],
                      "expert_norm": [p for n, p in named if ".expert_norm." in n],
                      "rows": [model.rows.weight], "columns": [model.columns.weight]}
            group_norms = {name: math.sqrt(math.fsum(float(p.grad.double().square().sum()) for p in parameters))
                           for name, parameters in groups.items()}
            check("active_path_gradient_groups", all(math.isfinite(value) and value > 0 for value in group_norms.values()),
                  aggregate_norms=group_norms)
            check("expert_features_detached", first_feature.grad is None)
            check("base_frozen_no_gradients", all(not p.requires_grad and p.grad is None for p in model.base_parameters))
            check("inactive_44_bytes_unchanged", parameter_digest(p for _, p in inactive) == inactive_before)
            first_feature.requires_grad_(False)
            check("nonzero_gate_binding_cleanup", binding_is_clear())

            phase = "left_padding"
            # Keep the full canonical prefixes. Padding never changes their
            # content and does not bypass the helper's prefix validation.
            with torch.no_grad():
                batched = _forward_logits(model, tokenizer, prefixes, routed, features).log_softmax(-1)
                individual = torch.cat([_forward_logits(model, tokenizer, [prefix], [q], features).log_softmax(-1)
                                        for prefix, q in zip(prefixes, routed)])
            batch_delta = float((batched[:, list(label_ids)] - individual[:, list(label_ids)]).abs().max())
            check("actual_batch_single_equivalence", batch_delta <= 1e-4
                  and torch.equal(batched.argmax(-1), individual.argmax(-1)),
                  max_label_logp_difference=batch_delta, tolerance=1e-4,
                  canonical_prefix_lengths=[len(prefix) for prefix in prefixes])
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            inputs = _left_pad(prefixes, pad_id, torch.device("cpu"))
            for name, value in list(inputs.items()):
                fill = pad_id if name == "input_ids" else 0
                inputs[name] = torch.cat([torch.full((2, 2), fill, dtype=value.dtype), value], dim=1)
            expert_batch = torch.stack([features[q["position_id"]] for q in selected])
            with torch.no_grad():
                padded = model(**inputs, expert_features=expert_batch, point_indices=point_tensor,
                               logits_to_keep=1, use_cache=False).logits[:, -1].float().log_softmax(-1)
            pad_delta = float((padded[:, list(label_ids)] - batched[:, list(label_ids)]).abs().max())
            check("explicit_left_padding_equivalence", pad_delta <= 1e-4
                  and torch.equal(padded.argmax(-1), batched.argmax(-1)),
                  max_label_logp_difference=pad_delta, tolerance=1e-4, extra_leading_pad_columns=2)

            phase = "checkpoint_restoration"
            checkpoint = output.parent / "adapter.safetensors"
            save_file(model.adapter_state_dict(), str(checkpoint), metadata=checkpoint_metadata)
            with torch.no_grad():
                for _, parameter in named:
                    parameter.zero_()
            check("before_restore_binding_cleanup", binding_is_clear())
            metadata = load_adapter_checkpoint(model, checkpoint)
            check("restored_APS_metadata", metadata == checkpoint_metadata)
            with torch.no_grad():
                restored = _forward_logits(model, tokenizer, prefixes, routed, features)
            restore_delta = float((restored - expected_logits).abs().max())
            check("manual_gate_checkpoint_output_equivalence", restore_delta == 0.0,
                  max_abs_logit_difference=restore_delta)
            check("restored_inactive_44_bytes", parameter_digest(p for _, p in inactive) == inactive_before)
            report["checkpoint_status"] = "engineering_only_saved_and_restored"
            report["checkpoint"] = relative(checkpoint)
            report["checkpoint_sha256"] = file_sha256(checkpoint)
            report["checkpoint_scope"] = "Gate=.01 manually; zero optimizer steps; not fitted task weights."

            phase = "cache_and_finally_cleanup"
            gen_inputs = {"input_ids": prefixes[0].unsqueeze(0), "max_new_tokens": 2, "min_new_tokens": 2,
                          "do_sample": False, "pad_token_id": tokenizer.eos_token_id}
            gen_features = features[selected[0]["position_id"]].unsqueeze(0)
            gen_points = point_tensor[:1]
            with torch.no_grad():
                cached = model.generate(expert_features=gen_features, point_indices=gen_points,
                                        use_cache=True, **gen_inputs)
                uncached = model.generate(expert_features=gen_features, point_indices=gen_points,
                                          use_cache=False, **gen_inputs)
            check("greedy_cache_generation_equivalence", torch.equal(cached, uncached)
                  and cached.shape[1] - prefixes[0].shape[0] == 2,
                  generated_tokens=int(cached.shape[1] - prefixes[0].shape[0]))
            class ExpectedBindingExit(Exception):
                pass
            try:
                with model.bind_expert(gen_features, point_indices=gen_points):
                    raise ExpectedBindingExit()
            except ExpectedBindingExit:
                pass
            check("exception_finally_binding_cleanup", binding_is_clear())

        phase = "final_hashes_and_source"
        check("exactly_two_real_backwards_zero_updates", report["backward_passes"] == 2
              and report["optimizer_updates"] == 0 and not report["optimizer_constructed"])
        after = parameter_digest(model.base_parameters)
        report["frozen_base_sha256_after"] = after
        report["frozen_base_final_hash_status"] = "measured"
        check("base_bytes_unchanged", after == before)
        features_after = parameter_digest(features.values())
        report["expert_feature_tensor_sha256_after"] = features_after
        check("expert_feature_bytes_unchanged", features_after == features_before)
        report["inactive_44_tensor_sha256_after"] = parameter_digest(p for _, p in inactive)
        check("inactive_QK_querynorm_bytes_final", report["inactive_44_tensor_sha256_after"] == inactive_before)
        check("all_binding_cleanup", binding_is_clear())
        check("imported_source_set_and_bytes_unchanged", imported_first_party() == source_pins)
        check("fact_source_files_unchanged", file_sha256(reference_path) == REFERENCE_SHA256
              and file_sha256(manifest_path) == report["source"]["manifest_sha256"]
              and file_sha256(audit_path) == report["source"]["conflict_audit_sha256"]
              and file_sha256(cfg_path) == report["source"]["fusion_config_sha256"]
              and all(file_sha256(ROOT / name) == digest for name, digest
                      in report["original_M_implementation_sha256"].items())
              and all(file_sha256(ROOT / entry["file"]) == entry["sha256"]
                      for entry in source_features.values()))
        report["status"] = "passed"
        report["engineering_completed"] = True
    except BaseException as exc:
        report.update(status="partial_deadline" if isinstance(exc, TimeoutError) else "failed",
                      engineering_completed=False, exception_type=type(exc).__name__, phase_at_stop=phase)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if first_feature is not None:
            first_feature.requires_grad_(False)
        report["phase_at_stop"] = phase
        if time.perf_counter() > deadline:
            report.update(status="failed_budget", engineering_completed=False)
        record()
        report["seconds_through_first_report_delivery"] = time.perf_counter() - PROGRAM_STARTED
        if report["seconds_through_first_report_delivery"] > args.seconds:
            report.update(status="failed_budget", engineering_completed=False)
        record()
        if time.perf_counter() > deadline and report["status"] != "failed_budget":
            report.update(status="failed_budget", engineering_completed=False)
            record()
        print(json.dumps({"status": report["status"], "checks": len(report["checks"]),
                          "seconds": report["seconds"], "backward_passes": report["backward_passes"],
                          "optimizer_updates": report["optimizer_updates"]}), flush=True)


if __name__ == "__main__":
    main()
