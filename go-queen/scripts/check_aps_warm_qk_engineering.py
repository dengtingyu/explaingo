#!/usr/bin/env python3
"""One APS-to-natural engineering check, with one backward and zero updates.

At most 24 model forwards: 21 assisted restoration samples, one natural batch
of four, and the same natural single sample before and after checkpoint load.
No optimizer is constructed. This does not test the future 1,000-update recipe.
"""
from __future__ import annotations

import time
PROGRAM_STARTED = time.perf_counter()

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from train_grounding_microfit import (LAYERS, content_sha256, file_sha256,
                                     parameter_digest, relative, require)
from eval_assisted_point_transfer import (MANIFEST, MANIFEST_SHA, FIT_REPORT, FIT_SHA,
                                         CHECKPOINT, CHECKPOINT_SHA, TOLERANCE,
                                         select_questions, restore_allows_validation)

OUTPUT = ROOT / "outputs/aps-warm-qk-engineering"
QK_SUFFIXES = (".query_norm.weight", ".query_norm.bias", ".query.weight", ".key.weight")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=240)
    parser.add_argument("--origin-start", type=float, default=PROGRAM_STARTED)
    args = parser.parse_args(argv)
    require(1 <= args.seconds <= 240 and math.isfinite(args.origin_start)
            and 0 < args.origin_start <= PROGRAM_STARTED, "Invalid engineering budget")
    origin, deadline = args.origin_start, args.origin_start + args.seconds
    output, temporary = OUTPUT / "report.json", OUTPUT / "report.tmp"
    source_copy, checkpoint = OUTPUT / "script-at-run.py", OUTPUT / "adapter.safetensors"
    reserved = (output, temporary, source_copy, checkpoint, OUTPUT / "run-start.json",
                OUTPUT / "source-at-run")
    require(not any(p.exists() for p in reserved), "Retain previous engineering attempt; no overwrite")
    OUTPUT.mkdir(parents=True, exist_ok=True)  # External process records may already exist.
    with (OUTPUT / "run-start.json").open("x") as stream:
        json.dump({"stage": "aps_warm_qk_engineering", "seconds": args.seconds,
                   "origin_monotonic": origin, "script_sha256": file_sha256(__file__),
                   "automatic_retry": False, "optimizer_updates": 0}, stream)
    with source_copy.open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    report = {
        "stage": "aps_warm_qk_engineering", "status": "running", "engineering_completed": False,
        "optimizer_constructed": False, "optimizer_updates": 0, "backward_passes": 0,
        "forward_calls": {"assisted_restoration": 0, "natural_batch4": 0, "natural_single": 0},
        "scope": "Zero-update integration only; no optimizer membership at runtime, fit, validation54, generation or mechanism inference.",
        "limits": {"model_forwards": 24, "sample_slots": 27, "backwards": 1, "optimizer_updates": 0},
        "runtime": {"device": "cpu", "threads": 4, "dtype": "float32", "sdpa": "math",
                    "soft_total_seconds": args.seconds},
        "script": relative(__file__), "script_sha256": file_sha256(__file__),
        "script_at_run_copy": relative(source_copy), "checks": [], "restoration_rows": [],
        "natural_sdpa": [], "checkpoint_status": "not_saved", "error": None,
    }
    phase, model, monitor, feature_with_grad = "imports", None, None, None

    def check(name, condition, **facts):
        report["checks"].append({"name": name, "passed": bool(condition), **facts})
        require(condition, name)

    def record():
        report["seconds_before_report_write"] = time.perf_counter() - origin
        report["within_soft_cap"] = report["seconds_before_report_write"] <= args.seconds
        if not report["within_soft_cap"]:
            report["engineering_completed"] = False
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(output)

    def available():
        if time.perf_counter() >= deadline - 10.0:
            raise TimeoutError("Engineering reserve reached")

    def expired(_signum, _frame):
        raise TimeoutError("Engineering soft deadline reached")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - time.perf_counter()))
    try:
        record()
        import numpy as np
        import torch
        import transformers
        from torch.nn import functional as F
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from goqueen.assisted_point import AssistedPointGoQueen
        from goqueen import assisted_point_eval, grounding_eval
        from goqueen.fusion import GoQueen, GatedBoardAttention
        from goqueen.checkpoints import load_adapter_checkpoint
        from goqueen.grounding_data import occupancy_prompt, point_to_gtp
        from goqueen.transfer_metrics import compare_restoration, summarize_transfer
        from goqueen.warm_qk import (adapter_parameters, convert_to_natural,
                                    freeze_except_qk, save_full_checkpoint)
        from train_grounding_assisted_point import RoutingMonitor, binding_clear, imported_first_party

        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        report["runtime_versions"] = {"python": sys.version.split()[0], "torch": torch.__version__,
                                      "transformers": transformers.__version__, "numpy": np.__version__}
        pins = {relative(p): file_sha256(p) for p in (MANIFEST, FIT_REPORT, CHECKPOINT)}
        check("pinned_inputs", pins[relative(MANIFEST)] == MANIFEST_SHA
              and pins[relative(FIT_REPORT)] == FIT_SHA and pins[relative(CHECKPOINT)] == CHECKPOINT_SHA)
        data, fit = [json.loads(p.read_text()) for p in (MANIFEST, FIT_REPORT)]
        restoration, _unused54, saved_final, _separation = select_questions(data, fit)
        by_id = {q["question_id"]: q for q in data["training_pool_questions"]}
        order = data["train_order"]
        check("stored_training_order", len(order) == 4000 and len(by_id) == len(data["training_pool_questions"])
              and all(qid in by_id for qid in order[:4]))
        batch = [by_id[qid] for qid in order[:4]]
        report.update(input_sha256=pins, restoration_questions=restoration,
                      engineering_batch_questions=batch, train_order_sha256=content_sha256(order))
        check("training_batch_only", len(batch) == 4 and all(q["split"] == "train" for q in batch))
        for q in restoration + batch:
            check("canonical_question:" + q["question_id"], q["point"] == point_to_gtp(q["point_index"])
                  and q["prompt"] == occupancy_prompt(q["to_play"], q["point"]))
        source_pins = imported_first_party()
        report["implementation_sha256"] = source_pins
        for name, digest in source_pins.items():
            copy = OUTPUT / "source-at-run" / name
            copy.parent.mkdir(parents=True, exist_ok=True)
            with copy.open("xb") as stream:
                stream.write((ROOT / name).read_bytes())
            check("source_copy:" + name, file_sha256(copy) == digest)
        check("producer_copy", file_sha256(source_copy) == report["script_sha256"])

        phase = "load_features"
        positions = {p["position_id"]: p for p in data["positions"]}
        features, feature_pins = {}, {}
        for pid in dict.fromkeys(q["position_id"] for q in restoration + batch):
            available()
            position = positions[pid]
            path = ROOT / position["source_feature_file"]
            digest = file_sha256(path)
            check("feature_source:" + pid, digest == position["feature_sha256"])
            with np.load(path, allow_pickle=False) as saved:
                arrays = [saved[f"trunk_after_block{i}"] for i in range(11)]
                check("feature_tensor:" + pid, all(a.shape == (361, 768) and a.dtype == np.float32
                      and np.isfinite(a).all() for a in arrays))
                features[pid] = torch.from_numpy(np.stack(arrays))
            check("feature_labels:" + pid, all(q["game_id"] == position["game_id"]
                  and q["to_play"] == position["to_play"]
                  and q["label"] == position["absolute_board_labels"][q["point_index"]]
                  for q in restoration + batch if q["position_id"] == pid))
            feature_pins[pid] = {"file": relative(path), "sha256": digest}
        report["feature_sources"] = feature_pins
        features_before = {pid: parameter_digest([value]) for pid, value in features.items()}
        report["feature_tensor_sha256_before"] = features_before

        phase = "load_model"
        available()
        model_path = ROOT / fit["local_model"]["directory"]
        check("base_config", file_sha256(model_path / "config.json") == fit["local_model"]["config_sha256"])
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        language = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                                                       dtype=torch.float32, attn_implementation="sdpa")
        language.gradient_checkpointing_disable()
        torch.manual_seed(42)
        model = AssistedPointGoQueen(language, LAYERS, 768, 8)
        named = adapter_parameters(model)
        check("full_adapter_layout", len(named) == 101 and sum(p.numel() for _, p in named) == 40438795)
        metadata = load_adapter_checkpoint(model, CHECKPOINT)
        check("fitted_APS_loaded", metadata == fit["checkpoint_metadata"]
              and metadata["engineering_only"] == "false" and metadata["updates"] == "180")
        model.requires_grad_(False)
        model.eval()
        check("frozen_CPU_FP32", all(not p.requires_grad and p.device.type == "cpu"
              and p.dtype == torch.float32 for p in model.parameters()))
        report["base_sha256_before"] = parameter_digest(model.base_parameters)
        check("base_matches_APS", report["base_sha256_before"] == fit["frozen_base_sha256_after"])
        report["adapter_sha256_before"] = parameter_digest(p for _, p in named)
        report["source_checkpoint_metadata"] = metadata

        phase = "assisted_restoration21"
        monitor = RoutingMonitor(model)
        for q in restoration:
            available()
            with monitor.expect([q["point_index"]], "evaluation"):
                rows = assisted_point_eval.score_questions(model, tokenizer, [q], features,
                                                            batch_size=1, deadline=deadline - 10.0)
            check("restoration_row:" + q["question_id"], len(rows) == 1)
            summarize_transfer(rows, [q])
            report["forward_calls"]["assisted_restoration"] += 1
            report["restoration_rows"].extend(rows)
            record()
        report["restoration"] = compare_restoration(report["restoration_rows"], saved_final, TOLERANCE)
        check("restoration_guard", restore_allows_validation(report["restoration"]))
        check("APS_binding_clear", binding_clear(model))
        report["assisted_adapter_calls"] = monitor.adapter_calls
        check("21x11_routed_calls", monitor.adapter_calls == 231)
        monitor.close()
        check("APS_observer_removed", not monitor.handles)

        phase = "in_place_conversion"
        signature = [(n, tuple(p.shape), id(p)) for n, p in named]
        tensor_before = {n: parameter_digest([p]) for n, p in named}
        rng = torch.get_rng_state().clone()
        converted = convert_to_natural(model)
        after = adapter_parameters(model)
        check("same_model_and_M_classes", converted is model and type(model) is GoQueen
              and all(type(layer.adapter) is GatedBoardAttention for layer in model.injected_layers))
        check("101_identity_names_shapes_unchanged", [(n, tuple(p.shape), id(p)) for n, p in after] == signature)
        check("conversion_bytes_unchanged", {n: parameter_digest([p]) for n, p in after} == tensor_before)
        check("conversion_no_RNG_draw", torch.equal(rng, torch.get_rng_state()))
        trainable = freeze_except_qk(model, named)
        expected = [(n, p) for n, p in named if n.endswith(QK_SUFFIXES)]
        trainable_ids = {id(p) for _, p in trainable}
        frozen = [(n, p) for n, p in named if id(p) not in trainable_ids]
        check("exact_QK44_selection", [(n, id(p)) for n, p in trainable] == [(n, id(p)) for n, p in expected]
              and len(trainable) == 44 and sum(p.numel() for _, p in trainable) == 20207616)
        check("44_trainable_57_frozen", len(frozen) == 57
              and {id(p) for p in model.parameters() if p.requires_grad} == trainable_ids
              and all(not p.requires_grad for _, p in frozen))
        report["trainable_names"] = [n for n, _ in trainable]
        report["ordered_adapter_signature"] = [{"name": n, "shape": list(p.shape), "dtype": str(p.dtype)} for n, p in named]

        @contextmanager
        def natural_sdpa(batch_size):
            original = F.scaled_dot_product_attention
            calls = []

            def observed(query, key, value, *positional, **kwargs):
                if not (key.ndim == 4 and key.shape[1] == 8 and key.shape[-2] == 361):
                    return original(query, key, value, *positional, **kwargs)
                mask = kwargs.get("attn_mask", positional[0] if positional else None)
                dropout = kwargs.get("dropout_p", positional[1] if len(positional) > 1 else 0.0)
                causal = kwargs.get("is_causal", positional[2] if len(positional) > 2 else False)
                require(query.ndim == value.ndim == 4 and query.shape[0] == batch_size
                        and key.shape == value.shape and mask is None and dropout == 0.0 and not causal,
                        "Natural SDPA must use all361 keys without a routing mask")
                require(all(bool(torch.isfinite(t).all()) for t in (query, key, value)), "Nonfinite natural QKV")
                out = original(query, key, value, *positional, **kwargs)
                require(bool(torch.isfinite(out).all()), "Nonfinite natural attention output")
                calls.append({"batch_size": batch_size, "query_tokens": query.shape[-2],
                              "key_tokens": key.shape[-2], "heads": key.shape[1], "mask": None})
                return out

            F.scaled_dot_product_attention = observed
            try:
                yield
                check("natural_11_adapter_SDPA", len(calls) == 11, batch_size=batch_size)
            finally:
                F.scaled_dot_product_attention = original
                report["natural_sdpa"].extend(calls)

        def clear_binding():
            return not model._bound and all(layer.expert_features is None
                and getattr(layer.adapter, "point_mask", None) is None
                and getattr(layer.adapter, "point_indices", None) is None for layer in model.injected_layers)

        phase = "natural_batch4_backward"
        available()
        encoded = [grounding_eval.encode_prefix(tokenizer, q["prompt"]) for q in batch]
        prefixes, label_ids = [x[0] for x in encoded], encoded[0][1]
        check("same_value_tokens", all(ids == label_ids for _, ids in encoded))
        report["batch_prefix_lengths"] = [len(p) for p in prefixes]
        report["mixed_length_batch_observed"] = len(set(report["batch_prefix_lengths"])) > 1
        report["padding_scope"] = "One real batch through the existing left-padding helper; no separate padding-equivalence experiment."
        # Forward receives only the board identity and label-independent token prefixes.
        forward_metadata = [{"position_id": q["position_id"]} for q in batch]
        feature_with_grad = features[batch[0]["position_id"]]
        report["feature_detach_probe_position_id"] = batch[0]["position_id"]
        feature_with_grad.requires_grad_(True)
        model.eval()  # Match the deterministic future runner; autograd remains enabled.
        model.zero_grad(set_to_none=True)
        with sdpa_kernel(SDPBackend.MATH), natural_sdpa(4):
            logits = grounding_eval._forward_logits(model, tokenizer, prefixes, forward_metadata, features)
            report["forward_calls"]["natural_batch4"] += 1
            check("natural_logits_finite", bool(torch.isfinite(logits).all()))
            targets = torch.tensor([label_ids[q["label"]] for q in batch], dtype=torch.long)
            loss = F.cross_entropy(logits, targets)
            check("value_CE_finite", bool(torch.isfinite(loss)))
            report["batch_full_vocabulary_value_CE"] = float(loss.detach())
            loss.backward()
            report["backward_passes"] += 1
        gradients = []
        for name, p in trainable:
            check("gradient_present_finite:" + name, p.grad is not None and bool(torch.isfinite(p.grad).all()))
            norm = float(p.grad.double().norm())
            check("gradient_norm_finite:" + name, math.isfinite(norm))
            gradients.append({"name": name, "l2_norm": norm,
                              "any_nonzero": bool(p.grad.ne(0).any())})
        report["trainable_gradients"] = gradients
        check("57_and_base_gradients_absent", all(p.grad is None and not p.requires_grad for _, p in frozen)
              and all(p.grad is None and not p.requires_grad for p in model.base_parameters))
        check("features_detached", feature_with_grad.grad is None)
        feature_with_grad.requires_grad_(False)
        check("natural_binding_cleared", clear_binding())
        model.zero_grad(set_to_none=True)
        model.eval()

        def single_score():
            available()
            with natural_sdpa(1):
                rows = grounding_eval.score_questions(model, tokenizer, [batch[0]], features,
                                                      batch_size=1, deadline=deadline - 10.0)
            check("natural_single_row", len(rows) == 1)
            report["forward_calls"]["natural_single"] += 1
            check("natural_single_finite", all(math.isfinite(x) for x in rows[0]["label_log_probs"]))
            return rows[0]

        phase = "checkpoint_save_and_load"
        before_save = single_score()
        metadata = {"variant": "m", "training_recipe": "aps_warm_qk", "engineering_only": "true",
                    "updates": "0", "source_aps_sha256": CHECKPOINT_SHA,
                    "trainable_names": json.dumps(report["trainable_names"], separators=(",", ":")),
                    "source_manifest_sha256": MANIFEST_SHA,
                    "source_features_sha256": content_sha256(feature_pins),
                    "source_implementation_sha256": content_sha256(source_pins),
                    "scope": "Zero-update engineering artifact from fitted APS; not a trained QK-only checkpoint"}
        report["checkpoint_metadata"] = metadata
        report["checkpoint_status"] = "saving"
        record()
        saved = save_full_checkpoint(model, named, checkpoint, metadata)
        report["checkpoint_save_result"] = saved
        check("full101_loader_verified", saved.get("loader_verified") is True and saved.get("count") == 101
              and saved.get("sha256") == file_sha256(checkpoint))
        report["checkpoint_status"] = "saved_and_loader_verified"
        after_save = single_score()
        delta = max(abs(a - b) for a, b in zip(before_save["label_log_probs"], after_save["label_log_probs"]))
        check("saved_same_model_score_restoration", delta == 0.0
              and before_save["forced_choice_label"] == after_save["forced_choice_label"]
              and before_save["unconstrained_token_id"] == after_save["unconstrained_token_id"], max_abs_logp_difference=delta)
        report["checkpoint_score_before"], report["checkpoint_score_after"] = before_save, after_save
        report["checkpoint"] = relative(checkpoint)

        phase = "final_checks"
        available()
        report["adapter_sha256_after"] = parameter_digest(p for _, p in named)
        report["base_sha256_after"] = parameter_digest(model.base_parameters)
        report["feature_tensor_sha256_after"] = {pid: parameter_digest([value]) for pid, value in features.items()}
        report["per_tensor_changed"] = {n: parameter_digest([p]) != tensor_before[n] for n, p in named}
        check("all101_unchanged_without_updates", report["adapter_sha256_after"] == report["adapter_sha256_before"]
              and not any(report["per_tensor_changed"].values()))
        check("base_unchanged", report["base_sha256_before"] == report["base_sha256_after"])
        check("features_unchanged", report["feature_tensor_sha256_after"] == features_before)
        check("freeze_flags_restored", {id(p) for p in model.parameters() if p.requires_grad} == trainable_ids)
        check("final_no_gradients_binding_clear", all(p.grad is None for p in model.parameters()) and clear_binding())
        check("sources_unchanged", imported_first_party() == source_pins
              and all(file_sha256(ROOT / name) == digest for name, digest in pins.items())
              and all(file_sha256(ROOT / item["file"]) == item["sha256"] for item in feature_pins.values()))
        check("exact_engineering_call_counts", report["forward_calls"] == {
              "assisted_restoration": 21, "natural_batch4": 1, "natural_single": 2}
              and report["backward_passes"] == 1 and len(report["natural_sdpa"]) == 33)
        report.update(status="complete", engineering_completed=True)
    except Exception as exc:
        report.update(status="stopped_budget" if isinstance(exc, TimeoutError) else "error",
                      engineering_completed=False, error={"phase": phase, "type": type(exc).__name__,
                      "message": str(exc).replace(str(ROOT), ".")})
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if monitor is not None:
            monitor.close()
        if feature_with_grad is not None:
            feature_with_grad.requires_grad_(False)
        if model is not None:
            model.zero_grad(set_to_none=True)
        report["last_phase"] = phase
        record()
        if time.perf_counter() > deadline:
            report.update(status="stopped_budget", engineering_completed=False)
            record()
    print(json.dumps({"status": report["status"], "engineering_completed": report["engineering_completed"],
                      "forward_calls": report["forward_calls"], "backward_passes": report["backward_passes"],
                      "optimizer_updates": 0, "seconds": time.perf_counter() - origin}), flush=True)
    return 0 if report["engineering_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
