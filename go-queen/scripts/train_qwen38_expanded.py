#!/usr/bin/env python3
"""Cloud-only five-epoch occupancy training draft; launch through run_qwen38_expanded.py.

Hardware choice changes only the frozen-base loader. No automatic fallback,
recipe change, retry, model download or local model execution. This entry has
not been run; a successful isolated capacity check is not five-epoch validation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import signal
import stat
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
STAGE = "qwen38_expanded_five_epoch"
MODEL = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
CONFIG_SHA = "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"
FIRST_PARTY = ("scripts/train_qwen38_expanded.py", "scripts/run_qwen38_expanded.py",
               "src/goqueen/qwen38_fusion.py", "src/goqueen/qwen38_training_runtime.py",
               "src/goqueen/expanded_training_data.py", "src/goqueen/epoch_schedule.py",
               "src/goqueen/cloud_training_state.py", "src/goqueen/cached_features.py",
               "src/goqueen/cloud_checkpoint_transport.py", "scripts/receive_cloud_checkpoint.py")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode()


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return [json_ready(item) for item in sorted(value, key=repr)]
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    return value


def write_json(path, value):
    temporary = Path(path).with_name(Path(path).name + ".part")
    with temporary.open("w") as stream:
        stream.write(json.dumps(json_ready(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def contained(relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Expected project-relative data/output path")
    result = (ROOT / path).resolve()
    if not result.is_relative_to(ROOT):
        raise ValueError("Path escapes project")
    return result


def cloud_asset(value):
    path = Path(value)
    if not path.is_absolute():
        return contained(value)
    path = path.resolve()
    if not path.is_relative_to(Path("/workspace")):
        raise ValueError("Model and asset record must be under cloud /workspace")
    return path


def learning_rate(update_number, total_updates):
    """One-based next Adam update; persisted global step is the scheduler state."""
    if not 1 <= update_number <= total_updates:
        raise ValueError("Learning-rate update index outside fixed plan")
    warmup = max(1, math.ceil(total_updates * 0.01))
    if update_number <= warmup:
        return 1e-4 * update_number / warmup
    progress = (update_number - warmup) / max(1, total_updates - warmup)
    return 1e-4 * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def example_passes(plan, completed_updates):
    if not 0 <= completed_updates <= plan.total_steps:
        raise ValueError("Cursor exceeds fixed plan")
    if completed_updates == plan.total_steps:
        return plan.example_passes
    epoch, within = divmod(completed_updates, plan.steps_per_epoch)
    return epoch * plan.examples + within * plan.batch_size


def cgroup_memory_estimate(limit, current, stat):
    """Conservative clean inactive-file credit, not a promise of reclaimability."""
    if limit < 0 or current < 0:
        raise ValueError("Invalid negative cgroup memory counter")
    required = {"inactive_file", "file", "file_dirty", "file_writeback", "shmem"}
    trusted = (isinstance(stat, dict) and required <= stat.keys()
               and all(type(stat[key]) is int and stat[key] >= 0 for key in required))
    credit = 0
    if trusted:
        # The dirty/writeback/shmem counters need not lie outside inactive_file.
        # Subtract all of them from that pool too; overlap may under-credit, but
        # must not turn a possibly all-dirty inactive pool into reclaim credit.
        excluded = stat["file_dirty"] + stat["file_writeback"] + stat["shmem"]
        credit = max(0, min(current, stat["inactive_file"] - excluded, stat["file"] - excluded))
    return {"limit_bytes": limit, "current_bytes": current,
            "raw_headroom_bytes": max(0, limit - current),
            "clean_inactive_file_credit_bytes": credit,
            "estimated_available_bytes": min(limit, max(0, limit - current + credit)),
            "stat": stat, "stat_credit_status": "conservative_clean_inactive" if trusted else "unknown_zero_credit",
            "formula": "min(limit,max(0,limit-current+max(0,min(current,inactive_file-dirty-writeback-shmem,file-dirty-writeback-shmem))))",
            "limitation": "estimate; reclaimability and concurrent resource usage are not guaranteed"}


def read_memory_counters(path):
    try:
        result = {key: int(value) for key, value in (line.split() for line in Path(path).read_text().splitlines())}
        return result if all(value >= 0 for value in result.values()) else None
    except (OSError, ValueError):
        return None


def read_oom_events():
    events = read_memory_counters("/sys/fs/cgroup/memory.events")
    if events is None or not {"oom", "oom_kill"} <= events.keys():
        return None
    return {name: events.get(name, 0) for name in ("oom", "oom_kill", "oom_group_kill")}


def available_host_memory():
    """Estimate allocatable Linux/cgroup RAM, including only clean inactive files."""
    fields = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        fields[key] = int(value.split()[0]) * 1024
    available = [fields["MemAvailable"]]
    evidence = {"system_mem_available_bytes": fields["MemAvailable"], "is_estimate": True}
    for name, limit_path, used_path in (
        ("cgroup_v2", "/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        ("cgroup_v1", "/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ):
        try:
            limit_text = Path(limit_path).read_text().strip()
            used = int(Path(used_path).read_text())
            if limit_text != "max":
                # v1 usage is hierarchical but stat layouts vary; give no credit
                # unless the active v2 controller supplies the exact counters.
                stat = read_memory_counters("/sys/fs/cgroup/memory.stat") if name == "cgroup_v2" else None
                record = cgroup_memory_estimate(int(limit_text), used, stat)
                evidence[name] = record
                available.append(record["estimated_available_bytes"])
        except (FileNotFoundError, PermissionError):
            pass
    return min(available), evidence


def advise_loaded_model_source_cache(model_dir, asset_hashes, feature_positions):
    """Once-only advice for the exact verified model files, never global cache.

    Caller must first finish asset SHA verification and fully resident CUDA
    model loading (no offload hooks). No file payload is read here: rehashing
    after advice would immediately warm the same page cache again. Success of
    the syscall is not proof of eviction; the unchanged preload guard follows.
    """
    model_dir = Path(model_dir).resolve()
    files, sizes = asset_hashes["files"], asset_hashes["file_sizes"]
    if (len(files) != 29 or set(files) != set(sizes)
            or asset_hashes["model_id"] != MODEL or asset_hashes["revision"] != REVISION
            or type(feature_positions) is not int or feature_positions < 1):
        raise ValueError("Targeted cache advice requires the exact verified model manifest")
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        raise RuntimeError("Targeted cache advice requires Linux posix_fadvise")
    reserve = 8 * 1024**3
    required = feature_positions * 11 * 361 * 768 * 4 + reserve
    evidence = {"status": "validating", "advice": "POSIX_FADV_DONTNEED", "offset": 0,
                "length": 0, "requested_files": len(files), "feature_positions": feature_positions,
                "reserve_bytes": reserve, "required_preload_bytes": required, "files": [],
                "content_read_or_written": False, "global_cache_operation": False,
                "limitation": "advice is not guaranteed eviction; metadata checks are not a new content SHA verification"}
    handles = []
    def metadata(fd):
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("Cache advice target must be a regular file")
        return {"device": info.st_dev, "inode": info.st_ino, "bytes": info.st_size,
                "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns}
    try:
        # Validate every target before advising any target. File descriptors pin
        # these inodes even if a pathname is subsequently changed.
        for name in sorted(files):
            if (not name or Path(name).name != name or name in (".", "..")
                    or type(sizes[name]) is not int or sizes[name] < 0):
                raise ValueError("Unexpected model file name or recorded size")
            fd = os.open(model_dir / name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            handles.append(fd)
            before = metadata(fd)
            if before["bytes"] != sizes[name]:
                raise ValueError("Model file size changed before targeted cache advice")
            evidence["files"].append({"name": name, "before": before, "advice_returned": False})
        available, sources = available_host_memory()
        evidence.update(status="advising", available_before_bytes=available, before_sources=sources,
                        oom_events_before=read_oom_events(), meets_preload_guard_before=available >= required)
        for fd, record in zip(handles, evidence["files"]):
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            record["advice_returned"] = True
            record["after"] = metadata(fd)
            if record["after"] != record["before"]:
                raise ValueError("Model file metadata changed during targeted cache advice")
        available, sources = available_host_memory()
        after_oom = read_oom_events()
        evidence.update(status="advised_once", available_after_bytes=available, after_sources=sources,
                        oom_events_after=after_oom, meets_preload_guard_after=available >= required)
        before_oom = evidence["oom_events_before"]
        if before_oom is not None and after_oom is not None and any(after_oom[k] > before_oom[k] for k in before_oom):
            raise MemoryError("Cgroup OOM event increased during targeted source-cache advice")
        return evidence
    except BaseException as error:
        evidence["status"] = "failed"
        error.source_cache_advice_evidence = evidence
        raise
    finally:
        for fd in handles:
            os.close(fd)


def preload_feature_cache(cache):
    """Keep every feature once without changing the fixed shuffled question order."""
    if cache.capacity != len(cache):
        raise ValueError("Explicit cache capacity must equal the complete feature position count")
    tensor_bytes = len(cache) * 11 * 361 * 768 * 4
    reserve = 8 * 1024**3
    before, before_sources = available_host_memory()
    oom_before = read_oom_events()
    evidence = {"position_count": len(cache), "tensor_bytes": 0, "required_tensor_bytes": tensor_bytes,
                "reserve_bytes": reserve, "available_before_bytes": before, "before_sources": before_sources,
                "oom_events_before": oom_before, "checkpoints": [], "status": "preflight",
                "scope": "estimated allocatable RAM; canonical tensors plus unchanged8GiB reserve, not an allocation guarantee"}
    def fail(message):
        evidence["status"] = "failed"
        error = MemoryError(message)
        error.memory_evidence = evidence
        raise error
    if before < tensor_bytes + reserve:
        fail("Insufficient estimated post-base RAM for all feature tensors plus 8 GiB reserve")
    started, actual_bytes = time.monotonic(), 0
    evidence["status"] = "preloading"
    try:
        for count, identity in enumerate(cache, 1):
            value = cache[identity]
            actual_bytes += value.numel() * value.element_size()
            evidence["tensor_bytes"] = actual_bytes
            if count % 64 == 0 or count == len(cache):
                after, after_sources = available_host_memory()
                oom_after = read_oom_events()
                remaining = max(0, tensor_bytes - actual_bytes)
                evidence["checkpoints"].append({"positions_loaded": count, "remaining_tensor_bytes": remaining,
                    "estimated_available_bytes": after, "sources": after_sources, "oom_events": oom_after})
                evidence.update(available_after_bytes=after, after_sources=after_sources, oom_events_after=oom_after)
                if oom_before is not None and oom_after is not None and any(oom_after[key] > oom_before[key] for key in oom_before):
                    fail("Cgroup OOM event increased during feature preloading")
                if after < remaining + reserve:
                    fail("Feature preload estimate no longer covers remaining tensors plus8GiB reserve")
        if actual_bytes != tensor_bytes or cache.stats()["evictions"] != 0:
            fail("Full feature preloading size/retention contract failed")
    except Exception as error:
        evidence["status"] = "failed"
        error.memory_evidence = evidence
        raise
    evidence.update(status="preloaded", seconds=time.monotonic() - started,
                    oom_monitoring_available=oom_before is not None and evidence.get("oom_events_after") is not None)
    return evidence


def base_digest(torch, named_base_parameters):
    """Hash actual resident base parameters and NF4 quantization-state tensors."""
    digest = hashlib.sha256()
    def add(name, value):
        digest.update(name.encode() + b"\0")
        if isinstance(value, torch.Tensor):
            digest.update(str(value.dtype).encode() + canonical(list(value.shape)))
            digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value):
                add(str(key), value[key])
        else:
            digest.update(repr(value).encode())
    for name, parameter in named_base_parameters:
        add(name, parameter)
        quant_state = getattr(parameter, "quant_state", None)
        if quant_state is not None:
            add(name + ".quant_state", quant_state.as_dict(packed=False))
    return digest.hexdigest()


def publish_durable_boundary(output, record, config, transport, report):
    """A local commit is not durable until its exact remote receipt is verified."""
    report["last_local_checkpoint"] = str(record.path.relative_to(output))
    report["last_local_updates"] = record.metadata["cursor"]["completed_updates"]
    report["checkpoint_publish_pending"] = True
    ack = transport.publish_checkpoint(output, record, config)
    ack = transport.validate_ack(ack, output, record, config)
    report.update(last_durable_checkpoint=str(record.path.relative_to(output)),
                  last_durable_updates=record.metadata["cursor"]["completed_updates"],
                  last_durable_ack=ack, checkpoint_publish_pending=False)
    return ack


def prune_periodic_checkpoints(checkpoint_root, newest, identities, receipt_directory, state,
                               durable_guard=None):
    """Prune only checked local payloads whose exact persistent receipt exists."""
    if durable_guard is None:
        raise ValueError("Periodic pruning requires verified persistent checkpoint receipts")
    verified = state.resolve_checkpoint(checkpoint_root, identities)
    if verified.path != newest.path or verified.manifest_sha256 != newest.manifest_sha256:
        raise ValueError("Newest checkpoint must be verified before periodic retention")
    replacement_ack = durable_guard(verified)
    periodic = []
    for directory in checkpoint_root.iterdir():
        if not directory.is_dir() or not directory.name.startswith("step-") or not directory.name.endswith("-periodic"):
            continue
        if directory.is_symlink() or directory.parent != checkpoint_root:
            raise ValueError("Unexpected periodic checkpoint target")
        raw = (directory / "MANIFEST.json").read_bytes()
        hashed = hashlib.sha256(raw).hexdigest()
        manifest = json.loads(raw)
        if ((directory / "READY").read_text().strip() != hashed
                or manifest["metadata"]["identities"] != identities
                or manifest["name"] != directory.name):
            raise ValueError("Cannot prune a checkpoint with different/unverified metadata")
        actual_files = {str(path.relative_to(directory)): path for path in directory.rglob("*") if path.is_file()}
        expected = manifest["files"]
        if set(actual_files) != set(expected) | {"MANIFEST.json", "READY"}:
            raise ValueError("Periodic checkpoint file inventory changed")
        for name, info in expected.items():
            path = actual_files[name]
            if path.is_symlink() or path.stat().st_size != info["bytes"] or sha(path) != info["sha256"]:
                raise ValueError("Cannot prune changed periodic checkpoint bytes")
        record = state.CheckpointRecord(directory, manifest["metadata"], expected, hashed)
        periodic.append((manifest["metadata"]["cursor"]["completed_updates"], directory, hashed, expected, record))
    removed = []
    for count, directory, digest, files, checkpoint_record in sorted(periodic)[:-2]:
        obsolete_ack = durable_guard(checkpoint_record)
        receipt = receipt_directory / (directory.name + ".json")
        receipt_directory.mkdir(exist_ok=True)
        record = {"checkpoint": directory.name, "completed_updates": count,
                  "manifest_sha256": digest, "payload_files": files,
                  "replacement_verified": newest.path.name,
                  "replacement_durable_unit_sha256": replacement_ack["unit_sha256"],
                  "obsolete_durable_unit_sha256": obsolete_ack["unit_sha256"],
                  "discard_directory": directory.name + ".discarding",
                  "reason": "retain initial, every epoch endpoint, and latest two periodic boundaries",
                  "status": "verified_before_removal"}
        if receipt.exists():
            previous = json.loads(receipt.read_text())
            if any(previous.get(key) != record[key] for key in
                    ("checkpoint", "completed_updates", "manifest_sha256", "payload_files", "discard_directory", "status")):
                raise ValueError("Existing prune receipt does not authorize this same obsolete boundary")
        else:
            with receipt.open("x") as stream:
                stream.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        # Move out of the committed root first. An interruption during deletion
        # must not leave a half-deleted old checkpoint blocking a newer restore.
        discard = receipt_directory / record["discard_directory"]
        if discard.exists():
            raise ValueError("Discard destination unexpectedly exists")
        os.rename(directory, discard)
        shutil.rmtree(discard)
        record["status"] = "removed"
        write_json(receipt, record)
        removed.append(directory.name)
    return removed


def arguments():
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
    parser.add_argument("--persistence-config", required=True,
                        help="Private cloud path to the fixed checkpoint receiver configuration")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--attempt", required=True, help=argparse.SUPPRESS)
    parser.add_argument("--outer-start", required=True, type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if platform.system() != "Linux" or not ROOT.is_relative_to(Path("/workspace")):
        parser.error("Cloud Linux /workspace execution required; no local model use")
    if args.expected_train < 100000 or args.cache_capacity < 1 or not math.isfinite(args.outer_start):
        parser.error("Expected training pool must contain at least 100000 distinct questions")
    return args


def load_base(torch, args, model_dir, report):
    from transformers import Qwen3_5ForCausalLM, BitsAndBytesConfig
    expected_gpus = 1 if args.hardware == "nf4-single-gpu" else 2
    if torch.cuda.device_count() != expected_gpus:
        raise ValueError("Visible device count differs from the fixed hardware recipe")
    if any("5090" not in torch.cuda.get_device_name(i) for i in range(expected_gpus)):
        raise ValueError("Only the requested RTX 5090 hardware is permitted")
    kwargs = {"local_files_only": True, "dtype": torch.bfloat16,
              "attn_implementation": "sdpa", "output_loading_info": True}
    if args.hardware == "bf16-two-gpu":
        kwargs["device_map"] = {"model.embed_tokens": 0, "model.rotary_emb": 0,
            "model.norm": 1, "lm_head": 1, **{f"model.layers.{i}": 0 if i < 32 else 1 for i in range(64)}}
    else:
        kwargs["device_map"] = {"": 0}
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=False,
            llm_int8_skip_modules=["lm_head", "in_proj_a", "in_proj_b"],
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_storage=torch.uint8)
    base, info = Qwen3_5ForCausalLM.from_pretrained(model_dir, **kwargs)
    report["loading_info"] = json_ready(info)
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("Unresolved text-only model loading diagnostics")
    if (base.config.model_type != "qwen3_5_text" or base.config.hidden_size != 5120
            or base.config.num_hidden_layers != 64 or base.config.vocab_size != 248320):
        raise ValueError("Loaded base has the wrong architecture")

    def check_hook(hook):
        if bool(getattr(hook, "no_grad", False)) or bool(getattr(hook, "offload", False)):
            raise ValueError("Offload/no-grad hook would break the selected resident training path")
        for nested in getattr(hook, "hooks", ()):
            check_hook(nested)

    for module in base.modules():
        if hasattr(module, "_hf_hook"):
            check_hook(module._hf_hook)
    if any(p.device.type != "cuda" for p in base.parameters()):
        raise ValueError("All frozen base parameters must be resident on CUDA")
    if args.hardware == "bf16-two-gpu":
        if any(p.dtype != torch.bfloat16 for p in base.parameters()):
            raise ValueError("BF16 recipe has non-BF16 base parameters")
    else:
        import bitsandbytes as bnb
        linears = [module for module in base.modules() if isinstance(module, bnb.nn.Linear4bit)]
        if not linears or not getattr(base, "is_loaded_in_4bit", False):
            raise ValueError("NF4 recipe was not realized")
        if any(module.compute_dtype != torch.bfloat16 or module.weight.quant_state is None
               or module.weight.quant_type != "nf4" or module.weight.quant_state.nested for module in linears):
            raise ValueError("NF4/BF16 quantized-linear contract mismatch")
        inventory = []
        skipped = []
        for name, module in base.named_modules():
            if isinstance(module, torch.nn.Linear):
                quantized = isinstance(module, bnb.nn.Linear4bit)
                inventory.append({"name": name, "class": type(module).__name__, "quantized": quantized,
                                  "weight_dtype": str(module.weight.dtype), "shape": list(module.weight.shape)})
                if name.rsplit(".", 1)[-1] in {"lm_head", "in_proj_a", "in_proj_b"}:
                    if quantized or module.weight.dtype != torch.bfloat16:
                        raise ValueError("Requested skipped projection was quantized or has wrong dtype")
                    skipped.append(name)
        if len(linears) != 400 or len(skipped) != 97:
            raise ValueError("Expected400 NF4 and97 explicitly skipped head/a/b projections")
        report["quantized_linear_count"] = len(linears)
        report["linear_module_inventory"] = inventory
        report["quantization_config"] = json_ready(kwargs["quantization_config"].to_dict())
    report["explicit_device_map"] = kwargs["device_map"]
    return base


def execute(args, report):
    output, attempt = contained(args.output), contained(args.attempt)
    if attempt.parent != output / "attempts":
        raise ValueError("Attempt must be inside this run's attempts directory")
    control = json.loads((attempt / "process-start.json").read_text())
    if (control["launcher_pid"] != os.getppid() or control["outer_start_monotonic"] != args.outer_start
            or control["runner_sha256"] != sha(__file__)):
        raise ValueError("Launcher/runner identity mismatch")
    source_pins = {name: sha(ROOT / name) for name in FIRST_PARTY}
    for name in FIRST_PARTY:
        target = attempt / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    sys.path.insert(0, str(ROOT / "src"))
    from goqueen.expanded_training_data import load_expanded_corpus
    from goqueen import epoch_schedule as schedule
    from goqueen import cloud_training_state as state
    from goqueen import cloud_checkpoint_transport as transport
    from goqueen.cached_features import BoundedFeatureCache
    from goqueen import qwen38_training_runtime as runtime
    persistence_config_path = cloud_asset(args.persistence_config)
    persistence_config = transport.load_config(persistence_config_path)
    persistence_config_sha = sha(persistence_config_path)
    persistence_identity = transport.public_identity(persistence_config)
    if persistence_identity["receiver_sha256"] != source_pins["scripts/receive_cloud_checkpoint.py"]:
        raise ValueError("Configured persistent receiver differs from the pinned receiver source")
    # This empty file is part of the initial checkpoint publication. Later
    # snapshots are flushed at the exact optimizer boundary before publishing.
    steps_path = attempt / "steps.jsonl"
    with steps_path.open("x") as empty_journal:
        empty_journal.flush()
        os.fsync(empty_journal.fileno())
    sources, features = contained(args.source_manifest), contained(args.feature_manifest)
    splits, feature_manifest, corpus = load_expanded_corpus(ROOT, sources, features)
    if len(splits["train"]) != args.expected_train:
        raise ValueError("Actual complete training pool differs from the declared count")
    positions = {item["position_id"]: item for item in feature_manifest["positions"]}
    for rows in splits.values():
        for q in rows:
            q["ply"] = positions[q["position_id"]]["ply"]
    train = {q["question_id"]: q for q in splits["train"]}
    train_ids = schedule.validate_ids(train)
    plan = schedule.make_plan(len(train_ids), 16, 5)
    legacy_path = contained(args.legacy_manifest)
    legacy_manifest = json.loads(legacy_path.read_text())
    legacy_rows = legacy_manifest["questions"]
    legacy_features_path = contained(legacy_manifest["feature_manifest"])
    legacy_features = json.loads(legacy_features_path.read_text())
    if (len(legacy_rows) != 54 or len({row["question_id"] for row in legacy_rows}) != 54
            or set(train_ids).intersection(row["question_id"] for row in legacy_rows)):
        raise ValueError("Legacy secondary set must contain exactly54 distinct non-training IDs")
    legacy_positions = {p["position_id"]: p for p in legacy_features["positions"]}
    for q in legacy_rows:
        p = legacy_positions[q["position_id"]]
        if (type(q["label"]) is not int or q["label"] not in (0, 1, 2)
                or q["label_name"] != ("empty", "black", "white")[q["label"]]
                or type(q["point_index"]) is not int or not 0 <= q["point_index"] < 361
                or any(q[field] != p[field] for field in ("game_id", "ply", "to_play"))):
            raise ValueError("Legacy secondary prepared row/feature metadata mismatch")
        coordinate = "ABCDEFGHJKLMNOPQRST"[q["point_index"] % 19] + str(19 - q["point_index"] // 19)
        if q["coordinate"] != coordinate or f"判断{coordinate}点" not in q["prompt"]:
            raise ValueError("Legacy point and new prompt differ")
        q["ply"] = p["ply"]
    recipe = {"model": MODEL, "revision": REVISION, "hardware": args.hardware,
        "seed": 42, "cpu_threads": 4, "epochs": 5, "microbatch": args.microbatch, "effective_batch": 16,
        "feature_cache_capacity": args.cache_capacity, "feature_cache_ram_reserve_bytes": 8 * 1024**3,
        "attention_width": 1024, "heads": 8, "expert_width": 768, "adapter_dtype": "float32",
        "base_precision": "NF4/BF16 compute; numerically distinct from BF16" if args.hardware.startswith("nf4") else "bfloat16",
        "optimizer": "torch.optim.AdamW", "peak_lr": 1e-4, "betas": [0.9, 0.999],
        "eps": 1e-8, "weight_decay": 0.0, "clip_norm": 1.0, "warmup_fraction": 0.01,
        "cosine_floor_fraction": 0.1, "gradient_checkpointing": "non-reentrant",
        "nf4_double_quant": False, "quantization_skip_modules": ["lm_head", "in_proj_a", "in_proj_b"],
        "loss_normalization_denominator": 16,
        "persistence": persistence_identity,
        "checkpoint_publication": "synchronous exact receiver ACK before next update or local pruning",
        "schedule": "one-based learning_rate(update_number,total_updates)",
        "checkpoint_every_updates": 500, "checkpoint_retention": "initial, five epoch endpoints, latest two periodic boundaries",
        "endpoint": "fifth full epoch, no best-epoch selection", "drop_last": False}
    model_dir, assets_path = cloud_asset(args.model_dir), cloud_asset(args.asset_record)
    assets = json.loads(assets_path.read_text())
    if (assets.get("qwen_model_id") != MODEL or assets.get("qwen_revision") != REVISION
            or assets.get("qwen_download_complete") is not True
            or Path(assets["qwen_local_remote_path"]).resolve() != model_dir
            or sha(model_dir / "config.json") != CONFIG_SHA):
        raise ValueError("Cloud base asset identity mismatch")
    for name, size in assets["qwen_file_sizes"].items():
        if Path(name).name != name or (model_dir / name).stat().st_size != size:
            raise ValueError("Recorded cloud model file size changed")
    asset_hash_path = cloud_asset(args.asset_hashes)
    asset_hashes = json.loads(asset_hash_path.read_text())
    if (asset_hashes["model_id"] != MODEL or asset_hashes["revision"] != REVISION
            or asset_hashes["asset_record_sha256"] != sha(assets_path)
            or set(asset_hashes["files"]) != set(assets["qwen_file_sizes"])):
        raise ValueError("Cloud model hash record does not match the original asset identity/inventory")
    for name, value in asset_hashes["files"].items():
        if sha(model_dir / name) != value:
            raise ValueError("Cloud model source file SHA256 changed")
    questions_path = contained(json.loads(sources.read_text())["questions_file"])
    pin_files = {**{name: ROOT / name for name in FIRST_PARTY},
                 "source_manifest": sources, "feature_manifest": features,
                 "legacy_manifest": legacy_path, "legacy_feature_manifest": legacy_features_path,
                 "cloud_asset_hash_record": asset_hash_path,
                 "questions": questions_path,
                 "cloud_model_config": model_dir / "config.json", "cloud_asset_record": assets_path}
    file_pins = {name: sha(path) for name, path in pin_files.items()}
    if any(file_pins[name] != source_pins[name] or sha(attempt / "source" / name) != source_pins[name]
           for name in FIRST_PARTY):
        raise ValueError("First-party source changed while capturing its run snapshot")
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "accelerate", "safetensors", "numpy", "causal-conv1d", "flash-linear-attention")}
    if args.hardware.startswith("nf4"):
        versions["bitsandbytes"] = importlib.metadata.version("bitsandbytes")
    if versions["transformers"] != "5.19.0" or versions["accelerate"] != "1.15.0":
        raise ValueError("Different model/dispatch versions require a new reviewed recipe")
    report.update(recipe=recipe, corpus=corpus, plan=asdict(plan), versions=versions,
                  input_source_sha256=file_pins, completed_updates=0, example_passes=0,
                  epochs_completed=0, evaluations=[], historical_54_evaluated=False)
    import torch
    torch.set_num_threads(4)
    import numpy as np
    from torch.nn import functional as F
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer
    from goqueen.qwen38_fusion import Qwen38GoQueen
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    encoded, label_ids = runtime.encode_questions(tokenizer, splits["train"] + splits["val"] + legacy_rows)
    encoding_hash = hashlib.sha256()
    for identity in sorted(encoded):
        encoding_hash.update(canonical([identity, encoded[identity]]) + b"\n")
    identities = {"recipe": recipe, "corpus": corpus, "versions": versions,
                  "files": file_pins, "schedule_fingerprint": schedule.fingerprint(train_ids, 42, 16, 5),
                  "encoded_questions_sha256": encoding_hash.hexdigest(), "label_token_ids": label_ids}
    run_identity = output / "RUN.json"
    if args.resume:
        existing_identities = json.loads(run_identity.read_text())
        expected_base_digest = existing_identities.get("actual_base_sha256")
        if {key: value for key, value in existing_identities.items() if key != "actual_base_sha256"} != identities:
            raise ValueError("Run recipe/data/source/tokenization changed; resume refused")
        identities = existing_identities
        # Validate every committed payload byte before spending time loading 27B.
        resume_record = state.resolve_checkpoint(output / "checkpoints", identities)
        # Reconcile the newest local boundary before loading the base model.
        # This may resend an immutable unit after an interrupted publication;
        # it does not replay an optimizer update or trust an unacked local save.
        publish_durable_boundary(output, resume_record, persistence_config, transport, report)
        if resume_record.metadata["cursor"]["completed_updates"] == plan.total_steps:
            raise ValueError("Run already has the final committed boundary; do not reload/re-evaluate it")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base = load_base(torch, args, model_dir, report)
    named_base_parameters = tuple(base.named_parameters())
    base_parameters = tuple(base.parameters())
    actual_base_digest = base_digest(torch, named_base_parameters)
    if args.resume and actual_base_digest != expected_base_digest:
        raise ValueError("Resident base/quantization-state digest differs from saved identity")
    identities["actual_base_sha256"] = actual_base_digest
    report["actual_base_sha256_before"] = actual_base_digest
    if not args.resume:
        with run_identity.open("x") as stream:
            stream.write(json.dumps(identities, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    report["identities_sha256"] = hashlib.sha256(canonical(identities)).hexdigest()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = Qwen38GoQueen(base, attention_width=1024, expert_width=768, heads=8,
                          adapter_dtype=torch.float32, position_device="cuda:0")
    adapters = model.adapter_named_parameters()
    if (len(adapters) != 101 or sum(p.numel() for p in adapters.values()) != 132803595
            or any(p.dtype != torch.float32 or not p.requires_grad for p in adapters.values())):
        raise ValueError("Expected complete fresh FP32 narrow adapter set")
    base_versions = tuple(p._version for p in base_parameters)
    initial_digest = hashlib.sha256()
    for name, parameter in adapters.items():
        initial_digest.update(name.encode() + b"\0" + canonical(list(parameter.shape)) + b"\0")
        initial_digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    report["fresh_initial_adapter_sha256"] = initial_digest.hexdigest()
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(list(adapters.values()), lr=1e-4, betas=(0.9, 0.999),
                                  eps=1e-8, weight_decay=0.0, foreach=False)
    ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    if len(ids) != 101 or set(ids) != {id(p) for p in adapters.values()}:
        raise ValueError("Optimizer contains wrong parameters")
    cache = BoundedFeatureCache(features.parent, feature_manifest["positions"], capacity=args.cache_capacity)
    report["model_source_cache_advice"] = advise_loaded_model_source_cache(model_dir, asset_hashes, len(cache))
    report["feature_preload"] = preload_feature_cache(cache)
    legacy_cache = BoundedFeatureCache(legacy_features_path.parent, legacy_features["positions"],
                                       capacity=len(legacy_features["positions"]))
    checkpoint_root = output / "checkpoints"
    evaluations, completed, passes, epoch_seen = [], 0, 0, []
    last_checkpoint = None
    journal = None

    def verify_sources():
        if any(sha(pin_files[name]) != value for name, value in file_pins.items()):
            raise ValueError("Pinned source/input bytes changed during run")
        if sha(persistence_config_path) != persistence_config_sha:
            raise ValueError("Private persistent receiver configuration changed during this attempt")
        if any(p.requires_grad or p.grad is not None for p in base_parameters):
            raise ValueError("Frozen base became trainable or acquired gradients")
        if tuple(p._version for p in base_parameters) != base_versions:
            raise ValueError("Base parameter version counters changed")
        if model._bound or any(layer.expert_features is not None for layer in model.injected_layers):
            raise ValueError("Leaked expert feature binding")

    def rng_state():
        return {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()}

    def save_boundary(name):
        nonlocal last_checkpoint
        verify_sources()
        if journal is not None:
            journal.flush()
            os.fsync(journal.fileno())
        if len(optimizer.state) != (101 if completed else 0):
            raise ValueError("Adam state must cover all101 updated parameters (or be empty at initialization)")
        for parameter in adapters.values():
            if not completed:
                break
            entry = optimizer.state[parameter]
            if (set(entry) != {"step", "exp_avg", "exp_avg_sq"}
                    or float(entry["step"]) != completed
                    or any(entry[key].shape != parameter.shape or not bool(torch.isfinite(entry[key]).all())
                           for key in ("exp_avg", "exp_avg_sq"))):
                raise ValueError("Adam step/moment shape/finite state mismatch")
        location = schedule.locate(plan, completed)
        metadata = {"identities": identities, "cursor": {"epoch_index": location.epoch,
                    "next_batch_index": location.step_in_epoch, "completed_updates": completed},
                    "example_passes": passes, "epoch_prefix_sha256": schedule.order_digest(epoch_seen),
                    "evaluations": evaluations, "initial_adapter_sha256": report["fresh_initial_adapter_sha256"],
                    "scheduler": {"completed_updates": completed, "total_updates": plan.total_steps,
                                  "last_lr": optimizer.param_groups[0]["lr"]},
                    "attempt": str(attempt.relative_to(output)),
                    "attempt_wall_seconds": time.monotonic() - args.outer_start}
        captured_rng = rng_state()
        last_checkpoint = state.atomic_checkpoint(checkpoint_root, name,
            lambda directory: runtime.save_payload(torch, save_file, model, optimizer, directory,
                              global_step=completed, rng_state=captured_rng), metadata)
        publish_durable_boundary(output, last_checkpoint, persistence_config, transport, report)
        removed = prune_periodic_checkpoints(checkpoint_root, last_checkpoint, identities,
                                              output / "retention-receipts", state, durable_guard)
        report.setdefault("pruned_periodic_checkpoints", []).extend(removed)

    def durable_guard(record):
        ack_path = output / "acks" / (record.path.name + ".json")
        return transport.validate_ack(json.loads(ack_path.read_text()), output, record, persistence_config)

    if args.resume:
        last_checkpoint = resume_record
        metadata = last_checkpoint.metadata
        cursor = metadata["cursor"]
        completed = cursor["completed_updates"]
        location = schedule.locate(plan, completed)
        if cursor != {"epoch_index": location.epoch, "next_batch_index": location.step_in_epoch,
                      "completed_updates": completed}:
            raise ValueError("Saved cursor disagrees with deterministic schedule")
        passes = example_passes(plan, completed)
        if metadata["example_passes"] != passes or metadata["initial_adapter_sha256"] != report["fresh_initial_adapter_sha256"]:
            raise ValueError("Saved exposure count or fresh initialization differs")
        if not location.finished:
            epoch_seen = schedule.epoch_order(train_ids, 42, location.epoch)[:location.step_in_epoch * 16]
        if metadata["epoch_prefix_sha256"] != schedule.order_digest(epoch_seen):
            raise ValueError("Saved actual epoch prefix differs from expected sequence")
        saved = load_file(str(last_checkpoint.path / "adapter.safetensors"), device="cpu")
        if set(saved) != set(adapters) or any(saved[name].shape != p.shape or saved[name].dtype != p.dtype
                                            or not bool(torch.isfinite(saved[name]).all()) for name, p in adapters.items()):
            raise ValueError("Saved complete adapter names/shapes/dtypes/finite contract failed")
        with torch.no_grad():
            for name, parameter in adapters.items():
                parameter.copy_(saved[name].to(parameter.device))
        del saved
        # Only own same-run committed bytes, verified by resolve_checkpoint above.
        trusted = torch.load(last_checkpoint.path / "training-state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(trusted["optimizer"])
        if (metadata["scheduler"]["completed_updates"] != completed
                or metadata["scheduler"]["total_updates"] != plan.total_steps
                or optimizer.param_groups[0]["lr"] != metadata["scheduler"]["last_lr"]):
            raise ValueError("Optimizer/scheduler cursor mismatch")
        random.setstate(trusted["rng"]["python"])
        np.random.set_state(trusted["rng"]["numpy"])
        torch.set_rng_state(trusted["rng"]["torch_cpu"])
        torch.cuda.set_rng_state_all(trusted["rng"]["torch_cuda"])
        evaluations = list(metadata["evaluations"])
        for evaluation in evaluations:
            path = output / evaluation["file"]
            if not path.resolve().is_relative_to(output) or sha(path) != evaluation["sha256"]:
                raise ValueError("Committed evaluation evidence missing/changed")
        report.update(resumed_from=str(last_checkpoint.path.relative_to(output)),
                      resumed_durable_updates=completed,
                      completed_updates=completed, example_passes=passes,
                      epochs_completed=completed // plan.steps_per_epoch,
                      last_durable_updates=completed, evaluations=evaluations,
                      resume_limit="Any uncommitted updates in prior attempts are discarded and may be recomputed")

    def evaluate(name):
        model.eval()
        rows, secondary_rows, start = [], [], time.monotonic()
        target = attempt / ("evaluation-" + name + ".json")
        with (attempt / ("evaluation-" + name + ".jsonl")).open("x") as stream, torch.no_grad():
            for offset in range(0, len(splits["val"]), args.microbatch):
                questions = splits["val"][offset:offset + args.microbatch]
                expert = runtime.feature_batch(torch, questions, cache)
                with model.bind_expert(expert):
                    report["evaluation_forward_attempts"] += 1
                    logits = runtime.forward_logits(torch, model, tokenizer, questions, encoded, cache, "cuda:0")
                    report["evaluation_forward_returns"] += 1
                scored = runtime.score_batch(torch, logits, questions, label_ids)
                rows.extend(scored)
                for row in scored:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                del expert, logits
            for offset in range(0, len(legacy_rows), args.microbatch):
                questions = legacy_rows[offset:offset + args.microbatch]
                expert = runtime.feature_batch(torch, questions, legacy_cache)
                with model.bind_expert(expert):
                    report["evaluation_forward_attempts"] += 1
                    logits = runtime.forward_logits(torch, model, tokenizer, questions, encoded, legacy_cache, "cuda:0")
                    report["evaluation_forward_returns"] += 1
                secondary_rows.extend(runtime.score_batch(torch, logits, questions, label_ids))
                del expert, logits
        if [row["question_id"] for row in rows] != [q["question_id"] for q in splits["val"]]:
            raise ValueError("Evaluation coverage/order mismatch")
        groups = {field: {str(value): runtime.aggregate([row for row in rows if row[field] == value])
                          for value in sorted({row[field] for row in rows}, key=str)}
                  for field in ("game_id", "to_play", "ply")}
        result = {"name": name, "completed_updates": completed, "coverage_complete": True,
                  "summary": runtime.aggregate(rows), "groups": groups, "rows": rows,
                  "seconds": time.monotonic() - start,
                  "historical_54": {"secondary_nonblind_reference": True,
                      "summary": runtime.aggregate(secondary_rows), "rows": secondary_rows}}
        if [row["question_id"] for row in secondary_rows] != [q["question_id"] for q in legacy_rows]:
            raise ValueError("Legacy evaluation coverage/order mismatch")
        write_json(target, result)
        reference = {"name": name, "file": str(target.relative_to(output)), "sha256": sha(target),
                     "completed_updates": completed, "summary": result["summary"],
                     "historical_54_summary": result["historical_54"]["summary"]}
        evaluations.append(reference)
        report["evaluations"] = evaluations
        model.train()

    if not args.resume:
        evaluate("initial")
        save_boundary("step-00000000-initial")
    report["restored_or_initial_updates"] = completed
    report["restored_or_initial_example_passes"] = passes
    report["last_durable_checkpoint"] = str(last_checkpoint.path.relative_to(output))
    model.train()
    with steps_path.open("a") as journal:
        for step in schedule.steps(train_ids, 42, 16, 5, start_step=completed):
            if step.global_step != completed:
                raise ValueError("Schedule cursor skipped or repeated a committed update")
            start = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            lr = learning_rate(completed + 1, plan.total_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            loss_sum = 0.0
            for offset in range(0, len(step.example_ids), args.microbatch):
                questions = [train[key] for key in step.example_ids[offset:offset + args.microbatch]]
                expert = runtime.feature_batch(torch, questions, cache)
                with model.bind_expert(expert):
                    report["training_forward_attempts"] += 1
                    logits = runtime.forward_logits(torch, model, tokenizer, questions, encoded, cache, "cuda:0")
                    report["training_forward_returns"] += 1
                    targets = torch.tensor([label_ids[q["label_name"]] for q in questions], device=logits.device)
                    batch_sum = F.cross_entropy(logits, targets, reduction="sum")
                    if not bool(torch.isfinite(batch_sum)):
                        raise FloatingPointError("Nonfinite training loss; no skip or retry")
                    loss = batch_sum / 16
                    loss_sum += float(batch_sum.detach())
                    report["backward_attempts"] += 1
                    loss.backward()
                    report["backward_returns"] += 1
                del expert, logits, batch_sum, loss, targets
            if any(p.grad is None for p in adapters.values()) or any(p.grad is not None for p in base_parameters):
                raise ValueError("Expected complete adapter gradients and no base gradients")
            grad_norm = torch.nn.utils.clip_grad_norm_(list(adapters.values()), 1.0,
                                                       error_if_nonfinite=True, foreach=False)
            report["optimizer_step_attempts"] += 1
            optimizer.step()
            report["optimizer_step_returns"] += 1
            completed += 1
            passes += len(step.example_ids)
            epoch_seen.extend(step.example_ids)
            report.update(completed_updates=completed, example_passes=passes,
                          epochs_completed=completed // plan.steps_per_epoch)
            entry = {"completed_updates": completed, "epoch": step.epoch, "step_in_epoch": step.step_in_epoch,
                     "question_ids": step.example_ids, "examples": len(step.example_ids),
                     "mean_full_vocab_loss": loss_sum / len(step.example_ids), "learning_rate": lr,
                     "gradient_norm_before_clip": float(grad_norm), "seconds": time.monotonic() - start}
            journal.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
            journal.flush()
            if step.final_in_epoch:
                if epoch_seen != schedule.epoch_order(train_ids, 42, step.epoch):
                    raise ValueError("Actual epoch traversal differs from exact fixed ID permutation")
                report.setdefault("completed_epoch_order_sha256", {})[str(step.epoch)] = schedule.order_digest(epoch_seen)
                epoch_seen = []
                evaluate(f"epoch-{step.epoch + 1}")
                save_boundary(f"step-{completed:08d}-epoch-{step.epoch + 1}")
            elif completed % 500 == 0:
                save_boundary(f"step-{completed:08d}-periodic")
            if completed % 100 == 0 or step.final_in_epoch:
                print(json.dumps({"stage": STAGE, "updates": completed, "example_passes": passes,
                                  "epochs_completed": report["epochs_completed"], "loss": entry["mean_full_vocab_loss"]}), flush=True)
    verify_sources()
    if (completed != plan.total_steps or passes != plan.example_passes
            or [entry["name"] for entry in evaluations] != ["initial", "epoch-1", "epoch-2", "epoch-3", "epoch-4", "epoch-5"]
            or any(not bool(torch.isfinite(p).all()) for p in adapters.values())
            or not cache.verify_files() or not legacy_cache.verify_files()):
        raise ValueError("Final full traversal/evaluation/finite/cache criterion failed")
    verified_final = state.resolve_checkpoint(checkpoint_root, identities)
    if verified_final.metadata["cursor"]["completed_updates"] != completed:
        raise ValueError("Final checkpoint not durably committed at final boundary")
    final_ack = durable_guard(verified_final)
    report["actual_base_sha256_after"] = base_digest(torch, named_base_parameters)
    if report["actual_base_sha256_after"] != actual_base_digest:
        raise ValueError("Resident base/quantization-state digest changed")
    if any(sha(model_dir / name) != value for name, value in asset_hashes["files"].items()):
        raise ValueError("Cloud model source file SHA256 changed during run")
    report.update(status="complete", valid_complete=True, completed_updates=completed, example_passes=passes,
                  epochs_completed=5, evaluations=evaluations, cache_files_unchanged=True,
                  all_base_gradients_absent=True, base_version_counters_unchanged=True,
                  base_integrity_limit="in-process before/after parameter and quantization-state byte hash; not independent model reload",
                  final_checkpoint=str(verified_final.path.relative_to(output)),
                  final_checkpoint_manifest_sha256=verified_final.manifest_sha256,
                  final_checkpoint_persistent_ack_verified=True,
                  final_checkpoint_durable_unit_sha256=final_ack["unit_sha256"],
                  automatic_next_experiment=False, cache_stats=cache.stats(), historical_54_evaluated=True)


def main():
    args = arguments()
    attempt = contained(args.attempt)
    report = {"stage": STAGE, "status": "running", "valid_complete": False,
              "training_forward_attempts": 0, "training_forward_returns": 0,
              "evaluation_forward_attempts": 0, "evaluation_forward_returns": 0,
              "backward_attempts": 0, "backward_returns": 0,
              "optimizer_step_attempts": 0, "optimizer_step_returns": 0}

    def interrupted(signum, frame):
        raise InterruptedError(f"Signal {signum}; preserve last committed checkpoint, no automatic retry")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        execute(args, report)
    except BaseException as error:
        report.update(status="stopped" if isinstance(error, (InterruptedError, KeyboardInterrupt)) else "error",
                      valid_complete=False, error_type=type(error).__name__,
                      error=str(error).replace(str(ROOT), "PROJECT"),
                      recovery="Only an explicit same-run resume from latest verified committed checkpoint is allowed")
        if hasattr(error, "memory_evidence"):
            report["feature_memory_evidence"] = error.memory_evidence
        if hasattr(error, "source_cache_advice_evidence"):
            report["model_source_cache_advice"] = error.source_cache_advice_evidence
        traceback.print_exc()
    report["attempt_total_seconds"] = time.monotonic() - args.outer_start
    write_json(attempt / "report.json", report)
    print(json.dumps({key: report.get(key) for key in
          ("stage", "status", "valid_complete", "completed_updates", "example_passes", "attempt_total_seconds", "error")}), flush=True)
    return 0 if report["valid_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
