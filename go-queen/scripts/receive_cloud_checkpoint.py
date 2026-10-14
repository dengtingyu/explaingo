#!/usr/bin/env python3
"""Bounded checkpoint receiver; Python 3.8 standard library, no model imports.

Deploy as a forced SSH command with a fixed --root. Input is a length-prefixed
JSON manifest followed by exactly the declared regular-file bytes. A committed
unit contains a complete restorable files/ tree; its atomic rename precedes the
receipt. Hash readback/fsync is filesystem evidence, not proof of physical media
durability or a second replica. Never deserialize optimizer/tensor payloads here.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import sys
import tempfile
import time

PROTOCOL = "goqueen-checkpoint-stream-v1"
MAGIC = b"GQCP001\n"
DEFAULT_ROOT = "/workspace/goqueen-persisted-training"
DEFAULT_TARGET = "goqueen-persisted-training-v1"
MAX_HEADER = 2 * 1024 * 1024
MAX_FILES = 4096
MAX_BYTES = 8 * 1024**3
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
CP_NAME = re.compile(r"step-(\d{8})-(initial|periodic|epoch-[1-5])\Z")
HEX = re.compile(r"[0-9a-f]{64}\Z")


class ProtocolError(ValueError):
    pass


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def safe_relative(value):
    if not isinstance(value, str) or not value or len(value) > 512 or "\\" in value:
        raise ProtocolError("Invalid relative file name")
    parts = PurePosixPath(value).parts
    if value.startswith("/") or any(p in ("", ".", "..") for p in value.split("/")):
        raise ProtocolError("Noncanonical relative file name")
    if any(not NAME.fullmatch(part) for part in parts):
        raise ProtocolError("Unsupported file name")
    return value


def file_info(path, sync=False):
    if path.is_symlink() or not path.is_file():
        raise ProtocolError("Expected a regular file")
    h, count = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            h.update(block)
            count += len(block)
        if sync:
            os.fsync(stream.fileno())
    return {"bytes": count, "sha256": h.hexdigest()}


def sync_dir(path):
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_new(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def read_exact(stream, count):
    chunks = []
    remaining = count
    while remaining:
        block = stream.read(remaining)
        if not block:
            raise ProtocolError("Interrupted stream")
        chunks.append(block)
        remaining -= len(block)
    return b"".join(chunks)


def checked_header(header, receiver_sha256, target_id):
    if not isinstance(header, dict) or header.get("protocol") != PROTOCOL:
        raise ProtocolError("Wrong protocol")
    if header.get("receiver_sha256") != receiver_sha256 or header.get("target_id") != target_id:
        raise ProtocolError("Receiver source/target identity conflict")
    if not isinstance(header.get("run_id"), str) or not NAME.fullmatch(header["run_id"]):
        raise ProtocolError("Invalid run identity")
    if header.get("kind") in ("query", "export_latest"):
        if set(header) != {"protocol", "kind", "run_id", "receiver_sha256", "target_id"}:
            raise ProtocolError("Unexpected query fields")
        return
    required = {"protocol", "kind", "run_id", "receiver_sha256", "target_id", "unit_id",
                "checkpoint_name", "completed_updates", "cursor", "run_identity_sha256",
                "checkpoint_manifest_sha256", "files", "unit_sha256"}
    if set(header) != required or header["kind"] not in ("checkpoint", "final"):
        raise ProtocolError("Unexpected unit fields")
    expected_hash = digest_bytes(canonical({k: v for k, v in header.items() if k != "unit_sha256"}))
    if header["unit_sha256"] != expected_hash:
        raise ProtocolError("Unit manifest hash mismatch")
    for key in ("run_identity_sha256", "checkpoint_manifest_sha256"):
        if not isinstance(header[key], str) or not HEX.fullmatch(header[key]):
            raise ProtocolError("Invalid identity digest")
    match = CP_NAME.fullmatch(header["checkpoint_name"] or "")
    engineering = header["checkpoint_name"] == "engineering-b5"
    step = header["completed_updates"]
    cursor = header["cursor"]
    if ((not match and not engineering) or type(step) is not int or step < 0
            or (engineering and step != 5) or (match and int(match.group(1)) != step)
            or not isinstance(cursor, dict) or set(cursor) != {"epoch_index", "next_batch_index", "completed_updates"}
            or any(type(v) is not int or v < 0 for v in cursor.values()) or cursor["completed_updates"] != step):
        raise ProtocolError("Invalid checkpoint cursor/name")
    if header["kind"] == "checkpoint":
        if header["unit_id"] != header["checkpoint_name"]:
            raise ProtocolError("Checkpoint unit name differs")
    elif not re.fullmatch(r"final-attempt-\d{6}", header["unit_id"] or ""):
        raise ProtocolError("Invalid final unit name")
    files = header["files"]
    if not isinstance(files, dict) or not 1 <= len(files) <= MAX_FILES or "RUN.json" not in files:
        raise ProtocolError("Invalid file inventory")
    total = 0
    for name, info in files.items():
        safe_relative(name)
        if (not isinstance(info, dict) or set(info) != {"bytes", "sha256"}
                or type(info["bytes"]) is not int or info["bytes"] < 0
                or not isinstance(info["sha256"], str) or not HEX.fullmatch(info["sha256"])):
            raise ProtocolError("Invalid file size/hash")
        total += info["bytes"]
        if total > MAX_BYTES:
            raise ProtocolError("Unit exceeds byte bound")


def verify_tree(unit, header):
    root = unit / "files"
    if root.is_symlink() or not root.is_dir():
        raise ProtocolError("Invalid payload directory")
    actual = {}
    for parent, dirs, names in os.walk(str(root), followlinks=False):
        for name in dirs:
            if (Path(parent) / name).is_symlink():
                raise ProtocolError("Symlink directory forbidden")
        for name in names:
            p = Path(parent) / name
            actual[p.relative_to(root).as_posix()] = file_info(p, sync=True)
    if actual != header["files"]:
        raise ProtocolError("Readback inventory or payload hash mismatch")
    run = json.loads((root / "RUN.json").read_bytes())
    if digest_bytes(canonical(run)) != header["run_identity_sha256"]:
        raise ProtocolError("Run identity hash mismatch")
    if header["kind"] == "checkpoint":
        cp = root / "checkpoints" / header["checkpoint_name"]
        raw = (cp / "MANIFEST.json").read_bytes()
        manifest = json.loads(raw)
        if (digest_bytes(raw) != header["checkpoint_manifest_sha256"]
                or (cp / "READY").read_bytes() != (digest_bytes(raw) + "\n").encode("ascii")
                or manifest.get("format_version") != 1 or manifest.get("name") != header["checkpoint_name"]):
            raise ProtocolError("Checkpoint MANIFEST/READY mismatch")
        metadata = manifest["metadata"]
        if metadata["identities"] != run or metadata["cursor"] != header["cursor"]:
            raise ProtocolError("Checkpoint cursor or run identity conflict")
        prefix = "checkpoints/" + header["checkpoint_name"] + "/"
        expected = {"RUN.json", prefix + "MANIFEST.json", prefix + "READY"}
        for name, info in manifest["files"].items():
            safe_relative(name)
            if header["files"].get(prefix + name) != info:
                raise ProtocolError("Checkpoint payload declaration mismatch")
            expected.add(prefix + name)
        if header["checkpoint_name"] == "engineering-b5":
            if (metadata.get("engineering_only") is not True
                    or run.get("formal_resume_forbidden") is not True
                    or run.get("stage") != "qwen38_expanded_contract_engineering_source_cache_repair"
                    or header["cursor"] != {"epoch_index": 0, "next_batch_index": 5, "completed_updates": 5}
                    or metadata.get("evaluations")):
                raise ProtocolError("Invalid isolated engineering checkpoint")
            if set(header["files"]) != expected:
                raise ProtocolError("Unexpected engineering payload")
            return run
        for item in metadata.get("evaluations", []):
            name = safe_relative(item["file"])
            if not re.fullmatch(r"attempts/\d{6}/evaluation-[A-Za-z0-9_.-]+\.json", name):
                raise ProtocolError("Invalid evaluation reference")
            if header["files"].get(name, {}).get("sha256") != item["sha256"]:
                raise ProtocolError("Evaluation reference hash mismatch")
            expected.add(name)
        attempt = metadata.get("attempt")
        if not isinstance(attempt, str) or not re.fullmatch(r"attempts/\d{6}", attempt):
            raise ProtocolError("Missing checkpoint attempt identity")
        expected.add(attempt + "/steps-snapshot-" + header["checkpoint_name"] + ".jsonl")
        if set(header["files"]) != expected:
            raise ProtocolError("Undeclared or missing checkpoint dependency")
    else:
        attempt = header["unit_id"].replace("final-attempt-", "attempts/")
        required = {attempt + "/report.json", attempt + "/process-completion.json"}
        if not required.issubset(header["files"]):
            raise ProtocolError("Final attempt completion/report absent")
        for name in header["files"]:
            if name != "RUN.json" and (not name.startswith(attempt + "/")
                    or Path(name).suffix not in (".json", ".jsonl", ".log", ".py")):
                raise ProtocolError("Unexpected final artifact path/type")
    return run


def read_unit(path):
    if path.is_symlink() or not path.is_dir():
        raise ProtocolError("Invalid committed unit")
    if {p.name for p in path.iterdir()} != {"UNIT.json", "ACK.json", "files"}:
        raise ProtocolError("Unexpected committed unit entries")
    for name in ("UNIT.json", "ACK.json"):
        if (path / name).is_symlink() or not (path / name).is_file():
            raise ProtocolError("Invalid unit metadata file")
    header = json.loads((path / "UNIT.json").read_bytes())
    checked_header(header, header.get("receiver_sha256"), header.get("target_id"))
    ack = json.loads((path / "ACK.json").read_bytes())
    expected = dict(header, ok=True)
    if any(ack.get(key) != value for key, value in expected.items()) or set(ack) != set(expected) | {"committed_unix"}:
        raise ProtocolError("Committed receipt mismatch")
    if type(ack["committed_unix"]) not in (int, float) or not math.isfinite(ack["committed_unix"]):
        raise ProtocolError("Invalid committed timestamp")
    return header, ack


def committed_units(run_root):
    directory = run_root / "commits"
    result = []
    if directory.is_symlink():
        raise ProtocolError("Committed directory symlink forbidden")
    if not directory.exists():
        return result
    for p in sorted(directory.iterdir()):
        if not NAME.fullmatch(p.name):
            raise ProtocolError("Unknown committed-unit name")
        h, ack = read_unit(p)
        if h["unit_id"] != p.name:
            raise ProtocolError("Committed name differs")
        result.append((p, h, ack))
    return result


def restore_candidate(root, run_id):
    """Read-only latest complete checkpoint unit; caller copies its files/ tree."""
    if not NAME.fullmatch(run_id or ""):
        raise ProtocolError("Invalid run id")
    reject_incomplete_retirement(Path(root) / run_id)
    units = [row for row in committed_units(Path(root) / run_id) if row[1]["kind"] == "checkpoint"]
    if not units:
        raise ProtocolError("No committed checkpoint receipt")
    if len({h["completed_updates"] for _, h, _ in units}) != len(units):
        raise ProtocolError("Ambiguous committed cursors")
    path, header, ack = max(units, key=lambda row: row[1]["completed_updates"])
    verify_tree(path, header)
    return path / "files", ack


def reject_incomplete_retirement(run_root):
    if run_root.exists() and any(p.name.startswith(".discarding-") for p in run_root.iterdir()):
        raise ProtocolError("Incomplete prior retirement requires inspection")


def prune(run_root, newest):
    reject_incomplete_retirement(run_root)
    periodic = [(h["completed_updates"], p, h) for p, h, _ in committed_units(run_root)
                if h["kind"] == "checkpoint" and h["checkpoint_name"].endswith("-periodic")]
    retired = run_root / "retired"
    if retired.is_symlink():
        raise ProtocolError("Retirement directory symlink forbidden")
    retired.mkdir(exist_ok=True)
    for step, path, header in sorted(periodic)[:-2]:
        if step >= newest["completed_updates"]:
            raise ProtocolError("Cannot prune current/future boundary")
        verify_tree(path, header)
        record = {"unit_id": path.name, "unit_sha256": header["unit_sha256"],
                  "replacement_unit_sha256": newest["unit_sha256"], "completed_updates": step}
        receipt = retired / (path.name + ".json")
        if receipt.exists():
            existing = json.loads(receipt.read_bytes())
            if any(existing[k] != record[k] for k in ("unit_id", "unit_sha256", "completed_updates")):
                raise ProtocolError("Retirement identity conflict")
        else:
            write_new(receipt, canonical(record))
            sync_dir(retired)
        discard = run_root / (".discarding-" + path.name)
        if discard.exists():
            raise ProtocolError("Incomplete prior retirement requires inspection")
        os.rename(str(path), str(discard))
        sync_dir(path.parent)
        sync_dir(run_root)
        shutil.rmtree(str(discard))
        sync_dir(run_root)


def receive(root, stream, receiver_sha256, target_id=DEFAULT_TARGET, export_stream=None):
    """Receive one bounded unit and return only its fully checked receipt."""
    if read_exact(stream, len(MAGIC)) != MAGIC:
        raise ProtocolError("Wrong stream magic")
    length = struct.unpack(">Q", read_exact(stream, 8))[0]
    if not 1 <= length <= MAX_HEADER:
        raise ProtocolError("Header exceeds bound")
    header = json.loads(read_exact(stream, length))
    checked_header(header, receiver_sha256, target_id)
    root = Path(root).absolute()
    if root.is_symlink():
        raise ProtocolError("Receiver root symlink forbidden")
    root.mkdir(parents=True, exist_ok=True)
    run_root = root / header["run_id"]
    if run_root.is_symlink():
        raise ProtocolError("Run root symlink forbidden")
    run_root.mkdir(exist_ok=True)
    sync_dir(root)
    if (run_root / "replica.lock").is_symlink():
        raise ProtocolError("Lock symlink forbidden")
    with (run_root / "replica.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        reject_incomplete_retirement(run_root)
        if header["kind"] in ("query", "export_latest"):
            if stream.read(1):
                raise ProtocolError("Unexpected query payload")
            payload_root, ack = restore_candidate(root, header["run_id"])
            if ack["receiver_sha256"] != receiver_sha256 or ack["target_id"] != target_id:
                raise ProtocolError("Latest receipt source/target differs")
            if header["kind"] == "query":
                return ack
            if export_stream is None:
                raise ProtocolError("Export output stream is required")
            raw = canonical(ack)
            export_stream.write(MAGIC + struct.pack(">Q", len(raw)) + raw)
            for relative, info in sorted(ack["files"].items()):
                h, count = hashlib.sha256(), 0
                with (payload_root / relative).open("rb") as source:
                    while True:
                        block = source.read(1024 * 1024)
                        if not block:
                            break
                        export_stream.write(block)
                        h.update(block)
                        count += len(block)
                if {"bytes": count, "sha256": h.hexdigest()} != info:
                    raise ProtocolError("Export source changed during read")
            export_stream.flush()
            return None
        staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=str(run_root)))
        files_root = staging / "files"
        try:
            for relative, info in sorted(header["files"].items()):
                path = files_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                h, remaining = hashlib.sha256(), info["bytes"]
                with path.open("xb") as destination:
                    while remaining:
                        block = stream.read(min(1024 * 1024, remaining))
                        if not block:
                            raise ProtocolError("Interrupted payload")
                        destination.write(block)
                        h.update(block)
                        remaining -= len(block)
                    destination.flush()
                    os.fsync(destination.fileno())
                if h.hexdigest() != info["sha256"]:
                    raise ProtocolError("Stream payload hash mismatch")
            if stream.read(1):
                raise ProtocolError("Unexpected trailing payload")
            verify_tree(staging, header)
            run_file = run_root / "RUN.json"
            if run_file.exists():
                if run_file.is_symlink() or run_file.read_bytes() != (files_root / "RUN.json").read_bytes():
                    raise ProtocolError("Run identity bytes conflict")
            else:
                write_new(run_file, (files_root / "RUN.json").read_bytes())
                sync_dir(run_root)
            units = committed_units(run_root)
            for _, old, _ in units:
                if any(old[k] != header[k] for k in ("run_identity_sha256", "run_id", "receiver_sha256", "target_id")):
                    raise ProtocolError("Existing run identity conflict")
            commits = run_root / "commits"
            if commits.is_symlink():
                raise ProtocolError("Committed directory symlink forbidden")
            commits.mkdir(exist_ok=True)
            destination = commits / header["unit_id"]
            if destination.exists():
                previous, ack = read_unit(destination)
                if previous != header:
                    raise ProtocolError("Same unit name with different bytes")
                verify_tree(destination, previous)
                latest = max((h for _, h, _ in units if h["kind"] == "checkpoint"),
                             key=lambda h: h["completed_updates"])
                prune(run_root, latest)
                return ack
            checkpoints = [h for _, h, _ in units if h["kind"] == "checkpoint"]
            if header["kind"] == "checkpoint":
                if checkpoints and header["completed_updates"] <= max(h["completed_updates"] for h in checkpoints):
                    raise ProtocolError("Checkpoint cursor cannot move backward or repeat")
                if checkpoints:
                    latest = max(checkpoints, key=lambda h: h["completed_updates"])
                    if tuple(header["cursor"][k] for k in ("epoch_index", "next_batch_index")) <= tuple(latest["cursor"][k] for k in ("epoch_index", "next_batch_index")):
                        raise ProtocolError("Next-batch cursor cannot move backward or repeat")
            else:
                latest = max(checkpoints, key=lambda h: h["completed_updates"]) if checkpoints else None
                if not latest or any(latest[k] != header[k] for k in
                       ("checkpoint_name", "completed_updates", "cursor", "checkpoint_manifest_sha256")):
                    raise ProtocolError("Final evidence lacks matching latest committed checkpoint")
                verify_tree(commits / latest["unit_id"], latest)
            ack = dict(header, ok=True, committed_unix=time.time())
            write_new(staging / "UNIT.json", canonical(header))
            write_new(staging / "ACK.json", canonical(ack))
            for parent, _, _ in os.walk(str(staging), topdown=False):
                sync_dir(Path(parent))
            os.rename(str(staging), str(destination))
            sync_dir(commits)
            pointer = run_root / ".latest.tmp"
            if pointer.exists():
                pointer.unlink()
            write_new(pointer, canonical({"unit_id": header["unit_id"], "unit_sha256": header["unit_sha256"]}))
            os.replace(str(pointer), str(run_root / "latest.json"))
            sync_dir(run_root)
            prune(run_root, header)
            return ack
        finally:
            if staging.exists():
                shutil.rmtree(str(staging))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--target-id", default=DEFAULT_TARGET)
    args = parser.parse_args()
    if str(Path(args.root).absolute()) != DEFAULT_ROOT:
        parser.error("Receiver CLI is restricted to the dedicated task root")
    try:
        source_sha = file_info(Path(__file__))["sha256"]
        result = receive(args.root, sys.stdin.buffer, source_sha, args.target_id, sys.stdout.buffer)
        if result is not None:
            print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
        return 0
    except Exception as error:
        # Do not echo command lines, hostnames, credentials, or filesystem errors.
        print(json.dumps({"ok": False, "error_class": type(error).__name__,
                          "error": str(error) if isinstance(error, ProtocolError) else "Receiver operation failed"}), flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
