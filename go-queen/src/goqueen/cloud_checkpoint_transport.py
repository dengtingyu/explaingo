"""Synchronous, byte-checked checkpoint replication; standard library only.

The caller stops optimizer work at a saved boundary until publication returns.
Remote receipts prove the receiver's specified hash/readback/fsync procedure,
not physical media independence. This module never decodes model/optimizer/RNG
payloads, consumes training RNG, starts training, provisions SSH, or stops a pod.
Private connection configuration is deliberately separate from public identity.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import subprocess
import tempfile
import threading
import time

from .cloud_training_state import resolve_checkpoint

PROTOCOL = "goqueen-checkpoint-stream-v1"
MAGIC = b"GQCP001\n"
TARGET = "goqueen-persisted-training-v1"
MAX_HEADER = 2 * 1024 * 1024
MAX_FILES = 4096
MAX_BYTES = 8 * 1024**3
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
HEX = re.compile(r"[0-9a-f]{64}\Z")


class TransportError(RuntimeError):
    """No valid receipt; caller must stop and leave compute resources intact."""


def _canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _relative(value):
    if (not isinstance(value, str) or not value or len(value) > 512 or "\\" in value
            or value.startswith("/") or any(p in ("", ".", "..") for p in value.split("/"))
            or any(not NAME.fullmatch(p) for p in PurePosixPath(value).parts)):
        raise TransportError("Invalid relative payload path")
    return value


def _file(path):
    if path.is_symlink() or not path.is_file():
        raise TransportError("Expected a regular payload file")
    h, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
            size += len(chunk)
    return {"bytes": size, "sha256": h.hexdigest()}


def _directory(path, create=False):
    if path.is_symlink():
        raise TransportError("Symlink directory forbidden")
    if create:
        existed = path.exists()
        path.mkdir(parents=True, exist_ok=True)
        if not existed:
            _sync_dir(path.parent)
    if not path.is_dir():
        raise TransportError("Required directory absent")
    return path


def _under(root, relative):
    path = root
    for part in PurePosixPath(_relative(relative)).parts:
        path = path / part
        if path.is_symlink():
            raise TransportError("Symlink payload path forbidden")
    return path


def _inventory(directory):
    result = {}
    _directory(directory)
    for parent, dirs, files in os.walk(directory, followlinks=False):
        for name in dirs:
            if (Path(parent) / name).is_symlink():
                raise TransportError("Symlink payload directory forbidden")
        for name in files:
            path = Path(parent) / name
            relative = _relative(path.relative_to(directory).as_posix())
            result[relative] = _file(path)
    return result


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_new(path, data):
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _json_file(path):
    if path.is_symlink() or not path.is_file():
        raise TransportError("Required JSON file is not regular")
    return json.loads(path.read_bytes())


def _config(value):
    if not isinstance(value, dict):
        raise TransportError("Transport config must be an object")
    required = {"host", "port", "identity_file", "known_hosts", "run_id", "receiver_sha256"}
    optional = {"user", "target_id", "timeout_seconds", "expected_run_identity_sha256"}
    if not required.issubset(value) or set(value) - required - optional:
        raise TransportError("Transport config fields differ")
    result = dict(value)
    result.setdefault("user", "root")
    result.setdefault("target_id", TARGET)
    result.setdefault("timeout_seconds", 900)
    for key in ("run_id", "target_id", "user"):
        if not isinstance(result[key], str) or not NAME.fullmatch(result[key]):
            raise TransportError("Invalid transport identity")
    if not isinstance(result["host"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]{0,252}", result["host"]):
        raise TransportError("Invalid destination host")
    if type(result["port"]) is not int or not 1 <= result["port"] <= 65535:
        raise TransportError("Invalid destination port")
    if type(result["timeout_seconds"]) is not int or not 1 <= result["timeout_seconds"] <= 1800:
        raise TransportError("Invalid bounded transport timeout")
    for key in ("receiver_sha256", "expected_run_identity_sha256"):
        if key in result and (not isinstance(result[key], str) or not HEX.fullmatch(result[key])):
            raise TransportError("Invalid pinned digest")
    for key in ("identity_file", "known_hosts"):
        path = Path(result[key])
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise TransportError("Private SSH file absent or not regular")
    return result


def load_config(path):
    """Read a local private JSON configuration; no SSH or other network action."""
    return _config(_json_file(Path(path)))


def public_identity(config):
    config = _config(config)
    return {"protocol": PROTOCOL, "run_id": config["run_id"],
            "receiver_sha256": config["receiver_sha256"], "target_id": config["target_id"]}


def _check_header(header, config):
    expected = public_identity(config)
    if any(header.get(k) != v for k, v in expected.items()):
        raise TransportError("Unit public identity differs")
    fields = set(expected) | {"kind", "unit_id", "checkpoint_name", "completed_updates", "cursor",
                             "run_identity_sha256", "checkpoint_manifest_sha256", "files", "unit_sha256"}
    if set(header) != fields or header["kind"] not in ("checkpoint", "final"):
        raise TransportError("Unexpected frozen unit fields")
    if _sha(_canonical({k: v for k, v in header.items() if k != "unit_sha256"})) != header["unit_sha256"]:
        raise TransportError("Frozen unit hash mismatch")
    if not NAME.fullmatch(header["unit_id"]) or not NAME.fullmatch(header["checkpoint_name"]):
        raise TransportError("Invalid frozen unit name")
    match = re.fullmatch(r"step-(\d{8})-(initial|periodic|epoch-[1-5])", header["checkpoint_name"])
    if not match and header["checkpoint_name"] != "engineering-b5":
        raise TransportError("Invalid frozen checkpoint name")
    if ((match and int(match.group(1)) != header["completed_updates"])
            or (not match and header["completed_updates"] != 5)
            or (header["kind"] == "checkpoint" and header["unit_id"] != header["checkpoint_name"])
            or (header["kind"] == "final" and not re.fullmatch(r"final-attempt-\d{6}", header["unit_id"]))):
        raise TransportError("Frozen name/counter differs")
    cursor = header["cursor"]
    if (not isinstance(cursor, dict) or set(cursor) != {"epoch_index", "next_batch_index", "completed_updates"}
            or any(type(v) is not int or v < 0 for v in cursor.values())
            or cursor["completed_updates"] != header["completed_updates"]):
        raise TransportError("Invalid frozen optimizer cursor")
    for key in ("run_identity_sha256", "checkpoint_manifest_sha256"):
        if not isinstance(header[key], str) or not HEX.fullmatch(header[key]):
            raise TransportError("Invalid frozen digest")
    if (config.get("expected_run_identity_sha256") is not None
            and config["expected_run_identity_sha256"] != header["run_identity_sha256"]):
        raise TransportError("Expected run identity differs")
    files = header["files"]
    if not isinstance(files, dict) or not 1 <= len(files) <= MAX_FILES or "RUN.json" not in files:
        raise TransportError("Invalid frozen inventory")
    total = 0
    for name, info in files.items():
        _relative(name)
        if (not isinstance(info, dict) or set(info) != {"bytes", "sha256"}
                or type(info["bytes"]) is not int or info["bytes"] < 0
                or not isinstance(info["sha256"], str) or not HEX.fullmatch(info["sha256"])):
            raise TransportError("Invalid frozen payload declaration")
        total += info["bytes"]
    if total > MAX_BYTES:
        raise TransportError("Frozen unit exceeds byte bound")
    if len(_canonical(header)) > MAX_HEADER:
        raise TransportError("Frozen header exceeds bound")


def _check_ack(ack, header):
    if (not isinstance(ack, dict) or set(ack) != set(header) | {"ok", "committed_unix"}
            or ack.get("ok") is not True or any(ack.get(k) != v for k, v in header.items())
            or type(ack.get("committed_unix")) not in (float, int)
            or not math.isfinite(ack["committed_unix"])):
        raise TransportError("Receiver acknowledgement differs from exact sent unit")
    return ack


def _record_matches(run_dir, record, header):
    run = _json_file(run_dir / "RUN.json")
    if (Path(record.path).absolute() != run_dir / "checkpoints" / record.path.name
            or record.metadata["identities"] != run
            or _sha(_canonical(run)) != header["run_identity_sha256"]
            or record.path.name != header["checkpoint_name"]
            or record.manifest_sha256 != header["checkpoint_manifest_sha256"]
            or record.metadata["cursor"] != header["cursor"]):
        raise TransportError("Checkpoint record/run does not match receipt")
    if record.path.exists():
        if record.path.is_symlink() or _file(record.path / "MANIFEST.json")["sha256"] != record.manifest_sha256:
            raise TransportError("Checkpoint manifest changed after publication")
    return run


def _unit_path(run_dir, unit_id):
    if not isinstance(unit_id, str) or not NAME.fullmatch(unit_id):
        raise TransportError("Invalid unit name")
    return _under(run_dir, "transport-units/" + unit_id)


def _read_unit(run_dir, unit_id, config):
    path = _directory(_unit_path(run_dir, unit_id))
    header = _json_file(path / "UNIT.json")
    _check_header(header, config)
    if header["unit_id"] != unit_id:
        raise TransportError("Frozen unit name mismatch")
    return header


def validate_ack(ack, run_dir, record, config):
    """Validate a cached exact receipt, NOT a fresh remote existence check.

    Retained unit descriptors survive local payload pruning. Remote retention
    may likewise have retired an older periodic unit after a newer valid ACK.
    """
    config = _config(config)
    run_dir = _directory(Path(run_dir).absolute())
    header = _read_unit(run_dir, record.path.name, config)
    if header["kind"] != "checkpoint":
        raise TransportError("Receipt is not a checkpoint")
    _record_matches(run_dir, record, header)
    return _check_ack(ack, header)


def _engineering(record):
    m = record.metadata
    return (record.path.name == "engineering-b5" and m.get("engineering_only") is True
            and m["identities"].get("formal_resume_forbidden") is True
            and m["identities"].get("stage") == "qwen38_expanded_contract_engineering_source_cache_repair"
            and m["cursor"] == {"epoch_index": 0, "next_batch_index": 5, "completed_updates": 5}
            and not m.get("evaluations"))


def _checkpoint_sources(run_dir, record):
    cp = _under(run_dir, "checkpoints/" + record.path.name)
    if cp != Path(record.path).absolute():
        raise TransportError("Checkpoint is outside this run")
    inventory = _inventory(cp)
    raw = (cp / "MANIFEST.json").read_bytes()
    manifest = json.loads(raw)
    if (_sha(raw) != record.manifest_sha256 or manifest.get("format_version") != 1
            or manifest.get("name") != cp.name or manifest.get("metadata") != record.metadata
            or manifest.get("files") != record.files
            or (cp / "READY").read_bytes() != (record.manifest_sha256 + "\n").encode("ascii")
            or {k: v for k, v in inventory.items() if k not in ("MANIFEST.json", "READY")} != record.files):
        raise TransportError("Checkpoint bytes/manifest/READY differ")
    sources = {"RUN.json": _under(run_dir, "RUN.json")}
    sources.update({"checkpoints/" + cp.name + "/" + name: cp / name for name in inventory})
    if _json_file(sources["RUN.json"]) != record.metadata["identities"]:
        raise TransportError("RUN does not equal checkpoint immutable identities")
    if _engineering(record):
        return sources
    if not re.fullmatch(r"step-\d{8}-(initial|periodic|epoch-[1-5])", cp.name):
        raise TransportError("Unsupported formal checkpoint name")
    for item in record.metadata.get("evaluations", []):
        name = _relative(item["file"])
        if not re.fullmatch(r"attempts/\d{6}/evaluation-[A-Za-z0-9_.-]+\.json", name):
            raise TransportError("Unexpected evaluation reference")
        path = _under(run_dir, name)
        if _file(path)["sha256"] != item["sha256"]:
            raise TransportError("Referenced evaluation changed")
        sources[name] = path
    attempt = record.metadata.get("attempt", "")
    if not re.fullmatch(r"attempts/\d{6}", attempt):
        raise TransportError("Checkpoint attempt identity absent")
    sources[attempt + "/steps-snapshot-" + cp.name + ".jsonl"] = _under(run_dir, attempt + "/steps.jsonl")
    return sources


def _freeze(run_dir, record, config, kind, unit_id, sources):
    directory = _directory(run_dir / "transport-units", create=True)
    destination = _unit_path(run_dir, unit_id)
    if destination.exists():
        header = _read_unit(run_dir, unit_id, config)
        _record_matches(run_dir, record, header)
        if header["kind"] != kind:
            raise TransportError("Existing frozen unit kind differs")
        return header
    staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=directory))
    try:
        files = {}
        for relative, source in sorted(sources.items()):
            _relative(relative)
            if relative.startswith("checkpoints/"):
                # Keep only the descriptor locally, not another 1.6 GB payload.
                files[relative] = _file(source)
            else:
                target = staging / "files" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                before = _file(source)
                with source.open("rb") as inp, target.open("xb") as out:
                    shutil.copyfileobj(inp, out, 1024 * 1024)
                    out.flush()
                    os.fsync(out.fileno())
                files[relative] = _file(target)
                if files[relative] != before or _file(source) != before:
                    raise TransportError("Dependency changed while freezing unit")
        header = dict(public_identity(config), kind=kind, unit_id=unit_id,
                      checkpoint_name=record.path.name,
                      completed_updates=record.metadata["cursor"]["completed_updates"],
                      cursor=record.metadata["cursor"],
                      run_identity_sha256=_sha(_canonical(record.metadata["identities"])),
                      checkpoint_manifest_sha256=record.manifest_sha256, files=files)
        header["unit_sha256"] = _sha(_canonical(header))
        _check_header(header, config)
        _write_new(staging / "UNIT.json", _canonical(header))
        for parent, _, _ in os.walk(staging, topdown=False):
            _sync_dir(Path(parent))
        os.rename(staging, destination)
        _sync_dir(directory)
        return header
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _sources_for_unit(run_dir, header):
    snapshots = _unit_path(run_dir, header["unit_id"]) / "files"
    return {relative: _under(run_dir if relative.startswith("checkpoints/") else snapshots, relative)
            for relative in header["files"]}


def _ssh_args(config):
    return ["ssh", "-T", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
            "-o", "StrictHostKeyChecking=yes", "-o", "UserKnownHostsFile=" + config["known_hosts"],
            "-o", "ConnectTimeout=20", "-o", "ConnectionAttempts=1",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2",
            "-o", "LogLevel=ERROR", "-i", config["identity_file"], "-p", str(config["port"]),
            "--", config["user"] + "@" + config["host"]]


def _connection_failure(code, stderr):
    # Authentication, host-key, protocol/hash and generic timeout failures stop.
    terms = (b"connection timed out", b"connection refused", b"no route to host",
             b"connection reset by peer", b"broken pipe", b"connection closed by")
    return code == 255 and any(term in stderr.lower() for term in terms)


def _write_stream(stream, header, sources):
    raw = _canonical(header)
    stream.write(MAGIC + struct.pack(">Q", len(raw)) + raw)
    for relative, info in sorted(header["files"].items()):
        path = sources[relative]
        if path.is_symlink() or not path.is_file():
            raise TransportError("Frozen payload missing or replaced")
        h, count = hashlib.sha256(), 0
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                stream.write(block)
                h.update(block)
                count += len(block)
                if count > info["bytes"]:
                    raise TransportError("Payload grew beyond frozen byte count")
        if {"bytes": count, "sha256": h.hexdigest()} != info:
            raise TransportError("Payload changed while streaming")
    stream.flush()


def _send_unit(config, header, sources):
    for attempt in range(2):
        errors = []
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            process = subprocess.Popen(_ssh_args(config), stdin=subprocess.PIPE, stdout=out, stderr=err)
            def writer():
                try:
                    _write_stream(process.stdin, header, sources)
                except Exception as error:
                    errors.append(error)
                finally:
                    try:
                        process.stdin.close()
                    except (BrokenPipeError, OSError):
                        pass
            thread = threading.Thread(target=writer, daemon=True)
            thread.start()
            try:
                code = process.wait(timeout=config["timeout_seconds"])
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                thread.join(timeout=5)
                raise TransportError("Bounded transport timed out; no receipt accepted") from None
            thread.join(timeout=5)
            if thread.is_alive():
                raise TransportError("Transport writer did not finish")
            out.seek(0)
            err.seek(0)
            raw, stderr = out.read(MAX_HEADER + 4097), err.read(65536)
            if errors and any(isinstance(e, TransportError) for e in errors):
                raise TransportError("Local payload verification failed; no retry")
            if code == 0 and not errors:
                if len(raw) > MAX_HEADER + 4096:
                    raise TransportError("Acknowledgement exceeds bound")
                try:
                    return _check_ack(json.loads(raw), header)
                except (ValueError, TypeError):
                    raise TransportError("Invalid receiver acknowledgement") from None
            if attempt == 0 and _connection_failure(code, stderr):
                continue
            raise TransportError("Checkpoint transport failed; no matching receipt accepted")
    raise TransportError("Transport attempts exhausted")


def _save_ack(run_dir, header, ack):
    directory = _directory(run_dir / "acks", create=True)
    target = directory / (header["unit_id"] + ".json")
    if target.exists() or target.is_symlink():
        if _json_file(target) != ack:
            raise TransportError("Cached receipt conflicts with receiver")
        return
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(_canonical(ack))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, target)
        _sync_dir(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _lock(run_dir):
    path = run_dir / "transport.lock"
    if path.is_symlink():
        raise TransportError("Transport lock symlink forbidden")
    stream = path.open("a+b")
    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
    return stream


def publish_checkpoint(run_dir, record, config):
    """Synchronously replicate/reconcile one immutable local boundary.

    Re-publication reuses frozen dependencies; a valid remote ACK is required
    even if a local receipt already exists. No tensor payload copy is retained.
    """
    config = _config(config)
    run_dir = _directory(Path(run_dir).absolute())
    with _lock(run_dir):
        unit = _unit_path(run_dir, record.path.name)
        sources = {} if unit.exists() else _checkpoint_sources(run_dir, record)
        header = _freeze(run_dir, record, config, "checkpoint", record.path.name, sources)
        ack = _send_unit(config, header, _sources_for_unit(run_dir, header))
        validate_ack(ack, run_dir, record, config)
        _save_ack(run_dir, header, ack)
        return ack


def _ended_attempts(run_dir):
    attempts = _directory(run_dir / "attempts")
    children = sorted(attempts.iterdir(), key=lambda path: path.name)
    if not children or any(not re.fullmatch(r"\d{6}", p.name) or p.is_symlink() or not p.is_dir() for p in children):
        raise TransportError("Invalid attempt directory inventory")
    if [p.name for p in children] != [f"{index:06d}" for index in range(len(children))]:
        raise TransportError("Local attempt history has gaps; terminal coverage unknown")
    for attempt in children:
        terminal = {name for name in ("report.json", "process-completion.json")
                    if (attempt / name).is_file()}
        if not terminal and not (attempt / "process-start.json").exists():
            raise TransportError("Dependency-only or unstarted attempt is not terminal history; coverage unknown")
        # A live/incomplete attempt is never treated as a frozen terminal record.
        for name in ("report.json", "process-completion.json"):
            if not isinstance(_json_file(attempt / name), dict):
                raise TransportError("Attempt lacks completed report/process evidence")
    return children


def _final_sources(run_dir, attempt):
    sources = {"RUN.json": run_dir / "RUN.json"}
    for relative in _inventory(attempt):
        if relative == "durability-completion.json":
            continue  # Written after the receipt; never introduce a hash cycle.
        if Path(relative).suffix not in (".json", ".jsonl", ".log", ".py"):
            raise TransportError("Unexpected final attempt artifact type")
        sources["attempts/" + attempt.name + "/" + relative] = attempt / relative
    return sources


def _final_snapshot(run_dir, attempt, config, run):
    header = _read_unit(run_dir, "final-attempt-" + attempt.name, config)
    if header["kind"] != "final" or header["run_identity_sha256"] != _sha(_canonical(run)):
        raise TransportError("Final snapshot identity differs")
    # Completed attempt evidence must remain immutable, except its later receipt.
    actual = {name: _file(path) for name, path in _final_sources(run_dir, attempt).items()}
    if actual != header["files"]:
        raise TransportError("Completed attempt changed after final snapshot")
    return header


def _final_context(run_dir, config):
    run = _json_file(run_dir / "RUN.json")
    record = resolve_checkpoint(run_dir / "checkpoints", run)
    if _engineering(record):
        raise TransportError("Engineering checkpoint is not a formal final run")
    ack = _json_file(run_dir / "acks" / (record.path.name + ".json"))
    validate_ack(ack, run_dir, record, config)
    if (run_dir / "RESTORE-ACK.json").exists() or (run_dir / "RESTORE-ACK.json").is_symlink():
        # restore_latest exports checkpoint dependencies, not a complete attempt
        # archive. Do not silently relabel those directories as ended attempts.
        raise TransportError("Checkpoint-only recovery leaves historical terminal coverage unknown; final shutdown refused")
    return run, record, _ended_attempts(run_dir)


def _final_coverage(run_dir, config, run, record, attempts):
    receipts, hashes = {}, {}
    for attempt in attempts:
        header = _final_snapshot(run_dir, attempt, config, run)
        ack = _json_file(run_dir / "acks" / (header["unit_id"] + ".json"))
        receipts[attempt.name] = _check_ack(ack, header)
        hashes[attempt.name] = header["unit_sha256"]
    latest = receipts[attempts[-1].name]
    _record_matches(run_dir, record, latest)
    return {"all_ended_attempts_acknowledged": True,
            "ended_attempts": [p.name for p in attempts],
            "coverage_scope": "Complete contiguous local terminal-attempt inventory; cached exact receipts, not a fresh remote retention check",
            "final_unit_sha256": hashes, "latest_final_ack": latest}


def validate_final_artifacts(run_dir, config):
    """Check exact cached receipts for EVERY ended attempt and latest local CP.

    This validates prior remote acknowledgements, not current remote existence or
    provider media guarantees. Incomplete or checkpoint-only recovered attempt
    history prevents shutdown readiness; this API does not reconstruct missing
    historical terminal records or certify a newly restored host's full history.
    """
    config = _config(config)
    run_dir = _directory(Path(run_dir).absolute())
    with _lock(run_dir):
        run, record, attempts = _final_context(run_dir, config)
        return _final_coverage(run_dir, config, run, record, attempts)


def publish_final_artifacts(run_dir, config):
    """Replicate all ended attempts lacking receipts; return latest exact ACK.

    The newest LOCAL checkpoint must already be acknowledged. Earlier exact
    receipts keep their original checkpoint binding and frozen bytes. A pending
    frozen unit is retransmitted unchanged: if the receiver cannot accept that
    old binding after a newer checkpoint, stop rather than rewrite history.
    All attempts must be terminal before any final evidence is sent. No live logs
    are captured, and later durability-completion files remain outside snapshots.
    """
    config = _config(config)
    run_dir = _directory(Path(run_dir).absolute())
    with _lock(run_dir):
        run, record, attempts = _final_context(run_dir, config)
        for attempt in attempts:
            unit_id = "final-attempt-" + attempt.name
            receipt = run_dir / "acks" / (unit_id + ".json")
            if receipt.exists() or receipt.is_symlink():
                header = _final_snapshot(run_dir, attempt, config, run)
                _check_ack(_json_file(receipt), header)
                continue
            if _unit_path(run_dir, unit_id).exists():
                header = _final_snapshot(run_dir, attempt, config, run)
            else:
                header = _freeze(run_dir, record, config, "final", unit_id,
                                 _final_sources(run_dir, attempt))
            final_ack = _send_unit(config, header, _sources_for_unit(run_dir, header))
            _check_ack(final_ack, header)
            _save_ack(run_dir, header, final_ack)
        return _final_coverage(run_dir, config, run, record, attempts)["latest_final_ack"]


def _read_exact(stream, size):
    data = bytearray()
    while len(data) < size:
        block = stream.read(size - len(data))
        if not block:
            raise TransportError("Interrupted export stream")
        data.extend(block)
    return bytes(data)


def _receive_export(stream, destination, config):
    if _read_exact(stream, len(MAGIC)) != MAGIC:
        raise TransportError("Invalid export stream magic")
    size = struct.unpack(">Q", _read_exact(stream, 8))[0]
    if not 1 <= size <= MAX_HEADER + 4096:
        raise TransportError("Export header exceeds bound")
    ack = json.loads(_read_exact(stream, size))
    header = {k: v for k, v in ack.items() if k not in ("ok", "committed_unix")}
    _check_header(header, config)
    _check_ack(ack, header)
    if header["kind"] != "checkpoint":
        raise TransportError("Export is not a checkpoint")
    for relative, info in sorted(header["files"].items()):
        path = _under(destination, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        h, remaining = hashlib.sha256(), info["bytes"]
        with path.open("xb") as out:
            while remaining:
                block = stream.read(min(1024 * 1024, remaining))
                if not block:
                    raise TransportError("Interrupted export payload")
                out.write(block)
                h.update(block)
                remaining -= len(block)
            out.flush()
            os.fsync(out.fileno())
        if h.hexdigest() != info["sha256"]:
            raise TransportError("Export payload hash mismatch")
    if stream.read(1):
        raise TransportError("Unexpected export trailing bytes")
    if _inventory(destination) != header["files"]:
        raise TransportError("Export readback inventory differs")
    run = _json_file(destination / "RUN.json")
    record = resolve_checkpoint(destination / "checkpoints", run)
    _record_matches(destination, record, header)
    expected = {"RUN.json"} | {"checkpoints/" + record.path.name + "/" + key
                                for key in (*record.files, "MANIFEST.json", "READY")}
    if not _engineering(record):
        attempt = record.metadata.get("attempt", "")
        if not re.fullmatch(r"attempts/\d{6}", attempt):
            raise TransportError("Restored attempt identity missing")
        expected.add(attempt + "/steps-snapshot-" + record.path.name + ".jsonl")
        for item in record.metadata.get("evaluations", []):
            if header["files"].get(item["file"], {}).get("sha256") != item["sha256"]:
                raise TransportError("Restored evaluation identity mismatch")
            expected.add(item["file"])
    if set(header["files"]) != expected:
        raise TransportError("Restored checkpoint dependencies differ")
    return ack


def restore_latest(config, destination):
    """Copy latest remote committed unit into a NEW directory; never auto-resume.

    Incomplete destinations remain visible without RESTORE-ACK.json. This is a
    bounded one-attempt readback operation; callers inspect failure explicitly.
    """
    config = _config(config)
    deadline = time.monotonic() + config["timeout_seconds"]
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise TransportError("Restore destination must not exist")
    destination.mkdir(parents=True)
    request = dict(public_identity(config), kind="export_latest")
    raw = _canonical(request)
    results, errors = [], []
    with tempfile.TemporaryFile() as err:
        process = subprocess.Popen(_ssh_args(config), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err)
        process.stdin.write(MAGIC + struct.pack(">Q", len(raw)) + raw)
        process.stdin.close()
        def reader():
            try:
                results.append(_receive_export(process.stdout, destination, config))
            except Exception as error:
                errors.append(error)
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        try:
            code = process.wait(timeout=max(0.001, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            thread.join(timeout=5)
            raise TransportError("Bounded restore timed out; no complete receipt") from None
        thread.join(timeout=max(0, deadline - time.monotonic()))
        if code != 0 or thread.is_alive() or errors or len(results) != 1:
            raise TransportError("Restore incomplete or mismatched; no complete receipt")
        ack = results[0]
    # Reconstruct only the small frozen descriptor/dependencies. The restored
    # checkpoint payload remains in its one canonical location, enabling the
    # trainer's mandatory re-publication before resume without duplicate bytes.
    header = {k: v for k, v in ack.items() if k not in ("ok", "committed_unix")}
    unit = destination / "transport-units" / header["unit_id"]
    unit.mkdir(parents=True)
    for relative in header["files"]:
        if not relative.startswith("checkpoints/"):
            target = unit / "files" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with (destination / relative).open("rb") as source, target.open("xb") as out:
                shutil.copyfileobj(source, out, 1024 * 1024)
                out.flush()
                os.fsync(out.fileno())
    _write_new(unit / "UNIT.json", _canonical(header))
    _save_ack(destination, header, ack)
    for parent, _, _ in os.walk(destination, topdown=False):
        _sync_dir(Path(parent))
    if time.monotonic() >= deadline:
        raise TransportError("Restore verification exceeded bounded total time")
    _write_new(destination / "RESTORE-ACK.json", _canonical(ack))
    _sync_dir(destination)
    return ack
