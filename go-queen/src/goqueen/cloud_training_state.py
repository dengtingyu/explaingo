"""Atomic, byte-checked training checkpoints without importing a tensor library.

The caller owns tensor/optimizer/RNG serialization and the epoch schedule. Call
this only at a completed optimizer boundary, with a cursor naming the *next*
batch. ``identities`` must pin the recipe, data/order, model and initialization.
Restoring an older in-memory state is never attempted here. Unsaved optimizer
updates can be recomputed after interruption; this module cannot make GPU work
and disk persistence one transaction. Only a published checkpoint is durable.

Each writer gets a fresh directory. Completed directories are immutable. READY
authenticates the manifest's bytes, and the manifest checks every payload file.
These checks detect accidental corruption, not malicious replacement of all
files. Never use them to justify unpickling an untrusted checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Callable


MANIFEST = "MANIFEST.json"
READY = "READY"
LATEST = "latest.json"
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


@dataclass(frozen=True)
class CheckpointRecord:
    path: Path
    metadata: dict
    files: dict
    manifest_sha256: str


def _json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _metadata_copy(metadata: dict) -> dict:
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")
    value = json.loads(_json_bytes(metadata))
    identities = value.get("identities")
    cursor = value.get("cursor")
    if not isinstance(identities, dict) or not identities or not all(isinstance(k, str) and k for k in identities):
        raise ValueError("metadata.identities must be a nonempty JSON object")
    if not isinstance(cursor, dict):
        raise ValueError("metadata.cursor is required")
    for key in ("epoch_index", "next_batch_index", "completed_updates"):
        if type(cursor.get(key)) is not int or cursor[key] < 0:
            raise ValueError(f"cursor.{key} must be a nonnegative integer")
    return value


def _root(path, create=False) -> Path:
    root = Path(path).absolute()
    if root.is_symlink():
        raise ValueError("checkpoint root cannot be a symlink")
    if create:
        root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise FileNotFoundError("checkpoint root does not exist")
    return root


def _file_info(path: Path, sync=False) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("checkpoint payload must contain only regular files")
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            hasher.update(chunk)
        if sync:
            os.fsync(stream.fileno())
    return {"sha256": hasher.hexdigest(), "bytes": size}


def _payload_inventory(directory: Path, sync=False) -> dict:
    files = {}
    for parent, directories, names in os.walk(directory, followlinks=False):
        base = Path(parent)
        for name in directories:
            if (base / name).is_symlink():
                raise ValueError("checkpoint symlink directory is forbidden")
        for name in sorted(names):
            path = base / name
            relative = path.relative_to(directory).as_posix()
            if relative in (MANIFEST, READY):
                continue
            files[relative] = _file_info(path, sync=sync)
    if not files:
        raise ValueError("checkpoint has no payload files")
    return files


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_file(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _read_manifest(directory: Path) -> CheckpointRecord:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("invalid committed checkpoint directory")
    for name in (MANIFEST, READY):
        if (directory / name).is_symlink() or not (directory / name).is_file():
            raise ValueError(f"committed checkpoint is missing a regular {name}")
    raw = (directory / MANIFEST).read_bytes()
    hashed = hashlib.sha256(raw).hexdigest()
    if (directory / READY).read_bytes() != (hashed + "\n").encode("ascii"):
        raise ValueError("READY does not match the checkpoint manifest")
    data = json.loads(raw)
    if data.get("format_version") != 1 or data.get("name") != directory.name:
        raise ValueError("checkpoint format/name mismatch")
    metadata = _metadata_copy(data.get("metadata"))
    files = data.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("checkpoint file manifest is empty")
    return CheckpointRecord(directory, metadata, files, hashed)


def _records(root: Path) -> list[CheckpointRecord]:
    records = []
    for path in root.iterdir():
        # Only our unpublished staging directories use this prefix. Do not
        # mistake a truncated committed directory for a recoverable old state.
        if path.name.startswith(".pending-") or path.name.startswith(".latest-"):
            continue
        if path.name == LATEST:
            if path.is_symlink() or not path.is_file():
                raise ValueError("latest pointer is not a regular file")
            continue
        if not _NAME.fullmatch(path.name):
            raise ValueError("unexpected entry in dedicated checkpoint root")
        records.append(_read_manifest(path))
    return records


def _latest_record(root: Path, expected_identities: dict) -> CheckpointRecord | None:
    expected = _json_bytes(expected_identities)
    records = _records(root)
    for record in records:
        if _json_bytes(record.metadata["identities"]) != expected:
            raise ValueError("immutable recipe/data/model identities differ")
    if not records:
        return None
    steps = [r.metadata["cursor"]["completed_updates"] for r in records]
    if len(set(steps)) != len(steps):
        raise ValueError("ambiguous checkpoint optimizer boundaries")
    return max(records, key=lambda r: r.metadata["cursor"]["completed_updates"])


def atomic_checkpoint(root, name: str, writer_callback: Callable[[Path], None], metadata: dict) -> CheckpointRecord:
    """Write and publish one immutable checkpoint; single-writer use is required.

    ``writer_callback(staging_path)`` writes and closes every payload file. It
    must include all adapter parameters (including frozen ones), optimizer,
    RNG, and any other runner state required at the supplied next-batch cursor.
    This function cannot inspect their tensor semantics. It refuses to replace
    existing names or to publish a non-increasing completed-update counter.

    An exception may leave a hidden ``.pending-*`` directory, deliberately
    ignored on recovery. A failure after publication can leave latest.json
    stale; resolve_checkpoint discovers the newest published boundary itself.
    Use a dedicated root; do not run concurrent saves/restores in that root.
    """
    if not isinstance(name, str) or not _NAME.fullmatch(name) or name == LATEST:
        raise ValueError("name must be a safe checkpoint directory basename")
    info = _metadata_copy(metadata)
    root = _root(root, create=True)
    destination = root / name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("completed checkpoint names are immutable")
    previous = _latest_record(root, info["identities"])
    if previous is not None and info["cursor"]["completed_updates"] <= previous.metadata["cursor"]["completed_updates"]:
        raise ValueError("checkpoint must advance the completed optimizer boundary")
    staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=root))
    writer_callback(staging)
    if staging.is_symlink() or not staging.is_dir():
        raise ValueError("writer replaced its staging directory")
    if any((staging / key).exists() or (staging / key).is_symlink() for key in (MANIFEST, READY)):
        raise ValueError("writer must not create reserved manifest/READY files")
    files = _payload_inventory(staging, sync=True)
    manifest = _json_bytes({"format_version": 1, "name": name, "metadata": info, "files": files})
    manifest_hash = hashlib.sha256(manifest).hexdigest()
    _write_file(staging / MANIFEST, manifest)
    _write_file(staging / READY, (manifest_hash + "\n").encode("ascii"))
    for parent, directories, _ in os.walk(staging, topdown=False):
        _sync_directory(Path(parent))
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("checkpoint destination appeared during save")
    os.rename(staging, destination)
    _sync_directory(root)
    pointer = root / (".latest-" + staging.name.removeprefix(".pending-"))
    _write_file(pointer, _json_bytes({"name": name, "manifest_sha256": manifest_hash}))
    os.replace(pointer, root / LATEST)
    _sync_directory(root)
    return CheckpointRecord(destination, info, files, manifest_hash)


def resolve_checkpoint(root, expected_identities: dict) -> CheckpointRecord:
    """Return the newest committed boundary after checking every payload byte.

    Never silently fall back if the latest checkpoint is corrupt. A stale or
    absent advisory latest.json is harmless because committed manifests carry
    the monotone optimizer counter. The caller then loads only its own trusted
    serialization, restores optimizer/RNG, and resumes the exact saved cursor.
    """
    if not isinstance(expected_identities, dict) or not expected_identities:
        raise ValueError("expected_identities must be a nonempty object")
    root = _root(root)
    record = _latest_record(root, expected_identities)
    if record is None:
        raise FileNotFoundError("no committed checkpoint")
    if _payload_inventory(record.path) != record.files:
        raise ValueError("checkpoint payload file set, size or SHA256 differs")
    return record
