"""Bounded, hash-checked CPU feature storage for position-keyed training.

Importing this module loads neither NumPy nor PyTorch. Each miss verifies one
NPZ before decoding its 11 saved FP32 blocks. Callers move only their current
batch to the GPU; this mapping never stores a GPU tensor.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import hashlib
from pathlib import Path
import re


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def feature_path(directory, relative):
    directory = Path(directory).resolve()
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Feature path must be relative and contained in its directory")
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory):
        raise ValueError("Feature path escapes its directory")
    return path


def load_cpu_blocks(path, record):
    """Decode only cached board activations, with no model construction."""
    import numpy as np
    import torch

    with np.load(path, allow_pickle=False) as saved:
        blocks = []
        for index in range(11):
            array = saved[f"trunk_after_block{index}"]
            if (array.shape != (361, 768) or array.dtype != np.float32
                    or not bool(np.isfinite(array).all())):
                raise ValueError("Expected finite FP32 [361,768] expert blocks")
            blocks.append(array)
        result = torch.from_numpy(np.stack(blocks))
    if result.device.type != "cpu" or result.requires_grad:
        raise ValueError("The position cache stores detached CPU features only")
    return result


class BoundedFeatureCache(Mapping):
    """Map position_id to a CPU [11,361,768] tensor using an LRU capacity.

    `loader` exists for small file/mapping tests; production uses load_cpu_blocks.
    Hits reuse the verified in-memory bytes. verify_files() rechecks source files
    separately, without loading tensors, when the caller closes a run.
    """
    def __init__(self, directory, positions, capacity=8, *, loader=None):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("Cache capacity must be a positive integer")
        self.directory, self.capacity = Path(directory).resolve(), capacity
        self.positions = {}
        for record in positions:
            item = dict(record)
            identity = item["position_id"]
            if not isinstance(identity, str) or not identity or identity in self.positions:
                raise ValueError("Position IDs must be nonempty and unique")
            if not re.fullmatch(r"[0-9a-f]{64}", item.get("feature_sha256", "")):
                raise ValueError("Each position needs an exact feature SHA256")
            feature_path(self.directory, item["feature_file"])
            self.positions[identity] = item
        self._loader = load_cpu_blocks if loader is None else loader
        self._cache = OrderedDict()
        self.hits = self.misses = self.evictions = self.peak_positions = 0

    def __len__(self):
        return len(self.positions)

    def __iter__(self):
        return iter(self.positions)

    def __getitem__(self, identity):
        record = self.positions[identity]
        if identity in self._cache:
            self.hits += 1
            self._cache.move_to_end(identity)
            return self._cache[identity]
        self.misses += 1
        path = feature_path(self.directory, record["feature_file"])
        if file_sha256(path) != record["feature_sha256"]:
            raise ValueError("Saved feature SHA256 changed: " + identity)
        # Evict before decoding, keeping retained feature storage bounded even
        # while the loader creates the next payload (caller-owned batches apart).
        if len(self._cache) >= self.capacity:
            self._cache.popitem(last=False)
            self.evictions += 1
        payload = self._loader(path, record)
        self._cache[identity] = payload
        self.peak_positions = max(self.peak_positions, len(self._cache))
        return payload

    def clear(self):
        self._cache.clear()

    def stats(self):
        return {"capacity_positions": self.capacity, "cached_positions": len(self._cache),
                "peak_cached_positions": self.peak_positions, "hits": self.hits,
                "misses": self.misses, "evictions": self.evictions}

    def verify_files(self):
        return all(file_sha256(feature_path(self.directory, record["feature_file"]))
                   == record["feature_sha256"] for record in self.positions.values())
