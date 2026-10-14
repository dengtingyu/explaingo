"""Disk transactions with tiny byte payloads; no torch, model or optimizer."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "src/goqueen/cloud_training_state.py"
SPEC = importlib.util.spec_from_file_location("cloud_state_under_test", SOURCE)
state = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = state
SPEC.loader.exec_module(state)
IDENTITIES = {"model": "fixed-model", "data_sha256": "a" * 64, "recipe_sha256": "b" * 64}


def metadata(updates=1, epoch=0, batch=1):
    return {"identities": IDENTITIES,
            "cursor": {"epoch_index": epoch, "next_batch_index": batch, "completed_updates": updates}}


def writer(path):
    (path / "adapter.bin").write_bytes(b"all-adapter-payload")
    (path / "state").mkdir()
    (path / "state/optimizer-and-rng.json").write_text('{"optimizer":{},"rng":42}')


class CloudTrainingStateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "checkpoints"

    def tearDown(self):
        self.temporary.cleanup()

    def test_roundtrip_checks_nested_payload_and_pointer(self):
        original = state.atomic_checkpoint(self.root, "epoch-001", writer, metadata())
        restored = state.resolve_checkpoint(self.root, IDENTITIES)
        self.assertEqual(restored, original)
        self.assertEqual(set(restored.files), {"adapter.bin", "state/optimizer-and-rng.json"})
        pointer = json.loads((self.root / "latest.json").read_text())
        self.assertEqual(pointer["manifest_sha256"], restored.manifest_sha256)
        self.assertNotIn("torch", sys.modules)

    def test_writer_failure_leaves_no_committed_boundary(self):
        def broken(path):
            (path / "partial.bin").write_bytes(b"partial")
            raise RuntimeError("simulated interruption")
        with self.assertRaises(RuntimeError):
            state.atomic_checkpoint(self.root, "epoch-001", broken, metadata())
        self.assertFalse((self.root / "epoch-001").exists())
        with self.assertRaises(FileNotFoundError):
            state.resolve_checkpoint(self.root, IDENTITIES)

    def test_crash_after_directory_publish_does_not_restore_stale_pointer(self):
        state.atomic_checkpoint(self.root, "step-001", writer, metadata(1))
        with patch.object(state.os, "replace", side_effect=OSError("pointer publication interrupted")):
            with self.assertRaises(OSError):
                state.atomic_checkpoint(self.root, "step-002", writer, metadata(2, batch=2))
        self.assertEqual(json.loads((self.root / "latest.json").read_text())["name"], "step-001")
        self.assertEqual(state.resolve_checkpoint(self.root, IDENTITIES).path.name, "step-002")

    def test_existing_checkpoint_is_not_overwritten_and_epochs_preserved(self):
        state.atomic_checkpoint(self.root, "epoch-001", writer, metadata(2, epoch=1, batch=0))
        state.atomic_checkpoint(self.root, "epoch-002", writer, metadata(4, epoch=2, batch=0))
        self.assertTrue((self.root / "epoch-001/READY").is_file())
        with self.assertRaises(FileExistsError):
            state.atomic_checkpoint(self.root, "epoch-001", writer, metadata(6))

    def test_non_increasing_optimizer_boundary_is_rejected(self):
        state.atomic_checkpoint(self.root, "step-002", writer, metadata(2))
        for step in (1, 2):
            with self.assertRaises(ValueError):
                state.atomic_checkpoint(self.root, "other", writer, metadata(step))

    def test_recipe_or_data_identity_change_is_rejected_before_writer(self):
        state.atomic_checkpoint(self.root, "step-001", writer, metadata())
        with self.assertRaises(ValueError):
            state.resolve_checkpoint(self.root, {**IDENTITIES, "data_sha256": "c" * 64})
        calls = []
        wrong = metadata(2); wrong["identities"] = {**IDENTITIES, "model": "other"}
        with self.assertRaises(ValueError):
            state.atomic_checkpoint(self.root, "step-002", lambda p: calls.append(p), wrong)
        self.assertEqual(calls, [])

    def test_corrupt_latest_payload_does_not_fall_back(self):
        state.atomic_checkpoint(self.root, "step-001", writer, metadata(1))
        state.atomic_checkpoint(self.root, "step-002", writer, metadata(2))
        (self.root / "step-002/adapter.bin").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "payload"):
            state.resolve_checkpoint(self.root, IDENTITIES)

    def test_added_or_missing_payload_is_rejected(self):
        state.atomic_checkpoint(self.root, "step-001", writer, metadata())
        extra = self.root / "step-001/unlisted.bin"; extra.write_bytes(b"extra")
        with self.assertRaises(ValueError):
            state.resolve_checkpoint(self.root, IDENTITIES)
        extra.unlink(); (self.root / "step-001/adapter.bin").unlink()
        with self.assertRaises(ValueError):
            state.resolve_checkpoint(self.root, IDENTITIES)

    def test_incomplete_committed_record_and_changed_manifest_are_rejected(self):
        state.atomic_checkpoint(self.root, "step-001", writer, metadata())
        manifest = self.root / "step-001/MANIFEST.json"
        manifest.write_text(manifest.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "READY"):
            state.resolve_checkpoint(self.root, IDENTITIES)
        (self.root / "step-001/READY").unlink()
        with self.assertRaises(ValueError):
            state.resolve_checkpoint(self.root, IDENTITIES)

    def test_reserved_files_symlinks_and_unsafe_names_are_rejected(self):
        with self.assertRaises(ValueError):
            state.atomic_checkpoint(self.root, "../escape", writer, metadata())
        def reserved(path):
            (path / "READY").write_bytes(b"fake")
        with self.assertRaises(ValueError):
            state.atomic_checkpoint(self.root, "reserved", reserved, metadata())
        external = Path(self.temporary.name) / "outside"; external.write_bytes(b"untouched")
        def linked(path):
            (path / "payload").symlink_to(external)
        with self.assertRaises(ValueError):
            state.atomic_checkpoint(self.root, "linked", linked, metadata())
        self.assertEqual(external.read_bytes(), b"untouched")

    def test_cursor_and_nonfinite_metadata_rejected(self):
        for bad in (-1, True, 1.5):
            value = metadata(); value["cursor"]["completed_updates"] = bad
            with self.assertRaises(ValueError):
                state.atomic_checkpoint(self.root, "bad", writer, value)
        value = metadata(); value["loss"] = float("nan")
        with self.assertRaises(ValueError):
            state.atomic_checkpoint(self.root, "bad", writer, value)

    def test_saved_next_batch_cursor_covers_five_epochs_without_duplicates(self):
        # A seven-ID schedule has one full and one partial batch each epoch.
        # The fake writer persists exactly the boundary after three batches.
        batches = [(epoch, batch, list(range(7))[batch * 4:(batch + 1) * 4])
                   for epoch in range(5) for batch in range(2)]
        completed = batches[:3]
        def snapshot(path):
            (path / "state.json").write_text(json.dumps(completed))
        state.atomic_checkpoint(self.root, "boundary-003", snapshot, metadata(3, epoch=1, batch=1))
        record = state.resolve_checkpoint(self.root, IDENTITIES)
        cursor = record.metadata["cursor"]
        next_index = cursor["epoch_index"] * 2 + cursor["next_batch_index"]
        restored_completed = json.loads((record.path / "state.json").read_text())
        rebuilt = restored_completed + [[e, b, ids] for e, b, ids in batches[next_index:]]
        self.assertEqual(rebuilt, [[e, b, ids] for e, b, ids in batches])
        for epoch in range(5):
            self.assertEqual([i for e, _, ids in rebuilt if e == epoch for i in ids], list(range(7)))


if __name__ == "__main__":
    unittest.main()
