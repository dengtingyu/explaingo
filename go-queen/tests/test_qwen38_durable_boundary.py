"""Stdlib-only boundary/retention checks; no tensors, child model or network."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


trainer = load("qwen38_durable_test_trainer", ROOT / "scripts/train_qwen38_expanded.py")
launcher = load("qwen38_durable_test_launcher", ROOT / "scripts/run_qwen38_expanded.py")
state = load("qwen38_durable_test_state", ROOT / "src/goqueen/cloud_training_state.py")


class DurableBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)
        self.identities = {"recipe": "synthetic-persistence-only"}

    def checkpoint(self, number, suffix="periodic"):
        name = f"step-{number:08d}-{suffix}"
        def writer(directory):
            (directory / "payload.bin").write_bytes(f"payload-{number}".encode())
        return state.atomic_checkpoint(self.run / "checkpoints", name, writer,
            {"identities": self.identities, "cursor": {"epoch_index": 0,
             "next_batch_index": number, "completed_updates": number},
             "evaluations": [], "attempt": "attempts/000000"})

    def transport(self, record, *, publish_error=None, invalid=False):
        calls = []
        ack = {"unit_sha256": "u" * 64, "checkpoint_manifest_sha256": record.manifest_sha256,
               "checkpoint_name": record.path.name}
        def publish(run, supplied_record, config):
            self.assertEqual(run, self.run)
            self.assertEqual(supplied_record, record)
            calls.append("publish")
            if publish_error:
                raise publish_error
            return ack
        def validate(received, run, supplied_record, config):
            calls.append("validate")
            self.assertEqual(supplied_record, record)
            if invalid or received != ack:
                raise ValueError("Mismatched exact receipt")
            return received
        return SimpleNamespace(publish_checkpoint=publish, validate_ack=validate), calls, ack

    def test_local_commit_never_advances_durable_counter_on_transport_failure(self):
        record = self.checkpoint(500)
        transport, calls, _ = self.transport(record, publish_error=OSError("transfer failed"))
        report = {"last_durable_updates": 0, "last_durable_checkpoint": "checkpoints/initial"}
        with self.assertRaises(OSError):
            trainer.publish_durable_boundary(self.run, record, {}, transport, report)
        self.assertEqual(calls, ["publish"])
        self.assertEqual(report["last_local_updates"], 500)
        self.assertEqual(report["last_durable_updates"], 0)
        self.assertTrue(report["checkpoint_publish_pending"])
        self.assertTrue(record.path.is_dir())

    def test_invalid_returned_receipt_does_not_mark_durable(self):
        record = self.checkpoint(500)
        transport, calls, _ = self.transport(record, invalid=True)
        report = {}
        with self.assertRaises(ValueError):
            trainer.publish_durable_boundary(self.run, record, {}, transport, report)
        self.assertEqual(calls, ["publish", "validate"])
        self.assertNotIn("last_durable_checkpoint", report)
        self.assertTrue(report["checkpoint_publish_pending"])

    def test_verified_receipt_advances_only_after_exact_validation(self):
        record = self.checkpoint(0, "initial")
        transport, calls, ack = self.transport(record)
        report = {}
        self.assertEqual(trainer.publish_durable_boundary(self.run, record, {}, transport, report), ack)
        self.assertEqual(calls, ["publish", "validate"])
        self.assertEqual(report["last_durable_updates"], 0)
        self.assertFalse(report["checkpoint_publish_pending"])
        self.assertEqual(report["last_durable_checkpoint"], str(record.path.relative_to(self.run)))

    def test_pruning_rejects_missing_ack_guard(self):
        newest = self.checkpoint(500)
        with self.assertRaisesRegex(ValueError, "persistent checkpoint receipts"):
            trainer.prune_periodic_checkpoints(self.run / "checkpoints", newest,
                self.identities, self.run / "receipts", state)
        self.assertTrue(newest.path.is_dir())

    def test_unacked_obsolete_checkpoint_is_preserved(self):
        old = self.checkpoint(500)
        middle = self.checkpoint(1000)
        newest = self.checkpoint(1500)
        def guard(record):
            if record.path == old.path:
                raise ValueError("No exact old receipt")
            return {"unit_sha256": str(record.metadata["cursor"]["completed_updates"])}
        with self.assertRaisesRegex(ValueError, "old receipt"):
            trainer.prune_periodic_checkpoints(self.run / "checkpoints", newest,
                self.identities, self.run / "receipts", state, guard)
        self.assertTrue(all(record.path.is_dir() for record in (old, middle, newest)))
        self.assertFalse((self.run / "receipts").exists())

    def test_verified_retention_keeps_initial_and_latest_two_periodic(self):
        initial = self.checkpoint(0, "initial")
        old = self.checkpoint(500)
        middle = self.checkpoint(1000)
        newest = self.checkpoint(1500)
        calls = []
        def guard(record):
            calls.append(record.path.name)
            return {"unit_sha256": str(record.metadata["cursor"]["completed_updates"])}
        removed = trainer.prune_periodic_checkpoints(self.run / "checkpoints", newest,
            self.identities, self.run / "receipts", state, guard)
        self.assertEqual(removed, [old.path.name])
        self.assertEqual(calls, [newest.path.name, old.path.name])
        self.assertTrue(all(record.path.is_dir() for record in (initial, middle, newest)))
        self.assertFalse(old.path.exists())
        receipt = json.loads((self.run / "receipts" / (old.path.name + ".json")).read_text())
        self.assertEqual(receipt["status"], "removed")
        self.assertEqual(receipt["replacement_durable_unit_sha256"], "1500")
        self.assertEqual(receipt["obsolete_durable_unit_sha256"], "500")

    def test_launcher_does_not_accept_local_only_final_success(self):
        args = SimpleNamespace(expected_train=126561, microbatch=16)
        total = 39555
        report = {"stage": launcher.STAGE, "status": "complete", "valid_complete": True,
            "restored_or_initial_updates": 0, "completed_updates": total,
            "example_passes": 632805, "epochs_completed": 5,
            "training_forward_attempts": total, "training_forward_returns": total,
            "backward_attempts": total, "backward_returns": total,
            "optimizer_step_attempts": total, "optimizer_step_returns": total,
            "evaluation_forward_attempts": 10, "evaluation_forward_returns": 10,
            "all_base_gradients_absent": True, "base_version_counters_unchanged": True,
            "cache_files_unchanged": True, "evaluations": [{"name": name} for name in
                ["initial", "epoch-1", "epoch-2", "epoch-3", "epoch-4", "epoch-5"]]}
        self.assertFalse(launcher.completed_contract(report, args))
        report.update(final_checkpoint_persistent_ack_verified=True,
                      checkpoint_publish_pending=False, last_durable_updates=total)
        self.assertTrue(launcher.completed_contract(report, args))
        report["last_durable_updates"] = total - 1
        self.assertFalse(launcher.completed_contract(report, args))

    def test_final_export_guard_does_not_substitute_an_older_checkpoint_receipt(self):
        old = self.checkpoint(500)
        newest = self.checkpoint(1000)
        (self.run / "RUN.json").write_text(json.dumps(self.identities))
        (self.run / "acks").mkdir()
        (self.run / "acks" / (old.path.name + ".json")).write_text("{}")
        transport = SimpleNamespace(validate_ack=lambda *args: self.fail("Newest receipt is absent"))
        with self.assertRaises(FileNotFoundError):
            launcher.latest_receipted_checkpoint(self.run, {}, transport, state)
        self.assertTrue(newest.path.is_dir())

    def test_final_artifact_ack_must_name_the_same_latest_checkpoint(self):
        old = self.checkpoint(500)
        newest = self.checkpoint(1000)
        older_ack = {"checkpoint_name": old.path.name,
                     "checkpoint_manifest_sha256": old.manifest_sha256,
                     "completed_updates": 500}
        with self.assertRaisesRegex(ValueError, "newest local"):
            launcher.validate_final_checkpoint_reference(older_ack, newest)
        current_ack = {"checkpoint_name": newest.path.name,
                       "checkpoint_manifest_sha256": newest.manifest_sha256,
                       "completed_updates": 1000}
        self.assertEqual(launcher.validate_final_checkpoint_reference(current_ack, newest), current_ack)

    def fake_parent_completion(self, *, receipt_error=False, complete_coverage=True):
        """Run the actual parent flow with a fake child; never invoke a process."""
        sys.path.insert(0, str(ROOT / "src"))
        self.addCleanup(lambda: sys.path.remove(str(ROOT / "src")))
        from goqueen import cloud_checkpoint_transport as real_transport
        project = (self.run / "project").resolve()
        for relative in ("scripts/train_qwen38_expanded.py", "scripts/receive_cloud_checkpoint.py",
                         "src/goqueen/cloud_checkpoint_transport.py", "private-config.json"):
            target = project / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("fixture bytes, not executed")
        output = project / "outputs/run"
        checkpoint_name = "step-00039555-epoch-5"
        manifest_sha = launcher.hashlib.sha256(b"{}").hexdigest()
        record = SimpleNamespace(path=output / "checkpoints" / checkpoint_name,
            manifest_sha256=manifest_sha, metadata={"cursor": {"completed_updates": 39555}})
        ack = {"checkpoint_name": checkpoint_name, "checkpoint_manifest_sha256": manifest_sha,
               "completed_updates": 39555, "unit_sha256": "f" * 64}
        class FakeChild:
            pid = 987654
            def __init__(self, *args, **kwargs):
                record.path.mkdir(parents=True)
                (record.path / "MANIFEST.json").write_bytes(b"{}")
                (record.path / "READY").write_text(manifest_sha + "\n")
                (output / "attempts/000000/report.json").write_text(json.dumps({
                    "final_checkpoint": "checkpoints/" + checkpoint_name,
                    "final_checkpoint_manifest_sha256": manifest_sha,
                    "final_checkpoint_durable_unit_sha256": ack["unit_sha256"]}))
            def poll(self):
                return 0
            def wait(self, timeout=None):
                return 0
        def latest(*args):
            if receipt_error:
                raise real_transport.TransportError("Exact receipt differs (fixture)")
            return record, ack
        def publish(*args):
            # Terminal evidence must exist before any artifact publication.
            completion = json.loads((output / "attempts/000000/process-completion.json").read_text())
            self.assertEqual(completion["exit_code"], 0)
            return ack
        original_is_relative = Path.is_relative_to
        def cloud_test_path(path, *other):
            return True if other == (Path("/workspace"),) else original_is_relative(path, *other)
        argv = ["launcher", "--hardware", "nf4-single-gpu", "--model-dir", "/workspace/model",
            "--asset-record", "/workspace/assets.json", "--asset-hashes", "/workspace/hashes.json",
            "--source-manifest", "data/source.json", "--feature-manifest", "data/features.json",
            "--legacy-manifest", "data/legacy.json", "--persistence-config", str(project / "private-config.json"),
            "--microbatch", "16", "--output", "outputs/run"]
        coverage = {"all_ended_attempts_acknowledged": complete_coverage,
                    "ended_attempts": ["000000"], "final_unit_sha256": {"000000": ack["unit_sha256"]},
                    "latest_final_ack": ack}
        with patch.object(launcher, "ROOT", project), patch.object(launcher, "RUNNER", project / "scripts/train_qwen38_expanded.py"), \
             patch.object(launcher.platform, "system", return_value="Linux"), patch.object(Path, "is_relative_to", cloud_test_path), \
             patch.object(sys, "argv", argv), patch.object(real_transport, "load_config", return_value={}), \
             patch.object(real_transport, "public_identity", return_value={"receiver_sha256": launcher.sha(project / "scripts/receive_cloud_checkpoint.py")}), \
             patch.object(launcher.subprocess, "Popen", FakeChild), patch.object(launcher, "completed_contract", return_value=True), \
             patch.object(launcher, "latest_receipted_checkpoint", side_effect=latest), \
             patch.object(real_transport, "publish_final_artifacts", side_effect=publish), \
             patch.object(real_transport, "validate_final_artifacts", return_value=coverage):
            code = launcher.main()
        attempt = output / "attempts/000000"
        return code, json.loads((attempt / "process-completion.json").read_text()), json.loads((attempt / "durability-completion.json").read_text())

    def test_real_transport_error_preserves_parent_terminal_records(self):
        code, completion, durability = self.fake_parent_completion(receipt_error=True)
        self.assertEqual(code, 1)
        self.assertEqual(completion["exit_code"], 0)
        self.assertFalse(completion["effective_complete"])
        self.assertIn("Exact receipt differs", completion["completion_verification_error"])
        self.assertFalse(durability["shutdown_ready"])
        self.assertFalse(durability["latest_local_checkpoint_ack_verified"])

    def test_missing_old_attempt_coverage_prevents_shutdown(self):
        code, completion, durability = self.fake_parent_completion(complete_coverage=False)
        self.assertEqual(code, 1)
        self.assertTrue(completion["effective_complete"])
        self.assertTrue(durability["latest_local_checkpoint_ack_verified"])
        self.assertFalse(durability["all_ended_attempts_acknowledged"])
        self.assertFalse(durability["shutdown_ready"])
        self.assertIn("every ended attempt", durability["export_error"])

    def test_exact_current_and_all_attempt_receipts_allow_shutdown(self):
        code, completion, durability = self.fake_parent_completion()
        self.assertEqual(code, 0)
        self.assertTrue(completion["effective_complete"])
        self.assertTrue(durability["all_ended_attempts_acknowledged"])
        self.assertTrue(durability["shutdown_ready"])


if __name__ == "__main__":
    unittest.main()
