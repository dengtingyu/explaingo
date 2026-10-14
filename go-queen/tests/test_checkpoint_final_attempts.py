"""Targeted final-attempt backlog fixtures; opaque bytes and in-memory receiver."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from goqueen import cloud_checkpoint_transport as transport
from goqueen.cloud_training_state import atomic_checkpoint

PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("final_attempt_receiver_fixture", PROJECT / "scripts/receive_cloud_checkpoint.py")
receiver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(receiver)


class FinalAttemptFixtures(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.local, self.remote = self.root / "local", self.root / "remote"
        self.local.mkdir()
        self.remote.mkdir()
        key, hosts = self.root / "fake-key", self.root / "fake-hosts"
        key.write_text("fixture-only")
        hosts.write_text("fixture-only")
        self.config = {"host": "fixture.invalid", "port": 22, "identity_file": str(key),
                       "known_hosts": str(hosts), "run_id": "final-attempt-fixture",
                       "receiver_sha256": hashlib.sha256((PROJECT / "scripts/receive_cloud_checkpoint.py").read_bytes()).hexdigest()}
        self.identity = {"fixture": True, "persistence": transport.public_identity(self.config)}
        (self.local / "RUN.json").write_bytes(transport._canonical(self.identity))
        self.sent = []
        sender = patch.object(transport, "_send_unit", side_effect=self.send)
        sender.start()
        self.addCleanup(sender.stop)

    def attempt(self, index):
        path = self.local / "attempts" / f"{index:06d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def checkpoint(self, step, attempt_index):
        attempt = self.attempt(attempt_index)
        (attempt / "steps.jsonl").write_text(json.dumps({"completed_updates": step}) + "\n")
        name = f"step-{step:08d}-" + ("initial" if step == 0 else "periodic")
        metadata = {"identities": self.identity,
                    "cursor": {"epoch_index": 0, "next_batch_index": step, "completed_updates": step},
                    "attempt": attempt.relative_to(self.local).as_posix(), "evaluations": []}
        def writer(path):
            (path / "adapter.safetensors").write_bytes(b"OPAQUE-NOT-TENSOR" + str(step).encode())
            (path / "training-state.pt").write_bytes(b"OPAQUE-NOT-PICKLE" + str(step).encode())
        record = atomic_checkpoint(self.local / "checkpoints", name, writer, metadata)
        transport.publish_checkpoint(self.local, record, self.config)
        return record

    def end(self, index, code):
        path = self.attempt(index)
        (path / "report.json").write_text(json.dumps({"status": "failed" if code else "complete"}))
        (path / "process-completion.json").write_text(json.dumps({"exit_code": code}))
        (path / "stdout.log").write_text(f"attempt {index} ended\n")
        (path / "runner-at-run.py").write_text("# fixture source only\n")

    def send(self, config, header, sources):
        self.sent.append(header["unit_id"])
        stream = io.BytesIO()
        transport._write_stream(stream, header, sources)
        return receiver.receive(self.remote, io.BytesIO(stream.getvalue()), config["receiver_sha256"])

    def test_resumed_run_publishes_previously_unacked_failed_attempt(self):
        self.checkpoint(0, 0)
        self.end(0, 1)
        latest = self.checkpoint(1, 1)
        self.end(1, 0)
        self.sent.clear()
        ack = transport.publish_final_artifacts(self.local, self.config)
        self.assertEqual(self.sent, ["final-attempt-000000", "final-attempt-000001"])
        self.assertEqual(ack["checkpoint_manifest_sha256"], latest.manifest_sha256)
        coverage = transport.validate_final_artifacts(self.local, self.config)
        self.assertTrue(coverage["all_ended_attempts_acknowledged"])
        self.assertEqual(coverage["ended_attempts"], ["000000", "000001"])
        self.assertEqual(coverage["latest_final_ack"], ack)
        prior = self.remote / self.config["run_id"] / "commits/final-attempt-000000/files/attempts/000000"
        self.assertEqual(json.loads((prior / "process-completion.json").read_text())["exit_code"], 1)
        self.assertTrue((prior / "runner-at-run.py").is_file())

    def test_partial_backlog_failure_retries_only_missing_receipt(self):
        self.checkpoint(0, 0)
        self.end(0, 1)
        self.checkpoint(1, 1)
        self.end(1, 0)
        def stop_second(config, header, sources):
            if header["unit_id"] == "final-attempt-000001":
                raise transport.TransportError("fixture connection failure")
            return self.send(config, header, sources)
        with patch.object(transport, "_send_unit", side_effect=stop_second):
            with self.assertRaises(transport.TransportError):
                transport.publish_final_artifacts(self.local, self.config)
        before = (self.local / "acks/final-attempt-000000.json").read_bytes()
        with self.assertRaises(transport.TransportError):
            transport.validate_final_artifacts(self.local, self.config)
        self.sent.clear()
        transport.publish_final_artifacts(self.local, self.config)
        self.assertEqual(self.sent, ["final-attempt-000001"])
        self.assertEqual((self.local / "acks/final-attempt-000000.json").read_bytes(), before)

    def test_existing_final_ack_keeps_old_checkpoint_binding(self):
        first = self.checkpoint(0, 0)
        self.end(0, 1)
        prior_ack = transport.publish_final_artifacts(self.local, self.config)
        self.assertEqual(prior_ack["checkpoint_manifest_sha256"], first.manifest_sha256)
        (self.attempt(0) / "durability-completion.json").write_text('{"later_receipt":true}')
        latest = self.checkpoint(1, 1)
        self.end(1, 0)
        self.sent.clear()
        ack = transport.publish_final_artifacts(self.local, self.config)
        self.assertEqual(self.sent, ["final-attempt-000001"])
        self.assertEqual(ack["checkpoint_manifest_sha256"], latest.manifest_sha256)
        self.assertEqual(json.loads((self.local / "acks/final-attempt-000000.json").read_text()), prior_ack)

    def test_incomplete_attempt_rejected_before_any_final_send(self):
        self.checkpoint(0, 0)
        self.end(0, 1)
        (self.attempt(1) / "stdout.log").write_text("still running")
        self.sent.clear()
        with self.assertRaises(transport.TransportError):
            transport.publish_final_artifacts(self.local, self.config)
        self.assertEqual(self.sent, [])
        self.assertFalse((self.local / "transport-units/final-attempt-000000").exists())

    def test_corrupt_earlier_cached_ack_prevents_complete_coverage(self):
        self.checkpoint(0, 0)
        self.end(0, 1)
        transport.publish_final_artifacts(self.local, self.config)
        latest = self.checkpoint(1, 1)
        self.end(1, 0)
        path = self.local / "acks/final-attempt-000000.json"
        ack = json.loads(path.read_text())
        ack["completed_updates"] = latest.metadata["cursor"]["completed_updates"]
        path.write_text(json.dumps(ack))
        self.sent.clear()
        with self.assertRaises(transport.TransportError):
            transport.publish_final_artifacts(self.local, self.config)
        self.assertEqual(self.sent, [])

    def test_pending_old_frozen_unit_is_not_rebound_or_silently_skipped(self):
        record = self.checkpoint(0, 0)
        self.end(0, 1)
        old = transport._freeze(self.local, record, self.config, "final", "final-attempt-000000",
                                transport._final_sources(self.local, self.attempt(0)))
        before = (self.local / "transport-units/final-attempt-000000/UNIT.json").read_bytes()
        self.checkpoint(1, 1)
        self.end(1, 0)
        with self.assertRaisesRegex(receiver.ProtocolError, "latest committed checkpoint"):
            transport.publish_final_artifacts(self.local, self.config)
        self.assertEqual((self.local / "transport-units/final-attempt-000000/UNIT.json").read_bytes(), before)
        self.assertEqual(old["completed_updates"], 0)
        self.assertFalse((self.local / "acks/final-attempt-000001.json").exists())

    def test_lost_old_local_ack_can_reconcile_existing_remote_commit(self):
        self.checkpoint(0, 0)
        self.end(0, 1)
        with patch.object(transport, "_save_ack", side_effect=OSError("fixture local write interruption")):
            with self.assertRaises(OSError):
                transport.publish_final_artifacts(self.local, self.config)
        self.checkpoint(1, 1)
        self.end(1, 0)
        transport.publish_final_artifacts(self.local, self.config)
        self.assertEqual(transport.validate_final_artifacts(self.local, self.config)["ended_attempts"], ["000000", "000001"])

    def test_changed_completed_evidence_does_not_inherit_old_receipt(self):
        self.checkpoint(0, 0)
        self.end(0, 1)
        transport.publish_final_artifacts(self.local, self.config)
        (self.attempt(0) / "stderr.log").write_text("new evidence after frozen end")
        with self.assertRaisesRegex(transport.TransportError, "changed after final snapshot"):
            transport.validate_final_artifacts(self.local, self.config)


    def test_checkpoint_only_restore_does_not_claim_terminal_history_coverage(self):
        self.checkpoint(0, 0)
        self.end(0, 1)
        transport.publish_final_artifacts(self.local, self.config)
        request = dict(transport.public_identity(self.config), kind="export_latest")
        raw = transport._canonical(request)
        import struct
        exported = io.BytesIO()
        receiver.receive(self.remote, io.BytesIO(transport.MAGIC + struct.pack(">Q", len(raw)) + raw),
                         self.config["receiver_sha256"], export_stream=exported)
        class FakeProcess:
            def __init__(self):
                self.stdin, self.stdout = io.BytesIO(), io.BytesIO(exported.getvalue())
            def wait(self, timeout=None):
                return 0
            def kill(self):
                pass
        destination = self.root / "fresh-restore"
        with patch.object(transport.subprocess, "Popen", return_value=FakeProcess()):
            transport.restore_latest(self.config, destination)
        self.assertFalse((destination / "attempts/000000/process-completion.json").exists())
        next_attempt = destination / "attempts/000001"
        next_attempt.mkdir()
        (next_attempt / "report.json").write_text('{"status":"complete"}')
        (next_attempt / "process-completion.json").write_text('{"exit_code":0}')
        self.sent.clear()
        with self.assertRaisesRegex(transport.TransportError, "historical terminal coverage unknown"):
            transport.publish_final_artifacts(destination, self.config)
        with self.assertRaisesRegex(transport.TransportError, "historical terminal coverage unknown"):
            transport.validate_final_artifacts(destination, self.config)
        self.assertEqual(self.sent, [])


if __name__ == "__main__":
    unittest.main()
