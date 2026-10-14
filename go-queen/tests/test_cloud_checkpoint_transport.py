"""Pure filesystem/protocol fault fixtures; fake payloads and mocked SSH only."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from unittest.mock import patch

from goqueen import cloud_checkpoint_transport as transport
from goqueen.cloud_training_state import atomic_checkpoint, resolve_checkpoint

PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("checkpoint_receiver_fixture", PROJECT / "scripts/receive_cloud_checkpoint.py")
receiver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(receiver)
REAL_SEND_UNIT = transport._send_unit


class TransportFixtures(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.local = self.root / "local"
        self.local.mkdir()
        self.remote = self.root / "remote"
        self.remote.mkdir()
        key, hosts = self.root / "fake-key", self.root / "fake-hosts"
        key.write_text("fixture, not a credential")
        hosts.write_text("fixture, not a host key")
        self.config = {"host": "fixture.invalid", "port": 22, "identity_file": str(key),
                       "known_hosts": str(hosts), "run_id": "fixture-run",
                       "receiver_sha256": hashlib.sha256((PROJECT / "scripts/receive_cloud_checkpoint.py").read_bytes()).hexdigest()}
        self.identity = {"fixture": True, "persistence": transport.public_identity(self.config)}
        (self.local / "RUN.json").write_bytes(transport._canonical(self.identity))
        self.attempt = self.local / "attempts/000000"
        self.attempt.mkdir(parents=True)
        (self.attempt / "steps.jsonl").write_text("")
        self.sender = patch.object(transport, "_send_unit", side_effect=self.send)
        self.sender.start()
        self.addCleanup(self.sender.stop)

    def checkpoint(self, step, kind="periodic", identity=None):
        name = "step-{:08d}-{}".format(step, kind)
        (self.attempt / "steps.jsonl").write_text(json.dumps({"completed_updates": step}) + "\n")
        evaluation = self.attempt / ("evaluation-{:08d}.json".format(step))
        evaluation.write_text(json.dumps({"fixture": True, "step": step}) + "\n")
        metadata = {"identities": self.identity if identity is None else identity,
                    "cursor": {"epoch_index": 0, "next_batch_index": step, "completed_updates": step},
                    "attempt": "attempts/000000", "evaluations": [{"file": evaluation.relative_to(self.local).as_posix(),
                      "sha256": hashlib.sha256(evaluation.read_bytes()).hexdigest()}]}
        def writer(path):
            (path / "adapter.safetensors").write_bytes(b"FAKE-NOT-A-TENSOR" + str(step).encode())
            (path / "training-state.pt").write_bytes(b"FAKE-NOT-PICKLE" + str(step).encode())
        return atomic_checkpoint(self.local / "checkpoints", name, writer, metadata)

    def wire(self, header, sources):
        output = io.BytesIO()
        transport._write_stream(output, header, sources)
        return output.getvalue()

    def send(self, config, header, sources):
        return receiver.receive(self.remote, io.BytesIO(self.wire(header, sources)),
                                config["receiver_sha256"], config.get("target_id", transport.TARGET))

    def freeze(self, record):
        return transport._freeze(self.local, record, self.config, "checkpoint", record.path.name,
                                 transport._checkpoint_sources(self.local, record))

    def request(self, kind):
        request = dict(transport.public_identity(self.config), kind=kind)
        raw = transport._canonical(request)
        return io.BytesIO(transport.MAGIC + struct.pack(">Q", len(raw)) + raw)

    def test_exact_ack_and_no_duplicate_checkpoint_payload(self):
        record = self.checkpoint(0, "initial")
        ack = transport.publish_checkpoint(self.local, record, self.config)
        self.assertEqual(transport.validate_ack(ack, self.local, record, self.config), ack)
        unit = self.local / "transport-units" / record.path.name
        self.assertFalse((unit / "files/checkpoints").exists())
        path, actual = receiver.restore_candidate(self.remote, self.config["run_id"])
        self.assertEqual(actual, ack)
        self.assertEqual(resolve_checkpoint(path / "checkpoints", self.identity).manifest_sha256, record.manifest_sha256)

    def test_corrupt_payload_never_commits(self):
        record = self.checkpoint(0, "initial")
        header = self.freeze(record)
        wire = bytearray(self.wire(header, transport._sources_for_unit(self.local, header)))
        wire[-1] ^= 1
        with self.assertRaises(receiver.ProtocolError):
            receiver.receive(self.remote, io.BytesIO(wire), self.config["receiver_sha256"])
        self.assertEqual(receiver.committed_units(self.remote / self.config["run_id"]), [])

    def test_interrupted_stream_never_commits(self):
        record = self.checkpoint(0, "initial")
        header = self.freeze(record)
        wire = self.wire(header, transport._sources_for_unit(self.local, header))
        with self.assertRaises(receiver.ProtocolError):
            receiver.receive(self.remote, io.BytesIO(wire[:-3]), self.config["receiver_sha256"])
        self.assertEqual(receiver.committed_units(self.remote / self.config["run_id"]), [])

    def test_idempotent_republish_reuses_frozen_log(self):
        record = self.checkpoint(0, "initial")
        ack = transport.publish_checkpoint(self.local, record, self.config)
        (self.attempt / "steps.jsonl").write_text("later entries are not recaptured\n")
        self.assertEqual(transport.publish_checkpoint(self.local, record, self.config), ack)

    def test_same_name_different_frozen_bytes_rejected(self):
        record = self.checkpoint(0, "initial")
        transport.publish_checkpoint(self.local, record, self.config)
        header = self.freeze(record)
        changed = json.loads(json.dumps(header))
        log = "attempts/000000/steps-snapshot-" + record.path.name + ".jsonl"
        source = self.root / "other-log"
        source.write_text("different\n")
        changed["files"][log] = transport._file(source)
        changed["unit_sha256"] = transport._sha(transport._canonical({k: v for k, v in changed.items() if k != "unit_sha256"}))
        sources = transport._sources_for_unit(self.local, header)
        sources[log] = source
        with self.assertRaisesRegex(receiver.ProtocolError, "Same unit name"):
            self.send(self.config, changed, sources)

    def test_retention_keeps_initial_epoch_and_latest_two_periodic(self):
        records, receipts = [], []
        for step, kind in [(0, "initial"), (1, "periodic"), (2, "periodic"), (3, "epoch-1"), (4, "periodic"), (5, "periodic")]:
            record = self.checkpoint(step, kind)
            records.append(record)
            receipts.append(transport.publish_checkpoint(self.local, record, self.config))
        retained = {p.name for p, _, _ in receiver.committed_units(self.remote / self.config["run_id"])}
        self.assertEqual(retained, {records[i].path.name for i in (0, 3, 4, 5)})
        # Historical local receipts remain valid descriptions even after remote retirement.
        transport.validate_ack(receipts[1], self.local, records[1], self.config)
        path, ack = receiver.restore_candidate(self.remote, self.config["run_id"])
        self.assertEqual(ack["completed_updates"], 5)
        self.assertEqual(resolve_checkpoint(path / "checkpoints", self.identity).metadata["cursor"]["completed_updates"], 5)

    def test_restore_rejects_corrupt_latest_without_falling_back(self):
        for step in (0, 1):
            record = self.checkpoint(step, "initial" if step == 0 else "periodic")
            transport.publish_checkpoint(self.local, record, self.config)
        unit = self.remote / self.config["run_id"] / "commits" / record.path.name
        (unit / "files/checkpoints" / record.path.name / "training-state.pt").write_bytes(b"corrupt")
        with self.assertRaises(receiver.ProtocolError):
            receiver.restore_candidate(self.remote, self.config["run_id"])

    def test_path_escape_and_source_identity_rejected(self):
        record = self.checkpoint(0, "initial")
        header = self.freeze(record)
        with self.assertRaises(receiver.ProtocolError):
            receiver.checked_header(header, "0" * 64, transport.TARGET)
        changed = json.loads(json.dumps(header))
        changed["files"]["../escape"] = {"bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}
        changed["unit_sha256"] = transport._sha(transport._canonical({k: v for k, v in changed.items() if k != "unit_sha256"}))
        with self.assertRaises(receiver.ProtocolError):
            receiver.checked_header(changed, self.config["receiver_sha256"], transport.TARGET)

    def test_mutated_evaluation_stops_before_network(self):
        record = self.checkpoint(0, "initial")
        (self.local / record.metadata["evaluations"][0]["file"]).write_text("changed")
        with self.assertRaises(transport.TransportError):
            transport.publish_checkpoint(self.local, record, self.config)

    def test_local_newest_unacked_refuses_final(self):
        record = self.checkpoint(0, "initial")
        transport.publish_checkpoint(self.local, record, self.config)
        self.checkpoint(1)
        (self.attempt / "report.json").write_text('{"status":"failed"}')
        (self.attempt / "process-completion.json").write_text('{"exit_code":1}')
        with self.assertRaises(transport.TransportError):
            transport.publish_final_artifacts(self.local, self.config)

    def test_failed_attempt_final_is_frozen_and_latest_matched(self):
        record = self.checkpoint(0, "initial")
        transport.publish_checkpoint(self.local, record, self.config)
        (self.attempt / "report.json").write_text('{"status":"failed"}')
        (self.attempt / "process-completion.json").write_text('{"exit_code":1}')
        ack = transport.publish_final_artifacts(self.local, self.config)
        self.assertEqual(ack["checkpoint_manifest_sha256"], record.manifest_sha256)
        (self.attempt / "durability-completion.json").write_text('{"ack":"later"}')
        self.assertEqual(transport.publish_final_artifacts(self.local, self.config), ack)
        self.assertNotIn("attempts/000000/durability-completion.json", ack["files"])

    def test_export_to_fresh_directory_and_resume_republication(self):
        record = self.checkpoint(0, "initial")
        ack = transport.publish_checkpoint(self.local, record, self.config)
        output = io.BytesIO()
        receiver.receive(self.remote, self.request("export_latest"), self.config["receiver_sha256"], export_stream=output)
        class FakeProcess:
            def __init__(self, raw):
                self.stdin, self.stdout = io.BytesIO(), io.BytesIO(raw)
            def wait(self, timeout=None):
                return 0
            def kill(self):
                pass
        destination = self.root / "restored"
        with patch.object(transport.subprocess, "Popen", return_value=FakeProcess(output.getvalue())):
            restored_ack = transport.restore_latest(self.config, destination)
        self.assertEqual(restored_ack, ack)
        self.assertTrue((destination / "RESTORE-ACK.json").is_file())
        restored = resolve_checkpoint(destination / "checkpoints", self.identity)
        transport.validate_ack(ack, destination, restored, self.config)
        self.assertEqual(transport.publish_checkpoint(destination, restored, self.config), ack)

    def test_explicit_connection_retry_classification_only(self):
        self.assertTrue(transport._connection_failure(255, b"Connection reset by peer"))
        self.assertFalse(transport._connection_failure(2, b"Connection reset by peer"))
        self.assertFalse(transport._connection_failure(255, b"Permission denied (publickey)"))
        self.assertFalse(transport._connection_failure(255, b"Host key verification failed"))

    def test_missing_checkpoint_cannot_claim_final(self):
        with self.assertRaises((transport.TransportError, FileNotFoundError)):
            transport.publish_final_artifacts(self.local, self.config)

    def test_failed_retirement_blocks_reack_and_export(self):
        for step in (0, 1, 2):
            record = self.checkpoint(step, "initial" if step == 0 else "periodic")
            transport.publish_checkpoint(self.local, record, self.config)
        record = self.checkpoint(3)
        original = receiver.shutil.rmtree
        def fail_discard(path, *args, **kwargs):
            if Path(path).name.startswith(".discarding-"):
                raise OSError("fixture interruption after retirement rename")
            return original(path, *args, **kwargs)
        with patch.object(receiver.shutil, "rmtree", side_effect=fail_discard):
            with self.assertRaises(OSError):
                transport.publish_checkpoint(self.local, record, self.config)
        with self.assertRaisesRegex(receiver.ProtocolError, "retirement"):
            transport.publish_checkpoint(self.local, record, self.config)
        with self.assertRaisesRegex(receiver.ProtocolError, "retirement"):
            receiver.receive(self.remote, self.request("export_latest"), self.config["receiver_sha256"], export_stream=io.BytesIO())

    def test_restore_metadata_failure_leaves_no_completion_marker(self):
        record = self.checkpoint(0, "initial")
        transport.publish_checkpoint(self.local, record, self.config)
        output = io.BytesIO()
        receiver.receive(self.remote, self.request("export_latest"), self.config["receiver_sha256"], export_stream=output)
        class FakeProcess:
            stdin = io.BytesIO()
            stdout = io.BytesIO(output.getvalue())
            def wait(self, timeout=None):
                return 0
            def kill(self):
                pass
        destination = self.root / "incomplete-restore"
        with patch.object(transport.subprocess, "Popen", return_value=FakeProcess()), patch.object(transport, "_save_ack", side_effect=OSError("fixture disk failure")):
            with self.assertRaises(OSError):
                transport.restore_latest(self.config, destination)
        self.assertFalse((destination / "RESTORE-ACK.json").exists())

    def test_network_only_retry_has_two_attempt_cap(self):
        record = self.checkpoint(0, "initial")
        header = self.freeze(record)
        sources = transport._sources_for_unit(self.local, header)
        config = transport._config(self.config)
        ack = dict(header, ok=True, committed_unix=1.0)
        class FakeProcess:
            def __init__(self, code):
                self.stdin = io.BytesIO()
                self.code = code
            def wait(self, timeout=None):
                return self.code
            def kill(self):
                pass
        calls = []
        def popen(args, stdin, stdout, stderr):
            calls.append(args)
            code = 255 if len(calls) == 1 else 0
            (stderr if code else stdout).write(b"Connection reset by peer" if code else transport._canonical(ack))
            return FakeProcess(code)
        with patch.object(transport.subprocess, "Popen", side_effect=popen):
            self.assertEqual(REAL_SEND_UNIT(config, header, sources), ack)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][-1], "root@fixture.invalid")
        calls.clear()
        def rejected(args, stdin, stdout, stderr):
            calls.append(args)
            stderr.write(b"Connection reset by peer")
            return FakeProcess(2)
        with patch.object(transport.subprocess, "Popen", side_effect=rejected):
            with self.assertRaises(transport.TransportError):
                REAL_SEND_UNIT(config, header, sources)
        self.assertEqual(len(calls), 1)

    def test_engineering_exception_preserves_original_manifest(self):
        identities = {"stage": "qwen38_expanded_contract_engineering_source_cache_repair",
                      "formal_resume_forbidden": True, "mode": "B", "updates": 5}
        (self.local / "RUN.json").write_bytes(transport._canonical(identities))
        metadata = {"identities": identities, "engineering_only": True,
                    "cursor": {"epoch_index": 0, "next_batch_index": 5, "completed_updates": 5}}
        def writer(path):
            (path / "adapter.safetensors").write_bytes(b"FAKE-ENGINEERING-PAYLOAD")
        record = atomic_checkpoint(self.local / "checkpoints", "engineering-b5", writer, metadata)
        original_manifest = (record.path / "MANIFEST.json").read_bytes()
        ack = transport.publish_checkpoint(self.local, record, self.config)
        restored, _ = receiver.restore_candidate(self.remote, self.config["run_id"])
        self.assertEqual((restored / "checkpoints/engineering-b5/MANIFEST.json").read_bytes(), original_manifest)
        self.assertEqual(ack["checkpoint_manifest_sha256"], record.manifest_sha256)
        with self.assertRaises(transport.TransportError):
            transport.publish_final_artifacts(self.local, self.config)


if __name__ == "__main__":
    unittest.main()
