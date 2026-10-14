"""Small standard-library/fake checks; no NumPy, Torch, models or CUDA run."""
from __future__ import annotations

import builtins
import contextlib
import copy
import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def import_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(data):
    return hashlib.sha256(data).hexdigest()


extract = import_file("cuda_extract_entry_for_test", "scripts/extract_features_cuda_batch.py")
cache_module = import_file("position_cache_for_test", "src/goqueen/cached_features.py")


class ImportAndSelectionTests(unittest.TestCase):
    def setUp(self):
        self.sources = {"games": [
            {"game_id": "train-a", "split": "train", "file": "data/a.sgf",
             "sgf_sha256": "a" * 64, "selected_plies": [73, 25]},
            {"game_id": "val-b", "split": "val", "file": "data/b.sgf",
             "sgf_sha256": "b" * 64, "selected_plies": [48]},
        ]}

    def test_import_has_no_heavy_dependencies(self):
        real_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "numpy", "sgfmill", "katago"}:
                raise AssertionError("Unexpected heavyweight import: " + name)
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded):
            import_file("cuda_extract_import_only", "scripts/extract_features_cuda_batch.py")
            import_file("position_cache_import_only", "src/goqueen/cached_features.py")

    def test_selection_preserves_whole_game_split_and_explicit_plies(self):
        plan = extract.selection_plan(self.sources)
        self.assertEqual([row["position_id"] for row in plan],
                         ["train-a-ply-025", "train-a-ply-073", "val-b-ply-048"])
        self.assertEqual([row["split"] for row in plan], ["train", "train", "val"])
        self.assertEqual(plan[0]["source_sgf"], "data/a.sgf")

    def test_no_implicit_sampling_and_no_override(self):
        del self.sources["games"][0]["selected_plies"]
        with self.assertRaises(ValueError):
            extract.selection_plan(self.sources)
        with self.assertRaises(ValueError):
            extract.selection_plan(self.sources, [25])
        for game in self.sources["games"]:
            game.pop("selected_plies", None)
        self.assertEqual(len(extract.selection_plan(self.sources, [25, 48])), 4)

    def test_invalid_selection_rejected(self):
        changes = [{"game_id": "../a"}, {"split": "implicit"}, {"sgf_sha256": "0"},
                   {"selected_plies": [1, 1]}, {"selected_plies": [True]},
                   {"selected_plies": [-1]}, {"selected_plies": []},
                   {"file": "/tmp/a.sgf"}, {"file": "data/../a.sgf"}]
        for change in changes:
            with self.subTest(change=change):
                value = copy.deepcopy(self.sources)
                value["games"][0].update(change)
                with self.assertRaises(ValueError):
                    extract.selection_plan(value)
        self.sources["games"][1]["game_id"] = "train-a"
        with self.assertRaises(ValueError):
            extract.selection_plan(self.sources)

    def test_paths_are_contained_in_project(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(extract.project_path("data/a.sgf", root), root.resolve() / "data/a.sgf")
            with self.assertRaises(ValueError):
                extract.project_path("../other.sgf", root)

    def test_invalid_cli_limits_rejected_before_inputs_or_gpu(self):
        for option in ("--batch-size", "--threads", "--seconds"):
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    extract.main(["--sources", "missing.json", "--output", "missing", option, "0"])
                self.assertEqual(raised.exception.code, 2)


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.output = Path(self.temporary.name)
        (self.output / "features").mkdir()
        self.planned = {"position_id": "a-ply-025", "game_id": "a", "split": "train",
                        "ply": 25, "source_sgf": "data/a.sgf", "sgf_sha256": "a" * 64}
        self.contract = {"selection": [self.planned], "implementation_sha256": {"entry.py": "1" * 64}}
        self.record = {**self.planned, "feature_file": "features/a-ply-025.npz",
                       "feature_sha256": sha(b"fake-cache"), "position_sha256": "b" * 64}
        (self.output / self.record["feature_file"]).write_bytes(b"fake-cache")
        self.manifest = {"contract": copy.deepcopy(self.contract),
                         "contract_sha256": extract.content_sha256(self.contract),
                         "positions": [copy.deepcopy(self.record)]}

    def tearDown(self):
        self.temporary.cleanup()

    def verify(self):
        return extract.verify_recorded_positions(self.manifest, self.contract, self.output)

    def test_verified_file_reused_without_decoding(self):
        rows, digests = self.verify()
        self.assertEqual(rows, {"a-ply-025": self.record})
        self.assertEqual(digests, {"b" * 64: "a-ply-025"})

    def test_resume_rejects_changed_contract(self):
        self.manifest["contract"]["implementation_sha256"]["entry.py"] = "2" * 64
        with self.assertRaises(ValueError):
            self.verify()

    def test_resume_rejects_changed_bytes_without_overwrite(self):
        path = self.output / self.record["feature_file"]
        path.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            self.verify()
        self.assertEqual(path.read_bytes(), b"changed")

    def test_resume_rejects_changed_split_path_and_missing_input_digest(self):
        for field, value in [("split", "val"), ("feature_file", "../a.npz"), ("position_sha256", "")]:
            with self.subTest(field=field):
                self.manifest["positions"] = [{**self.record, field: value}]
                with self.assertRaises(ValueError):
                    self.verify()

    def test_resume_rejects_duplicate_rows(self):
        self.manifest["positions"].append(copy.deepcopy(self.record))
        with self.assertRaises(ValueError):
            self.verify()

    def test_resume_rejects_orphan_without_overwrite_or_gpu(self):
        orphan = self.output / "features/unrecorded.npz.part"
        orphan.write_bytes(b"partial evidence")
        with self.assertRaisesRegex(ValueError, "Unrecorded"):
            self.verify()
        self.assertEqual(orphan.read_bytes(), b"partial evidence")


class BoundedCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.positions = []
        for identity in ("a", "b", "c"):
            data = ("fake-" + identity).encode()
            (self.directory / (identity + ".npz")).write_bytes(data)
            self.positions.append({"position_id": identity, "feature_file": identity + ".npz",
                                   "feature_sha256": sha(data)})
        self.loaded = []

        def loader(path, record):
            self.loaded.append(record["position_id"])
            return {"identity": record["position_id"], "bytes": path.read_bytes()}

        self.loader = loader

    def tearDown(self):
        self.temporary.cleanup()

    def test_lru_is_bounded_and_retains_recent_position(self):
        cache = cache_module.BoundedFeatureCache(self.directory, self.positions, 2, loader=self.loader)
        original = cache["a"]
        cache["b"]
        self.assertIs(cache["a"], original)
        cache["c"]
        self.assertEqual(list(cache._cache), ["a", "c"])
        cache["b"]
        self.assertEqual(self.loaded, ["a", "b", "c", "b"])
        self.assertEqual(cache.stats(), {"capacity_positions": 2, "cached_positions": 2,
                                        "peak_cached_positions": 2, "hits": 1,
                                        "misses": 4, "evictions": 2})
        self.assertEqual(list(cache), ["a", "b", "c"])
        self.assertEqual(len(cache), 3)

    def test_corrupt_file_rejected_before_loader(self):
        cache = cache_module.BoundedFeatureCache(self.directory, self.positions, 1, loader=self.loader)
        (self.directory / "a.npz").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            cache["a"]
        self.assertEqual(self.loaded, [])

    def test_hits_reuse_verified_payload_but_final_verification_sees_change(self):
        cache = cache_module.BoundedFeatureCache(self.directory, self.positions, 1, loader=self.loader)
        original = cache["a"]
        self.assertTrue(cache.verify_files())
        (self.directory / "a.npz").write_bytes(b"changed")
        self.assertIs(cache["a"], original)
        self.assertFalse(cache.verify_files())
        cache.clear()
        self.assertEqual(cache.stats()["cached_positions"], 0)
        with self.assertRaises(ValueError):
            cache["a"]

    def test_loader_failure_does_not_leave_more_than_capacity(self):
        cache = cache_module.BoundedFeatureCache(self.directory, self.positions, 1, loader=self.loader)
        cache["a"]
        cache._loader = lambda *_: (_ for _ in ()).throw(ValueError("decode failed"))
        with self.assertRaisesRegex(ValueError, "decode failed"):
            cache["b"]
        self.assertLessEqual(cache.stats()["cached_positions"], 1)

    def test_invalid_capacity_record_or_path_rejected(self):
        for capacity in (0, -1, True, 1.5):
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                cache_module.BoundedFeatureCache(self.directory, self.positions, capacity, loader=self.loader)
        for records in ([self.positions[0], self.positions[0]],
                        [{**self.positions[0], "feature_file": "../outside.npz"}],
                        [{**self.positions[0], "feature_sha256": "bad"}]):
            with self.subTest(records=records), self.assertRaises(ValueError):
                cache_module.BoundedFeatureCache(self.directory, records, loader=self.loader)


class PayloadBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.fp32 = object()
        self.arrays = {f"trunk_after_block{i}": SimpleNamespace(shape=(361, 768), dtype=self.fp32, finite=True)
                       for i in range(11)}
        self.tensor = SimpleNamespace(device=SimpleNamespace(type="cpu"), requires_grad=False)
        self.load_calls = []
        self.stacked = []

        def load(path, **kwargs):
            self.load_calls.append((path, kwargs))
            return contextlib.nullcontext(self.arrays)

        def stack(arrays):
            self.stacked.extend(arrays)
            return "fake-stacked-array"

        self.numpy = SimpleNamespace(load=load, float32=self.fp32, stack=stack,
                                     isfinite=lambda array: SimpleNamespace(all=lambda: array.finite))
        self.torch = SimpleNamespace(from_numpy=lambda _: self.tensor)

    def decode(self):
        with patch.dict(sys.modules, {"numpy": self.numpy, "torch": self.torch}):
            return cache_module.load_cpu_blocks(Path("fake.npz"), {})

    def test_decoder_requests_no_pickle_and_all_eleven_blocks(self):
        self.assertIs(self.decode(), self.tensor)
        self.assertEqual(self.load_calls, [(Path("fake.npz"), {"allow_pickle": False})])
        self.assertEqual(self.stacked, list(self.arrays.values()))

    def test_wrong_shape_dtype_or_nonfinite_is_rejected(self):
        for change in ({"shape": (768, 361)}, {"dtype": "float16"}, {"finite": False}):
            with self.subTest(change=change):
                self.arrays["trunk_after_block3"] = SimpleNamespace(
                    **({"shape": (361, 768), "dtype": self.fp32, "finite": True} | change))
                with self.assertRaises(ValueError):
                    self.decode()

    def test_device_and_detach_contract_is_checked(self):
        self.tensor.device.type = "cuda"
        with self.assertRaises(ValueError):
            self.decode()
        self.tensor.device.type = "cpu"
        self.tensor.requires_grad = True
        with self.assertRaises(ValueError):
            self.decode()


if __name__ == "__main__":
    unittest.main()
