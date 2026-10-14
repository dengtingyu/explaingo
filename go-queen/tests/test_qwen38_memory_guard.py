"""Stdlib-only fixtures for cgroup estimates; no NumPy/Torch/model import."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("qwen38_entry_memory_fixture", ROOT / "scripts/train_qwen38_expanded.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
GIB = 1024**3
PER_POSITION = 11 * 361 * 768 * 4


class FakeTensor:
    def numel(self): return 11 * 361 * 768
    def element_size(self): return 4


class FakeCache:
    def __init__(self, count=130):
        self.capacity, self.count, self.loaded = count, count, []
    def __len__(self): return self.count
    def __iter__(self): return iter(range(self.count))
    def __getitem__(self, key):
        self.loaded.append(key)
        return FakeTensor()
    def stats(self): return {"evictions": 0}


def file_stat(**overrides):
    result = dict(file=50*GIB, inactive_file=49*GIB, file_dirty=0, file_writeback=0, shmem=0)
    result.update(overrides)
    return result


class MemoryEstimateTests(unittest.TestCase):
    def test_actual_post_exit_cache_dominated_snapshot(self):
        record = entry.cgroup_memory_estimate(59999997952, 53666603008,
            file_stat(file=53500686336, inactive_file=53287743488, file_dirty=4096, shmem=8192))
        self.assertEqual(record["raw_headroom_bytes"], 6333394944)
        self.assertEqual(record["clean_inactive_file_credit_bytes"], 53287731200)
        self.assertEqual(record["estimated_available_bytes"], 59621126144)
        need = 1404 * PER_POSITION + 8*GIB
        self.assertLess(record["raw_headroom_bytes"], need)
        self.assertGreater(record["estimated_available_bytes"], need)

    def test_anonymous_pressure_does_not_receive_file_credit(self):
        record = entry.cgroup_memory_estimate(60*GIB, 55*GIB, file_stat(file=GIB, inactive_file=0))
        self.assertEqual(record["estimated_available_bytes"], 5*GIB)
        self.assertLess(record["estimated_available_bytes"], 8*GIB)

    def test_dirty_writeback_and_shmem_are_not_credited(self):
        record = entry.cgroup_memory_estimate(60*GIB, 50*GIB,
            file_stat(file=40*GIB, inactive_file=39*GIB, file_dirty=15*GIB, file_writeback=10*GIB, shmem=12*GIB))
        self.assertEqual(record["clean_inactive_file_credit_bytes"], 2*GIB)
        zero = entry.cgroup_memory_estimate(60*GIB, 50*GIB,
            file_stat(file=GIB, inactive_file=GIB, file_dirty=2*GIB))
        self.assertEqual(zero["clean_inactive_file_credit_bytes"], 0)

    def test_small_inactive_pool_may_be_entirely_dirty(self):
        record = entry.cgroup_memory_estimate(200, 100,
            file_stat(file=100, inactive_file=10, file_dirty=10))
        self.assertEqual(record["clean_inactive_file_credit_bytes"], 0)
        self.assertEqual(record["estimated_available_bytes"], 100)

    def test_unknown_or_incomplete_stat_has_zero_credit(self):
        for stat in (None, {}, {"file": 50*GIB}, file_stat(shmem=-1)):
            with self.subTest(stat=stat):
                record = entry.cgroup_memory_estimate(60*GIB, 55*GIB, stat)
                self.assertEqual(record["estimated_available_bytes"], 5*GIB)
                self.assertEqual(record["stat_credit_status"], "unknown_zero_credit")

    def test_credit_never_exceeds_current_or_limit(self):
        record = entry.cgroup_memory_estimate(10*GIB, 2*GIB, file_stat())
        self.assertEqual(record["clean_inactive_file_credit_bytes"], 2*GIB)
        self.assertEqual(record["estimated_available_bytes"], 10*GIB)

    def test_system_available_still_bounds_cgroup_estimate(self):
        files = {"/proc/meminfo": "MemAvailable: 10485760 kB\n",
                 "/sys/fs/cgroup/memory.max": str(60*GIB),
                 "/sys/fs/cgroup/memory.current": str(55*GIB),
                 "/sys/fs/cgroup/memory.stat": "\n".join(f"{k} {v}" for k,v in file_stat().items())}
        def read(path, *args, **kwargs):
            if str(path) not in files: raise FileNotFoundError(str(path))
            return files[str(path)]
        with patch.object(entry.Path, "read_text", read):
            available, evidence = entry.available_host_memory()
        self.assertEqual(available, 10*GIB)
        self.assertTrue(evidence["is_estimate"])
        self.assertGreater(evidence["cgroup_v2"]["estimated_available_bytes"], available)


class PreloadGuardTests(unittest.TestCase):
    def test_unchanged_eight_gib_preflight_rejects_one_byte_short(self):
        cache = FakeCache()
        required = len(cache)*PER_POSITION + 8*GIB
        with patch.object(entry, "available_host_memory", return_value=(required-1, {})), \
             patch.object(entry, "read_oom_events", return_value={"oom":0,"oom_kill":0,"oom_group_kill":0}):
            with self.assertRaises(MemoryError) as caught: entry.preload_feature_cache(cache)
        self.assertEqual(cache.loaded, [])
        self.assertEqual(caught.exception.memory_evidence["reserve_bytes"], 8*GIB)

    def test_checks_every64_and_final_without_changing_order(self):
        cache = FakeCache()
        oom = {"oom":0,"oom_kill":0,"oom_group_kill":0}
        with patch.object(entry, "available_host_memory", return_value=(60*GIB, {})), \
             patch.object(entry, "read_oom_events", return_value=oom):
            result = entry.preload_feature_cache(cache)
        self.assertEqual(cache.loaded, list(range(130)))
        self.assertEqual([r["positions_loaded"] for r in result["checkpoints"]], [64,128,130])
        self.assertEqual(result["tensor_bytes"], 130*PER_POSITION)
        self.assertEqual(result["reserve_bytes"], 8*GIB)
        self.assertTrue(result["oom_monitoring_available"])

    def test_midload_pressure_stops_before_loading_more(self):
        cache = FakeCache()
        oom = {"oom":0,"oom_kill":0,"oom_group_kill":0}
        remaining = (130-64)*PER_POSITION
        with patch.object(entry, "available_host_memory", side_effect=[(60*GIB,{}),(remaining+8*GIB-1,{})]), \
             patch.object(entry, "read_oom_events", return_value=oom):
            with self.assertRaises(MemoryError) as caught: entry.preload_feature_cache(cache)
        self.assertEqual(cache.loaded, list(range(64)))
        self.assertEqual(caught.exception.memory_evidence["status"], "failed")

    def test_oom_increment_stops_even_with_large_estimate(self):
        cache = FakeCache()
        before = {"oom":2,"oom_kill":1,"oom_group_kill":0}
        after = dict(before, oom=3)
        with patch.object(entry, "available_host_memory", return_value=(60*GIB,{})), \
             patch.object(entry, "read_oom_events", side_effect=[before,after]):
            with self.assertRaisesRegex(MemoryError,"OOM event"): entry.preload_feature_cache(cache)
        self.assertEqual(cache.loaded, list(range(64)))

    def test_unknown_events_remain_explicit_not_false_zero(self):
        cache = FakeCache(1)
        with patch.object(entry, "available_host_memory", return_value=(60*GIB,{})), \
             patch.object(entry, "read_oom_events", return_value=None):
            result = entry.preload_feature_cache(cache)
        self.assertIsNone(result["oom_events_before"])
        self.assertFalse(result["oom_monitoring_available"])


if __name__ == "__main__":
    unittest.main()
