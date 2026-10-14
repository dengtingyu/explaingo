"""Only mocked file descriptors and counters; no real fadvise, tensors or models."""
import importlib.util
from pathlib import Path
import stat
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('qwen_source_cache_fixture', ROOT/'scripts/train_qwen38_expanded.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
GIB = 1024**3


class SourceCacheAdviceTests(unittest.TestCase):
    def setUp(self):
        self.names = [f'asset-{i:02}.bin' for i in range(29)]
        self.assets = {'files':{n:'a'*64 for n in self.names}, 'file_sizes':{n:4096 for n in self.names},
                       'model_id':entry.MODEL, 'revision':entry.REVISION}
        self.opened, self.closed, self.advised = [], [], []
        def opened(path, flags):
            self.opened.append((path.name,flags))
            return 100+len(self.opened)
        def fstat(fd):
            return SimpleNamespace(st_mode=stat.S_IFREG|0o444,st_dev=1,st_ino=fd,st_size=4096,
                                   st_mtime_ns=123,st_ctime_ns=456)
        def advise(fd, offset, length, advice): self.advised.append((fd,offset,length,advice))
        self.patches = [patch.object(entry.os,'open',side_effect=opened),
            patch.object(entry.os,'fstat',side_effect=fstat), patch.object(entry.os,'close',side_effect=self.closed.append),
            patch.object(entry.os,'posix_fadvise',side_effect=advise,create=True),
            patch.object(entry.os,'POSIX_FADV_DONTNEED',4,create=True),
            patch.object(entry,'available_host_memory',side_effect=[(16*GIB,{'test':'before'}),(40*GIB,{'test':'after'})]),
            patch.object(entry,'read_oom_events',return_value={'oom':0,'oom_kill':0,'oom_group_kill':0}),
            patch.object(entry,'sha',side_effect=AssertionError('Payload SHA must not rewarm the cache'))]
        self.mocks=[p.start() for p in self.patches]
        self.addCleanup(lambda:[p.stop() for p in reversed(self.patches)])

    def test_exact_readonly_manifest_once_and_original_guard(self):
        result=entry.advise_loaded_model_source_cache('/workspace/model',self.assets,1404)
        self.assertEqual([n for n,_ in self.opened],self.names)
        self.assertTrue(all((flags & entry.os.O_ACCMODE)==entry.os.O_RDONLY and not flags & entry.os.O_TRUNC for _,flags in self.opened))
        self.assertEqual(self.advised,[(fd,0,0,4) for fd in range(101,130)])
        self.assertEqual(self.closed,list(range(101,130)))
        self.assertEqual(result['reserve_bytes'],8*GIB)
        self.assertEqual(result['required_preload_bytes'],1404*11*361*768*4+8*GIB)
        self.assertFalse(result['meets_preload_guard_before'])
        self.assertTrue(result['meets_preload_guard_after'])
        self.assertTrue(all(r['before']==r['after'] and r['advice_returned'] for r in result['files']))
        self.assertFalse(result['content_read_or_written'])
        self.assertFalse(result['global_cache_operation'])

    def test_insufficient_after_is_recorded_without_retry(self):
        self.mocks[5].side_effect=[(16*GIB,{}),(16*GIB,{})]
        result=entry.advise_loaded_model_source_cache('/workspace/model',self.assets,1404)
        self.assertFalse(result['meets_preload_guard_after'])
        self.assertEqual(result['status'],'advised_once')
        self.assertEqual(len(self.advised),29)

    def test_wrong_count_and_escape_refused_before_advice(self):
        bad=dict(self.assets,files={})
        with self.assertRaises(ValueError): entry.advise_loaded_model_source_cache('/workspace/model',bad,1404)
        self.assertEqual(self.opened,[])
        self.assets['files']['../escaped']=self.assets['files'].pop(self.names[-1])
        self.assets['file_sizes']['../escaped']=self.assets['file_sizes'].pop(self.names[-1])
        with self.assertRaises(ValueError): entry.advise_loaded_model_source_cache('/workspace/model',self.assets,1404)
        self.assertEqual(self.advised,[])
        self.assertEqual(len(self.closed),len(self.opened))

    def test_syscall_error_preserves_partial_evidence_and_closes_all(self):
        self.mocks[3].side_effect=OSError('unsupported filesystem')
        with self.assertRaises(OSError) as caught: entry.advise_loaded_model_source_cache('/workspace/model',self.assets,1404)
        self.assertEqual(self.mocks[3].call_count,1)
        self.assertEqual(len(self.closed),29)
        self.assertEqual(caught.exception.source_cache_advice_evidence['status'],'failed')
        self.assertFalse(any(r['advice_returned'] for r in caught.exception.source_cache_advice_evidence['files']))

    def test_oom_increment_stops_even_if_estimate_improved(self):
        self.mocks[6].side_effect=[{'oom':0,'oom_kill':0,'oom_group_kill':0},{'oom':1,'oom_kill':0,'oom_group_kill':0}]
        with self.assertRaisesRegex(MemoryError,'OOM event') as caught:
            entry.advise_loaded_model_source_cache('/workspace/model',self.assets,1404)
        self.assertEqual(caught.exception.source_cache_advice_evidence['oom_events_after']['oom'],1)
        self.assertEqual(len(self.closed),29)
        self.assertEqual(len(self.advised),29)


if __name__=='__main__': unittest.main()
