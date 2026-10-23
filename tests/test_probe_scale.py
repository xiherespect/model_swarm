"""Standard-library-only pilot tests. All models/data/children are tiny fixtures.

python -I -m unittest discover -s /data/lvliping/merge_swarm/tests -p test_probe_scale.py
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("probe_scale", ROOT / "tools/probe_scale.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def tiny_tensor(path, value=1.0, key="layer.lora_A.weight"):
    header = json.dumps({key: {"dtype": "F32", "shape": [1, 1], "data_offsets": [0, 4]}}).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", value))


def archive(task="nlgraph", count=50):
    return {"scoring_version": "strict_v1", "dataset": task, "split": "dev",
            "subset_start": 0, "subset_stop": 50, "n_samples": count, "score": 1.0,
            "records": [{"index": i, "raw_generation": "answer", "parsed_answer": "answer",
                         "gold": "answer", "score": 1.0} for i in range(count)]}


DUMMY = r'''
import json,os,pathlib,sys,time
job_path=pathlib.Path(sys.argv[1]); mode=sys.argv[2]
job=json.loads(job_path.read_text())
snapshot=json.loads(pathlib.Path(job['snapshot']).read_text())
row=next(r for r in snapshot['candidates'] if r['id']==job['candidate_id'])
assert os.environ['HF_HUB_OFFLINE']=='1'
assert os.environ['TRANSFORMERS_OFFLINE']=='1'
assert os.environ['WANDB_DISABLED']=='true'
assert os.environ['CUDA_VISIBLE_DEVICES']=='GPU-fixture'
result=job_path.parent/'results'; result.mkdir()
count=7 if mode in ('partial','timeout','signal','incomplete') else 50
records=[dict(index=i,raw_generation='ok',parsed_answer='ok',gold='ok',score=1.0,
              question_sha256=snapshot['data'][row['task']]['items'][i]['evaluation_question_sha256'])
         for i in range(count)]
payload=dict(scoring_version='strict_v1',dataset=row['task'],split='dev',subset_start=0,
             subset_stop=50,n_samples=count,score=1.0,records=records)
(result/'evaluation.json').write_text(json.dumps(payload))
if mode=='timeout': time.sleep(30)
if mode=='signal':
 import signal
 os.kill(os.getppid(),signal.SIGTERM)
 time.sleep(30)
if mode in ('fail','partial'): sys.exit(9)
'''


class FixtureTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="probe-scale-test-")
        self.root = Path(self.temporary.name)
        self.config = copy.deepcopy(probe.read_json(probe.DEFAULT_CONFIG))
        self.config.update(base_model=str(self.root / "base"), budget_seconds=20,
                           candidate_timeout_seconds=3)
        for item in self.config["experts"]:
            directory = self.root / "initial_experts" / item["name"]
            write_json(directory / "adapter_config.json", {
                "peft_type": "LORA", "bias": "none", "r": 1,
                "lora_alpha": 1, "target_modules": ["q_proj", "v_proj"]})
            tiny_tensor(directory / "adapter_model.safetensors")
            item["config_sha256"] = probe.file_identity(directory / "adapter_config.json")["sha256"]
            item["weights_sha256"] = probe.file_identity(directory / "adapter_model.safetensors")["sha256"]
        base = self.root / "base"
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"):
            write_json(base / name, {})
        (base / "merges.txt").write_text("fixture\n")
        tiny_tensor(base / "model.safetensors", key="model.weight")
        for task in probe.TASKS:
            write_json(self.root / "data/eval" / f"{task}.json", {
                "dev": [{"id": f"fixture-{i}", "question": str(i), "answer": "ok"} for i in range(60)],
                "test": [{"question": "MUST NEVER GENERATE ON TEST"}]})
        for source in probe.SOURCES:
            path = self.root / source
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# synthetic source\n")
        (self.root / "evaluate.py").write_text(
            "def evaluate(model_path, eval_type, dataset, gpu_id, base_model=None, "
            "subset_size=None, subset_start=None, protocol=None, result_dir=None):\n    pass\n")
        self.output = self.root / "new-run"
        self.gpu = mock.Mock(return_value={"index": 0, "uuid": "GPU-fixture"})

    def tearDown(self):
        self.temporary.cleanup()

    def execute(self, mode="success", **kwargs):
        return probe.execute(self.config, self.output, root=self.root, gpu_check=self.gpu,
                             command_factory=lambda job: [sys.executable, "-I", "-c", DUMMY, str(job), mode],
                             **kwargs)

    def test_matrix_frozen_and_smoke_is_first(self):
        matrix = probe.build_matrix(self.config)
        self.assertEqual(len(matrix), 20)
        self.assertEqual(len({row['id'] for row in matrix}), 20)
        self.assertEqual(sum(row['smoke'] for row in matrix), 1)
        self.assertEqual(matrix[0]['task'], 'nlgraph')
        self.assertEqual(matrix[0]['scale'], 1.0)
        self.assertEqual(matrix[0]['alpha'], [1.0] + [0.0] * 9)
        self.assertEqual(matrix[0]['onehot_expert'], 'tulu_code_alpaca')
        for row in matrix:
            self.assertEqual(row['effective_delta_multiplier'], row['scale'] ** 2)
            self.assertAlmostEqual(sum(row['alpha']), row['scale'])
        self.assertEqual(20 * self.config['subset_size'], 1000)

    def test_factor_merge_cross_terms_and_square_scale(self):
        # scalar A/B are a small factor fixture; factor mix != delta mix.
        a, b, weights = [2.0, 3.0], [5.0, 7.0], [0.25, 0.75]
        delta = sum(x * w for x, w in zip(a, weights)) * sum(x * w for x, w in zip(b, weights))
        self.assertNotEqual(delta, sum(w * x * y for w, x, y in zip(weights, a, b)))
        for c in probe.SCALES:
            scaled = sum(c * w * x for w, x in zip(weights, a)) * sum(c * w * y for w, y in zip(weights, b))
            self.assertAlmostEqual(scaled, c * c * delta)
        self.assertEqual(a[0] * b[0], sum(w * x for w, x in zip([1, 0], a)) * sum(w * y for w, y in zip([1, 0], b)))

    def test_task_whitelist_and_no_test_or_matrix_expansion(self):
        for field, value in [('tasks', ['test']), ('tasks', ['gsm8k', 'nlgraph']),
                             ('split', 'test'), ('subset_start', 1), ('subset_size', 51),
                             ('scales', [2]), ('directions', ['winner']), ('max_samples', 1001)]:
            with self.subTest(field=field, value=value):
                config = copy.deepcopy(self.config)
                config[field] = value
                with self.assertRaises(probe.ProbeError):
                    probe.build_matrix(config)

    def test_budget_validation(self):
        for key in ('budget_seconds', 'candidate_timeout_seconds'):
            for value in (0, -1, 7201, float('inf'), float('nan'), True):
                with self.subTest(key=key, value=value):
                    config = copy.deepcopy(self.config)
                    config[key] = value
                    with self.assertRaises(probe.ProbeError):
                        probe.validate_config(config)

    def test_preflight_content_identity_and_no_writes(self):
        before = set(self.root.rglob('*'))
        with mock.patch.object(probe, 'query_idle_gpu', side_effect=AssertionError('GPU forbidden')):
            first = probe.preflight(self.config, self.root)
        self.assertEqual(before, set(self.root.rglob('*')))
        self.assertTrue(first['strict_api_ready'])
        self.assertEqual(first['base_model']['layout'], 'single_safetensors_no_index_needed')
        self.assertEqual(len(first['data']['gsm8k']['items']), 50)
        self.assertEqual(first['identity_sha256'], probe.preflight(self.config, self.root)['identity_sha256'])
        path = self.root / 'evaluate.py'
        stat = path.stat()
        path.write_text(path.read_text().replace('pass', 'None'))
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertNotEqual(first['identity_sha256'], probe.preflight(self.config, self.root)['identity_sha256'])

    def test_data_content_hash_not_mtime(self):
        first = probe.preflight(self.config, self.root)
        path = self.root / 'data/eval/gsm8k.json'
        stat = path.stat()
        data = probe.read_json(path)
        data['dev'][0]['question'] = 'x'
        write_json(path, data)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        second = probe.preflight(self.config, self.root)
        self.assertNotEqual(first['data']['gsm8k']['items'][0]['sha256'], second['data']['gsm8k']['items'][0]['sha256'])
        self.assertNotEqual(first['identity_sha256'], second['identity_sha256'])

    def test_pinned_expert_hash_detects_same_size_mtime_change(self):
        path = self.root / 'initial_experts' / self.config['experts'][0]['name'] / 'adapter_model.safetensors'
        stat = path.stat()
        tiny_tensor(path, value=2.0)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with self.assertRaisesRegex(probe.ProbeError, 'hash mismatch'):
            probe.preflight(self.config, self.root)

    def test_pool_names_and_incompatible_configs_rejected(self):
        extra = self.root / 'initial_experts/extra'
        extra.mkdir()
        with self.assertRaisesRegex(probe.ProbeError, 'identity mismatch'):
            probe.preflight(self.config, self.root)
        extra.rmdir()
        item = self.config['experts'][1]
        path = self.root / 'initial_experts' / item['name'] / 'adapter_config.json'
        cfg = probe.read_json(path)
        cfg['r'] = 2
        write_json(path, cfg)
        item['config_sha256'] = probe.file_identity(path)['sha256']
        with self.assertRaisesRegex(probe.ProbeError, 'Incompatible'):
            probe.preflight(self.config, self.root)

    def test_config_target_module_order_is_semantic_set(self):
        item = self.config['experts'][1]
        path = self.root / 'initial_experts' / item['name'] / 'adapter_config.json'
        cfg = probe.read_json(path)
        cfg['target_modules'].reverse()
        write_json(path, cfg)
        item['config_sha256'] = probe.file_identity(path)['sha256']
        self.assertTrue(probe.preflight(self.config, self.root)['strict_api_ready'])

    def test_snapshot_missing_files_and_truncation_fail_closed(self):
        path = self.root / 'base/model.safetensors'
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaises(probe.ProbeError):
            probe.preflight(self.config, self.root)
        tiny_tensor(path, key='model.weight')
        (self.root / 'base/tokenizer.json').unlink()
        with self.assertRaises(probe.ProbeError):
            probe.preflight(self.config, self.root)

    def test_snapshot_indexed_shards(self):
        base = self.root / 'base'
        (base / 'model.safetensors').rename(base / 'model-00001-of-00001.safetensors')
        write_json(base / 'model.safetensors.index.json', {
            'weight_map': {'model.weight': 'model-00001-of-00001.safetensors'}})
        self.assertEqual(probe.inspect_snapshot(base)['layout'], 'indexed_shards')
        write_json(base / 'model.safetensors.index.json', {'weight_map': {'model.weight': '../outside.safetensors'}})
        with self.assertRaises(probe.ProbeError):
            probe.inspect_snapshot(base)

    def test_existing_output_and_symlink_never_touched(self):
        self.output.mkdir()
        sentinel = self.output / 'sentinel'
        sentinel.write_text('preserve')
        with self.assertRaises(probe.ProbeError):
            self.execute()
        self.assertEqual(probe.main(['--output', str(self.output)]), 2)
        self.assertEqual(sentinel.read_text(), 'preserve')
        self.gpu.assert_not_called()
        link = self.root / 'dangling'
        link.symlink_to(self.root / 'absent')
        with self.assertRaises(probe.ProbeError):
            probe.execute(self.config, link, root=self.root, gpu_check=self.gpu)

    def test_materialize_targets_new_and_contained(self):
        directory = self.root / 'candidate'
        directory.mkdir()
        probe.require_new_target(directory / 'adapter', directory)
        (directory / 'adapter').mkdir()
        for path in (directory / 'adapter', self.root / 'other'):
            with self.assertRaises(probe.ProbeError):
                probe.require_new_target(path, directory)

    def test_all_dummy_candidates_complete_and_ledger_is_1000(self):
        state = self.execute()
        self.assertEqual(state['status'], 'succeeded', state)
        self.assertEqual(state['actual_samples_known'], 1000)
        self.assertFalse(state['has_unknown_sample_counts'])
        self.assertEqual(len(list(self.output.glob('*/worker.log'))), 20)
        self.assertTrue(all(row['status'] == 'succeeded' for row in state['ledger']))
        self.assertTrue(all(row['elapsed_seconds'] > 0 for row in state['ledger']))
        self.assertEqual(probe.read_json(self.output / 'status.json')['status'], 'succeeded')
        self.assertTrue((self.output / 'metadata.json').is_file())
        self.assertEqual(len((self.output / 'summary.csv').read_text().splitlines()), 21)

    def test_worker_failure_stops_immediately_and_keeps_partial(self):
        state = self.execute('partial')
        self.assertEqual(state['status'], 'failed')
        self.assertEqual(state['ledger'][0]['actual_samples'], 7)
        self.assertEqual(state['ledger'][1]['status'], 'pending')
        self.assertEqual(len(list(self.output.glob('*/job.json'))), 1)
        self.assertEqual(state['ledger'][0]['returncode'], 9)
        self.assertTrue((self.output / state['ledger'][0]['id'] / 'results/evaluation.json').exists())

    def test_timeout_kills_only_owned_child_retains_ledger(self):
        self.config['candidate_timeout_seconds'] = 0.15
        unrelated = subprocess.Popen([sys.executable, '-I', '-c', 'import time; time.sleep(30)'])
        try:
            state = self.execute('timeout')
            self.assertEqual(state['status'], 'timeout', state)
            self.assertEqual(state['ledger'][0]['status'], 'timeout')
            self.assertEqual(state['ledger'][0]['actual_samples'], 7)
            self.assertLess(state['elapsed_seconds'], 3)
            self.assertIsNone(unrelated.poll())
            self.assertEqual(state['ledger'][1]['status'], 'pending')
        finally:
            unrelated.kill()
            unrelated.wait()

    def test_cumulative_budget_limits_first_child(self):
        self.config['budget_seconds'] = 0.15
        state = self.execute('timeout')
        self.assertEqual(state['status'], 'budget_exhausted', state)
        self.assertEqual(len(list(self.output.glob('*/job.json'))), 1)
        self.assertLess(state['elapsed_seconds'], 3)
        self.assertGreaterEqual(state['elapsed_seconds'], 0.1)

    def test_budget_counts_gaps_between_children(self):
        self.config['budget_seconds'] = 0.3
        calls = 0
        def gpu(config, requested):
            nonlocal calls
            calls += 1
            if calls == 3:
                # Simulates a slow idle check; budget includes the entire gap.
                import time
                time.sleep(0.35)
            return {'index': 0, 'uuid': 'GPU-fixture'}
        self.gpu = gpu
        state = self.execute()
        self.assertEqual(state['status'], 'budget_exhausted')
        self.assertEqual(len(list(self.output.glob('*/worker.log'))), 1)

    def test_sigterm_cancels_child_and_preserves_terminal_status(self):
        state = self.execute('signal')
        self.assertEqual(state['status'], 'cancelled', state)
        self.assertEqual(state['ledger'][0]['actual_samples'], 7)
        self.assertEqual(state['ledger'][1]['status'], 'pending')
        self.assertEqual(probe.read_json(self.output / 'status.json')['status'], 'cancelled')

    def test_no_gpu_is_blocked_without_starting_children(self):
        self.gpu.side_effect = probe.Blocked('no idle GPU')
        state = self.execute()
        self.assertEqual(state['status'], 'blocked')
        self.assertEqual(len(list(self.output.glob('*/job.json'))), 0)
        self.assertTrue((self.output / 'status.json').exists())

    def test_no_strict_api_blocks_before_gpu(self):
        (self.root / 'evaluate.py').write_text('def evaluate(model_path):\n    pass\n')
        state = self.execute()
        self.assertEqual(state['status'], 'blocked')
        self.gpu.assert_not_called()

    def test_preflight_failure_retains_status(self):
        (self.root / 'base/tokenizer.json').unlink()
        state = self.execute()
        self.assertEqual(state['status'], 'failed')
        self.assertTrue((self.output / 'status.json').is_file())
        self.gpu.assert_not_called()

    def test_archive_requires_exactly_50_raw_generations_and_matching_score(self):
        path = self.root / 'archive.json'
        write_json(path, archive())
        self.assertEqual(probe.validate_archive(path, {'task': 'nlgraph'}), 1)
        for mutate in (lambda d: d['records'].pop(),
                       lambda d: d['records'][0].pop('raw_generation'),
                       lambda d: d.update(score=0.5),
                       lambda d: d.update(split='test'),
                       lambda d: d.update(n_samples=49),
                       lambda d: d['records'][0].update(index=99)):
            data = archive()
            mutate(data)
            write_json(path, data)
            with self.assertRaises(probe.ProbeError):
                probe.validate_archive(path, {'task': 'nlgraph'})
        self.assertIsNone(probe.archive_samples(self.root / 'missing.json'))

    def test_zero_exit_with_partial_archive_still_fails(self):
        state = self.execute('incomplete')
        self.assertEqual(state['status'], 'failed')
        self.assertEqual(state['ledger'][0]['returncode'], 0)
        self.assertEqual(state['ledger'][0]['actual_samples'], 7)
        self.assertEqual(state['ledger'][1]['status'], 'pending')

    def test_private_worker_with_stdlib_fake_dependencies_and_mutation_guard(self):
        # Exercise real --worker control flow without importing torch or using a GPU.
        (self.root / 'alpha_space.py').write_text('''
from pathlib import Path
import shutil
class ExpertPool:
    def __init__(self, directory):
        self.directory=Path(directory)
        self.names=sorted(p.name for p in self.directory.iterdir() if p.is_dir())
        self.n=len(self.names)
        self._experts=[{'fixture': b'equal'}]*self.n
    def materialize(self, alpha, target):
        target=Path(target)
        assert not target.exists()
        target.mkdir()
        for name in ('adapter_config.json','adapter_model.safetensors'):
            shutil.copyfile(self.directory/self.names[0]/name,target/name)
''')
        (self.root / 'torch.py').write_text('def equal(a,b): return a==b\n')
        (self.root / 'safetensors').mkdir()
        (self.root / 'safetensors/__init__.py').write_text('')
        (self.root / 'safetensors/torch.py').write_text("def load_file(path,device): return {'fixture': b'equal'}\n")
        (self.root / 'evaluate.py').write_text('''
import json,hashlib
from pathlib import Path
def evaluate(model_path,eval_type,dataset,gpu_id,base_model=None,subset_size=None,subset_start=None,protocol=None,result_dir=None):
    assert gpu_id==0 and subset_size==50 and subset_start==0 and protocol=='strict_v1'
    assert Path(base_model).is_dir() and Path(model_path).is_dir()
    rows=json.loads((Path(__file__).parent/'data/eval'/f'{dataset}.json').read_text())['dev'][:50]
    records=[dict(index=i,question_sha256=hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),raw_generation='ok',parsed_answer='ok',gold='ok',score=1.0) for i,row in enumerate(rows)]
    destination=Path(result_dir); destination.mkdir()
    (destination/'evaluation.json').write_text(json.dumps(dict(scoring_version='strict_v1',dataset=dataset,split='dev',subset_start=0,subset_stop=50,n_samples=50,records=records,score=1.0)))
    return 1.0
''')
        snapshot = probe.preflight(self.config, self.root)
        self.output.mkdir()
        write_json(self.output / 'metadata.json', snapshot)
        candidate = snapshot['candidates'][0]
        job_path = self.output / candidate['id'] / 'job.json'
        write_json(job_path, {'snapshot': str(self.output / 'metadata.json'),
                              'candidate_id': candidate['id'], 'gpu_id': 0})
        command = [sys.executable, '-I', str(ROOT / 'tools/probe_scale.py'), '--worker', str(job_path)]
        result = subprocess.run(command, capture_output=True, text=True,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES='GPU-fixture'), timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        status = probe.read_json(job_path.parent / 'worker_status.json')
        self.assertEqual(status['onehot_tensor_equality'], 'exact_cpu_all_tensors')
        self.assertEqual(status['actual_samples'], 50)
        # A second job with a changed source must fail before materializing/loading.
        candidate = snapshot['candidates'][1]
        job_path = self.output / candidate['id'] / 'job.json'
        write_json(job_path, {'snapshot': str(self.output / 'metadata.json'),
                              'candidate_id': candidate['id'], 'gpu_id': 0})
        with (self.root / 'evaluate.py').open('a') as handle:
            handle.write('\n# changed source\n')
        result = subprocess.run(command[:-1] + [str(job_path)], capture_output=True, text=True,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES='GPU-fixture'), timeout=10)
        self.assertEqual(result.returncode, 1)
        self.assertFalse((job_path.parent / 'adapter').exists())
        self.assertIn('changed after snapshot', result.stderr)

    def test_archive_question_hash_must_match_frozen_dev(self):
        path = self.root / 'archive.json'
        data = archive()
        items = probe.preflight(self.config, self.root)['data']['nlgraph']['items']
        for record, item in zip(data['records'], items):
            record['question_sha256'] = item['evaluation_question_sha256']
        write_json(path, data)
        self.assertEqual(probe.validate_archive(path, {'task': 'nlgraph'}, items), 1)
        data['records'][0]['question_sha256'] = 'wrong'
        write_json(path, data)
        with self.assertRaisesRegex(probe.ProbeError, 'question hash'):
            probe.validate_archive(path, {'task': 'nlgraph'}, items)


class GpuTests(unittest.TestCase):
    def config(self):
        return probe.read_json(probe.DEFAULT_CONFIG)

    def test_idle_gpu_checks_memory_utilization_processes(self):
        devices = '0, GPU-busy, 0, 0\n1, GPU-idle, 0, 0\n'
        with mock.patch.object(probe.subprocess, 'run', side_effect=[
                subprocess.CompletedProcess([], 0, devices, ''),
                subprocess.CompletedProcess([], 0, 'GPU-busy, 123\n', '')]) as call:
            self.assertEqual(probe.query_idle_gpu(self.config())['index'], 1)
            self.assertEqual(call.call_count, 2)
        for devices, processes in [('0, GPU-x, 200, 0\n', ''),
                                   ('0, GPU-x, 0, 5\n', ''),
                                   ('0, GPU-x, 0, 0\n', 'GPU-x, 5\n'),
                                   ('0, GPU-x, N/A, N/A\n', '')]:
            with mock.patch.object(probe.subprocess, 'run', side_effect=[
                    subprocess.CompletedProcess([], 0, devices, ''),
                    subprocess.CompletedProcess([], 0, processes, '')]):
                with self.assertRaises(probe.Blocked):
                    probe.query_idle_gpu(self.config())

    def test_nvidia_smi_failure_blocks(self):
        with mock.patch.object(probe.subprocess, 'run', side_effect=FileNotFoundError()):
            with self.assertRaises(probe.Blocked):
                probe.query_idle_gpu(self.config())

    def test_parent_import_has_no_heavy_dependencies(self):
        code = ("import importlib.util,sys; "
                f"s=importlib.util.spec_from_file_location('probe',{str(ROOT / 'tools/probe_scale.py')!r}); "
                "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
                "assert not any(x in sys.modules for x in ('torch','numpy','evaluate','alpha_space','search_lp'))")
        result = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
