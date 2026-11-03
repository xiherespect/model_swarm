"""CPU-only worker/RealOracle contract tests; no model or GPU is loaded."""
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import search_frozen as worker


class FrozenRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        data = self.root / "data" / "eval"
        data.mkdir(parents=True)
        for name, dev, test in (("medqa", 120, 3), ("medmcqa", 125, 4),
                                ("nlgraph", 120, 3)):
            (data / f"{name}.json").write_text(json.dumps({"dev": [0] * dev,
                                                             "test": [0] * test}))
        root_patch = patch.object(worker, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        self.snapshot = {"task": "medical", "protocol": "racer_eval_v1",
                         "experts": {"path": str(self.root / "experts"),
                                     "names": ["a", "b"]},
                         "base_model": {"path": str(self.root / "base")}}
        self.instances = []
        tests = self

        class FakePool:
            def __init__(self, path):
                self.names = ["a", "b"]
                self.n = 2
                self.path = path
                tests.instances.append(self)

            def materialize(self, alpha, out_dir):
                # The real pool creates out_dir itself and refuses no preexisting path.
                assert not Path(out_dir).exists()
                Path(out_dir).mkdir()
                (Path(out_dir) / "weights").write_text(str(list(alpha)))

        def fake_evaluate(adapter, kind, dataset, gpu, *, subset_start,
                          subset_size, protocol, result_dir, **kwargs):
            assert Path(adapter).is_dir()
            assert protocol == "racer_eval_v1"
            assert not Path(result_dir).exists()
            dest = Path(result_dir)
            dest.mkdir()
            records = [{"index": i, "score": 0.5} for i in
                       range(subset_start, subset_start + subset_size)]
            (dest / "evaluation.json").write_text(json.dumps({
                "scoring_version": protocol, "split": "dev", "dataset": dataset,
                "subset_start": subset_start,
                "subset_stop": subset_start + subset_size,
                "n_samples": subset_size, "records": records}))

        def fake_test(adapter, kind, task, gpu, *, protocol, result_dir, **kwargs):
            dest = Path(result_dir)
            dest.mkdir()
            for dataset in worker.TASK_REGISTRY[task][1]:
                child = dest / dataset if len(worker.TASK_REGISTRY[task][1]) > 1 else dest
                child.mkdir(exist_ok=True)
                count = worker.load_split_size(dataset, "test")
                (child / "evaluation.json").write_text(json.dumps({
                    "dataset": dataset, "split": "test", "scoring_version": protocol,
                    "n_samples": count,
                    "records": [{"index": i} for i in range(count)]}))
            (dest / "evaluation.json").write_text(json.dumps({
                "split": "test", "scoring_version": protocol, "score": 0.75}))
            return 0.75

        modules = patch.dict(sys.modules, {
            "alpha_space": types.SimpleNamespace(ExpertPool=FakePool),
            "evaluate": types.SimpleNamespace(evaluate=fake_evaluate,
                                              evaluate_test=fake_test)})
        modules.start()
        self.addCleanup(modules.stop)

    def test_exact_alpha_identity(self):
        alpha = np.array([1., 0.])
        near = alpha.copy()
        near[0] = np.nextafter(1., 2.)
        self.assertNotEqual(worker.alpha_fingerprint(alpha), worker.alpha_fingerprint(near))
        oracle = worker.RealOracle(self.snapshot, self.root / "oracle")
        self.assertEqual(oracle.evaluate(alpha, 1)["new_samples"], 100)
        self.assertEqual(oracle.evaluate(near, 1)["new_samples"], 100)
        with self.assertRaisesRegex(ValueError, "finite"):
            worker.alpha_fingerprint([float("nan"), 0.])

    def test_incremental_rungs_and_exact_test_accounting(self):
        oracle = worker.RealOracle(self.snapshot, self.root / "oracle")
        alpha = np.array([1., 0.])
        self.assertEqual(oracle.evaluate(alpha, 1), {"score": 0.5, "new_samples": 100})
        self.assertEqual(oracle.evaluate(alpha, 2)["new_samples"], 100)
        self.assertEqual(oracle.evaluate(alpha, None)["new_samples"], 45)
        self.assertEqual(oracle.evaluate(alpha, 1)["new_samples"], 0)
        self.assertEqual(oracle.physical_samples, 245)
        self.assertEqual(len(self.instances), 1)
        self.assertEqual(len(list(oracle.adapters.iterdir())), 3)
        self.assertEqual(oracle.evaluate_test(alpha), 0.75)
        self.assertEqual(oracle.test_samples, 7)
        self.assertEqual(oracle.physical_samples, 252)
        self.assertEqual(len(list(oracle.adapters.iterdir())), 4)

    def test_expert_order_is_checked_before_materialization(self):
        snapshot = {**self.snapshot, "experts": {"path": "unused", "names": ["b", "a"]}}
        oracle = worker.RealOracle(snapshot, self.root / "oracle")
        with self.assertRaisesRegex(RuntimeError, "disagrees"):
            oracle.evaluate(np.array([1., 0.]), 1)
        self.assertEqual(list(oracle.adapters.iterdir()), [])

    def test_archive_mismatch_does_not_charge_or_cache(self):
        oracle = worker.RealOracle(self.snapshot, self.root / "oracle")
        eval_module = sys.modules["evaluate"]

        def wrong_archive(*args, result_dir, subset_start, subset_size, **kwargs):
            dest = Path(result_dir)
            dest.mkdir()
            (dest / "evaluation.json").write_text(json.dumps({
                "scoring_version": "racer_eval_v1", "split": "dev",
                "dataset": "medqa", "subset_start": subset_start,
                "subset_stop": subset_start + subset_size,
                "n_samples": subset_size + 1,
                "records": [{"index": i, "score": 0.5} for i in
                            range(subset_start, subset_start + subset_size)]}))

        with patch.object(eval_module, "evaluate", side_effect=wrong_archive):
            with self.assertRaisesRegex(RuntimeError, "archive slice mismatch"):
                oracle.evaluate(np.array([1., 0.]), 1)
        self.assertEqual(oracle.physical_samples, 0)
        self.assertEqual(oracle.item_cache[(worker.alpha_fingerprint([1., 0.]), "medqa", "racer_eval_v1")], {})

    def test_kcross_manifest_selection_and_archive_checks(self):
        import hashlib
        source_dir = self.root / "experiments"
        source_dir.mkdir()
        dataset = "knowledge_crosswords"
        suffix = "\nOptions:\nPlease select the correct answer from A, B, C, D."
        rows = [{"question": f"question {i}{suffix}"} for i in range(3)]
        source = self.root / "data" / "eval" / f"{dataset}.json"
        source.write_text(json.dumps({"dev": rows, "test": rows}))
        raw = source.read_bytes()
        from tools.audit_racer_data import stem
        entries = [{"index": i,
                    "question_sha256": hashlib.sha256(rows[i]["question"].encode()).hexdigest(),
                    "stem_sha256": hashlib.sha256(stem(rows[i]["question"], dataset).encode()).hexdigest()}
                   for i in (0, 2)]
        payload = json.dumps(entries, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False).encode()
        manifest = {"sources": {dataset: {"sha256": hashlib.sha256(raw).hexdigest()}},
                    "kcross_subset": {"selected": entries,
                                      "selected_sha256": hashlib.sha256(payload).hexdigest()}}
        audit = source_dir / "racer_data_audit_v1.json"
        audit.write_text(json.dumps(manifest))
        self.assertEqual(worker.kcross_test_selection(), entries)
        kcross = {**self.snapshot, "task": "kcross"}
        oracle = worker.RealOracle(kcross, self.root / "oracle")
        def fake_test(adapter, kind, task, gpu, *, protocol, result_dir,
                      subset_indices=None, **kwargs):
            # 回归锁：kcross 的 test 必须按数据集名 knowledge_crosswords 打开，
            # 不能把任务名当数据集名（曾导致 FileNotFoundError）。
            self.assertEqual(task, "knowledge_crosswords")
            self.assertEqual(kind, "multiple_choice")
            self.assertEqual(subset_indices, [0, 2])
            dest = Path(result_dir)
            dest.mkdir()
            (dest / "evaluation.json").write_text(json.dumps({
                "score": 0.75, "split": "test", "scoring_version": protocol,
                "dataset": dataset, "n_samples": 2, "subset_indices": [0, 2],
                "records": [{"index": entry["index"],
                             "question_text_sha256": entry["question_sha256"]}
                            for entry in entries]}))
            return 0.75
        with patch.object(sys.modules["evaluate"], "evaluate_test", side_effect=fake_test):
            self.assertEqual(oracle.evaluate_test(np.array([1., 0.]), selected=entries), 0.75)
        self.assertEqual(oracle.test_samples, 2)
        source.write_bytes(raw + b" ")
        with self.assertRaisesRegex(RuntimeError, "source disagrees"):
            worker.kcross_test_selection()

    def test_job_contract_and_worker_anytime(self):
        import hashlib
        expert_root = Path(self.snapshot["experts"]["path"])
        for name in self.snapshot["experts"]["names"]:
            folder = expert_root / name
            folder.mkdir(parents=True)
            (folder / "adapter_model.safetensors").write_bytes(b"weights")
        base_path = Path(self.snapshot["base_model"]["path"])
        base_path.mkdir()
        (base_path / "config.json").write_text("{}")
        from tools.run_racer_campaign import tree_identity
        self.snapshot["experts"]["files"] = {
            f"{name}/{relative}": entry
            for name in self.snapshot["experts"]["names"]
            for relative, entry in tree_identity(expert_root / name).items()}
        self.snapshot["base_model"]["files"] = tree_identity(base_path)
        campaign = {"spec": {"protocol": "racer_eval_v1",
                             "frozen_config_overrides": {"init_size": 2}},
                    "sources": {}, "data": {},
                    "experts": self.snapshot["experts"],
                    "base_model": self.snapshot["base_model"]}
        payload = json.dumps(campaign, sort_keys=True, separators=(",", ":")).encode()
        campaign["identity_sha256"] = hashlib.sha256(payload).hexdigest()
        campaign_file = self.root / "snapshot.json"
        campaign_file.write_text(json.dumps(campaign))
        run_dir = self.root / "run"
        run_dir.mkdir()
        job = run_dir / "job.json"
        job.write_text(json.dumps({"snapshot": str(campaign_file), "run": {
            "task": "nlgraph", "method": "random_box", "ablation": "full",
            "seed": 42, "budget_samples": 1000, "run_test": False}}))
        normalized = worker.load_run_snapshot(json.loads(job.read_text()))
        self.assertEqual(normalized["protocol"], "racer_eval_v1")
        self.assertEqual(normalized["task"], "nlgraph")
        self.assertFalse(normalized["run_test"])
        import search_core

        class FakeSearch:
            def __init__(self, costs):
                self.ledger = types.SimpleNamespace(budget_used=sum(costs),
                                                    candidates=[None, None])

            def snapshot_state(self):
                return {"mock": True}

            def run(self):
                return "budget_exhausted"

            def final_selection(self):
                return 0, np.array([1., 0.]), 0.5

            def anytime_at(self, marks):
                return [None, 0.5, 0.5]

            def _snapshot_base(self):
                return {"events": []}

        def build(*args, **kwargs):
            return FakeSearch(args[8])

        with patch.object(search_core, "build_search", side_effect=build), \
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "mock"}):
            self.assertEqual(worker.main(["--job", str(job)]), 0)
        summary = json.loads((run_dir / "summary.json").read_text())
        self.assertEqual(summary["anytime"], [
            {"budget": 300.0, "best": None},
            {"budget": 600.0, "best": 0.5},
            {"budget": 1000.0, "best": 0.5}])
        self.assertEqual(summary["physical_samples"], summary["budget_used"])
        self.assertEqual(summary["test_samples"], 0)

    def _gate_job(self):
        """Build a worker job using the real core and the fake evaluate module."""
        import hashlib
        expert_root = Path(self.snapshot["experts"]["path"])
        for name in self.snapshot["experts"]["names"]:
            folder = expert_root / name
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "adapter_model.safetensors").write_bytes(b"weights")
            (folder / "adapter_config.json").write_text("{}")
        base_path = Path(self.snapshot["base_model"]["path"])
        base_path.mkdir(exist_ok=True)
        (base_path / "config.json").write_text("{}")
        from tools.run_racer_campaign import tree_identity
        self.snapshot["experts"]["files"] = {
            f"{name}/{relative}": entry
            for name in self.snapshot["experts"]["names"]
            for relative, entry in tree_identity(expert_root / name).items()}
        self.snapshot["base_model"]["files"] = tree_identity(base_path)
        campaign = {"spec": {"protocol": "racer_eval_v1",
                             "frozen_config_overrides": {"init_size": 2}},
                    "sources": {}, "data": {},
                    "experts": self.snapshot["experts"],
                    "base_model": self.snapshot["base_model"]}
        payload = json.dumps(campaign, sort_keys=True, separators=(",", ":")).encode()
        campaign["identity_sha256"] = hashlib.sha256(payload).hexdigest()
        campaign_file = self.root / "gate_snapshot.json"
        campaign_file.write_text(json.dumps(campaign))
        run_dir = self.root / "gate_run"
        run_dir.mkdir()
        job = run_dir / "job.json"
        row = {"task": "nlgraph", "method": "random_box", "ablation": "full",
               "seed": 42, "budget_samples": 1000, "run_test": False}
        job.write_text(json.dumps({"snapshot": str(campaign_file), "run": row}))
        return job, run_dir, row

    def test_resume_rejects_mismatched_checkpoint_identity(self):
        job, run_dir, _ = self._gate_job()
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "mock"}):
            self.assertEqual(worker.main(["--job", str(job)]), 0)
        self.assertTrue((run_dir / "checkpoint.json").is_file())
        checkpoint = json.loads((run_dir / "checkpoint.json").read_text())
        checkpoint["identity_sha256"] = "0" * 64
        (run_dir / "checkpoint.json").write_text(json.dumps(checkpoint))
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "mock"}):
            self.assertEqual(worker.main(["--job", str(job), "--resume"]), 1)
        failed = json.loads((run_dir / "worker_status.json").read_text())
        self.assertEqual(failed["status"], "failed")
        self.assertIn("identity", failed["error"])

    def test_resume_rejects_oracle_drift_after_checkpoint(self):
        job, run_dir, _ = self._gate_job()
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "mock"}):
            self.assertEqual(worker.main(["--job", str(job)]), 0)
        # Tamper with a saved adapter file: resume must fail closed on drift.
        adapters = sorted((run_dir / "oracle" / "adapters").iterdir())
        self.assertTrue(adapters)
        (adapters[0] / "adapter_model.safetensors").write_bytes(b"tampered")
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "mock"}):
            self.assertEqual(worker.main(["--job", str(job), "--resume"]), 1)
        failed = json.loads((run_dir / "worker_status.json").read_text())
        self.assertEqual(failed["status"], "failed")


if __name__ == "__main__":
    unittest.main()
