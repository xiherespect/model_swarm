"""CPU-only campaign scheduler contract tests; no GPU or worker execution."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_racer_campaign as campaign


def row(name, test=True):
    return {"id": name, "group": "A_main", "task": "nlgraph",
            "method": "racer_frozen_v1", "ablation": "full", "seed": 42,
            "run_test": test, "budget_samples": 1100}


class FakeChild:
    next_pid = 4000

    def __init__(self, command, stdout, **kwargs):
        self.pid = FakeChild.next_pid
        FakeChild.next_pid += 1
        self.returncode = 0
        self.command = command
        self.env = kwargs["env"]
        self.session = kwargs["start_new_session"]
        self.stdout_closed_at_spawn = stdout.closed
        job = Path(command[-1])
        (job.parent / "summary.json").write_text(json.dumps({"status": "succeeded", "budget_used": 7}))

    def poll(self):
        return self.returncode

    def wait(self):
        return self.returncode


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.snapshot = {"spec": {"protocol": "racer_eval_v1", "frozen_config_overrides": {}},
                         "experts": {"path": "/mock/experts", "names": ["expert-a"]},
                         "base_model": {"path": "/mock/model"}, "identity_sha256": "abc"}

    def test_job_preserves_full_run_and_effective_test_override(self):
        directory = self.base / "campaign"
        directory.mkdir()
        run_dir = campaign.prepare_run(row("a"), self.snapshot, directory, False)
        job = campaign.read_json(run_dir / "job.json")
        self.assertEqual(job["snapshot"], str(directory / "snapshot.json"))
        self.assertEqual(job["run"], dict(row("a"), run_test=False))
        self.assertEqual(campaign.read_json(directory / "snapshot.json"), self.snapshot)

    def test_attempt_bounds_counts_completed_and_uncertain_batches(self):
        results = self.base / "oracle/results"
        first = results / "slice-a"
        second = results / "slice-b"
        first.mkdir(parents=True)
        second.mkdir(parents=True)
        events = [{"state": "started", "indices": [0, 1]},
                  {"state": "completed", "indices": [0, 1], "generated_count": 2},
                  {"state": "started", "indices": [2, 3, 4]}]
        (first / "attempt.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n")
        (second / "attempt.jsonl").write_text(
            json.dumps({"state": "started", "indices": [8]}) + "\n{" )
        bounds = campaign.journal_attempt_bounds(self.base)
        self.assertEqual(bounds["completed_lower_bound"], 2)
        self.assertEqual(bounds["attempted_upper_bound"], 6)
        self.assertEqual(bounds["uncertain_inflight"], 4)
        self.assertEqual(bounds["journals"], 2)
        self.assertEqual(len(bounds["malformed_entries"]), 1)

    def test_preflight_hashes_sources_experts_and_symlinked_base_contents(self):
        root = self.base / "repo"
        (root / "experiments").mkdir(parents=True)
        (root / "data/eval").mkdir(parents=True)
        (root / "initial_experts").mkdir()
        (root / "weights").mkdir()
        (root / "model").mkdir()
        (root / "weights/weights.bin").write_bytes(b"first")
        (root / "model/model.safetensors").symlink_to("../weights/weights.bin")
        for name in ("search_frozen.py",) + campaign.IDENTITY_DEPENDENCIES:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
        for index in range(10):
            expert = root / "initial_experts" / f"expert-{index}"
            expert.mkdir()
            (expert / "adapter_model.safetensors").write_bytes(bytes([index]))
            (expert / "adapter_config.json").write_text("{}")
        (root / "data/eval/nlgraph.json").write_text('{"dev": []}')
        spec = {"source_files": ["search_frozen.py"], "main_tasks": ["nlgraph"],
                "ablation_tasks": [], "scale_tasks": []}
        (root / "experiments/racer_frozen_v1.json").write_text(json.dumps(spec))
        with patch.object(campaign, "ROOT", root), \
                patch.object(campaign, "SPEC", root / "experiments/racer_frozen_v1.json"), \
                patch.object(campaign, "task_datasets", return_value=["nlgraph"]), \
                patch.dict(campaign.os.environ, {"RACER_BASE_MODEL": str(root / "model")}):
            first = campaign.preflight()
            self.assertIn("tools/run_racer_campaign.py", first["sources"])
            self.assertIn("experiments/racer_data_audit_v1.json", first["sources"])
            self.assertEqual(len(first["experts"]["files"]), 20)
            base = first["base_model"]["files"]["model.safetensors"]
            self.assertEqual(base["sha256"], campaign.file_identity(root / "weights/weights.bin")["sha256"])
            (root / "weights/weights.bin").write_bytes(b"later")
            second = campaign.preflight()
            self.assertNotEqual(first["identity_sha256"], second["identity_sha256"])
            (root / "initial_experts/expert-0/adapter_config.json").write_text("changed")
            third = campaign.preflight()
            self.assertNotEqual(second["identity_sha256"], third["identity_sha256"])
            (root / "experiments/racer_data_audit_v1.json").write_text("changed")
            fourth = campaign.preflight()
            self.assertNotEqual(third["identity_sha256"], fourth["identity_sha256"])
            (root / "model/model.safetensors").unlink()
            (root / "model/model.safetensors").symlink_to("missing.bin")
            with self.assertRaises((campaign.CampaignError, OSError)):
                campaign.preflight()

    def test_query_selects_second_card_and_archives_raw_checks(self):
        class Result:
            def __init__(self, stdout):
                self.stdout, self.stderr, self.returncode = stdout, "", 0
        def smi(command, **kwargs):
            if command[1].startswith("--query-gpu="):
                return Result("0, GPU-a, 0, 0\n1, GPU-b, 0, 0\n")
            return Result("GPU-a, 900\n")
        path = self.base / "telemetry.json"
        with patch.object(campaign.subprocess, "run", side_effect=smi):
            gpu = campaign.query_idle_gpu(telemetry_path=path, excluded={0})
        self.assertEqual(gpu, {"index": 1, "uuid": "GPU-b"})
        raw = campaign.read_json(path)
        self.assertEqual(raw["queries"]["compute-apps"]["stdout"], "GPU-a, 900\n")
        self.assertEqual(raw["excluded"], [0])

    def test_malformed_or_failed_gpu_telemetry_blocks_and_is_saved(self):
        class Result:
            stdout, stderr, returncode = "garbled", "", 0
        path = self.base / "bad.json"
        with patch.object(campaign.subprocess, "run", return_value=Result()):
            with self.assertRaises(campaign.Blocked):
                campaign.query_idle_gpu(telemetry_path=path)
        self.assertIn("error", campaign.read_json(path))
        with patch.object(campaign.subprocess, "run", side_effect=FileNotFoundError("nvidia-smi")):
            with self.assertRaises(campaign.Blocked):
                campaign.query_idle_gpu(telemetry_path=path)
        self.assertIn("nvidia-smi", campaign.read_json(path)["error"])

    def test_no_card_records_selected_subset_as_blocked_without_spawn(self):
        with patch.object(campaign, "query_idle_gpu", side_effect=campaign.Blocked("busy")), \
                patch.object(campaign.subprocess, "Popen") as popen:
            ledger = campaign.execute_campaign([row("a"), row("b")], self.base / "campaign",
                                               {}, self.snapshot, only_ids={"b"})
        popen.assert_not_called()
        self.assertEqual(ledger["status"], "blocked")
        self.assertEqual([r["id"] for r in ledger["rows"]], ["b"])
        self.assertEqual(ledger["rows"][0]["status"], "blocked")
        self.assertEqual(campaign.read_json(self.base / "campaign/status.json"), ledger)

    def test_two_distinct_cards_fill_parallel_slots_and_close_logs(self):
        cards = {0: "GPU-a", 1: "GPU-b"}
        children = []
        checks = []
        def select(requested=None, excluded=(), **kwargs):
            checks.append((requested, set(excluded)))
            for index in (0, 1):
                if (requested is None or requested == index) and index not in excluded:
                    return {"index": index, "uuid": cards[index]}
            raise campaign.Blocked("no card")
        def spawn(*args, **kwargs):
            child = FakeChild(*args, **kwargs)
            children.append(child)
            return child
        with patch.object(campaign, "query_idle_gpu", side_effect=select), \
                patch.object(campaign.subprocess, "Popen", side_effect=spawn), \
                patch.object(campaign.os, "killpg"), \
                patch.object(campaign.time, "sleep"):
            ledger = campaign.execute_campaign([row("a"), row("b")], self.base / "campaign",
                                               {}, self.snapshot)
        self.assertEqual(ledger["status"], "succeeded")
        self.assertEqual([r["gpu"] for r in ledger["rows"]], [0, 1])
        self.assertEqual({c.env["CUDA_VISIBLE_DEVICES"] for c in children}, {"GPU-a", "GPU-b"})
        self.assertTrue(all(c.session and not c.stdout_closed_at_spawn for c in children))
        self.assertTrue(any(excluded == {0} for _, excluded in checks))
        self.assertTrue(all(c.command[1] == "-I" for c in children))

    def test_completed_worker_is_reaped_and_card_reused(self):
        selected = []
        def select(requested=None, excluded=(), **kwargs):
            selected.append(requested)
            if requested not in (None, 0) or 0 in excluded:
                raise campaign.Blocked("no card")
            return {"index": 0, "uuid": "GPU-a"}
        with patch.object(campaign, "query_idle_gpu", side_effect=select), \
                patch.object(campaign.subprocess, "Popen", side_effect=FakeChild), \
                patch.object(campaign.os, "killpg") as kill, \
                patch.object(campaign.time, "sleep"):
            ledger = campaign.execute_campaign([row("a"), row("b")], self.base / "campaign",
                                               {}, self.snapshot, max_parallel=1)
        self.assertEqual([r["status"] for r in ledger["rows"]], ["succeeded", "succeeded"])
        self.assertEqual([r["gpu"] for r in ledger["rows"]], [0, 0])
        self.assertEqual(kill.call_count, 2)
        self.assertIn(0, selected)

    def test_spawn_failure_records_failure_and_closes_log(self):
        logs = []
        def fail_spawn(command, stdout, **kwargs):
            logs.append(stdout)
            raise OSError("spawn unavailable")
        with patch.object(campaign, "query_idle_gpu", return_value={"index": 0, "uuid": "GPU-a"}), \
                patch.object(campaign.subprocess, "Popen", side_effect=fail_spawn):
            with self.assertRaises(OSError):
                campaign.execute_campaign([row("a")], self.base / "campaign", {}, self.snapshot)
        self.assertTrue(logs[0].closed)
        ledger = campaign.read_json(self.base / "campaign/status.json")
        self.assertEqual(ledger["status"], "failed")
        self.assertEqual(ledger["rows"][0]["status"], "failed")

    def test_selection_and_parallelism_fail_before_output_creation(self):
        for options in ({"only_ids": {"unknown"}}, {"only_ids": set()},
                        {"max_parallel": 0}, {"gpus": [0, 0]}, {"gpus": [-1]}):
            output = self.base / "bad"
            with self.subTest(options=options), self.assertRaises(campaign.CampaignError):
                campaign.execute_campaign([row("a")], output, {}, self.snapshot, **options)
            self.assertFalse(output.exists())

    def test_resume_appends_rows_and_keeps_snapshot(self):
        directory = self.base / "campaign"
        directory.mkdir()
        campaign.save_json(directory / "snapshot.json", self.snapshot)
        campaign.save_json(directory / "status.json",
                           {"status": "failed", "created_at": "t0",
                            "rows": [dict(row("done"), status="succeeded")]})
        (directory / "telemetry").mkdir()
        with patch.object(campaign, "query_idle_gpu",
                          return_value={"index": 0, "uuid": "GPU-a"}), \
                patch.object(campaign.subprocess, "Popen", side_effect=FakeChild), \
                patch.object(campaign.os, "killpg"), \
                patch.object(campaign.time, "sleep"):
            ledger = campaign.execute_campaign([row("new")], directory, {}, self.snapshot,
                                               resume=True)
        self.assertEqual(ledger["status"], "succeeded")
        self.assertEqual([r["id"] for r in ledger["rows"]], ["done", "new"])
        self.assertEqual([r["status"] for r in ledger["rows"]],
                         ["succeeded", "succeeded"])
        self.assertIn("resumed_at", ledger)
        self.assertEqual(campaign.read_json(directory / "snapshot.json"), self.snapshot)
        self.assertEqual(campaign.read_json(directory / "status.json"), ledger)

    def test_resume_rejects_missing_ledger_and_row_overlap(self):
        with patch.object(campaign, "query_idle_gpu"), \
                patch.object(campaign.subprocess, "Popen"):
            with self.assertRaises(campaign.CampaignError):
                campaign.execute_campaign([row("a")], self.base / "missing", {},
                                          self.snapshot, resume=True)
        self.assertFalse((self.base / "missing").exists())
        directory = self.base / "campaign"
        directory.mkdir()
        campaign.save_json(directory / "status.json",
                           {"status": "failed", "rows": [dict(row("a"), status="failed")]})
        with patch.object(campaign, "query_idle_gpu"), \
                patch.object(campaign.subprocess, "Popen"):
            with self.assertRaises(campaign.CampaignError):
                campaign.execute_campaign([row("a")], directory, {}, self.snapshot,
                                          resume=True)
        # 拒绝发生在写账本之前：既有行保持 failed，不被覆盖。
        self.assertEqual(campaign.read_json(directory / "status.json")["rows"][0]["status"],
                         "failed")


if __name__ == "__main__":
    unittest.main()
