"""CPU-only contract tests for the read-only frozen campaign reporter."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import racer_results as results


class ResultsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.spec = results.read_json(results.SPEC)
        self.budgets = {task: 10 for task in results.DATASETS}
        self.matrix = results.matrix(self.spec, self.budgets)
        self.campaign = self.base / "campaign"
        self.campaign.mkdir()
        self.identity_root = self.base / "identity"
        sources = {name: self.fake_source(name) for name in
                   self.spec["source_files"] +
                   ["experiments/racer_frozen_v1.json", "experiments/racer_data_audit_v1.json"]}
        datasets = {dataset for task in
                    self.spec["main_tasks"] + self.spec["scale_tasks"] + ["kcross"]
                    for dataset in results.DATASETS[task]}
        data = {dataset: self.fake_source(f"data/{dataset}.json") for dataset in datasets}
        experts = {"names": [f"e{i}" for i in range(10)], "files": {
            f"e{i}/{relative}": self.fake_source(f"experts/e{i}/{relative}")
            for i in range(10) for relative in ("adapter_config.json",
                                                "adapter_model.safetensors")}}
        base_files = {relative: self.fake_source(f"base/{relative}")
                      for relative in ("config.json", "model.safetensors")}
        self.snapshot = {"spec": self.spec, "sources": sources, "data": data,
                         "experts": experts,
                         "base_model": {"files": base_files}}
        self.snapshot["identity_sha256"] = hashlib.sha256(json.dumps(
            self.snapshot, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.save(self.campaign / "snapshot.json", self.snapshot)

    def fake_source(self, relative):
        path = self.identity_root / relative
        self.save(path, {"content": relative})
        return {"path": str(path), "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def save(self, path, obj):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj), encoding="utf-8")

    def report(self, entries):
        self.save(self.campaign / "status.json", {"status": "completed", "rows": entries})
        with patch.object(results, "budgets_from_data", return_value=self.budgets):
            return results.build_report(self.spec, self.campaign)

    def row(self, method="racer_frozen_v1", group="A_main", ablation="full"):
        return next(r for r in self.matrix if r["task"] == "nlgraph" and
                    r["method"] == method and r["group"] == group and
                    r["seed"] == 42 and r["ablation"] == ablation)

    def succeeded(self, row, score):
        directory = self.campaign / row["id"]
        directory.mkdir()
        self.save(directory / "job.json", {"snapshot": str(self.campaign / "snapshot.json"),
                                           "run": row})
        alpha = [0.1] * 10
        summary = {"status": "succeeded", "method": row["method"],
                   "task": row["task"], "ablation": row["ablation"],
                   "seed": row["seed"], "budget_samples": row["budget_samples"],
                   "budget_used": 10, "best_candidate_id": 0, "best_alpha": alpha,
                   "best_dev": 0.6, "test_score": score,
                   "kcross_filtered_test_score": None, "n_candidates": 1}
        self.save(directory / "summary.json", summary)
        self.save(directory / "ledger.json", {"method": row["method"],
                  "ablation": row["ablation"], "budget_samples": row["budget_samples"],
                  "budget_used": 10, "candidates": [{"id": 0, "alpha": alpha}],
                  "events": [{"kind": "observe", "fidelity": "full",
                              "candidate_id": 0, "score": 0.6}]})
        self.save(directory / "oracle/results/test_000001/evaluation.json",
                  {"split": "test", "scoring_version": "racer_eval_v1", "dataset": "nlgraph",
                   "score": score, "n_samples": 1,
                   "records": [{"index": 0, "score": score}]})
        return {**row, "status": "succeeded", "run_dir": str(directory),
                "summary": summary}

    def test_empty_campaign_reports_all_240_and_explicit_incomplete_pairs(self):
        report = self.report([])
        self.assertEqual(report["counts"], {"not_scheduled": 240})
        self.assertEqual(report["pair_counts"], {"incomplete": 189})
        self.assertTrue(all(p["test_delta_reference_minus_comparison"] is None
                            for p in report["pairs"]))

    def test_blocked_and_failed_are_never_interpreted_as_results(self):
        blocked = {**self.row(), "status": "blocked", "error": "no GPU"}
        failed = {**self.row("cma_full"), "status": "failed", "returncode": 1}
        report = self.report([blocked, failed])
        self.assertEqual(report["counts"]["blocked"], 1)
        self.assertEqual(report["counts"]["failed"], 1)
        self.assertEqual(report["pairs"][0]["status"], "incomplete")
        self.assertEqual(report["rows"][0]["error"], "no GPU")

    def test_verified_paired_result_requires_both_run_artifacts(self):
        first = self.succeeded(self.row(), 0.8)
        second = self.succeeded(self.row("cma_full"), 0.5)
        report = self.report([first, second])
        pair = next(p for p in report["pairs"] if p["comparison_id"] == second["id"])
        self.assertEqual(pair["status"], "verified")
        self.assertAlmostEqual(pair["test_delta_reference_minus_comparison"], 0.3)
        self.assertEqual(report["counts"]["verified"], 2)

    def test_missing_evidence_and_identity_mismatch_fail_closed(self):
        first = self.succeeded(self.row(), 0.8)
        second = self.succeeded(self.row("cma_full"), 0.5)
        (self.campaign / second["id"] / "oracle/results/test_000001/evaluation.json").unlink()
        report = self.report([first, second])
        self.assertEqual(report["counts"]["unverified"], 1)
        self.assertEqual(report["pair_counts"], {"incomplete": 189})
        second["task"] = "gsm8k"
        with self.assertRaises(results.EvidenceError):
            self.report([first, second])

    def test_dev_selection_mismatch_and_stale_parent_summary_fail_closed(self):
        first = self.succeeded(self.row(), 0.8)
        directory = self.campaign / first["id"]
        ledger = results.read_json(directory / "ledger.json")
        ledger["events"][0]["score"] = 0.1
        self.save(directory / "ledger.json", ledger)
        self.assertEqual(self.report([first])["counts"]["unverified"], 1)
        ledger["events"][0]["score"] = 0.6
        self.save(directory / "ledger.json", ledger)
        first["summary"]["test_score"] = 0.3
        self.assertEqual(self.report([first])["counts"]["unverified"], 1)

    def test_duplicate_unknown_and_partial_row_identities_rejected(self):
        pending = {**self.row(), "status": "pending"}
        for entries in ([pending, pending], [{**pending, "budget_samples": 1}],
                        [{**pending, "id": "unknown"}]):
            with self.subTest(entries=entries), self.assertRaises(results.EvidenceError):
                self.report(entries)

    def test_bad_snapshot_is_rejected(self):
        self.snapshot["spec"]["protocol"] = "legacy"
        self.save(self.campaign / "snapshot.json", self.snapshot)
        with self.assertRaises(results.EvidenceError):
            self.report([])


if __name__ == "__main__":
    unittest.main()
