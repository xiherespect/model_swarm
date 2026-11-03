#!/usr/bin/env python3
"""search_frozen.py — racer_frozen_v1 单 run 运行器（薄入口）。

绑定 search_core 的 oracle 回调到真实物化+评测：
  - 逐题增量缓存：命中身份 = (float64 alpha 精确字节, 数据集, 生成配置, 协议)；同一候选
    从 rung1 升 rung2 只补评新增题目，只计真实新增推理样本；跨候选不复用
    （不同 alpha 是不同模型）。共享初始化点在同进程内跨方法不共享（各方法
    独立 run 进程），无跨 run 缓存。
  - 每个候选物化到本 run 全新目录；评测走 evaluate.py 的 racer_eval_v1。
  - 状态/账本原子落盘；父驱动（tools/run_racer_campaign.py）负责 GPU 选择、
    超时与 kill 自有进程组。CUDA_VISIBLE_DEVICES 由父进程固定为本进程 GPU 0。

用法（父驱动调用，不直接手敲）:
  python -I search_frozen.py --job <run_dir>/job.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

TASK_REGISTRY = {
    "nlgraph": ("exact_match", ["nlgraph"]),
    "gsm8k": ("exact_match", ["gsm8k"]),
    "hellaswag": ("multiple_choice", ["hellaswag"]),
    "mmlu": ("multiple_choice", ["mmlu"]),
    "mmlu_pro": ("multiple_choice", ["mmlu_pro"]),
    "truthfulqa": ("multiple_choice", ["truthfulqa"]),
    "medical": ("multitask", ["medqa", "medmcqa"]),
    "science": ("multitask", ["scifact", "stem"]),
    "culture": ("multitask", ["normad_country", "normad_value"]),
    "kcross": ("multiple_choice", ["knowledge_crosswords"]),
}
RUNG1_PER_COMPONENT = 50
RUNG2_PER_COMPONENT = 100
OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "WANDB_DISABLED": "true", "WANDB_MODE": "disabled",
               "TOKENIZERS_PARALLELISM": "false"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_json_atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def journal_sample_bounds(results_dir):
    """Count durable completed batches and started attempts; unjournaled work is unknown."""
    lower = upper = 0
    for journal in Path(results_dir).rglob("attempt.jsonl"):
        started, completed = set(), set()
        for line in journal.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # A torn trailing write never counts as completed.
            indices = event.get("indices", [])
            if not isinstance(indices, list) or not all(isinstance(i, int) for i in indices):
                continue
            if event.get("state") == "started":
                started.update(indices)
            elif event.get("state") == "completed" and event.get("generated_count") == len(indices):
                completed.update(indices)
        lower += len(completed)
        upper += len(started | completed)
    return lower, upper


def alpha_fingerprint(alpha):
    """Exact float64 coefficients; near-but-distinct candidates never share inference."""
    values = np.asarray(alpha, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("alpha must be a finite one-dimensional vector")
    return values.astype("<f8", copy=False).tobytes().hex()


def load_split_size(dataset, split):
    with (ROOT / "data" / "eval" / f"{dataset}.json").open() as handle:
        return len(json.load(handle)[split])


def load_dev_size(dataset):
    return load_split_size(dataset, "dev")


def kcross_test_selection():
    """Fail closed unless the frozen subset matches the current test source."""
    manifest = json.loads((ROOT / "experiments" / "racer_data_audit_v1.json").read_text())
    source = ROOT / "data" / "eval" / "knowledge_crosswords.json"
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["sources"]["knowledge_crosswords"]["sha256"]:
        raise RuntimeError("kcross test source disagrees with audit manifest")
    entries = manifest["kcross_subset"]["selected"]
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    if hashlib.sha256(payload).hexdigest() != manifest["kcross_subset"]["selected_sha256"]:
        raise RuntimeError("kcross selected entries disagree with audit manifest")
    from tools.audit_racer_data import stem
    rows = json.loads(raw)["test"]
    indices = []
    for entry in entries:
        index = entry["index"]
        if not isinstance(index, int) or index < 0 or index >= len(rows) or \
                (indices and index <= indices[-1]):
            raise RuntimeError("kcross selection indices must be ascending and unique")
        text = rows[index]["question"]
        if (hashlib.sha256(text.encode("utf-8")).hexdigest() != entry["question_sha256"] or
                hashlib.sha256(stem(text, "knowledge_crosswords").encode("utf-8")).hexdigest()
                != entry["stem_sha256"]):
            raise RuntimeError(f"kcross selected question hash mismatch at {index}")
        indices.append(index)
    if not indices:
        raise RuntimeError("kcross selection is empty")
    return entries


def _file_identity_matches(entry):
    path = Path(entry["path"])
    if not path.is_file() or path.stat().st_size != entry["bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == entry["sha256"]


def verify_campaign_identity(campaign):
    identity = campaign["identity_sha256"]
    payload = {key: value for key, value in campaign.items() if key != "identity_sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(encoded).hexdigest() != identity:
        raise RuntimeError("campaign snapshot identity mismatch")
    if any(not _file_identity_matches(entry) for entry in campaign["sources"].values()):
        raise RuntimeError("campaign source file identity mismatch")
    if any(not _file_identity_matches(entry) for entry in campaign["data"].values()):
        raise RuntimeError("campaign data file identity mismatch")
    expert = campaign["experts"]
    names = expert["names"]
    location = Path(expert["path"])
    if (not isinstance(names, list) or not all(isinstance(name, str) for name in names)
            or not location.is_dir() or
            sorted(p.name for p in location.iterdir() if p.is_dir() and
                   (p / "adapter_model.safetensors").is_file()) != names):
        raise RuntimeError("campaign expert identity mismatch")
    from tools.run_racer_campaign import tree_identity
    try:
        expert_files = {f"{name}/{relative}": entry
                        for name in names
                        for relative, entry in tree_identity(location / name).items()}
        base_files = tree_identity(Path(campaign["base_model"]["path"]))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"campaign model file identity mismatch: {exc}") from exc
    if expert_files != expert.get("files") or base_files != campaign["base_model"].get("files"):
        raise RuntimeError("campaign model file identity mismatch")


def load_run_snapshot(job):
    """Combine the campaign identity snapshot with this worker's per-run row."""
    campaign = json.loads(Path(job["snapshot"]).read_text(encoding="utf-8"))
    verify_campaign_identity(campaign)
    row = job["run"]
    spec = campaign["spec"]
    required = ("task", "method", "ablation", "seed", "budget_samples", "run_test")
    if any(key not in row for key in required):
        raise ValueError("job.run is missing required run fields")
    return {**{key: row[key] for key in required},
            "protocol": spec["protocol"],
            "frozen_config_overrides": spec.get("frozen_config_overrides", {}),
            "experts": campaign["experts"], "base_model": campaign["base_model"],
            "identity_sha256": campaign["identity_sha256"]}


def contiguous_runs(indices):
    """把升序索引切成连续区间 [(start, stop), ...]（evaluate 的切片语义）。"""
    runs = []
    for i in sorted(indices):
        if runs and i == runs[-1][1]:
            runs[-1][1] = i + 1
        else:
            runs.append([i, i + 1])
    return [(a, b) for a, b in runs]


class RealOracle:
    """search_core 的真实评测 oracle。"""

    def __init__(self, snapshot, run_dir, resume=False):
        self.task = snapshot["task"]
        self.eval_type, self.datasets = TASK_REGISTRY[self.task]
        self.protocol = snapshot["protocol"]
        self.snapshot = snapshot
        self.is_multitask = self.eval_type == "multitask"
        self.run_dir = Path(run_dir)
        if resume:
            if not self.run_dir.is_dir():
                raise FileNotFoundError(self.run_dir)
        else:
            self.run_dir.mkdir(parents=True, exist_ok=False)
        self.adapters = self.run_dir / "adapters"
        self.results = self.run_dir / "results"
        if not resume:
            self.adapters.mkdir()
            self.results.mkdir()
        elif not self.adapters.is_dir() or not self.results.is_dir():
            raise RuntimeError("resume oracle directories are missing")
        self._pool = None
        self._dev_sizes = {d: load_dev_size(d) for d in self.datasets}
        # (exact float64 fingerprint, dataset, protocol) -> {item_index: record}
        self.item_cache = {}
        self._adapter_seq = 0
        self.calls = 0
        self.physical_samples = 0
        self.test_samples = 0
        self.uncertain_samples_upper_bound = 0
        self.generation_seconds = 0.0

    # ---------- checkpoint ----------
    def snapshot_cache(self):
        return {"adapter_seq": self._adapter_seq, "calls": self.calls,
                "physical_samples": self.physical_samples,
                "test_samples": self.test_samples,
                "generation_seconds": self.generation_seconds,
                "adapter_names": sorted(p.name for p in self.adapters.iterdir()),
                "result_names": sorted(p.name for p in self.results.iterdir()),
                "result_files": {str(p.relative_to(self.results)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in self.results.rglob("*") if p.is_file()},
                "items": [{"alpha": fp, "dataset": dataset, "protocol": protocol,
                           "records": [cached[i] for i in sorted(cached)]}
                          for (fp, dataset, protocol), cached in self.item_cache.items()]}

    def restore_cache(self, state):
        if self.item_cache or self.physical_samples or self._adapter_seq:
            raise RuntimeError("oracle must be empty before cache restore")
        if state["test_samples"] or state["physical_samples"] < 0:
            raise ValueError("checkpoint test state or sample count invalid")
        self._adapter_seq = state["adapter_seq"]
        self.calls = state["calls"]
        self.physical_samples = state["physical_samples"]
        self.generation_seconds = state["generation_seconds"]
        for row in state["items"]:
            dataset, protocol, fp = row["dataset"], row["protocol"], row["alpha"]
            if dataset not in self.datasets or protocol != self.protocol or len(fp) != 16 * len(
                    self.snapshot["experts"]["names"]):
                raise ValueError("oracle cache identity mismatch")
            key = fp, dataset, protocol
            if key in self.item_cache:
                raise ValueError("duplicate oracle cache key")
            records = {r["index"]: r for r in row["records"]}
            if len(records) != len(row["records"]) or any(
                    not isinstance(i, int) or i < 0 or i >= self._dev_sizes[dataset]
                    for i in records):
                raise ValueError("invalid oracle cache records")
            self.item_cache[key] = records
        if sum(map(len, self.item_cache.values())) != self.physical_samples:
            raise ValueError("oracle cache sample count mismatch")
        adapter_names = {p.name for p in self.adapters.iterdir()}
        result_names = {p.name for p in self.results.iterdir()}
        if (adapter_names != set(state["adapter_names"]) or
                result_names != set(state["result_names"]) or
                len(adapter_names) != self._adapter_seq):
            raise RuntimeError("oracle directories exceed checkpoint")
        if any(p.is_symlink() for root in (self.adapters, self.results)
               for p in root.rglob("*")):
            raise RuntimeError("oracle resume refuses symlinked files")
        result_files = {str(p.relative_to(self.results)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in self.results.rglob("*") if p.is_file()}
        if result_files != state["result_files"]:
            raise RuntimeError("oracle result files changed after checkpoint")
        for adapter in self.adapters.iterdir():
            if not adapter.is_dir() or not (adapter / "adapter_model.safetensors").is_file() or \
                    not (adapter / "adapter_config.json").is_file():
                raise RuntimeError("oracle adapter incomplete after checkpoint")
        for result in self.results.iterdir():
            if not result.is_dir() or not (result / "evaluation.json").is_file():
                raise RuntimeError("oracle has incomplete evaluation after checkpoint")

    # ---------- 基础 ----------
    def _pool_obj(self):
        if self._pool is None:
            from alpha_space import ExpertPool
            self._pool = ExpertPool(self.snapshot["experts"]["path"])
            names = self.snapshot["experts"]["names"]
            if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
                raise ValueError("snapshot experts.names must be a list of names")
            if self._pool.names != names:
                raise RuntimeError("ExpertPool discovery disagrees with snapshot")
        return self._pool

    def _materialize(self, alpha):
        self._adapter_seq += 1
        out = self.adapters / f"cand_{self._adapter_seq:06d}"
        if out.exists():
            raise FileExistsError(out)
        self._pool_obj().materialize(alpha, str(out))
        return out

    def _rung_stop(self, dataset, rung):
        if rung is None:
            return self._dev_sizes[dataset]
        size = RUNG1_PER_COMPONENT if rung == 1 else RUNG2_PER_COMPONENT
        return min(size, self._dev_sizes[dataset])

    def estimate(self, alpha, rung):
        """最坏新增样本数（忽略已有缓存；实际只少不多）。"""
        return sum(self._rung_stop(d, rung) for d in self.datasets)

    # ---------- 评测 ----------
    def _evaluate_slice(self, adapter, dataset, start, stop):
        """对单组件连续切片跑一次 strict/racer 评测，返回逐题记录。"""
        from evaluate import evaluate
        result_dir = self.results / f"{self._adapter_seq:06d}_{dataset}_{start}_{stop}"
        if result_dir.exists():
            raise FileExistsError(result_dir)
        kind = "exact_match" if dataset in ("nlgraph", "gsm8k") else "multiple_choice"
        evaluate(str(adapter), kind, dataset, 0,
                 base_model=self.snapshot["base_model"]["path"],
                 subset_size=stop - start, subset_start=start,
                 protocol=self.protocol, result_dir=str(result_dir))
        archive = json.loads((result_dir / "evaluation.json").read_text())
        records = archive["records"]
        if (archive.get("scoring_version") != self.protocol or
                archive.get("split") != "dev" or archive.get("dataset") != dataset or
                archive.get("subset_start") != start or
                archive.get("subset_stop") != stop or
                archive.get("n_samples") != stop - start or
                len(records) != stop - start or
                [r["index"] for r in records] != list(range(start, stop))):
            raise RuntimeError("evaluation archive slice mismatch")
        return records

    def evaluate(self, alpha, rung):
        self.calls += 1
        started = time.monotonic()
        fp = alpha_fingerprint(alpha)
        adapter = None
        new_total = 0
        for dataset in self.datasets:
            stop = self._rung_stop(dataset, rung)
            cached = self.item_cache.setdefault((fp, dataset, self.protocol), {})
            need = [i for i in range(stop) if i not in cached]
            if not need:
                continue
            if adapter is None:
                adapter = self._materialize(alpha)
            for a, b in contiguous_runs(need):
                try:
                    records = self._evaluate_slice(adapter, dataset, a, b)
                except BaseException:
                    self.uncertain_samples_upper_bound += b - a
                    raise
                for record in records:
                    cached[record["index"]] = record
                new_total += b - a
                self.physical_samples += b - a
        self.generation_seconds += time.monotonic() - started
        return {"score": self._aggregate(fp, rung), "new_samples": new_total}

    def _aggregate(self, fp, rung):
        """按当前 rung 覆盖区间聚合缓存记录；multitask 组件调和均值。"""
        from eval_protocol import harmonic_mean
        scores = []
        for dataset in self.datasets:
            stop = self._rung_stop(dataset, rung)
            cached = self.item_cache[(fp, dataset, self.protocol)]
            missing = [i for i in range(stop) if i not in cached]
            if missing:
                raise RuntimeError(f"aggregation with missing items: {dataset} {missing[:5]}")
            scores.append(sum(cached[i]["score"] for i in range(stop)) / stop)
        if self.is_multitask:
            return float(harmonic_mean(scores, self.protocol))
        return float(scores[0])

    # ---------- 初始化与终选 ----------
    def evaluate_init(self, init_alphas):
        scores, costs = [], []
        for alpha in init_alphas:
            result = self.evaluate(alpha, None)
            scores.append(result["score"])
            costs.append(result["new_samples"])
        return {"scores": scores, "costs": costs}

    def evaluate_test(self, alpha, selected=None):
        """One full or prevalidated indexed test pass, outside the search budget."""
        started = time.monotonic()
        adapter = self._materialize(alpha)
        from evaluate import evaluate_test
        result_dir = self.results / f"test_{self._adapter_seq:06d}"
        indices = [entry["index"] for entry in selected] if selected is not None else None
        expected = len(indices) if indices is not None else sum(
            load_split_size(dataset, "test") for dataset in self.datasets)
        try:
            # multitask 由 evaluate.evaluate_test 经 multitask_domain_dataset_dict 展开；
            # 单数据集任务必须传数据集名（kcross 的文件是 knowledge_crosswords.json，
            # 任务名不是数据集名——曾因直接传 self.task 在 kcross test 上 FileNotFoundError）。
            test_dataset = self.task if self.is_multitask else self.datasets[0]
            value = evaluate_test(str(adapter), self.eval_type, test_dataset, 0,
                                  base_model=self.snapshot["base_model"]["path"],
                                  protocol=self.protocol, result_dir=str(result_dir),
                                  **({"subset_indices": indices} if indices is not None else {}))
            archive = json.loads((result_dir / "evaluation.json").read_text())
            if (archive.get("split") != "test" or
                    archive.get("scoring_version") != self.protocol or
                    archive.get("score") != value):
                raise RuntimeError("test evaluation archive mismatch")
            count = 0
            for dataset in self.datasets:
                child = result_dir / dataset if self.is_multitask else result_dir
                result = json.loads((child / "evaluation.json").read_text())
                selected_indices = indices if indices is not None else list(
                    range(load_split_size(dataset, "test")))
                if (result.get("dataset") != dataset or result.get("split") != "test" or
                        result.get("scoring_version") != self.protocol or
                        result.get("n_samples") != len(selected_indices) or
                        len(result.get("records", [])) != len(selected_indices) or
                        [r["index"] for r in result["records"]] != selected_indices or
                        (selected is not None and
                         (result.get("subset_indices") != selected_indices or
                          [r.get("question_text_sha256") for r in result["records"]] !=
                          [entry["question_sha256"] for entry in selected]))):
                    raise RuntimeError(f"test evaluation archive mismatch: {dataset}")
                count += len(selected_indices)
            self.test_samples += count
            self.physical_samples += count
            return float(value)
        except BaseException:
            self.uncertain_samples_upper_bound += expected
            raise
        finally:
            self.generation_seconds += time.monotonic() - started


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="restore a complete paired checkpoint")
    args = parser.parse_args(argv)
    job_path = args.job.resolve()
    job = json.loads(job_path.read_text())
    run_dir = job_path.parent
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("worker requires parent-fixed CUDA_VISIBLE_DEVICES")
    os.environ.update(OFFLINE_ENV)
    status_path = run_dir / "worker_status.json"
    status = {"status": "starting", "started_at": utc_now()}
    oracle = None
    save_json_atomic(status_path, status)
    try:
        snapshot = load_run_snapshot(job)
        from search_core import (RACER_VERSION, DEFAULT_CONFIG, build_search,
                                 BudgetExhausted)
        config = dict(DEFAULT_CONFIG)
        config.update(snapshot.get("frozen_config_overrides") or {})
        task = snapshot["task"]
        method = snapshot["method"]
        ablation = snapshot.get("ablation", "full")
        seed = snapshot["seed"]
        budget = snapshot["budget_samples"]
        oracle = RealOracle(snapshot, run_dir / "oracle", resume=args.resume)
        status.update(status="loading", method=method, ablation=ablation,
                      task=task, seed=seed)
        save_json_atomic(status_path, status)
        rng = np.random.default_rng(seed)
        n = len(snapshot["experts"]["names"])
        # 共享初始化：与 alpha_space.sample_initial_alphas 同公式的无 torch 版
        rng_init = np.random.default_rng(seed)
        inits = [np.eye(n)[i] for i in range(min(n, config["init_size"]))]
        while len(inits) < config["init_size"]:
            if rng_init.random() < 0.7:
                a = rng_init.dirichlet(np.ones(n) * 0.6)
            else:
                a = rng_init.dirichlet(np.ones(n) * 0.6) * rng_init.uniform(0.5, 1.5)
            inits.append(a)
        init_cost = sum(oracle.estimate(alpha, None) for alpha in inits)
        if init_cost > budget:
            raise BudgetExhausted(f"initialization requires {init_cost} samples, budget is {budget}")
        checkpoint_file = run_dir / "checkpoint.json"
        if args.resume:
            paired = json.loads(checkpoint_file.read_text(encoding="utf-8"))
            if paired.get("identity_sha256") != snapshot["identity_sha256"] or \
                    paired.get("job_run") != job["run"] or \
                    paired.get("initial_alphas") != [a.tolist() for a in inits]:
                raise ValueError("paired checkpoint identity mismatch")
            oracle.restore_cache(paired["oracle"])
            init_eval = paired["init_eval"]
        else:
            if checkpoint_file.exists():
                raise FileExistsError(checkpoint_file)
            init_eval = oracle.evaluate_init(inits)
        search = build_search(method, ablation, n, config, oracle, rng,
                              inits, init_eval["scores"], init_eval["costs"],
                              budget, seed=seed)
        if args.resume:
            search.restore_state(paired["search"])
        def checkpoint(current):
            save_json_atomic(checkpoint_file, {
                "identity_sha256": snapshot["identity_sha256"],
                "job_run": job["run"],
                "initial_alphas": [a.tolist() for a in inits],
                "init_eval": init_eval, "search": current.snapshot_state(),
                "oracle": oracle.snapshot_cache()})
        search.on_checkpoint = checkpoint
        if not args.resume:
            checkpoint(search)
        status["status"] = "searching"
        save_json_atomic(status_path, status)
        stop_reason = search.run()
        pick, alpha, best_dev = search.final_selection()
        test_score = None
        kcross_filtered_test_score = None
        if snapshot.get("run_test", False):
            selected = kcross_test_selection() if task == "kcross" else None
            test_score = oracle.evaluate_test(alpha)
            if selected is not None:
                kcross_filtered_test_score = oracle.evaluate_test(alpha, selected=selected)
        marks = [budget * f for f in (0.3, 0.6, 1.0)]
        summary = {
            "status": "succeeded",
            "method": method, "ablation": ablation, "task": task, "seed": seed,
            "stop_reason": stop_reason,
            "budget_samples": budget,
            "budget_used": search.ledger.budget_used,
            "best_dev": best_dev,
            "best_alpha": [float(v) for v in alpha],
            "best_candidate_id": pick,
            "test_score": test_score,
            "kcross_filtered_test_score": kcross_filtered_test_score,
            "n_candidates": len(search.ledger.candidates),
            "anytime": [{"budget": mark, "best": score} for mark, score in
                        zip(marks, search.anytime_at(marks))],
            "physical_samples": oracle.physical_samples,
            "test_samples": oracle.test_samples,
            "uncertain_samples_upper_bound": oracle.uncertain_samples_upper_bound,
            "generation_seconds": oracle.generation_seconds,
            "racer_version": RACER_VERSION,
            "finished_at": utc_now(),
        }
        save_json_atomic(run_dir / "summary.json", summary)
        save_json_atomic(run_dir / "ledger.json", search._snapshot_base())
        save_json_atomic(status_path, {**status, "status": "succeeded"})
        return 0
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        try:
            journal_lower, journal_upper = (journal_sample_bounds(oracle.results)
                                            if oracle is not None else (0, 0))
        except (OSError, ValueError):
            journal_lower = journal_upper = None
        counters = ({"physical_samples": oracle.physical_samples,
                     "test_samples": oracle.test_samples,
                     "uncertain_samples_upper_bound": oracle.uncertain_samples_upper_bound,
                     "journal_completed_lower_bound": journal_lower,
                     "journal_attempted_upper_bound": journal_upper}
                    if oracle is not None else {})
        save_json_atomic(status_path, {**status, **counters, "status": "failed", "error": error})
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
