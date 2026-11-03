#!/usr/bin/env python3
"""run_racer_gate.py — 阶段二 dev-only 单卡微型门禁（预注册、不碰 test）。

内容（批准计划第 6 节）：
  G1 9 主任务 × 2 个预指定候选（one-hot e0 与均匀 mix），dev 全量一次，
     验证真实物化/加载/MC与数值/tiny-dev 与领域/rung 切片。
  G2 4 开发任务 × 4 主方法 × 1 seed，每 run 预算 = 24 × D_t 全量等效样本，
     只跑搜索、不跑 test；验证预算账本/checkpoint/终选与逐题缓存。

只通过 tools/run_racer_campaign.py 的同款 worker（search_frozen.py --job）执行；
本脚本仅构造 job、检查 GPU 空闲（fail-closed）、逐个串行启动、汇总状态。
不访问 test split，不据此调分/改协议；输出吞吐供冻结 timeout。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SPEC = ROOT / "experiments/racer_frozen_v1.json"
OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "WANDB_DISABLED": "true", "WANDB_MODE": "disabled",
               "TOKENIZERS_PARALLELISM": "false"}
GATE_CONFIG_OVERRIDES = {"init_size": 2}  # 门禁专用：初始化 2 点（one-hot + mix）
G2_BUDGET_PER_DEV_ITEM = 24
G2_METHODS = ("racer_frozen_v1", "cma_full", "pso_box", "random_box")
G2_TASKS = ("nlgraph", "gsm8k", "hellaswag", "medical")


def utc_now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def query_idle_gpu(requested, telemetry_path):
    """与 campaign driver 相同的 fail-closed 空闲判定。"""
    raw = {"checked_at": utc_now(), "requested": requested, "queries": {}}
    try:
        for fields, kind in (("index,uuid,memory.used,utilization.gpu", "gpu"),
                             ("gpu_uuid,pid", "compute-apps")):
            command = ["nvidia-smi", f"--query-{kind}={fields}",
                       "--format=csv,noheader,nounits"]
            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=10, check=False)
            raw["queries"][kind] = {"stdout": result.stdout,
                                    "returncode": result.returncode}
            if result.returncode:
                raise RuntimeError(f"nvidia-smi {kind} rc={result.returncode}")
        devices = list(csv.reader(raw["queries"]["gpu"]["stdout"].splitlines(),
                                  skipinitialspace=True))
        processes = [row for row in csv.reader(
            raw["queries"]["compute-apps"]["stdout"].splitlines(),
            skipinitialspace=True) if row]
        busy = {row[0].strip() for row in processes}
        for row in devices:
            index, uuid, memory, utilization = [v.strip() for v in row]
            if int(index) == requested:
                if uuid not in busy and int(memory) <= 64 and int(utilization) == 0:
                    return {"index": requested, "uuid": uuid}
                raise RuntimeError(f"GPU {requested} busy: mem={memory} util={utilization}")
        raise RuntimeError(f"GPU {requested} not found")
    finally:
        save_json(telemetry_path, raw)


def terminate_owned(child):
    import signal
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


def run_worker(job_path, gpu, log_path, timeout):
    env = dict(os.environ, **OFFLINE_ENV, CUDA_VISIBLE_DEVICES=gpu["uuid"])
    command = [sys.executable, "-I", str(ROOT / "search_frozen.py"),
               "--job", str(job_path)]
    started = time.monotonic()
    with Path(log_path).open("xb") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                 env=env, cwd=str(ROOT), start_new_session=True)
        try:
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            terminate_owned(child)
            return {"status": "timeout", "returncode": child.returncode,
                    "elapsed_seconds": time.monotonic() - started}
        terminate_owned(child)
        return {"status": "exited", "returncode": code,
                "elapsed_seconds": time.monotonic() - started}


def build_gate_rows(spec, budgets):
    """G1 18 行 + G2 16 行；G1 传 budget=2×D_t 但 init 2 点后立即预算耗尽。"""
    rows = []
    for task in spec["main_tasks"]:
        rows.append({"id": f"G1_{task}_probe_s42", "group": "G1", "task": task,
                     "method": "random_box", "ablation": "full", "seed": 42,
                     "run_test": False,
                     "budget_samples": 2 * budgets[task],
                     "gate_config_overrides": GATE_CONFIG_OVERRIDES})
    for task in G2_TASKS:
        for method in G2_METHODS:
            rows.append({"id": f"G2_{task}_{method}_s42", "group": "G2",
                         "task": task, "method": method, "ablation": "full",
                         "seed": 42, "run_test": False,
                         "budget_samples": G2_BUDGET_PER_DEV_ITEM * budgets[task],
                         "gate_config_overrides": {}})
    seen = set()
    for row in rows:
        if row["id"] in seen:
            raise ValueError(f"duplicate gate id {row['id']}")
        seen.add(row["id"])
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, help="门禁目录（首次必须全新；已有账本则断点续跑）")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--run-timeout", type=float, default=2 * 3600)
    args = parser.parse_args(argv)
    spec = read_json(SPEC)

    from search_frozen import TASK_REGISTRY
    budgets = {}
    for task in set(spec["main_tasks"]) | set(G2_TASKS):
        total = 0
        for dataset in TASK_REGISTRY[task][1]:
            with (ROOT / "data/eval" / f"{dataset}.json").open() as handle:
                total += len(json.load(handle)["dev"])
        budgets[task] = total

    rows = build_gate_rows(spec, budgets)
    print(f"gate matrix: {len(rows)} runs (G1={len(spec['main_tasks'])}, "
          f"G2={len(G2_TASKS) * len(G2_METHODS)})")
    if not args.execute:
        for row in rows:
            print(f"  {row['id']}  budget={row['budget_samples']}")
        print("(dry run only; --execute to run)")
        return 0
    if args.output is None:
        raise SystemExit("--execute requires an --output directory")
    from tools.run_racer_campaign import preflight, wait_release_window

    ledger = None
    if args.output.exists():
        # 断点续跑：已有账本则只补未成功的 run；缺账本视为损坏目录。
        ledger_path = args.output / "gate_status.json"
        if not ledger_path.is_file():
            raise SystemExit(f"existing output lacks gate ledger: {args.output}")
        ledger = read_json(ledger_path)
        done_ids = {r["id"] for r in ledger["runs"] if r["status"] == "succeeded"}
        rows = [row for row in rows if row["id"] not in done_ids]
        ledger["status"] = "running"
        ledger["resumed_at"] = utc_now()
        save_json(ledger_path, ledger)
        print(f"resume: {len(rows)} runs remaining; snapshot reuse from ledger")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        base = os.environ.get("RACER_BASE_MODEL")
        if not base or not Path(base).is_dir():
            raise SystemExit("RACER_BASE_MODEL must point at the local Qwen snapshot")
        campaign = preflight()
        save_json(args.output / "snapshot.json", campaign)
        ledger = {"status": "running", "created_at": utc_now(),
                  "gpu": args.gpu, "runs": []}
        save_json(args.output / "gate_status.json", ledger)
    telemetry = args.output / "telemetry"
    telemetry.mkdir(exist_ok=True)

    failures = sum(1 for r in ledger["runs"] if r["status"] != "succeeded")
    try:
        for seq, row in enumerate(rows):
            run_dir = args.output / row["id"]
            fresh = not run_dir.exists()
            if fresh:
                run_dir.mkdir()
                job = {"snapshot": str(args.output / "snapshot.json"),
                       "run": {k: v for k, v in row.items()
                               if k not in ("gate_config_overrides", "group")}}
                save_json(run_dir / "job.json", job)
            else:
                job = read_json(run_dir / "job.json")
            overrides = row["gate_config_overrides"]
            if overrides and fresh:
                # 将门禁覆盖并入 campaign spec 的运行副本（不改冻结 spec 文件），
                # 并重算身份哈希：worker 会校验 identity_sha256 覆盖全部字段。
                snapshot = read_json(args.output / "snapshot.json")
                snapshot["spec"] = {**snapshot["spec"],
                                    "frozen_config_overrides": overrides}
                payload = {k: v for k, v in snapshot.items()
                           if k != "identity_sha256"}
                import hashlib
                snapshot["identity_sha256"] = hashlib.sha256(json.dumps(
                    payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                save_json(args.output / f"snapshot_{row['id']}.json", snapshot)
                job["snapshot"] = str(args.output / f"snapshot_{row['id']}.json")
                save_json(run_dir / "job.json", job)
            # 自己 worker 退出后的释放窗口：有界稳定等待，不降阈值（串行复用同一卡）。
            telemetry_path = telemetry / f"gpu_check_{row['id']}.json"
            gpu = wait_release_window(args.gpu, telemetry_path=telemetry_path)
            attempt = 1
            log_path = run_dir / "worker.log"
            while log_path.exists():
                attempt += 1
                log_path = run_dir / f"worker_retry{attempt}.log"
            result = run_worker(run_dir / "job.json", gpu, log_path, args.run_timeout)
            summary_path = run_dir / "summary.json"
            status = result["status"]
            if status == "exited":
                if result["returncode"] == 0 and summary_path.is_file():
                    summary = read_json(summary_path)
                    if summary.get("status") == "succeeded":
                        status = "succeeded"
                    else:
                        status = "failed"
                        failures += 1
                else:
                    status = "failed"
                    failures += 1
            else:
                failures += 1
            # 断点续跑时更新已有行而不是重复 append。
            existing = next((r for r in ledger["runs"] if r["id"] == row["id"]), None)
            record = {**row, "status": status,
                      "elapsed_seconds": result["elapsed_seconds"],
                      "gpu_uuid": gpu["uuid"]}
            if existing is not None:
                existing.update(record)
            else:
                ledger["runs"].append(record)
            ledger["failed"] = failures
            save_json(args.output / "gate_status.json", ledger)
            print(f"  {row['id']}: {status} ({result['elapsed_seconds']:.1f}s)")
        ledger["status"] = "succeeded" if failures == 0 else "failed"
    except BaseException as exc:
        ledger["status"] = "blocked"
        ledger["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save_json(args.output / "gate_status.json", ledger)
    print(json.dumps({"status": ledger["status"], "failed": failures}))
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
