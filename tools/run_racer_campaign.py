#!/usr/bin/env python3
"""run_racer_campaign.py — racer_frozen_v1 正式矩阵排程器（父驱动）。

职责（parent，stdlib only）：
  - 预检：冻结配置/源码/数据/专家/基模身份快照，拒绝 dirty 身份。
  - GPU：fail-closed 空闲检查（显存/利用率/无 compute 进程 + 原始遥测落盘）；
    自己 worker 正常退出后的释放窗口做有上限稳定等待，不降阈值、不自动换卡。
  - 排程：每次 1-2 张空闲卡，每卡一个独立 run 子进程（start_new_session），
    超时只 kill 自有进程组。
  - 账本：每 run 的状态/耗时/样本数/终态逐条落盘；blocked/failed 公开记录。

默认 dry-run：只打印矩阵与预检结果。--execute 才真跑；--job 可单发一个 run。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "experiments/racer_frozen_v1.json"
IDENTITY_DEPENDENCIES = (
    "tools/run_racer_campaign.py", "tools/audit_racer_data.py",
    "tests/test_racer_campaign.py", "experiments/racer_frozen_v1.json",
    "experiments/racer_data_audit_v1.json",
)
OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "WANDB_DISABLED": "true", "WANDB_MODE": "disabled",
               "TOKENIZERS_PARALLELISM": "false"}


def task_datasets(task):
    sys.path.insert(0, str(ROOT))
    from search_frozen import TASK_REGISTRY
    return TASK_REGISTRY[task][1]


def dev_budget(task):
    """D_t = 组件 dev 样本数之和；从数据文件实算，不用手抄表。"""
    total = 0
    for dataset in task_datasets(task):
        with (ROOT / "data/eval" / f"{dataset}.json").open() as handle:
            total += len(json.load(handle)["dev"])
    return total


class CampaignError(ValueError):
    pass


class Blocked(CampaignError):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def file_identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def tree_identity(directory):
    """Hash each effective file, following symlink targets within a finite tree."""
    directory = Path(directory)
    if not directory.is_dir():
        raise CampaignError(f"identity directory missing: {directory}")
    identities = {}

    def visit(current, ancestry):
        resolved = current.resolve(strict=True)
        if current.is_dir():
            if resolved in ancestry:
                raise CampaignError(f"symlink directory cycle: {current}")
            for child in sorted(current.iterdir(), key=lambda entry: entry.name):
                visit(child, ancestry | {resolved})
        elif current.is_file():
            relative = current.relative_to(directory).as_posix()
            identities[relative] = file_identity(current)
        else:
            raise CampaignError(f"invalid identity tree entry: {current}")

    try:
        visit(directory, set())
    except (OSError, RuntimeError) as exc:
        raise CampaignError(f"cannot inspect identity tree {directory}: {exc}") from exc
    if not identities:
        raise CampaignError(f"empty identity directory: {directory}")
    return identities


def build_matrix(spec):
    """返回预注册 run 列表（有序、去重、可校验）。"""
    seeds = spec["seeds"]
    rows = []

    def add(task, method, ablation, seed, group, run_test=True):
        rows.append({"id": f"{group}_{task}_{method}" + (f"_{ablation}" if ablation != "full" else "")
                                 + f"_s{seed}",
                     "group": group, "task": task, "method": method,
                     "ablation": ablation, "seed": seed, "run_test": run_test,
                     "budget_samples": 110 * dev_budget(task)})

    # A. 主比较 108
    for task in spec["main_tasks"]:
        for seed in seeds:
            add(task, "racer_frozen_v1", "full", seed, "A_main")
            add(task, "cma_full", "full", seed, "A_main")
            add(task, "pso_box", "full", seed, "A_main")
            add(task, "random_box", "full", seed, "A_main")
    # B. 消融 108（完整 RACER 行复用 A_main）
    for task in spec["ablation_tasks"]:
        for seed in seeds:
            for ab in spec["ablations"]:
                if ab == "cma_SH":
                    method = "cma_sh"
                    ablation = "full"
                else:
                    method, ablation = "racer_frozen_v1", ab
                add(task, method, ablation, seed, "B_ablation")
    # C. 尺度解混 12（实现后补充；此处占位登记为待实现组）
    for task in spec["scale_tasks"]:
        for seed in seeds:
            add(task, "pso_sum1", "full", seed, "C_scale", run_test=True)
            add(task, "pso_sum1_scalegrid", "full", seed, "C_scale", run_test=True)
    # D. kcross 12
    for seed in seeds:
        for method in ("racer_frozen_v1", "cma_full", "pso_box", "random_box"):
            add("kcross", method, "full", seed, "D_kcross")
    seen = set()
    for row in rows:
        if row["id"] in seen:
            raise CampaignError(f"duplicate run id {row['id']}")
        seen.add(row["id"])
    return rows


def preflight():
    spec = read_json(SPEC)
    sources = {}
    for name in sorted(set(spec["source_files"]) | set(IDENTITY_DEPENDENCIES)):
        path = ROOT / name
        if not path.is_file():
            raise CampaignError(f"missing source {name}")
        sources[name] = file_identity(path)
    data_files = {}
    for task in set(spec["main_tasks"] + spec["ablation_tasks"] +
                    spec["scale_tasks"] + ["kcross"]):
        for dataset in task_datasets(task):
            p = ROOT / "data/eval" / f"{dataset}.json"
            data_files[dataset] = file_identity(p)
    experts_dir = ROOT / "initial_experts"
    experts = sorted(p.name for p in experts_dir.iterdir()
                     if p.is_dir() and (p / "adapter_model.safetensors").is_file())
    if len(experts) != 10:
        raise CampaignError(f"expected 10 experts, found {len(experts)}")
    expert_files = {}
    for name in experts:
        for relative, identity in tree_identity(experts_dir / name).items():
            expert_files[f"{name}/{relative}"] = identity
    base = os.environ.get("RACER_BASE_MODEL")
    if not base:
        raise CampaignError("RACER_BASE_MODEL must point at the local Qwen snapshot")
    base_path = Path(base)
    snapshot = {"spec": spec, "sources": sources, "data": data_files,
                "experts": {"path": str(experts_dir), "names": experts,
                            "files": expert_files},
                "base_model": {"path": str(base_path), "files": tree_identity(base_path)},
                "created_at": utc_now()}
    snapshot["identity_sha256"] = hashlib.sha256(json.dumps(
        snapshot, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return snapshot


def query_idle_gpu(requested=None, max_memory_mib=64, telemetry_path=None,
                   excluded=()):
    """Select one idle card outside our running set; archive telemetry even on failure."""
    raw = {"checked_at": utc_now(), "requested": requested,
           "excluded": sorted(excluded), "queries": {}}
    try:
        for fields, kind in (("index,uuid,memory.used,utilization.gpu", "gpu"),
                             ("gpu_uuid,pid", "compute-apps")):
            command = ["nvidia-smi", f"--query-{kind}={fields}",
                       "--format=csv,noheader,nounits"]
            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=10, check=False)
            raw["queries"][kind] = {"command": command, "stdout": result.stdout,
                                     "stderr": result.stderr, "returncode": result.returncode}
            if result.returncode:
                raise Blocked(f"nvidia-smi {kind} returned {result.returncode}")
        devices = list(csv.reader(raw["queries"]["gpu"]["stdout"].splitlines(),
                                  skipinitialspace=True))
        processes = list(csv.reader(raw["queries"]["compute-apps"]["stdout"].splitlines(),
                                    skipinitialspace=True))
        if not devices or any(len(row) != 4 for row in devices) or \
                any(len(row) != 2 or not row[0].strip() or not row[1].strip()
                    for row in processes):
            raise Blocked("malformed nvidia-smi telemetry")
        busy = {row[0].strip() for row in processes}
        available = []
        seen_indices, seen_uuids = set(), set()
        for row in devices:
            try:
                index, uuid, memory, utilization = [value.strip() for value in row]
                index, memory, utilization = int(index), int(memory), int(utilization)
            except ValueError as exc:
                raise Blocked("malformed nvidia-smi GPU values") from exc
            if index < 0 or memory < 0 or not 0 <= utilization <= 100 or not uuid or \
                    index in seen_indices or uuid in seen_uuids:
                raise Blocked("invalid or duplicate nvidia-smi GPU")
            seen_indices.add(index)
            seen_uuids.add(uuid)
            if (requested is None or index == requested) and index not in excluded and \
                    uuid not in busy and memory <= max_memory_mib and utilization == 0:
                available.append({"index": index, "uuid": uuid})
        if not available:
            raise Blocked(f"no idle GPU (requested={requested}, excluded={sorted(excluded)})")
        return available[0]
    except (OSError, subprocess.SubprocessError) as exc:
        raw["error"] = f"cannot query nvidia-smi: {exc}"
        raise Blocked(raw["error"]) from exc
    except Blocked as exc:
        raw["error"] = str(exc)
        raise
    finally:
        if telemetry_path is not None:
            save_json(telemetry_path, raw)


def wait_release_window(gpu_index, max_wait=10.0, telemetry_path=None):
    """Bounded wait only after our own worker exits, without relaxing thresholds."""
    deadline = time.monotonic() + max_wait
    attempt = 0
    while True:
        attempt += 1
        path = None if telemetry_path is None else Path(telemetry_path).with_name(
            f"{Path(telemetry_path).stem}_{attempt}.json")
        try:
            return query_idle_gpu(gpu_index, telemetry_path=path)
        except Blocked as exc:
            if time.monotonic() >= deadline:
                raise Blocked(f"GPU {gpu_index} not idle after {max_wait}s release window: "
                              f"{exc}") from exc
            time.sleep(min(1.0, max(0, deadline - time.monotonic())))


def journal_attempt_bounds(run_dir):
    """Physical sample bounds from durable batch journals; never resume partial runs."""
    completed = attempted = uncertain = 0
    journals = 0
    malformed = []
    for path in sorted((Path(run_dir) / "oracle/results").glob("**/attempt.jsonl")):
        journals += 1
        started, done = set(), set()
        try:
            with path.open(encoding="utf-8") as handle:
                for number, line in enumerate(handle, 1):
                    try:
                        entry = json.loads(line)
                        indices = entry["indices"]
                        if (entry["state"] not in ("started", "completed") or
                                not isinstance(indices, list) or not indices or
                                any(type(i) is not int or i < 0 for i in indices)):
                            raise ValueError("invalid journal event")
                        key = tuple(indices)
                        if entry["state"] == "started":
                            started.add(key)
                        elif key in started and entry.get("generated_count") == len(indices):
                            done.add(key)
                        else:
                            raise ValueError("unmatched or incomplete completion")
                    except (ValueError, KeyError, TypeError):
                        malformed.append(f"{path}:{number}")
        except OSError:
            malformed.append(str(path))
        completed += sum(len(batch) for batch in done)
        attempted += sum(len(batch) for batch in started)
        uncertain += sum(len(batch) for batch in started - done)
    return {"completed_lower_bound": completed,
            "attempted_upper_bound": attempted,
            "uncertain_inflight": uncertain, "journals": journals,
            "malformed_entries": malformed,
            "note": "Journalled generation attempts only; unjournalled in-flight work is unknowable."}


def terminate_owned_child(child):
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


def run_child(command, timeout, log_path, env, cwd):
    started = time.monotonic()
    child = None
    try:
        with Path(log_path).open("xb") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     env=env, cwd=cwd, start_new_session=True)
            try:
                code = child.wait(timeout=max(0.001, timeout - (time.monotonic() - started)))
            except subprocess.TimeoutExpired:
                terminate_owned_child(child)
                return {"status": "timeout", "returncode": child.returncode,
                        "elapsed_seconds": time.monotonic() - started}
            terminate_owned_child(child)
            return {"status": "exited", "returncode": code,
                    "elapsed_seconds": time.monotonic() - started}
    except BaseException:
        if child is not None:
            terminate_owned_child(child)
        raise


def prepare_run(run_row, snapshot, campaign_dir, run_test=None):
    run_dir = campaign_dir / run_row["id"]
    if run_dir.exists():
        raise CampaignError(f"run dir exists: {run_dir}")
    run_dir.mkdir(parents=True)
    effective_run = dict(run_row, run_test=run_row["run_test"] if run_test is None
                         else run_test)
    snapshot_path = campaign_dir / "snapshot.json"
    if not snapshot_path.exists():
        save_json(snapshot_path, snapshot)
    save_json(run_dir / "job.json", {"snapshot": str(snapshot_path),
                                     "run": effective_run})
    return run_dir


def execute_campaign(rows, campaign_dir, spec, snapshot, gpus=None,
                     run_timeout=4 * 3600, max_parallel=2, only_ids=None,
                     run_test_override=None, resume=False):
    campaign_dir = Path(campaign_dir).resolve()
    if max_parallel not in (1, 2) or run_timeout <= 0:
        raise CampaignError("max_parallel must be 1 or 2 and run_timeout positive")
    if gpus is not None and (not gpus or len(set(gpus)) != len(gpus) or
                              len(gpus) > 2 or any(g < 0 for g in gpus)):
        raise CampaignError("GPU allowlist must contain one or two distinct nonnegative indices")
    ids = {row["id"] for row in rows}
    if only_ids is not None and (not only_ids or not only_ids <= ids):
        raise CampaignError(f"unknown or empty run selection: {sorted(only_ids - ids) if only_ids else []}")
    selected = [dict(row, status="pending") for row in rows
                if only_ids is None or row["id"] in only_ids]
    if resume:
        # 断点续跑：只接收未成功 run；行追加进既有账本，不重写 snapshot。
        if not campaign_dir.is_dir() or not (campaign_dir / "status.json").is_file():
            raise CampaignError("resume requires an existing campaign ledger")
        ledger = read_json(campaign_dir / "status.json")
        known = {r["id"] for r in ledger["rows"]}
        if known & ids:
            raise CampaignError("resume rows overlap the existing ledger")
        ledger["rows"].extend(selected)
        ledger["status"] = "running"
        ledger["resumed_at"] = utc_now()
    else:
        campaign_dir.mkdir(parents=True, exist_ok=False)
        ledger = {"status": "running", "created_at": utc_now(), "rows": selected}
        save_json(campaign_dir / "snapshot.json", snapshot)
    save_json(campaign_dir / "status.json", ledger)
    telemetry_dir = campaign_dir / "telemetry"
    telemetry_dir.mkdir(exist_ok=resume)
    pending = list(selected)
    running = {}  # index -> (row, child, started)
    check_seq = 0
    pinned = {}  # acquired physical card identities; do not switch cards later
    release_wait = set()

    def telemetry_path():
        nonlocal check_seq
        check_seq += 1
        return telemetry_dir / f"gpu_check_{check_seq:06d}.json"

    def next_gpu():
        allowed = gpus if gpus is not None else list(pinned)
        # Reuse acquired cards first. Only acquire a new card when another
        # parallel slot remains; never replace a previously acquired identity.
        if gpus is None and len(pinned) < max_parallel:
            allowed = list(pinned) + [None]
        errors = []
        for index in allowed:
            if index is not None and index in running:
                continue
            try:
                if index in release_wait:
                    gpu = wait_release_window(index, telemetry_path=telemetry_path())
                    release_wait.remove(index)
                else:
                    gpu = query_idle_gpu(requested=index,
                                         excluded=set(running) | (set(pinned) if index is None else set()),
                                         telemetry_path=telemetry_path())
                if index is not None and index in pinned and pinned[index] != gpu["uuid"]:
                    raise Blocked(f"GPU {index} UUID changed")
                pinned.setdefault(gpu["index"], gpu["uuid"])
                return gpu, None
            except Blocked as exc:
                errors.append(str(exc))
        return None, "; ".join(errors) or "no available campaign GPU"

    def launch(row, gpu):
        # Recheck the selected card immediately before spawning; another owner
        # may have claimed it between scheduling and launch.
        checked = query_idle_gpu(requested=gpu["index"],
                                 telemetry_path=telemetry_path())
        if checked["uuid"] != gpu["uuid"]:
            raise Blocked("selected GPU identity changed before launch")
        run_dir = prepare_run(row, snapshot, campaign_dir, run_test_override)
        env = dict(os.environ, **OFFLINE_ENV, CUDA_VISIBLE_DEVICES=gpu["uuid"])
        command = [sys.executable, "-I", str(ROOT / "search_frozen.py"),
                   "--job", str(run_dir / "job.json")]
        row.update(run_dir=str(run_dir), gpu=gpu["index"], gpu_uuid=gpu["uuid"])
        with (run_dir / "worker.log").open("xb") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     env=env, cwd=str(ROOT), start_new_session=True)
        row.update(status="running", started_at=utc_now(), pid=child.pid)
        save_json(campaign_dir / "status.json", ledger)
        return child

    try:
        while pending or running:
            for index, (row, child, started) in list(running.items()):
                code = child.poll()
                timed_out = code is None and time.monotonic() - started > run_timeout
                if code is None and not timed_out:
                    continue
                if timed_out:
                    terminate_owned_child(child)
                    row.update(status="timeout", error="run deadline exceeded")
                else:
                    # Even after a successful exit, kill surviving descendants of
                    # our own process group before assigning this GPU again.
                    terminate_owned_child(child)
                    summary_path = Path(row["run_dir"]) / "summary.json"
                    if code == 0 and summary_path.is_file():
                        try:
                            summary = read_json(summary_path)
                            if summary.get("status") != "succeeded":
                                raise ValueError("worker summary is not successful")
                            row.update(status="succeeded", summary=summary)
                        except (OSError, ValueError) as exc:
                            row.update(status="failed", error=str(exc), returncode=code)
                    else:
                        row.update(status="failed", returncode=code)
                row["elapsed_seconds"] = time.monotonic() - started
                row["attempt_bounds"] = journal_attempt_bounds(row["run_dir"])
                row["finished_at"] = utc_now()
                del running[index]
                release_wait.add(index)
                save_json(campaign_dir / "status.json", ledger)
            while pending and len(running) < max_parallel:
                gpu, reason = next_gpu()
                if gpu is None:
                    if not running:
                        for row in pending:
                            row.update(status="blocked", error=reason, finished_at=utc_now())
                        pending.clear()
                        save_json(campaign_dir / "status.json", ledger)
                    break
                row = pending[0]
                try:
                    child = launch(row, gpu)
                except Blocked as exc:
                    if not running:
                        for blocked_row in pending:
                            blocked_row.update(status="blocked", error=str(exc),
                                               finished_at=utc_now())
                        pending.clear()
                        save_json(campaign_dir / "status.json", ledger)
                    break
                pending.pop(0)
                running[gpu["index"]] = (row, child, time.monotonic())
            if running:
                time.sleep(5.0)
        ledger["status"] = ("succeeded" if all(r["status"] == "succeeded" for r in selected)
                            else "blocked" if any(r["status"] == "blocked" for r in selected)
                            else "completed")
    except BaseException as exc:
        ledger["status"] = "failed"
        ledger["error"] = f"{type(exc).__name__}: {exc}"
        for row in selected:
            if row["status"] == "pending" and "run_dir" in row:
                row.update(status="failed", error=ledger["error"], finished_at=utc_now())
        for row, child, started in running.values():
            terminate_owned_child(child)
            row.update(status="cancelled", elapsed_seconds=time.monotonic() - started,
                       attempt_bounds=journal_attempt_bounds(row["run_dir"]),
                       finished_at=utc_now())
        for row in pending:
            if row["status"] == "pending":
                row.update(status="blocked", error="campaign interrupted", finished_at=utc_now())
        raise
    finally:
        save_json(campaign_dir / "status.json", ledger)
    return ledger


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, help="campaign 目录（必须全新）")
    parser.add_argument("--only", help="逗号分隔 run id 子集")
    parser.add_argument("--gpus", help="one or two physical GPU indices, comma-separated")
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--run-timeout", type=float, default=6 * 3600)
    parser.add_argument("--resume", action="store_true",
                        help="断点续跑已有 campaign：身份一致时只重发未成功 run")
    args = parser.parse_args(argv)
    spec = read_json(SPEC)
    rows = build_matrix(spec)
    print(f"matrix: {len(rows)} runs")
    if not args.execute:
        for row in rows[:40]:
            print(f"  {row['id']}  budget={row['budget_samples']}")
        print(f"  ... ({len(rows)} total; dry run only)")
        return 0
    if args.output is None:
        raise CampaignError("--execute requires an --output directory")
    only = set(args.only.split(",")) if args.only is not None else None
    ids = {row["id"] for row in rows}
    if only is not None and (not only or not only <= ids):
        raise CampaignError(f"unknown or empty --only selection: {sorted(only - ids) if only else []}")
    if getattr(args, "resume", False):
        # 断点续跑：身份必须与已存 snapshot 一致（源码/数据/专家/基模无漂移），
        # 只重发未成功 run；已成功的不重跑、不重复计样本。
        if not args.output.is_dir() or not (args.output / "status.json").is_file():
            raise CampaignError("--resume requires an existing campaign with a status ledger")
        saved = read_json(args.output / "snapshot.json")
        if preflight()["identity_sha256"] != saved.get("identity_sha256"):
            raise CampaignError("identity drift: sources/data/experts/base changed since campaign start")
        existing = read_json(args.output / "status.json")
        by_id = {r["id"]: r for r in existing["rows"]}
        if set(by_id) - ids:
            raise CampaignError("ledger contains runs outside the frozen matrix")
        skip = {r["id"] for r in existing["rows"] if r["status"] == "succeeded"}
        rows = [row for row in rows if row["id"] not in skip]
        print(f"resume: {len(skip)} succeeded skipped, {len(rows)} to (re)run; "
              "failed run dirs must be archived before relaunch")
        gpus = [int(v) for v in args.gpus.split(",")] if args.gpus else None
        ledger = execute_campaign(rows, args.output, spec, saved, gpus=gpus,
                                  max_parallel=args.max_parallel,
                                  run_timeout=args.run_timeout, only_ids=only,
                                  resume=True)
        print(json.dumps({k: ledger[k] for k in ("status", "created_at")}, indent=2))
        return 0 if ledger["status"] in ("succeeded", "completed") else 2
    if args.output.exists():
        raise CampaignError("--execute requires a new --output directory")
    try:
        gpus = [int(value) for value in args.gpus.split(",")] if args.gpus else None
    except ValueError as exc:
        raise CampaignError("--gpus must be comma-separated integer indices") from exc
    if args.max_parallel not in (1, 2) or args.run_timeout <= 0 or \
            (gpus is not None and (not gpus or len(gpus) > 2 or
                                   len(set(gpus)) != len(gpus) or min(gpus) < 0)):
        raise CampaignError("invalid GPU selection, parallelism, or timeout")
    snapshot = preflight()
    ledger = execute_campaign(rows, args.output, spec, snapshot, gpus=gpus,
                              max_parallel=args.max_parallel,
                              run_timeout=args.run_timeout, only_ids=only)
    print(json.dumps({k: ledger[k] for k in ("status", "created_at")}, indent=2))
    return 0 if ledger["status"] == "succeeded" else 2


if __name__ == "__main__":
    raise SystemExit(main())
