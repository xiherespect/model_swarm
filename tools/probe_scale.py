#!/usr/bin/env python3
"""Offline, dev-only factor-LoRA scale pilot. Parent and dry-run use stdlib only.

Default: read-only preflight + matrix. --execute claims a NEW directory, checks
an idle GPU, and runs exactly one isolated child per candidate. No resume/retry,
no test evaluation, no plots, no search_lp import, and no downloads.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments/scale_pilot.json"
TASKS = ["nlgraph", "gsm8k"]
DIRECTIONS = ["onehot", "uniform"]
SCALES = [0.5, 0.75, 1.0, 1.25, 1.5]
OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1", "WANDB_DISABLED": "true",
    "WANDB_MODE": "disabled", "TOKENIZERS_PARALLELISM": "false",
}
SOURCES = ["tools/probe_scale.py", "alpha_space.py", "evaluate.py", "eval_protocol.py"]


class ProbeError(ValueError):
    pass


class Blocked(ProbeError):
    pass


class Cancelled(Exception):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_identity(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ProbeError(f"Missing/empty local file (no download attempted): {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.absolute()), "resolved_path": str(path.resolve()),
            "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def local_path(root, value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def validate_config(config):
    fixed = {"schema_version": 1, "tasks": TASKS, "directions": DIRECTIONS,
             "scales": SCALES, "split": "dev", "subset_start": 0,
             "subset_size": 50, "protocol": "strict_v1", "eval_type": "exact_match",
             "max_configurations": 20, "max_samples": 1000,
             "expert_selection": "first_alphabetic_not_test_selected", "data_dir": "data/eval"}
    for key, expected in fixed.items():
        if config.get(key) != expected:
            raise ProbeError(f"Frozen pilot field {key}: expected {expected!r}")
    for key, maximum in (("budget_seconds", 7200), ("candidate_timeout_seconds", 7200),
                         ("gpu_max_memory_mib", 64)):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= maximum:
            raise ProbeError(f"{key} must be > 0 and <= {maximum}")
    if config.get("gpu_max_utilization_percent") != 0:
        raise ProbeError("GPU must have zero utilization")
    experts = config.get("experts", [])
    names = [item["name"] for item in experts]
    if len(names) != 10 or len(set(names)) != 10 or names != sorted(names):
        raise ProbeError("Exactly 10 distinct alphabetically sorted expert names required")
    for item in experts:
        if Path(item["name"]).name != item["name"] or item["name"] in (".", ".."):
            raise ProbeError("Unsafe expert name")
        for field in ("config_sha256", "weights_sha256"):
            value = item.get(field, "")
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ProbeError(f"Invalid {field} for {item['name']}")
    return config


def build_matrix(config):
    validate_config(config)
    candidates = []
    for task in config["tasks"]:
        for direction in config["directions"]:
            for scale in config["scales"]:
                base = [1.0] + [0.0] * 9 if direction == "onehot" else [0.1] * 10
                candidates.append({
                    "id": f"{task}_{direction}_c{scale:g}", "task": task,
                    "direction": direction, "scale": scale,
                    "alpha": [scale * x for x in base],
                    "effective_delta_multiplier": scale ** 2,
                    "onehot_expert": config["experts"][0]["name"] if direction == "onehot" else None,
                    "smoke": task == TASKS[0] and direction == "onehot" and scale == 1,
                })
    candidates.sort(key=lambda row: not row["smoke"])  # stable; smoke IS one grid point
    return candidates


def tensor_signature(path):
    """Read safetensors headers without torch; reject truncated payloads."""
    path = Path(path)
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ProbeError(f"Truncated safetensors: {path}")
        size = struct.unpack("<Q", prefix)[0]
        if not 0 < size <= min(16 * 1024 * 1024, path.stat().st_size - 8):
            raise ProbeError(f"Invalid safetensors header: {path}")
        header = json.loads(handle.read(size))
    signature = {}
    end = 0
    widths = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8,
              "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
    intervals = []
    for key, value in header.items():
        if key == "__metadata__":
            continue
        dtype, shape, offsets = value["dtype"], value["shape"], value["data_offsets"]
        if dtype not in widths or any(type(d) is not int or d < 0 for d in shape):
            raise ProbeError(f"Unsupported tensor layout: {path}:{key}")
        start, stop = offsets
        if not 0 <= start <= stop or stop - start != math.prod(shape) * widths[dtype]:
            raise ProbeError(f"Invalid tensor offsets: {path}:{key}")
        intervals.append((start, stop))
        signature[key] = {"dtype": dtype, "shape": shape}
    for start, stop in sorted(intervals):
        if start != end:
            raise ProbeError(f"Noncontiguous tensor payload: {path}")
        end = stop
    if not signature or 8 + size + end != path.stat().st_size:
        raise ProbeError(f"Incomplete safetensors payload: {path}")
    return signature


def inspect_snapshot(base):
    if not base.is_dir():
        raise ProbeError(f"Local snapshot missing: {base}")
    files = ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
    index = base / "model.safetensors.index.json"
    if index.exists():
        mapping = read_json(index).get("weight_map", {})
        if not mapping:
            raise ProbeError("Empty model shard index")
        shards = sorted(set(mapping.values()))
        if any(Path(name).name != name or not name.endswith(".safetensors") for name in shards):
            raise ProbeError("Unsafe shard index paths")
        files += [index.name] + shards
        actual_keys = set()
        for name in shards:
            keys = set(tensor_signature(base / name))
            if actual_keys & keys or keys != {k for k, v in mapping.items() if v == name}:
                raise ProbeError(f"Shard keys disagree with index: {name}")
            actual_keys.update(keys)
        layout = "indexed_shards"
    else:
        files.append("model.safetensors")
        tensor_signature(base / "model.safetensors")
        layout = "single_safetensors_no_index_needed"
    if (base / "generation_config.json").exists():
        files.append("generation_config.json")
    return {"path": str(base), "layout": layout,
            "files": {name: file_identity(base / name) for name in files}}


def inspect_experts(config, root):
    pool_dir = local_path(root, config["expert_dir"])
    # Match ExpertPool's discovery, also rejecting incomplete expert directories.
    actual = sorted(p.name for p in pool_dir.iterdir() if p.is_dir())
    expected = [item["name"] for item in config["experts"]]
    if actual != expected:
        raise ProbeError(f"Expert pool identity mismatch: {actual!r} != {expected!r}")
    identities, reference_config, reference_signature = [], None, None
    for expected_item in config["experts"]:
        directory = pool_dir / expected_item["name"]
        cfg = file_identity(directory / "adapter_config.json")
        weights = file_identity(directory / "adapter_model.safetensors")
        if (cfg["sha256"] != expected_item["config_sha256"] or
                weights["sha256"] != expected_item["weights_sha256"]):
            raise ProbeError(f"Pinned expert content hash mismatch: {directory}")
        semantic = read_json(directory / "adapter_config.json")
        if isinstance(semantic.get("target_modules"), list):
            semantic["target_modules"] = sorted(semantic["target_modules"])
        if (semantic.get("peft_type") != "LORA" or semantic.get("bias", "none") != "none"
                or semantic.get("use_dora") or semantic.get("modules_to_save")):
            raise ProbeError("Only plain, bias-free factor LoRA is supported")
        signature = tensor_signature(directory / "adapter_model.safetensors")
        if not all(".lora_A." in key or ".lora_B." in key for key in signature):
            raise ProbeError("Expert contains non-factor tensors")
        if reference_config is None:
            reference_config, reference_signature = semantic, signature
        if semantic != reference_config or signature != reference_signature:
            raise ProbeError(f"Incompatible expert configuration/tensors: {directory}")
        identities.append({"name": directory.name, "config": cfg, "weights": weights})
    return {"path": str(pool_dir), "names": expected, "onehot_expert": expected[0],
            "selection": config["expert_selection"], "experts": identities,
            "semantic_config_sha256": canonical_hash(reference_config),
            "tensor_signature_sha256": canonical_hash(reference_signature)}


def check_evaluate_api(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "evaluate"]
    required = {"model_path", "eval_type", "dataset", "gpu_id", "base_model",
                "subset_size", "subset_start", "protocol", "result_dir"}
    return bool(functions and required <= {
        arg.arg for arg in functions[0].args.args + functions[0].args.kwonlyargs})


def preflight(config, root=ROOT):
    validate_config(config)
    root = Path(root).resolve()
    experts = inspect_experts(config, root)
    datasets = {}
    for task in TASKS:
        path = local_path(root, config["data_dir"]) / f"{task}.json"
        identity = file_identity(path)
        data = read_json(path)
        dev = data.get("dev")
        if not isinstance(dev, list) or len(dev) < 50:
            raise ProbeError(f"{task}: need at least 50 dev items")
        selected = dev[:50]  # Never inspect/select the test split.
        datasets[task] = {"file": identity, "split": "dev", "start": 0, "size": 50,
                          "items": [{"index": i, "id": item.get("id"),
                                     "sha256": canonical_hash(item),
                                     "evaluation_question_sha256": hashlib.sha256(json.dumps(
                                         item, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
                                    for i, item in enumerate(selected)]}
    sources = {name: file_identity(root / name) if (root / name).is_file() else None
               for name in SOURCES}
    api_ready = (all(sources.values()) and check_evaluate_api(root / "evaluate.py"))
    # Strict evaluate resolves data/eval alongside its source: do not permit a
    # driver-only data path override that would hash one file but evaluate another.
    if local_path(root, config["data_dir"]) != (root / "data/eval").resolve():
        raise ProbeError("Data path must match strict evaluate's repository data/eval")
    snapshot = {"config": config, "repo_root": str(root), "experts": experts,
                "data": datasets, "base_model": inspect_snapshot(local_path(root, config["base_model"])),
                "sources": sources, "strict_api_ready": bool(api_ready),
                "offline_environment": OFFLINE_ENV, "candidates": build_matrix(config),
                "planned_samples": 1000, "budget_accounting":
                "Conservative monotonic wall clock from before first child spawn through finalization; includes load, materialize, failures and gaps. No retry."}
    snapshot["identity_sha256"] = canonical_hash(snapshot)
    return snapshot


def query_idle_gpu(config, requested=None):
    """Fail closed on missing nvidia-smi, unknown metrics, or compute processes."""
    def query(fields, kind):
        try:
            result = subprocess.run(["nvidia-smi", f"--query-{kind}={fields}",
                                     "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, timeout=10, check=True)
        except (OSError, subprocess.SubprocessError) as exc:
            raise Blocked(f"Cannot establish GPU idleness: {exc}") from exc
        return list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    devices = query("index,uuid,memory.used,utilization.gpu", "gpu")
    processes = query("gpu_uuid,pid", "compute-apps")
    busy = {row[0].strip() for row in processes if row}
    for row in devices:
        try:
            index, uuid, memory, utilization = [value.strip() for value in row]
            index, memory, utilization = int(index), int(memory), int(utilization)
        except (ValueError, TypeError):
            continue
        if requested is not None and index != requested:
            continue
        if uuid not in busy and memory <= config["gpu_max_memory_mib"] and utilization == 0:
            return {"index": index, "uuid": uuid, "memory_used_mib": memory,
                    "utilization_percent": utilization, "compute_processes": [], "checked_at": utc_now()}
    raise Blocked("No idle GPU satisfies memory/utilization/process checks; no jobs started or stopped")


def terminate_owned_child(child):
    # start_new_session=True makes this PGID exclusively ours, including descendants.
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
            # A successful worker must not leave background descendants behind either.
            terminate_owned_child(child)
            return {"status": "exited", "returncode": code,
                    "elapsed_seconds": time.monotonic() - started}
    except BaseException:
        if child is not None:
            terminate_owned_child(child)
        raise


def archive_samples(path):
    if not path.is_file():
        return None  # Missing partial archive is UNKNOWN, not zero generations.
    try:
        records = read_json(path).get("records")
        return len(records) if isinstance(records, list) else None
    except (ValueError, OSError, AttributeError):
        return None


def validate_archive(path, candidate, expected_items=None):
    archive = read_json(path)
    metadata = archive.get("metadata", archive)  # strict writer uses flattened metadata
    records, score = archive.get("records"), archive.get("score")
    if not isinstance(metadata, dict) or not isinstance(records, list) or len(records) != 50:
        raise ProbeError("Strict success requires metadata and exactly 50 generated records")
    protocol = metadata.get("protocol", metadata.get("scoring_version"))
    if protocol != "strict_v1":
        raise ProbeError("evaluation.json must identify strict_v1 scoring")
    expected = {"dataset": candidate["task"], "split": "dev", "subset_start": 0}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ProbeError(f"evaluation.json metadata.{key} must equal {value!r}")
    if metadata.get("subset_stop", 50) != 50 or metadata.get("subset_size", 50) != 50:
        raise ProbeError("Archive slice must be dev[0:50]")
    if metadata.get("n_samples", len(records)) != 50:
        raise ProbeError("Archive n_samples must equal 50")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ProbeError("Invalid archive score")
    indices, item_scores = [], []
    for record in records:
        index = record.get("index", record.get("question_index", record.get("dataset_index")))
        raw = record.get("raw", record.get("raw_generation"))
        if (type(index) is not int or not isinstance(raw, str) or "gold" not in record
                or not ("parsed" in record or "parsed_answer" in record)):
            raise ProbeError("Each record needs index, raw, parsed answer and gold")
        value = record.get("score")
        if not isinstance(value, (int, float)) or value not in (0, 1):
            raise ProbeError("Each exact-match record needs a binary score")
        if expected_items is not None:
            expected_hash = expected_items[len(indices)]["evaluation_question_sha256"]
            if record.get("question_sha256") != expected_hash:
                raise ProbeError("Record question hash does not match frozen dev item")
        indices.append(index)
        item_scores.append(value)
    if indices != list(range(50)) or not math.isclose(score, sum(item_scores) / 50, abs_tol=1e-10):
        raise ProbeError("Record indices/score disagree with dev slice or aggregate")
    return float(score)


def persist_state(output, state):
    state["updated_at"] = utc_now()
    state["actual_samples_known"] = sum(row["actual_samples"] or 0 for row in state["ledger"])
    state["has_unknown_sample_counts"] = any(row["actual_samples"] is None for row in state["ledger"])
    save_json(output / "status.json", state)
    fields = ["id", "task", "direction", "scale", "effective_delta_multiplier", "smoke",
              "status", "actual_samples", "elapsed_seconds", "score", "error"]
    temporary = output / "summary.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(state["ledger"])
    os.replace(temporary, output / "summary.csv")


def execute(config, output, *, root=ROOT, gpu_id=None, config_path=None,
            gpu_check=query_idle_gpu, command_factory=None):
    """Internal injections permit CPU-only dummy-child tests, never CLI overrides."""
    validate_config(config)
    output = Path(output).absolute()
    # Refuse all existing paths, including empty directories and dangling symlinks.
    if os.path.lexists(output):
        raise ProbeError(f"Output already exists; refusing to touch it: {output}")
    output.mkdir(parents=True, exist_ok=False)
    output = output.resolve()
    state = {"status": "preflight", "created_at": utc_now(), "output": str(output),
             "elapsed_seconds": 0.0, "budget_seconds": config["budget_seconds"],
             "ledger": [dict(row, status="pending", actual_samples=0, elapsed_seconds=0.0,
                             score=None, error=None) for row in build_matrix(config)]}
    started, current = None, None
    old_handlers = {}

    def cancelled(signum, _frame):
        raise Cancelled(f"Received signal {signum}")

    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            old_handlers[sig] = signal.signal(sig, cancelled)
        persist_state(output, state)
        snapshot = preflight(config, root)
        snapshot["created_at"] = utc_now()
        if config_path is not None:
            snapshot["config_file"] = file_identity(config_path)
        save_json(output / "metadata.json", snapshot)
        if not snapshot["strict_api_ready"]:
            raise Blocked("Strict evaluate API/source files not ready; require protocol and result_dir")
        gpu = gpu_check(config, gpu_id)
        state["gpu"] = gpu
        state["identity_sha256"] = snapshot["identity_sha256"]
        env = dict(os.environ, **OFFLINE_ENV, CUDA_VISIBLE_DEVICES=gpu["uuid"])
        state["status"] = "running"
        for row in state["ledger"]:
            if started is not None and time.monotonic() - started >= config["budget_seconds"]:
                state["status"] = "budget_exhausted"
                break
            checked = gpu_check(config, gpu["index"])
            if checked["uuid"] != gpu["uuid"]:
                raise Blocked("Selected GPU identity changed")
            current = row
            directory = output / row["id"]
            directory.mkdir(exist_ok=False)
            job_path = directory / "job.json"
            save_json(job_path, {"snapshot": str(output / "metadata.json"),
                                 "candidate_id": row["id"], "gpu_id": 0})
            row.update(status="running", actual_samples=None, started_at=utc_now(),
                       result_dir=str(directory / "results"))
            if started is None:
                started = time.monotonic()  # Includes interpreter/import/load/materialize/failure.
            remaining = config["budget_seconds"] - (time.monotonic() - started)
            state["elapsed_seconds"] = time.monotonic() - started
            persist_state(output, state)
            if remaining <= 0:
                row.update(status="budget_exhausted", actual_samples=0)
                state["status"] = "budget_exhausted"
                break
            timeout = min(config["candidate_timeout_seconds"], remaining)
            command = (command_factory(job_path) if command_factory else
                       [sys.executable, "-I", str(Path(__file__).resolve()), "--worker", str(job_path)])
            child_start = time.monotonic()
            try:
                result = run_child(command, timeout, directory / "worker.log", env, str(root))
            finally:
                row["elapsed_seconds"] = time.monotonic() - child_start
                row["actual_samples"] = archive_samples(directory / "results/evaluation.json")
            row["returncode"] = result["returncode"]
            if result["status"] == "timeout":
                row["status"] = "timeout"
                row["error"] = "Owned worker killed on wall-clock timeout; partial files retained"
                state["status"] = ("budget_exhausted" if remaining <= config["candidate_timeout_seconds"] else "timeout")
                break
            if result["returncode"] != 0:
                raise ProbeError(f"Worker exit {result['returncode']}; see {directory / 'worker.log'}")
            row["score"] = validate_archive(directory / "results/evaluation.json", row,
                                              snapshot["data"][row["task"]]["items"])
            row["evaluation_file"] = file_identity(directory / "results/evaluation.json")
            row["status"] = "succeeded"
            current = None
            state["elapsed_seconds"] = time.monotonic() - started
            persist_state(output, state)
        else:
            state["status"] = "succeeded"
        if started is not None and time.monotonic() - started >= config["budget_seconds"]:
            state["status"] = "budget_exhausted"
    except (Exception, KeyboardInterrupt) as exc:
        state["status"] = ("blocked" if isinstance(exc, Blocked) else
                           "cancelled" if isinstance(exc, (Cancelled, KeyboardInterrupt)) else "failed")
        state["error"] = f"{type(exc).__name__}: {exc}"
        if current is not None:
            current.update(status=state["status"], error=state["error"])
    finally:
        if started is not None:
            state["elapsed_seconds"] = time.monotonic() - started
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        persist_state(output, state)
    return state


def require_new_target(path, parent):
    path, parent = Path(path), Path(parent).resolve()
    if os.path.lexists(path) or path.parent.resolve() != parent:
        raise ProbeError(f"Materialization/result target must be new and directly inside candidate: {path}")


def worker(job_path):
    """Private single-candidate process. Only here may heavy dependencies load."""
    job_path = Path(job_path).resolve()
    job = read_json(job_path)
    snapshot_path = Path(job["snapshot"]).resolve()
    snapshot = read_json(snapshot_path)
    config = snapshot["config"]
    validate_config(config)
    candidates = [row for row in build_matrix(config) if row["id"] == job["candidate_id"]]
    if len(candidates) != 1 or job_path.parent.parent != snapshot_path.parent:
        raise ProbeError("Invalid private worker job location/candidate")
    candidate = candidates[0]
    directory = job_path.parent
    if directory.name != candidate["id"] or job["gpu_id"] != 0:
        raise ProbeError("Invalid worker candidate/GPU")
    os.environ.update(OFFLINE_ENV)
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ProbeError("Worker requires parent's selected CUDA_VISIBLE_DEVICES")
    status_path = directory / "worker_status.json"
    status = {"status": "verifying", "started_at": utc_now(), "candidate": candidate}
    save_json(status_path, status)
    try:
        root = Path(snapshot["repo_root"])
        observed = preflight(config, root)
        if observed["identity_sha256"] != snapshot["identity_sha256"]:
            raise ProbeError("Source/data/expert/base content changed after snapshot; refusing run")
        adapter, result_dir = directory / "adapter", directory / "results"
        require_new_target(adapter, directory)
        require_new_target(result_dir, directory)
        sys.path.insert(0, str(root))
        from alpha_space import ExpertPool
        status["status"] = "materializing"
        save_json(status_path, status)
        pool = ExpertPool(snapshot["experts"]["path"])
        if pool.names != snapshot["experts"]["names"] or pool.n != 10:
            raise ProbeError("Heavy ExpertPool discovery disagrees with pinned pool")
        require_new_target(adapter, directory)  # materialize deletes existing targets: guard immediately.
        pool.materialize(candidate["alpha"], str(adapter))
        if candidate["smoke"]:
            from safetensors.torch import load_file
            import torch
            actual = load_file(str(adapter / "adapter_model.safetensors"), device="cpu")
            expected = pool._experts[0]
            if set(actual) != set(expected) or any(not torch.equal(actual[k], expected[k]) for k in expected):
                raise ProbeError("Scale-1 onehot smoke: materialized tensors differ from first expert")
            status["onehot_tensor_equality"] = "exact_cpu_all_tensors"
            del actual
        del pool
        status["materialized_adapter"] = {
            name: file_identity(adapter / name)
            for name in ("adapter_config.json", "adapter_model.safetensors")}
        status["status"] = "evaluating"
        save_json(status_path, status)
        os.chdir(root)  # evaluate's legacy data paths are repository-relative.
        from evaluate import evaluate
        result = evaluate(str(adapter), "exact_match", candidate["task"], 0,
                          base_model=snapshot["base_model"]["path"], subset_size=50,
                          subset_start=0, protocol="strict_v1", result_dir=str(result_dir))
        score = validate_archive(result_dir / "evaluation.json", candidate,
                                 snapshot["data"][candidate["task"]]["items"])
        if not isinstance(result, (int, float)) or not math.isclose(float(result), score, abs_tol=1e-10):
            raise ProbeError("evaluate return value disagrees with strict archive")
        status.update(status="succeeded", score=score, actual_samples=50)
        return 0
    except BaseException as exc:
        status.update(status="cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error=f"{type(exc).__name__}: {exc}",
                      actual_samples=archive_samples(directory / "results/evaluation.json"))
        traceback.print_exc()
        return 1
    finally:
        status["finished_at"] = utc_now()
        save_json(status_path, status)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, help="New run directory; existing paths always rejected")
    parser.add_argument("--execute", action="store_true", help="Explicitly allow real single-idle-GPU execution")
    parser.add_argument("--gpu", type=int, help="Physical nvidia-smi index; must be idle (otherwise auto-select)")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args.worker)
    try:
        config = validate_config(read_json(args.config))
        if args.gpu is not None and args.gpu < 0:
            raise ProbeError("GPU index must be nonnegative")
        if args.output is not None and os.path.lexists(args.output):
            raise ProbeError(f"Output already exists; refusing to touch it: {args.output}")
        if args.execute:
            if args.output is None:
                raise ProbeError("--execute requires an explicit NEW --output directory")
            state = execute(config, args.output, gpu_id=args.gpu, config_path=args.config)
            print(json.dumps({key: state.get(key) for key in
                              ("status", "output", "elapsed_seconds", "actual_samples_known", "has_unknown_sample_counts", "error")}, indent=2))
            return 0 if state["status"] == "succeeded" else 2
        snapshot = preflight(config)
        print("DRY RUN ONLY: no directory writes, GPU queries, heavy imports or generation.")
        print(f"Expert order: {', '.join(snapshot['experts']['names'])}")
        print(f"Onehot: {snapshot['experts']['onehot_expert']} (first alphabetic, not a test winner)")
        print(f"Snapshot layout: {snapshot['base_model']['layout']}")
        print(f"Source/data/expert/base SHA-256 identity: {snapshot['identity_sha256']}")
        print(f"Strict evaluate API ready: {snapshot['strict_api_ready']} (execute blocks if false)")
        print("task\tdirection\tscale\teffective_delta_c_squared\tsamples\tsmoke")
        for row in snapshot["candidates"]:
            print(f"{row['task']}\t{row['direction']}\t{row['scale']:g}\t{row['effective_delta_multiplier']:g}\t50\t{row['smoke']}")
        print(f"20 configurations / 1000 dev generations; one GPU; <= {config['budget_seconds']:g}s conservative wall clock; no retries.")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"probe_scale: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
