#!/usr/bin/env python3
"""Read-only, fail-closed status and paired evidence for the frozen RACER matrix.

Run with ``python -I tools/racer_results.py [--campaign PATH]``. JSON goes to
stdout; no model imports, evaluations, GPU access, or campaign writes occur.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "experiments/racer_frozen_v1.json"
DATASETS = {
    "nlgraph": ("nlgraph",), "gsm8k": ("gsm8k",),
    "hellaswag": ("hellaswag",), "mmlu": ("mmlu",),
    "mmlu_pro": ("mmlu_pro",), "truthfulqa": ("truthfulqa",),
    "medical": ("medqa", "medmcqa"), "science": ("scifact", "stem"),
    "culture": ("normad_country", "normad_value"),
    "kcross": ("knowledge_crosswords",),
}
STATUSES = {"pending", "running", "succeeded", "failed", "blocked",
            "timeout", "cancelled"}


class EvidenceError(ValueError):
    """The available evidence cannot establish the asserted identity/result."""


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def read_json(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"missing or symlinked file: {path}")
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, UnicodeError, ValueError) as exc:
        raise EvidenceError(f"unreadable JSON: {path}: {exc}") from exc


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def number(value):
    require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
            "score must be finite and in [0, 1]")
    return float(value)


def same(a, b):
    return math.isclose(number(a), number(b), rel_tol=0, abs_tol=1e-12)


def matrix(spec, budgets):
    require(spec.get("spec_version") == "racer_frozen_v1" and
            spec.get("protocol") == "racer_eval_v1" and
            spec.get("seeds") == [42, 43, 44] and
            spec.get("kcross_task") == "kcross", "not the frozen RACER protocol")
    tasks = spec["main_tasks"] + spec["ablation_tasks"] + spec["scale_tasks"] + ["kcross"]
    require(all(t in budgets for t in tasks) and len(set(spec["main_tasks"])) == 9 and
            len(set(spec["ablation_tasks"])) == 4 and len(set(spec["scale_tasks"])) == 2 and
            set(spec["ablation_tasks"]) <= set(spec["main_tasks"]) and
            set(spec["scale_tasks"]) <= set(spec["main_tasks"]) and
            spec["ablations"] == ["no_DE", "no_sum", "no_ES", "no_guard",
                                  "no_plateau", "no_rung2", "full_eval",
                                  "standard_SH", "cma_SH"], "unexpected matrix definition")
    rows = []

    def add(group, task, method, seed, ablation="full"):
        rows.append({"id": f"{group}_{task}_{method}" +
                     (f"_{ablation}" if ablation != "full" else "") + f"_s{seed}",
                     "group": group, "task": task, "method": method,
                     "ablation": ablation, "seed": seed, "run_test": True,
                     "budget_samples": 110 * budgets[task]})

    for task in spec["main_tasks"]:
        for seed in spec["seeds"]:
            for method in ("racer_frozen_v1", "cma_full", "pso_box", "random_box"):
                add("A_main", task, method, seed)
    for task in spec["ablation_tasks"]:
        for seed in spec["seeds"]:
            for ablation in spec["ablations"]:
                add("B_ablation", task, "cma_sh" if ablation == "cma_SH"
                    else "racer_frozen_v1", seed,
                    "full" if ablation == "cma_SH" else ablation)
    for task in spec["scale_tasks"]:
        for seed in spec["seeds"]:
            for method in ("pso_sum1", "pso_sum1_scalegrid"):
                add("C_scale", task, method, seed)
    for seed in spec["seeds"]:
        for method in ("racer_frozen_v1", "cma_full", "pso_box", "random_box"):
            add("D_kcross", "kcross", method, seed)
    require(len(rows) == 240 and len({r["id"] for r in rows}) == 240,
            "matrix is not 240 unique runs")
    return rows


def verify_file_identity(entry):
    require(type(entry) is dict and type(entry.get("path")) is str and
            type(entry.get("bytes")) is int and type(entry.get("sha256")) is str,
            "incomplete file identity")
    path = Path(entry["path"])
    # 与 campaign preflight 的 tree_identity 同语义：跟随符号链接按目标字节核验
    # （HF 缓存基模快照内全部是指向 blobs/ 的符号链接），不拒绝链接本身。
    require(path.is_file() and
            path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"],
            f"file identity mismatch: {path}")
    return path


def budgets_from_data(spec, snapshot=None):
    budgets = {}
    for task in set(spec["main_tasks"] + spec["ablation_tasks"] + spec["scale_tasks"] + ["kcross"]):
        require(task in DATASETS, f"unknown task: {task}")
        count = 0
        for dataset in DATASETS[task]:
            if snapshot is not None:
                identity = snapshot["data"][dataset]
                path = verify_file_identity(identity)
            else:
                path = ROOT / "data/eval" / f"{dataset}.json"
            data = read_json(path)
            require(type(data.get("dev")) is list and len(data["dev"]) > 0,
                    f"invalid dev dataset: {dataset}")
            count += len(data["dev"])
        budgets[task] = count
    return budgets


def verify_snapshot(snapshot, spec):
    require(type(snapshot) is dict and snapshot.get("spec") == spec,
            "snapshot differs from frozen specification")
    identity = snapshot.get("identity_sha256")
    payload = {k: v for k, v in snapshot.items() if k != "identity_sha256"}
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True,
                             separators=(",", ":")).encode()).hexdigest()
    require(identity == expected, "campaign snapshot identity mismatch")
    require(isinstance(snapshot.get("experts", {}).get("names"), list) and
            len(snapshot["experts"]["names"]) == 10 and
            snapshot.get("base_model", {}).get("files") and
            type(snapshot.get("sources")) is dict and
            type(snapshot.get("data")) is dict, "incomplete snapshot identity")
    require(set(snapshot["sources"]) >= set(spec["source_files"]) |
            {"experiments/racer_frozen_v1.json", "experiments/racer_data_audit_v1.json"},
            "missing frozen source identities")
    for entry in (*snapshot["sources"].values(), *snapshot["data"].values(),
                  *snapshot["experts"]["files"].values(),
                  *snapshot["base_model"]["files"].values()):
        verify_file_identity(entry)
    require(set(snapshot["data"]) >= {dataset for task in
            spec["main_tasks"] + spec["scale_tasks"] + ["kcross"]
            for dataset in DATASETS[task]}, "missing dataset identities")


def kcross_selection(snapshot):
    audit = read_json(verify_file_identity(
        snapshot["sources"]["experiments/racer_data_audit_v1.json"]))
    subset = audit.get("kcross_subset", {})
    selected = subset.get("selected")
    require(subset.get("dataset") == "knowledge_crosswords" and
            type(selected) is list and len(selected) == 564,
            "unexpected frozen kcross subset")
    payload = json.dumps(selected, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    require(hashlib.sha256(payload).hexdigest() == subset.get("selected_sha256"),
            "kcross subset digest mismatch")
    data = read_json(verify_file_identity(snapshot["data"]["knowledge_crosswords"]))
    questions = data["test"]
    last = -1
    for entry in selected:
        index = entry.get("index")
        require(type(index) is int and last < index < len(questions) and
                hashlib.sha256(questions[index]["question"].encode("utf-8")).hexdigest() ==
                entry.get("question_sha256"), "kcross selected question mismatch")
        last = index
    return selected


def verify_archive(path, expected, protocol, dataset=None, source=None):
    archive = read_json(path)
    require(archive.get("split") == "test" and
            archive.get("scoring_version") == protocol and
            same(archive.get("score"), expected), f"test archive mismatch: {path}")
    if dataset is not None:
        records = archive.get("records")
        require(archive.get("dataset") == dataset and type(records) is list and
                len(records) > 0 and archive.get("n_samples") == len(records) and
                [r.get("index") for r in records] == list(range(len(records))) and
                same(sum(number(r["score"]) for r in records) / len(records), expected),
                f"test records mismatch: {path}")
    return archive


def verify_result(run_dir, row, snapshot, campaign_dir):
    require(run_dir.is_dir() and not run_dir.is_symlink() and
            run_dir.parent == campaign_dir and run_dir.name == row["id"],
            "run directory escapes campaign")
    job = read_json(run_dir / "job.json")
    # prepare_run 在调度器把行标为 pending 后写 job，job.run 因此携带 status 字段；
    # 身份核验只比对矩阵字段本身，忽略调度器附加的执行状态键。
    job_run = job.get("run")
    require(job.get("snapshot") == str(campaign_dir / "snapshot.json") and
            isinstance(job_run, dict) and
            all(job_run.get(k) == v for k, v in row.items()),
            "job identity mismatch")
    summary = read_json(run_dir / "summary.json")
    for key in ("task", "method", "ablation", "seed", "budget_samples"):
        require(summary.get(key) == row[key], f"summary {key} mismatch")
    require(summary.get("status") == "succeeded" and row["run_test"] is True and
            type(summary.get("best_candidate_id")) is int and
            type(summary.get("n_candidates")) is int and
            summary["n_candidates"] > 0 and
            type(summary.get("best_alpha")) is list and
            len(summary["best_alpha"]) == len(snapshot["experts"]["names"]) and
            type(summary.get("budget_used")) in (int, float) and
            0 <= summary["budget_used"] <= row["budget_samples"],
            "incomplete successful summary")
    dev, test = number(summary.get("best_dev")), number(summary.get("test_score"))
    ledger = read_json(run_dir / "ledger.json")
    candidates, events = ledger.get("candidates"), ledger.get("events")
    require(ledger.get("method") == row["method"] and
            ledger.get("ablation") == row["ablation"] and
            ledger.get("budget_samples") == row["budget_samples"] and
            ledger.get("budget_used") == summary["budget_used"] and
            type(candidates) is list and len(candidates) == summary["n_candidates"] and
            type(events) is list, "search ledger mismatch")
    pick = summary["best_candidate_id"]
    require(0 <= pick < len(candidates) and candidates[pick].get("id") == pick and
            candidates[pick].get("alpha") == summary["best_alpha"],
            "selected candidate mismatch")
    full = {}
    for event in events:
        if event.get("kind") == "observe" and event.get("fidelity") == "full":
            cid = event.get("candidate_id")
            require(type(cid) is int and 0 <= cid < len(candidates) and cid not in full,
                    "invalid full-dev observation")
            full[cid] = number(event.get("score"))
    require(full and pick == max(full, key=lambda cid: (full[cid], -cid)) and
            same(full[pick], dev), "dev selection mismatch")
    results_dir = run_dir / "oracle/results"
    require(results_dir.is_dir() and not results_dir.is_symlink(), "missing oracle results")
    test_dirs = sorted(results_dir.glob("test_*"))
    require(len(test_dirs) == (2 if row["task"] == "kcross" else 1) and
            all(p.is_dir() and not p.is_symlink() for p in test_dirs),
            "missing or ambiguous test evidence")
    protocol = snapshot["spec"]["protocol"]
    first = verify_archive(test_dirs[0] / "evaluation.json", test, protocol,
                           DATASETS[row["task"]][0] if len(DATASETS[row["task"]]) == 1 else None)
    if len(DATASETS[row["task"]]) > 1:
        scores = first.get("per_dataset_scores")
        require(first.get("aggregation_version") == "harmonic_v1" and
                type(scores) is dict and set(scores) == set(DATASETS[row["task"]]),
                "multitask aggregation mismatch")
        components = []
        for dataset in DATASETS[row["task"]]:
            component = number(scores[dataset])
            verify_archive(test_dirs[0] / dataset / "evaluation.json", component,
                           protocol, dataset)
            components.append(component)
        harmonic = 0 if sum(components) == 0 else 2 * math.prod(components) / sum(components)
        require(same(harmonic, test), "multitask score mismatch")
    if row["task"] == "kcross":
        filtered = number(summary.get("kcross_filtered_test_score"))
        archive = verify_archive(test_dirs[1] / "evaluation.json", filtered, protocol)
        require(archive.get("dataset") == "knowledge_crosswords" and
                archive.get("subset_indices") is not None and
                archive.get("n_samples") == len(archive.get("records", [])) and
                [r.get("index") for r in archive["records"]] == archive["subset_indices"],
                "kcross subset evidence mismatch")
    else:
        require(summary.get("kcross_filtered_test_score") is None,
                "unexpected filtered test score")
    return {"dev": dev, "test": test,
            "kcross_filtered_test": summary.get("kcross_filtered_test_score")}


def build_report(spec, campaign=None, extra_campaigns=()):
    """单 campaign 或多实例合并报告；各实例用各自快照核验自己的行，
    不同实例快照只允许 created_at 与（经批准的修复带来的）源文件哈希不同，
    其余身份字段必须逐字段一致。"""
    campaign_dirs = []
    for path in (campaign, *extra_campaigns):
        if path is None:
            continue
        resolved = Path(path).absolute()
        require(resolved.is_dir() and not resolved.is_symlink(),
                f"missing or symlinked campaign directory: {resolved}")
        campaign_dirs.append(resolved)
    snapshots = {}
    ledgers = {}
    base_payload = None
    for campaign_dir in campaign_dirs:
        snapshot = read_json(campaign_dir / "snapshot.json")
        verify_snapshot(snapshot, spec)
        payload = {k: v for k, v in snapshot.items()
                   if k not in ("identity_sha256", "created_at")}
        if base_payload is None:
            base_payload = payload
        else:
            require(payload == base_payload,
                    f"campaign snapshot payload mismatch: {campaign_dir}")
        snapshots[campaign_dir] = snapshot
        ledger = read_json(campaign_dir / "status.json")
        require(type(ledger) is dict and type(ledger.get("rows")) is list,
                "invalid campaign ledger")
        ledgers[campaign_dir] = ledger
    primary = campaign_dirs[0] if campaign_dirs else None
    snapshot = snapshots.get(primary)
    expected = matrix(spec, budgets_from_data(spec, snapshot))
    by_id = {r["id"]: r for r in expected}
    listed = {}
    for campaign_dir in campaign_dirs:
        for item in ledgers[campaign_dir]["rows"]:
            require(type(item) is dict and item.get("id") in by_id and
                    item["id"] not in listed and item.get("status") in STATUSES,
                    "unknown, duplicate, or invalid ledger row")
            require(all(item.get(k) == v for k, v in by_id[item["id"]].items()),
                    f"ledger run identity mismatch: {item['id']}")
            item = dict(item, _campaign=str(campaign_dir))
            listed[item["id"]] = item
    rows = []
    for expected_row in expected:
        item = listed.get(expected_row["id"])
        status = item["status"] if item is not None else "not_scheduled"
        result = {**expected_row, "status": status, "protocol": spec["protocol"],
                  "snapshot_identity_sha256": snapshot["identity_sha256"] if snapshot else None,
                  "scores": None}
        if item is not None:
            result["campaign"] = item["_campaign"]
            for field in ("error", "returncode", "attempt_bounds", "elapsed_seconds"):
                if field in item:
                    result[field] = item[field]
        if status == "succeeded":
            try:
                run_campaign = Path(item["_campaign"])
                run_snapshot = snapshots[run_campaign]
                require(type(item.get("run_dir")) is str and
                        item["run_dir"] == str(run_campaign / expected_row["id"]),
                        "run_dir identity mismatch")
                evidence = verify_result(Path(item["run_dir"]), expected_row, run_snapshot,
                                         run_campaign)
                require(item.get("summary") == read_json(Path(item["run_dir"]) / "summary.json"),
                        "parent/worker summary mismatch")
                result["scores"] = evidence
                result["status"] = "verified"
            except (EvidenceError, KeyError, TypeError, OSError, IndexError) as exc:
                result["status"] = "unverified"
                result["error"] = str(exc)
        rows.append(result)
    verified = {(r["group"], r["task"], r["seed"], r["method"], r["ablation"]): r
                for r in rows if r["status"] == "verified"}
    pairs = []
    for row in rows:
        if row["group"] == "A_main" and row["method"] != "racer_frozen_v1":
            reference = verified.get(("A_main", row["task"], row["seed"],
                                      "racer_frozen_v1", "full"))
            kind = "main"
        elif row["group"] == "B_ablation":
            reference = verified.get(("A_main", row["task"], row["seed"],
                                      "racer_frozen_v1", "full"))
            kind = "ablation"
        else:
            continue
        paired = row["status"] == "verified" and reference is not None
        pairs.append({"kind": kind, "task": row["task"], "seed": row["seed"],
                      "reference_id": (f"A_main_{row['task']}_racer_frozen_v1_s{row['seed']}"),
                      "comparison_id": row["id"], "status": "verified" if paired else "incomplete",
                      "reference_status": reference["status"] if reference else
                      next(r["status"] for r in rows if r["id"] ==
                           f"A_main_{row['task']}_racer_frozen_v1_s{row['seed']}"),
                      "comparison_status": row["status"],
                      "test_delta_reference_minus_comparison":
                      reference["scores"]["test"] - row["scores"]["test"] if paired else None})
    return {"spec_version": spec["spec_version"], "protocol": spec["protocol"],
            "campaign": str(primary) if primary else None,
            "campaigns": [str(d) for d in campaign_dirs],
            "snapshot_identity_sha256": snapshot["identity_sha256"] if snapshot else None,
            "counts": dict(sorted(Counter(r["status"] for r in rows).items())),
            "pair_counts": dict(sorted(Counter(p["status"] for p in pairs).items())),
            "rows": rows, "pairs": pairs}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, action="append", dest="campaigns_arg",
                          help="existing scheduler campaign directory (repeatable)")
    args = parser.parse_args(argv)
    campaign_list = list(args.campaigns_arg or [])
    report = build_report(read_json(SPEC),
                          campaign_list[0] if campaign_list else None,
                          campaign_list[1:])
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
