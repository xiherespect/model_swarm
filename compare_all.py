#!/usr/bin/env python3
"""汇总所有算法 run 的终值对比表。

用法: python compare_all.py            # 扫描 search/ 下全部已完成 run
支持两种日志格式:
  - PSO (search.py):   "final metrics for test: {...dict...}"
  - LP/SGMS/GENOME:    "final metrics: starting_best_dev=... ending_best_dev=..."
"""
import glob
import json
import os
import re
import sys

RUNS = []


def parse_run(log_path):
    name = os.path.basename(os.path.dirname(log_path))
    text = open(log_path, errors="ignore").read()

    # PSO dict 格式
    m = re.findall(r"final metrics for test: (\{.*\})", text)
    if m:
        d = eval(m[-1])
        return {
            "run": name, "algo": "PSO",
            "dev": d.get("ending_best_validation_utility"),
            "test": d.get("ending_best_single_test_accuracy"),
            "dev_start": d.get("starting_best_validation_utility"),
            "budget": "",
        }
    # LP/SGMS/GENOME 行格式
    m = re.search(r"final metrics: starting_best_dev=([\d.]+) ending_best_dev=([\d.]+) "
                  r"ending_best_test=([\d.]+)(?: full_evals=(\d+) coarse_evals=(\d+))?", text)
    if m:
        full, coarse = m.group(4), m.group(5)
        equiv = ""
        if full is not None:
            equiv = f"{int(full) + 0.25 * int(coarse):.0f}"
        algo = "SGMS" if "sgms" in name else ("GENOME" if "genome" in name else "LP")
        return {
            "run": name, "algo": algo,
            "dev": float(m.group(2)), "test": float(m.group(3)),
            "dev_start": float(m.group(1)), "budget": equiv,
        }
    return None


def dataset_of(name):
    for ds in ["hellaswag", "nlgraph", "gsm8k", "medical"]:
        if ds in name:
            return ds
    return "?"


for log_path in sorted(glob.glob("search/*/log.txt")):
    r = parse_run(log_path)
    if r:
        r["dataset"] = dataset_of(r["run"])
        RUNS.append(r)

if not RUNS:
    print("无已完成 run"); sys.exit(1)

cur = {"hellaswag": None, "nlgraph": None}
for ds in ["hellaswag", "nlgraph", "gsm8k", "medical"]:
    rows = [r for r in RUNS if r["dataset"] == ds and r["test"] is not None]
    if not rows:
        continue
    print(f"\n=== {ds} ===")
    print(f"{'run':<34}{'algo':<7}{'dev':>7}{'test':>7}{'起点':>7}{'等效预算':>8}")
    rows.sort(key=lambda r: -(r["dev"] or 0))
    for r in rows:
        print(f"{r['run']:<34}{r['algo']:<7}{r['dev']:>7.3f}{r['test']:>7.3f}"
              f"{(r['dev_start'] or 0):>7.3f}{r['budget']:>8}")
    best_test = max(rows, key=lambda r: r["test"])
    print(f"  → test 最优: {best_test['run']} ({best_test['test']:.3f}, {best_test['algo']})")
