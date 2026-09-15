#!/usr/bin/env python3
"""汇总 sweep 结果：解析各 run 的 log.txt 末尾 final metrics + args.txt 超参 → CSV + 排序打印。

用法: python3 sweep_collect.py ["search/<glob>/log.txt"]
不传参数时默认匹配所有 sweep run。
"""
import ast
import csv
import glob
import os
import re
import sys

PATTERN = sys.argv[1] if len(sys.argv) > 1 else "search/*_sweep_*/log.txt"
# sweep_search.sh 的命名: {dataset}_{tag#sweep_}_{rNN}_{hostname}/log.txt，这里放宽匹配
if not glob.glob(PATTERN):
    PATTERN = "search/*_*_r[0-9][0-9]*/log.txt"


def parse_args_txt(path):
    """args.txt 是 Namespace(...) repr，用正则抽超参。"""
    if not os.path.exists(path):
        return {}
    raw = open(path).read()
    out = {}
    for k in ["inertia", "cognitive_coeff", "social_coeff", "repel_coeff", "step_length"]:
        m = re.search(rf"{k}='([^']*)'|{k}=([^,\)]+)", raw)
        if m:
            out[k] = m.group(1) if m.group(1) is not None else m.group(2)
    return out


def parse_final_metrics(log_path):
    text = open(log_path, errors="ignore").read()
    hits = re.findall(r"final metrics for test: (\{.*\})", text)
    if not hits:
        return None
    try:
        return ast.literal_eval(hits[-1])
    except (ValueError, SyntaxError):
        return None


rows = []
for log_path in sorted(glob.glob(PATTERN)):
    run_dir = os.path.dirname(log_path)
    name = os.path.basename(run_dir)
    metrics = parse_final_metrics(log_path)
    hp = parse_args_txt(os.path.join(run_dir, "args.txt"))
    rows.append({
        "run": name.rsplit("_", 1)[0],  # 去掉 hostname 后缀
        "status": "done" if metrics else "unfinished",
        "dev_start": (metrics or {}).get("starting_best_validation_utility", ""),
        "dev_end": (metrics or {}).get("ending_best_validation_utility", ""),
        "test_end": (metrics or {}).get("ending_best_single_test_accuracy", ""),
        "improve_count": (metrics or {}).get("g_history_improve_count", ""),
        "inertia": hp.get("inertia", ""),
        "cog": hp.get("cognitive_coeff", ""),
        "soc": hp.get("social_coeff", ""),
        "repel": hp.get("repel_coeff", ""),
        "step": hp.get("step_length", ""),
    })

if not rows:
    print("没有匹配到任何 run 目录"); sys.exit(1)

rows.sort(key=lambda r: float(r["dev_end"]) if r["dev_end"] != "" else -1, reverse=True)

with open("sweep_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"{'run':<34}{'dev_start':>9}{'dev_end':>9}{'test_end':>9}  hyperparams (inertia/cog/soc/repel/step)")
print("-" * 100)
for r in rows:
    if r["status"] != "done":
        print(f"{r['run']:<34}{'-- 未完成 --':>9}")
        continue
    print(f"{r['run']:<34}{r['dev_start']:>9}{r['dev_end']:>9}{r['test_end']:>9}  "
          f"{r['inertia']}/{r['cog']}/{r['soc']}/{r['repel']}/{r['step']}")

done = [r for r in rows if r["status"] == "done"]
print("-" * 100)
print(f"完成 {len(done)}/{len(rows)} 个。注意：按 dev_end 挑最优时请同时核对 test_end（赢者诅咒）。")
