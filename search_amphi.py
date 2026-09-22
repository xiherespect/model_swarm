"""search_amphi.py — AMPHI v3 两栖算法: 试探式路由(try-and-switch).

设计演化史(全部有实验数据背书):
  v1 探测后路由(全评17equiv)  → gsm8k v0.2 证明: 17等效探测直接毁掉SUMO纪录(-5分)
  v2 探测后路由(纯粗评13.75)  → 两地形粗评比均=1.00, 噪声吞掉信号, 判据失效
  v3 试探式路由(本版)         → 直接开跑SUMO, 轮询scratchpad: iter≥8且g未超起始值
                                → 判不可组合 → 换RACER; 否则继续, 探测费=0
依据: nlgraph的SUMO从第1轮死锁(g永远=起始值, 自身即探测器);
     gsm8k的SUMO iter8时g已0.46>起始0.395(正赛无缝继续).

子进程零改动: 只读它的 sumo_scratchpad.json 与 log.
"""
import argparse
import datetime
import json
import logging
import os
import re
import socket
import subprocess
import sys
import time


def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--name", required=True)
    ap.add_argument("-e", "--eval_type", required=True)
    ap.add_argument("-d", "--dataset", required=True)
    ap.add_argument("-g", "--gpus", default="0,1,2,3")
    ap.add_argument("-b", "--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("-i", "--expert_dir", default="./initial_experts")
    ap.add_argument("--budget", type=int, default=110)
    ap.add_argument("--probe_iters", type=int, default=8, help="SUMO 试探轮数")
    ap.add_argument("--stall_margin", type=float, default=0.005,
                    help="g 须超 起始最优+margin 才算'有生机'")
    ap.add_argument("--force_route", choices=["sumo", "racer"], default=None)
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--fake_eval", action="store_true")
    args = ap.parse_args()

    host = socket.gethostname()
    run_dir = os.path.join("search", args.name + "_" + host)
    os.makedirs(run_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(run_dir, "log.txt"), level=logging.INFO,
                        format="%(asctime)s INFO:root:%(message)s", datefmt="%H:%M:%S")
    log = logging.info
    log(f"AMPHI v3 (try-and-switch) initialized. {now_str()}")
    log(f"args: {vars(args)}")

    PY = sys.executable
    child_common = ["-g", args.gpus, "-b", args.base_model, "-i", args.expert_dir,
                    "--seed", "42"]
    if args.no_wandb:
        child_common.append("--no_wandb")
    if args.fake_eval:
        child_common.append("--fake_eval")

    def launch_sumo():
        cmd = [PY, "search_sumo.py", "-n", args.name + "_s", "-e", args.eval_type,
               "-d", args.dataset, "--budget", str(args.budget)] + child_common
        log(f"launch SUMO: {' '.join(cmd)}")
        return subprocess.Popen(cmd)

    def launch_racer(max_iter):
        cmd = [PY, "search_racer.py", "-n", args.name + "_r", "-e", args.eval_type,
               "-d", args.dataset, "--max_iter", str(max_iter)] + child_common
        log(f"launch RACER: {' '.join(cmd)}")
        return subprocess.Popen(cmd)

    # ---- 阶段1: 开跑 SUMO 并轮询 ----
    if args.force_route == "racer":
        proc = launch_racer(25)
        sys.exit(proc.wait())

    proc = launch_sumo()
    s_dir = os.path.join("search", args.name + "_s_" + host)
    pad = os.path.join(s_dir, "sumo_scratchpad.json")
    clog = os.path.join(s_dir, "log.txt")
    starting_best, decision = None, None
    while proc.poll() is None:
        time.sleep(30)
        try:
            txt = open(clog, errors="ignore").read()
            if starting_best is None:
                m = re.search(r"starting best ([0-9.]+)", txt)
                if m:
                    starting_best = float(m.group(1))
            if starting_best is not None and os.path.exists(pad):
                d = json.load(open(pad))
                if d.get("iter", 0) >= args.probe_iters:
                    if d["g"] <= starting_best + args.stall_margin:
                        decision = "switch"
                    else:
                        decision = "keep"
                    break
        except Exception:
            pass
    if decision is None:   # SUMO 自然跑完(fast mode或早停)
        log(f"SUMO finished naturally (no switch decision). exit={proc.returncode}")
        sys.exit(proc.returncode or 0)

    if decision == "keep":
        log(f"KEEP SUMO: iter8 g已超起始 ({starting_best:.3f} → alive), 探测费=0")
        sys.exit(proc.wait())

    # ---- 阶段2: 换道——杀 SUMO, RACER 带剩余预算上 ----
    try:
        d = json.load(open(pad))
        spent = d.get("full_evals", 20) + 0.25 * d.get("coarse_evals", 0)
    except Exception:
        spent = 25
    proc.terminate()
    try:
        proc.wait(timeout=60)
    except Exception:
        proc.kill()
    rem = max(15, args.budget - spent)
    mi = max(8, int(rem / 4.2))     # RACER 每轮约 4.2 等效
    log(f"SWITCH → RACER: SUMO 死锁于 iter{d.get('iter')} (g={d['g']:.3f} ≤ 起始{starting_best:.3f}), "
        f"已花 {spent:.0f} 等效, RACER 获 {rem:.0f} 等效 (~{mi} 轮)")
    with open(os.path.join(run_dir, "amphi_route.json"), "w") as f:
        json.dump({"route": "racer", "starting_best": starting_best,
                   "sumo_stall_g": d["g"], "spent": spent, "rem": rem}, f, indent=1)
    rp = launch_racer(mi)
    sys.exit(rp.wait())


if __name__ == "__main__":
    main()
