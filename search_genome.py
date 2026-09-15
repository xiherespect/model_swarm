"""search_genome.py — GENOME (arXiv:2503.01155, ACL 2026) 的忠实移植。

按官方仓库 ZhangYiqun018/GENOME 的 scripts/genome.sh 默认配置实现：
  init: N=10 个体，随机专家对 (a,b) 线性合并 t·A+(1-t)·B, t~U(0,1)
  每轮: 交叉(cross_rate=0 → 关闭) → 变异(im 0.1: 个体级；gm 0.1+σ0.001: 权重逐元素噪声)
        → 全量评测 → 选择(elite 1 + 锦标赛3)
  早停: 全局最优 >5 轮不变；iters=10
评测/专家池复用本项目 evaluate.py + alpha_space.py（同池同数据可比）。
个体存物化 adapter 目录（变异离开张成空间，无法用 α 表示）。
"""
import argparse
import datetime
import json
import os
import random
import shutil
import socket
import time
import uuid

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from alpha_space import ExpertPool
from evaluate import evaluate, evaluate_test
from search_lp import eval_batch  # spawn 池并行评测

try:
    import wandb
except ImportError:
    wandb = None


def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    logging.info(msg)
    logging.getLogger().handlers[0].flush()


import logging


class Individual:
    def __init__(self, ind_dir, score=None, parent=None):
        self.dir = ind_dir
        self.score = score
        self.parent = parent


def mutate_ind(ind, out_dir, gm_rate, sigma, rng_tensor=torch):
    """GENOME 变异：x' = x + mask·N(0,σ²)，mask~Bernoulli(gm_rate) 逐元素。"""
    sd = load_file(os.path.join(ind.dir, "adapter_model.safetensors"), device="cpu")
    mutated = {}
    with torch.no_grad():
        for k, t in sd.items():
            mask = (torch.rand(t.shape) < gm_rate).to(t.dtype)
            noise = torch.randn(t.shape) * sigma
            mutated[k] = t + noise * mask
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    save_file(mutated, os.path.join(out_dir, "adapter_model.safetensors"))
    shutil.copyfile(
        os.path.join(ind.dir, "adapter_config.json"),
        os.path.join(out_dir, "adapter_config.json"),
    )
    return Individual(out_dir, parent=ind.dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--name", required=True)
    ap.add_argument("-e", "--eval_type", required=True)
    ap.add_argument("-d", "--dataset", required=True)
    ap.add_argument("-g", "--gpus", default="0,1,2,3")
    ap.add_argument("-b", "--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("-i", "--expert_dir", default="./initial_experts")
    # GENOME 官方默认（scripts/genome.sh）
    ap.add_argument("--N", type=int, default=10, help="种群大小")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--cross_rate", type=float, default=0.0, help="官方脚本默认 0（关闭）")
    ap.add_argument("--im_rate", type=float, default=0.1, help="个体变异率")
    ap.add_argument("--gm_rate", type=float, default=0.1, help="基因(逐元素)变异率")
    ap.add_argument("--sigma", type=float, default=0.001)
    ap.add_argument("--elite_percent", type=float, default=0.1)
    ap.add_argument("--early_stop_iter", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--project_name_wb", default="keg218-swarm-qwen2.5b")
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--skip_test", action="store_true")
    ap.add_argument("--fake_eval", action="store_true", help="干跑：假分数遍历全部代码路径")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    search_pass_name = args.name + "_" + socket.gethostname()
    run_dir = os.path.join("search", search_pass_name)
    tmp_dir = os.path.join(run_dir, "pop")
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "global_best"), exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(run_dir, "log.txt"), level=logging.INFO,
        format="%(asctime)s INFO:root:%(message)s", datefmt="%H:%M:%S",
    )
    log(f"GENOME search initialized. {now_str()}")
    log(f"args: {vars(args)}")

    wb = None
    if wandb is not None and not args.no_wandb:
        try:
            wb = wandb.init(project=args.project_name_wb, name=f"genome_{args.name}", config=vars(args))
        except Exception as e:
            log(f"wandb init failed ({e}), continue without wandb")

    pool = ExpertPool(args.expert_dir)
    n = pool.n
    log(f"expert pool: {n} experts: {pool.names}")

    fake_rng = random.Random(7)
    if args.fake_eval:
        log("FAKE EVAL MODE: 仅代码路径测试")
        pool.materialize = lambda a, d: (os.makedirs(d, exist_ok=True), d)[1]

        def _fake_touch(d):
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "adapter_config.json"), "w") as f:
                f.write("{}")
            return d

        # 必须用 globals[] 重绑定（局部 def 会让真实路径 NameError——18:42 事故根因）
        def _fmute(ind, out_dir, gm_rate, sigma, rng_tensor=torch):
            return Individual(_fake_touch(out_dir), parent=ind.dir)

        def _fmerge(d1, d2, out, weights):
            _fake_touch(out)

        def _fe(*a, **k):
            return fake_rng.uniform(0.30, 0.75)

        def _fet(*a, **k):
            return fake_rng.uniform(0.55, 0.70)

        def _feb(dirs, *a, **k):
            return [fake_rng.uniform(0.30, 0.75) for _ in dirs]

        globals()["mutate_ind"] = _fmute
        globals()["_merge_two_dirs"] = _fmerge
        globals()["evaluate"] = _fe
        globals()["evaluate_test"] = _fet
        globals()["eval_batch"] = _feb

    gpus = [int(x) for x in args.gpus.split(",")]
    eval_count = 0

    def eval_all(inds):
        """评测所有未评测个体（并行），返回个体列表。"""
        nonlocal eval_count
        todo = [i for i in inds if i.score is None]
        if not todo:
            return todo
        scores = eval_batch([i.dir for i in todo], args.eval_type, args.dataset, gpus, args.base_model)
        eval_count += len(todo)
        for i, s in zip(todo, scores):
            i.score = s
        return todo

    # ---- init：N 个随机专家对的线性合并（t~U(0,1)）----
    log(f"initializing {args.N} individuals from random pairs...")
    pop = []
    for k in range(args.N):
        a, b = rng.sample(range(n), 2)
        t = rng.random()
        alpha = np.zeros(n)
        alpha[a], alpha[b] = t, 1 - t
        d = os.path.join(tmp_dir, f"init_{k}")
        pool.materialize(alpha, d)
        pop.append(Individual(d, parent=[pool.names[a], pool.names[b]]))
    eval_all(pop)
    g_score = max(i.score for i in pop)
    g_dir = max(pop, key=lambda i: i.score).dir
    starting_best = g_score
    shutil.copytree(g_dir, os.path.join(run_dir, "global_best"), dirs_exist_ok=True)
    log(f"starting best (dev): {g_score:.3f}")
    if wb is not None:
        wb.log({"g": g_score, "iter": 0, "evals": eval_count})

    elite_n = max(1, int(args.elite_percent * args.N))
    history = [g_score]
    stagnation = 0

    for it in range(1, args.iters + 1):
        log(f"--- iteration {it}! {now_str()} ---")

        # 1. 交叉（官方默认 cross_rate=0 → 跳过；保留路径以备 paper 配置）
        if args.cross_rate > 0:
            base = [i for i in pop if i.score is not None]  # 快照：只从已评测个体选亲代
            num_pairs = len(base) // 2
            for _ in range(num_pairs):
                if rng.random() <= args.cross_rate:
                    pa, pb = rng.sample(base, 2)
                    w1 = pa.score / (pa.score + pb.score + 1e-9)
                    # 个体可能已被变异（不在 span），交叉退化为权重层线性合并两个目录
                    d_new = os.path.join(tmp_dir, f"cross_{uuid.uuid4().hex[:8]}")
                    _merge_two_dirs(pa.dir, pb.dir, d_new, [w1, 1 - w1])
                    pop.append(Individual(d_new, parent=[pa.dir, pb.dir]))

        # 2. 变异：个体级概率 im，逐元素权重噪声
        new_inds = []
        for ind in pop:
            if rng.random() <= args.im_rate:
                d_new = os.path.join(tmp_dir, f"mut_{uuid.uuid4().hex[:8]}")
                new_inds.append(mutate_ind(ind, d_new, args.gm_rate, args.sigma))
        pop.extend(new_inds)
        log(f"mutation produced {len(new_inds)} new individuals, pop={len(pop)}")

        # 3. 评测新个体
        eval_all(pop)

        # 4. 选择：elite + 锦标赛(3)，淘汰者删除目录
        pop.sort(key=lambda i: -i.score)
        elites = pop[:elite_n]
        rest = pop[elite_n:]
        need = args.N - elite_n
        chosen = list(elites)
        avail = list(rest)
        while need > 0 and avail:
            k = min(3, len(avail))
            tour = rng.sample(avail, k)
            win = max(tour, key=lambda i: i.score)
            chosen.append(win)
            avail.remove(win)
            need -= 1
        for loser in avail:
            if os.path.exists(loser.dir) and loser.dir.startswith(tmp_dir):
                shutil.rmtree(loser.dir, ignore_errors=True)
        pop = chosen

        best = pop[0]
        if best.score > g_score + 1e-9:
            g_score, g_dir = best.score, best.dir
            shutil.rmtree(os.path.join(run_dir, "global_best"), ignore_errors=True)
            shutil.copytree(g_dir, os.path.join(run_dir, "global_best"))
            log(f"new global best: {g_score:.3f}")
            stagnation = 0
        else:
            stagnation += 1

        history.append(g_score)
        log(f"iter {it}: g={g_score:.3f} stagn={stagnation} pop={len(pop)} evals={eval_count}")
        if wb is not None:
            wb.log({"g": g_score, "iter": it, "evals": eval_count})
        with open(os.path.join(run_dir, "genome_scratchpad.json"), "w") as f:
            json.dump({"iter": it, "g": g_score, "history": history, "evals": eval_count,
                       "best_dir": g_dir}, f, indent=1)
        if stagnation > args.early_stop_iter:
            log("early stop triggered.")
            break

    # ---- 收尾 ----
    if args.skip_test:
        log(f"search ended (skip_test). {now_str()}")
        return
    log(f"search ended. evaluating global best on test... {now_str()}")
    test_score = evaluate_test(os.path.join(run_dir, "global_best"), args.eval_type, args.dataset,
                               gpus[0], args.base_model)
    log(f"final metrics: starting_best_dev={starting_best:.3f} ending_best_dev={g_score:.3f} "
        f"ending_best_test={test_score:.3f} full_evals={eval_count}")
    if wb is not None:
        wb.log({"final_dev": g_score, "final_test": test_score})
        wb.finish()
    log(f"the end of search... {now_str()}")


def _merge_two_dirs(d1, d2, out, weights):
    """两个 adapter 目录的线性合并（交叉用；个体可能已出 span，需在权重层做）。"""
    s1 = load_file(os.path.join(d1, "adapter_model.safetensors"), device="cpu")
    s2 = load_file(os.path.join(d2, "adapter_model.safetensors"), device="cpu")
    merged = {k: weights[0] * s1[k] + weights[1] * s2[k] for k in s1}
    os.makedirs(out, exist_ok=True)
    save_file(merged, os.path.join(out, "adapter_model.safetensors"))
    shutil.copyfile(os.path.join(d1, "adapter_config.json"), os.path.join(out, "adapter_config.json"))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        if logging.getLogger().handlers:
            logging.error("FATAL:\n" + traceback.format_exc())
            logging.getLogger().handlers[0].flush()
        else:
            traceback.print_exc()
        raise
