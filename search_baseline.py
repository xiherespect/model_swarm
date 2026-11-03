"""search_baseline.py — 论文基线三件套 + 免搜索锚点（与 RACER 同 eval 管线/预算口径）。

方法:
  random : 剩余预算全花在 scaled Dirichlet 全保真抽样（无漏斗——检验"随便搜行不行"）
  cma    : pycma 官方 CMA-ES（文献默认进化合并搜索器, Sakana 同款），ask-tell,
           x0=最优初始点, sigma0=0.3, popsize=8, box 约束, 同预算
  soup   : Greedy Soup 风格贪心前向选专家（起点=最优单专家, 逐个尝试并入
           w∈{0.3,0.6}, 无改进即止）
  anchors: 免搜索下界——Uniform(1/n) 与 Task-Arithmetic 缩放(λ∈{0.3,0.5,0.8,1.0}×Uniform)

共同协议: mix 初始化 20 点全保真(=RACER 同款, 含 Best Single 锚), 总等效预算
= full + 0.25×coarse 与 RACER 对齐; 终选=raw argmax(基线不配 LCB——选择机器
正是对比对象之一); 结尾 evaluate_test。
"""
import argparse
import datetime
import json
import logging
import os
import random
import socket
import time

import numpy as np

from alpha_space import ExpertPool, sample_initial_alphas
from evaluate import evaluate, evaluate_test
from search_lp import eval_batch

try:
    import wandb
except ImportError:
    wandb = None


def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    logging.info(msg)
    logging.getLogger().handlers[0].flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--name", required=True)
    ap.add_argument("-e", "--eval_type", required=True)
    ap.add_argument("-d", "--dataset", required=True)
    ap.add_argument("-m", "--method", required=True, choices=["random", "cma", "soup", "anchors"])
    ap.add_argument("-g", "--gpus", default="0,1,2,3")
    ap.add_argument("-b", "--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("-i", "--expert_dir", default="./initial_experts")
    ap.add_argument("--budget", type=int, default=110, help="等效评估总预算(含 init 20)")
    ap.add_argument("--init_size", type=int, default=20)
    ap.add_argument("--lb", type=float, default=-0.5)
    ap.add_argument("--ub", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--project_name_wb", default="keg218-swarm-qwen2.5b")
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--fake_eval", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)

    search_pass_name = args.name + "_" + socket.gethostname()
    run_dir = os.path.join("search", search_pass_name)
    os.makedirs(os.path.join(run_dir, "tmp"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "global_best"), exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(run_dir, "log.txt"), level=logging.INFO,
        format="%(asctime)s INFO:root:%(message)s", datefmt="%H:%M:%S",
    )
    log(f"BASELINE {args.method} initialized. {now_str()}")
    log(f"args: {vars(args)}")

    wb = None
    if wandb is not None and not args.no_wandb:
        try:
            wb = wandb.init(project=args.project_name_wb, name=f"base_{args.name}", config=vars(args))
        except Exception as e:
            log(f"wandb init failed ({e}), continue without wandb")

    pool = ExpertPool(args.expert_dir)
    n = pool.n
    log(f"expert pool: {n} experts")

    fake_rng = random.Random(7)
    if args.fake_eval:
        log("FAKE EVAL MODE")
        pool.materialize = lambda a, d: (os.makedirs(d, exist_ok=True), d)[1]

        def _fe(*a, **k):
            return fake_rng.uniform(0.30, 0.75)

        def _fet(*a, **k):
            return fake_rng.uniform(0.55, 0.70)

        def _feb(dirs, *a, **k):
            return [fake_rng.uniform(0.30, 0.75) for _ in dirs]

        globals()["evaluate"] = _fe
        globals()["evaluate_test"] = _fet
        globals()["eval_batch"] = _feb

    gpus = [int(x) for x in args.gpus.split(",")]
    full_evals = coarse_evals = 0

    def materialize(alpha, tag):
        return pool.materialize(alpha, os.path.join(run_dir, "tmp", tag))

    def full_eval_batch(alphas, tag):
        nonlocal full_evals
        dirs = [materialize(a, f"{tag}_{k}") for k, a in enumerate(alphas)]
        scores = eval_batch(dirs, args.eval_type, args.dataset, gpus, args.base_model)
        full_evals += len(scores)
        return scores

    def clip(a):
        return np.clip(np.asarray(a, dtype=np.float64), args.lb, args.ub)

    # ---- 公共初始化: mix 20 点全保真（与 RACER 完全一致, 公平）----
    init_alphas = [np.asarray(a) for a in sample_initial_alphas(n, args.init_size, seed=args.seed)]
    t0 = time.time()
    init_scores = full_eval_batch(init_alphas, "init")
    archive = (list(init_alphas), list(init_scores))
    g_idx = int(np.argmax(init_scores))
    g_alpha, g_score = init_alphas[g_idx], init_scores[g_idx]
    starting_best = g_score
    log(f"init done {(time.time()-t0)/60:.1f} min, starting best {g_score:.3f} "
        f"(full={full_evals})")
    pool.materialize(g_alpha, os.path.join(run_dir, "global_best"))

    # ---- anchors: Uniform + Task-Arithmetic 缩放（免搜索下界, 不耗后续预算）----
    if args.method == "anchors":
        uni = np.ones(n) / n
        cand = [uni] + [lam * uni for lam in (0.3, 0.5, 0.8)]
        sc = full_eval_batch([clip(c) for c in cand], "anchor")
        for c, s in zip(cand, sc):
            archive[0].append(clip(c)); archive[1].append(s)
        best_a = archive[0][int(np.argmax(archive[1]))]
        g_alpha, g_score = best_a, max(archive[1])
        log(f"anchors: uniform={sc[0]:.3f} ta_scaling={sc[1:]}, best={g_score:.3f}")
    elif args.method == "random":
        # 剩余预算全部用于 scaled Dirichlet 全保真抽样
        remaining = args.budget - full_evals
        seen = [np.asarray(a) for a in archive[0]]
        while remaining > 0:
            batch_n = min(len(gpus) * 3, remaining)
            batch = []
            for _ in range(batch_n):
                a = np.random.dirichlet(np.ones(n)) * rng.uniform(0.5, 1.5)
                tries = 0
                while any(np.linalg.norm(a - b) < 0.05 for b in seen) and tries < 10:
                    a = np.random.dirichlet(np.ones(n)) * rng.uniform(0.5, 1.5)
                    tries += 1
                seen.append(a)
                batch.append(clip(a))
            sc = full_eval_batch(batch, "rand")
            for a, s in zip(batch, sc):
                archive[0].append(a); archive[1].append(s)
            if max(sc) > g_score:
                g_idx = int(np.argmax(archive[1]))
                g_alpha, g_score = archive[0][g_idx], archive[1][g_idx]
                log(f"random new best: {g_score:.3f}")
            remaining -= batch_n
            log(f"random: {args.budget - remaining}/{args.budget} best={g_score:.3f}")
    elif args.method == "cma":
        import cma
        remaining = args.budget - full_evals
        es = cma.CMAEvolutionStrategy(
            g_alpha.astype(float), 0.3,
            {"popsize": 8, "bounds": [[args.lb] * n, [args.ub] * n],
             "seed": args.seed, "verbose": -9})
        gen = 0
        while remaining >= 8:
            X = np.array(es.ask())
            sc = full_eval_batch(list(X), f"cma{gen}")
            es.tell(list(X), [-s for s in sc])   # pycma 最小化
            for a, s in zip(X, sc):
                archive[0].append(a); archive[1].append(s)
            if max(sc) > g_score:
                g_idx = int(np.argmax(archive[1]))
                g_alpha, g_score = archive[0][g_idx], archive[1][g_idx]
                log(f"cma gen{gen} new best: {g_score:.3f}")
            remaining -= 8
            gen += 1
            log(f"cma gen{gen}: best={g_score:.3f} sigma={es.sigma:.3f} ({full_evals} evals)")
    elif args.method == "soup":
        remaining = args.budget - full_evals
        chosen = np.zeros(n)
        chosen[int(np.argmax(np.abs(g_alpha)))] = 1.0     # 起点: 最优单专家
        cur_score = starting_best
        improved = True
        rnd = 0
        while improved and remaining >= 4:
            improved = False
            rnd += 1
            cands, metas = [], []
            for i in range(n):
                if chosen[i] > 0.5:
                    continue
                for w in (0.3, 0.6):
                    cands.append(clip(chosen + w * np.eye(n)[i]))
                    metas.append((i, w))
            sc = full_eval_batch(cands, f"soup{rnd}")[: len(cands)]
            full_evals += len(cands) - min(len(gpus) * 3, len(cands)) if False else 0
            remaining -= len(cands)
            order = np.argsort(sc)[::-1]
            for j in order:
                i, w = metas[j]
                if sc[j] > cur_score:
                    chosen = cands[j]
                    cur_score = sc[j]
                    archive[0].append(cands[j]); archive[1].append(sc[j])
                    improved = True
                    log(f"soup round{rnd}: +expert{i}@{w} -> {cur_score:.3f}")
                    break
            if full_evals > args.budget + 20:
                break
        g_alpha, g_score = chosen, cur_score

    log(f"search ended. {now_str()} full={full_evals} coarse={coarse_evals} "
        f"equiv={full_evals + 0.25 * coarse_evals:.0f}")
    pool.materialize(np.asarray(g_alpha), os.path.join(run_dir, "global_best"))
    test_score = evaluate_test(os.path.join(run_dir, "global_best"), args.eval_type,
                               args.dataset, gpus[0], args.base_model)
    log(f"final metrics: starting_best_dev={starting_best:.3f} ending_best_dev={float(g_score):.3f} "
        f"ending_best_test={test_score:.3f} full_evals={full_evals} coarse_evals={coarse_evals}")
    if wb is not None:
        wb.log({"final_dev": float(g_score), "final_test": test_score})
        wb.finish()
    with open(os.path.join(run_dir, "baseline_scratchpad.json"), "w") as f:
        json.dump({"g": float(g_score), "g_alpha": list(map(float, np.asarray(g_alpha))),
                   "full_evals": full_evals}, f, indent=1)
    log(f"the end of search... {now_str()}")


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
