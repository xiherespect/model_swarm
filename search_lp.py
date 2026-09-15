"""search_lp.py — LP-Merge：在 α（合并系数）空间搜索最优专家合并。

实现 doc/lp_algorithm_design.md §3.2：
  四种生成器（记忆重组+定向外插 / 协同生成 / 稀疏重置 / 随机外插）
  + rank 截断选择 + 自适应步长 + 55 参数二次代理 + 多保真评测。

复用：evaluate.py（subset_size 多保真）、alpha_space.py（α 物化/反解）。
用法示例：
  python search_lp.py -n lp_hellaswag_smoke -e multiple_choice -d hellaswag \
      -g 0,1,2,3 -b Qwen/Qwen2.5-1.5B-Instruct --popsize 12 --max_iter 15
"""
import argparse
import datetime
import json
import logging
import math
import os
import random
import socket
import time

import numpy as np
import multiprocessing as mp

from alpha_space import ExpertPool, sample_initial_alphas
from evaluate import evaluate, evaluate_test

try:
    import wandb
except ImportError:
    wandb = None


# ---------------- 工具 ----------------
def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    logging.info(msg)
    logging.getLogger().handlers[0].flush()


class QuadSurrogate:
    """f(α) ≈ Σβᵢαᵢ + Σβᵢⱼαᵢαⱼ + b；n=10 时 56 维。岭回归。"""

    def __init__(self, n, ridge=1e-3):
        self.n = n
        self.pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        self.w = None
        self.ridge = ridge
        self.r2 = None

    def features(self, alphas):
        a = np.asarray(alphas, dtype=np.float64)
        cols = [a]
        for i, j in self.pairs:
            cols.append((a[:, i] * a[:, j])[:, None])
        cols.append(np.ones((a.shape[0], 1)))
        return np.hstack(cols)

    def fit(self, alphas, scores):
        X = self.features(alphas)
        y = np.asarray(scores, dtype=np.float64)
        G = X.T @ X + self.ridge * np.eye(X.shape[1])
        self.w = np.linalg.solve(G, X.T @ y)
        pred = X @ self.w
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
        self.r2 = 1.0 - ss_res / ss_tot
        return self.r2

    def beta_main(self):
        return self.w[: self.n].copy() if self.w is not None else None

    def synergy(self, alphas_history):
        """[(i,j, 分数)]：βᵢⱼ × 新颖度（历史共现越少越新）。无代理返回 []。"""
        if self.w is None:
            return []
        H = np.abs(np.asarray(alphas_history)) > 0.2
        out = []
        for k, (i, j) in enumerate(self.pairs):
            beta_ij = float(self.w[self.n + k])
            novelty = 1.0 / (1.0 + float(np.sum(H[:, i] & H[:, j])))
            out.append((i, j, beta_ij * novelty, beta_ij))
        out.sort(key=lambda t: -t[2])
        return out

    def contributions(self):
        """每专家贡献度：|βᵢ| + Σⱼ|βᵢⱼ|（无代理返回 None）。"""
        if self.w is None:
            return None
        main = np.abs(self.beta_main())
        inter = np.zeros(self.n)
        for k, (i, j) in enumerate(self.pairs):
            b = abs(float(self.w[self.n + k]))
            inter[i] += b
            inter[j] += b
        return main + inter


# ---------------- 评测 ----------------
def eval_batch(cand_dirs, eval_type, dataset, gpus, base_model, subset_size=None, subset_start=None):
    """并行评测一批已物化的 adapter 目录，返回分数列表。

    用 spawn 而非 fork：父进程持有 torch 张量/大 numpy 状态，fork 会继承
    已损坏的锁状态导致子进程死锁（实测复现），spawn 换全新解释器绕开。
    """
    eval_args = [
        (d, eval_type, dataset, gpus[i % len(gpus)], base_model, False, None, False, subset_size, subset_start)
        for i, d in enumerate(cand_dirs)
    ]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(len(gpus), len(eval_args))) as pool:
        results = pool.starmap(_eval_worker, [(args,) for args in eval_args])
    return results


def _eval_worker(args):
    """spawn 模式下 starmap 的单参数入口（spawn 子进程会重新 import 本模块）。"""
    return evaluate(*args)


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--name", required=True)
    ap.add_argument("-e", "--eval_type", required=True)
    ap.add_argument("-d", "--dataset", required=True)
    ap.add_argument("-g", "--gpus", default="0,1,2,3")
    ap.add_argument("-b", "--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("-i", "--expert_dir", default="./initial_experts")
    ap.add_argument("--popsize", type=int, default=20)          # v4: 12→20（初始=10单专家+10混合）
    ap.add_argument("--max_iter", type=int, default=25)         # v4: 15→25（给重启留空间）
    ap.add_argument("--restart_stagnation", type=int, default=3, help="停滞几轮触发重启（v4: 6→3）")
    ap.add_argument("--gen_ratios", default="0.3,0.3,0.2,0.2",
                    help="四生成器比例: 外插,协同,稀疏,随机（v4: 40/30/20/10→30/30/20/20）")
    ap.add_argument("--coarse_size", type=int, default=50, help="粗筛样本数；0 表示关多保真")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lb", type=float, default=-0.5)
    ap.add_argument("--ub", type=float, default=1.5)
    ap.add_argument("--project_name_wb", default="keg218-swarm-qwen2.5b")
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--skip_test", action="store_true", help="跳过结尾的 test 评测（冒烟用）")
    ap.add_argument("--fake_eval", action="store_true",
                    help="干跑模式：用假分数替换真评测，秒级跑完全部代码路径（改代码后必跑）")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)

    # v4 生成器比例（累积阈值）
    r_ext, r_syn, r_sp, _r = [float(x) for x in args.gen_ratios.split(",")]
    T_SYN, T_SPARSE = r_ext, r_ext + r_syn

    gpus = [int(x) for x in args.gpus.split(",")]
    search_pass_name = args.name + "_" + socket.gethostname()
    run_dir = os.path.join("search", search_pass_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "tmp"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "global_best"), exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(run_dir, "log.txt"), level=logging.INFO,
        format="%(asctime)s INFO:root:%(message)s", datefmt="%H:%M:%S",
    )
    log(f"LP-Merge search initialized. {now_str()}")
    log(f"args: {vars(args)}")

    if args.fake_eval:
        # 干跑模式：替换真评测与物化，秒级遍历所有代码路径（含晚激活分支）
        fake_rng = random.Random(7)

        def _fake_eval_one(*a, **k):
            return fake_rng.uniform(0.30, 0.75)

        def _fake_eval_batch(cand_dirs, *a, **k):
            return [fake_rng.uniform(0.30, 0.75) for _ in cand_dirs]

        def _fake_eval_test(*a, **k):
            return fake_rng.uniform(0.55, 0.70)

        globals()["evaluate"] = _fake_eval_one
        globals()["eval_batch"] = _fake_eval_batch
        globals()["evaluate_test"] = _fake_eval_test
        log("FAKE EVAL MODE: 所有分数为随机数，仅用于代码路径测试")

    wb = None
    if wandb is not None and not args.no_wandb:
        try:
            wb = wandb.init(project=args.project_name_wb, name=f"lp_{args.name}", config=vars(args))
        except Exception as e:  # wandb 不可用不影响搜索
            log(f"wandb init failed ({e}), continue without wandb")

    pool = ExpertPool(args.expert_dir)
    if args.fake_eval:
        pool.materialize = lambda alpha, out_dir: (os.makedirs(out_dir, exist_ok=True), out_dir)[1]
    n = pool.n
    log(f"expert pool: {n} experts: {pool.names}")
    surrogate = QuadSurrogate(n)

    def materialize(alpha, tag):
        return pool.materialize(alpha, os.path.join(run_dir, "tmp", tag))

    # ---- 初始种群：身份向量 + 混合，全部全量评测 ----
    pop_alphas = sample_initial_alphas(n, args.popsize, seed=args.seed)
    log(f"initializing... evaluating {len(pop_alphas)} candidates on full dev {now_str()}")
    cand_dirs = [materialize(a, f"init_{k}") for k, a in enumerate(pop_alphas)]
    t0 = time.time()
    scores = eval_batch(cand_dirs, args.eval_type, args.dataset, gpus, args.base_model)
    log(f"initial eval done in {(time.time()-t0)/60:.1f} min")

    pop = list(zip(pop_alphas, scores))              # [(alpha, full_score)]
    obs_a, obs_f = list(pop_alphas), list(scores)    # 全量观测（喂代理）
    g_alpha, g_score = max(pop, key=lambda t: t[1])
    starting_best = g_score
    log(f"starting best (dev, full): {g_score:.3f}")
    pool.materialize(g_alpha, os.path.join(run_dir, "global_best"))

    step_scale, stagnation = 1.0, 0
    full_evals, coarse_evals = len(pop), 0
    history = [g_score]
    if wb is not None:  # iter-0 上报，与 PSO 的 wandb 曲线起点对齐
        wb.log({"g": g_score, "iter": 0, "step_scale": step_scale,
                "full_evals": full_evals, "coarse_evals": coarse_evals})

    def scratchpad_dump(it):
        payload = {
            "iter": it, "g": g_score, "g_alpha": list(map(float, g_alpha)),
            "history": history, "obs_alphas": [list(map(float, a)) for a in obs_a],
            "obs_scores": list(map(float, obs_f)),
            "full_evals": full_evals, "coarse_evals": coarse_evals,
            "step_scale": step_scale,
            "surrogate_r2": surrogate.r2,
            "surrogate_beta_main": None if surrogate.w is None else list(map(float, surrogate.beta_main())),
        }
        with open(os.path.join(run_dir, "lp_scratchpad.json"), "w") as f:
            json.dump(payload, f, indent=1)

    # ---- 迭代 ----
    for it in range(1, args.max_iter + 1):
        log(f"--- iteration {it}! {now_str()} ---")
        pop.sort(key=lambda t: -t[1])
        keep = args.popsize // 2
        survivors = pop[:keep]
        n_new = args.popsize - keep

        # 代理更新（仅全量观测）
        if len(obs_a) >= 30:
            r2 = surrogate.fit(obs_a, obs_f)
            log(f"surrogate refit on {len(obs_a)} obs, R2={r2:.3f}")

        contrib = surrogate.contributions()
        syn = surrogate.synergy(obs_a)

        # ---- 四种生成器 ----
        new_alphas = []
        for k in range(n_new):
            r = rng.random()
            if r < T_SYN:  # ① 记忆重组 + 向 g 移动（v4: 锚点 parent 化，防 g 坍缩）
                parent = survivors[rng.randrange(len(survivors))][0]
                beta = rng.uniform(0.05, 0.30) * step_scale
                a = parent + beta * (g_alpha - parent)
            elif r < T_SPARSE:  # ② 协同生成
                if syn:
                    i, j = syn[0][0], syn[0][1]
                    a = np.full(n, 0.02)
                    a[i], a[j] = rng.uniform(0.4, 0.9), rng.uniform(0.4, 0.9)
                    syn = syn[1:] + syn[:1]  # 轮换使用 top 协同对
                else:  # 代理未就绪：凸混合两个幸存者
                    p1, p2 = (survivors[rng.randrange(len(survivors))][0] for _ in range(2))
                    t = rng.uniform(0.3, 0.7)
                    a = t * p1 + (1 - t) * p2
            elif r < T_SPARSE + r_sp:  # ③ 稀疏重置
                base = survivors[rng.randrange(len(survivors))][0].copy()
                if contrib is not None:
                    zero_idx = np.argsort(contrib)[: max(1, n // 3)]
                else:
                    zero_idx = rng.sample(range(n), max(1, n // 3))
                base[zero_idx] = 0.0
                a = base + rng.gauss(0, 0.05)
            else:  # ④ 随机外插
                a = np.random.dirichlet(np.ones(n) * 0.6) * rng.uniform(0.5, 1.5)
            new_alphas.append(np.clip(a, args.lb, args.ub))

        # ---- 多保真评测：粗筛 → 晋升 → 冠军复评 ----
        cand_dirs = [materialize(a, f"it{it}_c{k}") for k, a in enumerate(new_alphas)]
        if args.coarse_size and args.coarse_size > 0:
            coarse = eval_batch(cand_dirs, args.eval_type, args.dataset, gpus, args.base_model,
                                subset_size=args.coarse_size)
            coarse_evals += len(coarse)
            order = np.argsort(coarse)[::-1]
            n_promote = max(1, len(new_alphas) // 3)
            promote = [(new_alphas[i], cand_dirs[i], coarse[i]) for i in order[:n_promote]]
            log(f"coarse: {[f'{c:.3f}' for c in coarse]} -> promote {n_promote}")
        else:
            promote = [(new_alphas[i], cand_dirs[i], None) for i in range(len(new_alphas))]

        promoted = []
        for a, d, cs in promote:
            s = evaluate(d, args.eval_type, args.dataset, gpus[0], args.base_model)
            full_evals += 1
            if s > g_score:  # 冠军复评防噪声（两次均值）
                s2 = evaluate(d, args.eval_type, args.dataset, gpus[0], args.base_model)
                full_evals += 1
                s = (s + s2) / 2.0
            obs_a.append(a)
            obs_f.append(s)
            promoted.append((a, s))

        # ---- 选择 + 记忆 ----
        pop = survivors + promoted
        pop.sort(key=lambda t: -t[1])
        pop = pop[: args.popsize]

        best_this = pop[0][1]
        if best_this > g_score + 1e-9:
            g_alpha, g_score = pop[0]
            pool.materialize(g_alpha, os.path.join(run_dir, "global_best"))
            log(f"new global best: {g_score:.3f}")
            step_scale = min(step_scale * 1.1, 3.0)
            stagnation = 0
        else:
            stagnation += 1
            step_scale = max(step_scale * 0.85, 0.15)
            if stagnation >= args.restart_stagnation:  # v4: 停滞重启（默认 3 轮）
                log("stagnation restart: injecting new candidates")
                fresh = []
                for _ in range(args.popsize - args.popsize // 2):
                    if rng.random() < 0.5:
                        fresh.append(np.clip(g_alpha + np.random.normal(0, 0.1, n), args.lb, args.ub))
                    else:
                        fresh.append(np.random.dirichlet(np.ones(n) * 0.6) * rng.uniform(0.5, 1.5))
                cand_dirs = [materialize(a, f"it{it}_r{k}") for k, a in enumerate(fresh)]
                rs = eval_batch(cand_dirs, args.eval_type, args.dataset, gpus, args.base_model)
                full_evals += len(rs)
                for a, s in zip(fresh, rs):
                    obs_a.append(a)
                    obs_f.append(s)
                pop = pop[: args.popsize // 2] + list(zip(fresh, rs))
                pop.sort(key=lambda t: -t[1])
                if pop[0][1] > g_score:
                    g_alpha, g_score = pop[0]
                    pool.materialize(g_alpha, os.path.join(run_dir, "global_best"))
                    log(f"new global best (restart): {g_score:.3f}")
                step_scale, stagnation = 1.0, 0

        history.append(g_score)
        log(f"iter {it}: g={g_score:.3f} step={step_scale:.2f} stagn={stagnation} "
            f"full_evals={full_evals} coarse_evals={coarse_evals}")
        if wb is not None:
            wb.log({"g": g_score, "iter": it, "step_scale": step_scale,
                    "full_evals": full_evals, "coarse_evals": coarse_evals,
                    "surrogate_r2": surrogate.r2 or 0})
        scratchpad_dump(it)

    # ---- 收尾：冠军上 test ----
    if args.skip_test:
        log(f"search ended (skip_test). {now_str()}")
        scratchpad_dump(args.max_iter)
        return
    log(f"search ended. evaluating global best on test... {now_str()}")
    test_score = evaluate_test(os.path.join(run_dir, "global_best"), args.eval_type, args.dataset,
                               gpus[0], args.base_model)
    log(f"final metrics: starting_best_dev={starting_best:.3f} ending_best_dev={g_score:.3f} "
        f"ending_best_test={test_score:.3f} full_evals={full_evals} coarse_evals={coarse_evals}")
    if wb is not None:
        wb.log({"final_test": test_score, "final_dev": g_score})
        wb.finish()
    scratchpad_dump(args.max_iter)
    log(f"the end of search... {now_str()}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # 崩溃现场写进 run 目录的 log.txt（stdout/stderr 常被重定向到 /dev/null）
        import traceback
        if logging.getLogger().handlers:
            logging.error("FATAL:\n" + traceback.format_exc())
            logging.getLogger().handlers[0].flush()
        else:
            traceback.print_exc()
        raise
