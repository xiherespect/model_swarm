"""search_dualres.py — 双分辨率 RACER: 同时在 20D 全空间 + 稀疏支撑子空间搜索.

核心创新: 不是"先试一个再换"(AMPHI)也不是"交替执行"(SUMO), 而是两个种群
并行运行 + 动态预算分配 + 信息互通.

种群 E (Explorer): RACER 的三路引擎, 在 20D 平坦空间搜新支撑区域
种群 R (Refiner):  探针步进器, 在已知好支撑的 ~5D 子空间精调权重
桥梁: E 发现好解 → 提取支撑 → 传给 R; R 精修结果写回共享档案
预算: 按最近进步动态分配 (哪边最近有突破就多给, bandit 式)
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
from safetensors import safe_open

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


def active_set(a, thr=0.05):
    return frozenset(i for i in range(len(a)) if abs(a[i]) > thr)


def _expert_cosine(expert_dir, names):
    vecs = {}
    for nm in names:
        p = os.path.join(expert_dir, nm, "adapter_model.safetensors")
        try:
            with safe_open(p, framework="numpy") as f:
                ks = sorted(f.keys())[:50]
                v = np.concatenate([np.ravel(f.get_tensor(k)) for k in ks])
            vecs[nm] = v / (np.linalg.norm(v) + 1e-9)
        except Exception:
            vecs[nm] = None
    n = len(names)
    sim = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            if vecs[names[i]] is not None and vecs[names[j]] is not None:
                sim[i, j] = sim[j, i] = float(vecs[names[i]] @ vecs[names[j]])
    return sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--name", required=True)
    ap.add_argument("-e", "--eval_type", required=True)
    ap.add_argument("-d", "--dataset", required=True)
    ap.add_argument("-g", "--gpus", default="0,1,2,3")
    ap.add_argument("-b", "--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("-i", "--expert_dir", default="./initial_experts")
    ap.add_argument("--max_iter", type=int, default=25)
    ap.add_argument("--coarse_size", type=int, default=50)
    ap.add_argument("--budget", type=int, default=110)
    ap.add_argument("--k_lo", type=int, default=4)
    ap.add_argument("--k_hi", type=int, default=7)
    ap.add_argument("--sum_lo", type=float, default=1.2)
    ap.add_argument("--sum_hi", type=float, default=1.9)
    ap.add_argument("--lb", type=float, default=-0.5)
    ap.add_argument("--ub", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--elite_percent", type=float, default=0.25)
    ap.add_argument("--shade_p", type=float, default=0.11)
    ap.add_argument("--de_F", type=float, default=0.5)
    ap.add_argument("--de_CR", type=float, default=0.9)
    ap.add_argument("--project_name_wb", default="keg218-swarm-qwen2.5b")
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--fake_eval", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)

    run_dir = os.path.join("search", args.name + "_" + socket.gethostname())
    os.makedirs(os.path.join(run_dir, "tmp"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "global_best"), exist_ok=True)
    logging.basicConfig(filename=os.path.join(run_dir, "log.txt"), level=logging.INFO,
                        format="%(asctime)s INFO:root:%(message)s", datefmt="%H:%M:%S")
    log(f"DUAL-RES initialized. {now_str()}")
    log(f"args: {vars(args)}")

    wb = None
    if wandb is not None and not args.no_wandb:
        try:
            wb = wandb.init(project=args.project_name_wb, name=f"dual_{args.name}",
                            config=vars(args))
        except Exception:
            pass

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
    coarse_n = (args.coarse_size * 2) if args.eval_type == "multitask" else args.coarse_size

    def equiv():
        return full_evals + 0.25 * coarse_evals

    def materialize(alpha, tag):
        return pool.materialize(alpha, os.path.join(run_dir, "tmp", tag))

    def clip(a):
        return np.clip(np.asarray(a, dtype=np.float64), args.lb, args.ub)

    # 互补性预计算
    sim = _expert_cosine(args.expert_dir, pool.names)

    # ---- 公共初始化 ----
    init_alphas = [np.asarray(a) for a in sample_initial_alphas(n, 20, seed=args.seed)]
    init_scores = eval_batch([materialize(a, f"init_{k}") for k, a in enumerate(init_alphas)],
                             args.eval_type, args.dataset, gpus, args.base_model)
    full_evals += len(init_scores)
    archive_a, archive_f = list(init_alphas), list(init_scores)
    gi = int(np.argmax(init_scores))
    g_alpha, g_score = init_alphas[gi], init_scores[gi]
    starting_best = g_score
    pool.materialize(g_alpha, os.path.join(run_dir, "global_best"))
    log(f"init done, starting best {g_score:.3f}")

    # ---- 种群 E (Explorer): RACER 引擎在 20D ----
    sigma_E = 1.0
    evo_path = np.zeros(n)
    M_F = [args.de_F] * 5
    M_CR = [args.de_CR] * 5
    mem_ptr = 0

    # ---- 种群 R (Refiner): 支撑登记表 + 探针步进器 ----
    registry = {}
    for a, s in zip(archive_a, archive_f):
        S = active_set(a)
        if len(S) >= 2 and (S not in registry or s > registry[S]["score"]):
            registry[S] = {"w": np.array([a[i] for i in sorted(S)]), "score": float(s),
                           "d": None, "sigma": 0.15, "csum": 0}

    # ---- 预算分配器状态 ----
    alpha_E = 0.7   # 初始: 70% 预算给探索 (先找区域)
    last_E_improve, last_R_improve = 0, 0

    # ---- 主循环 ----
    for it in range(1, args.max_iter + 1):
        if equiv() > args.budget:
            log(f"budget hit at iter {it}")
            break
        log(f"--- iter {it}! α_E={alpha_E:.2f} E_full={full_evals} c={coarse_evals} eq={equiv():.0f} ---")

        E_idx = np.argsort(archive_f)[::-1][:max(3, int(args.elite_percent * len(archive_a)))]
        E = [archive_a[i] for i in E_idx]
        Ef = [archive_f[i] for i in E_idx]

        # ============ 种群 E: 探索 (RACER 三路引擎) ============
        if rng.random() < alpha_E:
            wts = np.linspace(1.0, 0.3, len(E))
            wts /= wts.sum()
            m_es = np.average(np.stack(E), axis=0, weights=wts)
            sd_es = np.clip(np.std(np.stack(E), axis=0), 0.02, 0.5)
            proposals = []
            for q in range(4):
                r = rng.random()
                if r < 0.4:  # sep-CMA-lite
                    a = m_es + sigma_E * sd_es * np.random.normal(0, 1, n) + 0.15 * evo_path
                elif r < 0.8:  # SHADE-DE
                    F = float(np.clip(np.random.standard_cauchy() * 0.1 + M_F[rng.randrange(5)], 1e-3, 1))
                    CR = float(np.clip(np.random.normal(M_CR[rng.randrange(5)], 0.1), 0, 1))
                    pbn = max(2, int(args.shade_p * len(archive_a)))
                    pb = archive_a[np.argsort(archive_f)[::-1][:pbn][rng.randrange(pbn)]]
                    r1 = archive_a[rng.randrange(len(archive_a))]
                    r2 = archive_a[rng.randrange(len(archive_a))]
                    cur = archive_a[rng.randrange(len(archive_a))]
                    v = cur + F * (pb - cur) + F * (r1 - r2)
                    cross = np.random.random(n) < CR
                    cross[rng.randrange(n)] = True
                    a = np.where(cross, v, cur)
                else:  # sum 定向
                    a_e = E[rng.randrange(len(E))]
                    s_t = rng.uniform(args.sum_lo, args.sum_hi)
                    a = a_e * (s_t / max(a_e.sum(), 0.3)) + np.random.uniform(-0.05, 0.05, n)
                proposals.append(clip(a))

            dirs = [materialize(a, f"E{it}_{q}") for q, a in enumerate(proposals)]
            cs = eval_batch(dirs, args.eval_type, args.dataset, gpus, args.base_model,
                            subset_size=coarse_n)
            coarse_evals += len(cs)
            order = np.argsort(cs)[::-1]
            # X2: 粗分接近者加测
            near = [i for i in range(len(cs)) if cs[order[0]] - cs[i] <= 0.03]
            if len(near) > 1:
                r2 = eval_batch([dirs[i] for i in near], args.eval_type, args.dataset,
                                gpus, args.base_model, subset_size=coarse_n * 2)
                coarse_evals += 2 * len(r2)
                for i, s2 in zip(near, r2):
                    cs[i] = s2
            order = np.argsort(cs)[::-1]
            promoted_E = []
            for j in order[:2]:
                s = evaluate(dirs[j], args.eval_type, args.dataset, gpus[0], args.base_model)
                full_evals += 1
                if s > g_score:
                    s2 = evaluate(dirs[j], args.eval_type, args.dataset, gpus[1], args.base_model)
                    full_evals += 1
                    s = (s + s2) / 2.0
                archive_a.append(proposals[j]); archive_f.append(s)
                promoted_E.append((proposals[j], s))
                if s > g_score:
                    g_alpha, g_score = proposals[j], s
                    old_g = evo_path  # keep reference for path update
                    pool.materialize(g_alpha, os.path.join(run_dir, "global_best"))
                    log(f"E new best: {g_score:.3f}")
                    last_E_improve = it
                    # 桥梁: 发现好解 → 提取支撑传给 R
                    S_new = active_set(np.asarray(g_alpha))
                    if len(S_new) >= 2:
                        w_new = np.array([g_alpha[i] for i in sorted(S_new)])
                        if S_new not in registry:
                            registry[S_new] = {"w": w_new, "score": s, "d": None,
                                               "sigma": 0.15, "csum": 0}
                            log(f"  bridge: new support {sorted(S_new)} -> R")
            if promoted_E and max(s for _, s in promoted_E) > g_score - 0.005:
                sigma_E = min(sigma_E * 1.1, 3.0)
            else:
                sigma_E = max(sigma_E * 0.85, 0.15)
            log(f"E: coarse={['%.2f' % c for c in cs[:2]]} best_g={g_score:.3f}")

        # ============ 种群 R: 精修 (探针步进器) ============
        if rng.random() < (1 - alpha_E) and registry:
            S = max((s for s in registry if registry[s]["score"] > 0),
                    key=lambda s: registry[s]["score"], default=None)
            if S is None:
                continue
            rec = registry[S]
            k = len(S)
            if rec["d"] is None:
                rec["d"] = np.random.randn(k)
                rec["d"] /= (np.linalg.norm(rec["d"]) + 1e-9)

            def embed(S_, w_):
                a = np.zeros(n)
                for pos, i_ in enumerate(sorted(S_)):
                    a[i_] = w_[pos]
                return clip(a)

            W = np.asarray(rec["w"], dtype=float)
            A_probe = [embed(S, W + rec["sigma"] * rec["d"]),
                       embed(S, W - rec["sigma"] * rec["d"])]
            d_probe = [materialize(a, f"R{it}_{q}") for q, a in enumerate(A_probe)]
            c_probe = eval_batch(d_probe, args.eval_type, args.dataset, gpus, args.base_model,
                                 subset_size=coarse_n)
            coarse_evals += len(c_probe)
            c_cur = eval_batch([materialize(embed(S, W), f"R{it}_cur")],
                               args.eval_type, args.dataset, gpus, args.base_model,
                               subset_size=coarse_n)[0]
            coarse_evals += 1

            j = int(np.argmax(c_probe))
            improved = False
            if c_probe[j] > c_cur + 1e-4:
                step = (1 if j == 0 else -1) * rec["sigma"] * rec["d"]
                W_new = np.clip(W + step, args.lb, args.ub)
                rec["d"] = 0.7 * rec["d"] + 0.3 * step / (np.linalg.norm(step) + 1e-9)
                rec["d"] /= (np.linalg.norm(rec["d"]) + 1e-9)
                rec["sigma"] = min(rec["sigma"] * 1.15, 0.4)
                improved = True
            else:
                rec["sigma"] = max(rec["sigma"] * 0.8, 0.03)
                if rng.random() < 0.3:
                    rec["d"] = np.random.randn(k)
                    rec["d"] /= (np.linalg.norm(rec["d"]) + 1e-9)
                W_new = W
            rec["csum"] += 1
            if rec["csum"] % 3 == 0:
                ssum = float(W_new.sum())
                if ssum > 0.3 and (ssum < args.sum_lo or ssum > args.sum_hi + 0.1):
                    W_new = np.clip(W_new * (rng.uniform(args.sum_lo, args.sum_hi) / ssum),
                                    args.lb, args.ub)

            if improved:
                cand = embed(S, W_new)
                s = evaluate(materialize(cand, f"R{it}_opt"), args.eval_type, args.dataset,
                             gpus[0], args.base_model)
                full_evals += 1
                if s > g_score:
                    s2 = evaluate(materialize(cand, f"R{it}_opt"), args.eval_type, args.dataset,
                                  gpus[1], args.base_model)
                    full_evals += 1
                    s = (s + s2) / 2.0
                archive_a.append(cand); archive_f.append(s)
                if s > rec["score"]:
                    rec["score"], rec["w"] = s, np.array(W_new)
                if s > g_score:
                    g_alpha, g_score = cand, s
                    pool.materialize(g_alpha, os.path.join(run_dir, "global_best"))
                    log(f"R new best: {g_score:.3f} (support={sorted(S)})")
                    last_R_improve = it
            log(f"R: support={sorted(S)} best_in={rec['score']:.3f} σ={rec['sigma']:.2f}")

        # ---- 动态预算分配 v2: 自适应集中 ----
        # R 无贡献 → α_E 可升至 1.0(退化为纯 RACER, 治预算稀释)
        # R 有贡献 → α_E 保底 0.5(E 必须持续探索喂 R)
        if it % 3 == 0:
            if last_R_improve > last_E_improve:
                alpha_E = max(0.5, alpha_E - 0.1)   # R 有突破 → 多给 R
            elif last_E_improve > last_R_improve:
                alpha_E = min(1.0, alpha_E + 0.15)  # E 有突破 → 多给 E(可到 1.0)
            # 长期无 R 贡献 → 直接集中
            if it - last_R_improve >= 9:
                alpha_E = 1.0
            log(f"budget: α_E → {alpha_E:.1f} (E_last={last_E_improve}, R_last={last_R_improve})")

    # ---- 终选复评 ----
    top_idx = np.argsort(archive_f)[::-1][:3]
    recheck = []
    for k2, i in enumerate(top_idx):
        d2 = materialize(archive_a[i], f"recheck_{k2}")
        s2 = evaluate(d2, args.eval_type, args.dataset, gpus[k2 % len(gpus)], args.base_model)
        full_evals += 1
        recheck.append((i, float(np.mean([archive_f[i], s2]))))
    pick = max(recheck, key=lambda t: t[1])[0]
    final_a, final_s = np.asarray(archive_a[pick]), archive_f[pick]
    log(f"FINAL RECHECK: {[f'{x[1]:.3f}' for x in recheck]} -> pick {final_s:.3f}")
    pool.materialize(final_a, os.path.join(run_dir, "global_best"))

    test_score = evaluate_test(os.path.join(run_dir, "global_best"), args.eval_type,
                               args.dataset, gpus[0], args.base_model)
    log(f"final metrics: starting_best_dev={starting_best:.3f} ending_best_dev={float(final_s):.3f} "
        f"ending_best_test={test_score:.3f} full_evals={full_evals} coarse_evals={coarse_evals} "
        f"equiv_total={equiv():.0f} supports={len(registry)}")
    if wb is not None:
        wb.log({"final_dev": float(final_s), "final_test": test_score})
        wb.finish()
    log(f"the end. {now_str()}")


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
