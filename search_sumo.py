"""search_sumo.py — SUMO: Sparse-sUpspace Mining and Optimization（脊空间双层搜索）.

问题表述的转换（15 冠军解定律：k=4-7 活跃成分, sum∈[1.3,1.8]）:
  旧表述(PSO/CMA/DE/RACER/LoRAHub): 平铺搜 20 维连续空间
  SUMO: 拆成两层 —— 上层"选哪 4-7 个专家"(离散支撑空间, 竞速分配)
                       下层"选中的 ~5 维上配权重"(探针步进器, 低维无需协方差学习)

上层支撑生成器(三个, 全部吃独家数据):
  ① 突变: 精英支撑 ±加/减/换一个专家
  ② 共现: 从 archive 高分解的活跃集学专家对共现, 序贯采样
  ③ 互补: LoRA 权重余弦相似度一次性预计算, 优先互补对(低相似)
下层: 自研"探针步进器"(Probe-Stepper) —— 双向探针+动量方向记忆+sum锚定,
     零外部优化器(5-7维无需协方差学习, 这是双层表述的回报)
噪声协议继承: 竞速漏斗(粗评→全评) + 冠军复评 + 终选复评
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
    """一次性预计算专家间 LoRA 权重余弦相似度(互补性生成器用)."""
    vecs = {}
    for nm in names:
        p = os.path.join(expert_dir, nm, "adapter_model.safetensors")
        try:
            with safe_open(p, framework="numpy") as f:  # type: ignore
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
    ap.add_argument("--budget", type=int, default=110, help="等效预算上限")
    ap.add_argument("--k_lo", type=int, default=4)
    ap.add_argument("--k_hi", type=int, default=7)
    ap.add_argument("--sum_lo", type=float, default=1.2)
    ap.add_argument("--sum_hi", type=float, default=1.9)
    ap.add_argument("--lb", type=float, default=-0.5)
    ap.add_argument("--ub", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_cooccur", action="store_true", help="消融: 关共现生成器")
    ap.add_argument("--no_complement", action="store_true", help="消融: 关互补生成器")
    ap.add_argument("--no_pair_seed", action="store_true",
                    help="v0.2 消融: 关配对种子引导(45对中点粗扫+top6全评注入登记表)")
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
    log(f"SUMO search initialized. {now_str()}")
    log(f"args: {vars(args)}")

    wb = None
    if wandb is not None and not args.no_wandb:
        try:
            wb = wandb.init(project=args.project_name_wb, name=f"sumo_{args.name}", config=vars(args))
        except Exception as e:
            log(f"wandb init failed ({e})")

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

    def full_eval_batch(alphas, tag):
        nonlocal full_evals
        dirs = [materialize(a, f"{tag}_{k}") for k, a in enumerate(alphas)]
        sc = eval_batch(dirs, args.eval_type, args.dataset, gpus, args.base_model)
        full_evals += len(sc)
        return sc, dirs

    # ---- 互补性预计算 ----
    t0 = time.time()
    sim = None if args.no_complement else _expert_cosine(args.expert_dir, pool.names)
    if sim is not None:
        log(f"expert cosine sim computed in {time.time()-t0:.1f}s "
            f"(mean_offdiag={float((sim.sum()-n)/(n*n-1)):.3f})")

    # ---- 公共 mix 初始化（与 RACER 一致, 供起点/共现学习）----
    init_alphas = [np.asarray(a) for a in sample_initial_alphas(n, 20, seed=args.seed)]
    init_scores = eval_batch([materialize(a, f"init_{k}") for k, a in enumerate(init_alphas)],
                             args.eval_type, args.dataset, gpus, args.base_model)
    full_evals += len(init_scores)
    archive_a, archive_f = list(init_alphas), list(init_scores)
    gi = int(np.argmax(init_scores))
    g_alpha, g_score = init_alphas[gi], init_scores[gi]
    starting_best = g_score
    log(f"init done, starting best {g_score:.3f}")

    # ---- 支撑登记表: {frozenset: {"w","score","d","sigma","csum"}} ----
    #    d/sigma = 探针步进器状态(动量方向+步长); csum = 步数计数(sum锚定用)
    registry = {}
    for a, s in zip(archive_a, archive_f):
        S = active_set(a)
        if len(S) >= 2 and (S not in registry or s > registry[S]["score"]):
            k = len(S)
            registry[S] = {"w": np.array([a[i] for i in sorted(S)]), "score": float(s),
                           "d": None, "sigma": 0.15, "csum": 0}

    def learn_cooccur():
        C = np.ones((n, n)) * 0.5   # 平滑先验
        idx = np.argsort(archive_f)[::-1][: max(6, len(archive_f) // 3)]
        for t, i in enumerate(idx):
            a = np.asarray(archive_a[i])
            S = sorted(active_set(a))
            wgt = archive_f[i] - min(archive_f) + 1e-3
            for x in S:
                for y in S:
                    if x != y:
                        C[x, y] += wgt
        return C

    def single_scores():
        s = np.zeros(n)
        for a, f in zip(archive_a, archive_f):
            S = active_set(a)
            if len(S) == 1:
                s[list(S)[0]] = f
        return np.clip(s, 0, None) + 1e-3   # AbstainQA 等负值效用域: 截负保概率合法

    def gen_support(cooccur, s_single, elite_S):
        """三合一支撑生成: 突变 / 共现序贯 / 互补加权."""
        k_target = rng.randint(args.k_lo, args.k_hi)
        mode = rng.random()
        S = set(elite_S if (mode < 0.4 and elite_S) else [])
        if mode < 0.4 and S:                       # ① 突变
            op = rng.random()
            if op < 0.4:
                cands = [i for i in range(n) if i not in S]
                S.add(rng.choice(cands))
            elif op < 0.7 and len(S) > args.k_lo:
                S.remove(rng.choice(sorted(S)))
            else:
                S.remove(rng.choice(sorted(S)))
                cands = [i for i in range(n) if i not in S]
                S.add(rng.choice(cands))
        else:                                       # ②+③ 共现/互补 序贯采样
            w0 = s_single.copy()
            first = int(np.random.choice(n, p=w0 / w0.sum()))
            S = {first}
            while len(S) < k_target:
                w = np.full(n, 1e-3)
                for j in range(n):
                    if j in S:
                        continue
                    attr = 1.0
                    if not args.no_cooccur:
                        attr *= np.mean([cooccur[j, x] for x in S])
                    if sim is not None:            # 互补: 相似度低 → 吸引力高
                        attr *= (1.2 - np.mean([sim[j, x] for x in S]))
                    attr *= s_single[j]
                    w[j] = attr
                S.add(int(np.random.choice(n, p=w / w.sum())))
        return frozenset(S)

    def embed(S, w_sub):
        a = np.zeros(n)
        w_sub = np.asarray(w_sub, dtype=float)
        for pos, i in enumerate(sorted(S)):
            a[i] = w_sub[pos]
        return clip(a)

    def init_weights(S):
        base = np.array([rng.uniform(0.25, 0.55) for _ in S])
        s_target = rng.uniform(args.sum_lo, args.sum_hi)
        w = base * (s_target / base.sum())
        return np.clip(w, args.lb, args.ub)

    # ---- v0.2: 配对种子引导（治 nlgraph 冷启动死锁）----
    # 45 个专家对中点全粗扫(11.25 equiv) + top-6 全评注入——共现学习器第一轮就有真实素材
    if not args.no_pair_seed:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        pm_dirs = [materialize(embed({i, j}, [0.5, 0.5]), f"pair_{k}") for k, (i, j) in enumerate(pairs)]
        pm_c = eval_batch(pm_dirs, args.eval_type, args.dataset, gpus, args.base_model,
                          subset_size=coarse_n)
        coarse_evals += len(pm_c)
        top6 = np.argsort(pm_c)[::-1][:6]
        for t in top6:
            i, j = pairs[t]
            s = evaluate(pm_dirs[t], args.eval_type, args.dataset, gpus[0], args.base_model)
            full_evals += 1
            a = embed({i, j}, [0.5, 0.5])
            archive_a.append(a)
            archive_f.append(s)
            S = frozenset({i, j})
            registry[S] = {"w": np.array([0.5, 0.5]), "score": float(s),
                           "d": None, "sigma": 0.15, "csum": 0}
            if s > g_score:
                g_alpha, g_score = a, s
                pool.materialize(g_alpha, os.path.join(run_dir, "global_best"))
        log(f"pair-seed: top6={[(pairs[t][0], pairs[t][1], round(pm_c[t], 3)) for t in top6]}, "
            f"best_full={max(registry[S]['score'] for S in registry if len(S) == 2):.3f}")

    # ---- 主循环: 奇数轮探索新支撑(2个), 偶数轮精修最优支撑 ----
    history = [g_score]
    for it in range(1, args.max_iter + 1):
        if equiv() > args.budget:
            log(f"budget hit at iter {it}, stop")
            break
        log(f"--- iteration {it}! {now_str()} ---")
        cooccur = learn_cooccur()
        s_single = single_scores()
        elite_S = max(registry, key=lambda S: registry[S]["score"])

        if it % 2 == 1:   # ---- 探索: 2 个新支撑 ----
            new_Ss = []
            for q in range(2):
                S = gen_support(cooccur, s_single, elite_S)
                tries = 0
                while S in registry and tries < 6:
                    S = gen_support(cooccur, s_single, elite_S)
                    tries += 1
                if S in registry:
                    continue
                w0 = init_weights(S)
                a0 = embed(S, w0)
                d0 = materialize(a0, f"exp{it}_{q}")
                c0 = eval_batch([d0], args.eval_type, args.dataset, gpus, args.base_model,
                                subset_size=coarse_n)[0]
                coarse_evals += 1
                registry[S] = {"w": w0, "score": -1.0, "d": None, "sigma": 0.15, "csum": 0}
                if c0 > float(np.median(archive_f)) - 0.02:   # 竞速: 有戏才全评
                    s0 = evaluate(d0, args.eval_type, args.dataset, gpus[0], args.base_model)
                    full_evals += 1
                    registry[S]["score"] = s0
                    archive_a.append(a0); archive_f.append(s0)
                    if s0 > g_score:
                        s2 = evaluate(d0, args.eval_type, args.dataset, gpus[1], args.base_model)
                        full_evals += 1
                        s0 = (s0 + s2) / 2.0
                        archive_f[-1] = s0
                        registry[S]["score"] = s0
                    if s0 > g_score:
                        g_alpha, g_score = a0, s0
                        pool.materialize(g_alpha, os.path.join(run_dir, "global_best"))
                        log(f"new best {g_score:.3f} support={sorted(S)}")
                    new_Ss.append(S)
                else:
                    registry[S]["score"] = 0.5 * c0   # 粗分记账, 未见全评
            log(f"explore: {[sorted(s) for s in new_Ss]}")
        else:             # ---- 精修: 最优支撑上的探针步进器(自研, 零外部优化器) ----
            S = max((s for s in registry if registry[s]["score"] > 0),
                    key=lambda s: registry[s]["score"], default=None)
            if S is None:
                continue
            rec = registry[S]
            k = len(S)
            if rec["d"] is None:
                rec["d"] = np.random.randn(k)
                rec["d"] /= (np.linalg.norm(rec["d"]) + 1e-9)
            # ① 双向探针: w±σd
            W = np.asarray(rec["w"], dtype=float)
            A = [embed(S, W + rec["sigma"] * rec["d"]),
                 embed(S, W - rec["sigma"] * rec["d"])]
            dirs = [materialize(a, f"probe{it}_{q}") for q, a in enumerate(A)]
            cs = eval_batch(dirs, args.eval_type, args.dataset, gpus, args.base_model,
                            subset_size=coarse_n)
            coarse_evals += len(cs)
            cur_c = eval_batch([materialize(embed(S, W), f"cur{it}")],
                               args.eval_type, args.dataset, gpus, args.base_model,
                               subset_size=coarse_n)[0]
            coarse_evals += 1
            j = int(np.argmax(cs))
            if cs[j] > cur_c + 1e-4:                             # ② 采纳好侧+动量更新
                step = (1 if j == 0 else -1) * rec["sigma"] * rec["d"]
                W_new = np.clip(W + step, args.lb, args.ub)
                rec["d"] = 0.7 * rec["d"] + 0.3 * step / (np.linalg.norm(step) + 1e-9)
                rec["d"] /= (np.linalg.norm(rec["d"]) + 1e-9)
                rec["sigma"] = min(rec["sigma"] * 1.15, 0.4)
                improved = True
            else:                                                # ③ 收缩/换向
                rec["sigma"] = max(rec["sigma"] * 0.8, 0.03)
                if rng.random() < 0.3:
                    rec["d"] = np.random.randn(k)
                    rec["d"] /= (np.linalg.norm(rec["d"]) + 1e-9)
                W_new, improved = W, False
            rec["csum"] += 1
            if rec["csum"] % 3 == 0:                             # ④ sum 锚定回赢家带
                ssum = float(W_new.sum())
                if ssum > 0.3 and (ssum < args.sum_lo or ssum > args.sum_hi + 0.1):
                    W_new = np.clip(W_new * (rng.uniform(args.sum_lo, args.sum_hi) / ssum),
                                    args.lb, args.ub)
            # ⑤ 破纪录触发全评+复评
            cand = embed(S, W_new)
            if improved or cs[j] > cur_c:
                s = evaluate(materialize(cand, f"opt{it}"), args.eval_type, args.dataset,
                             gpus[0], args.base_model)
                full_evals += 1
                if s > g_score:
                    s2 = evaluate(materialize(cand, f"opt{it}"), args.eval_type, args.dataset,
                                  gpus[1], args.base_model)
                    full_evals += 1
                    s = (s + s2) / 2.0
                archive_a.append(cand); archive_f.append(s)
                if s > rec["score"]:
                    rec["score"], rec["w"] = s, np.array(W_new)
                if s > g_score:
                    g_alpha, g_score = cand, s
                    pool.materialize(g_alpha, os.path.join(run_dir, "global_best"))
                    log(f"new best {g_score:.3f} (probe-stepper, support={sorted(S)})")
                if s > g_score:
                    g_alpha, g_score = A[j], s
                    pool.materialize(g_alpha, os.path.join(run_dir, "global_best"))
                    log(f"new best {g_score:.3f} (subspace opt, support={sorted(S)})")
            log(f"exploit support={sorted(S)}: coarse={['%.3f' % c for c in cs]} "
                f"best_in_support={rec['score']:.3f}")

        history.append(g_score)
        log(f"iter {it}: g={g_score:.3f} supports={len(registry)} full={full_evals} "
            f"coarse={coarse_evals} equiv={equiv():.0f}")
        # 逐轮落盘(AMPHI try-and-switch 轮询依赖)
        with open(os.path.join(run_dir, "sumo_scratchpad.json"), "w") as f:
            json.dump({"iter": it, "g": float(g_score), "g_alpha": list(map(float, g_alpha)),
                       "full_evals": full_evals, "coarse_evals": coarse_evals}, f)
        if wb is not None:
            wb.log({"g": g_score, "iter": it, "equiv": equiv()})

    # ---- 终选复评(继承 B'): top-3 各加测一次(换卡), 按均值重排 ----
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
        f"equiv_total={equiv():.0f} supports_tried={len(registry)}")
    if wb is not None:
        wb.log({"final_dev": float(final_s), "final_test": test_score})
        wb.finish()
    with open(os.path.join(run_dir, "sumo_scratchpad.json"), "w") as f:
        json.dump({"g": float(final_s), "g_alpha": list(map(float, final_a)),
                   "supports": {",".join(map(str, sorted(s))): r["score"] for s, r in registry.items()},
                   "full_evals": full_evals, "coarse_evals": coarse_evals}, f, indent=1)
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
