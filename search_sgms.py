"""search_sgms.py — SGMS v1：代理引导模型群体搜索（doc/SGMS 方案 + v4 实验修订版）。

相对 SGMS 方案文档的修订（依据 2026-09-05 v3/v4 实测）：
  R1 去掉 Σλ=1 自缚 → box 约束 [-0.5,1.5]（v4 冠军 sum=0.80 在仿射包外）
  R2 ES 提案直接在 box 内做（CMA 不需要约束面参数化）
  R3 吸收 LP 四件套：混合体初始化 / 多保真粗筛 / 协同第四路提案 / 冠军复评

每轮: Q=3k 提案(ES 30% + DE 30% + Dirichlet 20% + 协同 20%)
      → GP 采集预筛 μ+κσ (κ 2.0→0.3) → k 个粗评(50) → 半数全评
      → 精英 archive → 停滞膨胀重启；结尾 LCB(μ−0.5σ) 终选 → test
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

from alpha_space import ExpertPool, sample_initial_alphas, sample_initial_alphas_pso
from evaluate import evaluate, evaluate_test
from search_lp import QuadSurrogate, eval_batch

try:
    import wandb
except ImportError:
    wandb = None

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    SK_OK = True
except ImportError:
    SK_OK = False


def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    logging.info(msg)
    logging.getLogger().handlers[0].flush()


class GPSurrogate:
    """GP 代理：μ(λ), σ(λ)。观测不足时 not ready。"""

    def __init__(self):
        self.gp = None

    def fit(self, alphas, scores):
        X = np.asarray(alphas, dtype=np.float64)
        y = np.asarray(scores, dtype=np.float64)
        if len(y) < 15:
            return False
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0, length_scale_bounds=(0.2, 5.0)) \
            + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-4, 0.2))
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=1)
        self.gp.fit(X, y)
        return True

    def predict(self, X):
        if self.gp is None:
            return None, None
        mu, sd = self.gp.predict(np.asarray(X, dtype=np.float64), return_std=True)
        return mu, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--name", required=True)
    ap.add_argument("-e", "--eval_type", required=True)
    ap.add_argument("-d", "--dataset", required=True)
    ap.add_argument("-g", "--gpus", default="0,1,2,3")
    ap.add_argument("-b", "--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("-i", "--expert_dir", default="./initial_experts")
    ap.add_argument("--init_size", type=int, default=20, help="初始池=10 单专家+混合")
    ap.add_argument("--init_pair_scan", action="store_true",
                    help="v1p: 初始化追加 45 个两两中点粗筛（~11 等效预算），系统覆盖 2-专家脊")
    ap.add_argument("--init_protocol", default="mix", choices=["mix", "pso"],
                    help="初始点协议: mix=v4(单位+Dirichlet混合体) / pso=复刻 PSO populate 子代链(与基线对齐)")
    ap.add_argument("--expand_quota", type=float, default=0.0,
                    help="v2: 成分扩张生成器配额（对 g 的低|α|维加±0.1 扰动=补第4-6成分），从 ES/DE 各扣一半")
    ap.add_argument("--synth_bg", action="store_true",
                    help="v2: 协同生成器加背景扰动（pair 上叠加 1-2 个 ±0.1 低幅成分）")
    ap.add_argument("--round_evals", type=int, default=4, help="每轮真实评估名额 k")
    ap.add_argument("--max_iter", type=int, default=25)
    ap.add_argument("--coarse_size", type=int, default=50)
    ap.add_argument("--kappa_start", type=float, default=2.0)
    ap.add_argument("--kappa_end", type=float, default=0.3)
    ap.add_argument("--de_F", type=float, default=0.5)
    ap.add_argument("--de_CR", type=float, default=0.9)
    ap.add_argument("--elite_percent", type=float, default=0.25)
    ap.add_argument("--restart_stagnation", type=int, default=3)
    ap.add_argument("--lb", type=float, default=-0.5)
    ap.add_argument("--ub", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=42)
    # ---- 消融旗标（论文消融表用，默认全关）----
    ap.add_argument("--no_gp", action="store_true", help="消融: GP预筛→随机筛, LCB终选→raw best")
    ap.add_argument("--no_syn", action="store_true", help="消融: 协同生成器配额归零(并入ES/DE)")
    ap.add_argument("--simplex", action="store_true", help="消融: 所有提案投影回Σλ=1单纯形(仿射包自缚)")
    ap.add_argument("--project_name_wb", default="model-swarms-qwen2.5-1.5b")  # 与 PSO 基线同 project，便于对比
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--skip_test", action="store_true")
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
    log(f"SGMS v1 search initialized. {now_str()}")
    log(f"args: {vars(args)}")

    wb = None
    if wandb is not None and not args.no_wandb:
        try:
            wb = wandb.init(project=args.project_name_wb, name=f"sgms_{args.name}", config=vars(args))
        except Exception as e:
            log(f"wandb init failed ({e}), continue without wandb")

    pool = ExpertPool(args.expert_dir)
    n = pool.n
    log(f"expert pool: {n} experts: {pool.names}")

    fake_rng = random.Random(7)
    if args.fake_eval:
        log("FAKE EVAL MODE: 仅代码路径测试")
        pool.materialize = lambda a, d: (os.makedirs(d, exist_ok=True), d)[1]

        # 注意：必须用 globals[] 重绑定而非局部 def——局部 def 会让这些名字
        # 成为 main 的局部变量，真实路径(不执行本块)调用时 NameError（GENOME 的教训）
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
    gp = GPSurrogate()
    quad = QuadSurrogate(n)          # 第四路（协同）用二次代理
    archive_a, archive_f = [], []     # 全量观测
    full_evals = coarse_evals = 0

    def materialize(alpha, tag):
        return pool.materialize(alpha, os.path.join(run_dir, "tmp", tag))

    # ---- 初始化：10 单专家 + 混合（LP v4 教训），全量评测 ----
    if args.init_protocol == "pso":  # 与 PSO 基线对齐：单位向量 + populate 子代链（clip 进 box）
        init_alphas = [np.clip(a, args.lb, args.ub) for a in
                       sample_initial_alphas_pso(n, args.init_size, seed=args.seed)]
    else:
        init_alphas = list(sample_initial_alphas(n, args.init_size, seed=args.seed))
    if args.simplex:
        init_alphas = [project_simplex(np.asarray(a, dtype=np.float64)) for a in init_alphas]
    dirs = [materialize(a, f"init_{k}") for k, a in enumerate(init_alphas)]
    t0 = time.time()
    scores = eval_batch(dirs, args.eval_type, args.dataset, gpus, args.base_model)
    full_evals += len(scores)
    archive_a, archive_f = list(init_alphas), list(scores)
    g_idx = int(np.argmax(archive_f))
    g_alpha, g_score = np.asarray(archive_a[g_idx]), archive_f[g_idx]
    starting_best = g_score
    pool.materialize(g_alpha, os.path.join(run_dir, "global_best"))
    log(f"init done in {(time.time()-t0)/60:.1f} min, starting best: {g_score:.3f}")

    # ---- v1p: 全对中点粗筛扫描（覆盖 2-专家脊，死因③的修复）----
    if args.init_pair_scan and not args.fake_eval:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        pm_dirs, pm_alphas = [], []
        for k, (i, j) in enumerate(pairs):
            a = np.zeros(n)
            a[i] = a[j] = 0.5
            pm_alphas.append(a)
            pm_dirs.append(materialize(a, f"pair_{k}"))
        pm_coarse = eval_batch(pm_dirs, args.eval_type, args.dataset, gpus, args.base_model,
                               subset_size=args.coarse_size)
        coarse_evals += len(pm_coarse)
        top3 = np.argsort(pm_coarse)[::-1][:3]
        log(f"pair scan (45 coarse): top5={[(pool.names[pairs[i][0]][:6], pool.names[pairs[i][1]][:6], round(pm_coarse[i],3)) for i in np.argsort(pm_coarse)[::-1][:5]]}")
        for i in top3:
            s = evaluate(pm_dirs[i], args.eval_type, args.dataset, gpus[0], args.base_model)
            full_evals += 1
            archive_a.append(pm_alphas[i])
            archive_f.append(s)
        if max(archive_f) > g_score:
            g_idx = int(np.argmax(archive_f))
            g_alpha, g_score = np.asarray(archive_a[g_idx]), archive_f[g_idx]
            pool.materialize(g_alpha, os.path.join(run_dir, "global_best"))
            log(f"pair scan new best: {g_score:.3f}")

    step_scale, stagnation, dirichlet_boost = 1.0, 0, False
    kappa = args.kappa_start
    history = [g_score]
    if wb is not None:
        wb.log({"g": g_score, "iter": 0, "full_evals": full_evals})

    def elites():
        k = max(3, int(args.elite_percent * len(archive_a)))
        k = min(k, len(archive_a))
        idx = np.argsort(archive_f)[::-1][:k]
        return [np.asarray(archive_a[i]) for i in idx], [archive_f[i] for i in idx]

    def clip(a):
        if args.simplex:
            a = project_simplex(np.asarray(a, dtype=np.float64))
        return np.clip(a, args.lb, args.ub)

    def near_archive(a, eps=0.05):
        return any(np.linalg.norm(a - np.asarray(b)) < eps for b in archive_a)

    for it in range(1, args.max_iter + 1):
        log(f"--- iteration {it}! {now_str()} ---")
        E, Ef = elites()
        kappa = args.kappa_start + (args.kappa_end - args.kappa_start) * (it / args.max_iter)

        # ---- 1. 提案 Q=3k（四路 + v2 成分扩张）----
        Q = 3 * args.round_evals
        eq = min(args.expand_quota, 0.3)
        if args.no_syn:
            ratios5 = [0.4 - eq / 2, 0.4 - eq / 2, 0.2, 0.0, eq]  # 协同配额并入 ES/DE
        else:
            ratios5 = [0.3 - eq / 2, 0.3 - eq / 2, 0.2, 0.2, eq]  # ES/DE/Dir/协同/扩张
        if dirichlet_boost:
            if args.no_syn:
                ratios5 = [0.3 - eq / 3, 0.3 - eq / 3, 0.4, 0.0, eq]
            else:
                ratios5 = [0.2 - eq / 3, 0.2 - eq / 3, 0.4, 0.2, eq]
            dirichlet_boost = False
        cum = np.cumsum(ratios5)
        # ES 状态：精英秩加权均值 + 每维std
        wts = np.linspace(1.0, 0.3, len(E))
        wts = wts / wts.sum()
        m_es = np.average(np.stack(E), axis=0, weights=wts)
        sd_es = np.clip(np.std(np.stack(E), axis=0), 0.02, 0.5)
        syn = quad.synergy(archive_a) if len(archive_a) >= 30 else []
        proposals = []
        for _ in range(Q):
            r = rng.random()
            if r < cum[0]:  # ES 提案（自适应均值/方差 + 步长）
                a = m_es + step_scale * sd_es * np.random.normal(0, 1, n)
            elif r < cum[1]:  # DE rand-to-best/1/bin
                if len(E) >= 3:
                    r1, r2, r3 = (E[rng.randrange(len(E))] for _ in range(3))
                    lam_cur = np.asarray(archive_a[rng.randrange(len(archive_a))])
                    v = r1 + args.de_F * (np.asarray(g_alpha) - r1) + args.de_F * (r2 - r3)
                    cross = np.random.random(n) < args.de_CR
                    cross[rng.randrange(n)] = True
                    a = np.where(cross, v, lam_cur)
                else:
                    a = rng_dirichlet(rng, n)
            elif r < cum[2]:  # Dirichlet（凸组合整体缩放，可出仿射包）
                a = rng_dirichlet(rng, n)
            elif r < cum[3]:  # 协同提案（二次代理 βᵢⱼ；v2 可加背景扰动）
                if syn:
                    i, j = syn[0][0], syn[0][1]
                    syn = syn[1:] + syn[:1]
                    a = np.full(n, 0.02)
                    a[i], a[j] = rng.uniform(0.4, 0.9), rng.uniform(0.4, 0.9)
                    if args.synth_bg:  # v2: 叠加 1-2 个低幅第 4-6 成分
                        for k2 in rng.sample([x for x in range(n) if x not in (i, j)],
                                             rng.choice([1, 2])):
                            a[k2] += rng.uniform(-0.12, 0.12)
                else:
                    a = rng_dirichlet(rng, n)
            else:  # v2 成分扩张：对当前最优补 2-3 个低幅成分（冠军=脊点+第4-6成分的模式）
                a = np.asarray(g_alpha).copy()
                low_dims = np.argsort(np.abs(a))[: max(2, n // 3)]
                for k2 in rng.sample(list(low_dims), min(3, len(low_dims))):
                    a[k2] += rng.uniform(0.08, 0.15) * rng.choice([-1, 1])
            proposals.append(clip(a))
        proposals = [p for p in proposals if not near_archive(p)] or [clip(rng_dirichlet(rng, n))]

        # ---- 2. GP 采集预筛 → k 个 ----
        mu, sd = gp.predict(proposals)
        if mu is not None:
            acq = mu + kappa * sd
            order = np.argsort(acq)[::-1]
        else:
            order = np.random.permutation(len(proposals))
        chosen = [proposals[i] for i in order[: args.round_evals]]
        log(f"proposals {len(proposals)} -> surrogate screen -> {len(chosen)} (kappa={kappa:.2f})")

        # ---- 3. 多保真：粗筛 → 半数全评 + 冠军复评 ----
        cdirs = [materialize(a, f"it{it}_c{k}") for k, a in enumerate(chosen)]
        if args.coarse_size and args.coarse_size > 0:
            coarse = eval_batch(cdirs, args.eval_type, args.dataset, gpus, args.base_model,
                                subset_size=args.coarse_size)
            coarse_evals += len(coarse)
            order2 = np.argsort(coarse)[::-1]
            promote = [(chosen[i], cdirs[i]) for i in order2[: max(1, len(chosen) // 2)]]
            log(f"coarse: {[f'{c:.3f}' for c in coarse]} -> promote {len(promote)}")
        else:
            promote = list(zip(chosen, cdirs))

        promoted = []
        for a, d in promote:
            s = evaluate(d, args.eval_type, args.dataset, gpus[0], args.base_model)
            full_evals += 1
            if s > g_score:
                s2 = evaluate(d, args.eval_type, args.dataset, gpus[0], args.base_model)
                full_evals += 1
                s = (s + s2) / 2.0
            archive_a.append(a)
            archive_f.append(s)
            promoted.append((a, s))

        best_a, best_s = max(promoted, key=lambda t: t[1])
        if best_s > g_score + 1e-9:
            g_alpha, g_score = best_a, best_s
            pool.materialize(g_alpha, os.path.join(run_dir, "global_best"))
            log(f"new global best: {g_score:.3f}")
            step_scale = min(step_scale * 1.1, 3.0)
            stagnation = 0
        else:
            stagnation += 1
            step_scale = max(step_scale * 0.85, 0.15)
            if stagnation >= args.restart_stagnation:
                step_scale = min(step_scale * 1.5, 3.0)
                dirichlet_boost = True
                log(f"stagnation restart: sigma -> {step_scale:.2f}, dirichlet boost")
                stagnation = 0

        # ---- 4. 代理更新（--no_gp 下不拟合: 预筛自动退化为随机筛、LCB 终选退化为 raw best）----
        if SK_OK and not args.no_gp and it % 2 == 0:
            gp.fit(archive_a, archive_f)
        if len(archive_a) >= 30:
            quad.fit(archive_a, archive_f)

        history.append(g_score)
        log(f"iter {it}: g={g_score:.3f} step={step_scale:.2f} full={full_evals} "
            f"coarse={coarse_evals} equiv={full_evals + 0.25*coarse_evals:.0f}")
        if wb is not None:
            wb.log({"g": g_score, "iter": it, "step_scale": step_scale,
                    "full_evals": full_evals, "equiv_evals": full_evals + 0.25 * coarse_evals})
        with open(os.path.join(run_dir, "sgms_scratchpad.json"), "w") as f:
            json.dump({"iter": it, "g": g_score, "history": history,
                       "g_alpha": list(map(float, g_alpha)),
                       "full_evals": full_evals, "coarse_evals": coarse_evals}, f, indent=1)

    # ---- 5. LCB 终选 + test ----
    if args.skip_test:
        log(f"search ended (skip_test). {now_str()}")
        return
    top_idx = np.argsort(archive_f)[::-1][:10]
    final_a, final_s = g_alpha, g_score
    mu, sd = gp.predict([archive_a[i] for i in top_idx])
    if mu is not None:
        lcb = mu - 0.5 * sd
        pick = top_idx[int(np.argmax(lcb))]
        final_a, final_s = np.asarray(archive_a[pick]), archive_f[pick]
        log(f"LCB final selection: raw best {archive_f[top_idx[0]]:.3f} -> LCB pick {final_s:.3f}")
        pool.materialize(final_a, os.path.join(run_dir, "global_best"))
    log(f"search ended. evaluating final model on test... {now_str()}")
    test_score = evaluate_test(os.path.join(run_dir, "global_best"), args.eval_type, args.dataset,
                               gpus[0], args.base_model)
    log(f"final metrics: starting_best_dev={starting_best:.3f} ending_best_dev={final_s:.3f} "
        f"ending_best_test={test_score:.3f} full_evals={full_evals} coarse_evals={coarse_evals}")
    if wb is not None:
        wb.log({"final_dev": final_s, "final_test": test_score})
        wb.finish()
    log(f"the end of search... {now_str()}")


def project_simplex(v):
    """欧氏投影到 Σv=1, v≥0 单纯形（Duchi et al. 2008）。--simplex 消融用。"""
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (css - 1))[0][-1]
    theta = (css[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def rng_dirichlet(rng, n):
    """凸组合 × U(0.5,1.5) 缩放：Dirichlet 保多样 + 可出仿射包。"""
    a = np.random.dirichlet(np.ones(n))
    return a * rng.uniform(0.5, 1.5)


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
