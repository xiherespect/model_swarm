"""search_racer.py — RACER: Ridge-Adaptive Covariance Evolution with Racing.

SGMS 消融四刀（09-07）+ 15 冠军解结构分析 + 跨机器（214）双种子数据的合成体：
  空间: box [-0.5,1.5]（simplex 消融 −18 → 几何修正是核心）
  初始化: mix 协议（10 单专家 + scaled Dirichlet；pso-init 4 run 方差爆炸已被否决）
  提案 40/40/20:
    sep-CMA-lite 40% —— 秩加权精英均值 + 每维 std + 全局步长 σ + 方向路径
    SHADE-DE 40% —— 成功历史自适应 F/CR，current-to-pbest/1，零超参
    sum 定向 20% —— 精英锚点重标度到 s~U(1.2,1.9)（15 冠军成功带 sum∈[1.3,1.8]）
  竞速漏斗: Q=3k 提案 → 随机选 k（GP 消融 ±0 已移除）→ 粗评(multitask×100/其余×50)
    → 晋升半数全评 → 新冠军复评取均值
  防塌缩守卫: g 连续 3 轮呈失败签名(sum<1.15|k≤3|top2>0.8) → 换脊重启
  终选: top-10 内 0.5·(观测+二次代理μ) − 0.5·残差σ
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


def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    logging.info(msg)
    logging.getLogger().handlers[0].flush()


def _top3_gap(coarse):
    """best 与 third 的差: 抗垃圾候选离群的'强信号'判据(v1.1 教训)."""
    s = sorted(coarse)[::-1]
    return float(s[0] - s[2]) if len(s) >= 3 else 0.0


def top2_mass(a):
    return np.sort(np.abs(a))[::-1][:2].sum() / (np.abs(a).sum() + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--name", required=True)
    ap.add_argument("-e", "--eval_type", required=True)
    ap.add_argument("-d", "--dataset", required=True)
    ap.add_argument("-g", "--gpus", default="0,1,2,3")
    ap.add_argument("-b", "--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("-i", "--expert_dir", default="./initial_experts")
    ap.add_argument("--init_size", type=int, default=20, help="初始池=10 单专家+混合")
    ap.add_argument("--init_protocol", default="mix", choices=["mix", "pso"],
                    help="初始点协议: mix=v4(单位+Dirichlet混合体) / pso=复刻 PSO populate 子代链(与基线对齐)")
    ap.add_argument("--round_evals", type=int, default=4, help="每轮真实评估名额 k")
    ap.add_argument("--max_iter", type=int, default=25)
    ap.add_argument("--coarse_size", type=int, default=50, help="粗评样本数(multitask 自动翻倍)")
    ap.add_argument("--de_F", type=float, default=0.5, help="SHADE F 记忆初值")
    ap.add_argument("--de_CR", type=float, default=0.9, help="SHADE CR 记忆初值")
    ap.add_argument("--shade_p", type=float, default=0.11, help="current-to-pbest 的 pbest 池比例")
    ap.add_argument("--sum_lo", type=float, default=1.2, help="sum 定向生成器下界")
    ap.add_argument("--sum_hi", type=float, default=1.9, help="sum 定向生成器上界")
    ap.add_argument("--guard_patience", type=int, default=3, help="防塌缩守卫触发轮数")
    ap.add_argument("--elite_percent", type=float, default=0.25)
    ap.add_argument("--restart_stagnation", type=int, default=3)
    # ---- v2 三手术（默认全开，消融可关）----
    ap.add_argument("--search_dev", type=int, default=0,
                    help="B:搜索用 dev 前 N 条(0=全量)。判据验证无效已默认关,显式传入开启")
    ap.add_argument("--holdout_dev", type=int, default=200, help="B:终选留出集条数(与search_dev合成700)")
    ap.add_argument("--no_holdout", action="store_true", help="消融B: 关留出集终选,回raw LCB")
    ap.add_argument("--no_plateau", action="store_true", help="消融A: 关地形自适应评估(永远漏斗)")
    ap.add_argument("--no_ridgegate", action="store_true", help="消融C: 关换脊接受准则(回无条件换脊)")
    ap.add_argument("--plateau_patience", type=int, default=2, help="A:平顶判定所需连续轮数")
    ap.add_argument("--no_rung2", action="store_true", help="消融X2: 关双档粗筛(rung1距最优0.03内者加测2×rung2再排晋升)")
    # ---- v1.1 四刀（默认全开，独立开关）----
    ap.add_argument("--highway", action="store_true", help="v1.2-A: 开高速档(top3差>0.10且ρ>0.5连续2轮→k个全保真直评; v1.1版spread含垃圾候选会误触发已回滚)")
    ap.add_argument("--accept_margin", type=float, default=0.0,
                    help="v1.1-B(已废弃,默认0): 门槛挡不住大彩票跳却误杀小步爬升, 由 --no_finalrecheck 取代")
    ap.add_argument("--no_finalrecheck", action="store_true",
                    help="v1.1-B': 关终选三次复评(top-3 各加测一次,按全观测均值排名——彩票冠军回归均值即降级)")
    ap.add_argument("--adaptive_sum", action="store_true", help="v1.1-C: 开自适应sum带(v1.1验证有害,默认关)")
    ap.add_argument("--stretch", action="store_true", help="v1.2-D: 开秩-1方向拉伸(单刀待验)")
    ap.add_argument("--focus_from", type=int, default=10,
                    help="v3: 从第N轮起启用聚焦引擎(提取g支撑+探针精修,0=关)")
    ap.add_argument("--focus_quota", type=float, default=0.15,
                    help="v3: 聚焦引擎提案配额(从ES/DE各扣一半)")
    # ---- 对局1 选手开关 ----
    ap.add_argument("--full_cma", action="store_true",
                    help="对局C: sep-CMA-lite 换全量 pycma CMA-ES(完整协方差, 粗评代理反馈)")
    ap.add_argument("--no_de", action="store_true", help="对局B: 关 SHADE-DE 引擎")
    ap.add_argument("--no_sum", action="store_true", help="对局B: 关 sum 定向引擎")
    ap.add_argument("--no_guard", action="store_true", help="对局B: 关防塌缩守卫")
    ap.add_argument("--lb", type=float, default=-0.5)
    ap.add_argument("--ub", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=42)
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
    log(f"RACER search initialized. {now_str()}")
    log(f"args: {vars(args)}")

    wb = None
    if wandb is not None and not args.no_wandb:
        try:
            wb = wandb.init(project=args.project_name_wb, name=f"racer_{args.name}", config=vars(args))
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
    quad = QuadSurrogate(n)          # 终选 LCB 用二次代理（μ + 残差σ）
    archive_a, archive_f = [], []     # 全量观测
    full_evals = coarse_evals = 0
    # ---- RACER 状态：sep-CMA-lite 步长/路径 + SHADE 记忆 + 守卫计数 ----
    sigma = 1.0
    evo_path = np.zeros(n)
    M_F = [args.de_F] * 5
    M_CR = [args.de_CR] * 5
    mem_ptr = 0
    collapse_rounds = 0
    # v2 状态：C 换脊记账 / A 平原监测
    visited_pairs = []               # C: 最近访问过的脊(top2 维对)， maxlen 3
    plateau_rounds, plateau_mode = 0, False   # A: 平顶计数与模式开关
    highway_rounds, highway_mode = 0, False   # v1.1-A: 高速档计数与模式开关
    coarse_full_pairs = []           # A: (粗分, 全分) 滑窗，算秩相关
    holdout_evals = 0                # B: 留出集复评记账
    focus_d = None                   # v3: 聚焦引擎方向记忆
    focus_sigma = 0.15               # v3: 聚焦步长
    focus_active = False             # v3: 是否已激活
    # 对局1: 全量 CMA 引擎状态（粗评代理反馈: ES 只吃自己 ask 的粗分）
    es_obj, es_pending, es_buf = None, [], []   # pycma ES / 未消费的 ask / (x, coarse) 缓冲
    es_quota = 0   # 每轮由引擎构成决定: cma_only=4(全部) / 混合=2
    # B: 动态切分——各任务 dev 长度不一(gsm8k 200 / normad 500 / hearsay 94...)，
    #    留出集取末尾 N_h = min(--holdout_dev, len//4)，搜索集 = 其前全部
    if args.search_dev and args.holdout_dev:
        try:
            if os.path.exists(f"data/eval/{args.dataset}.json"):
                _dev_len = len(json.load(open(f"data/eval/{args.dataset}.json"))["dev"])
            else:  # multitask 域: 取子任务最小 dev
                from evaluate import multitask_domain_dataset_dict as _mddd
                _dev_len = min(len(json.load(open(f"data/eval/{s}.json"))["dev"])
                               for s in _mddd.get(args.dataset, [args.dataset]))
        except Exception:
            _dev_len = 200
        N_h = min(args.holdout_dev, max(20, _dev_len // 4))
        holdout_start = max(0, _dev_len - N_h)
        N_s = max(20, holdout_start) if holdout_start > 0 else _dev_len
        log(f"dev split: total={_dev_len} search={N_s} holdout={N_h}@[{holdout_start}:{_dev_len}]")
    else:
        N_h, holdout_start, N_s = 0, 0, (args.search_dev or None)

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
    scores = eval_batch(dirs, args.eval_type, args.dataset, gpus, args.base_model,
                        subset_size=(N_s if (args.search_dev and args.holdout_dev) else (args.search_dev or None)))
    full_evals += len(scores)
    archive_a, archive_f = list(init_alphas), list(scores)
    g_idx = int(np.argmax(archive_f))
    g_alpha, g_score = np.asarray(archive_a[g_idx]), archive_f[g_idx]
    starting_best = g_score
    pool.materialize(g_alpha, os.path.join(run_dir, "global_best"))
    log(f"init done in {(time.time()-t0)/60:.1f} min, starting best: {g_score:.3f}")

    stagnation, sumdir_boost = 0, False
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

        # ---- 0. 防塌缩守卫：g 呈失败签名(sum<1.15|k≤3|top2>0.8) 连续 guard_patience 轮 → 换脊 ----
        if args.no_guard:
            g_sum, g_k, g_t2 = 9.9, 9, 0.0   # B: 守卫关闭——签名永不触发
        else:
            g_sum, g_k = float(np.sum(g_alpha)), int((np.abs(g_alpha) > 0.05).sum())
            g_t2 = top2_mass(np.asarray(g_alpha))
        if (g_sum < 1.15) or (g_k <= 3) or (g_t2 > 0.8):
            collapse_rounds += 1
        else:
            collapse_rounds = 0
        if collapse_rounds >= args.guard_patience:
            gpair = tuple(sorted(np.argsort(np.abs(g_alpha))[::-1][:2].tolist()))
            all_cand = [(i, tuple(sorted(np.argsort(np.abs(np.asarray(archive_a[i])))[::-1][:2].tolist())))
                        for i in range(len(archive_a))]
            # C: 换脊接受准则 —— ① 换入分不低于 g−0.02（不许越换越差）
            #    ② 优先未访问过的脊；全部访问过则清空黑名单重探
            if args.no_ridgegate:
                cand = [i for i, p in all_cand if p != gpair]
            else:
                cand = [i for i, p in all_cand
                        if p != gpair and p not in visited_pairs and archive_f[i] >= g_score - 0.02]
                if not cand:  # 放宽: 允许分稍低(-0.05)但脊必须新
                    cand = [i for i, p in all_cand
                            if p != gpair and p not in visited_pairs and archive_f[i] >= g_score - 0.05]
                if not cand:
                    visited_pairs.clear()
                    cand = [i for i, p in all_cand if p != gpair and archive_f[i] >= g_score - 0.02]
            if cand:
                j = max(cand, key=lambda i: archive_f[i])
                visited_pairs.append(gpair)
                visited_pairs = visited_pairs[-3:]
                g_alpha, g_score = np.asarray(archive_a[j]), archive_f[j]
                pool.materialize(g_alpha, os.path.join(run_dir, "global_best"))
                log(f"ridge-switch restart: g -> {g_score:.3f} (sum={g_sum:.2f} k={g_k} t2={g_t2:.2f}, "
                    f"blacklist={len(visited_pairs)})")
            sigma = max(sigma, 1.2)
            sumdir_boost = True
            collapse_rounds = 0

        # ---- v3: 聚焦引擎激活(第N轮且有稀疏支撑) ----
        if args.focus_from and it >= args.focus_from:
            S_g = np.abs(g_alpha) > 0.05
            k_f = int(S_g.sum())
            if 2 <= k_f <= 10:
                if not focus_active:
                    focus_active = True
                    focus_d = np.random.randn(k_f)
                    focus_d /= (np.linalg.norm(focus_d) + 1e-9)
                    log(f"FOCUS ON: support={np.where(S_g)[0].tolist()} k={k_f}")
                elif focus_d is not None and len(focus_d) != k_f:
                    # 支撑大小变了 → 重新初始化方向
                    focus_d = np.random.randn(k_f)
                    focus_d /= (np.linalg.norm(focus_d) + 1e-9)
                    focus_sigma = 0.15
                    log(f"FOCUS RESET: support size {len(focus_d)}→{k_f}")

        # ---- 1. 提案 Q=3k（sep-CMA-lite / SHADE-DE / sum定向 / v3聚焦）----
        #   对局1: --full_cma 时 sep-CMA-lite 路由被全量 ES ask 取代(配额让给 ES)
        if args.full_cma:
            es_quota = 4 if (args.no_de and args.no_sum) else 2   # B=独占 / C=共享
            if es_obj is None:
                import cma as _pycma
                es_obj = _pycma.CMAEvolutionStrategy(
                    np.asarray(g_alpha, dtype=float), 0.3,
                    {"popsize": 4, "bounds": [[args.lb] * n, [args.ub] * n],
                     "seed": args.seed, "verbose": -9})
                log(f"FULL-CMA engine ON (popsize4, sigma0=0.3, quota={es_quota}/轮)")
            while len(es_pending) < es_quota:
                es_pending.extend(es_obj.ask())
        else:
            es_quota = 0
        Q = 3 * args.round_evals
        fq = args.focus_quota if focus_active else 0.0
        ratios = [0.4 - fq/2, 0.4 - fq/2, 0.2, fq]  # ES/DE/sum/focus
        if sumdir_boost:
            ratios = [0.3 - fq/2, 0.3 - fq/2, 0.4, fq]
        if args.full_cma:  # sep 路由让位: 配额并入 ES, DE/sum 分掉 sep 份额
            ratios = [0.0, ratios[0] + ratios[1], ratios[2]] if not args.no_de else [0.0, 0.0, 1.0]
            if args.no_de and not args.no_sum:
                ratios = [0.0, 0.0, 1.0]
            if args.no_de and args.no_sum:
                ratios = [0.0, 0.0, 0.0]  # B: 无普通提案, 全靠 ES ask
        sumdir_boost = False
        cum = np.cumsum(ratios)
        # sep-CMA-lite 状态：秩加权精英均值 + 每维 std（协方差的对角近似）
        wts = np.linspace(1.0, 0.3, len(E))
        wts = wts / wts.sum()
        m_es = np.average(np.stack(E), axis=0, weights=wts)
        sd_es = np.clip(np.std(np.stack(E), axis=0), 0.02, 0.5)
        proposals, prop_meta = [], []
        for _ in range(Q):
            r = rng.random()
            if r < cum[0]:  # sep-CMA-lite：自适应均值/方差 + 全局步长 + 方向路径
                if not args.stretch:
                    sd_eff = sd_es
                else:  # v1.1-D: 秩-1 方向拉伸——沿 evo_path 把采样椭圆拉斜(斜脊可达)
                    sd_eff = sd_es * (1.0 + 0.5 * np.abs(evo_path) / (np.abs(evo_path).max() + 1e-12))
                a = m_es + sigma * sd_eff * np.random.normal(0, 1, n) + 0.15 * evo_path
                meta = ("cma", None, None)
            elif r < cum[1]:  # SHADE-DE current-to-pbest/1/bin（成功历史自适应 F/CR）
                F = np.clip(np.random.standard_cauchy() * 0.1 + M_F[rng.randrange(5)], 1e-3, 1.0)
                CR = np.clip(np.random.normal(M_CR[rng.randrange(5)], 0.1), 0.0, 1.0)
                if len(E) >= 3:
                    pbn = max(2, int(args.shade_p * len(archive_a)))
                    pb_order = np.argsort(archive_f)[::-1][:pbn]
                    pbest = np.asarray(archive_a[pb_order[rng.randrange(pbn)]])
                    r1 = np.asarray(archive_a[rng.randrange(len(archive_a))])
                    r2 = np.asarray(archive_a[rng.randrange(len(archive_a))])
                    lam_cur = np.asarray(archive_a[rng.randrange(len(archive_a))])
                    v = lam_cur + F * (pbest - lam_cur) + F * (r1 - r2)
                    cross = np.random.random(n) < CR
                    cross[rng.randrange(n)] = True
                    a = np.where(cross, v, lam_cur)
                else:
                    a = rng_dirichlet(rng, n)
                meta = ("de", float(F), float(CR))
            elif r < cum[2]:  # sum 定向：精英锚点重标度到目标 sum 带（15 冠军成功带 1.3-1.8）
                a_e = E[rng.randrange(len(E))]
                if (not args.adaptive_sum) or len(Ef) < 10:
                    s_target = rng.uniform(args.sum_lo, args.sum_hi)
                else:  # v1.1-C(v2): 自适应带——只从多成分精英学, 且夹在[1.1,2.0]保底离包压力
                      # (v1教训: 早期精英=初始化单专家(sum≈1.0), 学错方向拆掉了逃逸压力)
                    cand = [(float(x.sum()), f) for x, f in zip(E, Ef)
                            if int((np.abs(x) > 0.05).sum()) > 2]
                    if len(cand) >= 4:
                        e_sums = np.array([max(c, 0.3) for c, _ in cand])
                        w = np.array([f for _, f in cand], dtype=float)
                        w = w - w.min() + 1e-3
                        j = int(np.random.choice(len(cand), p=w / w.sum()))
                        s_target = float(np.clip(e_sums[j] + np.random.normal(0, 0.1), 1.1, 2.0))
                    else:
                        s_target = rng.uniform(args.sum_lo, args.sum_hi)
                a = a_e * (s_target / max(a_e.sum(), 0.3)) + np.random.uniform(-0.05, 0.05, n)
                meta = ("sum", None, None)
            else:  # v3 聚焦引擎: 在 g 的支撑子空间内探针步进
                S_idx = np.where(np.abs(g_alpha) > 0.05)[0]
                w_sub = np.array([g_alpha[i] for i in S_idx])
                step = focus_sigma * focus_d * rng.choice([-1, 1])
                w_new = np.clip(w_sub + step, args.lb, args.ub)
                a = np.zeros(n)
                a[S_idx] = w_new
                meta = ("focus", None, None)
            proposals.append(clip(a))
            prop_meta.append(meta)
        keep = [i for i, p in enumerate(proposals) if not near_archive(p)]
        if not keep:
            keep = [rng.randrange(len(proposals))]
        proposals = [proposals[i] for i in keep]
        prop_meta = [prop_meta[i] for i in keep]

        # ---- 2. 竞速漏斗第一级：随机选 k（GP 消融 ±0 → 移除代理筛）----
        #   对局1: ES ask 优先占位(保证 CMA 反馈自洽), 其余名额从普通提案随机
        cma_picks = [np.asarray(x, dtype=float) for x in es_pending[:es_quota]]
        es_pending = es_pending[es_quota:]
        n_rand = max(0, args.round_evals - len(cma_picks))
        sel = rng.sample(range(len(proposals)), min(n_rand, len(proposals))) if proposals else []
        chosen = cma_picks + [proposals[i] for i in sel]
        chosen_meta = [("es", None, None)] * len(cma_picks) + [prop_meta[i] for i in sel]
        es_round = cma_picks   # 本轮需回收粗分的 ES ask

        # ---- 3. 竞速漏斗：粗评(multitask×2) → 晋升半数全评 + 冠军复评 ----
        #     A(v2): 平原模式检测——平顶(粗分离散<eps) 且 粗→全秩相关塌(ρ<0.2) 连续 patience 轮
        #            → 切换为全保真直评模式(PSO式暴力, k/2 个)；信号恢复自动切回
        coarse_n = (args.coarse_size * 2) if args.eval_type == "multitask" else args.coarse_size
        cdirs = [materialize(a, f"it{it}_c{k}") for k, a in enumerate(chosen)]

        def _rankcorr(pairs):
            if len(pairs) < 4:
                return None
            x = np.array([p[0] for p in pairs]); y = np.array([p[1] for p in pairs])
            rx, ry = np.argsort(np.argsort(x)).astype(float), np.argsort(np.argsort(y)).astype(float)
            return float(np.corrcoef(rx, ry)[0, 1])

        if plateau_mode or (not args.no_plateau and plateau_rounds >= args.plateau_patience):
            if not plateau_mode:
                plateau_mode = True
                log(f"PLATEAU MODE ON (rounds={plateau_rounds}): 漏斗→全保真直评")
            plateau_rounds = 0
            kk = max(1, args.round_evals // 2)
            promote = [(chosen[i], cdirs[i], chosen_meta[i], None) for i in range(min(kk, len(chosen)))]
            coarse = None
        elif highway_mode or (args.highway and highway_rounds >= 2):
            # v1.1-A 高速档: 信号强且粗评可信 → 粗筛是纯损耗, k 个全部直接全保真
            if not highway_mode:
                highway_mode = True
                log("HIGHWAY MODE ON: 跳过粗筛, 全保真直评")
            highway_rounds = 0
            promote = [(chosen[i], cdirs[i], chosen_meta[i], None) for i in range(len(chosen))]
            coarse = None
        elif coarse_n and coarse_n > 0:
            coarse = eval_batch(cdirs, args.eval_type, args.dataset, gpus, args.base_model,
                                subset_size=coarse_n)
            coarse_evals += len(coarse)
            # X2 双档粗筛: rung1 距最优 0.03 内的"接近者"加测 rung2(2×样本) 再排晋升
            #    （治"粗筛运气好"错杀——接近者用更准的测量分高下, 非接近者不浪费预算）
            rank_score = list(coarse)
            if not args.no_rung2 and len(coarse) > 1:
                near = [i for i in range(len(coarse)) if max(coarse) - coarse[i] <= 0.03]
                if len(near) > 1:
                    r2 = eval_batch([cdirs[i] for i in near], args.eval_type, args.dataset,
                                    gpus, args.base_model, subset_size=coarse_n * 2)
                    coarse_evals += 2 * len(r2)
                    for i, s2 in zip(near, r2):
                        rank_score[i] = s2
                    log(f"rung2 ({len(near)}个接近者, {coarse_n*2}样本): "
                        f"{[f'{s2:.3f}' for s2 in r2]}")
            order2 = np.argsort(rank_score)[::-1]
            promote = [(chosen[i], cdirs[i], chosen_meta[i], i) for i in order2[: max(1, len(chosen) // 2)]]
            log(f"coarse: {[f'{c:.3f}' for c in coarse]} -> promote {len(promote)}")
            if es_obj is not None and es_round:   # 对局1: ES 吃自己 ask 的粗分(代理地形)
                for x in es_round:
                    j = next(k2 for k2, c in enumerate(chosen) if c is x)
                    es_buf.append((x, float(coarse[j])))
                if len(es_buf) >= 4:
                    es_obj.tell([b[0] for b in es_buf], [-b[1] for b in es_buf])
                    es_buf = []
            # A: 平原统计更新
            spread = float(np.max(coarse) - np.min(coarse)) if len(coarse) > 1 else 0.0
            rho = _rankcorr(coarse_full_pairs[-6:])
            plateau_eps = 0.10 if args.eval_type == "multitask" else 0.14
            if spread < plateau_eps and (rho is None or rho < 0.2):
                plateau_rounds += 1
            else:
                plateau_rounds = 0
            # v1.1-A: 高速档统计更新（信号强且粗评排名可信）
            if _top3_gap(coarse) > 0.10 and (rho is not None and rho > 0.5):
                highway_rounds += 1
            else:
                highway_rounds = 0
        else:
            coarse = None
            promote = [(a, d, m, None) for a, d, m in zip(chosen, cdirs, chosen_meta)]

        promoted = []
        succ_F, succ_CR = [], []
        med_f = float(np.median(archive_f))
        # v1.3: 晋升全评批量化——多卡并行(与逐个单卡等价结果, 快~2x; 全保真不传 subset)
        _sub = (N_s if (args.search_dev and args.holdout_dev) else (args.search_dev or None))
        pscores = eval_batch([d for _, d, _, _ in promote], args.eval_type, args.dataset,
                             gpus, args.base_model, subset_size=_sub)
        full_evals += len(pscores)
        for (a, d, meta, ci), s in zip(promote, pscores):
            if s > g_score:
                s2 = evaluate(d, args.eval_type, args.dataset, gpus[0], args.base_model,
                              subset_size=_sub)
                full_evals += 1
                s = (s + s2) / 2.0
            if ci is not None and coarse is not None:   # A: (粗分,全分) 对入滑窗
                coarse_full_pairs.append((float(coarse[ci]), float(s)))
                coarse_full_pairs = coarse_full_pairs[-12:]
            if meta[0] == "de" and s > med_f:   # SHADE 成功判定：DE 儿女过 archive 中位
                succ_F.append(meta[1])
                succ_CR.append(meta[2])
            archive_a.append(a)
            archive_f.append(s)
            promoted.append((a, s))

        if plateau_mode and any(s > g_score + 1e-9 for _, s in promoted):
            plateau_mode = False
            log("PLATEAU MODE OFF: 信号恢复, 切回漏斗")
        if highway_mode and promoted:  # v1.1-A: 档内自检——全评分极差缩水=信号消失
            fs = [s for _, s in promoted]
            if max(fs) - min(fs) < 0.10:
                highway_mode = False
                log("HIGHWAY MODE OFF: 全评极差缩水, 切回漏斗")

        # SHADE 记忆更新（Lehmer mean for F, arithmetic for CR）
        if succ_F:
            M_F[mem_ptr] = float(np.sum(np.array(succ_F) ** 2) / np.sum(succ_F))
            M_CR[mem_ptr] = float(np.mean(succ_CR))
            mem_ptr = (mem_ptr + 1) % 5

        # v3: 聚焦步长自适应
        for a, s in promoted:
            pass  # 在 promoted 中找 focus 来源的成功/失败
        focus_succ = any(s > g_score for a, s in promoted if True)
        if focus_active:
            if focus_succ:
                focus_sigma = min(focus_sigma * 1.2, 0.4)
                focus_d = 0.8 * focus_d + 0.2 * (np.asarray(g_alpha)[np.abs(g_alpha) > 0.05] / max(np.abs(np.asarray(g_alpha)[np.abs(g_alpha) > 0.05]).max(), 1e-9))
                fd_norm = np.linalg.norm(focus_d)
                if fd_norm > 0:
                    focus_d /= fd_norm
            else:
                focus_sigma = max(focus_sigma * 0.85, 0.02)

        best_a, best_s = max(promoted, key=lambda t: t[1])
        # v1.1-B 显著性门槛: 复评均值须超旧冠军 accept_margin 才登基(挡噪声彩票; 0=旧行为)
        if best_s > g_score + max(args.accept_margin, 1e-9):
            old_g = np.asarray(g_alpha, dtype=np.float64)
            g_alpha, g_score = best_a, best_s
            pool.materialize(g_alpha, os.path.join(run_dir, "global_best"))
            log(f"new global best: {g_score:.3f}")
            evo_path = 0.7 * evo_path + 0.3 * (np.asarray(g_alpha, dtype=np.float64) - old_g) / max(sigma, 1e-6)
            sigma = min(sigma * 1.1, 3.0)
            stagnation = 0
        else:
            stagnation += 1
            sigma = max(sigma * 0.85, 0.15)
            if stagnation >= args.restart_stagnation:
                sigma = min(sigma * 1.5, 3.0)
                sumdir_boost = True
                log(f"stagnation restart: sigma -> {sigma:.2f}, sumdir boost")
                stagnation = 0

        # ---- 4. 二次代理（终选 LCB 用）----
        if len(archive_a) >= 30:
            quad.fit(archive_a, archive_f)

        history.append(g_score)
        log(f"iter {it}: g={g_score:.3f} sigma={sigma:.2f} full={full_evals} "
            f"coarse={coarse_evals} equiv={full_evals + 0.25*coarse_evals:.0f} "
            f"gsum={float(np.sum(g_alpha)):.2f} gk={int((np.abs(g_alpha) > 0.05).sum())}")
        if wb is not None:
            wb.log({"g": g_score, "iter": it, "sigma": sigma,
                    "full_evals": full_evals, "equiv_evals": full_evals + 0.25 * coarse_evals})
        with open(os.path.join(run_dir, "racer_scratchpad.json"), "w") as f:
            json.dump({"iter": it, "g": g_score, "history": history,
                       "g_alpha": list(map(float, g_alpha)),
                       "full_evals": full_evals, "coarse_evals": coarse_evals}, f, indent=1)

    # ---- 5. LCB 终选 + test ----
    if args.skip_test:
        log(f"search ended (skip_test). {now_str()}")
        return
    top_idx = np.argsort(archive_f)[::-1][:10]
    final_a, final_s = g_alpha, g_score
    # v1.1-B': 终选三次复评——top-3 各加测一次, 按该点全部观测的均值重排(彩票冠军回归均值即降级)
    rechecked = False
    if not args.no_finalrecheck and not (args.search_dev and args.holdout_dev and not args.no_holdout):
        ck_idx = top_idx[:3]
        recheck = []
        for k2, i in enumerate(ck_idx):
            d2 = materialize(archive_a[i], f"recheck_{k2}")
            s2 = evaluate(d2, args.eval_type, args.dataset, gpus[0], args.base_model,
                          subset_size=(N_s if (args.search_dev and args.holdout_dev) else None))
            full_evals += 1
            recheck.append((i, float(np.mean([archive_f[i], s2]))))  # 该点已有观测+新观测的均值
        pick = max(recheck, key=lambda t: t[1])[0]
        final_a, final_s = np.asarray(archive_a[pick]), archive_f[pick]
        rechecked = True
        log(f"FINAL RECHECK: top3 复评均 {[f'{x[1]:.3f}' for x in recheck]} -> pick {final_s:.3f}")
        pool.materialize(final_a, os.path.join(run_dir, "global_best"))
    if (args.no_holdout or not (args.search_dev and args.holdout_dev)) and not rechecked:
        # v1 路径: 二次代理 LCB 终选
        if quad.w is not None:
            X = quad.features([archive_a[i] for i in top_idx])
            y_all = quad.features(archive_a) @ quad.w
            resid_std = float(np.std(np.asarray(archive_f) - y_all))
            mu = (X @ quad.w).ravel()
            lcb = 0.5 * (np.asarray([archive_f[i] for i in top_idx]) + mu) - 0.5 * resid_std
            pick = top_idx[int(np.argmax(lcb))]
            final_a, final_s = np.asarray(archive_a[pick]), archive_f[pick]
            log(f"LCB final selection: raw best {archive_f[top_idx[0]]:.3f} -> LCB pick {final_s:.3f} "
                f"(resid_std={resid_std:.3f})")
            pool.materialize(final_a, os.path.join(run_dir, "global_best"))
    else:
        # B(v2): 留出集复选——top-10 在 dev[500:700] 重评, 新鲜噪声下打破赢者诅咒
        hd = [materialize(archive_a[i], f"hold_{k}") for k, i in enumerate(top_idx)]
        h_scores = []
        for d in hd:
            hs = evaluate(d, args.eval_type, args.dataset, gpus[0], args.base_model,
                          subset_size=N_h, subset_start=holdout_start)
            holdout_evals += 1
            h_scores.append(hs)
        order_h = np.argsort(h_scores)[::-1]
        pick = top_idx[order_h[0]]
        final_a, final_s = np.asarray(archive_a[pick]), h_scores[order_h[0]]
        log(f"HOLDOUT final selection: search best {archive_f[top_idx[0]]:.3f} -> "
            f"holdout best {final_s:.3f} (top10 holdout: {[f'{x:.3f}' for x in h_scores]})")
        pool.materialize(final_a, os.path.join(run_dir, "global_best"))
    log(f"search ended. evaluating final model on test... {now_str()}")
    test_score = evaluate_test(os.path.join(run_dir, "global_best"), args.eval_type, args.dataset,
                               gpus[0], args.base_model)
    log(f"final metrics: starting_best_dev={starting_best:.3f} ending_best_dev={final_s:.3f} "
        f"ending_best_test={test_score:.3f} full_evals={full_evals} coarse_evals={coarse_evals} "
        f"search_best_dev={float(np.max(archive_f)):.3f} holdout_evals={holdout_evals} "
        f"equiv_total={full_evals + 0.25*coarse_evals + 0.29*holdout_evals:.0f}")
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
