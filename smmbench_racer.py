"""smmbench_racer.py — 把 RACER 验证过的搜索协议移植到 SMM-Bench-PS(64维, 毫秒级代理).

目的: 外部验证——我们的引擎组合与噪声协议在别人的基准/别人的已发表参照
     (Sep-CMA-ES, DE, Akizuki et al. 2025) 上是否成立. 零 GPU 成本.

移植内容(保持机制, 适配空间):
  三路引擎: sep-CMA-lite / SHADE-DE / 定向采样(20维sum带先验→64维改为 box 自适应带)
  竞速漏斗: 粗评=代理加噪? 代理无噪声 → 漏斗在此不适用, 改为纯全评
             (协议层中漏斗是评测贵的产物; 代理便宜, 只测引擎+记忆层)
  终选: 观测即无噪 → 直接 argmax
即: 本测试隔离的是【引擎组合 vs 单引擎】在 64 维光滑/崎岖代理地形上的差异.
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "/data/lvliping/.claude/jobs/e49790df/tmp/SMM-Bench")
from smmbench import SMMBenchPS


def run_engine(f, D, seed, budget=200, mode="racer", pop_init=20):
    """mode: racer(三路) / sep_cma / de / random — 同预算对照."""
    rng = np.random.default_rng(seed)
    archive_x, archive_f = [], []
    lb, ub = 0.0, 1.0

    def ev(x):
        return float(f(np.clip(x, lb, ub)))

    # 初始化: 均匀随机
    for _ in range(pop_init):
        x = rng.uniform(lb, ub, D)
        archive_x.append(x); archive_f.append(ev(x))
    g_i = int(np.argmax(archive_f))
    g_x, g_f = archive_x[g_i].copy(), archive_f[g_i]
    spent = pop_init
    sigma = 0.25
    M_F, M_CR = [0.5] * 5, [0.9] * 5

    while spent < budget:
        E_idx = np.argsort(archive_f)[::-1][:max(3, len(archive_f) // 4)]
        E = [archive_x[i] for i in E_idx]
        Ef = [archive_f[i] for i in E_idx]
        wts = np.linspace(1.0, 0.3, len(E)); wts /= wts.sum()
        m_es = np.average(np.stack(E), axis=0, weights=wts)
        sd_es = np.clip(np.std(np.stack(E), axis=0), 0.02, 0.4)
        cands = []
        for q in range(4):
            r = rng.random()
            if mode in ("racer", "sep") and r < (0.4 if mode == "racer" else 1.0):
                cands.append(m_es + sigma * sd_es * rng.standard_normal(D))
            elif mode in ("racer", "de") and (r < 0.8 or mode == "de"):
                F = float(np.clip(rng.standard_cauchy() * 0.1 + M_F[rng.integers(5)], 1e-3, 1))
                CR = float(np.clip(rng.normal(M_CR[rng.integers(5)], 0.1), 0, 1))
                pbn = max(2, int(0.11 * len(archive_x)))
                pb = archive_x[np.argsort(archive_f)[::-1][:pbn][rng.integers(pbn)]]
                r1 = archive_x[rng.integers(len(archive_x))]
                r2 = archive_x[rng.integers(len(archive_x))]
                cur = archive_x[rng.integers(len(archive_x))]
                v = cur + F * (pb - cur) + F * (r1 - r2)
                cross = rng.random(D) < CR
                cross[rng.integers(D)] = True
                cands.append(np.where(cross, v, cur))
            elif mode == "racer":
                cands.append(rng.uniform(lb, ub, D))
            else:
                cands.append(rng.uniform(lb, ub, D))
        scores = [ev(x) for x in cands]
        spent += len(cands)
        for x, s in zip(cands, scores):
            archive_x.append(x); archive_f.append(s)
        best_q = int(np.argmax(scores))
        if scores[best_q] > g_f + 1e-9:
            g_x, g_f = cands[best_q].copy(), scores[best_q]
            sigma = min(sigma * 1.1, 0.5)
        else:
            sigma = max(sigma * 0.85, 0.03)
    return g_f, float(f(g_x, mode="test_acc"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--modes", default="racer,sep,de,random")
    args = ap.parse_args()
    f = SMMBenchPS("/data/lvliping/.claude/jobs/e49790df/tmp/SMM-Bench/surrogate/smm-bench-ps")
    results = {}
    for mode in args.modes.split(","):
        tr, te = [], []
        for s in range(args.repeats):
            g_f, g_t = run_engine(f, f.D, seed=1000 + s, budget=args.budget, mode=mode)
            tr.append(g_f); te.append(g_t)
        results[mode] = {"train": float(np.mean(tr)), "train_std": float(np.std(tr)),
                         "test": float(np.mean(te)), "test_std": float(np.std(te))}
        print(f"{mode:8s} train={np.mean(tr):.4f}±{np.std(tr):.4f} "
              f"test={np.mean(te):.4f}±{np.std(te):.4f}")
    with open("smmbench_results.json", "w") as fp:
        json.dump(results, fp, indent=1)


if __name__ == "__main__":
    main()
