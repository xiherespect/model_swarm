"""CPU 冒烟测试：α 物化层正确性 + 反解闭环 + 二次代理恢复。

不占 GPU。运行: python tests/smoke_cpu.py
"""
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from alpha_space import ExpertPool, sample_initial_alphas
from safetensors.torch import load_file

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASS, FAIL = "✅", "❌"
failures = 0


def check(name, cond, detail=""):
    global failures
    print(f"{PASS if cond else FAIL} {name} {detail}")
    if not cond:
        failures += 1


# ---- 1. 专家池加载 ----
pool = ExpertPool(os.path.join(REPO, "initial_experts"))
check("专家池: 10 个专家", pool.n == 10, f"(n={pool.n})")
check("专家池: key 一致", len(pool.keys) > 0, f"({len(pool.keys)} tensors)")

# ---- 2. one-hot 物化 = 原专家 ----
tmp = tempfile.mkdtemp()
d = pool.materialize(np.eye(pool.n)[3], os.path.join(tmp, "onehot3"))
sd = load_file(os.path.join(d, "adapter_model.safetensors"))
ref = load_file(os.path.join(REPO, "initial_experts", pool.names[3], "adapter_model.safetensors"))
max_diff = max(float((sd[k] - ref[k]).abs().max()) for k in pool.keys)
check("one-hot 物化 == 原专家(4号)", max_diff < 1e-5, f"(max_diff={max_diff:.2e})")

# ---- 3. 随机 α 物化 → 反解闭环（span 检验） ----
rng = np.random.default_rng(0)
alpha_true = rng.dirichlet(np.ones(pool.n) * 0.6) * 1.2  # 系数和非 1，出仿射包
d = pool.materialize(alpha_true, os.path.join(tmp, "mix"))
alpha_rec, residual = pool.recover_alpha(merged_dir=d)
err = float(np.abs(alpha_rec - alpha_true).max())
check("α 反解闭环: 复原随机 α", err < 1e-3, f"(max|Δα|={err:.2e})")
check("span 检验: 残差≈0", residual < 1e-3, f"(rel_resid={residual:.2e})")

# ---- 4. 初始种群采样 ----
alphas = sample_initial_alphas(pool.n, 12, seed=42)
check("初始采样: 形状", alphas.shape == (12, pool.n))
check("初始采样: 前 10 个是单位向量", np.allclose(alphas[:10], np.eye(10)))
sums = alphas[10:].sum(axis=1)
check("初始采样: 混合项系数和在合理区间", bool(((sums > 0.2) & (sums < 2.0)).all()), f"(sums={sums.round(2)})")

# ---- 5. 二次代理恢复已知交互项 ----
from search_lp import QuadSurrogate

n = pool.n
w_true = np.random.default_rng(1).normal(0, 0.5, n + n * (n - 1) // 2 + 1)
surr = QuadSurrogate(n)
# 注意：单纯形上（Σα=1）特征与截距共线，w 不可唯一识别（预测不受影响）。
# 参数恢复测试用无约束 box 采样（满秩设计）检验求解器本身。
Xa = np.random.default_rng(2).uniform(0, 1, (200, n))
y = surr.features(Xa) @ w_true
r2 = surr.fit(Xa, y)
check("二次代理: 合成数据 R²=1", r2 > 0.999, f"(R²={r2:.4f})")
w_err = float(np.abs(surr.w - w_true).max())
check("二次代理: 参数恢复", w_err < 5e-3, f"(max|Δw|={w_err:.2e}, 岭回归 λ=1e-3 的收缩偏差)")
syn = surr.synergy(list(Xa[:20]))
check("协同打分: 非空且有序", len(syn) == 45 and syn[0][2] >= syn[-1][2])
contrib = surr.contributions()
check("贡献度: 形状", contrib is not None and contrib.shape == (n,))

# ---- 6. 噪声鲁棒性：加噪后协同 top-5 仍大致正确 ----
y_noisy = y + np.random.default_rng(3).normal(0, 0.03, 200)
surr.fit(Xa, y_noisy)
true_top = set(np.argsort(-w_true[n:n + 45])[:5])
fit_top = set(np.argsort(-surr.w[n:n + 45])[:5])
check("噪声鲁棒: top-5 交互项命中≥3", len(true_top & fit_top) >= 3,
      f"(命中 {len(true_top & fit_top)}/5)")

print("\n" + ("全部通过 ✅" if failures == 0 else f"{failures} 项失败 ❌"))
sys.exit(1 if failures else 0)
