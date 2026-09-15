"""alpha_space.py — LP-Merge 的 α 物化层。

将"合并系数向量 α ∈ R^n"与"物化 LoRA adapter 目录"互相转换：
- ExpertPool: 一次性加载 n 个专家的 LoRA safetensors（常驻内存 ~n×37MB）
- materialize(α, out_dir): Σ αᵢ·eᵢ → 可直接喂给 evaluate.py 的 adapter 目录
- recover_alpha(state_dict/dir): 最小二乘反解 α（span 检验：残差应≈0）

与 merge.py 的 lora_merge(fast_merge=1) 等价，但专家只加载一次，
且在系数空间操作（doc/lp_algorithm_design.md §6）。
"""
import glob
import json
import os
import random
import shutil

import numpy as np
import torch
from safetensors.torch import load_file, save_file


class ExpertPool:
    """加载并常驻 n 个同构 LoRA 专家。"""

    def __init__(self, expert_dir="./initial_experts"):
        dirs = sorted(
            d for d in glob.glob(os.path.join(expert_dir, "*"))
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "adapter_model.safetensors"))
        )
        if not dirs:
            raise FileNotFoundError(f"{expert_dir} 下没有含 adapter_model.safetensors 的专家目录")
        self.names = [os.path.basename(d) for d in dirs]
        self.n = len(dirs)

        self._experts = []          # list[state_dict]，与 names 对齐
        keys = None
        for d in dirs:
            sd = load_file(os.path.join(d, "adapter_model.safetensors"), device="cpu")
            if keys is None:
                keys = set(sd.keys())
            else:
                assert set(sd.keys()) == keys, f"专家 {d} 的 LoRA key 与其他专家不一致"
            self._experts.append(sd)
        self.keys = sorted(keys)
        self._config_src = os.path.join(dirs[0], "adapter_config.json")

        # 展平的专家矩阵 (n, D)，供反解 α 用；惰性构建
        self._flat = np.stack([
            torch.cat([sd[k].reshape(-1) for k in self.keys]).numpy().astype(np.float64)
            for sd in self._experts
        ])  # (n, D)

    # ---------- 物化 ----------
    def materialize(self, alpha, out_dir):
        """α → adapter 目录（含 adapter_config.json + 加权求和的 safetensors）。"""
        alpha = np.asarray(alpha, dtype=np.float64)
        assert alpha.shape == (self.n,), f"alpha 维度应为 ({self.n},)"
        merged = {}
        with torch.no_grad():
            for k in self.keys:
                acc = None
                for a, sd in zip(alpha, self._experts):
                    if a == 0.0:
                        continue
                    t = sd[k] * a
                    acc = t if acc is None else acc + t
                merged[k] = acc if acc is not None else torch.zeros_like(self._experts[0][k])
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        save_file(merged, os.path.join(out_dir, "adapter_model.safetensors"))
        shutil.copyfile(self._config_src, os.path.join(out_dir, "adapter_config.json"))
        return out_dir

    # ---------- 反解 ----------
    def recover_alpha(self, merged_dir=None, state_dict=None):
        """从合并后的权重反解 α（span 检验：residual 应≈0）。"""
        sd = state_dict or load_file(os.path.join(merged_dir, "adapter_model.safetensors"), device="cpu")
        target = torch.cat([sd[k].reshape(-1) for k in self.keys]).numpy().astype(np.float64)
        alpha, *_ = np.linalg.lstsq(self._flat.T, target, rcond=None)
        residual = float(np.linalg.norm(self._flat.T @ alpha - target) / max(np.linalg.norm(target), 1e-12))
        return alpha, residual

    # ---------- 供 surrogate/分析用的便捷 ----------
    def flat(self):
        """(n, D) 专家展平矩阵。"""
        return self._flat


# ---------- 初始种群采样 ----------
def sample_initial_alphas(n, popsize, seed=0):
    """身份向量 + 混合采样：凸组合 70% / 外插（系数和≠1）30%。

    前 n 个是单位向量（保证初始最优单专家进入评测，可与 PSO run 对齐起点）。
    """
    rng = np.random.default_rng(seed)
    alphas = [np.eye(n)[i] for i in range(min(n, popsize))]
    while len(alphas) < popsize:
        if rng.random() < 0.7:  # 凸组合
            a = rng.dirichlet(np.ones(n) * 0.6)  # 较稀疏的 Dirichlet
        else:                    # 外插：凸组合整体缩放，系数和 ∈ [0.5, 1.5]
            a = rng.dirichlet(np.ones(n) * 0.6) * rng.uniform(0.5, 1.5)
        alphas.append(a)
    return np.stack(alphas)


def sample_initial_alphas_pso(n, popsize, seed=0):
    """复刻 search.py populate_initial_experts 的初始化分布（α 空间版，与 PSO 基线对齐）：
    前 n 个单位向量；后续每个新点 = 从『已有全部点（含新生成的子代）』随机挑两个
    不同父代，w1~U(0,2)，α = w1·p1 + (1−w1)·p2 —— 子代可再当父代，故后期点可为
    3-4 稀疏。超出搜索 box 的系数由调用方 clip（PSO 本身无 box，这是唯一差异）。
    """
    rng = random.Random(seed)
    pts = [np.eye(n)[i] for i in range(min(n, popsize))]
    while len(pts) < popsize:
        i1 = rng.randrange(len(pts))
        i2 = rng.randrange(len(pts))
        while i2 == i1:
            i2 = rng.randrange(len(pts))
        w1 = rng.random() * 2.0
        pts.append(w1 * pts[i1] + (1.0 - w1) * pts[i2])
    return np.stack(pts)
