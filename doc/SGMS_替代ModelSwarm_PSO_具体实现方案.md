# SGMS：代理引导模型群体搜索 —— 替代 Model Swarms PSO 的可行方案

> 文档日期：2026-09-04
> 目标：在 Model Swarms 官方框架内，把 PSO 更新替换为预期更强的搜索器，
> 在等预算下稳定优于原版 PSO，不涉及 EvoGM，不引入 LLM 生成组件。
> 代码依据：https://github.com/BunsenFeng/model_swarm（2026-09 检查的 main 分支）

---

## 0. 核心观察：Model Swarms 的 PSO 隐式上就是 N 维系数搜索

官方实现的事实（`search.py` + `merge.py`）：

1. 粒子 = LoRA 权重目录；一切"运动"都是 `lora_merge(weights, models, out)`，
   即 state dict 的**线性加权求和**；
2. 初始位置 = N 个专家（one-hot）；初始 velocity = 两个专家的 `[-1, 1]` 组合，
   系数和为 0；
3. 每轮的三个差分项 `p−x`、`g−x`、`x−w` 系数和均为 0，四项权重归一化后求和为 1。

归纳可得：**任意时刻任意粒子的权重都严格等于 N 个专家的仿射组合**
`θ = Σᵢ λᵢ τᵢ`，且 `Σλᵢ = 1`。PSO 只是隐式地在
`Λ = {λ ∈ R^N : Σλᵢ = 1}` 这个 N 维空间里做随机游走。

### 0.1 这意味着什么

- 可以**显式参数化** λ 并换成任何低维黑盒优化器，权重仍由
  `lora_merge(λ, experts, out)` 一步合成——不是近似，是精确等价；
- 替换后"合并 + 评估 + 预算协议 + 最终输出"全部保持不变，
  性能差异可以干净归因于搜索器本身（这正是审稿人要的公平性）；
- 工程 bonus：原 PSO 每粒子每轮要做 5 次 `lora_merge`
  （`p_x`、`g_x`、`x_w`、`velocity`、`now`），显式化后每个被评估候选
  只需 **1 次**从专家直接合成，合并开销约降为 1/5。

### 0.2 PSO 的五个具体弱点（对应到代码）

| # | 弱点 | 代码证据 | SGMS 对策 |
|---|---|---|---|
| W1 | 每轮评估全部粒子，从不预筛 | 主循环 `eval_args` 收集所有 particle 的 `now` | 代理模型 top-k 预筛，只评估有希望的提案 |
| W2 | gbest 引力导致多样性坍缩 | `social_coeff` 项 + 权重归一化把所有粒子拉向同一点 | CMA 协方差 + DE + Dirichlet 三路提案 |
| W3 | 只用 pbest/gbest/gworst 三个记忆点，丢弃其余全部历史 | scratchpad 只存 best/worst | 精英 archive 全量喂给代理 |
| W4 | 小验证集噪声下直接选 gbest，val-test gap 大 | `if results[i] > best` 逐点比较 | 采集函数带不确定性（后期 LCB 保守选择） |
| W5 | 固定 inertia/c1/c2，手工 `step_length *= 0.95` 调度 | argparse 默认值 | CMA 自适应步长 + 停滞重启 |

---

## 1. SGMS 算法设计

**SGMS = Surrogate-Guided Merge Swarm（代理引导模型群体搜索）**
= 三路提案算子 + 代理预筛 + 精英 archive + 停滞重启。

### 1.1 状态与数据结构

```text
experts:        τ_1..τ_N                     # 初始专家 LoRA，只读
archive D:      {(λ, F(λ), per_example_scores)}  # 所有已真实评估的候选
elite E:        D 中 top-25%                 # 喂 CMA 与 DE
CMA 状态:       (m, σ, C)                    # 建议用 separable CMA 或 (μ/μ_w,λ)-ES
全局最优 g:     archive 中按采集函数复核后的最优
```

### 1.2 每轮流程（替代 particle_update + 全量评估）

```text
输入: archive D, 剩余预算 b, 本轮评估名额 k

1. 生成 Q = 3k 个提案（系数向量 λ，Σλ=1）:
     a) CMA 提案 40%: m + σ·N(0,C)，围绕精英均值，带协方差方向信息
     b) DE 提案  40%: v = λ_r1 + F(λ_best − λ_r1) + F(λ_r2 − λ_r3)
                      与 λ_current 二项交叉（CR=0.9, F=0.5）
     c) Dirichlet 20%: Dir(α=1) 重采样，纯探索保险
2. 代理预筛:
     在 D 上拟合 GP（N≤30）或 RF（N>30），得到 μ(λ), σ(λ)
     采集函数 A(λ) = μ(λ) + κ·σ(λ)，κ 随预算消耗从 2.0 线性衰减到 0.3
     选 A 最高的 k 个提案（去重、且与 archive 中已评估点的距离 > ε）
3. 真实评估:
     对每个选中 λ: lora_merge(λ, experts, particle_i/now) → evaluate(dev)
     逐样本分数写入 archive
4. 更新:
     重算 elite；CMA 用 top-μ 精英做加权均值/步长更新
     若 gbest 连续 3 轮无提升: σ ← 1.5σ，并把下轮提案的 Dirichlet 配额提至 40%
5. 预算耗尽后:
     在 archive 上用 LCB 复核（μ − 0.5σ），选最终输出 g
```

### 1.3 各组件与弱点的对应（消融依据）

| 组件 | 针对弱点 | 单独可关 |
|---|---|---|
| 代理预筛 | W1 预算浪费 | `--surrogate off` 退化为纯多路进化 |
| CMA 协方差 | W5 步长 + W2 | 换均匀采样 |
| DE 提案 | W2 多样性 + W3 历史 | 关闭该路 |
| LCB 复核 | W4 val-test gap | 直接用 argmax |
| 停滞重启 | W2/W5 | 固定 σ |

### 1.4 为什么在 Model Swarms 设定下有胜算

- 有效维度 = 专家数 N（典型 5–10，最多几十），代理模型在这个维度**完全成立**，
  这是 BO/代理辅助方法最强的区间；
- 评估是整条流水线里唯一大头（模型推理），预筛直接换算成候选预算；
- 原版 PSO 每轮把全部预算花在"盲飞"出的位置上，SGMS 每轮只在代理认为
  有希望的位置花预算——同样 100 次评估，SGMS 相当于搜了 ~300 个提案。

---

## 2. 代码集成方案（改动点）

仓库结构不动，新增一个文件，改 `search.py` 两处。

### 2.1 新增 `sgms.py`

```text
sgms.py
  class CoefficientSpace:      # λ 约束 Σ=1、去重、距离
  class Surrogate:             # sklearn GP / RF，标准化 y
  class CMAKernel:             # sep-CMA 或简化 ES；含 σ 重启
  def de_propose(D, elite):    # rand-to-best/1/bin
  def dirichlet_propose(N):    # α=1
  def acquisition(mu, sig, kappa): ...
  class SGMS:                  # step(archive, k) -> List[λ]
```

依赖只有 `numpy/scipy/sklearn`，全部 CPU、毫秒级开销。

### 2.2 改 `search.py`

1. `initialize_search_records`：
   - 保留粒子目录结构（`now`/`personal_best` 等复用现有清理与日志逻辑）；
   - 每个粒子写 `coeff.json` 存当前 λ（初始 one-hot）；
   - 跳过 velocity 初始化。
2. 主循环：
   ```python
   proposals = sgms.step(archive, k=per_round_evals)   # 替代 particle_update
   for j, lam in enumerate(proposals):
       lora_merge(lam, expert_paths, f".../particle_{j}/now", gpu, fast_merge)
   # 只 evaluate 被选中的 k 个粒子；其余粒子跳过评估
   # 分数写入 archive；global_best 用 LCB 复核后的 archive 最优
   ```
3. `evaluate.py`、`merge.py`、wandb 记录、最终 `evaluate_test` 流程**零改动**。

### 2.3 CLI 兼容

原 PSO 参数（inertia/cognitive/...）保留但忽略；新增：

```text
--searcher {pso, sgms}        # pso 保留为冻结基线
--round_evals k               # 每轮真实评估数（默认 4）
--proposal_ratio Q/k          # 默认 3
--surrogate {gp, rf, off}
--kappa_start / --kappa_end   # 2.0 / 0.3
--restart_patience_rounds 3
```

---

## 3. 实验设计

### 3.1 阶梯（必须按顺序跑，先便宜后复杂）

```text
Rung 0  Random-λ（等预算随机系数）
        Uniform average（1/N）
        Best single expert
Rung 1  原版 PSO（frozen，官方默认超参）
Rung 2  纯 CMA-on-λ（无代理）      # 检验"显式系数化"本身的收益
Rung 3  SGMS-lite = CMA + 代理预筛  # 检验代理的边际贡献
Rung 4  SGMS-full = + DE 提案 + 停滞重启 + LCB 复核
Rung 5（可选） + 多保真：先小子集粗评，successive halving 加样本
```

每一级都和 Rung 1 在**相同 evaluate() 次数 + 相同 candidate-example pairs**
下对比；同时报告 wall time（SGMS 合并次数约为 PSO 的 1/5，理应更快）。

### 3.2 设置

- 框架/数据：直接用官方 repo 的 `data/eval/*.json`
  （human_* 未见任务为主战场，加 gsm8k、hellaswag 等标准任务）；
- 专家池：沿用 `initial_experts/` 的 LoRA 流程（gemma-7b-it 基座），
  算力不够换小基座 + Tulu LoRA，但 PSO 与 SGMS 必须同池同评估器；
- 种子：每配置 ≥5；
- 预算档位：50 / 100 / 200 次候选评估。

### 3.3 指标

```text
final test score                     # 主指标
best-so-far AUC vs 候选评估数         # 样本效率
evaluations to reach PSO-final-score # 追平所需预算
validation-test gap                  # W4 是否被修复
λ 多样性（archive pairwise distance）# W2 是否被修复
wall time / merge 次数
```

统计：配对 A12 + seed 级 Wilcoxon + bootstrap CI（沿用 rlo_ea 的 Gate 风格）。

### 3.4 成功标准（预注册）

- 主判据：SGMS-full 在 ≥2/3 预算档、≥60% 任务上 test 分数优于 PSO，
  且 A12 > 0.56；
- 机制验证：−代理 / −DE / −重启三个消融中，每个至少在一个预算档
  显著变差（证明不是缝合怪）；
- 诚实报告：若只有 Rung 2 赢（纯 CMA 就够），论文主张改为
  "显式系数化 + 标准自适应优化器即可超过权重空间 PSO"，
  这本身也是干净且有价值的结论。

---

## 4. 风险与对策

| 风险 | 对策 |
|---|---|
| 小样本下代理误导 | Dirichlet 恒占 20%；κ>0；保留 Rung 2 纯 CMA 对照 |
| 专家数 N > 30，GP 失效 | 自动切 RF 或 TPE（nevergrad 一行切换） |
| dev 集很小、分数噪声大 | LCB 复核 + 单独报告 val-test gap |
| 被质疑"改了搜索空间" | 附录给等价性证明（§0 的归纳论证），强调协议不变、合并次数反而更少 |
| 被质疑"hybrid 无归因" | §3.1 阶梯就是 factorial ablation |

---

## 5. 论文贡献框架

1. **分析**：证明 Model Swarms 的权重空间 PSO 隐式等价于 N 维仿射系数
   搜索（附录引理 + λ 轨迹可视化验证）；
2. **方法**：SGMS——三路提案 + 代理预筛 + LCB 复核的模型群体搜索，
   每个组件对应 PSO 一个已诊断弱点；
3. **实证**：等预算、同专家池、同评估器下系统优于 PSO，
   并给出组件级消融。

与 RMM/PPO 路线的关系：本方案不训练策略、无 RL 调参负担，
是"最小可行且最可能赢 PSO"的路线；若后续要做策略迁移，
SGMS 的 archive 可直接作为 PPO 的预热数据。
