# 领域前沿档案（2026-09-21 三轮扫描汇总，论文 related work 素材）

## 一、合并搜索三条先行线（各有盲区，无人交叉对照）

| 线 | 代表 | 设定 | 盲区 |
|---|---|---|---|
| CMA-loss-极简预算 | LoRAHub (COLM'24) | 20 LoRA、NGOpt(40 evals)、5-shot CE loss+L1、box[-1.5,1.5]零初始化 | 无噪声防御、无优化器对照 |
| CMA-大预算-层配方 | Sakana EvolMerge (Nature MI'25) | 2-3 模型层间系数、1000+ evals | 同上；且预算是我们的 5-100 倍 |
| PSO-专家池 | Model Swarms (ICML'25) | 20 专家、准确率效用、~200 evals | 几何锁死+选择崩溃（我们诊断）；无优化器对照 |

**我们的位置**：三线交叉处的第一个受控研究（同预算 5 方法 × 12 任务 × 噪声取证）。

## 二、2026 新动向（均不构成必超基线，全部"引用+一句话切割"）

| 工作 | 要点 | 切割话术 |
|---|---|---|
| MOBO-Merge (arXiv 2608) | 合并权重=多目标 BO，Pareto 前沿，胜 random/grid | "多目标权衡 vs 我们的单效用搜索器科学；其基线仅 random/grid，我们提供更强的受控对照" |
| MERGEvolve (arXiv 2606) | "凸包外有高性能区"——独立佐证我们的仿射包几何！解法=全参数 ES(贵) | "他们逃出包外靠全参数噪声(重)，我们靠系数空间 box 松弛(轻)"；几何发现互相印证 |
| Inconvenient Truth (2026.03) + Hitit et al. (2025) | 系统评测：朴素 Task Arithmetic 常胜花哨方法 | 我们地形地图+噪声机理的理论化解释——intro 引用立"领域已有怀疑，我们给出解释" |
| LM-as-a-Judge 黑盒合并 (2025.09) | 进化搜索合并权重做裁判模型 | 不同效用函数，引用 |
| GimmBO / Fisher-BO 合并 | 人机交互/BO 变体 | 引用 |

## 三、基准动态（考场未变）

- FusionBench (JMLR'25)：评配方不评搜索
- MergeBench (2025)：领域 LLM 大规模合并评测——可选外部验证场，非必须
- 我们沿 Model Swarms 协议（原生设定）+ 比两者都细的噪声分析

## 四、必超基线终版清单（三轮扫描零新增）

✅ PSO(原版+强化)、CMA-ES(官方pycma)、Greedy Soup、Random、Anchors(Uniform/TA) —— 全部已在我们 12 任务表内被 RACER 覆盖或击败
🔶 可选：LoRAHub NGOpt(40)（半天工作量）
➖ 引用不对比：MERGEvolve(全参ES)、MOBO-Merge(多目标)

## 五、给审稿人的三行答案

1. "和 LoRAHub/Sakana 什么关系？" —— 他们证明进化合并可行(各自设定)，我们回答"该用哪个搜索器、为什么"——领域没人做过的受控对照
2. "和 MOBO-Merge 什么关系？" —— 多目标 Pareto 设定 vs 我们的单效用+噪声科学；其基线(random/grid)弱于我们的全套受控
3. "贡献是什么？" —— 诊断(几何/噪声/地形三定律) + 双算法(RACER 平铺搜索/SUMO 双层支撑) + 受控证据链(消融/对局/跨机复现)
