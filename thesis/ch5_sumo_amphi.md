# 第 5 章 探索性扩展：稀疏支撑搜索与试探式路由

## 5.1 与论文主线的关系

RACER 是本文主线；SUMO 和 AMPHI 是围绕同一 factor 合并空间进行的探索性扩展，不再与 RACER 并列包装为已经充分验证的第二套冠军方法。十个原始专家对应十维系数；支撑 S 是其中非零或超过事前阈值的维度集合，而不是从二十个独立专家中选择。

对支撑 S 和权重 w，可将搜索写为

\[
 \max_{S\subseteq\{1,\ldots,10\}}\max_w f(\operatorname{embed}(S,w)).
\]

这只是问题分解，并不能推出实际最优解必然只有四到七维。该范围是启发式搜索配置；历史 `g_alpha` 与最终物化 alpha 未必相同，不能由部分搜索期冠军把它提升为全部最终模型的结构定律。清零一个 factor 系数会同时移除自项及其所有交叉项，所以支撑收益还与尺度、交叉项纠缠。

## 5.2 SUMO 的设计动机

SUMO 将候选生成分为支撑变更与支撑内连续步进。上层可通过加/减/换专家、精英共现和专家互补性启发式生成支撑；下层通过双向探针、方向记忆和步长适应调整系数。部分版本增加专家对种子，另一些显式关闭。它继承粗评/完整 dev 与终选流程的某些工程部件，但不能只因“继承”就假定已经有独立 holdout 或相同成本。

所谓互补性若来自 A/B factor 的余弦而非真实 ΔW 行为，需要明确参数化依赖。探针每次调用含多少样本、多领域如何聚合、是否复用相同题目及启动成本，都需由账本证明；不宣称它天然比经典优化器信息密度高三倍，也不宣称经典探针机制为首次发明。

## 5.3 按真实配置拆分历史结果

### 5.3.1 不能构造统一“SUMO 十二任务表”

显式清单记录 17 个 SUMO run，按 args 中的 pair-seed 开关及批次分为四个 cohort。配置标签是 **日志可知性**，不是据当前源码反推版本哈希。对配置不同的行，不计算跨组平均或跨版本双 seed 稳健性。

| cohort | run 与任务范围 | 配置事实 | 日志 test |
|---|---|---|---|
| `sumo_original` | `sumo_{nlgraph,gsm8k,hsw}_s42` | args 没有 `no_pair_seed` 字段 | nlgraph .371；gsm8k .469；hellaswag .595 |
| `sumo2_pair_enabled` | `sumo2_{nlgraph,gsm8k,legal}_s42` | `no_pair_seed=False` | .371；.419；.553 |
| `sumo2_no_pair` | `sumo2_{science,kcross,mmlu,mmlu_pro,truthfulqa,culture,medical,abstainqa}_s42` | `no_pair_seed=True` | .304；.393；.576；.250；.425；.539；.404；.161 |
| `sumo43_no_pair` | `sumo43_{nlgraph,gsm8k,hsw}_s43` | `no_pair_seed=True`，后续 seed43 批次 | .371；.427；.582 |

所有目录均在 `search/` 下并带 `_218keg` 后缀；完整 args、来源行、SHA 和缺失项见 `experiments/legacy_manifest.json`。`tools/thesis_results.py` 可重建各 cohort，不把其中 test 更好的 .469 替换进 pair-enabled 的 .419。

### 5.3.2 校正结论

原始 gsm8k 的 .469 是一项真实 legacy 结果，但它不能证明“可组合任务普遍获益”或跨配置双 seed 稳健。将 .469 与后续 .427 平均得到 .448 在算术上成立，在协议上不是匹配复制；更不能拿 RACER seed42 .435 冒充其双种子均值 .4215。pair-enabled seed42 的 .419 同样必须保留，不能因不利而隐去。

nlgraph 的多条 .371 提示该探索机制在这些配置下未改善起点，但不足以证明任务本质“不可组合”。SUMO 没有一套冻结配置覆盖十二任务，因此旧稿 .420 的“全谱均值”撤回，相关跨算法全谱冠军表一并删除。

## 5.4 可组合性指标尚未验证

专家对/最佳单专家比值只是一种候选诊断，两个最大值有选择偏差，factor midpoint 还改变有效 ΔW 强度。旧稿把比率 .937 归入 `ρ≥.95` 是自相矛盾；本文不再用该阈值判类，也不把两个任务上的观察上升为可组合性理论。

小 dev 粗评未能区分两个例子，并不从理论上证明所有廉价路由都不可能。要检验指标，应固定开发任务、阈值和误判成本，在未参与阈值调节的任务上评估；尺度配平先于此工作。

## 5.5 AMPHI 的真实运行与缺失反事实

AMPHI v3 不是虚构的实验。父 run 日志明确保存启动 SUMO、观察迭代、KEEP 或 SWITCH 到 RACER 的事件。例如 `search/amphi_nlgraph_218keg/log.txt:4` 在 iter8 记录 SWITCH，已花约 43 等效、给 RACER 约 67 等效（这是历史计数代理，并非统一真实成本）。

清单固定以下十二条父 run 的路由事件，不混入 test 最佳重跑：

| 路由 | 任务 | 父 run 标签 |
|---|---|---|
| SWITCH | nlgraph、legal、kcross | `amphi_nlgraph`、`amphi12_legal`、`amphi12_kcross` |
| KEEP | gsm8k、science、culture | `amphi_gsm8k`、`amphi_science_f`、`amphi_culture_f` |
| KEEP | hellaswag、mmlu、mmlu_pro、truthfulqa、abstainqa、medical | 对应 `amphi12_hsw/mmlu/mmlu_pro/truthfulqa/abstainqa/medical` |

这是 **3 SWITCH + 9 KEEP 的决策分布**，不是“12/12 正确率”。日志中的停滞条件就是触发规则本身，不能再把规则满足当成正确性的 oracle。每个任务还缺同一初始状态、相同剩余预算下 KEEP 与 SWITCH 两分支的反事实结果，故工具将 `counterfactual_accuracy` 设为 null，不生成路由准确率或完整 final-test 主表。

KEEP 不额外执行一个独立探测阶段，不代表路由零成本：前八轮、父进程协调、观察与选择机会成本仍属于总预算。SWITCH 的两段搜索初始化、终选和已花成本是否严格按统一预算执行，也尚需验收。

## 5.6 该章的保留价值与后续条件

可保留的贡献是：双层表示的工程实现、配置变化的可追溯结果、以及真实试探路由事件。其地位是提出假说和展示失败模式，而非提供第二套已成立的理论。

升级证据需依次完成：尺度/方向配平；单一冻结 SUMO 配置的多 seed；以开发集决策、独立最终集评估的两分支预算匹配反事实；包含初始化、探测、失败和终选的完整成本账本。上述不在首批自动扩跑范围内，本文不以单次历史高点替代这些证据。
