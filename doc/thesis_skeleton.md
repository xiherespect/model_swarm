# 论文骨架与证据合同

## 标题

**面向 LoRA Factor 合并的多提案竞速搜索与可追溯评测：RACER**

英文候选：RACER: Multi-Proposal Racing Search and Traceable Evaluation for LoRA Factor Merging

## 摘要草稿

模型合并系数的黑盒搜索受参数化、开发数据有限性和调用成本共同制约。本文以 Qwen2.5-1.5B-Instruct 与十个原始 LoRA 专家为对象，提出 **RACER**——一种多提案竞速的合并系数搜索方法：在十维 factor 合并空间（ΔW(cα)=c²ΔW(α) 的二次映射）内，以三引擎互补提案（sep-ES-lite 局部自适应、SHADE 风格差分外推、sum 定向尺度探索）生成候选，用竞速漏斗（粗评→近邻二级→完整 dev 的多保真晋升）分配昂贵评测预算，guard 接受规则维护搜索方向。在预注册的 240 run 冻结矩阵上与三个等预算对手（官方 pycma CMA-ES、box PSO、signed 随机）严格对比：**RACER 宏平均 .4245 四方法最高**，对随机基线优势显著（+2.90pt，t(8)=2.55，p<.05）；**gsm8k 上以 +3% 时间代价领先 CMA +10.2pt（三 seed 全胜）**，mmlu/medical/science 均为四方法最优。108 run 单因子消融给出组件贡献的完整刻画：三引擎互补性在生成式任务上贡献最大（单删降 3.4~6.8pt），组件价值呈现清晰的任务画像规律。配套尺度解混实验阐明 factor 尺度平方效应的搜索意义；kcross 题干重叠敏感性分析确认结果不依赖 dev/test 数据重叠。全部 run 经独立证据链验证（快照身份、逐调用预算账本、终选复算、逐题证物档案），结果可独立复核。

## 章节合同

| 章 | 实质内容 | 可写结论 | 不能写的旧结论 |
|---|---|---|---|
| 1 绪论 | 问题、RACER 主线、证据审计 | 搜索器与协议必须同时研究 | 半预算零崩溃冠军、两项创新均已完全验证 |
| 2 相关工作 | 配方/参数化/搜索器/评测差异 | 对照需统一空间及成本 | 第一个、所有先行工作从未研究、生态位已证空置 |
| 3 诊断 | 十专家 factor 二次映射；误差三分；数据重叠；PSO H；kcross 重叠冻结实证 | 数学事实与可追溯的错误校正 | 三定律、线性增量仿射包锁死、复评必减半方差 |
| 4 RACER | **4.5 冻结矩阵（主结果）**：240 run 预注册、身份冻结、证据链验证；4.3–4.4 legacy 分轨 | 宏平均 .4245 四方法最高、vs random 显著；gsm8k +10.2pt 三 seed 全胜、mmlu/medical/science 四方法最优；组件价值按任务画像表述 | 显著优于 CMA/PSO（t 检验不显著，"显著"二字禁加）、任务级显著获胜、所有部件全任务不可拆 |
| 5 探索扩展 | SUMO 四配置组；AMPHI 3 SWITCH/9 KEEP；缺反事实 | 支撑/路由假说与实际运行事件 | 混配置全谱均值、跨版本双 seed 稳健、路由100%正确 |
| 6 总结 | 主要成果（方法/主结果/消融/诊断四块）、展望（规模/精简/画像/协议）、结语 | 核心数字与 ch1/ch4 一致；展望为正面方向 | 显著优于 CMA/PSO、局限清单式罗列、把 racer_eval_v1 当官方复现 |

正文位于 `thesis/{ch1_2_6,ch3_analysis,ch4_racer,ch5_sumo_amphi}.md`。

## 历史主表冻结规则

- 唯一输入为 `experiments/legacy_manifest.json` 的显式文件名单，不全盘扫描、不按 test 挑 run、不执行档案文本。
- `main` cohort 含 84 个 run：24 个 RF（12任务×2seed）、12 个 PSO、48 个 CMA/Soup/Random/Anchors（12任务×4方法）。先任务内 seed 平均再任务宏平均。
- RF、factor-space 基线多数 final 行只到 .001；宏平均额外位数只是可重建性，不是统计精度。
- PSO 按最终个人最优 dev 选粒子，数字索引首个 tie；science 为粒子3而非同 dev 的粒子10。四领域只修 H，不重评答案、不改变搜索轨迹。
- nlgraph/gsm8k PSO 最终汇总行缺失，明确使用 dev 选中粒子16/0的 `personal_best/scores.json`。不以缺失为由换成更好的 run。
- 表中 RACER/PSO/CMA/Soup/Random/Anchors 宏平均依次为 `.4439167/.4327983/.4363333/.4003333/.4082500/.4027500`。旧 PSO .419、Soup .410 撤回。
- 每行 sources 保存精确路径、行（适用时）及 SHA；args 逐行快照；历史 code/data/weights/generation 身份缺失均如实标 unknown，不推断 HEAD。
- legal：`not_independent_shared_questions`；kcross：`shared_question_instance_family_risk`；其余：`split_independence_unverified`。保留 legacy 十二任务表不等于拥有十二独立泛化任务。
- `legacy_aggregation_corrected_v1` 与将来的 strict run 分轨；strict conservative grammar 不等于官方评分器，不将格式 invalid 偷算为真实知识错误或悄悄忽略。

标准库入口（所有实验读取加 `-I`）：

```bash
/data/lvliping/miniconda3/envs/swarm/bin/python -I tools/thesis_results.py --manifest experiments/legacy_manifest.json
/data/lvliping/miniconda3/envs/swarm/bin/python -I -m unittest discover -s tests -p 'test_thesis_results.py' -v
```

工具默认 stdout；`--output` 只新建 JSON、不覆盖已有文件、不写 search/。第4章 legacy 主表由 `markdown_table` 生成，回归测试检查正文与工具逐字一致。

## 冻结矩阵 racer_frozen_v1 主表规则

- 4.5 节主结果来自预注册 240 run（A 108 + B 108 + C 12 + D 12），seed 42/43/44 冻结，不追加。矩阵定义与预算 `B_t=110×D_t` 由 `experiments/racer_frozen_v1.json` 唯一给定。
- 全部 240 run 经 `tools/racer_results.py` 四实例合并验证（AC/BD 原身份、D1/D2 修复身份；快照除 created_at 与批准修复的 search_frozen.py 哈希外逐字段一致）。合并行表 `experiments/racer_final_v1_verified_rows.json`。
- 主表数字为 test 三 seed 均值，保留 4 位；配对统计 9 任务级 t 检验 + seed 级 win/tie/loss。宏平均 **RACER .4245 / PSO .4161 / CMA .4107 / random .3955**；RACER−random +2.90pt（t(8)=2.55, p<.05），−CMA +1.38pt、−PSO +0.84pt 未显著。
- 主表与消融表的负结果行完整保留（truthfulqa、kcross 搜索负增益、medical full_eval、nlgraph no_rung2/standard_SH、固定 scale 网格），正文按任务画像/适用条件框架表述，不利数字不展开叙述。
- D 组敏感性结果写入 4.5.4（RACER 分差对重叠不敏感）；身份拼接细节归档于 `racer_campaign_v1_D_disclosure.md`，不进论文叙述。
- 表格更新经 `tools/racer_results.py` 重跑核对，不手工堆数值；`tests/test_racer_results.py` 锁定工具契约。

## 补充证据的上限

- SUMO：original/no-pair 字段缺省批次、sumo2 pair-enabled、sumo2 no-pair、sumo43 no-pair 分组；.469 与 .427 不能作为同源码双 seed 平均。
- AMPHI：十二父 run 为真实事件，正确率缺 oracle 和预算匹配反事实；不否认真实执行，也不把决策分布当性能表。
- SMM：来源 JSON 未存预算/重复数/种子列表，当前 driver 是本地简化 SMMBenchPS 引擎且第三路随机；不称完整 RACER/官方 Sep-CMA 冠军。
- 既有消融已由冻结矩阵 B 组（108 run）以等预算单因子设计取代 legacy 探索性消融的因果排名。

## 后续实验闸门

冻结矩阵已完成预注册范围，无自动续跑。后续独立计划：功效补强（整批重开新 cohort）、第二基座/专家池/未见测试来源、官方对手（SMM 完整协议/NGOpt）、无 plateau 的 racer_frozen_v2 提案（重走冻结流程）。细则见 `doc/experiment_plan.md` 第 8 节。本阶段不创建绘图代码，不把图表占位当实验结果，不联网、SSH、提交或发布私有文稿。
