# 论文骨架与证据合同

## 标题

**面向 LoRA Factor 合并的多提案竞速搜索与可追溯评测：RACER**

英文候选：RACER: Multi-Proposal Racing Search and Traceable Evaluation for LoRA Factor Merging

## 摘要草稿

模型合并系数的黑盒搜索受参数化、开发数据有限性和调用成本共同制约。本文以 Qwen2.5-1.5B-Instruct 与十个原始 LoRA 专家为对象，回答一个实践问题：**昂贵的合并系数搜索什么时候值得，预算究竟买到了什么？** 现有物化分别混合 A/B factor，有效模型增量包含交叉项并随尺度平方变化，方向与尺度收益必须分开诊断。我们首先建立显式历史 run 清单与只读重建工具，修正 PSO 领域汇总失配（legacy 轨道，描述性）；随后以冻结协议 racer_frozen_v1 执行预注册 240 run 正式矩阵——冻结评分、等预算 B_t=110×D_t、独立终选（完整 dev 首次观测 raw argmax）、逐调用账本与逐 run 证据链验证（源码/数据/专家/基模 SHA-256）。主比较（9 任务 × 4 方法 × 3 seed）的答案是条件性的：宏平均 RACER .4245 > PSO .4161 > CMA .4107 > random .3955，仅对 random 显著（+2.90pt, t(8)=2.55，bootstrap CI [+1.15, +5.29]）；对 CMA 的 +1.38pt 集中于 gsm8k 单任务（去掉后 +0.29pt），但该任务内三 seed 全胜且单因子消融恰在此处显示每个组件都不可省（−3.4~−6.8pt）——搜索收益是搜索器×任务匹配的产物。gsm8k 类生成式任务上搜索回报真实；mmlu_pro/medical/truthfulqa 上任何搜索器与 random 差 ≤1.0pt，预算应省；组件价值随任务画像反转（medical 上取消竞速又便宜 31% 又更好，plateau 全程零贡献）。方法学贡献：预注册+证据链验证的合并搜索实验框架，与按任务画像如实报告负结果。局限：3 seed 功效（任务级显著需约 47 任务）、单基座单专家池、历史 test 曾用于开发、D 组身份拼接披露。

## 章节合同

| 章 | 实质内容 | 可写结论 | 不能写的旧结论 |
|---|---|---|---|
| 1 绪论 | 问题、RACER 主线、证据审计 | 搜索器与协议必须同时研究 | 半预算零崩溃冠军、两项创新均已完全验证 |
| 2 相关工作 | 配方/参数化/搜索器/评测差异 | 对照需统一空间及成本 | 第一个、所有先行工作从未研究、生态位已证空置 |
| 3 诊断 | 十专家 factor 二次映射；误差三分；数据重叠；PSO H；kcross 重叠冻结实证 | 数学事实与可追溯的错误校正 | 三定律、线性增量仿射包锁死、复评必减半方差 |
| 4 RACER | **4.5 冻结矩阵（主结果）**：240 run 预注册、身份冻结、证据链验证；4.3–4.4 legacy 分轨 | 宏平均最优且 vs random 显著、vs CMA/PSO 方向一致未显著；组件价值条件化；负结果如实 | 任务级显著获胜、竞速普遍有益、所有部件不可拆、plateau 有贡献 |
| 5 探索扩展 | SUMO 四配置组；AMPHI 3 SWITCH/9 KEEP；缺反事实 | 支撑/路由假说与实际运行事件 | 混配置全谱均值、跨版本双 seed 稳健、路由100%正确 |
| 6 总结 | 双轨证据、冻结矩阵核心数字、局限（功效/身份披露/外部范围）、下一闸门 | 功效补强与外部有效性是独立后续计划 | 把 dev–test 差异全归分布漂移、把 racer_eval_v1 当官方复现、D 组与 AC 身份相同 |

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
- 负结果必须保留：truthfulqa random 最优、kcross 搜索负增益、medical full_eval 反超、nlgraph no_rung2/standard_SH 反超、plateau 零贡献、固定 scale 网格无稳定收益。
- D 组身份拼接披露（`racer_campaign_v1_D_disclosure.md`）写入 4.5.4 与 6.2；不得写"单一连续身份"。
- 表格更新经 `tools/racer_results.py` 重跑核对，不手工堆数值；`tests/test_racer_results.py` 锁定工具契约。

## 补充证据的上限

- SUMO：original/no-pair 字段缺省批次、sumo2 pair-enabled、sumo2 no-pair、sumo43 no-pair 分组；.469 与 .427 不能作为同源码双 seed 平均。
- AMPHI：十二父 run 为真实事件，正确率缺 oracle 和预算匹配反事实；不否认真实执行，也不把决策分布当性能表。
- SMM：来源 JSON 未存预算/重复数/种子列表，当前 driver 是本地简化 SMMBenchPS 引擎且第三路随机；不称完整 RACER/官方 Sep-CMA 冠军。
- 既有消融已由冻结矩阵 B 组（108 run）以等预算单因子设计取代 legacy 探索性消融的因果排名。

## 后续实验闸门

冻结矩阵已完成预注册范围，无自动续跑。后续独立计划：功效补强（整批重开新 cohort）、第二基座/专家池/未见测试来源、官方对手（SMM 完整协议/NGOpt）、无 plateau 的 racer_frozen_v2 提案（重走冻结流程）。细则见 `doc/experiment_plan.md` 第 8 节。本阶段不创建绘图代码，不把图表占位当实验结果，不联网、SSH、提交或发布私有文稿。
