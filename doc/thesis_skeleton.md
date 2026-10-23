# 论文骨架与证据合同

## 标题

**面向 LoRA Factor 合并的多提案竞速搜索与可追溯评测：RACER**

英文候选：RACER: Multi-Proposal Racing Search and Traceable Evaluation for LoRA Factor Merging

## 摘要草稿

模型合并系数的黑盒搜索受参数化、开发数据有限性和调用成本共同制约。本文以 Qwen2.5-1.5B-Instruct 与十个原始 LoRA 专家为对象，研究多提案与分层评测的 RACER 搜索流程。现有物化分别混合 A/B factor，因此有效模型增量包含交叉项，并随整体系数尺度平方变化；该性质要求区分方向搜索与尺度收益。我们建立显式历史 run 清单、安全解析工具和 CPU 回归，将 PSO 领域指标按同一 dev 选中粒子的两个 test 分项恢复调和均值。在固定十二任务历史表中，RACER 双 seed 宏平均 .4439167，校正 PSO .4327983，CMA .4363333。该表是 legacy 描述性证据，而非严格等预算、显著性或独立泛化证明：legal 题干非独立，kcross 有共享题干/实例族风险，评分与终选仍有协议债务。SUMO 按日志配置分组，AMPHI 报告真实路由事件而不报告无反事实支持的正确率，SMM 仅作简化 PS 引擎的补充。下一验证优先检验 dev 尺度敏感性及冻结评分、选择与成本协议。

## 章节合同

| 章 | 实质内容 | 可写结论 | 不能写的旧结论 |
|---|---|---|---|
| 1 绪论 | 问题、RACER 主线、证据审计 | 搜索器与协议必须同时研究 | 半预算零崩溃冠军、两项创新均已完全验证 |
| 2 相关工作 | 配方/参数化/搜索器/评测差异 | 对照需统一空间及成本 | 第一个、所有先行工作从未研究、生态位已证空置 |
| 3 诊断 | 十专家 factor 二次映射；误差三分；数据重叠；PSO H | 数学事实与可追溯的错误校正 | 三定律、线性增量仿射包锁死、复评必减半方差 |
| 4 RACER | 三路提案、竞速、版本化终选；自动生成主表 | 固定历史名单中平均较高、值得继续验证 | 显著获胜、严格预算因果、所有部件不可拆、十六失败证明局部最优 |
| 5 探索扩展 | SUMO 四配置组；AMPHI 3 SWITCH/9 KEEP；缺反事实 | 支撑/路由假说与实际运行事件 | 混配置全谱均值、跨版本双 seed 稳健、路由100%正确 |
| 6 总结 | 已完成证据、协议债务、下一闸门 | 先尺度诊断再匹配对照 | 把 dev–test 差异全归分布漂移、把新 strict 当官方复现 |

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

工具默认 stdout；`--output` 只新建 JSON、不覆盖已有文件、不写 search/。第4章主表由 `markdown_table` 生成，回归测试检查正文与工具逐字一致。其余表若扩充必须先加入独立 cohort 和来源，而不是手工堆数值。

## 补充证据的上限

- SUMO：original/no-pair 字段缺省批次、sumo2 pair-enabled、sumo2 no-pair、sumo43 no-pair 分组；.469 与 .427 不能作为同源码双 seed 平均。
- AMPHI：十二父 run 为真实事件，正确率缺 oracle 和预算匹配反事实；不否认真实执行，也不把决策分布当性能表。
- SMM：来源 JSON 未存预算/重复数/种子列表，当前 driver 是本地简化 SMMBenchPS 引擎且第三路随机；不称完整 RACER/官方 Sep-CMA 冠军。
- 既有消融/对局尚未进入匹配清单，不再保留强因果表。失败探索可作为设计史，不能证明局部最优。

## 后续实验闸门

首批仅 `nlgraph/gsm8k × 固定onehot/均匀方向 × [.5,.75,1,1.25,1.5]` 的 dev 尺度敏感性；至多20配置、1000生成、单卡累计2 GPU小时。效应可能正、负或不显著，不预设涨分。该工作包的 driver/实际 run 不由本文稿任务执行。

之后先报告，再决定统一 selector/空间/初始化/预算后的多 seed 搜索比较。细则见 `doc/experiment_plan.md`。本阶段不创建绘图代码，不把图表占位当实验结果，不联网、SSH、提交或发布私有文稿。
