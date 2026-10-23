# RACER 结论—证据—实验验收表

## 1. 首批范围与证据分轨

本计划承接已批准的 `warm-wandering-candle.md`。阶段 A 负责论文、显式 legacy 清单与 CPU 重建测试；阶段 B 负责评分/切片纯函数及评测集成；阶段 C 负责有硬上限的 dev 尺度 pilot。A 的完成不等于 B/C 已完成，也不授权自动扩大任务矩阵。

证据必须保留三个层次：

1. **legacy 原始输出**：旧日志原地不改，旧评分缺陷仍是事实。
2. **legacy 聚合校正**：仅将同一最终 dev 选中粒子的领域分项恢复 H，或由该粒子保存的 scores 恢复平均。`legacy_aggregation_corrected_v1` 不称 strict。
3. **新冻结协议**：明确各 scoring/slicing/aggregation/selection 版本、raw 档案及新唯一 run 身份；不得无标签拼接进旧表。

## 2. 可执行验收矩阵

| 拟检验结论 | 已有证据 | 主要缺口 | 下一实验/工程 | 验收与失败判定 |
|---|---|---|---|---|
| RACER 历史宏平均较高 | 显式84主表run；.4439167 vs PSO .4327983、CMA .4363333 | seed、成本、评分、终选不匹配；非独立任务 | 先重建历史，之后批准matched比较 | 无扫描/按test挑run；逐行路径、行、SHA可核验；不能写显著性 |
| PSO 领域旧汇总失配 | 四领域日志两分项与最终dev粒子 | 只修汇总，搜索轨迹仍legacy | A按H校正；后续PSO返回值接线 | science .2583263191、legal .5886990802、medical .4284014002、culture .5532408495；tie首数字索引 |
| factor尺度可能解释部分收益 | A/B分别线性物化，ΔW(cα)=c²ΔW(α) | 方向、尺度、交叉项混杂 | 首批dev尺度pilot | 报告全20配置或partial；效应正负皆可，不以涨分为成功条件 |
| RACER竞速降低成本 | full/coarse历史计数 | 样本长度/二级/失败/加载未统一 | 新逐调用账本与matched全评对照 | 真实样本数、墙钟、失败和终选均计；非“预算参数相同即公平” |
| 复评缓解执行敏感性 | 同题重复的机制动机 | 不独立抽样，旧RF无真实holdout | 互斥selector、固定题目身份、untouched holdout | search/final alpha与selector分别保存；同题复评不称减半抽样方差 |
| legal/kcross泛化 | 当前数据精确题干交集审计 | 历史数据哈希缺失；实例族共享 | 后续按来源/族去重与新最终集 | legal不计独立泛化；kcross不误称全整题泄漏；当前快照不冒充历史身份 |
| 单个RACER部件必要 | 旧探索消融设计史 | 分支反馈、尺度、初始化、预算未验收 | full-CMA/no_de/no_sum等先分支验收 | 相同空间/尺度/候选与记账，不由一次失败证明必要性 |
| SUMO支撑搜索有益 | 17个run按4配置组记录 | 混版本、非匹配seed、全谱缺配置 | 后续冻结单配置后matched支撑消融 | 不混原始.469和后续.427算稳健双seed；不报伪全谱均值 |
| AMPHI路由改善效用 | 12真实父run：3 SWITCH/9 KEEP | 无同预算两分支反事实/oracle | 后续预注册开发路由+独立终评 | 决策分布不是准确率；失败、启动与试探成本全部计入 |
| SMM外部引擎收益 | 本地简化SMMBenchPS聚合JSON | 预算/种子/代理哈希缺；非官方对手 | 后续独立冻结PS实验和正规优化器 | 不宣称完整RACER/全SMM冠军，不从当前默认值补造历史元数据 |

## 3. A 阶段：显式清单与重建工具

`experiments/legacy_manifest.json` 共113行：84主表、17 SUMO、12 AMPHI。每行记录 args、名义/日志seed来源、metric、cohort/config、independence、unknown版本与缺失元数据、sources及expected重建值。清单不靠目录名猜方法；目录名只作为已冻结的路径。

`tools/thesis_results.py` 默认只输出主表，可 `--output` 新建完整 JSON 报告；拒绝既有目标、search/内写入、路径穿越/通配符/逃逸符号链接、来源哈希变化、重复任务方法seed、不完整cohort、无标签混协议或混SUMO配置。不会执行 `eval`、模型加载、目录全扫或按test排序择优。工具可直接生成论文主表，不需要手抄数字。

RF和CMA等以final metrics来源行重建；PSO两项缺最终行则按scratchpad最终个人最优dev选择16/0粒子再读对应scores。所有实验数据读取的Python命令必须带 `-I`。

```bash
/data/lvliping/miniconda3/envs/swarm/bin/python -I tools/thesis_results.py
/data/lvliping/miniconda3/envs/swarm/bin/python -I -m unittest discover -s tests -p 'test_thesis_results.py' -v
```

CPU测试验收包括恶意字面量不执行、H边界、首数字索引tie、重复/混配置/标签拒绝、真实六方法均值、SUMO与路由范围、数据题干审计、正文表格一致、新输出不可覆盖和旧证据SHA未改。标准库测试不声称执行了官方JSON Schema引擎；schema提供机器可读结构，reader另外执行跨行语义验证。

## 4. B 阶段：strict 的准确解释

由相应负责人实施评分/切片集成，本阶段文稿不替其改代码。验收应覆盖 MC 明确格式及冲突、数值完整token（gold 2不匹配12/20）、负数/小数/千位、非零切片起点、越界/空集先拒绝、领域独立记录与H(0,0)=0，以及legacy黄金行为保留。strict拒绝只优化某一领域的component override，而非静默忽略。

strict 是 **conservative grammar**，不是官方评分器的等价复现。明确格式外的答案、单位/解释不被grammar支持的输出可能invalid；需报告invalid率、分任务raw例子和规则版本，不能暗中改规则追分。新dev/test对同raw使用一致规则，但旧raw缺失时不能声称已重评历史。

**当前生成器限制**：仅返回decoded文本，尚无完整停止原因/token数。因此只能拒绝语法不完整或歧义输出，不能声称识别所有max-token截断；一个被截断但语法看似完整的答案可能无法判别。后续需要生成器记录stop reason/token count才能审计该问题。

strict元数据、逐题ID/hash/raw/parsed/gold/score及生成配置必须能在独立run目录追溯。依赖惰性加载和CPU小tensor物化测试只验证相应接口，不替代真实离线模型冒烟。不得因纯函数/集成测试通过便称RACER/PSO终选或硬预算已经修复。

## 5. C 阶段：固定开发集尺度矩阵

| 项目 | 冻结内容 |
|---|---|
| 任务 | nlgraph、gsm8k |
| 数据 | 每任务50条dev；明确切片起点与ID；不调用test |
| 方向 | 按名称排序的第一个原始专家onehot；十维均匀系数向量（每维.1） |
| 专家身份 | 执行前写出10个确切名称/顺序、模型与adapter身份；不能按历史test选方向 |
| 尺度c | [.5,.75,1,1.25,1.5]；有效ΔW强度倍数为[.25,.5625,1,1.5625,2.25] |
| 矩阵上限 | 2×2×5=20配置；最多1000条生成；首次加载/onehot冒烟复用其中配置 |
| 资源 | 至多本机一张空闲卡；从首次模型加载起累计占卡墙钟≤2 GPU小时，失败/重试全部计入 |
| 模型 | 仅现有离线Qwen snapshot；先检查完整性；缺资源、缺文件、加载失败即blocked/failed，不联网补下载 |
| 安全 | 默认dry-run，显式execute；新唯一run目录，已存在拒绝；不执行会rm-rf同名run的旧队列 |
| 档案 | 每候选完整配置、raw、解析、逐题得分/ID、实际调用样本数、耗时、终态和失败记录 |

pilot目标是 **尺度敏感性**，不是预期必涨。允许观察到降分、无变化、格式invalid增多或资源失败；这些都应保留并报告。不能通过选择最高test或只展示成功候选制造收益。该数据来自反复使用过的dev，不是独立泛化证据，更不能单独证明PSO+scale追平、RACER机制成立或全局尺度最优。

只有同一当前run内完整成功且全部身份一致的记录才能跳过重复调用；失败不能当缓存命中。停止只能作用于自己启动的作业，不管理他人任务。partial必须报告完成数与已耗预算；超过硬上限不自动延长。

## 6. 停止点与扩展闸门

首批结束或阻塞时报告实际成本、完成矩阵、invalid率、尺度效应、协议遗留问题，再决策。不会自动启动“4任务×4方法×3seed”的48次扩展。

扩大前必需：互斥终选与真实holdout、PSO域级返回及tie、各搜索器统一初始化/空间/成本/终选、full-CMA/no_de/no_sum分支验收。之后才可批准比较PSO+scale、RACER、CMA、matched Random。第二专家池、新测试来源、delta merge、SUMO/AMPHI反事实和官方SMM对手另行立项。

阶段A整理不创建绘图代码、不运行GPU/SSH/网络、不commit、不发布私有文稿；既有日志、模型和其他用户改动保持原状。阶段B/C的实际验收另行记录，不能将A阶段完成视为全部实验完成。

## 7. 首批实际执行状态（2026-10-21）

- 协议、接线、物化、driver和清单共103项CPU测试通过；严格评测真实离线导入已验证，无云/reward初始化。
- 已执行批准的真实pilot，run为 `runs/scale_pilot_20261021T090325_1282358/`；3/20配置成功、150条dev、64.190秒，终态blocked。前三项均为nlgraph onehot，c=1/.5/.75分数.26/.22/.22；不据partial宣布最优尺度或算法收益。
- 第四配置前GPU空闲检查未通过；失败时原始遥测未落盘，尚不能确认是释放尾窗或其他占用。未重试、未切卡、未启动剩余17项或扩展搜索。
- 详见 `doc/scale_pilot_report.md`，包含invalid率、实际来源、未完成项及需要重新批准的恢复闸门。
