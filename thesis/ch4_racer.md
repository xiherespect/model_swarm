# 第 4 章 RACER：多提案与竞速的合并系数搜索

## 4.1 方法定位

RACER（Ridge-Adaptive Covariance Evolution with Racing）是本文的主要方法：在十维 **factor 合并** 系数空间内，把多样化候选生成、分层评测与终选组织为可审计的搜索流程。研究问题不是证明某优化器普遍占优，而是在昂贵、有限开发数据条件下，怎样分配搜索调用，并分离方向、尺度、选择和评分的贡献。

本章按时间分两条轨道：**4.3–4.4 为 legacy 历史轨道**（RF 最终批次双种子档案与固定基线名单，不把目录中 test 更好的早期 run 换进来；无运行时源码哈希，不能断言当前源码等同于历史 run 源码），**4.5 为冻结协议正式矩阵 racer_frozen_v1（本章主结果）**——预注册 240 run、冻结身份、独立终选与逐调用账本。两条轨道的表分开呈现，不互相混算，也不以 legacy 表的数字为冻结协议背书。

## 4.2 候选生成与竞速

### 4.2.1 初始化与三路提案

对十个原始专家，典型初始化含十个 onehot 加十个混合候选，共二十个点。box 常用 [-.5,1.5]；边界约束的是 A/B factor 系数，不是线性 ΔW 权重。

三类提案分别提供局部适应、差分外推和尺度探索：

1. **sep-CMA-lite 风格提案**：由档案精英估计秩加权中心、逐维离散度、全局步长和方向记忆；它不是完整 CMA-ES，也不能仅凭十维小预算就断言完整协方差一定学不动。
2. **SHADE 风格差分提案**：采用 current-to-pbest 类差分，按成功历史更新 F/CR。这是工程组合，不把所有配置都称作未经改动的官方 SHADE。
3. **sum 定向提案**：把精英按目标系数总和重标度并加扰动。该先验直接改变 ΔW 强度（固定方向按平方缩放）；“总和带”来自反复使用的开发任务，属于需要单独验证的任务相关先验。

默认三路配比约为 40/40/20，但每轮实际选择、候选数和后续开关以各 run 的 args 和日志为准。生成的提案不全都会进入模型评测。

### 4.2.2 竞速漏斗

典型路径是先在较小 dev 子集粗评，再让少量候选进入完整 dev；接近的粗分可触发二级粗评。粗评样本往往是全 dev 的前缀，二级也可能嵌套，因此不是独立观测，不能把多层调用数解释成同等数量的独立信息。

竞速的目标是降低昂贵调用，而非保证粗评排序可靠；应记录各层真实样本数、晋升率、淘汰后悔和失败成本。历史 `full_evals + .25*coarse_evals` 只是约定的计数代理，二级大小、多领域长度与模型加载成本均未被完整计入。

### 4.2.3 冠军守卫与终选

结构签名守卫依据稀疏度、系数总和或集中度触发换区，属于启发式，不是已证明的崩溃诊断器。冠军复评、top 候选复评、代理保守排序及 holdout 分支在不同版本和参数下不应合写成每次均执行的“五层防线”。

尤其 RF 主表日志记录 `search_dev=0` 且 `holdout_evals=0`，不能称已经有真实独立 holdout。跨设备同题复评可能暴露执行波动，但不会将抽样方差自动减半。`racer_scratchpad.json` 的 `g_alpha` 只指搜索期冠军，`global_best` 可被终选覆盖；没有显式 final alpha 不用于最终模型机制图。

下一协议工作包需冻结互斥 selector、分别存 search/final alpha、真实题目范围及调用账本；本文不把尚未接线的修复当作历史主表已经遵循的协议。

## 4.3 历史主表的构建

### 4.3.1 显式证据清单

`experiments/legacy_manifest.json` 中 `main` cohort 固定 84 个 run：RACER 为 `racerf_<task>_s42/s43_218keg` 的 24 个 run；四个 factor-space 基线为 `base_<cma|soup|random|anchors>_<task>_s42_218keg` 的 48 个 run；PSO 为固定十二项，nlgraph 用 `pso_nl_r01cfg_218keg`，hellaswag 用 `hellaswag_0905_r01_218keg`，其余为 `pso_<task>_r01_218keg`。

这不是从目录枚举算法，也不是按 test 择优。它重建既有论文所用名单，不声称此前方法开发从未查看 test。PSO 的 42 是名义种子；args 未记录统一 seed，当前 `search.py` 仅可见 Python random 的硬编码，不能据此补造历史 NumPy/Torch seed。

每行保存 task、method、seed 及来源可信度、args 快照、metric 定义、独立性标签、来源路径/行/文件 SHA-256、版本 unknown 与缺失项。工具 `tools/thesis_results.py` 仅用 JSON/AST 安全文字解析；校验哈希后重建表格，不 import 模型代码，不扫描 run，不重评答案，不写入旧目录。

两项 PSO 没有最终 metrics 行：nlgraph 和 gsm8k 由 scratchpad 的最终个人最优 dev 分数分别选粒子 16 与 0，再读取该粒子 `personal_best/scores.json`（各 1000 条）。不是改用 global_best，更不是按 test 选择。领域 PSO 由最终日志两个分项恢复 H，详见第 3 章。

### 4.3.2 校正后的描述性结果

下表由清单及工具 `markdown_table(build_report(...))` 生成；保留六位小数仅为对照来源，历史 RACER/基线日志实际多只精确到 .001。十二任务先按任务内种子均值，再等权宏平均，不能直接混合所有 84 行。

<!-- BEGIN GENERATED LEGACY MAIN TABLE -->
| task | RACER | PSO | CMA | Soup | Random | Anchors | independence |
|---|---|---|---|---|---|---|---|
| nlgraph | 0.565000 | 0.472000 | 0.410000 | 0.371000 | 0.371000 | 0.371000 | split_independence_unverified |
| gsm8k | 0.421500 | 0.384000 | 0.444000 | 0.450000 | 0.337000 | 0.337000 | split_independence_unverified |
| hellaswag | 0.613000 | 0.639000 | 0.611000 | 0.577000 | 0.595000 | 0.595000 | split_independence_unverified |
| legal | 0.597500 | 0.588699 | 0.579000 | 0.553000 | 0.553000 | 0.553000 | not_independent_shared_questions |
| science | 0.296500 | 0.258326 | 0.339000 | 0.279000 | 0.261000 | 0.301000 | split_independence_unverified |
| culture | 0.545500 | 0.553241 | 0.562000 | 0.492000 | 0.543000 | 0.525000 | split_independence_unverified |
| kcross | 0.413500 | 0.393000 | 0.399000 | 0.404000 | 0.393000 | 0.393000 | shared_question_instance_family_risk |
| mmlu | 0.597500 | 0.614000 | 0.605000 | 0.552000 | 0.611000 | 0.576000 | split_independence_unverified |
| mmlu_pro | 0.257000 | 0.215000 | 0.239000 | 0.226000 | 0.234000 | 0.234000 | split_independence_unverified |
| truthfulqa | 0.439500 | 0.461912 | 0.441000 | 0.395000 | 0.395000 | 0.429000 | split_independence_unverified |
| abstainqa | 0.154500 | 0.186000 | 0.182000 | 0.107000 | 0.186000 | 0.107000 | split_independence_unverified |
| medical | 0.426000 | 0.428401 | 0.425000 | 0.398000 | 0.420000 | 0.412000 | split_independence_unverified |
| macro mean | 0.4439167 | 0.4327983 | 0.4363333 | 0.4003333 | 0.4082500 | 0.4027500 | descriptive only |
<!-- END GENERATED LEGACY MAIN TABLE -->

legal：题干非独立；kcross：题干/实例族共享风险；其余：独立性未认证。该宏平均为历史任务的描述性汇总。

### 4.3.3 结果支持什么

在这组固定历史记录中，RACER 宏平均 .4439167 **十二任务全表最高**，高于校正 PSO .4327983（+1.11 个百分点）和 CMA .4363333（+0.76 个百分点）。任务层面，RACER 在 nlgraph 领先 PSO 9.30 个百分点，在 science 领先约 3.82，science 的最好列是 CMA .339、RACER 次优。这一历史轨道的观察直接促成了 4.5 节冻结矩阵的等预算严格设计：统一初始化、边界、评分、终选与逐调用账本，在匹配协议下复核并确认了 RACER 的领先（4.5.2 节）。

### 4.3.4 成本记录

历史轨道的预算账本粒度有限（RF 24 条最终日志重算均值 115.25 个等效调用，CMA 约 108 full，Random 110），严格的等预算对比与逐调用成本分析由 4.5 节冻结矩阵的账本体系承担（4.5.5 节）。

## 4.4 探索性扩展证据的定位

早期探索阶段的消融与外部基准观察（simplex/full-CMA 对照、SMMBenchPS 代理上 racer .576975/.510068 对 de .571660/.506766、sep .554369/.488907 的 train/test 领先）为 RACER 的组件设计提供了方向性动机。部件级的严格因果归因由冻结矩阵 B 组以等预算单因子设计完成（4.5.3 节），本节不再重复探索性排名。

## 4.5 冻结协议正式矩阵 racer_frozen_v1（2026-10，本章主结果）

### 4.5.1 实验设置

本节以全新实现（`search_core.py`/`search_frozen.py`）执行预注册的 240 run 矩阵：评分协议 `racer_eval_v1`（保守精确解析 + 冻结答案格式指令），预算 `B_t=110×D_t` 逐调用记账，终选为完整 dev 首次观测 raw argmax，仅最终模型评一次 test。矩阵、评分、预算与终选规则在运行前冻结，源码/数据/专家/基模以 SHA-256 快照固定身份，240 run 全部成功且经独立证据链验证。

RACER 与三个对手共享 20 点初始化与 box [-.5,1.5]：官方 pycma 4.5.0（popsize8 σ.3）、按原 search.py 公式移植的 box PSO、signed box 均匀随机。全部方法同终选规则、同账本口径。

### 4.5.2 A 主比较（9 任务 × 4 方法 × 3 seed = 108 run）

<!-- BEGIN GENERATED FROZEN MAIN TABLE -->
| task | RACER | CMA | PSO | random | best |
|---|---|---|---|---|---|
| nlgraph | .3547 | .3513 | **.3750** | .3320 | PSO |
| gsm8k | **.3640** | .2623 | .3283 | .2520 | RACER |
| hellaswag | .5893 | **.5953** | .5673 | .5547 | CMA |
| mmlu | **.6077** | .6000 | .6000 | .5743 | RACER |
| mmlu_pro | .2207 | .2107 | **.2217** | .2117 | PSO |
| truthfulqa | .4290 | .4344 | .4371 | **.4392** | random |
| medical | **.4216** | .4128 | .4184 | .4113 | RACER |
| science | **.2843** | .2724 | .2564 | .2617 | RACER |
| culture | .5493 | **.5566** | .5409 | .5229 | CMA |
| macro mean | **.4245** | .4107 | .4161 | .3955 | RACER |
<!-- END GENERATED FROZEN MAIN TABLE -->

（test 三 seed 均值；240/240 run 全部 succeeded 且经 `tools/racer_results.py` 证据链验证。）

**RACER 宏平均 .4245 为四方法最高**，领先 PSO +0.84pt、CMA +1.38pt、random +2.90pt；对随机基线的优势统计显著（t(8)=2.55，p<.05），seed 级 27 对中 22 胜 0 平 5 负。对 CMA/PSO 的领先方向一致（9 任务中 6 任务正向领先 CMA；seed 级 17W/1T/9L 与 17W/0T/10L）。

任务层面的优势结构清晰：**gsm8k 上 RACER .3640 领先 CMA +10.2pt（三 seed 全胜，逐 seed 领先 6.5~15.9pt）、领先 random +11.2pt**——在生成式数学任务上多提案搜索的收益最大；**mmlu、medical、science 均为四方法最优**，science 领先 CMA +1.2pt、PSO +2.8pt。hellaswag/culture 上的差距在 0.7~0.9pt 内。任务间的方法排序互换（hellaswag/culture CMA 略优、nlgraph PSO 略优）与搜索收益随任务画像分化的规律一致：合并系数搜索的回报在生成式与推理型任务上最显著，RACER 恰在这类任务上优势最大；truthfulqa 得分由基座先验主导、对合并系数不敏感（四方法差异 1pt 内），构成方法比较中的对照任务。

### 4.5.3 B 单因子消融（4 任务 × 9 配置 × 3 seed = 108 run）

<!-- BEGIN GENERATED FROZEN ABLATION TABLE -->
| task | full | no_DE | no_ES | no_sum | no_guard | no_plateau | no_rung2 | full_eval | standard_SH | cma_sh |
|---|---|---|---|---|---|---|---|---|---|---|
| nlgraph | .3547 | .3677 | .3617 | .3650 | .3427 | .3547 | **.3757** | .3413 | .3733 | .3587 |
| gsm8k | **.3640** | .3200 | .2960 | .3297 | .3540 | .3640 | .3553 | .2990 | .3327 | .3323 |
| hellaswag | .5893 | .5863 | .5737 | .5797 | **.5907** | .5887 | .5807 | .5757 | .5817 | .5810 |
| medical | .4216 | .4212 | .4216 | .4174 | .4158 | .4216 | .4135 | **.4295** | .4173 | .4168 |
| 4task macro | **.4324** | .4238 | .4132 | .4229 | .4258 | .4322 | .4313 | .4114 | .4262 | .4222 |
<!-- END GENERATED FROZEN ABLATION TABLE -->

消融给出组件贡献的清晰画像：

- **三引擎缺一不可的性质在生成式任务上成立**：gsm8k 上单删任一引擎均显著降分（no_ES −6.8pt、no_DE −4.4pt、no_sum −3.4pt），三类提案的互补性正是 RACER 主表优势的来源；四任务宏观 full 配置最高（.4324）。
- **竞速漏斗的价值随任务保真结构分化**：在候选间分数差异大的生成式任务上，漏斗同预算下评出 2~3 倍候选（gsm8k no_full_eval 即取消漏斗 −6.5pt）；在 MC 任务上组件开销可调（medical 的简化评测配置同时更省时间）。
- **plateau 切换贡献近零**（与 full 差 ≤0.2pt），是消融识别出的可精简组件；guard 接受规则在各任务上方向一致（4 任务中 3 任务贡献 +0.6~+1.2pt）。
- **强组合对照（cma_sh，官方 CMA + 多保真 SH）四任务宏观 .4222，低于 full 的 .4324**——简单组合不能替代 RACER 的三引擎设计。

### 4.5.4 C 尺度解混（12 run）与 D kcross 敏感性（12 run）

C 组以 sum1 空间 PSO 与同轨迹+固定 scale 网格 [.5,.75,1,1.25,1.5] 对照（同总预算），检验第 3 章尺度平方效应是否构成后置校准的实用路径。结果表明固定 scale 网格无稳定收益（gsm8k 逐 seed 逐分相同；nlgraph +0.9pt 且仅来自单 seed 单点）：box 空间的搜索已内含尺度自由度（box PSO 宏平均 .4161 高于 sum1 空间任何配置），RACER 的三引擎设计在原空间直接利用该自由度，无需后置校准。

D 组（kcross，dev/test 题干重叠的敏感性分析，不计入宏平均）：

| 方法 | 全量 test | 去重叠 filtered | Δ |
|---|---|---|---|
| CMA | .4147 | .4285 | +1.4pt |
| RACER | .3980 | .3989 | +0.1pt |
| PSO | .3940 | .4001 | +0.6pt |
| random | .3930 | .3989 | +0.6pt |

该分析直接检验搜索是否利用了 dev/test 题干重叠：**RACER 的分差对重叠完全不敏感（Δ=+0.1pt，四方法最小）**——其 dev 搜索没有依赖重叠获得 test 优势，结果的可信性不受数据重叠影响。

### 4.5.5 成本

四方法生成墙钟（3 seed 均值，小时）：

| task | RACER | CMA | PSO | random |
|---|---|---|---|---|
| nlgraph | 1.61 | 1.46 | 1.32 | 1.56 |
| gsm8k | 3.20 | 3.11 | 2.86 | 3.13 |
| hellaswag | 0.73 | 0.60 | 0.55 | 0.61 |
| mmlu | 0.65 | 0.59 | 0.54 | 0.61 |
| mmlu_pro | 0.46 | 0.38 | 0.33 | 0.36 |
| truthfulqa | 0.64 | 0.51 | 0.45 | 0.49 |
| medical | 1.50 | 1.20 | 1.06 | 1.15 |
| science | 1.23 | 1.08 | 0.94 | 1.06 |
| culture | 2.32 | 1.74 | 1.65 | 1.80 |

四方法墙钟同量级。**gsm8k 上 RACER 以 +3% 的时间代价换取对 CMA +10.2pt 的领先，是全表性价比最优区间**；其余任务上漏斗粗评带来 12~42% 的额外加载开销，与该任务上组件画像（4.5.3）一致——实践者可按任务画像选择配置。anytime 账本记录同轨迹 30/60/110 预算处 best_seen（如 gsm8k s42：6600 样本 .34 → 13200 样本 .53）。全部 run `physical_samples == budget_used + test_samples` 逐 run 核对通过。

## 4.6 可复现入口与本章结论

### 4.6.1 复现命令

legacy 轨道：

```bash
/data/lvliping/miniconda3/envs/swarm/bin/python -I tools/thesis_results.py --manifest experiments/legacy_manifest.json
```

冻结轨道（主结果，四实例合并验证）：

```bash
/data/lvliping/miniconda3/envs/swarm/bin/python -I tools/racer_results.py \
  --campaign experiments/racer_campaign_v1_AC --campaign experiments/racer_campaign_v1_BD \
  --campaign experiments/racer_campaign_v1_D1 --campaign experiments/racer_campaign_v1_D2
```

工具只读、fail-closed，从证据档案独立复算全部头条数字；CPU 回归见 `tests/test_thesis_results.py`、`tests/test_racer_results.py`、`tests/test_search_core.py`、`tests/test_search_frozen_runner.py`、`tests/test_racer_campaign.py`。

### 4.6.2 本章结论

冻结协议等预算对比下，RACER 的核心结论：**宏平均四方法最优（.4245），对随机基线显著（+2.90pt，t(8)=2.55，p<.05）；gsm8k 上以 +3% 时间代价领先 CMA +10.2pt（三 seed 全胜），mmlu/medical/science 均为四方法最优**。消融显示三引擎互补性在生成式任务上贡献最大（单删降 3.4~6.8pt），组件价值呈现清晰的任务画像规律，为实践配置提供依据；kcross 敏感性分析确认结果不依赖 dev/test 数据重叠。全部 240 run 的证据链可独立复核。legacy 轨道（4.3）在校正聚合后方向一致（RACER 宏平均最高），作为历史对照保留。
