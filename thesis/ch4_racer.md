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

legal：题干非独立；kcross：题干/实例族共享风险；其余：独立性未认证。该宏平均包括这些历史任务，不能改称独立泛化均值。

### 4.3.3 结果支持什么

在这组固定历史记录中，RACER 宏平均 .4439167，高于校正 PSO .4327983（约 **1.11 个百分点**）和 CMA .4363333（约 **.76 个百分点**）。旧稿 PSO .419 和 Soup .410 不成立；Soup 按既有十二行实际为 .4003333。

任务层面，RACER 在 nlgraph 领先 PSO 9.30 个百分点，在 science 领先约 3.82，在 legal 领先约 .88；science 的最好列是 CMA .339，legal 的差异不具独立泛化意义。RACER 在 hellaswag、mmlu、truthfulqa、abstainqa 等列低于 PSO。不能把所有有利差异称为显著，把所有不利差异称为噪声。

这些结果支持“RACER 值得在匹配协议下继续验证”的描述性判断，**不支持** 严格显著、零崩溃、全面击败 PSO、独立泛化冠军或每个部件不可替代。RACER 双 seed 而基线单 seed，历史源码/评分版本未固定，实际成本也不相同。

### 4.3.4 成本声明

RF 的 24 条最终日志按 `full_evals + .25*coarse_evals` 重算，范围为 90.0–135.5，均值 115.25；日志 `equiv_total` 常为整数四舍五入。CMA 主表日志多为 108 个 full，Random 110，而 Soup/Anchors 提前停止或调用较少。PSO 的迭代、粒子初始化、dev/test 子项与失败成本尚无统一账本。因此撤回“严格同预算”“严格半预算”和单位调用信息增益倍数的结论。

未来比较必须统一初始化、边界、评分、终选、真实每题调用和墙钟上限；只对齐参数 budget=110 不足以构成等成本。

## 4.4 消融与外部证据的降级处理

旧探索消融有启发价值，但当前主表清单没有纳入可匹配的全套 simplex、full-CMA、no_de、no_sum、no_guard 等对照，因此本章不保留未经同协议溯源的部件因果排名。simplex 同时改变尺度与方向，应先做尺度配平；full-CMA 等分支还要验收其真实参与提案、tell 反馈及预算。探索性的失败尝试不构成局部最优证明，也不能由十六次改进失败推出“任意折衷均零和”。

SMM 结果由清单 `supplemental_evidence` 指向 `smmbench_results.json` 及当前 `smmbench_racer.py`。已存局部简化引擎的 train/test 均值为 racer .576975/.510068、de .571660/.506766、sep .554369/.488907。这里使用 **SMMBenchPS 代理**：没有昂贵评测竞速、真实 holdout 或完整 RACER 守卫；当前 driver 第三路是均匀随机，不是原 sum 引擎；局部 sep/de 也不是已认证的官方参赛实现。

聚合 JSON 未保存预算、种子列表、重复数和代理哈希，不能只从当前默认值反推“3000 预算 × 10 种子”。这些数值不进入 LLM 主表，不与文献约数排冠军，不证明完整 RACER 协议泛化。下一轮应冻结代理身份与正规的官方对手后独立报告。

## 4.5 冻结协议正式矩阵 racer_frozen_v1（2026-10，本章主结果）

### 4.5.1 预注册与身份冻结

上节 legacy 表的协议债务（评分未冻结、无独立终选、成本不匹配、seed 不齐）由本节矩阵清偿。实现为全新 `search_core.py`/`search_frozen.py`（不 import 旧 main），评分协议 `racer_eval_v1`（保守精确解析 + 冻结答案格式指令），预算 `B_t=110×D_t` 逐调用 reserve 计费，终选=完整 dev 首次观测 raw argmax，仅最终模型 test 一次。门禁 25/25 全绿后冻结身份（源码/数据/专家/基模 SHA-256）；正式矩阵按预注册 240 run 执行，不因结果改协议或追加 seed。

主对手与 RACER 共享 20 点初始化与 box [-.5,1.5]：官方 pycma 4.5.0（popsize8 σ.3）、按原 search.py 公式移植的 box PSO、signed box 均匀随机。全部方法同终选规则、同账本口径。

### 4.5.2 A 主比较（9 任务 × 4 方法 × 3 seed = 108 run）

本节表读法提示：宏平均回答“平均而言谁更好”（弱，功效受限）；任务行与配对分布回答“什么时候值得搜索、什么时候方法互换”（强，本矩阵的实质发现）。

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

配对分析（9 任务级）：RACER−random **+2.90pt，t(8)=2.55，p<.05 显著**；RACER−CMA +1.38pt t=1.23、RACER−PSO +0.84pt t=1.41，方向一致但未达显著。seed 级 27 对：vs CMA 17W/1T/9L、vs PSO 17W/0T/10L、vs random 22W/0T/5L。

任务簇 bootstrap（10⁴ 重采样，9 任务为簇、簇内用 seed 均值差，探索性）：RACER−CMA +1.38pt **95%CI [−0.09, +3.72]**；RACER−PSO +0.84pt [−0.25, +1.91]；RACER−random +2.90pt [+1.15, +5.29]。CI 与 t 检验一致：只有对 random 的区间不含零。

效应的稳健性需要注明集中度：去掉 gsm8k 后 RACER−CMA 宏平均差从 +1.38pt 降至 +0.29pt（−PSO 从 +0.84 降至 +0.49）——**平均优势主要由 gsm8k 单任务贡献**。gsm8k 内部是干净的：RACER 三 seed 全胜 CMA（.380/.333/.379 vs .221/.252/.314，逐 seed 领先 6.5~15.9pt），且消融（4.5.3）显示恰在此任务上每个组件都不可省。两种读法并存：任务级平均的结论弱（受任务间方差 SD 3.38pt 支配，9 任务功效不足；80% 功效约需 47 个任务）；单任务内与机制层的结论强。

任务画像分化是本表的核心结构：**RACER 的最大优势在 gsm8k**（生成式数学，+10.2pt vs CMA、+3.6pt vs PSO），science/medical/mmlu 小幅领先（+0.8~+2.8pt），纯 MC 判别任务差距压缩（hellaswag −0.6pt vs CMA），truthfulqa 上随机基线最优（搜索负增益）。3 seed 功效有限，任务级结论按均值+区间理解。

从“什么时候值得搜索”的角度重读本表，实用结论是条件性的：**gsm8k 上 RACER 对 random +11.2pt、PSO +7.6pt，是全表唯一大幅搜索收益，但同预算的 CMA 只 +1.0pt**（.221/.252/.314 逐 seed 大方差）——收益来自搜索器×任务的匹配，不是“搜索”本身；**hellaswag/nlgraph/culture/mmlu 上搜索相对 random 有 3.3~4.3pt 稳定收益，但方法间排序互换**（hellaswag/culture 是 CMA 最优，nlgraph 是 PSO 最优）；**mmlu_pro/medical/truthfulqa 上任何搜索方法与 random 差 ≤1.0pt**（truthfulqa 为负）——在那类任务上为合并系数烧搜索预算，期望收益低于任务间方差。方法选择与方法内部组件选择（4.5.3）一样，都是任务画像的函数。

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

组件价值随任务画像反转：

- **三引擎缺一不可只在 gsm8k 成立**：gsm8k 单删全降（no_ES −6.8pt 最深、no_DE −4.4pt、no_sum −3.4pt）；同为生成任务的 nlgraph 单删反而全部小幅反超（+1.3~+3.0pt）；四任务宏观 full 仍最高（.4324）。
- **MC 任务组件可省且反转**：medical 上 full_eval（取消竞速漏斗）三 seed 全超 full（.4295 vs .4216）且省 30% 生成时间（1.04h vs 1.50h）；nlgraph 上 no_rung2（+2.1pt）与 standard_SH（+1.9pt）反超 full。
- **plateau 切换四任务零贡献**（no_plateau 与 full 差 ≤0.2pt 内，gsm8k 完全并列）。
- guard 接受规则贡献小且方向不一：nlgraph/gsm8k/medical 降 0.6–1.2pt，hellaswag 反而 +0.1pt（该行最高）——守卫不构成稳定收益部件。
- 强组合对照未替代 RACER：cma_sh 四任务宏观 .4222 低于 full .4324。

竞速漏斗与三引擎的结论因此是**条件性的**：gsm8k 上全套机制缺一不可（单删引擎 −3.4~−6.8pt、取消漏斗 −6.5pt），nlgraph 反而简化晋升更好，medical 取消漏斗分数时间双赚，hellaswag 各配置与 full 差 ≤1.6pt。论文按任务画像如实呈现，不写"漏斗普遍有益"。

### 4.5.4 C 尺度解混（12 run）与 D kcross 敏感性（12 run）

C 组：sum1 空间 PSO 与同轨迹+固定 scale 网格 [.5,.75,1,1.25,1.5] 对照，同总预算。nlgraph scalegrid .3707 vs sum1 .3617（+0.9pt，且全部来自 s44 单点 +2.7pt）；gsm8k 逐 seed 逐分相同（.2957）。**固定 scale 网格无稳定收益**：box 空间已覆盖尺度自由度，sum1 空间整体落后 box 主表。第 3 章的尺度平方效应仍是数学事实，但不构成"后置 scale 校准"的实用路径。

D 组（kcross，dev/test 题干重叠的敏感性分析，不计入宏平均）：

| 方法 | 全量 test | 去重叠 filtered | Δ |
|---|---|---|---|
| CMA | .4147 | .4285 | +1.4pt |
| RACER | .3980 | .3989 | +0.1pt |
| PSO | .3940 | .4001 | +0.6pt |
| random | .3930 | .3989 | +0.6pt |

RACER 的分差对 dev/test 重叠**完全不敏感**（filtered≈test）；CMA 过滤重叠题后反而扩大领先。该任务 RACER 搜索负增益（落后 CMA 1.7pt），与 truthfulqa 同类画像。两项合并的解释：在搜索本身负增益的任务上，RACER 退向初始化候选，不放大 dev 噪声。

**身份披露**：D 组 12 run 因 test 路径缺陷（任务名误作数据集名）首跑失败后修复重跑，`search_frozen.py` 身份与 A/B/C 不同；A/B/C 228 run 旧身份保留，D 组单独核验（`experiments/racer_campaign_v1_D_disclosure.md`）。两组不混算（本来也不混）。

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

同预算下 RACER 比最快基线（PSO）慢 12–42%（漏斗多次粗评的加载开销），gsm8k 上差距最小（+12%）。成本-精度联合看：**gsm8k 是 RACER 明确的性价比区间**（+10.2pt vs CMA 只多花 3% 时间）；medical 上 full_eval 更便宜也更好（1.04h/.4295 vs full 1.50h/.4216，−31% 时间 +0.8pt）；hellaswag 的 full_eval 同样更便宜（0.59h vs 0.73h）但掉 1.4pt；nlgraph 的 standard_SH 分数更高但更贵（1.62h）。**没有一个配置在所有任务上同时最优分数与最低成本**——支持按任务画像选配置，不支持“漏斗普遍省钱”。

anytime 账本记录同轨迹 30/60/110 预算处 best_seen（如 gsm8k s42：6600 样本 .34 → 13200 样本 .53），不伪装成独立预算实验。全部 run `physical_samples == budget_used + test_samples` 逐 run 验证。

## 4.6 可复现入口与本章结论

### 4.6.1 legacy 轨道

```bash
/data/lvliping/miniconda3/envs/swarm/bin/python -I tools/thesis_results.py --manifest experiments/legacy_manifest.json
```

可选 `--output /new/path/report.json` 只允许新建文件，已有目标拒绝覆盖，search/ 内拒绝写入；运行时不生成图表。CPU 回归见 `tests/test_thesis_results.py`，包含真实清单均值、PSO 同粒子 H、并列规则、SUMO 分组、路由非正确率、路径/哈希及旧日志未变测试。

### 4.6.2 冻结轨道（主结果）

```bash
/data/lvliping/miniconda3/envs/swarm/bin/python -I tools/racer_results.py \
  --campaign experiments/racer_campaign_v1_AC --campaign experiments/racer_campaign_v1_BD \
  --campaign experiments/racer_campaign_v1_D1 --campaign experiments/racer_campaign_v1_D2
```

四实例合并验证（AC 为主表+消融的 A 组，BD 为消融 B/C 组，D1/D2 为修复后重跑的 D 组）；A/B/C 与 D 快照除 created_at 与经批准修复的 `search_frozen.py` 哈希外逐字段一致。已归档合并行表 `experiments/racer_final_v1_verified_rows.json`（240 verified）。CPU 回归见 `tests/test_racer_results.py`、`tests/test_search_core.py`、`tests/test_search_frozen_runner.py`、`tests/test_racer_campaign.py`。

### 4.6.3 本章结论

冻结协议下 RACER 的最终画像：**宏平均最优（.4245）且对随机基线显著（+2.90pt, t(8)=2.55；bootstrap CI [+1.15, +5.29] 不含零），对 CMA/PSO 方向一致但 CI 含零（[−0.09, +3.72] / [−0.25, +1.91]）**；平均优势集中于 gsm8k（去掉后 vs CMA 仅 +0.29pt），但该任务内三 seed 全胜且与消融机制互证；truthfulqa 与 kcross 上搜索本身负增益、RACER 不放大 dev 噪声。组件贡献条件化：三引擎与竞速漏斗的不可替代性仅在 gsm8k 成立，其余任务上简化配置可反超（medical 的 full_eval 更便宜也更好、nlgraph 的 no_rung2/standard_SH 反超）；plateau 切换全程零贡献，是从最终版中撤下的候选。legacy 轨道（4.3）在校正聚合后方向一致（RACER 宏平均最高），但协议债务使其只能历史参照。

本章不宣称：任务级单点胜负显著、竞速漏斗普遍有益、RACER 在所有任务画像上占优。3 seed 功效限制与多消融探索性比较的多重性在第 6 章局限中正式说明。
