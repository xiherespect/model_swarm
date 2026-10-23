# 第 4 章 RACER：多提案与竞速的合并系数搜索

## 4.1 方法定位

RACER（Ridge-Adaptive Covariance Evolution with Racing）是本文的主要方法：在十维 **factor 合并** 系数空间内，把多样化候选生成、分层评测与终选组织为可审计的搜索流程。研究问题不是证明某优化器普遍占优，而是在昂贵、有限开发数据条件下，怎样分配搜索调用，并分离方向、尺度、选择和评分的贡献。

历史实现包含多个演化版本。本章主表只采用明确指定的 RF 最终批次双种子档案与固定基线名单，不把目录中 test 更好的早期 run 换进来。当前源码可用于解释机制，但没有运行时源码哈希，不能断言等同于每个历史 run 的源码。

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

## 4.5 可复现入口与本章结论

标准库只读重建：

```bash
/data/lvliping/miniconda3/envs/swarm/bin/python -I tools/thesis_results.py --manifest experiments/legacy_manifest.json
```

可选 `--output /new/path/report.json` 只允许新建文件，已有目标拒绝覆盖，search/ 内拒绝写入；运行时不生成图表。CPU 回归见 `tests/test_thesis_results.py`，包含真实清单均值、PSO 同粒子 H、并列规则、SUMO 分组、路由非正确率、路径/哈希及旧日志未变测试。

本章形成的核心结果是 **RACER 方法描述与一张可追溯、校正聚合但仍属 legacy 的历史表**。独立终选、严格评分的新 run 与等成本因果消融属于后续闸门，不能借由清单测试通过而宣称已完成。
