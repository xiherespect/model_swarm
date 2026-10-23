# 首批尺度诊断：partial / blocked

## 执行身份与范围

- 日期：2026-10-21。
- run：`runs/scale_pilot_20261021T090325_1282358/`。
- 主账本：`status.json`；配置/内容身份：`metadata.json`；逐候选表：`summary.csv`。
- 预注册范围：nlgraph/gsm8k × first-alphabetic onehot/uniform × c=[.5,.75,1,1.25,1.5]，每配置dev[0:50]，最多20配置/1000条，单卡累计上限7200秒。
- 实际终态：**blocked**，完成 **3/20配置、150条dev生成**；累计保守墙钟 **64.19007183192298秒**。未调用test、未联网下载、未重试或切卡，也未启动完整搜索。
- 选择GPU 0；first-alphabetic专家为 `tulu_code_alpaca`。首配置scale=1的真实物化与原专家全tensor逐项精确一致，见该候选的 `worker_status.json:25`。

## 完成结果

下表按尺度排序，实际执行顺序为1、.5、.75。

| 任务 | 方向 | 系数尺度c | 有效delta倍数c² | strict准确率 | 格式invalid数/总数 |
|---|---|---:|---:|---:|---:|
| nlgraph | tulu_code_alpaca onehot | .5 | .25 | .22 | 2/50 |
| nlgraph | tulu_code_alpaca onehot | .75 | .5625 | .22 | 3/50 |
| nlgraph | tulu_code_alpaca onehot | 1 | 1 | .26 | 25/50 |

invalid以逐题 `parsed_answer is None` 计数。每行逐题raw、parsed、gold、score与题目哈希见对应候选 `results/evaluation.json`。该统计仅用于当前strict_v1保守语法诊断，不等同于模型回答语义正确率或官方评分器等价结果。

## 阻塞原因与证据边界

父进程启动第四配置前，选定GPU未满足空闲检查，报：

`Blocked: No idle GPU satisfies memory/utilization/process checks; no jobs started or stopped`

三次worker均正常成功退出；第四候选目录未创建。后续只读检查发现四卡均1MiB、0%利用率且无compute进程，但**不能据此证明失败检查当刻也空闲**。

当前driver在子进程退出后立即检查选定GPU，阈值为显存≤64MiB、利用率=0、无compute进程；未保存不合格时的原始遥测。约一秒内的遥测尾窗或资源释放滞后可能造成自阻塞，但尚未确认，不将猜测写成根因。

## 可得与不可得的结论

- 已验证真实本地模型加载、LoRA物化、严格解析、raw归档及父进程账本的成功路径。
- 三个已完成尺度存在分数与invalid率差异；**样本少、矩阵未完成，不能推断最优尺度、泛化收益、PSO+scale结果或RACER因果机制**。
- c=1的invalid率高，提示输出格式/生成长度与解析规则可能混杂尺度效应。应先审计raw，不能为涨分事后放宽规则再冒充同一预注册协议。
- 尚未评测c=1.25/1.5、uniform或gsm8k，不填补缺失、不报告20配置完成。

## 下一闸门（未执行）

1. 增补GPU检查的原始遥测及拒绝原因记录，不降低空闲阈值。
2. 如获进一步批准，在自己worker正常退出后做有上限、计入预算的资源稳定检查。
3. 如果继续，必须保留并核验已成功3项，只补剩余17项，承接64.190秒和150条已耗账；不能重跑20项后从零计价。当前driver没有恢复入口，不自动重启。
4. 现阶段停止动态loop，等待后续资源/恢复行为的明确决定；不扩展48次搜索矩阵。

## 已完成工程验收

首批 `test_*.py` 共 **103项CPU测试通过**（9.535秒），覆盖协议39、strict接线5、实际物化1、driver29、论文清单29；`git diff --check`通过。AST/假后端测试不代替真实GPU推理，真实推理证据仅限以上三配置。
