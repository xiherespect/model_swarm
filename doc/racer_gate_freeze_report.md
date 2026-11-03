# RACER 冻结门禁报告（ racer_gate_v1 ）

生成：2026-10-24 定稿（门禁 25/25 全绿，2026-10-23 12:32 → 2026-10-24 01:20 UTC）
状态：**通过——正式矩阵身份据此冻结**

## 1. 目的与范围

按批准计划第 6 节阶段二：dev-only 单卡微型门禁，验证真实物化/加载/协议/预算账本/
checkpoint/终选，**不访问 test、不以算法排名作通过门槛**。通过后冻结正式矩阵身份。

- G1：9 主任务 × 2 预指定候选（one-hot e0 + 均匀 mix），dev 全量各一次（budget=2×D_t，init_size=2 后立即耗尽）
- G2：4 开发任务（nlgraph/gsm8k/hellaswag/medical）× 4 主方法 × seed 42，budget=24×D_t，只搜索不 test

## 2. 身份冻结

- 快照：`experiments/racer_gate_v1/snapshot.json`，identity_sha256 `c346785dc4422154…`
- 身份依赖（改动任一则 cohort 作废，按计划整批重跑）：
  `tools/run_racer_campaign.py`、`tools/audit_racer_data.py`、`tests/test_racer_campaign.py`、
  `experiments/racer_frozen_v1.json`、`experiments/racer_data_audit_v1.json` + spec.source_files
  （search_core.py / search_frozen.py / eval_protocol.py / evaluate.py / alpha_space.py）
- 前代 cohort：`experiments/racer_gate_v1_superseded/`（身份漂移归档，成功 run 不拼接，见其 SUPERSEDED.md；
  其 adapter 物化已按 PRUNED_MANIFEST 清理，summary/results/日志保留）

## 3. 门禁结果（实测，2026-10-24）

**25/25 全部 succeeded，failed=0，driver exit 0。**

| 组 | 通过/总数 | 失败 |
|---|---|---|
| G1 | 9/9 | 0 |
| G2 | 16/16 | 0 |

不变量核验（25/25 逐 run）：`physical_samples == budget_used`、`test_samples == 0`、
`status == succeeded`、终选=完整 dev 首次观测 raw argmax——全部成立，无例外。

G2 各任务耗时（budget=24×D_t；秒）：

| 任务 | racer | cma_full | pso_box | random_box |
|---|---|---|---|---|
| nlgraph (4800) | 843.3 | 828.9 | 826.2 | 1020.1 |
| gsm8k (4000) | 1888.7 | 1905.5 | 1875.9 | 2261.5 |
| hellaswag (4800) | 346.1 | 351.2 | 349.8 | 414.6 |
| medical (4800) | 652.7 | 646.1 | 648.0 | 774.5 |

逐 run 明细见 `experiments/racer_gate_v1/gate_status.json`。

## 4. 实测吞吐与正式矩阵 timeout 冻结

- 总实测：4.49 生成小时 / 89,400 样本 = **5.5 样本/秒**（单卡、混合 MC+生成任务）
- 任务类型差异：MC 类（hellaswag/medical/mmlu 等）~13.9 样本/秒；生成类 gsm8k ~2.1 样本/秒；
  nlgraph（短生成）~5.7 样本/秒
- random_box 恒比其余方法慢 ~20-25%（无 rung 漏斗、评满全部样本）——排程时长预估已计入
- 按 B_t=110×D_t 最重任务 culture（D_t=1000×2 数据集=2000，B_t=220000）：MC 速率外推 ≈ 4.4h；
  **正式 run timeout 冻结为 6h**（`--run-timeout 21600`），留 ~35% 余量覆盖加载/物化/慢尾

## 5. 判定标准（预注册，不按分数）

- 25/25 run `status=succeeded`；`physical_samples == budget_used`；`test_samples == 0`
- 终选=完整 dev 首次观测 raw argmax（并列取最早 candidate ID）；attempt.jsonl 存在且可解析
- 失败即停：不放宽协议、不以门禁分数调参、不追加 seed

## 6. 正式矩阵启动命令（门禁全绿后执行）

**并行方案（用户 2026-10-23 批准放宽至 4 卡）：双实例，每实例 2 卡。**
不改代码、身份哈希不变（身份只由源文件决定）；两实例各自独立账本/目录，
GPU 白名单互斥（0,1 vs 2,3），他人占卡时该实例 fail-closed blocked 不抢卡。
矩阵切分（ID 清单已存 `experiments/campaign_instance{1,2}_ids.txt`）：
- 实例1（`--gpus 0,1`）：A 主比较 108 + C 尺度 12 = 120 run ≈ 319 GPU·h → ~6.6 天
- 实例2（`--gpus 2,3`）：B 消融 108 + D kcross 12 = 120 run ≈ 283 GPU·h → ~5.9 天

```
RACER_BASE_MODEL=/data/lvliping/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  /data/lvliping/miniconda3/envs/swarm/bin/python -I tools/run_racer_campaign.py \
  --execute --output experiments/racer_campaign_v1_AC --gpus 0,1 \
  --max-parallel 2 --run-timeout 21600 \
  --only "$(cat experiments/campaign_instance1_ids.txt)" &

RACER_BASE_MODEL=... /data/lvliping/miniconda3/envs/swarm/bin/python -I tools/run_racer_campaign.py \
  --execute --output experiments/racer_campaign_v1_BD --gpus 2,3 \
  --max-parallel 2 --run-timeout 21600 \
  --only "$(cat experiments/campaign_instance2_ids.txt)" &
```

磁盘纪律（本轮实测）：每 run adapter 物化 ~37MB×候选数（G2 20 候选≈740MB）。
已在跑 cohort 的已终结 run 逐个清 `oracle/adapters/cand_*` 并留 PRUNED_MANIFEST.txt；
正式矩阵期间按同样规则例行清理（只清 worker_status 终态的 run）。

启动前置条件（缺一不启）：
1. 门禁 25/25 全绿且本报告第 3/4 节填完实测值
2. 磁盘水位 >20G（/data 曾到 100%，教训 L2.9/L7.1）
3. 启动前 preflight 身份 == 门禁快照身份（无源文件漂移）
