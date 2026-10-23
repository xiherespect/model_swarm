# 仓库布局与整理边界

本文说明已批准整理计划的目标布局、入口身份与恢复约束。文件移动和逐项验收由主整理任务负责；本文档不宣称迁移已验收完成。实际旧/新路径、hash、权限和迁移记录以主任务维护的 `repository_cleanup_manifest.json` 为准，不能只凭目录名或本文计数判断完成。

## 1. 当前研究与证据入口

| 路径 | 身份 / 使用边界 |
|---|---|
| `README.md` | 当前工作区导航；所有相对仓库路径从根目录解释 |
| `doc/upstream_model_swarms.md` | 整理前 README 的完整原文副本，含原有 Qwen 补充；保留来源，不承诺历史命令当前可用 |
| `evaluate.py` | dev/test 评测、生成与结果归档；实际评分协议必须显式核验 |
| `eval_protocol.py` | 无重依赖的评分、切片、聚合等协议纯函数，保留 legacy 并提供 strict_v1 |
| `alpha_space.py` | ExpertPool、系数采样与 LoRA factor 物化；A/B 分别混合，不是线性 delta merge |
| `search_racer.py` | RACER 研究主线；目录整理不修复终选/预算等协议债务 |
| `tools/thesis_results.py` | 根据显式 legacy manifest 安全重建旧结果，不重新推理、不按 test 选 run |
| `experiments/legacy_manifest.json` | 历史证据白名单与来源身份；不是整理迁移清单 |
| `experiments/run_manifest.schema.json` | 实验清单结构说明 |
| `tools/probe_scale.py`、`experiments/scale_pilot.json` | 尺度 pilot：默认只读 dry-run；真实 execute 需另外批准 |
| `tests/test_*.py` | CPU 回归基线入口；首批记录为 103 项，见 `doc/scale_pilot_report.md` |
| `tests/smoke_cpu.py` | 独立重专家脚本，不是无副作用的默认 CPU 回归 |
| `doc/`、`thesis/` | 既有设计、验收、相关工作与论文文稿；此次不改这些既有研究正文 |

`strict_v1` 是保守答案语法，不等价于官方评分器。legacy 聚合校正不等于 strict 重评分；目录归档不能改变旧实验的协议身份。详细证据边界见 [experiment_plan.md](experiment_plan.md)。

## 2. 根目录 Python 不移动

所有根目录 `.py` 原位保留，不打包、不加兼容层、不改 import。

- **核心**：`evaluate.py`、`eval_protocol.py`、`alpha_space.py`、`search_racer.py`。
- **上游 / 历史搜索、对照与探索**：
  - `search.py`：Model Swarms / PSO。
  - `search_baseline.py`：基线与免搜索锚点。
  - `search_sgms.py`：SGMS 历史路线。
  - `search_lp.py`：LP-Merge 与二次代理。
  - `search_genome.py`：GENOME 对照移植。
  - `search_dualres.py`：双分辨率探索。
  - `search_sumo.py`：SUMO 支撑子空间搜索。
  - `search_amphi.py`：AMPHI 试探式路由。
- **合并 / 评测辅助与旧汇总**：`merge.py`、`reward_modeling.py`、`overall_metrics.py`、`compare_all.py`、`sweep_collect.py`。
- **旧探针**：`probe_composability.py`、`probe_comp_full.py`、`probe_singles.py`。源码输出路径不改，不能视为新的安全探针驱动器。
- **补充外部验证**：`smmbench_racer.py`。它和根目录 `smmbench_results.json` 都被 legacy manifest 引用，留在原位；简化 PS 引擎证据不代表完整 RACER 或官方全 SMM 结果。

保留源码是为了维护可追溯性，不是推荐逐个执行；各搜索器的初始化、预算、评分、终选和依赖必须单独核验。

## 3. 归档范围与不动范围

### 产物归档

主任务按批准计划将以下 **515 个目录**保持原字节归档到 `artifacts/legacy_probes/`，通常保留原 basename；每一项映射必须查迁移清单，不依赖通配符推测或现场重新扫描。

| 原根目录前缀 | 历史来源 | 目标类别 |
|---|---|---|
| `cmp_pair_*`、`cmp_single_*` | `probe_composability.py` | `artifacts/legacy_probes/` |
| `full_p_*`、`full_s_*` | `probe_comp_full.py` | `artifacts/legacy_probes/` |
| `single_*` | `probe_singles.py` | `artifacts/legacy_probes/` |

归档不改模型、配置、统计或时间记录，不重新计算、不重命名内部文件、不压缩替换、不把这些缓存宣称为新实验。旧 `probe_*.py` 仍在根目录，输出是相对 cwd 的旧名字；从根目录再次运行会**重新生成根目录产物**，不会自动重定向到 `artifacts/`。详见 [artifacts/README.md](../artifacts/README.md)。

### shell 归档

主任务将 **56 个**旧实验队列、停止/重启、探针和 sweep shell 移至 `scripts/legacy/`。根目录保留四个上游示例：

- `search_nlgraph.sh`
- `search_legal.sh`
- `search_concise.sh`
- `search_phd_application.sh`

所有这些旧 shell 都是**历史参考**，包括仍在根目录的四个示例。迁移是保留字节和权限的归档，不修复路径；不保证 `scripts/legacy/` 中的脚本在新位置或任何 cwd 下可运行。部分脚本包含 `rm -rf`、`pkill`、`$0` 自重启、绝对路径、相对 cwd 和旧日志路径依赖。不得将它们作为新实验推荐入口，也不得在整理中执行。详见 [scripts/legacy/README.md](../scripts/legacy/README.md)。

### 原位保留

- `search/`：旧搜索轨迹、模型和日志；不改内容，不重跑覆盖。
- `runs/`：独立 run 账本与 pilot 结果；不搬移、不续跑。
- `data/`、`initial_experts/`、`wandb/`：数据、真实专家与追踪记录原位不动。
- 根目录 `.log` / `.out`：引用尚未全部确认，**保留原位**；不以“已归档 shell”为由移动、删除或截断。
- `smmbench_racer.py`、`smmbench_results.json` 及其他 manifest 引用来源：保持原位和原内容。
- 既有 `doc/`、`thesis/` 文稿、实验 manifest 和旧实验内容：不顺手修改。

## 4. 安全的离线导航命令

在仓库根目录使用已有依赖环境，Python 读取实验数据使用 `-I`：

```bash
python -I tools/thesis_results.py --manifest experiments/legacy_manifest.json
python -I tools/probe_scale.py --config experiments/scale_pilot.json
python -I -m unittest discover -s tests -p 'test_*.py' -v
```

- legacy 重建只消费 allowlist，核验来源身份，不全盘发现 / 排名 run。默认输出终端；可选 `--output` 必须是新 JSON 文件，不得覆盖证据或写入 `search/`。
- probe 默认 dry-run：不写目录、不查询 GPU、不导入重型模型依赖、不生成。显式 `--execute` 和全新 `--output` 才进入真实执行；导航不是 GPU 授权。
- 103 项是已记录的 CPU 基线，不是本次文档整理的新测试结果。小 tensor / AST / 假 worker 测试不等于真实 GPU 搜索验证。
- `tests/smoke_cpu.py` 不匹配 `test_*.py`：它直接加载十个真实专家，导入 torch/safetensors，物化 adapter、反解，并以 `tempfile.mkdtemp()` 写临时目录且没有自动清理。即使不占 GPU，也有重权重、CPU 内存 / I/O 和写文件副作用，不能作为默认只读冒烟入口。

## 5. 清单与恢复规则

`doc/repository_cleanup_manifest.json` 由主整理任务单独维护，记录逐项旧路径、新路径、hash、权限与迁移信息。文档维护不编辑该清单；它不同于 `experiments/legacy_manifest.json` 的研究证据白名单。

恢复只针对清单所列迁移项，不能恢复掉工作区其他人的改动：

1. **只读预检全部待恢复项**。确认归档的新路径存在，原旧路径不存在；新旧不能冲突，不能覆盖、合并目录或自动删除根目录新生成的 probe 结果。
2. 按清单核验归档内容的逐项 hash 与权限。hash 不一致、缺项、权限变化或路径冲突时停止，人工核实，不跳过校验或猜测另一份副本。
3. 在校验通过后，按已完成迁移记录的**逆序**，将每个新路径 `rename` 回旧路径。执行每一步前再次确认目标旧路径空缺且来源仍匹配；保留内容字节与权限，不复制重建或重写脚本。
4. 恢复后再次核验旧位置 hash / 权限与清单一致，并确认对应新路径不再存在；由主任务记录实际恢复状态。

**不用 `git reset`、`git clean` 或批量强制覆盖恢复**；它们不能替代未跟踪实验产物的逐项逆序 rename，并可能损坏工作区其他未提交内容。此说明不是在本次文档任务中实际执行恢复的授权。
