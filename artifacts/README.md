# 历史探针产物归档

`artifacts/legacy_probes/` 用于保留旧探针生成的 adapter / 对比产物，**不是新的实验结果目录或默认输出路径**。按批准计划，主整理任务将 515 个根目录产物目录原字节归档到此处；具体旧 / 新路径、逐项 hash、权限与完成状态由 [doc/repository_cleanup_manifest.json](../doc/repository_cleanup_manifest.json) 记录，此文档不代替迁移验收。

## 归档来源

| 原根目录目录名 | 生成源码（保留在根目录） |
|---|---|
| `cmp_pair_*`、`cmp_single_*` | `probe_composability.py` |
| `full_p_*`、`full_s_*` | `probe_comp_full.py` |
| `single_*` | `probe_singles.py` |

这些前缀仅说明历史类别；归档和恢复的精确范围以清单逐项记录为准，不能用宽泛 glob 把后续新产物混入同一次迁移。

- 归档不修改模型 tensor、adapter 配置、实验统计或内部文件名，不重新计算，不以压缩 / 转换替代原文件，不补造新 run 身份。
- 这些目录不自动成为独立实验、strict 评分证据或论文主表来源。历史主表依赖 `experiments/legacy_manifest.json` 的显式证据白名单；它不是迁移清单。
- **旧探针源码没有改动**，输出仍使用相对 cwd 的旧目录名。从仓库根目录再次运行会重新生成根目录产物，不会自动写到 `artifacts/legacy_probes/`；不要为了“填充归档”重跑 probe。
- 旧 shell 位于 `scripts/legacy/`，全部只作历史参考，不保证新位置可运行；可能涉及 `rm -rf`、`pkill`、`$0` 和 cwd 依赖，不能当新入口执行。

## 不在本次归档范围

`search/`、`runs/`、`data/`、`initial_experts/`、`wandb/` 原位保留。根目录 `.log` / `.out` 因引用尚未全部确认而不移动、不删除。根目录 Python 全部保留，尤其 legacy manifest 引用的 `smmbench_racer.py` 与 `smmbench_results.json` 不迁入本目录。既有论文文稿、实验来源与清单不作顺手清理。

新的尺度诊断使用 `tools/probe_scale.py`，默认仅 dry-run；真实 GPU 执行需要显式 execute、全新输出路径和单独授权。目录归档不启动任何推理或网络请求。

## 校验与恢复

迁移清单由主任务维护；文档任务不修改它。恢复前必须确认新归档路径存在、旧根目录路径不存在，且逐项 hash / 权限与清单一致。若旧 probe 已重新生成同名根目录，即构成冲突：停止，不删除或覆盖任一份。

通过预检后按已完成迁移记录**逆序 rename** 回旧路径，逐步重新检查冲突，结束后复核 hash / 权限与新路径移除状态。缺失、hash 变化或权限不符都应停止并核实；不能靠 `git reset`、`git clean`、合并目录或强制覆盖恢复。完整约束见 [仓库布局与恢复规则](../doc/repository_layout.md)。
