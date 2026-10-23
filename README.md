# Model Swarms 衍生研究工作区

本仓库基于 [Model Swarms: Collaborative Search to Adapt LLM Experts via Swarm Intelligence](https://arxiv.org/abs/2410.11163)（ICML 2025），当前用于 Qwen2.5-1.5B-Instruct / 十个 Tulu v2 LoRA 专家的模型合并搜索、RACER 方法研究、历史证据重建与严格评测协议诊断。它是研究工作区，不是所有历史脚本均可直接运行的发行包。

整理前的 README **全文原样保存**于 [doc/upstream_model_swarms.md](doc/upstream_model_swarms.md)，包括原始快速开始、引用和本地 Qwen 补充；它是来源与历史说明，不是当前环境的运行保证。上游推荐的后续实现见 [MoCo](https://github.com/BunsenFeng/model_collaboration)。

## 从哪里开始

| 目的 | 入口 | 边界 |
|---|---|---|
| 了解目录、历史入口与恢复规则 | [仓库布局](doc/repository_layout.md) | 整理不重构代码、不修复历史脚本路径 |
| 查看结论、证据边界和实验闸门 | [实验验收表](doc/experiment_plan.md)、[论文骨架](doc/thesis_skeleton.md) | legacy 与 strict 分轨，不能由旧表推断等预算或独立泛化 |
| 重建显式 legacy 结果 | [tools/thesis_results.py](tools/thesis_results.py)、[experiments/legacy_manifest.json](experiments/legacy_manifest.json) | 只读白名单证据；不是重新推理或 strict 重评分 |
| 查看评测与物化实现 | `evaluate.py`、`eval_protocol.py`、`alpha_space.py` | factor-LoRA 的 A/B 分别混合；整体系数尺度 c 对有效增量产生 c² 缩放 |
| 查看当前搜索研究主线 | `search_racer.py` | RACER；现存搜索/终选/预算仍须按实验验收表审计 |
| 查看尺度诊断 | [tools/probe_scale.py](tools/probe_scale.py)、[experiments/scale_pilot.json](experiments/scale_pilot.json)、[pilot 报告](doc/scale_pilot_report.md) | 默认 dry-run；真实执行必须另行批准 |
| 查看历史队列和探针产物 | [scripts/legacy/README.md](scripts/legacy/README.md)、[artifacts/README.md](artifacts/README.md) | 历史参考，不作为新实验启动入口 |

既有论文正文在 `thesis/`；研究设计、经验及相关工作在 `doc/`。导航整理不修改这些文稿或旧实验结果。

## 代码身份：根目录 Python 全部保留

**核心实现**：`evaluate.py` 是 dev/test 评测与结果归档入口；`eval_protocol.py` 提供评分、切片、聚合等纯函数及 legacy/strict_v1 区分；`alpha_space.py` 提供专家池与系数物化；`search_racer.py` 是 RACER 搜索主线。`strict_v1` 是保守语法，不等同于官方评分器；保留 legacy 默认行为的路径不会因目录整理自动升级为 strict。

**历史搜索与对照/探索入口**（保留源码和路径，不作为统一推荐启动命令）：

- `search.py`：上游 Model Swarms / PSO 搜索。
- `search_baseline.py`：Random、CMA 等基线与免搜索锚点。
- `search_sgms.py`、`search_lp.py`：SGMS、LP-Merge 历史路线。
- `search_genome.py`：GENOME 对照移植。
- `search_dualres.py`：双分辨率搜索探索。
- `search_sumo.py`、`search_amphi.py`：SUMO 支撑子空间搜索与 AMPHI 试探式路由探索。

其余根目录 `.py` 也全部原位保留，包括 `merge.py`、`reward_modeling.py`、`overall_metrics.py`、`compare_all.py`、`sweep_collect.py`、`smmbench_racer.py` 以及旧 `probe_*.py`；保留不代表推荐直接运行。

## 离线 CPU 入口

以下命令在仓库根目录、使用已有依赖环境运行；不会为你安装依赖或下载模型。

```bash
# 显式 legacy 重建：默认只向终端输出，不启动模型、不扫描择优 run
python -I tools/thesis_results.py --manifest experiments/legacy_manifest.json

# 默认 dry-run：只读预检与候选矩阵，不查询 GPU、不生成、不写目录
python -I tools/probe_scale.py --config experiments/scale_pilot.json

# 仅发现 test_*.py 的 CPU 回归；不包含 smoke_cpu.py
python -I -m unittest discover -s tests -p 'test_*.py' -v
```

- legacy reader 的 `--output` 只允许**新 JSON 文件**，拒绝覆盖来源和在 `search/` 内写入。`experiments/legacy_manifest.json` 是证据白名单，和整理清单不是同一个文件。
- `probe_scale.py` 只有显式 `--execute` 加全新 `--output` 才进入真实 GPU 执行；这不是默认模式，也不属于文档整理授权。不能通过旧队列绕过预算、安全检查或批准流程。
- 本次整理后已重新运行 **103 项 CPU 测试，全部通过**；历史证据表和 pilot dry-run 身份/矩阵保持不变，56 个归档 shell 的 `bash -n` 均通过。这不等于完整搜索或 GPU 推理已通过；逐项验收见 `doc/repository_cleanup_manifest.json`。
- `tests/smoke_cpu.py` **不属于上述轻量回归入口**：虽然标称 CPU，但会加载 `initial_experts/` 的十个真实专家，导入 torch/safetensors，执行物化和反解并写临时 adapter；临时目录没有自动清理逻辑。不要把它当成无重权重、无写入的默认 smoke，也不要在整理时执行。

## 整理后的归档边界

已将 **515 个**根目录 `cmp_pair_*`、`cmp_single_*`、`full_p_*`、`full_s_*`、`single_*` 产物目录原字节归档至 `artifacts/legacy_probes/`，将 **56 个**旧 shell 移至 `scripts/legacy/`。共1526个文件的内容哈希、大小及权限核验通过，未删除科研资产。根目录目录数从529降至16；归档仅改变位置，不释放权重占用空间。逐项旧/新路径和验收记录见 `doc/repository_cleanup_manifest.json`。

根目录仅保留四个上游示例 shell：`search_nlgraph.sh`、`search_legal.sh`、`search_concise.sh`、`search_phd_application.sh`。**所有旧 shell（包括这四个示例）都只是历史参考**：可能包含 `rm -rf`、`pkill`、`$0` 自重启、固定路径和 cwd 假设；不保证迁移后可运行，不能当成新入口直接执行。

根目录 `.log` / `.out` 因引用尚未全部确认而保留。`search/`、`runs/`、`data/`、`initial_experts/`、`wandb/` 原位不动；legacy 清单引用的 `smmbench_results.json`、`smmbench_racer.py` 也原位保留。旧探针源码未改，若从根目录再次运行，仍会生成上述根目录产物，不会自动写入归档。

恢复须逐项核验 hash、权限和路径，确保旧/新路径无冲突，再按迁移记录**逆序 rename**；不使用 `git reset`。详见 [仓库布局与恢复规则](doc/repository_layout.md)。

## Model Swarms 来源引用

```bibtex
@article{feng2024model,
  title={Model Swarms: Collaborative Search to Adapt LLM Experts via Swarm Intelligence},
  author={Feng, Shangbin and Wang, Zifeng and Wang, Yike and Ebrahimi, Sayna and Palangi, Hamid and Miculicich, Lesly and Kulshrestha, Achin and Rauschmayr, Nathalie and Choi, Yejin and Tsvetkov, Yulia and others},
  journal={arXiv preprint arXiv:2410.11163},
  year={2024}
}
```

本地采用的 Qwen 专家来源：[TaoJiangCN/qwen2.5-1.5b-tulu-v2-lora-experts](https://huggingface.co/TaoJiangCN/qwen2.5-1.5b-tulu-v2-lora-experts)。模型准备与 GPU/云 API 配置须单独审查，不属于仓库导航或归档操作。
