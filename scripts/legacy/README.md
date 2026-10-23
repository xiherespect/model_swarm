# 历史 shell 归档

本目录是旧实验启动、串行队列、探针、sweep、停止和重启 shell 的**历史参考区**，不是新实验入口。按批准计划，主整理任务将 56 个旧 shell 原字节、保留权限迁入此处；实际名单和迁移验收以 [doc/repository_cleanup_manifest.json](../../doc/repository_cleanup_manifest.json) 为准。此 README 不表示迁移已全部完成。

## 不能直接照搬执行

- 归档只移动文件，不改源码、不修路径，**不保证脚本在新位置可运行**。
- 部分脚本含 `rm -rf`，可能删掉同名旧 run、模型或临时目录；不能把这种清理机制作为新实验默认行为。
- 部分停止 / 修复脚本含 `pkill -f`，可能匹配同名其他进程；不能用它们管理当前任务。
- `$0`、`setsid`、`nohup` 自重启 / 后台排队可能依赖旧位置；不要因表面“只启动一个 shell”忽略后续作业。
- 固定绝对路径、仓库根 cwd、相对 `search/` / `data/` / 日志路径与其他脚本互相调用，都可能在迁移后失效或指向错误位置。

以上警告适用于全部旧 shell，不是说每个脚本都含全部危险语句。不要在本目录批量执行、补执行权限或用软链接 / wrapper 假装修复兼容性。任何后续运行都需要单独审查脚本、确认路径和副作用、得到相应资源授权；归档本身不授权运行。

## 根目录仍保留的上游示例

`search_nlgraph.sh`、`search_legal.sh`、`search_concise.sh`、`search_phd_application.sh` 留在根目录，便于对照 [整理前 README 原文](../../doc/upstream_model_swarms.md)。它们也只是历史示例，不是当前推荐的一键启动命令；可能涉及 GPU、模型下载、WandB 或云 API，必须另行审查。

## 当前导航

- 当前研究入口见 [根 README](../../README.md) 和 [仓库布局](../../doc/repository_layout.md)。所有根目录 `.py` 都保留原位；不从 shell 迁移推断 Python 已重构。
- `tools/thesis_results.py` 是显式 legacy 证据重建工具，不启动搜索；`tools/probe_scale.py` 默认只读 dry-run，真实 execute 另行批准。
- 旧 probe 源码未改输出路径。从根目录再次运行仍会生成根目录 `cmp_pair_*`、`cmp_single_*`、`full_p_*`、`full_s_*`、`single_*`，不会自动写入归档。
- 根目录 `.log` / `.out` 因引用未全部确认而保持原位；`search/`、`runs/`、`data/`、`initial_experts/`、`wandb/` 与 manifest 引用的 `smmbench` 文件也保持原位。不能因脚本已归档而删除它们。

## 恢复

由主任务维护迁移清单，不编辑清单来绕过校验。恢复前确认归档新路径存在、原旧路径不存在、逐项 hash 和权限一致；发现冲突、缺失或变化即停止。校验通过后按迁移记录**逆序 rename** 回旧路径，再核验内容和权限。不得覆盖、合并、删除冲突路径，也不得使用 `git reset` / `git clean` 替代恢复。详见 [恢复规则](../../doc/repository_layout.md#5-清单与恢复规则)。
