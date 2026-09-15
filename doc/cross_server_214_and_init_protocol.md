# 跨服务器（214）实验分析与 init_protocol 改动记录

日期：2026-09-07。涉及机器：218keg（本机，4×GPU）/ 214（211.81.248.214，4×GPU）。

## 1. 背景与时间线

- 09-07 00:34 本机向 214 传送 `search_sgms.py / alpha_space.py / search_lp.py`（scp 免密已配）。
- 214 侧随后自行扩展了 `search_sgms.py`（12 行 diff）：新增 `--init_protocol {mix,pso}` 与 `alpha_space.sample_initial_alphas_pso()`，并用其跑完 **12 任务 × PSO/SGMS 对照矩阵**（原作者专家池 TaoJiangCN/qwen2.5-1.5b-tulu-v2-lora-experts，与本机同一池）。
- 09-07 15:45 本机将 214 的改动同步回来（直接采纳其文件，保留我们全部消融旗标），并启动本机验证链。

## 2. 214 的改动内容（已同步回本机）

### 2.1 `alpha_space.py` 新增 `sample_initial_alphas_pso(n, popsize, seed)`

复刻 `search.py` 的 `populate_initial_experts` 初始化分布（α 空间版）：
前 n 个单位向量；此后每个新点从「已有全部点（含子代）」随机挑两个不同父代，
`w1~U(0,2)`，α = w1·p1 + (1−w1)·p2。子代可再当父代 → 后期点自然呈 3-4 稀疏。
超出 box 的系数由调用方 clip（PSO 本身无 box，这是与基线的唯一差异）。

### 2.2 `search_sgms.py` 新增 `--init_protocol`

- `mix`（默认，= LP v4 协议）：10 单专家 + Dirichlet 混合体
- `pso`：上述 populate 子代链，**与 PSO 基线共享初始点**

### 2.3 方法学意义

PSO 与 SGMS 使用**同一套初始点**后，两者的性能差可完全归因于搜索动力学本身，
消除"初始化协议"这一混淆变量。本机原有对比存在该混淆（PSO 用 populate 链、
SGMS 用 mix）。**论文主表采用 pso-init 口径**，mix/pso 之差作为独立消融行。

## 3. 214 实验数据要点（12 任务，dev）

| 任务 | PSO | SGMS(pso-init) | Δ | 本机对照(mix-init) |
|---|---|---|---|---|
| nlgraph | 0.460 | **0.615** | +15.5 | 0.480→0.550 (+7) |
| gsm8k | 0.430 | 0.490 | +6 | 0.430→0.495 (+6.5) |
| hellaswag | 0.665 | 0.665 | 0 | 0.705→0.650 |
| medical | 0.533 | 0.527 | −0.6 | 0.507→0.509 |
| mmlu | 0.640 | 0.650 | +1 | （新战场） |
| mmlu_pro | 0.400 | 0.429 | +2.9 | （新战场） |
| kcross | 0.430 | 0.440 | +1 | （新战场） |
| truthfulqa | 0.555 | 0.545 | −1 | （新战场） |
| legal | 0.624 | 0.619 | −0.5 | （新战场；test +12.8 但小集高方差） |
| science | 0.519 | 0.512 | −0.7 | （新战场；test +15.9 同上） |
| abstainqa | 0.280 | 0.260 | −2 | （新战场） |
| **culture** | **0.561** | 0.510 | **−5.1** | （新战场；唯一显著失利，待种子确认） |

要点判读：
1. **PSO nlgraph 天花板跨机器成立**（0.460/0.480 同代码同池），仿射包论点无恙。
2. **初始化效应任务相关**：nlgraph 上 pso-init vs 本机 mix-init 差 +6.5（0.615/0.550，
   单种子待证）；gsm8k 无差（0.490/0.495）；hsw 约 +1.5。init 协议本身成为新消融维度。
3. 光滑地形（hsw）平手、饱和地形（medical）躺平——地形地图跨机器复现。
4. 214 的 `--no_gp` 消融三任务连击（gsm8k 0.485 vs 完整版 0.490；hsw 0.685 vs 0.665
   还更高；medical 在跑）→ GP 可砍结论加固，SGMS v3 方向再确认。

## 4. 环境差异清单（214 vs 本机）

- **专家池同一性已哈希验证**（09-07 18:40）：tulu_cot / tulu_sharegpt 的 adapter_model.safetensors 两机 md5 逐字节一致
- **同代码+同池+同seed+同init_protocol 的跨机器复现**（09-07 本机 sgms_ipso_nlgraph_s42）：
  214 得 0.615 / 本机 0.535，差 8 分——可控变量全部对齐后差距仍在，
  归因于跨机器评测噪声 × 轨迹混沌。**nlgraph 上单次 run 结果本身是 0.53-0.62 的分布**，
  任何单点数字（含本机 0.550）都不可单独引用，必须 mean±std（≥3 种子）。
  init_protocol 解释 0.615 的假设因此排除（本机 pso-init 0.535 < mix-init 0.550）。

- GPU 分配：214 每 run 用 2 卡（0,1 或 2,3 双 run 并行）；本机每 run 4 卡
- 模型路径：214 用本地路径 `/data/lvliping/src/evogm/models/qwen25-1.5b-lora-experts/{base,experts}`
- wandb project：214 默认 `model-swarms-qwen2.5-1.5b`（与 PSO 同 project 便于对比）
- 跨机器评测噪声：同 α 同分集 dev 分差 ~±2（PSO nlgraph 0.460 vs 0.480 即此量级）

## 5. 本机验证链（run_ipso.sh，09-07 15:53 起）

1. `sgms_ipso_nlgraph_s42`：同种子同配置跨机器复现 214 的 0.615
2. `sgms_ipso_nlgraph_s43`：pso-init 的种子方差
3. `sgms_ipso_gsm8k_s43`：gsm8k 种子扩展（pso-init 口径）

预期产出：论文主表 nlgraph 行的双协议、双机器、多种子定数。

## 6. 214 侧待办

- `sgms_s43_nlgraph` 未跑（其 s43 队列脚本中途失败：`run_sgms_s43.sh: No such file`）——0.615 的第二种子，最关键缺口
- culture s43/s44：唯一显著输 PSO 的任务，需种子确认后定性

## 7. 已知坑（同步教训）

- 214 只修了 evaluate.py `subset_size` 透传一处；**`base_model` 两处（dev/test 递归）未同步**——multitask 任务在他们环境会崩（本机 09-07 凌晨已全修）。
- 214 部分任务 eval_type 口径特殊：abstainqa 用 `-e AbstainQA -d mmlu`；kcross 用 `-d knowledge_crosswords`；mmlu_pro/science/legal/culture/truthfulqa 走 multiple_choice 或 multitask。并表时按 214 的 run 名映射，勿想当然。
