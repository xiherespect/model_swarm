# RACER v1.1 设计（四刀，2026-09-17）

判据：验证任务（nlgraph/gsm8k/hsw/culture × s42/s43）均值 ≥ v1.0 且单任务跌幅 ≤1 分；无效刀单独关闭。

## A 三模式评估（推广平原刀）
- 已有统计量：spread（粗分极差）、ρ（粗→全秩相关，滑窗6对）
- 高速档 HIGHWAY：spread>0.15 且 ρ>0.5 连续 2 轮 → 关粗筛，k=4 个全部直接全保真；
  档内自检：全评分极差 <0.10 → 退出（信号消失）
- 高原档 PLATEAU：现状（spread<0.14/0.10 且 ρ<0.2 连续 2 轮 → 2 个直评）
- 其余漏斗档（现状）。开关 --no_highway
## B 显著性门槛
- 冠军接受准则：复评均值 > g + accept_margin(0.02)；archive 照记。
- 开关 --accept_margin 0
## C 自适应 sum 带
- 观测 ≥10 条后：目标 sum = 按分数加权抽一个精英的 sum + N(0,0.1)；
  早期回落 U(1.2,1.9)。开关 --fixed_sum
## D 秩-1 方向拉伸
- CMA 路：sd_eff = sd_es × (1 + 0.5·|evo_path|/‖evo_path‖∞)。开关 --no_stretch

依据（v1.0 复盘）：gsm8k iter18 +5 噪声跳被照收（test 反输 soup/cma）；hsw 粗筛晋升
错误造成回撤（0.675→0.640）；gsm8k 冠军 sum 1.91 顶穿固定上界；15 冠军斜脊 vs 轴对齐采样。
