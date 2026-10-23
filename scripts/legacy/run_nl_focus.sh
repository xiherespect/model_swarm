#!/usr/bin/env bash
# nlgraph 主战场强化：等 GENOME 修复链 (PID ) 结束后跑
#   PSO nlgraph 强化基线 ×2（给对手最强配置，堵审稿人）+ 双算法多种子 ×2（统计稳健性）
set -euo pipefail
cd /data/lvliping/merge_swarm
if [[ -z "${NLF:-}" ]]; then
    NLF=1 setsid nohup "$0" >> nl_focus.log 2>&1 < /dev/null &
    echo "nlgraph 强化链已挂起 (等 PID )"; exit 0
fi
while kill -0  2>/dev/null; do sleep 60; done
sleep 30
pgrep -f "envs/swarm/bin/python search" > /dev/null && { echo "[nlf] 忙，放弃"; exit 1; }
export WANDB_ENTITY=llp031004-yanshan
PY=/data/lvliping/miniconda3/envs/swarm/bin/python

run () { echo "[nlf] $(date '+%H:%M') $1 开始"; shift
  if "$PY" "$@" > /dev/null 2>&1; then echo "[nlf] $(date '+%H:%M') 完成: $(grep -h 'final metrics' search/$(echo \"$@\" | grep -oP '(?<=-n )\S+')_\218keg/log.txt 2>/dev/null | tail -1)"; else echo "[nlf] 失败"; fi }

# 1. PSO nlgraph 强化基线（论文Table8暗示nlgraph喜大repel，故测 repel 0.1 与 0.05 两档）
run pso_nl_r01cfg search.py -n pso_nl_r01cfg -e exact_match -d nlgraph -g 0,1,2,3 -b Qwen/Qwen2.5-1.5B-Instruct --inertia 0.3 --cognitive_coeff 0.3 --social_coeff 0.3 --repel_coeff 0.1 --step_length 0.8 --populate_initial_experts 1 --initial_experts_num 20 --starting_test_set_eval 0 --fast_merge 1 --starting_velocity_mode random --repel_term 1 --step_length_factor 0.95 --restart_stray_particles 1 --restart_patience 0.5 --weight_randomess 1 --project_name_wb keg218-swarm-qwen2.5b -p 10 -m 30
run pso_nl_repel05 search.py -n pso_nl_repel05 -e exact_match -d nlgraph -g 0,1,2,3 -b Qwen/Qwen2.5-1.5B-Instruct --inertia 0.2 --cognitive_coeff 0.2 --social_coeff 0.4 --repel_coeff 0.05 --step_length 0.9 --populate_initial_experts 1 --initial_experts_num 20 --starting_test_set_eval 0 --fast_merge 1 --starting_velocity_mode random --repel_term 1 --step_length_factor 0.95 --restart_stray_particles 1 --restart_patience 0.5 --weight_randomess 1 --project_name_wb keg218-swarm-qwen2.5b -p 10 -m 30
# 2. 赢家的多种子稳健性
run lp_nl_s43 search_lp.py -n lp_v4_nlgraph_s43 -e exact_match -d nlgraph -g 0,1,2,3 --popsize 20 --max_iter 25 --seed 43
run sgms_nl_s43 search_sgms.py -n sgms_v1_nlgraph_s43 -e exact_match -d nlgraph -g 0,1,2,3 --lb -1.5 --ub 2.0 --seed 43
echo "[nlf] 全部结束 $(date)"
$PY compare_all.py || true
