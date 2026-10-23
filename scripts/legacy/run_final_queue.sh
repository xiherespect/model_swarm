#!/usr/bin/env bash
# 终极队列：等主链(PID $1)结束后 → GENOME修复×2 → PSO nlgraph强化基线×2 → 双算法多种子×2 → 总表
set -euo pipefail
cd /data/lvliping/merge_swarm
if [[ -z "${FQ:-}" ]]; then
    FQ=1 setsid nohup "$0" "$1" >> final_queue.log 2>&1 < /dev/null &
    echo "终极链已挂起 (等主链 PID $1)"; exit 0
fi
while kill -0 "$1" 2>/dev/null; do sleep 60; done
sleep 30
pgrep -f "envs/swarm/bin/python search" > /dev/null && { echo "[fq] 忙，放弃"; exit 1; }
export WANDB_ENTITY="llp031004-yanshan"
PY=/data/lvliping/miniconda3/envs/swarm/bin/python
run () { local nm="$1"; shift
  echo "[fq] $(date '+%m-%d %H:%M') $nm 开始"
  if "$PY" "$@" > /dev/null 2>&1; then
    echo "[fq] $nm 完成: $(grep 'final metrics' search/${nm}_$(hostname)/log.txt 2>/dev/null | tail -1)"
  else echo "[fq] $nm 失败"; fi }

run genome_hellaswag search_genome.py -n genome_hellaswag -e multiple_choice -d hellaswag -g 0,1,2,3 --seed 42
run genome_nlgraph  search_genome.py -n genome_nlgraph  -e exact_match   -d nlgraph  -g 0,1,2,3 --seed 42
run pso_nl_r01cfg search.py -n pso_nl_r01cfg -e exact_match -d nlgraph -g 0,1,2,3 -b Qwen/Qwen2.5-1.5B-Instruct --inertia 0.3 --cognitive_coeff 0.3 --social_coeff 0.3 --repel_coeff 0.1 --step_length 0.8 --populate_initial_experts 1 --initial_experts_num 20 --starting_test_set_eval 0 --fast_merge 1 --starting_velocity_mode random --repel_term 1 --step_length_factor 0.95 --restart_stray_particles 1 --restart_patience 0.5 --weight_randomess 1 --project_name_wb keg218-swarm-qwen2.5b -p 10 -m 30
run pso_nl_repel05 search.py -n pso_nl_repel05 -e exact_match -d nlgraph -g 0,1,2,3 -b Qwen/Qwen2.5-1.5B-Instruct --inertia 0.2 --cognitive_coeff 0.2 --social_coeff 0.4 --repel_coeff 0.05 --step_length 0.9 --populate_initial_experts 1 --initial_experts_num 20 --starting_test_set_eval 0 --fast_merge 1 --starting_velocity_mode random --repel_term 1 --step_length_factor 0.95 --restart_stray_particles 1 --restart_patience 0.5 --weight_randomess 1 --project_name_wb keg218-swarm-qwen2.5b -p 10 -m 30
run lp_v4_nlgraph_s43 search_lp.py -n lp_v4_nlgraph_s43 -e exact_match -d nlgraph -g 0,1,2,3 --popsize 20 --max_iter 25 --seed 43
run sgms_v1_nlgraph_s43 search_sgms.py -n sgms_v1_nlgraph_s43 -e exact_match -d nlgraph -g 0,1,2,3 --lb -1.5 --ub 2.0 --seed 43
echo "[fq] 全部结束 $(date)"
"$PY" compare_all.py || true
