#!/usr/bin/env bash
# 第三棒：等 SGMS 链 (PID 2380202) 结束后跑 medical 多任务×4算法 + gsm8k×2
set -euo pipefail
REPO_DIR="/data/lvliping/merge_swarm"
PYTHON="/data/lvliping/miniconda3/envs/swarm/bin/python"
export WANDB_ENTITY="llp031004-yanshan"
cd "$REPO_DIR"
if [[ -z "${M2_CHAIN:-}" ]]; then
    M2_CHAIN=1 setsid nohup "$0" >> meal2_chain.log 2>&1 < /dev/null &
    echo "meal2 链已挂起 (等 PID 2380202)"; exit 0
fi
while kill -0 2380202 2>/dev/null; do sleep 30; done
sleep 30
if pgrep -f "envs/swarm/bin/python search" > /dev/null; then echo "[meal2] 仍有进程，放弃"; exit 1; fi

run () { echo "[meal2] $(date '+%H:%M') $1 开始"
  shift; if "$PYTHON" "$@" > /dev/null 2>&1; then echo "[meal2] $(date '+%H:%M') 完成"; else echo "[meal2] 失败"; fi }

# medical 多任务 ×4
run pso_medical search.py -n pso_medical_r01 -e multitask -d medical -g 0,1,2,3 -b Qwen/Qwen2.5-1.5B-Instruct --inertia 0.3 --cognitive_coeff 0.3 --social_coeff 0.3 --repel_coeff 0.1 --step_length 0.8 --populate_initial_experts 1 --initial_experts_num 20 --starting_test_set_eval 0 --fast_merge 1 --starting_velocity_mode random --repel_term 1 --step_length_factor 0.95 --restart_stray_particles 1 --restart_patience 0.5 --weight_randomess 1 --project_name_wb keg218-swarm-qwen2.5b -p 10 -m 30
run lp_medical search_lp.py -n lp_v4_medical -e multitask -d medical -g 0,1,2,3 --popsize 20 --max_iter 25 --seed 42
run genome_medical search_genome.py -n genome_medical -e multitask -d medical -g 0,1,2,3 --seed 42
run sgms_medical search_sgms.py -n sgms_v1_medical -e multitask -d medical -g 0,1,2,3 --seed 42
# gsm8k 关键对照
run pso_gsm8k search.py -n pso_gsm8k_r01 -e exact_match -d gsm8k -g 0,1,2,3 -b Qwen/Qwen2.5-1.5B-Instruct --inertia 0.3 --cognitive_coeff 0.3 --social_coeff 0.3 --repel_coeff 0.1 --step_length 0.8 --populate_initial_experts 1 --initial_experts_num 20 --starting_test_set_eval 0 --fast_merge 1 --starting_velocity_mode random --repel_term 1 --step_length_factor 0.95 --restart_stray_particles 1 --restart_patience 0.5 --weight_randomess 1 --project_name_wb keg218-swarm-qwen2.5b -p 10 -m 30
run sgms_gsm8k search_sgms.py -n sgms_v1_gsm8k -e exact_match -d gsm8k -g 0,1,2,3 --seed 42
echo "[meal2] 全部结束 $(date)"
