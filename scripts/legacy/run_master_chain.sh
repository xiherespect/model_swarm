#!/usr/bin/env bash
# 主决策链：SGMS链(PID 2380202)结束后 → v1<0.70则自动跑v2 → v1p → meal2(medical×4+gsm8k×2)
set -euo pipefail
REPO_DIR="/data/lvliping/merge_swarm"
PYTHON="/data/lvliping/miniconda3/envs/swarm/bin/python"
export WANDB_ENTITY="llp031004-yanshan"
cd "$REPO_DIR"
if [[ -z "${MASTER_CHAIN:-}" ]]; then
    MASTER_CHAIN=1 setsid nohup "$0" >> master_chain.log 2>&1 < /dev/null &
    echo "主决策链已挂起 (等 SGMS链 PID 2380202)"; exit 0
fi
echo "[master] $(date '+%m-%d %H:%M') 等待 SGMS 链..."
while kill -0 2390016 2>/dev/null; do sleep 30; done
sleep 30
pgrep -f "envs/swarm/bin/python search" > /dev/null && { echo "[master] 忙，放弃"; exit 1; }

# 读取 sgms_v1_hellaswag 终值，决定是否跑 v2
V1DEV=$("$PYTHON" -c "
import re
t = open('search/sgms_v1_hellaswag_218keg/log.txt', errors='ignore').read()
m = re.search(r'ending_best_dev=([\d.]+)', t)
print(m.group(1) if m else '0')" 2>/dev/null || echo 0)
echo "[master] SGMS v1 hellaswag ending_best_dev=$V1DEV"

if "$PYTHON" -c "exit(0 if float('$V1DEV' or 0) < 0.70 else 1)"; then
    echo "[master] v1 < 0.70 → 自动启动 v2（成分扩张+协同背景扰动）"
    "$PYTHON" search_sgms.py -n sgms_v2_hellaswag -e multiple_choice -d hellaswag -g 0,1,2,3         --expand_quota 0.15 --synth_bg --round_evals 6 --kappa_start 1.2 --seed 42 > /dev/null 2>&1         && echo "[master] v2 完成: $(grep 'final metrics' search/sgms_v2_hellaswag_$(hostname)/log.txt)"         || echo "[master] v2 失败"
else
    echo "[master] v1 ≥ 0.70 → 跳过 v2"
fi

echo "[master] v1p (pair-scan) 开始"
"$PYTHON" search_sgms.py -n sgms_v1p_hellaswag -e multiple_choice -d hellaswag -g 0,1,2,3     --init_pair_scan --seed 42 > /dev/null 2>&1     && echo "[master] v1p 完成: $(grep 'final metrics' search/sgms_v1p_hellaswag_$(hostname)/log.txt)"     || echo "[master] v1p 失败"

run () { echo "[master] $(date '+%H:%M') $1 开始"
  shift; if "$PYTHON" "$@" > /dev/null 2>&1; then echo "[master] 完成"; else echo "[master] 失败"; fi }
run pso_medical search.py -n pso_medical_r01 -e multitask -d medical -g 0,1,2,3 -b Qwen/Qwen2.5-1.5B-Instruct --inertia 0.3 --cognitive_coeff 0.3 --social_coeff 0.3 --repel_coeff 0.1 --step_length 0.8 --populate_initial_experts 1 --initial_experts_num 20 --starting_test_set_eval 0 --fast_merge 1 --starting_velocity_mode random --repel_term 1 --step_length_factor 0.95 --restart_stray_particles 1 --restart_patience 0.5 --weight_randomess 1 --project_name_wb keg218-swarm-qwen2.5b -p 10 -m 30
run lp_medical search_lp.py -n lp_v4_medical -e multitask -d medical -g 0,1,2,3 --popsize 20 --max_iter 25 --seed 42
run genome_medical search_genome.py -n genome_medical -e multitask -d medical -g 0,1,2,3 --seed 42
run sgms_medical search_sgms.py -n sgms_v1_medical -e multitask -d medical -g 0,1,2,3 --seed 42
run pso_gsm8k search.py -n pso_gsm8k_r01 -e exact_match -d gsm8k -g 0,1,2,3 -b Qwen/Qwen2.5-1.5B-Instruct --inertia 0.3 --cognitive_coeff 0.3 --social_coeff 0.3 --repel_coeff 0.1 --step_length 0.8 --populate_initial_experts 1 --initial_experts_num 20 --starting_test_set_eval 0 --fast_merge 1 --starting_velocity_mode random --repel_term 1 --step_length_factor 0.95 --restart_stray_particles 1 --restart_patience 0.5 --weight_randomess 1 --project_name_wb keg218-swarm-qwen2.5b -p 10 -m 30
run sgms_gsm8k search_sgms.py -n sgms_v1_gsm8k -e exact_match -d gsm8k -g 0,1,2,3 --seed 42 --lb -1.5 --ub 2.0
echo "[master] 全部结束 $(date)"
"$PYTHON" compare_all.py || true
