#!/usr/bin/env bash
# 后续链：sgms_gsm8k 结束后接力。优先解决 LP 种子方差问题。
set -euo pipefail
cd /data/lvliping/merge_swarm
if [[ -z "${FU:-}" ]]; then FU=1 setsid nohup "$0" >> followup.log 2>&1 < /dev/null & echo "后续链已启动"; exit 0; fi
# 等待现有搜索结束
while pgrep -f "swarm/bin/pytho[n] search" > /dev/null; do sleep 120; done
df -h /data | tail -1
PY=/data/lvliping/miniconda3/envs/swarm/bin/python
run () { local nm="$1"; shift
  echo "[fu] $(date '+%m-%d %H:%M') $nm 开始 (剩余: $(df --output=avail -h /data | tail -1 | tr -d ' '))"
  rm -rf "search/${nm}_218keg"
  if "$PY" "$@" > /dev/null 2>&1; then
    echo "[fu] $nm 完成: $(grep 'final metrics' search/${nm}_$(hostname)/log.txt 2>/dev/null | tail -1 | grep -oE 'starting_best_dev=[0-9.]+ ending_best_dev=[0-9.]+ ending_best_test=[0-9.]+')"
  else echo "[fu] $nm 失败"; fi }
run lp_v4_nlgraph_s44 search_lp.py -n lp_v4_nlgraph_s44 -e exact_match -d nlgraph -g 0,1,2,3 --popsize 20 --max_iter 25 --seed 44
run lp_v4_gsm8k search_lp.py -n lp_v4_gsm8k -e exact_match -d gsm8k -g 0,1,2,3 --popsize 20 --max_iter 25 --seed 42
run pso_medical_r01 search.py -n pso_medical_r01 -e multitask -d medical -g 0,1,2,3 -b Qwen/Qwen2.5-1.5B-Instruct --inertia 0.3 --cognitive_coeff 0.3 --social_coeff 0.3 --repel_coeff 0.1 --step_length 0.8 --populate_initial_experts 1 --initial_experts_num 20 --starting_test_set_eval 0 --fast_merge 1 --starting_velocity_mode random --repel_term 1 --step_length_factor 0.95 --restart_stray_particles 1 --restart_patience 0.5 --weight_randomess 1 --project_name_wb keg218-swarm-qwen2.5b -p 10 -m 30
run sgms_v1_medical search_sgms.py -n sgms_v1_medical -e multitask -d medical -g 0,1,2,3 --seed 42
run lp_v4_medical search_lp.py -n lp_v4_medical -e multitask -d medical -g 0,1,2,3 --popsize 20 --max_iter 25 --seed 42
run genome_medical search_genome.py -n genome_medical -e multitask -d medical -g 0,1,2,3 --seed 42
echo "[fu] 全部结束 $(date)"
"$PY" compare_all.py || true
