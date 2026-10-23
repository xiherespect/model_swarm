#!/usr/bin/env bash
# legal 对照实验：PSO vs RACER（seed 42，本机4卡）
set -euo pipefail
cd /data/lvliping/merge_swarm
if [[ -z "${LG:-}" ]]; then LG=1 setsid nohup "$0" >> legal.log 2>&1 < /dev/null & echo "legal对照链已启动"; exit 0; fi
while pgrep -f "swarm/bin/pytho[n] search" > /dev/null; do sleep 120; done
PY=/data/lvliping/miniconda3/envs/swarm/bin/python
run () {
  local nm="$1"; shift
  echo "[lg] $(date '+%m-%d %H:%M') $nm 开始"
  rm -rf "search/${nm}_218keg"
  if "$PY" "$@" > /dev/null 2>&1; then
    echo "[lg] $nm 完成: $(grep 'final metrics' search/${nm}_$(hostname)/log.txt | tail -1 | grep -oE 'starting_best_dev=[0-9.]+ ending_best_dev=[0-9.]+ ending_best_test=[0-9.]+')"
  else
    echo "[lg] $nm 失败: $(grep -A2 FATAL search/${nm}_$(hostname)/log.txt 2>/dev/null | tail -1 | head -c 120)"
  fi
  rm -rf "search/${nm}_218keg/tmp" 2>/dev/null || true
}
run pso_legal_r01 search.py -n pso_legal_r01 -e multitask -d legal -g 0,1,2,3 -b Qwen/Qwen2.5-1.5B-Instruct --inertia 0.3 --cognitive_coeff 0.3 --social_coeff 0.3 --repel_coeff 0.1 --step_length 0.8 --populate_initial_experts 1 --initial_experts_num 20 --starting_test_set_eval 0 --fast_merge 1 --starting_velocity_mode random --repel_term 1 --step_length_factor 0.95 --restart_stray_particles 1 --restart_patience 0.5 --weight_randomess 1 --project_name_wb keg218-swarm-qwen2.5b -p 10 -m 30
run racer_legal_s42 search_racer.py -n racer_legal_s42 -e multitask -d legal -g 0,1,2,3 --seed 42
echo "[lg] 全部结束 $(date)"
