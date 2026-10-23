#!/usr/bin/env bash
# PSO 基线全量补齐：4 单任务MC + 1 AbstainQA + 2 multitask（r01 配置, seed 42）
set -euo pipefail
cd /data/lvliping/merge_swarm
if [[ -z "${PF:-}" ]]; then PF=1 setsid nohup "$0" >> pso_full.log 2>&1 < /dev/null & echo "PSO基线补齐链已启动"; exit 0; fi
while pgrep -f "swarm/bin/pytho[n] search" > /dev/null; do sleep 120; done
PY=/data/lvliping/miniconda3/envs/swarm/bin/python
run () {
  local nm="$1"; shift
  echo "[pf] $(date '+%m-%d %H:%M') $nm 开始 (剩余: $(df --output=avail -h /data | tail -1 | tr -d ' '))"
  rm -rf "search/${nm}_218keg"
  if "$PY" "$@" > /dev/null 2>&1; then
    echo "[pf] $nm 完成"
  else
    echo "[pf] $nm 退出码非零(cosmetic 收尾崩溃可后处理): $(grep -c 'final metrics' search/${nm}_$(hostname)/log.txt 2>/dev/null) 条 final 记录"
  fi
  rm -rf "search/${nm}_218keg/tmp" "search/${nm}_218keg/particle_"{now,velocity,personal_best,x_w,g_x,p_x} 2>/dev/null || true
}
PSO_ARGS="-g 0,1,2,3 -b Qwen/Qwen2.5-1.5B-Instruct --inertia 0.3 --cognitive_coeff 0.3 --social_coeff 0.3 --repel_coeff 0.1 --step_length 0.8 --populate_initial_experts 1 --initial_experts_num 20 --starting_test_set_eval 0 --fast_merge 1 --starting_velocity_mode random --repel_term 1 --step_length_factor 0.95 --restart_stray_particles 1 --restart_patience 0.5 --weight_randomess 1 --project_name_wb keg218-swarm-qwen2.5b -p 10 -m 30"
run pso_kcross_r01      search.py -n pso_kcross_r01      -e multiple_choice -d knowledge_crosswords $PSO_ARGS
run pso_mmlu_r01        search.py -n pso_mmlu_r01        -e multiple_choice -d mmlu               $PSO_ARGS
run pso_truthfulqa_r01  search.py -n pso_truthfulqa_r01  -e multiple_choice -d truthfulqa         $PSO_ARGS
run pso_mmlu_pro_r01    search.py -n pso_mmlu_pro_r01    -e multiple_choice -d mmlu_pro           $PSO_ARGS
run pso_abstainqa_r01   search.py -n pso_abstainqa_r01   -e AbstainQA       -d mmlu               $PSO_ARGS
run pso_science_r01     search.py -n pso_science_r01     -e multitask       -d science            $PSO_ARGS
run pso_culture_r01     search.py -n pso_culture_r01     -e multitask       -d culture            $PSO_ARGS
echo "[pf] 全部结束 $(date)"
