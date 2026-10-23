#!/usr/bin/env bash
set -euo pipefail
cd /data/lvliping/merge_swarm
if [[ -z "${AF12:-}" ]]; then AF12=1 setsid nohup "$0" >> amphi_full.log 2>&1 < /dev/null & echo "AMPHI全量链已启动"; exit 0; fi
PY=/data/lvliping/miniconda3/envs/swarm/bin/python

laneA () {
  run_amphi laneA 0,1 legal multitask legal
  run_amphi laneA 0,1 medical multitask medical
  run_amphi laneA 0,1 kcross multiple_choice knowledge_crosswords
  run_amphi laneA 0,1 mmlu multiple_choice mmlu
  echo "[laneA] DONE $(date)"
}
laneB () {
  run_amphi laneB 2,3 mmlu_pro multiple_choice mmlu_pro
  run_amphi laneB 2,3 truthfulqa multiple_choice truthfulqa
  run_amphi laneB 2,3 abstainqa AbstainQA mmlu
  # hsw 也要补(之前用了错的eval_type)
  run_amphi laneB 2,3 hsw multiple_choice hellaswag
  echo "[laneB] DONE $(date)"
}
run_amphi () {
  local lane=$1 gp=$2 nm=$3 et=$4 ds=$5
  local full="amphi12_${nm}"
  if [[ -f "search/${full}_$(hostname)/log.txt" ]] && grep -q "child exit" "search/${full}_$(hostname)/log.txt" 2>/dev/null; then
    echo "[$lane] $nm 已完成,跳过"; return 0
  fi
  echo "[$lane] $(date '+%m-%d %H:%M') $nm 开始"
  rm -rf "search/${full}_218keg" "search/${full}_s_218keg" "search/${full}_r_218keg"
  if "$PY" search_amphi.py -n $full -e $et -d $ds -g $gp --no_wandb > /tmp/${lane}_${nm}.out 2>&1; then
    local route=$(grep -oE 'KEEP SUMO|SWITCH → RACER' "search/${full}_$(hostname)/log.txt" 2>/dev/null | head -1)
    local result=$(grep 'final metrics' "search/${full}_{s,r}_$(hostname)/log.txt" 2>/dev/null | tail -1 | grep -oE 'ending_best_dev=[0-9.]+ ending_best_test=[0-9.]+')
    echo "[$lane] $nm 完成 | 路由: $route | $result"
  else
    echo "[$lane] $nm 失败"
  fi
  rm -rf "search/${full}_s_218keg/tmp" "search/${full}_r_218keg/tmp" 2>/dev/null || true
}

laneA & laneB & wait
echo "[AF12] 全部结束 $(date)"
