#!/usr/bin/env bash
# SUMO v0.2 全量: 剩余 9 任务 × s42（gsm8k/nlgraph/hsw 已有 v0.2 数据）
set -euo pipefail
cd /data/lvliping/merge_swarm
if [[ -z "${SF:-}" ]]; then SF=1 setsid nohup "$0" >> sumo_full.log 2>&1 < /dev/null & echo "SUMO全量链已排队"; exit 0; fi
while pgrep -f "swarm/bin/pytho[n] search" > /dev/null || pgrep -f "probe_singles.p[y]" > /dev/null; do sleep 180; done
PY=/data/lvliping/miniconda3/envs/swarm/bin/python
run () {
  local nm="$1"; shift
  if [[ -f "search/${nm}_$(hostname)/log.txt" ]] && grep -q "final metrics" "search/${nm}_$(hostname)/log.txt" 2>/dev/null; then echo "[sf] $nm 已完成,跳过"; return 0; fi
  echo "[sf] $(date '+%m-%d %H:%M') $nm 开始"
  rm -rf "search/${nm}_218keg"
  if "$PY" search_sumo.py "$@" > /tmp/sf_last.out 2> /tmp/sf_last.err; then
    echo "[sf] $nm 完成: $(grep 'final metrics' search/${nm}_$(hostname)/log.txt | tail -1 | grep -oE 'ending_best_dev=[0-9.]+ ending_best_test=[0-9.]+')"
  else
    echo "[sf] $nm 失败: $(tail -1 /tmp/sf_last.err | head -c 120)"
  fi
  rm -rf "search/${nm}_218keg/tmp" 2>/dev/null || true
}
T () {
  local t=$1; local R=()
  case "$t" in
    legal)     R=(-e multitask -d legal);;
    science)   R=(-e multitask -d science);;
    medical)   R=(-e multitask -d medical);;
    culture)   R=(-e multitask -d culture);;
    kcross)    R=(-e multiple_choice -d knowledge_crosswords);;
    mmlu)      R=(-e multiple_choice -d mmlu);;
    mmlu_pro)  R=(-e multiple_choice -d mmlu_pro);;
    truthfulqa) R=(-e multiple_choice -d truthfulqa);;
    abstainqa) R=(-e AbstainQA -d mmlu);;
  esac
  run sumo2_${t}_s42 -n sumo2_${t}_s42 "${R[@]}" -g 0,1,2,3 --seed 42
}
for t in legal science kcross mmlu mmlu_pro truthfulqa culture medical abstainqa; do T $t; done
echo "[sf] 全部结束 $(date)"
