#!/usr/bin/env bash
# laneB 独立版: GPU 2,3 专用, 无需等待(探测器已完成, GPU 2,3 空闲)
set -euo pipefail
cd /data/lvliping/merge_swarm
if [[ -z "${LB:-}" ]]; then LB=1 setsid nohup "$0" >> sumo_full.log 2>&1 < /dev/null & echo "laneB已启动"; exit 0; fi
PY=/data/lvliping/miniconda3/envs/swarm/bin/python
run () {
  local t=$1; local R=()
  case "$t" in
    truthfulqa) R=(-e multiple_choice -d truthfulqa);;
    culture)   R=(-e multitask -d culture);;
    medical)   R=(-e multitask -d medical);;
    abstainqa) R=(-e AbstainQA -d mmlu);;
  esac
  local nm="sumo2_${t}_s42"
  if [[ -f "search/${nm}_$(hostname)/log.txt" ]] && grep -q "final metrics" "search/${nm}_$(hostname)/log.txt" 2>/dev/null; then echo "[laneB] $nm 已完成,跳过"; return 0; fi
  echo "[laneB] $(date '+%m-%d %H:%M') $nm 开始"
  rm -rf "search/${nm}_218keg"
  if "$PY" search_sumo.py -n $nm "${R[@]}" -g 2,3 --seed 42 --no_pair_seed > /tmp/laneB_last.out 2> /tmp/laneB_last.err; then
    echo "[laneB] $nm 完成: $(grep 'final metrics' search/${nm}_$(hostname)/log.txt | tail -1 | grep -oE 'ending_best_dev=[0-9.]+ ending_best_test=[0-9.]+')"
  else
    echo "[laneB] $nm 失败: $(tail -1 /tmp/laneB_last.err | head -c 120)"
  fi
  rm -rf "search/${nm}_218keg/tmp" 2>/dev/null || true
}
for t in truthfulqa culture medical abstainqa; do run $t; done
echo "[laneB] ALL DONE $(date)"
