#!/usr/bin/env bash
# laneA 续接版: GPU 0,1, 等 legal(在跑)完后续接后续任务
set -euo pipefail
cd /data/lvliping/merge_swarm
if [[ -z "${LA:-}" ]]; then LA=1 setsid nohup "$0" >> sumo_full.log 2>&1 < /dev/null & echo "laneA续接已启动"; exit 0; fi
while pgrep -f "search_sum[o].py.*legal" > /dev/null; do sleep 120; done
PY=/data/lvliping/miniconda3/envs/swarm/bin/python
run () {
  local t=$1; local R=()
  case "$t" in
    science)   R=(-e multitask -d science);;
    kcross)    R=(-e multiple_choice -d knowledge_crosswords);;
    mmlu)      R=(-e multiple_choice -d mmlu);;
    mmlu_pro)  R=(-e multiple_choice -d mmlu_pro);;
  esac
  local nm="sumo2_${t}_s42"
  if [[ -f "search/${nm}_$(hostname)/log.txt" ]] && grep -q "final metrics" "search/${nm}_$(hostname)/log.txt" 2>/dev/null; then echo "[laneA] $nm 已完成,跳过"; return 0; fi
  echo "[laneA] $(date '+%m-%d %H:%M') $nm 开始"
  rm -rf "search/${nm}_218keg"
  if "$PY" search_sumo.py -n $nm "${R[@]}" -g 0,1 --seed 42 --no_pair_seed > /tmp/laneA_last.out 2> /tmp/laneA_last.err; then
    echo "[laneA] $nm 完成: $(grep 'final metrics' search/${nm}_$(hostname)/log.txt | tail -1 | grep -oE 'ending_best_dev=[0-9.]+ ending_best_test=[0-9.]+')"
  else
    echo "[laneA] $nm 失败: $(tail -1 /tmp/laneA_last.err | head -c 120)"
  fi
  rm -rf "search/${nm}_218keg/tmp" 2>/dev/null || true
}
for t in science kcross mmlu mmlu_pro; do run $t; done
echo "[laneA] ALL DONE $(date)"
