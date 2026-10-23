#!/usr/bin/env bash
# 补跑链: mmlu/medical/abstainqa (被误清tmp害死的3场)
set -euo pipefail
cd /data/lvliping/merge_swarm
if [[ -z "${R3:-}" ]]; then R3=1 setsid nohup "$0" >> repair3.log 2>&1 < /dev/null & echo "补跑链已启动"; exit 0; fi
while pgrep -f "swarm/bin/pytho[n] search" > /dev/null; do sleep 180; done
PY=/data/lvliping/miniconda3/envs/swarm/bin/python
run () {
  local t=$1; local R=()
  case "$t" in
    mmlu)      R=(-e multiple_choice -d mmlu);;
    medical)   R=(-e multitask -d medical);;
    abstainqa) R=(-e AbstainQA -d mmlu);;
  esac
  local nm="sumo2_${t}_s42"
  if [[ -f "search/${nm}_$(hostname)/log.txt" ]] && grep -q "final metrics" "search/${nm}_$(hostname)/log.txt" 2>/dev/null; then echo "[r3] $nm 已完成,跳过"; return 0; fi
  echo "[r3] $(date '+%m-%d %H:%M') $nm 开始"
  rm -rf "search/${nm}_218keg"
  if "$PY" search_sumo.py -n $nm "${R[@]}" -g 0,1,2,3 --seed 42 --no_pair_seed > /tmp/r3_last.out 2> /tmp/r3_last.err; then
    echo "[r3] $nm 完成: $(grep 'final metrics' search/${nm}_$(hostname)/log.txt | tail -1 | grep -oE 'ending_best_dev=[0-9.]+ ending_best_test=[0-9.]+')"
  else
    echo "[r3] $nm 失败"
  fi
  rm -rf "search/${nm}_218keg/tmp" 2>/dev/null || true
}
for t in mmlu medical abstainqa; do run $t; done
echo "[r3] 全部结束 $(date)"
