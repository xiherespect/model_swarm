#!/usr/bin/env bash
# 预算 scaling 曲线: 3任务 × {55,110,220} × s42 = 9 run
set -euo pipefail
cd /data/lvliping/merge_swarm
if [[ -z "${SC:-}" ]]; then SC=1 setsid nohup "$0" >> scaling.log 2>&1 < /dev/null & echo "scaling链已启动"; exit 0; fi
while pgrep -f "swarm/bin/pytho[n] search" > /dev/null; do sleep 120; done
PY=/data/lvliping/miniconda3/envs/swarm/bin/python
run () {
  local nm="$1"; shift
  if [[ -f "search/${nm}_$(hostname)/log.txt" ]] && grep -q "final metrics" "search/${nm}_$(hostname)/log.txt" 2>/dev/null; then echo "[sc] $nm 已完成,跳过"; return 0; fi
  echo "[sc] $(date '+%m-%d %H:%M') $nm 开始"
  rm -rf "search/${nm}_218keg"
  if "$PY" search_racer.py "$@" > /tmp/sc_last.out 2> /tmp/sc_last.err; then
    echo "[sc] $nm 完成: $(grep 'final metrics' search/${nm}_$(hostname)/log.txt | tail -1 | grep -oE 'ending_best_dev=[0-9.]+ ending_best_test=[0-9.]+ .*equiv=[0-9]+')"
  else
    echo "[sc] $nm 失败: $(tail -1 /tmp/sc_last.err | head -c 120)"
  fi
  rm -rf "search/${nm}_218keg/tmp" 2>/dev/null || true
}
T () {  # task iter tag
  local t=$1 it=$2 tg=$3; local R=()
  case "$t" in
    nlgraph) R=(-e exact_match -d nlgraph);;
    gsm8k)   R=(-e exact_match -d gsm8k);;
    legal)   R=(-e multitask -d legal);;
  esac
  run scale_${tg}_${t}_s42 -n scale_${tg}_${t}_s42 "${R[@]}" -g 0,1,2,3 --seed 42 --max_iter ${it}
}
# 先 220（最长, 最先出关键信息）再 110 已有(racerf 复用) 再 55
for t in nlgraph gsm8k legal; do T $t 50 b220; done
for t in nlgraph gsm8k legal; do T $t 12 b55; done
echo "[sc] 全部结束 $(date)"
