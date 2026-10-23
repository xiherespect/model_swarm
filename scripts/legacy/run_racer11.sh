#!/usr/bin/env bash
# RACER v1.1 验证: 4任务 × s42/s43（判据: 均值≥v1.0且单任务跌幅≤1分）
set -euo pipefail
cd /data/lvliping/merge_swarm
if [[ -z "${R11:-}" ]]; then R11=1 setsid nohup "$0" >> racer11.log 2>&1 < /dev/null & echo "v1.1验证链已启动"; exit 0; fi
while pgrep -f "swarm/bin/pytho[n] search" > /dev/null; do sleep 120; done
PY=/data/lvliping/miniconda3/envs/swarm/bin/python
run () {
  local nm="$1"; shift
  if [[ -f "search/${nm}_$(hostname)/log.txt" ]] && grep -q "final metrics" "search/${nm}_$(hostname)/log.txt" 2>/dev/null; then echo "[r11] $nm 已完成,跳过"; return 0; fi
  echo "[r11] $(date '+%m-%d %H:%M') $nm 开始"
  rm -rf "search/${nm}_218keg"
  if "$PY" search_racer.py "$@" > /tmp/racer11_last.out 2> /tmp/racer11_last.err; then
    echo "[r11] $nm 完成: $(grep 'final metrics' search/${nm}_$(hostname)/log.txt | tail -1 | grep -oE 'ending_best_dev=[0-9.]+ ending_best_test=[0-9.]+')"
  else
    echo "[r11] $nm 失败: $(tail -1 /tmp/racer11_last.err | head -c 120)"
  fi
  rm -rf "search/${nm}_218keg/tmp" 2>/dev/null || true
}
T () {
  local t=$1 s=$2; local R=()
  case "$t" in
    nlgraph)   R=(-e exact_match -d nlgraph);;
    gsm8k)     R=(-e exact_match -d gsm8k);;
    hellaswag) R=(-e multiple_choice -d hellaswag);;
    culture)   R=(-e multitask -d culture);;
  esac
  run racer11_${t}_s${s} -n racer11_${t}_s${s} "${R[@]}" -g 0,1,2,3 --seed ${s}
}
for s in 42 43; do for t in nlgraph gsm8k hellaswag culture; do T $t $s; done; done
echo "[r11] 全部结束 $(date)"
