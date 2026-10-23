#!/usr/bin/env bash
# 对局1: A(复用已有) vs B(极简嫁接) vs C(混合) —— 单种子s42
set -euo pipefail
cd /data/lvliping/merge_swarm
if [[ -z "${M1:-}" ]]; then M1=1 setsid nohup "$0" >> match1.log 2>&1 < /dev/null & echo "对局1链已启动(待命)"; exit 0; fi
while pgrep -f "swarm/bin/pytho[n] search" > /dev/null || pgrep -f "run_scaling.s[h]" > /dev/null; do sleep 180; done
PY=/data/lvliping/miniconda3/envs/swarm/bin/python
run () {
  local nm="$1"; shift
  if [[ -f "search/${nm}_$(hostname)/log.txt" ]] && grep -q "final metrics" "search/${nm}_$(hostname)/log.txt" 2>/dev/null; then echo "[m1] $nm 已完成,跳过"; return 0; fi
  echo "[m1] $(date '+%m-%d %H:%M') $nm 开始"
  rm -rf "search/${nm}_218keg"
  if "$PY" search_racer.py "$@" > /tmp/m1_last.out 2> /tmp/m1_last.err; then
    echo "[m1] $nm 完成: $(grep 'final metrics' search/${nm}_$(hostname)/log.txt | tail -1 | grep -oE 'ending_best_dev=[0-9.]+ ending_best_test=[0-9.]+')"
  else
    echo "[m1] $nm 失败: $(tail -1 /tmp/m1_last.err | head -c 120)"
  fi
  rm -rf "search/${nm}_218keg/tmp" 2>/dev/null || true
}
T () {  # $1=task $2=选手B/C $3..=旗标
  local t=$1 p=$2; shift 2; local R=()
  case "$t" in
    nlgraph)   R=(-e exact_match -d nlgraph);;
    gsm8k)     R=(-e exact_match -d gsm8k);;
    science)   R=(-e multitask -d science);;
    hellaswag) R=(-e multiple_choice -d hellaswag);;
  esac
  run m1${p}_${t}_s42 -n m1${p}_${t}_s42 "${R[@]}" -g 0,1,2,3 --seed 42 "$@"
}
# 主裁决两场先打
T nlgraph  B --full_cma --no_de --no_sum --no_guard --no_plateau
T nlgraph  C --full_cma
T hellaswag B --full_cma --no_de --no_sum --no_guard --no_plateau
T hellaswag C --full_cma
# 参考场
T gsm8k   B --full_cma --no_de --no_sum --no_guard --no_plateau
T gsm8k   C --full_cma
T science B --full_cma --no_de --no_sum --no_guard --no_plateau
T science C --full_cma
echo "[m1] 全部结束 $(date)"
