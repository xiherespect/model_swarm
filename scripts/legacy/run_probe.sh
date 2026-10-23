#!/usr/bin/env bash
# 归因探针: nlgraph s42 逐刀开关, 定位 v1.1 在 nlgraph 的拖累源
set -euo pipefail
cd /data/lvliping/merge_swarm
if [[ -z "${PB:-}" ]]; then PB=1 setsid nohup "$0" >> probe.log 2>&1 < /dev/null & echo "探针链已排队"; exit 0; fi
while pgrep -f "run_racer1[1].sh" > /dev/null || pgrep -f "swarm/bin/pytho[n] search" > /dev/null; do sleep 120; done
PY=/data/lvliping/miniconda3/envs/swarm/bin/python
run () {
  local nm="$1"; shift
  echo "[pb] $(date '+%m-%d %H:%M') $nm 开始"
  rm -rf "search/${nm}_218keg"
  if "$PY" search_racer.py "$@" > /tmp/probe_last.out 2> /tmp/probe_last.err; then
    echo "[pb] $nm 完成: $(grep 'final metrics' search/${nm}_$(hostname)/log.txt | tail -1 | grep -oE 'ending_best_dev=[0-9.]+ ending_best_test=[0-9.]+')"
  else echo "[pb] $nm 失败"; fi
  rm -rf "search/${nm}_218keg/tmp" 2>/dev/null || true
}
# 探针1: 只关 C 和 D (A+B' 单独在场) —— 若回到 0.60+, 拖累= C 或 D
run probe_nlg_noCD -n probe_nlg_noCD -e exact_match -d nlgraph -g 0,1,2,3 --seed 42 --fixed_sum --no_stretch
# 探针2: 只关 C (D 在场) —— 区分 C vs D
run probe_nlg_noC -n probe_nlg_noC -e exact_match -d nlgraph -g 0,1,2,3 --seed 42 --fixed_sum
echo "[pb] 全部结束 $(date)"
