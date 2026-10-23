#!/usr/bin/env bash
# SUMO 首发验证: 3任务×s42, 对照 RACER v1.3 (nlgraph .592/gsm8k .435/hsw .615)
set -euo pipefail
cd /data/lvliping/merge_swarm
if [[ -z "${SU:-}" ]]; then SU=1 setsid nohup "$0" >> sumo.log 2>&1 < /dev/null & echo "SUMO链已启动(待命)"; exit 0; fi
while pgrep -f "swarm/bin/pytho[n] search" > /dev/null || pgrep -f "run_match1.s[h]" > /dev/null || pgrep -f "run_scaling.s[h]" > /dev/null; do sleep 180; done
PY=/data/lvliping/miniconda3/envs/swarm/bin/python
run () {
  local nm="$1"; shift
  if [[ -f "search/${nm}_$(hostname)/log.txt" ]] && grep -q "final metrics" "search/${nm}_$(hostname)/log.txt" 2>/dev/null; then echo "[su] $nm 已完成,跳过"; return 0; fi
  echo "[su] $(date '+%m-%d %H:%M') $nm 开始"
  rm -rf "search/${nm}_218keg"
  if "$PY" search_sumo.py "$@" > /tmp/sumo_last.out 2> /tmp/sumo_last.err; then
    echo "[su] $nm 完成: $(grep 'final metrics' search/${nm}_$(hostname)/log.txt | tail -1 | grep -oE 'ending_best_dev=[0-9.]+ ending_best_test=[0-9.]+ equiv_total=[0-9]+')"
  else
    echo "[su] $nm 失败: $(tail -1 /tmp/sumo_last.err | head -c 120)"
  fi
  rm -rf "search/${nm}_218keg/tmp" 2>/dev/null || true
}
run sumo_nlgraph_s42 -n sumo_nlgraph_s42 -e exact_match -d nlgraph -g 0,1,2,3 --seed 42
run sumo_gsm8k_s42   -n sumo_gsm8k_s42   -e exact_match -d gsm8k   -g 0,1,2,3 --seed 42
run sumo_hsw_s42     -n sumo_hsw_s42     -e multiple_choice -d hellaswag -g 0,1,2,3 --seed 42
echo "[su] 全部结束 $(date)"
