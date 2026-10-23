#!/usr/bin/env bash
# SUMO v0.2 验证: nlgraph(冷启动修复?) + gsm8k(配对种子挤占预算是否伤及) × s42
set -euo pipefail
cd /data/lvliping/merge_swarm
if [[ -z "${S2:-}" ]]; then S2=1 setsid nohup "$0" >> sumo2.log 2>&1 < /dev/null & echo "SUMO v0.2链已启动"; exit 0; fi
while pgrep -f "swarm/bin/pytho[n] search" > /dev/null; do sleep 120; done
PY=/data/lvliping/miniconda3/envs/swarm/bin/python
run () {
  local nm="$1"; shift
  if [[ -f "search/${nm}_$(hostname)/log.txt" ]] && grep -q "final metrics" "search/${nm}_$(hostname)/log.txt" 2>/dev/null; then echo "[s2] $nm 已完成,跳过"; return 0; fi
  echo "[s2] $(date '+%m-%d %H:%M') $nm 开始"
  rm -rf "search/${nm}_218keg"
  if "$PY" search_sumo.py "$@" > /tmp/sumo2_last.out 2> /tmp/sumo2_last.err; then
    echo "[s2] $nm 完成: $(grep 'final metrics' search/${nm}_$(hostname)/log.txt | tail -1 | grep -oE 'ending_best_dev=[0-9.]+ ending_best_test=[0-9.]+')"
  else
    echo "[s2] $nm 失败: $(tail -1 /tmp/sumo2_last.err | head -c 120)"
  fi
  rm -rf "search/${nm}_218keg/tmp" 2>/dev/null || true
}
run sumo2_nlgraph_s42 -n sumo2_nlgraph_s42 -e exact_match -d nlgraph -g 0,1,2,3 --seed 42
run sumo2_gsm8k_s42   -n sumo2_gsm8k_s42   -e exact_match -d gsm8k   -g 0,1,2,3 --seed 42
echo "[s2] 全部结束 $(date)"
