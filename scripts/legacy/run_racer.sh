#!/usr/bin/env bash
# RACER 首发验证链：nlgraph/gsm8k/culture/medical 各 seed 42
set -euo pipefail
cd /data/lvliping/merge_swarm
if [[ -z "${RC:-}" ]]; then RC=1 setsid nohup "$0" >> racer.log 2>&1 < /dev/null & echo "RACER链已启动(待命)"; exit 0; fi
while pgrep -f "swarm/bin/pytho[n] search" > /dev/null; do sleep 120; done
PY=/data/lvliping/miniconda3/envs/swarm/bin/python
run () {
  local nm="$1"; shift
  echo "[rc] $(date '+%m-%d %H:%M') $nm 开始 (剩余: $(df --output=avail -h /data | tail -1 | tr -d ' '))"
  rm -rf "search/${nm}_218keg"
  if "$PY" "$@" > /dev/null 2>&1; then
    echo "[rc] $nm 完成: $(grep 'final metrics' search/${nm}_$(hostname)/log.txt | tail -1 | grep -oE 'starting_best_dev=[0-9.]+ ending_best_dev=[0-9.]+ ending_best_test=[0-9.]+')"
  else
    echo "[rc] $nm 失败: $(grep -A2 FATAL search/${nm}_$(hostname)/log.txt 2>/dev/null | tail -1 | head -c 120)"
  fi
  rm -rf "search/${nm}_218keg/tmp" 2>/dev/null || true
}
run racer_nlgraph_s42 search_racer.py -n racer_nlgraph_s42 -e exact_match -d nlgraph -g 0,1,2,3 --seed 42
run racer_gsm8k_s42   search_racer.py -n racer_gsm8k_s42   -e exact_match -d gsm8k   -g 0,1,2,3 --seed 42
run racer_culture_s42 search_racer.py -n racer_culture_s42 -e multitask   -d culture -g 0,1,2,3 --seed 42
run racer_medical_s42 search_racer.py -n racer_medical_s42 -e multitask   -d medical -g 0,1,2,3 --seed 42
echo "[rc] 全部结束 $(date)"
