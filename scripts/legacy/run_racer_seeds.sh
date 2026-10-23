#!/usr/bin/env bash
# RACER 多种子验证：nlgraph/gsm8k 各 s43/s44（方差是否被守卫杀死）
set -euo pipefail
cd /data/lvliping/merge_swarm
if [[ -z "${RS:-}" ]]; then RS=1 setsid nohup "$0" >> racer_seeds.log 2>&1 < /dev/null & echo "RACER种子链已启动"; exit 0; fi
while pgrep -f "swarm/bin/pytho[n] search" > /dev/null; do sleep 120; done
PY=/data/lvliping/miniconda3/envs/swarm/bin/python
run () {
  local nm="$1"; shift
  echo "[rs] $(date '+%m-%d %H:%M') $nm 开始 (剩余: $(df --output=avail -h /data | tail -1 | tr -d ' '))"
  rm -rf "search/${nm}_218keg"
  if "$PY" "$@" > /dev/null 2>&1; then
    echo "[rs] $nm 完成: $(grep 'final metrics' search/${nm}_$(hostname)/log.txt | tail -1 | grep -oE 'starting_best_dev=[0-9.]+ ending_best_dev=[0-9.]+ ending_best_test=[0-9.]+')"
  else
    echo "[rs] $nm 失败"
  fi
  rm -rf "search/${nm}_218keg/tmp" 2>/dev/null || true
}
run racer_nlgraph_s43 search_racer.py -n racer_nlgraph_s43 -e exact_match -d nlgraph -g 0,1,2,3 --seed 43
run racer_nlgraph_s44 search_racer.py -n racer_nlgraph_s44 -e exact_match -d nlgraph -g 0,1,2,3 --seed 44
run racer_gsm8k_s43   search_racer.py -n racer_gsm8k_s43   -e exact_match -d gsm8k   -g 0,1,2,3 --seed 43
run racer_gsm8k_s44   search_racer.py -n racer_gsm8k_s44   -e exact_match -d gsm8k   -g 0,1,2,3 --seed 44
echo "[rs] 全部结束 $(date)"
