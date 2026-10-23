#!/usr/bin/env bash
# pso-init 验证链：214 的 init_protocol=pso 在本机复现 + 种子扩展
set -euo pipefail
cd /data/lvliping/merge_swarm
if [[ -z "${IP:-}" ]]; then IP=1 setsid nohup "$0" >> ipso.log 2>&1 < /dev/null & echo "ipso链已启动"; exit 0; fi
while pgrep -f "swarm/bin/pytho[n] search" > /dev/null; do sleep 120; done
PY=/data/lvliping/miniconda3/envs/swarm/bin/python
run () { local nm="$1"; shift
  echo "[ip] $(date '+%m-%d %H:%M') $nm 开始 (剩余: $(df --output=avail -h /data | tail -1 | tr -d ' '))"
  rm -rf "search/${nm}_218keg"
  if "$PY" "$@" > /dev/null 2>&1; then
    echo "[ip] $nm 完成: $(grep 'final metrics' search/${nm}_$(hostname)/log.txt | tail -1 | grep -oE 'starting_best_dev=[0-9.]+ ending_best_dev=[0-9.]+ ending_best_test=[0-9.]+')"
  else echo "[ip] $nm 失败"; fi }
run sgms_ipso_nlgraph_s42 search_sgms.py -n sgms_ipso_nlgraph_s42 -e exact_match -d nlgraph -g 0,1,2,3 --init_protocol pso --seed 42
run sgms_ipso_nlgraph_s43 search_sgms.py -n sgms_ipso_nlgraph_s43 -e exact_match -d nlgraph -g 0,1,2,3 --init_protocol pso --seed 43
run sgms_ipso_gsm8k_s43   search_sgms.py -n sgms_ipso_gsm8k_s43   -e exact_match -d gsm8k  -g 0,1,2,3 --init_protocol pso --seed 43
echo "[ip] 全部结束 $(date)"
