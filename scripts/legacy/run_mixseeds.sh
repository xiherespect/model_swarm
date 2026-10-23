#!/usr/bin/env bash
# mix-init nlgraph 种子补强链（s44/s45）——主表锚定协议
set -euo pipefail
cd /data/lvliping/merge_swarm
if [[ -z "${MX:-}" ]]; then MX=1 setsid nohup "$0" >> mixseeds.log 2>&1 < /dev/null & echo "mix种子链已启动"; exit 0; fi
while pgrep -f "swarm/bin/pytho[n] search" > /dev/null; do sleep 120; done
PY=/data/lvliping/miniconda3/envs/swarm/bin/python
run () { local nm="$1"; shift
  echo "[mx] $(date '+%m-%d %H:%M') $nm 开始"
  rm -rf "search/${nm}_218keg"
  if "$PY" "$@" > /dev/null 2>&1; then
    echo "[mx] $nm 完成: $(grep 'final metrics' search/${nm}_$(hostname)/log.txt | tail -1 | grep -oE 'ending_best_dev=[0-9.]+ ending_best_test=[0-9.]+')"
  else echo "[mx] $nm 失败"; fi }
run sgms_mix_nlgraph_s44 search_sgms.py -n sgms_mix_nlgraph_s44 -e exact_match -d nlgraph -g 0,1,2,3 --init_protocol mix --seed 44
run sgms_mix_nlgraph_s45 search_sgms.py -n sgms_mix_nlgraph_s45 -e exact_match -d nlgraph -g 0,1,2,3 --init_protocol mix --seed 45
echo "[mx] 全部结束 $(date)"
