#!/usr/bin/env bash
# 消融链：SGMS v1 四部件消融（nlgraph, seed 42, 对照基线 sgms_v1_nlgraph 0.550/103evals）
# 用法: ./run_ablation.sh  （等待 followup 链结束后自动开始）
set -euo pipefail
cd /data/lvliping/merge_swarm
if [[ -z "${ABL:-}" ]]; then
    ABL=1 setsid nohup "$0" >> ablation.log 2>&1 < /dev/null & echo "消融链已启动(待命)"; exit 0
fi
while pgrep -f "swarm/bin/pytho[n] search" > /dev/null || pgrep -f "run_followu[p].sh" > /dev/null; do sleep 180; done
df -h /data | tail -1
PY=/data/lvliping/miniconda3/envs/swarm/bin/python
run () { local nm="$1"; shift
  echo "[abl] $(date '+%m-%d %H:%M') $nm 开始 (剩余: $(df --output=avail -h /data | tail -1 | tr -d ' '))"
  rm -rf "search/${nm}_218keg"
  if "$PY" "$@" > /dev/null 2>&1; then
    echo "[abl] $nm 完成: $(grep 'final metrics' search/${nm}_$(hostname)/log.txt 2>/dev/null | tail -1 | grep -oE 'starting_best_dev=[0-9.]+ ending_best_dev=[0-9.]+ ending_best_test=[0-9.]+')"
  else echo "[abl] $nm 失败"; fi }
run sgms_abl_simplex  search_sgms.py -n sgms_abl_simplex  -e exact_match -d nlgraph -g 0,1,2,3 --simplex --lb 0 --ub 1 --seed 42
run sgms_abl_nogp     search_sgms.py -n sgms_abl_nogp     -e exact_match -d nlgraph -g 0,1,2,3 --no_gp --seed 42
run sgms_abl_nosyn    search_sgms.py -n sgms_abl_nosyn    -e exact_match -d nlgraph -g 0,1,2,3 --no_syn --seed 42
run sgms_abl_nocoarse search_sgms.py -n sgms_abl_nocoarse -e exact_match -d nlgraph -g 0,1,2,3 --coarse_size 0 --round_evals 3 --seed 42
echo "[abl] 全部结束 $(date)"
"$PY" compare_all.py || true
