#!/usr/bin/env bash
# 快速单种子验证协议: 逐刀单测(s42), 大效应才采纳
set -euo pipefail
cd /data/lvliping/merge_swarm
if [[ -z "${FS:-}" ]]; then FS=1 setsid nohup "$0" >> fast.log 2>&1 < /dev/null & echo "快速验证链已启动"; exit 0; fi
while pgrep -f "swarm/bin/pytho[n] search" > /dev/null; do sleep 120; done
PY=/data/lvliping/miniconda3/envs/swarm/bin/python
run () {
  local nm="$1"; shift
  if [[ -f "search/${nm}_$(hostname)/log.txt" ]] && grep -q "final metrics" "search/${nm}_$(hostname)/log.txt" 2>/dev/null; then echo "[fx] $nm 已完成,跳过"; return 0; fi
  echo "[fx] $(date '+%m-%d %H:%M') $nm 开始"
  rm -rf "search/${nm}_218keg"
  if "$PY" search_racer.py "$@" > /tmp/fast_last.out 2> /tmp/fast_last.err; then
    echo "[fx] $nm 完成: $(grep 'final metrics' search/${nm}_$(hostname)/log.txt | tail -1 | grep -oE 'ending_best_dev=[0-9.]+ ending_best_test=[0-9.]+') | highway触发$(grep -c 'HIGHWAY MODE ON' search/${nm}_$(hostname)/log.txt)次"
  else
    echo "[fx] $nm 失败: $(tail -1 /tmp/fast_last.err | head -c 120)"
  fi
  rm -rf "search/${nm}_218keg/tmp" 2>/dev/null || true
}
# 刀D单独: nlgraph s42 (v1.0=0.620/0.592)
run fastD_nlg_s42 -n fastD_nlg_s42 -e exact_match -d nlgraph -g 0,1,2,3 --seed 42 --stretch
# 刀A-v2单独: hsw s42 (v1.0=0.690/0.615, PSO 0.639)
run fastA_hsw_s42 -n fastA_hsw_s42 -e multiple_choice -d hellaswag -g 0,1,2,3 --seed 42 --highway
# 刀A-v2误触发检查: nlgraph s42 (v1.0=0.620/0.592; 期望 highway 0 次)
run fastA_nlg_s42 -n fastA_nlg_s42 -e exact_match -d nlgraph -g 0,1,2,3 --seed 42 --highway
echo "[fx] 全部结束 $(date)"
