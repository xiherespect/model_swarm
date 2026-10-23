#!/usr/bin/env bash
# AMPHI 多地形修复版（eval_type 各任务正确）
set -euo pipefail
cd /data/lvliping/merge_swarm
if [[ -z "${AF:-}" ]]; then AF=1 setsid nohup "$0" >> amphi_fix.log 2>&1 < /dev/null & echo "AMPHI修复链已启动"; exit 0; fi
while pgrep -f "swarm/bin/pytho[n] search" > /dev/null; do sleep 180; done
PY=/data/lvliping/miniconda3/envs/swarm/bin/python
run () {
  local nm=$1 et=$2 ds=$3
  echo "[af] $(date '+%m-%d %H:%M') $nm 开始"
  rm -rf "search/${nm}_218keg" "search/${nm}_s_218keg" "search/${nm}_r_218keg"
  if "$PY" search_amphi.py -n $nm -e $et -d $ds -g 0,1,2,3 --no_wandb > /tmp/af_last.out 2> /tmp/af_last.err; then
    echo "[af] $nm 完成 | 路由: $(grep -oE 'KEEP SUMO|SWITCH → RACER' search/${nm}_$(hostname)/log.txt | head -1) | $(grep 'final metrics' search/${nm}_{s,r}_$(hostname)/log.txt 2>/dev/null | tail -1 | grep -oE 'ending_best_dev=[0-9.]+ ending_best_test=[0-9.]+')"
  else echo "[af] $nm 失败"; fi
  rm -rf "search/${nm}_s_218keg/tmp" "search/${nm}_r_218keg/tmp" 2>/dev/null || true
}
run amphi_hsw_f multiple_choice hellaswag
run amphi_science_f multitask science
run amphi_culture_f multitask culture
echo "[af] 全部结束 $(date)"
