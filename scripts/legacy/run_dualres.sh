#!/usr/bin/env bash
set -euo pipefail
cd /data/lvliping/merge_swarm
if [[ -z "${DR:-}" ]]; then DR=1 setsid nohup "$0" >> dualres.log 2>&1 < /dev/null & echo "双分辨率链已启动"; exit 0; fi
PY=/data/lvliping/miniconda3/envs/swarm/bin/python
run () {
  local nm=$1 et=$2 ds=$3
  echo "[dr] $(date '+%m-%d %H:%M') $nm 开始"
  rm -rf "search/${nm}_218keg"
  if "$PY" search_dualres.py -n $nm -e $et -d $ds -g 0,1,2,3 --seed 42 --no_wandb > /tmp/dr_last.out 2> /tmp/dr_last.err; then
    echo "[dr] $nm 完成: $(grep 'final metrics' search/${nm}_$(hostname)/log.txt | tail -1 | grep -oE 'ending_best_dev=[0-9.]+ ending_best_test=[0-9.]+')"
  else echo "[dr] $nm 失败: $(tail -1 /tmp/dr_last.err | head -c 120)"; fi
  rm -rf "search/${nm}_218keg/tmp" 2>/dev/null || true
}
run dual_nlgraph exact_match nlgraph
run dual_gsm8k exact_match gsm8k
run dual_hsw multiple_choice hellaswag
echo "[dr] 全部结束 $(date)"
