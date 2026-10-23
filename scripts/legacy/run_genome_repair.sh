#!/usr/bin/env bash
# GENOME 修复重跑：等主链 (PID 由 $1) 全部结束后补跑两个 GENOME run
set -euo pipefail
cd /data/lvliping/merge_swarm
if [[ -z "${GR_CHAIN:-}" ]]; then
    GR_CHAIN=1 setsid nohup "$0" "$1" >> genome_repair.log 2>&1 < /dev/null &
    echo "GENOME 修复链已挂起 (等主链 PID $1)"; exit 0
fi
while kill -0 "$1" 2>/dev/null; do sleep 60; done
sleep 30
pgrep -f "envs/swarm/bin/python search" > /dev/null && { echo "[gr] 忙，放弃"; exit 1; }
export WANDB_ENTITY=llp031004-yanshan
PY=/data/lvliping/miniconda3/envs/swarm/bin/python
for cfg in "genome_hellaswag hellaswag multiple_choice" "genome_nlgraph nlgraph exact_match"; do
    set -- $cfg
    echo "[gr] $(date '+%H:%M') $1 开始"
    $PY search_genome.py -n "$1" -e "$3" -d "$2" -g 0,1,2,3 --seed 42 > /dev/null 2>&1 \
      && echo "[gr] $1 完成: $(grep 'final metrics' search/$1_$(hostname)/log.txt | tail -1)" \
      || echo "[gr] $1 失败"
done
echo "[gr] 全部结束 $(date)"
