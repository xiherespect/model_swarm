#!/usr/bin/env bash
set -euo pipefail
cd /data/lvliping/merge_swarm
if [[ -z "${V1P_CHAIN:-}" ]]; then
    V1P_CHAIN=1 setsid nohup "$0" >> v1p_chain.log 2>&1 < /dev/null &
    echo "v1p 链已挂起 (等 meal2 PID 2382976)"; exit 0
fi
while kill -0 2382976 2>/dev/null; do sleep 30; done
sleep 30
pgrep -f "envs/swarm/bin/python search" > /dev/null && { echo "[v1p] 忙，放弃"; exit 1; }
export WANDB_ENTITY=llp031004-yanshan
echo "[v1p] $(date '+%H:%M') sgms_v1p_hellaswag 开始"
/data/lvliping/miniconda3/envs/swarm/bin/python search_sgms.py -n sgms_v1p_hellaswag -e multiple_choice -d hellaswag -g 0,1,2,3 --init_pair_scan --seed 42 > /dev/null 2>&1   && echo "[v1p] 完成: $(grep 'final metrics' search/sgms_v1p_hellaswag_$(hostname)/log.txt)" || echo "[v1p] 失败"
