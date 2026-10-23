#!/usr/bin/env bash
# 链式：等 GENOME 链 (PID $1) 结束后跑 SGMS v1 的 hellaswag 和 nlgraph
set -euo pipefail
REPO_DIR="/data/lvliping/merge_swarm"
PYTHON="/data/lvliping/miniconda3/envs/swarm/bin/python"
WAIT_PID="${1:?用法: ./run_sgms_chain.sh <等待的PID>}"
export WANDB_ENTITY="llp031004-yanshan"
cd "$REPO_DIR"
if [[ -z "${SG_CHAIN:-}" ]]; then
    SG_CHAIN=1 setsid nohup "$0" "$WAIT_PID" >> sgms_chain.log 2>&1 < /dev/null &
    echo "SGMS 链已挂起: 等 PID ${WAIT_PID} (GENOME链) 结束后跑 hsw→nl"
    exit 0
fi
echo "[sgms-chain] $(date '+%m-%d %H:%M') 等待 GENOME 链 (PID ${WAIT_PID})..."
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
sleep 30
if pgrep -f "envs/swarm/bin/python search" > /dev/null; then
    echo "[sgms-chain] 仍有 search 进程，放弃"; exit 1
fi
run_sgms () {
    echo "[sgms-chain] $(date '+%m-%d %H:%M') SGMS $1 ($2) 开始"
    if "$PYTHON" search_sgms.py -n "$1" -e "$3" -d "$2" -g 0,1,2,3 \
        --init_size 20 --round_evals 4 --max_iter 25 --seed 42 "${@:4}" > /dev/null 2>&1; then
        echo "[sgms-chain] $(date '+%m-%d %H:%M') $1 完成: $(grep 'final metrics' search/$1_$(hostname)/log.txt | tail -1)"
    else
        echo "[sgms-chain] $(date '+%m-%d %H:%M') $1 失败 (FATAL 见 search/$1_$(hostname)/log.txt)"
    fi
}
run_sgms sgms_v1_hellaswag hellaswag multiple_choice
run_sgms sgms_v1_nlgraph  nlgraph  exact_match --lb -1.5 --ub 2.0
echo "[sgms-chain] $(date '+%m-%d %H:%M') 全部结束"
