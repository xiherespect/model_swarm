#!/usr/bin/env bash
# 链式启动器：等 lp_v4_nlgraph (PID 由参数指定) 结束后，顺序跑 GENOME 的 hellaswag 和 nlgraph。
# 用法: ./run_genome_chain.sh <等待的PID>    （自动 setsid 脱离）
set -euo pipefail

REPO_DIR="/data/lvliping/merge_swarm"
PYTHON="/data/lvliping/miniconda3/envs/swarm/bin/python"
WAIT_PID="${1:?用法: ./run_genome_chain.sh <等待的PID>}"
export WANDB_ENTITY="llp031004-yanshan"
cd "$REPO_DIR"

if [[ -z "${CHAIN_DETACHED:-}" ]]; then
    CHAIN_DETACHED=1 setsid nohup "$0" "$WAIT_PID" >> genome_chain.log 2>&1 < /dev/null &
    echo "GENOME 链已挂起: 等 PID ${WAIT_PID} 结束后跑 hellaswag→nlgraph"
    echo "  日志: tail -f ${REPO_DIR}/genome_chain.log"
    exit 0
fi

echo "[chain] $(date '+%m-%d %H:%M') 等待 PID ${WAIT_PID} 结束..."
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
sleep 30  # 等 GPU 释放 + test 评测子进程退出

if pgrep -f "envs/swarm/bin/python search" > /dev/null; then
    echo "[chain] 仍有 search 进程，放弃启动"; exit 1
fi

run_genome () {  # $1=name $2=dataset $3=eval_type
    echo "[chain] $(date '+%m-%d %H:%M') GENOME $1 ($2) 开始"
    if "$PYTHON" search_genome.py -n "$1" -e "$3" -d "$2" -g 0,1,2,3 \
        --N 10 --iters 10 --cross_rate 0 --im_rate 0.1 --gm_rate 0.1 \
        --sigma 0.001 --elite_percent 0.1 --early_stop_iter 5 --seed 42 \
        > /dev/null 2>&1; then
        echo "[chain] $(date '+%m-%d %H:%M') $1 完成: $(grep 'final metrics' search/$1_$(hostname)/log.txt | tail -1)"
    else
        echo "[chain] $(date '+%m-%d %H:%M') $1 失败 (FATAL 见 search/$1_$(hostname)/log.txt)"
    fi
}

run_genome genome_hellaswag hellaswag multiple_choice
run_genome genome_nlgraph  nlgraph  exact_match
echo "[chain] $(date '+%m-%d %H:%M') 全部结束"
