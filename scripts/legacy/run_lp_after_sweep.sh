#!/usr/bin/env bash
# 链式启动器：等 hellaswag 超参 sweep 跑完后，自动启动 LP-Merge 完整验证 run。
# 用法: ./run_lp_after_sweep.sh   （自动 setsid 脱离，断网不影响）
set -euo pipefail

REPO_DIR="/data/lvliping/merge_swarm"
PYTHON="/data/lvliping/miniconda3/envs/swarm/bin/python"
SWEEP_PID="${SWEEP_PID:-2075758}"          # sweep 驱动进程
RUN_NAME="${RUN_NAME:-lp_hellaswag_v1}"
export WANDB_ENTITY="llp031004-yanshan"
cd "$REPO_DIR"

# ---- 自脱离 ----
if [[ -z "${CHAIN_DETACHED:-}" ]]; then
    CHAIN_DETACHED=1 setsid nohup "$0" "$@" >> lp_chain.log 2>&1 < /dev/null &
    echo "链式启动器已挂起: 等 sweep (PID ${SWEEP_PID}) 结束后自动跑 LP-Merge (${RUN_NAME})"
    echo "  链式日志: tail -f ${REPO_DIR}/lp_chain.log"
    exit 0
fi

echo "[chain] $(date) 等待 sweep (PID ${SWEEP_PID}) 结束..."
while kill -0 "$SWEEP_PID" 2>/dev/null; do sleep 60; done
echo "[chain] $(date) sweep 已结束，等待 GPU 释放..."
sleep 30

if pgrep -f "envs/swarm/bin/python search" > /dev/null; then
    echo "[chain] 仍有 search 进程在跑，放弃自动启动（避免抢卡）"
    exit 1
fi

echo "[chain] $(date) 启动 LP-Merge 验证: ${RUN_NAME}"
"$PYTHON" search_lp.py \
    -n "$RUN_NAME" -e multiple_choice -d hellaswag -g 0,1,2,3 \
    --popsize 12 --max_iter 15 --coarse_size 50 --seed 42 \
    > /dev/null 2>&1
echo "[chain] $(date) LP-Merge 结束。日志: search/${RUN_NAME}_$(hostname)/log.txt"
"$PYTHON" sweep_collect.py "search/${RUN_NAME}_$(hostname)/log.txt" 2>/dev/null || true
