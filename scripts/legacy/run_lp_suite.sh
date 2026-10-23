#!/usr/bin/env bash
# LP-Merge v4 三数据集连跑套件：hellaswag → nlgraph → gsm8k（顺序，独占 GPU）
# 用法: ./run_lp_suite.sh    （自动 setsid 脱离）
set -euo pipefail

REPO_DIR="/data/lvliping/merge_swarm"
PYTHON="/data/lvliping/miniconda3/envs/swarm/bin/python"
export WANDB_ENTITY="llp031004-yanshan"
cd "$REPO_DIR"

if [[ -z "${SUITE_DETACHED:-}" ]]; then
    SUITE_DETACHED=1 setsid nohup "$0" "$@" >> lp_suite.log 2>&1 < /dev/null &
    echo "suite 已后台启动: tail -f ${REPO_DIR}/lp_suite.log"
    exit 0
fi

# 防撞卡
if pgrep -f "envs/swarm/bin/python search" > /dev/null; then
    echo "[suite] 已有 search 进程在跑，退出"; exit 1
fi

run_lp () {  # $1=name $2=dataset $3=eval_type
    echo "[suite] $(date '+%m-%d %H:%M') LP $1 ($2) 开始"
    if "$PYTHON" search_lp.py -n "$1" -e "$3" -d "$2" -g 0,1,2,3 \
        --popsize 20 --max_iter 25 --coarse_size 50 --seed 42 > /dev/null 2>&1; then
        echo "[suite] $(date '+%H:%M') $1 完成: $(grep 'final metrics' search/$1_$(hostname)/log.txt | tail -1 | sed 's/.*final metrics/final metrics/')"
    else
        echo "[suite] $(date '+%H:%M') $1 失败 (FATAL 见 search/$1_$(hostname)/log.txt)"
    fi
}

run_lp lp_v4_hellaswag hellaswag multiple_choice
run_lp lp_v4_nlgraph  nlgraph  exact_match
run_lp lp_v4_gsm8k    gsm8k    exact_match

echo "[suite] $(date '+%m-%d %H:%M') 全部结束。汇总："
"$PYTHON" sweep_collect.py "search/lp_v4_*/log.txt" 2>/dev/null || true
