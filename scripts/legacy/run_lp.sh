#!/usr/bin/env bash
# LP-Merge 实验启动脚本（对照 run_search.sh 的 PSO 版）
#
# 用法:
#   ./run_lp.sh                # 按下面配置区的默认参数启动
#   CLEAN=1 ./run_lp.sh        # 先删旧 run 目录再启动
#   DATASET=nlgraph EVAL_TYPE=exact_match ./run_lp.sh   # 换数据集（env 覆盖）
#
# 启动方式 setsid + nohup + disown，断网/退出会话不影响。
# 要改的参数基本都在下面配置区；四种生成器比例/自适应规则在 search_lp.py 里。
set -euo pipefail

# ---------------- 配置区（可直接改，也可用环境变量临时覆盖） ----------------
REPO_DIR="/data/lvliping/merge_swarm"
PYTHON="/data/lvliping/miniconda3/envs/swarm/bin/python"

RUN_NAME="${RUN_NAME:-lp_hellaswag}"           # run 目录 = search/${RUN_NAME}_$(hostname)
DATASET="${DATASET:-hellaswag}"                # nlgraph / hellaswag / gsm8k ...
EVAL_TYPE="${EVAL_TYPE:-multiple_choice}"      # 必须与 DATASET 配套（下方有校验）
GPUS="${GPUS:-0,1,2,3}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
EXPERT_DIR="${EXPERT_DIR:-./initial_experts}"  # 10 专家池

POPSIZE="${POPSIZE:-20}"        # v4: 种群（初始=10 单专家+10 混合）；每轮保 top-一半、生成一半
MAX_ITER="${MAX_ITER:-25}"      # v4: 迭代上限（给重启留空间）
COARSE_SIZE="${COARSE_SIZE:-50}"  # 多保真粗筛样本数；设 0 关闭多保真
SEED="${SEED:-42}"
LB="${LB:--0.5}"                # α 下界（注意 ${VAR:-负数} 的写法）
UB="${UB:-1.5}"                 # α 上界（>1 即允许外插出凸包）
PROJECT_WB="${PROJECT_WB:-keg218-swarm-qwen2.5b}"  # 同一项目方便对比
# --------------------------------------------------------------------------

cd "$REPO_DIR"

# ---- dataset 与 eval_type 配套校验 ----
case "$DATASET" in
    nlgraph|nlgraph_mini|gsm8k)  REQUIRED_ET="exact_match" ;;
    hellaswag|mmlu|mmlu_pro|truthfulqa|medqa|medmcqa|scifact|stem|hearsay|citation_prediction_classification|normad_country|normad_value)
        REQUIRED_ET="multiple_choice" ;;
    *)  echo "未知 dataset=${DATASET}"; exit 1 ;;
esac
if [[ "$EVAL_TYPE" != "$REQUIRED_ET" ]]; then
    echo "配置错误: ${DATASET} 必须配 EVAL_TYPE=${REQUIRED_ET}（当前=${EVAL_TYPE}）"; exit 1
fi

export WANDB_ENTITY="llp031004-yanshan"
RUN_DIR="search/${RUN_NAME}_$(hostname)"
LOG_OUT="/dev/null"

# ---- 防重复启动 ----
RUNNING_PIDS=""
for p in $(pgrep -f "search_lp.py -n" || true); do
    exe="$(readlink -f "/proc/${p}/exe" 2>/dev/null || true)"
    [[ "$exe" == "$(readlink -f "$PYTHON")" ]] && RUNNING_PIDS="${RUNNING_PIDS} ${p}"
done
if [[ -n "$RUNNING_PIDS" ]]; then
    echo "已有 LP-Merge 在运行 (PID:${RUNNING_PIDS})，退出。"; exit 1
fi

# ---- 可选清理 ----
if [[ "${CLEAN:-0}" == "1" ]]; then
    echo "清理旧结果: ${RUN_DIR}"
    rm -rf "${RUN_DIR}"
fi

setsid nohup "$PYTHON" search_lp.py \
    -n "$RUN_NAME" \
    -e "$EVAL_TYPE" \
    -d "$DATASET" \
    -g "$GPUS" \
    -b "$BASE_MODEL" \
    -i "$EXPERT_DIR" \
    --popsize "$POPSIZE" \
    --max_iter "$MAX_ITER" \
    --coarse_size "$COARSE_SIZE" \
    --seed "$SEED" \
    --lb "$LB" \
    --ub "$UB" \
    --project_name_wb "$PROJECT_WB" \
    > "$LOG_OUT" 2>&1 < /dev/null &
PID=$!
disown

echo "LP-Merge 已后台启动并脱离会话 (PID: ${PID})"
echo "  迭代日志:  tail -f ${REPO_DIR}/${RUN_DIR}/log.txt"
echo "  状态文件:  ${REPO_DIR}/${RUN_DIR}/lp_scratchpad.json   (含 g、g_alpha、history、评测计数)"
echo "  停止:      kill ${PID}"
