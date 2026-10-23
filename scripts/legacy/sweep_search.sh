#!/usr/bin/env bash
# 超参随机搜索（对标 Model Swarms 论文 Appendix C：grid 内随机采样，按 dev 挑最优）
#
# 用法:
#   ./sweep_search.sh                                    # 默认: nlgraph, 20 个配置
#   NUM_RUNS=30 ./sweep_search.sh                        # 30 个配置
#   DATASET=hellaswag EVAL_TYPE=multiple_choice ./sweep_search.sh
#   SWEEP_TAG=sweep_nlgraph_0905 ./sweep_search.sh       # 手动指定 tag（跨天续跑时用）
#
# 特性:
#   - 首次运行生成并固化随机配置表（*_configs.tsv），续跑不换配置
#   - 断点续跑：已完成的 run 自动跳过，半途中断的 run 自动清理重跑
#   - 自脱离（setsid）：断网/退出会话不影响 sweep
#   - 每个 run 用 --starting_test_set_eval 0 省约 17 分钟（结尾仍会测冠军的 test）
#   - 跑完自动调用 sweep_collect.py 汇总出 CSV
#
# 注意 -d 与 -e 必须配套: nlgraph/gsm8k→exact_match, hellaswag/mmlu 等→multiple_choice
set -euo pipefail

REPO_DIR="/data/lvliping/merge_swarm"
PYTHON="/data/lvliping/miniconda3/envs/swarm/bin/python"
DATASET="${DATASET:-nlgraph}"
EVAL_TYPE="${EVAL_TYPE:-exact_match}"
NUM_RUNS="${NUM_RUNS:-20}"
MAX_ITER="${MAX_ITER:-30}"          # 论文 K=50；30 是算力折中，patience=10 会早停大部分 run
GPUS="${GPUS:-0,1,2,3}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PROJECT="${PROJECT:-keg218-swarm-qwen2.5b}"
export WANDB_ENTITY="llp031004-yanshan"

cd "$REPO_DIR"

# ---- dataset 与 eval_type 配套校验（防 exact_match 跑选择题这类静默错误）----
case "$DATASET" in
    nlgraph|nlgraph_mini|gsm8k)  REQUIRED_ET="exact_match" ;;
    hellaswag|mmlu|mmlu_pro|truthfulqa|medqa|medmcqa|scifact|stem|hearsay|citation_prediction_classification|normad_country|normad_value)
        REQUIRED_ET="multiple_choice" ;;
    *)  echo "未知 dataset=${DATASET}，请先确认 eval_type 配套关系"; exit 1 ;;
esac
if [[ "$EVAL_TYPE" != "$REQUIRED_ET" ]]; then
    echo "配置错误: ${DATASET} 必须配 EVAL_TYPE=${REQUIRED_ET}（当前=${EVAL_TYPE}）"; exit 1
fi

SWEEP_TAG="${SWEEP_TAG:-sweep_${DATASET}_$(date +%m%d)}"
CONFIG_FILE="${SWEEP_TAG}_configs.tsv"
SWEEP_LOG="${SWEEP_TAG}.log"

# ---- 自脱离启动 ----
if [[ -z "${SWEEP_DETACHED:-}" ]]; then
    SWEEP_DETACHED=1 setsid nohup "$0" "$@" >> "$SWEEP_LOG" 2>&1 < /dev/null &
    echo "sweep 已后台启动并脱离会话"
    echo "  驱动日志:  tail -f ${REPO_DIR}/${SWEEP_LOG}"
    echo "  配置表:    ${REPO_DIR}/${CONFIG_FILE}"
    exit 0
fi

echo "=== sweep 开始: $(date) | dataset=${DATASET} eval=${EVAL_TYPE} runs=${NUM_RUNS} m=${MAX_ITER} ==="

# ---- 防止与其他实验撞卡 ----
RUNNING=$(pgrep -f "envs/swarm/bin/python search.py" || true)
if [[ -n "$RUNNING" ]]; then
    echo "已有 search.py 在运行 (PID ${RUNNING})，为避免抢卡退出。先停掉它再启动 sweep。"; exit 1
fi

# ---- 生成并固化随机配置（论文 grid）----
if [[ ! -f "$CONFIG_FILE" ]]; then
    "$PYTHON" - "$CONFIG_FILE" "$NUM_RUNS" <<'PYEOF'
import random, sys
out, n = sys.argv[1], int(sys.argv[2])
grid = {
    "inertia": [0.1, 0.2, 0.3],                  # φv
    "cog":     [0.1, 0.2, 0.3, 0.4, 0.5],        # φp
    "soc":     [0.2, 0.3, 0.4, 0.5, 0.6],        # φg
    "repel":   [0.01, 0.05, 0.1],                # φw
    "step":    [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],   # λ
}
with open(out, "w") as f:
    f.write("name\tinertia\tcog\tsoc\trepel\tstep\n")
    for i in range(1, n + 1):
        # 第 1 个固定为贴近论文的对照配置，其余随机
        if i == 1:
            vals = ["0.3", "0.3", "0.3", "0.1", "0.8"]
        else:
            vals = [str(random.choice(grid[k])) for k in ["inertia", "cog", "soc", "repel", "step"]]
        f.write(f"r{i:02d}\t" + "\t".join(vals) + "\n")
print(f"已生成 {n} 个配置 → {out}")
PYEOF
fi

# ---- 顺序执行 ----
DONE=0; SKIP=0; FAIL=0
while IFS=$'\t' read -r name inertia cog soc repel step; do
    [[ "$name" == "name" ]] && continue
    RUN_NAME="${SWEEP_TAG#sweep_}_${name}"                 # 如 nlgraph_0905_r01
    RUN_DIR="search/${RUN_NAME}_$(hostname)"
    LOG_TXT="${RUN_DIR}/log.txt"

    if [[ -f "$LOG_TXT" ]] && grep -qE "the end of search|final metrics for test" "$LOG_TXT"; then
        echo "[skip] ${RUN_NAME} 已完成"; SKIP=$((SKIP+1)); continue
    fi
    if [[ -d "$RUN_DIR" ]]; then
        echo "[clean] ${RUN_NAME} 存在未完成残留，清理重跑"; rm -rf "$RUN_DIR"
    fi

    echo "[run ] ${RUN_NAME}: inertia=$inertia cog=$cog soc=$soc repel=$repel step=$step  $(date +%H:%M)"
    START=$(date +%s)
    if "$PYTHON" search.py \
        -n "$RUN_NAME" \
        -e "$EVAL_TYPE" -d "$DATASET" -g "$GPUS" -b "$BASE_MODEL" \
        --inertia "$inertia" --cognitive_coeff "$cog" --social_coeff "$soc" \
        --repel_coeff "$repel" --step_length "$step" \
        --populate_initial_experts 1 --initial_experts_num 20 \
        --starting_test_set_eval 0 --fast_merge 1 \
        --starting_velocity_mode random --repel_term 1 \
        --step_length_factor 0.95 --restart_stray_particles 1 \
        --restart_patience 0.5 --weight_randomess 1 \
        --project_name_wb "$PROJECT" \
        -p 10 -m "$MAX_ITER" > /dev/null 2>&1; then
        if grep -q "the end of search" "$LOG_TXT" 2>/dev/null; then
            echo "[done] ${RUN_NAME} 用时 $(( ($(date +%s) - START) / 60 )) 分钟"; DONE=$((DONE+1))
        else
            echo "[fail] ${RUN_NAME} 退出但未见完成标记（看 ${LOG_TXT}）"; FAIL=$((FAIL+1))
        fi
    else
        echo "[fail] ${RUN_NAME} 进程异常退出（看 ${LOG_TXT}）"; FAIL=$((FAIL+1))
    fi
done < "$CONFIG_FILE"

echo "=== sweep 结束: $(date) | 完成 ${DONE} / 跳过 ${SKIP} / 失败 ${FAIL} ==="

# ---- 汇总 ----
"$PYTHON" sweep_collect.py "search/${SWEEP_TAG#sweep_}_r*/log.txt" || true
echo "结果已写入 ${REPO_DIR}/sweep_results.csv（按 dev 降序；挑最优请同时看 test 列，防赢者诅咒）"
