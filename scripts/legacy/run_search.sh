#!/usr/bin/env bash
# Model Swarms (Qwen2.5-1.5B + Tulu LoRA experts) 正式实验启动脚本

set -euo pipefail

# ---- 配置（与 2026-09-04 正式 run 保持一致）----
REPO_DIR="/data/lvliping/merge_swarm"
PYTHON="/data/lvliping/miniconda3/envs/swarm/bin/python"
RUN_NAME="hellaswag"       #本次实验的名字
RUN_DIR="search/${RUN_NAME}_$(hostname)"   # search.py 会自动加主机名后缀
LOG_OUT="/dev/null"   # stdout/stderr 直接丢弃（崩溃无线索；想留就改成 search/ 下的路径）
export WANDB_ENTITY="llp031004-yanshan"    # 个人团队；默认 entity 是别人的，必须显式指定

cd "$REPO_DIR"

# 防止重复启动：按 /proc/PID/exe 只认 swarm 环境的 python 进程。
# 不用 comm 判断（PyTorch 主线程 comm 是 pt_main_thread），也不用裸 pgrep
# （会误匹配含同样字符串的 bash 包装进程和本脚本自身）。
PYEXE="$(readlink -f "$PYTHON")"
RUNNING_PIDS=""
for p in $(pgrep -f "search.py -n ${RUN_NAME}" || true); do
    exe="$(readlink -f "/proc/${p}/exe" 2>/dev/null || true)"
    [[ "$exe" == "$PYEXE" ]] && RUNNING_PIDS="${RUNNING_PIDS} ${p}"
done
if [[ -n "$RUNNING_PIDS" ]]; then
    echo "已有同名实验在运行 (PID:${RUNNING_PIDS})，退出。"
    exit 1
fi

# 可选清理（默认关闭）。LOG_OUT=/dev/null 时绝不能 rm，否则会删掉系统设备节点
if [[ "${CLEAN:-0}" == "1" ]]; then
    echo "清理旧结果: ${RUN_DIR}"
    rm -rf "${RUN_DIR}"
    if [[ -f "$LOG_OUT" ]]; then
        rm -f "$LOG_OUT"
    fi
fi
#   self_weight     = r_w * inertia         # φv  惯性：保留当前速度
#   cognitive_weight = r_p * cognitive_coeff # φp  认知：拉向个人最优 p_i
#   social_weight    = r_s * social_coeff    # φg  社会：拉向全局最优 g
#   repel_weight     = r_b * repel_coeff     # φw  排斥：推离全局最差 g_w
setsid nohup "$PYTHON" search.py \
    -n "$RUN_NAME" \
    -e multiple_choice \
    -d hellaswag \
    -g 0,1,2,3 \
    -b Qwen/Qwen2.5-1.5B-Instruct \
    --inertia 0.4 \
    --cognitive_coeff 0.3 \
    --social_coeff 0.3 \
    --repel_coeff 0.3 \
    --step_length 0.8 \
    --populate_initial_experts 1 \
    --initial_experts_num 20 \
    --starting_test_set_eval 1 \
    --fast_merge 1 \
    --starting_velocity_mode random \
    --repel_term 1 \
    --step_length_factor 0.95 \
    --restart_stray_particles 1 \
    --restart_patience 0.67 \
    --weight_randomess 1 \
    --project_name_wb keg218-swarm-qwen2.5b \
    -p 10 -m 50 \
    > "$LOG_OUT" 2>&1 < /dev/null &
PID=$!
disown

echo "search.py 已后台启动并脱离会话 (PID: ${PID})"
[[ "$LOG_OUT" != "/dev/null" ]] && echo "  总日志:    tail -f ${LOG_OUT}"
echo "  迭代日志:  tail -f ${REPO_DIR}/${RUN_DIR}/log.txt"
echo "  停止:      kill ${PID}"
