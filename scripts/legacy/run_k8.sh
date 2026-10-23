#!/usr/bin/env bash
set -euo pipefail
cd /data/lvliping/merge_swarm
if [[ -z "${K8:-}" ]]; then K8=1 setsid nohup "$0" >> k8.log 2>&1 < /dev/null & echo "k8链已启动(串行)"; exit 0; fi
PY=/data/lvliping/miniconda3/envs/swarm/bin/python
echo "[k8] hsw 开始 $(date '+%H:%M')"
$PY search_racer.py -n racer_k8_hsw -e multiple_choice -d hellaswag -g 0,1,2,3 --seed 42 --round_evals 8 --max_iter 18 --no_wandb > /dev/null 2>&1
echo "[k8] hsw 完成: $(grep 'final metrics' search/racer_k8_hsw_$(hostname)/log.txt 2>/dev/null | grep -oE 'ending_best_dev=[0-9.]+ ending_best_test=[0-9.]+')"
echo "[k8] culture 开始 $(date '+%H:%M')"
$PY search_racer.py -n racer_k8_culture -e multitask -d culture -g 0,1,2,3 --seed 42 --round_evals 8 --max_iter 18 --no_wandb > /dev/null 2>&1
echo "[k8] culture 完成: $(grep 'final metrics' search/racer_k8_culture_$(hostname)/log.txt 2>/dev/null | grep -oE 'ending_best_dev=[0-9.]+ ending_best_test=[0-9.]+')"
echo "[k8] 全部结束 $(date)"
