#!/usr/bin/env bash
# culture 重试（evaluate.py except 分支修复后）
set -euo pipefail
cd /data/lvliping/merge_swarm
if [[ -z "${CT:-}" ]]; then CT=1 setsid nohup "$0" >> culture_retry.log 2>&1 < /dev/null & echo "culture重试已排队"; exit 0; fi
while pgrep -f "swarm/bin/pytho[n] search" > /dev/null || pgrep -f "run_race[r].sh" > /dev/null; do sleep 120; done
PY=/data/lvliping/miniconda3/envs/swarm/bin/python
echo "[ct] $(date '+%m-%d %H:%M') racer_culture_s42 开始"
rm -rf search/racer_culture_s42_218keg
if "$PY" search_racer.py -n racer_culture_s42 -e multitask -d culture -g 0,1,2,3 --seed 42 > /dev/null 2>&1; then
  echo "[ct] 完成: $(grep 'final metrics' search/racer_culture_s42_$(hostname)/log.txt | tail -1 | grep -oE 'starting_best_dev=[0-9.]+ ending_best_dev=[0-9.]+ ending_best_test=[0-9.]+')"
else echo "[ct] 失败"; fi
rm -rf search/racer_culture_s42_218keg/tmp 2>/dev/null || true
