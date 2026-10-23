#!/usr/bin/env bash
# 基线电池: 6任务 × {random, cma, soup, anchors} = 24 run（预算110等效, seed42）
set -euo pipefail
cd /data/lvliping/merge_swarm
if [[ -z "${BL:-}" ]]; then BL=1 setsid nohup "$0" >> baselines.log 2>&1 < /dev/null & echo "基线链已启动"; exit 0; fi
while pgrep -f "swarm/bin/pytho[n] search" > /dev/null; do sleep 120; done
PY=/data/lvliping/miniconda3/envs/swarm/bin/python
run () {
  local nm="$1"; shift
  if [[ -f "search/${nm}_$(hostname)/log.txt" ]] && grep -q "final metrics" "search/${nm}_$(hostname)/log.txt" 2>/dev/null; then
    echo "[bl] $nm 已完成,跳过"
    return 0
  fi
  echo "[bl] $(date '+%m-%d %H:%M') $nm 开始 (剩余: $(df --output=avail -h /data | tail -1 | tr -d ' '))"
  rm -rf "search/${nm}_218keg"
  if "$PY" search_baseline.py "$@" > /tmp/base_last.out 2> /tmp/base_last.err; then
    echo "[bl] $nm 完成: $(grep 'final metrics' search/${nm}_$(hostname)/log.txt | tail -1 | grep -oE 'ending_best_dev=[0-9.]+ ending_best_test=[0-9.]+')"
  else
    echo "[bl] $nm 失败: $(tail -1 /tmp/base_last.err | head -c 120)"
  fi
  rm -rf "search/${nm}_218keg/tmp" 2>/dev/null || true
}
T () {  # task
  local t=$1; local R=()
  case "$t" in
    nlgraph)   R=(-e exact_match -d nlgraph);;
    gsm8k)     R=(-e exact_match -d gsm8k);;
    legal)     R=(-e multitask -d legal);;
    science)   R=(-e multitask -d science);;
    hellaswag) R=(-e multiple_choice -d hellaswag);;
    culture)   R=(-e multitask -d culture);;
    kcross)    R=(-e multiple_choice -d knowledge_crosswords);;
    mmlu)      R=(-e multiple_choice -d mmlu);;
    mmlu_pro)  R=(-e multiple_choice -d mmlu_pro);;
    truthfulqa) R=(-e multiple_choice -d truthfulqa);;
    abstainqa) R=(-e AbstainQA -d mmlu);;
    medical)   R=(-e multitask -d medical);;
  esac
  for m in random cma soup anchors; do
    run base_${m}_${t}_s42 -n base_${m}_${t}_s42 -m $m "${R[@]}" -g 0,1,2,3 --seed 42
  done
}
for t in nlgraph gsm8k legal science hellaswag culture kcross mmlu mmlu_pro truthfulqa abstainqa medical; do T $t; done
echo "[bl] 全部结束 $(date)"
