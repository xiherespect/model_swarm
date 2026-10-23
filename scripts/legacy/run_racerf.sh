#!/usr/bin/env bash
# RACER 合成终版(A+C+X2) 全量确认: 12任务×s42 全跑完 → 12任务×s43
set -euo pipefail
cd /data/lvliping/merge_swarm
if [[ -z "${RF:-}" ]]; then RF=1 setsid nohup "$0" >> racerf.log 2>&1 < /dev/null & echo "合成终版全量链已启动(待命)"; exit 0; fi
while pgrep -f "swarm/bin/pytho[n] search" > /dev/null || pgrep -f "run_pso_ful[l].sh" > /dev/null; do sleep 180; done
PY=/data/lvliping/miniconda3/envs/swarm/bin/python
run () {
  local nm="$1"; shift
  echo "[rf] $(date '+%m-%d %H:%M') $nm 开始 (剩余: $(df --output=avail -h /data | tail -1 | tr -d ' '))"
  rm -rf "search/${nm}_218keg"
  if "$PY" search_racer.py "$@" > /tmp/racer_last.out 2> /tmp/racer_last.err; then
    echo "[rf] $nm 完成: $(grep 'final metrics' search/${nm}_$(hostname)/log.txt | tail -1 | grep -oE 'ending_best_dev=[0-9.]+ ending_best_test=[0-9.]+')"
  else
    echo "[rf] $nm 失败"
  fi
  rm -rf "search/${nm}_218keg/tmp" 2>/dev/null || true
}
T () {
  local t=$1 s=$2; local R=()
  case "$t" in
    nlgraph)    R=(-e exact_match -d nlgraph);;
    gsm8k)      R=(-e exact_match -d gsm8k);;
    hellaswag)  R=(-e multiple_choice -d hellaswag);;
    legal)      R=(-e multitask -d legal);;
    medical)    R=(-e multitask -d medical);;
    science)    R=(-e multitask -d science);;
    culture)    R=(-e multitask -d culture);;
    kcross)     R=(-e multiple_choice -d knowledge_crosswords);;
    mmlu)       R=(-e multiple_choice -d mmlu);;
    mmlu_pro)   R=(-e multiple_choice -d mmlu_pro);;
    truthfulqa) R=(-e multiple_choice -d truthfulqa);;
    abstainqa)  R=(-e AbstainQA -d mmlu);;
  esac
  run racerf_${t}_s${s} -n racerf_${t}_s${s} "${R[@]}" -g 0,1,2,3 --seed ${s}
}
for t in nlgraph gsm8k hellaswag legal medical culture kcross mmlu mmlu_pro truthfulqa science abstainqa; do T $t 42; done
for t in nlgraph gsm8k hellaswag legal medical culture kcross mmlu mmlu_pro truthfulqa science abstainqa; do T $t 43; done
echo "[rf] 全部结束 $(date)"
