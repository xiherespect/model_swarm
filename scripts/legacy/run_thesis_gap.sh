#!/usr/bin/env bash
# 毕业论文补强链: ①SUMO多种子 ②AMPHI多地形 ③LoRAHub NGOpt基线
set -euo pipefail
cd /data/lvliping/merge_swarm
if [[ -z "${TG:-}" ]]; then TG=1 setsid nohup "$0" >> thesis_gap.log 2>&1 < /dev/null & echo "补强链已启动"; exit 0; fi
PY=/data/lvliping/miniconda3/envs/swarm/bin/python

run_sumo () {
  local nm=$1 et=$2 ds=$3 sd=$4
  if [[ -f "search/${nm}_$(hostname)/log.txt" ]] && grep -q "final metrics" "search/${nm}_$(hostname)/log.txt" 2>/dev/null; then echo "[tg] $nm 已完成"; return 0; fi
  echo "[tg] $(date '+%m-%d %H:%M') $nm 开始"
  rm -rf "search/${nm}_218keg"
  if "$PY" search_sumo.py -n $nm -e $et -d $ds -g 0,1,2,3 --seed $sd --no_pair_seed --no_wandb > /tmp/tg_last.out 2> /tmp/tg_last.err; then
    echo "[tg] $nm 完成: $(grep 'final metrics' search/${nm}_$(hostname)/log.txt | tail -1 | grep -oE 'ending_best_dev=[0-9.]+ ending_best_test=[0-9.]+')"
  else echo "[tg] $nm 失败"; fi
  rm -rf "search/${nm}_218keg/tmp" 2>/dev/null || true
}

run_amphi () {
  local nm=$1 ds=$2
  if [[ -f "search/${nm}_$(hostname)/log.txt" ]] && grep -q "child exit" "search/${nm}_$(hostname)/log.txt" 2>/dev/null; then echo "[tg] $nm 已完成"; return 0; fi
  echo "[tg] $(date '+%m-%d %H:%M') $nm 开始"
  rm -rf "search/${nm}_218keg" "search/${nm}_s_218keg" "search/${nm}_r_218keg"
  if "$PY" search_amphi.py -n $nm -e exact_match -d $ds -g 0,1,2,3 --no_wandb > /tmp/tg_last.out 2> /tmp/tg_last.err; then
    echo "[tg] $nm 完成: $(grep -oE 'KEEP SUMO|SWITCH → RACER' search/${nm}_$(hostname)/log.txt | head -1)"
  else echo "[tg] $nm 失败"; fi
  rm -rf "search/${nm}_s_218keg/tmp" "search/${nm}_r_218keg/tmp" 2>/dev/null || true
}

run_lorahub () {
  local t=$1
  local nm="lorahub_${t}_s42"
  if [[ -f "search/${nm}_$(hostname)/log.txt" ]] && grep -q "final metrics" "search/${nm}_$(hostname)/log.txt" 2>/dev/null; then echo "[tg] $nm 已完成"; return 0; fi
  echo "[tg] $(date '+%m-%d %H:%M') $nm 开始"
  rm -rf "search/${nm}_218keg"
  if "$PY" search_baseline.py -n $nm -m cma -e $(case $t in nlgraph|gsm8k) echo exact_match;; *) echo multiple_choice;; esac) -d $t -g 0,1,2,3 --budget 40 --seed 42 --no_wandb > /tmp/tg_last.out 2> /tmp/tg_last.err; then
    echo "[tg] $nm 完成: $(grep 'final metrics' search/${nm}_$(hostname)/log.txt | tail -1 | grep -oE 'ending_best_dev=[0-9.]+ ending_best_test=[0-9.]+')"
  else echo "[tg] $nm 失败"; fi
  rm -rf "search/${nm}_218keg/tmp" 2>/dev/null || true
}

# ---- ① SUMO 多种子 (s43) ----
run_sumo sumo43_gsm8k_s43 exact_match gsm8k 43
run_sumo sumo43_nlgraph_s43 exact_match nlgraph 43
run_sumo sumo43_hsw_s43 multiple_choice hellaswag 43

# ---- ② AMPHI 多地形 ----
run_amphi amphi_hsw hellaswag
run_amphi amphi_science science
run_amphi amphi_culture culture

# ---- ③ LoRAHub NGOpt(40) ----
for t in nlgraph gsm8k hellaswag mmlu mmlu_pro truthfulqa kcross; do run_lorahub $t; done

echo "[tg] 全部结束 $(date)"
