#!/usr/bin/env bash
# AMPHI v3 实战: nlgraph(应换道) + gsm8k(应保持) —— 试探式路由终验
set -euo pipefail
cd /data/lvliping/merge_swarm
if [[ -z "${AP:-}" ]]; then AP=1 setsid nohup "$0" >> amphi.log 2>&1 < /dev/null & echo "AMPHI链已排队"; exit 0; fi
while pgrep -f "search_sum[o].py" > /dev/null || pgrep -f "search_race[r].py" > /dev/null; do sleep 180; done
PY=/data/lvliping/miniconda3/envs/swarm/bin/python
for t in nlgraph gsm8k; do
  echo "[ap] $(date '+%m-%d %H:%M') amphi_$t 开始"
  rm -rf search/amphi_${t}_218keg search/amphi_${t}_s_218keg search/amphi_${t}_r_218keg
  if "$PY" search_amphi.py -n amphi_$t -e exact_match -d $t -g 0,1,2,3 --no_wandb > /tmp/ap_$t.out 2>&1; then
    echo "[ap] amphi_$t 完成: $(grep 'final metrics' search/amphi_${t}_{s,r}_218keg/log.txt 2>/dev/null | tail -1 | grep -oE 'starting_best_dev=[0-9.]+ ending_best_dev=[0-9.]+ ending_best_test=[0-9.]+') | 路由: $(grep -oE 'KEEP SUMO|SWITCH → RACER' search/amphi_${t}_218keg/log.txt | head -1)"
  else
    echo "[ap] amphi_$t 失败"
  fi
  rm -rf search/amphi_${t}_s_218keg/tmp search/amphi_${t}_r_218keg/tmp 2>/dev/null || true
done
echo "[ap] 全部结束 $(date)"
