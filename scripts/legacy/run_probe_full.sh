#!/usr/bin/env bash
cd /data/lvliping/merge_swarm
if [[ -z "${PF:-}" ]]; then PF=1 setsid nohup "$0" > probe_full.out 2>&1 < /dev/null & echo "全保真探测已排队"; exit 0; fi
while pgrep -f "probe_composability.p[y]" > /dev/null; do sleep 300; done
/data/lvliping/miniconda3/envs/swarm/bin/python probe_comp_full.py
