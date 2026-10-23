#!/usr/bin/env bash
cd /data/lvliping/merge_swarm
if [[ -z "${PC:-}" ]]; then PC=1 setsid nohup "$0" > probe_cmp.out 2>&1 < /dev/null & echo "判据探测已排队"; exit 0; fi
while pgrep -f "search_amphi.p[y]" > /dev/null; do sleep 300; done
/data/lvliping/miniconda3/envs/swarm/bin/python probe_composability.py
