#!/usr/bin/env bash
cd /data/lvliping/merge_swarm
if [[ -z "${PS:-}" ]]; then PS=1 setsid nohup "$0" > probe_singles.out 2>&1 & echo "探测器已排队"; exit 0; fi
while pgrep -f "swarm/bin/pytho[n] search" > /dev/null; do sleep 120; done
PY=/data/lvliping/miniconda3/envs/swarm/bin/python
$PY probe_singles.py gsm8k
$PY probe_singles.py nlgraph
echo PROBE_DONE
