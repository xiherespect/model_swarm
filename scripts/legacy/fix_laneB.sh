#!/bin/bash
pkill -f "run_sumo_full.s[h]"
sleep 2
pgrep -af "search_sum[o]" | wc -l
