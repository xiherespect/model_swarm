#!/bin/bash
pkill -f "run_amphi.s[h]"
pkill -f "search_amphi.p[y]"
pkill -f "search_sum[o].py.*amphi"
sleep 3
echo stopped
