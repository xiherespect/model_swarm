#!/bin/bash
pkill -f "run_racerf.s[h]"
sleep 2
pgrep -f "run_racerf.s[h]" | wc -l
