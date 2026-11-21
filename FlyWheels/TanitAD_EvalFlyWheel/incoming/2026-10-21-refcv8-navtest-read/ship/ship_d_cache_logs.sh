#!/bin/bash
# Ship navtest log pickles (1.1 GB) + the navtest v1.1 metric cache (12 GB) dev box D: -> Thor /home/nvidia/r8ns/navtest/{nsdata,nsexp}.
set -u
NT=/home/nvidia/r8ns/navtest
t0=$(date +%s)
cd /d/Archive/devbox-C/navsim/data/openscene || exit 2
tar -cf - navsim_logs/test | ssh tanitad-thor "mkdir -p $NT/nsdata/openscene && tar -xf - -C $NT/nsdata/openscene"
echo "logs rc_pipe=${PIPESTATUS[*]} wall_s=$(( $(date +%s) - t0 ))"
t1=$(date +%s)
cd /d/Archive/devbox-C/navsim/exp/w3_navtest_v1 || exit 2
tar -cf - metric_cache_navtest | ssh tanitad-thor "mkdir -p $NT/nsexp/w3_navtest_v1 && tar -xf - -C $NT/nsexp/w3_navtest_v1"
echo "cache rc_pipe=${PIPESTATUS[*]} wall_s=$(( $(date +%s) - t1 ))"
echo QQSHIPD2_END
