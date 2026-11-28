#!/bin/bash
# Ship the warmup / navhard 416 banks (dev box C: -> Thor /home/nvidia/r8ns/banks/) -- priority 2 (N3, N2).
set -u
cd /c/Users/Admin/tanitad-caches/refcv6-navsim-20260923 || exit 2
t0=$(date +%s)
tar -cf - warmup_two_stage navhard_s1 navhard_s2 | ssh tanitad-thor "mkdir -p /home/nvidia/r8ns/banks && tar -xf - -C /home/nvidia/r8ns/banks"
echo "rc_pipe=${PIPESTATUS[*]} wall_s=$(( $(date +%s) - t0 ))"
echo QQSHIPCBANKS_END
