#!/bin/bash
# Ship the navtest 416 frame bank (dev box D: -> Thor /home/nvidia/r8ns/navtest/frame_bank) over the LAN alias tanitad-thor.
# The 4 shards already on Thor (s01 s02 s28 s31) are NOT re-sent; they are md5-verified with the rest afterwards.
# MSYS tar reads "C:/x" as a remote host -> relative paths after cd.
set -u
SRC=/d/Archive/devbox-C/navsim/exp/refcv6_navtest416/frame_bank
DST=/home/nvidia/r8ns/navtest/frame_bank
cd $SRC || exit 2
FILES=$(ls frames_s*.npy | grep -v -E 'frames_s(01|02|28|31)\.npy'; echo index.json; ls shard_*.DONE.json; echo rigs)
t0=$(date +%s)
tar -cf - $FILES | ssh tanitad-thor "mkdir -p $DST && tar -xf - -C $DST"
echo "rc_pipe=${PIPESTATUS[*]} wall_s=$(( $(date +%s) - t0 ))"
echo QQSHIPBANK_END
