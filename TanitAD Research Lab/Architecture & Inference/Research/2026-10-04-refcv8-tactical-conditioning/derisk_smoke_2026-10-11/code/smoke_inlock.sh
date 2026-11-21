#!/usr/bin/env bash
# runs INSIDE the GPU lock (mirrors ladder_chain.sh locked(): MemAvailable asserted INSIDE the lock; exit 97 = retry).
# usage: smoke_inlock.sh <MEM_MIN_GB> <cmd...>
S=/home/nvidia/refcv8_smoke
MIN=$1; shift
m=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
if [ -z "$m" ] || [ "$m" -lt "$MIN" ]; then exit 97; fi
echo "[$(date -u +%FT%TZ)] ZZSMOKE-LOCKEDZZ MemAvailable=${m}GB (lock held, trainer exec next)" >> "$S/smoke.log"
exec "$@"
