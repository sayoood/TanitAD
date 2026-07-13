#!/usr/bin/env bash
# pod2 ops daemon (supervised + boot-persistent via runs.d/ops-daemon.env):
#  1) MEMORY RELIEF: fadvise-DONTNEED the build's mp4 read-cache. The flagship's
#     epcache guard does NOT cover r0/camera_front_wide, so that clean cache
#     accumulates, pushes the cgroup toward the ~51GB cap, and forces the
#     flagship guard to sweep hard -> GPU stalls. Dropping already-read, clean,
#     re-readable pages is safe (same technique as the trainer guard).
#  2) DISK MONITOR: real 500MB dd quota test (the df-invisible MooseFS quota) +
#     build progress -> /workspace/ops/disk_status.json for external watch.
set +e
STATUS=/workspace/ops/disk_status.json
MP4_DIR=/workspace/data/physicalai_phase0/r0/camera_front_wide
EPCACHE=/workspace/data/physicalai_phase0/_epcache
i=0
while true; do
  python3 -c 'import os,glob,sys
for p in glob.glob(sys.argv[1]+"/*.mp4"):
    try:
        fd=os.open(p,os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd)
    except OSError: pass' "$MP4_DIR" 2>/dev/null
  if [ $(( i % 20 )) -eq 0 ]; then
    t="/workspace/_quota_probe_$$.bin"
    if dd if=/dev/zero of="$t" bs=1M count=500 conv=fsync 2>/dev/null; then dd_ok=true; else dd_ok=false; fi
    rm -f "$t"
    tr_n=$(ls "$EPCACHE"/physicalai-train-*/ep_*.pt 2>/dev/null | wc -l)
    va_n=$(ls "$EPCACHE"/physicalai-val-*/ep_*.pt 2>/dev/null | wc -l)
    mem=$(( $(cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null||echo 0)/1024/1024/1024 ))
    oom=$(grep -o "oom_kill [0-9]*" /sys/fs/cgroup/memory/memory.oom_control 2>/dev/null | awk '{print $2}')
    tight=false; [ "$dd_ok" = "false" ] && tight=true
    printf '{"ts":"%s","epoch":%s,"dd_500mb_ok":%s,"train_ep":%s,"val_ep":%s,"target":"2400tr+600va","cgroup_gib":%s,"oom_kill_total":%s,"tight":%s}\n' \
      "$(date -u +%FT%TZ)" "$(date +%s)" "$dd_ok" "${tr_n:-0}" "${va_n:-0}" "$mem" "${oom:-0}" "$tight" > "$STATUS.tmp"
    mv -f "$STATUS.tmp" "$STATUS"
  fi
  i=$((i+1)); sleep 30
done
