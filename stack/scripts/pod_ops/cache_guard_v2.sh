#!/bin/bash
# cgroup-v2 OOM guard for pod2/pod3 (cap read from memory/memory.limit_in_bytes). Pre-armed
# BEFORE REF-B launch so there is no crash-loop honeymoon. Drops the episode
# caches' clean pages via `fadvise` (python posix_fadvise) when usage crosses
# LIMIT. No privileges needed.
CAP=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)
LIMIT=$(( CAP - 8 * 1024*1024*1024 ))   # sweep 8 GB below the hard cap
echo "[guard] cap $(( CAP/1024/1024/1024 ))GB, sweep above $(( LIMIT/1024/1024/1024 ))GB"
while true; do
  U=$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null) || break
  if [ "$U" -gt "$LIMIT" ]; then
    python3 - <<'PY'
import glob, os
n=0
for pat in ("/opt/comma_epcache/*/ep_*.pt",
            "/workspace/data/physicalai/_epcache/*/ep_*.pt"):
    for p in glob.glob(pat):
        try:
            fd=os.open(p, os.O_RDONLY)
            os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd); n+=1
        except OSError: pass
print(f"[guard] swept {n} files", flush=True)
PY
  fi
  sleep 20
done
