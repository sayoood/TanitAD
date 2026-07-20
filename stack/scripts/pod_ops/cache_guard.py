"""OOM guard for pod1 (2026-07-12 incident: memcg 62 GB cap, silent kills at
~26.55k). Userspace page-cache relief: when cgroup usage crosses LIMIT, drop
the episode caches' clean pages via posix_fadvise(DONTNEED) — no privileges
needed, unlike /proc/sys/vm/drop_caches (RO) or memory.force_empty (denied).
Cost: periodic cold re-reads (data_s spikes); benefit: the trainer survives.
"""
import glob
import os
import time

LIMIT = 60 * 1024**3            # sweep above 54 GB (cap: 62 GB)
USAGE = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
PATTERNS = (
    "/workspace/data/comma2k19/_epcache/*/ep_*.pt",
    "/workspace/data/physicalai/_epcache/*/ep_*.pt",
)


def sweep() -> int:
    n = 0
    for pat in PATTERNS:
        for p in glob.glob(pat):
            try:
                fd = os.open(p, os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                os.close(fd)
                n += 1
            except OSError:
                pass
    return n


while True:
    try:
        u = int(open(USAGE).read())
    except OSError:
        break
    if u > LIMIT:
        n = sweep()
        u2 = int(open(USAGE).read())
        print(f"[guard] {u/1e9:.1f} -> {u2/1e9:.1f} GB ({n} files)",
              flush=True)
    time.sleep(20)
