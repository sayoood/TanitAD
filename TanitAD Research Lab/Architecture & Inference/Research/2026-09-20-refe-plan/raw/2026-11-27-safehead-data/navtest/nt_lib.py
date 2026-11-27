"""NAVTEST dump helpers: memory gate (own log, never the navtrain chain's CHAIN.log)."""
import ctypes, os, sys, time
W = "F:/safehead_cache"
LOG = W + "/navtest/NAVTEST_CHAIN.log"

def say(m):
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {m}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def memstat():
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32), ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64), ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64)]
    s = MS(); s.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.ullAvailPhys // (1 << 20), s.ullAvailPageFile // (1 << 30)

def memgate(tag, need_commit=None, samples=None, every=None, give_up_s=36 * 3600):
    # MM 2026-11-27 15:15Z: floors are env-configurable (SH_NEED_COMMIT_GB, SH_GATE_SAMPLES, SH_GATE_EVERY_S); defaults unchanged (8, 5, 20).
    # 5 x 20 s per log cost >= 100 s x 136 logs, and an 8 GB commit floor stalled beside the regression; the dump already holds its model.
    need_commit = float(os.environ.get("SH_NEED_COMMIT_GB", "8")) if need_commit is None else need_commit
    samples = int(os.environ.get("SH_GATE_SAMPLES", "5")) if samples is None else samples
    every = int(os.environ.get("SH_GATE_EVERY_S", "20")) if every is None else every
    need_mb = int(os.environ.get("SH_NEED_MB", "2000"))
    ok, t0, n = 0, time.time(), 0
    while time.time() - t0 < give_up_s:
        a, c = memstat()
        ok = ok + 1 if (a >= need_mb and c >= need_commit) else 0
        if n % 15 == 0:
            say(f"memgate {tag}: available {a} MB (need {need_mb}), commit free {c} GB (need {need_commit}) ok {ok}/{samples}")
        n += 1
        if ok >= samples:
            return True
        time.sleep(every)
    say(f"memgate {tag}: gave up")
    sys.exit(8)
