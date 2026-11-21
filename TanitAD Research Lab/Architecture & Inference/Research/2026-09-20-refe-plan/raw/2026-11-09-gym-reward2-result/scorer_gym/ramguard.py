"""RAM guard for the REWARD-2 chain (brief, allowed by PREREG_REWARD_2 section 5): a new state (or test group) is STARTED only while the available
physical RAM is at least MIN_AVAIL_GIB (another session's job pauses itself below 6 GiB). Every wait is logged. At most 4 workers.
`avail_gib()` uses GlobalMemoryStatusEx (ctypes, no dependency, works under python -I)."""
import ctypes, multiprocessing as mp, os, time

MIN_AVAIL_GIB = 7.0
MAX_WORKERS = 4


def avail_gib():
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2 ** 30


def min_gib_now():
    """7 GiB, always, except in a DRY chain (chain_v6 --dry sets MGYM6_DRY_CHAIN=1) where MGYM6_MIN_RAM_GIB may lower it for a tester's check; the real launch can never lower it."""
    if os.environ.get("MGYM6_DRY_CHAIN") == "1" and os.environ.get("MGYM6_MIN_RAM_GIB"):
        return float(os.environ["MGYM6_MIN_RAM_GIB"])
    return MIN_AVAIL_GIB


def may_start(avail, min_gib=MIN_AVAIL_GIB):
    """the guard rule, pure so a test can mutate it"""
    return avail >= min_gib


def wait_for_ram(label, log, min_gib=None, avail_fn=None, poll_s=20.0, sleep=time.sleep, rule=may_start):
    """blocks until the rule admits a start; logs one line per wait poll. Returns the available GiB that admitted the start."""
    avail_fn = avail_fn or avail_gib
    min_gib = min_gib_now() if min_gib is None else min_gib
    while True:
        a = avail_fn()
        if rule(a, min_gib):
            return a
        log(f"RAM_WAIT avail={a:.2f} GiB < {min_gib:.1f} GiB before starting {label}")
        sleep(poll_s)


def run_tasks(fn, tasks, workers, on_result, label_fn=str, log=print, min_gib=None, avail_fn=None, poll_s=20.0, settle_s=30.0,
              sleep=time.sleep, rule=may_start):
    """spawn pool of `workers` (<= 4). A task is submitted only when a worker slot is free AND the RAM guard admits it; after a start the next start waits
    `settle_s` so the new worker's memory shows up in the next reading. Results are handed to on_result(result, task) in completion order."""
    assert 1 <= workers <= MAX_WORKERS, f"at most {MAX_WORKERS} workers"
    pending, inflight = list(tasks), []
    with mp.get_context("spawn").Pool(workers) as pool:
        while pending or inflight:
            still = []
            for ar, t in inflight:
                if ar.ready():
                    on_result(ar.get(), t)
                else:
                    still.append((ar, t))
            inflight = still
            if pending and len(inflight) < workers:
                wait_for_ram(label_fn(pending[0]), log, min_gib, avail_fn, poll_s, sleep, rule)
                t = pending.pop(0)
                inflight.append((pool.apply_async(fn, (t,)), t))
                sleep(settle_s if pending else 0.0)
                continue
            sleep(1.0)


def _echo(x):          # module-level so the spawn pool can import it (used by the guard test)
    return x
