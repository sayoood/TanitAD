"""Wait until the dev box has enough FREE COMMIT (not free RAM) for a job, then exit.

    python wait_commit.py --min-gb 6 --max-wait-s 10800 --out <rec.json>

MEASURED 2026-10-04 00:33 Berlin: the shared dev box sat at 2.0 GB free commit of a 50 GB limit (desktop
apps + three other agents' python jobs) while "free RAM" read 4.3 GB, and a G0 diagnostic died at its
first 4 MiB read with MemoryError. Windows grows the system-managed pagefile on demand, but not fast
enough for a burst. The commit counter (`Win32_OperatingSystem.FreeVirtualMemory`) is the one that
decides whether an allocation succeeds, so it is the one waited on. Nothing is killed, ever.
The record (`--out`) says whether the threshold was met; assert on it, never on the exit code.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time


def free_commit_gb() -> tuple[float, float, float]:
    ps = subprocess.run(["powershell.exe", "-NoProfile", "-Command",
                         "$o=Get-CimInstance Win32_OperatingSystem; "
                         "\"$($o.FreeVirtualMemory) $($o.TotalVirtualMemorySize) $($o.FreePhysicalMemory)\""],
                        capture_output=True, text=True, timeout=120).stdout.split()
    fc, tc, fp = (int(x) / 1024 / 1024 for x in ps[:3])
    return round(fc, 2), round(tc, 2), round(fp, 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-gb", type=float, required=True)
    ap.add_argument("--max-wait-s", type=int, default=10800)
    ap.add_argument("--poll-s", type=int, default=60)
    ap.add_argument("--stable-polls", type=int, default=2, help="consecutive polls at/above the bar")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0, n, ok_run, hist = time.time(), 0, 0, []
    while True:
        n += 1
        try:
            fc, tc, fp = free_commit_gb()
        except Exception as exc:                      # noqa: BLE001 -- recorded, retried
            fc, tc, fp = -1.0, -1.0, -1.0
            hist.append({"t": time.strftime("%FT%T%z"), "error": str(exc)[:200]})
        hist = (hist + [{"t": time.strftime("%FT%T%z"), "free_commit_gb": fc, "commit_limit_gb": tc,
                         "free_phys_gb": fp}])[-30:]
        ok_run = ok_run + 1 if fc >= a.min_gb else 0
        met = ok_run >= a.stable_polls
        if met or time.time() - t0 > a.max_wait_s:
            rec = {"met": met, "min_gb": a.min_gb, "waited_s": round(time.time() - t0, 1), "polls": n,
                   "last": hist[-1], "history_tail": hist}
            json.dump(rec, open(a.out, "w", encoding="utf-8"), indent=1)
            print(json.dumps({k: rec[k] for k in ("met", "min_gb", "waited_s", "last")}), flush=True)
            return 0
        time.sleep(a.poll_s)


if __name__ == "__main__":
    raise SystemExit(main())
