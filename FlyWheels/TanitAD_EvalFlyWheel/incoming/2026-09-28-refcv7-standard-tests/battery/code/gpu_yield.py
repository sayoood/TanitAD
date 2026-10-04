"""The evening YIELD rule (Master Mind + Research Lab, 2026-09-28; a scheduling rule, not a SPEC change):
after every arm/stage that finishes after 20:00 Berlin, the milestone chain RELEASES the GPU lock, sleeps
>= 90 s, then RE-ACQUIRES it -- so REFe's snapshot waiter (polling every 60 s) can take the card for its
<= 20 min seam in between. Each gap is logged with who took the lock.

    python gpu_yield.py --job <chain job> --pid <chain pid> --stage <name> --log <jsonl> [--after 20:00] [--gap 90]

Before 20:00 this is a no-op (logged). The lock is re-acquired under the SAME job and pid, so the chain's
EXIT trap still releases it. Never breaks another session's lock; never kills anything.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_lock  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--after", default="20:00")
    ap.add_argument("--gap", type=int, default=90)
    ap.add_argument("--max-wait-s", type=int, default=12 * 3600)
    a = ap.parse_args()
    hh, mm = (int(x) for x in a.after.split(":"))
    now = time.localtime()
    rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "stage": a.stage, "job": a.job}
    if (now.tm_hour, now.tm_min) < (hh, mm):
        rec["action"] = f"none (before {a.after})"
    else:
        rel = gpu_lock.release(a.job, a.pid)
        rec["released"] = rel
        if not rel.get("released"):
            rec["action"] = "NOT RELEASED (the lock was not ours) -- no yield"
        else:
            time.sleep(max(90, a.gap))
            seen = []

            def log(*x, **k):
                h = gpu_lock.read_lock()
                if isinstance(h, dict) and (not seen or seen[-1] != h.get("job")):
                    seen.append(h.get("job"))
            t0 = time.time()
            got = gpu_lock.acquire(a.job, a.pid, a.max_wait_s, poll_s=30, log=log)
            rec.update(action="yielded", gap_s=round(max(90, a.gap) + time.time() - t0, 1),
                       taken_by_in_gap=seen, reacquired=bool(got.get("ok")))
    with open(a.log, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    print(json.dumps(rec, default=str), flush=True)
    sys.exit(0 if rec.get("reacquired", True) else 3)


if __name__ == "__main__":
    main()
