"""Block until BOTH of my arms have produced a complete record, or until they are
provably dead. Exits on FAILURE as well as success -- an arm gone with its output
missing is REPORTED, never waited on (`wait-loops-must-match-failure`).

⛔ Gates on the ARTIFACT (record present AND 8/8 episode dumps), not on a success
marker in a log, and treats a FAILED process probe as "unknown", never as "gone".
ASCII only.
"""
import glob
import os
import re
import subprocess
import sys
import time

P4 = "C:/Users/Admin/refav1_margin/p4out"
MINE = ["combined_seed1", "bestlad"]
PS = ["powershell.exe", "-NoProfile", "-Command",
      "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
      "ForEach-Object { $_.CommandLine }"]


def complete(tag):
    return (os.path.exists("%s/rec_%s.json" % (P4, tag)) and
            len(glob.glob("%s/dump_%s/ep*.npz" % (P4, tag))) == 8)


def live():
    """(set of live arm tags, set of live queue markers) or (None, None)."""
    try:
        r = subprocess.run(PS, capture_output=True, text=True, timeout=180)
    except Exception:                                             # noqa: BLE001
        return None, None
    arms, q = set(), 0
    for line in r.stdout.splitlines():
        if "refav1" + "_arm.py" in line:
            m = re.search(r"--arm\s+(\S+)", line)
            if m:
                arms.add(m.group(1).split("p4-")[-1])
        if "best_queue" in line:
            q += 1
    return arms, q


DEADLINE = time.time() + 8 * 3600
i = 0
while time.time() < DEADLINE:
    i += 1
    done = [t for t in MINE if complete(t)]
    if len(done) == len(MINE):
        print("ZZMINE-ALL-COMPLETE-ZZ %s" % time.strftime("%H:%M:%S"), flush=True)
        sys.exit(0)
    arms, q = live()
    if arms is None:
        print("ZZPROBE-FAIL-%d-ZZ not treating as gone" % i, flush=True)
        time.sleep(60)
        continue
    pending = [t for t in MINE if t not in done]
    stalled = [t for t in pending if t not in arms]
    # a pending arm is only DEAD if the queue is also gone (it cannot be relaunched)
    if stalled and q == 0:
        print("ZZMINE-STALLED-ZZ done=%s pending=%s live_arms=%s queue=%d %s"
              % (done, pending, sorted(arms), q, time.strftime("%H:%M:%S")), flush=True)
        sys.exit(1)
    if i % 20 == 0:
        print("ZZWAITMINE-%d-ZZ done=%s live=%s queue=%d %s"
              % (i, done, sorted(arms), q, time.strftime("%H:%M:%S")), flush=True)
    time.sleep(45)
print("ZZWAITMINE-DEADLINE-ZZ", flush=True)
sys.exit(2)
