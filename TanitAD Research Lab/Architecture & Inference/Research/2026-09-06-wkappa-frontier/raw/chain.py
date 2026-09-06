#!/usr/bin/env python3
"""Run batch 2 then batch 3 as soon as the GPU frees, so it never idles.

⚠️ The wait must MATCH THE FAILURE, not only the success: it polls for the
ABSENCE of this package's arm processes AND caps its own wall-clock, so a batch
that dies early advances the chain instead of polling forever.
⛔ It never greps a stream for a token its own command contains; the arm count is
computed here and emitted as an opaque ZZ...ZZ marker.
ASCII only.
"""
import os
import subprocess
import sys
import time

HERE = r"C:\Users\Admin\wkfront"
PY = r"C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
MAX_WAIT_S = 5400          # 1.5 h per batch, then advance regardless

# Batch 2 completes the goal-conditioned curve at seed 1 and gives both of its
# endpoints a second inference seed.
BATCH2 = ["t5_s1:5.03748:1", "t10_s1:10.07497:1", "t0_s1:0.0:1", "t15_s1:SCALAR:1"]
# ⭐ Batch 3 was re-planned after the mechanism read: `wk3` is the arm that turns
# MOST (recall 0.3636 = 4 of 11) while still reading kamm_over 0.0000
# non-vacuously, so it BRACKETS the frontier on the recall side and its
# replication is worth more than a third seed for the endpoints.
# ⭐⭐ Batch 3 was re-planned a second time, after the paired curvature read
# measured that the BINDING interval for the curvature claim is the EPISODE one
# (RESULT.md §4) -- so the cheapest experiment that can move it is INDEPENDENT
# EPISODES, not a third seed. `wk7` is run on two EPISODE-DISJOINT 8-episode
# draws (dA, dB), giving three independent episode draws of the wk7-vs-ha0
# comparison; `wk3` keeps its replicate because it brackets the frontier on the
# recall side. The by-goal rungs therefore stop at two seeds, and that is stated.
BATCH3 = ["wk7_dA:S7.0:0:dA", "wk7_dB:S7.0:0:dB", "wk3_s1:S3.0:1", "wk3_s2:S3.0:2"]


def n_arms():
    """Count THIS package's live arm processes. Scoped to the wkfront out dir so
    a sibling's refav1 work is never counted (and never killed)."""
    # ⛔ NOT `wmic`: it is removed on Windows 11 26200 and returns nothing, so the
    # probe read -1 forever and the wait could only ever advance at its cap.
    ps = ("Get-CimInstance Win32_Process -Filter \"Name LIKE '%python%'\" | "
          "ForEach-Object { $_.CommandLine }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=90).stdout
    except Exception:  # noqa: BLE001
        return -1
    if not out.strip():
        return -1
    n = 0
    for line in out.splitlines():
        if "refav1_arm.py" in line and "wkfront" in line and "mktree" not in line:
            n += 1
    return n


def wait_idle(tag):
    t0 = time.time()
    while time.time() - t0 < MAX_WAIT_S:
        n = n_arms()
        print("ZZWAIT-%s-%d-%dZZ" % (tag, n, int(time.time() - t0)), flush=True)
        if n == 0:
            return True
        time.sleep(60)
    print("ZZWAITCAP-%s-ZZ" % tag, flush=True)
    return False


def run_batch(name, specs):
    log = os.path.join(HERE, "out", "%s.driver.log" % name)
    with open(log, "w", encoding="utf-8") as fh:
        p = subprocess.run([PY, os.path.join(HERE, "launch_rungs.py")] + specs,
                           cwd=HERE, stdout=fh, stderr=subprocess.STDOUT)
    print("ZZBATCH-%s-rc%d-ZZ" % (name, p.returncode), flush=True)


if __name__ == "__main__":
    wait_idle("b1")
    run_batch("batch2", BATCH2)
    wait_idle("b2")
    run_batch("batch3", BATCH3)
    print("ZZCHAINDONE-ZZ", flush=True)
    sys.exit(0)
