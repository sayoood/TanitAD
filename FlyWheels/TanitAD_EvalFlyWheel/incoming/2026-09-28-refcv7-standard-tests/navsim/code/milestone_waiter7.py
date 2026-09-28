#!/usr/bin/env python3
"""EVENT-DRIVEN milestone waiter for the refcv7 NavSim suite (TANITAD VENV). Detached; UNATTENDED.

    python code/milestone_waiter7.py 5000              # arm for step 5,000
    python code/milestone_waiter7.py 5000 --status     # print the waiter's state file

Waits for EVENTS, never for a clock:
  1. THE CHECKPOINT: ``D:/refcv7_eval_kit/ckpt/ckpt_<N>.pt`` pulled by the battery stream's waiter,
     AND a ``MD5SUMS`` line naming it, AND the local md5 == that line, AND the file's own ``step``
     key == N. (Nothing is pulled from Thor here -- the Master Mind's rule for the milestones.)
  2. THE BATTERY GOES FIRST: after (1), the dev-box GPU lock must be seen HELD by the battery (a job
     naming this step, not this suite's), and then be ABSENT for ``--quiet-s`` continuous seconds (default 30 min: a gap
     between two battery stages is not a release). Python compute apps on the card are LOGGED but
     do not hold the waiter (an unrelated lock-less GPU job would otherwise deadlock it; the
     runner's lock acquisition still requires the card free, then falls back to CPU per split).
     Fallbacks, each recorded: no battery lock seen within ``--battery-grace-h`` of (1) ->
     ``BATTERY_NOT_OBSERVED``; the lock never quiet within ``--max-battery-h`` -> ``BATTERY_TIMEOUT``.
  3. THE RUNNER: ``run_navsim_refcv7.py`` on all three splits (warmup -> navtest -> navhard).
  4. THE VERDICT IS READ FROM ARTIFACTS, never from an exit code: ``MILESTONE_SUMMARY.json`` with
     this md5, ``summary_<split>.json`` per split and ``BARS.json`` -> ``DONE`` or ``INCOMPLETE``
     with the missing artifacts named, in ``raw/milestones/waiter_<N>.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import gpu_lock  # noqa: E402

PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
KIT = "D:/refcv7_eval_kit/ckpt"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class State:
    def __init__(self, step: int):
        d = os.path.join(PKG, "raw", "milestones")
        os.makedirs(d, exist_ok=True)
        self.path = os.path.join(d, f"waiter_{step}.json")
        self.log = os.path.join(d, f"waiter_{step}.log")
        self.s = {"step": step, "pid": os.getpid(), "armed_utc": now(), "status": "ARMED",
                  "events": []}
        self.save()

    def ev(self, what: str, **kw) -> None:
        e = {"utc": now(), "event": what, **kw}
        self.s["events"].append(e)
        with open(self.log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(e) + "\n")
        self.save()

    def save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.s, fh, indent=1)
        os.replace(tmp, self.path)


def md5_file(p: str) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def md5sums_entry(name: str):
    p = os.path.join(KIT, "MD5SUMS")
    try:
        for ln in open(p, encoding="utf-8", errors="replace"):
            m = re.match(r"^([0-9a-f]{32})\s+\*?(\S+)", ln.strip())
            if m and os.path.basename(m.group(2)) == name:
                return m.group(1)
    except FileNotFoundError:
        return None
    return None


def ckpt_ready(step: int, st: State) -> tuple:
    name = f"ckpt_{step}.pt"
    p = os.path.join(KIT, name)
    want = md5sums_entry(name)
    if not (os.path.exists(p) and want):
        return None, None
    got = md5_file(p)
    if got != want:
        st.ev("CKPT_MD5_MISMATCH_WAITING", local=got, md5sums=want)
        return None, None
    import torch
    s = int(torch.load(p, map_location="cpu", weights_only=False, mmap=True).get("step", -1))
    if s != step:
        st.ev("CKPT_STEP_MISMATCH", step_key=s)
        return None, None
    return p, got


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", type=int)
    ap.add_argument("--quiet-s", type=int, default=1800)
    ap.add_argument("--battery-grace-h", type=float, default=4.0)
    ap.add_argument("--max-battery-h", type=float, default=10.0)
    ap.add_argument("--poll-s", type=int, default=60)
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args(argv)
    if a.status:
        print(open(os.path.join(PKG, "raw", "milestones", f"waiter_{a.step}.json"),
                   encoding="utf-8").read())
        return 0
    st = State(a.step)
    st.ev("ARMED", ckpt=os.path.join(KIT, f"ckpt_{a.step}.pt"), quiet_s=a.quiet_s,
          battery_grace_h=a.battery_grace_h)
    # 1. the checkpoint
    while True:
        p, md5 = ckpt_ready(a.step, st)
        if p:
            break
        time.sleep(a.poll_s)
    st.s["ckpt"], st.s["md5"] = p, md5
    st.ev("CKPT_READY", path=p, md5=md5)
    t_ready = time.time()
    # 2. the battery goes first
    seen, quiet_since = None, None
    while True:
        cur = gpu_lock.read()
        job = str((cur or {}).get("job", ""))
        # ⭐ THE BATTERY is identified by a lock job naming THIS milestone's step and not this suite
        # (MEASURED battery jobs: refcv7-g0-1500, refcv7-g0probe-1500, refcv7-smoke-1500). The
        # Research Lab takes the SAME lock (Master Mind, 2026-09-28 06:45: GPU runs 08:00-12:30, a
        # REFe snapshot eval ~13:25 for 50-70 min): its lock is LOGGED and waited out, never
        # mistaken for the battery's release. Any foreign lock fields are tolerated; a lock that is
        # not ours is never touched.
        tags = (str(a.step), f"{a.step // 1000}k", f"{a.step:,}")
        other = (cur is not None and not job.startswith("refcv7-navsim")
                 and any(t in job for t in tags))
        if cur is not None and not other and not job.startswith("refcv7-navsim") and                 st.s.get("_last_foreign_job") != job:
            st.s["_last_foreign_job"] = job
            st.ev("LOCK_SEEN_OTHER", job=job)
        try:
            busy = gpu_lock.other_python_compute()
        except Exception as e:                                            # noqa: BLE001
            busy = [{"nvidia_smi_error": repr(e)}]
        if other and seen is None:
            seen = cur.get("job")
            st.ev("BATTERY_LOCK_SEEN", job=seen)
        if cur is None:
            quiet_since = quiet_since or time.time()
        else:
            quiet_since = None
        st.s["last_poll"] = {"utc": now(), "lock": cur, "python_compute": busy}
        if seen and quiet_since and time.time() - quiet_since >= a.quiet_s:
            st.ev("BATTERY_RELEASED", job=seen, quiet_s=a.quiet_s, python_compute=busy)
            break
        if not seen and time.time() - t_ready >= a.battery_grace_h * 3600:
            st.ev("BATTERY_NOT_OBSERVED", after_h=a.battery_grace_h)
            break
        if time.time() - t_ready >= a.max_battery_h * 3600:
            st.ev("BATTERY_TIMEOUT", after_h=a.max_battery_h, lock=cur)
            break
        st.save()
        time.sleep(a.poll_s)
    # 3. the runner
    out = os.path.join(PKG, "raw", "milestones", f"step{a.step}")
    st.s["status"] = "RUNNING"
    st.ev("RUNNER_START", out=out)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    with open(os.path.join(PKG, "raw", "milestones", f"runner_{a.step}.out.txt"), "a",
              encoding="utf-8") as fh:
        subprocess.run([PY, os.path.join(HERE, "run_navsim_refcv7.py"), "--ckpt", p, "--md5", md5,
                        "--splits", "warmup,navtest,navhard", "--device", "auto", "--out", out],
                       stdout=fh, stderr=subprocess.STDOUT, env=env)
    # 4. the verdict, from the artifacts
    missing = []
    ms = os.path.join(out, "MILESTONE_SUMMARY.json")
    try:
        msd = json.load(open(ms, encoding="utf-8"))
        if msd.get("md5") != md5:
            missing.append(f"MILESTONE_SUMMARY.json md5 {msd.get('md5')} != {md5}")
    except Exception:                                                     # noqa: BLE001
        missing.append("MILESTONE_SUMMARY.json")
    for sk in ("warmup", "navtest", "navhard"):
        if not os.path.exists(os.path.join(out, f"summary_{sk}.json")):
            missing.append(f"summary_{sk}.json")
    if not os.path.exists(os.path.join(out, "BARS.json")):
        missing.append("BARS.json")
    st.s["status"] = "DONE" if not missing else "INCOMPLETE"
    st.s["missing"] = missing
    st.ev(f"ZZWAITER{a.step}{st.s['status']}ZZ", missing=missing)
    return 0


if __name__ == "__main__":
    sys.exit(main())
