#!/usr/bin/env python3
"""EVENT-DRIVEN waiter for the navtest EPDMS re-score (TANITAD VENV). Detached; UNATTENDED.

    python code/epdms_waiter7.py --detach     # start the waiter as a DETACHED process and print its PID
    python code/epdms_waiter7.py              # run in the foreground (what --detach launches)
    python code/epdms_waiter7.py --status     # print raw/milestones/epdms_waiter.json
    python code/epdms_waiter7.py --once       # print the readiness table and the next work item, then exit

Pre-registration: ``SPEC_ADDENDUM_NAVTEST_EPDMS.md``. The pattern is ``milestone_waiter7.py``'s: it waits for
EVENTS, never for a clock, and reads every verdict from ARTIFACTS, never from an exit code.

WHAT IT RUNS, in this order, ONE scorer at a time (each a child process of ``score_navtest_epdms7.py score``):
  step 5,000 : STOP, R7_A1, R7_A1_s1, CV, PRIOR_ha0p, R7_CEILDECL_d, HUMAN  (the floors are checkpoint-independent
               and are scored once, here, on the same cache and tokens)
  step 30,000: R7_A1, R7_A1_s1, PRIOR_ha0p, R7_CEILDECL_d
  step 50,400: R7_A1, R7_A1_s1, PRIOR_ha0p, R7_CEILDECL_d
then ``summarize --step N`` per milestone.

AN ARM IS ELIGIBLE only when
  (1) its milestone is DONE -- step 5,000: ``MILESTONE_SUMMARY.json`` with the right md5 and ``BARS.json``;
      step 30,000: ``complete.log`` carries ``ZZCOMPLETE7DONEZZ step=30000``; step 50,400: ``MILESTONE_SUMMARY.json``
      with the right md5 and ``BARS.json`` (the runner writes them last) --
  (2) its banked navtest seam exists and passes the addendum's G4 (12,146 unique tokens whose set hash is the
      cache's), and
  (3) the box is quiet: ``FreePhysicalMemory >= 9 GB`` AND ``FreeVirtualMemory >= 6 GB`` (commit headroom) on
      5 CONSECUTIVE samples 30 s apart, re-read before EVERY arm and every retry.
The scorer's own RAM guard is 4,000 MB (the other registered jobs' floor is 3,000 MB), so this job yields first;
a guard abort is retried after a fresh gate. It runs at BELOW_NORMAL priority, uses no GPU, and writes nothing
under ``raw/milestones/step<N>/``.

State: ``raw/milestones/epdms_waiter.json``. Log (JSON lines): ``raw/milestones/epdms_waiter.log``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import score_navtest_epdms7 as E  # noqa: E402

PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
SCORE = os.path.join(HERE, "score_navtest_epdms7.py")
MS = os.path.join(PKG, "raw", "milestones")
STATE = os.path.join(MS, "epdms_waiter.json")
LOG = os.path.join(MS, "epdms_waiter.log")
CHILD_OUT = os.path.join(MS, "epdms_waiter_child.out.txt")

#: milestone -> (md5 of the checkpoint, done rule)
MILESTONES = {
    5000: {"md5": "06eb9dfde9f3783a782cebef22259ce4", "rule": "summary+bars"},
    30000: {"md5": "ac4e4fab87e35b20de24d8d94910d910", "rule": "complete_log_marker"},
    50400: {"md5": "b418d0fc4a92a6848c246a6a7c50207b", "rule": "summary+bars"},
}
#: (step-or-None for a floor, arm). None = checkpoint-independent.
WORK = ([(None, "STOP"), (5000, "R7_A1"), (5000, "R7_A1_s1"), (None, "CV"), (5000, "PRIOR_ha0p"),
         (5000, "R7_CEILDECL_d"), (None, "HUMAN")]
        + [(s, a) for s in (30000, 50400) for a in E.STEP_ORDER])
MAX_TRIES_RETRYABLE = 30
MAX_TRIES_HARD = 2


def now() -> str:
    return E.now()


# --------------------------------------------------------------------------- #
# readiness (pure functions of ARTIFACTS -- tested)                            #
# --------------------------------------------------------------------------- #
def _read_json(p: str):
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:                                                       # noqa: BLE001
        return None


def milestone_done(step: int, ms_root: str = MS) -> tuple:
    """-> (done: bool, why: str). Verdicts from artifacts only."""
    cfg = MILESTONES[step]
    d = os.path.join(ms_root, f"step{step}")
    if cfg["rule"] == "complete_log_marker":
        p = os.path.join(d, "complete.log")
        try:
            txt = open(p, encoding="utf-8", errors="replace").read()
        except OSError:
            return False, "complete.log absent"
        return (f"ZZCOMPLETE7DONEZZ step={step}" in txt,
                "marker present" if f"ZZCOMPLETE7DONEZZ step={step}" in txt else "marker ZZCOMPLETE7DONEZZ absent")
    ms = _read_json(os.path.join(d, "MILESTONE_SUMMARY.json"))
    if ms is None:
        return False, "MILESTONE_SUMMARY.json absent"
    if ms.get("md5") != cfg["md5"]:
        return False, f"MILESTONE_SUMMARY.json md5 {ms.get('md5')} != {cfg['md5']}"
    if not os.path.exists(os.path.join(d, "BARS.json")):
        return False, "BARS.json absent"
    return True, "MILESTONE_SUMMARY.json + BARS.json present"


_SEAM_CACHE: dict = {}


def seam_ok(step: int, arm: str) -> tuple:
    """G4 on a banked seam, cached on (path, size, mtime): the waiter polls every 30-60 s and a seam is never
    rewritten by the milestone drivers (``np.savez`` stamps zip times, so a rewrite WOULD change the key)."""
    p = E.seam_path(step, arm)
    if not os.path.exists(p):
        return False, "seam absent"
    key = (p, os.path.getsize(p), os.path.getmtime(p))
    if key in _SEAM_CACHE:
        return _SEAM_CACHE[key]
    try:
        bad = E.seam_refusals(E.seam_info(p), full=True)
    except Exception as e:                                                  # noqa: BLE001
        return False, f"seam unreadable: {type(e).__name__}"          # not cached: it may be mid-write
    res = ((not bad), ("ok" if not bad else "; ".join(bad)))
    _SEAM_CACHE[key] = res
    return res


def absent_final(step, arm: str, ms_root: str = MS) -> bool:
    """A model arm whose milestone is DONE but whose seam is absent/invalid will never appear: terminal."""
    if step is None:
        return False
    return milestone_done(step, ms_root)[0] and not seam_ok(step, arm)[0]


def item_done(step, arm: str) -> bool:
    return E.read_done(step, arm) is not None


def eligible(step, arm: str, ms_root: str = MS) -> tuple:
    """-> (eligible, why). Floors need only the cache; model arms need a DONE milestone and a valid seam."""
    if step is None:
        return True, "floor"
    ok, why = milestone_done(step, ms_root)
    if not ok:
        return False, f"milestone step{step} not done: {why}"
    ok, why = seam_ok(step, arm)
    return ok, ("seam ok" if ok else f"seam: {why}")


def next_item(tries: dict, given_up: set, ms_root: str = MS):
    """The first item in WORK order that is not done, not given up, and eligible. -> (item, table)."""
    table = []
    pick = None
    for step, arm in WORK:
        key = f"{'floors' if step is None else 'step' + str(step)}/{arm}"
        if item_done(step, arm):
            table.append((key, "DONE"))
            continue
        if key in given_up:
            table.append((key, "GAVE_UP"))
            continue
        if absent_final(step, arm, ms_root):
            table.append((key, "ABSENT"))
            continue
        ok, why = eligible(step, arm, ms_root)
        table.append((key, "ELIGIBLE" if ok else f"waiting: {why}"))
        if ok and pick is None:
            pick = (step, arm, key)
    return pick, table


# --------------------------------------------------------------------------- #
class State:
    def __init__(self):
        os.makedirs(MS, exist_ok=True)
        self.s = {"pid": os.getpid(), "armed_utc": now(), "status": "ARMED", "tries": {}, "given_up": [],
                  "summarized": [], "last_poll_utc": None, "events_tail": [], "addendum": E.addendum_state()}
        self.save()

    def ev(self, what: str, **kw) -> None:
        e = {"utc": now(), "event": what, **kw}
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(e, default=str) + "\n")
        self.s["events_tail"] = (self.s["events_tail"] + [e])[-12:]
        self.save()

    def save(self) -> None:
        tmp = STATE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.s, fh, indent=1, default=str)
        os.replace(tmp, STATE)


def alive_other() -> int:
    import psutil
    d = _read_json(STATE)
    if d and d.get("status") in ("ARMED", "WAITING", "GATING", "RUNNING") and int(d.get("pid", -1)) != os.getpid():
        pid = int(d["pid"])
        if psutil.pid_exists(pid):
            try:
                cl = " ".join(psutil.Process(pid).cmdline())
            except Exception:                                               # noqa: BLE001
                cl = ""
            if "epdms_waiter7" in cl:
                return pid
    return 0


def run_summary(step: int, st: State) -> None:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", OMP_NUM_THREADS="2")
    with open(CHILD_OUT, "a", encoding="utf-8") as fh:
        rc = subprocess.run([PY, SCORE, "summarize", "--step", str(step)], stdout=fh, stderr=subprocess.STDOUT,
                            env=env).returncode
    p = os.path.join(E.OUT, f"step{step}", "summary_navtest_epdms.json")
    d = _read_json(p)
    st.s["summarized"] = sorted(set(st.s["summarized"]) | ({step} if d else set()))
    st.ev("SUMMARY", step=step, status=(d or {}).get("status"), artifact=os.path.exists(p), rc_not_trusted=rc)


def milestone_exhausted(step: int, tries, given_up) -> bool:
    """All of a milestone's model arms done or given up (floors are summarised with step 5,000)."""
    for s, arm in WORK:
        key = f"step{s}/{arm}"
        if s == step and not item_done(s, arm) and key not in given_up and not absent_final(s, arm):
            return False
    if step == 5000:
        for arm in E.FLOOR_ORDER:
            if not item_done(None, arm) and f"floors/{arm}" not in given_up:
                return False
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detach", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--poll-s", type=int, default=60)
    ap.add_argument("--max-hours", type=float, default=120.0)
    a = ap.parse_args(argv)
    if a.status:
        print(open(STATE, encoding="utf-8").read())
        return 0
    if a.once:
        pick, table = next_item({}, set())
        print(json.dumps({"mem": E.mem_status(), "gate_open_now": E.gate_ok(E.mem_status()), "next": pick,
                          "table": table, "addendum_ok": E.addendum_state()["ok"],
                          "milestones": {s: milestone_done(s) for s in MILESTONES}}, indent=1, default=str))
        return 0
    if a.detach:
        other = alive_other()
        if other:
            print(f"REFUSED: a waiter is already alive (pid {other})")
            return 2
        flags = 0x00000008 | 0x00000200                # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        args = [PY, os.path.abspath(__file__), "--poll-s", str(a.poll_s), "--max-hours", str(a.max_hours)]
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", OMP_NUM_THREADS="2")
        out = open(os.path.join(MS, "epdms_waiter.nohup.txt"), "a", encoding="utf-8")
        try:
            p = subprocess.Popen(args, creationflags=flags | 0x01000000, stdout=out, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, env=env, close_fds=True, cwd=PKG)   # + BREAKAWAY_FROM_JOB
        except OSError:
            p = subprocess.Popen(args, creationflags=flags, stdout=out, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, env=env, close_fds=True, cwd=PKG)
        print(f"detached waiter pid {p.pid}")
        return 0
    other = alive_other()
    if other:
        print(f"REFUSED: a waiter is already alive (pid {other})")
        return 2
    E.lower_priority()
    st = State()
    ads = E.addendum_state()
    st.ev("ARMED", poll_s=a.poll_s, addendum_ok=ads["ok"], addendum_sha=ads["current"],
          gate={"phys_gb": E.GATE_PHYS_GB, "virt_gb": E.GATE_VIRT_GB, "samples": E.GATE_SAMPLES,
                "every_s": E.GATE_EVERY_S}, ram_floor_mb_wrapper=4000, work=[f"{s}/{x}" for s, x in WORK])
    if not ads["ok"]:
        st.s["status"] = "REFUSED_ADDENDUM_CHANGED"
        st.ev("ZZEPDMSWAITERREFUSEDZZ", why="addendum sha256 is not the last recorded hash line")
        return 2
    t_start = time.time()
    tries: dict = st.s["tries"]
    given_up: set = set()
    last_hb = 0.0
    samples: list = []
    while True:
        if time.time() - t_start > a.max_hours * 3600:
            st.s["status"] = "TIMEOUT"
            st.ev("ZZEPDMSWAITERTIMEOUTZZ", hours=a.max_hours)
            return 1
        if not E.addendum_state()["ok"]:
            # an amendment is written and its hash line appended a few ms apart: re-read before refusing
            time.sleep(3)
            if not E.addendum_state()["ok"]:
                time.sleep(10)
            if not E.addendum_state()["ok"]:
                st.s["status"] = "REFUSED_ADDENDUM_CHANGED"
                st.ev("ZZEPDMSWAITERREFUSEDZZ", why="addendum sha256 changed while the waiter was running")
                return 2
        # summaries first: a milestone whose arms are all done/given up is summarised exactly once
        for step in MILESTONES:
            if step not in st.s["summarized"] and milestone_done(step)[0] and milestone_exhausted(step, tries, given_up):
                run_summary(step, st)
        pick, table = next_item(tries, given_up)
        mem = E.mem_status()
        st.s.update({"last_poll_utc": now(), "mem": mem, "table": table, "next": pick,
                     "gate_open_now": E.gate_ok(mem), "tries": tries, "given_up": sorted(given_up)})
        if time.time() - last_hb > 1800:
            last_hb = time.time()
            st.ev("HEARTBEAT", mem=mem, next=pick, n_done=sum(1 for _, v in table if v == "DONE"),
                  n_total=len(table))
        if pick is None:
            n_pending = sum(1 for _, v in table if v not in ("DONE", "GAVE_UP", "ABSENT"))
            if n_pending == 0 and all(s in st.s["summarized"] for s in MILESTONES):
                st.s["status"] = "DONE"
                st.ev("ZZEPDMSWAITERDONEZZ", gave_up=sorted(given_up))
                return 0
            st.s["status"] = "WAITING"
            st.save()
            samples = []
            time.sleep(a.poll_s)
            continue
        step, arm, key = pick
        # ---- the quiet-box gate: 5 consecutive samples, re-read before every attempt
        st.s["status"] = "GATING"
        samples = (samples + [mem])[-E.GATE_SAMPLES:]
        st.save()
        if not E.gate_sustained(samples):
            time.sleep(E.GATE_EVERY_S)
            continue
        # ---- run ONE arm
        st.s["status"] = "RUNNING"
        st.ev("ARM_START", key=key, attempt=int(tries.get(key, 0)) + 1, mem=E.mem_status())
        cmd = [PY, SCORE, "score", "--arm", arm] + (["--step", str(step)] if step is not None else [])
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", OMP_NUM_THREADS="2")
        t0 = time.time()
        with open(CHILD_OUT, "a", encoding="utf-8") as fh:
            fh.write(f"\n=== {now()} {key} attempt {int(tries.get(key, 0)) + 1} ===\n")
            fh.flush()
            subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
        samples = []
        d = E.read_done(step, arm)
        if d is not None:
            tries[key] = int(tries.get(key, 0)) + 1
            st.ev("ARM_DONE", key=key, epdms_x100=d.get("headline_x100"), wall_s=round(time.time() - t0, 1))
            continue
        # the verdict is the ARTIFACT: ARM_LAST_ATTEMPT.json written by the child (a closed gate / low disk /
        # a running twin are NOT attempts and cost no retry; a crash that left no artifact IS one)
        la = _read_json(os.path.join(E.OUT, "floors" if step is None else f"step{step}", arm,
                                     "ARM_LAST_ATTEMPT.json")) or {}
        fresh = float(la.get("epoch", 0)) >= t0 - 2
        status = la.get("status") if fresh else None
        if status in ("GATE_CLOSED", "DISK_LOW", "ALREADY_RUNNING"):
            st.ev("ARM_DEFERRED", key=key, status=status, wall_s=round(time.time() - t0, 1))
            time.sleep(60)
            continue
        tries[key] = int(tries.get(key, 0)) + 1
        failed = _read_json(os.path.join(E.OUT, "floors" if step is None else f"step{step}", arm,
                                         "ARM_FAILED.json")) or {}
        retry = bool(failed.get("retryable", la.get("retryable", True))) if status else True
        st.ev("ARM_NOT_DONE", key=key, status=status or "NO_ARTIFACT (child crashed)", retryable=retry,
              attempt=tries[key], wall_s=round(time.time() - t0, 1))
        limit = MAX_TRIES_RETRYABLE if retry else MAX_TRIES_HARD
        if tries[key] >= limit:
            given_up.add(key)
            st.ev("GAVE_UP", key=key, tries=tries[key], last_status=status)
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
