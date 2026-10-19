"""Tests of the compute-type GPU guard in launch_stepA_ctype.py (PI decision 2026-10-19, "Allow compute-type check").  NOTHING is launched and the real dev-box lock is never touched:
temp lock/marker, injected memory / compute / pid / sleep / run functions.  The nvidia-smi text is a LIVE capture of this box (fixtures/nvidia_smi_live_20261019.txt: 35 C+G rows + 1 C row).
Same discipline as test_launch_stepA.py: `chk_*` is True when healthy; `test_red_under_mutation` applies one deliberate fault and requires the SAME check to turn False.

    CUDA_VISIBLE_DEVICES=-1 python -m pytest -q test_launch_stepA_ctype.py
"""
import json
import os
import re
import sys
import tempfile
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))                              # repo layout: the frozen landed launch_stepA.py sits one level up (l1_stepA/); run layout: beside this file
sys.path.insert(0, HERE)
import launch_stepA as LANDED  # noqa: E402  (the frozen original; only READ here)
import launch_stepA_ctype as L  # noqa: E402

FIXTURE = next(p for p in (os.path.join(HERE, "fixtures", "nvidia_smi_live_20261019.txt"), os.path.join(HERE, "raw", "nvidia_smi_live_20261019.txt")) if os.path.exists(p))
LIVE = open(FIXTURE, encoding="utf-8").read()
C_PID = 62136                                                          # the single type-C row of the live capture
DESKTOP_MIB = 2300                                                     # what the C+G desktop apps hold with no compute job (MEASURED 2026-10-19, hn34 + this box)


def strip_c_rows(text):
    return "\n".join(x for x in text.splitlines() if not re.match(r"^\|\s+\d+\s+N/A\s+N/A\s+\d+\s+C\s", x))


class Env:
    def __init__(self, d, used=DESKTOP_MIB, compute=None):
        self.lock, self.mark = os.path.join(d, "devbox_gpu.lock"), os.path.join(d, "devbox_gpu.chain_active")
        self.t, self.ran, self.log, self.used, self.compute = time.time(), [], [], used, compute if compute is not None else []
        self.ram, self.compute_calls = 16.0, 0

    def sleep(self, s):
        self.t += s

    def try_lock(self, mod=None, compute_fn="env"):
        m = mod or L
        kw = {} if compute_fn is None else {"compute_fn": (lambda: self.compute) if compute_fn == "env" else compute_fn}
        return m.try_lock(self.lock, self.mark, lambda: self.used, lambda p: False, self.sleep, lambda: self.t, **kw)

    def _compute(self):
        self.compute_calls += 1
        return self.compute

    def run(self, max_wait=120):
        return L.run_locked(["echo", "gpu-job"], max_wait, self.lock, self.mark, used_fn=lambda: self.used, alive_fn=lambda p: False, sleep_fn=self.sleep, now_fn=lambda: self.t,
                            run_fn=lambda c: (self.ran.append(c), 0)[1], poll_s=30.0, log=self.log.append, compute_fn=self._compute, avail_fn=lambda: self.ram)


def fresh(**k):
    return Env(tempfile.mkdtemp(prefix="l1ctype_"), **k)


# ============================================================================================ checks (True = healthy)
def chk_live_table_c_row_is_detected():
    return bool(L.compute_pids(LIVE) == [C_PID])


def chk_cg_rows_are_ignored():
    # the live table with its one C row removed is 35 C+G rows: none counts
    return bool(LIVE.count("C+G") == 35 and L.compute_pids(strip_c_rows(LIVE)) == [] and L.compute_pids("") == [])


def chk_cg_only_desktop_gpu_launches():
    e = fresh(used=DESKTOP_MIB, compute=L.compute_pids(strip_c_rows(LIVE)))               # 2300 MiB held by C+G apps, no C row
    body, why = e.try_lock()
    ok = bool(body and why == "ok" and os.path.exists(e.lock))
    L.release(body, e.lock) if body else None
    return ok and not os.path.exists(e.lock)


def chk_c_row_refuses_and_gives_the_lock_back():
    e = fresh(used=DESKTOP_MIB, compute=L.compute_pids(LIVE))                              # the LIVE table: one C row
    body, why = e.try_lock()
    return bool(body is None and "compute process" in why and str(C_PID) in why and not os.path.exists(e.lock))


def chk_run_waits_and_never_runs_the_job_while_c_is_present():
    e = fresh(used=DESKTOP_MIB, compute=[C_PID])
    rc = e.run(max_wait=90)
    return bool(rc == L.EX_BUSY and e.ran == [] and not os.path.exists(e.lock) and any("compute process" in x for x in e.log) and any("ZZGPU_NOT_RUN" in x for x in e.log))


def chk_run_launches_once_the_c_row_disappears():
    e = fresh(used=DESKTOP_MIB, compute=[C_PID])
    state = {"n": 0}
    orig_sleep = e.sleep

    def sleep(s):
        orig_sleep(s)
        state["n"] += 1
        if state["n"] >= 3:                                                                # the other session's chain finished its job
            e.compute = []
    e.sleep = sleep
    rc = e.run(max_wait=3600)
    return bool(rc == 0 and len(e.ran) == 1 and not os.path.exists(e.lock) and any("ZZGPUWAIT" in x for x in e.log) and any("ZZGPULOCKED" in x for x in e.log))


def chk_backstop_refuses_a_big_unlisted_user():
    e = fresh(used=7000, compute=[])                                                       # no C row but 7 GB resident: a user nvidia-smi does not classify
    body, why = e.try_lock()
    return bool(body is None and "memory.used" in why and not os.path.exists(e.lock))


def chk_unreadable_nvidia_smi_fails_closed():
    import subprocess
    real = subprocess.run

    def boom(*a, **k):
        raise OSError("nvidia-smi gone")
    subprocess.run = boom
    try:
        cp = L.compute_pids()                                                              # the real code path, no injected text
    finally:
        subprocess.run = real
    return bool(cp == [-1])


def chk_chain_active_still_blocks():
    e = fresh()
    open(e.mark, "w").close()
    rc = e.run(max_wait=90)
    return bool(rc == L.EX_BUSY and e.ran == [] and any("chain_active" in x for x in e.log))


def chk_foreign_lock_still_never_removed():
    e = fresh()
    foreign = {"pid": 7852, "owner": "srttc_s0", "t": 1.0}
    open(e.lock, "w", encoding="utf-8").write(json.dumps(foreign))
    os.utime(e.lock, (e.t - 7200, e.t - 7200))
    before = open(e.lock, "rb").read()
    rc = e.run(max_wait=90)
    return bool(rc == L.EX_BUSY and e.ran == [] and open(e.lock, "rb").read() == before)


def chk_probe_reports_compute_pids_and_takes_nothing():
    e = fresh()
    p = L.probe(e.lock, e.mark, lambda: DESKTOP_MIB, lambda: [])
    q = L.probe(e.lock, e.mark, lambda: DESKTOP_MIB, lambda: [C_PID])
    return bool(p["would_launch"] and p["compute_pids"] == [] and not q["would_launch"] and q["compute_pids"] == [C_PID] and not os.path.exists(e.lock))


def chk_landed_launcher_can_never_pass_on_the_desktop_gpu():
    # the CONTROL for the whole patch: the frozen original, given the SAME C+G-only desktop (2300 MiB), refuses forever; the patched copy launches (chk_cg_only_desktop_gpu_launches)
    e = fresh(used=DESKTOP_MIB)
    body, why = e.try_lock(mod=LANDED, compute_fn=None)
    return bool(body is None and "memory.used" in why and not os.path.exists(e.lock))


def chk_plan_uses_the_patched_launcher_for_the_one_gpu_step():
    gpu = [x for x in L.plan("W", "T") if "  GPU " in x]
    return bool(len(gpu) == 1 and 'launch_stepA_ctype.py" run' in gpu[0] and "make_stepA_bundle.py" in gpu[0] and "launch_stepA.py" not in gpu[0])


def chk_low_ram_waits_and_never_takes_the_lock():
    e = fresh(used=DESKTOP_MIB, compute=[])
    e.ram = 4.2                                                                            # this box at 06:09: 4.26 GB available
    rc = e.run(max_wait=300)
    return bool(rc == L.EX_BUSY and e.ran == [] and e.compute_calls == 0 and not os.path.exists(e.lock) and any("RAM available 4 GB < 8 GB" in x for x in e.log))


def chk_enough_ram_launches():
    e = fresh(used=DESKTOP_MIB, compute=[])
    e.ram = 8.5
    rc = e.run(max_wait=300)
    return bool(rc == 0 and len(e.ran) == 1 and not os.path.exists(e.lock))


def chk_ram_probe_fails_closed():
    saved = sys.modules.get("psutil", "ABSENT")
    sys.modules["psutil"] = None                                                           # `import psutil` now raises ImportError
    try:
        v = L.avail_gb()
    finally:
        if saved == "ABSENT":
            sys.modules.pop("psutil", None)
        else:
            sys.modules["psutil"] = saved
    return bool(v == 0.0)


def chk_live_ram_probe_is_a_real_number():
    v = L.avail_gb()
    return bool(0.1 < v < 4096)


def chk_thresholds():
    return bool(L.MIN_AVAIL_GB == 8.0 and L.MAX_USED_MIB == 6000 and LANDED.MAX_USED_MIB == 1500 and L.OWNER == "l1_stepA" and L.LOCK == LANDED.LOCK and L.MARK == LANDED.MARK)


# ============================================================================================ mutations
def mut_ctype_blind(orig):
    return lambda text=None: []                                                            # never sees a compute process


def mut_cg_counts_as_compute(orig):
    def f(text=None):
        if text is None:
            return orig(None)
        out = []
        for line in text.splitlines():
            t = line.strip().strip("|").split()
            if len(t) >= 6 and t[0].isdigit() and t[3].isdigit() and t[4].startswith("C"):          # C+G counted: the landed rule's blindness in a new costume
                out.append(int(t[3]))
        return out
    return f


def mut_fail_open(orig):
    def f(text=None):
        r = orig(text)
        return [] if r == [-1] else r                                                      # an unreadable GPU becomes an empty one
    return f


def mut_skip_backstop(orig):
    def f(lock=L.LOCK, mark=L.MARK, used_fn=L.used_mib, *a, **k):
        return orig(lock, mark, lambda: 0, *a, **k)
    return f


def mut_keep_lock_on_refusal(orig):
    return lambda body, lock=L.LOCK: False                                                 # a refused launcher leaves its lock behind


def mut_plan_old_launcher(orig):
    return lambda *a, **k: [x.replace("launch_stepA_ctype.py", "launch_stepA.py") for x in orig(*a, **k)]


def mut_ignore_chain_active(orig):
    def f(lock=L.LOCK, mark=L.MARK, *a, **k):
        return orig(lock, mark + ".never", *a, **k)
    return f


def mut_steal_any_lock(orig):
    def f(lock=L.LOCK, mark=L.MARK, *a, **k):
        if os.path.exists(lock):
            os.remove(lock)
        return orig(lock, mark, *a, **k)
    return f


def mut_probe_ignores_compute(orig):
    def f(lock=L.LOCK, mark=L.MARK, used_fn=L.used_mib, compute_fn=None):
        r = orig(lock, mark, used_fn, compute_fn)
        r["would_launch"] = bool(not os.path.exists(mark) and not os.path.exists(lock) and used_fn() <= L.MAX_USED_MIB)
        return r
    return f


def mut_try_lock_skips_compute_check(orig):
    def f(lock=L.LOCK, mark=L.MARK, used_fn=L.used_mib, alive_fn=L.pid_alive, sleep_fn=time.sleep, now_fn=time.time, compute_fn=None):
        return orig(lock, mark, used_fn, alive_fn, sleep_fn, now_fn, lambda: [])             # the memory test stays; the compute-type test is gone
    return f


def mut_ram_gate_off(orig):
    def f(*a, **k):
        k["avail_fn"] = lambda: 99.0
        return orig(*a, **k)
    return f


def mut_ram_always_starved(orig):
    def f(*a, **k):
        k["avail_fn"] = lambda: 0.0
        return orig(*a, **k)
    return f


def mut_ram_probe_fail_open(orig):
    return lambda: orig() or 99.0                                                          # an unreadable RAM probe becomes plenty of RAM


TABLE = [
    ("live_table_c_row_is_detected", chk_live_table_c_row_is_detected, "compute_pids", mut_ctype_blind),
    ("cg_rows_are_ignored", chk_cg_rows_are_ignored, "compute_pids", mut_cg_counts_as_compute),
    ("cg_only_desktop_gpu_launches", chk_cg_only_desktop_gpu_launches, "compute_pids", mut_cg_counts_as_compute),
    ("c_row_refuses_and_gives_the_lock_back", chk_c_row_refuses_and_gives_the_lock_back, "try_lock", mut_try_lock_skips_compute_check),
    ("c_row_refuses_and_gives_the_lock_back/blind", chk_c_row_refuses_and_gives_the_lock_back, "compute_pids", mut_ctype_blind),
    ("c_row_refuses_and_gives_the_lock_back/release", chk_c_row_refuses_and_gives_the_lock_back, "release", mut_keep_lock_on_refusal),
    ("run_never_runs_the_job_while_c_present", chk_run_waits_and_never_runs_the_job_while_c_is_present, "try_lock", mut_try_lock_skips_compute_check),
    ("run_never_runs_the_job_while_c_present/release", chk_run_waits_and_never_runs_the_job_while_c_is_present, "release", mut_keep_lock_on_refusal),
    ("run_launches_once_c_disappears", chk_run_launches_once_the_c_row_disappears, "release", mut_keep_lock_on_refusal),
    ("backstop_refuses_a_big_unlisted_user", chk_backstop_refuses_a_big_unlisted_user, "try_lock", mut_skip_backstop),
    ("unreadable_nvidia_smi_fails_closed", chk_unreadable_nvidia_smi_fails_closed, "compute_pids", mut_fail_open),
    ("chain_active_still_blocks", chk_chain_active_still_blocks, "try_lock", mut_ignore_chain_active),
    ("foreign_lock_still_never_removed", chk_foreign_lock_still_never_removed, "try_lock", mut_steal_any_lock),
    ("probe_reports_compute_pids", chk_probe_reports_compute_pids_and_takes_nothing, "probe", mut_probe_ignores_compute),
    ("low_ram_waits_and_never_takes_the_lock", chk_low_ram_waits_and_never_takes_the_lock, "run_locked", mut_ram_gate_off),
    ("enough_ram_launches", chk_enough_ram_launches, "run_locked", mut_ram_always_starved),
    ("ram_probe_fails_closed", chk_ram_probe_fails_closed, "avail_gb", mut_ram_probe_fail_open),
    ("plan_uses_the_patched_launcher", chk_plan_uses_the_patched_launcher_for_the_one_gpu_step, "plan", mut_plan_old_launcher),
]


@pytest.mark.parametrize("name,check", [(t[0], t[1]) for t in TABLE] + [("landed_launcher_can_never_pass_on_the_desktop_gpu", chk_landed_launcher_can_never_pass_on_the_desktop_gpu),
                                                                       ("thresholds", chk_thresholds), ("live_ram_probe_is_a_real_number", chk_live_ram_probe_is_a_real_number)],
                         ids=[t[0] for t in TABLE] + ["landed_launcher_can_never_pass_on_the_desktop_gpu", "thresholds", "live_ram_probe_is_a_real_number"])
def test_green(name, check):
    assert check(), name


@pytest.mark.parametrize("name,check,attr,mutant", TABLE, ids=[t[0] for t in TABLE])
def test_red_under_mutation(name, check, attr, mutant, monkeypatch):
    monkeypatch.setattr(L, attr, mutant(getattr(L, attr)))
    try:
        healthy = check()
    except Exception:
        healthy = False
    assert not healthy, f"{name}: stayed GREEN under its deliberate mutation of launch_stepA_ctype.{attr}"


def test_threshold_red_when_backstop_is_the_old_1500_and_cg_apps_hold_2300(monkeypatch):
    # the C+G-only launch must go RED if the numeric rule is put back to the landed 1500 (the patch's whole point)
    monkeypatch.setattr(L, "MAX_USED_MIB", 1500)
    assert not chk_cg_only_desktop_gpu_launches()


def test_the_frozen_original_is_untouched():
    import hashlib
    p = LANDED.__file__
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == "96133d51b7b74257c19a4f45a75874a84d678df781ba0184280a4d51576879ab"
