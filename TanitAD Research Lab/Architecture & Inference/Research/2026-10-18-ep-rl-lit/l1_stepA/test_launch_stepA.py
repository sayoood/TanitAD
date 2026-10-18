"""Refusal-path tests of launch_stepA.py.  NOTHING is launched and the real dev-box lock is never touched: every test uses a temp lock/marker and injected GPU-memory / pid / sleep / run functions.
Same discipline as test_stepA.py: `chk_*` returns True when healthy; `test_red_under_mutation` applies one deliberate fault and requires the SAME check to turn False.

    python -m pytest -q test_launch_stepA.py
"""
import json
import os
import sys
import tempfile
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import launch_stepA as L  # noqa: E402

FOREIGN_MSYS = {"job": "refcv8-battery", "pid": 861623, "win_pid_of_locker": 4242, "host": "box", "acquired": "2026-10-18T10:00:00"}
FOREIGN_OWNER = {"pid": 7852, "owner": "srttc_s0", "t": 1.0}


class Env:
    def __init__(self, d, used=0, alive=False):
        self.lock, self.mark = os.path.join(d, "devbox_gpu.lock"), os.path.join(d, "devbox_gpu.chain_active")
        self.t, self.ran, self.log, self.used, self.alive = time.time(), [], [], used, alive

    def sleep(self, s):
        self.t += s

    def run(self, max_wait=120, run_fn=None):
        rf = run_fn or (lambda c: (self.ran.append(c), 0)[1])
        return L.run_locked(["echo", "gpu-job"], max_wait, self.lock, self.mark, used_fn=lambda: self.used, alive_fn=lambda p: self.alive, sleep_fn=self.sleep, now_fn=lambda: self.t,
                            run_fn=rf, poll_s=30.0, log=self.log.append)

    def put_lock(self, body, age_s=0.0):
        open(self.lock, "w", encoding="utf-8").write(json.dumps(body))
        if age_s:
            os.utime(self.lock, (self.t - age_s, self.t - age_s))


def fresh():
    return Env(tempfile.mkdtemp(prefix="l1launch_"))


# ============================================================================================ checks (True = healthy)
def chk_happy_path_lock_is_ours_and_released():
    e = fresh()
    seen = {}
    rc = e.run(run_fn=lambda c: (seen.update(held=json.loads(open(e.lock).read())), 0)[1])
    return bool(rc == 0 and seen["held"]["owner"] == L.OWNER and seen["held"]["pid"] == os.getpid() and not os.path.exists(e.lock))


def chk_chain_active_blocks():
    e = fresh()
    open(e.mark, "w").close()
    rc = e.run(max_wait=90)
    return bool(rc == L.EX_BUSY and e.ran == [] and not os.path.exists(e.lock) and any("chain_active" in x for x in e.log))


def chk_foreign_msys_lock_never_removed():
    e = fresh()
    e.put_lock(FOREIGN_MSYS, age_s=7200)               # old, and its MSYS pid is invisible to tasklist: a dead-pid rule would steal it
    before = open(e.lock, "rb").read()
    rc = e.run(max_wait=90)
    return bool(rc == L.EX_BUSY and e.ran == [] and open(e.lock, "rb").read() == before)


def chk_foreign_owner_lock_never_removed():
    e = fresh()
    e.put_lock(FOREIGN_OWNER, age_s=7200)
    before = open(e.lock, "rb").read()
    rc = e.run(max_wait=90)
    return bool(rc == L.EX_BUSY and e.ran == [] and open(e.lock, "rb").read() == before)


def chk_unreadable_lock_left_alone():
    e = fresh()
    open(e.lock, "w").write("{not json")
    rc = e.run(max_wait=90)
    return bool(rc == L.EX_BUSY and e.ran == [] and open(e.lock).read() == "{not json")


def chk_busy_gpu_gives_the_lock_back():
    e = fresh()
    e.used = 20000                                      # an unlocked GPU user (S0) holds the memory
    rc = e.run(max_wait=90)
    return bool(rc == L.EX_BUSY and e.ran == [] and not os.path.exists(e.lock) and any("memory.used" in x for x in e.log))


def chk_release_only_exact_body():
    e = fresh()

    def job(c):
        e.ran.append(c)
        open(e.lock, "w", encoding="utf-8").write(json.dumps(FOREIGN_OWNER))             # someone else took the lock while we ran
        return 0
    rc = e.run(run_fn=job)
    return bool(rc == 0 and os.path.exists(e.lock) and json.loads(open(e.lock).read()) == FOREIGN_OWNER and any("NOT" in x for x in e.log))


def chk_released_even_when_the_job_raises():
    e = fresh()

    def boom(c):
        raise RuntimeError("job died")
    try:
        e.run(run_fn=boom)
    except RuntimeError:
        pass
    return bool(not os.path.exists(e.lock))


def chk_own_stale_lock_only():
    a = fresh(); a.alive = False                                             # (a) our marker, dead pid, old: reclaimed
    a.put_lock({"pid": 1, "owner": L.OWNER, "t": 0}, age_s=3600)
    ra = a.run(max_wait=0)
    b = fresh(); b.alive = True                                              # (b) our marker, live pid: refused
    b.put_lock({"pid": 1, "owner": L.OWNER, "t": 0}, age_s=3600)
    rb = b.run(max_wait=0)
    c = fresh(); c.alive = False                                             # (c) our marker, dead pid, young: refused
    c.put_lock({"pid": 1, "owner": L.OWNER, "t": 0}, age_s=60)
    rc = c.run(max_wait=0)
    return bool(ra == 0 and a.ran and rb == L.EX_BUSY and not b.ran and rc == L.EX_BUSY and not c.ran)


def chk_o_excl_second_launcher_refused():
    e = fresh()
    b1, _ = L.try_lock(e.lock, e.mark, lambda: 0, lambda p: False, e.sleep, lambda: e.t)
    b2, w2 = L.try_lock(e.lock, e.mark, lambda: 0, lambda p: True, e.sleep, lambda: e.t)         # the second sees OUR owner marker with a live pid
    return bool(b1 and b2 is None and "live l1_stepA" in w2 and json.loads(open(e.lock).read())["pid"] == os.getpid())


def chk_probe_takes_nothing():
    e = fresh()
    p = L.probe(e.lock, e.mark, lambda: 0)
    no_lock_created = not os.path.exists(e.lock)
    e.put_lock(FOREIGN_OWNER)
    q = L.probe(e.lock, e.mark, lambda: 0)
    return bool(p["would_launch"] and no_lock_created and q["lock_exists"] and q["lock_holder"] == FOREIGN_OWNER and not q["would_launch"] and json.loads(open(e.lock).read()) == FOREIGN_OWNER)


def chk_plan_has_exactly_one_gpu_step():
    p = L.plan("W", "T")
    gpu = [x for x in p if "  GPU " in x]
    cpu = [x for x in p if "  CPU " in x]
    return bool(len(gpu) == 1 and 'launch_stepA.py" run' in gpu[0] and "make_stepA_bundle.py" in gpu[0] and cpu and all(("CUDA_VISIBLE_DEVICES=-1" in x) or ("navsim env" in x) for x in cpu))


def chk_child_env_marks_the_lock():
    e = L.child_env({"PATH": "x", "CUDA_VISIBLE_DEVICES": "-1"})
    return bool(e.get("L1_GPU_LOCKED") == "1" and "CUDA_VISIBLE_DEVICES" not in e and e["PATH"] == "x")


# ============================================================================================ mutations
def mut_child_env_unmarked(orig):
    return lambda base=None: {k: v for k, v in (os.environ if base is None else base).items() if k != "CUDA_VISIBLE_DEVICES"}


def mut_release_never(orig):
    return lambda body, lock=L.LOCK: False


def mut_ignore_chain_active(orig):
    def f(lock=L.LOCK, mark=L.MARK, *a, **k):
        return orig(lock, mark + ".never", *a, **k)
    return f


def mut_steal_any_lock(orig):
    def f(lock=L.LOCK, mark=L.MARK, *a, **k):
        if os.path.exists(lock):
            os.remove(lock)                                        # the dead-pid / old-lock "stale" rule that stole live batteries
        return orig(lock, mark, *a, **k)
    return f


def mut_reclaim_when_pid_dead_ignoring_owner(orig):
    def f(lock=L.LOCK, mark=L.MARK, used_fn=L.used_mib, alive_fn=L.pid_alive, *a, **k):
        h, age = L.read_holder(lock)
        if isinstance(h, dict) and age > L.OWN_STALE_S and not alive_fn(h.get("pid")):
            os.remove(lock)                                        # ignores the owner marker
        return orig(lock, mark, used_fn, alive_fn, *a, **k)
    return f


def mut_skip_memory_check(orig):
    def f(lock=L.LOCK, mark=L.MARK, used_fn=L.used_mib, *a, **k):
        return orig(lock, mark, lambda: 0, *a, **k)
    return f


def mut_release_unconditionally(orig):
    def f(body, lock=L.LOCK):
        try:
            os.remove(lock)
            return True
        except OSError:
            return False
    return f


def mut_no_finally(orig):
    def f(cmd, max_wait, lock=L.LOCK, mark=L.MARK, used_fn=L.used_mib, alive_fn=L.pid_alive, sleep_fn=L.time.sleep, now_fn=L.time.time, run_fn=None, poll_s=60.0, log=print):
        body, why = L.try_lock(lock, mark, used_fn, alive_fn, sleep_fn, now_fn)
        if not body:
            return L.EX_BUSY
        rc = run_fn(cmd)                                           # no try/finally: a dying job leaks the lock
        L.release(body, lock)
        return rc
    return f


def mut_reclaim_young_own_lock(orig):
    def f(lock=L.LOCK, mark=L.MARK, used_fn=L.used_mib, alive_fn=L.pid_alive, *a, **k):
        h, age = L.read_holder(lock)
        if isinstance(h, dict) and h.get("owner") == L.OWNER and not alive_fn(h.get("pid")):
            os.remove(lock)                                        # a launcher that just started is not 'dead'
        return orig(lock, mark, used_fn, alive_fn, *a, **k)
    return f


def mut_probe_takes_the_lock(orig):
    def f(lock=L.LOCK, mark=L.MARK, used_fn=L.used_mib):
        r = orig(lock, mark, used_fn)
        if not os.path.exists(lock):
            open(lock, "w").write("{}")
        return r
    return f


def mut_plan_two_gpu(orig):
    return lambda *a, **k: orig(*a, **k) + ["# 9  GPU  extra"]


TABLE = [
    ("happy_path_lock_is_ours_and_released", chk_happy_path_lock_is_ours_and_released, "release", mut_release_never),
    ("chain_active_blocks", chk_chain_active_blocks, "try_lock", mut_ignore_chain_active),
    ("foreign_msys_lock_never_removed", chk_foreign_msys_lock_never_removed, "try_lock", mut_steal_any_lock),
    ("foreign_owner_lock_never_removed", chk_foreign_owner_lock_never_removed, "try_lock", mut_reclaim_when_pid_dead_ignoring_owner),
    ("unreadable_lock_left_alone", chk_unreadable_lock_left_alone, "try_lock", mut_steal_any_lock),
    ("busy_gpu_gives_lock_back", chk_busy_gpu_gives_the_lock_back, "try_lock", mut_skip_memory_check),
    ("release_only_exact_body", chk_release_only_exact_body, "release", mut_release_unconditionally),
    ("released_when_job_raises", chk_released_even_when_the_job_raises, "run_locked", mut_no_finally),
    ("own_stale_lock_needs_age", chk_own_stale_lock_only, "try_lock", mut_reclaim_young_own_lock),
    ("own_stale_lock_needs_dead_pid_and_own_marker", chk_own_stale_lock_only, "try_lock", mut_steal_any_lock),
    ("second_launcher_refused", chk_o_excl_second_launcher_refused, "try_lock", mut_steal_any_lock),
    ("probe_takes_nothing", chk_probe_takes_nothing, "probe", mut_probe_takes_the_lock),
    ("plan_has_exactly_one_gpu_step", chk_plan_has_exactly_one_gpu_step, "plan", mut_plan_two_gpu),
    ("child_is_marked_as_locked", chk_child_env_marks_the_lock, "child_env", mut_child_env_unmarked),
]


@pytest.mark.parametrize("name,check", [(t[0], t[1]) for t in TABLE], ids=[t[0] for t in TABLE])
def test_green(name, check):
    assert check(), name


@pytest.mark.parametrize("name,check,attr,mutant", TABLE, ids=[t[0] for t in TABLE])
def test_red_under_mutation(name, check, attr, mutant, monkeypatch):
    monkeypatch.setattr(L, attr, mutant(getattr(L, attr)))
    try:
        healthy = check()
    except Exception:
        healthy = False
    assert not healthy, f"{name}: stayed GREEN under its deliberate mutation of launch_stepA.{attr}"


def test_defaults_are_the_landed_lock_paths():
    assert L.LOCK == "C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock" and L.MARK == "C:/Users/Admin/qland/work/refcv7/devbox_gpu.chain_active" and L.OWNER == "l1_stepA"


def test_cli_run_requires_a_command():
    assert L.main(["run", "--max-wait", "0", "--lock", os.path.join(tempfile.mkdtemp(), "l"), "--mark", os.path.join(tempfile.mkdtemp(), "m")]) == 2
