"""LOCK-STEAL-2 regression: c3_run_p2.take_lock must clear ONLY a lock carrying its own job marker (2026-10-18).

The shared dev-box lock `devbox_gpu.lock` is written by TWO schemas: this helper's {"job": "p2-own-table-<tag>", "pid", "host", "acquired"} and the refcv8 battery's
{"job", "pid" = an MSYS pid, "win_pid_of_locker", "host", "acquired"}.  An MSYS pid is invisible to `tasklist`, so the old rule ("holder pid not alive and the
lock older than 600 s -> os.remove") deleted a LIVE battery lock (MEASURED 2026-10-18: pid 861623, 2 h old) and let this helper take the GPU.

    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I test_c3_run_p2_foreign_lock.py      -> prints ZZTEST_PASS <n> (exit 0) or ZZTEST_FAIL (exit 1)

take_lock() runs against a TEMP lock with `time.sleep` stubbed (the 4 s settle returns, the 3 s poll raises _Stop) and the GPU probe stubbed idle, so the call
returns "TOOK" or "WAITED" in milliseconds.  The real shared lock and chain_active are only `stat`ed (read-only), never opened.
(c) Mutations are applied IN-PROCESS to the source text (the file on disk is never edited) and must make the named check go RED.
"""
import json
import os
import re
import sys
import tempfile
import time
import traceback
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "c3_run_p2.py"
REAL = ["C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock", "C:/Users/Admin/qland/work/refcv7/devbox_gpu.chain_active"]
ME = os.getpid()
MSYS_PID = 861623                       # not a multiple of 4, so never a Windows pid; the refcv8 battery's `pid` is an MSYS pid like this
DEAD_PID = 99999999
TWO_H = 7200
OWN_JOB = "p2-own-table-main"                # a body this helper's stage_dump writes (job = "p2-own-table-{ctx.tag}")


class _Stop(Exception):
    pass


class _Ctx:
    def __init__(self):
        self.status = {}

    def set(self, **kw):
        self.status.update(kw)


def load(text=None):
    m = types.ModuleType("c3_run_p2_under_test")
    m.__file__ = str(SRC)
    exec(compile(text if text is not None else SRC.read_text(encoding="utf-8"), str(SRC), "exec"), m.__dict__)
    return m


def battery(pid=MSYS_PID):
    return {"job": "refcv8-battery-step30000ema", "pid": pid, "win_pid_of_locker": 23276, "host": "H", "acquired": "2026-10-18T14:56:21+0200"}


def own(pid=DEAD_PID, job=OWN_JOB):
    return {"job": job, "pid": pid, "host": "H", "acquired": "2026-10-18T10:00:00+0200"}


def snap(p):
    return p.read_bytes(), os.stat(p).st_mtime_ns


def put(d, body, age_s=TWO_H, raw=None):
    p = Path(d) / "devbox_gpu.lock"
    p.write_text(raw if raw is not None else json.dumps(body), encoding="utf-8")
    t = time.time() - age_s
    os.utime(p, (t, t))
    return p


def run_take_lock(C, d, me="MY-LOCK-BODY"):
    """-> ("TOOK" | "WAITED", lock path).  Everything shared is redirected to the temp dir `d`."""
    lock, mark = str(Path(d) / "devbox_gpu.lock"), str(Path(d) / "devbox_gpu.chain_active")
    assert Path(lock).resolve() != Path(REAL[0]).resolve() and Path(mark).resolve() != Path(REAL[1]).resolve()
    real_time = C.time

    class T:
        def __getattr__(self, k):
            return getattr(real_time, k)

        @staticmethod
        def sleep(s):
            if s == 4:
                return                                        # the settle after taking the lock
            raise _Stop()                                     # the poll: take_lock decided to WAIT

    saved = (C.LOCK, C.MARK, C.gpu_used_mib, C.time)
    C.LOCK, C.MARK, C.gpu_used_mib, C.time = lock, mark, (lambda: 0), T()
    try:
        C.take_lock(me, _Ctx())
        return "TOOK", Path(lock)
    except _Stop:
        return "WAITED", Path(lock)
    finally:
        C.LOCK, C.MARK, C.gpu_used_mib, C.time = saved


def waits_untouched(C, body=None, raw=None, why="", age_s=TWO_H):
    with tempfile.TemporaryDirectory() as d:
        p = put(d, body, age_s=age_s, raw=raw)
        before = snap(p)
        out, _ = run_take_lock(C, d)
        assert out == "WAITED", f"{why}: take_lock TOOK the GPU although a lock that must be waited on exists"
        assert p.exists() and snap(p) == before, f"{why}: the lock was removed or modified"
        assert sorted(x.name for x in Path(d).iterdir()) == ["devbox_gpu.lock"], f"{why}: stray files {sorted(x.name for x in Path(d).iterdir())}"


# ---------------------------------------------------------------------------------------------------------- the checks (each takes the module under test)
def chk_a1_battery_lock_msys_pid_2h_old(C):
    assert not C.pid_alive(MSYS_PID), "precondition: the MSYS pid must be invisible to tasklist"
    waits_untouched(C, battery(), why="battery schema, MSYS pid, 2 h old")


def chk_a2_other_owner_dead_pid_2h_old(C):
    waits_untouched(C, {"pid": DEAD_PID, "owner": "aux_navtest_forward", "t": time.time() - TWO_H}, why="owner/pid/t schema, dead pid, 2 h old")


def chk_a3_other_job_family_same_schema(C):           # same {job,pid,host,acquired} schema, another family's job name: not ours
    waits_untouched(C, own(job="c3-selection-headroom-x"), why="job of another family (c3-selection-headroom-x), dead pid, 2 h old")
    waits_untouched(C, own(job="refe-stage2-x"), why="job of stage2_sft (refe-stage2-x), dead pid, 2 h old")


def chk_a4_no_job_pid_less_non_dict_and_torn_bodies(C):
    waits_untouched(C, {"pid": DEAD_PID}, why="no job key")
    waits_untouched(C, {"job": OWN_JOB}, why="own job marker but no pid")
    waits_untouched(C, ["not", "a", "dict"], why="non-object JSON body")
    waits_untouched(C, raw="{torn", why="torn / non-JSON body")


def chk_b1_ZERO_CONTROL_own_marker_dead_pid_old_is_cleared(C):
    with tempfile.TemporaryDirectory() as d:
        p = put(d, own())
        assert not C.pid_alive(DEAD_PID)
        out, lock = run_take_lock(C, d, me="MY-LOCK-BODY")
        assert out == "TOOK", "zero control: our own dead-pid 2 h old lock was NOT cleared -- the probe cannot see a removal"
        assert lock.read_text(encoding="utf-8") == "MY-LOCK-BODY", "zero control: the lock now present is not the one this call wrote"


def chk_b2_own_marker_dead_pid_but_fresh_is_kept(C):
    waits_untouched(C, own(), why="own marker, dead pid, only 60 s old", age_s=60)


def chk_b3_own_marker_live_pid_is_kept(C):
    waits_untouched(C, own(pid=ME), why="own marker, LIVE pid, 2 h old")


def chk_b4_the_body_stage_dump_writes_carries_the_marker(C):
    src = SRC.read_text(encoding="utf-8")
    m = re.search(r'"job": f"([^"{]*)\{ctx\.tag\}"', src)
    assert m, "stage_dump's lock body no longer has the `\"job\": f\"<prefix>{ctx.tag}\"` form this test reads"
    assert m.group(1) == C.OWN_JOB_PREFIX, f"stage_dump writes job prefix {m.group(1)!r} but take_lock clears {C.OWN_JOB_PREFIX!r}: its own lock would never be cleared"


def chk_c1_release_only_the_body_this_call_wrote(C):
    with tempfile.TemporaryDirectory() as d:
        p = put(d, battery(pid=ME), age_s=0)
        saved = C.LOCK
        C.LOCK = str(p)
        try:
            before = snap(p)
            C.release(json.dumps(own(pid=ME)))
            assert p.exists() and snap(p) == before, "release removed a lock whose body differs from the one this call wrote"
            C.release(p.read_text(encoding="utf-8"))
            assert not p.exists(), "release did not remove the lock whose body it wrote"
        finally:
            C.LOCK = saved


CHECKS = [chk_a1_battery_lock_msys_pid_2h_old, chk_a2_other_owner_dead_pid_2h_old, chk_a3_other_job_family_same_schema,
          chk_a4_no_job_pid_less_non_dict_and_torn_bodies, chk_b1_ZERO_CONTROL_own_marker_dead_pid_old_is_cleared, chk_b2_own_marker_dead_pid_but_fresh_is_kept,
          chk_b3_own_marker_live_pid_is_kept, chk_b4_the_body_stage_dump_writes_carries_the_marker, chk_c1_release_only_the_body_this_call_wrote]

# ----------------------------------------------------------------------------------------------------- (c) mutations on the source text, in-process
GATE = 'isinstance(holder, dict) and str(holder.get("job", "")).startswith(OWN_JOB_PREFIX) and not pid_alive(holder["pid"]) and age > 600'
MUTATIONS = [
    # (name, old text, new text, checks that must go RED)
    ("old_removal_branch_restored__no_job_marker_test", GATE, 'not pid_alive(holder["pid"]) and age > 600',
     [chk_a1_battery_lock_msys_pid_2h_old, chk_a2_other_owner_dead_pid_2h_old, chk_a3_other_job_family_same_schema]),
    ("marker_is_any_job__prefix_ignored", 'str(holder.get("job", "")).startswith(OWN_JOB_PREFIX)', 'str(holder.get("job", "")).startswith("")',
     [chk_a1_battery_lock_msys_pid_2h_old, chk_a3_other_job_family_same_schema]),
    ("age_rule_dropped", 'and age > 600', 'and age > -1', [chk_b2_own_marker_dead_pid_but_fresh_is_kept]),
    ("own_lock_never_cleared__the_zero_control_must_see_it", GATE, 'False and not pid_alive(holder["pid"]) and age > 600',
     [chk_b1_ZERO_CONTROL_own_marker_dead_pid_old_is_cleared]),
]


def main() -> int:
    real_before = [os.stat(p).st_mtime_ns if os.path.exists(p) else None for p in REAL]
    passed = failed = 0
    C = load()
    for c in CHECKS:
        try:
            c(C)
            passed += 1
            print(f"ok   {c.__name__}")
        except Exception as e:                                     # noqa: BLE001
            failed += 1
            print(f"FAIL {c.__name__}: {type(e).__name__}: {e}")
            if not isinstance(e, AssertionError):
                traceback.print_exc()
    src = SRC.read_text(encoding="utf-8")
    for name, old, new, must_go_red in MUTATIONS:
        if src.count(old) != 1:
            failed += 1
            print(f"FAIL mutation {name}: the target text is not present exactly once ({src.count(old)}) -- the test is stale, or the helper still has the old removal branch")
            continue
        M = load(src.replace(old, new))
        for c in must_go_red:
            try:
                c(M)
            except AssertionError as e:
                passed += 1
                print(f"RED  {name} -> {c.__name__}: {str(e)[:100]}")
            except Exception as e:                                 # noqa: BLE001  (a crash is not a clean RED)
                failed += 1
                print(f"FAIL mutation {name}: {c.__name__} CRASHED ({type(e).__name__}: {e}) -- not a clean RED")
            else:
                failed += 1
                print(f"FAIL mutation {name}: {c.__name__} stayed GREEN (the check cannot see this regression)")
    real_after = [os.stat(p).st_mtime_ns if os.path.exists(p) else None for p in REAL]
    for p, b, a in zip(REAL, real_before, real_after):
        if b == a:
            passed += 1
            print(f"ok   real {os.path.basename(p)}: mtime unchanged (stat only)")
        else:
            failed += 1
            print(f"FAIL the REAL {p} changed: {b} -> {a}")
    print(f"ZZTEST_PASS {passed}" if not failed else f"ZZTEST_FAIL passed={passed} failed={failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
