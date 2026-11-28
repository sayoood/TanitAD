"""LOCK-STEAL-2 regression: p2s1_autoread.Job.lock_state must remove ONLY a lock carrying its own marker (2026-10-18).

The shared dev-box lock `devbox_gpu.lock` is written by TWO schemas.  This script never takes the lock itself; the only lock it may clear is the one the stage 2 it launches writes
(stage2_sft.py: {"job": "refe-stage2-<name>", "pid", "host", "acquired"}).  The refcv8 battery writes {"job", "pid" = an MSYS pid, "win_pid_of_locker", "host", "acquired"}; an MSYS pid is
invisible to `tasklist`, so the old rule ("holder pid not alive and the lock older than 600 s -> unlink") removed a LIVE battery lock (MEASURED 2026-10-18: pid 861623, 2 h old).

    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I test_p2s1_autoread_foreign_lock.py      -> prints ZZTEST_PASS <n> (exit 0) or ZZTEST_FAIL (exit 1)

Every lock is a TEMP file (A.LOCK / A.MARK are redirected); the real shared lock and chain_active are only `stat`ed (read-only), never opened.  `pid_alive` is the REAL one (tasklist).
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
SRC = HERE / "p2s1_autoread.py"
STAGE2 = HERE.parent.parent / "eval" / "stage2_sft.py"                    # raw/<date>-p2s1-read -> the refe-plan package root -> eval/
REAL = ["C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock", "C:/Users/Admin/qland/work/refcv7/devbox_gpu.chain_active"]
ME = os.getpid()
MSYS_PID = 861623                       # not a multiple of 4, so never a Windows pid; the refcv8 battery's `pid` is an MSYS pid like this
DEAD_PID = 99999999
TWO_H = 7200
NAME = "n"
OWN_JOB = f"refe-stage2-{NAME}"
_N = [0]


def load(text=None):
    _N[0] += 1
    m = types.ModuleType(f"p2s1_autoread_under_test_{_N[0]}")
    m.__file__ = str(SRC)
    sys.modules[m.__name__] = m
    exec(compile(text if text is not None else SRC.read_text(encoding="utf-8"), str(SRC), "exec"), m.__dict__)
    return m


def battery(pid=MSYS_PID):
    return {"job": "refcv8-battery-step30000ema", "pid": pid, "win_pid_of_locker": 23276, "host": "H", "acquired": "2026-10-18T14:56:21+0200"}


def own(pid=DEAD_PID, job=OWN_JOB):
    return {"job": job, "pid": pid, "host": "H", "acquired": "2026-10-18T10:00:00+0200"}


def snap(p):
    return p.read_bytes(), os.stat(p).st_mtime_ns


def lock_state_of(A, body=None, raw=None, age_s=TWO_H):
    """-> (lock_state() result, lock path, bytes+mtime before, dir listing after).  A.LOCK / A.MARK point into a temp dir for the duration of the call."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        p = d / "devbox_gpu.lock"
        p.write_text(raw if raw is not None else json.dumps(body), encoding="utf-8")
        t = time.time() - age_s
        os.utime(p, (t, t))
        before = snap(p)
        saved = (A.LOCK, A.MARK)
        A.LOCK, A.MARK = str(p), str(d / "devbox_gpu.chain_active")
        assert Path(A.LOCK).resolve() != Path(REAL[0]).resolve()
        try:
            job = A.Job({"out": str(d / "o"), "repo": "r", "name": NAME}, None, sleep=lambda s: None)
            res = job.lock_state()
        finally:
            A.LOCK, A.MARK = saved
        present = p.exists()
        unchanged = present and snap(p) == before
        return res, present, unchanged


def held_untouched(A, why, body=None, raw=None, age_s=TWO_H):
    res, present, unchanged = lock_state_of(A, body=body, raw=raw, age_s=age_s)
    assert present and unchanged, f"{why}: the lock was removed or modified (lock_state returned {res!r})"
    assert res is not None, f"{why}: lock_state reported the GPU lock FREE although a lock it must not remove exists"


# ---------------------------------------------------------------------------------------------------------- the checks (each takes the module under test)
def chk_a1_battery_lock_msys_pid_2h_old(A):
    assert not A.pid_alive(MSYS_PID), "precondition: the MSYS pid must be invisible to tasklist"
    held_untouched(A, "battery schema, MSYS pid, 2 h old", body=battery())


def chk_a2_other_owner_dead_pid_2h_old(A):
    held_untouched(A, "owner/pid/t schema, dead pid, 2 h old", body={"pid": DEAD_PID, "owner": "aux_navtest_forward", "t": time.time() - TWO_H})


def chk_a3_other_stage2_name_or_other_family(A):
    held_untouched(A, "stage2_sft lock of ANOTHER name, dead pid, 2 h old", body=own(job="refe-stage2-other"))
    held_untouched(A, "c3 family job, dead pid, 2 h old", body=own(job="c3-selection-headroom-main"))
    held_untouched(A, "p2 family job, dead pid, 2 h old", body=own(job="p2-own-table-main"))


def chk_a4_no_job_pid_less_non_dict_and_torn_bodies(A):
    held_untouched(A, "no job key", body={"pid": DEAD_PID})
    held_untouched(A, "own job marker but no pid", body={"job": OWN_JOB})
    try:
        held_untouched(A, "non-object JSON body", body=["not", "a", "dict"])
    except (TypeError, AttributeError) as e:                                 # pragma: no cover  (a crash is fail-closed too, but it is not a pass)
        raise AssertionError(f"non-object JSON body crashed lock_state: {type(e).__name__}: {e}")
    res, present, unchanged = lock_state_of(A, raw="{torn")
    assert present and unchanged and res == "lock unreadable", f"torn body: {res!r}"


def chk_b1_ZERO_CONTROL_own_marker_dead_pid_old_is_removed(A):
    assert not A.pid_alive(DEAD_PID)
    res, present, _ = lock_state_of(A, body=own())
    assert res is None and not present, f"zero control: our own dead-pid 2 h old lock was NOT removed (lock_state {res!r}, present {present}) -- the probe cannot see a removal"


def chk_b2_own_marker_dead_pid_but_fresh_is_kept(A):
    held_untouched(A, "own marker, dead pid, only 60 s old", body=own(), age_s=60)


def chk_b3_own_marker_live_pid_is_kept(A):
    held_untouched(A, "own marker, LIVE pid, 2 h old", body=own(pid=ME))


def chk_b4_marker_matches_what_stage2_sft_writes(A):
    s2 = STAGE2.read_text(encoding="utf-8")
    m = re.search(r'"job": f"([^"{]*)\{a\.name\}"', s2)
    assert m, f"{STAGE2}: stage2_sft's lock body no longer has the `\"job\": f\"<prefix>{{a.name}}\"` form (the marker would never match)"
    src = SRC.read_text(encoding="utf-8")
    assert f'f"{m.group(1)}{{self.c[\'name\']}}"' in src, f"lock_state's marker is not f\"{m.group(1)}{{name}}\": it would never clear the lock stage2_sft writes"
    assert f"{m.group(1)}{NAME}" == OWN_JOB


CHECKS = [chk_a1_battery_lock_msys_pid_2h_old, chk_a2_other_owner_dead_pid_2h_old, chk_a3_other_stage2_name_or_other_family,
          chk_a4_no_job_pid_less_non_dict_and_torn_bodies, chk_b1_ZERO_CONTROL_own_marker_dead_pid_old_is_removed, chk_b2_own_marker_dead_pid_but_fresh_is_kept,
          chk_b3_own_marker_live_pid_is_kept, chk_b4_marker_matches_what_stage2_sft_writes]

# ----------------------------------------------------------------------------------------------------- (c) mutations on the source text, in-process
GATE = '''isinstance(holder, dict) and holder.get("job") == f"refe-stage2-{self.c['name']}" and "pid" in holder and not pid_alive(holder["pid"]) and age > 600'''
MUTATIONS = [
    # (name, old text, new text, checks that must go RED)
    ("old_removal_branch_restored__no_marker_test", GATE, '"pid" in holder and not pid_alive(holder["pid"]) and age > 600',
     [chk_a1_battery_lock_msys_pid_2h_old, chk_a2_other_owner_dead_pid_2h_old, chk_a3_other_stage2_name_or_other_family]),
    ("marker_is_any_job", '''holder.get("job") == f"refe-stage2-{self.c['name']}"''', 'holder.get("job")',
     [chk_a1_battery_lock_msys_pid_2h_old, chk_a3_other_stage2_name_or_other_family]),
    ("age_rule_dropped", 'and age > 600', 'and age > -1', [chk_b2_own_marker_dead_pid_but_fresh_is_kept]),
    ("own_lock_never_removed__the_zero_control_must_see_it", GATE, 'False and "pid" in holder and not pid_alive(holder["pid"]) and age > 600',
     [chk_b1_ZERO_CONTROL_own_marker_dead_pid_old_is_removed]),
]


def main() -> int:
    real_before = [os.stat(p).st_mtime_ns if os.path.exists(p) else None for p in REAL]
    passed = failed = 0
    A = load()
    for c in CHECKS:
        try:
            c(A)
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
