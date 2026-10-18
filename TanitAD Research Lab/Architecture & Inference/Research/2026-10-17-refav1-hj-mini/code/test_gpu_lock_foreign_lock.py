"""LOCK-STEAL-2 regression: gpu_lock.py must clear ONLY a lock that carries its own owner marker (2026-10-18).

The shared dev-box lock `devbox_gpu.lock` is written by TWO schemas: ours {"pid", "owner", ...} and the refcv8 battery's
{"job", "pid" = an MSYS pid, "win_pid_of_locker", "host", "acquired"}.  An MSYS pid is invisible to the Windows pid probe, and the battery schema has
no `t`, so the old rules ("pid not running" / "`t` before the last boot") renamed a LIVE battery lock aside and let the encode job take the GPU.

    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I test_gpu_lock_foreign_lock.py      -> prints ZZTEST_PASS <n> (exit 0) or ZZTEST_FAIL (exit 1)

Every lock is a TEMP file; the real shared lock and chain_active are only `stat`ed (read-only) before and after, never opened.
(c) Mutations are applied IN-PROCESS to the source text (the file on disk is never edited) and must make the named check go RED.
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import time
import traceback
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "gpu_lock.py"
REAL = ["C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock", "C:/Users/Admin/qland/work/refcv7/devbox_gpu.chain_active"]
OWNER = "ra1a"                         # run_encode.sh acquires with `--owner ra1a` and releases with the default
ME = os.getpid()
MSYS_PID = 861623                       # not a multiple of 4, so never a Windows pid; the refcv8 battery's `pid` is an MSYS pid like this
DEAD_PID = 99999999
TWO_H = 7200


def load(text=None):
    m = types.ModuleType("gpu_lock_under_test")
    m.__file__ = str(SRC)
    exec(compile(text if text is not None else SRC.read_text(encoding="utf-8"), str(SRC), "exec"), m.__dict__)
    return m


def foreign_battery(pid=MSYS_PID):
    return {"job": "refcv8-battery-step30000ema", "pid": pid, "win_pid_of_locker": 23276, "host": "H", "acquired": "2026-10-18T14:56:21+0200"}


def put(d, body, age_s=TWO_H):
    p = Path(d) / "devbox_gpu.lock"
    p.write_text(json.dumps(body), encoding="utf-8")
    t = time.time() - age_s
    os.utime(p, (t, t))
    return p


def snap(p):
    return p.read_bytes(), os.stat(p).st_mtime_ns


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def asides(d):
    return sorted(x.name for x in Path(d).iterdir() if x.name != "devbox_gpu.lock")


def acquire_refused_untouched(G, body, why):
    with tempfile.TemporaryDirectory() as d:
        p = put(d, body)
        before = snap(p)
        rc = quiet(G.acquire, p, ME, OWNER, "j", 1.0, 0.2)
        assert rc == 2, f"{why}: acquire returned {rc}, expected 2 (timeout: it must WAIT on a live job's lock, not take it)"
        assert p.exists() and snap(p) == before, f"{why}: the lock was removed or modified"
        assert asides(d) == [], f"{why}: the lock was renamed aside: {asides(d)}"


# ---------------------------------------------------------------------------------------------------------- the checks (each takes the module under test)
def chk_a1_battery_lock_dead_msys_pid_old(G):
    assert not G.pid_alive(MSYS_PID), "precondition: the MSYS pid must be invisible to the Windows probe"
    acquire_refused_untouched(G, foreign_battery(), "battery schema, MSYS pid, 2 h old")


def chk_a2_battery_lock_live_pid_no_t_field(G):      # the refav1-specific bug: `float(info.get("t", 0)) < boot - 5` is True for any lock without a `t`
    acquire_refused_untouched(G, foreign_battery(pid=ME), "battery schema, LIVE pid, no `t` field")


def chk_a3_other_owner_dead_pid_old(G):
    acquire_refused_untouched(G, {"pid": DEAD_PID, "owner": "aux_navtest_forward", "t": time.time() - TWO_H}, "other owner, dead pid, 2 h old")


def chk_a4_release_foreign_lock_with_our_pid(G):     # a foreign body that happens to carry our pid is not ours
    with tempfile.TemporaryDirectory() as d:
        p = put(d, foreign_battery(pid=ME))
        before = snap(p)
        rc = quiet(G.release, p, ME)
        assert rc == 3 and p.exists() and snap(p) == before, f"release removed/touched a foreign lock that carries our pid (rc {rc})"


def chk_a5_status_does_not_call_a_foreign_lock_stale(G):
    with tempfile.TemporaryDirectory() as d:
        p = put(d, foreign_battery())
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            G.main(["status", "--lock", str(p)])
        out = json.loads(buf.getvalue())
        assert out["held"] is True and out["stale"] is None, f"status calls a foreign lock stale: {out['stale']}"


def chk_b1_ZERO_CONTROL_own_marker_dead_pid_old_is_cleared(G):
    with tempfile.TemporaryDirectory() as d:
        body = {"pid": DEAD_PID, "owner": OWNER, "job": "refav1-hj-pooled-encode", "t": time.time() - TWO_H, "acquired": "old"}
        p = put(d, body)
        raw = p.read_bytes()
        rc = quiet(G.acquire, p, ME, OWNER, "j", 3.0, 0.2)
        assert rc == 0, f"zero control: our own dead-pid old lock was NOT cleared (acquire rc {rc}) -- the probe cannot see a removal"
        a = asides(d)
        assert len(a) == 1 and (Path(d) / a[0]).read_bytes() == raw, f"zero control: the stale lock was not renamed aside intact: {a}"
        cur = json.loads(p.read_text(encoding="utf-8"))
        assert cur["pid"] == ME and cur["owner"] == OWNER, f"zero control: the new lock is not ours: {cur}"


def chk_b2_own_marker_pre_boot_live_pid_is_cleared(G):
    with tempfile.TemporaryDirectory() as d:
        p = put(d, {"pid": ME, "owner": OWNER, "job": "x", "t": G.boot_time() - 3600, "acquired": "old"})
        rc = quiet(G.acquire, p, ME, OWNER, "j", 3.0, 0.2)
        assert rc == 0 and len(asides(d)) == 1, f"our own pre-boot lock was not renamed aside (rc {rc}, asides {asides(d)})"


def chk_b3_release_own_lock(G):
    with tempfile.TemporaryDirectory() as d:
        p = put(d, {"pid": ME, "owner": OWNER, "job": "x", "t": time.time()}, age_s=0)
        rc = quiet(G.release, p, ME)
        assert rc == 0 and not p.exists(), f"release did not remove our own lock (rc {rc})"


def chk_c1_live_own_lock_is_waited_on(G):
    with tempfile.TemporaryDirectory() as d:
        body = {"pid": ME, "owner": OWNER, "job": "x", "t": time.time()}
        p = put(d, body, age_s=0)
        before = snap(p)
        rc = quiet(G.acquire, p, ME + 4, OWNER, "j", 1.0, 0.2)
        assert rc == 2 and snap(p) == before and asides(d) == [], f"a live own lock was not waited on (rc {rc})"


CHECKS = [chk_a1_battery_lock_dead_msys_pid_old, chk_a2_battery_lock_live_pid_no_t_field, chk_a3_other_owner_dead_pid_old,
          chk_a4_release_foreign_lock_with_our_pid, chk_a5_status_does_not_call_a_foreign_lock_stale,
          chk_b1_ZERO_CONTROL_own_marker_dead_pid_old_is_cleared, chk_b2_own_marker_pre_boot_live_pid_is_cleared, chk_b3_release_own_lock,
          chk_c1_live_own_lock_is_waited_on]

# ----------------------------------------------------------------------------------------------------- (c) mutations on the source text, in-process
GATE = '    if not isinstance(info, dict) or info.get("owner") != owner:'
REL = '    if info.get("owner") != owner or int(info.get("pid", -1)) != int(pid):'
MUTATIONS = [
    # (name, old text, new text, checks that must go RED)
    ("owner_gate_removed__old_removal_branch_restored", GATE, "    if False:",
     [chk_a1_battery_lock_dead_msys_pid_old, chk_a2_battery_lock_live_pid_no_t_field, chk_a3_other_owner_dead_pid_old, chk_a5_status_does_not_call_a_foreign_lock_stale]),
    ("release_ignores_owner", REL, '    if int(info.get("pid", -1)) != int(pid):', [chk_a4_release_foreign_lock_with_our_pid]),
    ("own_lock_never_cleared__the_zero_control_must_see_it", GATE, "    if True:",
     [chk_b1_ZERO_CONTROL_own_marker_dead_pid_old_is_cleared, chk_b2_own_marker_pre_boot_live_pid_is_cleared]),
]


def main() -> int:
    real_before = [os.stat(p).st_mtime_ns if os.path.exists(p) else None for p in REAL]
    passed = failed = 0
    G = load()
    for c in CHECKS:
        try:
            c(G)
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
            print(f"FAIL mutation {name}: the target line is not present exactly once ({src.count(old)}) -- the test is stale")
            continue
        M = load(src.replace(old, new))
        for c in must_go_red:
            try:
                c(M)
            except AssertionError as e:
                passed += 1
                print(f"RED  {name} -> {c.__name__}: {str(e)[:110]}")
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
