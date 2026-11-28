#!/usr/bin/env python3
"""One-shot patcher (SFT-4R readiness, 2026-10-16): the line that started the runaway, and a no-spawn proof on the REAL profile.  usage: patch_test_sft4r_gate.py <refe dir>

1. test_sft4r_gate.py section 7, the `run` CLI call: it ran `launch_gate.py run --profile refe_sft4r` with EVERY check (no --only).  The real refe_sft4r suite contains this very file, so G-SUITE ran
   this file, which ran `run`, which ran G-SUITE ... (~80 python processes).  The call now excludes G-SUITE (`--only`, the pattern test_launch_gate.py R12 / A8 use) and writes its gate key under the
   test's own temp dir (the default key_file is created on first use -- a side effect on whatever box ran the test).
2. a new section 8 proves, on the real profile and WITHOUT spawning (subprocess.run is replaced by a function that raises), that check_suite refuses when this file is already running above it.
"""
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "test_sft4r_gate.py"
s = open(p, encoding="utf-8", newline="").read()
assert "\r\n" not in s

old = '''    rc = cli("run", "--profile", "refe_sft4r", "--args", " ".join(base), "--pkg", str(work / "nopkg"), "--out-dir", str(work / "o"))
    ok(rc.returncode != 0 and "mints for ONE named arm" not in (rc.stdout + rc.stderr), (rc.stdout + rc.stderr)[-300:])             # minting needs no --arm-name (it fails later, on the missing package/preflight)
'''
assert s.count(old) == 1
new = '''    # G-SUITE is EXCLUDED here (--only): the REAL refe_sft4r suite contains THIS FILE, so a `run` that reached G-SUITE would run this file, which would run `run`, which would run G-SUITE ... (the 2026-10-16
    # runaway: ~80 python processes, RAM 8.6 GB).  refe_sft4r has no arm table, so -- unlike refe_rls1 / refe_aux_s1 -- nothing refuses an unnamed arm before the checks run.  Section 8 + test_suite_nesting.py
    # pin the engine guard; this call keeps the property it was written for.  The gate key goes under the test's own dir: the default key_file is CREATED on first use.
    rc = cli("run", "--profile", "refe_sft4r", "--args", " ".join(base), "--pkg", str(work / "nopkg"), "--out-dir", str(work / "o"), "--key-file", str(work / "k2" / "gate.key"),
             "--only", "G-HYG,G-DVB,G-PREFLIGHT")
    ok(rc.returncode != 0 and "mints for ONE named arm" not in (rc.stdout + rc.stderr), (rc.stdout + rc.stderr)[-300:])             # minting needs no --arm-name (it fails later, on the missing package/preflight)
    ok("G-SUITE" not in "".join(ln for ln in rc.stdout.splitlines(True) if ln.startswith("  G-")), rc.stdout[-300:])                  # no G-SUITE row was run
    # ======================================================================== 8. the real suite cannot re-enter itself: check_suite refuses to nest (the full proof, with mutations, is test_suite_nesting.py)
    me = {"script": "test_sft4r_gate.py", "token": "ZZTEST_PASS"}
    ok(me in L.SFT4R_SUITE and {"script": "test_suite_nesting.py", "token": "ZZTEST_PASS"} in L.SFT4R_SUITE)                       # the file the runaway came from IS an entry of the suite it used to run
    spawned = []
    real_run, real_env = C.subprocess.run, os.environ.get(C.SUITE_NEST_ENV)

    def never_spawn(*a, **k):
        spawned.append(a)
        raise AssertionError("check_suite SPAWNED a process although the suite contains the script that is running it")
    C.subprocess.run = never_spawn
    os.environ[C.SUITE_NEST_ENV] = json.dumps([C.suite_entry_id(me)])                                                                 # exactly what the real G-SUITE sets for this file's own process
    try:
        ev = C.check_suite(PROF)
    finally:
        C.subprocess.run = real_run
        if real_env is None:
            os.environ.pop(C.SUITE_NEST_ENV, None)
        else:
            os.environ[C.SUITE_NEST_ENV] = real_env
    ok(spawned == [] and ev["status"] == "FAIL" and any("refuses to nest" in r and "test_sft4r_gate.py" in r for r in ev["reasons"]), (spawned, ev["reasons"]))
'''
s = s.replace(old, new)
open(p, "w", encoding="utf-8", newline="").write(s)
print("patched", p.name)
