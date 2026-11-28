#!/usr/bin/env python3
"""NAVTEST feature-dump chain (detached; resumable). Stages: wait for markers/chain.DONE|FAILED (navtrain chain) -> prep -> smoke (2 logs + controls)
-> full dump (per-log memgate + per-log GPU lock) -> full controls + index -> navtest.DONE|FAILED -> ALWAYS relaunch the TRK-SCORE chain (WMI, 0x08000200).
Never touches the navtrain chain or any TRK file. Log: F:/safehead_cache/navtest/NAVTEST_CHAIN.log"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, "F:/safehead_cache/code")
import nt_lib as NT  # noqa: E402
import sh_lib as L  # noqa: E402
import sh_env as E  # noqa: E402

say = NT.say
MK = L.wp("markers")
ND = L.wp("navtest")
C = "F:/safehead_cache/code/"
START = time.time()
GIVE_UP = 36 * 3600
os.environ["SH_NEED_MB"] = "800"   # MM 2026-11-27 13:50Z: 2000 self-blocked (the dump holds ~2.4 GB itself; Chrome 15.6 GB); commit floor 8 GB stays
os.environ["SH_NEED_COMMIT_GB"] = "4"; os.environ["SH_GATE_SAMPLES"] = "2"; os.environ["SH_GATE_EVERY_S"] = "10"   # MM 2026-11-27 15:15Z


def run(cmd, env, log):
    with open(log, "a", encoding="utf-8") as f:
        return subprocess.call(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)


def chain_alive():
    q = "(Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | Where-Object { $_.CommandLine -match 'sh_run.py' } | Measure-Object).Count"
    out = subprocess.run(["powershell", "-NoProfile", "-Command", q], capture_output=True, text=True).stdout.strip()
    return int(out or 0) > 0


def relaunch_trk():
    ps = ("$su = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{CreateFlags=[uint32]0x08000200}; "
          "$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd.exe /c E:\\Projects\\TanitAD-artifacts\\trkscore\\run_trk_chain.cmd'; ProcessStartupInformation=$su}; "
          "'ReturnValue=' + $r.ReturnValue + ' ProcessId=' + $r.ProcessId")
    o = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)
    say("TRK-SCORE relaunch: " + o.stdout.strip() + " " + o.stderr.strip()[:300])


def finish(ok, why):
    open(f"{MK}/navtest.{'DONE' if ok else 'FAILED'}", "w").write(time.strftime("%FT%TZ") + " " + why)
    say(f"navtest {'DONE' if ok else 'FAILED'}: {why}")
    relaunch_trk()
    sys.exit(0 if ok else 1)


def newer_failed():
    p = f"{MK}/chain.FAILED"
    return os.path.exists(p) and os.path.getmtime(p) > START


def main():
    say(f"navtest chain start pid {os.getpid()}")
    os.makedirs(ND, exist_ok=True)
    if os.path.exists(f"{MK}/navtest.DONE"):
        say("navtest.DONE already present: only relaunching TRK")
        relaunch_trk()
        return
    dead_since = None
    while True:                                           # stage 1: wait for the navtrain chain
        if os.path.exists(f"{MK}/chain.DONE"):
            say("chain.DONE seen")
            break
        if newer_failed():
            say("chain.FAILED (newer than my start) seen: proceeding, navtest does not depend on navtrain")
            break
        if time.time() - START > GIVE_UP:
            finish(False, "give-up waiting for the navtrain chain")
        if not chain_alive():
            dead_since = dead_since or time.time()
            if time.time() - dead_since > 1800:
                say("no sh_run.py process for 30 min and no marker: proceeding")
                break
        else:
            dead_since = None
        time.sleep(60)
    rc = run([L.PY_T, C + "nt_prep.py"], os.environ.copy(), ND + "/prep.log")
    if rc:
        finish(False, f"prep rc {rc}")
    env = E.denv()
    job = ND + "/plan/job_navtest.json"
    sm = ND + "/ctrl_smoke.json"
    if not (os.path.exists(sm) and json.load(open(sm)).get("overall_pass")):   # stage 2: smoke = first 2 logs of the full job (the full run skips their files)
        j = json.load(open(job))
        j["logs"] = j["logs"][:2]
        json.dump(j, open(ND + "/plan/job_smoke.json", "w"))
        say("smoke: 2 logs")
        rc = run([L.PY_D, C + "sh_dump_nt.py", "--job", ND + "/plan/job_smoke.json", "--ckpt", L.C1_CKPT, "--expect-sha256", L.C1_SHA, "--out-dir", ND,
                  "--jobname", "refe-safehead-navtest"], env, ND + "/dump.log")
        if rc:
            finish(False, f"smoke dump rc {rc}")
        rc = run([L.PY_T, C + "nt_ctrl.py", "smoke", "--n", "2"], os.environ.copy(), ND + "/ctrl.log")
        say(f"smoke controls rc {rc} (navtest/ctrl_smoke.json)")
        if rc:
            finish(False, "smoke controls FAILED")
    say("full dump")                                      # stage 3
    rc = run([L.PY_D, C + "sh_dump_nt.py", "--job", job, "--ckpt", L.C1_CKPT, "--expect-sha256", L.C1_SHA, "--out-dir", ND, "--jobname", "refe-safehead-navtest"], env, ND + "/dump.log")
    if rc:
        finish(False, f"full dump rc {rc}")
    rc = run([L.PY_T, C + "nt_ctrl.py", "full"], os.environ.copy(), ND + "/ctrl.log")
    finish(rc == 0, f"full controls rc {rc} (navtest/ctrl_full.json)")


try:
    main()
except SystemExit:
    raise
except Exception as e:  # noqa: BLE001
    say(f"EXCEPTION {type(e).__name__}: {e}")
    open(f"{MK}/navtest.FAILED", "w").write(f"exception {e}")
    relaunch_trk()
    sys.exit(1)
