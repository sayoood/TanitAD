"""Control-flow tests for chain_battery_r8.sh + chain_helpers_r8.py -- a SANDBOX dry run with stub scripts (no GPU, no real lock).

Run:  python -m pytest -q -p no:cacheprovider test_chain_battery_r8.py      (needs Git-for-Windows `sh`)

The chain's paths are overridable by R8_* env vars for exactly this purpose.  The stubs record every call into `calls.log`, so
the ORDER of events (lock before battery, release after the end, pass 1 on the standalone clock WITH baselines, pass 2 on the
corrected clock with --no-baselines --os-only-all) is asserted on the artifact, never on an exit code.
Each guard has a REGRESSION arm: the skip paths (no g0.json / M1 undetected / other checkpoint), a dead G0 shell, and a failing
pass (the lock must still be released by the EXIT trap).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chain_helpers_r8 as H  # noqa: E402

SH = shutil.which("sh") or "C:/Program Files/Git/bin/sh.exe"
needs_sh = pytest.mark.skipif(not Path(SH).exists() and not shutil.which("sh"), reason="no sh")
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"


def msys(p: Path) -> str:
    s = str(p).replace("\\", "/")
    return "/" + s[0].lower() + s[2:] if len(s) > 1 and s[1] == ":" else s


# ------------------------------------------------------------------------------------------ helpers (pure)
def write_log(p: Path, lines):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_g0state_cases(tmp_path):
    log = tmp_path / "c.log"
    assert H.g0state(str(log), 5000) == "nostart:-:-"
    write_log(log, ["ZZR8CHAINSTART5000ZZ 2026-10-15T21:46:36+0200 pid 2699", "ZZR8PRECONDOKZZ ckpt x", "[gpu_lock] WAIT job=x"])
    assert H.g0state(str(log), 5000) == "running:PRECONDOK:2699"
    write_log(log, ["ZZR8CHAINSTART5000ZZ t pid 2699"])
    assert H.g0state(str(log), 5000) == "running:CHAINSTART5000:2699"
    write_log(log, ["ZZR8CHAINSTART5000ZZ t pid 2699", "ZZR8LOCKEDZZ t", "ZZR8G0DONEZZ t"])
    assert H.g0state(str(log), 5000) == "running:G0DONE:2699"
    write_log(log, ["ZZR8CHAINSTART5000ZZ t pid 2699", "ZZR8LOCKEDZZ t", "ZZR8G0DONEZZ t", "ZZR8BANKEDZZ t", "ZZR8CHAINENDZZ t"])
    assert H.g0state(str(log), 5000) == "ended:CHAINEND:2699"
    write_log(log, ["ZZR8CHAINSTART5000ZZ t pid 11", "ZZR8PRECONDFAILZZ ckpt md5 x"])
    assert H.g0state(str(log), 5000) == "ended:PRECONDFAIL:11"
    for term in ("NOLOCK", "NOCOMMIT", "PROBEFAIL"):
        write_log(log, ["ZZR8CHAINSTART5000ZZ t pid 11", f"ZZR8{term}ZZ t"])
        assert H.g0state(str(log), 5000) == f"ended:{term}:11"
    # a RELAUNCH: the terminal marker of the OLD run must not end the new one
    write_log(log, ["ZZR8CHAINSTART5000ZZ t pid 11", "ZZR8PROBEFAILZZ t", "ZZR8CHAINSTART5000ZZ t pid 22", "ZZR8PRECONDOKZZ"])
    assert H.g0state(str(log), 5000) == "running:PRECONDOK:22"
    # another step's log never ends this one
    write_log(log, ["ZZR8CHAINSTART10000ZZ t pid 5", "ZZR8CHAINENDZZ"])
    assert H.g0state(str(log), 5000) == "nostart:-:-"


def test_g0gate_cases(tmp_path):
    ck = tmp_path / "ckpt.pt"
    ck.write_bytes(b"checkpoint-bytes")
    md5 = hashlib.md5(b"checkpoint-bytes").hexdigest()
    g0 = tmp_path / "g0.json"
    out = tmp_path / "gate.json"
    assert H.g0gate(str(g0), str(ck), str(out)).startswith("skip:g0.json is missing")                       # REGRESSION arm
    g0.write_text("", encoding="utf-8")
    assert H.g0gate(str(g0), str(ck), str(out)).startswith("skip:g0.json is missing or empty")
    g0.write_text("{not json", encoding="utf-8")
    assert H.g0gate(str(g0), str(ck), str(out)).startswith("skip:g0.json unreadable")

    def put(m1, g0v="FAIL", md=md5):
        g0.write_text(json.dumps({"ckpt_md5": md, "verdict": {"G0": g0v, "amendment": "A7",
                                                              "mutation_detection": {"m1": {"detected": m1}}}}), encoding="utf-8")
    put(True, "FAIL", md="0" * 32)
    assert H.g0gate(str(g0), str(ck), str(out)).startswith("skip:the g0.json is for another checkpoint")
    put(False)
    assert H.g0gate(str(g0), str(ck), str(out)).startswith("skip:M1_detected is False")
    put(None)
    assert H.g0gate(str(g0), str(ck), str(out)).startswith("skip:M1_detected is None")
    put(True, "FAIL")
    assert H.g0gate(str(g0), str(ck), str(out)) == "ok:FAIL"                                                # early read proceeds, stamped later
    assert H.g0gate(str(g0), str(ck), str(out), require_pass=True).startswith("skip:G0 is FAIL")
    put(True, "PASS")
    assert H.g0gate(str(g0), str(ck), str(out), require_pass=True) == "ok:PASS"
    assert json.load(open(out))["result"] == "ok:PASS" and json.load(open(out))["M1_detected"] is True


# ------------------------------------------------------------------------------------------ the chain, in a sandbox
STUB_LOCK = r'''
import json, os, sys
a = sys.argv[1:]
log = os.environ["STUB_CALLS"]
open(log, "a").write("gpu_lock %s job=%s\n" % (a[0], a[a.index("--job") + 1]))
if a[0] == "acquire":
    out = a[a.index("--out") + 1]
    json.dump({"ok": True}, open(out, "w"))
'''
STUB_COMMIT = r'''
import json, sys
a = sys.argv[1:]
json.dump({"met": True}, open(a[a.index("--out") + 1], "w"))
'''
STUB_NOP = r'''
import sys
'''
STUB_BATTERY = r'''
import json, os, sys, time
a = sys.argv[1:]
def v(f): return a[a.index(f) + 1]
tag, root = v("--tag"), v("--out-root")
open(os.environ["STUB_CALLS"], "a").write("run_battery tag=%s nopose=%s nobase=%s osall=%s skiproll=%s g0nb=%s lockheld=%s\n" % (
    tag, os.environ.get("R8_NO_POSE_SYNC", "-"), "--no-baselines" in a, "--os-only-all" in a, "--skip-roll" in a,
    "--g0-nonblocking" in a, "--lock-held" in a))
os.makedirs(os.path.join(root, tag), exist_ok=True)
if os.environ.get("STUB_FAIL_PASS") and tag.endswith(os.environ["STUB_FAIL_PASS"]):
    sys.exit(1)
json.dump({"tag": tag, "bars": [{"id": "BAR-R7-1", "verdict": "x"}], "finished": "t", "r8": {"pose_sync_policy": os.environ.get("R8_NO_POSE_SYNC", "ON")},
           "stages": {}}, open(os.path.join(root, tag, "battery_summary.json"), "w"))
'''
STUB_PAIR = r'''
import json, os, sys
a = sys.argv[1:]
def v(f): return a[a.index(f) + 1]
open(os.environ["STUB_CALLS"], "a").write("pair_read clock=%s root=%s\n" % (v("--clock"), os.path.basename(v("--root"))))
json.dump({"clock": v("--clock"), "g0": {"G0": "FAIL"}, "bars": {"BAR-R8-E1": {"verdict": "FAILED"}, "BAR-R8-NR": {"verdict": "x", "per_family": {}}}},
          open(v("--out-json"), "w"))
open(v("--out-md"), "w").write("x")
'''


STUB_PERC = r'''
import json, os, sys
a = sys.argv[1:]
out = a[a.index("--out") + 1]
open(os.environ["STUB_CALLS"], "a").write("perc_pass out=%s\n" % os.path.basename(os.path.dirname(out)))
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump({"ok": True}, open(out + ".json", "w"))
'''


def build_sandbox(tmp: Path, *, g0_log, g0_json=None, m1=True, ckpt_md5_ok=True):
    pkg = tmp / "pkg"
    (pkg / "code").mkdir(parents=True)
    (pkg / "raw").mkdir()
    for f in ("chain_helpers_r8.py", "bank_battery_r8.py", "chain_battery_r8.sh"):
        shutil.copy(HERE / f, pkg / "code" / f)
    (pkg / "code" / "run_battery_r8.py").write_text(STUB_BATTERY)
    (pkg / "code" / "pair_read_r8.py").write_text(STUB_PAIR)
    bat = tmp / "bat"
    bat.mkdir()
    (bat / "gpu_lock.py").write_text(STUB_LOCK)
    (bat / "wait_commit.py").write_text(STUB_COMMIT)
    (bat / "chain_marker.py").write_text(STUB_NOP)
    (bat / "gpu_watch.py").write_text(STUB_NOP)
    kit = tmp / "kit"
    (kit / "ckpt").mkdir(parents=True)
    (kit / "chain").mkdir()
    (kit / "battery" / "step5000").mkdir(parents=True)
    (kit / "ckpt" / "ckpt_5000.pt").write_bytes(b"checkpoint-bytes")
    (kit / "ckpt" / "config.json").write_text("{}")
    write_log(kit / "chain" / "chain_step5000.log", g0_log)
    if g0_json is not False:
        md5 = hashlib.md5(b"checkpoint-bytes").hexdigest() if ckpt_md5_ok else "f" * 32
        (kit / "battery" / "step5000" / "g0.json").write_text(json.dumps(
            {"ckpt_md5": md5, "verdict": {"G0": "FAIL", "amendment": "A7", "mutation_detection": {"m1": {"detected": m1}}}}))
    env = dict(os.environ, R8_PKGW=str(pkg).replace("\\", "/"), R8_PKGL=msys(pkg), R8_BAT=str(bat).replace("\\", "/"),
               R8_KITW=str(kit).replace("\\", "/"), R8_KIT=msys(kit), R8_PY=msys(Path(PY)), R8_POLL_S="1",
               STUB_CALLS=str(tmp / "calls.log"))
    env.pop("R8_NO_POSE_SYNC", None)
    return pkg, kit, env


def run_chain(pkg, env, timeout=120):
    return subprocess.run([SH, str(pkg / "code" / "chain_battery_r8.sh"), "5000"], env=env, timeout=timeout,
                          capture_output=True, text=True)


def markers(kit: Path):
    p = kit / "chain" / "battery_step5000.log"
    return [ln.split()[0] for ln in p.read_text(encoding="utf-8", errors="replace").splitlines() if ln.startswith("ZZR8")]


G0_ENDED = ["ZZR8CHAINSTART5000ZZ t pid 11", "ZZR8G0DONEZZ t", "ZZR8CHAINENDZZ t"]


@needs_sh
def test_chain_skip_paths_never_touch_the_lock(tmp_path):
    for name, kw, why in (("nog0", {"g0_json": False}, "g0.json is missing"),
                          ("nom1", {"m1": False}, "M1_detected is False"),
                          ("otherck", {"ckpt_md5_ok": False}, "another checkpoint")):
        pkg, kit, env = build_sandbox(tmp_path / name, g0_log=G0_ENDED, **kw)
        run_chain(pkg, env)
        mk = markers(kit)
        assert "ZZR8BATTSKIPZZ" in mk and "ZZR8BATTLOCKEDZZ" not in mk and "ZZR8BATTDONEZZ" not in mk, (name, mk)
        assert not (tmp_path / name / "calls.log").exists(), "a skipped chain must not call the lock or the battery"
        summ = json.load(open(pkg / "raw" / "step5000" / "BATTERY_SUMMARY_step5000.json"))
        assert summ["status"] == "SKIPPED" and why in summ["reason"]


@needs_sh
def test_chain_happy_path_order_and_clock_arguments(tmp_path):
    pkg, kit, env = build_sandbox(tmp_path, g0_log=G0_ENDED)
    run_chain(pkg, env)
    mk = markers(kit)
    want = ["ZZR8BATTSTARTZZ", "ZZR8BATTG0ENDEDZZ", "ZZR8BATTGATEOKZZ", "ZZR8BATTLOCKEDZZ", "ZZR8BATTPASSDONEZZ", "ZZR8BATTPAIREDZZ",
            "ZZR8BATTDONEZZ", "ZZR8BATTPASSDONEZZ", "ZZR8BATTCORRDONEZZ", "ZZR8BATTCHAINENDZZ"]
    assert mk == want, mk
    calls = (tmp_path / "calls.log").read_text().splitlines()
    assert calls[0] == "gpu_lock acquire job=refcv8-battery-step5000"                                      # lock BEFORE any battery
    assert calls[-1] == "gpu_lock release job=refcv8-battery-step5000"                                     # released at the end (EXIT trap)
    unc = next(c for c in calls if "tag=step5000_unc" in c)
    cor = next(c for c in calls if "tag=step5000_corr" in c)
    assert "nopose=1 nobase=False osall=False skiproll=True g0nb=True lockheld=True" in unc               # standalone clock, WITH baselines
    assert "nopose=- nobase=True osall=True skiproll=True g0nb=True lockheld=True" in cor                  # corrected clock, NO baselines
    assert "pair_read clock=standalone root=step5000_unc" in calls and "pair_read clock=corrected root=step5000_corr" in calls
    assert calls.index(unc) < calls.index("pair_read clock=standalone root=step5000_unc") < calls.index(cor)
    summ = json.load(open(pkg / "raw" / "step5000" / "BATTERY_SUMMARY_step5000.json"))
    assert summ["passes"]["unc"]["status"] == "COMPLETE" and summ["passes"]["corr"]["status"] == "COMPLETE"
    assert summ["passes"]["unc"]["paired_read"]["clock"] == "standalone"
    assert summ["g0_gate"]["result"] == "ok:FAIL"                                                          # provisional, proceeded


@needs_sh
def test_chain_corrected_pass_can_be_disabled(tmp_path):
    pkg, kit, env = build_sandbox(tmp_path, g0_log=G0_ENDED)
    env["R8_CORRECTED"] = "0"
    run_chain(pkg, env)
    assert "ZZR8BATTDONEZZ" in markers(kit) and "ZZR8BATTCORRDONEZZ" not in markers(kit)
    assert not any("tag=step5000_corr" in c for c in (tmp_path / "calls.log").read_text().splitlines())


@needs_sh
def test_chain_optional_perception_stage_is_off_by_default(tmp_path):
    pkg, kit, env = build_sandbox(tmp_path / "off", g0_log=G0_ENDED)
    (pkg / "code" / "perc_pass_r8.py").write_text(STUB_PERC)
    run_chain(pkg, env)
    assert "ZZR8BATTPERCDONEZZ" not in markers(kit) and not any("perc_pass" in c for c in (tmp_path / "off" / "calls.log").read_text().splitlines())
    pkg, kit, env = build_sandbox(tmp_path / "on", g0_log=G0_ENDED)
    (pkg / "code" / "perc_pass_r8.py").write_text(STUB_PERC)
    env["R8_PERC"] = "1"
    run_chain(pkg, env)
    mk = markers(kit)
    assert "ZZR8BATTPERCDONEZZ" in mk and mk[-1] == "ZZR8BATTCHAINENDZZ" and mk.index("ZZR8BATTPERCDONEZZ") > mk.index("ZZR8BATTCORRDONEZZ")


@needs_sh
def test_REGRESSION_failing_pass_retries_once_then_fails_and_still_releases_the_lock(tmp_path):
    pkg, kit, env = build_sandbox(tmp_path, g0_log=G0_ENDED)
    env["STUB_FAIL_PASS"] = "unc"
    run_chain(pkg, env)
    mk = markers(kit)
    assert "ZZR8BATTRETRYZZ" in mk and "ZZR8BATTFAILZZ" in mk and "ZZR8BATTDONEZZ" not in mk and "ZZR8BATTCORRDONEZZ" not in mk
    calls = (tmp_path / "calls.log").read_text().splitlines()
    assert sum("tag=step5000_unc" in c for c in calls) == 2                                               # exactly one retry
    assert not any("tag=step5000_corr" in c for c in calls)                                               # no second pass after a failed first
    assert calls[-1] == "gpu_lock release job=refcv8-battery-step5000"                                     # the trap released it


@needs_sh
def test_chain_waits_for_the_g0_terminal_marker(tmp_path):
    pkg, kit, env = build_sandbox(tmp_path, g0_log=["ZZR8CHAINSTART5000ZZ t pid -", "ZZR8PRECONDOKZZ"])
    p = subprocess.Popen([SH, str(pkg / "code" / "chain_battery_r8.sh"), "5000"], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True)
    time.sleep(4)
    assert "ZZR8BATTGATEOKZZ" not in markers(kit) and p.poll() is None                                     # still waiting
    assert not (tmp_path / "calls.log").exists()
    with open(kit / "chain" / "chain_step5000.log", "a") as f:
        f.write("ZZR8G0DONEZZ t\nZZR8CHAINENDZZ t\n")
    p.wait(timeout=120)
    assert "ZZR8BATTDONEZZ" in markers(kit)


@needs_sh
def test_chain_declares_a_dead_g0_shell_but_still_gates_on_the_artifact(tmp_path):
    pkg, kit, env = build_sandbox(tmp_path, g0_log=["ZZR8CHAINSTART5000ZZ t pid 999999", "ZZR8PRECONDOKZZ"], g0_json=False)
    run_chain(pkg, env)
    mk = markers(kit)
    assert "ZZR8BATTG0DEADZZ" in mk and "ZZR8BATTSKIPZZ" in mk and "ZZR8BATTLOCKEDZZ" not in mk
