"""Control-flow tests for queue_k2k4_r8.sh -- a SANDBOX dry run with stub scripts (no GPU, no real lock, no model).  NEW file.

    python -m pytest -q -p no:cacheprovider test_queue_k2k4_r8.py      (needs Git-for-Windows `sh`)

The stubs record every call into `calls.log`, so the ORDER of events is asserted on the artifact, never an exit code: the provenance check and the plan
come BEFORE the lock, both rolls run inside the lock with the right mode / clock / PYTHONPATH, the compare runs with the GPU post-hoc flag, and the EXIT
trap releases the lock on every path.  Each guard has a REGRESSION arm: provenance refusal, a missing precondition, a failing OFF roll (the ON roll must
not run), a failing compare, the optional control.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SH = shutil.which("sh") or "C:/Program Files/Git/bin/sh.exe"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
pytestmark = pytest.mark.skipif(not Path(SH).exists() and not shutil.which("sh"), reason="no sh")


def msys(p: Path) -> str:
    s = str(p).replace("\\", "/")
    return "/" + s[0].lower() + s[2:] if len(s) > 1 and s[1] == ":" else s


STUBS = {
    "code/r8clamp_common.py": r'''
import json, os, sys
a = sys.argv[1:]
open(os.environ["STUB_CALLS"], "a").write("provenance_check\n")
ok = os.environ.get("STUB_PROV_FAIL") != "1"
json.dump({"ok": True} if ok else {"ok": False, "reason": "REFUSING: stub"}, open(a[a.index("--out") + 1], "w"))
''',
    "code/k2k4_compare.py": r'''
import json, os, sys
a = sys.argv[1:]
def v(f): return a[a.index(f) + 1]
calls = os.environ["STUB_CALLS"]
if a[0] == "plan":
    open(calls, "a").write("plan nepisodes=%s pp=%s\n" % (v("--n-episodes") if "--n-episodes" in a else "auto", os.environ.get("PYTHONPATH", "")))
    json.dump({"clipA": [7, 12]}, open(v("--out-windows"), "w"))
    json.dump({"n_episodes": 10, "n_windows": 342, "n_binding_posthoc_nav_true_in_prefix": 12}, open(v("--out-plan"), "w"))
else:
    open(calls, "a").write("compare off=%s on=%s control=%s gpu_posthoc=%s\n" % (v("--off-dump").endswith("off/dump_s0"), v("--on-dump").endswith("on/dump_s0"),
                                                                          "--control-dump" in a, "--gpu-posthoc" in a))
    if os.environ.get("STUB_COMPARE_NOJSON") != "1":
        json.dump({"K2": {"verdict": "PASS"}, "K4": {"verdict": "INCONCLUSIVE"}, "OVERALL": {"verdict": "INCONCLUSIVE"}}, open(v("--out"), "w"))
''',
    "code/roll_seed_r8clamp.py": r'''
import json, os, sys
a = sys.argv[1:]
def v(f): return a[a.index(f) + 1]
mode = os.environ.get("R8_CLAMP_MODE")
open(os.environ["STUB_CALLS"], "a").write("roll mode=%s nopose=%s seed=%s pp_tree=%s windows=%s\n" % (
    mode, os.environ.get("R8_NO_POSE_SYNC"), v("--seed"), "tree_de4e15b" in os.environ.get("PYTHONPATH", ""), os.path.basename(v("--windows-json"))))
if os.environ.get("STUB_FAIL_ON") == mode:
    sys.exit(7)
d = v("--dump-dir"); os.makedirs(d, exist_ok=True)
open(os.path.join(d, "refcv6_extras.npz"), "wb").write(b"npz-bytes")
json.dump({"seed": 0}, open(v("--out-json"), "w"))
''',
    "code/roll_seed_r8.py": r'''
import json, os, sys
a = sys.argv[1:]
def v(f): return a[a.index(f) + 1]
open(os.environ["STUB_CALLS"], "a").write("roll_ctl mode=%s nopose=%s pp_launch=%s\n" % (
    os.environ.get("R8_CLAMP_MODE"), os.environ.get("R8_NO_POSE_SYNC"), "Lfb47a58376" in os.environ.get("PYTHONPATH", "")))
d = v("--dump-dir"); os.makedirs(d, exist_ok=True)
open(os.path.join(d, "refcv6_extras.npz"), "wb").write(b"npz-bytes")
json.dump({"seed": 0}, open(v("--out-json"), "w"))
''',
    "bat/wait_commit.py": r'''
import json, sys
a = sys.argv[1:]
json.dump({"met": True}, open(a[a.index("--out") + 1], "w"))
''',
    "bat/gpu_lock.py": r'''
import json, os, sys
a = sys.argv[1:]
open(os.environ["STUB_CALLS"], "a").write("gpu_lock %s job=%s\n" % (a[0], a[a.index("--job") + 1]))
if a[0] == "acquire":
    json.dump({"ok": os.environ.get("STUB_NOLOCK") != "1"}, open(a[a.index("--out") + 1], "w"))
''',
    "bat/gpu_watch.py": "import sys\n",
    "bat/chain_marker.py": r'''
import os, sys
open(os.environ["STUB_CALLS"], "a").write("chain_marker %s\n" % sys.argv[1])
''',
}


@pytest.fixture()
def sandbox(tmp_path):
    root = tmp_path / "sb"
    for rel, body in STUBS.items():
        p = root / "pkg" / rel if rel.startswith("code") else root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    kit = root / "kit"
    for f in ("ckpt/ckpt_ema_30000.pt", "ckpt/config.json", "windows/s2_windows.json"):
        (kit / f).parent.mkdir(parents=True, exist_ok=True)
        (kit / f).write_bytes(b"x")
    banked = root / "banked"
    for f in ("refcv6_extras.npz", "manifest.json", "refcv6_extras.npz.json"):
        (banked / "dump_s0" / f).parent.mkdir(parents=True, exist_ok=True)
        (banked / "dump_s0" / f).write_bytes(b"x")
    (kit / "chain").mkdir(parents=True, exist_ok=True)
    calls = root / "calls.log"
    return root, kit, banked, calls


def run_queue(sandbox, label="30000ema", **extra):
    root, kit, banked, calls = sandbox
    env = dict(os.environ, R8_PKGW=str((root / "pkg")).replace("\\", "/"), R8_BAT=str(root / "bat").replace("\\", "/"),
               R8_KITW=str(kit).replace("\\", "/"), R8_KIT=msys(kit), R8_PY=msys(Path(PY)), R8_CLAMP_TREE="E:/x/tree_de4e15b",
               K2K4_BANKED=str(banked).replace("\\", "/"), STUB_CALLS=str(calls).replace("\\", "/"), **extra)
    p = subprocess.run([SH, str(HERE / "queue_k2k4_r8.sh"), label], capture_output=True, text=True, env=env, timeout=300)
    log = (kit / "chain" / f"k2k4_step{label}.log")
    return p, (log.read_text(encoding="utf-8") if log.exists() else ""), (calls.read_text(encoding="utf-8") if calls.exists() else "")


def markers(log: str):
    return [ln.split()[0] for ln in log.splitlines() if ln.startswith("ZZK2K4")]


def test_happy_path_order_modes_and_release(sandbox):
    root = sandbox[0]
    p, log, calls = run_queue(sandbox)
    c = [ln.split()[0] + (" " + ln.split()[1] if ln.startswith(("gpu_lock", "chain_marker")) else "") for ln in calls.splitlines()]
    assert c.index("provenance_check") < c.index("plan") < c.index("gpu_lock acquire")                    # CPU gates BEFORE the lock
    rolls = [ln for ln in calls.splitlines() if ln.startswith("roll ")]
    assert rolls[0].startswith("roll mode=off nopose=1 seed=0 pp_tree=True windows=prefix_windows.json")   # K2: flag OFF, standalone clock, clamp tree
    assert rolls[1].startswith("roll mode=on nopose=1 seed=0 pp_tree=True")                               # K4: flag ON
    assert len(rolls) == 2 and not any(ln.startswith("roll_ctl") for ln in calls.splitlines())            # the control is OFF by default
    ia, ir = c.index("gpu_lock acquire"), c.index("gpu_lock release")
    assert ia < c.index("roll") < c.index("compare") < ir                                                 # rolls + compare INSIDE the lock; released after
    assert "compare off=True on=True control=False gpu_posthoc=True" in calls
    m = markers(log)
    assert m == ["ZZK2K4STARTZZ", "ZZK2K4PROVOKZZ", "ZZK2K4PLANOKZZ", "ZZK2K4LOCKEDZZ", "ZZK2K4OFFDONEZZ", "ZZK2K4ONDONEZZ", "ZZK2K4VERDICTZZ", "ZZK2K4CHAINENDZZ"]
    assert "K2=PASS K4=INCONCLUSIVE OVERALL=INCONCLUSIVE" in log and "episodes=10 windows=342 binding=12" in log
    assert (root / "pkg" / "raw" / "clamp_gpu" / "K2K4_step30000ema.json").exists()


def test_control_roll_uses_the_launch_tree_and_no_clamp_mode(sandbox):
    p, log, calls = run_queue(sandbox, K2K4_CONTROL="1")
    assert "roll_ctl mode=None nopose=1 pp_launch=True" in calls
    assert "compare off=True on=True control=True gpu_posthoc=True" in calls and "ZZK2K4CTLDONEZZ" in markers(log)


def test_n_episodes_override_is_passed_to_the_planner(sandbox):
    p, log, calls = run_queue(sandbox, K2K4_N_EPISODES="12")
    assert "plan nepisodes=12" in calls


def test_provenance_refusal_stops_before_the_lock(sandbox):
    p, log, calls = run_queue(sandbox, STUB_PROV_FAIL="1")
    assert "ZZK2K4PROVFAILZZ" in markers(log) and "REFUSING: stub" in log
    assert "gpu_lock" not in calls and "roll" not in calls and "plan" not in calls                         # REGRESSION arm: nothing ran


def test_missing_precondition_stops_before_anything(sandbox):
    root, kit, banked, calls = sandbox
    (kit / "ckpt" / "ckpt_ema_30000.pt").unlink()
    p, log, calls_txt = run_queue(sandbox)
    assert "ZZK2K4PRECONDFAILZZ" in markers(log) and "provenance_check" not in calls_txt and "gpu_lock" not in calls_txt


def test_failed_off_roll_skips_the_on_roll_and_still_releases_the_lock(sandbox):
    p, log, calls = run_queue(sandbox, STUB_FAIL_ON="off")
    m = markers(log)
    assert "ZZK2K4OFFFAILZZ" in m and "ZZK2K4ONDONEZZ" not in m and "ZZK2K4NOCOMPAREZZ" in m
    assert not any(ln.startswith("roll mode=on") for ln in calls.splitlines())
    assert "gpu_lock release" in calls and "chain_marker remove" in calls                                  # the EXIT trap ran


def test_failed_on_roll_means_no_compare_and_the_lock_is_released(sandbox):
    p, log, calls = run_queue(sandbox, STUB_FAIL_ON="on")
    m = markers(log)
    assert "ZZK2K4OFFDONEZZ" in m and "ZZK2K4ONFAILZZ" in m and "ZZK2K4NOCOMPAREZZ" in m and "compare" not in calls
    assert "gpu_lock release" in calls


def test_compare_without_json_is_reported_not_assumed(sandbox):
    p, log, calls = run_queue(sandbox, STUB_COMPARE_NOJSON="1")
    assert "ZZK2K4NOJSONZZ" in markers(log) and "ZZK2K4VERDICTZZ" not in markers(log)


def test_no_lock_means_no_roll(sandbox):
    p, log, calls = run_queue(sandbox, STUB_NOLOCK="1")
    assert "ZZK2K4NOLOCKZZ" in markers(log) and "roll" not in calls
