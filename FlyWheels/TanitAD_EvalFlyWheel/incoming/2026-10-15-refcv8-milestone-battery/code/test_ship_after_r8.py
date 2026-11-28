"""Pins for ship_after_r8.sh (the SHIP-R8 waiter): done / timeout / memory-gate / WITHHELD / stage-failure branches, run for real through `sh` with STUB python stages in a temp package.
Expectations are LITERAL marker names; every guard has a mutation (a not-ALL_PASS trainfit forward must NOT release the chain; a stale artifact must not pass for a stage's output).
    python -m pytest -q test_ship_after_r8.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "ship_after_r8.sh"
SH = shutil.which("sh") or next((p for p in ("C:/Program Files/Git/usr/bin/sh.exe", "C:/Program Files/Git/bin/sh.exe") if Path(p).exists()), None)
pytestmark = pytest.mark.skipif(SH is None, reason="no POSIX sh")

STUB = '''
import json, os, sys
from pathlib import Path
name = Path(__file__).stem
pkg = Path(__file__).resolve().parent.parent
tree = sys.argv[sys.argv.index("--tree") + 1]
mode = os.environ.get("STUB_" + name.split("_")[0].upper(), "ok")
(pkg / "stub_calls.log").open("a").write(name + " " + tree + " " + mode + "\\n")
dep = pkg / "raw" / "retime_deploy"
dep.mkdir(parents=True, exist_ok=True)
if name == "deploy8_forward_r8":
    if mode == "ok":
        (dep / f"FORWARD_DEPLOY8_{tree}.json").write_text(json.dumps({"controls": {"ALL_PASS": True}}))
        for sd in (0, 1):
            (dep / f"DEPLOY8_POS_{tree}_s{sd}.npz").write_bytes(b"x")
    elif mode == "nonpz":
        (dep / f"FORWARD_DEPLOY8_{tree}.json").write_text(json.dumps({"controls": {"ALL_PASS": True}}))
elif name == "deploy_forward_r8":
    if mode == "ok":
        (dep / f"FORWARD_DEPLOY_{tree}.json").write_text(json.dumps({"controls": {"ALL_PASS": True}}))
    elif mode == "withheld":
        (dep / f"WITHHELD_DEPLOY_{tree}.json").write_text("{}")
elif name == "tac_retime_r8":
    if mode == "ok":
        (dep / f"TAC_{tree}.json").write_text(json.dumps({"clause5": {"status": "PASS"}, "controls": {"ALL_PASS": True}}))
elif name == "ship_r8_verdict":
    if mode == "ok":
        d = pkg / "raw" / "ship_r8"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"SHIP_R8_{tree}.json").write_text(json.dumps({"overall": "PASS", "label": "stub label", "per_clause": {"1": "PASS"}}))
'''


def msys(p: Path) -> str:
    s = str(p).replace("\\", "/")
    return "/" + s[0].lower() + s[2:] if len(s) > 2 and s[1] == ":" else s


def make_pkg(tmp: Path, tf="allpass", withheld=False, d8=True) -> Path:
    pkg = tmp / "pkg"
    (pkg / "code").mkdir(parents=True)
    (pkg / "raw" / "retime_trainfit").mkdir(parents=True)
    for n in ("deploy8_forward_r8.py", "deploy_forward_r8.py", "tac_retime_r8.py", "ship_r8_verdict.py") if d8 else ("deploy_forward_r8.py", "tac_retime_r8.py", "ship_r8_verdict.py"):
        (pkg / "code" / n).write_text(STUB)
    if tf == "allpass":
        (pkg / "raw" / "retime_trainfit" / "FORWARD_TRAINFIT_t1.json").write_text(json.dumps({"controls": {"ALL_PASS": True}}))
    elif tf == "notpass":
        (pkg / "raw" / "retime_trainfit" / "FORWARD_TRAINFIT_t1.json").write_text(json.dumps({"controls": {"ALL_PASS": False}}))
    if withheld:
        (pkg / "raw" / "retime_trainfit" / "WITHHELD_t1.json").write_text("{}")
    return pkg


def run(tmp: Path, pkg: Path, free="50", maxwait="6", extra_env=None, timeout=60):
    kit = tmp / "kit"
    (kit / "chain").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({"R8_PKGW": str(pkg).replace("\\", "/"), "R8_PKGL": msys(pkg), "R8_KIT": msys(kit), "R8_PY": msys(Path(sys.executable)),
                "SHIP_LOG": msys(kit / "chain" / "ship_r8.log"), "SHIP_OUT": msys(kit / "chain" / "ship_t1.out"), "SHIP_POLL_S": "1", "SHIP_CONFIRM_S": "1", "SHIP_MEMPOLL_S": "1",
                "SHIP_MAXWAIT_S": maxwait, "SHIP_NEED_GB": "5", "SHIP_FREE_GB_STUB": free, "SHIP_WITHHELD_GRACE_S": "2"})
    env.update(extra_env or {})
    p = subprocess.run([SH, str(SCRIPT), "t1"], env=env, capture_output=True, text=True, timeout=timeout)
    log = (kit / "chain" / "ship_r8.log").read_text() if (kit / "chain" / "ship_r8.log").exists() else ""
    calls = (pkg / "stub_calls.log").read_text().split("\n") if (pkg / "stub_calls.log").exists() else []
    return p.returncode, log, [c for c in calls if c]


def markers(log: str) -> list:
    return [ln.split()[0] for ln in log.splitlines() if ln.startswith("ZZSHIP")]


def test_done_branch_runs_2_3_4_in_order_and_ends_with_a_verdict(tmp_path):
    pkg = make_pkg(tmp_path)
    rc, log, calls = run(tmp_path, pkg)
    m = markers(log)
    assert m == ["ZZSHIPSTARTZZ", "ZZSHIPTFSEENZZ", "ZZSHIPRUNDEP8ZZ", "ZZSHIPDEP8OKZZ", "ZZSHIPRUNDEPZZ", "ZZSHIPDEPOKZZ", "ZZSHIPRUNTACZZ", "ZZSHIPTACOKZZ", "ZZSHIPRUNVERDICTZZ",
                 "ZZSHIPVERDICTZZ", "ZZSHIPDONEZZ", "ZZSHIPENDZZ"], m
    assert [c.split()[0] for c in calls] == ["deploy8_forward_r8", "deploy_forward_r8", "tac_retime_r8", "ship_r8_verdict"]       # the scored arm first, the per-surface form beside, then tac, then the verdict
    assert "overall=PASS" in log and "clause5=PASS" in log and "t1" in log
    assert (pkg / "raw" / "ship_r8" / "SHIP_R8_t1.json").exists()


def test_timeout_branch_when_the_trainfit_forward_never_appears(tmp_path):
    pkg = make_pkg(tmp_path, tf="none")
    rc, log, calls = run(tmp_path, pkg, maxwait="3")
    assert rc == 4 and "ZZSHIPTIMEOUTZZ" in markers(log) and markers(log)[-1] == "ZZSHIPENDZZ" and calls == []


def test_mutation_forward_trainfit_without_all_pass_does_not_release_the_chain(tmp_path):
    pkg = make_pkg(tmp_path, tf="notpass")
    rc, log, calls = run(tmp_path, pkg, maxwait="3")
    assert rc == 4 and "ZZSHIPTFSEENZZ" not in markers(log) and "ZZSHIPTIMEOUTZZ" in markers(log) and calls == []


def test_memory_gate_blocks_then_times_out_without_running_a_stage(tmp_path):
    pkg = make_pkg(tmp_path)
    rc, log, calls = run(tmp_path, pkg, free="1.5", maxwait="3")
    m = markers(log)
    assert rc == 4 and "ZZSHIPTFSEENZZ" in m and "ZZSHIPMEMWAITZZ" in m and "ZZSHIPTIMEOUTZZ" in m and "ZZSHIPRUNDEPZZ" not in m and calls == []


def test_memory_gate_threshold_is_inclusive_at_five(tmp_path):
    pkg = make_pkg(tmp_path)
    rc, log, calls = run(tmp_path, pkg, free="5")
    assert "ZZSHIPDONEZZ" in markers(log) and len(calls) == 4
    pkg2 = make_pkg(tmp_path / "b")
    rc, log, calls = run(tmp_path / "b", pkg2, free="4.99", maxwait="3")
    assert "ZZSHIPDONEZZ" not in markers(log) and calls == []


def test_a_failed_deploy_stage_does_not_stop_the_verdict_and_a_stale_artifact_does_not_pass(tmp_path):
    pkg = make_pkg(tmp_path)
    dep = pkg / "raw" / "retime_deploy"
    dep.mkdir(parents=True)
    (dep / "FORWARD_DEPLOY_t1.json").write_text(json.dumps({"controls": {"ALL_PASS": True}}))     # STALE: a previous run's output
    rc, log, calls = run(tmp_path, pkg, extra_env={"STUB_DEPLOY": "none"})
    m = markers(log)
    assert "ZZSHIPDEPFAILZZ" in m and "ZZSHIPDEPOKZZ" not in m                                      # the stale file was deleted before the stage ran
    assert "ZZSHIPRUNVERDICTZZ" in m and "ZZSHIPDONEZZ" in m and len(calls) == 4


def test_withheld_deploy_and_missing_tac_are_logged_and_the_verdict_still_runs(tmp_path):
    pkg = make_pkg(tmp_path)
    rc, log, calls = run(tmp_path, pkg, extra_env={"STUB_DEPLOY": "withheld", "STUB_TAC": "none"})
    m = markers(log)
    assert "ZZSHIPDEPWITHHELDZZ" in m and "ZZSHIPTACFAILZZ" in m and "ZZSHIPDONEZZ" in m


def test_verdict_stage_failure_is_a_terminal_marker(tmp_path):
    pkg = make_pkg(tmp_path)
    rc, log, calls = run(tmp_path, pkg, extra_env={"STUB_SHIP": "none"})
    m = markers(log)
    assert "ZZSHIPVERDICTFAILZZ" in m and "ZZSHIPDONEZZ" not in m and m[-1] == "ZZSHIPENDZZ"


def test_withheld_trainfit_goes_to_stage_4_only_after_the_grace_period(tmp_path):
    pkg = make_pkg(tmp_path, tf="none", withheld=True)
    rc, log, calls = run(tmp_path, pkg, maxwait="30")
    m = markers(log)
    assert "ZZSHIPTFWITHHELDSEENZZ" in m and "ZZSHIPTFWITHHELDZZ" in m and "ZZSHIPRUNDEPZZ" not in m and "ZZSHIPRUNTACZZ" not in m
    assert [c.split()[0] for c in calls] == ["ship_r8_verdict"] and "ZZSHIPDONEZZ" in m


def test_script_is_lf_only_and_posix_clean():
    b = SCRIPT.read_bytes()
    assert b"\r" not in b
    assert subprocess.run([SH, "-n", str(SCRIPT)], capture_output=True).returncode == 0


def test_missing_sibling_deploy8_script_is_checked_at_run_time_and_logged(tmp_path):
    pkg = make_pkg(tmp_path, d8=False)
    rc, log, calls = run(tmp_path, pkg)
    m = markers(log)
    assert "ZZSHIPDEP8MISSINGZZ" in m and "ZZSHIPRUNDEP8ZZ" not in m and "ZZSHIPDEP8OKZZ" not in m
    assert [c.split()[0] for c in calls] == ["deploy_forward_r8", "tac_retime_r8", "ship_r8_verdict"] and "ZZSHIPDONEZZ" in m      # the chain still ends in a verdict (it reads INCOMPLETE)


def test_deploy8_stage_without_its_npz_is_a_failure_and_a_stale_forward_is_deleted(tmp_path):
    pkg = make_pkg(tmp_path)
    dep = pkg / "raw" / "retime_deploy"
    dep.mkdir(parents=True)
    (dep / "FORWARD_DEPLOY8_t1.json").write_text(json.dumps({"controls": {"ALL_PASS": True}}))   # STALE
    for sd in (0, 1):
        (dep / f"DEPLOY8_POS_t1_s{sd}.npz").write_bytes(b"stale")
    rc, log, calls = run(tmp_path, pkg, extra_env={"STUB_DEPLOY8": "none"})
    m = markers(log)
    assert "ZZSHIPDEP8FAILZZ" in m and "ZZSHIPDEP8OKZZ" not in m and not (dep / "FORWARD_DEPLOY8_t1.json").exists()
    rc, log, calls = run(tmp_path / "b", make_pkg(tmp_path / "b"), extra_env={"STUB_DEPLOY8": "nonpz"})    # a forward file without its positions is not a pass either
    assert "ZZSHIPDEP8FAILZZ" in markers(log) and "ZZSHIPDONEZZ" in markers(log)
