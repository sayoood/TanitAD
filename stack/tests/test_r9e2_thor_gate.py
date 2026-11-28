"""test_r9e2_thor_gate.py -- the battery GPU gate on Jetson Thor (run_battery_r9_e2.py (iv), FIXED 2026-11-12).

MEASURED 2026-11-12, the first REAL E2 roll on Thor: the dev-box gate (gpu_gate.py) int()s `nvidia-smi --query-gpu=memory.used`, Thor answers "[N/A]", the probe
fails, and run_battery_r7.gate_wait WAITS up to 12 h holding thor_gpu.lock.  The roll's dry run skips the gate, so nothing reached it before.
Every Thor fact here is INJECTED (the marker path, the nvidia-smi text): this suite also runs ON Thor, where /etc/nv_tegra_release exists.
Stdlib + pytest only (the Thor venv has no scipy).
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
R9DIR = ROOT / "ops" / "r9e2_read"
for _p in (str(ROOT), str(R9DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_battery_r9_e2 as R9  # noqa: E402

#: MEASURED 2026-11-12 on Thor (a CUDA tensor alive in pid 1985011): the ONE row `nvidia-smi --query-compute-apps=pid,process_name,used_memory` printed.
THOR_PY = "/home/nvidia/venvs/tanitad-train/bin/python"
THOR_ROW = f"1985011, {THOR_PY}, 134 MiB\n"
GG_REAL = ROOT.parent / "FlyWheels" / "TanitAD_EvalFlyWheel" / "incoming" / "2026-09-28-refcv7-standard-tests" / "battery" / "code" / "gpu_gate.py"
FAKE_DEVBOX_GATE = "def gate(max_used_mib=4300, min_free_ram_gb=8.0, self_pid=None, self_pids=None):\n    return {'ok': False, 'dev_box': True}\n"


@pytest.fixture(autouse=True)
def _restore_imports():
    """The import-hook / sys.path tests must not leak into the next test."""
    mp, path, mods = list(sys.meta_path), list(sys.path), set(sys.modules)
    yield
    sys.meta_path[:] = mp
    sys.path[:] = path
    for m in set(sys.modules) - mods:
        sys.modules.pop(m, None)


def _q(text):
    return lambda: text


def _marker(tmp_path, present=True):
    p = tmp_path / "nv_tegra_release"
    if present:
        p.write_text("# R38 (release), REVISION: test\n", encoding="utf-8")
    return p


def _fake_battery(tmp_path):
    bat = tmp_path / "bat"
    bat.mkdir()
    (bat / "gpu_gate.py").write_text(FAKE_DEVBOX_GATE, encoding="utf-8")
    sys.modules.pop("gpu_gate", None)
    sys.path.insert(0, str(bat))
    return bat


# ============================================================ the Thor gate's decision
def test_idle_thor_passes_and_reads_no_memory():
    g = R9.thor_gate_factory(_q(""))(self_pid=111)
    assert g["ok"] is True and g["probe_ok"] is True and g["platform"] == "thor"
    assert g["gpu_used_mib"] is None and g["free_ram_gb"] is None and g["compute_apps"] == []
    assert "no memory probe is admissible" in g["rule"]


def test_the_measured_thor_row_passes_for_its_own_pid_and_waits_for_anyone_else():
    gate = R9.thor_gate_factory(_q(THOR_ROW))
    assert gate(self_pid=1985011)["ok"] is True
    w = gate(self_pid=42)
    assert w["ok"] is False and w["probe_ok"] is True
    assert w["compute_apps"] == [(1985011, THOR_PY)] and w["python_compute"] == [(1985011, THOR_PY)]


def test_every_self_pid_is_excluded():
    assert R9.thor_gate_factory(_q(THOR_ROW))(self_pid=1, self_pids=[1985011])["ok"] is True


def test_a_non_python_compute_app_is_a_wait_on_thor():
    w = R9.thor_gate_factory(_q("777, /usr/src/tensorrt/bin/trtexec, 900 MiB\n"))(self_pid=1)
    assert w["ok"] is False and w["python_compute"] == [] and w["compute_apps"] == [(777, "/usr/src/tensorrt/bin/trtexec")]


@pytest.mark.parametrize("exc", [subprocess.TimeoutExpired("nvidia-smi", 60), subprocess.CalledProcessError(9, "nvidia-smi"),
                                 FileNotFoundError("nvidia-smi")], ids=["timeout", "rc", "missing"])
def test_an_unreadable_probe_fails_closed(exc):
    def boom():
        raise exc
    g = R9.thor_gate_factory(boom)(self_pid=1)
    assert g["ok"] is False and g["probe_ok"] is False and type(exc).__name__ in g["probe_error"]


def test_an_unparsable_row_fails_closed():
    g = R9.thor_gate_factory(_q("[N/A], python, [N/A]\n"))(self_pid=1)
    assert g["ok"] is False and g["probe_ok"] is False and "ValueError" in g["probe_error"]


def test_the_keywords_of_both_battery_call_sites_are_accepted():
    gate = R9.thor_gate_factory(_q(""))
    assert gate(self_pid=os.getpid(), min_free_ram_gb=6.0)["ok"] is True          # run_battery_r7.gate_wait
    assert gate(max_used_mib=4300, min_free_ram_gb=8.0)["self_pids_excluded"] == [os.getpid()]   # gpu_lock style: no self_pid => the caller


# ============================================================ where it is installed
def test_off_thor_nothing_is_installed(tmp_path):
    mp = list(sys.meta_path)
    assert R9.install_thor_gate(["--lock-held"], marker=_marker(tmp_path, present=False)) is None
    assert sys.meta_path == mp


def test_on_thor_without_lock_held_it_refuses(tmp_path):
    with pytest.raises(SystemExit, match="--lock-held"):
        R9.install_thor_gate(["--no-baselines", "--tag", "t"], marker=_marker(tmp_path))


def test_on_thor_the_gate_is_replaced_when_the_battery_imports_gpu_gate(tmp_path):
    _fake_battery(tmp_path)
    R9.install_thor_gate(["--lock-held"], marker=_marker(tmp_path), query=_q(""))
    gg = importlib.import_module("gpu_gate")
    assert getattr(gg.gate, "r9e2_thor", False) is True
    assert gg.gate(self_pid=1)["ok"] is True and gg.gate(self_pid=1)["platform"] == "thor"


def test_regression_arm_without_the_install_the_dev_box_gate_survives(tmp_path):
    _fake_battery(tmp_path)
    gg = importlib.import_module("gpu_gate")
    assert gg.gate(self_pid=1) == {"ok": False, "dev_box": True} and not hasattr(gg.gate, "r9e2_thor")


def test_already_imported_gpu_gate_is_patched_at_once(tmp_path):
    _fake_battery(tmp_path)
    importlib.import_module("gpu_gate")
    R9.install_thor_gate(["--lock-held"], marker=_marker(tmp_path), query=_q(""))
    assert getattr(sys.modules["gpu_gate"].gate, "r9e2_thor", False) is True


# ============================================================ reachable from the caller (main), in the right order
def _pins(monkeypatch, tmp_path):
    for k in R9.ENV_PINS:
        monkeypatch.setenv(k, str(tmp_path / f"pin_{k}"))
    monkeypatch.delenv("R8_NO_POSE_SYNC", raising=False)


def test_main_keeps_the_env_refusals_first_on_thor(tmp_path, monkeypatch):
    monkeypatch.setattr(R9, "THOR_MARKER", _marker(tmp_path))
    for k in R9.ENV_PINS:
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(SystemExit, match="unset env pin"):
        R9.main(["--no-baselines", "--tag", "t"])


def test_main_refuses_on_thor_without_lock_held(tmp_path, monkeypatch):
    monkeypatch.setattr(R9, "THOR_MARKER", _marker(tmp_path))
    _pins(monkeypatch, tmp_path)
    with pytest.raises(SystemExit, match="--lock-held"):
        R9.main(["--no-baselines", "--tag", "t"])


def test_main_installs_the_thor_gate_before_the_battery_loads(tmp_path, monkeypatch):
    monkeypatch.setattr(R9, "THOR_MARKER", _marker(tmp_path))
    _pins(monkeypatch, tmp_path)
    with pytest.raises((SystemExit, OSError)):                  # the dummy split cannot load: by then the hook must already be in
        R9.main(["--no-baselines", "--tag", "t", "--lock-held"])
    assert [f for f in sys.meta_path if getattr(f, "name", None) == "gpu_gate"]


def test_main_off_thor_installs_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(R9, "THOR_MARKER", _marker(tmp_path, present=False))
    _pins(monkeypatch, tmp_path)
    with pytest.raises((SystemExit, OSError)):
        R9.main(["--no-baselines", "--tag", "t", "--lock-held"])
    assert not [f for f in sys.meta_path if getattr(f, "name", None) == "gpu_gate"]


# ============================================================ the defect itself, on the REAL dev-box gate (skipped where the battery package is not in the tree)
@pytest.mark.skipif(not GG_REAL.exists(), reason="the battery package is not in this tree (Thor launch trees carry stack/ + taniteval/ only)")
def test_the_real_dev_box_gate_can_never_pass_on_thor_facts():
    spec = importlib.util.spec_from_file_location("gpu_gate_real_r9e2test", GG_REAL)
    gg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gg)

    def thor_q(args):
        if any("memory.used" in a for a in args):
            return "[N/A]\n"                                    # MEASURED on Thor 2026-11-12
        if any("compute-apps" in a for a in args):
            return ""
        raise FileNotFoundError(args[0])                        # powershell.exe does not exist on Thor
    gg._q = thor_q
    g = gg.gate(self_pid=1, min_free_ram_gb=6.0)
    assert g["ok"] is False and g["probe_ok"] is False and "ValueError" in g["probe_error"]
    gg.gate = R9.thor_gate_factory(_q(""))
    assert gg.gate(self_pid=1, min_free_ram_gb=6.0)["ok"] is True
