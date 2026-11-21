"""No arm inherits another arm's description — the builder describes each arm explicitly or refuses.

    PYTHONPATH=D:/Projects/TanitAD/stack;D:/Projects/TanitAD/taniteval \
      C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q tests/test_artifact_arm_meta.py

⛔ CAUGHT BEFORE IT RAN ON THE MODEL ARM. ``goal_source`` fell through to "PRIVILEGED: the logged
human future itself" for any arm outside (CV, STOP), and ``sensor_set`` was "NONE" for every arm
— so refcv4b's first criteria-checked artifact would have described a camera model as reading no
camera and as a privileged reference arm. Both expectations below are LITERALS.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("w3_artifacts", PKG / "code" / "artifacts_navtest.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["w3_artifacts"] = m
    spec.loader.exec_module(m)
    return m


M = _mod()


def test_the_model_arm_is_described_as_what_it_is():
    m = M.arm_meta("A1")
    assert "CAM_F0" in m["sensors"] and "CAM_L0" in m["sensors"] and "CAM_R0" in m["sensors"]
    assert not m["goal"].startswith("PRIVILEGED")
    assert m["goal"].startswith("NavSim driving_command[t0]")
    assert m["inputs"]["cameras"] is True and m["inputs"]["driving_command"] is True
    assert m["inputs"]["ego_pose_history"] is False
    assert m["enforce"]["declared_fields"] == ["ego_velocity[t0]", "ego_acceleration[t0]",
                                               "driving_command[t0]"]
    assert m["enforce"]["vision_only_claimed"] is False


def test_an_undescribed_arm_is_refused_not_defaulted():
    with pytest.raises(SystemExit, match="no explicit"):
        M.arm_meta("A2_vision_pure")


def test_the_reference_arms_read_exactly_what_they_read_before_the_refactor():
    """⭐ THE CONTROL: the refactor must change nothing for the three banked arms. These literals
    are the strings the pre-refactor code emitted for them."""
    assert M.arm_meta("CV")["goal"] == "none — this arm reads no route/goal signal"
    assert M.arm_meta("STOP")["goal"] == "none — this arm reads no route/goal signal"
    assert M.arm_meta("HUMAN")["goal"] == ("PRIVILEGED: the logged human future itself "
                                           "(reference arm, never deployable)")
    for arm in ("CV", "STOP", "HUMAN"):
        assert M.arm_meta(arm)["sensors"] == "NONE (no camera/LiDAR is read by this arm)"
        assert M.arm_meta(arm)["inputs"]["cameras"] is False


def test_the_model_arm_declares_the_same_inputs_as_e2s_arm_table():
    """Cross-check against an INDEPENDENT source: E2's own ARMS table, read from E2's file — not
    re-derived from W3's entry, which would only measure that W3 agrees with itself."""
    e2 = (PKG.parents[0] / "2026-09-19-navsim-refcv4b-bridge" / "code" /
          "tanitad_navsim_bridge.py")
    if not e2.exists():
        pytest.skip(f"E2's bridge not on disk: {e2}")
    txt = e2.read_text(encoding="utf-8")
    block = txt[txt.index('"A1_ego_cmd": {'):][:400]
    for f in ("ego_velocity[t0]", "ego_acceleration[t0]", "driving_command[t0]"):
        assert f in block
    assert M.arm_meta("A1")["enforce"]["declared_fields"] == [
        "ego_velocity[t0]", "ego_acceleration[t0]", "driving_command[t0]"]
