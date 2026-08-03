"""Standalone offline tests for the L2D -> contract mapping spec (no network, no video).

Run: pytest "TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-07-11-semantic-label-survey/tests"
Uses synthetic rows in the MEASURED L2D schema (probe_l2d_taxonomy.py, 2026-07-11).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from l2d_contract_map import (  # noqa: E402
    FRONT_CAMERA_KEY,
    map_continuous_action,
    nav_command_class,
    label_entropy,
    build_contract_row,
    _CLASS_NAMES,
)

# --- verbatim instructions measured off yaak-ai/L2D (l2d_taxonomy_result.json) ----------------
REAL_INSTRUCTIONS = {
    "Go straight on the secondary road for 0.7 km, observe the speed limit of 70 km/h.": "follow",
    "Make a U-turn and then go straight for 50 m.": "u_turn",
    "Go straight on the residential road for 150 m, observe the speed limit of 50 km/h and turn "
    "right at the intersection following the right before left rule and then go straight for 100 m.": "turn_right",
    "Go straight on the residential road for 150 m, observe the speed limit of 30 km/h and exit "
    "the roundabout using the first exit and then go straight on the residential road for 50 m.": "roundabout",
    "reverse out and then go straight for 10 m and turn left at the intersection following the "
    "right before left rule and then go straight for 100 m.": "reverse",
}


def test_front_camera_key_is_a_real_l2d_key():
    # measured camera_keys from the probe
    measured = {
        "observation.images.front_left", "observation.images.left_forward",
        "observation.images.right_forward", "observation.images.left_backward",
        "observation.images.rear", "observation.images.right_backward",
    }
    assert FRONT_CAMERA_KEY in measured


def test_action_3_to_2_channel_algebra():
    # steer passes through; accel channel = accel - brake
    out = map_continuous_action([0.3, 0.8, 0.0])   # steer .3, throttle .8, no brake
    assert out.shape == (2,)
    assert out[0] == pytest.approx(0.3)
    assert out[1] == pytest.approx(0.8)
    braking = map_continuous_action([-0.1, 0.0, 0.9])  # hard brake -> negative accel
    assert braking[1] == pytest.approx(-0.9)
    assert braking[0] == pytest.approx(-0.1)


def test_action_wrong_dim_fails_loud():
    with pytest.raises(ValueError):
        map_continuous_action([0.1, 0.2])  # dim-2, not L2D


def test_nav_class_on_real_instructions():
    for instr, expected in REAL_INSTRUCTIONS.items():
        assert nav_command_class(instr) == expected, instr


def test_compositional_instruction_takes_decisive_maneuver_not_follow():
    # "go straight ... and turn right ..." must label as the maneuver, not follow
    instr = "Go straight on the tertiary road for 150 m and turn right at the intersection."
    assert nav_command_class(instr) == "turn_right"


def test_all_classes_are_known():
    for instr in REAL_INSTRUCTIONS:
        assert nav_command_class(instr) in _CLASS_NAMES


def test_label_entropy_captures_comma_starvation_vs_l2d():
    # comma2k19 proxy: ~all follow -> effective classes ~1
    comma = ["follow"] * 1000
    assert label_entropy(comma) == pytest.approx(1.0, abs=1e-6)
    # L2D proxy: diverse maneuvers -> effective classes well above 1
    l2d = (["follow"] * 400 + ["turn_left"] * 150 + ["turn_right"] * 150 +
           ["roundabout"] * 120 + ["u_turn"] * 80 + ["lane_change"] * 100)
    assert label_entropy(l2d) > 3.0


def test_build_contract_row_shapes_and_labels():
    row = {
        "action.continuous": np.array([0.2, 0.5, 0.0], dtype=np.float32),
        "action.discrete": np.array([1, 0]),
        "observation.state.waypoints": np.arange(10, dtype=np.float32),
        "task.instructions": "Make a U-turn and then go straight for 50 m.",
    }
    out = build_contract_row(row)
    assert out["action"].shape == (2,)
    assert out["nav_class"] == "u_turn"
    assert out["waypoints"].shape == (10,)
    assert out["front_camera_key"] == FRONT_CAMERA_KEY


def test_build_contract_row_missing_action_fails_loud():
    with pytest.raises(ValueError):
        build_contract_row({"task.instructions": "Go straight."})


def test_instruction_dict_form_is_unwrapped():
    # tasks.jsonl packs the instruction inside a dict — build_contract_row must handle it
    row = {
        "action.continuous": [0.0, 0.3, 0.1],
        "task.instructions": {"task_index": 6, "__index_level_0__": "Make a U-turn and then go straight for 50 m."},
    }
    out = build_contract_row(row)
    assert out["nav_class"] == "u_turn"
