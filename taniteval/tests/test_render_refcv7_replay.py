"""Pins ``render_refcv7_replay``'s PURE parts: the plan -> (lat3, lon3) reading, the agreement rules, the data
encoders, the box matcher / NMS ports, and the replay page's wiring (every layer key has a control).

Every expected value is a LITERAL worked out by hand -- never an expression over the code under test -- and the
direction-sensitive checks ship with a mirror mutation that must go RED (left drawn as right is the classic defect).

Runs without the model, the data or the ``tanitad`` package: the tool is loaded by path (numpy at module level, PIL
inside the encoders). The JavaScript is only syntax-checked here (``node --check``) and wiring-checked by regex; its
numerical behaviour is checked against Python values by the page's own "self-test" button on the real data.
"""
from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess

import numpy as np
import pytest

pytest.importorskip("PIL")

_HERE = os.path.dirname(os.path.abspath(__file__))
_TOOLS = os.path.join(os.path.dirname(_HERE), "tools")
_TOOL = os.path.join(_TOOLS, "render_refcv7_replay.py")
_JS = os.path.join(_TOOLS, "refcv7_replay", "replay.js")
_HTML = os.path.join(_TOOLS, "refcv7_replay", "index.html")


def _load():
    spec = importlib.util.spec_from_file_location("render_refcv7_replay_under_test", _TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load()


# ------------------------------------------------------------------------------------------- #
# the class tables are the programme's, in the programme's ORDER                              #
# ------------------------------------------------------------------------------------------- #
def test_lat3_lon3_orders_are_refc_tactical_orders():
    # refc_tactical.py:136-145 -- LAT_CLASSES / LON_CLASSES, ids 0, 1, 2
    assert R.LAT3 == ("lane_keep", "turn_left", "turn_right")
    assert R.LON3 == ("brake_stop", "steady", "accelerate")


def test_threshold_literals_are_refc_tactical_literals():
    assert R.CURV_TURN_MAN_PER_M == pytest.approx(1.0 / 60.0, abs=1e-15)
    assert (R.DV_ACCEL_MS, R.DV_BRAKE_MS, R.STOP_V_MS, R.MOVING_V_MS, R.MIN_ARC_M) == (1.0, -1.0, 0.3, 1.0, 0.10)


# ------------------------------------------------------------------------------------------- #
# plan -> (lat3, lon3)                                                                        #
# ------------------------------------------------------------------------------------------- #
STRAIGHT = [(2.5, 0.0), (5.0, 0.0), (7.5, 0.0), (10.0, 0.0)]            # 5 m/s, no turn
LEFT = [(2.5, 0.0), (5.0, 0.0), (7.5, 0.2), (9.9, 1.2)]                 # last segment (2.4, 1.0): 0.3948 rad left


def test_straight_constant_speed_is_lane_keep_steady():
    f = R.plan_factor(STRAIGHT, v0=5.0)
    assert (f["lat3"], f["lon3"]) == (0, 1)
    assert f["dv"] == pytest.approx(0.0, abs=1e-12) and f["v1"] == pytest.approx(5.0, abs=1e-12)


def test_left_curve_is_turn_left_and_its_mirror_is_turn_right():
    f = R.plan_factor(LEFT, v0=5.0)
    assert f["lat3"] == 1                                   # kappa = 0.3948 / 10.11 m = 0.039 >= 1/60
    assert f["dyaw"] == pytest.approx(0.39479, abs=1e-4)
    mirrored = [(x, -y) for x, y in LEFT]
    assert R.plan_factor(mirrored, v0=5.0)["lat3"] == 2     # the lateral-mirror mutation must flip the class
    assert R.plan_factor(mirrored, v0=5.0)["dyaw"] == pytest.approx(-0.39479, abs=1e-4)


def test_gentle_curve_below_the_curvature_gate_is_lane_keep():
    # radius ~300 m: last segment heading 0.0083 rad over ~10 m -> kappa 0.0008 < 1/60
    gentle = [(2.5, 0.0), (5.0, 0.01), (7.5, 0.03), (10.0, 0.0 + 0.0833 * 0.5)]
    assert R.plan_factor(gentle, v0=5.0)["lat3"] == 0


def test_braking_accelerating_and_stopped():
    brake = [(2.0, 0.0), (3.5, 0.0), (4.5, 0.0), (4.6, 0.0)]            # last segment 0.1 m -> v1 = 0.2 m/s
    f = R.plan_factor(brake, v0=10.0)
    assert f["lon3"] == 0 and f["dv"] == pytest.approx(-9.8, abs=1e-9)    # dv < -1
    accel = [(1.0, 0.0), (2.2, 0.0), (3.9, 0.0), (5.9, 0.0)]            # last segment 2.0 m -> v1 = 4 m/s
    assert R.plan_factor(accel, v0=2.0)["lon3"] == 2                     # dv = +2 > +1
    stopped = [(0.0, 0.0)] * 4
    f0 = R.plan_factor(stopped, v0=0.0)
    assert (f0["lat3"], f0["lon3"]) == (0, 1)                            # standing still from rest is 'steady'
    assert f0["dyaw"] == 0.0                                             # a stalled last segment reads heading 0


def test_brake_overwrites_accelerate_when_it_stops_a_moving_ego():
    # v0 = 1.5 m/s (moving), v1 = 0.2 m/s < STOP -> brake_stop even though |dv| = 1.3 > 1 would also be 'brake'
    f = R.plan_factor([(0.3, 0.0), (0.5, 0.0), (0.6, 0.0), (0.7, 0.0)], v0=1.5)
    assert f["lon3"] == 0


def test_invalid_slot_gives_none():
    f = R.plan_factor(STRAIGHT, v0=5.0, valid=[True, True, True, False])
    assert f["lat3"] is None and f["lon3"] is None


# ------------------------------------------------------------------------------------------- #
# agreement rules                                                                             #
# ------------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("lat3, lat, want", [(1, "TURN_L", True), (2, "TURN_L", False), (0, "LANE_KEEP", True),
                                              (0, "LANE_CHANGE_L", None), (1, "NUDGE_L", None), (None, "TURN_L", None)])
def test_lat_agreement(lat3, lat, want):
    assert R.agree_tactical(lat3, 1, lat, "CRUISE")["lat_ok"] is want


@pytest.mark.parametrize("lon3, lon, want", [(2, "ACCELERATE", True), (1, "ACCELERATE", False), (0, "BRAKE_TO", True),
                                              (1, "CRUISE", True), (0, "CRUISE", False), (1, "HOLD", True),
                                              (0, "HOLD", True), (2, "HOLD", False), (None, "CRUISE", None)])
def test_lon_agreement(lon3, lon, want):
    assert R.agree_tactical(0, lon3, "LANE_KEEP", lon)["lon_ok"] is want


def test_every_v7_action_has_a_mapping_entry():
    assert set(R.LAT_TO_LAT3) == {"LANE_KEEP", "LANE_CHANGE_L", "LANE_CHANGE_R", "ABORT_LC", "NUDGE_L", "NUDGE_R",
                                  "TURN_L", "TURN_R"}
    assert set(R.LON_TO_LON3) == {"FOLLOW", "CRUISE", "YIELD_MERGE", "BRAKE_TO", "CREEP", "HOLD",
                                  "ADAPT_SPEED_FOR_CURVE", "ACCELERATE"}


def test_nav_agreement():
    assert R.agree_nav("left", 0, True) is True and R.agree_nav("left", 1, False) is False
    assert R.agree_nav("right", 2, None) is None                   # the predicate could not be evaluated
    assert R.agree_nav("follow", 0, None) is True                  # follow: agree iff the plan does not turn
    assert R.agree_nav("follow", 1, None) is False and R.agree_nav("follow", 2, None) is False
    assert R.agree_nav("follow", None, None) is None


# ------------------------------------------------------------------------------------------- #
# encoders                                                                                    #
# ------------------------------------------------------------------------------------------- #
def test_fan_cm_literal_bytes_and_roundtrip():
    fan = np.array([[[1.0, -2.5]]])
    b64, n_clip = R.encode_fan_cm(fan)
    assert b64 == "ZAAG/w==" and n_clip == 0                       # 100 -> 64 00 ; -250 -> 06 FF (little endian)
    rt = R.decode_fan_cm(b64, 1, 1)
    assert rt.shape == (1, 1, 2) and rt[0, 0, 0] == 1.0 and rt[0, 0, 1] == -2.5
    rng = np.random.RandomState(0)
    f = rng.uniform(-60, 60, (7, 8, 2))
    back = R.decode_fan_cm(R.encode_fan_cm(f)[0], 7, 8)
    assert np.abs(back - f).max() <= 0.0050001                     # centimetre quantisation


def test_fan_cm_counts_clipped_values():
    b64, n_clip = R.encode_fan_cm(np.array([[[400.0, 0.0]]]))      # 40,000 cm > int16
    assert n_clip == 1


def test_map_png_roundtrip_with_flags():
    codes = np.zeros((1000, 600), np.uint8)
    codes[10:20, 5:9] = 3
    codes[0, 0] = 255                                              # SAM3 never saw this cell
    lv = np.ones((1000, 600), bool)
    lv[500:510, 100:110] = False                                   # the lift cannot reach these cells
    png = R.encode_map(codes, lv)
    c2, lv2 = R.decode_map(png)
    assert np.array_equal(c2, codes) and np.array_equal(lv2, lv)
    # raw bytes the browser will read: class | 8 (not seen) | 16 (lift invalid)
    from PIL import Image
    import io
    raw = np.asarray(Image.open(io.BytesIO(png)))
    assert raw[0, 0] == 8 and raw[15, 6] == 3 and raw[505, 105] == 16 and raw[1, 1] == 0


def test_prediction_png_carries_the_gt_not_seen_flag():
    pred = np.full((1000, 600), 1, np.uint8)
    gt = np.full((1000, 600), 1, np.uint8)
    gt[3, 4] = 255
    lv = np.ones((1000, 600), bool)
    png = R.encode_map(pred, lv, notseen=(gt == 255))
    from PIL import Image
    import io
    raw = np.asarray(Image.open(io.BytesIO(png)))
    assert raw[3, 4] == 1 + 8 and raw[0, 0] == 1


def test_map_class_above_seven_is_refused():
    with pytest.raises(ValueError):
        R.encode_map(np.full((1000, 600), 9, np.uint8), np.ones((1000, 600), bool))


# ------------------------------------------------------------------------------------------- #
# detection matcher + NMS                                                                     #
# ------------------------------------------------------------------------------------------- #
def test_greedy_match_literal():
    det_xy = [(0.0, 0.0), (10.0, 0.0), (50.0, 50.0)]
    rows = R.greedy_match(det_xy, [0.9, 0.8, 0.7], [(0.5, 0.0), (10.5, 0.0)], [True, False], [False, True])
    assert rows == [(0, 0, 1), (1, 1, -1), (2, -1, 0)]             # TP, DontCare (an IGNORE), FP


def test_greedy_match_a_taken_gt_cannot_be_matched_twice():
    rows = R.greedy_match([(0.0, 0.0), (0.2, 0.0)], [0.9, 0.8], [(0.0, 0.0)], [True], [False])
    assert rows == [(0, 0, 1), (1, -1, 0)]                         # the second detection is a false positive


def test_greedy_match_no_gt_is_all_false_positive():
    assert R.greedy_match([(1.0, 1.0)], [0.5], np.zeros((0, 2)), [], []) == [(0, -1, 0)]


def test_greedy_match_min_conf_is_a_prefix_of_the_full_run():
    det = [(0.0, 0.0), (1.0, 0.0), (30.0, 0.0), (31.0, 0.0)]
    p = [0.9, 0.6, 0.4, 0.2]
    gt = [(0.1, 0.0), (30.2, 0.0)]
    full = R.greedy_match(det, p, gt, [True, True], [False, False])
    part = R.greedy_match(det, p, gt, [True, True], [False, False], min_conf=0.5)
    assert part == full[:2]


def test_bev_nms_literal():
    assert R.bev_nms([(0.0, 0.0), (1.0, 0.0), (5.0, 0.0)], [0.9, 0.8, 0.7], 2.0) == [0, 2]
    assert R.bev_nms([(0.0, 0.0), (1.0, 0.0), (5.0, 0.0)], [0.9, 0.8, 0.7], 0.5) == [0, 1, 2]
    assert R.bev_nms([(0.0, 0.0), (1.0, 0.0), (5.0, 0.0)], [0.7, 0.8, 0.9], 2.0) == [2, 1]


def test_top_k_by_score_respects_the_mask_and_exclusions():
    assert R.top_k_by_score([1.0, 5.0, 3.0, 4.0], [True, False, True, True], 2) == [3, 2]
    assert R.top_k_by_score([1.0, 5.0, 3.0, 4.0], [True, True, True, True], 2, exclude=[1]) == [3, 2]


def test_fan_spread_literal():
    assert R.fan_spread_m([(0.0, 0.0), (3.0, 4.0)]) == pytest.approx(5.0, abs=1e-12)
    assert R.fan_spread_m([(0.0, 0.0), (3.0, 4.0), (100.0, 0.0)], keep=[True, True, False]) == pytest.approx(5.0)
    assert R.fan_spread_m([(1.0, 1.0)]) == 0.0


def test_bits():
    assert R.bits([True, False, True]) == "101"


# ------------------------------------------------------------------------------------------- #
# colour semantics: one rule everywhere                                                       #
# ------------------------------------------------------------------------------------------- #
def test_gt_green_is_not_close_to_any_model_colour():
    gt = np.array(R.C_GT, float)
    for name in ("C_PLAN", "C_DEC", "C_BOX", "C_GOAL", "FAN_HI", "FAN_LO"):
        c = np.array(getattr(R, name), float)
        assert np.abs(c - gt).sum() > 150, name
    # the whole fan ramp stays away from the GT green and the plan orange
    for t in np.linspace(0, 1, 11):
        c = np.array(R.lerp_colour(R.FAN_HI, R.FAN_LO, t), float)
        assert np.abs(c - gt).sum() > 120 and np.abs(c - np.array(R.C_PLAN, float)).sum() > 200


def test_ceiling_stamp_is_the_briefed_text():
    assert R.CEIL_STAMP == "speed ceiling not applied to the emitted plan (SPEC_REFCV7 26.1)"


# ------------------------------------------------------------------------------------------- #
# the page: syntax + wiring                                                                   #
# ------------------------------------------------------------------------------------------- #
def _js() -> str:
    with open(_JS, encoding="utf-8") as fh:
        return fh.read()


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_replay_js_parses():
    r = subprocess.run(["node", "--check", _JS], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_every_layer_key_has_a_control_and_every_control_has_a_key():
    js = _js()
    block = re.search(r"L: \{(.*?)\n  \},\n\};", js, re.S).group(1)
    keys = set(re.findall(r"\b([a-z_0-9]+):", block))
    keys -= {"cls"}                                                # the 8 class chips are built from M.mapcls
    wired = set(re.findall(r"add(?:Check|Range|Select)\([a-zA-Z]+, '([a-z_0-9]+)'", js))
    wired |= {"view"}                                              # the radio group
    assert keys == wired, (sorted(keys - wired), sorted(wired - keys))


def test_page_loads_its_data_by_script_tags_not_fetch():
    html = open(_HTML, encoding="utf-8").read()
    js = _js()
    assert 'src="data/manifest.js"' in html and 'src="replay.js"' in html
    assert not re.search(r"(?<![A-Za-z_])fetch\(", js) and "XMLHttpRequest" not in js


def test_every_planner_overlay_carries_the_ceiling_stamp():
    js = _js()
    assert js.count("planStamp(w)") >= 3                           # BEV x2 code paths + camera
    assert "CEIL_STAMP" in js and "w.pex" in js


def test_gt_only_view_hides_model_layers_and_the_reverse():
    js = _js()
    assert "gt: v !== 'model', model: v !== 'gt'" in js


# ------------------------------------------------------------------------------------------- #
# the asset compaction                                                                        #
# ------------------------------------------------------------------------------------------- #
def _row(valid):
    inter = [[0, 0]] * 8
    return {
        "clip_rank": 1, "win": 0, "t_label_s": 1.0, "v0_ms": 5.0, "nav_cmd": "left", "v_max_raw_ms": 6.0,
        "v_max_valid": 1.0, "ceil_kmh": 30, "v_lim_ms": 8.3, "traj": [[1, 0]] * 8, "sel_idx": 3, "core_sel_idx": 3,
        "fan_b64": "AAAA", "e9": [0.0], "dec": [0.0], "reach": "1", "ceilk": "1", "navc": "0", "navc_inf": True,
        "vmx": [1.0], "gdist": [1.0], "plan_vmax_ms": 5.0, "plan_exceeds_ceiling": False,
        "gt_dense": [[1.0, 0.0]] * 60, "gt_dense_valid": valid, "tac_lat_p": [1.0] * 8, "tac_lon_p": [1.0] * 8,
        "tac_lat_gt": None, "tac_lon_gt": None, "goal_p": [0.0] * 22, "goal_conf": [0.0] * 22, "goal_gt_pos": [],
        "goal_scored": "0" * 22, "g_tac": [[0, 0, 0, 0]] * 3, "g_tac_gt": [[0, 0, 0, 0]] * 3,
        "g_tac_gt_valid": [True, True, True],
        "plan3": {"lat3": 0, "lon3": 1, "lat_ok": True, "lon_ok": None, "nav_ok": False},
        "ade_m": 1.0, "fde_m": 2.0, "speed_mae_0_2s": 0.1, "heading_mae_0_2s_deg": 0.2, "curv_mae_0_2s": 0.0,
        "map_inter": inter, "map_union": [[3, 4]] * 8, "n_map_scored": 7,
        "pred": {"idx": [], "p": [], "box": [], "yaw": [], "cz": [], "h": [], "cls": []},
        "gt": {"box": [], "yaw": [], "cls": [], "cz": [], "h": [], "zh": [], "pos": [], "ign": [], "hidden": []},
        "det": {"label": False},
    }


def test_compact_window_keeps_the_valid_prefix_and_sums_bands():
    w = R.compact_window(_row([True] * 40 + [False] * 20))
    assert len(w["gtd"]) == 40
    assert w["m"]["un"] == [7] * 8 and w["m"]["it"] == [0] * 8       # 3 + 4 summed over the two bands
    assert w["p3"] == {"lat3": 0, "lon3": 1, "latok": True, "lonok": None, "navok": False}
    assert w["dt"] == {"lab": False, "np": 0, "pairs": [], "tp0": 0, "n0": 0}


def test_compact_window_refuses_a_non_prefix_validity_mask():
    with pytest.raises(SystemExit):
        R.compact_window(_row([True] * 10 + [False] * 5 + [True] * 45))


def test_test_vector_probe_is_deterministic_and_clip_specific():
    a, b = R.test_vector_probe(0), R.test_vector_probe(0)
    assert a.shape == (68, 3) and np.array_equal(a, b)
    assert not np.array_equal(a, R.test_vector_probe(1))
    assert a[:60, 0].min() >= 2.0 and a[:60, 0].max() <= 90.0
