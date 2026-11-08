"""Tests for stream A priorities 2 and 3 (v10 tactical labels, strategic tokens)."""
import difflib
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_strategic_tok as T  # noqa: E402
import build_v9_verbatim as B9  # noqa: E402

OUT = os.environ.get("BAND_OUT", "F:/strat_out/band_labels")
V9_EVAL = "E:/Projects/TanitAD-artifacts/v9labels/v9_labels_eval139.npz"


def drive(speed, yawrate, t_end=60.0, hz=100.0):
    """Integrate a log from callables speed(t) [m/s] and yawrate(t) [rad/s]."""
    ts = np.arange(0.0, t_end + 1e-9, 1.0 / hz)
    v = np.array([speed(t) for t in ts])
    w = np.array([yawrate(t) for t in ts])
    yaw = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) / hz)])
    x = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] * np.cos(yaw[1:]) + v[:-1] * np.cos(yaw[:-1])) / hz)])
    y = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] * np.sin(yaw[1:]) + v[:-1] * np.sin(yaw[:-1])) / hz)])
    return B9.log_from_arrays(ts, x, y, yaw, v)


@pytest.fixture(scope="module")
def emit():
    S2 = T.emitter()
    G, A = T.vocab()
    return S2, G, A


def test_vocab_is_the_frozen_v7_set(emit):
    _, G, A = emit
    assert len(G) == 8 and len(A) == 7 and G[0] == "FOLLOW_ROUTE" and A[0] == "HOLD_MAIN_ROAD"


def test_band_override_is_6_30(emit):
    S2, _, _ = emit
    assert S2.STRATEGIC_S == (6.0, 30.0) and S2.GAP_S == (6.0, 6.0) and S2.TACTICAL_S == (2.0, 6.0)


def test_straight_is_follow_route_and_absence_needs_full_band(emit):
    S2, G, A = emit
    log = drive(lambda t: 12.0, lambda t: 0.0, t_end=60.0)
    f = T.frame_tokens(S2, log, 5.0, G, A)
    assert G[f["g"]] == "FOLLOW_ROUTE" and A[f["a"]] == "HOLD_MAIN_ROAD" and f["gv"] and f["av"]
    f = T.frame_tokens(S2, log, 45.0, G, A)               # only 15 s observed: an absence claim is not made
    assert G[f["g"]] == "FOLLOW_ROUTE" and not f["gv"] and not f["av"] and abs(f["obs"] - 15.0) < 0.2
    f = T.frame_tokens(S2, log, 58.0, G, A)               # < 3 s observed: nothing claimed
    assert f["g"] == -1 and not f["gv"]


def _turn_log(t_turn, v=4.0, R=12.0, dyaw=np.pi / 2):
    dur = dyaw * R / v
    return drive(lambda t: v, lambda t: (v / R) if t_turn <= t < t_turn + dur else 0.0, t_end=60.0)


def test_turn_in_the_6_to_8s_span_is_now_a_strategic_token(emit):
    """The 6-8 s span: the v7.2 emitter (8, 30) cannot see a turn that starts at NOW + 6.5; the (6, 30) band does."""
    S2, G, A = emit
    log = _turn_log(6.2, R=4.0)                                 # 90 deg in 1.57 s, wholly inside 6.2 - 7.8 s
    f = T.frame_tokens(S2, log, 0.0, G, A)
    assert G[f["g"]] == "TURN_LEFT_FOLLOW_ROUTE" and 6.0 <= f["by"] < 8.0 and f["gv"]
    assert abs(f["within"] - 4.0 * f["by"]) < 1.0              # arc to the turn start at 4 m/s
    S2.STRATEGIC_S, S2.GAP_S = (8.0, 30.0), (6.0, 8.0)         # the v7.2 bands: the same frame has no strategic token
    try:
        p = np.stack(B9.sample(log, np.arange(0, 30.01, 0.1)), axis=1)
        g0, _ = S2.strategic(p, 0, S2.manoeuvre_sequence(p, 0, hz=10.0), 10.0)
        assert g0["token"] == "FOLLOW_ROUTE"
    finally:
        S2.STRATEGIC_S, S2.GAP_S = T.BAND, (T.BAND[0], T.BAND[0])


def test_right_turn_and_time_of_turn_tracks_now(emit):
    S2, G, A = emit
    log = _turn_log(20.0, dyaw=np.pi / 2)
    log_r = drive(lambda t: 4.0, lambda t: -(4.0 / 12.0) if 20.0 <= t < 20.0 + np.pi / 2 * 3.0 else 0.0)
    f = T.frame_tokens(S2, log_r, 4.0, G, A)
    assert G[f["g"]] == "TURN_RIGHT_FOLLOW_ROUTE" and abs(f["by"] - 16.0) < 0.7
    f2 = T.frame_tokens(S2, log_r, 8.0, G, A)                  # NOW moves 4 s later: the turn is 4 s closer
    assert abs(f2["by"] - (f["by"] - 4.0)) < 0.3


def test_stop_in_band(emit):
    S2, G, A = emit
    log = drive(lambda t: 10.0 if t < 20.0 else (0.0 if t < 40 else 10.0), lambda t: 0.0)
    f = T.frame_tokens(S2, log, 0.0, G, A)
    assert G[f["g"]] == "STOP_AT_FOLLOW_ROUTE" and A[f["a"]] == "PREPARE_STOP_FOLLOW_ROUTE" and 18.0 <= f["by"] <= 22.0


# ------------------------------------------------------------------------------------------------ v10 builder
def test_v10_builder_differs_from_v9_only_in_the_listed_lines():
    a = open(os.path.join(HERE, "build_v9_verbatim.py"), encoding="utf-8").read().splitlines()
    b = open(os.path.join(HERE, "build_v10_labels.py"), encoding="utf-8").read().splitlines()
    changed = [l for l in difflib.unified_diff(a, b, lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    assert len(changed) == 26, len(changed)                    # 11 hunks, 11 lines out, 15 in: the band edits plus the 3-line one-point-grid guard of the 4,572-clip builder; see SCHEMA.md
    added = " ".join(l for l in changed if l[0] == "+")
    for must in ('"band_s": (2.0, 6.0)', '"h_abs_min_s": 6.0', "j_end", "tanitad.v10_labels/1"):
        assert must in added


@pytest.mark.parametrize("split", ["eval139", "train"])
def test_v10_schema_pin_and_literals(split):
    p = f"{OUT}/v10/v10_labels_{split}.npz"
    if not os.path.exists(p):
        pytest.skip("not built yet")
    man = json.load(open(f"{OUT}/v10/v10_labels_{split}.manifest.json", encoding="utf-8"))
    ref = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "v9_labels_train_4572.manifest.json") if split == "train" else "E:/Projects/TanitAD-artifacts/v9labels/v9_labels_%s.manifest.json" % split, encoding="utf-8"))
    assert man["schema"] == "tanitad.v10_labels/1"
    assert man["fields"] == ref["fields"]                        # same schema: the existing join reads it unchanged
    assert man["literals"]["band_s"] == [2.0, 6.0] and man["literals"]["h_abs_min_s"] == 6.0
    assert {k: v for k, v in man["literals"].items() if k not in ("band_s", "h_abs_min_s")} == \
           {k: v for k, v in ref["literals"].items() if k not in ("band_s", "h_abs_min_s")}
    assert man["vocab"] == ref["vocab"] and man["sources"]["lead"] == ref["sources"]["lead"]
    assert B9.file_md5(p) == man["npz_md5"]
    z = np.load(p)
    assert set(z.files) == set(man["fields"])


def test_control_rebuild_of_v9_is_bit_identical_to_the_banked_eval139():
    c = "F:/TanitAD-scratch/hwm_bands/a/ctl/x/v9_labels_eval139.npz"
    if not os.path.exists(c):
        pytest.skip("control not built")
    assert B9.file_md5(c) == B9.file_md5(V9_EVAL) == "6b5c7f207cffc3b7eb3cd527fd433599"


def test_band_end_is_enforced_in_v10_eval139():
    """Mutation guard: under band (2, 6) no row's reversing mask or v_end reads past NOW + 6 s: v_end_ms is the speed at
    NOW + 6 s (v9: NOW + 8 s)."""
    p = f"{OUT}/v10/v10_labels_eval139.npz"
    if not os.path.exists(p):
        pytest.skip("not built yet")
    z = np.load(p)
    z9 = np.load(V9_EVAL)
    assert np.nanmax(z["row__h_obs_s"]) <= 6.0 + 1e-9 and np.nanmax(z9["row__h_obs_s"]) <= 8.0 + 1e-9
    assert np.nanmax(z9["row__h_obs_s"]) > 6.5
