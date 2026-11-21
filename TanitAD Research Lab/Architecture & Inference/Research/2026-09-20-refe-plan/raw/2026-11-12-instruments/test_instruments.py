"""Tests for the W-TRANSP / W-FAM instruments.

Unit tests need only numpy + scipy (a fake ego state). `test_plan_views_*` need the REFe `refe/` dir on REFE_DIR (navsim_dac).
`test_banked_controls_pass` runs validate_controls.py on banked rows (env INSTR_ROWS = glob of rows, INSTR_TF1 = glob(s) of TF-1 rows,
INSTR_P5S = p5s_cubic.npz) and is skipped when they are not set.

  C:/Users/Admin/navsim-crun/venv/Scripts/python.exe -m pytest -q test_instruments.py
"""
import hashlib
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import transp_score as T  # noqa: E402


def fake_ego(v=10.0, w=0.0, x=100.0, y=-50.0, h=0.7):
    return SimpleNamespace(
        dynamic_car_state=SimpleNamespace(rear_axle_velocity_2d=SimpleNamespace(x=v, y=0.0), angular_velocity=w),
        rear_axle=SimpleNamespace(x=x, y=y, heading=h))


def test_transparent_straight_line_is_exact():
    ego = fake_ego(v=10.0)
    plan = np.stack([10.0 * 0.2 * np.arange(1, 21), np.zeros(20), np.zeros(20)], 1)
    s0 = np.zeros(11)
    st = T.transparent_states(plan, ego, s0, 0.2)
    assert st.shape == (41, 11)
    t = 0.1 * np.arange(41)
    ch, sh = np.cos(0.7), np.sin(0.7)
    np.testing.assert_allclose(st[1:, 0], 100.0 + 10.0 * t[1:] * ch, atol=1e-9)
    np.testing.assert_allclose(st[1:, 1], -50.0 + 10.0 * t[1:] * sh, atol=1e-9)
    np.testing.assert_allclose(st[1:, 3], 10.0, atol=1e-9)           # body-frame velocity
    np.testing.assert_allclose(st[1:, 9], 0.0, atol=1e-9)            # yaw rate
    assert (st[0] == s0).all()                                       # t = 0 row is the logged ego state, untouched


def test_transparent_dt_parameter_is_consistent():
    """The same constant-yaw-rate arc given at dt 0.2 (20 poses) and dt 0.5 (8 poses) builds the same states up to the spline error."""
    ego = fake_ego(v=8.0, w=0.1)
    R = 8.0 / 0.1

    def arc(ts):
        th = 0.1 * ts
        return np.stack([R * np.sin(th), R * (1 - np.cos(th)), th], 1)

    a = T.transparent_states(arc(0.2 * np.arange(1, 21)), ego, np.zeros(11), 0.2)
    b = T.transparent_states(arc(0.5 * np.arange(1, 9)), ego, np.zeros(11), 0.5)
    assert np.abs(a[:, :3] - b[:, :3]).max() < 3e-2                # 8 coarse poses: ~1.5 cm spline error is expected
    assert np.abs(a[1:, 9] - 0.1).max() < 5e-3                       # yaw rate ~ 0.1 rad/s


def test_world_to_ego8_roundtrip():
    ego = fake_ego()
    st = np.zeros((41, 11))
    ra = ego.rear_axle
    for k in range(41):
        d = 0.5 * k
        st[k, 0], st[k, 1], st[k, 2] = ra.x + d * np.cos(ra.heading), ra.y + d * np.sin(ra.heading), ra.heading
    e = T.world_to_ego8(st, ego)
    np.testing.assert_allclose(e[:, 0], 0.5 * np.arange(5, 41, 5)[:8], atol=1e-9)
    np.testing.assert_allclose(e[:, 1:], 0.0, atol=1e-9)


def test_shards_are_a_partition_and_stable():
    toks = [f"{i:016x}" for i in range(5000)]
    for sn in (2, 3, 5):
        owner = {t: int(hashlib.sha1(t.encode()).hexdigest(), 16) % sn for t in toks}
        assert sorted(set(owner.values())) == list(range(sn))
        assert sum(1 for t in toks if owner[t] == 0) > 0.8 * len(toks) / sn


def test_resume_skips_done_and_retries_errors(tmp_path):
    p = tmp_path / "rows.jsonl"
    p.write_text(json.dumps({"token": "a", "arms": {}}) + "\n" + json.dumps({"token": "b", "error": "x"}) + "\n" + "{broken\n")
    assert T.done_tokens(str(p)) == {"a"}


@pytest.mark.skipif(not os.environ.get("REFE_DIR"), reason="REFE_DIR (refe/ with navsim_dac.py) not set")
def test_plan_views_a7_and_shift():
    sys.path.insert(0, os.environ["REFE_DIR"])
    p20 = np.zeros((20, 3))
    p20[:, 0] = np.arange(1, 21)
    p20[-1, 2] = 0.5
    tr8, (rel, dt) = T.plan_views({"plan20": p20, "plan8": None, "log": "x"}, a7=True, shift_lat=0.0)
    assert dt == 0.2 and rel[-1, 2] == 0.0 and tr8.shape == (8, 3)
    tr8s, (rels, _) = T.plan_views({"plan20": p20, "plan8": None, "log": "x"}, a7=False, shift_lat=0.5)
    assert np.allclose(rels[:, 1], 0.5) and rels[-1, 2] == 0.5            # no A7 by default: the plan is scored as given
    assert np.allclose(tr8s[:, 1], 0.5)
    tr8b, (relb, dtb) = T.plan_views({"plan20": None, "plan8": tr8, "log": "x"}, a7=False, shift_lat=0.0)
    assert dtb == 0.5 and relb.shape == (8, 3) and (tr8b == tr8).all()    # plans8-only: transparent builder runs at 0.5 s


@pytest.mark.skipif(not os.environ.get("INSTR_ROWS"), reason="INSTR_ROWS not set")
def test_banked_controls_pass(tmp_path):
    cmd = [sys.executable, os.path.join(HERE, "validate_controls.py"), "--rows", os.environ["INSTR_ROWS"], "--tf1",
           *os.environ["INSTR_TF1"].split(";"), "--out", str(tmp_path / "controls.json")]
    if os.environ.get("INSTR_P5S"):
        cmd += ["--p5s", os.environ["INSTR_P5S"]]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout + r.stderr
