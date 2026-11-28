"""X10 -- the POSE-TO-IMAGE timing correction: known values, zero-offset identity, a deliberate regression.

Every expected value below is a LITERAL worked by hand from the stated numbers -- never an expression over the
code under test (a check that shares the defect it checks is green forever; CLAUDE.md, 2026-09-07).

KNOWN-VALUE CLIPS
  * constant velocity v = 12 m/s sampled every DT = 0.1006666 s; shifted by exactly 16.5 ms: absolute position
    moves by v * 0.0165 = 0.198 m, and the ego-frame waypoint TARGETS do not move at all (translation
    invariance -- the reason the defect is second-order for a displacement target).
  * constant acceleration a = 2 m/s^2: the h = 10 tick waypoint moves by a * delta * H with H = 10 * DT:
    2 * 0.0165 * 1.006666 = 0.033220 m.  (Linear interpolation of a quadratic errs by 0.5 a f (1-f) DT^2, the
    SAME at the anchor and at the waypoint, so it cancels in the displacement.)
  * the grid replica: a 25 fps camera from t = 0 over exactly 20 s, 10 Hz grid => n_target 200, dt_q =
    20e6/199 us, delta[1..3] = 19497.4874, 38994.9749, 18492.4623 us (hand-worked, see test).

DELIBERATE REGRESSION: the sign of delta flipped must go RED on the very check that passes with the right sign.
ZERO OFFSET: an all-zero sidecar must be BIT-IDENTICAL to the path with the correction OFF.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tanitad.data import pose_sync as PS  # noqa: E402
from tanitad.data.v2_dataset import stable_episode_id  # noqa: E402

DT = 0.1006666          # s per row (the corpus median, MEASURED)
V = 12.0                # m/s
LIT_CV_SHIFT_M = 0.198  # = 12 * 0.0165, worked by hand
LIT_ACC_DELTA_M = 0.033220   # = 2 * 0.0165 * (10 * 0.1006666)
ROWS = 130
W = 8
N_RAW = ROWS + 2        # n_stack 3 => raw_offset 2


# --------------------------------------------------------------------------------------------- fixtures
def _cv_poses(n=ROWS, v=V):
    r = np.arange(n, dtype=np.float64)
    p = np.zeros((n, 4))
    p[:, 0], p[:, 3] = v * DT * r, v
    return p.astype(np.float32)


def _acc_poses(n=ROWS, a=2.0, v0=5.0):
    t = DT * np.arange(n, dtype=np.float64)
    p = np.zeros((n, 4))
    p[:, 0] = v0 * t + 0.5 * a * t * t
    p[:, 3] = v0 + a * t
    return p.astype(np.float32)


def _write(tmp: Path, rows, name="sync.jsonl"):
    p = tmp / name
    PS.write_pose_sync_sidecar(str(p), rows, {"test": True})
    return str(p)


def _const_rows(sids, delta_us, n_raw=N_RAW):
    return [{"sid": s, "dt_s": DT, "delta_us": [delta_us] * n_raw} for s in sids]


def _ego_frame_wp(poses, now, h):
    """numpy ego-frame waypoint at tick h -- written out, NOT imported (refb_labels.waypoint_targets is checked
    against this in the dataset-level test)."""
    d = poses[now + h, :2].astype(np.float64) - poses[now, :2].astype(np.float64)
    c, s = np.cos(-float(poses[now, 2])), np.sin(-float(poses[now, 2]))
    return np.array([d[0] * c - d[1] * s, d[0] * s + d[1] * c])


def _known_value_check(ps: PS.PoseSync) -> None:
    """The check whose sign-flipped twin must go RED (the deliberate-regression arm)."""
    P0 = _cv_poses()
    got = PS.window_shifted_tracks(ps, 7, 20, 2, P0)
    assert got is not None
    P1 = got[0]
    np.testing.assert_allclose(P1[:, 0].astype(np.float64) - P0[:, 0].astype(np.float64),
                               LIT_CV_SHIFT_M, atol=2e-6)


# ------------------------------------------------------------------------------------ known values
def test_constant_velocity_shift_is_exactly_v_times_dt(tmp_path):
    ps = PS.read_pose_sync_sidecar(_write(tmp_path, _const_rows([7], 16500)))
    _known_value_check(ps)                      # +0.198 m on EVERY row (interior and the extrapolated tail)


def test_constant_velocity_targets_do_not_move(tmp_path):
    """Translation invariance: the displacement target of a constant-velocity track is untouched, to 1e-6 m."""
    ps = PS.read_pose_sync_sidecar(_write(tmp_path, _const_rows([7], 16500)))
    P0 = _cv_poses()
    P1 = PS.window_shifted_tracks(ps, 7, 20, 2, P0)[0]
    for h in (5, 10, 20, 60):
        np.testing.assert_allclose(_ego_frame_wp(P1, 20, h), _ego_frame_wp(P0, 20, h), atol=2e-6)
        assert _ego_frame_wp(P0, 20, h)[0] == pytest.approx(V * DT * h, abs=1e-4)        # sanity: literal geometry


def test_constant_acceleration_waypoint_moves_by_a_delta_H(tmp_path):
    ps = PS.read_pose_sync_sidecar(_write(tmp_path, _const_rows([7], 16500)))
    P0 = _acc_poses()
    P1 = PS.window_shifted_tracks(ps, 7, 20, 2, P0)[0]
    d = _ego_frame_wp(P1, 20, 10)[0] - _ego_frame_wp(P0, 20, 10)[0]
    assert d == pytest.approx(LIT_ACC_DELTA_M, abs=2e-5)     # 0.033220 m, hand-worked above


def test_yaw_is_shifted_on_the_shortest_arc_across_the_seam(tmp_path):
    n = ROWS
    r = np.arange(n, dtype=np.float64)
    w = 0.2                                                   # rad/s
    yaw0 = 3.1 - 0.5 * DT * 0 + w * DT * r                   # starts at 3.1 rad, crosses +pi at row ~0.45
    yaw = np.arctan2(np.sin(yaw0), np.cos(yaw0))
    P0 = np.zeros((n, 4)); P0[:, 2] = yaw; P0[:, 3] = 1.0
    P0 = P0.astype(np.float32)
    ps = PS.read_pose_sync_sidecar(_write(tmp_path, _const_rows([7], 20000)))
    P1 = PS.window_shifted_tracks(ps, 7, 20, 2, P0)[0]
    dy = np.arctan2(np.sin(P1[:, 2].astype(np.float64) - P0[:, 2].astype(np.float64)),
                    np.cos(P1[:, 2].astype(np.float64) - P0[:, 2].astype(np.float64)))
    assert dy[:-1].max() == pytest.approx(0.2 * 0.02, abs=2e-6) and dy[:-1].min() == pytest.approx(0.004, abs=2e-6)
    assert np.all(np.abs(P1[:, 2]) <= np.pi + 1e-6)


# --------------------------------------------------------------------------- zero offset = identity
def test_zero_offset_is_bit_identical(tmp_path):
    ps = PS.read_pose_sync_sidecar(_write(tmp_path, _const_rows([7], 0)))
    P0 = _acc_poses()
    P0[:, 2] = np.random.default_rng(3).uniform(-np.pi, np.pi, ROWS).astype(np.float32)   # a NON-trivial yaw: the seam must not re-round it
    A0 = np.stack([np.linspace(-0.1, 0.1, ROWS), np.linspace(-2, 2, ROWS)], 1).astype(np.float32)
    P1, A1, s = PS.window_shifted_tracks(ps, 7, 20, 2, P0, A0)
    assert s == 0.0
    assert P1.dtype == np.float32 and A1.dtype == np.float32
    assert P1.tobytes() == P0.tobytes() and A1.tobytes() == A0.tobytes()


def test_grid_replica_matches_hand_worked_literals():
    t_frames = np.arange(0, 20_000_001, 40_000, dtype=np.float64)       # 25 fps, exactly 20 s, 501 frames
    d, dt, n = PS.grid_delta(t_frames)
    assert n == 200 and dt == pytest.approx(20.0 / 199, abs=1e-12)
    assert d[0] == 0.0
    assert d[1] * 1e6 == pytest.approx(19497.4874, abs=1e-3)           # 120000 - 100502.5126
    assert d[2] * 1e6 == pytest.approx(38994.9749, abs=1e-3)           # 240000 - 201005.0251
    assert d[3] * 1e6 == pytest.approx(18492.4623, abs=1e-3)           # 320000 - 301507.5377
    assert d.min() >= 0.0 and d.max() * 1e6 < 40_000.0                 # image never BEFORE its pose; < 1 period
    # the closed form, derived from four numbers, agrees to < 1 us on an ideal camera
    da = PS.analytic_delta_s(t_frames[0], t_frames[-1], len(t_frames), n)
    assert np.abs(d - da).max() * 1e6 < 1.0


# ---------------------------------------------------------------------- deliberate regression arms
def test_REGRESSION_flipped_sign_goes_red_on_the_known_value_check(tmp_path):
    good = PS.read_pose_sync_sidecar(_write(tmp_path, _const_rows([7], 16500)), sign=+1.0)
    bad = PS.read_pose_sync_sidecar(_write(tmp_path, _const_rows([7], 16500), "sync2.jsonl"), sign=-1.0)
    _known_value_check(good)                                  # passes
    with pytest.raises(AssertionError):                       # the SAME check must fail on the flipped sign
        _known_value_check(bad)
    # and the flip is not a no-op: it moves the track the OTHER way by the same amount
    P0 = _cv_poses()
    Pb = PS.window_shifted_tracks(bad, 7, 20, 2, P0)[0]
    assert (Pb[10:-3, 0].astype(np.float64) - P0[10:-3, 0].astype(np.float64)) == pytest.approx(-LIT_CV_SHIFT_M, abs=2e-6)


def test_REGRESSION_per_row_shift_injects_first_order_timing_jitter():
    """The literal 'interpolate each row to ITS camera timestamp' is NOT the correction.  A constant-velocity track
    with a sawtooth delta moves every waypoint by v * (delta_{r+h} - delta_r): here 12 m/s * 20 ms = 0.24 m."""
    P0 = _cv_poses()
    delta_rows = np.zeros(ROWS)
    delta_rows[30:] = 0.020 / DT                              # a 20 ms step in delta between the anchor and the target
    Pa = PS.shift_rows_per_row(P0, delta_rows, angle_cols=(2,))
    wp_a = _ego_frame_wp(Pa, 20, 20)[0]                       # anchor row 20 (delta 0), waypoint row 40 (delta 20 ms)
    wp_0 = _ego_frame_wp(P0, 20, 20)[0]
    assert wp_a - wp_0 == pytest.approx(12.0 * 0.020, abs=1e-5)     # 0.24 m of target noise (first order)
    ps = PS.PoseSync(table={7: PS.SyncRow(delta_s=np.full(N_RAW, 0.0165), dt_s=DT)})
    Pb = PS.window_shifted_tracks(ps, 7, 20, 2, P0)[0]        # the window shift: zero target noise on the same track
    assert _ego_frame_wp(Pb, 20, 20)[0] - wp_0 == pytest.approx(0.0, abs=2e-6)


# ------------------------------------------------------------------------------------------ refusals
@pytest.mark.parametrize("row,why", [
    ({"sid": 1, "dt_s": DT, "n_rows": 5, "delta_us": [0, 1, 2, 3, 4]}, "ok-shape-but-see-below"),
])
def test_reader_accepts_a_wellformed_row(tmp_path, row, why):
    p = tmp_path / "ok.jsonl"
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert PS.read_pose_sync_sidecar(str(p)).n_rows == 1


@pytest.mark.parametrize("row", [
    {"sid": 1, "dt_s": DT, "n_rows": 5, "delta_us": [0, 1, -1, 3, 4]},          # negative: a sign error
    {"sid": 1, "dt_s": DT, "n_rows": 5, "delta_us": [0, 1, 2, 3, 400000]},      # > 50 ms: a unit error
    {"sid": 1, "dt_s": DT, "n_rows": 6, "delta_us": [0, 1, 2, 3, 4]},           # length != n_rows
    {"sid": 1, "dt_s": 0.5, "n_rows": 5, "delta_us": [0, 1, 2, 3, 4]},          # dt outside the cache grid's band
    {"dt_s": DT, "n_rows": 5, "delta_us": [0, 1, 2, 3, 4]},                      # no sid
])
def test_reader_refuses_untrustworthy_rows(tmp_path, row):
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(PS.PoseSyncError):
        PS.read_pose_sync_sidecar(str(p))


def test_reader_refuses_missing_empty_and_conflicting_duplicates(tmp_path):
    with pytest.raises(PS.PoseSyncError):
        PS.read_pose_sync_sidecar(str(tmp_path / "nope.jsonl"))
    e = tmp_path / "empty.jsonl"
    e.write_text("", encoding="utf-8")
    with pytest.raises(PS.PoseSyncError):
        PS.read_pose_sync_sidecar(str(e))
    a = {"sid": 1, "dt_s": DT, "n_rows": 4, "delta_us": [0, 1, 2, 3]}
    b = dict(a, delta_us=[0, 1, 2, 9])
    d = tmp_path / "dup.jsonl"
    d.write_text(json.dumps(a) + "\n" + json.dumps(b) + "\n", encoding="utf-8")
    with pytest.raises(PS.PoseSyncError):
        PS.read_pose_sync_sidecar(str(d))


def test_uncovered_clip_returns_none_so_the_caller_can_count_it(tmp_path):
    ps = PS.read_pose_sync_sidecar(_write(tmp_path, _const_rows([7], 16500)))
    assert PS.window_shifted_tracks(ps, 8, 20, 2, _cv_poses()) is None             # unknown sid
    assert PS.window_shifted_tracks(ps, 7, N_RAW + 5, 0, _cv_poses()) is None      # row outside the sidecar


# ---------------------------------------------------------------- the DATASET level (V3Dataset hook)
def _episode(clip: str, poses: np.ndarray):
    actions = np.stack([np.linspace(0.0, 0.05, ROWS), np.linspace(0.0, 1.0, ROWS)], 1).astype(np.float32)
    return SimpleNamespace(frames=torch.zeros(ROWS, 9, 8, 8, dtype=torch.uint8),
                           actions=torch.from_numpy(actions), poses=torch.from_numpy(poses.copy()),
                           episode_id=stable_episode_id(clip))


def _dataset(eps):
    import refc_v3_train as T
    ds = T.V3Dataset(eps, window=W, max_horizon=20, channels=9)
    ds.v7_by_sid, ds.v7_dt = {}, 0.1
    return ds


def _same(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    return a == b


@pytest.fixture(scope="module")
def eps():
    return [_episode("clip-aaa", _acc_poses()), _episode("clip-bbb", _cv_poses())]


def test_OFF_path_is_untouched(eps):
    ds = _dataset(eps)
    assert ds.pose_sync is None
    it = ds[3]
    assert set(it) == set(_dataset(eps)[3])                    # no new key, no config
    assert "pose_sync" not in set(it)


def test_dataset_known_value_waypoint_shift_and_row_indices_unmoved(eps, tmp_path):
    import refb_labels as RL
    ds0 = _dataset(eps)
    ds1 = _dataset(eps)
    sidecar = _write(tmp_path, _const_rows([int(e.episode_id) for e in eps], 16500))
    rep = ds1.enable_pose_sync(sidecar)
    assert rep["n_covered"] == 2 and rep["n_uncovered_read_unshifted"] == 0 and rep["sign"] == 1.0
    i = 3                                                      # clip 0 (accelerating), t = 3 => NOW row 10
    a, b = ds0[i], ds1[i]
    h = torch.tensor([10])
    wa = RL.waypoint_targets(a["pose_last"][None], a["future_poses_ext"][None], (10,))[0, 0]
    wb = RL.waypoint_targets(b["pose_last"][None], b["future_poses_ext"][None], (10,))[0, 0]
    assert float(wb[0] - wa[0]) == pytest.approx(LIT_ACC_DELTA_M, abs=2e-5)
    assert float(wb[1] - wa[1]) == pytest.approx(0.0, abs=2e-5)
    # v0 = pose_last[3] moves by a * delta = 2 * 0.0165 = 0.033 m/s (the speed channel is linear in t)
    assert float(b["pose_last"][3] - a["pose_last"][3]) == pytest.approx(0.033, abs=2e-5)
    for k in ("pose_last", "future_poses", "future_poses_ext", "goal_tac", "actions", "future_actions"):
        assert a[k].shape == b[k].shape and a[k].dtype == b[k].dtype, k          # same contract, new values
    # nothing that is not a pose/action moved; the row/clock-keyed fields are untouched
    for k in a:
        if k not in ("pose_last", "future_poses", "future_poses_ext", "goal_tac", "actions", "future_actions",
                     "pose_hist"):
            assert _same(a[k], b[k]), k
    assert _same(a["goal_tac_valid"], b["goal_tac_valid"]) and _same(a["future_valid_ext"], b["future_valid_ext"])
    assert torch.equal(a["frames"], b["frames"])
    # the LABEL CLOCK and the (sid, k) key the v9 join checks to 1e-6 s are untouched by the correction
    for t_ in (0, 3, 40, 100):
        assert ds1._now_s(eps[0], t_) == ds0._now_s(eps[0], t_)
    assert ds1._raw_offset(eps[0]) == ds0._raw_offset(eps[0]) == 2


def test_dataset_uses_the_NOW_rows_own_delta(eps, tmp_path):
    """delta varies by row (1000 us per raw row, mod 30 ms): the window shift must be the NOW row's, and ONLY that.
    Literals (hand-worked): clip 0 accelerates at a = 2 m/s^2, H = 10 ticks = 1.006666 s.
      t = 3  => NOW provider row 10 => raw row 12 => delta 12000 us => a*delta*H = 2*0.012*1.006666 = 0.024160 m
      t = 40 => NOW provider row 47 => raw row 49 => 49000 mod 30000 = 19000 us => 2*0.019*1.006666 = 0.038253 m"""
    import refb_labels as RL
    sids = [int(e.episode_id) for e in eps]
    rows = [{"sid": s_, "dt_s": DT, "delta_us": [(1000 * k) % 30000 for k in range(N_RAW)]} for s_ in sids]
    ds0, ds1 = _dataset(eps), _dataset(eps)
    ds1.enable_pose_sync(_write(tmp_path, rows))
    for t, lit in ((3, 0.024160), (40, 0.038253)):
        a, b = ds0[t], ds1[t]
        wa = RL.waypoint_targets(a["pose_last"][None], a["future_poses_ext"][None], (10,))[0, 0]
        wb = RL.waypoint_targets(b["pose_last"][None], b["future_poses_ext"][None], (10,))[0, 0]
        assert float(wb[0] - wa[0]) == pytest.approx(lit, abs=3e-5), t


def test_dataset_zero_offset_is_bit_identical_to_OFF(eps, tmp_path):
    ds0, ds1 = _dataset(eps), _dataset(eps)
    ds1.enable_pose_sync(_write(tmp_path, _const_rows([int(e.episode_id) for e in eps], 0)))
    for i in (0, 3, 40, len(ds0) - 1, len(ds0) // 2 + 5):
        a, b = ds0[i], ds1[i]
        assert set(a) == set(b)
        for k in a:
            assert _same(a[k], b[k]), (i, k)


def test_dataset_REGRESSION_flipped_sign_moves_the_target_the_wrong_way(eps, tmp_path):
    import refb_labels as RL
    ds0, ds1 = _dataset(eps), _dataset(eps)
    ds1.enable_pose_sync(_write(tmp_path, _const_rows([int(e.episode_id) for e in eps], 16500)), sign=-1.0)
    a, b = ds0[3], ds1[3]
    wa = RL.waypoint_targets(a["pose_last"][None], a["future_poses_ext"][None], (10,))[0, 0]
    wb = RL.waypoint_targets(b["pose_last"][None], b["future_poses_ext"][None], (10,))[0, 0]
    assert float(wb[0] - wa[0]) == pytest.approx(-LIT_ACC_DELTA_M, abs=2e-5)       # the opposite of the right answer
    assert abs(float(wb[0] - wa[0]) - LIT_ACC_DELTA_M) > 0.06                       # i.e. the +0.033 check goes RED


def test_dataset_ego_history_rides_the_same_shift(eps, tmp_path):
    ds0, ds1 = _dataset(eps), _dataset(eps)
    ds0.ego_history = ds1.ego_history = True
    ds1.enable_pose_sync(_write(tmp_path, _const_rows([int(e.episode_id) for e in eps], 16500)))
    a, b = ds0[3], ds1[3]
    assert a["pose_hist"].shape == b["pose_hist"].shape == (W, 4)
    assert torch.equal(b["pose_hist"][-1], b["pose_last"])                          # the history ENDS at the NOW pose
    assert float((b["pose_hist"] - a["pose_hist"])[:, 3].min()) > 0.0               # all rows moved (accelerating clip)


def test_enable_refuses_a_sidecar_that_covers_nothing_or_another_timeline(eps, tmp_path):
    ds = _dataset(eps)
    with pytest.raises(SystemExit):
        ds.enable_pose_sync(_write(tmp_path, _const_rows([123456789], 16500), "other.jsonl"))
    with pytest.raises(SystemExit):                                                  # right sids, WRONG row count
        ds.enable_pose_sync(_write(tmp_path, _const_rows([int(e.episode_id) for e in eps], 16500, n_raw=N_RAW + 7),
                                   "rows.jsonl"))
    assert ds.pose_sync is None                                                      # a refusal leaves it OFF


def test_uncovered_clip_is_read_unshifted_and_counted(eps, tmp_path):
    ds0, ds1 = _dataset(eps), _dataset(eps)
    sids = [int(e.episode_id) for e in eps]
    rep = ds1.enable_pose_sync(_write(tmp_path, _const_rows(sids[:1], 16500)), max_uncovered_frac=0.6)
    assert rep["n_covered"] == 1 and rep["n_uncovered_read_unshifted"] == 1
    n0 = len([1 for e_i, _ in ds0.index if e_i == 0])
    a, b = ds0[n0 + 3], ds1[n0 + 3]                                                  # a clip-1 window: uncovered
    for k in a:
        assert _same(a[k], b[k]), k
    with pytest.raises(SystemExit):                                                  # and the DEFAULT cap (1 %) refuses it
        _dataset(eps).enable_pose_sync(_write(tmp_path, _const_rows(sids[:1], 16500), "cap.jsonl"))


# ------------------------------------------------------------------------------- the sidecar BUILDER
def _synthetic_corpus(tmp: Path, t_out: int):
    """One clip: a 25 fps camera from t = 0 over exactly 20 s (501 frames) => n_target 200, n_stack 3 => T_out 198."""
    import pandas as pd
    cid = "clip-synth-0001"
    cam = tmp / "cam"
    cam.mkdir()
    pd.DataFrame({"timestamp": np.arange(0, 20_000_001, 40_000, dtype=np.int64)}).to_parquet(cam / f"{cid}.timestamps.parquet")
    man = {"clip_id": [cid], "episode_uid": [stable_episode_id(cid)], "n_stack": [3], "T_out": [t_out]}
    torch.save(man, tmp / "_v2manifest.pt")
    return str(tmp / "_v2manifest.pt"), str(cam)


def test_builder_writes_the_hand_worked_deltas(tmp_path):
    import build_pose_sync_sidecar as B
    man, cam = _synthetic_corpus(tmp_path, 198)
    out = tmp_path / "o.jsonl"
    assert B.main(["--manifest", man, "--camera-dir", cam, "--out", str(out)]) == 0
    ps = PS.read_pose_sync_sidecar(str(out))
    row = ps.lookup(stable_episode_id("clip-synth-0001"))
    assert row.delta_s.shape == (200,) and row.dt_s == pytest.approx(20.0 / 199, abs=1e-12)
    assert [round(float(v) * 1e6) for v in row.delta_s[:4]] == [0, 19497, 38995, 18492]     # the hand-worked literals
    assert "clip-synth-0001" not in out.read_text(encoding="utf-8")                          # the clip-id rule
    assert PS.SCHEMA in (tmp_path / "o.jsonl.meta.json").read_text(encoding="utf-8")


def test_builder_refuses_a_timeline_that_is_not_the_caches(tmp_path):
    import build_pose_sync_sidecar as B
    man, cam = _synthetic_corpus(tmp_path, 190)                  # the cache has 190 rows; this camera grid has 198
    with pytest.raises(SystemExit):
        B.main(["--manifest", man, "--camera-dir", cam, "--out", str(tmp_path / "o.jsonl")])
    assert not (tmp_path / "o.jsonl").exists()                   # REFUSED = nothing written


def test_builder_refuses_when_the_independent_clock_disagrees(tmp_path):
    import build_pose_sync_sidecar as B
    man, cam = _synthetic_corpus(tmp_path, 198)
    ck = tmp_path / "clock.jsonl"
    ck.write_text(json.dumps({"sid": stable_episode_id("clip-synth-0001"), "grid_start_s": 0.1, "dt_s": 0.1006666}) + chr(10),
                  encoding="utf-8")
    with pytest.raises(SystemExit):                              # K4: 0.1006666 vs the rebuilt 20/199 = 0.100502...
        B.main(["--manifest", man, "--camera-dir", cam, "--out", str(tmp_path / "o2.jsonl"), "--clock-sidecar", str(ck)])
