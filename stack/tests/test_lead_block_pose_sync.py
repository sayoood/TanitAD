"""The B1 lead block on the POSE-SYNC clock (2026-11-12) -- ``taniteval/tools/build_lead_block_b1.py --pose-sync-sidecar``.

MEASURED 2026-11-12, the first real refcv9 E2 read on Thor: the distance-keeping join's label-free speed proof (``r9_e2_distkeep.load_arm``) REFUSED all
70 DEV episodes, max |dump v0 - block speed| 0.008-0.060 m/s per episode.  K0 and K2M1N20 train AND evaluate with the pose-sync sidecar
(``tanitad/data/pose_sync.py``): the dataset reads its ego state ``delta`` (0..35 ms) LATER than the pose row, so the dump's ``v0`` is not the block's
pose-clock speed.  The eval cache and the block agreed to <= 2e-5 m/s and every whole-frame offset was worse: the frame map was right, the clock was not.

Pinned here, CPU only, synthetic world with per-row deltas:
  (1) ``pose_sync_track`` IS the dataset's v0 -- equal, bit for bit, to ``V3Dataset[i]["pose_last"][3]`` with the sidecar enabled (the real hook);
  (2) the ZERO-SHIFT control: delta = 0 reproduces every column of the unshifted block exactly;
  (3) a shifted build: t0_s = t0_pose + s*dt_grid, speeds = the dataset's v0, speeds_egomotion = egomotion at the shifted instant;
  (4) the ACTUAL CALLER: ``r9_e2_distkeep.load_arm`` PASSES its unchanged 1e-3 m/s proof on the pose-synced block and REFUSES the unshifted one
      (the 2026-11-12 defect, reproduced);
  (5) REGRESSION ARM: a block built with the opposite sign is refused by the same proof;
  (6) guards: an uncovered clip is built unshifted; a v0 that the shifted instant does not reproduce refuses the build; main() refuses a sidecar
      without its sign / raw offset, and either of those without a sidecar.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]                     # <tree>/stack
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tanitad.data import pose_sync as PS  # noqa: E402
from tanitad.data.v2_dataset import stable_episode_id  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bb = _load("build_lead_block_b1_psync_under_test", REPO / "taniteval" / "tools" / "build_lead_block_b1.py")

DT = 0.1                    # grid spacing == sidecar dt_s (inside PS.DT_BAND_S)
OFF = 2                     # n_stack 3 => raw_offset 2
ROWS = 130                  # PROVIDER rows
N_RAW = ROWS + OFF
W = 8
CLIP = "clip-psync-aaa"
SID = int(stable_episode_id(CLIP))
K_DK, DT_DK = 4, 0.5        # the E2 roll's distance-keeping grid


def _world():
    """An accelerating, oscillating straight drive: a = 0.8 + 1.5*(2pi/4)*cos(2pi t/4) m/s^2, so every shift moves v0."""
    t = DT * np.arange(N_RAW, dtype=np.float64)
    v = 8.0 + 0.8 * t + 1.5 * np.sin(2 * np.pi * t / 4.0)
    x = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * DT)])
    p = np.zeros((N_RAW, 4))
    p[:, 0], p[:, 3] = x, v
    poses = p.astype(np.float32)
    pf = poses.astype(np.float64)
    ego = {"t": t, "x": pf[:, 0], "y": pf[:, 1], "yaw": pf[:, 2], "v": pf[:, 3]}
    return t, poses, ego


def _sidecar(tmp: Path, delta_us, name="sync.jsonl", sid=SID):
    p = tmp / name
    PS.write_pose_sync_sidecar(str(p), [{"sid": sid, "dt_s": DT, "delta_us": list(delta_us)}], {"test": True})
    return str(p)


def _sawtooth():
    return [(k * 1300) % 33000 for k in range(N_RAW)]               # per-row delta, 0..32.5 ms


def _dataset(poses_raw, sidecar):
    import refc_v3_train as T
    acts = np.stack([np.linspace(0.0, 0.05, ROWS), np.linspace(0.0, 1.0, ROWS)], 1).astype(np.float32)
    ep = SimpleNamespace(frames=torch.zeros(ROWS, 9, 8, 8, dtype=torch.uint8), actions=torch.from_numpy(acts),
                         poses=torch.from_numpy(poses_raw[OFF:].copy()), episode_id=SID)
    ds = T.V3Dataset([ep], window=W, max_horizon=20, channels=9)
    ds.v7_by_sid, ds.v7_dt = {}, 0.1
    if sidecar is not None:
        ds.enable_pose_sync(sidecar)
    assert ds._raw_offset(ep) == OFF
    return ds


def _dataset_v0(ds, stride=1):
    """[(raw row, v0)] for the dataset's windows: v0 = pose_last[3] at NOW = t + W - 1 (PROVIDER), raw = NOW + OFF."""
    out = []
    for i in range(0, len(ds.index), stride):
        _e, t = ds.index[i]
        out.append((int(t) + W - 1 + OFF, float(ds[i]["pose_last"][3])))
    return out


def _rows(tmp, sidecar_path, *, sign=1.0, k=K_DK, dt=DT_DK, sid=SID):
    t, poses, ego = _world()
    ps = PS.read_pose_sync_sidecar(sidecar_path, sign=sign)
    shift, v0s, info = bb.pose_sync_clip(ps, sid, poses, t, OFF)
    rows, cov = bb.build_clip_rows(CLIP, ego, t, None, k=k, dt=dt, obs_reason="synthetic", t0_shift_s=shift, v0_sync=v0s)
    return rows, cov, info, shift, v0s


def _plain_rows(k=K_DK, dt=DT_DK):
    t, poses, ego = _world()
    return bb.build_clip_rows(CLIP, ego, t, None, k=k, dt=dt, poses=poses, obs_reason="synthetic")


def _block(path: Path, rows, cov, k=K_DK, dt=DT_DK):
    bb.write_block(str(path), [rows], {CLIP: cov}, {"version": "test", "tool": "test"}, k=k, dt=dt)
    return path


def _dump(root: Path, pairs):
    d = root / "dump_s0"
    d.mkdir(parents=True)
    ws = np.asarray([r - OFF for r, _ in pairs], dtype=np.int64)
    v0 = np.asarray([v for _, v in pairs], dtype=np.float32)
    g = np.zeros((ws.size, K_DK, 2))
    g[..., 0] = v0[:, None].astype(np.float64) * DT_DK * np.arange(1, K_DK + 1)   # constant-speed GT on the published 0.5 s clock: the reader's
    # dt_check (r9_e2_families, 3624402e) refuses an all-zero GT for a moving ego (median |g[:, 0]| / v0 = 0 against grid.dt_s 0.5) -- MEASURED on the
    # Thor E2 suite 2026-11-13 at d0974ba6; the speed proof this file tests is unchanged
    np.savez(d / "ep000.npz", g=g, os=g, ws=ws, v0=v0, eid=np.asarray([0]))
    (d / "manifest.json").write_text(json.dumps({"episodes": [{"file_index": 0, "clip_id": CLIP, "n_windows": int(ws.size)}],
                                                 "grid": {"dt_s": DT_DK, "k": K_DK, "n_windows": int(ws.size)},
                                                 "corpus": {"frames": {"provider_to_raw_frame_offset": OFF}}}), encoding="utf-8")
    return d


# ---------------------------------------------------------------------------------------------------------------- (1)
def test_pose_sync_track_IS_the_datasets_v0_bit_for_bit(tmp_path):
    _t, poses, _e = _world()
    side = _sidecar(tmp_path, _sawtooth())
    pairs = _dataset_v0(_dataset(poses, side))
    v0s, sh = bb.pose_sync_track(PS.read_pose_sync_sidecar(side), SID, poses, OFF)
    assert len(pairs) > 50
    for raw, v in pairs:
        assert v0s[raw] == v, (raw, v0s[raw], v)                   # EXACT: the same function on the same track
    moved = [abs(v - float(poses[raw, 3])) for raw, v in pairs]
    assert max(moved) > 0.02                                        # the shift is not a no-op on this world
    assert np.isnan(v0s[:OFF]).all() and np.isfinite(v0s[OFF:]).all()
    assert np.nanmax(sh) * DT * 1e3 <= 33.0 and np.nanmin(sh) >= 0.0


# ---------------------------------------------------------------------------------------------------------------- (2)
def test_ZERO_shift_control_reproduces_every_column_of_the_unshifted_block(tmp_path):
    rows, cov, info, shift, _v = _rows(tmp_path, _sidecar(tmp_path, [0] * N_RAW))
    plain, _c = _plain_rows()
    assert info["covered"] and not shift.any()
    for k, a in plain.items():
        b = rows[k]
        assert a.dtype == b.dtype and a.shape == b.shape, k
        assert np.array_equal(a, b, equal_nan=a.dtype.kind == "f"), k
    assert set(rows) - set(plain) == {"t0_pose_s", "pose_sync_shift_s", "speeds_egomotion"}
    assert cov["pose_sync"]["max_abs_v0_sync_minus_10hz_resample_mps"] == 0.0
    assert cov["pose_sync"]["interp_10hz_vs_egomotion_mps"]["max"] == 0.0


# ---------------------------------------------------------------------------------------------------------------- (3)
def test_shifted_build_carries_the_shifted_instant_and_the_datasets_v0(tmp_path):
    t, _p, ego = _world()
    rows, cov, info, shift, v0s = _rows(tmp_path, _sidecar(tmp_path, _sawtooth()))
    assert np.array_equal(rows["t0_pose_s"], t)
    assert np.array_equal(rows["t0_s"], t + shift) and np.array_equal(rows["pose_sync_shift_s"], shift)
    assert np.array_equal(rows["speeds"][OFF:], v0s[OFF:])           # the dataset's v0, exactly
    assert np.array_equal(rows["speeds"][:OFF], rows["speeds_egomotion"][:OFF])
    assert np.allclose(rows["speeds_egomotion"], np.interp(rows["t0_s"], ego["t"], ego["v"]), atol=1e-12, rtol=0)
    ps = cov["pose_sync"]
    assert ps["n_rows_synced"] == N_RAW - OFF and ps["n_rows_unsynced"] == OFF
    # the gate's identity holds to float32 rounding; on this 10 Hz-linear world the egomotion lookup agrees in-grid too.
    # The LAST row's shifted instant lies beyond the grid (delta > 0 there): extrapolated by the dataset, clamped by the lookup -> counted
    assert 0.0 < ps["shift_ms"]["max"] <= 33.0 and ps["max_abs_v0_sync_minus_10hz_resample_mps"] < 1e-5
    assert ps["interp_10hz_vs_egomotion_mps"]["max"] < 1e-5
    assert ps["n_rows_beyond_grid"] == 1 and rows["t0_s"][-1] > t[-1]
    assert info["dt_grid_s"] == pytest.approx(DT) and info["dt_sidecar_s"] == DT


def test_interpolation_error_is_REPORTED_not_gated_but_a_wrong_instant_is_refused(tmp_path):
    """MEASURED on the real eval clips: the 10 Hz track re-sampled at the shifted instant differs from the 100 Hz log by up to 0.163 m/s
    (median 0.0012).  A non-linear egomotion between grid rows must therefore NOT refuse the build -- it is a property of what the model reads --
    while a v0 taken at the wrong instant (the pose instant, i.e. the old block's clock) must."""
    t, poses, ego = _world()
    ps = PS.read_pose_sync_sidecar(_sidecar(tmp_path, _sawtooth()))
    shift, v0s, _i = bb.pose_sync_clip(ps, SID, poses, t, OFF)
    fine_t = np.arange(0.0, t[-1] + 1e-9, 0.01)                      # a 100 Hz log with a 5 Hz ripple between the 10 Hz samples
    ripple = 0.05 * np.sin(2 * np.pi * 5.0 * fine_t)                 # zero at every 0.1 s grid instant
    ego100 = {"t": fine_t, "x": np.interp(fine_t, t, ego["x"]), "y": np.zeros_like(fine_t), "yaw": np.zeros_like(fine_t),
              "v": np.interp(fine_t, t, ego["v"]) + ripple}
    rows, cov = bb.build_clip_rows(CLIP, ego100, t, None, k=K_DK, dt=DT_DK, t0_shift_s=shift, v0_sync=v0s)
    rep = cov["pose_sync"]
    assert rep["max_abs_v0_sync_minus_10hz_resample_mps"] < 1e-5 and rep["interp_10hz_vs_egomotion_mps"]["max"] > 0.01
    assert np.array_equal(rows["speeds"][OFF:], v0s[OFF:])
    wrong = v0s.copy()
    wrong[OFF:] = poses[OFF:, 3]                                     # v0 at the POSE instant (no shift)
    with pytest.raises(RuntimeError, match="is not the 10 Hz speed track"):
        bb.build_clip_rows(CLIP, ego100, t, None, k=K_DK, dt=DT_DK, t0_shift_s=shift, v0_sync=wrong)


# ---------------------------------------------------------------------------------------------------------------- (4) + (5)
def _load_arm(dump, block):
    import r9_e2_distkeep as D
    return D.load_arm(dump, block)


def test_the_ACTUAL_caller_passes_on_the_pose_synced_block_and_refuses_the_unshifted_one(tmp_path):
    _t, poses, _e = _world()
    side = _sidecar(tmp_path, _sawtooth())
    dump = _dump(tmp_path, _dataset_v0(_dataset(poses, side), stride=3))
    synced = _block(tmp_path / "psync.npz", *_rows(tmp_path, side)[:2])
    out = _load_arm(dump, synced)
    ep_cov = list(out["coverage"]["episodes"].values())[0]
    assert ep_cov.get("status") != "SPEED_MISMATCH" and ep_cov["speed_check_max_mps"] <= 1e-5
    plain = _block(tmp_path / "plain.npz", *_plain_rows())
    with pytest.raises(SystemExit, match="speed proof FAILED"):     # the 2026-11-12 defect, reproduced
        _load_arm(dump, plain)


def test_REGRESSION_the_opposite_sign_is_refused_by_the_same_proof(tmp_path):
    _t, poses, _e = _world()
    side = _sidecar(tmp_path, _sawtooth())
    dump = _dump(tmp_path, _dataset_v0(_dataset(poses, side), stride=3))
    flipped = _block(tmp_path / "flip.npz", *_rows(tmp_path, side, sign=-1.0)[:2])
    with pytest.raises(SystemExit, match="speed proof FAILED"):
        _load_arm(dump, flipped)


# ---------------------------------------------------------------------------------------------------------------- (6)
def test_an_uncovered_clip_is_built_unshifted(tmp_path):
    rows, cov, info, shift, v0s = _rows(tmp_path, _sidecar(tmp_path, _sawtooth(), sid=SID + 1))
    plain, _c = _plain_rows()
    assert info == {"sid": SID, "covered": False} and not shift.any() and np.isnan(v0s).all()
    assert np.array_equal(rows["speeds"], plain["speeds"]) and np.array_equal(rows["t0_s"], plain["t0_s"])
    assert cov["pose_sync"]["n_rows_synced"] == 0


def test_a_v0_the_shifted_instant_does_not_reproduce_refuses_the_build(tmp_path):
    t, poses, ego = _world()
    ps = PS.read_pose_sync_sidecar(_sidecar(tmp_path, _sawtooth()))
    shift, v0s, _i = bb.pose_sync_clip(ps, SID, poses, t, OFF)
    with pytest.raises(RuntimeError, match="is not the 10 Hz speed track"):
        bb.build_clip_rows(CLIP, ego, t, None, k=K_DK, dt=DT_DK, t0_shift_s=shift, v0_sync=v0s + 0.02)
    bad = v0s.copy()
    bad[60] += 0.02                                                 # ONE in-grid row is enough
    with pytest.raises(RuntimeError, match="is not the 10 Hz speed track"):
        bb.build_clip_rows(CLIP, ego, t, None, k=K_DK, dt=DT_DK, t0_shift_s=shift, v0_sync=bad)
    with pytest.raises(ValueError, match="come together"):
        bb.build_clip_rows(CLIP, ego, t, None, k=K_DK, dt=DT_DK, t0_shift_s=shift)


def test_the_E2_and_D1_read_env_names_the_pose_synced_block():
    """Both runs the E2 read compares (and the D1 read, whose env sources this one) train with the pose-sync sidecar."""
    env = (ROOT / "ops" / "r9e2_read" / "r9e2_read_env.thor.sh").read_text(encoding="utf-8")
    lines = [ln for ln in env.splitlines() if ln.startswith("export R9E2_LEAD_BLOCK=")]
    assert len(lines) == 1
    assert '"$R/inputs/b1_eval_lead_block_k4_dt0p5_psync.npz"' in lines[0].split("#")[0]


def test_main_refuses_a_sidecar_without_its_sign_or_raw_offset(tmp_path):
    side = _sidecar(tmp_path, _sawtooth())
    base = ["--clips", "x", "--ego-tar", "x", "--ts-tar", "x", "--obs-dir", "x", "--out", str(tmp_path / "o.npz")]
    with pytest.raises(SystemExit, match="needs --pose-sync-sign AND --pose-sync-raw-offset"):
        bb.main(base + ["--pose-sync-sidecar", side, "--pose-sync-sign", "1"])
    with pytest.raises(SystemExit, match="needs --pose-sync-sign AND --pose-sync-raw-offset"):
        bb.main(base + ["--pose-sync-sidecar", side, "--pose-sync-raw-offset", "2"])
    with pytest.raises(SystemExit, match="without --pose-sync-sidecar"):
        bb.main(base + ["--pose-sync-sign", "1"])
