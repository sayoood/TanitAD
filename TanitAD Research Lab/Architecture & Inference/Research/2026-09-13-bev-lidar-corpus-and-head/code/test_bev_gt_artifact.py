"""P3 - tests for the LiDAR BEV GT artifacts and their loader.

Run:  PYTHONPATH=<stack> pytest -q test_bev_gt_artifact.py
      (BEVGT_DIR overrides the artifact directory; default = the P2 output on the dev box)
      From the repo alone: BEVGT_DIR=../raw/sample_gt  (2 staged artifacts + manifest; 10/10 green,
      MEASURED 2026-09-13). The full 134-clip set lives on the dev box C: cache and its D: mirror.

Every expectation below is a LITERAL or a PHYSICAL derivation independent of the code under
test. Each mutation test re-introduces a real failure mode and asserts the check goes RED:
  * the 09-11 deskew rotation sign                  (measured wrong in P0)
  * a MIRRORED artifact (y -> -y / az -> -az)       (R-2026-09-08-wpa-mirror)
  * a WRONG CELL SIZE in meta_json                  (units that do not match the bytes)
"""
from __future__ import annotations

import copy
import json
import lzma
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bev_gt_loader as L  # noqa: E402
from lidar_bev import (  # noqa: E402
    CartesianBEVSpec, PolarBEVSpec, ZBand, deskew_rigid, rasterise_cartesian, rasterise_polar,
)

BEVGT_DIR = Path(os.environ.get("BEVGT_DIR", r"C:\Users\Admin\tanitad-caches\bevhead-20260913\bev_gt"))
JOIN = (HERE.parents[3] / "Benchmarks & Evals" / "Research" / "2026-09-06-b1-agent-join"
        / "raw" / "b1eval_agents.jsonl.xz")


# ---------------------------------------------------------------------------
# 1. deskew: physically exact arc motion, old sign as the mutation
# ---------------------------------------------------------------------------
def _simulate_spin(v=10.0, r=0.30, t_ref_us=1_000_000, spin_us=100_000):
    """Static world points observed by an ego on an EXACT circular arc (constant v, r).
    World frame = ego frame at t_ref. Returns (p_observed, pt_ts_us, P_world)."""
    rng = np.random.default_rng(0)
    n = 4000
    ang = rng.uniform(-math.pi, math.pi, n)
    rad = rng.uniform(5.0, 40.0, n)
    P = np.stack([rad * np.cos(ang), rad * np.sin(ang), rng.uniform(0.3, 2.5, n)], 1)
    # each point captured at its own time, spread over one spin centred on t_ref
    ts = (t_ref_us + rng.uniform(-spin_us / 2, spin_us / 2, n)).astype(np.int64)
    dt = (ts - t_ref_us) * 1e-6
    th = r * dt
    ex = v * np.sin(th) / r           # exact arc (r != 0)
    ey = v * (1.0 - np.cos(th)) / r
    c, s = np.cos(-th), np.sin(-th)   # world -> ego(t): R(-theta) (P - E)
    dx, dy = P[:, 0] - ex, P[:, 1] - ey
    obs = np.stack([dx * c - dy * s, dx * s + dy * c, P[:, 2]], 1)
    return obs, ts, P, t_ref_us, v, r


def _deskew_error(fn) -> float:
    obs, ts, P, t_ref, v, r = _simulate_spin()
    out = fn(obs, ts, t_ref, v, r)
    return float(np.hypot(out[:, 0] - P[:, 0], out[:, 1] - P[:, 1]).max())


def test_deskew_recovers_static_scene_on_an_exact_arc():
    # second-order residual of the first-order model: v*dt*dth/2 ~ 10*0.05*0.015/2 = 4 mm,
    # plus the rotation-of-translation term; bound it with a literal 2 cm.
    assert _deskew_error(deskew_rigid) < 0.02


def test_MUTATION_old_09_11_rotation_sign_goes_red():
    # deskew_rigid's translation does not depend on the yaw rate, so calling it with -r is
    # EXACTLY the 09-11 function (rotation by -r*dt). At 40 m, 2*dth*r ~ 2*0.015*40 = 1.2 m.
    def old_sign(p, ts, t_ref, v, r):
        return deskew_rigid(p, ts, t_ref, v, -r)
    err = _deskew_error(old_sign)
    assert err > 0.10, f"the old sign must be detectably wrong, got {err:.4f} m"
    assert not (err < 0.02), "RED expected: the analytic bound must reject the old sign"


# ---------------------------------------------------------------------------
# 2. rasteriser: a pole at a KNOWN rig coordinate lands in LITERAL cells
# ---------------------------------------------------------------------------
def _pole(x, y):
    z = np.linspace(0.35, 2.5, 40)
    return np.stack([np.full_like(z, x), np.full_like(z, y), z], 1)


def test_pole_10m_ahead_5m_left_lands_in_literal_cells():
    pts = _pole(10.0, 5.0)
    P48 = rasterise_polar(pts, PolarBEVSpec(n_az=40, n_rng=48), ZBand())
    # r = hypot(10,5) = 11.180 -> bin floor(11.180/1.25) = 8
    # az = atan2(5,10) = 26.565 deg LEFT -> col floor((60 - 26.565)/3) = 11
    assert np.argwhere(P48["occ"] > 0).tolist() == [[8, 11]]
    C = rasterise_cartesian(pts, np.zeros(len(pts), np.uint8), CartesianBEVSpec(), ZBand())
    # ix = floor(10/0.5) = 20 ; iy = floor((5 + 16)/0.5) = 42
    assert np.argwhere(C["occ"] > 0).tolist() == [[20, 42]]


def test_MUTATION_mirrored_pole_goes_red():
    pts = _pole(10.0, -5.0)       # the same pole, mirrored
    P48 = rasterise_polar(pts, PolarBEVSpec(n_az=40, n_rng=48), ZBand())
    got = np.argwhere(P48["occ"] > 0).tolist()
    assert got != [[8, 11]], "RED expected: a mirrored input must not satisfy the LEFT literal"
    assert got == [[8, 28]]       # floor((60 + 26.565)/3) = 28


def test_three_polar_observed_rules_literal():
    """One column: a pole at r=10 m, bare ground returns at r=20 m, a wall at r=30 m."""
    z = np.linspace(0.35, 2.5, 40)
    pole = np.stack([np.full_like(z, 10.0), np.zeros_like(z), z], 1)
    wall = np.stack([np.full_like(z, 30.0), np.zeros_like(z), z], 1)
    gx = np.linspace(20.3, 20.7, 30)      # strictly inside bin 16 = [20.0, 21.25)
    ground = np.stack([gx, np.zeros_like(gx), np.zeros_like(gx)], 1)
    P = rasterise_polar(np.concatenate([pole, ground, wall]), PolarBEVSpec(n_az=40, n_rng=48), ZBand())
    col = int(np.argwhere(P["occ"] > 0)[0][1])
    assert col == 20
    occ = np.nonzero(P["occ"][:, col])[0].tolist()
    A = np.nonzero(P["observed"][:, col])[0].tolist()
    B = np.nonzero(P["observed_npts"][:, col])[0].tolist()
    C = np.nonzero(P["visible_first_hit"][:, col])[0].tolist()
    # y = 0 is az 0: col floor((60 - 0)/3) = 20
    # bins: pole floor(10/1.25)=8, ground floor(20.3..20.7/1.25)=16, wall floor(30/1.25)=24
    # first hit centre 10.625; shadow where centre > 11.875 -> bins >= 10
    assert occ == [8, 24]
    assert C == list(range(10))                          # up to the first hit + 1 cell
    assert A == list(range(10)) + [24]                   # shipped: keeps the occupied wall
    assert B == list(range(10)) + [16, 24]               # + the free ground with returns
    # MUTATION: the biased rule A drops the free-with-returns cell B keeps -> RED on it
    assert 16 not in A


def test_state_encoding_literal():
    occ = np.array([[[True, False, False, False]]])
    obs = np.array([[[True, True, False, False]]])
    clip = L.BevClip(meta={}, grid="polar48", occ=occ, observed=obs,
                     cam_vis=np.array([[255, 255, 255, 0]], np.uint8),
                     label_valid=np.array([True]), t_img_us=np.array([0]),
                     raw_frame=np.array([0], np.int16))
    assert clip.state().tolist() == [[[2, 1, 3, 0]]]
    assert clip.scored_mask().tolist() == [[[True, True, False, False]]]
    assert clip.rows_to_frames(np.array([0, 5])).tolist() == [2, 7]


# ---------------------------------------------------------------------------
# 3. the REAL artifacts
# ---------------------------------------------------------------------------
def _manifest_ok() -> list[dict]:
    mp = BEVGT_DIR / "manifest.jsonl"
    if not mp.exists():
        return []
    rows = {}
    for line in mp.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        rows[r["clip_sha12"]] = r
    return [r for r in rows.values() if r.get("ok")]


@pytest.fixture(scope="module")
def artifacts():
    ok = _manifest_ok()
    if not ok:
        pytest.skip(f"no P2 artifacts under {BEVGT_DIR} -- NOT a pass")
    return ok


def test_every_ok_artifact_verifies_against_literals_and_builder(artifacts):
    import p2_build_corpus as B
    n_read = 0
    for r in artifacts:
        for grid in ("polar48", "polar24", "cart"):
            clip = L.load_clip(BEVGT_DIR / r["artifact"], grid=grid, verify=True,
                               builder_specs=B.SPECS, zband=B.ZB)
            assert clip.occ.shape[0] == r["T"]
        n_read += 1
    # positive assertion that files were READ -- "0 failures" over 0 files is not a pass
    assert n_read == len(artifacts) and n_read > 0


def test_physical_camera_field_literals(artifacts):
    """Independent physical facts: the 0-1.25 m ring around the rear axle is under the car
    (the camera cannot see it); a cell 50 m dead ahead is visible."""
    for r in artifacts[:20]:
        with np.load(BEVGT_DIR / r["artifact"]) as z:
            vis = z["polar48_cam_vis"]
        assert (vis[0, :] < L.CAM_VIS_IN_FIELD).all()
        assert (vis[40, 19:21] >= L.CAM_VIS_IN_FIELD).all()   # r ~ 50.6 m, az ~ 0


def test_MUTATION_wrong_cell_size_goes_red(artifacts):
    with np.load(BEVGT_DIR / artifacts[0]["artifact"]) as z:
        meta = json.loads(str(z["meta_json"]))
    L.verify_meta(meta)                              # the original passes
    bad = copy.deepcopy(meta)
    bad["cartesian"]["cell_m"] = 0.25
    with pytest.raises(L.ArtifactSpecError, match="cell_m"):
        L.verify_meta(bad)
    bad2 = copy.deepcopy(meta)
    bad2["polar48"]["cell_deg"] = 6.0
    with pytest.raises(L.ArtifactSpecError, match="cell_deg"):
        L.verify_meta(bad2)


def _join_rows_by_sha12() -> dict:
    import hashlib
    out: dict = {}
    with lzma.open(JOIN, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            k = hashlib.sha256(r["clip_id"].encode()).hexdigest()[:12]
            out.setdefault(k, []).append((float(r["t_s"]), r.get("agents", [])))
    return out


def _mirror_copy(src: Path, dst: Path) -> None:
    with np.load(src) as z:
        d = {k: z[k] for k in z.files}
    for k in list(d):
        if k.startswith("cart_") and d[k].ndim == 3:
            d[k] = d[k][:, :, ::-1].copy()           # y -> -y
        if (k.startswith("polar48_") or k.startswith("polar24_")) and d[k].ndim == 3:
            d[k] = d[k][:, :, ::-1].copy()           # az -> -az
    np.savez_compressed(dst, **d)


def test_orientation_real_boxes_beat_mirror_and_MUTATION_goes_red(artifacts, tmp_path):
    join = _join_rows_by_sha12()
    use = [r for r in artifacts if r["clip_sha12"] in join][:4]
    assert len(use) >= 2, "need >= 2 joined artifacts for a pooled orientation check"
    tot = {"real": [0.0, 0], "mirror": [0.0, 0], "marg": [0.0, 0]}
    tot_m = {"real": [0.0, 0], "mirror": [0.0, 0]}
    for r in use:
        src = BEVGT_DIR / r["artifact"]
        res = L.registration_check(src, join[r["clip_sha12"]])
        for k in ("real", "mirror"):
            tot[k][0] += res[f"{k}_hit_rate"] * res[f"{k}_n_cells"]
            tot[k][1] += res[f"{k}_n_cells"]
        tot["marg"][0] += res["marginal"] * res["n_observed_cells"]
        tot["marg"][1] += res["n_observed_cells"]
        dst = tmp_path / f"mirrored_{r['clip_sha12']}.npz"
        _mirror_copy(src, dst)
        rm = L.registration_check(dst, join[r["clip_sha12"]])
        for k in ("real", "mirror"):
            tot_m[k][0] += rm[f"{k}_hit_rate"] * rm[f"{k}_n_cells"]
            tot_m[k][1] += rm[f"{k}_n_cells"]
    real = tot["real"][0] / tot["real"][1]
    mirror = tot["mirror"][0] / tot["mirror"][1]
    marg = tot["marg"][0] / tot["marg"][1]
    # the gate (literals): real >= 2x marginal AND real >= 1.5x mirror
    assert real >= 2.0 * marg, (real, marg)
    assert real >= 1.5 * mirror, (real, mirror)
    # the MUTATED artifact must FAIL the same gate
    real_m = tot_m["real"][0] / tot_m["real"][1]
    mirror_m = tot_m["mirror"][0] / tot_m["mirror"][1]
    assert not (real_m >= 1.5 * mirror_m), ("RED expected on the mirrored artifact", real_m, mirror_m)
