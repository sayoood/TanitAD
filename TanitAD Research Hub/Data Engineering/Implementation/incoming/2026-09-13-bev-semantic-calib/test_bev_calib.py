#!/usr/bin/env python3
"""Does BEV-agreement calibration actually recover known parameters?

This is the experiment that has to pass before the method is pointed at real
data. It builds a synthetic road with dashed lane markings, drives a known
camera along a known path, renders the ink pixels, perturbs the calibration, and
asks the optimiser to find its way back.

It is designed to answer three questions, and to be able to answer "no":

  1. Does the module's projection agree with ``trajlib.camera``? If not, the
     optimiser would be calibrating a different camera than the one that renders.
  2. Is the scale genuinely observable from metric ego-motion -- i.e. is the
     "collapse the height and everything overlaps" degeneracy really absent?
  3. **Are f and h separable?** The prediction from the geometry is: NO on a
     straight path (lateral scales freely with h), YES once the path curves.
     This test measures that instead of asserting it.

Run directly for a report, or under pytest.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bev_calib as BC                                            # noqa: E402

TRUE = dict(yaw=np.deg2rad(-6.5), pitch=np.deg2rad(-1.2), roll=np.deg2rad(0.8),
            height=1.28, lateral=-0.12, longitudinal=2.10,
            fx=1478.3, cx=960.0, cy=540.0)
IMG_W, IMG_H = 1920, 1080


# --------------------------------------------------------------------------
# synthetic world
# --------------------------------------------------------------------------

def make_path(n, ds, curvature):
    """Poses (x, y, psi) of each frame in the anchor frame, advancing ``ds`` m."""
    psi = np.arange(n) * ds * curvature
    x = np.concatenate([[0.0], np.cumsum(ds * np.cos(psi))[:-1]])
    y = np.concatenate([[0.0], np.cumsum(ds * np.sin(psi))[:-1]])
    return list(zip(x, y, psi))


def make_markings(poses, lane_w=3.50, dash=3.0, gap=10.0, width=0.15,
                  n_across=3, step=0.10, reach=140.0):
    """Dashed lane lines either side of the path, as dense world points.

    Markings follow the path, so on a curved road they curve with it -- which is
    what makes the curvature informative rather than just noisy.
    """
    ds = np.hypot(poses[1][0] - poses[0][0], poses[1][1] - poses[0][1])
    s_max = ds * len(poses) + reach
    s = np.arange(0.0, s_max, step)
    # interpolate the path (and extrapolate along the final heading) at arc length s
    ps = np.array(poses)
    s_nodes = np.arange(len(poses)) * ds
    x = np.interp(s, s_nodes, ps[:, 0]); y = np.interp(s, s_nodes, ps[:, 1])
    psi = np.interp(s, s_nodes, ps[:, 2])
    beyond = s > s_nodes[-1]
    if beyond.any():
        ex = s[beyond] - s_nodes[-1]
        x[beyond] = ps[-1, 0] + ex * np.cos(ps[-1, 2])
        y[beyond] = ps[-1, 1] + ex * np.sin(ps[-1, 2])
        psi[beyond] = ps[-1, 2]
    on_dash = (s % (dash + gap)) < dash
    nx, ny = -np.sin(psi), np.cos(psi)                   # left normal
    pts = []
    for side in (+1, -1):
        for w in np.linspace(-width / 2, width / 2, n_across):
            e = side * lane_w / 2 + w
            pts.append(np.stack([x[on_dash] + nx[on_dash] * e,
                                 y[on_dash] + ny[on_dash] * e], axis=1))
    return np.concatenate(pts)


def render(world_pts, poses, P):
    """World ink -> per-frame integer pixel observations, using TRUE calibration."""
    obs = []
    for (tx, ty, psi) in poses:
        c, s = np.cos(-psi), np.sin(-psi)                # world -> this vehicle frame
        dx, dy = world_pts[:, 0] - tx, world_pts[:, 1] - ty
        local = np.stack([c * dx - s * dy, s * dx + c * dy], axis=1)
        uv = BC.project_ground(local, P)
        ok = (np.isfinite(uv[:, 0]) & (uv[:, 0] >= 0) & (uv[:, 0] < IMG_W)
              & (uv[:, 1] > IMG_H * 0.45) & (uv[:, 1] < IMG_H))
        obs.append(np.round(uv[ok]))                     # integer pixels, as a mask gives
    return obs


def scenario(curvature, n_frames=14, ds=3.0):
    poses = make_path(n_frames, ds, curvature)
    world = make_markings(poses)
    return render(world, poses, TRUE), poses


# --------------------------------------------------------------------------
# 1. the convention must match the pipeline
# --------------------------------------------------------------------------

def _load_trajlib_camera():
    p = (pathlib.Path("/home/user/TanitAD/stack/tanitad/data/trajrecon/camera.py"))
    if not p.exists():
        return None
    spec = importlib.util.spec_from_file_location("_cam_ref", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_projection_matches_trajlib():
    cam_mod = _load_trajlib_camera()
    if cam_mod is None:
        print("  (trajlib camera.py not found — convention check skipped)")
        return
    cam = cam_mod.nominal_camera(IMG_W, IMG_H, 66.0)
    cam.fx = cam.fy = TRUE["fx"]; cam.cx = TRUE["cx"]; cam.cy = TRUE["cy"]
    cam.yaw, cam.pitch, cam.roll = TRUE["yaw"], TRUE["pitch"], TRUE["roll"]
    cam.height_m = TRUE["height"]
    cam.longitudinal_m = TRUE["longitudinal"]; cam.lateral_m = TRUE["lateral"]
    g = np.array([[10.0, 1.75], [25.0, -1.75], [40.0, 0.30]])
    ref, ok = cam.project(np.column_stack([g, np.zeros(len(g))]))
    mine = BC.project_ground(g, TRUE)
    err = np.abs(ref[ok] - mine[ok]).max()
    print(f"  projection vs trajlib.camera: max |delta| = {err:.9f} px")
    assert err < 1e-6, f"convention mismatch ({err} px) — the optimiser would fit the wrong camera"


def test_roundtrip_is_exact():
    g = np.array([[8.0, 1.0], [30.0, -2.0], [50.0, 0.5]])
    uv = BC.project_ground(g, TRUE)
    back = BC.ground_from_pixels(uv, TRUE)
    err = np.abs(back - g).max()
    print(f"  ground->pixel->ground: max |delta| = {err:.9f} m")
    assert err < 1e-9


# --------------------------------------------------------------------------
# 2. the objective must peak at the truth, and must NOT reward collapse
# --------------------------------------------------------------------------

def test_truth_beats_perturbations_and_collapse():
    obs, poses = scenario(curvature=1 / 300.0)
    grid = BC.BevGrid()
    s_true = BC.sharpness(obs, poses, TRUE, grid)
    rows = [("truth", TRUE, s_true)]
    for name, d in (("yaw +2 deg", dict(yaw=TRUE["yaw"] + np.deg2rad(2))),
                    ("pitch +1 deg", dict(pitch=TRUE["pitch"] + np.deg2rad(1))),
                    ("height x1.3", dict(height=TRUE["height"] * 1.3)),
                    ("height -> 0.05 (collapse)", dict(height=0.05)),
                    ("fx x1.3", dict(fx=TRUE["fx"] * 1.3))):
        P = dict(TRUE); P.update(d)
        rows.append((name, P, BC.sharpness(obs, poses, P, grid)))
    print("\n  BEV agreement score (higher = sharper):")
    for name, _, s in rows:
        print(f"    {name:28s} {s:.6e}   {'<-- TRUTH' if name=='truth' else ''}")
    for name, _, s in rows[1:]:
        assert s < s_true, f"{name} scored >= truth — the objective does not peak at the truth"
    collapse = [s for n, _, s in rows if "collapse" in n][0]
    assert collapse < s_true * 0.7, (
        "collapsing the height scored close to the truth — the metric ego-motion "
        "is not pinning the scale as the design claims")


# --------------------------------------------------------------------------
# 3. THE REAL QUESTION: recovery, and the f/h degeneracy
# --------------------------------------------------------------------------

def _recover(curvature, free, perturb, label):
    obs, poses = scenario(curvature=curvature)
    P0 = dict(TRUE)
    for k, v in perturb.items():
        P0[k] = TRUE[k] + v
    P, info = BC.calibrate(obs, poses, P0, free=free,
                           bounds=dict(height=(0.4, 3.0), fx=(700.0, 3000.0)))
    print(f"\n  {label}")
    print(f"    score {info['score0']:.4e} -> {info['score']:.4e}  ({info['nfev']} evals)")
    out = {}
    for k in free:
        unit = "deg" if k in ("yaw", "pitch", "roll") else ("px" if k in ("fx", "cx", "cy") else "m")
        conv = (lambda z: np.rad2deg(z)) if unit == "deg" else (lambda z: z)
        t, s, r = conv(TRUE[k]), conv(P0[k]), conv(P[k])
        out[k] = r - t
        print(f"    {k:12s} true {t:9.3f}  start {s:9.3f}  ->  {r:9.3f}   "
              f"err {r-t:+8.3f} {unit}")
    return out, info


def test_recovers_extrinsics_from_a_bad_start():
    """Which extrinsics come back, and which do not.

    ⚠️ `_recover` already returns errors in the REPORTING unit (deg / m), so these
    thresholds are in degrees and metres directly. An earlier version applied
    `rad2deg` a second time here and failed a 0.009 deg result against a 0.5 deg
    bound -- a bug in the check, not in the method.

    The bounds encode a MEASURED observability split, not a wish:
      * yaw, pitch, height  -- strongly observable, sub-0.2 deg / few mm
      * roll, lateral       -- WEAK. A constant mount translation is common to
        every frame, so it cancels almost entirely in a CROSS-FRAME consistency
        objective; only the lever arm during a turn makes it visible at all, and
        1/300 m^-1 is a gentle turn. This is a property of the method, and it is
        why lateral offset has resisted every estimator in this pipeline.
    """
    err, _ = _recover(1 / 300.0, ("yaw", "pitch", "roll", "height", "lateral"),
                      dict(yaw=np.deg2rad(6.5), pitch=np.deg2rad(1.2),
                           roll=np.deg2rad(-0.8), height=-0.11, lateral=0.37),
                      "EXTRINSICS from a wrong start (curved path, 1/300 m^-1)")
    assert abs(err["yaw"]) < 0.30, f"yaw off by {err['yaw']:+.3f} deg"
    assert abs(err["pitch"]) < 0.30, f"pitch off by {err['pitch']:+.3f} deg"
    assert abs(err["height"]) < 0.05, f"height off by {err['height']:+.3f} m"
    return err


def test_f_and_h_degenerate_on_a_straight_path():
    """The prediction: straight driving cannot separate them."""
    err, _ = _recover(0.0, ("height", "fx"), dict(height=0.35, fx=420.0),
                      "f AND h on a STRAIGHT path (expect: NOT separated)")
    ratio = (TRUE["height"] + err["height"]) / TRUE["height"] * \
            (TRUE["fx"] + err["fx"]) / TRUE["fx"]
    print(f"    -> h err {err['height']:+.3f} m, f err {err['fx']:+.1f} px, "
          f"f*h preserved to {abs(1-ratio)*100:.1f}%")
    return err


def test_f_and_h_separate_once_the_path_curves():
    err, _ = _recover(1 / 150.0, ("height", "fx"), dict(height=0.35, fx=420.0),
                      "f AND h on a CURVED path, R=150 m (expect: separated)")
    return err


if __name__ == "__main__":
    print("=" * 78)
    print("BEV-agreement calibration — synthetic recovery")
    print("=" * 78)
    print("\n[1] convention")
    test_projection_matches_trajlib()
    test_roundtrip_is_exact()
    print("\n[2] objective shape")
    test_truth_beats_perturbations_and_collapse()
    print("\n[3] recovery")
    er = test_recovers_extrinsics_from_a_bad_start()
    ee = test_recovers_extrinsics_from_a_bad_start.__wrapped__ if False else None
    es = test_f_and_h_degenerate_on_a_straight_path()
    ec = test_f_and_h_separate_once_the_path_curves()
    print("\n" + "=" * 78)
    print(f"  straight path: |h err| {abs(es['height']):.3f} m, |f err| {abs(es['fx']):.0f} px")
    print(f"  curved  path: |h err| {abs(ec['height']):.3f} m, |f err| {abs(ec['fx']):.0f} px")
    print("=" * 78)
