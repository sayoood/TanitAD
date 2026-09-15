#!/usr/bin/env python3
"""Calibrate from TRACKED GROUND POINTS against the odometer — f*h and the horizon.

WHY A THIRD ESTIMATOR, AND WHY THIS ONE
---------------------------------------
Two have now failed on this recording, each for a diagnosed reason:

  * **BEV agreement** (``run_real.py``) fixes the extrinsics (+41% over the
    shipped nominal) but its focal scan is MONOTONE to the bound, because
    stretching the map longitudinally makes the fixed 2.2 m of inter-frame travel
    relatively smaller and any overlap score rewards that.
  * **BEV lag** (``lag_scale.py``) removes that bias by construction -- it reads a
    displacement against the metric odometer, and its synthetic self-test recovers
    an injected ``f*h`` to within 5% across a 3.3x range. On the real recording it
    REFUSES: MEASURED, at D = 2.99 m the correlation decays monotonically from
    ``r = 0.211`` at lag 0 and is only ``r = 0.062`` at the true displacement.
    Content that is static in the IMAGE back-projects to the same range in every
    frame, so it correlates at lag 0 and outweighs the moving ground.

That last diagnosis is also the design of this module. **A whole-map correlation
has to average the ground and the clutter together; per-feature tracking does
not.** A point on the road 10 m ahead moves hundreds of pixels between frames; a
guardrail post, the bonnet, a distant hillside move a handful. The two classes
separate by their motion, so a robust fit can keep the ground and discard the
rest -- which is exactly what the correlation could not do.

THE MODEL
---------
No approximations and no small-angle expansion: the same projection the renderer
uses. For a tracked point ``uv0`` in frame 0, with the odometer's relative pose
``(dx, dy, dpsi)`` to frame k::

    g      = ground_from_pixels(uv0, P)            # assume it lies on the road
    g_k    = R(-dpsi) @ (g - [dx, dy])             # METRIC, does not scale with P
    uv_pred = project_ground(g_k, P)

and the calibration is the ``P`` that makes ``uv_pred`` land on where the tracker
actually found the point. A point that is NOT on the road violates the first line
and lands far away -- which is the signal used to reject it, not a nuisance.

WHAT IS OBSERVABLE, MEASURED RATHER THAN ASSUMED
------------------------------------------------
``--self-test`` injects known errors and reports recovery, and :func:`observability`
reports the singular values of the parameter Jacobian on the REAL tracks, so the
degeneracy is quantified on this data instead of inherited from the textbook. The
expectation from ``scale_calib``'s own derivation is that straight driving sees
``f*h`` and the horizon row but not ``f`` and ``h`` separately; the point of
printing the spectrum is to find out whether this clip's turning is enough to
break that, rather than assuming either way.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bev_calib as BC                                              # noqa: E402
import lag_scale as LS                                              # noqa: E402


# --------------------------------------------------------------------------
# tracking
# --------------------------------------------------------------------------
LK = dict(winSize=(21, 21), maxLevel=4,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))


def road_mask(img: np.ndarray, row_band=(0.55, 0.77), dilate=11) -> np.ndarray:
    """Where to look for trackable points: ON THE ROAD MARKINGS, nowhere else.

    ⚠️ **A plain corner detector does not find the road, it avoids it.** MEASURED
    on this recording: ``goodFeaturesToTrack`` over the whole lower image returned
    6807 forward-backward-checked tracks of which **2.3% fit any road-plane model**,
    with a 400 px median reprojection error. That is not an outlier problem a
    robust loss can absorb -- it is the detector working correctly. Corners live
    where texture is, asphalt is smooth, so the features were the guardrail, the
    rock face, other vehicles and the bonnet: everything except the plane being
    calibrated.

    The mask is built from the pipeline's OWN ridge detector -- the same marking
    detector ``lane_calib`` uses -- and then dilated, so corners are sought at
    dash ends, line breaks and arrow tips. Those are on the ground by
    construction and, unlike a point in the middle of a lane line, they are not
    free to slide.
    """
    from trajlib import lane_calib as LC
    h, w = img.shape[:2]
    u, v = LC._ridge_points(img, int(row_band[0] * h), int(row_band[1] * h))
    m = np.zeros((h, w), np.uint8)
    if len(u):
        m[np.clip(v, 0, h - 1).astype(int), np.clip(u, 0, w - 1).astype(int)] = 255
        m = cv2.dilate(m, np.ones((dilate, dilate), np.uint8))
    return m


def track_pair(img0: np.ndarray, img1: np.ndarray, n_pts=600,
               row_band=(0.55, 0.77), fb_tol=0.6, mask=None):
    """Forward-backward-checked LK tracks from ``img0`` to ``img1``.

    The forward-backward check is not optional on road texture: lane lines are
    locally one-dimensional, so LK can slide along them for free and return a
    confident, wrong correspondence. Tracking back and requiring the round trip to
    land within ``fb_tol`` px removes exactly that failure, and it is cheap.

    ``row_band`` stops at 0.77 of image height, and that bound is MEASURED: below
    row ~830 (0.77 x 1080) the tracked flow collapses to ~0 while the ground-plane
    model predicts 59-135 px, because that part of the frame is the bonnet or a
    windscreen reflection -- static in the image. It contributed 6294 seeds, which
    is why an earlier 0.97 bound produced a fit with 0.8% inliers.
    """
    h, w = img0.shape[:2]
    if mask is None:
        mask = np.zeros((h, w), np.uint8)
        mask[int(row_band[0] * h):int(row_band[1] * h), :] = 255
    if mask.sum() == 0:
        return np.zeros((0, 2)), np.zeros((0, 2))
    p0 = cv2.goodFeaturesToTrack(img0, maxCorners=n_pts, qualityLevel=0.005,
                                 minDistance=6, mask=mask, blockSize=7)
    if p0 is None or len(p0) < 20:
        return np.zeros((0, 2)), np.zeros((0, 2))
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **LK)
    p0b, st0, _ = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **LK)
    ok = (st1.ravel() == 1) & (st0.ravel() == 1)
    fb = np.linalg.norm((p0b - p0).reshape(-1, 2), axis=1)
    ok &= fb < fb_tol
    return p0.reshape(-1, 2)[ok], p1.reshape(-1, 2)[ok]


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------
def predict(uv0: np.ndarray, P: dict, pose: tuple) -> np.ndarray:
    """Where a ground point seen at ``uv0`` should appear after ``pose``.

    ``pose = (dx, dy, dpsi)`` is frame k's origin in frame 0's vehicle frame --
    the same convention ``trajectory.jsonl`` windows use and ``BC.to_anchor``
    consumes, so this is its exact inverse.
    """
    g = BC.ground_from_pixels(uv0, P)
    if len(g) != len(uv0):                 # some rays missed the ground plane
        return np.full((len(uv0), 2), np.nan)
    dx, dy, dpsi = pose
    c, s = np.cos(-dpsi), np.sin(-dpsi)
    gx, gy = g[:, 0] - dx, g[:, 1] - dy
    gk = np.stack([c * gx - s * gy, s * gx + c * gy], axis=1)
    return BC.project_ground(gk, P)


def residuals(tracks, P: dict, img_wh=(1920, 1080), margin=400.0) -> np.ndarray:
    """Stacked prediction errors in pixels over every (uv0, uv1, pose) triple.

    ⚠️ Predictions that land far outside the frame are DROPPED, not scored. A
    point close to the bonnet is 3 m ahead; move 2.2 m and its predicted row is
    off the bottom of the image, so the "error" is an arbitrary number set by how
    far past the edge the projection went, not by the calibration. Including them
    made the median reprojection read 348 px before the fit and 423 px after --
    a statistic that says nothing about either calibration. The same applies to
    rays that miss the plane entirely, which arrive here as NaN.
    """
    w, h = img_wh
    out = []
    for uv0, uv1, pose in tracks:
        pr = predict(uv0, P, pose)
        d = np.linalg.norm(pr - uv1, axis=1)
        ok = (np.isfinite(d)
              & (pr[:, 0] > -margin) & (pr[:, 0] < w + margin)
              & (pr[:, 1] > -margin) & (pr[:, 1] < h + margin))
        out.append(d[ok])
    return np.concatenate(out) if out else np.zeros(0)


def robust_cost(tracks, P: dict, c_px: float = 3.0, n_ref: int | None = None) -> float:
    """Cauchy loss on the reprojection error.

    Cauchy rather than Huber because the outliers here are not mild: a tracked
    point on a passing car or on the rock face is not a noisy ground point, it is
    a different object obeying a different equation, and its error grows without
    bound as the fit improves. Cauchy's influence falls to zero; Huber's does not.
    """
    r = residuals(tracks, P)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return 1e9
    # ⚠️ divide by a FIXED count, not by len(r). Dropping off-image predictions is
    # necessary (see `residuals`) but it hands the optimiser a way to cheat: push
    # the calibration until most points project off-frame and the mean is taken
    # over the few that remain. Scoring the dropped ones at the loss ceiling makes
    # discarding data cost exactly what it should.
    n = n_ref if n_ref is not None else len(r)
    ceil_ = np.log1p((50.0 / c_px) ** 2)
    return float((np.log1p((r / c_px) ** 2).sum() + ceil_ * max(0, n - len(r))) / max(n, 1))


def inlier_stats(tracks, P: dict, tol_px: float = 3.0, n_ref: int | None = None):
    """Inlier fraction is over ALL correspondences, including the ones whose
    prediction fell off-frame -- otherwise a calibration that projects most of the
    data into the void scores a flattering fraction over the remnant."""
    r = residuals(tracks, P)
    r = r[np.isfinite(r)]
    n = n_ref if n_ref is not None else len(r)
    if len(r) == 0:
        return dict(n=0, n_scored=int(n), inlier_frac=0.0, med_px=float("nan"),
                    med_inlier_px=float("nan"))
    ins = r < tol_px
    return dict(n=int(len(r)), n_scored=int(n),
                inlier_frac=float(ins.sum() / max(n, 1)),
                med_px=float(np.median(r)),
                med_inlier_px=float(np.median(r[ins])) if ins.any() else float("nan"))


# --------------------------------------------------------------------------
# fitting, parameterised by (f*h, horizon row) — the observable pair
# --------------------------------------------------------------------------
def fit(tracks, P0: dict, height: float, fh0: float, vh0: float,
        fit_yaw=True, c_px=3.0, maxfev=800):
    """Fit ``(f*h, horizon row)`` -- and optionally yaw -- by robust reprojection.

    ⚠️ The free variables are ``f*h`` and the HORIZON ROW, not ``fx`` and ``pitch``.
    Those are the same two numbers re-coordinated, but the re-coordination matters:
    changing ``fx`` at fixed pitch also moves the horizon (see
    ``lag_scale.horizon_row``), so an optimiser over ``(fx, pitch)`` walks a
    diagonal valley and reports a focal length that is partly a pitch error. In
    ``(f*h, v_h)`` the two axes are the things the geometry actually separates.

    ``height`` is HELD, because straight driving cannot separate it from ``f``;
    the caller scans it and reads ``f = f*h / h``.
    """
    from scipy.optimize import minimize

    def build(x):
        fh, vh = float(x[0]) * 1000.0, float(x[1]) * 100.0
        Q = dict(P0)
        Q["height"] = height
        Q["fx"] = fh / height
        if fit_yaw:
            Q["yaw"] = float(x[2]) * 0.01
        try:
            Q["pitch"] = LS.pitch_for_horizon(Q, vh)
        except Exception:
            return None
        return Q

    n_ref = sum(len(u) for u, _, _ in tracks)

    def neg(x):
        Q = build(x)
        return 1e9 if Q is None else robust_cost(tracks, Q, c_px, n_ref)

    x0 = [fh0 / 1000.0, vh0 / 100.0] + ([P0["yaw"] / 0.01] if fit_yaw else [])
    res = minimize(neg, np.array(x0, float), method="Nelder-Mead",
                   options=dict(maxfev=maxfev, xatol=1e-4, fatol=1e-7))
    Q = build(res.x)
    return Q, dict(cost0=neg(np.array(x0, float)), cost=float(res.fun),
                   nfev=int(res.nfev), fh=float(res.x[0] * 1000.0),
                   vh=float(res.x[1] * 100.0),
                   yaw_deg=float(np.rad2deg(Q["yaw"])) if Q else None)


def row_flow_fit(tracks, grid=np.arange(380.0, 581.0, 2.0), c_px=2.0,
                 n_boot=300, seed=0):
    """``f*h`` and the horizon row from the VERTICAL flow alone. The decisive one.

    THE EXACT ONE-DIMENSIONAL MODEL. With ``q = v - v_h``, ``A = f*h`` and an
    odometer step ``D``, a ground point's row obeys ``q' = q A / (A - D q)``, so::

        dv  =  D q^2 / (A - D q)              and, inverted per point,
        A   =  D q q' / (q' - q)

    Two parameters, no yaw, no roll, no lateral offset, no principal point. That
    narrowness is the point: the 2-D fit in :func:`fit` has to explain the
    horizontal flow too, and on this recording it never found a credible optimum
    (0.8% inliers, yaw +3.9 deg against a well-established -7.0). The row equation
    keeps only the two things this geometry actually determines.

    **The horizon is found by CONSISTENCY, not by fitting a curve.** Given the
    right ``v_h`` the per-point ``A`` estimates agree across image rows; given a
    wrong one they trend with row. So ``v_h`` is the value where the robust slope
    of ``A`` against ``q`` is zero -- which is the same "constant across range"
    logic the Mobius fit uses, but exact and per point.

    ⚠️ **Row band matters and it was measured, not chosen.** Below row ~830 the
    measured flow COLLAPSES to ~0 while the plane model predicts 59-135 px, and
    the tracking rate falls to 15-20%: that region is static in the image (bonnet
    or windscreen reflection) and it supplied 6294 of the seeds. Callers must stop
    at ~0.77 of image height. Above that, rows 580-830 follow the model closely
    (at 730-780, measured 27.46 px against a predicted 27.27).

    ⚠️ **Tracking selection was tested and REFUTED as the driver.** Points that move
    far fail LK, so the survivors are biased slow, which would inflate ``A``.
    MEASURED: across steps of 1/2/3 frames the tracking rate falls 61.8% -> 24.8%
    -> 8.8%, a 7x change in selection pressure, while the fitted horizon holds at
    423.7 / 432.5 / 426.6 px. A selection artefact cannot survive that.

    ⚠️ **A per-pair constant row offset (pitch jitter, EIS) was also tested and NOT
    supported.** Eliminating one by within-pair de-meaning made the robust cost
    WORSE (1.348 against 1.340) and sent the fit to a degenerate horizon of 120 px.
    The offset is therefore not modelled -- deliberately, and on evidence.
    """
    V = np.concatenate([u[:, 1] for u, _, _ in tracks])
    DV = np.concatenate([b[:, 1] - a[:, 1] for a, b, _ in tracks])
    DD = np.concatenate([np.full(len(a), p[0]) for a, _, p in tracks])
    GRP = np.concatenate([np.full(len(a), i) for i, (a, _, _) in enumerate(tracks)])
    rng = np.random.default_rng(seed)

    def at(vh, sel=None):
        v, dv, d = (V, DV, DD) if sel is None else (V[sel], DV[sel], DD[sel])
        q = v - vh
        qp = q + dv
        ok = (q > 25) & (dv > 0.5) & (qp > q)
        if ok.sum() < 150:
            return None
        a = d[ok] * q[ok] * qp[ok] / (qp[ok] - q[ok])
        qq = q[ok]
        med = np.median(a)
        keep = (a > 0.3 * med) & (a < 3.0 * med)
        if keep.sum() < 100:
            return None
        aa, qq = a[keep], qq[keep]
        i, j = rng.integers(0, len(aa), 5000), rng.integers(0, len(aa), 5000)
        s = np.abs(qq[i] - qq[j]) > 20
        if s.sum() < 100:
            return None
        slope = float(np.median((aa[i][s] - aa[j][s]) / (qq[i][s] - qq[j][s])))
        return dict(slope=slope, fh=float(np.median(aa)), n=int(keep.sum()),
                    rel_iqr=float((np.percentile(aa, 75) - np.percentile(aa, 25))
                                  / np.median(aa)))

    def solve(sel=None):
        sl, vs = [], []
        for vh in grid:
            r = at(float(vh), sel)
            if r:
                sl.append(r["slope"])
                vs.append(vh)
        if len(sl) < 4:
            return None
        sl, vs = np.array(sl), np.array(vs)
        if not (sl.min() < 0 < sl.max()):
            return None                       # zero crossing not bracketed
        vh = float(np.interp(0.0, sl, vs))
        r = at(vh, sel)
        return None if r is None else (vh, r)

    base = solve()
    if base is None:
        return None
    vh, r = base
    out = dict(horizon=vh, fh=r["fh"], n=r["n"], rel_iqr=r["rel_iqr"],
               n_points=int(len(V)), n_pairs=len(tracks),
               row_span=[float(V.min()), float(V.max())])

    groups = np.unique(GRP)
    if len(groups) >= 5 and n_boot:
        boot = []
        for _ in range(n_boot):
            pick = rng.choice(len(groups), len(groups), replace=True)
            sel = np.concatenate([np.flatnonzero(GRP == groups[p]) for p in pick])
            got = solve(sel)
            if got:
                boot.append((got[0], got[1]["fh"]))
        if len(boot) >= 30:
            b = np.array(boot)
            out["horizon_ci"] = [float(np.percentile(b[:, 0], 2.5)),
                                 float(np.percentile(b[:, 0], 97.5))]
            out["fh_ci"] = [float(np.percentile(b[:, 1], 2.5)),
                            float(np.percentile(b[:, 1], 97.5))]
            out["n_boot"] = len(boot)
    return out


def observability(tracks, P: dict, height: float):
    """Singular values of the (f*h, v_h, yaw, height) Jacobian, in pixels per unit.

    This is the honest version of "f and h are degenerate": rather than asserting
    it, differentiate the actual reprojection on the actual tracks and report the
    spectrum. A direction whose singular value is ~0 is not measurable from this
    data no matter which optimiser is pointed at it.
    """
    fh = P["fx"] * height
    base = dict(P)
    steps = dict(fh=fh * 0.01, vh=2.0, yaw=np.deg2rad(0.2), height=height * 0.02)

    def r_of(fh_, vh_, yaw_, h_):
        Q = dict(base)
        Q["height"] = h_
        Q["fx"] = fh_ / h_
        Q["yaw"] = yaw_
        try:
            Q["pitch"] = LS.pitch_for_horizon(Q, vh_)
        except Exception:
            return None
        out = []
        for uv0, uv1, pose in tracks:
            out.append((predict(uv0, Q, pose) - uv1).ravel())
        return np.concatenate(out)

    vh = LS.horizon_row(P)
    r0 = r_of(fh, vh, P["yaw"], height)
    if r0 is None:
        return None
    cols, names = [], ["f*h(+1%)", "horizon(+2px)", "yaw(+0.2deg)", "height(+2%)"]
    for nm, args in zip(names, [(fh + steps["fh"], vh, P["yaw"], height),
                                (fh, vh + steps["vh"], P["yaw"], height),
                                (fh, vh, P["yaw"] + steps["yaw"], height),
                                (fh, vh, P["yaw"], height + steps["height"])]):
        r1 = r_of(*args)
        cols.append(np.zeros_like(r0) if r1 is None else (r1 - r0))
    J = np.stack(cols, axis=1)
    keep = np.isfinite(J).all(axis=1)
    J = J[keep]
    if len(J) < 10:
        return None
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    return dict(names=names, sv=S.tolist(), n_rows=int(len(J)),
                rms_px_per_step=[float(np.sqrt((c[keep] ** 2).mean())) for c in cols],
                worst_dir=dict(zip(names, [float(v) for v in Vt[-1]])),
                cond=float(S[0] / S[-1]) if S[-1] > 0 else float("inf"))


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def build_tracks(run: pathlib.Path, n_anchors: int, steps=(3, 6), max_pts=600,
                 min_speed=8.0, fps=29.922, use_road_mask=True, verbose=False):
    """Tracked correspondences with their odometer pose, over the clip."""
    import run_real as RR
    recs = RR.load_records(run)
    frames = run / "frames"
    usable = [r for r in recs if r["complete"] and r["speed_ms"] > min_speed
              and (frames / f"{int(r['frame']):06d}.jpg").exists()]
    picks = np.linspace(0, len(usable) - 1, n_anchors).astype(int)
    tracks = []
    for pi in picks:
        a = usable[pi]
        f0 = int(a["frame"])
        img0 = cv2.imread(str(frames / f"{f0:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        if img0 is None:
            continue
        m0 = road_mask(img0) if use_road_mask else None
        if use_road_mask and m0.sum() == 0:
            continue
        t = np.asarray(a["t"], float)
        ax, ay, ayaw = (np.asarray(a[k], float) for k in ("x", "y", "yaw"))
        for st in steps:
            fk = f0 + st
            p = frames / f"{fk:06d}.jpg"
            dt = st / fps
            if not p.exists() or dt > t.max():
                continue
            img1 = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img1 is None:
                continue
            uv0, uv1 = track_pair(img0, img1, n_pts=max_pts, mask=m0)
            if len(uv0) < 30:
                continue
            pose = (float(np.interp(dt, t, ax)), float(np.interp(dt, t, ay)),
                    float(np.interp(dt, t, ayaw)))
            tracks.append((uv0, uv1, pose))
            if verbose:
                print(f"    frame {f0}+{st}  tracks {len(uv0):4d}  "
                      f"pose ({pose[0]:+.2f}, {pose[1]:+.2f}, "
                      f"{np.rad2deg(pose[2]):+.2f} deg)")
    return tracks


# --------------------------------------------------------------------------
# synthetic self-test
# --------------------------------------------------------------------------
def synth_tracks(P_true, n=8, speed=22.0, dt=0.1, seed=0, outlier_frac=0.35):
    """Exact ground correspondences plus a realistic dose of NON-ground outliers.

    The outliers are the point of the test. A fit that recovers the truth on clean
    ground points proves nothing about this recording, where the diagnosed problem
    IS the non-ground content -- so a third of the points here are static in the
    image, which is what a guardrail, a bonnet edge or a distant hillside look
    like to the tracker.
    """
    rng = np.random.default_rng(seed)
    tracks = []
    for i in range(n):
        g = np.stack([rng.uniform(6.0, 45.0, 400), rng.uniform(-8.0, 8.0, 400)], axis=1)
        uv0 = BC.project_ground(g, P_true)
        D = speed * dt * (1 + i % 3)
        pose = (D, 0.0, 0.0)
        c, s = 1.0, 0.0
        gk = np.stack([g[:, 0] - D, g[:, 1]], axis=1)
        uv1 = BC.project_ground(gk, P_true)
        ok = np.isfinite(uv0).all(1) & np.isfinite(uv1).all(1)
        ok &= (uv0[:, 1] > 0.55 * 1080) & (uv0[:, 1] < 1080) & (uv0[:, 0] > 0) & (uv0[:, 0] < 1920)
        ok &= (uv1[:, 1] > 0.5 * 1080) & (uv1[:, 1] < 1080)
        a, b = uv0[ok], uv1[ok]
        m = max(1, int(outlier_frac / (1 - outlier_frac) * len(a)))
        sa = np.stack([rng.uniform(0, 1920, m), rng.uniform(0.55 * 1080, 1080, m)], axis=1)
        a = np.concatenate([a, sa])
        b = np.concatenate([b, sa + rng.normal(0, 0.3, sa.shape)])   # static in image
        if len(a) > 40:
            tracks.append((a, b, pose))
    return tracks


def self_test() -> int:
    P_true = dict(BC.DEFAULTS)
    P_true.update(yaw=np.deg2rad(-7.0), pitch=np.deg2rad(-0.6), height=1.03,
                  fx=1560.0, lateral=-0.12, longitudinal=2.10)
    vh_true = LS.horizon_row(P_true)
    fh_true = P_true["fx"] * P_true["height"]
    tracks = synth_tracks(P_true)
    npts = sum(len(a) for a, _, _ in tracks)
    print(f"synthetic: {len(tracks)} pairs, {npts} correspondences "
          f"(35% static-in-image outliers)")
    print(f"truth: f*h {fh_true:.1f}   horizon {vh_true:.2f}   "
          f"yaw {np.rad2deg(P_true['yaw']):.2f} deg\n")

    obs = observability(tracks, P_true, P_true["height"])
    if obs:
        print("observability on the truth (pixels of reprojection per step):")
        for nm, v in zip(obs["names"], obs["rms_px_per_step"]):
            print(f"    {nm:16s} {v:8.3f} px rms")
        print(f"    singular values {['%.1f' % v for v in obs['sv']]}  "
              f"cond {obs['cond']:.3g}")
        print("    least-observable direction: "
              + "  ".join(f"{k} {v:+.2f}" for k, v in obs["worst_dir"].items()))

    ok = True
    print("\nrecovery from deliberately wrong starts:")
    for dfh, dvh in ((0.0, 0.0), (+0.30, 0.0), (-0.25, 0.0),
                     (0.0, +25.0), (+0.30, -25.0)):
        P0 = dict(P_true)
        Q, info = fit(tracks, P0, P_true["height"],
                      fh_true * (1 + dfh), vh_true + dvh)
        e_fh = info["fh"] / fh_true - 1.0
        e_vh = info["vh"] - vh_true
        good = abs(e_fh) < 0.03 and abs(e_vh) < 3.0
        ok &= good
        print(f"   {'OK ' if good else 'BAD'} start f*h {100*dfh:+5.0f}% "
              f"horizon {dvh:+5.1f} px  ->  f*h {info['fh']:7.1f} ({100*e_fh:+5.1f}%)"
              f"   horizon {info['vh']:7.2f} ({e_vh:+5.2f} px)"
              f"   yaw {info['yaw_deg']:+.2f} deg   cost {info['cost0']:.4f}"
              f"->{info['cost']:.4f}")
    rf = row_flow_fit(tracks, n_boot=0)
    if rf is None:
        print("\nrow-flow fit: NOT BRACKETED on synthetic — estimator broken")
        ok = False
    else:
        e_fh = rf["fh"] / fh_true - 1.0
        e_vh = rf["horizon"] - vh_true
        good = abs(e_fh) < 0.05 and abs(e_vh) < 5.0
        ok &= good
        print(f"\nrow-flow (vertical flow only, 2 parameters):")
        print(f"   {'OK ' if good else 'BAD'} f*h {rf['fh']:7.1f} ({100*e_fh:+.1f}%)"
              f"   horizon {rf['horizon']:7.2f} ({e_vh:+.2f} px)"
              f"   n {rf['n']}  relative IQR {rf['rel_iqr']:.3f}")

    print("\nself-test", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--run", type=pathlib.Path, default=None)
    ap.add_argument("--anchors", type=int, default=40)
    ap.add_argument("--steps", type=int, nargs="+", default=[3, 6, 9])
    ap.add_argument("--max-pts", type=int, default=600)
    ap.add_argument("--height", type=float, default=1.03)
    ap.add_argument("--fx", type=float, default=1478.3)
    ap.add_argument("--yaw", type=float, default=-7.01)
    ap.add_argument("--horizon", type=float, default=523.4)
    ap.add_argument("--height-scan", type=float, nargs="+",
                    default=[0.90, 1.03, 1.17, 1.30, 1.45, 1.60])
    ap.add_argument("--no-road-mask", action="store_true",
                    help="track anywhere in the lower image instead of on the "
                         "markings — kept so the 2.3%%-inlier failure is reproducible")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a = ap.parse_args()

    if a.self_test:
        return self_test()

    import run_real as RR
    run = a.run or RR.RUN
    print("building tracks…")
    tracks = build_tracks(run, a.anchors, tuple(a.steps), a.max_pts,
                          use_road_mask=not a.no_road_mask, verbose=a.verbose)
    npts = sum(len(u) for u, _, _ in tracks)
    print(f"{len(tracks)} frame pairs, {npts} forward-backward-checked "
          f"correspondences\n")
    if npts < 500:
        print("too few tracks to fit — the tracker, not the geometry, is the blocker")
        return 1

    P0 = dict(RR.NOMINAL)
    P0.update(yaw=np.deg2rad(a.yaw), height=a.height, fx=a.fx)
    P0["pitch"] = LS.pitch_for_horizon(P0, a.horizon)
    fh0 = a.fx * a.height
    out = dict(n_pairs=len(tracks), n_points=int(npts), fh_start=fh0,
               horizon_start=a.horizon)

    obs = observability(tracks, P0, a.height)
    if obs:
        out["observability"] = obs
        print("observability on the REAL tracks (reprojection px per step):")
        for nm, v in zip(obs["names"], obs["rms_px_per_step"]):
            print(f"    {nm:16s} {v:8.3f} px rms")
        print(f"    singular values {['%.1f' % v for v in obs['sv']]}  "
              f"cond {obs['cond']:.3g}")
        print("    least-observable direction: "
              + "  ".join(f"{k} {v:+.2f}" for k, v in obs["worst_dir"].items()))

    rf = row_flow_fit(tracks)
    if rf:
        out["row_flow"] = rf
        print(f"\nROW-FLOW fit (vertical flow only — the 2 parameters this geometry has)")
        print(f"    {rf['n_points']} points, {rf['n_pairs']} pairs, rows "
              f"{rf['row_span'][0]:.0f}-{rf['row_span'][1]:.0f}, {rf['n']} after trimming")
        print(f"    horizon  {rf['horizon']:8.2f} px"
              + (f"   95% CI [{rf['horizon_ci'][0]:.1f}, {rf['horizon_ci'][1]:.1f}]"
                 if "horizon_ci" in rf else ""))
        print(f"    f*h      {rf['fh']:8.1f} px*m   (relative IQR {rf['rel_iqr']:.3f})"
              + (f"   95% CI [{rf['fh_ci'][0]:.0f}, {rf['fh_ci'][1]:.0f}]"
                 if "fh_ci" in rf else ""))
        print("    f for a range of heights — the height is NOT from this fit:")
        for hh in (1.17, 1.45, 1.60, 1.75, 1.90):
            f_ = rf["fh"] / hh
            print(f"       h {hh:4.2f} m -> f {f_:7.1f} px  HFOV "
                  f"{np.rad2deg(2*np.arctan(1920/(2*f_))):5.1f} deg"
                  + ("   <== inside the 1356-1628 device band"
                     if 1356 <= f_ <= 1628 else ""))
    else:
        print("\nROW-FLOW fit: zero crossing NOT bracketed — no horizon from the flow")

    print(f"\nfit (f*h, horizon, yaw) at h = {a.height:.3f} m")
    Q, info = fit(tracks, P0, a.height, fh0, a.horizon)
    n_ref = int(npts)
    st = inlier_stats(tracks, Q, n_ref=n_ref)
    st0 = inlier_stats(tracks, P0, n_ref=n_ref)
    print(f"    cost {info['cost0']:.5f} -> {info['cost']:.5f}  ({info['nfev']} evals)")
    print(f"    f*h      {fh0:8.1f}  ->  {info['fh']:8.1f} px*m")
    print(f"    horizon  {a.horizon:8.2f}  ->  {info['vh']:8.2f} px")
    print(f"    yaw      {a.yaw:8.2f}  ->  {info['yaw_deg']:8.2f} deg")
    print(f"    f = f*h/h = {info['fh']/a.height:.0f} px  "
          f"(HFOV {np.rad2deg(2*np.arctan(1920/(2*info['fh']/a.height))):.1f} deg)")
    print(f"    reprojection: median {st0['med_px']:.2f} -> {st['med_px']:.2f} px "
          f"(scored {st0['n']}/{n_ref} -> {st['n']}/{n_ref} on-frame), "
          f"inliers(<3px) {100*st0['inlier_frac']:.1f}% -> {100*st['inlier_frac']:.1f}%")
    out["fit"] = info
    out["inliers"] = st
    out["inliers_start"] = st0

    # f*h is what the geometry gives; h is what it cannot. Scan it and show that
    # the SAME f*h comes back, so the reported f carries h's uncertainty openly
    # rather than pretending the fit resolved it.
    print(f"\nheight scan — does f*h move when h is forced?")
    scan = []
    for h in a.height_scan:
        Qh, ih = fit(tracks, P0, float(h), fh0, a.horizon)
        sh = inlier_stats(tracks, Qh, n_ref=n_ref)
        scan.append(dict(height=float(h), fh=ih["fh"], vh=ih["vh"],
                         cost=ih["cost"], f=ih["fh"] / h,
                         inlier_frac=sh["inlier_frac"], med_px=sh["med_px"]))
        print(f"    h {h:5.2f} m  ->  f*h {ih['fh']:7.1f}  f {ih['fh']/h:7.1f} px  "
              f"horizon {ih['vh']:7.2f}  cost {ih['cost']:.5f}  "
              f"median {sh['med_px']:5.2f} px  inliers {100*sh['inlier_frac']:4.1f}%")
    out["height_scan"] = scan
    c = np.array([s["cost"] for s in scan])
    hh = np.array([s["height"] for s in scan])
    print(f"    cost range over the scan {c.min():.5f}-{c.max():.5f} "
          f"({100*(c.max()/c.min()-1):.1f}% spread), best h {hh[c.argmin()]:.2f} m")
    if c.max() / c.min() < 1.05:
        print("    => h is NOT determined by this fit (as the geometry predicts); "
              "f*h is the result and f needs h from the lateral metric.")

    if a.json:
        a.json.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
