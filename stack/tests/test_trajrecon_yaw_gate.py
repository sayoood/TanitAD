"""The lane-yaw credibility gate must not judge a measurement against a default.

WHY THIS EXISTS
---------------
MEASURED 2026-08-08 on the `14-19-54` recording. `lane_calib` measured a mount
yaw of **-7.01 deg** with a half-split spread of **0.20 deg** — its own stability
gate is 0.6 deg, so the measurement was comfortably stable — and the pipeline
threw it away with:

    yaw declined: -7.01 deg is 7.0 deg from the FOE, not credible

There was no FOE. `camera.py`'s FOE fit had already failed
("FOE fit produced no usable flow - kept nominal"), which leaves `cam.yaw` at the
**nominal 0.0**. The gate at `lane_calib.py:243` compares against `cam.yaw`, so
with no FOE it degenerates into *"reject any mount yaw beyond 4 deg of dead
ahead"* — and it rejects **hardest exactly when the mount is most crooked**,
which is precisely when the correction is worth having.

The cost was not subtle. An independent VP fit over 154 straight frames / 6689
segments gives -6.05 deg [95% CI -6.28, -5.83], confirming the pipeline's own
number. Shipping 0.0 instead put **4.9 m of lateral error at 40 m** into the
rendered overlay — outside the ego lane from ~15 m onward.

Three layers made it quiet, and the third is the one to remember:

  1. the gate rejected a good measurement against a placeholder;
  2. the warning *called* the placeholder "the FOE", so the log looked like a
     real disagreement between two estimates;
  3. `lane_calib` returns `yaw_deg if yaw_ok else None`, so `pipeline.py:494`'s
     `if res.yaw_deg is not None` was False and the one diagnostic that spelled
     out the damage — `"... (-7.01 deg, 4.89 m at 40 m)"` — **never printed.**
     The louder the error, the quieter the log.

These are source-level checks on purpose: they need neither the trajrecon extras
nor torch, so they run everywhere the rest of the suite runs.
"""

from __future__ import annotations

import pathlib
import re

PKG = pathlib.Path(__file__).resolve().parents[1] / "tanitad" / "data" / "trajrecon"
PIPELINE = (PKG / "pipeline.py").read_text(encoding="utf-8")
CAMERA = (PKG / "camera.py").read_text(encoding="utf-8")
LANE = (PKG / "lane_calib.py").read_text(encoding="utf-8")


def test_caller_passes_the_gate_threshold_rather_than_taking_the_default():
    """Upstream let `max_yaw_correction_deg` default to 4.0 unconditionally."""
    assert "max_yaw_correction_deg=" in PIPELINE, (
        "pipeline.py no longer sets max_yaw_correction_deg — the lane-VP yaw is "
        "back to being gated against cam.yaw even when no FOE was measured")


def test_the_threshold_is_conditioned_on_whether_a_foe_actually_exists():
    """The whole point: the gate is only meaningful with a real FOE to agree with."""
    assert re.search(r'foe_measured\s*=\s*"extrinsics"\s+in\s+cam\.source', PIPELINE), (
        "the FOE-measured test is gone; without it the caller cannot know "
        "whether cam.yaw is a measurement or a nominal placeholder")
    assert re.search(r"max_yaw_correction_deg\s*=\s*\(?\s*4\.0\s+if\s+foe_measured\s+else\s+15\.0",
                     PIPELINE), (
        "the conditional threshold is gone — with no FOE the bound must be a "
        "plausibility limit on mount yaw, not an agreement limit against 0.0")


def test_the_foe_measured_key_still_means_what_we_think():
    """Guard the detection itself.

    `"extrinsics" in cam.source` is only a valid "the FOE succeeded" test while
    `camera.py` writes that key **exclusively** in the success branch. If upstream
    ever sets it on a failure path too, the fix above silently stops working —
    so pin the invariant rather than trusting it.
    """
    success = re.findall(r'source\["extrinsics"\]\s*=', CAMERA)
    failures = re.findall(r'source\["extrinsics_note"\]\s*=', CAMERA)
    assert len(success) == 1, (
        f'camera.py writes source["extrinsics"] {len(success)} times; the '
        f'FOE-measured test assumes exactly one (the success branch)')
    assert len(failures) >= 3, (
        f'camera.py writes source["extrinsics_note"] only {len(failures)} times; '
        f'the nominal-fallback paths appear to have changed')


def test_the_plausibility_bound_matches_cameras_own_constant():
    """15 deg is not arbitrary — camera.py:405 uses it to sanity-check the FOE.

    Two different numbers for "a dashcam mount is never more crooked than this"
    would drift apart; this asserts they are still the same number.
    """
    assert "np.deg2rad(15)" in CAMERA, (
        "camera.py's mount-yaw plausibility constant changed; the 15.0 passed "
        "from pipeline.py was chosen to match it and must be updated together")


def test_the_gate_being_guarded_still_exists():
    """If upstream fixes this properly, this whole file should be revisited.

    A guard for a defect that no longer exists is rot — it should fail loudly
    rather than pass vacuously forever.
    """
    assert "max_yaw_correction_deg" in LANE and "not credible" in LANE, (
        "lane_calib's credibility gate has changed shape upstream — re-derive "
        "whether the pipeline.py workaround is still the right fix")


def test_every_operator_override_is_labelled_as_not_measured():
    """An override must never be readable as a measurement.

    The ground plane cannot separate focal length from camera height
    (`scale_calib.py`), so settling this geometry needs externally supplied values
    pinned and the rest re-derived. That is legitimate — but a validation run whose
    provenance says "yaw_deg: -7.01" with no qualifier is indistinguishable from a
    measured one three weeks later, and `calibration.json` is exactly the artifact
    someone will quote. So every override writes "OPERATOR OVERRIDE ... (not
    measured)" into the provenance, and this pins that.
    """
    for flag in ("--cam-yaw", "--cam-pitch", "--cam-roll", "--horizon-row",
                 "--lock-lateral", "--focal-px"):
        assert flag in PIPELINE, f"{flag} override is gone"
    n_override = PIPELINE.count("OPERATOR OVERRIDE")
    assert n_override >= 5, (
        f"only {n_override} provenance strings say OPERATOR OVERRIDE; every "
        f"override path must label itself as not measured")
    assert "(not measured)" in PIPELINE


def test_focal_override_is_applied_before_the_horizon_row_override():
    """Order is load-bearing, not cosmetic.

    `--horizon-row` is converted to a pitch by ``atan((row - cy) / fy)``. If the
    focal override ran afterwards, that conversion would use the OLD focal and the
    horizon would land on a different row than the operator asked for — silently,
    because both flags would still appear in the provenance with the values that
    were requested.

    Concrete on this corpus: the row flow gives ``f*h = 2668`` px·m
    (95% CI [2547, 2798]) against the shipped ``1478.3 x 1.17 = 1729``, so a
    validation run supplies BOTH a new focal and a horizon row. At f 1438 vs 1478
    a horizon of 465 px would be misplaced by ~2 px, and at the shipped-vs-measured
    focal ratio the error grows with how wrong the focal was.
    """
    i_focal = PIPELINE.index('"focal_px", None')
    i_horiz = PIPELINE.index('"horizon_row", None')
    assert i_focal < i_horiz, (
        "--focal-px must be applied BEFORE --horizon-row: the horizon->pitch "
        "conversion divides by fy, so a later focal override silently moves the "
        "horizon away from the row that was requested")


def test_locked_lateral_actually_blocks_the_lane_calib_override():
    """`--lock-lateral` is pointless unless it beats the estimator.

    `lane_calib` assigns `args.lateral_offset = res.lateral_offset_m`, so without
    the guard the operator value is silently replaced and the run reports a number
    the operator never asked for.
    """
    assert re.search(r"res\.lateral_offset_m is not None and not "
                     r"getattr\(args, \"lock_lateral\", False\)", PIPELINE), (
        "the lock-lateral guard on lane_calib's assignment is missing")


def test_horizon_row_override_is_expressed_in_the_observable_quantity():
    """Pitch is not directly observable; the row the lane markings converge to is.

    MEASURED 2026-08-08: lane VP row 523.4 px, 95% CI [513.3, 534.8], against the
    pipeline's 464.4 — 59 px, 2.29 deg of pitch. Being able to set the row directly
    is what makes that measurement usable without hand-converting it.
    """
    assert "np.arctan2(row - cam.cy, cam.fy)" in PIPELINE


def test_declined_yaw_still_collapses_to_none_so_the_diagnostic_is_suppressed():
    """Pin layer 3, the one that hid the error.

    `pipeline.py` only logs the metres-of-error line when `res.yaw_deg is not
    None`, and `lane_calib` nulls it on decline. So a rejected yaw prints no
    magnitude. This test documents that coupling; if either side changes, the
    suppression may be gone (good) and this note should be revisited.
    """
    assert "yaw_deg if yaw_ok else None" in LANE
    assert "if res.yaw_deg is not None:" in PIPELINE


def test_nominal_focal_is_labelled_as_the_uncropped_lens():
    """The nominal HFOV describes the LENS, not the recorded video.

    Video stabilisation crops, so the focal actually in force is larger than the
    nominal by the crop factor. MEASURED on `2026-08-08_14-19-54-android`
    (SM-G990B, EIS confirmed on by the operator): the lens is 26 mm-equivalent =
    79.5 deg diagonal = HFOV 67.3 deg = 1442 px at 1920 wide, against a true
    **1713 px** — a **1.19x crop**, so the nominal ran 16% low.

    That error does not announce itself. The ground plane sees only the product
    `f*h`, so a low focal is absorbed by an inflated HEIGHT — 1.17 m recorded
    against a true ~1.43 m — and every number downstream stays self-consistent
    while the overlay is visibly wrong. Days went into hunting a focal error that
    was never there. The provenance string is the only thing standing between a
    later reader and that same mistake, so it is pinned here.
    """
    assert "UNCROPPED" in CAMERA, (
        "camera.py's nominal intrinsics no longer warn that the quoted HFOV is the "
        "uncropped lens; with EIS on it is a LOWER BOUND on the recorded focal")
    assert "EIS crops" in CAMERA
    assert re.search(r'"intrinsics":\s*f?"nominal HFOV=.*UNCROPPED', CAMERA, re.S), (
        "the nominal-focal provenance string no longer carries the warning")


def test_flow_calib_solves_for_the_horizon_rather_than_taking_it():
    """The whole reason this module exists next to `scale_calib`.

    A road point's range is `f*h / (v - v_horizon)`. `scale_calib.estimate_fh` is
    PASSED `cam.horizon_v()`, so a horizon that is wrong biases every track by an
    amount depending on where in the frame it sat — which is how it returned a 62%
    spread over 1576 tracks on 2026-08-08 and declined. `flow_calib` scans the
    horizon and takes the zero-trend crossing, so it cannot be poisoned that way.

    Also pinned: it must WARN when its horizon disagrees with the camera's. On that
    recording the flow family lands at 431–448 px and the paint family at 465–485,
    ~40 px apart, and only the paint set is validated against the markings. An
    operator who silently inherits whichever ran last gets a 12% error in f*h.
    """
    FLOW = (PKG / "flow_calib.py").read_text(encoding="utf-8")
    assert "_solve" in FLOW and "zero crossing" in FLOW, (
        "flow_calib no longer brackets a zero-trend horizon — if it now takes the "
        "horizon as an input it has the same defect as scale_calib")
    assert "LARGE disagreement" in FLOW, (
        "flow_calib no longer warns when its horizon disagrees with the camera's")
    assert "--flow-calib" in PIPELINE and "flow_calib" in PIPELINE, (
        "the pipeline no longer calls flow_calib")
    i_flow = PIPELINE.index("from trajlib import flow_calib")
    i_sc = PIPELINE.index("vals, note = SC.estimate_fh")
    assert i_flow < i_sc, (
        "flow_calib must be tried BEFORE scale_calib: it is the one that does not "
        "inherit a wrong horizon")


def test_camera_height_can_be_made_authoritative():
    """`--cam-height` alone does not survive the estimators.

    It seeds the camera before `scale_calib` runs, and `scale_calib.solve` then
    RESETS `cam.height_m` from f*h and the lane width. MEASURED 2026-09-13: a run
    given `--cam-height 1.60` rendered at **1.622**, because scale_calib recomputed
    it and nothing put the operator's value back.

    That matters more than 1.4%: on this corpus the height is the one parameter no
    instrument could measure (every lateral estimator turned out to confirm whatever
    height it was given), so it is supplied from outside — the VW Caddy's mount
    geometry, or a tape measure. A supplied value that an estimator can silently
    discard is worse than no override at all, because the provenance still shows
    what was asked for.

    Every other override sits in the post-estimator block for exactly this reason.
    `--lock-height` puts the height there too.
    """
    assert "--lock-height" in PIPELINE, "the height override is gone"
    i_lock = PIPELINE.index('"lock_height", False')
    i_scale = PIPELINE.index("ScaleResult") if "ScaleResult" in PIPELINE else None
    assert 'cam.height_m = float(args.cam_height)' in PIPELINE, (
        "--lock-height no longer actually reassigns cam.height_m")
    assert "OPERATOR OVERRIDE --lock-height (not measured)" in PIPELINE, (
        "a locked height must be labelled as not measured, like every other override")


def test_the_ridge_operator_is_sized_by_the_paint_not_by_the_rows_rank():
    """`w` must come from the geometry, or the far field is detected out of existence.

    The operator is `2*row - row(-w) - row(+w)`: it responds only when `w` straddles
    the marking and returns almost nothing when it samples entirely INSIDE the paint.
    The original schedule ramps `w` from 2 to 28 across whatever band it is handed,
    which is right at the near edge of `lane_calib`'s own 0.55-0.86 H band and asks
    for `w = 2` at the far edge where this geometry wants ~10.

    MEASURED 2026-09-14 on the 14-19-54 recording: in a narrow far band, association
    to the painted line succeeded at 40 m in **2 frames of 150** with the default
    schedule, while the line is plainly visible in every one of them. That is the
    far field — where the lane's angular information lives, and where the yaw fit
    that this whole test file is about gets its leverage.

    The schedule is `w ~ 2 + 0.5 * paint_m * (y - v_h) / h`, which carries NO focal
    length: eliminating range between `v = v_h + f*h/x` and a width `0.5*paint*f/x`
    cancels `f` exactly.
    """
    assert "def paint_width_schedule" in LANE, (
        "the physical ridge-width schedule is gone; the far field goes with it")
    assert "w_of_row=paint_width_schedule(cam)" in LANE, (
        "_ridge_points no longer receives the physical schedule — the lateral-offset "
        "fit is back to the rank-based ramp")
    assert "paint_width_schedule(cam)" in LANE.split("HoughLinesP")[1][:200] or \
           "_ridge_mask(g, r0, r1, paint_width_schedule(cam))" in LANE, (
        "the yaw fit's Hough segments are still built from the rank-based mask")
    assert "cam.horizon_v()" in LANE, (
        "the schedule must be anchored on the horizon row, not on the band edges")


def test_the_physical_ridge_width_restores_the_response_the_default_zeroes():
    """Behavioural, not source-level: build the failure and show the fix removes it.

    ⚠️ The first version of this test asserted on the NUMBER of ridge points and it
    was too weak — on a clean synthetic line the rank-based schedule still fires on
    the line's two EDGES (it found 417 points against the fix's 433). The mechanism
    is not "the line is invisible", it is that the response AT THE LINE'S CENTRE
    collapses to zero when `w` sits inside the paint, so all that survives is a pair
    of weak edge responses that a 99.2nd-percentile threshold then discards on a real,
    textured road. The centre response is the quantity to assert on.
    """
    import importlib
    import sys
    import types
    import numpy as np
    # `tanitad.data.__init__` imports torch, which this container does not have, and
    # the ridge operator needs none of it. Stub the two parent packages so the
    # module's own relative imports still resolve, then import it for real.
    root = PKG.parents[2]
    for name, path in (("tanitad", root / "tanitad"),
                       ("tanitad.data", root / "tanitad" / "data"),
                       ("tanitad.data.trajrecon", PKG)):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [str(path)]
            sys.modules[name] = mod
    LC = importlib.import_module("tanitad.data.trajrecon.lane_calib")

    class Cam:
        fx = fy = 1533.0
        cx, cy = 960.0, 540.0
        height_m = 1.586

        def horizon_v(self):
            return 448.4

    r0, r1 = 594, 928             # lane_calib's own band, 0.55-0.86 H at 1080
    y = 596                       # the FAR edge of it: ~16.6 m for this geometry
    row = np.full(1920, 90, np.int16)
    row[700:716] = 230            # a 0.18 m marking is ~16 px wide at that range

    def response(w, at):
        return int(2 * row[at] - row[at - w] - row[at + w])

    w_default = max(2, int(round(2 + 26 * (y - r0) / max(1, r1 - r0))))
    w_fixed = LC.paint_width_schedule(Cam())(y)
    centre = 708

    assert w_default < 4, f"the rank-based schedule no longer undersizes here (w={w_default})"
    assert w_fixed >= 5, f"the physical schedule is not straddling the paint (w={w_fixed})"
    assert response(w_default, centre) == 0, (
        "the premise of the fix is wrong: an operator inside the paint should see "
        "no ridge at the line's centre")
    assert response(w_fixed, centre) > 100, (
        f"the physical schedule recovers only {response(w_fixed, centre)} at the line "
        f"centre — it is not straddling the marking")


def test_a_vehicle_override_renames_the_vehicle():
    """A label must not outlive the values it describes.

    MEASURED 2026-09-14: a run given `--wheelbase 3.105` (VW Caddy Maxi) printed

        Steering [Audi A6 e-tron, L=3.105 m, ratio 15.9:1]

    The wheelbase was applied correctly; only the NAME was stale. But a reader sees a
    car's name next to numbers and takes the numbers as that car's, and the remaining
    fields (steering ratio, understeer gradient, lock, track) really are still the
    Audi's -- so the line asserts more than the run knows.

    This is the same failure as the yaw gate calling a nominal placeholder "the FOE":
    the defect was never the number, it was a label that made a placeholder look like
    a measurement. Overriding either dimension now relabels the vehicle unless the
    operator names it.
    """
    assert "--vehicle-name" in PIPELINE, (
        "no way to name the vehicle, so an override silently keeps the Audi's label")
    assert 'veh["name"] = args.vehicle_name' in PIPELINE, (
        "--vehicle-name is declared but never applied")
    assert "unnamed vehicle" in PIPELINE, (
        "an overridden wheelbase/ratio no longer relabels the vehicle — the panel is "
        "back to printing a car's name beside numbers that are not that car's")
    assert "other" in PIPELINE.split("unnamed vehicle")[1][:200], (
        "the relabel must still say which values are inherited, or it trades a wrong "
        "name for a claim that nothing is inherited")


def test_road_tracks_reject_correspondences_that_do_not_move_like_the_road():
    """The corridor mask reaches under the bonnet, and the bonnet is static.

    MEASURED 2026-09-14 on the 14-19-54 recording: `collect_road_tracks` projects a
    corridor of x in (5, 32) m, and that polygon spans source rows 529-1079 while the
    bonnet line sits at row 832 -- so 45 % of the mask's rows are not road. RANSAC
    fits the homography to the static part, and `plane_calib` then reports a camera
    height of 42.0 m, a pitch of +17.8 deg and 0 of 90 pairs admissible. All of that
    surfaces as one line: "plane calibration produced too few usable homographies".

    A bonnet-row mask would need the bonnet row. This does not: a point on the road
    must move by roughly what the KNOWN vehicle displacement predicts, so anything
    moving far less is not on the road -- bonnet, wiper, dashboard reflection, or a
    stopped vehicle ahead.

    The gate must stay ONE-SIDED. Rejecting points that move too MUCH as well would
    make it a fit against `cam`, and `cam` is what the estimator exists to produce.
    """
    GROUND = (PKG / "ground_calib.py").read_text(encoding="utf-8")
    assert "def _drop_static" in GROUND, (
        "the static-content filter is gone; the bonnet is back in the homography")
    assert "_drop_static(cam, a, b, dp, dpsi" in GROUND, (
        "_drop_static exists but collect_road_tracks no longer calls it")
    assert "d_meas > min_frac * d_pred" in GROUND, (
        "the motion test is no longer the one-sided 'moves too little' comparison")
    assert "d_meas <" not in GROUND, (
        "a two-sided motion gate turns the filter into a fit against cam, which is "
        "the very thing the estimator is supposed to measure")


def test_the_corridor_fades_with_range_and_the_fade_is_per_pixel():
    """Beyond ~53 m a solid ribbon asserts a precision the geometry does not have.

    MEASURED on the 14-19-54 recording: after removing the camera (common-mode
    rotation <= 0.25 deg) and the instrument (0.088 deg), the per-frame angle
    between the drawn corridor and the lane still has a real spread of about
    1 deg. One degree is 0.52 m of lateral uncertainty at 30 m and **0.93 m at
    53 m** -- exactly half the 1.855 m ribbon's width. Drawing a crisp edge past
    that point states more than is known.

    The renderer could not express this before: it blended the whole band with a
    single `addWeighted`, so alpha was one number for the entire ribbon. The fade
    needs a PER-PIXEL alpha mask, and the fill has to be rasterised into it
    without antialiasing -- abutting quads antialias against each other and a
    later quad replaces rather than accumulates, which would leave a lower-alpha
    seam along every quad boundary.
    """
    VIZ = (PKG / "viz.py").read_text(encoding="utf-8")
    assert "fade_start_m" in VIZ and "fade_end_m" in VIZ, (
        "the range fade is gone; the corridor is solid to the vanishing point again")
    assert "amask" in VIZ, (
        "no per-pixel alpha mask — a single addWeighted cannot fade with range")
    assert "cv2.addWeighted(overlay, alpha, out" not in VIZ, (
        "the single global-alpha blend is back, which silently disables the fade")
    assert "lineType=cv2.LINE_8" in VIZ, (
        "the alpha mask is antialiased again — abutting quads will seam")
    assert "fade_start_m=args.fade_start_m" in PIPELINE, (
        "the pipeline no longer passes the fade through, so the default is unreachable")


def test_a_faded_tick_is_not_labelled():
    """A crisp '4s / 88m' floating over nothing reads as a fault in the render."""
    VIZ = (PKG / "viz.py").read_text(encoding="utf-8")
    assert "if afade < 0.35:" in VIZ, (
        "tick labels are drawn regardless of the fade again")
    i_blend = VIZ.index("out[:] = np.clip(overlay")
    i_label = VIZ.index("for lab, org in labels:")
    assert i_label > i_blend, (
        "labels are emitted before the blend, so the alpha mask washes the text out")
