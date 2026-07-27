"""The closed-loop / pseudo-sim RE-RENDER, on the frame v5 actually uses.

WHAT WAS BROKEN (MEASURED 2026-07-27, ``V5_GATEABLE.md`` §1.4)
--------------------------------------------------------------
``clhorizon.sampling_homography`` is the ONE re-render behind both the gate
co-primary surface (``corridor_rollout``) and the mid-run held-out early-stop
(``pseudosim.pseudo_evaluate``). Its ``f=266`` / ``c=128`` are the DEPLOYED
256x256 pinhole crop's intrinsics. On v5's **176x624 cylindrical** frame at
``f_ref 305.5775`` it misplaces source pixels by a mean of **46.30 px** at the
mid-run gate's own ``+/-8 deg`` probe against a true shift of **42.67 px** —
99.08 % of pixels wrong by more than 1 px — while its CONTROL on the deployed
frame is **max 0.118 px**.

THE FOUR LOAD-BEARING TESTS
---------------------------
1. :func:`test_deployed_path_is_bit_identical` — a canonical frame (or
   ``frame=None``) routes to ``sampling_homography`` + ``warp_batch`` VERBATIM
   and the warped tensors are ``torch.equal``. **No published number moves.**
2. :func:`test_pinhole_field_reduces_to_the_shipped_homography` — the new
   projection-aware field IS the shipped homography on a pinhole frame, to
   ``< 1e-9`` px. The generalisation adds a projection; it does not change the
   physical model.
3. :func:`test_cylindrical_yaw_is_the_exact_pixel_shift` — on a cylinder a yaw
   is ``u -> u + f_ref*psi``, ``v -> v``, exactly and at arbitrary depth.
4. :func:`test_cylindrical_lateral_is_not_expressible_as_a_homography` — and
   the lateral axis is NOT, so the 3x3 representation genuinely cannot carry it
   (the resampler can).

Everything here is exact-vs-exact arithmetic: no data, no GPU, no cache.
"""
import math

import numpy as np
import pytest
import torch

from taniteval import clhorizon as CH
from taniteval import pseudosim as PS

# --- the v5 geometry, READ from the cache manifest, not re-derived --------- #
# /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl/_geometry.json
V5_CACHE_FRAME = {"height": 256, "width": 640, "f_ref": 305.5774907364391,
                  "projection": "cylindrical"}
#: `--v2-subframe 176x624` — a CENTRED slice, so f_ref and the projection are
#: inherited and the boresight stays at the child's own geometric centre
#: (`calib.centred_subframe` / `subframe_slice`, exact float identity).
V5_TRAIN_FRAME = {"height": 176, "width": 624, "f_ref": 305.5774907364391,
                  "projection": "cylindrical"}
PROBE_DYAW_DEG = 8.0            # tanitad.train.heldout_gate.probe_grid

# MEASURED anchors from raw/warp_geometry_audit_2026-07-27.json — named so a
# drift in either direction goes red.
DEFECT_MEAN_PX = 46.30
DEFECT_MAX_PX = 189.20
DEFECT_FRAC_GT1 = 0.9908
CONTROL_MAX_PX = 0.118


def _dest_grid(h, w):
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float64),
                            torch.arange(w, dtype=torch.float64), indexing="ij")
    return xs, ys


def _apply_H(H, u, v):
    """``warp_batch``'s source lookup, on a coordinate grid."""
    P = torch.stack([u, v, torch.ones_like(u)], dim=-1).reshape(-1, 3).T
    s = H.to(torch.float64) @ P
    return ((s[0] / s[2]).reshape(u.shape), (s[1] / s[2]).reshape(u.shape))


def _window(b=2, wn=3, c=3, h=256, w=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(b, wn, c, h, w, generator=g)


# =========================================================================== #
# 1. THE DEPLOYED PATH DOES NOT MOVE                                          #
# =========================================================================== #
class TestDeployedPathUnchanged:

    @pytest.mark.parametrize("dlat,dyaw", [(0.0, 0.0), (0.0, 8.0), (0.0, -8.0),
                                           (1.5, 3.0), (-2.0, -5.0),
                                           (3.0, 12.0)])
    def test_deployed_path_is_bit_identical(self, dlat, dyaw):
        """``frame=None`` == the pre-2026-07-27 call, tensor-for-tensor."""
        fw = _window()
        legacy = CH.warp_batch(
            fw, torch.stack([CH.sampling_homography(dlat, dyaw)] * fw.shape[0]))
        new = CH.warp_frames(fw, dlat, dyaw, None)
        assert torch.equal(legacy, new)

    def test_a_declared_canonical_frame_takes_the_same_verbatim_path(self):
        """Declaring the deployed geometry must not change one bit either."""
        fw = _window()
        dep = {"height": 256, "width": 256, "f_ref": 266.0,
               "projection": "pinhole"}
        assert torch.equal(CH.warp_frames(fw, 1.0, 4.0, None),
                           CH.warp_frames(fw, 1.0, 4.0, dep))
        prov = CH.assert_warp_frame(dep, fw)
        assert prov["is_deployed_frame"] is True
        assert "VERBATIM" in prov["path"]

    def test_a_real_tanitad_CanonicalFrame_is_accepted(self):
        """The object the cache/checkpoint actually carries, duck-typed."""
        from tanitad.data.calib import CANONICAL_256, CanonicalFrame
        fw = _window()
        assert torch.equal(CH.warp_frames(fw, 0.5, 2.0, None),
                           CH.warp_frames(fw, 0.5, 2.0, CANONICAL_256))
        f5 = CanonicalFrame(**V5_TRAIN_FRAME)
        assert CH.as_warp_frame(f5).tag() == CH.as_warp_frame(
            V5_TRAIN_FRAME).tag()

    def test_legacy_sentinel_bypasses_the_shape_check_and_says_so(self):
        fw = _window(h=16, w=16)
        prov = CH.assert_warp_frame(CH.LEGACY_WARP, fw)
        assert prov["legacy_unvalidated"] is True
        assert torch.equal(
            CH.warp_frames(fw, 0.0, 3.0, CH.LEGACY_WARP),
            CH.warp_batch(fw, torch.stack(
                [CH.sampling_homography(0.0, 3.0)] * fw.shape[0])))


# =========================================================================== #
# 2. THE GENERALISATION IS THE SAME PHYSICAL MODEL                            #
# =========================================================================== #
class TestPinholeReduction:

    @pytest.mark.parametrize("dlat,dyaw", [(0.0, 8.0), (2.0, 0.0), (1.0, -3.0),
                                           (-3.0, 12.0)])
    def test_pinhole_field_reduces_to_the_shipped_homography(self, dlat, dyaw):
        """Square pinhole frame ⇒ the field IS ``sampling_homography``'s map.

        Sherman-Morrison: ``(I - C n^T/d)^-1 == I + C n^T/(d - n^T C)``. If this
        ever fails, the new path has silently changed the PHYSICS, not the
        projection."""
        fr = {"height": 128, "width": 128, "f_ref": 200.0,
              "projection": "pinhole"}
        su, sv, _ = CH.sampling_source_grid(dlat, dyaw, fr)
        H = CH.sampling_homography(dlat, dyaw, f=200.0, c=(128 - 1) / 2.0)
        xs, ys = _dest_grid(128, 128)
        hu, hv = _apply_H(H, xs, ys)
        assert float((su - hu).abs().max()) < 1e-9
        assert float((sv - hv).abs().max()) < 1e-9

    def test_a_pinhole_frame_uses_independent_H_and_W(self):
        """``c`` was ONE scalar for cx and cy — structurally wrong off-square."""
        fr = {"height": 176, "width": 624, "f_ref": 305.5774907364391,
              "projection": "pinhole"}
        su, sv, _ = CH.sampling_source_grid(0.0, 0.0, fr)
        xs, ys = _dest_grid(176, 624)
        assert float((su - xs).abs().max()) < 1e-9      # identity at zero
        assert float((sv - ys).abs().max()) < 1e-9
        assert CH.as_warp_frame(fr).cx == (624 - 1) / 2.0
        assert CH.as_warp_frame(fr).cy == (176 - 1) / 2.0


# =========================================================================== #
# 3. THE CYLINDER — yaw is exact, lateral is not a homography                 #
# =========================================================================== #
class TestCylinder:

    @pytest.mark.parametrize("dyaw", [2.0, 8.0, -8.0, 12.0])
    def test_cylindrical_yaw_is_the_exact_pixel_shift(self, dyaw):
        """``u -> u + f_ref*psi``, ``v -> v``. Rows do not move AT ALL."""
        fr = CH.as_warp_frame(V5_TRAIN_FRAME)
        su, sv, valid = CH.sampling_source_grid(0.0, dyaw, V5_TRAIN_FRAME)
        xs, ys = _dest_grid(fr.height, fr.width)
        shift = fr.f_ref * math.radians(dyaw)
        assert float((su - (xs + shift)).abs().max()) < 1e-9
        # rows do not move: exact to float64 round-off (~3e-14 px, i.e. 1e-16 of
        # the frame) — the residual is `sqrt(sin^2+cos^2) != 1.0` in binary, not
        # a model term. Compare with the shipped warp's 47 px of INVENTED dv.
        assert float((sv - ys).abs().max()) < 1e-12
        assert bool(valid.all())            # a pure rotation is TOTAL

    def test_cylindrical_yaw_is_depth_and_pitch_independent(self):
        """A pure camera rotation involves NO scene structure. If the ground
        plane leaked into the yaw axis this changes with ``h_cam``."""
        a = CH.sampling_source_grid(0.0, 8.0, V5_TRAIN_FRAME, h_cam=1.5)
        b = CH.sampling_source_grid(0.0, 8.0, V5_TRAIN_FRAME, h_cam=17.0)
        c = CH.sampling_source_grid(0.0, 8.0, V5_TRAIN_FRAME, pitch_deg=6.0)
        assert float((a[0] - b[0]).abs().max()) == 0.0
        assert float((a[1] - b[1]).abs().max()) == 0.0
        assert float((a[0] - c[0]).abs().max()) == 0.0

    def test_cylindrical_lateral_is_not_expressible_as_a_homography(self):
        """⭐ The representation claim, MEASURED not argued.

        Fit the BEST 3x3 (DLT, exact least squares over the whole 176x624
        field) to the true cylindrical lateral map. A large residual means no
        homography — the shipped one or any other — can carry this axis."""
        fr = CH.as_warp_frame(V5_TRAIN_FRAME)
        su, sv, valid = CH.sampling_source_grid(2.0, 0.0, V5_TRAIN_FRAME)
        xs, ys = _dest_grid(fr.height, fr.width)
        m = valid & torch.isfinite(su) & torch.isfinite(sv)
        res = _best_homography_residual_px(xs[m], ys[m], su[m], sv[m])
        assert res["max_px"] > 8.0, res
        # ... and the yaw axis, by contrast, IS a homography (a translation):
        su2, sv2, v2 = CH.sampling_source_grid(0.0, 8.0, V5_TRAIN_FRAME)
        res2 = _best_homography_residual_px(xs[v2], ys[v2], su2[v2], sv2[v2])
        assert res2["max_px"] < 1e-6, res2

    def test_no_ground_preimage_above_the_horizon_is_reported(self):
        """At ``dlat != 0`` the rows above the horizon have no preimage. The
        shipped warp produced a finite meaningless coordinate there in silence;
        the fraction is now a number."""
        _, _, valid_lat = CH.sampling_source_grid(2.0, 0.0, V5_TRAIN_FRAME)
        _, _, valid_yaw = CH.sampling_source_grid(0.0, 8.0, V5_TRAIN_FRAME)
        assert 0.4 < float((~valid_lat).double().mean()) < 0.6   # ~half a frame
        assert float((~valid_yaw).double().mean()) == 0.0


def _best_homography_residual_px(u, v, su, sv):
    """Residual of the LEAST-SQUARES 3x3 that best explains ``(u,v)->(su,sv)``.

    Standard DLT: each correspondence gives two rows of ``A h = 0``; the
    solution is the smallest right singular vector. Reported in PIXELS after
    re-projecting, so the number is directly comparable to the audit's px
    errors."""
    u = u.reshape(-1).double().numpy()
    v = v.reshape(-1).double().numpy()
    su = su.reshape(-1).double().numpy()
    sv = sv.reshape(-1).double().numpy()
    # subsample for conditioning + speed; the map is smooth so this is exact
    ix = np.linspace(0, len(u) - 1, min(len(u), 4000)).astype(int)
    u, v, su, sv = u[ix], v[ix], su[ix], sv[ix]
    n = len(u)
    o, z = np.ones(n), np.zeros(n)
    A = np.empty((2 * n, 9))
    A[0::2] = np.stack([u, v, o, z, z, z, -su * u, -su * v, -su], axis=1)
    A[1::2] = np.stack([z, z, z, u, v, o, -sv * u, -sv * v, -sv], axis=1)
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    H = Vt[-1].reshape(3, 3)
    d = H[2, 0] * u + H[2, 1] * v + H[2, 2]
    pu = (H[0, 0] * u + H[0, 1] * v + H[0, 2]) / d
    pv = (H[1, 0] * u + H[1, 1] * v + H[1, 2]) / d
    e = np.hypot(pu - su, pv - sv)
    return {"max_px": float(e.max()), "mean_px": float(e.mean()),
            "p95_px": float(np.percentile(e, 95)), "n": int(n)}


# =========================================================================== #
# 4. THE DEFECT AND ITS CONTROL, reproduced by the SHIPPED code               #
# =========================================================================== #
class TestDefectAndControl:

    def test_the_shipped_warp_on_the_v5_frame_reproduces_the_measured_defect(self):
        """Regression anchor: if someone 'fixes' this by swapping a constant,
        these numbers move and the test says so."""
        fr = CH.as_warp_frame(V5_TRAIN_FRAME)
        xs, ys = _dest_grid(fr.height, fr.width)
        hu, hv = _apply_H(CH.sampling_homography(0.0, PROBE_DYAW_DEG), xs, ys)
        tu, tv, _ = CH.sampling_source_grid(0.0, PROBE_DYAW_DEG, V5_TRAIN_FRAME)
        e = torch.hypot(hu - tu, hv - tv)
        assert abs(float(e.mean()) - DEFECT_MEAN_PX) < 0.05
        assert abs(float(e.max()) - DEFECT_MAX_PX) < 0.05
        assert abs(float((e > 1.0).double().mean()) - DEFECT_FRAC_GT1) < 1e-3
        # the error EXCEEDS the whole correct displacement
        assert float(e.mean()) > fr.f_ref * math.radians(PROBE_DYAW_DEG)
        # and it invents VERTICAL motion where the truth is exactly zero
        assert float((hv - tv).abs().max()) > 40.0

    def test_the_control_on_the_deployed_frame_is_unregressed(self):
        """⭐ THE CONTROL. On the frame the warp was BUILT for, the shipped
        homography and the new field agree to ``max 0.118 px`` — the half-pixel
        ``c=128`` vs ``(W-1)/2 = 127.5`` convention, and nothing else. A guard
        that fired everywhere would prove nothing; this one discriminates."""
        dep = {"height": 256, "width": 256, "f_ref": 266.0,
               "projection": "pinhole"}
        xs, ys = _dest_grid(256, 256)
        hu, hv = _apply_H(CH.sampling_homography(0.0, PROBE_DYAW_DEG), xs, ys)
        tu, tv, _ = CH.sampling_source_grid(0.0, PROBE_DYAW_DEG, dep)
        e = torch.hypot(hu - tu, hv - tv)
        assert float(e.max()) <= CONTROL_MAX_PX
        assert float((e > 1.0).double().mean()) == 0.0


# =========================================================================== #
# 5. THE GUARD CAN FAIL, AND HERE IS WHAT MAKES IT                            #
# =========================================================================== #
class TestGuard:

    def test_no_frame_on_a_non_deployed_raster_is_refused(self):
        """The v5 defect verbatim: the 266/128 pinhole warp on a 176x624
        raster. Before this guard it produced numbers."""
        fw = _window(h=176, w=624)
        with pytest.raises(CH.WarpFrameRefused, match="176x624"):
            CH.warp_frames(fw, 0.0, 8.0, None)

    def test_a_declared_frame_that_is_not_the_raster_is_refused(self):
        fw = _window(h=176, w=624)
        with pytest.raises(CH.WarpFrameRefused, match="declared frame"):
            CH.warp_frames(fw, 0.0, 8.0, V5_CACHE_FRAME)   # 256x640 != 176x624

    def test_an_unknown_projection_is_refused_not_approximated(self):
        with pytest.raises(CH.WarpFrameRefused, match="projection"):
            CH.as_warp_frame({"height": 176, "width": 624, "f_ref": 305.0,
                              "projection": "equirectangular"})

    def test_an_object_with_no_geometry_is_refused(self):
        with pytest.raises(CH.WarpFrameRefused, match="cannot read a geometry"):
            CH.as_warp_frame(object())

    def test_the_guard_passes_on_the_matched_v5_frame(self):
        """It must not fire everywhere: the correct pairing is accepted."""
        fw = _window(h=176, w=624)
        prov = CH.assert_warp_frame(V5_TRAIN_FRAME, fw)
        assert prov["warp_model"] == "cylindrical_ground_plane_field"
        assert prov["is_deployed_frame"] is False
        out = CH.warp_frames(fw, 0.0, 8.0, V5_TRAIN_FRAME)
        assert out.shape == fw.shape


# =========================================================================== #
# 6. THE ROLLOUTS CARRY IT END TO END                                         #
# =========================================================================== #
def _episode(T=60, seed=0, H=176, Wd=624):
    from types import SimpleNamespace
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(T, dtype=torch.float32)
    yaw = 0.02 * torch.sin(t * 0.05) + 0.001 * seed
    v = 8.0 + 0.5 * torch.sin(t * 0.03)
    x = torch.cumsum(v * torch.cos(yaw) * CH.DT, 0)
    y = torch.cumsum(v * torch.sin(yaw) * CH.DT, 0)
    return SimpleNamespace(poses=torch.stack([x, y, yaw, v], dim=-1),
                           frames=torch.rand(T, 3, H, Wd, generator=g),
                           episode_id=f"ep{seed}")


class _StubPlanner:
    ego_input_declared = True                       # taniteval.ego_guard

    def traj(self, fw, v0, goal_batch):
        b = fw.shape[0]
        sig = fw.reshape(b, -1).mean(dim=1).float().cpu()
        steps = torch.arange(1, 21, dtype=torch.float32)[None]
        x = v0.float().cpu()[:, None] * steps * CH.DT
        y = 0.4 * torch.sin(sig[:, None] * 12.0) * steps * CH.DT
        return torch.stack([x, y], dim=-1)


class TestRolloutsCarryTheFrame:

    def test_corridor_rollout_refuses_a_cylindrical_raster_with_no_frame(self):
        with pytest.raises(CH.WarpFrameRefused):
            CH.corridor_rollout(_StubPlanner(), [_episode(T=60)], None, "cpu",
                                30, stride=16, batch=2)

    def test_corridor_rollout_runs_on_the_v5_frame_and_records_provenance(self):
        pw = CH.corridor_rollout(_StubPlanner(), [_episode(T=60, seed=1)], None,
                                 "cpu", 30, stride=16, batch=2,
                                 frame=V5_TRAIN_FRAME)
        assert pw is not None and pw["lat"].shape[1] == 30
        w = pw["_warp"]
        assert w["frame"]["projection"] == "cylindrical"
        assert w["warp_model"] == "cylindrical_ground_plane_field"
        assert w["is_deployed_frame"] is False

    def test_the_frame_changes_the_pixels_and_therefore_the_numbers(self):
        """If the frame were plumbed but not APPLIED this passes silently —
        so assert the rollout actually differs."""
        eps = [_episode(T=60, seed=2)]
        a = CH.corridor_rollout(_StubPlanner(), eps, None, "cpu", 30, stride=16,
                                batch=2, frame=V5_TRAIN_FRAME)
        b = CH.corridor_rollout(_StubPlanner(), eps, None, "cpu", 30, stride=16,
                                batch=2, frame=CH.LEGACY_WARP)
        assert not torch.allclose(a["lat"], b["lat"])

    def test_pseudo_evaluate_refuses_and_then_runs_on_the_v5_frame(self):
        eps = [_episode(T=60, seed=3)]
        grid = PS.GridSpec(dyaw_deg=(-8.0, 0.0, 8.0), dlon_steps=(0,))
        with pytest.raises(CH.WarpFrameRefused):
            PS.pseudo_evaluate(_StubPlanner(), eps, grid, stride=16, batch=2)
        pw = PS.pseudo_evaluate(_StubPlanner(), eps, grid, stride=16, batch=2,
                                frame=V5_TRAIN_FRAME)
        assert pw["warp"]["frame"]["projection"] == "cylindrical"
        assert pw["warp"]["max_no_ground_preimage_frac"] == 0.0   # yaw only
