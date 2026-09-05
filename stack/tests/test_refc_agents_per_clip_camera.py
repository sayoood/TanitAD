"""PER-CLIP RIG CAMERAS, and the ground prior that trains nothing.

Two defects are pinned here, and each gate is shown to FAIL its defect — a
control that has never read the wrong value certifies nothing.

**D1 — ONE camera for a corpus whose mount pose is PER CLIP.**
``agent_losses(cam=...)`` took a single :class:`RigCamera`, so a run declared
one mount pose for every clip. That is retraction class **C28** ("a constant
where the quantity is per-clip", ``RETRACTION_LOG.md``): the front-wide camera
height is **1.245–1.607 m** across 40 clips (37 distinct values, CV 7.4 %,
INHERITED from ``taniteval/tools/pai_extrinsics_table.py``), and MEASURED here
on the repo's own banked 3-clip table it is **1.2922 / 1.2968 / 1.5758 m** —
which straddles the shipped guard band ``(1.43, 1.56)`` at *both* ends.
⛔ It matters because :func:`ground_range_prior` back-projects through the road
plane, so its supervised range is directly proportional to the mount height.

**D2 — ``ground_range_prior`` is a TAUTOLOGY.** MEASURED 2026-09-05: it reads
``2.61e-08`` with a parameter gradient of ``8.73e-11``. It projects a foot at
rig ``z = 0`` and back-projects that pixel onto ``z = ROAD_PLANE_Z_M`` which
**is 0.0**, so ``r_back == r_pred`` by construction. An arm passing
``--agent-w-ground 0.5`` stamps the weight and adds exactly zero.
⭐ This is the M18 dead-flag defect ONE LEVEL IN: M18 fixed *"the camera is
None so the term never runs"*; the term now runs and is identically zero, and
only a gradient probe can see the difference.
"""
from __future__ import annotations

import math

import pytest
import torch

from tanitad.data import calib
from tanitad.data.rig_projection import (
    CAM_HEIGHT_RANGE_M, CAM_HEIGHT_SAMPLES, ROAD_PLANE_Z_M, RigCamera,
)
from tanitad.refs import refc_agents as ra

FRAME = calib.PHYSICALAI_WIDE120_256x640
B, N = 4, 16


def _cam(h):
    return RigCamera.nominal(FRAME, height_m=float(h))


def _boxes(seed=0, b=B, n=N, requires_grad=False):
    g = torch.Generator().manual_seed(seed)
    cx = torch.rand(b, n, generator=g) * 50.0 + 5.0
    cy = (torch.rand(b, n, generator=g) * 2.0 - 1.0) * 14.0
    box = torch.stack([cx, cy, torch.full_like(cx, 4.5),
                       torch.full_like(cx, 1.9)], dim=-1)
    return box.requires_grad_(requires_grad)


def _match(b=B, n=N):
    idx = torch.arange(n)
    return {"rows": [idx] * b, "cols": [idx] * b,
            "n_dropped": [0] * b, "n_target": [n] * b}


# =========================================================================
# D1 — the signature accepts a per-ROW camera, and it is REALLY per row
# =========================================================================

def test_row_cameras_normalises_the_three_admissible_shapes():
    c = _cam(1.5)
    assert ra.row_cameras(None, 4) == [None] * 4
    assert ra.row_cameras(c, 4) == [c] * 4
    seq = [c, None, c, None]
    assert ra.row_cameras(seq, 4) == seq


def test_a_partial_camera_list_is_REFUSED_not_padded():
    """⛔ THE DELIBERATE REGRESSION. Padding a short list would attach clip
    k's mount pose to clip k+1's row — a label corruption no downstream metric
    could attribute, because every count would still look healthy."""
    with pytest.raises(ValueError, match="entries for a batch"):
        ra.row_cameras([_cam(1.5)] * 3, 4)
    with pytest.raises(ValueError):
        ra.row_cameras([_cam(1.5)] * 5, 4)


def test_a_bank_reaching_the_loss_unresolved_is_REFUSED():
    """The loss has no episode ids. Inventing a lookup there is how a camera
    silently attaches to the wrong clip, so the bank must be resolved by the
    caller that HAS the ids."""
    bank = ra.RigCameraBank(by_episode={7: _cam(1.5)})
    with pytest.raises(TypeError, match="resolved BEFORE the loss"):
        ra.row_cameras(bank, 4)


def test_a_list_of_IDENTICAL_cameras_reproduces_the_single_camera():
    """⭐ THE KNOWN-VALUE CONTROL. If the per-row path did not reproduce the
    single-camera path on identical cameras, every banked comparison would
    shift for a reason unrelated to the mount pose."""
    c = _cam(1.5)
    box, tgt = _boxes(0), _boxes(1)
    m = _match()
    one = ra.monocular_projection_loss(box, tgt, c, m)
    many = ra.monocular_projection_loss(box, tgt, [c] * B, m)
    assert one["n"] == many["n"] > 0
    assert abs(float(one["loss"]) - float(many["loss"])) < 1e-12

    g_one = ra.ground_range_prior(box, c)
    g_many = ra.ground_range_prior(box, [c] * B)
    assert g_one["n"] == g_many["n"] > 0
    assert abs(float(g_one["loss"]) - float(g_many["loss"])) < 1e-12


def test_per_clip_cameras_REALLY_change_the_loss():
    """⛔ The positive assertion that the threading is not decoration. Two
    mount heights must give two different losses, and a MIXED batch must land
    strictly between the two uniform ones — otherwise `cams[b]` is being
    ignored and the test above would pass on a no-op."""
    lo, hi = _cam(1.245), _cam(1.607)          # the observed corpus extremes
    box, tgt = _boxes(0), _boxes(1)
    m = _match()
    l_lo = float(ra.monocular_projection_loss(box, tgt, [lo] * B, m)["loss"])
    l_hi = float(ra.monocular_projection_loss(box, tgt, [hi] * B, m)["loss"])
    assert abs(l_lo - l_hi) > 1e-6, "the mount height did not reach the loss"
    mixed = float(ra.monocular_projection_loss(
        box, tgt, [lo, lo, hi, hi], m)["loss"])
    assert min(l_lo, l_hi) < mixed < max(l_lo, l_hi)


def test_rows_without_a_camera_are_COUNTED_never_silently_dropped():
    """A term computed over 2 of 4 rows is not the term config.json says it
    trained. The count is the only thing that can say so."""
    c = _cam(1.5)
    box, tgt = _boxes(0), _boxes(1)
    out = ra.monocular_projection_loss(box, tgt, [c, None, c, None], _match())
    assert out["n_rows"] == B
    assert out["n_rows_no_cam"] == 2
    g = ra.ground_range_prior(box, [c, None, c, None])
    assert g["n_rows_no_cam"] == 2


def test_agent_losses_states_its_camera_SCOPE_and_its_row_coverage():
    cfg = ra.AgentSeamConfig(enable=True, queries=N, d_model=64, depth=1,
                             n_heads=4, enforce_band=False, w_project=1.0)
    c = _cam(1.5)
    slots = {"box": _boxes(0), "yaw_vec": torch.zeros(B, N, 2),
             "cls_logits": torch.zeros(B, N, ra.N_AGENT_CLASSES),
             "presence_logit": torch.zeros(B, N),
             "occ_logit": torch.zeros(B, N),
             "rates": torch.zeros(B, N, 3)}
    slots["yaw_vec"][..., 1] = 1.0
    tgt = {"box": _boxes(1), "yaw": torch.zeros(B, N),
           "cls": torch.zeros(B, N, dtype=torch.long),
           "valid": torch.ones(B, N, dtype=torch.bool),
           "occ": torch.zeros(B, N),
           "rates": torch.zeros(B, N, 3),
           "rates_mask": torch.zeros(B, N, dtype=torch.bool)}
    for cam, want, no_cam in ((None, "none", B), (c, "single", 0),
                              ([c, None, c, None], "per-row", 2)):
        out = ra.agent_losses(slots, tgt, cfg, cam=cam)
        assert out["cam_scope"] == want
        assert out["n"]["rows_no_cam"] == no_cam
        assert out["n"]["rows_with_cam"] == B - no_cam
        # the term exists only where a camera does
        assert ("loss_project" in out) == (no_cam < B)


def test_the_bank_resolves_by_episode_and_REPORTS_its_gap():
    a, b = _cam(1.3), _cam(1.5)
    bank = ra.RigCameraBank(by_episode={11: a, 22: b})
    got = bank.for_episodes(torch.tensor([11, 22, 33, 11]))
    assert got == [a, b, None, a]
    cov = bank.coverage([11, 22, 33, 44])
    assert cov["n"] == 4 and cov["n_covered"] == 2 and cov["n_missing"] == 2
    assert cov["frac"] == 0.5 and sorted(cov["missing_sample"]) == [33, 44]
    assert cov["has_default"] is False


# =========================================================================
# D2 — the ground prior is a tautology, and it is PINNED as one
# =========================================================================

def test_the_ground_prior_is_a_TAUTOLOGY_with_a_live_positive_control():
    """⛔⛔ THE FINDING, pinned so it cannot be quietly re-shipped as a live
    term. ``ROAD_PLANE_Z_M == 0.0`` and the foot is projected at z = 0, so
    ``project`` and ``ground_intersection`` are exact inverses.

    ⚠️ The same-breath POSITIVE CONTROL is what makes the zero admissible: a
    flat gradient is otherwise indistinguishable from a probe that never ran.
    """
    assert ROAD_PLANE_Z_M == 0.0
    cam = _cam(1.5)
    box = _boxes(0, requires_grad=True)
    gp = ra.ground_range_prior(box, cam)
    assert gp["n"] > 0, "the probe read nothing -- INCONCLUSIVE, not a pass"
    g_dead = float(torch.autograd.grad(gp["loss"], box)[0].abs().max())

    box2 = _boxes(0, requires_grad=True)
    pj = ra.monocular_projection_loss(box2, _boxes(1), cam, _match())
    assert pj["n"] > 0
    g_live = float(torch.autograd.grad(pj["loss"], box2)[0].abs().max())

    assert g_live > 1e-6, ("the CONTROL is flat too -- the probe is broken, "
                           "so the dead reading proves nothing")
    assert float(gp["loss"]) < 1e-6
    assert g_dead < 1e-7, (
        "ground_range_prior now has a real gradient (%.3e). If that is "
        "deliberate, REMOVE the refusal in refc_v3_train."
        "assert_ground_prior_is_supervised in the same change -- do not "
        "leave a live term refused." % g_dead)


def test_the_tautology_survives_a_wrong_mount_height_which_is_the_point():
    """⭐ Why the tautology is not merely harmless. Under the flat-road
    nominal geometry the back-projected range scales EXACTLY with the assumed
    height, ``r' = r * h'/h`` — so a wrong camera would bias the taught range
    by a computable factor. The term cannot see it, because it uses the SAME
    camera on both sides. That is what makes it a tautology rather than a
    weak signal."""
    for h in (1.245, 1.306, 1.5, 1.607):
        gp = ra.ground_range_prior(_boxes(0), _cam(h))
        assert float(gp["loss"]) < 1e-6, h
    # and the bias it is blind to, computed from the geometry it uses:
    assert abs((1.5 / 1.245 - 1.0) - 0.2048) < 1e-3


# =========================================================================
# The constant that circulates, and the sample each band came from
# =========================================================================

def test_the_camera_height_band_carries_every_sample_it_came_from():
    """⛔ C28. The band was ``(1.43, 1.56)`` from **12 clips, one chunk**, and
    the repo's own banked 3-clip table straddles it at both ends. Each sample
    is named so the next widening is a table row, not a silent edit."""
    assert CAM_HEIGHT_SAMPLES["chunk12_cuboids"] == (1.43, 1.56)
    lo3, hi3 = CAM_HEIGHT_SAMPLES["banked_render_table_3clip"]
    assert lo3 < 1.43 and hi3 > 1.56, "the 3-clip table must straddle the old band"
    par = CAM_HEIGHT_SAMPLES["train2400_parity"]
    assert par is not None, (
        "the parity corpus band must be MEASURED -- it is the corpus a real "
        "arm trains on, and every smaller sample under-stated the spread")
    # ⭐ the prediction the earlier bands invited: each widening strictly
    # CONTAINED the previous one, and the parity corpus widened it AGAIN at
    # both ends. MEASURED over all 2,400 clips, 554 distinct heights.
    lo40, hi40 = CAM_HEIGHT_SAMPLES["sensor_extrinsics_40clip"]
    assert par[0] < lo40 and par[1] > hi40
    lo = min(v[0] for v in CAM_HEIGHT_SAMPLES.values() if v)
    hi = max(v[1] for v in CAM_HEIGHT_SAMPLES.values() if v)
    assert CAM_HEIGHT_RANGE_M == (lo, hi), "the guard is the UNION of samples"
    # ⚠️ CORRECTED BY OUR OWN MEASUREMENT, same day. The 40-clip band
    # supported "1.22 m is BELOW every observed minimum"; the parity corpus
    # reads a minimum of 1.2131 m, so 1.22 IS inside the observed range. It
    # remains wrong AS A CONSTANT -- 554 distinct heights over 2,400 clips --
    # but the stronger claim was a small-sample artifact and is retracted.
    assert lo < 1.22 < hi
    assert lo == 1.2131


def test_the_fov_half_angle_is_the_rig_not_the_pinhole_formula():
    """The pinhole formula reads 92.6 deg for this 120 deg camera and looks
    entirely plausible. Kept here because the visibility filter — which is ON
    by default — is an azimuth cut at exactly this angle."""
    assert abs(ra.FOV_HALF_ANGLE_RAD - math.radians(60.0)) < 1e-12
    pinhole = 2.0 * math.atan((FRAME.width / 2.0) / FRAME.f_ref)
    assert abs(math.degrees(pinhole) - 92.6) < 0.5
    assert abs(math.degrees(2 * ra.FOV_HALF_ANGLE_RAD) - 120.0) < 1e-9
