"""Focal-invariance validation for D-016 camera canonicalization (H7).

`tanitad.data.calib.focal_crop_resize` crops each camera so the EFFECTIVE focal
at the model input is a shared constant ``F_REF`` (266 px @ 256). That geometry
is unit-tested for *arithmetic* (`stack/tests/test_calib.py`) but has never been
validated end-to-end: does using the correct per-camera focal actually make the
*trained encoder's* representation invariant to focal length on real frames? If
it does not, the D-010 real+sim mix is feeding the world model inconsistent
action->pixel-motion geometry across corpora (exactly what calib.py warns about),
and the I7 fingerprint's ``f_eff_px = 266`` claim is only nominal.

This module supplies (a) the controlled *measurement* and (b) a fail-loud
data-card self-check the loaders can call at ingest.

Controlled design (single scene -> no cross-corpus content confound)
--------------------------------------------------------------------
A real frame captured at focal ``f0`` is turned into a virtual capture of the
SAME scene at a LONGER focal ``f1 = z * f0`` (a narrower-FOV camera) by
central-crop-by-1/z then resize (``virtual_focal_zoom`` — geometrically exact
for a pinhole). Two intrinsics policies then map both captures to the model
input via ``focal_crop_resize``:

    correct : canon(raw_f0, f0)   vs   canon(raw_f1, z*f0)
    wrong   : canon(raw_f0, f0)   vs   canon(raw_f1, f0)     # assumes both are f0

The ONLY difference between the two policies is whether the perturbed camera's
TRUE focal is used. D-016 delivers focal-invariance iff, in the encoder's latent
space, ``drift_correct << drift_wrong``. Falsifier: ``drift_correct >=
drift_wrong`` (canonicalization does not remove the focal nuisance on real data).

Note the base corpus must have FOV headroom (``focal_crop_resize`` only crops
DOWN to the reference): comma2k19 (~50 deg) sits right at F_REF with none, so the
controlled perturbation runs on a wide 120-deg corpus (Cosmos / PhysicalAI). That
headroom asymmetry is itself the reason `assert_effective_focal` exists.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from tanitad.data.calib import F_REF, focal_crop_size


def virtual_focal_zoom(vid: Tensor, z: float) -> Tensor:
    """[N, C, H, W] -> same shape: the SAME scene at focal ``f1 = z * f0``.

    A longer-focal (narrower-FOV) pinhole camera sees the central ``1/z`` of the
    wider camera's field, magnified to fill the sensor. Exact for a pinhole:
    central-crop by ``1/z`` then bilinear resize back to ``(H, W)``. Content
    outside the central ``1/z`` is not observable by the narrower camera and is
    correctly discarded (not invented). Returns float32.
    """
    assert vid.ndim == 4, vid.shape
    if z < 1.0:
        raise ValueError(f"z must be >= 1 (a narrower/longer-focal camera); got {z}")
    n, c, h, w = vid.shape
    ch, cw = max(2, int(round(h / z))), max(2, int(round(w / z)))
    top, left = (h - ch) // 2, (w - cw) // 2
    crop = vid[..., top:top + ch, left:left + cw].float()
    return F.interpolate(crop, size=(h, w), mode="bilinear", align_corners=False)


def achieved_focal(f_px: float, h: int, w: int, size: int,
                   f_ref: float = F_REF) -> float:
    """Effective focal [px] actually achieved by `focal_crop_resize` (post-clamp).

    ``focal_crop_resize`` crops a centered square of side ``c =
    focal_crop_size(...)`` (clamped to the frame) then resizes to ``size``, so
    ``f_eff = f_px * size / c``. When the camera is narrower than the reference
    the clamp bites and ``f_eff > f_ref``.
    """
    c = focal_crop_size(f_px, h, w, size, f_ref)
    return f_px * size / c


def assert_effective_focal(f_px: float, h: int, w: int, size: int,
                           f_ref: float = F_REF, tol: float = 0.03) -> float:
    """Fail loud if a camera's (focal, resolution) cannot reach ``f_ref``.

    ``focal_crop_resize`` silently CLAMPS the crop to the frame, so a camera
    NARROWER than the reference (no FOV headroom) lands ABOVE ``f_ref`` — an
    ``f_eff`` inconsistent with the ``f_eff_px = 266`` that every corpus's
    ``CORPUS_META`` / I7 fingerprint asserts. A loader or data card calls this at
    ingest so the mismatch surfaces as a ``ValueError`` at ingest, not as
    corrupted cross-corpus geometry deep in training. Returns the achieved
    ``f_eff``.
    """
    f_eff = achieved_focal(f_px, h, w, size, f_ref)
    if abs(f_eff / f_ref - 1.0) > tol:
        raise ValueError(
            f"focal canonicalization cannot reach F_REF={f_ref:.0f}px: camera "
            f"f={f_px:.0f}px @ {h}x{w} -> f_eff={f_eff:.1f}px "
            f"({100 * (f_eff / f_ref - 1):+.1f}%). The camera is narrower than "
            f"the reference (no FOV headroom to crop down); its pixel<->metric "
            f"scale is NOT comparable to F_REF and must not be mixed as if it were."
        )
    return f_eff


def relative_latent_drift(za: Tensor, zb: Tensor) -> Tensor:
    """Per-row ``||za - zb|| / ||za||`` (row = one state vector)."""
    return (za - zb).norm(dim=-1) / za.norm(dim=-1).clamp_min(1e-8)


def cosine_sim(za: Tensor, zb: Tensor) -> Tensor:
    """Per-row cosine similarity."""
    return F.cosine_similarity(za, zb, dim=-1)
