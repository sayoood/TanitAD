"""In-envelope geometric recovery augmentation — the shared geometry.

The single idea of stream D2: teach REF-C's anchored-diffusion planner to RECOVER
from an off-path pose it can only sense through pixels, by SYNTHESISING the
covariate-shifted state analytically from a real frame — a ground-plane homography
warp bounded to the MEASURED low-OOD envelope (P1: lowood_flagship_ci.json) — and
supervising the return-to-path trajectory. No renderer, no world-model rollout, no
scarce junction scenes: every window becomes a recovery example.

WHY THIS IS TRAIN==TEST BY CONSTRUCTION. The closed-loop lane-keeping instrument
(`lowood_lanekeep.py`, abe82f1f) drives the ego off the recorded corridor and, at
each on-policy tick, shows the planner the SAME real window warped by the on-policy
(dlat, dpsi) through `sampling_homography`. This module warps with a byte-copy of
that exact operator, so the perturbations the planner learns to recover from are
drawn from the identical distribution the instrument scores it on. `_assert_warp_
matches_harness()` verifies the byte-copy against the live harness at run time.

RECOVERY TARGET (the ChauffeurNet "synthesise the worst" trajectory, built from the
program's own trained target function). At tick t the ego truly sits ON its recorded
path (dlat=dpsi=0). We place it synthetically at (+dlat left, +dpsi heading) and ask:
from THERE, what is the trajectory back onto the recorded path? That is exactly the
recorded future re-expressed in the PERTURBED ego frame:
    recovery = refb_labels.waypoint_targets(perturbed_pose_last, future_poses, H)
`waypoint_targets` is the same function REF-C trained its traj/anchor heads on, so the
recovery target is sign- and convention-identical to the base objective — only the
ego frame moved. At (dlat, dpsi)=(0,0) it is byte-identical to the base target
(`validate_identity`, the DAgger-probe discipline: max|err|==0).

Evidence class: MEASURED (this file's `validate_identity` / `_assert_warp_matches_
harness` CPU checks) for the geometry; the envelope bound is MEASURED (P1
`lower-ood-closedloop-source/P1_DECISION_GRADE_FINDINGS.md` §1.2/1.3).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

# f-theta canonical pinhole of the 256x256 phase-0 cache (lowood_lanekeep.py verbatim)
F_EFF = 266.0
CXY = 128.0
_H_CAM = 1.5          # camera height (m) used by the instrument's warp
_PITCH = 0.0

# ---------------------------------------------------------------------------- #
# Warp operator — BYTE-COPY of lowood_lanekeep.sampling_homography / warp_batch  #
# (kept inline so train==test does not depend on the harness file's location;   #
#  _assert_warp_matches_harness verifies the copy is exact at run time).         #
# ---------------------------------------------------------------------------- #
def sampling_homography(dlat_m: float, dyaw_deg: float, h_cam: float = _H_CAM,
                        pitch_deg: float = _PITCH, f: float = F_EFF,
                        c: float = CXY) -> Tensor:
    """cam2(offset)->cam1(real) sampling homography for grid_sample. dlat_m to the
    RIGHT, dyaw_deg to the LEFT; Delta=0 -> identity. (lowood_probe.py verbatim.)"""
    Kk = torch.tensor([[f, 0, c], [0, f, c], [0, 0, 1.0]], dtype=torch.float64)
    Ki = torch.linalg.inv(Kk)
    p = math.radians(pitch_deg)
    n = torch.tensor([0.0, math.cos(p), math.sin(p)], dtype=torch.float64)
    d = float(h_cam)
    psi = math.radians(dyaw_deg)
    Ry = torch.tensor([[math.cos(-psi), 0, math.sin(-psi)],
                       [0, 1.0, 0],
                       [-math.sin(-psi), 0, math.cos(-psi)]], dtype=torch.float64)
    Cc = torch.tensor([dlat_m, 0.0, 0.0], dtype=torch.float64)
    t = -(Ry @ Cc)
    H_1to2 = Kk @ (Ry + torch.outer(t, n) / d) @ Ki
    return torch.linalg.inv(H_1to2)


def warp_batch(fw: Tensor, Hs: Tensor) -> Tensor:
    """fw [b,W,C,Hh,Ww] in [0,1]; Hs [b,3,3] per-window cam2->cam1 sampling
    homography. Per-window H, border-replicate bilinear. (lowood_lanekeep.py.)"""
    b, Wn, C, Hh, Ww = fw.shape
    dev = fw.device
    ys, xs = torch.meshgrid(torch.arange(Hh, dtype=torch.float64, device=dev),
                            torch.arange(Ww, dtype=torch.float64, device=dev),
                            indexing="ij")
    ones = torch.ones_like(xs)
    P = torch.stack([xs, ys, ones], dim=-1).reshape(-1, 3).T
    src = Hs.to(dev).to(torch.float64) @ P
    su = (src[:, 0] / src[:, 2]).reshape(b, Hh, Ww)
    sv = (src[:, 1] / src[:, 2]).reshape(b, Hh, Ww)
    gx = 2.0 * su / (Ww - 1) - 1.0
    gy = 2.0 * sv / (Hh - 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1)
    grid = grid[:, None].expand(-1, Wn, -1, -1, -1).reshape(b * Wn, Hh, Ww, 2).float()
    out = F.grid_sample(fw.reshape(b * Wn, C, Hh, Ww), grid, mode="bilinear",
                        padding_mode="border", align_corners=True)
    return out.reshape(b, Wn, C, Hh, Ww)


# ---------------------------------------------------------------------------- #
# Envelope sampler                                                              #
# ---------------------------------------------------------------------------- #
@dataclass
class EnvelopeCfg:
    """In-envelope perturbation magnitudes. Bounds are MEASURED-justified:

    P1 (`P1_DECISION_GRADE_FINDINGS.md` §1.2/1.3, flagship v1, episode-cluster CI):
      * lateral offset carries NO CI-separated observation-OOD out to 2.0 m
        (first separation at 3.0 m, and even there ADE stays <=0.47, ~17x below
        NuRec's 1.52 gap). -> lat_max <= 1.75 m (also the corridor half-width:
        we teach recovery right up to the departure boundary) stays in the flat
        region.
      * yaw is the EXACT-geometry axis (ground-plane homography is optimistic on
        lateral, exact on yaw) and the more sensitive one: it separates at 3 deg
        (+0.017) and stays <=1.16x baseline to 12 deg. -> yaw_max 5 deg is inside
        the <=1.16x region, far below the 3.2x reconstruction-OOD the instrument
        exists to escape.
    `clean_frac` mixes un-perturbed windows in (BC anchor; the MGAIL/Urban-Driver
    "mix closed-loop with open-loop BC" law) so recovery never overwrites the
    on-path behaviour.
    """
    lat_max: float = 1.75
    yaw_max_deg: float = 5.0
    clean_frac: float = 0.30
    seed: int = 0


def sample_perturbation(b: int, cfg: EnvelopeCfg, gen: torch.Generator | None = None
                        ) -> tuple[Tensor, Tensor]:
    """Sample per-window (dlat_m [left +], dpsi_rad). A `clean_frac` fraction is
    exactly (0,0) so the batch mixes on-path BC with recovery."""
    dlat = (torch.rand(b, generator=gen) * 2 - 1) * cfg.lat_max
    dpsi = (torch.rand(b, generator=gen) * 2 - 1) * math.radians(cfg.yaw_max_deg)
    clean = torch.rand(b, generator=gen) < cfg.clean_frac
    dlat = torch.where(clean, torch.zeros_like(dlat), dlat)
    dpsi = torch.where(clean, torch.zeros_like(dpsi), dpsi)
    return dlat, dpsi


# ---------------------------------------------------------------------------- #
# Perturbed pose + recovery target                                              #
# ---------------------------------------------------------------------------- #
def perturbed_pose_last(pose_last: Tensor, dlat: Tensor, dpsi: Tensor) -> Tensor:
    """pose_last [B,4]=(x,y,yaw,v); move the ego +dlat along its LEFT axis and
    +dpsi in heading. Left unit vector of heading yaw = (-sin yaw, cos yaw).
    Matches the instrument's dlat = -sin*dx + cos*dy (left +) convention."""
    x, y, yaw, v = pose_last[:, 0], pose_last[:, 1], pose_last[:, 2], pose_last[:, 3]
    xp = x + dlat * (-torch.sin(yaw))
    yp = y + dlat * torch.cos(yaw)
    yawp = yaw + dpsi
    return torch.stack([xp, yp, yawp, v], dim=-1)


def recovery_targets(pose_last: Tensor, future_poses: Tensor,
                     horizons: tuple[int, ...], dlat: Tensor, dpsi: Tensor,
                     waypoint_targets_fn) -> Tensor:
    """The return-to-path trajectory from the perturbed pose: the recorded future
    re-expressed in the PERTURBED ego frame. Uses the program's OWN trained target
    function so the recovery target is convention-identical to the base objective.
    At (dlat,dpsi)=(0,0) equals waypoint_targets_fn(pose_last, ...) exactly."""
    pp = perturbed_pose_last(pose_last, dlat, dpsi)
    return waypoint_targets_fn(pp, future_poses, horizons)


def warp_windows(frames: Tensor, dlat: Tensor, dpsi: Tensor,
                 h_cam: float = _H_CAM, pitch_deg: float = _PITCH) -> Tensor:
    """frames [B,W,C,H,W'] -> warped by per-sample (dlat, dpsi). Same operator the
    instrument applies on-policy. dlat in metres (left +, passed straight to the
    harness op), dpsi in RADIANS (converted to deg for the op)."""
    b = frames.shape[0]
    Hs = torch.stack([
        sampling_homography(float(dlat[i]), float(math.degrees(dpsi[i])),
                            h_cam, pitch_deg)
        for i in range(b)])
    return warp_batch(frames, Hs)


# ---------------------------------------------------------------------------- #
# Self-validation (MEASURED discipline)                                         #
# ---------------------------------------------------------------------------- #
def validate_identity(waypoint_targets_fn, horizons=(5, 10, 15, 20), n=8,
                      seed=0) -> dict:
    """Δ=0 identity check (DAgger-probe discipline): with (dlat,dpsi)=(0,0) the
    recovery target must equal the base target byte-for-byte, and the identity
    homography must be ~identity. Returns max abs errors (must be ~0)."""
    g = torch.Generator().manual_seed(seed)
    pose_last = torch.randn(n, 4, generator=g)
    pose_last[:, 3] = pose_last[:, 3].abs() * 10          # v >= 0
    fut = torch.randn(n, max(horizons), 4, generator=g)
    base = waypoint_targets_fn(pose_last, fut, horizons)
    z = torch.zeros(n)
    rec0 = recovery_targets(pose_last, fut, horizons, z, z, waypoint_targets_fn)
    tgt_err = float((rec0 - base).abs().max())
    Hid = sampling_homography(0.0, 0.0)
    H_err = float((Hid - torch.eye(3, dtype=Hid.dtype)).abs().max())
    # a non-trivial perturbation must MOVE the target (else the transform is dead)
    rec1 = recovery_targets(pose_last, fut, horizons,
                            torch.full((n,), 1.0), torch.full((n,), 0.05),
                            waypoint_targets_fn)
    moved = float((rec1 - base).abs().max())
    return {"identity_target_maxerr": tgt_err, "identity_H_maxerr": H_err,
            "perturb_moves_target_maxabs": moved,
            "ok": tgt_err < 1e-5 and H_err < 1e-6 and moved > 1e-3}


def _assert_warp_matches_harness(harness_sampling_homography, harness_warp_batch,
                                 seed=0) -> dict:
    """Verify the inlined warp is a byte-copy of the live abe82f1f harness op, so
    train==test is not merely asserted but MEASURED. Call from the eval/FT entry
    with the harness's own functions imported."""
    g = torch.Generator().manual_seed(seed)
    dH = float((sampling_homography(0.9, 3.0)
                - harness_sampling_homography(0.9, 3.0, _H_CAM, _PITCH)).abs().max())
    fw = torch.rand(2, 4, 3, 32, 32, generator=g)
    Hs = torch.stack([sampling_homography(0.9, 3.0), sampling_homography(-0.5, -2.0)])
    dW = float((warp_batch(fw, Hs) - harness_warp_batch(fw, Hs)).abs().max())
    return {"homography_maxdiff": dH, "warp_maxdiff": dW,
            "ok": dH < 1e-9 and dW < 1e-9}


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "stack" / "scripts"))
    from refb_labels import waypoint_targets  # noqa: E402
    r = validate_identity(waypoint_targets)
    print("validate_identity:", r)
    assert r["ok"], "geometry identity check FAILED"
    print("PERTURB_SELFCHECK_OK")
