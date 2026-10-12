"""Proposal-level drivable-area FIELD LOSS (SPEC_AUX_DRIVABLE_HEAD.md section 8; PI decision D-AUX-4, 2026-10-12).

    L_field = mean over proposals k, 0.1 s steps t, footprint corners c  of  relu(margin - SDF_gt(corner_{k,t,c}))

WHAT. REFe's winner-takes-all loss trains only the best-matching proposal per sample, so the other 63 proposals are free to
leave the road ("dirty fans" hold 72 % of the zero picks, INHERITED). This loss puts a hinge on EVERY proposal's footprint
corners against the stored ground-truth signed distance field (SDF), so a fan that is entirely on the road by >= margin gets
EXACTLY 0.0 loss and EXACTLY 0.0 gradient, and a corner off the road (or closer than `margin` to the boundary) is pushed toward
the road by the gradient of a bilinear lookup. Published precedent: ReflectDrive-2 Table 6 (`2605.04647`), DAC 93.9 -> 97.0,
PDMS 84.8 -> 87.2 (PUBLISHED, one run, an output-side loss on a waypoint distribution).

TRAINING-ONLY. Nothing here is read at inference: the label is a map-derived training target (PI 2026-10-11 admits map-derived
drivable area as a TRAINING label / reward, never as an inference input). The module imports no map API.

INPUTS, in the model's NATIVE layout (refe/model.py `REFe.forward`, `traj_head(...).view(B, M, horizon_steps, traj_dim)`):
  traj      [B, 64, 20, 3]  float   x [m], y [m], yaw [rad] of the REAR AXLE in the rear-axle frame at t0 (x forward, y left),
                                    step n at t = 0.2 * (n + 1) s (20 steps @ 5 Hz = 4 s). t = 0 is the origin (0, 0, 0).
  sdf_int8  [B, 160, 128]   int8   the stored label, SPEC section 3.1: signed distance to the drivable-area boundary,
                                    POSITIVE INSIDE, unit 0.1 m, clipped +-12.7 m; index [ix, iy]; cell centres at
                                    x = -8 + (ix + 0.5) * 0.5, y = -32 + (iy + 0.5) * 0.5 (the SAME frame as `traj`).
  valid     [B] bool (optional)  False = this sample has no label: it contributes exactly 0 (loss and gradient).

DEFINITIONS (fixed here so a pre-registration can name them):
  * steps: the 40 steps t = 0.1 ... 4.0 s. t = 0 is excluded (the origin footprint is the same for every proposal: a constant
    that carries no gradient and would only inflate the logged value).
  * 5 Hz -> 10 Hz: linear interpolation of x, y and the WRAPPED heading from the origin pose at t = 0 (`interp="5hz"`, the SPEC).
    `interp="navsim2hz"` instead runs NAVSIM's own chain (REFe 20 poses -> 8 poses at 2 Hz by `navsim_dac.to_navsim` -> 41 poses at
    10 Hz by `navsim_dac.at_10hz`), the representation NAVSIM's verdict is judged in. Both are differentiable.
  * corners: the four corners of the nuPlan Pacifica footprint, identical to `navsim_dac.corners_global` (same order).
  * lookup: bilinear `grid_sample` on the SDF in metres (int8 * 0.1). A corner is IN EXTENT iff its fractional cell-centre
    index is inside [0, 159] x [0, 127] (the hull the bilinear lookup is fully supported on; the same test as
    `proto_label.raster_inside`). An out-of-extent corner contributes exactly 0 loss and exactly 0 gradient.
  * reduction: sum of the hinge over valid samples / (n_valid * 64 * 40 * 4). Out-of-extent corners count in the denominator as
    zeros, so the value is a plain mean over every corner of every proposal of every labelled sample.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

# ---- the label grid (SPEC_AUX_DRIVABLE_HEAD.md section 3.1; asserted against proto_label_results.json in the test) ----
X0, X1, Y0, Y1, RES = -8.0, 72.0, -32.0, 32.0, 0.5       # rear-axle frame, x forward, y left
NX, NY = 160, 128
SDF_UNIT_M = 0.1                                          # int8 unit
SDF_CLIP_M = 12.7

# ---- the footprint (nuPlan get_pacifica_parameters(): front 4.049, rear 1.127, width 2.297; asserted equal to navsim_dac's) ----
HALF_LENGTH = (4.049 + 1.127) / 2.0
HALF_WIDTH = 2.297 / 2.0
REAR_AXLE_TO_CENTER = (4.049 - 1.127) / 2.0

# ---- the proposal time base (REFeConfig.horizon_steps = 20 @ 5 Hz) ----
N_SRC, SRC_DT = 20, 0.2
N_STEPS = 40                                              # 0.1 ... 4.0 s
K_PROPOSALS = 64


def _wrap(d: torch.Tensor) -> torch.Tensor:
    """Wrap an angle difference to [-pi, pi); differentiable (slope 1 everywhere but the jump)."""
    return torch.remainder(d + math.pi, 2.0 * math.pi) - math.pi


def poses_10hz(traj: torch.Tensor) -> torch.Tensor:
    """[..., 20, 3] (5 Hz, t = 0.2 .. 4.0) -> [..., 41, 3] (10 Hz, t = 0.0 .. 4.0) from the origin pose at t = 0."""
    zero = torch.zeros_like(traj[..., :1, :])
    full = torch.cat([zero, traj], dim=-2)                                   # [..., 21, 3]  t = 0, 0.2, ..., 4.0
    a, b = full[..., :-1, :], full[..., 1:, :]
    mid_xy = 0.5 * (a[..., :2] + b[..., :2])
    mid_h = a[..., 2:] + 0.5 * _wrap(b[..., 2:] - a[..., 2:])
    mid = torch.cat([mid_xy, mid_h], dim=-1)                                 # [..., 20, 3]  t = 0.1, 0.3, ..., 3.9
    out = torch.stack([a, mid], dim=-2).flatten(-3, -2)                      # [..., 40, 3]  t = 0.0, 0.1, ..., 3.9
    return torch.cat([out, full[..., -1:, :]], dim=-2)                       # [..., 41, 3]  + t = 4.0


def _navsim_tables():
    """Index / weight tables of `navsim_dac.to_navsim` (8 poses at 0.5 ... 4.0 s from the 20 at 0.2 ... 4.0 s), same arithmetic."""
    t_src = SRC_DT * np.arange(1, N_SRC + 1)
    i0, i1, w = [], [], []
    for k in range(8):
        t = 0.5 * (k + 1)
        j = int(np.searchsorted(t_src, t - 1e-9))
        if j < N_SRC and abs(t_src[j] - t) < 1e-9:
            i0.append(j), i1.append(j), w.append(0.0)
        else:
            i0.append(j - 1), i1.append(j), w.append((t - t_src[j - 1]) / (t_src[j] - t_src[j - 1]))
    return i0, i1, w


def poses_10hz_navsim(traj: torch.Tensor) -> torch.Tensor:
    """NAVSIM's representation: 20 @ 5 Hz -> 8 @ 2 Hz (`to_navsim`) -> 41 @ 10 Hz from the origin (`at_10hz`); [..., 41, 3]."""
    i0, i1, w = _navsim_tables()
    p0, p1 = traj[..., i0, :], traj[..., i1, :]
    wt = torch.tensor(w, dtype=traj.dtype, device=traj.device).unsqueeze(-1)
    p8_xy = (1.0 - wt) * p0[..., :2] + wt * p1[..., :2]
    p8_h = p0[..., 2:] + wt * _wrap(p1[..., 2:] - p0[..., 2:])
    p8 = torch.cat([p8_xy, p8_h], dim=-1)                                    # [..., 8, 3]
    knots = torch.cat([torch.zeros_like(p8[..., :1, :]), p8], dim=-2)        # [..., 9, 3]  t = 0, 0.5, ..., 4.0
    q = np.arange(41)
    k = np.minimum(q // 5, 7)
    frac = torch.tensor((q - 5 * k) / 5.0, dtype=traj.dtype, device=traj.device).unsqueeze(-1)
    ka, kb = knots[..., k, :], knots[..., k + 1, :]
    out_xy = (1.0 - frac) * ka[..., :2] + frac * kb[..., :2]
    out_h = ka[..., 2:] + frac * _wrap(kb[..., 2:] - ka[..., 2:])
    return torch.cat([out_xy, out_h], dim=-1)


def footprint_corners(poses: torch.Tensor) -> torch.Tensor:
    """Rear-axle poses [..., T, 3] (x, y, yaw) -> the footprint's 4 corners [..., T, 4, 2], the order and the arithmetic of
    `navsim_dac.corners_global(rel, 0, 0, 0)`: (+L, +W), (+L, -W), (-L, +W), (-L, -W) about the centre, which is
    `REAR_AXLE_TO_CENTER` ahead of the rear axle along the heading."""
    x, y, h = poses[..., 0], poses[..., 1], poses[..., 2]
    ch, sh = torch.cos(h), torch.sin(h)
    cx, cy = x + REAR_AXLE_TO_CENTER * ch, y + REAR_AXLE_TO_CENTER * sh
    lo = torch.tensor([HALF_LENGTH, HALF_LENGTH, -HALF_LENGTH, -HALF_LENGTH], dtype=poses.dtype, device=poses.device)
    la = torch.tensor([HALF_WIDTH, -HALF_WIDTH, HALF_WIDTH, -HALF_WIDTH], dtype=poses.dtype, device=poses.device)
    px = cx.unsqueeze(-1) + lo * ch.unsqueeze(-1) - la * sh.unsqueeze(-1)
    py = cy.unsqueeze(-1) + lo * sh.unsqueeze(-1) + la * ch.unsqueeze(-1)
    return torch.stack([px, py], dim=-1)


def sample_sdf(sdf_int8: torch.Tensor, pts: torch.Tensor):
    """Bilinear SDF lookup. sdf_int8 [B, 160, 128] int8, pts [B, N, 2] (x, y) in the label frame ->
    (s [B, N] metres, in_extent [B, N] bool)."""
    img = (sdf_int8.to(pts.dtype) * SDF_UNIT_M).transpose(1, 2).unsqueeze(1)            # [B, 1, NY, NX]: H = y, W = x
    gx = 2.0 * (pts[..., 0] - X0) / (X1 - X0) - 1.0
    gy = 2.0 * (pts[..., 1] - Y0) / (Y1 - Y0) - 1.0
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(2)                                   # [B, N, 1, 2]: (x_norm, y_norm)
    s = F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=False)
    fx = (pts[..., 0] - X0) / RES - 0.5
    fy = (pts[..., 1] - Y0) / RES - 0.5
    in_ext = (fx >= 0) & (fx <= NX - 1) & (fy >= 0) & (fy <= NY - 1)
    return s.reshape(pts.shape[0], pts.shape[1]), in_ext


def field_loss(traj: torch.Tensor, sdf_int8: torch.Tensor, *, margin: float = 0.25, valid=None,
               interp: str = "5hz", return_parts: bool = False):
    """The proposal-level field loss (module docstring). `traj` [B, 64, 20, 3] in the model's native layout and units,
    `sdf_int8` [B, 160, 128] int8 -> scalar. With `return_parts=True` -> (scalar, dict) where the dict has
    `per_proposal` [B, K] (mean over steps and corners of the hinge, zeros for invalid samples), `min_sdf` [B, K] (metres, the
    minimum over in-extent corners; +12.7 where none), `in_extent_frac`, `violating_frac` (share of in-extent corners with SDF < 0),
    `n_valid`."""
    if traj.dim() != 4 or tuple(traj.shape[-2:]) != (N_SRC, 3):
        raise ValueError(f"field_loss: traj must be [B, K, {N_SRC}, 3] in the model's native layout, got {tuple(traj.shape)}")
    if sdf_int8.dtype != torch.int8:
        raise TypeError(f"field_loss: the label is the stored int8 SDF (unit {SDF_UNIT_M} m), got {sdf_int8.dtype}; "
                        "passing metres here would silently rescale the loss")
    if tuple(sdf_int8.shape) != (traj.shape[0], NX, NY):
        raise ValueError(f"field_loss: sdf_int8 must be [B={traj.shape[0]}, {NX}, {NY}], got {tuple(sdf_int8.shape)}")
    if interp not in ("5hz", "navsim2hz"):
        raise ValueError(f"field_loss: interp must be '5hz' or 'navsim2hz', got {interp!r}")
    if not math.isfinite(margin):
        raise ValueError("field_loss: margin must be finite")
    if not traj.dtype.is_floating_point or traj.dtype in (torch.float16, torch.bfloat16):
        traj = traj.float()                                                              # every loss in FP32 (train.py:1496)
    B, K = traj.shape[0], traj.shape[1]
    poses = poses_10hz(traj) if interp == "5hz" else poses_10hz_navsim(traj)             # [B, K, 41, 3]
    poses = poses[..., 1:, :]                                                            # t = 0.1 ... 4.0
    corners = footprint_corners(poses)                                                   # [B, K, 40, 4, 2]
    s, in_ext = sample_sdf(sdf_int8, corners.reshape(B, -1, 2))
    s = s.reshape(B, K, N_STEPS, 4)
    in_ext = in_ext.reshape(B, K, N_STEPS, 4)
    hinge = F.relu(margin - s)
    hinge = hinge * in_ext.to(hinge.dtype)
    if valid is None:
        vm = torch.ones(B, dtype=hinge.dtype, device=hinge.device)
    else:
        vm = torch.as_tensor(valid, device=hinge.device).reshape(B).to(hinge.dtype)
    per_prop = (hinge * vm[:, None, None, None]).mean(dim=(2, 3))                        # [B, K]
    n_valid = vm.sum().clamp(min=1.0)
    loss = per_prop.sum() / (n_valid * K)
    if not return_parts:
        return loss
    with torch.no_grad():
        ve = in_ext & (vm > 0)[:, None, None, None]
        n_e = ve.sum().clamp(min=1).to(hinge.dtype)
        parts = {
            "per_proposal": per_prop.detach(),
            "min_sdf": torch.where(in_ext, s, torch.full_like(s, SDF_CLIP_M)).amin(dim=(2, 3)).detach(),
            "in_extent_frac": (ve.sum().to(hinge.dtype) / (vm.sum().clamp(min=1.0) * K * N_STEPS * 4)),
            "violating_frac": ((s < 0) & ve).sum().to(hinge.dtype) / n_e,
            "n_valid": vm.sum(),
        }
    return loss, parts
