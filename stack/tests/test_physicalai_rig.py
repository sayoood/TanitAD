"""Two-rig VERTICAL principal-point fix (D-016 R1).

The PhysicalAI front-wide has two rigs distinguished by the vertical principal
point cy: rig A cy~543 (horizon ~image center) and rig B cy~755 (horizon ~215 px
lower in the 1080-row native frame). The old GEOMETRIC-center crop put the
horizon at DIFFERENT output rows for the two rigs (rig B ~66 rows below rig A),
corrupting cross-rig vertical consistency. The fix centers the crop on each
clip's per-clip (cx, cy); these tests pin that the horizon then lands at the SAME
output row for both rigs — analytically and end-to-end through the real
crop+resize (which replicate-pads rig B's below-frame overflow).
"""

import math

import pytest
import torch

from tanitad.data.calib import (F_REF, PHYSICALAI_FRONT_WIDE_FTHETA,
                                FThetaIntrinsics, ftheta_crop_resize,
                                ftheta_horizon_row, ftheta_project_ray)
from tanitad.data.physicalai import FrontWideExtrinsics

RIG_A_CY = 543.0                       # measured rig-A vertical principal point
RIG_B_CY = 755.0                       # measured rig-B (215 px lower)
# a REAL measured front-wide f-theta poly (rig A, chunk 36) — theta[rad]->px@1920
SHARED_POLY = (0.0, 925.281996, -5.741668, -10.263973, -0.798363)
# the REAL front-wide extrinsics of reference rig-A clip bf3cb5ea (chunk 36):
# ~0.59 deg down-pitch (cam->vehicle quaternion), so the horizon sits a few px
# above the optical axis.
EXTR = FrontWideExtrinsics(qx=-0.5006219795862275, qy=0.5045234002825358,
                           qz=-0.49930204087713487, qw=0.49551109381974756,
                           x=1.697, y=-0.010, z=1.426)


def _rig(cy: float, per_clip: bool = True) -> FThetaIntrinsics:
    """Two intrinsics that differ ONLY in cy — isolating the principal-point fix
    from every other geometric factor (same poly, same focal, same extrinsics)."""
    return FThetaIntrinsics(poly=SHARED_POLY, cx=958.0, cy=cy,
                            width=1920, height=1080, per_clip=per_clip)


def test_extrinsics_forward_ray_is_sensible():
    """The vehicle-forward ray in the camera frame is ~boresight (+z) with a
    small upward (negative-y) component (camera is pitched slightly DOWN, so the
    horizon sits just above the optical axis)."""
    dx, dy, dz = EXTR.vehicle_forward_in_cam()
    assert dz > 0.99                                   # ~straight ahead
    assert -0.03 < dy < 0.0                            # horizon just above axis
    assert 0.3 < math.degrees(EXTR.optical_axis_pitch_rad()) < 1.0


def test_horizon_same_output_row_across_rigs_principal():
    """THE FIX: with the per-clip (cx,cy) crop the horizon lands at the same
    output row for rig A and rig B; with the legacy geometric crop it does not."""
    a, b = _rig(RIG_A_CY), _rig(RIG_B_CY)
    d = EXTR.vehicle_forward_in_cam()

    ra = ftheta_horizon_row(a, d, center="principal")
    rb = ftheta_horizon_row(b, d, center="principal")
    assert abs(ra - rb) < 3.0                          # rigs AGREE (few px)
    assert abs(ra - 128.0) < 12.0 and abs(rb - 128.0) < 12.0   # ~image center

    ga = ftheta_horizon_row(a, d, center="geometric")
    gb = ftheta_horizon_row(b, d, center="geometric")
    assert abs(ga - gb) > 30.0                         # the BUG (rig-inconsistent)
    assert gb > ga + 30.0                              # rig B lands lower


def test_horizon_end_to_end_through_crop_resize():
    """Paint the horizon band into a native frame, run the REAL crop+resize
    (center='principal', which replicate-pads rig B's below-frame overflow), and
    confirm the bright row matches the analytic prediction AND agrees across
    rigs — and that centering leaves the achieved f_eff at the canonical F_REF."""
    d = EXTR.vehicle_forward_in_cam()
    found = {}
    for name, cy in (("A", RIG_A_CY), ("B", RIG_B_CY)):
        intr = _rig(cy)
        _u, v = ftheta_project_ray(intr, d)            # native horizon row
        vi = int(round(v))
        frame = torch.zeros(1, 3, 1080, 1920, dtype=torch.uint8)
        frame[..., max(0, vi - 3):vi + 4, :] = 255     # 7-px bright band
        out = ftheta_crop_resize(frame, intr, 256, center="principal")
        assert out.shape == (1, 3, 256, 256) and out.dtype == torch.uint8
        bright_row = out[0].float().mean(dim=0).mean(dim=1)   # per-output-row mean
        found[name] = int(torch.argmax(bright_row))
        pred = ftheta_horizon_row(intr, d, center="principal")
        assert abs(found[name] - pred) < 3.0           # pipeline == analytic
        assert abs(ftheta_crop_resize.last_f_eff - F_REF) / F_REF < 0.02
    assert abs(found["A"] - found["B"]) < 3            # rigs AGREE end-to-end


def test_rig_b_crop_overflows_and_is_padded_not_shifted():
    """Rig B's (cx,cy)-centered crop overflows the native bottom (cy~755, side
    ~825 -> bottom row ~1168 > 1080); the crop must PAD (keep cy at the true crop
    center) rather than shift the box up (which would reintroduce the offset)."""
    b = _rig(RIG_B_CY)
    from tanitad.data.calib import ftheta_crop_box
    c, top, left = ftheta_crop_box(b, 1080, 1920, 256, center="principal")
    assert top + c > 1080                              # genuinely spills past bottom
    # the box top stays at cy - c/2 (padding handles the shortfall), so mapping
    # the principal point through it lands at the exact output center:
    center_row = ftheta_horizon_row(b, (0.0, 0.0, 1.0), center="principal")  # axis
    assert abs(center_row - 128.0) < 1.5
    # and the padded crop still decodes to the right shape without error
    frame = torch.randint(0, 255, (2, 3, 1080, 1920), dtype=torch.uint8)
    out = ftheta_crop_resize(frame, b, 256, center="principal")
    assert out.shape == (2, 3, 256, 256)


def test_principal_requires_per_clip_cy_else_warns_and_reverts():
    """The centered path REQUIRES a per-clip cy. Given the corpus-median fallback
    (per_clip=False, cy is a rig-B value) it warns and reverts to geometric —
    never silently centers on a wrong global cy."""
    assert PHYSICALAI_FRONT_WIDE_FTHETA.per_clip is False
    frame = torch.randint(0, 255, (1, 3, 1080, 1920), dtype=torch.uint8)
    from tanitad.data import calib
    calib._warned_geometric[0] = False                 # reset one-shot guard
    with pytest.warns(RuntimeWarning, match="per-clip principal point"):
        out = ftheta_crop_resize(frame, PHYSICALAI_FRONT_WIDE_FTHETA, 256,
                                 center="principal")
    assert out.shape == (1, 3, 256, 256)
    # reverted path == explicit geometric path (byte-identical)
    geo = ftheta_crop_resize(frame, PHYSICALAI_FRONT_WIDE_FTHETA, 256,
                             center="geometric")
    assert torch.equal(out, geo)
