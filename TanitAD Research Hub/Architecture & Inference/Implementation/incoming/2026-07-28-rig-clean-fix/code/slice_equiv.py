"""IS THE CLEAN FIX A SLICE OR A REBUILD? — measured, not argued.

Claim under test: rebuilding the corpus at a CENTRED sub-frame of the frame it
was built at produces pixels BIT-IDENTICAL to slicing the built frames.

Three independent legs, each able to fail:
  L1  ray/grid identity      — the sampling grid of the sub-frame equals the
                               sub-block of the parent's grid, EXACTLY (float ==)
  L2  mask identity          — likewise for the observed mask
  L3  pixel identity         — `cylindrical_rectify` at the sub-frame equals the
                               sliced output of `cylindrical_rectify` at the
                               parent, on real per-clip intrinsics, uint8-exact
  L4  NEGATIVE CONTROL       — a NON-centred slice, and a slice with a changed
                               f_ref, must DIFFER. A check that cannot fail is
                               not a check (class C13).
"""
from __future__ import annotations
import json
import sys
import torch

from tanitad.data.calib import (CanonicalFrame, FThetaIntrinsics,
                                centred_subframe, cylindrical_grid,
                                cylindrical_rectify, subframe_slice)

PARENT = CanonicalFrame(256, 640, 305.5774907364391, "cylindrical")
CANDS = [(176, 624), (128, 576), (192, 640), (160, 592)]


def probe_intrinsics() -> list[FThetaIntrinsics]:
    """Real per-clip values, one worst-case clip per rig (from band_full_3000)."""
    return [
        # rig A, the clip with the largest 256x640 mask on that rig
        FThetaIntrinsics(poly=(0.0, 940.2709762, 23.1353, -58.5012, 16.5067),
                         cx=951.31021821, cy=533.9515397, per_clip=True),
        # rig B, the largest cy in the corpus
        FThetaIntrinsics(poly=(0.0, 927.606943876695, 23.1353, -58.5012, 16.5067),
                         cx=952.5651245, cy=764.5196535, per_clip=True),
    ]


def main() -> None:
    torch.manual_seed(0)
    out: dict = {"parent": PARENT.tag(), "legs": []}
    vid = torch.randint(0, 256, (3, 3, 1080, 1920), dtype=torch.uint8)
    ok_all = True
    for intr in probe_intrinsics():
        rig = "B" if intr.cy >= 650 else "A"
        gp, mp = cylindrical_grid(intr, 1080, 1920, PARENT)
        yp = cylindrical_rectify(vid, intr, PARENT)
        for h, w in CANDS:
            sub = centred_subframe(PARENT, h, w)
            rs, cs = subframe_slice(PARENT, sub)
            gc, mc = cylindrical_grid(intr, 1080, 1920, sub)
            yc = cylindrical_rectify(vid, intr, sub)
            l1 = bool(torch.equal(gc, gp[:, rs, cs, :]))
            l2 = bool(torch.equal(mc, mp[rs, cs]))
            l3 = bool(torch.equal(yc, yp[..., rs, cs]))
            maxabs = int((yc.int() - yp[..., rs, cs].int()).abs().max())
            ok_all &= l1 and l2 and l3
            out["legs"].append({
                "rig": rig, "sub": sub.tag(), "h": h, "w": w,
                "rows": [rs.start, rs.stop], "cols": [cs.start, cs.stop],
                "L1_grid_identical": l1, "L2_mask_identical": l2,
                "L3_pixels_identical": l3, "max_abs_pixel_diff": maxabs,
                "masked_frac_sub": float(1.0 - mc.float().mean()),
            })

    # ---- L4: negative controls, which MUST differ --------------------------
    intr = probe_intrinsics()[1]
    yp = cylindrical_rectify(vid, intr, PARENT)
    neg = []
    sub = centred_subframe(PARENT, 176, 624)
    yc = cylindrical_rectify(vid, intr, sub)
    # (a) off-centre slice of the parent
    off = yp[..., 42:42 + 176, 8:8 + 624]
    neg.append({"control": "off_centre_row_offset_42",
                "identical": bool(torch.equal(yc, off)),
                "must_be": False})
    # (b) same size, different f_ref (a RESAMPLE, not a slice)
    sub2 = CanonicalFrame(176, 624, PARENT.f_ref * 1.01, "cylindrical")
    yc2 = cylindrical_rectify(vid, intr, sub2)
    neg.append({"control": "f_ref_x1.01",
                "identical": bool(torch.equal(yc2, yp[..., 40:216, 8:632])),
                "must_be": False})
    # (c) subframe_slice must REFUSE a changed f_ref
    try:
        subframe_slice(PARENT, sub2)
        refused = False
    except ValueError:
        refused = True
    neg.append({"control": "subframe_slice_refuses_changed_f_ref",
                "refused": refused, "must_be": True})
    # (d) odd margin must be refused
    try:
        centred_subframe(PARENT, 175, 624)
        refused_odd = False
    except ValueError:
        refused_odd = True
    neg.append({"control": "centred_subframe_refuses_odd_margin",
                "refused": refused_odd, "must_be": True})
    out["negative_controls"] = neg
    neg_ok = (neg[0]["identical"] is False and neg[1]["identical"] is False
              and neg[2]["refused"] and neg[3]["refused"])
    out["verdict"] = {"all_slices_bit_identical": bool(ok_all),
                      "negative_controls_behaved": bool(neg_ok),
                      "SLICE_NOT_REBUILD": bool(ok_all and neg_ok)}
    print(json.dumps(out, indent=1))
    sys.exit(0 if out["verdict"]["SLICE_NOT_REBUILD"] else 1)


if __name__ == "__main__":
    main()
