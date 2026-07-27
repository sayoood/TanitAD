"""Two things the FOV-audit stream raised, settled by measurement.

(A) THE ROUTE CONFLICT. That stream reports: *"a non-square input need NOT change
    state_dim. Adaptive-4x4 pooling keeps 2048 ... The grid_w route a sibling
    stream is building is the expensive one."* This re-derives BOTH pooling
    routes against this implementation and reports whether they are the same
    operation, and what each costs.

(B) ⛔ A LIVE DEFECT IN THE DEPLOYED INPUT. Same stream: *"today's crop pads
    11.3 % of rows on rig B, 0 % on rig A (~71 % of the corpus)."* Reproduced
    here from the real per-clip intrinsics, then re-measured through the
    rectangular crop and the cylindrical path to answer the only question that
    matters for v5: does the WIDE rebuild inherit the asymmetry or remove it?

⚠️ No clip UUID is emitted (PhysicalAI-AV is gated).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn

STACK = Path(__file__).resolve().parents[5] / "stack"
sys.path.insert(0, str(STACK))

from tanitad.data import calib as C                                # noqa: E402
from tanitad.models.readout import SpatialGridReadout              # noqa: E402


# --------------------------------------------------------------------------- #
# (A) pooling routes                                                            #
# --------------------------------------------------------------------------- #
def route_check() -> dict:
    g = torch.Generator().manual_seed(7)
    rows = []
    # (token_h, token_w) for every live candidate at patch 16, plus a
    # deliberately NON-DIVISIBLE grid to find where the two routes differ.
    grids = [(16, 16, "deployed 256x256"), (16, 40, "256x640"),
             (24, 60, "384x960"), (24, 24, "384x384"),
             (16, 42, "672px wide - NOT divisible by 4")]
    for th, tw, label in grids:
        x = torch.randn(4, 128, th, tw, generator=g)
        adaptive = nn.AdaptiveAvgPool2d((4, 4))(x)
        row = {"token_grid": [th, tw], "label": label,
               "adaptive_out": list(adaptive.shape[-2:]),
               "adaptive_state_dim": 4 * 4 * 128}
        if th % 4 == 0 and tw % 4 == 0:
            exact = nn.AvgPool2d((th // 4, tw // 4))(x)
            row["exact_kernel"] = [th // 4, tw // 4]
            row["exact_out"] = list(exact.shape[-2:])
            row["exact_state_dim"] = 4 * 4 * 128
            row["bit_identical"] = bool(torch.equal(adaptive, exact))
            row["max_abs_diff"] = float((adaptive - exact).abs().max())
        else:
            row["exact_kernel"] = None
            row["bit_identical"] = None
            row["note"] = ("AvgPool2d cannot tile this grid; AdaptiveAvgPool2d "
                           "can. This is the ONLY case where the routes differ.")
        rows.append(row)

    # what THIS implementation's default actually does, end to end
    defaults = []
    for th, tw, label in grids[:4]:
        ro = SpatialGridReadout(th * tw, 128, grid=4, d_readout=128,
                                token_grid=(th, tw))          # grid_w defaults None
        out = ro(torch.randn(2, th * tw, 128, generator=g))
        defaults.append({"token_grid": [th, tw], "label": label,
                         "pool_kernel": list(ro.pool.kernel_size)
                         if hasattr(ro.pool, "kernel_size") else "adaptive",
                         "out_dim": ro.out_dim,
                         "forward_shape": list(out.shape),
                         "state_dim_2048": ro.out_dim == 2048})
    # ...and what the OPT-IN grid_w knob does when explicitly switched on
    ro_w = SpatialGridReadout(16 * 40, 128, grid=4, d_readout=128,
                              token_grid=(16, 40), grid_w=10)
    return {
        "pooling_routes": rows,
        "this_impl_default": defaults,
        "grid_w_knob": {
            "is_default": False,
            "default_value": None,
            "out_dim_when_set_to_10": ro_w.out_dim,
            "note": "grid_w is an OPT-IN knob that spends readout cells "
                    "asymmetrically; it is None by default, so the DEFAULT path "
                    "is the 4x4 route. Setting it DOES change state_dim (and "
                    "therefore breaks checkpoint loading) — which is exactly the "
                    "cost the FOV-audit stream is warning about.",
        },
    }


# --------------------------------------------------------------------------- #
# (B) rig asymmetry                                                             #
# --------------------------------------------------------------------------- #
_POLY = (0.0, 927.5032, 23.1353, -58.5012, 16.5067)


def pad_fraction(intr: C.FThetaIntrinsics, frame: C.CanonicalFrame,
                 h: int = 1080, w: int = 1920) -> dict:
    """Fraction of OUTPUT rows/cols that are replicate-padded by the crop path
    because the principal-point-centred box spills past the native edge."""
    c_h, c_w, top, left = C.ftheta_crop_box_hw(intr, h, w, center="principal",
                                               frame=frame)
    pt, pb = max(0, -top), max(0, (top + c_h) - h)
    pl, pr = max(0, -left), max(0, (left + c_w) - w)
    return {"crop_hw": [c_h, c_w], "top": top, "left": left,
            "pad_top": pt, "pad_bottom": pb, "pad_left": pl, "pad_right": pr,
            "padded_row_frac": round((pt + pb) / c_h, 5),
            "padded_col_frac": round((pl + pr) / c_w, 5)}


def rig_check(root: str | None, n_clips: int) -> dict:
    frames = {
        "deployed_256sq": C.CANONICAL_256,
        "100deg_256x640_pin": C.CanonicalFrame.from_hfov(100.0, 256, 640),
        "120deg_256x640_pin": C.CanonicalFrame.from_hfov(120.0, 256, 640),
        "100deg_256x640_cyl": C.CanonicalFrame.from_hfov(100.0, 256, 640,
                                                         "cylindrical"),
        "120deg_256x640_cyl": C.CanonicalFrame.from_hfov(120.0, 256, 640,
                                                         "cylindrical"),
    }
    # nominal rigs (the documented cy split) — deterministic, no data needed
    nominal = {}
    for rig, cy in (("A", 543.0), ("B", 755.0)):
        intr = C.FThetaIntrinsics(poly=_POLY, cx=958.0, cy=cy, width=1920,
                                  height=1080, per_clip=True)
        per_frame = {}
        for fname, fr in frames.items():
            if fr.projection == "cylindrical":
                probe = torch.zeros(1, 3, 1080, 1920, dtype=torch.uint8)
                C.cylindrical_rectify(probe, intr, fr)
                mask = C.cylindrical_rectify.last_mask
                per_frame[fname] = {
                    "path": "cylindrical",
                    "observed_frac": round(float(mask.float().mean()), 5),
                    "unobserved_frac": round(1 - float(mask.float().mean()), 5),
                    "fully_unobserved_rows": int((~mask).all(dim=1).sum()),
                    "rows": int(mask.shape[0]),
                    "fabricated_pixels": False,
                }
            else:
                pf = pad_fraction(intr, fr)
                per_frame[fname] = {"path": "ftheta_crop", **pf,
                                    "fabricated_pixels": pf["padded_row_frac"] > 0}
        nominal[f"rig{rig}_cy{cy:.0f}"] = per_frame

    out = {"nominal_rigs": nominal}

    # real per-clip intrinsics, if the corpus is reachable
    if root:
        from tanitad.data.physicalai import (discover_r0_clips,
                                             intrinsics_for_clip)
        clips = discover_r0_clips(root)[:n_clips]
        rows = []
        for i, cl in enumerate(clips):
            intr = intrinsics_for_clip(cl["clip_id"], root)
            r = {"index": i,
                 "clip_sha1_8": hashlib.sha1(cl["clip_id"].encode()
                                             ).hexdigest()[:8],
                 "rig": "B" if intr.cy > 650 else "A",
                 "cy": round(intr.cy, 1), "per_clip": bool(intr.per_clip)}
            r["deployed_padded_row_frac"] = pad_fraction(
                intr, C.CANONICAL_256)["padded_row_frac"]
            rows.append(r)
        a = [r["deployed_padded_row_frac"] for r in rows if r["rig"] == "A"]
        b = [r["deployed_padded_row_frac"] for r in rows if r["rig"] == "B"]
        out["real_clips"] = {
            "n": len(rows), "clips": rows,
            "rigA_mean_padded_row_frac": round(sum(a) / max(1, len(a)), 5),
            "rigB_mean_padded_row_frac": round(sum(b) / max(1, len(b)), 5),
            "asymmetry_confirmed": bool(a and b and max(a) < min(b)),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--clips", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"artifact": "route conflict + rig-asymmetry resolution",
           "date": "2026-07-27",
           "route": route_check(),
           "rig": rig_check(a.root, a.clips)}
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")

    print("=== (A) POOLING ROUTES ===")
    for r in res["route"]["pooling_routes"]:
        print(f"  {str(r['token_grid']):10s} {r['label']:34s} "
              f"exact_kernel={str(r['exact_kernel']):9s} "
              f"bit_identical={r['bit_identical']}")
    print("  default path state_dim:",
          [d["out_dim"] for d in res["route"]["this_impl_default"]])
    print("\n=== (B) RIG ASYMMETRY ===")
    for rig, d in res["rig"]["nominal_rigs"].items():
        print(f"  {rig}")
        for fn, v in d.items():
            if v["path"] == "ftheta_crop":
                print(f"    {fn:22s} crop padded_rows="
                      f"{v['padded_row_frac']*100:6.2f}%  fabricated="
                      f"{v['fabricated_pixels']}")
            else:
                print(f"    {fn:22s} cyl  unobserved="
                      f"{v['unobserved_frac']*100:6.2f}%  fabricated="
                      f"{v['fabricated_pixels']}")
    if "real_clips" in res["rig"]:
        rc = res["rig"]["real_clips"]
        print(f"  REAL clips n={rc['n']}: rigA {rc['rigA_mean_padded_row_frac']*100:.2f}%"
              f"  rigB {rc['rigB_mean_padded_row_frac']*100:.2f}%"
              f"  asymmetry_confirmed={rc['asymmetry_confirmed']}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
