"""Extract the C2 GEOMETRY slice from the full v5 dump that lives only on the
eval pod, so the shipped rule can be verified against the real fan + reference
roll (not just against the pre-reduced cost matrix).

Keeps float32 so the check can be BIT-EXACT. Drops `imag` and `ctrv`
(2 x 36 MB, rules A1/C1 — not needed for C2).
"""
import hashlib
import json
import sys

import torch

src, dst = sys.argv[1], sys.argv[2]
d = torch.load(src, map_location="cpu", weights_only=False)
out = {
    "fan": d["fan"],                                  # [881, 256, 20, 2] f32
    "imag_ref": d["imag_ref"],                        # [881, 20, 2]      f32
    "tgt": d["tgt"],                                  # [881, 20, 2]      f32
    "fan_err4": d["fan_err4"],                        # [881, 256]        f32
    "cost_C2_published": d["costs"]["C2_wm_ref_proximity"],
    "pick_C2_published": d["picks"]["C2_wm_ref_proximity"],
    "pick_A0_as_trained": d["picks"]["A0_as_trained"],
    "ade_C2_published": d["ade_by_arm"]["C2_wm_ref_proximity"],
    "ade_A0_published": d["ade_by_arm"]["A0_as_trained"],
    "ep": d["ep"], "t": d["t"], "v0": d["v0"],
    "_src": "tanitad-eval:/workspace/_v5/v5_v1_windows.pt (scorer = v1's WM)",
    "_dropped": "imag[881,256,20,2] and ctrv[881,256,20,2] — rules A1/C1, not C2",
    "_wp_steps": [5, 10, 15, 20],
}
torch.save(out, dst)
h = hashlib.sha256(open(dst, "rb").read()).hexdigest()
print(json.dumps({"dst": dst, "sha256": h,
                  "shapes": {k: list(v.shape) for k, v in out.items()
                             if hasattr(v, "shape")}}, indent=1))
