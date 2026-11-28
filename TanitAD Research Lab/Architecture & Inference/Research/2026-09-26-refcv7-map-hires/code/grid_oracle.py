"""GT-ONLY ceiling of a coarse BEV grid for the 10 cm map (no model, no training, no tuning).

For a cell size c (0.25 m = the NEW-2 lift / shared encoder grid; 0.5 m = refcv6's map grid;
0.1 m = identity CONTROL, must read exactly 1.0 for every present class):
  1. the 10 cm GT one-hot is AREA-AVERAGED onto the c grid (exact: 0.05 m supersampling, then
     c/0.05 average pooling), over SEEN cells only (a coarse cell's class fractions);
  2. decoded back to 10 cm the way HiresRefine does -- bilinear (align_corners=False) to the fine
     grid -- then argmax;
  3. scored on the SAME cells as the G-MAP-OVERFIT harness (fine GT != 255 AND lift-valid, the
     0-20 m band), IoU POOLED over frames, plus the 0.2 m tolerance P/R/F1 for the thin classes.
This is the best a decoder can do from c-grid CLASS FRACTIONS alone (8 numbers per cell). A
learned 64-channel code can carry more (sub-cell position), so it is a REFERENCE, not a hard cap.

Mode "frames16" runs on Thor over the prereg's 16 frames through the harness's own load_frames
(time guard on); prints sha12s and numbers only.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CODE = Path(sys.argv[1])
sys.path[:0] = [str(CODE / "stack"), str(CODE / "stack" / "scripts")]
sys.path.insert(0, "/home/nvidia/gmo_early_0327/code_t1/taniteval")      # scipy-free metrics
import map_hires_overfit as M  # noqa: E402
from tanitad.data.semantic_map_gt_fine import EXTENT_REFCV7  # noqa: E402
from tanitad.models import map_head_hires as H  # noqa: E402
from tanitad.models.trunk_shapes import frame_for_width  # noqa: E402
from taniteval import map_hires_metrics as MET  # noqa: E402

assert "code_t1" in MET.__file__
AUD = CODE / "TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-map-signal-audit/raw"
spec = M.load_spec(AUD / "gmo_spec.json", class_weights="uniform",
                   decision_rule="prior_corrected", band_keys=EXTENT_REFCV7.band_keys)
hcfg = H.MapHiresConfig(w_map_hires=1.0, x_max_m=100.0, y_half_m=30.0)
data = M.load_frames(spec, Path("/home/nvidia/data/refcv6-b1-416x1024-train"),
                     Path("/home/nvidia/data/sam3_gt_v3"),
                     Path("/home/nvidia/data/refcv6_train_eval139_extrinsics.json"),
                     frame_for_width(1024, 416), hcfg, 43,
                     cam_ts_dir=Path("/home/nvidia/data/_b1stage416/r0/camera_front_wide"))
codes = data["codes"].long()                                   # [16, 1000, 600]
lv = H.lift_valid_to_fine(data["valid"].any(dim=1), tuple(hcfg.out_hw))
N, HH, WW = codes.shape
BAND = 200                                                     # rows of 0-20 m


def decode_via(cell_m: float, c: torch.Tensor) -> torch.Tensor:
    """[H, W] codes -> [H, W] codes through a cell_m class-fraction grid."""
    if abs(cell_m - 0.1) < 1e-9:
        out = c.clone()
        out[out == 255] = 0              # unseen cells are never scored
        return out
    seen = (c != 255)
    oh = torch.zeros(8, HH, WW)
    oh.scatter_(0, c.clamp(max=7).unsqueeze(0), seen.unsqueeze(0).float())
    k = int(round(cell_m / 0.05))
    sup = oh.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)          # 0.05 m
    s_seen = seen.float().repeat_interleave(2, 0).repeat_interleave(2, 1)
    num = F.avg_pool2d(sup.unsqueeze(0), k, k)[0]                            # [8, h, w]
    den = F.avg_pool2d(s_seen[None, None], k, k)[0, 0]
    frac = num / den.clamp(min=1e-9)
    up = F.interpolate(frac.unsqueeze(0), size=(HH, WW), mode="bilinear",
                       align_corners=False)[0]
    return up.argmax(dim=0)


res = {}
for cell in (0.1, 0.25, 0.5):
    inter = np.zeros(8)
    union = np.zeros(8)
    stats = []
    for i in range(N):
        g = codes[i]
        p = decode_via(cell, g)
        sc = (g != 255) & lv[i]
        sc[BAND:] = False
        for k in range(8):
            gk, pk = (g == k) & sc, (p == k) & sc
            inter[k] += int((gk & pk).sum())
            union[k] += int((gk | pk).sum())
        gm = g.clone().numpy().astype(np.uint8)
        stats.append(MET.window_stats(p.numpy().astype(np.uint8), gm,
                                      valid=(sc.numpy())))
    st = {kk: np.stack([x[kk] for x in stats]) for kk in stats[0]}
    pl = MET.pooled(st)
    res[f"{cell}m"] = {
        "iou": {H.CLASS_KEYS[k]: (float(inter[k] / union[k]) if union[k] else None)
                for k in range(8)},
        "tol_0.2m_band0_20": {H.CLASS_KEYS[k]: {"P": float(pl["P"][k][0]), "R": float(pl["R"][k][0]),
                                                "F1": float(pl["F1"][k][0])}
                              for k in MET.THIN_CLASSES}}
print(json.dumps({"frames": list(zip(data["sha12"], data["raw_frame"])),
                  "axis_guard": data["axis_guard"], "band_rows": BAND,
                  "scored": "fine GT != 255 AND lift-valid, rows 0-200 (0-20 m)",
                  "oracle": res}, indent=1))
