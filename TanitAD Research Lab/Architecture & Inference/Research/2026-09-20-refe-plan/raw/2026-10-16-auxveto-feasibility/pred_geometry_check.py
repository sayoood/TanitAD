#!/usr/bin/env python3
"""pred_geometry_check.py -- do the banked PREDICTED rasters (A2, 10k updates, held-out) sit in the SAME frame as the GT store?

A geometry check only: pooled IoU of the predicted mask against the GT mask on the head's own 40 x 32 grid (2 m cells), under the identity
convention and under axis flips / shifts that a frame bug would produce.  The identity convention must reproduce A2's banked held-out IoU
[0,40) = 0.675218 (A2_RESULT.json).  No proposal, no DAC, no pick is read here.
"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
PK = HERE.parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(PK / "refe"))
import agreement_core as AC
import aux_label_store as A

d = AC.load_c3()
N = len(d["token"])
gt_frac = np.zeros((N, 40, 32))
for i, (t, lg) in enumerate(zip(d["token"], d["log"])):
    m = (A.read_label(AC.STORE, lg, t) > 0).astype(np.float64)                  # [160, 128] fine mask
    gt_frac[i] = m.reshape(40, 4, 32, 4).mean(axis=(1, 3))                      # 2 m cells = 4 x 4 fine cells
gt = gt_frac >= 0.5                                                              # aux_s1_readout.py:321 (mask_frac >= 0.5)

def load_units(root):
    logit = np.full((N, 40, 32), np.nan, np.float32); sdf = np.full((N, 40, 32), np.nan, np.float32); seen = np.zeros(N, int)
    for k in range(5):
        z = np.load(Path(root) / f"V_s0_f{k}.npz", allow_pickle=False)
        logit[z["held_idx"]] = z["logit"]; sdf[z["held_idx"]] = z["sdf"]; seen[z["held_idx"]] += 1
    assert (seen == 1).all()
    return logit, sdf

def iou(pred, truth, lo=4, hi=24):
    p, t = pred[:, lo:hi], truth[:, lo:hi]
    return float((p & t).sum() / (p | t).sum())

res = {}
for name, root in (("A2_10000_updates", "D:/Projects/TanitAD-artifacts/scene_tap_probe_a2/units"), ("A3R_400_updates", "D:/Projects/TanitAD-artifacts/scene_tap_probe_a3r/units"),
                   ("PROBE_2000_updates_V_s0", "D:/Projects/TanitAD-artifacts/scene_tap_probe/units")):
    logit, sdf = load_units(root)
    pm = logit > 0
    r = {"sign_agreement_logit_vs_sdf": float(((logit > 0) == (sdf > 0)).mean()), "iou_x0_40m_identity": iou(pm, gt), "iou_all_x": iou(pm, gt, 0, 40)}
    r["flip_y"] = iou(pm[:, :, ::-1], gt); r["flip_x"] = iou(pm[:, ::-1, :], gt)
    r["shift_x_plus1cell"] = iou(np.roll(pm, 1, 1), gt); r["shift_x_minus1cell"] = iou(np.roll(pm, -1, 1), gt)
    r["shift_y_plus1cell"] = iou(np.roll(pm, 1, 2), gt); r["shift_y_minus1cell"] = iou(np.roll(pm, -1, 2), gt)
    r["sdf_range_m"] = [float(np.nanmin(sdf)), float(np.nanmax(sdf))]
    res[name] = r
res["banked_A2_H_iou_0_40"] = json.load(open(PK / "raw/2026-10-16-scene-tap-a2/A2_RESULT.json")).get("H") if (PK / "raw/2026-10-16-scene-tap-a2/A2_RESULT.json").exists() else None
res["gt_drivable_share_fine"] = float(gt_frac.mean())
json.dump(res, open(HERE / "PRED_GEOMETRY.json", "w"), indent=1)
print(json.dumps(res, indent=1))
