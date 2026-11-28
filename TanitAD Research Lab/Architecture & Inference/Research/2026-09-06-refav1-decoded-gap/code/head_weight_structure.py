#!/usr/bin/env python3
"""Where in the image does the gap head's weight actually live?

⭐ WHY THIS IS WORTH A ZERO-GPU MINUTE. The named next lever is a FINER POOLING:
the bank pools 16x40 -> 8x20 to match M84 exactly, which HALVES the vertical
resolution -- and vertical image position is the classic monocular range cue (a
more distant vehicle sits higher in the frame). That lever is worth a GPU only if
the head is in fact reading vertical structure.

⛔ THIS IS DESCRIPTIVE, NOT A CLAIM. A weight map says where a LINEAR functional
puts its mass; it does not establish what the trunk encodes, and it cannot settle
whether a finer pool would help. It is reported to say whether the lever is worth
pre-registering, and nothing more.

⚠️ CONTROL: the same statistic is computed for the PRESENCE head, whose decode is
almost entirely CLIP IDENTITY (within-clip skill NEGATIVE, TRUE - SHUFFLED
+0.0192). If the two maps look alike, the structure is not telling us about range.
"""
import json
import os
import sys

import numpy as np
import torch

R = r"C:\Users\Admin\dkhead_run"
B = torch.load(os.path.join(R, "head", "gap_head_bundle.pt"),
               map_location="cpu", weights_only=False)
PH, PW = B["pool"]                       # 8 x 20
out = {"task": "D-REFAV1-DK-DECODED -- gap-head weight structure (descriptive)",
       "evidence_class": "MEASURED (ours)", "pool": [PH, PW],
       "scope": ("a weight map locates a LINEAR functional's mass; it does NOT "
                 "establish what the trunk encodes and does NOT settle whether a "
                 "finer pooling would help"),
       "targets": {}}

for tgt in ("gap_all_lead", "present"):
    if tgt not in B["targets"]:
        continue
    rows, cols = [], []
    for f, h in sorted(B["targets"][tgt]["by_fold"].items()):
        V = h["V"].numpy().astype(np.float64).reshape(PH, PW, -1)
        e = (V ** 2).sum(-1)                       # energy per pooled cell
        e = e / max(e.sum(), 1e-30)
        rows.append(e.sum(1))                      # per VERTICAL band
        cols.append(e.sum(0))                      # per AZIMUTH bin
    rows = np.mean(rows, 0)
    cols = np.mean(cols, 0)
    # how concentrated is it? 1.0 = uniform across bands
    def conc(v):
        return float(v.max() / (1.0 / len(v)))
    out["targets"][tgt] = {
        "vertical_band_energy_top_to_bottom": [round(float(x), 4) for x in rows],
        "azimuth_bin_energy_left_to_right": [round(float(x), 4) for x in cols],
        "vertical_peak_band": int(np.argmax(rows)),
        "vertical_concentration_vs_uniform": round(conc(rows), 3),
        "azimuth_peak_bin": int(np.argmax(cols)),
        "azimuth_concentration_vs_uniform": round(conc(cols), 3),
        "centre_two_azimuth_bins_share": round(
            float(cols[PW // 2 - 1] + cols[PW // 2]), 4)}
    print("%-14s vertical (top->bottom) %s" % (tgt, np.round(rows, 3).tolist()))
    print("%-14s peak band %d (x%.2f uniform) | azimuth peak bin %d (x%.2f) | "
          "centre 2 bins %.3f"
          % ("", int(np.argmax(rows)), conc(rows), int(np.argmax(cols)),
             conc(cols), float(cols[PW // 2 - 1] + cols[PW // 2])))

g = out["targets"].get("gap_all_lead", {})
p = out["targets"].get("present", {})
if g and p:
    a = np.array(g["vertical_band_energy_top_to_bottom"])
    b = np.array(p["vertical_band_energy_top_to_bottom"])
    out["gap_vs_present_vertical_cosine"] = round(
        float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))), 4)
    print("gap vs present vertical-profile cosine: %.4f  (near 1.0 would mean "
          "the structure is NOT specific to range)"
          % out["gap_vs_present_vertical_cosine"])

d = os.path.join(R, "pkg", "raw", "head_weight_structure.json")
json.dump(out, open(d, "w"), indent=1)
print("wrote", d)
