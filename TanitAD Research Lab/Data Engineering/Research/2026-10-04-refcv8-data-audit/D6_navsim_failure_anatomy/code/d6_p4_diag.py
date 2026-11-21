"""D6 P4 DIAGNOSTIC (post-hoc, not registered, enters no bar): WHY does the gate reject the deployed pick?   NAVSIM VENV.

For every scene whose deployed pick FAILS the G1 gate, the failing check points of that pick: their GT drivability (metric-cache
drivable_area_map bits, raw/p4_orc), distance band, point type and the head's prior-corrected class there; split by whether the
deployed pick is DAC-clean in the banked G0 frame (a rejection of a DAC-clean pick is a map FALSE ALARM at the pick level).
Writes raw/p4_diag_rejections.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d6_p4_common as C  # noqa: E402
import d6_p4_gate as G  # noqa: E402

NAMES = ("nocls", "drivable", "lane", "crosswalk", "arrow", "edge", "hatched", "sidewalk")
XB = (5.0, 10.0, 20.0, 40.0)
XL = ("<5", "5-10", "10-20", "20-40", ">=40")


def main():
    fan = G.load_fan_checked()
    gt = G.load_orc_bits()
    ms = C.MaskStore(os.path.join(C.RAW, "p4_mask"))
    g0 = pd.read_csv(os.path.join(C.RAW, "p4_frame_G0.csv")).set_index("token")
    picks = json.load(open(os.path.join(C.RAW, "p4_picks_G1.json"), encoding="utf-8"))["picks"]
    out = {}
    for grp in ("pick_DAC_clean", "pick_DAC_zero"):
        out[grp] = {"n_scenes_rejected": 0, "n_scenes": 0, "fail_points": 0, "fail_points_gt_drivable": 0,
                    "by_x": {x: 0 for x in XL}, "by_x_gt_drivable": {x: 0 for x in XL}, "by_point": {p: 0 for p in C.PT_NAMES},
                    "by_class_gt_drivable": {n: 0 for n in NAMES}, "first_fail_time_s": {}, "changed": 0, "fallback": 0}
    for t, f in fan.items():
        grp = "pick_DAC_clean" if g0.loc[t, "drivable_area_compliance"] == 1.0 else "pick_DAC_zero"
        o = out[grp]
        o["n_scenes"] += 1
        if picks[t]["deployed_passes"]:
            continue
        o["n_scenes_rejected"] += 1
        o["changed"] += int(picks[t]["changed"])
        o["fallback"] += int(picks[t]["fallback"])
        rec = ms.get(t, what=("m_hat", "cls"))
        pts = G.devkit_points(f["poses"][f["sel_idx"]][None])[0]           # [8, 5, 2]
        i, j, ins = C.cell_index(pts)
        drv = np.ones(ins.shape, bool)
        drv[ins] = rec["m_hat"][i[ins], j[ins]]
        bad = ins & ~drv
        g = gt[t][f["sel_idx"]]
        o["fail_points"] += int(bad.sum())
        o["fail_points_gt_drivable"] += int((bad & g).sum())
        tf = int(np.argmax(bad.any(axis=1)))
        o["first_fail_time_s"][str(0.5 * (tf + 1))] = o["first_fail_time_s"].get(str(0.5 * (tf + 1)), 0) + 1
        xb = np.digitize(pts[..., 0], XB)
        for k, lab in enumerate(XL):
            o["by_x"][lab] += int((bad & (xb == k)).sum())
            o["by_x_gt_drivable"][lab] += int((bad & g & (xb == k)).sum())
        for k, p in enumerate(C.PT_NAMES):
            o["by_point"][p] += int(bad[:, k].sum())
        cl = rec["cls"][i, j]
        for c in range(8):
            o["by_class_gt_drivable"][NAMES[c]] += int((bad & g & (cl == c)).sum())
    for o in out.values():
        o["share_rejected"] = o["n_scenes_rejected"] / max(o["n_scenes"], 1)
        o["share_fail_points_gt_drivable"] = o["fail_points_gt_drivable"] / max(o["fail_points"], 1)
    out["note"] = "POST-HOC DIAGNOSTIC; GT = metric-cache drivable polygons at the PLAN's check points (DAC itself tests the LQR-tracked corners)."
    json.dump(out, open(os.path.join(C.RAW, "p4_diag_rejections.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
