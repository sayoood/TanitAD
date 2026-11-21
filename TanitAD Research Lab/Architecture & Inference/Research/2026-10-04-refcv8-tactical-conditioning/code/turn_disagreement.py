"""MM pre-landing check on P8 (2026-10-04): DECOMPOSE the v9 TURN label <-> plan TAG disagreement.

`label_tag_agreement.py` measured, on the GT path, lateral agreement 0.928 on the [2, 6] s overlap but TURN_L 0.65 /
TURN_R 0.60. R8-4 (iii)'s consistency bar is defined on these tags, so the gap is decomposed with v9's OWN fields
(`turn_t_start_s`, `turn_t_end_s`, `turn_in_progress`, `curve`, `lat_theta_deg`) into causes, each disagreeing row
assigned to the FIRST matching cause in the Master Mind's order:

  direction T (a TURN label the tag does not call that turn):
    (1) band     -- the v9 turn segment starts after the tag window ends (t_start > 6 s);
    (2) early    -- the segment starts before NOW + 2 s (t_start < 2 s: partially inside the band);
    (3) curve    -- (the curve variant applies to the other direction; listed for completeness, 0 by definition);
    (4) thresh   -- the segment starts inside the tag window but the heading the TAG can see there is < 30 deg:
                    (4a) the segment runs past the window end (t_end > 6 s: the tag sees part of the turn),
                    (4b) the segment lies inside the window and the tag's max segment heading is still < 30 deg
                         (the tag's per-segment heading vs v9's segment dpsi -- the definitions differ);
    (5) residual -- none of the above (incl. a tag of the OPPOSITE side).
  direction L (a LANE_KEEP label the tag calls a TURN):
    (1) band     -- n/a (a LANE_KEEP band holds no turn segment after 6 s by construction) -- counted if t_start > 6;
    (2) early    -- the tag's turn lies before the band: the tag window's max heading is reached before NOW + 2 s;
    (3) curve    -- v9 `curve` = 1: |Theta| >= 30 deg on a non-junction curve, which variant a labels LANE_KEEP;
    (4) thresh   -- v9's band excursion |Theta| < 30 deg while the tag's >= 30 deg (two windows, one threshold);
    (5) residual.
Tag windows: PLAN [0, 6] s (what R8-4 (iii) tags the fan with) and OVERLAP [2, 6] s. ⛔ No clip id is written.
Run:  PYTHONPATH=<tree>/stack python code/turn_disagreement.py --split train|eval139
Writes raw/turn_disagreement_<split>.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import label_tag_agreement as LTA  # noqa: E402

TURN_V7 = (6, 7)
SIDE = {6: 1, 7: -1}
WINDOWS = {"plan_0_6s": (0.0, 6.0), "overlap_2_6s": (2.0, 6.0)}


def seg_headings(paths):
    """The tagger's own per-segment headings (rad) of origin -> slot 1 -> ... (refcv8_conditioning.tag_paths)."""
    q = np.concatenate([np.zeros((len(paths), 1, 2)), paths], 1)
    d = q[:, 1:] - q[:, :-1]
    seg = np.hypot(d[..., 0], d[..., 1])
    return np.where(seg < 0.05, 0.0, np.arctan2(d[..., 1], d[..., 0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("train", "eval139"), default="eval139")
    a = ap.parse_args()
    import torch
    import tanitad
    if str(Path(tanitad.__file__).resolve()).upper().startswith("G:"):
        raise SystemExit("tanitad imported from G:")
    from tanitad.data import v9_labels as V9L
    from tanitad.data.v2_dataset import stable_episode_id
    from tanitad.refs import refcv8_conditioning as r8c
    t0 = time.time()
    man = torch.load(LTA.MAN[a.split], map_location="cpu", weights_only=False)
    rel = V9L.load_v9_release(LTA.V9[a.split])
    R = rel.rows
    plans, ovls, v0p, v0o, rows = [], [], [], [], []
    for i, cid in enumerate(man["clip_id"]):
        P = man["poses"][i].numpy().astype(np.float64)
        T = len(P)
        n = T - (LTA.W + 20)
        if n <= 0:
            continue
        r = np.arange(n) + LTA.W - 1
        ok = r + LTA.FUT <= T - 1
        c = rel.clips.get(int(stable_episode_id(cid)))
        if c is None or not ok.any():
            continue
        r = r[ok]
        r0, nr, k0 = c
        j = r + 2 - k0
        inr = (j >= 0) & (j < nr)
        r, j = r[inr], j[inr]
        plans.append(LTA.frame_slots(P, r, LTA.PLAN_H))
        ovls.append(LTA.frame_slots(P, r + LTA.OVL_ANCHOR, LTA.OVL_H))
        v0p.append(P[r, 3])
        v0o.append(P[r + LTA.OVL_ANCHOR, 3])
        rows.append(r0 + j)
    paths = {"plan_0_6s": (np.concatenate(plans), np.concatenate(v0p), LTA.PLAN_H, 0.0),
             "overlap_2_6s": (np.concatenate(ovls), np.concatenate(v0o), LTA.OVL_H, 2.0)}
    rows = np.concatenate(rows)
    lab = np.asarray(R["lat_v7id_a"]).astype(np.int64)[rows]
    f = {k: np.asarray(R[k]).astype(np.float64)[rows] for k in
         ("turn_t_start_s", "turn_t_end_s", "lat_theta_deg", "turn_dyaw_deg")}
    curve = np.asarray(R["curve"]).astype(np.int64)[rows]
    t_s, t_e = f["turn_t_start_s"], f["turn_t_end_s"]
    out = {"_evidence": "MEASURED (WP-B; GT path, the trainer's tagger, v9 fields; no model)", "split": a.split,
           "n_windows": int(len(rows)), "windows": {}}
    for wname, (pth, v0, hz, off) in paths.items():
        tl_idx, _ = r8c.tag_paths(torch.tensor(pth, dtype=torch.float32)[:, None], torch.tensor(v0, dtype=torch.float32),
                                  [h * 0.1 for h in hz])
        tag = np.asarray(r8c.LAT3_V7_IDS)[tl_idx[:, 0].numpy()]
        th = seg_headings(pth)                                  # [n, S] rad, the tagger's
        exc = np.abs(th).max(1)                                 # the tag's max |heading| in its window
        arg = np.abs(th).argmax(1)
        t_arg = off + np.asarray(hz)[arg] * 0.1                 # when (s after NOW) that max is reached
        w1 = WINDOWS[wname][1]
        # ---- direction T: a TURN label the tag does not call that turn ------------------------------------
        dT = np.isin(lab, TURN_V7) & (tag != lab)
        nT_lab = int(np.isin(lab, TURN_V7).sum())
        cT = np.full(len(lab), "", object)
        rem = dT.copy()

        def take(mask, name, cell, rem=rem):
            m = rem & mask
            cell[m] = name
            rem &= ~m
        take(np.isfinite(t_s) & (t_s > w1), "1_band", cT)
        take(np.isfinite(t_s) & (t_s < 2.0), "2_early", cT)
        take(np.isfinite(t_s) & (t_s <= w1) & np.isfinite(t_e) & (t_e > w1), "4a_thresh_truncated", cT)
        take(np.isfinite(t_s) & (t_s <= w1) & (exc < math.radians(r8c.TAG_TURN_DEG)), "4b_thresh_definition", cT)
        opp = rem & np.isin(tag, TURN_V7)
        cT[opp] = "5_residual_opposite_side"
        rem &= ~opp
        cT[rem] = "5_residual"
        # ---- direction L: a LANE_KEEP label the tag calls a TURN ------------------------------------------
        dL = (lab == 0) & np.isin(tag, TURN_V7)
        nL_lab = int((lab == 0).sum())
        cL = np.full(len(lab), "", object)
        remL = dL.copy()

        def takeL(mask, name):
            m = remL & mask
            cL[m] = name
            remL[:] = remL & ~m
        takeL(np.isfinite(t_s) & (t_s > w1), "1_band")
        takeL(t_arg < 2.0, "2_early")
        takeL(curve == 1, "3_curve")
        takeL(np.isfinite(f["lat_theta_deg"]) & (np.abs(f["lat_theta_deg"]) < r8c.TAG_TURN_DEG), "4_thresh")
        cL[remL] = "5_residual"

        def table(cells, mask, n_lab):
            names = sorted({c for c in cells[mask]})
            n = int(mask.sum())
            return {"n_label_rows": n_lab, "n_disagree": n, "disagree_share_of_label": round(n / max(n_lab, 1), 4),
                    "cells": {k: {"n": int((cells[mask] == k).sum()), "share": round(float((cells[mask] == k).mean()), 4)}
                              for k in names}}
        # ---- the RESTRICTED agreement the consistency metric can use: only label rows whose hypothesis is
        # OBSERVABLE in the tag window -- a TURN whose v9 segment lies inside it, a LANE_KEEP that is not a v9 curve
        w0 = WINDOWS[wname][0]
        obsT = np.isin(lab, TURN_V7) & np.isfinite(t_s) & np.isfinite(t_e) & (t_s >= w0) & (t_e <= w1)
        obsL = (lab == 0) & (curve == 0)
        restricted = {"turn_rows_segment_inside_window": int(obsT.sum()),
                      "turn_agree_restricted": round(float((tag[obsT] == lab[obsT]).mean()), 4) if obsT.any() else None,
                      "turn_share_observable": round(float(obsT.sum()) / max(nT_lab, 1), 4),
                      "lanekeep_rows_not_curve": int(obsL.sum()),
                      "lanekeep_agree_restricted": round(float((tag[obsL] == 0).mean()), 4) if obsL.any() else None}
        out["windows"][wname] = {"T_turn_label_missed": table(cT, dT, nT_lab),
                                 "L_lanekeep_label_tagged_turn": table(cL, dL, nL_lab),
                                 "restricted_agreement": restricted}
    out["wall_s"] = round(time.time() - t0, 1)
    p = HERE.parent / "raw" / f"turn_disagreement_{a.split}.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for wname, w in out["windows"].items():
        for d, b in w.items():
            print(f"{wname:13s} {d:30s} label n={b['n_label_rows']:7d} disagree n={b['n_disagree']:6d} "
                  f"({b['disagree_share_of_label']})")
            for k, v in b["cells"].items():
                print(f"      {k:28s} n={v['n']:6d}  share={v['share']}")
    print("wrote", p, out["wall_s"], "s")


if __name__ == "__main__":
    sys.exit(main())
