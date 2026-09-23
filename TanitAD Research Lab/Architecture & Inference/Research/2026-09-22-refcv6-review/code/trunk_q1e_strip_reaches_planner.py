"""Q1e — does F1's rig-correlated black strip reach the PLANNER, or only the
perception branch?

The planner's scene feature is `pooled = s32.mean(dim=(2,3))`
(`timm_trunk.py:590`) — a GLOBAL spatial mean over the 13x32 stride-32 map. If
the bottom ~31 of 416 rows are a constant, they occupy ~1 of 13 stride-32 rows
and enter that mean.

⭐ THE INTERVENTION, not a correlation. Take clips that have **no** strip, black
out the bottom 31 rows, and measure how far `pooled` moves. Controls, each
matched on PIXEL COUNT so the comparison is about *where*, not *how much*:
  * **C1** black out an equal-area strip at the TOP (sky) — a region that is
    genuinely observed, so any shortcut there is content, not rig;
  * **C2** black out an equal-area VERTICAL strip at the left edge;
  * **C3** the null: the same frame twice (must read exactly 0).
And the reference scale: the distance between two DIFFERENT clips' pooled
features, which is what "a meaningful difference" looks like on this rig.

Evidence class: MEASURED (ours), real 416x1024 PNG frames, resnet34 ImageNet,
eval mode (so BatchNorm cannot confound — see trunk_q5b).
"""
from __future__ import annotations

import glob
import json
import os
import sys

import torch
import torchvision.io as tvio

CACHE = "D:/Projects/TanitAD-artifacts/v2ep-eval139-416x1024cyl"
N_ROWS = 31          # the measured mean strip height (q1d: mean 31.1)


def main() -> int:
    from tanitad.models.timm_trunk import TimmResNetTrunk, TimmTrunkConfig

    q1d = json.load(open(os.path.join(os.path.dirname(__file__), "..", "raw",
                                      "q1d_rig_black_strip.json")))
    clean = [r["clip_id"] for r in q1d["per_clip"]
             if r["n_rows_fully_black"] == 0][:6]
    if len(clean) < 2:
        print(json.dumps({"INCONCLUSIVE": "fewer than 2 strip-free clips"}))
        return 1

    t = TimmResNetTrunk(TimmTrunkConfig(
        model_name="resnet34.a1_in1k", frames=3, mode="shared",
        image_hw=(416, 1024), pretrained=True)).eval()

    def frame_of(cid):
        p = os.path.join(CACHE, cid + ".v2ep.pt")
        d = torch.load(p, map_location="cpu", weights_only=False)
        offs = torch.cat([torch.zeros(1, dtype=torch.int64),
                          torch.cumsum(d["jpeg_len"], 0)])
        dec = tvio.decode_png if d.get("codec") == "png" else tvio.decode_jpeg
        f = dec(d["jpeg_buf"][0:int(offs[1])], mode=tvio.ImageReadMode.RGB)
        return f.float().div(255.0)                   # [3, 416, 1024] in [0,1]

    def pooled_of(x3):
        x = torch.cat([x3, x3, x3], dim=0)[None]      # K=3 stack, [1,9,H,W]
        with torch.no_grad():
            _s16, _s32, p = t.forward_features(x)
        return p[0].double()

    def rel(a, b):
        return float((a - b).norm() / a.norm())

    rows = []
    base_pool = {}
    for cid in clean:
        f = frame_of(cid)
        p0 = pooled_of(f)
        base_pool[cid] = p0

        bot = f.clone(); bot[:, -N_ROWS:, :] = 0.0             # the INTERVENTION
        top = f.clone(); top[:, :N_ROWS, :] = 0.0              # C1
        ncol = int(round(N_ROWS * 1024 / 416))                 # equal AREA
        lef = f.clone(); lef[:, :, :ncol] = 0.0                # C2
        rows.append({
            "clip_id": cid,
            "n_rows_blacked": N_ROWS,
            "equal_area_cols": ncol,
            "rel_move_BOTTOM_strip": round(rel(p0, pooled_of(bot)), 6),
            "C1_rel_move_TOP_strip": round(rel(p0, pooled_of(top)), 6),
            "C2_rel_move_LEFT_strip": round(rel(p0, pooled_of(lef)), 6),
            "C3_null_same_frame_twice": round(rel(p0, pooled_of(f)), 12),
        })

    # the reference scale: distance BETWEEN clips
    ids = list(base_pool)
    between = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            between.append(rel(base_pool[ids[i]], base_pool[ids[j]]))
    bt = torch.tensor(between)
    tt = torch.tensor([r["rel_move_BOTTOM_strip"] for r in rows])
    out = {
        "n_clips": len(rows), "n_rows_blacked": N_ROWS,
        "pooled_dim": int(base_pool[ids[0]].numel()),
        "INTERVENTION_bottom_strip": {
            "mean_rel_move": round(float(tt.mean()), 6),
            "min": round(float(tt.min()), 6), "max": round(float(tt.max()), 6)},
        "CONTROL_top_strip_mean": round(float(torch.tensor(
            [r["C1_rel_move_TOP_strip"] for r in rows]).mean()), 6),
        "CONTROL_left_strip_mean": round(float(torch.tensor(
            [r["C2_rel_move_LEFT_strip"] for r in rows]).mean()), 6),
        "CONTROL_null_max": max(r["C3_null_same_frame_twice"] for r in rows),
        "REFERENCE_between_clip_distance": {
            "n_pairs": len(between),
            "mean": round(float(bt.mean()), 6),
            "min": round(float(bt.min()), 6), "max": round(float(bt.max()), 6)},
        "strip_move_as_frac_of_between_clip_distance":
            round(float(tt.mean() / bt.mean()), 4),
        "per_clip": rows,
    }
    json.dump(out, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
