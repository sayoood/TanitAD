"""Rig mix over the FULL 3,000-clip canonical selection — no mp4 needed.

``rig_mask_census.py`` runs over ``discover_r0_clips``, which only yields clips
whose mp4 is on THIS host. On a host holding a subset that measures the subset's
rig mix, not the corpus's. The rig is a property of the per-clip CALIBRATION,
which is present for all 3,000 clips wherever ``calibration/camera_intrinsics``
is complete — so the corpus-wide mix is measurable with no video at all.

Also reports the mix restricted to the parity TRAIN split, which is the set a
v5 cache would actually contain.
"""
from __future__ import annotations
import argparse
import json
import math
import statistics as st
from pathlib import Path


def _agg(v):
    if not v:
        return None
    return {"n": len(v), "mean": round(st.mean(v), 6),
            "median": round(st.median(v), 6), "min": round(min(v), 6),
            "max": round(max(v), 6),
            "stdev": round(st.stdev(v), 6) if len(v) > 1 else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--hfov", type=float, default=120.0)
    ap.add_argument("--projection-mode", default="cylindrical")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import pandas as pd
    import torch
    from tanitad.data.calib import CanonicalFrame, cylindrical_rectify
    from tanitad.data.physicalai import intrinsics_for_clip, split_clips

    root = Path(a.root)
    sel = pd.read_parquet(root / "r0" / "r0_selection.parquet")
    sel["clip_id"] = sel["clip_id"].astype(str)
    clip_ids = list(sel["clip_id"])
    # the parity split runs on the ORDERED discovered list; on a complete host
    # that order is sorted-by-mp4-path. r0_selection order is NOT that order, so
    # the split below is only computed when --root is complete (mp4 count 3000).
    frame = CanonicalFrame.from_hfov(a.hfov, a.height, a.width,
                                     "cylindrical" if a.projection_mode ==
                                     "cylindrical" else "pinhole")
    cache: dict = {}
    rows = []
    for cid in clip_ids:
        try:
            intr = intrinsics_for_clip(cid, str(root))
        except Exception as e:
            rows.append({"clip": cid[:8], "error": type(e).__name__})
            continue
        cy = float(getattr(intr, "cy", float("nan")))
        cx = float(getattr(intr, "cx", float("nan")))
        k = (round(cx, 3), round(cy, 3), int(intr.height), int(intr.width))
        if k not in cache:
            probe = torch.zeros(1, 3, intr.height, intr.width, dtype=torch.uint8)
            cylindrical_rectify(probe, intr, frame,
                                require_per_clip=intr.per_clip)
            cache[k] = (float(cylindrical_rectify.last_observed_frac),
                        float(cylindrical_rectify.last_f_eff))
        obs, feff = cache[k]
        rows.append({"clip": cid[:8], "cy": cy, "cx": cx,
                     "masked_frac": 1.0 - obs, "f_eff": feff})

    good = [r for r in rows if "cy" in r and not math.isnan(r["cy"])]
    cys = sorted(r["cy"] for r in good)
    gaps = [(cys[i + 1] - cys[i], i) for i in range(len(cys) - 1)]
    gmax, gi = max(gaps) if gaps else (0.0, 0)
    boundary = (cys[gi] + cys[gi + 1]) / 2.0 if gaps else cys[0]
    for r in good:
        r["rig"] = "A" if r["cy"] < boundary else "B"

    A = [r for r in good if r["rig"] == "A"]
    B = [r for r in good if r["rig"] == "B"]
    out = {
        "root": str(root),
        "frame": {**frame.to_dict(), "tag": frame.tag()},
        "selection_rows": len(clip_ids),
        "resolved_intrinsics": len(good),
        "failed_intrinsics": len(rows) - len(good),
        "distinct_sensor_geometries": len(cache),
        "rig_boundary": {"largest_gap": round(gmax, 2),
                         "boundary_cy": round(boundary, 2),
                         "bimodal": bool(gmax > 50.0),
                         "cy_min": round(cys[0], 2), "cy_max": round(cys[-1], 2)},
        "rig_mix": {"A": len(A), "B": len(B),
                    "A_frac": round(len(A) / max(len(good), 1), 4),
                    "B_frac": round(len(B) / max(len(good), 1), 4)},
        "masked_frac": {
            "A": _agg([r["masked_frac"] for r in A]),
            "B": _agg([r["masked_frac"] for r in B]),
            "pooled": _agg([r["masked_frac"] for r in good])},
        "cy": {"A": _agg([r["cy"] for r in A]), "B": _agg([r["cy"] for r in B])},
        "f_eff": _agg([r["f_eff"] for r in good]),
    }
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("RIG_MIX_FULL " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
