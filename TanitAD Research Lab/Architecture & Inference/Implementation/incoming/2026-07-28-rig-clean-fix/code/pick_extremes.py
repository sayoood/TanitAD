"""Pick real per-clip intrinsics for the test fixtures: the worst clip of each
rig at 256x640, plus a typical rig-B clip. No clip ids printed."""
import json, sys
import pandas as pd
from tanitad.data.calib import CanonicalFrame, observed_report
from tanitad.data.physicalai import intrinsics_for_clip
root = sys.argv[1]
P = CanonicalFrame(256, 640, 305.5774907364391, "cylindrical")
sel = pd.read_parquet(f"{root}/r0/r0_selection.parquet")
rows = []
seen = set()
for cid in sel["clip_id"].astype(str):
    it = intrinsics_for_clip(cid, root)
    if not it.per_clip: continue
    k = (round(it.cx,4), round(it.cy,4))
    if k in seen: continue
    seen.add(k)
    m = observed_report(it, P)["masked_frac"]
    rows.append({"cx": it.cx, "cy": it.cy, "poly": list(it.poly),
                 "w": it.width, "h": it.height, "masked": m,
                 "rig": "B" if it.cy >= 650.872 else "A"})
A = [r for r in rows if r["rig"]=="A"]; B=[r for r in rows if r["rig"]=="B"]
out = {"A_worst": max(A, key=lambda r: r["masked"]),
       "B_worst": max(B, key=lambda r: r["masked"]),
       "B_typical": sorted(B, key=lambda r: r["masked"])[len(B)//2],
       "A_typical": sorted(A, key=lambda r: r["masked"])[len(A)//2],
       "n_distinct": len(rows)}
# the 192x* and 176x640 binding cases, so the tests carry the real worst case
for tag,(hh,ww) in {"192x576":(192,576),"176x640":(176,640)}.items():
    F = CanonicalFrame(hh, ww, P.f_ref, "cylindrical")
    w = max(rows, key=lambda r: observed_report(
        __import__("tanitad.data.calib", fromlist=["FThetaIntrinsics"]).FThetaIntrinsics(
            poly=tuple(r["poly"]), cx=r["cx"], cy=r["cy"], width=r["w"], height=r["h"],
            per_clip=True), F)["masked_frac"])
    out[f"worst_{tag}"] = w
print(json.dumps(out, indent=1))
