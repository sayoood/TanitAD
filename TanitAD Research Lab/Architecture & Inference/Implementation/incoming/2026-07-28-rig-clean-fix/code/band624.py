"""The vertical band at 624 columns — where the horizontal deficit is gone, so
the rig split is the ONLY thing left and the row statement is unambiguous."""
import json, sys, torch, pandas as pd
from pathlib import Path
from tanitad.data.calib import CanonicalFrame, cylindrical_grid
from tanitad.data.physicalai import intrinsics_for_clip
root = sys.argv[1]
P = CanonicalFrame(256, 640, 305.5774907364391, "cylindrical")
sel = pd.read_parquet(f"{root}/r0/r0_selection.parquet")
seen, out = set(), {"A": [], "B": []}
for cid in sel["clip_id"].astype(str):
    it = intrinsics_for_clip(cid, root)
    if not it.per_clip: continue
    k = (round(it.cx,4), round(it.cy,4), tuple(round(x,6) for x in it.poly))
    if k in seen: continue
    seen.add(k)
    _, m = cylindrical_grid(it, it.height, it.width, P)
    m624 = m[:, 8:632]                      # the 624-column window
    ok = m624.all(dim=1)
    idx = torch.nonzero(ok).flatten().tolist()
    first_bad = next((i for i in range(256) if not bool(ok[i])), None)
    out["B" if it.cy >= 650.872 else "A"].append(
        {"cy": it.cy, "n_rows": len(idx), "first_row": idx[0] if idx else None,
         "last_row": idx[-1] if idx else None, "first_bad": first_bad,
         "contiguous": bool(idx) and len(idx) == idx[-1]-idx[0]+1})
res = {rg: {"n_distinct": len(v),
            "all_contiguous": all(r["contiguous"] for r in v),
            "first_row_max": max(r["first_row"] for r in v),
            "last_row_min": min(r["last_row"] for r in v),
            "last_row_max": max(r["last_row"] for r in v),
            "n_rows_min": min(r["n_rows"] for r in v)} for rg, v in out.items()}
print(json.dumps(res, indent=1))
