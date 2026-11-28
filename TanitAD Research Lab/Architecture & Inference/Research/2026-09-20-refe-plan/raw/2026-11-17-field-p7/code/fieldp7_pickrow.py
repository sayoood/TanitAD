#!/usr/bin/env python3
"""FIELDP7: per-row L_field (w=1) of the rows of the weight rule's micro-batches (given by index list), so the reach probe can use ONE row whose fan is not already clean.
A batch-1 first-step probe on a row with L_field == 0.0 exactly (a clean fan, the draft's 'clean fan' identity) cannot show reach.  No parameter is trained; no selection on any metric."""
import json
import sys
from pathlib import Path

refe = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(refe))
import weight_rule as W

targv = sys.argv[2].split()
T, ns, model, ds, dev = W.build_instrumented(refe, targv, None)
import torch
from torch.utils.data import default_collate
for p in model.parameters():
    p.requires_grad_(False)
g = torch.Generator().manual_seed(W.SEED)
perm = torch.randperm(len(ds), generator=g).tolist()[: W.N_MICRO * ns.batch]
want = [int(x) for x in sys.argv[3].split(",")]          # positions in the perm
out = []
for pos in want:
    i = perm[pos]
    batch = default_collate([ds[i]])
    img, ego, goal, tgt, _a, _b, _c, _d, cal = batch[:9]
    sdf, valid = batch[-2], batch[-1]
    with torch.no_grad():
        traj, _ = model(img, ego, goal, calib=cal if cal.numel() else None)[:2]
        lf, parts = T.field_loss(traj.float(), sdf, margin=ns.field_margin, valid=valid, interp=ns.field_interp, return_parts=True)
    out.append({"perm_pos": pos, "row": i, "log": ds.rows[i].get("log_name"), "token": ds.rows[i].get("token"), "l_field": float(lf), "violating_frac": float(parts["violating_frac"])})
    print("PICK", json.dumps(out[-1]), flush=True)
Path(sys.argv[4]).write_text(json.dumps(out, indent=1))
