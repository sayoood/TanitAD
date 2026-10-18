#!/usr/bin/env python3
"""POST-HOC (not pre-registered) descriptives of R0d's own in-sample log: dispersion of the 25-update rows around the registered window mean.  -> stdout"""
import json, numpy as np
rows=[json.loads(l) for l in open("C:/Users/Admin/r0d_out/raw/sft.jsonl",encoding="utf-8") if '"event": "train"' in l[:40]]
def stat(name,sel):
    for k in ("con_g_ttc","con_p1_gap_ttc"):
        v=np.array([r[k] for r in rows if sel(r["update"])])*100
        print(f"{name:24s} {k:15s} n={len(v):2d} mean {v.mean():+.3f}  sd {v.std(ddof=1):.3f}  se {v.std(ddof=1)/np.sqrt(len(v)):.3f}  (x100 points)")
stat("window 651-750 (registered)",lambda u:651<=u<=750)
stat("post-ramp 226-750",lambda u:226<=u<=750)
stat("last window 901-1000",lambda u:901<=u<=1000)
stat("whole run 25-1000",lambda u:True)
