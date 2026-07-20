import io
import contextlib

import pandas as pd
from tanitad.data.physicalai import intrinsics_for_clip
from tanitad.data.calib import F_REF, ftheta_feff_report

R = "/workspace/data/physicalai_phase0"
sel = pd.read_parquet(R + "/r0/r0_selection.parquet")
ok = True
for i in [0, 1500, 2999]:
    cid = str(sel.iloc[i]["clip_id"])
    ch = int(sel.iloc[i]["chunk"])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        intr = intrinsics_for_clip(cid, R)
    warned = "no per-clip intrinsics" in buf.getvalue()
    rep = ftheta_feff_report(intr)
    feff = rep["f_eff_after"]
    cx = getattr(intr, "cx", None)
    cy = getattr(intr, "cy", None)
    tag = "OK" if (abs(feff - F_REF) < 8.0 and not warned) else "ISSUE"
    if tag != "OK":
        ok = False
    print("clip={} chunk={} f_eff={:.2f} F_REF={} cx={} cy={} fallback={} -> {}".format(
        cid, ch, feff, F_REF, cx, cy, warned, tag))
print("fields:", [a for a in dir(intr) if not a.startswith("_")][:15])
print("PREFLIGHT_ALL_OK" if ok else "PREFLIGHT_HAS_ISSUES")
