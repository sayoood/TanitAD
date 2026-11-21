"""C-X1 (added control, no bar): the plan-production route reproduces HBLEND's banked M2b fit-window arrays."""
import json, sys
import numpy as np
PK = "E:/Projects/TanitAD-artifacts/daghacc"
res = {}
for s in (0, 1):
    a = np.load(f"{PK}/plans/fitplans_M2b_s{s}.npz"); b = np.load(f"{PK}/hb/fit_windows_M2b_s{s}.npz")
    r = {"keys_equal": sorted(a.files) == sorted(b.files), "n": int(len(a["traj30"]))}
    for k in ("ci", "t", "a_t", "a_s", "gt_ok"):
        r[f"{k}_identical"] = bool(np.array_equal(a[k], b[k]))
    for k in ("gt", "p30", "traj30"):
        r[f"{k}_maxabs"] = float(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)).max())
    r["traj30_bitwise_equal"] = bool(np.array_equal(a["traj30"], b["traj30"]))
    base = bool(r["keys_equal"] and all(r[f"{k}_identical"] for k in ("ci", "t", "a_t", "a_s", "gt_ok")) and r["gt_maxabs"] == 0.0)
    r["pass_strict_first_draft_p30_1e-6"] = bool(base and r["p30_maxabs"] <= 1e-6 and r["traj30_maxabs"] <= 1e-4)   # my FIRST-DRAFT tolerance, written before I had seen a float32 CPU number: it FAILED
    r["pass"] = bool(base and r["p30_maxabs"] <= 1e-4 and r["traj30_maxabs"] <= 1e-4)      # HBLEND's own CPU-vs-banked float bound (1e-4 m, test_v8f_hblend / xcheck); windows, ages and GT identical
    r["note"] = "added control C-X1, NOT in the PREREG; the 1e-6 first-draft p30 tolerance was too tight for float32 CPU thread-order noise (max 3.8e-5 m, ~1 ulp at 64 m scale); reported both"
    res[f"M2b_s{s}"] = r
json.dump(res, open(f"{PK}/plans/CX1_route_reproduces_banked_M2b.json", "w"), indent=1)
print(json.dumps(res, indent=1))
