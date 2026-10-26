"""Step 3e: tightness counterweight (compliance is gameable: an all-130 ceiling is 100 % compliant and useless).
slack = ceiling - human max [NOW, NOW+6] (km/h) on compliant windows; 'binds' = human within 10 km/h below the ceiling."""
import json, sys, numpy as np
sys.path.insert(0, "F:/refcv9_ceiling/code")
import l4_common as C
from l4_candidates import train_windows, eval_windows
from l4_c7a import prep, lift

out = {}
for tag, w, sp, vp in (("eval139", eval_windows(), C.SIDECAR_EVAL, C.V9_EVAL), ("train", train_windows(), C.SIDECAR_TRAIN, C.V9_TRAIN)):
    a0, c7 = prep(w, sp, vp)
    gk = w["gt"] * 3.6
    c0 = C.snap_up_kmh(w["vhi"])
    cands = {"C0_current": c0, "C3_drop30": C.snap_up_kmh(w["vhi"], (50.0, 100.0, 120.0)), "C6_clip_span_max": C.snap_up_kmh(w["cmax"]),
             "C7_N2_past_only": c7, "C7a_N2+a0>0.25_lift": lift(c7, a0 > 0.25),
             "C0_tol_+10kmh (scoring)": c0 + 10.0, "CONTROL_all_130 (must be ~100 % compliant, loosest)": np.full(len(gk), 130.0)}
    out[tag] = {}
    for k, c in cands.items():
        comp = gk <= c + 1e-6
        sl = (c - gk)[comp]
        out[tag][k] = {"compliance": round(float(comp.mean()), 4), "slack_kmh_p50": round(float(np.median(sl)), 2),
                       "slack_kmh_p90": round(float(np.percentile(sl, 90)), 2),
                       "binds_within_10kmh": round(float((comp & (c - gk <= 10)).mean()), 4)}
json.dump(out, open("F:/refcv9_ceiling/raw/l4_tightness.json", "w", encoding="utf-8"), indent=1)
for t, r in out.items():
    print("==", t)
    for k, v in r.items(): print(f"  {k:55s} {v}")
