"""DRY-RUN bank for l2_score.py: pseudo traces built from BANKED plans (seed-swapped), so the pipeline runs end to end. NOT results."""
import sys, json
import numpy as np
sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_l2/code")
import l2_common as C, l2_fit as F
from pathlib import Path
out = Path("C:/Users/Admin/refcv8_pf/refcv9_l2/dev30k/synth_bank"); out.mkdir(parents=True, exist_ok=True)
cal = C.jload("C:/Users/Admin/refcv8_pf/refcv9_l2/dev30k/calibrator_30k_DEV.json")
def ex(seed): return np.load(C.ARCH / C.EVAL_TREE / f"dump_s{seed}" / "refcv6_extras.npz")
def write(name, arm, plan_seed, seed_of_rows=0, n=None, app_kind="cal"):
    e = ex(plan_seed)
    clip, ws = e["clip_index"], e["ws"]
    raw = e["nav_true.tacv6_lon_logits"].astype(np.float32)
    plan = e["nav_true.traj"].astype(np.float32)
    if n: clip, ws, raw, plan = clip[:n], ws[:n], raw[:n], plan[:n]
    if app_kind == "cal": app = F.apply8(raw.astype(np.float64), b8=cal["b8"], T=cal["T"]).astype(np.float32)
    elif app_kind == "ident": app = raw.copy()
    else:
        app = np.full_like(raw, -20.0); app[:, 1] = 20.0
    meta = {"arm": arm, "final": True, "n_windows": int(len(ws)), "counters": {"hook_calls_armed": int(len(ws))}, "DRYRUN": True,
            "calibrator": {"file": "C:/Users/Admin/refcv8_pf/refcv9_l2/dev30k/calibrator_30k_DEV.json"}}
    np.savez_compressed(out / name, e_i=clip, t=ws - 7, ws=ws, raw_lon_logits=raw, applied_lon_logits=app, plan8=plan, sel=np.zeros(len(ws), np.int64), v0_in=np.zeros(len(ws)), meta_json=np.array(json.dumps(meta)))
write("trace_cal_s0.npz", "cal", 1)           # pseudo-CAL for seed 0 = banked seed-1 plans
write("trace_cal_s1.npz", "cal", 0)
write("trace_const_cruise_s0.npz", "const_cruise", 1, app_kind="const")
write("trace_native300_s0.npz", "native", 0, n=300, app_kind="ident")
write("trace_ident300_s0.npz", "ident", 0, n=300, app_kind="ident")
print("synthetic bank written")
