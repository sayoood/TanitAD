"""transfer_diag.py -- DESCRIPTIVE, post-hoc, 0 GPU (NOT a bar): does a forced DECEL label become a DECEL plan after the factored composition + stage?
On the K-HEAD union windows (eval139 S6 compliant GT-decel + GT-cruise, inference seed 0): P(plan class == decel | label decel / not decel), split by GT,
for LHS_FACT8 (LH argmax labels) and the LANDED FACTORED_DEPLOY8 (native argmax = SELFSHARP labels).  Plan class = the battery's s6_lon_decomp.plan_lon_class
(via l2_common.lon_indicators).  T1 open-loop planner read, descriptive.  NEW file."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_l2/code")
import k_head as K
import l2_common as C
K.assert_registered()
E = C.load_eval_pf(0); comp = np.asarray(E["compliant"], bool)
_fd, _md, gt, _ = C.lon_indicators(E["os"], E, comp)
keys = list(zip(E["ex_clip"][E["rows"]].tolist(), E["ex_ws"][E["rows"]].tolist()))
def lab_of(f):
    z = np.load(f); m = {(int(c), int(w)): int(v) for c, w, v in zip(z["clip_index"], z["ws"], z["lon_v7"])}
    return np.isin(np.array([m[k] for k in keys]), (3, 4, 5))
T = "step48852ema_pf"
arms = {"LHS_FACT8": (ROOT / f"fwd_lhs/out/arms/factored_scored/FACTORED_DEPLOY8/DEPLOY8_POS_{T}_s0.npz", ROOT / "labels/labels_LHS.npz"),
        "FACTORED_DEPLOY8": (Path(f"C:/Users/Admin/refcv8_pf/ship_r8_sub_fx/out/arms/factored_scored/FACTORED_DEPLOY8/DEPLOY8_POS_{T}_s0.npz"),
                             Path("C:/Users/Admin/refcv8_pf/refcv9_l2/selfsharp/labels_selfsharp_s0.npz"))}
base_pc = C.plan_class(np.asarray(np.load(f"C:/Users/Admin/refcv8_pf/ship_r8_sub_fx/out/arms/factored_scored/OSONLY_DEPLOY8/DEPLOY8_POS_{T}_s0.npz")["S6_pos"], np.float64), E["v0"], E["dt"])
out = {"tier": "T1 self-action OPEN LOOP on logged eval139 frames (not driving performance); descriptive, seed 0", "arms": {}}
for nm, (pf, lf) in arms.items():
    P = np.asarray(np.load(pf)["S6_pos"], np.float64); pc = C.plan_class(P, E["v0"], E["dt"]); ld = lab_of(lf)
    r = {}
    for g in ("decel", "cruise"):
        m = comp & (gt == g)
        for lv, lm in (("label_decel", ld), ("label_not_decel", ~ld)):
            mm = m & lm
            r[f"GT_{g}|{lv}"] = {"n": int(mm.sum()), "P_plan_decel": float((pc[mm] == "decel").mean()) if mm.any() else None,
                                 "P_base_os_plan_decel": float((base_pc[mm] == "decel").mean()) if mm.any() else None}
    out["arms"][nm] = r
(ROOT / "result" / "TRANSFER_DIAG.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
for nm, r in out["arms"].items():
    for k, v in r.items(): print(f"{nm:18s} {k:28s} n {v['n']:5d} P(plan decel) {v['P_plan_decel']:.4f}  base-os {v['P_base_os_plan_decel']:.4f}")
