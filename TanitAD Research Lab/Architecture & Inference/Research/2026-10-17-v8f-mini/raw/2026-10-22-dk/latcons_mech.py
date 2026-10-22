"""Why does T's own LATERAL constraint group (latcons) bring back early closing on the lead? Two candidate mechanisms, told apart on the
plans themselves (EXPLORATORY, zero GPU): (L) O drives FASTER with T's latcons -> more along-track progress by 2 s; (G) the plan's HEADING
differs -> the instrument's corridor frame (the predicted path's own local heading, lead_metrics) rotates and the projected gap changes.
Arms: eval_<c>-tacorc-all-latcons (T's latcons, GT elsewhere) minus eval_<c>-tacorc (whole GT code), LEAD windows of the K=10 block.
  python latcons_mech.py <pkg> <dk_perwindow-free: uses v8f_dk_read's join> <out.json>"""
import importlib.util, json, os, sys
from types import SimpleNamespace
import numpy as np
PKG, OUT = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("r", os.path.join(PKG, "code", "v8f_dk_read.py")); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
from taniteval import ci
RAW = os.path.join(PKG, "raw")
G = R.ground_truth(SimpleNamespace(gt_npz=None, eval_grid=os.path.join(RAW, "eval_grid_windows.json"), eval_kin=os.path.join(RAW, "eval141_kin.npz"), eval_join=os.path.join(RAW, "join_eval141.npz")))
blk = R.load_block(str(R.LEAD_BLOCK_DEFAULT))
J = R.join_block(blk, G["clip"], G["frame"], G["v0"], k=10)
lead, eid, gt = J["is_lead"], G["eid"], G["gt"]
def feats(p):
    p = np.asarray(p, np.float64)
    seg = p[:, 9] - p[:, 8]
    return {"along_2s_m": p[:, 9, 0], "abs_lat_2s_m": np.abs(p[:, 9, 1]), "heading_2s_deg": np.degrees(np.arctan2(seg[:, 1], seg[:, 0])),
            "speed_2s_mps": np.linalg.norm(seg, axis=-1) / 0.2, "lat_err_2s_m": np.abs(p[:, 9, 1] - gt[:, 9, 1]), "along_err_2s_m": p[:, 9, 0] - gt[:, 9, 0]}
out = {"exploratory": True, "tier": "T0 (both arms read GT code)", "n_lead": int(lead.sum()), "reads": {}}
for c in ("M2b_s0", "M2b_s1", "M1_s0", "M1_s1"):
    A = feats(np.load(os.path.join(RAW, "2026-10-21-tacgrp2", "eval", f"eval_{c}-tacorc-all-latcons.npz"))["pred"])
    B = feats(np.load(os.path.join(RAW, "2026-10-20-tac", "eval", f"eval_{c}-tacorc.npz"))["pred"])
    r = {}
    for k in A:
        q = ci.paired_episode_cluster_bootstrap(A[k][lead], B[k][lead], eid[lead], n_boot=2000, seed=0)
        r[k] = {"a": float(A[k][lead].mean()), "b": float(B[k][lead].mean()), "delta": q["delta"], "lo": q["lo"], "hi": q["hi"], "sep": bool(q["lo"] > 0 or q["hi"] < 0)}
    out["reads"][c] = r
json.dump(out, open(OUT, "w"), indent=1)
for c, r in out["reads"].items():
    print(c, " | ".join(f"{k} {v['delta']:+.3f} [{v['lo']:+.3f},{v['hi']:+.3f}]{'*' if v['sep'] else ''}" for k, v in r.items()))
