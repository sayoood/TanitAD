"""sel_coupling_diag.py -- DESCRIPTIVE: does the lon force move the PATH by changing the SELECTED fan candidate?  S6 compliant, seed 0."""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ssguard_score as SG  # noqa: E402
C, S, G = SG.C, SG.S, SG.G
E = C.load_eval_pf(0)
b1 = S.load_l1_base("C:/Users/Admin/refcv8_pf/refcv9_l1/bank", 0)
tss = S.load_trace(Path("F:/refcv9_selfsharp") / "trace_selfsharp_s0.npz")
keys = list(zip(E["ex_clip"][E["rows"]].tolist(), E["ex_ws"][E["rows"]].tolist()))
sb = np.array([np.asarray(b1["sel"])[b1["idx"][k]] for k in keys]).reshape(len(keys), -1)[:, 0]
ss = np.array([np.asarray(tss["sel"])[tss["idx"][k]] for k in keys]).reshape(len(keys), -1)[:, 0]
comp = np.asarray(E["compliant"], bool)
ch = (sb != ss)
pb, ps = S.join_plan8(b1, E), S.join_plan8(tss, E)
cb, cs = S.arm_cells(E, pb)["stage"], S.arm_cells(E, ps)["stage"]
print("sel shape check", np.asarray(b1["sel"]).shape, np.asarray(tss["sel"]).shape)
print(f"compliant n {comp.sum()}; selected candidate CHANGED {int((ch & comp).sum())} ({(ch & comp).sum() / comp.sum():.3f})")
for k in ("LAT_heading_mae_deg", "LAT_cross_mae_m", "missed_decel", "false_decel"):
    d = cs[k] - cb[k]
    for nm, m in (("changed", ch & comp), ("same", ~ch & comp)):
        v = d[m]; v = v[np.isfinite(v)]
        print(f"{k:28s} {nm:8s} n {len(v):5d} mean delta {v.mean():+.4f}" if len(v) else f"{k} {nm} n 0")
# lateral-only shape change when the candidate is the SAME: max |lateral offset change| along the path
lat = np.abs(ps[..., 1] - pb[..., 1]).max(1)
print("max |dy| m: changed median %.3f, same median %.3f" % (np.median(lat[ch & comp]), np.median(lat[~ch & comp])))
