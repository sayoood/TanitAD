"""P1c - THE LONGITUDINAL ORACLE-CHOOSER TABLE, with a REALISED column beside
every ORACLE column (M19 / retraction #29: an oracle row may not appear in the
sentence that approves anything).

Read on medAE over the GT-LON stratum (M15's binding ruling: NOT RMSE).
Zero GPU, no model: every row is a control profile rolled through the
programme's ONE unicycle integrator.
"""
import glob, json, os, sys, math
import numpy as np, torch
from tanitad.refs import refa_v1 as R
from tanitad.models.vocab_v7 import (TACTICAL_LAT_ACTIONS_V7,
                                     TACTICAL_LON_ACTIONS_V7)
from taniteval import four_families as ff
sys.path.insert(0, os.environ.get("LON_TOOLS", "C:/Users/Admin/tanitad-wt/taniteval/tools"))
from refav1_arm import paths_from_controls, _components

LAT_V70, LON_V70 = list(TACTICAL_LAT_ACTIONS_V7), list(TACTICAL_LON_ACTIONS_V7)
DT, K, OP = 0.2, 10, 30
P = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/Admin/refav1_margin/p4out/dump_wk15"

eps = sorted(glob.glob(os.path.join(P, "ep*.npz")))
dec = sorted(glob.glob(os.path.join(P, "decisions", "ep*.npz")))
G, V0, CL, HA0E, HA0, A0, LAT, LON, EID = [], [], [], [], [], [], [], [], []
for e, d in zip(eps, dec):
    with np.load(e) as z, np.load(d) as w:
        G.append(z["g"]); V0.append(z["v0"]); CL.append(z["cl"])
        HA0E.append(z["ha0_ext"]); HA0.append(z["ha0"])
        A0.append(w["ha0_ext_controls"][:, 0, 0])
        LAT.append(w["goal_lat_cl"]); LON.append(w["goal_lon_cl"])
        EID += [os.path.basename(e)] * int(z["v0"].shape[0])
G=np.concatenate(G); V0=np.concatenate(V0); CL=np.concatenate(CL)
HA0E=np.concatenate(HA0E); HA0=np.concatenate(HA0); A0=np.concatenate(A0)
LAT=np.concatenate(LAT); LON=np.concatenate(LON); EID=np.array(EID)
n = G.shape[0]
lat_s = [LAT_V70[i] for i in LAT]; lon_s = [LON_V70[i] for i in LON]

Gg = ff._seq_geometry(torch.as_tensor(G).float(), DT)
dv_gt = (Gg["speed"][:, -1].numpy() - V0)

# ---- the STRATUM, declared before any row is scored ----------------------- #
THR = 1.0
stratum = np.abs(dv_gt) >= THR
print("# dump %s  n=%d windows, %d episodes" % (P, n, len(eps)))
print("# STRATUM 'GT-LON' = |GT dv over the 2.0 s plan window| >= %.2f m/s"
      "  ->  n=%d / %d (%.1f %%)" % (THR, stratum.sum(), n, 100*stratum.mean()))
print("# sensitivity: >=0.50 -> n=%d ; >=1.50 -> n=%d"
      % ((np.abs(dv_gt) >= 0.5).sum(), (np.abs(dv_gt) >= 1.5).sum()))
print("# METRIC: medAE = MEDIAN over stratum windows of the per-window family")
print("#         metric (M15: RMSE is tail-dominated and ranks designs the")
print("#         other way). ADE median given for orientation only.")
print()

# ---- profile builders ---------------------------------------------------- #
def canon_paths(lon_of, a_sustain_of=None):
    """roll `canonical_controls(decoded lat, lon_of[i], v0_i)`; when
    `a_sustain_of` is given the LON channel is REPLACED by a CONSTANT a on the
    MAINTAIN branch only (v_t == v0), which is D1's proposed semantics."""
    out = np.zeros((n, K, 2), np.float32)
    for i in range(n):
        lon = lon_of[i]
        c = R.canonical_controls(lat_s[i], lon, float(V0[i]), OP, DT)[:K].clone()
        if a_sustain_of is not None:
            v0 = float(V0[i])
            v_t = (0.0 if lon == "HOLD" else R.GOAL_CREEP_MPS if lon == "CREEP"
                   else min(v0, R.GOAL_CURVE_VMAX_MPS) if lon == "ADAPT_SPEED_FOR_CURVE"
                   else max(0.0, v0 + R.GOAL_LON_DV_MPS.get(lon, 0.0)))
            if abs(v_t - v0) < 1e-9:                      # THE MAINTAIN BRANCH
                a = float(np.clip(a_sustain_of[i], -R.GOAL_A_MAX, R.GOAL_A_MAX))
                c[:, 0] = a
        out[i] = paths_from_controls(c, float(V0[i]), DT, K)[0].numpy()
    return out

def row(name, Pth, note=""):
    comp = _components(Pth.astype(np.float32), G.astype(np.float32), DT)
    def med(k): return float(np.median(comp[k][stratum]))
    return (name, med("LON_speed_mae_mps"), med("LON_accel_mae_mps2"),
            med("LON_along_mae_m"), med("ade_m"), note), comp

rows = []
rows.append(row("GT (identity control)", G, "must read EXACTLY 0"))
rows.append(row("cl  = wk15 REALISED plan", CL, "the arm"))
rows.append(row("ha0_ext (the floor, M11)", HA0E, "constant (a0,kappa0)"))
rows.append(row("ha0 (constant velocity)", HA0, "a=0,kappa=0"))
rows.append(row("canon @ DECODED token", canon_paths(lon_s), "REALISED vocab"))

# --- shipped-vocabulary ORACLE: best LON token per window, chosen on GT ----- #
allp = {t: canon_paths([t]*n) for t in LON_V70}
err = np.stack([np.abs(ff._seq_geometry(torch.as_tensor(allp[t]).float(), DT)["speed"]
                       - Gg["speed"]).mean(1).numpy() for t in LON_V70])   # [T,n]
best = err.argmin(0)
oracle_paths = np.stack([allp[LON_V70[best[i]]][i] for i in range(n)])
rows.append(row("canon @ ORACLE LON token", oracle_paths, "CEILING of shipped vocab"))

# --- D1: a_sustain = MEASURED a0. No level set, no chooser, no tuning ------ #
rows.append(row("D1 a_sustain = a0 MEASURED", canon_paths(lon_s, A0),
                "REALISED (hint is measured at t0)"))
rows.append(row("D1 ctl a_sustain = 0", canon_paths(lon_s, np.zeros(n)),
                "must equal 'canon @ DECODED'"))
rows.append(row("D1 regression a_sustain=-a0", canon_paths(lon_s, -A0),
                "deliberate regression: must be WORSE"))
# --- D1 + oracle token, i.e. the ceiling of the combined design ------------ #
bestp = {}
cand = np.stack([np.abs(ff._seq_geometry(torch.as_tensor(canon_paths([t]*n, A0)).float(), DT)["speed"]
                        - Gg["speed"]).mean(1).numpy() for t in LON_V70])
b2 = cand.argmin(0)
d1o = np.stack([canon_paths([LON_V70[b2[i]]]*n, A0)[i] for i in range(n)])
rows.append(row("D1 @ ORACLE LON token", d1o, "CEILING of D1"))

hdr = ("%-30s %11s %11s %11s %9s  %s"
       % ("row", "LONspd medAE", "LONacc medAE", "LONalong", "ADE med", "note"))
print("== TABLE G. LONGITUDINAL ORACLE vs REALISED, medAE on the GT-LON stratum ==")
print(hdr); print("-" * len(hdr))
for (nm, s, a, al, ad, note), _ in rows:
    print("%-30s %11.4f %11.4f %11.4f %9.4f  %s" % (nm, s, a, al, ad, note))
print()
print("== the same rows on ALL %d windows (no stratum) ==" % n)
print(hdr); print("-" * len(hdr))
for (nm, _, _, _, _, note), comp in rows:
    print("%-30s %11.4f %11.4f %11.4f %9.4f  %s"
          % (nm, np.median(comp["LON_speed_mae_mps"]), np.median(comp["LON_accel_mae_mps2"]),
             np.median(comp["LON_along_mae_m"]), np.median(comp["ade_m"]), note))
np.savez(os.path.join(os.path.dirname(P), "lon_oracle_arrays.npz"),
         A0=A0, V0=V0, dv_gt=dv_gt, stratum=stratum, eid=EID,
         lat=np.array(lat_s), lon=np.array(lon_s))
