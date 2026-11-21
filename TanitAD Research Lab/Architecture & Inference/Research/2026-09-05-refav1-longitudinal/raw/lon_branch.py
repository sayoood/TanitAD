"""P1d - COUNT ON THE PREDICATE YOU MEAN (M28 2). 'ADAPT_SPEED_FOR_CURVE is
decoded' is NOT 'the goal commands a == 0': the token is a==0 only while
v0 <= GOAL_CURVE_VMAX_MPS. Zero GPU."""
import glob, os, sys, numpy as np, torch
from tanitad.refs import refa_v1 as R
from tanitad.models.vocab_v7 import TACTICAL_LAT_ACTIONS_V7, TACTICAL_LON_ACTIONS_V7
from taniteval import four_families as ff
LAT_V70, LON_V70 = list(TACTICAL_LAT_ACTIONS_V7), list(TACTICAL_LON_ACTIONS_V7)
DT, K, OP = 0.2, 10, 30
P = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/Admin/refav1_margin/p4out/dump_wk15"
G,V0,A0,LAT,LON,CLC = [],[],[],[],[],[]
for e,d in zip(sorted(glob.glob(P+"/ep*.npz")), sorted(glob.glob(P+"/decisions/ep*.npz"))):
    with np.load(e) as z, np.load(d) as w:
        G.append(z["g"]); V0.append(z["v0"]); A0.append(w["ha0_ext_controls"][:,0,0])
        LAT.append(w["goal_lat_cl"]); LON.append(w["goal_lon_cl"]); CLC.append(w["cl_controls"])
G=np.concatenate(G);V0=np.concatenate(V0);A0=np.concatenate(A0)
LAT=np.concatenate(LAT);LON=np.concatenate(LON);CLC=np.concatenate(CLC)
n=len(V0); lon_s=[LON_V70[i] for i in LON]; lat_s=[LAT_V70[i] for i in LAT]
dv_gt=(ff._seq_geometry(torch.as_tensor(G).float(),DT)["speed"][:,-1].numpy()-V0)

def v_t_of(lon, v0):
    if lon=="HOLD": return 0.0
    if lon=="CREEP": return R.GOAL_CREEP_MPS
    if lon=="ADAPT_SPEED_FOR_CURVE": return min(v0, R.GOAL_CURVE_VMAX_MPS)
    return max(0.0, v0 + R.GOAL_LON_DV_MPS.get(lon, 0.0))

print("== TABLE H. decoded LON token x what it COMMANDS at that window's own v0 ==")
print("%-24s %4s %8s %9s %9s %11s %11s" % ("decoded LON","n","med v0","n maintain","n a==0","med a_cmd0","med a0_meas"))
maint=np.zeros(n,bool); azero=np.zeros(n,bool)
for i in range(n):
    v0=float(V0[i]); vt=v_t_of(lon_s[i],v0)
    maint[i]= abs(vt-v0)<1e-9
    azero[i]= float(np.abs(R.canonical_controls(lat_s[i],lon_s[i],v0,OP,DT)[:K,0]).max())<1e-9
for t in sorted(set(lon_s)):
    m=np.array([x==t for x in lon_s])
    ac=np.array([float(R.canonical_controls(lat_s[i],t,float(V0[i]),OP,DT)[0,0]) for i in np.where(m)[0]])
    print("%-24s %4d %8.3f %9d %9d %11.4f %11.4f" % (t,m.sum(),np.median(V0[m]),
          maint[m].sum(), azero[m].sum(), np.median(ac), np.median(A0[m])))
print()
print("  MAINTAIN branch (v_t == v0, the branch D1 touches): %d / %d  (%.1f %%)"
      % (maint.sum(), n, 100*maint.mean()))
print("  goal commands a == 0 over the whole plan window     : %d / %d  (%.1f %%)"
      % (azero.sum(), n, 100*azero.mean()))
print("  ... and on the GT-LON stratum (|dv_gt| >= 1.0)      : %d / %d  (%.1f %%)"
      % ((azero & (np.abs(dv_gt)>=1.0)).sum(), (np.abs(dv_gt)>=1.0).sum(),
         100*(azero & (np.abs(dv_gt)>=1.0)).mean()/max(np.mean(np.abs(dv_gt)>=1.0),1e-9)))
print()
print("== TABLE I. REACHABILITY: is the GT demand inside the vocabulary's box? ==")
dv_reach = np.array([[float(R.canonical_controls(lat_s[i],t,float(V0[i]),OP,DT)[:K,0].sum()*DT)
                      for t in LON_V70] for i in range(n)])
lo, hi = dv_reach.min(1), dv_reach.max(1)
print("  GT dv ABOVE the vocabulary's most-positive reachable dv: %d / %d (%.1f %%)"
      % ((dv_gt>hi).sum(), n, 100*(dv_gt>hi).mean()))
print("  GT dv BELOW the vocabulary's most-negative reachable dv: %d / %d (%.1f %%)"
      % ((dv_gt<lo).sum(), n, 100*(dv_gt<lo).mean()))
print("  median |quantisation residual| to the NEAREST reachable dv: %.4f m/s"
      % np.median(np.abs(dv_reach - dv_gt[:,None]).min(1)))
print("  ... with D1 (a_sustain=a0 on the maintain branch) the maintain rows")
d1 = dv_reach.copy()
for i in range(n):
    for j,t in enumerate(LON_V70):
        if abs(v_t_of(t,float(V0[i]))-float(V0[i]))<1e-9:
            d1[i,j] = float(np.clip(A0[i],-R.GOAL_A_MAX,R.GOAL_A_MAX))*K*DT
print("      become a0*2s, and the residual falls to             : %.4f m/s"
      % np.median(np.abs(d1 - dv_gt[:,None]).min(1)))
print()
print("== TABLE J. the POSITIVE-side asymmetry (the analogue of kappa in {0,0.08}) ==")
print("  most-positive reachable dv over 2 s, median over windows: %+0.4f m/s" % np.median(hi))
print("  most-negative reachable dv over 2 s, median over windows: %+0.4f m/s" % np.median(lo))
print("  GT dv p90 = %+0.4f  p10 = %+0.4f   => positive demand exceeds supply by %.2fx"
      % (np.percentile(dv_gt,90), np.percentile(dv_gt,10),
         np.percentile(dv_gt,90)/max(np.median(hi),1e-9)))
print("  windows where GT dv > 0 : %d ; of those, UNREACHABLE : %d (%.1f %%)"
      % ((dv_gt>0).sum(), ((dv_gt>0)&(dv_gt>hi)).sum(),
         100*((dv_gt>0)&(dv_gt>hi)).sum()/max((dv_gt>0).sum(),1)))
