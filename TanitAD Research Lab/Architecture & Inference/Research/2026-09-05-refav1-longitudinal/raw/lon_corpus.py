"""P1b - WHAT THE CORPUS ACTUALLY DOES longitudinally, on the SAME 40 windows.

Reuses `refav1_arm._components`' own geometry (`taniteval.four_families.
_seq_geometry`) - never a re-derivation. Zero GPU.
"""
import glob, json, os, sys, math
import numpy as np, torch
from tanitad.refs import refa_v1 as R
from taniteval import four_families as ff

P = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/Admin/refav1_margin/p4out/dump_wk15"
DT, K = 0.2, 10
LATN = list(R.TAC_LAT_ACTIONS) if hasattr(R, "TAC_LAT_ACTIONS") else None

eps = sorted(glob.glob(os.path.join(P, "ep*.npz")))
dec = sorted(glob.glob(os.path.join(P, "decisions", "ep*.npz")))
assert len(eps) == len(dec), (len(eps), len(dec))
G, V0, CL, HA0E, A0, LAT, LON, EID = [], [], [], [], [], [], [], []
for e, d in zip(eps, dec):
    with np.load(e) as z, np.load(d) as w:
        G.append(z["g"]); V0.append(z["v0"]); CL.append(z["cl"]); HA0E.append(z["ha0_ext"])
        A0.append(w["ha0_ext_controls"][:, 0, 0])
        LAT.append(w["goal_lat_cl"]); LON.append(w["goal_lon_cl"])
        EID += [os.path.basename(e)] * int(z["v0"].shape[0])
G = np.concatenate(G); V0 = np.concatenate(V0); CL = np.concatenate(CL)
HA0E = np.concatenate(HA0E); A0 = np.concatenate(A0)
LAT = np.concatenate(LAT); LON = np.concatenate(LON)
n = G.shape[0]
print("# dump: %s   windows n=%d  episodes=%d" % (P, n, len(eps)))

# --- token names, from the dump's own manifest (never guessed) ------------- #
man = json.load(open(os.path.join(P, "manifest.json")))
def _find(d, key, depth=0):
    if depth > 6 or not isinstance(d, dict): return None
    for k, v in d.items():
        if k == key: return v
        r = _find(v, key, depth + 1)
        if r is not None: return r
    return None
lat_names = _find(man, "goal_lat_names") or _find(man, "lat_names") or _find(man, "tac_lat_actions")
lon_names = _find(man, "goal_lon_names") or _find(man, "lon_names") or _find(man, "tac_lon_actions")
print("# lat_names from manifest:", lat_names)
print("# lon_names from manifest:", lon_names)

Gg = ff._seq_geometry(torch.as_tensor(G).float(), DT)
gt_v = Gg["speed"].numpy()                       # [n, K]
gt_a = Gg["accel"].numpy()
dv_gt = gt_v[:, -1] - V0                         # GT speed change over 2 s
print()
print("== TABLE D. The corpus's own longitudinal demand over the 2.0 s plan window ==")
def q(x, name):
    x = np.asarray(x, float)
    print("  %-22s n=%3d  min %+7.3f  p10 %+7.3f  p25 %+7.3f  med %+7.3f  "
          "p75 %+7.3f  p90 %+7.3f  max %+7.3f  mean|.| %6.3f"
          % (name, x.size, np.min(x), *np.percentile(x, [10, 25, 50, 75, 90]),
             np.max(x), np.abs(x).mean()))
q(V0, "v0 (m/s)")
q(A0, "a0 measured (m/s2)")
q(dv_gt, "GT dv over 2 s (m/s)")
q(gt_a.mean(1), "GT mean accel (m/s2)")
q(2.0 * A0, "ha0_ext dv over 2 s")

print()
print("== TABLE E. v0 stratification - does ADAPT_SPEED_FOR_CURVE really mean a==0? ==")
print("# `ADAPT_SPEED_FOR_CURVE` -> v_t = min(v0, GOAL_CURVE_VMAX_MPS=8.0), so it is")
print("# a==0 ONLY while v0 <= 8.0 m/s. Above that it is a SUSTAINED BRAKE to 8 m/s.")
print("  windows with v0 <= 8.0 : %d / %d  (%.1f %%)" % ((V0 <= 8.0).sum(), n, 100.0 * (V0 <= 8.0).mean()))
print("  windows with v0 >  8.0 : %d / %d  (%.1f %%)" % ((V0 > 8.0).sum(), n, 100.0 * (V0 > 8.0).mean()))

if lon_names:
    lon_s = np.array([lon_names[i] for i in LON])
    print()
    print("== TABLE F. decoded LON token x v0 stratum, and the a it commands ==")
    print("%-24s %5s %8s %8s %10s %10s" % ("decoded LON", "n", "med v0", "n(v0<=8)", "med a_cmd0", "med a0_meas"))
    for t in sorted(set(lon_s.tolist())):
        m = lon_s == t
        acmd = np.array([float(R.canonical_controls("LANE_KEEP", t, float(v), 30, DT)[0, 0])
                         for v in V0[m]])
        print("%-24s %5d %8.3f %8d %10.4f %10.4f"
              % (t, m.sum(), np.median(V0[m]), int((V0[m] <= 8.0).sum()),
                 np.median(acmd), np.median(A0[m])))
    print()
    print("  frac of windows whose decoded LON commands EXACTLY a==0 at its own v0: %.3f"
          % np.mean([abs(float(R.canonical_controls("LANE_KEEP", t, float(v), 30, DT)[:K, 0]).max()) < 1e-9
                     for t, v in zip(lon_s, V0)]))
np.savez(os.path.join(os.path.dirname(P), "lon_corpus_%s.npz" % os.path.basename(P)),
         G=G, V0=V0, CL=CL, HA0E=HA0E, A0=A0, LAT=LAT, LON=LON,
         gt_v=gt_v, gt_a=gt_a, dv_gt=dv_gt, eid=np.array(EID))
print("\n# banked arrays -> lon_corpus_%s.npz" % os.path.basename(P))
