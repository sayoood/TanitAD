"""THE NUMBER THAT DECIDES re-balance vs re-form: the NON-SELECTIVE tipping
weight, and whether it is a weight or a deletion.

⚠️ The "best constant-curvature candidate beats cv on 67.7 % of windows" figure is
a BEST-OF-10 selection on the same data it scores — and it is BELOW the 90.9 %
no-information rate, so it is an artefact, not evidence. The admissible version
fixes the candidate BEFORE looking at the cost: the sign comes from the DECODED
manoeuvre and the magnitude from GOAL_KAPPA_TURN. LANE_KEEP is the control: the
same computation there must read no advantage.

DELETION RULE (pre-registered by D-REFAV1-COST-REPAIR, applied verbatim):
`w` is a deletion in disguise iff  w * kappa_max^2  <  q(metric), the metric's
own realised float32 quantum (q(cos) = 2^-24 = 5.9605e-08).
"""
import glob, json, os, sys
import numpy as np

SC = r"C:\Users\Admin\AppData\Local\Temp\claude\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD\8e7cfa33-c625-47cf-88aa-711db80ac113\scratchpad"
D = r"C:\Users\Admin\tanitad-bench\refav1_eval_21109\full_dump"
REPO = r"C:\Users\Admin\tanitad-bench\refav1_eval_21109\repo"
sys.path.insert(0, os.path.join(REPO, "stack"))
sys.path.insert(0, os.path.join(REPO, "taniteval"))
from tanitad.models.v6 import tactical_lat_actions
from tanitad.refs import refa_v1 as R
from taniteval.ci import episode_cluster_bootstrap
LAT = list(tactical_lat_actions("v7.0"))
print(f"GOAL_KAPPA_TURN = {R.GOAL_KAPPA_TURN}   GOAL_TURN_S = {R.GOAL_TURN_S}   "
      f"W_KAPPA = {R.W_KAPPA}  W_JERK = {R.W_JERK}")

J = json.load(open(os.path.join(SC, "cost_anatomy_full.json")))
BN = J["box_names"]
rows = sorted(J["rows"], key=lambda r: (r["ep"], r["t"]))
A = lambda k: np.array([r[k] for r in rows], dtype=np.float64)
goal32, goal64, chord64, kap = A("goal32"), A("goal64"), A("chord64"), A("kap_raw")
N = goal64.shape[0]
KMAX = J["plan_cfg"]["kappa_max"]
ULP = 2.0 ** -24

gl, eid = [], []
for f in sorted(glob.glob(os.path.join(D, "decisions", "ep*.npz"))):
    z = np.load(f)
    gl.append(z["goal_lat_cl"])
    eid.append(np.full(z["goal_lat_cl"].shape[0], int(os.path.basename(f)[2:5])))
GL = np.concatenate(gl); EID = np.concatenate(eid)
lat = np.array([LAT[i] if 0 <= i < len(LAT) else "NONE" for i in GL])

i_cv = BN.index("cv")
MAG = "0.1"                      # nearest box magnitude to GOAL_KAPPA_TURN = 0.08
iL, iR = BN.index(f"kap+{MAG}"), BN.index(f"kap-{MAG}")
kbar = kap[0, iL]                # mean(kappa^2) for a constant |kappa| = 0.1
print(f"candidate fixed a priori: constant |kappa| = {MAG} "
      f"(mean kappa^2 = {kbar:.4g}); sign from the DECODED manoeuvre")
DEL_W = ULP / (KMAX ** 2)
print(f"deletion threshold: w < q/kappa_max^2 = {ULP:.4g}/{KMAX**2:.4g} = "
      f"{DEL_W:.4g}\n")

for lbl, G, q in (("SHIPPED  1-cos / float32", goal32, ULP),
                  ("         1-cos / float64", goal64, ULP),
                  ("         chord",           chord64, None)):
    print("="*96)
    print(lbl)
    print("="*96)
    for stratum, want in (("TURN_L", +1), ("TURN_R", -1), ("LANE_KEEP", 0)):
        m = lat == stratum
        if not m.sum():
            continue
        if want == 0:
            # CONTROL: average the two signs, which must read no advantage
            adv = 0.5 * (G[m, iL] + G[m, iR]) - G[m, i_cv]
        else:
            j = iL if want > 0 else iR
            adv = G[m, j] - G[m, i_cv]      # negative => the correct turn is CHEAPER
        w_tip = -adv / kbar                  # weight at which it ties cv
        r = episode_cluster_bootstrap(adv, EID[m], reduce="mean",
                                      n_boot=4000, seed=0, dp=10)
        sep = (r["lo"] > 0) or (r["hi"] < 0)
        print(f"  {stratum:10s} n={int(m.sum()):3d}/{len(set(EID[m])):3d}cl  "
              f"goal-term advantage of the CORRECT turn over cv: "
              f"{r['mean']:+.4g} [{r['lo']:+.4g}, {r['hi']:+.4g}] "
              f"{'SEPARATED' if sep else 'straddles 0'}")
        good = w_tip > 0
        if good.sum():
            wm = float(np.median(w_tip[good]))
            print(f"             the turn is CHEAPER on {int(good.sum())}/"
                  f"{int(m.sum())} windows; tipping W_KAPPA there: median "
                  f"{wm:.4g}  p10 {np.percentile(w_tip[good],10):.4g}  "
                  f"p90 {np.percentile(w_tip[good],90):.4g}")
            if q is not None:
                n_del = int((w_tip[good] * KMAX**2 < q).sum())
                print(f"             DELETION TEST: {n_del}/{int(good.sum())} of "
                      f"those tipping weights are DELETIONS "
                      f"(w*kappa_max^2 < {q:.3g});  median w is "
                      f"{wm/DEL_W:.4g}x the deletion threshold")
            print(f"             SHIPPED W_KAPPA {R.W_KAPPA} is {R.W_KAPPA/wm:.4g}x "
                  f"the median tipping weight")
        else:
            print(f"             the correct turn is cheaper on 0/"
                  f"{int(m.sum())} windows")
    print()

print("="*96)
print("SUMMARY ARITHMETIC (step 21,109, n=282 windows / 141 clusters)")
print("="*96)
m = (lat == "TURN_L") | (lat == "TURN_R")
sgn = np.where(lat[m] == "TURN_L", iL, iR)
adv = np.array([goal64[i, s] for i, s in zip(np.where(m)[0], sgn)]) - goal64[m, i_cv]
w = -adv / kbar
good = w > 0
print(f"manoeuvre stratum n={int(m.sum())} windows / {len(set(EID[m]))} clusters")
print(f"  correct turn cheaper on {int(good.sum())}/{int(m.sum())} = "
      f"{good.mean()*100:.1f}% of windows")
print(f"  median tipping W_KAPPA = {np.median(w[good]):.4g}")
print(f"  SHIPPED W_KAPPA        = {R.W_KAPPA}   ({R.W_KAPPA/np.median(w[good]):.4g}x too large)")
print(f"  deletion threshold     = {DEL_W:.4g}   (the tipping weight is "
      f"{np.median(w[good])/DEL_W:.4g}x ABOVE it)")
print(f"  => at step 21,109 a curvature weight that lets the correct turn win is "
      f"a WEIGHT, not a deletion." if np.median(w[good]) > DEL_W else
      "  => it IS a deletion.")
