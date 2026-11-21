"""The base-rate correction to the sign control.

⚠️ 81.6% "sign agreement" is EXACTLY what an ALWAYS-RIGHT constant predictor
scores on this stratum (31 of 38 manoeuvre windows are TURN_R). The headline
number is therefore uninformative; only the COMPOSITION against the LANE_KEEP
base rate is. This is the constant-control rule from CLAUDE.md applied to a
sign test.
"""
import glob, json, os, sys
import numpy as np
from math import comb
from collections import Counter

SC = r"C:\Users\Admin\AppData\Local\Temp\claude\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD\8e7cfa33-c625-47cf-88aa-711db80ac113\scratchpad"
D = r"C:\Users\Admin\tanitad-bench\refav1_eval_21109\full_dump"
sys.path.insert(0, r"C:\Users\Admin\tanitad-bench\refav1_eval_21109\repo\stack")
from tanitad.models.v6 import tactical_lat_actions
LAT = list(tactical_lat_actions("v7.0"))

J = json.load(open(os.path.join(SC, "cost_anatomy_full.json")))
BN = J["box_names"]
R = sorted(J["rows"], key=lambda r: (r["ep"], r["t"]))
A = lambda k: np.array([r[k] for r in R], dtype=np.float64)
goal32, goal64 = A("goal32"), A("goal64")
N = goal32.shape[0]

gl, eid = [], []
for f in sorted(glob.glob(os.path.join(D, "decisions", "ep*.npz"))):
    z = np.load(f)
    gl.append(z["goal_lat_cl"])
    eid.append(np.full(z["goal_lat_cl"].shape[0], int(os.path.basename(f)[2:5])))
GL = np.concatenate(gl); EID = np.concatenate(eid)
lat = np.array([LAT[i] if 0 <= i < len(LAT) else "NONE" for i in GL])

kidx = [i for i, n in enumerate(BN) if n.startswith("kap")]
ksign = np.array([+1.0 if BN[i].startswith("kap+") else -1.0 for i in kidx])
kmag = np.array([float(BN[i][4:]) for i in kidx])
x = ksign * kmag

def grad_left(G):
    return np.array([-np.sign(np.polyfit(x, G[n, kidx], 1)[0]) for n in range(N)])

def fisher(a, b, c, d):
    """two-sided Fisher exact on [[a,b],[c,d]]"""
    n = a + b + c + d
    r1, c1 = a + b, a + c
    def p(k):
        return (comb(r1, k) * comb(n - r1, c1 - k) / comb(n, c1)
                if 0 <= k <= r1 and 0 <= c1 - k <= n - r1 else 0.0)
    p0 = p(a)
    return sum(p(k) for k in range(0, r1 + 1) if p(k) <= p0 * (1 + 1e-12))

for lbl, G in (("goal32 (SHIPPED fp32)", goal32), ("goal64", goal64)):
    gl_ = grad_left(G)
    print("\n" + "="*76)
    print(f"{lbl} — box-gradient preference along signed curvature")
    print("="*76)
    tab = {}
    for s in ("LANE_KEEP", "TURN_L", "TURN_R"):
        m = lat == s
        tab[s] = (int((gl_[m] > 0).sum()), int(m.sum()))
        print(f"  {s:10s} n={int(m.sum()):4d}   prefers LEFT (+kappa): "
              f"{int((gl_[m]>0).sum()):4d} = {(gl_[m]>0).mean()*100:5.1f}%")
    lk_l, lk_n = tab["LANE_KEEP"]
    base = lk_l / lk_n
    print(f"\n  ⭐ CONSTANT CONTROL: the LANE_KEEP base rate is {base*100:.1f}% LEFT, "
          f"NOT 50% —")
    print(f"     the terminal field has a systematic RIGHT-curvature preference. "
          f"An 'always RIGHT'")
    print(f"     predictor scores {(1-base)*100:.1f}% on LANE_KEEP and "
          f"{tab['TURN_R'][1]}/{tab['TURN_L'][1]+tab['TURN_R'][1]} = "
          f"{tab['TURN_R'][1]/(tab['TURN_L'][1]+tab['TURN_R'][1])*100:.1f}% on the "
          f"manoeuvre stratum,")
    print(f"     which is why the raw 'sign agreement' headline is uninformative.")
    for s in ("TURN_L", "TURN_R"):
        a, n_ = tab[s]
        pv = fisher(a, n_ - a, lk_l, lk_n - lk_l)
        lift = a / n_ - base
        print(f"\n  {s} vs LANE_KEEP: {a}/{n_} = {a/n_*100:.1f}% LEFT vs base "
              f"{base*100:.1f}%   lift {lift*100:+.1f} pp   Fisher two-sided "
              f"p = {pv:.4g}")
        if s == "TURN_L":
            print(f"     P(all {n_} LEFT | base rate) = {base**n_:.3g}")
    # episode-cluster robustness for TURN_L (are the 7 windows independent?)
    m = lat == "TURN_L"
    print(f"\n  TURN_L windows sit in {len(set(EID[m]))} distinct episode clusters "
          f"(n_windows={int(m.sum())}) -> effective n is the CLUSTER count")
    m = lat == "TURN_R"
    print(f"  TURN_R windows sit in {len(set(EID[m]))} distinct episode clusters "
          f"(n_windows={int(m.sum())})")

print("\n" + "="*76)
print("WHY fp32 AND fp64 DISAGREE ON THE 'ALL' ROW")
print("="*76)
ties = np.array([(goal32[n, kidx] == goal32[n, kidx].min()).sum() for n in range(N)])
print(f"fp32 argmin over the 10 curvature candidates is TIED on "
      f"{np.mean(ties>1)*100:.1f}% of windows (median tie multiplicity "
      f"{int(np.median(ties[ties>1]))} of 10); fp64 ties on "
      f"{np.mean([(goal64[n,kidx]==goal64[n,kidx].min()).sum()>1 for n in range(N)])*100:.1f}%.")
print("np.argmin returns the FIRST tied index, and the box lists kap+ before "
      "kap-, so under fp32 a tie silently resolves to LEFT. That is the whole of "
      "the fp32-vs-fp64 gap on the ALL row (55.3% vs 25.2% LEFT) — an "
      "ORDERING ARTEFACT of the float32 subtraction, not a difference of opinion.")
