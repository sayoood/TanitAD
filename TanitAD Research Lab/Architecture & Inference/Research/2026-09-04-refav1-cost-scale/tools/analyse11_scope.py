"""HOW BIG IS THE DEFECT, HONESTLY? — the banked plan cross-tabbed against the
DECODED goal, on all 282 windows.

⚠️ "the plan is a trivial baseline on 282/282" is TRUE and is not the same as
"the plan is WRONG on 282/282". On a LANE_KEEP + CRUISE goal the canonical
controls ARE zero, so the imagined goal field IS the zero-action rollout and
`cv` is the CORRECT answer. The defect is confined to the windows where the goal
asks for something else — and that is the number a repair should be sized against.
"""
import glob, os, sys
import numpy as np
from collections import Counter
D = r"C:\Users\Admin\tanitad-bench\refav1_eval_21109\full_dump"
REPO = r"C:\Users\Admin\tanitad-bench\refav1_eval_21109\repo"
sys.path.insert(0, os.path.join(REPO, "stack"))
from tanitad.models.v6 import tactical_lat_actions, tactical_lon_actions_v
from tanitad.refs.refa_v1 import canonical_controls
LAT = list(tactical_lat_actions("v7.0")); LON = list(tactical_lon_actions_v("v7.0"))

C, GL, GN, EID, V0 = [], [], [], [], []
for f in sorted(glob.glob(os.path.join(D, "decisions", "ep*.npz"))):
    z = np.load(f); ep = int(os.path.basename(f)[2:5])
    C.append(z["cl_controls"]); GL.append(z["goal_lat_cl"]); GN.append(z["goal_lon_cl"])
    EID.append(np.full(len(z["goal_lat_cl"]), ep))
for f in sorted(glob.glob(os.path.join(D, "ep*.npz"))):
    z = np.load(f); V0.append(z["v0"])
C = np.concatenate(C); GL = np.concatenate(GL); GN = np.concatenate(GN)
EID = np.concatenate(EID); V0 = np.concatenate(V0)
N = len(C)
lat = np.array([LAT[i] if 0 <= i < len(LAT) else "NONE" for i in GL])
lon = np.array([LON[i] if 0 <= i < len(LON) else "NONE" for i in GN])
H = C.shape[1]
ZER = np.zeros((H, 2), np.float32); DEC = ZER.copy(); DEC[:, 0] = -1.5
is_z = np.all(C == ZER, axis=(1, 2)); is_d = np.all(C == DEC, axis=(1, 2))
print(f"n = {N} windows / {len(set(EID))} clusters;  plan == zeros {is_z.sum()}, "
      f"== decel_1.5 {is_d.sum()}, other {int((~(is_z|is_d)).sum())}")

print("\n" + "="*94)
print("WHAT DID THE PLANNER'S OWN IMAGINED GOAL ASK FOR?")
print("="*94)
print(f"{'lat x lon':46s} {'n':>5s} {'plan=zeros':>11s} {'plan=decel':>11s}")
tab = Counter(zip(lat, lon))
for (a, b), n in tab.most_common():
    m = (lat == a) & (lon == b)
    print(f"{a + ' x ' + b:46s} {n:5d} {int(is_z[m].sum()):11d} "
          f"{int(is_d[m].sum()):11d}")

print("\n" + "="*94)
print("THE CANONICAL CONTROLS OF EACH DECODED GOAL — is `cv` the RIGHT answer?")
print("="*94)
print("(`canonical_controls(lat, lon, v0, 10, 0.2)` is what `plan()` itself "
      "builds the goal from)")
print(f"{'lat x lon':46s} {'n':>5s} {'canon kappa==0':>15s} {'canon a==0':>12s} "
      f"{'canon == ZERO':>14s}")
n_goal_trivial = 0
for (a, b), n in tab.most_common():
    m = (lat == a) & (lon == b)
    kz = az = zz = 0
    for i in np.where(m)[0]:
        cc = canonical_controls(a, b, float(V0[i]), H, 0.2).numpy()
        kz += int(np.all(cc[:, 1] == 0)); az += int(np.all(cc[:, 0] == 0))
        zz += int(np.all(cc == 0))
    n_goal_trivial += zz
    print(f"{a + ' x ' + b:46s} {n:5d} {kz:15d} {az:12d} {zz:14d}")

print("\n" + "="*94)
print("⭐ THE HONEST SCOPE OF THE DEFECT")
print("="*94)
kap_nonzero = np.zeros(N, bool); a_nonzero = np.zeros(N, bool)
for i in range(N):
    cc = canonical_controls(lat[i], lon[i], float(V0[i]), H, 0.2).numpy()
    kap_nonzero[i] = not np.all(cc[:, 1] == 0)
    a_nonzero[i] = not np.all(cc[:, 0] == 0)
print(f"  the decoded goal wants NON-ZERO CURVATURE on   "
      f"{int(kap_nonzero.sum()):3d}/{N} = {kap_nonzero.mean()*100:.1f} % of windows")
print(f"  the decoded goal wants NON-ZERO ACCELERATION on "
      f"{int(a_nonzero.sum()):3d}/{N} = {a_nonzero.mean()*100:.1f} % of windows")
print(f"  the decoded goal is EXACTLY the zero control on "
      f"{int(n_goal_trivial):3d}/{N} = {n_goal_trivial/N*100:.1f} % of windows")
print()
print(f"  ⇒ on the {int(n_goal_trivial)} zero-goal windows `cv` is the CORRECT "
      f"plan and the cost is behaving")
print(f"    correctly. `kappa == 0 on 282/282` is a true statement about the "
      f"OUTPUT; the")
print(f"    number a repair must be sized against is the "
      f"{int(kap_nonzero.sum())} curvature-wanting windows")
print(f"    ({kap_nonzero.mean()*100:.1f} %), where the planner emitted "
      f"kappa == 0 on {int((kap_nonzero & (C[:,:,1]==0).all(1)).sum())}"
      f"/{int(kap_nonzero.sum())} = 100.0 %.")
print()
# did the planner get the LONGITUDINAL half right where the goal asked to brake?
brake = np.array([b in ("BRAKE_TO",) for b in lon])
print(f"  LONGITUDINAL, the one axis where the planner is NOT flat:")
print(f"    goal = BRAKE_TO on {int(brake.sum())} windows; the planner emitted "
      f"the decel baseline on {int(is_d[brake].sum())} of them")
print(f"    ({is_d[brake].mean()*100:.1f} %), and on "
      f"{int(is_d[~brake].sum())}/{int((~brake).sum())} "
      f"({is_d[~brake].mean()*100:.1f} %) of the windows where the goal did NOT "
      f"ask to brake.")
from math import comb
k, n1, K, n0 = int(is_d[brake].sum()), int(brake.sum()), int(is_d.sum()), N
p = sum(comb(n1, i) * comb(n0 - n1, K - i) for i in range(k, min(n1, K) + 1)) / comb(n0, K)
print(f"    one-sided hypergeometric p (the 12 decels landing on BRAKE_TO "
      f"windows by chance) = {p:.4g}")
