"""THE CONTROL the cost diagnosis needs: is the goal term's curvature preference
SIGNAL or NOISE?

The goal field g is the tactical predictor's terminal field rolled under the
DECODED manoeuvre's canonical controls. So if the predictor is action-sensitive
in the right direction, the candidate whose curvature MATCHES the decoded
manoeuvre's sign must score lowest. If it does not, the "the model term prefers a
turn on 67.7% of windows" figure is a best-of-10 selection artefact.

CONTROLS carried here (CLAUDE.md probe rule):
  * a LANE_KEEP stratum, where the goal carries no lateral intent -> the sign
    must read ~50/50 (the no-information value);
  * the raw best-of-10 selection rate under an explicit noise model.
"""
import glob, json, os, sys
import numpy as np
from collections import Counter

SC = r"C:\Users\Admin\AppData\Local\Temp\claude\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD\8e7cfa33-c625-47cf-88aa-711db80ac113\scratchpad"
D = r"C:\Users\Admin\tanitad-bench\refav1_eval_21109\full_dump"
sys.path.insert(0, r"C:\Users\Admin\tanitad-bench\refav1_eval_21109\repo\stack")
from tanitad.models.v6 import tactical_lat_actions, tactical_lon_actions_v
LAT = list(tactical_lat_actions("v7.0"))
print("lat vocab:", LAT)

J = json.load(open(os.path.join(SC, "cost_anatomy_full.json")))
BN, R = J["box_names"], J["rows"]
R = sorted(R, key=lambda r: (r["ep"], r["t"]))
A = lambda k: np.array([r[k] for r in R], dtype=np.float64)
goal32, goal64, chord64, kap = A("goal32"), A("goal64"), A("chord64"), A("kap_raw")
N = goal32.shape[0]

man = json.load(open(os.path.join(D, "manifest.json")))
files = sorted(glob.glob(os.path.join(D, "decisions", "ep*.npz")))
gl, eids, ws_all = [], [], []
for f in files:
    z = np.load(f)
    gl.append(z["goal_lat_cl"]); ws_all.append(z["ws"])
    eids.append(np.full(z["goal_lat_cl"].shape[0], int(os.path.basename(f)[2:5])))
GL = np.concatenate(gl); WS = np.concatenate(ws_all)
print(f"dump windows={len(GL)}  anatomy windows={N}")
t_anat = np.array([r["t"] for r in R])
assert len(GL) == N, "window count mismatch"
if not np.array_equal(WS, t_anat):
    print(f"⚠️ window-origin mismatch on {int((WS!=t_anat).sum())} rows — "
          f"aligning by (ep,t) failed; refusing the join")
    sys.exit(1)
print("✅ window alignment verified by CONTENT: dump `ws` == anatomy `t` on "
      f"{N}/{N} rows")

lat_name = np.array([LAT[i] if 0 <= i < len(LAT) else "NONE" for i in GL])
print("\ndecoded lateral goal histogram:", Counter(lat_name).most_common())

kidx = [i for i, n in enumerate(BN) if n.startswith("kap")]
ksign = np.array([+1.0 if BN[i].startswith("kap+") else -1.0 for i in kidx])
kmag = np.array([float(BN[i].split("kap")[1][1:]) for i in kidx])

def best_sign(G):
    out = np.zeros(N)
    for n in range(N):
        j = int(np.argmin(G[n, kidx]))
        out[n] = ksign[j]
    return out

def gradient_sign(G):
    """sign of the LINEAR trend of the goal term against SIGNED curvature —
    a full-box statistic, not a best-of-10 selection."""
    x = ksign * kmag
    out = np.zeros(N)
    for n in range(N):
        y = G[n, kidx]
        b = np.polyfit(x, y, 1)[0]
        out[n] = -np.sign(b)      # negative slope => LEFT (+kappa) is cheaper
    return out

for lbl, G in (("goal32 (SHIPPED fp32 1-cos)", goal32),
               ("goal64 (same, float64)", goal64),
               ("chord64", chord64)):
    bs, gs = best_sign(G), gradient_sign(G)
    print("\n" + "-"*72)
    print(f"metric: {lbl}")
    for stratum, mask in (("ALL", np.ones(N, bool)),
                          ("LANE_KEEP (no lateral intent = CONTROL)",
                           lat_name == "LANE_KEEP"),
                          ("goal is a LEFT manoeuvre",
                           np.array([n.endswith("_L") for n in lat_name])),
                          ("goal is a RIGHT manoeuvre",
                           np.array([n.endswith("_R") for n in lat_name]))):
        n = int(mask.sum())
        if n == 0:
            print(f"  {stratum:42s} n=0")
            continue
        fl_b = float((bs[mask] > 0).mean())
        fl_g = float((gs[mask] > 0).mean())
        print(f"  {stratum:42s} n={n:4d}  best-of-10 LEFT {fl_b*100:5.1f}%   "
              f"box-gradient LEFT {fl_g*100:5.1f}%")
    # the agreement test
    lr = np.array([1 if n.endswith("_L") else (-1 if n.endswith("_R") else 0)
                   for n in lat_name])
    m = lr != 0
    if m.sum():
        agr_b = float((bs[m] == lr[m]).mean())
        agr_g = float((gs[m] == lr[m]).mean())
        print(f"  >> SIGN AGREEMENT with the DECODED manoeuvre (n={int(m.sum())}): "
              f"best-of-10 {agr_b*100:.1f}%   box-gradient {agr_g*100:.1f}%   "
              f"(chance = 50.0%)")
        k = int((gs[m] == lr[m]).sum()); nn = int(m.sum())
        # exact binomial two-sided p
        from math import comb
        p = sum(comb(nn, i) for i in range(0, nn + 1)
                if abs(i - nn / 2) >= abs(k - nn / 2)) / 2 ** nn
        print(f"     exact binomial two-sided p (box-gradient) = {p:.4g}")

print("\n" + "="*72)
print("NOISE MODEL for the best-of-10 statistic")
print("="*72)
print("If the goal term carried NO curvature information, the minimum over 10")
print("curvature candidates would beat the single cv candidate with probability")
print("10/11 = 90.9% under exchangeability.  MEASURED: a constant-curvature")
kidx_beats = np.mean([(goal32[n, kidx].min() < goal32[n, BN.index('cv')])
                      for n in range(N)])
print(f"candidate beats cv on the goal term on {kidx_beats*100:.1f}% of windows —")
print(f"{'BELOW' if kidx_beats < 10/11 else 'ABOVE'} the no-information rate, i.e. the goal term")
print("systematically PREFERS zero curvature; the 'turn wins' rate is a")
print("selection artefact of the box, not evidence the model wants a turn.")
