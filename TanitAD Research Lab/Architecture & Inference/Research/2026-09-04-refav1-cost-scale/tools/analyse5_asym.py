"""Does the world-model term carry the GOAL's direction? A paired, magnitude-
matched asymmetry with episode-cluster bootstrap intervals.

⭐ WHY NOT THE SIGN TEST. The sign of the box gradient is coarse and its headline
(81.6 % agreement) is EXACTLY what an always-RIGHT constant predictor scores on
this stratum. The instrument here is a within-window CONTRAST between two
candidates of equal |kappa| and opposite sign, so the field's common-mode drops
out, plus a LANE_KEEP stratum that measures the field's INTRINSIC asymmetry —
the constant control that says what "no goal information" reads.

Estimator: episode-cluster bootstrap (`taniteval.ci`). NEVER overlapping_holdout_se.
"""
import glob, json, os, sys
import numpy as np

SC = r"C:\Users\Admin\AppData\Local\Temp\claude\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD\8e7cfa33-c625-47cf-88aa-711db80ac113\scratchpad"
D = r"C:\Users\Admin\tanitad-bench\refav1_eval_21109\full_dump"
REPO = r"C:\Users\Admin\tanitad-bench\refav1_eval_21109\repo"
sys.path.insert(0, os.path.join(REPO, "stack"))
sys.path.insert(0, os.path.join(REPO, "taniteval"))
from tanitad.models.v6 import tactical_lat_actions, tactical_lon_actions_v
from taniteval.ci import episode_cluster_bootstrap
LAT = list(tactical_lat_actions("v7.0")); LON = list(tactical_lon_actions_v("v7.0"))
print("lon vocab:", LON)

J = json.load(open(os.path.join(SC, "cost_anatomy_full.json")))
BN = J["box_names"]
R = sorted(J["rows"], key=lambda r: (r["ep"], r["t"]))
A = lambda k: np.array([r[k] for r in R], dtype=np.float64)
goal64, chord64, goal32 = A("goal64"), A("chord64"), A("goal32")
N = goal64.shape[0]

gl, gn, eid = [], [], []
for f in sorted(glob.glob(os.path.join(D, "decisions", "ep*.npz"))):
    z = np.load(f)
    gl.append(z["goal_lat_cl"]); gn.append(z["goal_lon_cl"])
    eid.append(np.full(z["goal_lat_cl"].shape[0], int(os.path.basename(f)[2:5])))
GL, GN = np.concatenate(gl), np.concatenate(gn)
EID = np.concatenate(eid)
lat = np.array([LAT[i] if 0 <= i < len(LAT) else "NONE" for i in GL])
lon = np.array([LON[i] if 0 <= i < len(LON) else "NONE" for i in GN])
print("lon histogram:", {k: int((lon == k).sum()) for k in sorted(set(lon))})

def ci(v, e, lbl, pad=44):
    r = episode_cluster_bootstrap(v, e, reduce="mean", n_boot=4000, seed=0, dp=8)
    lo, hi = r["lo"], r["hi"]
    sep = (lo > 0) or (hi < 0)
    print(f"  {lbl:{pad}s} n={len(v):4d} clusters={len(set(e)):3d}  "
          f"mean {r['mean']:+.5f}  [{lo:+.5f}, {hi:+.5f}] "
          f"{'SEPARATED from 0' if sep else 'straddles 0'}")
    return r

def asym(G, plus, minus):
    a, b = G[:, BN.index(plus)], G[:, BN.index(minus)]
    return (b - a) / np.maximum(a + b, 1e-300)     # >0 => `plus` is cheaper

print("\n" + "="*84)
print("LATERAL — normalised L/R contrast of the goal term, per |kappa|")
print("  >0 means the model term prefers LEFT (+kappa).  Strata by DECODED goal.")
print("="*84)
for mag in ("0.2", "0.1", "0.05", "0.02", "0.01"):
    for lbl, G in (("chord", chord64), ("1-cos f64", goal64)):
        v = asym(G, f"kap+{mag}", f"kap-{mag}")
        print(f"\n|kappa|={mag}  metric={lbl}")
        for s in ("LANE_KEEP", "TURN_L", "TURN_R"):
            m = lat == s
            if m.sum():
                ci(v[m], EID[m], s)
    break                      # the contrast is scale-free; one magnitude shown
print("\n(the contrast is a ratio, so it is near-identical across |kappa| and "
      "across the two monotone metrics — one magnitude is shown; the full sweep "
      "is in the JSON)")

print("\n" + "="*84)
print("THE DECISIVE COMPARISON: manoeuvre stratum vs its OWN LANE_KEEP control")
print("="*84)
v = asym(chord64, "kap+0.1", "kap-0.1")
lk = v[lat == "LANE_KEEP"]; lke = EID[lat == "LANE_KEEP"]
r_lk = episode_cluster_bootstrap(lk, lke, reduce="mean", n_boot=4000, seed=0, dp=8)
print(f"  CONTROL  LANE_KEEP           : {r_lk['mean']:+.5f} "
      f"[{r_lk['lo']:+.5f}, {r_lk['hi']:+.5f}]  <- the field's INTRINSIC "
      f"asymmetry ({'RIGHT' if r_lk['mean']<0 else 'LEFT'}-biased)")
rng = np.random.default_rng(0)
for s, want in (("TURN_L", +1), ("TURN_R", -1)):
    m = lat == s
    if not m.sum():
        continue
    r_s = episode_cluster_bootstrap(v[m], EID[m], reduce="mean", n_boot=4000, seed=0, dp=8)
    # difference of two independent cluster bootstraps
    def draw(vals, e, n):
        u = np.unique(e); idx = {k: np.where(e == k)[0] for k in u}
        out = np.empty(n)
        for i in range(n):
            pick = rng.choice(u, size=len(u), replace=True)
            sel = np.concatenate([idx[k] for k in pick])
            out[i] = vals[sel].mean()
        return out
    d = draw(v[m], EID[m], 4000) - draw(lk, lke, 4000)
    lo, hi = np.percentile(d, [2.5, 97.5])
    sep = (lo > 0) or (hi < 0)
    print(f"  {s:8s} {r_s['mean']:+.5f} [{r_s['lo']:+.5f}, {r_s['hi']:+.5f}]"
          f"   LIFT vs control {d.mean():+.5f} [{lo:+.5f}, {hi:+.5f}] "
          f"{'SEPARATED' if sep else 'straddles 0'}"
          f"   (expected sign {'+' if want>0 else '-'})")

print("\n" + "="*84)
print("LONGITUDINAL — normalised accel/brake contrast (>0 prefers ACCELERATE)")
print("="*84)
v = asym(chord64, "acc+0.5", "acc-0.5")
for s in sorted(set(lon)):
    m = lon == s
    if m.sum() >= 3:
        ci(v[m], EID[m], f"decoded lon = {s}")
v2 = asym(chord64, "acc+2.0", "acc-2.0")
print("\n  same contrast at |a| = 2.0:")
for s in sorted(set(lon)):
    m = lon == s
    if m.sum() >= 3:
        ci(v2[m], EID[m], f"decoded lon = {s}")
