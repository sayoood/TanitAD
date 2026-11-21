"""WP-6 refcv4 evidence, v2: the REPRESENTATIONAL CEILING of a unicycle-constrained
output head.  (v1 was REFUTED by its own control: a path that IS a unicycle rollout
fitted to 3.36 m instead of ~0, because the float32 `rollout_unicycle` was being
finite-differenced with a float64 step -- the Jacobian was pure noise.  Fixed by a
float64 numpy integrator VERIFIED line-for-line against `rollout_unicycle`.)

If refcv3 emitted (accel, curvature) integrated through the unicycle -- the Alpamayo
action space -- instead of 8 free 2-D waypoints, how close to the real GT path could
it get?  16 controls vs 16 free coordinates: the SAME parameter count.
"""
import glob, json, sys, numpy as np, torch
sys.path.insert(0, r"C:/Users/Admin/_wp56"); sys.path.insert(0, r"C:/Users/Admin/_wp56/wp56")
import smooth_lib as S
from scipy.optimize import least_squares
from tanitad.models.kinematic import rollout_unicycle
DT = 0.1


def roll64(v0, a, kp, seg):
    """float64 numpy twin of rollout_unicycle (SAME update order: x,y on the
    START speed; yaw next; v LAST).  No squash -- the fit is unbounded so the
    ceiling is not confounded with a saturation choice; feasibility of the
    RESULT is checked separately."""
    x = y = yaw = 0.0; v = float(v0); out = []
    for j, n in enumerate(seg):
        for _ in range(n):
            x += v * np.cos(yaw) * DT
            y += v * np.sin(yaw) * DT
            yaw += v * kp[j] * DT
            v = max(v + a[j] * DT, 0.0)
        out.append((x, y))
    return np.array(out)


# ---- CONTROL: roll64 must equal rollout_unicycle ---------------------------
seg_t = [5, 5, 5, 5, 10, 10, 10, 10]
a_t = np.array([0.3, -0.2, 0.1, 0.0, -0.4, 0.2, 0.0, 0.1])
k_t = np.array([0.01, -0.02, 0.005, 0.0, 0.03, -0.01, 0.0, 0.02])
mine = roll64(16.0, a_t, k_t, seg_t)
ctrl = torch.tensor(np.stack([np.repeat(a_t, seg_t), np.repeat(k_t, seg_t)], 1),
                    dtype=torch.float64)[None]
ref = rollout_unicycle(torch.tensor([[0., 0., 0., 16.]], dtype=torch.float64),
                       ctrl, dt=DT)[0].numpy()[np.cumsum(seg_t) - 1, :2]
print(f"CONTROL roll64 vs rollout_unicycle: max |diff| = "
      f"{np.abs(mine-ref).max():.3e} m  (must be ~0)")


def fit(gt, v0, t):
    seg = list(np.diff([0] + [int(round(x / DT)) for x in t[1:]]))
    K = len(seg)
    f = lambda p: (roll64(v0, p[:K], p[K:], seg) - gt).ravel()
    r = least_squares(f, np.zeros(2 * K), method="lm", max_nfev=20000,
                      xtol=1e-12, ftol=1e-12)
    return roll64(v0, r.x[:K], r.x[K:], seg), r.x


# ---- CONTROLS on the fitter ------------------------------------------------
for nm, P, v in (("ARC (is a unicycle rollout)", S.arc_path(np.array([16.]), np.array([.02]), S.T8)[0], 16.),
                 ("CV  (is a unicycle rollout)", S.cv_path(np.array([16.]), S.T8)[0], 16.)):
    p, _ = fit(P, v, S.T8)
    print(f"CONTROL fitter on {nm}: ADE {np.linalg.norm(p-P,axis=-1).mean():.3e} m  (must be ~0)")


def report(GT, V0, t, tag, n=None, other=None, othername=""):
    idx = (np.arange(len(GT)) if n is None
           else np.random.default_rng(0).choice(len(GT), min(n, len(GT)), False))
    err, sm = [], []
    for i in idx:
        p, _ = fit(GT[i], V0[i], t); sm.append(p)
        err.append(np.linalg.norm(p - GT[i], axis=-1).mean())
    sm = np.stack(sm); g = S.geom(S.with_origin(sm), t)
    gg = S.geom(S.with_origin(GT[idx]), t)
    print(f"\n== {tag}   n={len(idx)} ==")
    print(f"  BEST unicycle-constrained path : ADE {np.mean(err):7.4f} m  p50 "
          f"{np.median(err):7.4f}  p90 {np.percentile(err,90):7.4f}")
    if other is not None:
        e2 = np.linalg.norm(other[idx] - GT[idx], axis=-1).mean()
        print(f"  {othername:<31s}: ADE {e2:7.4f} m")
    print(f"  smoothness  fit: mean|jerk| {np.abs(g['jerk']).mean():6.3f}  "
          f"sign-flip {S.sign_flip(g['turn']):.4f}")
    print(f"  smoothness  GT : mean|jerk| {np.abs(gg['jerk']).mean():6.3f}  "
          f"sign-flip {S.sign_flip(gg['turn']):.4f}")
    return float(np.mean(err))


RAW = r"C:/Users/Admin/_wp56/taniteval/results/2026-09-04-refcv3-closedloop/raw/"
d = json.load(open(RAW + "rollouts_refcv3_openloop.json", encoding="utf-8"))
poses = np.array([[g["x"], g["y"], g["yaw"]] for g in d["gt"]])
st = {int(s["k"]): s for r in d["rollouts"] for s in r["steps"]}
ks = [k for k in sorted(st) if k + 60 < len(poses)]
G6 = np.stack([S.ego_future(poses, k, S.T8)[0] for k in ks])
v06 = np.array([st[k]["v"] for k in ks])
sel6 = np.array([st[k]["extra"]["traj_full_6s"] for k in ks])
report(G6, v06, S.T8, "6 s / 8 slots (GT-complete rollout windows)",
       other=sel6, othername="refcv3's OWN emitted path")

A = {k: [] for k in ("g", "v0", "os")}
for f in sorted(glob.glob(r"C:/Users/Admin/_wp56/dump/refcv3_40284_dump/ep*.npz")):
    dd = np.load(f)
    for k in A: A[k].append(dd[k])
A = {k: np.concatenate(v).astype(np.float64) for k, v in A.items()}
report(A["g"], A["v0"], S.T4, "2 s / 4 slots (banked 40,284 dump)", n=600,
       other=A["os"], othername="refcv3's OWN emitted path")
