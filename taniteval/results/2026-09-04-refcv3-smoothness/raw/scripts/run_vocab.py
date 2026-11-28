"""WP-6: IS THE ANCHOR VOCABULARY THE PROBLEM?  synthetic vs data-driven FPS.

refcv3 trains on `default_anchors` = FPS over 4096 SYNTHETIC constant-(yaw_rate,
accel) unicycle rollouts, because `refcv3_b1_launch.sh` passes no --anchors and
NO 6 s vocabulary file exists in the repo (every banked table is horizons
[5,10,15,20]).  refc-base/XL/small all trained on `refc_anchors_*.pt`, built by
`build_refc_anchors.py --data-root` = FPS over REAL ego-frame GT trajectories.

This measures the ORACLE-IN-VOCABULARY floor of each construction: the best ADE
any selector could possibly achieve. Split-half, so the data-FPS arm is scored
OUT OF SAMPLE.
"""
import glob, sys, numpy as np, torch
sys.path.insert(0, r"C:/Users/Admin/_wp56"); sys.path.insert(0, r"C:/Users/Admin/_wp56/wp56")
import smooth_lib as S
from tanitad.refs.refc import default_anchors, furthest_point_sample

# --- real GT 2 s targets, 4,823 windows, from the banked 40284 open-loop dump -
G, V0 = [], []
for f in sorted(glob.glob(r"C:/Users/Admin/_wp56/dump/refcv3_40284_dump/ep*.npz")):
    d = np.load(f); G.append(d["g"]); V0.append(d["v0"])
G = torch.tensor(np.concatenate(G), dtype=torch.float32)      # [N, 4, 2]
V0 = torch.tensor(np.concatenate(V0), dtype=torch.float32)
print(f"real GT windows: {G.shape}  v0 mean {V0.mean():.2f}")

def best_in_vocab(anchors, gt):
    """min_i mean_s ||a_i,s - gt_s||  -> [N] the oracle-in-vocabulary ADE."""
    d = (anchors[None] - gt[:, None]).norm(dim=-1).mean(-1)    # [N, K]
    return d.min(1).values

rng = np.random.default_rng(0)
perm = rng.permutation(len(G)); half = len(G) // 2
tr, te = torch.tensor(perm[:half]), torch.tensor(perm[half:])

A_syn = default_anchors((5, 10, 15, 20), 128, 4096, 0)
A_dat = furthest_point_sample(G[tr], 128, seed=0)              # data-driven FPS
cv = S.cv_path(V0.numpy(), S.T4)
CV = torch.tensor(cv, dtype=torch.float32)

print("\n== ORACLE-IN-VOCABULARY ADE over 0-2 s (metres, lower = better) — "
      f"scored on the HELD-OUT half, n={len(te)} ==")
rows = [("SYNTHETIC 128  (refcv3's actual vocabulary)", best_in_vocab(A_syn, G[te])),
        ("DATA-FPS   128 (refc-base's construction)  ", best_in_vocab(A_dat, G[te])),
        ("CONTROL: constant-velocity single path     ", (CV[te] - G[te]).norm(dim=-1).mean(-1)),
        ("CONTROL: zero path (stand still)           ", G[te].norm(dim=-1).mean(-1))]
for nm, v in rows:
    print(f"  {nm}  mean {v.mean():7.4f}  p50 {v.median():7.4f}  "
          f"p90 {np.percentile(v.numpy(),90):7.4f}  max {v.max():7.3f}")
syn, dat = rows[0][1], rows[1][1]
print(f"\n  ==> the SYNTHETIC vocabulary's oracle floor is "
      f"{(syn.mean()/dat.mean()):.2f}x the data-driven one "
      f"(+{(syn.mean()-dat.mean()):.4f} m), PAIRED on the same {len(te)} windows; "
      f"synthetic worse on {(syn>dat).float().mean()*100:.1f} % of them")

# ---- and the same question at 6 s, on the 130 GT-complete rollout windows ---
import json
RAW = r"C:/Users/Admin/_wp56/taniteval/results/2026-09-04-refcv3-closedloop/raw/"
d = json.load(open(RAW + "rollouts_refcv3_openloop.json", encoding="utf-8"))
poses = np.array([[g["x"], g["y"], g["yaw"]] for g in d["gt"]])
steps = {int(s["k"]): s for r in d["rollouts"] for s in r["steps"]}
ks = [k for k in sorted(steps) if k + 60 < len(poses)]
G6 = torch.tensor(np.stack([S.ego_future(poses, k, S.T8)[0] for k in ks]), dtype=torch.float32)
v06 = torch.tensor([steps[k]["v"] for k in ks], dtype=torch.float32)
A_syn6 = default_anchors((5, 10, 15, 20, 30, 40, 50, 60), 128, 4096, 0)
CV6 = torch.tensor(S.cv_path(v06.numpy(), S.T8), dtype=torch.float32)
sel6 = torch.tensor(np.array([steps[k]["extra"]["traj_full_6s"] for k in ks]), dtype=torch.float32)
print(f"\n== the SAME question at 6 s (n={len(ks)} GT-complete windows, one clip) ==")
print(f"  SYNTHETIC 128 oracle-in-vocab ADE 0-6 s : {best_in_vocab(A_syn6, G6).mean():7.4f} m")
print(f"  CONTROL constant-velocity               : {(CV6-G6).norm(dim=-1).mean():7.4f} m")
print(f"  the model's SELECTED path (for scale)   : {(sel6-G6).norm(dim=-1).mean():7.4f} m")
print(f"  CONTROL zero path                       : {G6.norm(dim=-1).mean():7.4f} m")
