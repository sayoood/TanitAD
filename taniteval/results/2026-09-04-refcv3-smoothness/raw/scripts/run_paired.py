"""WP-5/6: paired smoothness of refcv3 / refc-base / flagship-v1 vs GT + controls.

ZERO GPU. Substrate: the banked open-loop rollout JSONs from the 2026-09-04
closed-loop floor campaign (the same forwards the PI's video was rendered from).
"""
import json, sys, numpy as np
sys.path.insert(0, r"C:/Users/Admin/_wp56/wp56")
import smooth_lib as S

RAW = r"C:/Users/Admin/_wp56/taniteval/results/2026-09-04-refcv3-closedloop/raw/"
ARMS = {"refcv3": "rollouts_refcv3_openloop.json",
        "refc-base": "rollouts_refc-base_openloop.json",
        "flagship-v1": "rollouts_flagship-v1_openloop.json"}

data = {a: json.load(open(RAW + f, encoding="utf-8")) for a, f in ARMS.items()}
gt = data["refcv3"]["gt"]
poses = np.array([[g["x"], g["y"], g["yaw"]] for g in gt])
NP = len(poses)

def steps_by_k(d):
    out = {}
    for r in d["rollouts"]:
        for s in r["steps"]:
            out[int(s["k"])] = s
    return out

bk = {a: steps_by_k(d) for a, d in data.items()}
ks = sorted(set(bk["refcv3"]) & set(bk["refc-base"]) & set(bk["flagship-v1"]))
print(f"windows present in all three arms: {len(ks)}  (k {ks[0]}..{ks[-1]})")

# GT admission: a window needs the full horizon of LOGGED poses.
k8 = [k for k in ks if k + 60 < NP]
k4 = [k for k in ks if k + 20 < NP]
print(f"GT-complete windows: 6 s grid n={len(k8)}   2 s grid n={len(k4)}")

# ------------------------------------------------------------------ build --
def build8():
    rows = {}
    v0 = np.array([bk["refcv3"][k]["v"] for k in k8])
    kap0 = np.array([S.gt_curvature0(poses, k) for k in k8])
    rows["refcv3(8slot)"] = np.array(
        [bk["refcv3"][k]["extra"]["traj_full_6s"] for k in k8])
    g = np.zeros((len(k8), 8, 2))
    for i, k in enumerate(k8):
        g[i], ok = S.ego_future(poses, k, S.T8)
        assert ok.all()
    rows["GT(8slot)"] = g
    rows["CV(8slot)"] = S.cv_path(v0, S.T8)
    rows["ARC-gt-kappa(8slot)"] = S.arc_path(v0, kap0, S.T8)
    return rows, v0, kap0

def build4():
    rows = {}
    v0 = np.array([bk["refcv3"][k]["v"] for k in k4])
    kap0 = np.array([S.gt_curvature0(poses, k) for k in k4])
    for a in ARMS:
        rows[f"{a}(4slot)"] = np.array([bk[a][k]["plan"] for k in k4])
    g = np.zeros((len(k4), 4, 2))
    for i, k in enumerate(k4):
        g[i], ok = S.ego_future(poses, k, S.T4)
        assert ok.all()
    rows["GT(4slot)"] = g
    rows["CV(4slot)"] = S.cv_path(v0, S.T4)
    rows["ARC-gt-kappa(4slot)"] = S.arc_path(v0, kap0, S.T4)
    return rows, v0, kap0

def table(rows, t, title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    hdr = (f"{'path':<24s} {'|turn| deg':>26s} {'|kappa| 1/m':>22s} "
           f"{'|dkappa/ds| 1/m^2':>20s} {'|jerk| m/s^3':>18s}")
    print(hdr)
    print(f"{'':<24s} {'mean   p90    max':>26s} {'mean     p90':>22s} "
          f"{'mean      p90':>20s} {'mean     p90':>18s}")
    res = {}
    for name, P in rows.items():
        g = S.geom(S.with_origin(P), t)
        tn = np.degrees(np.abs(g["turn"]))
        kp = np.abs(g["kappa"])
        Lm = 0.5 * (g["L"][..., :-1] + g["L"][..., 1:])
        dk = np.abs(np.diff(g["kappa"], axis=-1)) / np.maximum(
            0.5 * (Lm[..., :-1] + Lm[..., 1:]), 1e-6)
        jk = np.abs(g["jerk"])
        res[name] = dict(turn=S.q(tn, "turn_deg"), kappa=S.q(kp, "kappa"),
                         dkds=S.q(dk, "dkappa_ds"), jerk=S.q(jk, "jerk"),
                         turn_per_vertex=[float(np.nanmean(tn[:, j]))
                                          for j in range(tn.shape[1])],
                         kappa_per_vertex=[float(np.nanmean(kp[:, j]))
                                           for j in range(kp.shape[1])])
        r = res[name]
        print(f"{name:<24s} {r['turn']['mean']:8.3f} {r['turn']['p90']:7.3f} "
              f"{r['turn']['max']:7.2f} {r['kappa']['mean']:12.5f} "
              f"{r['kappa']['p90']:9.5f} {r['dkds']['mean']:12.6f} "
              f"{r['dkds']['p90']:7.6f} {r['jerk']['mean']:10.3f} "
              f"{r['jerk']['p90']:7.3f}")
    return res

r8, v0_8, kap0_8 = build8()
res8 = table(r8, S.T8, "A. THE 6 s / 8-SLOT GRID (refcv3's real output) "
                       f"— n={len(k8)} paired windows")
print("\n  per-VERTEX mean |turn| (deg)   vertices at t = 0.5 1.0 1.5 [2.0=SEAM] 3.0 4.0 5.0 s")
for n, r in res8.items():
    print(f"   {n:<24s}", " ".join(f"{x:7.3f}" for x in r["turn_per_vertex"]))
print("\n  per-VERTEX mean |kappa| (1/m)  — dt-INVARIANT: a smooth path is FLAT here")
for n, r in res8.items():
    print(f"   {n:<24s}", " ".join(f"{x:8.5f}" for x in r["kappa_per_vertex"]))

r4, v0_4, kap0_4 = build4()
res4 = table(r4, S.T4, "B. THE 2 s / 4-SLOT GRID — refcv3 vs refc-base vs "
                       f"flagship-v1, PAIRED, n={len(k4)} windows")
print("\n  per-VERTEX mean |turn| (deg)   vertices at t = 0.5 1.0 1.5 s (UNIFORM grid, no seam)")
for n, r in res4.items():
    print(f"   {n:<24s}", " ".join(f"{x:7.3f}" for x in r["turn_per_vertex"]))

out = {"n_windows_8slot": len(k8), "n_windows_4slot": len(k4),
       "k8": k8, "k4": k4,
       "v0_mean": float(v0_8.mean()), "grid_8": list(S.T8), "grid_4": list(S.T4),
       "A_6s_grid": res8, "B_2s_grid": res4}
json.dump(out, open(r"C:/Users/Admin/_wp56/wp56/paired_smoothness.json", "w"),
          indent=1)
print("\nwrote paired_smoothness.json")
