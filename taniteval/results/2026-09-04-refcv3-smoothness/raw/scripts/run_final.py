"""WP-5/6 CONSOLIDATED — the numbers of record. ZERO GPU, zero checkpoint.

GUARD: a window is admitted only where EVERY segment of EVERY compared path is
>= MIN_DS_MPS * dt (kinematic.MIN_DS_MPS = 0.5 m/s) — below that the path tangent,
and therefore curvature, carries no information (the same constant and the same
reason as `four_families.MIN_DS_MPS`). Without it a stopped window divides by ~0
and manufactures a 1e5 curvature-rate.
"""
import json, sys, numpy as np
sys.path.insert(0, r"C:/Users/Admin/_wp56"); sys.path.insert(0, r"C:/Users/Admin/_wp56/wp56")
import smooth_lib as S

MIN_DS_MPS = 0.5
RAW = r"C:/Users/Admin/_wp56/taniteval/results/2026-09-04-refcv3-closedloop/raw/"
ARMS = {"refcv3": "rollouts_refcv3_openloop.json",
        "refc-base": "rollouts_refc-base_openloop.json",
        "flagship-v1": "rollouts_flagship-v1_openloop.json"}
data = {a: json.load(open(RAW + f, encoding="utf-8")) for a, f in ARMS.items()}
poses = np.array([[g["x"], g["y"], g["yaw"]] for g in data["refcv3"]["gt"]])
bk = {a: {int(s["k"]): s for r in d["rollouts"] for s in r["steps"]}
      for a, d in data.items()}
ks_all = sorted(set(bk["refcv3"]) & set(bk["refc-base"]) & set(bk["flagship-v1"]))


def metrics(P, t, tag):
    g = S.geom(S.with_origin(P), t)
    turn = np.degrees(np.abs(g["turn"]))
    kap = np.abs(g["kappa"])
    Lm = 0.5 * (g["L"][..., :-1] + g["L"][..., 1:])
    dk = np.abs(np.diff(g["kappa"], axis=-1)) / np.maximum(
        0.5 * (Lm[..., :-1] + Lm[..., 1:]), 1e-6)
    flip_frac = S.sign_flip(g["turn"])
    return {"arm": tag, "n": int(P.shape[0]),
            "mean_abs_turn_deg": float(turn.mean()),
            "p90_abs_turn_deg": float(np.percentile(turn, 90)),
            "mean_abs_kappa_1pm": float(np.nanmean(kap)),
            "mean_abs_dkappa_ds_1pm2": float(np.nanmean(dk)),
            "mean_abs_jerk_mps3": float(np.abs(g["jerk"]).mean()),
            "p90_abs_jerk_mps3": float(np.percentile(np.abs(g["jerk"]), 90)),
            "sign_flip_frac": flip_frac,
            "mean_abs_a_mps2": float(np.abs(g["a"]).mean())}


def run(t, get, label, extra_paths=()):
    S_ = len(t) - 1
    ks = [k for k in ks_all if k + int(round(t[-1] / 0.1)) < len(poses)]
    paths, names = {}, []
    for a in ARMS:
        P = get(a, ks)
        if P is not None:
            paths[a] = P; names.append(a)
    G = np.stack([S.ego_future(poses, k, t)[0] for k in ks])
    v0 = np.array([bk["refcv3"][k]["v"] for k in ks])
    kap0 = np.array([S.gt_curvature0(poses, k) for k in ks])
    paths["GT"] = G
    paths["CONTROL constant-velocity"] = S.cv_path(v0, t)
    paths["CONTROL constant-curvature arc"] = S.arc_path(v0, kap0, t)
    # guard
    ok = np.ones(len(ks), bool)
    for nm, P in paths.items():
        if nm.startswith("CONTROL"):
            continue
        L = np.linalg.norm(np.diff(S.with_origin(P), axis=-2), axis=-1)
        ok &= (L >= MIN_DS_MPS * np.diff(t)[None]).all(1)
    print(f"\n{'='*104}\n{label}   admitted {ok.sum()}/{len(ks)} windows "
          f"(guard: every segment >= {MIN_DS_MPS} m/s)\n{'='*104}")
    print(f"{'path':<32s}{'mean|turn|':>11s}{'mean|k|':>10s}{'mean|dk/ds|':>13s}"
          f"{'mean|jerk|':>12s}{'p90|jerk|':>11s}{'SIGN-FLIP':>11s}")
    print(f"{'':<32s}{'deg':>11s}{'1/m':>10s}{'1/m^2':>13s}{'m/s^3':>12s}"
          f"{'m/s^3':>11s}{'frac':>11s}")
    res = {}
    for nm, P in paths.items():
        m = metrics(P[ok], t, nm); res[nm] = m
        print(f"{nm:<32s}{m['mean_abs_turn_deg']:11.3f}{m['mean_abs_kappa_1pm']:10.5f}"
              f"{m['mean_abs_dkappa_ds_1pm2']:13.6f}{m['mean_abs_jerk_mps3']:12.3f}"
              f"{m['p90_abs_jerk_mps3']:11.3f}{m['sign_flip_frac']:11.4f}")
    return res, int(ok.sum())


r8, n8 = run(S.T8,
             lambda a, ks: (np.array([bk[a][k]["extra"]["traj_full_6s"] for k in ks])
                            if a == "refcv3" else None),
             "A. THE 6 s / 8-SLOT GRID — refcv3's real emitted output vs GT and controls")
r4, n4 = run(S.T4, lambda a, ks: np.array([bk[a][k]["plan"] for k in ks]),
             "B. THE 2 s / 4-SLOT GRID — refcv3 vs refc-base vs flagship-v1, PAIRED, "
             "UNIFORM grid (no seam)")

print("\n  ratios vs GT (6 s grid):  refcv3 jerk "
      f"{r8['refcv3']['mean_abs_jerk_mps3']/r8['GT']['mean_abs_jerk_mps3']:.2f}x  "
      f"dk/ds {r8['refcv3']['mean_abs_dkappa_ds_1pm2']/r8['GT']['mean_abs_dkappa_ds_1pm2']:.2f}x  "
      f"sign-flip {r8['refcv3']['sign_flip_frac']/r8['GT']['sign_flip_frac']:.2f}x  "
      f"curvature {r8['refcv3']['mean_abs_kappa_1pm']/r8['GT']['mean_abs_kappa_1pm']:.2f}x")
print("  ratios vs GT (2 s grid):")
for a in ARMS:
    print(f"    {a:<14s} jerk {r4[a]['mean_abs_jerk_mps3']/r4['GT']['mean_abs_jerk_mps3']:5.2f}x  "
          f"dk/ds {r4[a]['mean_abs_dkappa_ds_1pm2']/r4['GT']['mean_abs_dkappa_ds_1pm2']:5.2f}x  "
          f"sign-flip {r4[a]['sign_flip_frac']/max(r4['GT']['sign_flip_frac'],1e-9):5.2f}x")
print(f"  refcv3 vs refc-base (2 s, paired): jerk "
      f"{r4['refcv3']['mean_abs_jerk_mps3']/r4['refc-base']['mean_abs_jerk_mps3']:.2f}x  "
      f"dk/ds {r4['refcv3']['mean_abs_dkappa_ds_1pm2']/r4['refc-base']['mean_abs_dkappa_ds_1pm2']:.2f}x  "
      f"sign-flip {r4['refcv3']['sign_flip_frac']/r4['refc-base']['sign_flip_frac']:.2f}x")

json.dump({"guard_min_ds_mps": MIN_DS_MPS,
           "source": "taniteval/results/2026-09-04-refcv3-closedloop/raw/rollouts_*_openloop.json",
           "A_6s_grid": {"n_windows": n8, "rows": r8},
           "B_2s_grid": {"n_windows": n4, "rows": r4}},
          open(r"C:/Users/Admin/_wp56/wp56/FINAL_smoothness.json", "w"), indent=1)
print("\nwrote FINAL_smoothness.json")
