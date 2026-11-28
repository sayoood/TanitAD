#!/usr/bin/env python3
"""refav1_independent_probe.py -- a THIRD, from-scratch read of a refav1 dump.

Written from scratch against the raw npz files -- a different code path from
refav1_arm.py's own identity_probe/trivial_profile, so agreement between the two
is a genuine second probe (different PATH-BINDING), not the same query re-run.
"""
import glob, io, json, os, sys
import numpy as np

DUMP = sys.argv[1] if len(sys.argv) > 1 else "full_dump"
eps = sorted(glob.glob(os.path.join(DUMP, "ep*.npz")))
dec = {os.path.basename(p): p for p in sorted(glob.glob(os.path.join(DUMP, "decisions", "ep*.npz")))}
assert eps, "no episodes"

ARMS = ["cl", "ha", "ha0", "ol"]
acc = {a: [] for a in ARMS}
G, V0, EID, WS, CLIP = [], [], [], [], []
CTRL_CL, CTRL_HA = [], []
PSRC, PCOST, PAGREE, PNEVAL = [], [], [], []
NAV, NAVV, NAVS = [], [], []
LAB = {k: [] for k in ("lat_label", "lon_label", "route_label")}
PRED = {}
for h in ("lat", "lon", "route"):
    for c in ("nav_true", "nav_shuffled", "nav_zero"):
        PRED[f"{h}_{c}"] = []
WM = {k: [] for k in ("wm_mse_model", "wm_mse_const", "wm_mse_zero")}
WMSTD = []
EPIDX = []

for i, p in enumerate(eps):
    d = np.load(p)
    n = d["g"].shape[0]
    for a in ARMS:
        acc[a].append(d[a])
    G.append(d["g"]); V0.append(d["v0"]); WS.append(d["ws"])
    EID.append(np.repeat(d["eid"], n)); CLIP.append(np.repeat(d["clip_index"], n))
    EPIDX.append(np.full(n, i, dtype=np.int64))
    dp = np.load(dec[os.path.basename(p)])
    CTRL_CL.append(dp["cl_controls"]); CTRL_HA.append(dp["ha_controls"])
    PSRC.append(dp["plan_source_cl"]); PCOST.append(dp["plan_cost_cl"])
    PAGREE.append(dp["plan_agree_cl"]); PNEVAL.append(dp["plan_neval_cl"])
    NAV.append(dp["nav_cmd"]); NAVV.append(dp["nav_valid"]); NAVS.append(dp["nav_cmd_shuf"])
    for k in LAB: LAB[k].append(dp[k])
    for h in ("lat", "lon", "route"):
        for c in ("nav_true", "nav_shuffled", "nav_zero"):
            PRED[f"{h}_{c}"].append(dp[f"{h}_pred_{c}"])
    for k in WM: WM[k].append(dp[k])
    WMSTD.append(dp["wm_tgt_std"])

cat = lambda L: np.concatenate(L, axis=0)
A = {a: cat(acc[a]) for a in ARMS}
G = cat(G); V0 = cat(V0); WS = cat(WS); EID = cat(EID); CLIP = cat(CLIP); EPIDX = cat(EPIDX)
CTRL_CL = cat(CTRL_CL); CTRL_HA = cat(CTRL_HA)
PSRC = cat(PSRC); PCOST = cat(PCOST); PAGREE = cat(PAGREE); PNEVAL = cat(PNEVAL)
NAV = cat(NAV); NAVV = cat(NAVV); NAVS = cat(NAVS)
LAB = {k: cat(v) for k, v in LAB.items()}
PRED = {k: cat(v) for k, v in PRED.items()}
WM = {k: cat(v) for k, v in WM.items()}; WMSTD = cat(WMSTD)
N = G.shape[0]
print(f"N_WINDOWS = {N}   N_EPISODES = {len(eps)}   K = {G.shape[1]}")

out = {"n_windows": int(N), "n_episodes": len(eps)}

# ---------------------------------------------------------------- 1. IDENTITY
print("\n=== 1. BIT-IDENTITY (independent) ===")
ident = {}
for a in ARMS:
    for b in ARMS:
        if a >= b: continue
        exact = np.all(A[a] == A[b], axis=(1, 2))
        resid = np.abs(A[a] - A[b]).reshape(N, -1).max(axis=1)
        tol = resid < 1e-9
        ident[f"{a}_vs_{b}"] = dict(
            n_exact_equal=int(exact.sum()), frac_exact=float(exact.mean()),
            n_within_1e9=int(tol.sum()), frac_within_1e9=float(tol.mean()),
            max_resid_m=float(resid.max()), mean_resid_m=float(resid.mean()),
            median_resid_m=float(np.median(resid)))
        print(f"  {a:4s} vs {b:4s}: EXACT array-equal {exact.sum():4d}/{N} = {exact.mean():.4f}"
              f"   max|d| {resid.max():.4f} m   mean|d| {resid.mean():.4f} m")
out["identity"] = ident

# per-episode identity for cl vs ha0
d_cl_ha0 = np.all(A["cl"] == A["ha0"], axis=(1, 2))
byep = {}
for i in range(len(eps)):
    m = EPIDX == i
    byep[os.path.basename(eps[i])] = dict(n=int(m.sum()), n_identical=int(d_cl_ha0[m].sum()))
n_ep_all_ident = sum(1 for v in byep.values() if v["n_identical"] == v["n"])
print(f"  episodes where EVERY window is cl==ha0: {n_ep_all_ident}/{len(eps)}")
out["cl_vs_ha0_episodes_fully_identical"] = n_ep_all_ident

# ------------------------------------------------------- 2. SHAPE OF THE PLAN
print("\n=== 2. SHAPE OF THE EMITTED PLAN (cl_controls = (a, kappa)) ===")
a_ch, k_ch = CTRL_CL[:, :, 0], CTRL_CL[:, :, 1]
kzero = np.all(k_ch == 0.0, axis=1)
aconst = np.all(np.abs(a_ch - a_ch[:, :1]) < 1e-12, axis=1)
print(f"  kappa identically 0 on ALL 10 steps : {kzero.sum():4d}/{N} = {kzero.mean():.4f}")
print(f"  accel constant across all 10 steps  : {aconst.sum():4d}/{N} = {aconst.mean():.4f}")
print(f"  BOTH (straight + constant accel)    : {(kzero&aconst).sum():4d}/{N} = {(kzero&aconst).mean():.4f}")
vals, cnts = np.unique(np.round(a_ch[:, 0], 6), return_counts=True)
print(f"  distinct accel values emitted       : {dict(zip(vals.tolist(), cnts.tolist()))}")
print(f"  distinct kappa values emitted       : {np.unique(np.round(k_ch, 9)).tolist()}")
out["plan_shape"] = dict(frac_kappa_zero=float(kzero.mean()), frac_accel_const=float(aconst.mean()),
                         frac_straight_const_accel=float((kzero & aconst).mean()),
                         distinct_accels={str(v): int(c) for v, c in zip(vals.tolist(), cnts.tolist())},
                         distinct_kappas=[float(x) for x in np.unique(np.round(k_ch, 9))])

# hold-action controls for contrast
ka_ha = CTRL_HA[:, :, 1]
print(f"  [contrast] ha kappa identically 0   : {np.all(ka_ha==0,axis=1).sum():4d}/{N}")

# --------------------------------------------------------- 3. PLANNER SOURCE
names = ["cem", "baseline:cv", "baseline:hold_v0", "baseline:proposal", "baseline:decel_1.5"]
print("\n=== 3. WHERE THE PLAN CAME FROM ===")
sv, sc = np.unique(PSRC, return_counts=True)
src = {}
for v, c in zip(sv, sc):
    nm = names[v] if 0 <= v < len(names) else f"idx{v}"
    src[nm] = int(c)
    print(f"  {nm:22s} {c:4d}/{N} = {c/N:.4f}")
print(f"  mean plan cost {PCOST.mean():.6f}  std {PCOST.std():.6f}  distinct {len(np.unique(np.round(PCOST,6)))}")
print(f"  mean candidates evaluated {PNEVAL.mean():.1f}   coarse/fine agree {PAGREE.mean():.4f}")
out["plan_source"] = src
out["plan_cost"] = dict(mean=float(PCOST.mean()), std=float(PCOST.std()),
                        n_distinct=int(len(np.unique(np.round(PCOST, 6)))),
                        n_exactly_zero=int((PCOST == 0).sum()))
out["plan_neval_mean"] = float(PNEVAL.mean())

# ------------------------------------------------------------ 4. GT VARIETY
print("\n=== 4. IS THE GROUND TRUTH ITSELF TRIVIAL? (sanity: is 282 windows of straight road?) ===")
lat_extent = np.abs(G[:, :, 1]).max(axis=1)
chord = np.linalg.norm(np.diff(G, axis=1), axis=2)
spd = chord / 0.2
spd_rng = spd.max(axis=1) - spd.min(axis=1)
print(f"  GT |lateral| max over horizon: median {np.median(lat_extent):.3f} m  p90 {np.percentile(lat_extent,90):.3f} m  max {lat_extent.max():.3f} m")
print(f"  windows with GT |lat| > 0.5 m : {(lat_extent>0.5).sum()}/{N} = {(lat_extent>0.5).mean():.4f}")
print(f"  windows with GT |lat| > 2.0 m : {(lat_extent>2.0).sum()}/{N}")
print(f"  GT speed range within window  : median {np.median(spd_rng):.3f} m/s  p90 {np.percentile(spd_rng,90):.3f}")
print(f"  windows with GT speed range > 1 m/s: {(spd_rng>1.0).sum()}/{N}")
print(f"  v0 distribution: min {V0.min():.2f}  median {np.median(V0):.2f}  max {V0.max():.2f} m/s ; v0==0 on {(V0==0).sum()}")
out["gt_variety"] = dict(lat_med=float(np.median(lat_extent)), lat_p90=float(np.percentile(lat_extent, 90)),
                         lat_max=float(lat_extent.max()), n_lat_gt_0p5=int((lat_extent > 0.5).sum()),
                         n_lat_gt_2=int((lat_extent > 2.0).sum()),
                         spd_rng_med=float(np.median(spd_rng)), n_spd_rng_gt1=int((spd_rng > 1.0).sum()),
                         v0_min=float(V0.min()), v0_med=float(np.median(V0)), v0_max=float(V0.max()),
                         n_v0_zero=int((V0 == 0).sum()))

# --------------------------------------------------------- 5. DECISION HEADS
print("\n=== 5. DECISION HEADS (n and agreement, independent count) ===")
heads = {}
for h, lk in (("lat", "lat_label"), ("lon", "lon_label"), ("route", "route_label")):
    lab = LAB[lk]
    for c in ("nav_true", "nav_shuffled", "nav_zero"):
        pr = PRED[f"{h}_{c}"]
        m = (lab != -100)
        if h == "route":
            m = m & NAVV
        n = int(m.sum())
        acc_ = float((pr[m] == lab[m]).mean()) if n else float("nan")
        maj = 0.0
        if n:
            _, cc = np.unique(lab[m], return_counts=True); maj = float(cc.max() / n)
        heads[f"{h}_{c}"] = dict(n=n, acc=acc_, majority=maj,
                                 n_distinct_pred=int(len(np.unique(pr[m]))) if n else 0)
        print(f"  {h:6s} {c:13s} n={n:4d}  acc={acc_:.4f}  majority={maj:.4f}  distinct_preds={len(np.unique(pr[m])) if n else 0}")
out["heads"] = heads
out["nav_valid_frac"] = float(NAVV.mean())
print(f"  nav_valid on {NAVV.sum()}/{N} windows")

# --------------------------------------------------------------- 6. WM (T0)
print("\n=== 6. WORLD-MODEL T0 DIAGNOSTIC (feature MSE, frozen space) ===")
mm, mc, mz = WM["wm_mse_model"].mean(), WM["wm_mse_const"].mean(), WM["wm_mse_zero"].mean()
print(f"  model {mm:.5f}   persist-last {mc:.5f}   zero(=variance control) {mz:.5f}   tgt_std {WMSTD.mean():.5f}")
print(f"  model beats persist-last on {int((WM['wm_mse_model'].mean(1)<WM['wm_mse_const'].mean(1)).sum())}/{N} windows")
print(f"  model beats zero        on {int((WM['wm_mse_model'].mean(1)<WM['wm_mse_zero'].mean(1)).sum())}/{N} windows")
out["wm"] = dict(model=float(mm), const=float(mc), zero=float(mz), tgt_std=float(WMSTD.mean()),
                 n_beats_const=int((WM['wm_mse_model'].mean(1) < WM['wm_mse_const'].mean(1)).sum()),
                 n_beats_zero=int((WM['wm_mse_model'].mean(1) < WM['wm_mse_zero'].mean(1)).sum()),
                 per_step_model=[float(x) for x in WM["wm_mse_model"].mean(0)],
                 per_step_const=[float(x) for x in WM["wm_mse_const"].mean(0)],
                 per_step_zero=[float(x) for x in WM["wm_mse_zero"].mean(0)])

json.dump(out, io.open("indep_probe.json", "w", encoding="utf-8"), indent=1)
print("\nwrote indep_probe.json")
