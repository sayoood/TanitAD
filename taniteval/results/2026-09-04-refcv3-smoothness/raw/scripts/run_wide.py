"""WP-5: the SAME smoothness metrics at large n, on the in-repo banked dumps.
refcv3 : taniteval/results/refcv3-40284-openloop-dump.tar.gz (141 eps, 4823 windows)
refc-base: taniteval/results/fan_refc-base-30k.pt (40 eps, 881 canonical val windows)
Different window sets -> NOT paired with each other; each is paired with ITS OWN GT.
"""
import glob, sys, numpy as np, torch
sys.path.insert(0, r"C:/Users/Admin/_wp56"); sys.path.insert(0, r"C:/Users/Admin/_wp56/wp56")
import smooth_lib as S
MIN_DS = 0.5

def metrics(P, t):
    g = S.geom(S.with_origin(P), t)
    turn = np.degrees(np.abs(g["turn"])); kap = np.abs(g["kappa"])
    Lm = 0.5*(g["L"][..., :-1]+g["L"][..., 1:])
    dk = np.abs(np.diff(g["kappa"], axis=-1))/np.maximum(0.5*(Lm[..., :-1]+Lm[..., 1:]), 1e-6)
    fl = S.sign_flip(g["turn"])
    return dict(n=len(P), turn=float(turn.mean()), kappa=float(np.nanmean(kap)),
                dkds=float(np.nanmean(dk)), jerk=float(np.abs(g["jerk"]).mean()),
                flip=fl)

def show(rows, title, t):
    print(f"\n{'='*92}\n{title}\n{'='*92}")
    print(f"{'path':<40s}{'n':>7s}{'mean|turn|':>12s}{'mean|k|':>10s}"
          f"{'mean|dk/ds|':>13s}{'mean|jerk|':>11s}{'SIGN-FLIP':>11s}")
    for nm, P in rows.items():
        m = metrics(P, t)
        print(f"{nm:<40s}{m['n']:7d}{m['turn']:12.3f}{m['kappa']:10.5f}"
              f"{m['dkds']:13.6f}{m['jerk']:11.3f}{m['flip']:11.4f}")
    return {nm: metrics(P, t) for nm, P in rows.items()}

# ---------------- refcv3, 4823 windows, 2 s grid ---------------------------
A = {k: [] for k in ("os", "g", "ha", "ha0", "oracle_sel", "v0")}
for f in sorted(glob.glob(r"C:/Users/Admin/_wp56/dump/refcv3_40284_dump/ep*.npz")):
    d = np.load(f)
    for k in A: A[k].append(d[k])
A = {k: np.concatenate(v).astype(np.float64) for k, v in A.items()}
ok = np.ones(len(A["v0"]), bool)
for k in ("os", "g", "ha", "ha0"):
    L = np.linalg.norm(np.diff(S.with_origin(A[k]), axis=-2), axis=-1)
    ok &= (L >= MIN_DS*np.diff(S.T4)[None]).all(1)
print(f"refcv3 dump: {len(ok)} windows, admitted {ok.sum()} "
      f"({100*ok.mean():.1f} %) after the {MIN_DS} m/s guard")
r1 = show({"refcv3 SELECTED (os)": A["os"][ok], "GROUND TRUTH (g)": A["g"][ok],
           "CONTROL hold-action (ha)": A["ha"][ok],
           "CONTROL constant-velocity (ha0)": A["ha0"][ok],
           "refcv3 ORACLE-selected anchor": A["oracle_sel"][ok]},
          f"C. refcv3 @ 40,284 — 141 episodes, 2 s grid (the banked open-loop dump)", S.T4)

# ---------------- refc-base, 881 canonical val windows ---------------------
fb = torch.load(r"C:/Users/Admin/_wp56/taniteval/results/fan_refc-base-30k.pt",
                map_location="cpu", weights_only=False)
fan = fb["fan"].numpy().astype(np.float64); sel = fb["sel"].numpy()
selp = fan[np.arange(len(sel)), sel]; gtb = fb["gt"].numpy().astype(np.float64)
cvb = fb["cv"].numpy().astype(np.float64)
ok2 = np.ones(len(sel), bool)
for P in (selp, gtb, cvb):
    L = np.linalg.norm(np.diff(S.with_origin(P), axis=-2), axis=-1)
    ok2 &= (L >= MIN_DS*np.diff(S.T4)[None]).all(1)
print(f"\nrefc-base fan bank: {len(ok2)} windows, admitted {ok2.sum()} ({100*ok2.mean():.1f} %)")
r2 = show({"refc-base SELECTED": selp[ok2], "GROUND TRUTH": gtb[ok2],
           "CONTROL constant-velocity (banked cv)": cvb[ok2],
           "refc-base WHOLE FAN (128 candidates)": fan[ok2].reshape(-1, 4, 2)},
          "D. refc-base @ 29,999 — 40 canonical val episodes, 2 s grid", S.T4)
print(f"\n  refcv3/GT   jerk {r1['refcv3 SELECTED (os)']['jerk']/r1['GROUND TRUTH (g)']['jerk']:.2f}x"
      f"  sign-flip {r1['refcv3 SELECTED (os)']['flip']/r1['GROUND TRUTH (g)']['flip']:.2f}x"
      f"  dk/ds {r1['refcv3 SELECTED (os)']['dkds']/r1['GROUND TRUTH (g)']['dkds']:.2f}x")
print(f"  base/GT     jerk {r2['refc-base SELECTED']['jerk']/r2['GROUND TRUTH']['jerk']:.2f}x"
      f"  sign-flip {r2['refc-base SELECTED']['flip']/r2['GROUND TRUTH']['flip']:.2f}x"
      f"  dk/ds {r2['refc-base SELECTED']['dkds']/r2['GROUND TRUTH']['dkds']:.2f}x")
import json
json.dump({"C_refcv3_dump": r1, "D_refcbase_fanbank": r2},
          open(r"C:/Users/Admin/_wp56/wp56/WIDE_smoothness.json", "w"), indent=1)
