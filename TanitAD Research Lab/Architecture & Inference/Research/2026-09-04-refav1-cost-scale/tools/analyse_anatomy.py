"""The units arithmetic: what the refav1 planner cost is, term by term."""
import json, math, sys
import numpy as np

P = r"C:\Users\Admin\AppData\Local\Temp\claude\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD\8e7cfa33-c625-47cf-88aa-711db80ac113\scratchpad\cost_anatomy_full.json"
J = json.load(open(P))
BN = J["box_names"]
R = J["rows"]
W_J, W_K = J["weights_shipped"]["W_JERK"], J["weights_shipped"]["W_KAPPA"]
print(f"n_windows={len(R)}  box={len(BN)} candidates  W_JERK={W_J} W_KAPPA={W_K}")
print("box:", BN)

def A(key):
    return np.array([r[key] for r in R], dtype=np.float64)   # [N, box]

cost32 = A("cost32"); goal32 = A("goal32"); goal64 = A("goal64")
cos64 = A("cos64"); chord64 = A("chord64"); jerk = A("jerk_raw"); kap = A("kap_raw")
xn = A("xnorm"); dev_rel = A("dev_rel"); dcf = A("dc_frac"); cvc = A("cos_vs_cv")
yn = np.array([r["ynorm"] for r in R])
N, B = cost32.shape

print("\n" + "="*78)
print("1.  THE FIELD GEOMETRY  (what the cosine is taken between)")
print("="*78)
print(f"||zt|| (terminal tactical field, per candidate): "
      f"median {np.median(xn):.4g}  min {xn.min():.4g}  max {xn.max():.4g}")
print(f"||g||  (imagined goal field):                    "
      f"median {np.median(yn):.4g}")
print(f"D = tac_queries*d_state = 64*1024 = 65536")
print(f"token-mean (DC) fraction of ||zt||: median {np.nanmedian(dcf):.6f}  "
      f"[{np.nanmin(dcf):.6f}, {np.nanmax(dcf):.6f}]")
print(f"relative spread of candidate fields ||x_i - xbar||/||xbar||: "
      f"median {np.median(dev_rel):.4g}  max {dev_rel.max():.4g}")
print(f"cos(x_i, x_cv) across the box: min {cvc.min():.12f}  "
      f"median {np.median(cvc):.12f}")

print("\n" + "="*78)
print("2.  THE GOAL TERM 1-cos  -- TRUE VALUE (float64) vs SHIPPED (float32)")
print("="*78)
print(f"cos64(zt, g):  median {np.median(cos64):.12f}   "
      f"min {cos64.min():.12f}  max {cos64.max():.12f}")
print(f"1-cos64 (TRUE goal term): median {np.median(goal64):.6g}  "
      f"min {goal64.min():.6g}  max {goal64.max():.6g}")
ULP = 2.0 ** -24
print(f"float32 ulp at 1.0 = 2^-24 = {ULP:.10g}")
print(f"median TRUE goal term / ulp = {np.median(goal64)/ULP:.4g}")
q = goal32 / ULP
print(f"SHIPPED goal32 / ulp: unique values seen = "
      f"{sorted(set(np.round(q.ravel(),6)))[:12]}")
frac_int = np.mean(np.abs(q - np.round(q)) < 1e-6)
print(f"fraction of SHIPPED goal32 that are EXACT INTEGER multiples of the ulp: "
      f"{frac_int*100:.4f}%   <-- the term is QUANTISED, i.e. it carries no "
      f"sub-ulp information")
print(f"goal32 == 0.0 exactly on {np.mean(goal32==0.0)*100:.2f}% of "
      f"(window,candidate) pairs; goal32 < 0 (cos>1, impossible in exact "
      f"arithmetic) on {np.mean(goal32<0)*100:.2f}%")

print("\n" + "="*78)
print("3.  THE MODEL TERM ALONG KAPPA  (the 1.63e-10 number)")
print("="*78)
kidx = [i for i, n in enumerate(BN) if n.startswith("kap")]
cvi = BN.index("cv")
sub = [cvi] + kidx
sp64 = goal64[:, sub].max(1) - goal64[:, sub].min(1)
sp32 = goal32[:, sub].max(1) - goal32[:, sub].min(1)
print(f"kappa sub-box = cv + {len(kidx)} constant-curvature candidates "
      f"(|kappa| <= {J['plan_cfg']['kappa_max']})")
print(f"TRUE spread of the goal term over the kappa sub-box (float64):")
print(f"    median {np.median(sp64):.6g}   mean {sp64.mean():.6g}   "
      f"p90 {np.percentile(sp64,90):.6g}   max {sp64.max():.6g}")
print(f"SHIPPED spread (float32): median {np.median(sp32):.6g}  "
      f"(quantised to {ULP:.4g})")
sp64_all = goal64.max(1) - goal64.min(1)
print(f"TRUE spread over the WHOLE box (incl. accel candidates): "
      f"median {np.median(sp64_all):.6g}  max {sp64_all.max():.6g}")

print("\n  ---- THE ARITHMETIC ----")
kmax = J["plan_cfg"]["kappa_max"]
pen_kmax = W_K * kmax**2
print(f"  penalty at |kappa| = kappa_max = {kmax}:  W_KAPPA*kappa^2 = "
      f"{W_K}*{kmax}^2 = {pen_kmax:.6g}")
print(f"  model term available to pay for it (median true spread) = "
      f"{np.median(sp64):.6g}")
print(f"  RATIO penalty / model-term = {pen_kmax/np.median(sp64):.6g}  "
      f"({pen_kmax/np.median(sp64):.3g}x)")
w_break = np.median(sp64) / kmax**2
print(f"  => W_KAPPA at which a max-curvature candidate could break even: "
      f"{w_break:.6g}")
print(f"     and the whole penalty at that weight, over the whole box, is "
      f"{w_break*kmax**2:.6g} = {w_break*kmax**2/ULP:.4g} ulp -- i.e. "
      f"{ULP/(w_break*kmax**2):.4g}x SMALLER than ONE representable step of "
      f"the term it trades against.")
pen_j = W_J * jerk
pen_k = W_K * kap
print(f"\n  jerk penalty over the box:  median {np.median(pen_j):.6g}  "
      f"max {pen_j.max():.6g}")
print(f"  kappa penalty over the box: median {np.median(pen_k):.6g}  "
      f"max {pen_k.max():.6g}")

print("\n" + "="*78)
print("4.  WHY cos ~ 1:  decomposition")
print("="*78)
# 1-cos ~ 1/2 * ||xhat - yhat||^2 ; chord = ||xhat-yhat||
print(f"chord64 = ||xhat - yhat||: median {np.median(chord64):.6g}  "
      f"min {chord64.min():.6g}  max {chord64.max():.6g}")
print(f"check  1-cos = chord^2/2 :  max |1-cos - chord^2/2| = "
      f"{np.abs(goal64 - chord64**2/2).max():.3g}")
spc = chord64[:, sub].max(1) - chord64[:, sub].min(1)
print(f"TRUE spread of the CHORD over the kappa sub-box: median {np.median(spc):.6g}")
print(f"  => chord amplifies the discriminating signal by "
      f"{np.median(spc)/np.median(sp64):.4g}x (sqrt-law: at x~{np.median(goal64):.2g}, "
      f"d(sqrt(2x))/dx = 1/sqrt(2x) = {1/math.sqrt(2*np.median(goal64)):.4g})")

print("\n" + "="*78)
print("5.  WHICH CANDIDATE WINS, PER SETTING  (offline screen, full n=%d)" % N)
print("="*78)

def screen(wj, wk, metric="cos", prec="fp32"):
    if metric == "cos":
        g = goal32 if prec == "fp32" else goal64
    else:
        g = chord64
    tot = g + wj * jerk + wk * kap
    win = tot.argmin(1)
    # kappa==0 fraction: candidate rows whose kappa is 0
    kap_of = np.array([kap[0, i] for i in range(B)])
    k0 = np.mean([kap[n, win[n]] == 0.0 for n in range(N)])
    names = [BN[i] for i in win]
    uniq = len(set(names))
    return k0, uniq, names

grid = []
for prec, metric, lbl in (("fp32", "cos", "cos/fp32 (SHIPPED)"),
                          ("fp64", "cos", "cos/fp64"),
                          ("fp64", "chord", "chord/fp64")):
    for wj, wk in ((0.02, 0.05), (0.02, 5e-3), (0.02, 5e-4), (0.02, 5e-5),
                   (0.02, 1.66e-8), (0.0, 0.05), (0.0, 0.0), (0.02, 0.0)):
        k0, uq, names = screen(wj, wk, metric, prec)
        from collections import Counter
        top = ", ".join(f"{n}:{c}" for n, c in Counter(names).most_common(4))
        grid.append((lbl, wj, wk, k0, uq, top))
print(f"{'metric/precision':22s} {'W_JERK':>8s} {'W_KAPPA':>10s} "
      f"{'kappa==0 frac':>13s} {'distinct':>9s}  winners")
for lbl, wj, wk, k0, uq, top in grid:
    print(f"{lbl:22s} {wj:8.3g} {wk:10.3g} {k0*100:12.2f}% {uq:9d}  {top}")
