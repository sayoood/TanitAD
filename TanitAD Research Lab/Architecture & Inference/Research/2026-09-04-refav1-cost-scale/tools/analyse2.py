"""Second pass: who is priced out by which term, and is the correct plan even
REACHABLE by the shipped iCEM proposal?"""
import json, math
import numpy as np
from collections import Counter

P = r"C:\Users\Admin\AppData\Local\Temp\claude\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD\8e7cfa33-c625-47cf-88aa-711db80ac113\scratchpad\cost_anatomy_full.json"
J = json.load(open(P))
BN = J["box_names"]; R = J["rows"]
W_J, W_K = J["weights_shipped"]["W_JERK"], J["weights_shipped"]["W_KAPPA"]
A = lambda k: np.array([r[k] for r in R], dtype=np.float64)
goal32, goal64, chord64 = A("goal32"), A("goal64"), A("chord64")
jerk, kap = A("jerk_raw"), A("kap_raw")
N, B = goal32.shape
i_cv = BN.index("cv"); i_seed = BN.index("seed0") if "seed0" in BN else None
ULP = 2.0 ** -24

print("="*80)
print("A.  PER-CANDIDATE LEDGER  (medians over n=%d windows)" % N)
print("="*80)
print(f"{'candidate':12s} {'jerk_raw':>10s} {'kap_raw':>9s} "
      f"{'W_J*jerk':>10s} {'W_K*kap':>10s} {'goal32':>11s} "
      f"{'goal32-cv':>12s} {'chord-cv':>11s} {'TOTAL-cv':>11s}")
for i, nm in enumerate(BN):
    dj = W_J * np.median(jerk[:, i]); dk = W_K * np.median(kap[:, i])
    dg = np.median(goal32[:, i] - goal32[:, i_cv])
    dc = np.median(chord64[:, i] - chord64[:, i_cv])
    tot = np.median((goal32[:, i] + W_J*jerk[:, i] + W_K*kap[:, i])
                    - (goal32[:, i_cv] + W_J*jerk[:, i_cv] + W_K*kap[:, i_cv]))
    print(f"{nm:12s} {np.median(jerk[:,i]):10.4g} {np.median(kap[:,i]):9.4g} "
          f"{dj:10.4g} {dk:10.4g} {np.median(goal32[:,i]):11.4g} "
          f"{dg:12.4g} {dc:11.4g} {tot:11.4g}")

print("\n" + "="*80)
print("B.  THE IMAGINED GOAL'S OWN PLAN (seed0) — is it priced out, and by which term?")
print("="*80)
if i_seed is not None:
    dg = goal32[:, i_seed] - goal32[:, i_cv]         # goal advantage (negative = better)
    pj = W_J * (jerk[:, i_seed] - jerk[:, i_cv])
    pk = W_K * (kap[:, i_seed] - kap[:, i_cv])
    tot = dg + pj + pk
    print(f"goal32(seed) - goal32(cv):  median {np.median(dg):+.6g}   "
          f"wins goal on {np.mean(dg<0)*100:.1f}% of windows")
    print(f"W_JERK  penalty diff:       median {np.median(pj):+.6g}  "
          f"(seed jerk_raw median {np.median(jerk[:,i_seed]):.4g})")
    print(f"W_KAPPA penalty diff:       median {np.median(pk):+.6g}  "
          f"(seed kap_raw  median {np.median(kap[:,i_seed]):.4g})")
    print(f"TOTAL diff at SHIPPED weights: median {np.median(tot):+.6g}  "
          f"seed BEATS cv on {np.mean(tot<0)*100:.1f}% of windows")
    nz = jerk[:, i_seed] > 0
    print(f"\nseed has NON-ZERO jerk on {nz.mean()*100:.1f}% of windows, "
          f"non-zero kappa on {(kap[:,i_seed]>0).mean()*100:.1f}%")
    # break-even weights, per window, holding the other at 0
    with np.errstate(divide="ignore", invalid="ignore"):
        wj_be = np.where(jerk[:, i_seed] > 0, -dg / jerk[:, i_seed], np.nan)
        wk_be = np.where(kap[:, i_seed] > 0, -dg / kap[:, i_seed], np.nan)
    print(f"W_JERK  at which seed breaks even vs cv (other term 0): "
          f"median {np.nanmedian(wj_be):.4g}   (SHIPPED {W_J})  "
          f"=> shipped is {W_J/np.nanmedian(wj_be):.4g}x too big")
    print(f"W_KAPPA at which seed breaks even vs cv (other term 0): "
          f"median {np.nanmedian(wk_be):.4g}   (SHIPPED {W_K})  "
          f"=> shipped is {W_K/np.nanmedian(wk_be):.4g}x too big")

print("\n" + "="*80)
print("C.  CAN A CURVATURE CANDIDATE EVER WIN?  break-even W_KAPPA per window")
print("="*80)
kidx = [i for i, n in enumerate(BN) if n.startswith("kap")]
best_k = np.full(N, np.nan); best_name = []
for n in range(N):
    adv = goal32[n, kidx] - goal32[n, i_cv]       # negative = curvature helps
    j = int(np.argmin(adv))
    if adv[j] < 0:
        best_k[n] = -adv[j] / kap[n, kidx[j]]
        best_name.append(BN[kidx[j]])
    else:
        best_name.append(None)
ok = ~np.isnan(best_k)
print(f"a constant-curvature candidate beats cv ON THE GOAL TERM on "
      f"{ok.sum()}/{N} = {ok.mean()*100:.1f}% of windows")
print(f"break-even W_KAPPA on those: median {np.nanmedian(best_k):.6g}  "
      f"p10 {np.nanpercentile(best_k,10):.6g}  p90 {np.nanpercentile(best_k,90):.6g}")
print(f"SHIPPED W_KAPPA = {W_K}  =>  {W_K/np.nanmedian(best_k):.4g}x the "
      f"median break-even")
print("winning curvature (goal term only):", Counter([b for b in best_name if b]).most_common(6))
# same under chord
best_kc = np.full(N, np.nan)
for n in range(N):
    adv = chord64[n, kidx] - chord64[n, i_cv]
    j = int(np.argmin(adv))
    if adv[j] < 0:
        best_kc[n] = -adv[j] / kap[n, kidx[j]]
print(f"\nunder CHORD: break-even W_KAPPA median {np.nanmedian(best_kc):.6g} "
      f"({np.nanmedian(best_kc)/np.nanmedian(best_k):.4g}x larger than under cos) "
      f"=> shipped is {W_K/np.nanmedian(best_kc):.4g}x too big")

print("\n" + "="*80)
print("D.  REACHABILITY — what the shipped iCEM proposal can even PROPOSE")
print("="*80)
import torch, sys
sys.path.insert(0, r"C:\Users\Admin\tanitad-bench\refav1_eval_21109\repo\stack")
from tanitad.refs.refa_v1_plan import colored_noise, PlanConfig, _clip
pc = PlanConfig(**{k: J["plan_cfg"][k] for k in
                   ("n_samples","n_iters","n_elites","horizon","dt","a_max",
                    "kappa_max","beta","inject_baselines","seed")})
gen = torch.Generator().manual_seed(0)
noise = colored_noise((pc.n_samples, pc.horizon, 2), pc.beta, generator=gen)
mean0 = torch.zeros(pc.horizon, 2)
samples = _clip(mean0 + noise * 1.0, pc)
tm = samples.mean(1)                       # time-mean per sample, per channel
print(f"iteration-0 population: n={samples.shape[0]}, mean init = ZEROS "
      f"(refa_v1_plan.py:211)")
print(f"|time-mean| of the accel channel  : max {tm[:,0].abs().max():.3g}")
print(f"|time-mean| of the kappa channel  : max {tm[:,1].abs().max():.3g}")
print(f"=> colored_noise subtracts the time-mean (refa_v1_plan.py:157), so EVERY "
      f"iteration-0 sample has EXACTLY zero mean accel and zero mean curvature.")
print(f"   A SUSTAINED accel or curvature is UNREACHABLE at iteration 0 except "
      f"via the injected baselines (cv, hold_v0, decel_1.5 — none carries "
      f"curvature) and the goal SEED.")
print(f"   n_elites={pc.n_elites} of ~{pc.n_samples}: the seed must survive the "
      f"elite cut to move the mean; the cut is made by the SAME cost.")
# how far does clipping bite
print(f"   (clip at kappa_max={pc.kappa_max}: {float((noise[...,1].abs()>pc.kappa_max).float().mean())*100:.1f}% "
      f"of raw kappa samples are clipped, so the population is effectively "
      f"bang-bang in curvature)")

print("\n" + "="*80)
print("E.  FLOAT32 QUANTISATION — where it bites and where it does not")
print("="*80)
sub = [i_cv] + kidx
sp64 = goal64[:, sub].max(1) - goal64[:, sub].min(1)
sp32 = goal32[:, sub].max(1) - goal32[:, sub].min(1)
print(f"kappa sub-box goal spread: fp64 median {np.median(sp64):.4g} = "
      f"{np.median(sp64)/ULP:.1f} ulp ; fp32 median {np.median(sp32):.4g}")
frac_lost = np.mean(sp64 < ULP)
print(f"windows where the WHOLE kappa sub-box fits inside ONE ulp (signal "
      f"destroyed by the fp32 subtraction): {frac_lost*100:.2f}%")
# rank agreement between fp32 and fp64 over the kappa sub-box
agree = np.mean([np.argmin(goal32[n, sub]) == np.argmin(goal64[n, sub])
                 for n in range(N)])
print(f"argmin over the kappa sub-box agrees fp32 vs fp64 on {agree*100:.2f}% "
      f"of windows")
ties = np.mean([ (goal32[n, sub] == goal32[n, sub].min()).sum() > 1
                 for n in range(N)])
print(f"fp32 produces a TIE for the argmin on {ties*100:.2f}% of windows "
      f"(fp64: {np.mean([(goal64[n,sub]==goal64[n,sub].min()).sum()>1 for n in range(N)])*100:.2f}%)")
