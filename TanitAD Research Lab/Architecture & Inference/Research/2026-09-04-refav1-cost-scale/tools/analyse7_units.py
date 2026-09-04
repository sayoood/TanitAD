"""The DIMENSIONAL statement, and what the penalties charge a real iCEM
population. No GPU: `colored_noise` and `_clip` are pure torch."""
import sys, json
import numpy as np, torch
REPO = r"C:\Users\Admin\tanitad-bench\refav1_eval_21109\repo"
sys.path.insert(0, REPO + r"\stack")
from tanitad.refs.refa_v1_plan import colored_noise, PlanConfig, _clip, _baseline_controls
from tanitad.refs import refa_v1 as R

pc = PlanConfig(horizon=10, dt=0.2, seed=0)
print(f"W_JERK={R.W_JERK} (charges (m/s^3)^2)   W_KAPPA={R.W_KAPPA} (charges (1/m)^2)")
print(f"the goal term 1-cos is DIMENSIONLESS and bounded in [0, 2]; "
      f"the chord in [0, 2].\n")

print("="*88)
print("WHAT THE PENALTIES CHARGE A REAL iCEM POPULATION (iteration 0, n=300)")
print("="*88)
g = torch.Generator().manual_seed(0)
noise = colored_noise((pc.n_samples, pc.horizon, 2), pc.beta, generator=g)
S = _clip(torch.zeros(pc.horizon, 2) + noise * 1.0, pc)
jerk = ((S[:, 1:, 0] - S[:, :-1, 0]) / pc.dt).pow(2).mean(-1)
kap = S[..., 1].pow(2).mean(-1)
pj, pk = R.W_JERK * jerk, R.W_KAPPA * kap
for nm, v in (("jerk_raw  mean((da/dt)^2)  (m/s^3)^2", jerk),
              ("kappa_raw mean(kappa^2)    (1/m)^2", kap),
              ("W_JERK * jerk_raw", pj),
              ("W_KAPPA * kappa_raw", pk),
              ("total penalty", pj + pk)):
    v = v.numpy()
    print(f"  {nm:38s} median {np.median(v):10.4g}  p10 {np.percentile(v,10):10.4g}"
          f"  p90 {np.percentile(v,90):10.4g}  max {v.max():10.4g}")
print(f"\n  time-mean of the accel channel:  max |mean_t a| = "
      f"{S[:,:,0].mean(1).abs().max():.4g}   -> a SUSTAINED acceleration is "
      f"NOT REACHABLE at iteration 0")
print(f"  time-mean of the kappa channel:  max |mean_t k| = "
      f"{S[:,:,1].mean(1).abs().max():.4g}   -> sustained CURVATURE *is* "
      f"reachable, because {float((noise[...,1].abs()>pc.kappa_max).float().mean())*100:.1f}% "
      f"of raw kappa samples hit the +-{pc.kappa_max} clip and the clip breaks "
      f"colored_noise's zero-mean property")

print("\n" + "="*88)
print("THE COMPARISON THAT SETTLES IT")
print("="*88)
J = json.load(open(r"C:\Users\Admin\AppData\Local\Temp\claude\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD\8e7cfa33-c625-47cf-88aa-711db80ac113\scratchpad\cost_anatomy_full.json"))
rows = J["rows"]; BN = J["box_names"]
g64 = np.array([r["goal64"] for r in rows]); ch = np.array([r["chord64"] for r in rows])
print(f"  goal term 1-cos over the WHOLE candidate box: median "
      f"{np.median(g64):.4g}, p99 {np.percentile(g64,99):.4g}, max {g64.max():.4g}")
print(f"  chord            over the WHOLE candidate box: median "
      f"{np.median(ch):.4g}, p99 {np.percentile(ch,99):.4g}, max {ch.max():.4g}")
print(f"  median penalty charged to a REAL iCEM candidate: "
      f"{float(np.median((pj+pk).numpy())):.4g}")
print(f"\n  => a typical iCEM candidate pays "
      f"{float(np.median((pj+pk).numpy()))/np.median(g64):.4g}x its ENTIRE goal "
      f"term in penalty under 1-cos, and "
      f"{float(np.median((pj+pk).numpy()))/np.median(ch):.4g}x under the chord.")
print(f"  => of that penalty, the JERK share is "
      f"{float(np.median(pj.numpy())/np.median((pj+pk).numpy()))*100:.1f}% "
      f"and the KAPPA share {float(np.median(pk.numpy())/np.median((pj+pk).numpy()))*100:.1f}%.")

print("\n" + "="*88)
print("AND WHAT THE INJECTED BASELINES PAY")
print("="*88)
b = _baseline_controls(pc, 10.0, "cpu", None)
for k, c in b.items():
    jj = float(((c[1:,0]-c[:-1,0])/pc.dt).pow(2).mean())
    kk = float(c[...,1].pow(2).mean())
    print(f"  {k:12s} jerk_raw {jj:.6g}  kappa_raw {kk:.6g}  "
          f"penalty {R.W_JERK*jj + R.W_KAPPA*kk:.6g}")
print("  => cv / hold_v0 / decel_1.5 all pay EXACTLY ZERO. The cost's minimum "
      "over the\n     penalty term alone is the set {constant accel, zero "
      "curvature} — which is\n     exactly the 2 distinct plans the banked eval "
      "emitted on 282/282 windows.")
