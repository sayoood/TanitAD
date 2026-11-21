"""THE CLOSED-FORM EXCLUSION BOUND — why the search space is collapsed before
any world model is consulted, and a PREDICTION for each swept setting.

The goal term is NON-NEGATIVE. So a candidate `n` can beat `cv` only if

        penalty(n)  <  goal(cv) - goal(n)  <=  goal(cv).

`goal(cv)` is MEASURED per window. That turns "can a noisy candidate ever win?"
into an arithmetic question about `jerk_raw` with no model in it at all.
"""
import json, sys
import numpy as np, torch
SC = r"C:\Users\Admin\AppData\Local\Temp\claude\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD\8e7cfa33-c625-47cf-88aa-711db80ac113\scratchpad"
sys.path.insert(0, r"C:\Users\Admin\tanitad-bench\refav1_eval_21109\repo\stack")
from tanitad.refs.refa_v1_plan import colored_noise, PlanConfig, _clip

J = json.load(open(SC + r"\cost_anatomy_full.json"))
BN, rows = J["box_names"], J["rows"]
i_cv = BN.index("cv")
g32 = np.array([r["goal32"] for r in rows])[:, i_cv]
g64 = np.array([r["goal64"] for r in rows])[:, i_cv]
ch = np.array([r["chord64"] for r in rows])[:, i_cv]
print(f"goal term of `cv` (the incumbent), n={len(g32)} windows")
print(f"  1-cos fp32: median {np.median(g32):.4g}  p90 {np.percentile(g32,90):.4g}"
      f"  max {g32.max():.4g}")
print(f"  1-cos fp64: median {np.median(g64):.4g}  p90 {np.percentile(g64,90):.4g}"
      f"  max {g64.max():.4g}")
print(f"  chord     : median {np.median(ch):.4g}  p90 {np.percentile(ch,90):.4g}"
      f"  max {ch.max():.4g}")

pc = PlanConfig(horizon=10, dt=0.2, seed=0)
gen = torch.Generator().manual_seed(0)
S = _clip(torch.zeros(pc.horizon, 2) + colored_noise(
    (pc.n_samples, pc.horizon, 2), pc.beta, generator=gen), pc)
jerk = ((S[:, 1:, 0] - S[:, :-1, 0]) / pc.dt).pow(2).mean(-1).numpy()
kap = S[..., 1].pow(2).mean(-1).numpy()
print(f"\niCEM iteration-0 population (n={len(jerk)}): jerk_raw min "
      f"{jerk.min():.4g}, p1 {np.percentile(jerk,1):.4g}, median "
      f"{np.median(jerk):.4g}")

SETTINGS = [
    ("S0_shipped",      "cos",   0.02,   0.05),
    ("S3_both_zero",    "cos",   0.0,    0.0),
    ("S4_jerk_zero",    "cos",   0.0,    0.05),
    ("S5_kap_zero",     "cos",   0.02,   0.0),
    ("S6_chord_shipw",  "chord", 0.02,   0.05),
    ("S7_chord_rescal", "chord", 2e-5,   5e-5),
    ("S8_rebalance",    "cos",   1e-4,   2e-4),
]
print("\n" + "="*104)
print("PER-SETTING ITERATION-0 EXCLUSION BOUND  —  what fraction of a REAL "
      "iCEM population is arithmetically")
print("excluded from beating `cv` no matter what the world model says "
      "(goal >= 0 => penalty < goal(cv))")
print("="*104)
print(f"{'setting':18s} {'metric':6s} {'W_JERK':>8s} {'W_KAPPA':>9s} "
      f"{'median goal(cv)':>16s} {'excluded @ median':>18s} {'excluded @ max':>15s}"
      f"  PREDICTION")
for lbl, met, wj, wk in SETTINGS:
    g = ch if met == "chord" else g32
    pen = wj * jerk + wk * kap
    ex_med = float((pen >= np.median(g)).mean())
    ex_max = float((pen >= g.max()).mean())
    # the bound is an ITERATION-0 bound. iCEM shrinks the variance each
    # iteration, so a penalty on ONE channel can be escaped later by driving
    # that channel to ~0 while the OTHER channel stays free. A penalty on the
    # ACCEL channel cannot be escaped, because colored_noise's zero time-mean
    # makes a constant (zero-jerk) acceleration unreachable by the proposal at
    # any iteration -- only the injected baselines carry one.
    if wj == 0 and wk == 0:
        pred = "no penalty: cost is the pure goal term"
    elif wj > 0 and ex_med >= 0.999:
        pred = "FLAT (accel channel is inescapable: zero-jerk is unreachable)"
    elif ex_med >= 0.999:
        pred = "iter-0 excluded, but kappa-only -> escapable by shrinkage"
    else:
        pred = "noisy candidates compete from iteration 0"
    print(f"{lbl:18s} {met:6s} {wj:8.3g} {wk:9.3g} {np.median(g):16.4g} "
          f"{ex_med*100:17.2f}% {ex_max*100:14.2f}%  {pred}")

print("\n⭐ THE ARGUMENT IN ONE LINE (SHIPPED setting):")
pen = 0.02 * jerk + 0.05 * kap
need = np.median(g32) / 0.02
print(f"   To beat `cv` a candidate needs 0.02*jerk_raw < goal32(cv), i.e. "
      f"jerk_raw < {need:.4g} (m/s^3)^2 at the median window.")
print(f"   The SMALLEST jerk_raw in a 300-sample iCEM population is "
      f"{jerk.min():.4g} — {jerk.min()/need:.3g}x too large.")
print(f"   => {float((pen >= np.median(g32)).mean())*100:.2f}% of the population "
      f"is excluded BEFORE the world model is consulted, on the median window; "
      f"{float((pen >= g32.max()).mean())*100:.2f}% is excluded even on the "
      f"most favourable window in the whole eval set.")
print(f"   The only candidates with penalty 0 are cv / hold_v0 / decel_1.5 (and "
      f"any constant-accel, zero-curvature sequence).")
print(f"   ⇒ The 2-distinct-plan output is a THEOREM about the cost, not a "
      f"property of the weights.")
