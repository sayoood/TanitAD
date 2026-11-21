"""ESTIMATED arithmetic cited by SPEC_DDV2_RL_PAPER.md §2.5, §2.6, §3.4, §7.4.

Re-derives, from the published schedule constants only (diffusers `scaled_linear`,
beta 1e-4..0.02, T = 1000) and the released code's rollout (`rl.py:801-807`: labels
[18, 16, ..., 0], one-unit transitions t -> t-1 because `set_timesteps(1000)`):

  * sigma_t(eta=1) per step and whether DDv2's floors (0.04 sampler, 0.1 likelihood) bind;
  * d(mu)/d(x0_hat) and d(mu)/d(x_t) of the DDIM mean, times the gamma = 0.8 discount,
    i.e. how the policy gradient's weight on the network output is spread over the chain;
  * the WarmupCosLR per-epoch learning rate under per-epoch stepping.

Assumption stated, not hidden: `final_alpha_cumprod = 1.0` at t_prev = -1 (diffusers'
`set_alpha_to_one=True` default; the framework file is not banked -> UNVERIFIED).
Run: python ddv2_step_sensitivity.py > ../raw/ddv2_step_sensitivity.txt
"""
import math

import numpy as np

T = 1000
betas = np.linspace(0.0001 ** 0.5, 0.02 ** 0.5, T, dtype=np.float64) ** 2
abar = np.cumprod(1.0 - betas)
roll = [18, 16, 14, 12, 10, 8, 6, 4, 2, 0]
step_num = len(roll)

print("# DDv2 RL rollout: per-step noise, floors, and gradient weight on x0_hat")
print("t  t_prev  sigma_t(eta=1)  sampler_std  lik_std  dmu/dx0  dmu/dxt  gamma  weight")
total = 0.0
rows = []
for i, t in enumerate(roll):
    tp = t - 1
    a_t = abar[t]
    a_p = abar[tp] if tp >= 0 else 1.0
    var = (1.0 - a_p) / (1.0 - a_t) * (1.0 - a_t / a_p)
    s = max(math.sqrt(max(var, 0.0)), 1e-10)
    direction = math.sqrt(max(1.0 - a_p - s * s, 0.0))
    c_x0 = math.sqrt(a_p) - direction * math.sqrt(a_t) / math.sqrt(1.0 - a_t)
    c_xt = direction / math.sqrt(1.0 - a_t)
    gamma = 0.8 ** (step_num - i - 1)
    w = c_x0 * gamma
    total += w
    rows.append(w)
    print(f"{t:2d} {tp:3d}  {s:.5f}  {max(s, 0.04):.3f}  {max(s, 0.1):.3f}  "
          f"{c_x0:.4f}  {c_xt:.4f}  {gamma:.4f}  {w:.4f}")
print(f"share of total weight on the final step: {rows[-1] / total:.3f}")
print(f"sqrt(1-abar_8) = {math.sqrt(1 - abar[8]):.5f}")

print("\n# WarmupCosLR(lr=2e-4, min_lr=1e-6, epochs=10, warmup_epochs=1), stepped per epoch")
lr, mn, E, W = 2e-4, 1e-6, 10, 1
for e in range(E):
    v = lr * (e + 1) / W if e < W else mn + 0.5 * (lr - mn) * (
        1 + math.cos(math.pi * (e - W) / (E - W)))
    print(f"epoch {e}: {v:.3e}")
