"""DiffusionDrive truncation noise and step semantics, computed from the released configuration.

Sources (PUBLISHED-CODE, hustvl DiffusionDrive, banked under
`…/2026-09-05-diffusiondrive-v2-analysis/raw/ddv2_src/`):
  * V1 `v1/transfuser_model_v2.py:400-404` DDIMScheduler(1000, "scaled_linear", "sample")
    with diffusers' default beta_start/beta_end; norm_odo `:433-441`
    (x: 2(x+1.2)/56.9-1, y: 2(y+20)/46-1); train t ~ U[0,50) `:465-468`;
    inference t=8 `:518`; set_timesteps(1000) + roll [10, 0] `:505-554`.
  * V2 `diffusiondrivev2_model_rl.py:697-702` (+ steps_offset=1); norm_odo x/50, y/20 `:755-762`.
Ours (refcv5-v2): control space, 8 slots ending at 0.5/1/1.5/2/3/4/5/6 s
(`refc_anchors_6s_v0cond_alat_117.pt.json` horizons_steps x dt 0.1), control_norm (4.0, 3.0)
(`MODEL_REGISTRY.md` seam stamp), i.i.d. noise per slot (`refc_sampler.py:490-493`).
Position noise of ours is LINEARISED (small angle, v0 >= alat_v_floor 4 m/s): ESTIMATED.
"""
import json
import sys

import numpy as np
from diffusers.schedulers import DDIMScheduler

s = DDIMScheduler(num_train_timesteps=1000, beta_schedule="scaled_linear", prediction_type="sample")
ab = s.alphas_cumprod.numpy().astype(np.float64)
sig = lambda t: float(np.sqrt(1.0 - ab[t]))  # noqa: E731

out = {"diffusers_defaults": {"beta_start": s.config.beta_start, "beta_end": s.config.beta_end},
       "sqrt_one_minus_alpha_bar": {str(t): round(sig(t), 5) for t in (0, 8, 9, 10, 49)}}
v1_half_span = {"x_m": 56.9 / 2, "y_m": 46 / 2}
v2_scale = {"x_m": 50.0, "y_m": 20.0}
out["dd_v1_waypoint_noise_std_m"] = {
    "t8": {k: round(sig(8) * v, 3) for k, v in v1_half_span.items()},
    "t49_train_max": {k: round(sig(49) * v, 3) for k, v in v1_half_span.items()}}
out["dd_v2_waypoint_noise_std_m_t8"] = {k: round(sig(8) * v, 3) for k, v in v2_scale.items()}
out["dd_step_10_to_9_fraction_of_gap_kept"] = round(sig(9) / sig(10), 4)
out["ours_step_10_to_0_fraction_of_gap_kept"] = round(sig(0) / sig(10), 4)

# ours: per-slot constant-acceleration noise -> along-track / lateral position noise at T
slots_end = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]
slots_start = [0.0] + slots_end[:-1]
sig_a = {"along_mps2": sig(8) * 4.0, "lateral_mps2": sig(8) * 3.0}


def pos_std(T, sa):
    acc = 0.0
    for a, b in zip(slots_start, slots_end):
        if T <= a:
            continue
        e = min(b, T)
        w = (T - a) ** 2 / 2 - (T - e) ** 2 / 2      # integral over the slot of (T - tau) d tau
        acc += w ** 2
    return float(np.sqrt(acc) * sa)


out["ours_control_noise_std_t8"] = {k: round(v, 4) for k, v in sig_a.items()}
out["ours_position_noise_std_m_linearised"] = {
    f"{T}s": {k.split("_")[0]: round(pos_std(T, v), 3) for k, v in sig_a.items()} for T in (0.5, 1.0, 2.0, 4.0, 6.0)}
json.dump(out, open(sys.argv[1], "w"), indent=2)
print(json.dumps(out, indent=2))
