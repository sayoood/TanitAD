"""EMA arithmetic for the E1/E2/E3 schedules (WeightEMA: fixed decay, no warm-up, shadow = weights at step 0).
shadow_T = d^T * w_0 + sum_{t=1..T} (1-d) d^(T-t) w_t. Read-only arithmetic, no data."""
import json

cases = {"E1 rig (0.999, 4000)": (0.999, 4000), "E2 screen (0.9998, 6000)": (0.9998, 6000),
         "E3 D1 read (0.9998, 36000)": (0.9998, 36000), "E3 full (0.9998, 73280)": (0.9998, 73280)}
out = {}
for name, (d, T) in cases.items():
    init = d ** T
    # weight on steps (T-1000, T]
    last1k = 1 - d ** 1000
    # weights on [1,1000] (warm-up of E2), on the second half
    first1k = d ** (T - 1000) - d ** T
    # a zero-init parameter that ramps linearly w_t = t/T * w_T: shadow / w_T
    ramp = sum((1 - d) * d ** (T - t) * (t / T) for t in range(1, T + 1))
    # a randomly-initialised readout that converges exponentially to w* with time constant tau=1000 steps from w_0:
    # shadow - w* = (w_0 - w*) * [d^T + sum (1-d) d^(T-t) exp(-t/tau)]
    import math
    tau = 1000.0
    resid = d ** T + sum((1 - d) * d ** (T - t) * math.exp(-t / tau) for t in range(1, T + 1))
    raw_resid = math.exp(-T / tau)
    out[name] = {"decay": d, "steps": T, "init_weight": init, "weight_last_1000": last1k, "weight_first_1000": first1k,
                 "zero_init_linear_ramp_shadow_over_final": ramp,
                 "exp_converging_readout_residual_ema(tau=1000)": resid, "same_residual_raw": raw_resid,
                 "ema_age_mean_steps": d / (1 - d)}
    print("%-30s init %.4f | last1k %.3f | first1k %.3f | ramp-gate shadow/final %.3f | residual EMA %.3f vs raw %.5f"
          % (name, init, last1k, first1k, ramp, resid, raw_resid))
json.dump(out, open("F:/refcv9_e3/k2diag/ema_math.json", "w"), indent=1)
