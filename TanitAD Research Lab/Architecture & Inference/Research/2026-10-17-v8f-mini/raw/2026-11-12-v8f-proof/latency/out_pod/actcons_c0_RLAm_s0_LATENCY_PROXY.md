Latency proxy on the pod GPU (not Thor, not a product latency; the image encoder is not included). Tier: none (a cost measurement, not a driving metric).
LONGITUDINAL, LATERAL, TACTICAL, STRATEGIC: not applicable (a cost measurement)

Generated 2026-11-19T09:14:04Z by latency_proxy.py (sha256 d1a5a743de1c). All times in ms; p50 / p95 / mean over n timed repetitions after warm-up; every timed region is bracketed by a device synchronisation.

## Setup

* subject: real; device cuda; GPU NVIDIA A40; torch 2.8.0+cu128 (cuda 12.8); cudnn 91002
* checkpoint sha256 51a280fedb5a8226fc9768f8dcf540d37694cbe1e2ed8a4f93c5e89626963bf3; pinned tool sha256 61e140edb9da01acbbba4eef81e74bbb68b57e2ff31842e05958759446d97397 (expected 61e140edb9da01acbbba4eef81e74bbb68b57e2ff31842e05958759446d97397)
* state shape at batch 1 [1, 640]; predictor context 3; cached tokens at batch 1 [1, 16, 40, 1024] (tokens per frame 40, frames 16, feature width 1024)
* planner horizon 10 steps of 0.2 s = 2.0 s; source: imag_plan.N_ROLL (imag_plan.py:41): 10 rollout steps of 0.2 s = 2.0 s, the horizon the planner rolls
* parameters 9191726; per top-level module {"prior": 17724, "adapter": 264448, "e_ego": 2304, "e_nav": 1536, "e_lim": 1536, "e_age": 768, "e_ck": 3328, "e_code": 15616, "read_s": 792320, "read_t": 792320, "read_o": 1583104, "head_s": 529417, "trunk_t": 656128, "h_goal": 5654, "h_lat": 1285, "h_lon": 1799, "h_cons": 6682, "head_o": 537112, "wm": 3967381, "(direct parameters, e.g. pos, time)": 11264}; world-model parts {"stack": 3880258, "proj": 4624, "lead_head": 82499}; world-model stack {"act_embed": 1152, "encoders": 131584, "macro_encoders": 6400, "predictors": 3067264, "idm_heads": 673858, "sigreg": 0}
* windows: 80 of the 2399 grid windows; n 200, warm-up 20, K list [1, 8, 16, 32]

## Self-checks (all passed, or no table is written)

* s1 timer measures a known 50 ms sleep: PASS ({"value_p50_ms": 50.27670149866026, "bar": "|p50 - 50| <= 5.0 ms"})
* s1a timer waits for an asynchronous device (50 ms of async work): PASS ({"value_p50_ms": 50.09225249523297, "bar": "|p50 - 50| <= 5.0 ms"})
* s2 batched and sequential rollouts give the same states: PASS ({"value_max_abs_diff": 9.5367431640625e-07, "max_abs_state": 4.625880718231201, "K": 32, "H": 10, "bar": "max abs diff <= 0.0001 and states not all zero"})
* s3 a doubled horizon takes between 1.6 and 2.4 times as long: PASS ({"value_ratio": 2.089567373922049, "p50_ms_H": 11.488870994071476, "p50_ms_2H": 24.006769992411137, "K": 8, "H": 10, "bar": "1.6 <= ratio <= 2.4"})
* s4 counted predictor calls per tick: PASS ({"value": [10, 10], "input_index": 1, "bar": "calls > 0 and equal on two ticks of one input"})
* s5 statistics of two known lists (1..100 ms; 1,1,1,1,100 ms): PASS ({"value": {"p50": 50.5, "p95": 95.05, "mean": 50.5, "n": 100}, "value_skewed": {"p50": 1.0, "p95": 80.19999999999999, "mean": 20.8, "max": 100.0}, "bar": "p50 50.5, p95 95.05, mean 50.5, n 100; skewed list p50 1, p95 80.2, mean 20.8, max 100"})

## Variant: eager

### (a) one predictor step at batch 1

| region | p50 | p95 | mean | n |
|---|---|---|---|---|
| predictor step, batch 1 | 1.244 | 1.486 | 1.287 | 200 |

### (b) imagined rollout, K candidates

| H (steps / s) | K | batch p50 | batch p95 | batch mean | K x batch-1 total p50 | total p95 | per rollout mean | seq / batch | real-time factor (batch p50) |
|---|---|---|---|---|---|---|---|---|---|
| 10 / 2.0 (planner horizon) | 1 | 11.823 | 13.190 | 11.988 | 11.293 | 14.719 | 11.649 | 0.96 | 169.2 |
| 10 / 2.0 (planner horizon) | 8 | 11.768 | 12.828 | 11.908 | 90.092 | 103.542 | 11.611 | 7.66 | 170.0 |
| 10 / 2.0 (planner horizon) | 16 | 11.543 | 12.314 | 11.549 | 151.622 | 183.788 | 10.056 | 13.14 | 173.3 |
| 10 / 2.0 (planner horizon) | 32 | 9.711 | 10.762 | 9.851 | 295.961 | 306.069 | 9.283 | 30.48 | 205.9 |
| 30 / 6.0 | 1 | 27.522 | 28.162 | 27.531 | 28.055 | 28.851 | 28.152 | 1.02 | 218.0 |
| 30 / 6.0 | 8 | 29.316 | 30.239 | 29.422 | 221.691 | 261.470 | 28.739 | 7.56 | 204.7 |
| 30 / 6.0 | 16 | 29.471 | 30.642 | 29.618 | 500.713 | 541.224 | 30.130 | 16.99 | 203.6 |
| 30 / 6.0 | 32 | 28.328 | 28.929 | 28.390 | 859.909 | 1021.012 | 27.668 | 30.36 | 211.8 |

real-time factor = simulated seconds rolled / wall seconds spent (above 1 = faster than real time).

### (c) heads

| region | p50 | p95 | mean | n |
|---|---|---|---|---|
| c1_lead_head_one_state | 0.106 | 0.117 | 0.107 | 200 |
| c2_lead_head_at_1s_2s_on_rolled_state (2 head calls) | 0.360 | 0.371 | 0.361 | 200 |
| c3_lead_head_on_cached_tokens (head_values: adapter, z1, head, host copy) | 0.396 | 0.408 | 0.396 | 200 |
| c4_trajectory_heads (forward_infer, S+T+O readers and all heads, batch 1) | 5.773 | 5.959 | 5.789 | 200 |

### (d) one full planning tick of the planner tool as it runs today (batch 1)

**cand-imag/scale (card default set)**: 11 candidates; the planner rolled the world model on 23 of 80 inputs (29 %); predictor calls per tick (counted) {'0': 57, '10': 23}; predictor rows per call on a rolling tick [11, 11, 11, 11, 11, 11, 11, 11, 11, 11]; rollout calls [0, 1]; lead-head calls [1, 3].

| region | p50 | p95 | mean | n |
|---|---|---|---|---|
| tick, every input | 7.235 | 24.404 | 12.011 | 200 |
| tick, inputs where it rolls | 24.437 | 24.937 | 24.448 | 200 |

Arithmetic share of the PI's 100 ms end-to-end budget (Thor, 4 cameras; the encoder is not in this tick, so this is NOT a pass or fail): p50 24 %, p95 25 %.

**cand-imagkin/decel (note ADD40_N3, N4)**: 8 candidates; the planner rolled the world model on 23 of 80 inputs (29 %); predictor calls per tick (counted) {'0': 57, '10': 23}; predictor rows per call on a rolling tick [8, 8, 8, 8, 8, 8, 8, 8, 8, 8]; rollout calls [0, 1]; lead-head calls [1, 3].

| region | p50 | p95 | mean | n |
|---|---|---|---|---|
| tick, every input | 7.506 | 21.240 | 11.242 | 200 |
| tick, inputs where it rolls | 20.972 | 29.602 | 21.762 | 200 |

Arithmetic share of the PI's 100 ms end-to-end budget (Thor, 4 cameras; the encoder is not in this tick, so this is NOT a pass or fail): p50 21 %, p95 30 %.


## Variant: fp16

### (a) one predictor step at batch 1

| region | p50 | p95 | mean | n |
|---|---|---|---|---|
| predictor step, batch 1 | 1.571 | 2.652 | 1.983 | 200 |

### (b) imagined rollout, K candidates

| H (steps / s) | K | batch p50 | batch p95 | batch mean | K x batch-1 total p50 | total p95 | per rollout mean | seq / batch | real-time factor (batch p50) |
|---|---|---|---|---|---|---|---|---|---|
| 10 / 2.0 (planner horizon) | 1 | 13.908 | 15.832 | 13.941 | 13.185 | 14.017 | 13.076 | 0.95 | 143.8 |
| 10 / 2.0 (planner horizon) | 8 | 11.753 | 14.015 | 12.355 | 101.498 | 109.169 | 12.656 | 8.64 | 170.2 |
| 10 / 2.0 (planner horizon) | 16 | 13.992 | 16.339 | 14.343 | 183.220 | 211.169 | 11.676 | 13.09 | 142.9 |
| 10 / 2.0 (planner horizon) | 32 | 14.074 | 16.458 | 13.941 | 348.489 | 372.845 | 10.993 | 24.76 | 142.1 |
| 30 / 6.0 | 1 | 32.360 | 34.835 | 33.121 | 32.306 | 34.722 | 32.993 | 1.00 | 185.4 |
| 30 / 6.0 | 8 | 34.746 | 37.393 | 35.463 | 262.484 | 277.166 | 32.975 | 7.55 | 172.7 |
| 30 / 6.0 | 16 | 35.370 | 39.486 | 36.130 | 511.993 | 522.076 | 32.047 | 14.48 | 169.6 |
| 30 / 6.0 | 32 | 35.376 | 37.538 | 36.061 | 1036.617 | 1217.336 | 33.524 | 29.30 | 169.6 |

real-time factor = simulated seconds rolled / wall seconds spent (above 1 = faster than real time).

### (c) heads

| region | p50 | p95 | mean | n |
|---|---|---|---|---|
| c1_lead_head_one_state | 0.218 | 2.392 | 0.753 | 200 |
| c2_lead_head_at_1s_2s_on_rolled_state (2 head calls) | 0.553 | 2.431 | 1.390 | 200 |
| c3_lead_head_on_cached_tokens (head_values: adapter, z1, head, host copy) | 0.604 | 4.801 | 1.619 | 200 |
| c4_trajectory_heads (forward_infer, S+T+O readers and all heads, batch 1) | 8.194 | 10.638 | 8.348 | 200 |

### (d) one full planning tick of the planner tool as it runs today (batch 1)

**cand-imag/scale (card default set)**: 11 candidates; the planner rolled the world model on 23 of 80 inputs (29 %); predictor calls per tick (counted) {'0': 57, '10': 23}; predictor rows per call on a rolling tick [11, 11, 11, 11, 11, 11, 11, 11, 11, 11]; rollout calls [0, 1]; lead-head calls [1, 3].

| region | p50 | p95 | mean | n |
|---|---|---|---|---|
| tick, every input | 17.680 | 41.121 | 21.396 | 200 |
| tick, inputs where it rolls | 35.808 | 42.858 | 35.148 | 200 |

Arithmetic share of the PI's 100 ms end-to-end budget (Thor, 4 cameras; the encoder is not in this tick, so this is NOT a pass or fail): p50 36 %, p95 43 %.

**cand-imagkin/decel (note ADD40_N3, N4)**: 8 candidates; the planner rolled the world model on 23 of 80 inputs (29 %); predictor calls per tick (counted) {'0': 57, '10': 23}; predictor rows per call on a rolling tick [8, 8, 8, 8, 8, 8, 8, 8, 8, 8]; rollout calls [0, 1]; lead-head calls [1, 3].

| region | p50 | p95 | mean | n |
|---|---|---|---|---|
| tick, every input | 11.599 | 38.663 | 17.433 | 200 |
| tick, inputs where it rolls | 34.622 | 38.877 | 32.747 | 200 |

Arithmetic share of the PI's 100 ms end-to-end budget (Thor, 4 cameras; the encoder is not in this tick, so this is NOT a pass or fail): p50 35 %, p95 39 %.


Notes: {"max_abs_state_diff_vs_eager_K8_H10": 0.001595478504896164}

## Variant: compile

### (a) one predictor step at batch 1

| region | p50 | p95 | mean | n |
|---|---|---|---|---|
| predictor step, batch 1 | 0.667 | 2.489 | 1.286 | 200 |

### (b) imagined rollout, K candidates

| H (steps / s) | K | batch p50 | batch p95 | batch mean | K x batch-1 total p50 | total p95 | per rollout mean | seq / batch | real-time factor (batch p50) |
|---|---|---|---|---|---|---|---|---|---|
| 10 / 2.0 (planner horizon) | 1 | 5.867 | 5.937 | 5.740 | 5.859 | 6.574 | 5.702 | 1.00 | 340.9 |
| 10 / 2.0 (planner horizon) | 8 | 6.288 | 6.345 | 5.906 | 41.334 | 52.475 | 5.306 | 6.57 | 318.0 |
| 10 / 2.0 (planner horizon) | 16 | 6.385 | 8.834 | 6.792 | 77.474 | 82.684 | 4.712 | 12.13 | 313.3 |
| 10 / 2.0 (planner horizon) | 32 | 6.628 | 6.700 | 6.067 | 134.144 | 176.608 | 4.368 | 20.24 | 301.8 |

real-time factor = simulated seconds rolled / wall seconds spent (above 1 = faster than real time).

### (c) heads

| region | p50 | p95 | mean | n |
|---|---|---|---|---|
| NOT RUN: the compile variant covers (a) and (b) only | | | | |

### (d) one full planning tick of the planner tool as it runs today (batch 1)

NOT RUN: the planning tick has host-side numpy work and is not compiled

Notes: {"max_abs_state_diff_vs_eager_K8_H10": 5.960464477539062e-07}

Definitions and open choices: LATENCY_DEFS.md.
