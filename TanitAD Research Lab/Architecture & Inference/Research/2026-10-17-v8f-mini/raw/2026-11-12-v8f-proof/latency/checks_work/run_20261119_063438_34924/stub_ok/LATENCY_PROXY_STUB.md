Latency proxy on the pod GPU (not Thor, not a product latency; the image encoder is not included). Tier: none (a cost measurement, not a driving metric).
LONGITUDINAL, LATERAL, TACTICAL, STRATEGIC: not applicable (a cost measurement)

Generated 2026-11-19T06:34:49Z by latency_proxy.py (sha256 d1a5a743de1c). All times in ms; p50 / p95 / mean over n timed repetitions after warm-up; every timed region is bracketed by a device synchronisation.

## Setup

* subject: stub; device None; GPU None; torch None (cuda None); cudnn None
* checkpoint sha256 None; pinned tool sha256 None (expected None)
* state shape at batch 1 None; predictor context None; cached tokens at batch 1 None (tokens per frame None, frames None, feature width None)
* planner horizon 10 steps of 0.2 s = 2.0 s; source: stub
* parameters None; per top-level module null; world-model parts null; world-model stack null
* windows: None of the None grid windows; n 8, warm-up 2, K list [1, 4]

## Self-checks (all passed, or no table is written)

* s1 timer measures a known 50 ms sleep: PASS ({"value_p50_ms": 50.2627499809023, "bar": "|p50 - 50| <= 5.0 ms"})
* s1a timer waits for an asynchronous device (50 ms of async work): PASS ({"value_p50_ms": 50.27849998441525, "bar": "|p50 - 50| <= 5.0 ms"})
* s2 batched and sequential rollouts give the same states: PASS ({"value_max_abs_diff": 0.0, "max_abs_state": 0.7781371319973295, "K": 4, "H": 10, "bar": "max abs diff <= 0.0001 and states not all zero"})
* s3 a doubled horizon takes between 1.6 and 2.4 times as long: PASS ({"value_ratio": 2.0121361214262157, "p50_ms_H": 15.346749976743013, "p50_ms_2H": 30.879749974701554, "K": 1, "H": 10, "bar": "1.6 <= ratio <= 2.4"})
* s4 counted predictor calls per tick: PASS ({"value": [10, 10], "input_index": 0, "bar": "calls > 0 and equal on two ticks of one input"})
* s5 statistics of two known lists (1..100 ms; 1,1,1,1,100 ms): PASS ({"value": {"p50": 50.5, "p95": 95.05, "mean": 50.5, "n": 100}, "value_skewed": {"p50": 1.0, "p95": 80.19999999999999, "mean": 20.8, "max": 100.0}, "bar": "p50 50.5, p95 95.05, mean 50.5, n 100; skewed list p50 1, p95 80.2, mean 20.8, max 100"})

## Variant: eager

### (a) one predictor step at batch 1

| region | p50 | p95 | mean | n |
|---|---|---|---|---|
| predictor step, batch 1 | 1.524 | 1.571 | 1.499 | 8 |

### (b) imagined rollout, K candidates

| H (steps / s) | K | batch p50 | batch p95 | batch mean | K x batch-1 total p50 | total p95 | per rollout mean | seq / batch | real-time factor (batch p50) |
|---|---|---|---|---|---|---|---|---|---|
| 10 / 2.0 (planner horizon) | 1 | 15.316 | 15.567 | 15.264 | 15.419 | 15.573 | 15.413 | 1.01 | 130.6 |
| 10 / 2.0 (planner horizon) | 4 | 15.485 | 15.919 | 15.503 | 60.885 | 61.803 | 15.242 | 3.93 | 129.2 |
| 20 / 4.0 | 1 | 30.992 | 31.630 | 30.992 | 30.698 | 31.113 | 30.692 | 0.99 | 129.1 |
| 20 / 4.0 | 4 | 30.220 | 30.781 | 30.225 | 122.869 | 123.408 | 30.618 | 4.07 | 132.4 |

real-time factor = simulated seconds rolled / wall seconds spent (above 1 = faster than real time).

### (c) heads

| region | p50 | p95 | mean | n |
|---|---|---|---|---|
| c1_lead_head_one_state | 1.038 | 1.128 | 1.055 | 8 |
| c4_forward_infer_b1 | 2.539 | 2.726 | 2.581 | 8 |

### (d) one full planning tick of the planner tool as it runs today (batch 1)

**stub-tick**: 3 candidates; the planner rolled the world model on 2 of 4 inputs (50 %); predictor calls per tick (counted) {'0': 2, '10': 2}; predictor rows per call on a rolling tick []; rollout calls [0]; lead-head calls [0].

| region | p50 | p95 | mean | n |
|---|---|---|---|---|
| tick, every input | 10.352 | 18.432 | 10.415 | 8 |
| tick, inputs where it rolls | 18.203 | 18.429 | 18.119 | 8 |

Arithmetic share of the PI's 100 ms end-to-end budget (Thor, 4 cameras; the encoder is not in this tick, so this is NOT a pass or fail): p50 18 %, p95 18 %.


## Variant: fp16

NOT RUN: fp16: no such variant for the stub

## Variant: compile

NOT RUN: compile: no such variant for the stub

Definitions and open choices: LATENCY_DEFS.md.
