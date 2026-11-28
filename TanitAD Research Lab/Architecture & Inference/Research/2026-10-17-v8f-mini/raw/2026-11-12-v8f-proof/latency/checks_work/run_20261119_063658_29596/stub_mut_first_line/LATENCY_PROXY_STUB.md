Latency proxy (changed text)
LONGITUDINAL, LATERAL, TACTICAL, STRATEGIC: not applicable (a cost measurement)

Generated 2026-11-19T06:37:31Z by latency_proxy.py (sha256 d1a5a743de1c). All times in ms; p50 / p95 / mean over n timed repetitions after warm-up; every timed region is bracketed by a device synchronisation.

## Setup

* subject: stub; device None; GPU None; torch None (cuda None); cudnn None
* checkpoint sha256 None; pinned tool sha256 None (expected None)
* state shape at batch 1 None; predictor context None; cached tokens at batch 1 None (tokens per frame None, frames None, feature width None)
* planner horizon 10 steps of 0.2 s = 2.0 s; source: stub
* parameters None; per top-level module null; world-model parts null; world-model stack null
* windows: None of the None grid windows; n 8, warm-up 2, K list [1, 4]

## Self-checks (all passed, or no table is written)

* s1 timer measures a known 50 ms sleep: PASS ({"value_p50_ms": 50.39494999800809, "bar": "|p50 - 50| <= 5.0 ms"})
* s1a timer waits for an asynchronous device (50 ms of async work): PASS ({"value_p50_ms": 50.25560001377016, "bar": "|p50 - 50| <= 5.0 ms"})
* s2 batched and sequential rollouts give the same states: PASS ({"value_max_abs_diff": 0.0, "max_abs_state": 0.7781371319973295, "K": 4, "H": 10, "bar": "max abs diff <= 0.0001 and states not all zero"})
* s3 a doubled horizon takes between 1.6 and 2.4 times as long: PASS ({"value_ratio": 1.9981978635745166, "p50_ms_H": 15.648100001271814, "p50_ms_2H": 31.26799999154173, "K": 1, "H": 10, "bar": "1.6 <= ratio <= 2.4"})
* s4 counted predictor calls per tick: PASS ({"value": [10, 10], "input_index": 0, "bar": "calls > 0 and equal on two ticks of one input"})
* s5 statistics of two known lists (1..100 ms; 1,1,1,1,100 ms): PASS ({"value": {"p50": 50.5, "p95": 95.05, "mean": 50.5, "n": 100}, "value_skewed": {"p50": 1.0, "p95": 80.19999999999999, "mean": 20.8, "max": 100.0}, "bar": "p50 50.5, p95 95.05, mean 50.5, n 100; skewed list p50 1, p95 80.2, mean 20.8, max 100"})

## Variant: eager

### (a) one predictor step at batch 1

| region | p50 | p95 | mean | n |
|---|---|---|---|---|
| predictor step, batch 1 | 1.557 | 1.821 | 1.514 | 8 |

### (b) imagined rollout, K candidates

| H (steps / s) | K | batch p50 | batch p95 | batch mean | K x batch-1 total p50 | total p95 | per rollout mean | seq / batch | real-time factor (batch p50) |
|---|---|---|---|---|---|---|---|---|---|
| 10 / 2.0 (planner horizon) | 1 | 15.294 | 15.796 | 15.342 | 15.719 | 16.101 | 15.684 | 1.03 | 130.8 |
| 10 / 2.0 (planner horizon) | 4 | 15.690 | 16.197 | 15.771 | 61.469 | 62.805 | 15.373 | 3.92 | 127.5 |
| 30 / 6.0 | 1 | 47.062 | 47.366 | 46.994 | 46.755 | 47.891 | 47.012 | 0.99 | 127.5 |
| 30 / 6.0 | 4 | 46.913 | 48.003 | 46.941 | 185.017 | 186.767 | 46.228 | 3.94 | 127.9 |

real-time factor = simulated seconds rolled / wall seconds spent (above 1 = faster than real time).

### (c) heads

| region | p50 | p95 | mean | n |
|---|---|---|---|---|
| c1_lead_head_one_state | 1.042 | 1.141 | 0.996 | 8 |
| c4_forward_infer_b1 | 2.550 | 2.554 | 2.453 | 8 |

### (d) one full planning tick of the planner tool as it runs today (batch 1)

**stub-tick**: 3 candidates; the planner rolled the world model on 2 of 4 inputs (50 %); predictor calls per tick (counted) {'0': 2, '10': 2}; predictor rows per call on a rolling tick []; rollout calls [0]; lead-head calls [0].

| region | p50 | p95 | mean | n |
|---|---|---|---|---|
| tick, every input | 10.048 | 18.271 | 10.244 | 8 |
| tick, inputs where it rolls | 18.046 | 18.410 | 18.072 | 8 |

Arithmetic share of the PI's 100 ms end-to-end budget (Thor, 4 cameras; the encoder is not in this tick, so this is NOT a pass or fail): p50 18 %, p95 18 %.


## Variant: fp16

NOT RUN: fp16: no such variant for the stub

## Variant: compile

NOT RUN: compile: no such variant for the stub

Definitions and open choices: LATENCY_DEFS.md.
