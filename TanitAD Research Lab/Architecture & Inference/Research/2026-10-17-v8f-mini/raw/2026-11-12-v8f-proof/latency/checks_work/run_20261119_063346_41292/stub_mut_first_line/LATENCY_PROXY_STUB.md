Latency proxy (changed text)
LONGITUDINAL, LATERAL, TACTICAL, STRATEGIC: not applicable (a cost measurement)

Generated 2026-11-19T06:34:19Z by latency_proxy.py (sha256 0487a8b7801e). All times in ms; p50 / p95 / mean over n timed repetitions after warm-up; every timed region is bracketed by a device synchronisation.

## Setup

* subject: stub; device None; GPU None; torch None (cuda None); cudnn None
* checkpoint sha256 None; pinned tool sha256 None (expected None)
* state shape at batch 1 None; predictor context None; cached tokens at batch 1 None (tokens per frame None, frames None, feature width None)
* planner horizon 10 steps of 0.2 s = 2.0 s; source: stub
* parameters None; per top-level module null; world-model parts null; world-model stack null
* windows: None of the None grid windows; n 8, warm-up 2, K list [1, 4]

## Self-checks (all passed, or no table is written)

* s1 timer measures a known 50 ms sleep: PASS ({"value_p50_ms": 50.365750008495525, "bar": "|p50 - 50| <= 5.0 ms"})
* s1a timer waits for an asynchronous device (50 ms of async work): PASS ({"value_p50_ms": 50.362650013994426, "bar": "|p50 - 50| <= 5.0 ms"})
* s2 batched and sequential rollouts give the same states: PASS ({"value_max_abs_diff": 0.0, "max_abs_state": 0.7781371319973295, "K": 4, "H": 10, "bar": "max abs diff <= 0.0001 and states not all zero"})
* s3 a doubled horizon takes between 1.6 and 2.4 times as long: PASS ({"value_ratio": 1.9936026818009505, "p50_ms_H": 15.483049995964393, "p50_ms_2H": 30.86704999441281, "K": 1, "H": 10, "bar": "1.6 <= ratio <= 2.4"})
* s4 counted predictor calls per tick: PASS ({"value": [10, 10], "input_index": 0, "bar": "calls > 0 and equal on two ticks of one input"})
* s5 statistics of the known list 1..100 ms: PASS ({"value": {"p50": 50.5, "p95": 95.05, "mean": 50.5, "n": 100}, "bar": "p50 50.5, p95 95.05, mean 50.5, n 100"})

## Variant: eager

### (a) one predictor step at batch 1

| region | p50 | p95 | mean | n |
|---|---|---|---|---|
| predictor step, batch 1 | 1.622 | 1.649 | 1.543 | 8 |

### (b) imagined rollout, K candidates

| H (steps / s) | K | batch p50 | batch p95 | batch mean | K x batch-1 total p50 | total p95 | per rollout mean | seq / batch | real-time factor (batch p50) |
|---|---|---|---|---|---|---|---|---|---|
| 10 / 2.0 (planner horizon) | 1 | 15.258 | 15.798 | 15.302 | 15.230 | 15.627 | 15.213 | 1.00 | 131.1 |
| 10 / 2.0 (planner horizon) | 4 | 15.326 | 15.454 | 15.243 | 61.408 | 62.151 | 15.342 | 4.01 | 130.5 |
| 30 / 6.0 | 1 | 45.173 | 46.552 | 45.513 | 46.489 | 47.035 | 46.538 | 1.03 | 132.8 |
| 30 / 6.0 | 4 | 46.252 | 46.951 | 46.153 | 183.755 | 186.399 | 45.895 | 3.97 | 129.7 |

real-time factor = simulated seconds rolled / wall seconds spent (above 1 = faster than real time).

### (c) heads

| region | p50 | p95 | mean | n |
|---|---|---|---|---|
| c1_lead_head_one_state | 1.052 | 1.093 | 1.017 | 8 |
| c4_forward_infer_b1 | 2.525 | 2.650 | 2.491 | 8 |

### (d) one full planning tick of the planner tool as it runs today (batch 1)

**stub-tick**: 3 candidates; the planner rolled the world model on 2 of 4 inputs (50 %); predictor calls per tick (counted) {'0': 2, '10': 2}; predictor rows per call on a rolling tick []; rollout calls [0]; lead-head calls [0].

| region | p50 | p95 | mean | n |
|---|---|---|---|---|
| tick, every input | 9.946 | 18.164 | 10.208 | 8 |
| tick, inputs where it rolls | 17.853 | 18.390 | 17.926 | 8 |

Arithmetic share of the PI's 100 ms end-to-end budget (Thor, 4 cameras; the encoder is not in this tick, so this is NOT a pass or fail): p50 18 %, p95 18 %.


## Variant: fp16

NOT RUN: fp16: no such variant for the stub

## Variant: compile

NOT RUN: compile: no such variant for the stub

Definitions and open choices: LATENCY_DEFS.md.
