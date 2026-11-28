# LATENCY_DEFS : definitions and decisions of the latency proxy tool (step S0 of COORD_D5_SCOPING_2026-11-19.md)

Latency proxy on the pod GPU (not Thor, not a product latency; the image encoder is not included). Tier: none (a cost measurement, not a driving metric).
LONGITUDINAL, LATERAL, TACTICAL, STRATEGIC: not applicable (a cost measurement).

## What is timed

| id | region | callable |
|---|---|---|
| a | one predictor step, batch 1 | `stack.predictors[0](z_ctx [1,3,D], act_embed(a) [1,3,32])`, the predictor's own context (`imag_dump.predictor_context`) |
| b | imagined rollout of K candidates | `stack.rollout(1, z0 [K,D], A [K,H,2], context=ctx)` as ONE batch, and K calls with batch 1 (total and per rollout); the rollout includes the action embedding |
| c1 | lead head on one state | `model.wm.lead_head(Z5 [1,5,D])` |
| c2 | lead head at 1.0 s and 2.0 s on a rolled state | `imag_dump.head_reads_rolled` (2 head calls), the read the planner makes |
| c3 | lead head on cached tokens | `v8f_t1_eval_rule.head_values` (adapter, z1, head, host copy) |
| c4 | trajectory heads | `model.forward_infer(batch)` at batch 1: the S, T and O readers over the cached tokens and all heads |
| d | one planning tick | `ImagPlanner.__call__(batch)`: pinned model planner (= c4) + head read + gate + candidate plans and actions on the host + imagined rollouts of the candidate set + rule |

Every region is `sync(); t0; fn(); sync(); t1` (`torch.cuda.synchronize()` on cuda). Warm-up calls are untimed; the first warm-up call is reported as `first_call_ms` (cold start, or compile time).
Statistics are p50 / p95 / mean over n repetitions (linear-interpolated percentiles of the sorted samples), in ms.

## Decisions

1. **Horizon.** The planner rolls `imag_plan.N_ROLL = 10` steps of 0.2 s = 2.0 s (`imag_plan.py:41`); that is the default of `--horizon-steps` (0 = the planner's own). The scoping card says "to 6 s"; the plan is 30 knots = 6 s, so `--extra-horizons 30` is timed as well (eager and fp16 only), labelled in the table. The tick (d) always uses the planner's own horizon.
2. **Candidate sets of the tick.** Two ImagPlanner configurations, both timed: `cand-imag/scale` with the card set `CAND_CARD` (11 candidates, c = 0.0 included) as the default, and `cand-imagkin/decel` with `DECEL_CARD` (8 candidates, note ADD40_N3/N4), the form of the latest chain runs.
3. **Real windows.** The tick uses tick 0 of `--n-windows` evenly spaced windows of the eval grid, built by the T1 tool's own functions (`make_states`, `build_item`, `collate`), batch 1. The planner rolls the world model only on windows with a lead above the gate, so the tick is reported for every window AND for the windows where it rolls (the counted predictor calls decide which). Both are real; the second is the loaded case.
4. **Count.** Predictor calls, `stack.rollout` calls and lead-head calls per tick are counted by wrappers set on the instances (`forward`, `rollout`) for an untimed pass only and removed after it. The timed pass runs unwrapped. One rolling tick at batch 1 is expected to show 10 predictor calls at batch C (= candidates), 1 rollout call, 2 lead-head calls in the rollout read plus the gate read; the file holds the counted values.
5. **Inputs of a, b, c1, c2.** The state z0 is the real level-1 state of window 0's tick frame; the actions are random (fixed seed) raw actions of the logged size (acceleration sd 1.0, yaw rate sd 0.05), scaled by `wm.scale_actions`. Shapes come from the real batch. Random actions change no shape and no cost of a dense network.
6. **Loader.** The model is built by `imag_dump.load_model` (the study's loader: `V8FMiniConfig(**ck["cfg"])`, `V8FMini`, strict `load_state_dict`), which is the construction of `T.cmd_run` lines 505-513 (those lines are inline there, not a function). The pinned tool is imported from `--tree` by `load_pinned`; its sha256 is recorded next to the expected one.
7. **Variants.** `eager` always. `fp16`: `torch.autocast(cuda, float16)` around every region (weights stay fp32; the cast cost is inside the timing); refused on cpu. `compile`: `torch.compile` of the rollout function and of the predictor, regions a and b only, planner horizon only; its first-call time is the compile time; a recompile per K shape is expected (`dynamo.config.cache_size_limit` raised to 64). A variant that fails is written as `NOT RUN` with the error text. The max absolute state difference of a variant against eager (K = 8, planner horizon) is recorded, information only.
8. **Real-time factor** = simulated seconds rolled (H x 0.2) / wall seconds of the batched rollout at p50.
9. **100 ms share.** The tick table also prints the arithmetic share of the PI's 100 ms budget (Thor, 4 cameras). The encoder is not in this tick and the GPU is not Thor, so it is context, not a pass or a fail.
10. **No overwrite.** Output files are opened with mode x; an existing result gives exit 4.

## Self-checks (gate; a failure writes SELF_CHECKS_FAILED.json only, exit 2)

* s1 the timer reads a known 50 ms sleep within 5 ms (p50 of 20); s1a the same against an asynchronous stub device (work is done at `sync`), which catches a missing synchronisation.
* s2 batched and sequential rollouts agree within 1e-4 and the states are not all zero (largest K, planner horizon).
* s3 the batched rollout at twice the horizon takes 1.6 to 2.4 times as long (K = 8, or the first K of the list).
* s4 the counted predictor calls of a rolling tick are above 0 and equal on two ticks of the same input (the first input that rolls).
* s5 the statistics of two known lists (1..100 ms and 1,1,1,1,100 ms).

## Mutation list (run_checks_pod.py; each must turn a check red, and the harness must read green again after it)

M1 unit off by 1000; M2 timed function never called; M3 no synchronisation (async stub); M4 constant offset of 30 ms; M5 p95 is the maximum; M6 p50 is the mean; M7 call counter does not count; M8 counter never reset; M9 rollout ignores the horizon; M10 batched rollout drops a step; M11 both rollouts return zeros; M12 one repeated sample instead of n (sync count); plus a changed first line caught by the writer test. The runner also tests the gate (exit 2, no table), the no-overwrite refusal (exit 4), both header lines of the MD and the JSON, and that the MD holds none of the banned words.

## Limits

A proxy on the pod GPU; kernel timings of an L40-class card are not Thor timings (the Thor engine path is TensorRT with its own batching). Batch 1 only for the tick. Host-side numpy work of the planner is inside the tick time and is part of what the planner costs today. The encoder (about 54 ms of the earlier line's 60 ms on Thor, INHERITED from the scoping card) is not measured here.
