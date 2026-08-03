# INTAKE — Thor batch-9 predictor engine + the first end-to-end measured tick (2026-08-03)

**Owner:** Production & Optimization. **Box:** `thor6` (Blackwell sm_110, aarch64), torch 2.13.0+cu130,
TensorRT 10.13.3.9, `~/venvs/tanitad-edge`. **Cost $0** (owned hardware).
**Write-up:** `Research/2026-08-03-thor-batch9-engine-and-wired-tick.md`.
**Runbook:** annotated in place (`THOR_DEPLOYMENT_RUNBOOK.md`, block before §4 + §4's procedure).
**Retraction:** `Project Steering/RETRACTION_LOG.md` — `R-2026-08-03-e`.

## What landed in `stack/` (needs review, then merge)

| file | change |
|---|---|
| `stack/tanitad/models/fourbrain.py` | `TacticalSelector.propose_and_score(..., batch_fan=False)` + `_fan_batched`. **Default is unchanged**, so no existing caller moves. |
| `stack/tests/test_flagship4b.py` | 3 tests: batched == loop (index + score tol), **K calls not N×K**, loud refusal of a ragged vocabulary |
| `stack/scripts/build_predictor_trt.py` | **new** — export + build + verify-by-loading for the predictor engine. Dynamic batch profile, `--intent-dim`, `set_fastpath_enabled(False)`, and a `TRTPredictor` that **raises rather than silently dropping the intent token** |

`pytest -q` on `stack/`: **1722 passed, 12 skipped, 2 xfailed** (was 1719 + 3 new).

## Raw results (this directory)

| file | what |
|---|---|
| `thor_d1_batch9_engine.{py,json}` | engine builds + profile verification + batch sweep 1..9 (in-process **and** `trtexec`) + the fan measured 4 ways |
| `thor_d2_full_tick.py`, `thor_d2_full_tick_K20.json` | first END-TO-END wired tick (4 arms, p50/p95/p99) + batch-consistency + compounding + a first decision gate ⚠️ whose engine lacked the intent input |
| `thor_d4_decision_decomposition.{py,json}` | the one-factor-at-a-time decomposition that located the wiring bug |
| `thor_d5_selector_regret.{py,json}` | agreement **and regret** for: batching alone, the correct engine, and the intent-less engine as a deliberate control |
| `thor_d6_tick_intent_engine.{py,json}` | the tick re-measured on the **intent-carrying** engine |

## Artifacts that stayed on the box (too large for git — rebuild recipe, not a copy)

| thor path | what |
|---|---|
| `~/trt_deploy/predictor_v1_intent_dyn1-9_fp16.plan` | ⭐ **the deployment engine** — real v1 step-29999 weights, dynamic batch 1..9, **intent input**, fp16 (rel-err 3.66e-4/3.67e-4, `intent_is_live` 0.0522) |
| `~/trt_deploy/predictor_v5f_intent_dyn1-9_fp16.plan` | ⭐ the same for the **176×624 deployed geometry** arm (rel-err 8.49e-4/6.06e-4, `intent_is_live` 0.0470) |
| `~/trt_deploy/predictor_v1_dyn1-9_fp16.plan`, `…_v5f_dyn1-9_fp16.plan` | **no intent input** — kept only as the control arm for R-2026-08-03-e |
| `~/trt_deploy/MANIFEST.md` | provenance + supersession + rebuild command (copy in this dir as `TRT_DEPLOY_MANIFEST.md`) |
| `~/trt/predictor_fp16.plan` | ⛔ **superseded** — batch-1 **and random-weight**. Kept, not deleted |

Rebuild (~38-40 s, 174 MB): `stack/scripts/build_predictor_trt.py --ckpt <weights> --out <path>
--max-batch 9 --fp16 --intent-dim 256`; add `--v2-subframe 176x624` for v5f, without which the
**STRICT** load correctly REFUSES the checkpoint (`encoder.pos` 429 vs 256).

## ⛔ Escalations (not "please merge" buried in a README)

1. **The batched fan is a DEPLOYMENT REQUIREMENT, not a nicety.** Whoever wires the selector into a
   driver must pass `batch_fan=True`; a batch-9 engine with the serialised caller measures *worse*
   than the batch-1 engine.
2. **`TacticalSelector` has no production caller today** (two probes). The only closed-loop driver,
   `closedloop_drive.py::FlagshipV1Policy.plan`, is heads-only. The 100 ms budget in the runbook
   prices the fan path, which is designed but not yet wired — that gap belongs to Architecture &
   Inference, not to this stream.
3. **The four-family accuracy gate does not need re-running for the batch change** (b9 vs b1 =
   1.57e-4, half the fp16 error) — but it *was* run through an **intent-less** engine on 2026-08-03,
   and `rollout_decode` does not pass intent either. ⇒ **the existing gate measures the
   unconditioned predictor**. That is a question for its owner: is the deployed rollout meant to be
   intent-conditioned (it is in `propose_and_score`) or not (it is not in `rollout_decode`)?
