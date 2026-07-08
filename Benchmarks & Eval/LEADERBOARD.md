# TanitAD Leaderboard

> Our checkpoints vs published competitor/SOTA numbers. Every number carries source + date +
> eval-condition footnote (gate G-B1). Maintained by the Benchmarks & Eval agent (Thursdays).

> **⚠ Open-loop ⊥ closed-loop (standing footnote, G-B1).** arXiv 2605.00066 (Apr-2026, 15 methods):
> ADE/FDE have **no reliable correlation** with closed-loop Driving Score, and NAVSIM PDMS correlates
> *non-monotonically* with Bench2Drive DS (ranking inversions; paired n=8). **Never rank a TanitAD
> checkpoint on an open-loop number alone.** Open-loop and closed-loop blocks below are kept separate.

## Open-loop (NAVSIM v2 EPDMS, navtest/navhard) — context numbers, not yet re-run by us

| System | EPDMS | Source / date | Note |
|---|---|---|---|
| HAD | 88.6 | arXiv 2604.03581, 2026-04 | diffusion + metric-decoupled RL |
| SOTA claim (survey) | 89.3 | arXiv 2606.19641, 2026-06 | self-play scaling paper's citation |
| Drive-JEPA | 93.3 (PDMS, NAVSIM **v1**) | arXiv 2601.22032, 2026-01 | v1 metric — not comparable to EPDMS |
| PDM-Closed (baseline) | 51.3 (EPDMS, **navhard**) | arXiv 2506.04218 / leaderboard 2026-03 | pseudo-sim (3DGS aug), R²≈0.8 vs CL; navhard ≠ navtest |
| **TanitAD** | — | — | Phase 1 target: first entry |

## Closed-loop (Bench2Drive, CARLA — the arbiter block) — context numbers

| System | Driving Score | Success Rate | Source / date | Note |
|---|---|---|---|---|
| TF++ (VLAAD-MIL) | 86.97 | 71.97 % | arXiv 2603.25946 (context), 2026 | Bench2Drive-220, 1 safety-critical scenario/route |
| ADT | 77.90 | 55.0 % | Bench2Drive leaderboard, 2026 | Bench2Drive220, low-latency |
| **TanitAD** | — | — | — | Phase 1 target; MetaDrive closed-loop first (G0.5), then CARLA/Bench2Drive |

> Eval-condition note: Bench2Drive = 220 short (~150 m) routes, 44 interactive categories × 23 weathers ×
> 12 towns. **CARLA seed variance ≈ 5 DS** same-model → our closed-loop rows will report mean ± CI over
> ≥3 seeds; a "beats baseline" claim requires separated CIs.

## TanitAD internal gate ladder (Phase 0)

| Gate | Status | Value | Exp-ID | Date |
|---|---|---|---|---|
| D1 (probe ADE) | **FAIL** (mid-training preview, step 5k/30k) | ADE@1s 10.94 m (bar < 1.0) | p0-sB01-gates-step5000 | 2026-07-08 |
| D2 (imagination ranking) | **PASS** (mid-training preview, step 5k/30k) | dir-acc P1 **0.872** / P4 **0.940** (bar 0.7, chance 0.5); imag-rel 9.73 diagnostic — the A13 pattern ON REAL DATA | p0-sB01-gates-step5000 | 2026-07-08 |
| D3 (imagined-ADE ratio) | **BLOCKED** (I4 3.83 — doctrine refuses the claim) | @0.4 s horizon | p0-sB01-gates-step5000 | 2026-07-08 |
| D4 (tactical lift) | not run | — | — | — |
| D5 (routing edge) | not run | — | — | — |
| D6 (generalization slope) | not run | — | — | — |
| D8 (OOD AUROC) | not run (gate) — **paired preview at step 6.5k**: 16/23 same-scene clear→degraded pairs show higher imagination error (p≈0.047); unpaired scores ~chance | `stack/experiments/p0-d8-preview/` | d8-preview-sc05-v2 | 2026-07-08 |
| D9 (H15 hidden-sector imagination) | not run | — | — | — |

## Efficiency ledger (CNCE inputs)

| Checkpoint | Params | FLOPs/decision | Batch-1 latency 4060 | Orin projection | Date |
|---|---|---|---|---|---|
| p0-sB01 step 6500 | 262.8 M | TBD | **decision tick 15.07 ms p50** (encode 9.38 + K9 select 5.69; predictor pass 6.14; p95 ≈ 17.2), fp32 un-optimized, peak VRAM **1.08 GB**, strict numerics, I8 batch-1 | TBD (Phase 1; TRT/quant headroom on top of ~66 Hz fp32) | 2026-07-08 |

**First real TMS/CNCE rows** (diagnostic; `stack/experiments/p0-latency-baseline/`): CNCE on 12
comma-val log replays with the measured tick = median **2.02×10⁵ m/(s·B-params)** (collisions=0 by
construction — log replay, footnote stands); TMS expert-log reference band median 0.044 (min 0.024,
max 0.083) — a *reference band* for later closed-loop comparison, not a policy claim. LAL/OKRI/LOPS
remain unmeasured pending CARLA occluder telemetry (W31–32) — P8: no telemetry, no number.
