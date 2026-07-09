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
| HAD | 88.6 | arXiv 2604.03581, 2026-04 | diffusion + metric-decoupled RL (navtest) |
| SOTA claim (survey) | 89.3 | arXiv 2606.19641, 2026-06 | self-play scaling paper's citation (navtest) |
| Drive-JEPA | 93.3 (PDMS, NAVSIM **v1**) | arXiv 2601.22032, 2026-01 | v1 metric — not comparable to EPDMS |
| DrivoR (test-time opt) | **56.3** (EPDMS, **navhard**) | arXiv 2606.07170, 2026-06 | navhard #1; test-time trajectory optimization |
| DriveFuture | **55.5** (EPDMS, **navhard**) | arXiv 2605.09701, 2026-04 | navhard #1 *learned*; future-aware latent WM |
| PDM-Closed (baseline) | 51.3 (EPDMS, **navhard**) | arXiv 2506.04218 / leaderboard 2026-03 | pseudo-sim (3DGS aug), R²≈0.8 vs CL; navhard ≠ navtest |
| **TanitAD** | — | — | Phase 1 target: first entry |

> EPDMS (NAVSIM v2) extends PDMS with **compliance sub-metrics** — Driving-direction Compliance (DDC),
> Traffic-line Compliance (TLC), Lane-Keeping (LK) — and splits comfort into History/Extended Comfort
> (HC/EC) ([navsim/docs/metrics.md](https://github.com/autonomousvision/navsim/blob/main/docs/metrics.md),
> 2026). DDC/TLC/LK are the recognizable analogue of our H9 violation-rate / closure-incursion signal.

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
| p0-sB01 step 6500 — **precision sweep** | 262.8 M | — | **fp16 ~1.6× / bf16 ~1.8× vs fp32** (speedup *ratios*; absolute Hz withheld — 4060 was CarlaUE4-contended this run). **Accuracy Δ (contention-immune, 64 real windows):** fp16 imagine-select agreement **95.3 %** / enc rel-err 7.8e-4 / wp-shift 3.9 cm; bf16 **67.2 %** / 7.2e-3 / 47.7 cm | **Deploy fp16, not bf16** (Prod&Opt precision policy); TRT-fp16 acceptance bar pre-registered | 2026-07-09 |

**First real TMS/CNCE rows** (diagnostic; `stack/experiments/p0-latency-baseline/`): CNCE on 12
comma-val log replays with the measured tick = median **2.02×10⁵ m/(s·B-params)** (collisions=0 by
construction — log replay, footnote stands); TMS expert-log reference band median 0.044 (min 0.024,
max 0.083) — a *reference band* for later closed-loop comparison, not a policy claim.

## Live scenario metrics — SC-01 Work-Zone Phantom (first light, 2026-07-08)

> First run of the custom suite on **live CARLA physics** (`stack/scripts/carla_work_zone.py`, pod2
> nullrhi, real dynamics + raycast occlusion + measured tick). ⚠ **Weak rows:** policies are *scripted
> archetypes* (NOT our checkpoint — camera-driven ego is host-blocked), **single seed**, and **LAL-v1 is
> non-discriminative here** (superseded by LAL-v2, intake `2026-07-09-lal-v2-anticipation`). Not an edge
> claim; replaced when the checkpoint-driven, ≥3-seed rollout lands. Source
> `stack/experiments/p0-carla-workzone/suite_results_v1.json`.

| Policy (scripted) | OKRI ↓ | LOPS ↑ | TMS ↑ | CNCE ↑ | LAL-v1 | Note |
|---|---|---|---|---|---|---|
| reactive (E2E-like) | 32.37 | 0.00 | 0.006 | 8.68e5 | −0.7 | holds cruise into blind edge; no latent track |
| world_model (anticip.) | **12.83** | **0.834** | 0.023 | 1.06e6 | −0.7 | eases off early; holds noisy latent estimate |

- **OKRI 2.5× lower** for the anticipating policy (12.83 vs 32.37) — throttles kinetic energy into the
  blind spot. Seed-power (audit Result C): the 19.5 gap separates at 95 % with ~2 seeds if per-seed SD≈5,
  but OKRI's SD must be **measured** on the ≥3-seed re-run (units ≠ DS scale).
- **LOPS 0.834 vs 0.0**: independently recomputed (audit Result B) — 0.834 matches the analytic
  E=0.8325 of the injected σ=0.3 tracking-noise model (inside the 95 % CI for all n_occ) → *reproducible*.
  **But** the 0.0 is structural (no-estimate policy scores 0 by definition), so this proves latent-track
  **presence, not quality**; and it reflects an injected noise model, not our real model (P8).
- **LAL-v1 = −0.7 for both** → superseded. On a realistic comfort-bounded anticipatory ease-off (|jerk|
  < ~2 m/s³) LAL-v1's −1.5 m/s³ trigger never fires; **LAL-v2** returns +0.3…+3.1 s anticipation lead vs
  −0.3 s reactive (audit Result A). Next SC-01 run reports LAL-v2.

## Competitor efficiency (CNCE differentiation — W-05) — context numbers

> Operationalizes weakness **W-05** (our 261 M vs multi-billion competitors). Params are *active/total as
> published*; compute class differs by deployment (offline generative WM vs on-car policy) — **not an
> apples-to-apples score**, a parameter/compute-envelope comparison only (G-B1). Sourced to the Opponent
> Analyzer profiles (`TanitAD Research Hub/Opponent Analyzer/OPPONENT_PROFILES.md`, 2026-07).

| System | Params | Deployment class | Source / date |
|---|---|---|---|
| NVIDIA Alpamayo-2 | **32 B** | on-car VLA policy | Opponent profiles, 2026-07 (W-05) |
| Wayve GAIA-3 | **15 B** | offline generative world model | Opponent profiles, 2026-07 |
| DriveFuture | (latent WM) | NAVSIM-v2 open-loop planner | arXiv 2605.09701, 2026-04 |
| **TanitAD-4B-M** | **261 M** (measured) | on-car hierarchical latent WM + tactical | `stack/`, step-6500 ckpt; 15.07 ms tick / 1.08 GB @ 4060 fp32 |

The efficiency wedge (CNCE) is credible only *at matched safe-progress*: a 32 B / 15 B model must buy
proportionally more safe metres to justify its denominator. TanitAD's live decision tick (15 ms, 1.08 GB,
un-optimized fp32) is the honest numerator-per-compute anchor; the closed-loop CNCE comparison is Phase 1.
