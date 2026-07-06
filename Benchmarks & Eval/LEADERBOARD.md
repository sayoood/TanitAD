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
| D1 (probe ADE) | not run | — | — | — |
| D2 (imagination ranking) | not run | — | — | — |
| D3 (imagined-ADE ratio) | not run | — | — | — |
| D4 (tactical lift) | not run | — | — | — |
| D5 (routing edge) | not run | — | — | — |
| D6 (generalization slope) | not run | — | — | — |
| D8 (OOD AUROC) | not run | — | — | — |
| D9 (H15 hidden-sector imagination) | not run | — | — | — |

## Efficiency ledger (CNCE inputs)

| Checkpoint | Params | FLOPs/decision | Batch-1 latency 4060 | Orin projection | Date |
|---|---|---|---|---|---|
| — | — | — | — | — | — |
