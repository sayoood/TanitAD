# TanitAD Leaderboard

> Our checkpoints vs published competitor/SOTA numbers. Every number carries source + date +
> eval-condition footnote (gate G-B1). Maintained by the Benchmarks & Eval agent (Thursdays).

## Open-loop (NAVSIM v2 EPDMS, navtest) — context numbers, not yet re-run by us

| System | EPDMS | Source / date | Note |
|---|---|---|---|
| HAD | 88.6 | arXiv 2604.03581, 2026-04 | diffusion + metric-decoupled RL |
| SOTA claim (survey) | 89.3 | arXiv 2606.19641, 2026-06 | self-play scaling paper's citation |
| Drive-JEPA | 93.3 (PDMS, NAVSIM **v1**) | arXiv 2601.22032, 2026-01 | v1 metric — not comparable to EPDMS |
| **TanitAD** | — | — | Phase 1 target: first entry |

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
