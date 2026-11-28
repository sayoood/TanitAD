# P2-s1 read: p2s1_full  ->  **NOT REPLICATED**

> ZZP2REP_READ p2s1_full NOT REPLICATED RR1 +0.90 [+0.36, +1.59] adverse=TTC | F=0.33 dseed +0.33 [+0.04, +0.64] D_s0=+0.57 D_s1=+0.90 gates_ok=True

Tier: T1 = NAVSIM v1.1 PDMS, an ego pseudo-simulation of an open-loop plan against logged agents (self-action open loop, not closed loop).
Estimator: paired log-cluster bootstrap over the 136 navtest logs (cluster unit log_name), 10,000 resamples, 95 %, seed 20260927 (proxy_eval.boot via stage3_full.read_pair). It answers 'another draw of LOGS'. Delta_seed is ONE draw of the run-to-run difference of two training seeds: F is a one-draw estimate, never a variance (PREREG_P2_REPLICATE section 2).

Claim on P2's lever: NOT supported: P2's +0.57 lies within run-to-run variance.  Next: the next lever is the NOT PROVEN row of amendment section 4: SFT-4R (scorer-only on complete labels), then a seed replicate of the better arm (Rule Zero: never a stopping point).

## Six sub-scores (x100, paired, 95 % log-cluster CI)

| sub-score | RR1: P2-s1 - u400 | adverse | RR2: P2-s1 - P2-s0 | P2-s0 - u400 (banked) |
|---|---|---|---|---|
| PDMS | +0.90 [+0.36, +1.59] |  | +0.33 [+0.04, +0.64] | +0.57 [+0.02, +1.18] |
| NC | -0.17 [-0.36, +0.03] |  | -0.04 [-0.16, +0.09] | -0.13 [-0.31, +0.08] |
| DAC | +0.65 [+0.11, +1.37] |  | +0.35 [+0.09, +0.64] | +0.30 [-0.23, +0.89] |
| EP | +1.80 [+1.33, +2.41] |  | +0.65 [+0.33, +1.01] | +1.14 [+0.61, +1.70] |
| TTC | -0.49 [-0.97, -0.07] | ADVERSE | -0.18 [-0.47, +0.09] | -0.31 [-0.80, +0.14] |
| C | +0.02 [+0.00, +0.05] |  | +0.01 [+0.00, +0.03] | +0.01 [+0.00, +0.03] |
| DDC | -0.07 [-0.36, +0.23] |  | +0.08 [-0.07, +0.24] | -0.16 [-0.43, +0.13] |

PDMS means x100: u400 88.2469, P2-s0 88.8184, P2-s1 89.1493.
RR2: F = |+0.33| = 0.3309; registered comparison F < min(D_s0, D_s1) = 0.5715: True.
Mean of the two seeds' Delta vs u400: +0.7369.

## Families (RR3; reported, not gating)

- s1_vs_u400: 21 longitudinal/lateral components compared; adverse: none. Tactical rows (no interval): see JSON.
- s1_vs_p2_s0: 21 longitudinal/lateral components compared; adverse: none. Tactical rows (no interval): see JSON.
- STRATEGIC: PAUSED by the PI (2026-09-19, PI_DECISION_QUEUE item 25) and UNAVAILABLE in NAVSIM (no strategic decision is scored); reported as such, never as a pass.

## Gates

- PASS  pin:p2_s0_csv
- PASS  pin:u400_csv
- PASS  pin:tokens
- PASS  stage3:gates_ok
- PASS  stage3:tokens
- PASS  stage3:tokens_minus_c1
- PASS  stage3:base_pinned
- PASS  stage3:also_pinned
- PASS  stage3:harness_status
- PASS  stage3:seam_report
- PASS  stage3:checkpoint
- PASS  stage3:rows
- PASS  stage3:consistency_vs_base
- PASS  stage3:r2b_equals_stage2_primary
- PASS  stage3:not_dry_run
- PASS  stage3:name
- PASS  stage3:same_table
- PASS  stage3:checkpoint_sha256_was_checked
- PASS  estimator_pinned
- PASS  rows
- PASS  finite
- PASS  s1_not_s0
- PASS  s1_not_u400
- PASS  p2s0_vs_u400_reproduces_banked_R2
- PASS  rr1_equals_stage3_R2

## Inputs

- s1_csv: E:\Projects\TanitAD\data\refe_navtest\score\refe_stage2_p2s1_full\refe_stage2_p2s1_full.csv
- s1_csv_md5: 8dce98b80b7254f24d8816f9b35656fc
- p2_s0_csv: E:\Projects\TanitAD\TanitAD Research Lab\Architecture & Inference\Research\2026-09-20-refe-plan\raw\p2_readout\p2_full\arm\refe_stage2_p2_full.csv
- u400_csv: E:\Projects\TanitAD\data\refe_navtest\score\refe_stage2_sft4_B_u400_navtest_full\refe_stage2_sft4_B_u400_navtest_full.csv
- tokens: E:\Projects\TanitAD\data\refe_navtest\navtest_full_tokens.json
- r2_result: E:\Projects\TanitAD\TanitAD Research Lab\Architecture & Inference\Research\2026-09-20-refe-plan\raw\p2_readout\p2s1_full\r2_result.json
- families: ['None', 'None', 'None']
