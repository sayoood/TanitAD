# S1 read (`s1_read/1`)

**Status: INCOMPLETE.** Tier: T0-like: held-out selection and perception on teacher-forced banked sets (S1_POD_RUNBOOK l.5); never driving performance. Estimator: paired log-cluster bootstrap, 10000 resamples, 95 %, seed 20260927 (aux_s1_readout / proxy_eval.boot's draw scheme). It answers 'would another draw of LOGS say this?' ONLY; training-run variance is the N-vs-N seed floor (H-ESTIM-SEED-1); inference variance: REFe scoring is deterministic, ASSERTED not measured.

Registered arms (9): present and clean ['N-s0', 'N-s1', 'N-s2']; missing ['H1-s0', 'H1-s1', 'H1-s2', 'H2-s0', 'H2-s1', 'H2-s2']; refused none.

> S1 prereg A5: the first read happens after all registered arms exist; no outcome label (K2_*, H2_*, P3_*, *_PROMOTED) is emitted from a partial arm set

## Reads

| read | status | arms needed | arms missing | prereg clause |
|---|---|---|---|---|
| SEED_FLOOR | READY | N-s0, N-s1, N-s2 | - | S1 prereg 4.3 l.108 + draft 'Replicates': floor = largest |delta| between two N seeds, measured only with all three N runs; l.126: fewer than three N => capped |
| SEL_H1_s0 | INCOMPLETE | H1-s0, N-s0, N-s1, N-s2 | H1-s0 | S1 prereg 4.3 l.108: each seed pair Hx-sK minus N-sK on held-out pick PDMS and pick DAC; comparator of record = the SAME-seed N (section 2 l.48) |
| SEL_H2_s0 | INCOMPLETE | H2-s0, N-s0, N-s1, N-s2 | H2-s0 | S1 prereg 4.3 l.108: each seed pair Hx-sK minus N-sK on held-out pick PDMS and pick DAC; comparator of record = the SAME-seed N (section 2 l.48) |
| SEL_H1_s1 | INCOMPLETE | H1-s1, N-s0, N-s1, N-s2 | H1-s1 | S1 prereg 4.3 l.108: each seed pair Hx-sK minus N-sK on held-out pick PDMS and pick DAC; comparator of record = the SAME-seed N (section 2 l.48) |
| SEL_H2_s1 | INCOMPLETE | H2-s1, N-s0, N-s1, N-s2 | H2-s1 | S1 prereg 4.3 l.108: each seed pair Hx-sK minus N-sK on held-out pick PDMS and pick DAC; comparator of record = the SAME-seed N (section 2 l.48) |
| SEL_H1_s2 | INCOMPLETE | H1-s2, N-s0, N-s1, N-s2 | H1-s2 | S1 prereg 4.3 l.108: each seed pair Hx-sK minus N-sK on held-out pick PDMS and pick DAC; comparator of record = the SAME-seed N (section 2 l.48) |
| SEL_H2_s2 | INCOMPLETE | H2-s2, N-s0, N-s1, N-s2 | H2-s2 | S1 prereg 4.3 l.108: each seed pair Hx-sK minus N-sK on held-out pick PDMS and pick DAC; comparator of record = the SAME-seed N (section 2 l.48) |
| K2_H1_s0 | INCOMPLETE | H1-s0 | H1-s0 | draft K2 + P1; S1 prereg 4.1 table l.91-96 (P1 row l.93: navtest P1 on the 1,058 C3 tokens; P1-HO is the fallback) and l.98 (navtest governs) |
| P3_H1_s0 | INCOMPLETE | H1-s0 | H1-s0 | S1 prereg 3 R2 (pooled AUC), R3 (H2_PROVISIONAL), A2.1 l.154-156 (paired floor), A2.3 (fp32 planner ctx) |
| K6_H1_s0 | INCOMPLETE | H1-s0 | H1-s0 | draft K6: aux_bce >= 0.9 x its step-0 value after 10 % of the arm's steps => stop |
| K3_H1_s0 | INCOMPLETE | H1-s0, N-s0 | H1-s0 | draft K3: measured step time rises > 3 % or peak memory > 1 GB; proposal 2 / S1_POD_RUNBOOK S1-D6 (read afterwards from the arm evidence) |
| K5_H1_s0 | INCOMPLETE | H1-s0 | H1-s0 | draft K5: the zero-weight / detached exact controls must be bit-identical (the G12 preflight fields); the shuffle limb cannot fire (no C-shuffle arm, S1 prereg 1 l.39) |
| K2_H2_s0 | INCOMPLETE | H2-s0 | H2-s0 | draft K2 + P1; S1 prereg 4.1 table l.91-96 (P1 row l.93: navtest P1 on the 1,058 C3 tokens; P1-HO is the fallback) and l.98 (navtest governs) |
| P3_H2_s0 | INCOMPLETE | H2-s0 | H2-s0 | S1 prereg 3 R2 (pooled AUC), R3 (H2_PROVISIONAL), A2.1 l.154-156 (paired floor), A2.3 (fp32 planner ctx) |
| K6_H2_s0 | INCOMPLETE | H2-s0 | H2-s0 | draft K6: aux_bce >= 0.9 x its step-0 value after 10 % of the arm's steps => stop |
| K3_H2_s0 | INCOMPLETE | H2-s0, N-s0 | H2-s0 | draft K3: measured step time rises > 3 % or peak memory > 1 GB; proposal 2 / S1_POD_RUNBOOK S1-D6 (read afterwards from the arm evidence) |
| K5_H2_s0 | INCOMPLETE | H2-s0 | H2-s0 | draft K5: the zero-weight / detached exact controls must be bit-identical (the G12 preflight fields); the shuffle limb cannot fire (no C-shuffle arm, S1 prereg 1 l.39) |
| K2_H1_s1 | INCOMPLETE | H1-s1 | H1-s1 | draft K2 + P1; S1 prereg 4.1 table l.91-96 (P1 row l.93: navtest P1 on the 1,058 C3 tokens; P1-HO is the fallback) and l.98 (navtest governs) |
| P3_H1_s1 | INCOMPLETE | H1-s1 | H1-s1 | S1 prereg 3 R2 (pooled AUC), R3 (H2_PROVISIONAL), A2.1 l.154-156 (paired floor), A2.3 (fp32 planner ctx) |
| K6_H1_s1 | INCOMPLETE | H1-s1 | H1-s1 | draft K6: aux_bce >= 0.9 x its step-0 value after 10 % of the arm's steps => stop |
| K3_H1_s1 | INCOMPLETE | H1-s1, N-s1 | H1-s1 | draft K3: measured step time rises > 3 % or peak memory > 1 GB; proposal 2 / S1_POD_RUNBOOK S1-D6 (read afterwards from the arm evidence) |
| K5_H1_s1 | INCOMPLETE | H1-s1 | H1-s1 | draft K5: the zero-weight / detached exact controls must be bit-identical (the G12 preflight fields); the shuffle limb cannot fire (no C-shuffle arm, S1 prereg 1 l.39) |
| K2_H2_s1 | INCOMPLETE | H2-s1 | H2-s1 | draft K2 + P1; S1 prereg 4.1 table l.91-96 (P1 row l.93: navtest P1 on the 1,058 C3 tokens; P1-HO is the fallback) and l.98 (navtest governs) |
| P3_H2_s1 | INCOMPLETE | H2-s1 | H2-s1 | S1 prereg 3 R2 (pooled AUC), R3 (H2_PROVISIONAL), A2.1 l.154-156 (paired floor), A2.3 (fp32 planner ctx) |
| K6_H2_s1 | INCOMPLETE | H2-s1 | H2-s1 | draft K6: aux_bce >= 0.9 x its step-0 value after 10 % of the arm's steps => stop |
| K3_H2_s1 | INCOMPLETE | H2-s1, N-s1 | H2-s1 | draft K3: measured step time rises > 3 % or peak memory > 1 GB; proposal 2 / S1_POD_RUNBOOK S1-D6 (read afterwards from the arm evidence) |
| K5_H2_s1 | INCOMPLETE | H2-s1 | H2-s1 | draft K5: the zero-weight / detached exact controls must be bit-identical (the G12 preflight fields); the shuffle limb cannot fire (no C-shuffle arm, S1 prereg 1 l.39) |
| K2_H1_s2 | INCOMPLETE | H1-s2 | H1-s2 | draft K2 + P1; S1 prereg 4.1 table l.91-96 (P1 row l.93: navtest P1 on the 1,058 C3 tokens; P1-HO is the fallback) and l.98 (navtest governs) |
| P3_H1_s2 | INCOMPLETE | H1-s2 | H1-s2 | S1 prereg 3 R2 (pooled AUC), R3 (H2_PROVISIONAL), A2.1 l.154-156 (paired floor), A2.3 (fp32 planner ctx) |
| K6_H1_s2 | INCOMPLETE | H1-s2 | H1-s2 | draft K6: aux_bce >= 0.9 x its step-0 value after 10 % of the arm's steps => stop |
| K3_H1_s2 | INCOMPLETE | H1-s2, N-s2 | H1-s2 | draft K3: measured step time rises > 3 % or peak memory > 1 GB; proposal 2 / S1_POD_RUNBOOK S1-D6 (read afterwards from the arm evidence) |
| K5_H1_s2 | INCOMPLETE | H1-s2 | H1-s2 | draft K5: the zero-weight / detached exact controls must be bit-identical (the G12 preflight fields); the shuffle limb cannot fire (no C-shuffle arm, S1 prereg 1 l.39) |
| K2_H2_s2 | INCOMPLETE | H2-s2 | H2-s2 | draft K2 + P1; S1 prereg 4.1 table l.91-96 (P1 row l.93: navtest P1 on the 1,058 C3 tokens; P1-HO is the fallback) and l.98 (navtest governs) |
| P3_H2_s2 | INCOMPLETE | H2-s2 | H2-s2 | S1 prereg 3 R2 (pooled AUC), R3 (H2_PROVISIONAL), A2.1 l.154-156 (paired floor), A2.3 (fp32 planner ctx) |
| K6_H2_s2 | INCOMPLETE | H2-s2 | H2-s2 | draft K6: aux_bce >= 0.9 x its step-0 value after 10 % of the arm's steps => stop |
| K3_H2_s2 | INCOMPLETE | H2-s2, N-s2 | H2-s2 | draft K3: measured step time rises > 3 % or peak memory > 1 GB; proposal 2 / S1_POD_RUNBOOK S1-D6 (read afterwards from the arm evidence) |
| K5_H2_s2 | INCOMPLETE | H2-s2 | H2-s2 | draft K5: the zero-weight / detached exact controls must be bit-identical (the G12 preflight fields); the shuffle limb cannot fire (no C-shuffle arm, S1 prereg 1 l.39) |
| H1_IDENTITY_s0 | INCOMPLETE | H1-s0, N-s0 | H1-s0 | S1 prereg 4.3 l.124: H1's scorer is bit-identical to N by construction; any H1-vs-N difference in selection is a defect, investigated before any read |
| H1_IDENTITY_s1 | INCOMPLETE | H1-s1, N-s1 | H1-s1 | S1 prereg 4.3 l.124: H1's scorer is bit-identical to N by construction; any H1-vs-N difference in selection is a defect, investigated before any read |
| H1_IDENTITY_s2 | INCOMPLETE | H1-s2, N-s2 | H1-s2 | S1 prereg 4.3 l.124: H1's scorer is bit-identical to N by construction; any H1-vs-N difference in selection is a defect, investigated before any read |
| PROMOTE_H2 | INCOMPLETE | H2-s0, H2-s1, H2-s2, N-s0, N-s1, N-s2 | H2-s0, H2-s1, H2-s2 | S1 prereg 4.3 l.110-118: conditions 1-4 (l.111-114), otherwise-not-promoted (l.116), C-shuffle amendment (l.118) |
| PROMOTE_H1 | INCOMPLETE | H1-s0, H1-s1, H1-s2, N-s0, N-s1, N-s2 | H1-s0, H1-s1, H1-s2 | S1 prereg 4.3 l.120-122 |
| FAMILIES | INCOMPLETE | N-s0, N-s1, N-s2, H1-s0, H1-s1, H1-s2, H2-s0, H2-s1, H2-s2 | H1-s0, H1-s1, H1-s2, H2-s0, H2-s1, H2-s2 | draft 'Four families' + S1 prereg 4.2 l.100-104 (navtest families not computed at S1; STRATEGIC PAUSED, PI 2026-09-19 item 25) |
| S1_OVERALL | INCOMPLETE | N-s0, N-s1, N-s2, H1-s0, H1-s1, H1-s2, H2-s0, H2-s1, H2-s2 | H1-s0, H1-s1, H1-s2, H2-s0, H2-s1, H2-s2 | S1 prereg A5 l.298: 'The first S1 read happens after all registered arms exist'; A3.3 l.233-237: all nine registered arms run, both H1 replicates are reported |

## Component (a rig property of the N arms; NOT an outcome)

Seed floor, measured=True (n N seeds 3): `{"DAC": 0.25502072043353524, "EP": 0.5462196365954735, "NC": 0.22314313037934333, "PDMS": 0.3524454096270321, "TTC": 0.22314313037934333}`

## Families (binding; per family, never pooled)

| family | status at S1 | not computed |
|---|---|---|
| LONGITUDINAL | EP and TTC sub-scores of the PICK on the 3,137 held-out sets (paired H - N, same seed) | speed MAE/bias, along-track MAE/bias, headway/time-gap (navtest families instrument; the planner seam does not apply H2's feed, S1 prereg 4.2) |
| LATERAL | DAC sub-score of the PICK on the held-out sets (paired) | heading / cross-track / curvature / yaw-rate MAE (navtest families instrument; same reason) |
| TACTICAL | selection quality as pick / random / best-of-64 on the held-out sets, per arm (the random control is the arm's own fan mean) | lateral/longitudinal/5-way decision accuracy and kappa, command adherence (navtest families instrument); the paired CI of (actual - random) is not produced by the harness |
| STRATEGIC | PAUSED (PI 2026-09-19, PI_DECISION_QUEUE.md item 25: switched off temporarily until tactical and operative drive with high quality; reported as PAUSED, not a pass, not omitted, not extended to the other three) | - |

## Prereg ambiguities (none resolved silently)

| id | where | text | used | not used |
|---|---|---|---|---|
| AMB-1 | S1 prereg 4.3 cond 1 (l.111) and cond 2 (l.112) | 'the P3 decision on C3 is H2_EVALUATED or H2_PROVISIONAL' - there are three H2 runs, each with its own P3 decision; the text has one decision | all three H2 seeds must read EVALUATED or PROVISIONAL; the regime is PROVISIONAL (3 of 3 pairs) if ANY seed is PROVISIONAL | at least 2 of 3 seeds in {EVALUATED, PROVISIONAL} |
| AMB-2 | S1 prereg 4.3 H1 paragraph (l.120-122) | 'K2 does not fire on H1's head' / 'K6 did not fire' - there are three H1 runs | no K2 / K6 on ANY of the three H1 seeds (the phrase 'any H2 run' in cond 4 is the model) | H1-s0 only (the S1_POD_RUNBOOK S1-D3 drop rule reads H1-s0) |
| AMB-3 | draft K6 | 'aux_bce >= 0.9 x its step-0 value after 10 % of the arm's steps' - which logged number is the step-0 value and which row is 'after 10 %' | step-0 value = G12 bce_declared (the head's BCE on a real batch before any update); the row = the first 'train' event with update >= ceil(0.10 x 400) = 40, i.e. update 50 (a 25-update mean) | the update-25 row; or ln 2 as the step-0 value |
| AMB-4 | draft K3 | 'step time rises > 3 % or peak memory > 1 GB at micro-batch 4' - the S1 argv uses --batch 8; the comparison arm and the memory quantity are not named | memory: G12 head_retained_gb > 1.0 OR preflight gpu_mem_gb(H) - gpu_mem_gb(N same seed) > 1.0; time: run-mean s_per_update at update 400, H vs the same-seed N, > +3 %; either limb fires K3 | N = the mean of the three N seeds; micro-batch 4 timing (not measured at S1) |
| AMB-5 | draft K5 | 'the zero-weight / detached exact controls are not bit-identical to the comparator' is a check on an exact test, not on a banked number | read from the arm's PREFLIGHT_PASS.json G12 raw fields (a head arm cannot have trained without them); a missing or false field REFUSES the arm (an evidence-integrity failure, not an outcome); the shuffle limb is NOT_APPLICABLE | treat a K5 failure as a fifth promotion condition |
| AMB-6 | S1 prereg 4.3 (pick PDMS / pick DAC) | the held-out selection rule is not named in the S1 prereg | navsim_v1 (the draft's rule; the harness docstring's invocation; selection_pair passes L=None, which navsim_v1_lane cannot use) | navsim_v1_lane |
| AMB-7 | S1 prereg 4.3 l.124 | 'investigated before any read' - blocks which reads | an H1 identity defect makes the H1 outcome H1_IDENTITY_DEFECT (no promotion label) and is stamped on the H2 outcome; the H2 conditions are still computed | withhold the H2 outcome as well |
| AMB-8 | S1 prereg 3 R3 / A2.1 | the P3 decision names are H2_*; an H1 arm also gets a P3 number from the harness | the P3 DECISION is taken for H2 arms only; an H1 arm's P3 is reported, undecided (H1's promotion rule does not use P3) | decide P3 on H1 too |
| AMB-9 | S1 prereg 4.3 cond 2 + the harness classify() | 'separated and above the seed floor': floor = largest |delta| between two N seeds of the paired MEAN delta; the test is the pair's CI lower bound > floor | the harness's classify() (re-derived here and compared); the capped label 'separated, single seed, not yet a lever claim' counts as NOT above the floor | - |

## Not computed

- E1 / E2 / E3 on the full navtest (PDMS, DAC-zero share, zeroed share): the planner seam does not apply H2's feed (S1 prereg 4.2); H2's verdict is a HELD-OUT SELECTION verdict, never an ADOPT
- inference variance: REFe scoring is deterministic, ASSERTED by a second pass of the same checkpoint, not measured here
- C-shuffle: not an S1 arm (Master Mind ruling 2026-10-13) => every H2 gain carries 'capacity not excluded'; K5's shuffle limb cannot fire
- navtest PDMS for the H arms; the speed / along-track / heading / cross-track / curvature / yaw-rate / headway / decision-accuracy components of the four families (they need the navtest planner seam)

## Timings (s)

```
{
 "fetch": 47.0,
 "listing": 0.56,
 "selection": 21.58,
 "total": 71.68,
 "untar_pdm": 1.0
}
```
