# VETO_PREREG: does P7's own drivable-area head, used as a veto or a fixed-weight score term over P7's 64 proposals, improve the picked proposal's harness PDMS / DAC? (written 2026-11-17, BEFORE any per-proposal harness score of P7 was computed or read)

Status of what existed when this was written: P7 proposals + scorer logits (12,146 tokens), head predictions (1,058 C3 tokens, 3 seeds), banked per-token harness CSV of P7's pick. NOT computed, NOT looked at: any harness score of any P7 proposal other than its pick; any head-vs-proposal footprint statistic; any veto outcome. After this file is saved only a dated amendment appended at the end may change it.

## 1. Population and data (index by token string, never by row position)
- Tokens: the 1,058 C3 tokens (`.../raw/2026-10-11-c3-selection-headroom/tokens.json`, sha256 03072566...e2254d), 123 logs. Strata from that file: `zero` = 529 tokens whose **u400** pick had NC x DAC = 0 (census), `random` = 529 drawn uniformly from the 11,617 other tokens.
- P7 proposals and scorer logits: `E:/Projects/TanitAD/data/refe_navtest/seams/stage2_p7_full.proposals.npz` (proposals [12146,64,8,3], logits [12146,64,6], rule `navsim_v1`).
- Head: `p7_next/headiou/main/head_pred_H1-s0.npz` (trained head; `logit` [1058,40,32], 2 m cells, x0 -8, y0 -32). Seeds s1 (zeroed) and s2 (fresh init) are not used except as the reported-only floor controls in 7.
- Ground truth: harness (NAVSIM v1.1 PDMS, `score_e.py`, UNCHANGED, W3 pins) score of EVERY P7 proposal on the 1,058 tokens: 64 single-proposal seams, the same method as `raw/2026-10-11-c3-selection-headroom/c3_run.py stage_score`. The existing C3 table is u400's proposals and cannot be reused (verified: 0 of 67,712 P7 proposals equal any u400 proposal of the same token).

## 2. Arms (same tokens, same table)
- **A** = P7's actual pick: argmax of the `navsim_v1` selection score NC*DAC*(5EP+5TTC+2C)/12 on sigmoid(logits) (`rl_val_eval.select`). Must equal the dump's `pick` on 1,058/1,058.
- **Raster**: the head's drivable MASK = `logit > 0` (the AUXVETO convention); signed distance by Euclidean distance transform of the mask on the 2 m grid: for a cell inside, `+2.0 * (edt_to_outside_cell - 0.5)` m; outside, `-2.0 * (edt_to_inside_cell - 0.5)` m (edt in cell units, scipy.ndimage.distance_transform_edt); if the mask is all-inside or all-outside the field is +6 m / -6 m. Sampled at footprint corners by `geodac.geodac_corners` (bilinear, `unknown="drivable"`, `PRED_GEOM`), footprint and 41-step 0.1 s densification exactly as `raw/2026-10-16-auxveto-feasibility/geodac.py` (Pacifica 2.588 / 1.1485 / centre +1.461 m).
- **B (veto)**: a proposal is VETOED if the minimum corner signed distance over its 41 steps is `< -0.25 m` (the AUXVETO margin, fixed there before any outcome, not tuned here). The pick is the highest-selection-score proposal among the NON-vetoed; if all 64 are vetoed, the pick is A.
- **C (score term)**: pick = argmax of `sel + lambda * f_in`, with `sel` the navsim_v1 selection score above and `f_in` the share of the 41 x 4 corner samples with signed distance > 0 (unknown counts as inside), **lambda = 0.5 fixed**. lambda is not varied for the decision.
- **N (control)**: the identical B and C computations with the head rasters PERMUTED across tokens (numpy default_rng(20261117) permutation of the 1,058 head maps, redrawn until no token keeps its own map and no token receives a map from its own log). Reported as N_B and N_C. The proposals and scorer stay each token's own.
- Oracle reported only: **O** = veto proposals whose harness DAC = 0 (picks the best-score survivor), the ceiling of any DAC veto on this proposal set.

## 3. Endpoint and estimator
- Endpoint: harness PDMS x100 of the picked proposal, and harness DAC x100, from the table (lookup, no model).
- **Primary estimand: the navtest-population delta** = (529/12,146) * mean_zero(delta) + (11,617/12,146) * mean_random(delta), the stratified estimator of PREREG_AUXVETO section 5 (the C3 set is failure-enriched; the unweighted pooled 1,058 mean is reported alongside, labelled DESCRIPTIVE).
- Interval: paired log-cluster bootstrap over the 123 logs, 10,000 resamples, seed 20260927, 95 % percentile, resampling logs with the stratum weights fixed (`proxy_eval.boot` scheme). It answers "another draw of LOGS" only; it is blind to head TRAINING variance (one head, H-ESTIM-SEED-1) and to scorer training variance. Reported deltas: B-A, C-A, N_B-A, N_C-A, each for PDMS and DAC.

## 4. Validity gates (any failure => the read is INVALID, reported as such, not interpreted)
1. All 64 columns PASS the harness with 1,058 valid rows; table finite.
2. The table at A's pick reproduces the banked per-token score `refe_stage2_p7_full.csv` for all 1,058 tokens (max abs diff <= 1e-9). (Done for u400 in C3: diff 0.0.)
3. A reproduces the dump's `pick` 1,058/1,058.
4. Zero control: an all-drivable raster vetoes nothing and delta B-A = C-A = 0.0 exactly (lambda term constant).
5. Head file tokens equal the table tokens, in the same set.
6. Mutation check: forcing the veto to select the WORST surviving proposal must make the delta negative (instrument is wired to the endpoint).

## 5. Committed outcomes (decision list, read top-down, on the primary PDMS deltas, unrounded)
| row | condition | reading | next |
|---|---|---|---|
| **VH-GO** | (B-A lower bound > 0 AND N_B-A lower bound <= 0) OR (C-A lower bound > 0 AND N_C-A lower bound <= 0) | the head's own map adds information over P7's picks; **the H2 feed-on arm is worth a card** (still "separated, one head seed, not a lever claim") | H2 scorer-only feed-on arm with N control + replicate, as in P7_HEAD_IOU.md |
| **VH-CONTROL** | any of N_B-A, N_C-A has lower bound > 0 | a shuffled map "helps": the effect is a veto-by-geometry or reward artifact, not the head's content; no H2 card from this read | audit what the generic veto does (path-length / curvature prior) before any head claim |
| **VH-HARM** | B-A and C-A both have upper bound < 0 | false vetoes cost more than recoveries | head errors sit where the scorer fails; 1 m grid / `--w-field` arm is next |
| **VH-NULL** | otherwise (neither separated above 0, no harm) | the head as it stands does not move the pick; **H2 feed-on is not justified by this read**, and the oracle row says whether any DAC veto could (if O-A also has lower bound <= +0.3 PDMS the DAC headroom itself is the limit) | 1 m grid / `--w-field` arm (head quality) or stop |
Two tests (B, C) share one rule: a GO needs one of them separated; the family-wise false-positive rate is therefore up to ~2x the nominal 5 %; this is stated here and the result is worded accordingly. A point estimate below +0.10 PDMS is called "separated but small" even if its bound clears 0.

## 6. Reported with every read (four families; STRATEGIC paused earlier, PI 2026-11-16 unpaused: STRATEGIC cannot be computed from this table, no route label per proposal: stated, not silently omitted)
- LONGITUDINAL: delta EP, delta TTC. LATERAL: delta DAC, delta DDC. TACTICAL (selection quality): random / A / best-of-64 / each arm PDMS, per stratum, plus Z-stratum recovery (zero-score picks of A that become non-zero) and the false-veto rate (A's pick vetoed although its harness DAC = 1). Also delta NC, delta comfort; share of tokens where all 64 are vetoed; share of tokens where B changes the pick.

## 7. Reported only (never gating)
margin curve m in {0, 0.5, 1.0, 2.0} for B; lambda in {0.25, 1.0, 2.0} for C; the head's `sdf_m` channel as the field instead of the mask-EDT; head seeds s1 (zeroed, must veto nothing: expected equals A or, with an empty mask, vetoes everything and falls back to A) and s2 (fresh init) as floor controls; per-stratum deltas; the oracle O.

## 8. Admissibility
Vision-only: the veto reads the head raster (camera-derived) and the proposals: admissible at inference. The oracle O reads harness DAC: never deployable. Goal/situation rules: the head is not the goal and not the situation classifier. The head reads the planner's fp32 context (deployment path, `P7_HEAD_IOU.md`).

## 9. Budget gate
If the 64-column harness scoring is estimated above ~4 CPU-h, stop and report a plan (estimate recorded in HANDOFF.md before the scoring starts).

## Amendment 1 (2026-11-17, before any P7 proposal score was read)
The shuffle in section 2 ("redrawn until no token receives a map from its own log") cannot be done by rejection sampling: with 123 logs of about 8.6 tokens each the acceptance probability is effectively zero. Replaced, with the same properties and the same seed 20261117: order the tokens by a random permutation of the logs (random order inside a log) and give token i in that order the map of the token `max_log_size` places later (cyclic). Every token gets another token's map, from another log; the map multiset is unchanged. Nothing else changes.
