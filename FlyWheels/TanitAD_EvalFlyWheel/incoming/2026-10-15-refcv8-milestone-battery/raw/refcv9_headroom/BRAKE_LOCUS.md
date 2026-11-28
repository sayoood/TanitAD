# BRAKE LOCUS: where is refcv8's cruise false-brake decided?  (HEAD / SELECTION / CANDIDATE)

**LABEL (banked data): UNDETERMINED**  |  HEAD test (association): **FAIL**  |  SELECTION vs CANDIDATE: **NOT MEASURABLE from banked dumps (the refined fan is not banked)**

* tier: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)**; refcv8 EMA-30k (`step30000ema_unc`) vs refcv7-50,400; surface = S6 compliant windows (3,251 / 138 ep), GT-cruise 1,856; evidence class MEASURED (ours), CPU only, no model forward.
* estimator: episode-cluster bootstrap (pair_read_r8.paired_cell -> taniteval.ci), B 2000, seed pair_read_r8.SEED: answers 'another draw of EPISODES' only; NOT another training run (single training seed per arm, H-ESTIM-SEED-1), NOT another inference run (seeds 0 and 1 shown separately, never pooled).
* pre-registered rule: `raw/refcv9_headroom/PREREG_BRAKE_LOCUS.md` (sha256 `157bc6e5da5c` sealed before any locus number was computed; Amendment 1 `876a044eccd0`, also sealed before the first run: it corrects the `sel_bank_*` inventory row, adds S7 and the ALONG-RAND row); script sha256 `65fc6c26c060`; wall 9 s.
* **Pre-seal look disclosed in the prereg**: head-class counts were inspected before sealing, so the HEAD clause outcomes were not blind.

## 1. Inventory: what the banked dumps carry

| item | banked? |
|---|---|
| refined fan (all candidates, full trajectories) | NO (only terminal heading [W,149] + reach mask) |
| per-candidate selection scores | NO (only the max, sel_score_max) |
| candidate tags (lon6 / alloc / kind) | NO |
| selected index, pre-graft index, GT-nearest base anchor | yes (decisions/sel_idx, sel_idx_base, a_star) |
| selected plan, 8 slots | yes (decisions/plan_full_nav_true) |
| tactical lon head logits (tacv6 = planner-feeding; lon_logits_tac = legacy) | yes (refcv6_extras.npz) |
| head constraint outputs (cons_lon8) | NO |
| oracle_sel (refined GT-nearest plan) | S2 grid only (0-2 s) |
| sel_bank_* (selected anchor's UNREFINED 8-slot path) | yes, for windows with sel_idx < 117 only (rows = those windows in order; AMENDMENT 1 A1 corrects the original prereg's inadmissible); S6 files carry the S2 array unsliced |
| ablated arms (filter_off, vmax_zero) | seed 0 only |

## 2. HEAD test (the pre-registered rule, per inference seed)

| seed | (i) share of r8 false-brake windows with head in D [CI] | clause 1 (>= 0.6) | clause 2: r8 - r7 false-brake rate on r8-head-says-cruise windows (paired) [CI] | literal (<= 0) | not-separated-worse (lo <= 0) | P1 permutation null (97.5th pct) |
|---|---|---|---|---|---|---|
| 0 | 0.655 [0.513, 0.765] (n 493) | True | +0.0437 [+0.0190, +0.0711]  (r8 0.124 vs r7 0.080, n 1374) | False | False | obs 0.655 vs null 0.292 -> True |
| 1 | 0.649 [0.500, 0.764] (n 478) | True | +0.0429 [+0.0181, +0.0699]  (r8 0.122 vs r7 0.079, n 1374) | False | False | obs 0.649 vs null 0.295 -> True |

**HEAD test: FAIL.**  Clause 1 holds ((i) = 0.655 / 0.649 at seeds 0 / 1, chance 0.259) but clause 2 does not: where r8's own head says cruise, r8 brakes 0.124 vs refcv7's 0.080 (paired delta +0.044 [+0.019, +0.071], seed 1 +0.043 [+0.018, +0.070]).  The label stays UNDETERMINED: SELECTION vs CANDIDATE needs the refined fan.  What the banked data DO say (association): 81 % / 81 % of the r8-minus-r7 false-brake excess sits in windows where r8's head says decel; r8's plan follows its own decel head 0.67 / 0.64 of the time against refcv7's 0.12 / 0.13, although refcv7's head says decel MORE often on GT-cruise windows (0.36 vs 0.26) -- the difference is the head-to-plan coupling, not the head's decel rate.  The braking is already in the SELECTED UNREFINED anchor in 86 % / 89 % of in-bank false-brake windows (vs 7 % on other GT-cruise windows), so it is decided in WHICH ANCHOR is chosen, not in refinement; the ceiling filter, the speed-label input and the E9 graft are not the locus (S5, S4).  POST-HOC: where the selector did pick the GT-nearest base anchor on a false-brake window, that anchor itself brakes in 63 % / 67 % of cases (base-bank coverage is thin there, for refcv7 too).  NOT separable from banked data: whether the head CAUSES the pick (needs the force_lon F2 / head-neutral-ranking interventions), and whether the refined fan holds a good cruising candidate (needs the fan).

## 3. Descriptive supplements (not part of the rule)

| seed | head-decel share on GT-cruise: r8 / r7 | P(plan brakes \| own head in D): r8 / r7 | P(plan brakes \| own head NOT in D): r8 / r7 | excess r8-r7 false-brake rate | of which r8-head-in-D stratum / not-D stratum |
|---|---|---|---|---|---|
| 0 | 0.260 / 0.360 | 0.670 [0.545, 0.768] (n 482) / 0.118 [0.075, 0.172] (n 668) | 0.124 [0.087, 0.166] (n 1374) / 0.076 [0.047, 0.114] (n 1188) | +0.1746 | +0.1422 (81 %) / +0.0323 (19 %) |
| 1 | 0.260 / 0.360 | 0.643 [0.507, 0.749] (n 482) / 0.132 [0.086, 0.193] (n 668) | 0.122 [0.085, 0.165] (n 1374) / 0.072 [0.044, 0.108] (n 1188) | +0.1638 | +0.1320 (81 %) / +0.0318 (19 %) |

Selection provenance (S4, descriptive).  `sel_idx >= 117` = the selected candidate is one of the 32 APPENDED (head-allocated) candidates; `sel == a_star` = the selected candidate is the GT-nearest BASE-bank anchor; `sel != base` = the E9 goal-selection graft moved the pick (CORRECTED by Amendment 2 P-1: the core pick `sel_idx_base` already includes the refcv8 head-coupled ranking term).

| seed | population | n | sel >= 117 | sel == a_star | sel != sel_idx_base (E9 graft moved it) |
|---|---|---|---|---|---|
| 0 | r8 r8_false_brake | 493 | 0.105 | 0.185 | 0.144 |
| 0 | r8 r8_gt_cruise_not_false_brake | 1363 | 0.061 | 0.647 | 0.111 |
| 0 | r8 r8_false_brake_head_in_D | 323 | 0.108 | 0.133 | 0.108 |
| 0 | r8 r8_false_brake_head_not_D | 170 | 0.100 | 0.282 | 0.212 |
| 0 | r7 sel == a_star: false-brake / other GT-cruise | - | - | 0.290 / 0.706 | - |
| 1 | r8 r8_false_brake | 478 | 0.079 | 0.180 | 0.151 |
| 1 | r8 r8_gt_cruise_not_false_brake | 1378 | 0.067 | 0.657 | 0.103 |
| 1 | r8 r8_false_brake_head_in_D | 310 | 0.094 | 0.135 | 0.106 |
| 1 | r8 r8_false_brake_head_not_D | 168 | 0.054 | 0.262 | 0.232 |
| 1 | r7 sel == a_star: false-brake / other GT-cruise | - | - | 0.282 / 0.711 | - |

Seed-0 banked ablation arms (S5): false-decel rate native 0.266, ceiling filter OFF 0.263, speed label zeroed 0.257; the ceiling filter changes the selected candidate on 1.2 % of r8 false-brake windows; head-in-D share on GT-cruise native vs label-zeroed 0.260 / 0.266.

Sensitivities (S6): seed 0: D+FOLLOW (i) 0.819, clause-2 delta +0.0255; head==CRUISE only delta +0.0392; legacy head (i) 0.702, delta +0.0429; unpaired (each arm's own head-says-cruise) r8 0.124 vs r7 0.076; seed 1: D+FOLLOW (i) 0.816, clause-2 delta +0.0299; head==CRUISE only delta +0.0480; legacy head (i) 0.703, delta +0.0421; unpaired (each arm's own head-says-cruise) r8 0.122 vs r7 0.072.


Learned ranking-term scalars in the EMA-30k checkpoint (MEASURED, `BRAKE_LOCUS_ckpt_scalars.json`): beta_lat -0.0388, beta_lon -0.0068, gamma_prog +0.1010, gamma_head +0.3811.  beta_lon is ~0: the head's lon POSTERIOR hardly enters the ranking, so the argmax-based HEAD test is a proxy; the head can reach the pick through its lon CONSTRAINT output (gamma_prog, NOT banked), through candidate allocation (appended candidates are chosen on ~10 % of false-brake windows) or through shared scene features (common cause).


### S7 (Amendment 1 A2): is the braking already in the SELECTED UNREFINED anchor?  (windows with sel_idx < 117 only)

| seed | r8 false-brake windows in-bank (share of all r8 FB) | unrefined selected anchor ALREADY decel | unrefined cruise/accel but REFINED decel | ... when head in D / not in D (bank already decel) | r8 GT-cruise non-FB: bank decel | r7 GT-cruise: bank decel | r8 - r7 bank-decel, paired [CI] |
|---|---|---|---|---|---|---|---|
| 0 | 441 (0.895) | 0.864 | 0.136 | 0.861 / 0.869 | 0.069 | 0.132 | +0.141 [+0.094, +0.194] |
| 1 | 440 (0.921) | 0.889 | 0.111 | 0.868 / 0.925 | 0.068 | 0.138 | +0.136 [+0.086, +0.191] |

### POST-HOC (Amendment 2, written after run 1; exploratory): is the GT-nearest BASE anchor itself braking where the model selected it?

| seed | r8 false-brake, sel == a_star, in-bank: n / a_star bank path decel [CI] | r8 GT-cruise non-FB, sel == a_star: n / decel | r7 false-brake, sel == a_star: n / decel | r7 non-FB: n / decel |
|---|---|---|---|---|
| 0 | 91 / 0.626 [0.494, 0.750] | 882 / 0.034 | 49 / 0.755 | 1191 / 0.048 |
| 1 | 86 / 0.674 [0.529, 0.812] | 905 / 0.040 | 49 / 0.776 | 1196 / 0.049 |

### Row ALONG-RAND (coordinator add-on; sibling `refcv9/L1_DESIGN_ANNEX.md` 1.2 'SELECTION' verdict): NOT MEASURABLE -- the refined fan is not banked.  Random-candidate (`RAND`) and median-progress (`MED`) alignment r / g* on the same S2 compliant windows, both seeds, need a re-roll; definitions, reading rule (SELECTOR-ADDS-VALUE iff `r_os - r_RAND` separated above 0 at both seeds, SELECTION-SUPPORTED iff not) and controls are pre-registered in `PREREG_BRAKE_LOCUS_AMENDMENT1.md`, the statistics are implemented and self-tested (control C7), the design is `BRAKE_LOCUS_REROLL_DESIGN.md`.

## 4. Controls

* **ALL_PASS: True** (C1 True, C2 True, C3 True, C4 True, C5 True, C6 True, C7 True).
* C1 literals (from `L2_HEADROOM.json`): false-decel r8 0.265625 / 0.257543, r7 0.091056 / 0.093750, n 3,251 / 138 and 1,856 / 135: {'s0_n_compliant': True, 's0_n_cruise': True, 's0_fd_r8': True, 's0_fd_r7': True, 's1_n_compliant': True, 's1_n_cruise': True, 's1_fd_r8': True, 's1_fd_r7': True}.
* C2 alignment + head join (extras `nav_true.traj` bit-equal `os`, so a mis-join cannot pass), C3 identities, C4 decision-function arms (oracle head, never-decel head, rule-logic table), see JSON.
* C5 mutations: m1_decel_threshold_1p0: RED; m2_head_join_shifted_one_window: RED; m3_CRUISE_BRAKE_TO_columns_swapped: RED.
* C7 along-track statistics self-test (K1 / K2 / K3 known values, g* mutation RED, iid residual r ~ 0, verdict logic): pass True.
* C6 fan function self-test (synthetic fan, known E 0.25 / 0.5 / 0.25): pass True; mutations X<0 RED True, X huge RED True.

## 5. Caveats

* single TRAINING seed per arm: the intervals are episode draws only and are blind to training variance (H-ESTIM-SEED-1); the two inference seeds are the replicate for sampler noise and are not pooled.
* T1 open loop on logged eval139 frames: a statement about plans scored against the logged future, not about driving.
* ASSOCIATION, not intervention.  In refcv8 the head reaches the planner through candidate allocation, the ranking score (beta_lon * log p) and the constraint terms, so HEAD and SELECTION are coupled by design; only an intervention (force_lon F2, head-neutral ranking S0) separates them.
* the speed-label input differs between refcv7 (per-clip continuous v_max_ms) and refcv8 (4-class road-type label): every refcv8-vs-refcv7 row crosses that input change.
* the head argmax is a hard decision; head confidence, the constraint outputs and the allocation histogram are not used here.
* the two tactical lon heads (tacv6, legacy lon_logits_tac) disagree on 22.6 % of windows; the primary is tacv6 (the planner-feeding head, the one H-R8-S6-LON-1 used); the legacy head is a sensitivity.
* (ii) / (iii) are degenerate as worded (see prereg section 0); the fan stage is defined and self-tested but cannot run on banked data.
* S7 covers only in-bank selections (sel_idx < 117); the share is reported.  The unrefined bank path of the selected anchor is the residual family rolled from the prior (v0-conditioned); its class uses the same +-1.5 m/s rule on the last chord (5 s -> 6 s).
