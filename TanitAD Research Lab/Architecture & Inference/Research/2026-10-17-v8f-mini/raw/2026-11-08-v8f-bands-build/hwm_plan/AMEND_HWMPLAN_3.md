# AMEND_HWMPLAN_3 (DRAFT for the Master Mind's freeze): LON-PLAN, the HOLD-anchored REG(λ) dose on R35, selected on TRAIN windows before any SCORE window is planned

**Status.** DRAFT, 2026-11-16, Arch+Inference agent. Nothing here was run, scored or trained. No LON-PLAN window, plan or token exists. The Master Mind (MM) freezes it and writes the λ* line into the Log before any SCORE window is planned at λ ≠ 0.

**Frozen inputs this amends** (paths under `raw/2026-11-08-v8f-bands-build/hwm_plan/` unless named):
* `PREREG_V8F_HWMPLAN_DRAFT.md` `ac20d690`, `FREEZE_HWMPLAN.md` `ae2bbe40`, `AMEND_HWMPLAN_1.md` `dcbd55a9` (world model R35), `AMEND_HWMPLAN_2.md` `b9a7602a` (G3 `l1_full`), `IMPL_CHOICES_HWMPLAN.md` `c478d6cf`.
* ACTRSP card `raw/2026-11-16-v8f-actrsp-prereg/PREREG_V8F_ACTRSP.md` `ce2d02df`, sections 4.3 and 4.4.
* ACTRSP Amendment 1 `.../amend1/AMENDMENT_1_BCAL_DRAFT.md`, frozen at `4e122aa5…` (commit `76a8012f`): section 2.2 item 2 names this lever and its blocker.
* LAT-TOK `.../lattok/RESULT_LATTOK.md` (commit `45e6c130`): FAIL, admissible.

**Tier.** Every planner row here is **T0**: the goal is the encoded true-future latent and the planner has no trained part. A T0 row is a world-model diagnostic and never a driving-performance number. HOLD and PRIOR are T1-kin references. The TRAIN selection record (section 3) is a T0 diagnostic on windows the world model was trained on. It is a selection record, never a result.

## 0 What changes (the whole amendment, in one table)

| item | frozen today | this amendment |
|---|---|---|
| solver of the LON-PLAN arms | `seed_mode prior`, `lam 0` (FREEZE fill) | `--seed-mode hold`, `--lam λ`, named in every argv; REG refuses without the HOLD seed (`hwm_plan.py:420-421`) |
| λ selection data | the card's slot: PLANTEST2's TUNE half (an eval half) | **TRAIN windows of R35's own TRAIN latents only**; no eval window is read for selection |
| arms | phase 1, G3 | LON-PLAN arms A0, A1, R, S (section 4), HWM-ORACLE only, on R35_s0 and R35_s1 |
| deciding read | full 6 s band | **0-2 s band, LONGITUDINAL `acc_mae`**, as ACTRSP section 4.2 reads level 1's own plan |
| everything else | | unchanged: windows, priors, levers at their defaults, families, estimator, controls 1-15, the solver of record of every other arm |

## 1 Why, from measurements

* **The planner is the limit at T0, and the gap sits at 0-2 s too.** Phase 1 on R35 read PLANNER-LIMITED at both chassis. In the 0-2 s band, where level 1 plans and no decode runs, HWM-ORACLE `acc_mae` is 1.2877 [1.2160, 1.3582] / 1.1888 [1.1194, 1.2560] m/s², against HOLD's 0.2970 / 0.2971 (HOLD = arm minus its contrast row). HWM − HOLD is +0.9907 [0.9249, 1.0566] / +0.8917 [0.8178, 0.9634]. MEASURED: `phase1_R35/RESULT_HWMPLAN_R35_s{0,1}.md`. The G3 arm `l1_full` stays PLANNER-LIMITED on the decode-free path (MEASURED: `phase2_l1full/VERDICT_HWMPLAN_L1FULL.md`).
* **R35's acceleration response is not deficient.** Read at the 2 s endpoint through RA, the readout with clip-held-out transition skill, R35's response is G_aa 1.244 [1.116, 1.376] (s0) and 1.202 [1.066, 1.333] (s1) of the real one. MEASURED in ACTRSP Amendment 1, section 1.5. So on acceleration, the world model responds about as strongly as the real transitions. The longitudinal question moves to the planner (Amendment 1 section 2.2 item 2).
* **The planner leaves the data.** Phase-1 plans sit 1.1-1.6 prior-std from HOLD (VERDICT phase 1, HOLD-COLLAPSE off). Planned level-2 macros lie 6.77 / 6.79 times farther from the TRAIN macro cloud than held-out in-image macros, and 94 % of their decodes hit the action bounds. MEASURED: `AMEND_HWMPLAN_2.md`, `phase1_R35/decode_check/`.
* **The only REG grid read so far stopped before the regime asked about here.** PLANTEST2 on M2b, a different world model: MAE fell with λ in every row of the grid {0, 0.01, 0.03, 0.1, 0.3, 1}. The largest value, λ = 1, was chosen in all 8 cells. At λ = 1 the plans still sat 1.067-1.442 prior-std from HOLD. TD3 s0 TUNE `acc_mae` went from 1.4100 at λ 0 to 1.2174 at λ 1. MEASURED: `raw/2026-11-09-v8f-plantest2/RESULT_PLANTEST2.md`, TUNE table and lines 137-139. The grid holds no interior optimum, and its top is far from HOLD.
* **HYPOTHESIS H-LONPLAN.** Near HOLD, R35's level-1 search direction toward the goal carries acceleration information, because the 2 s response is calibrated (1.2×). Far from HOLD, the optimiser exploits the world model where it is off its data. A HOLD-anchored prior strong enough to keep plans inside the calibrated neighbourhood can then lower 0-2 s `acc_mae` below HOLD's. This needs a dose range that reaches close to HOLD-COLLAPSE, which M2b's grid did not.
* **What is expected, stated before any data (ESTIMATED).** The CEILING outcome below needs 0-2 s `acc_mae` to fall by about 75 %, from 1.19-1.29 to under 0.297. M2b's grid lowered it by at most 14 %. So CEILING is not the expected outcome. The read that decides most is the matched-shrinkage control S (section 4): does REG's solve beat a plain pull toward HOLD at the same distance?

## 2 The lever as built (MEASURED from the code of record)

* **What REG is.** `reg_term = λ · mean over steps and BOTH channels of ((a − HOLD_c) / prior_std)²` (`plantest2_lib.py:155-158`). The reference is `seeds[1]` = HOLD_c, the logged a0 held and clipped to the level-1 prior box (`hwm_plan.py:162-166`, `:424-425`). Amendment 1 calls it "the prior-regularised planner". Precisely, it is a **HOLD-anchored** regulariser measured in prior-std units. It does not pull toward the prior mean.
* **Both channels.** The term pulls yaw rate toward HOLD as well as acceleration. So LON-PLAN also moves the LATERAL family, which is read as a guard (section 6).
* **Level 1 only** (`plantest2_lib.py:129`). REG also enters candidate selection, because the argmin is taken over the full cost (`:137-138`).
* **Geometry on R35** (MEASURED, `Geometry.from_stack` with strides (3, 5), `n_op` 10, `n_traj` 30): K1 = 4, so level 1 plans 12 knots and REG spans all 12. Knots 1-10 enter the trajectory. Knots 11-30 come from the level-2 decode, which reads level 2's macros and HOLD_c, never level 1's actions (`hwm_plan.py:212-224`). Levels 3 and 2 are planned before level 1 and draw their random numbers first.
  * ⇒ **For HWM-ORACLE with `stitch decode`, knots 11-30 are bit-identical between λ and λ = 0 at the same inference seed.** That identity is a control (C-L2).
  * ⇒ The per-knot metrics (`acc_mae`, `yaw_mae`) in the 2-6 s band read exactly 0.0 for A1 − A0. Integrated metrics (speed, heading, cross-track, ADE, lead metrics, tactical classes) carry the 0-2 s change forward and are reported.
* **FLAT and `l1_full`.** REG spans all 30 knots there. Neither is a LON-PLAN arm; both are named as next steps (section 6).
* **HOLD versus HOLD_c.** REG pulls toward HOLD_c. When a0 lies outside the prior box, the λ → ∞ plan is HOLD_c and not HOLD, so A1 − HOLD then stays slightly above 0 at the limit.

## 3 The dose: a TRAIN-only selection procedure, fixed now

### 3.1 Selection data (MEASURED today, CPU, no eval file opened for selection)
* **Source.** `F:/TanitAD-scratch/hwm_plan/latents_train_R35_s{0,1}.pt`, sha256 `76a2d4f5…` / `692eb002…`. Both equal the phase-1 record (`phase1_R35/HANDOFF_HWMPLAN_PHASE1.md` lines 8-9).
  * Layout: `Z1[R35_s{k}]` (2,397, 16, 128), `A_raw` (2,397, 16, 2), `ok` (2,397, 16), `t`, `eid`.
  * 141 TRAIN clips. **0 of them are among the 141 eval clips** (`wmact/latents.pt`).
  * These are the same latents the priors are fitted on (FREEZE fill 2).
* **Frame table.** The 16-frame windows overlap. A frame's latent and actions are identical in every window that holds it: max |Δz| = 0.0 and max |Δa| = 0.0 over 11,690 unique (clip, frame) entries, at each chassis.
* **Planning windows.** NOW = g, with frames g … g+30 all present and `ok` (31 frames: T3 = 2 puts the goal at frame 30, and `n_traj` is 30).
  * 5,486 such NOW positions exist, 34-39 per clip, and every clip has at least 4.
  * **Selection set: 4 NOW positions per clip.** They are the valid positions at indices `unique(round(linspace(0, n_valid − 1, 4)))`, so 564 windows. The set is the same at both chassis.
* **Same inputs and code as the SCORE read.**
  * Goal: `OracleGoal(z[:, 30], 30)`, as `run_hwm.goal_fn_for`.
  * Priors: `hwm_data.fit_priors` on the same file, as `run_hwm.prepare`.
  * Planner: built by `run_hwm.solver_from` / `make_planner` from an argparse namespace that `run_hwm.add_common` parses from the run's own argv, so hops 1-5 of section 8.1 are the SCORE run's code.
  * Run settings: inference seed 0, `P.BATCH`, `OMP_NUM_THREADS=2`. `v0` is not read: the criterion uses logged accelerations only, and the constraints are off.
* **Declared bias (HYPOTHESIS).** R35 was trained on these clips. Its rollouts are therefore more accurate here than on eval clips, so the TRAIN optimum may sit at a smaller λ than the held-out optimum. That shifts which λ is read. It does not touch the SCORE read's validity.

### 3.2 Grid and extension rule
* **Base grid**, per chassis: λ ∈ {0, 0.3, 1, 3, 10, 30, 100}, all with `--seed-mode hold`.
* **Sign check:** λ = −1 is run beside the grid. It is never a candidate.
* **Extension up.** If the chosen λ is the largest value run so far, and its plans are not collapsed (D ≥ 0.10, section 3.3), run 300, 1000 and 3000 in that order. Stop at the first value whose C is above the running minimum or whose D < 0.10. At most 3 extra values.
* **Extension down.** If the chosen λ is 0.3, run 0.1, 0.03 and 0.01 in that order. Stop at the first value whose C is above the running minimum. At most 3 extra values.
* The procedure reads TRAIN windows only. It is fixed here and is not changed after any value is seen.

### 3.3 Criterion, admissibility, tie rule
* **C_k(λ)**: the mean over the 564 windows of the 0-2 s `acc_mae`, the `hwm_eval.band_metrics` expression (`hwm_eval.py:94`): mean over knots 1-10 of |plan acc − logged acc|.
* **D_k(λ)**: the mean |plan acc − HOLD acc| over the windows and knots 1-10, in units of the TRAIN prior std. This is `plantest2_lib.hold_distance` / `std_raw_of`, restricted to knots 1-10. D_yaw is the same on the yaw channel.
* **Admissible:** λ > 0 and D_k(λ) ≥ 0.10. This is the card's HOLD-COLLAPSE threshold, and a collapsed plan is not planning.
* **λ\*_k = the admissible λ with the lowest C_k.** C_k is rounded to 4 decimals (0.0001 m/s²). A tie goes to the smaller λ (PLANTEST2's `choose_lambda` rule, `plantest2_lib.py:214-216`, plus the rounding).
* **If no λ > 0 is admissible,** the downward extension runs until one is. If none is admissible after it, the label is LAMBDA-COLLAPSE (section 6).
* **Reported beside on TRAIN, no bar:** C_k(0), C_k(HOLD), C_k(PRIOR), D and D_yaw per λ, DIR-ACC per λ (section 5), and the sign check C_k(−1) − C_k(0).

### 3.4 Record and freeze
* The selection writes `LAMBDA_SELECTION_R35_s{k}.json`. It holds the grid as run, C, D, D_yaw, the rule trace, λ\*_k, every argv, the sha256 of every input, and the sha256 of every code-of-record file.
* The MM writes λ\*_0 and λ\*_1 into this amendment's Log before any SCORE plan at λ ≠ 0 exists.
* **A0 (λ = 0) does not depend on λ\*.** It may run on SCORE while the selection runs.

## 4 Arms (SCORE: the 994 phase-1 SCORE windows in 71 episodes, R35_s0 and R35_s1)

All arms are HWM-ORACLE with phase 1's levers: T3 2, `plan-ctx 1`, `stitch decode`, `vocab off`, `constr-weight 0`. They use each chassis's own checkpoint (sha256 `d920d62c…` / `a97e6017…`, equal to the phase-1 handoff) and its own TRAIN priors.

| arm | argv difference | inference seeds | role |
|---|---|---|---|
| **A0** SEED | `--seed-mode hold --lam 0` | 0, 1 | control; the REG effect is A1 − A0 |
| **A1** REG | `--seed-mode hold --lam λ*_k` | 0, 1 | the lever |
| **R** REG-NEG | `--seed-mode hold --lam −max(λ*_k, 1)` | 0 | deliberate regression: it pushes plans away from HOLD and **must read worse** than A0 (C-REG) |
| **S** SHRINK | none: post-hoc, zero compute | 0 | matched-shrinkage control (below) |
| HOLD, PRIOR | references from `run_hwm.py report` | — | T1-kin |
| phase-1 HWM-ORACLE | banked plans (`seed-mode prior`), from the HF phase-1 tar | 0, 1 | beside: the SEED effect, A0 − phase 1 |

* **How S is built.**
  * Knots 1-10: `HOLD + α_k,c · (A0_plan − HOLD)` per channel c. α_k,c = D_TRAIN,k,c(λ\*_k) / D_TRAIN,k,c(0), so α is fixed on TRAIN, never on SCORE.
  * Knots 11-30: A0's, which equal A1's (C-L2).
  * S asks one question: **does REG's solve move the plan toward the logged drive more than a plain pull toward HOLD of the same size?** The SCORE distances of S and A1 are printed beside, to show how well the TRAIN match holds there.
* **No FLAT arm, no shuffled-goal arm.**
  * FLAT needs its own selection (about +5 h) and answers the 2-6 s question. It is the committed next step on CEILING (section 6).
  * A shuffled-goal arm needs new code in `run_hwm.py`, which accepts the goal kinds `oracle` and `plan` only (`run_hwm.py:119-135`). R covers the regression requirement through the lever's own flag.

## 5 Read

* **Instrument.**
  * Each arm runs `run_hwm.py windows / plan / report` with its argv, in its own output directory (section 8.2).
  * The paired contrasts use `phase2_l1full/contrast_l1full.py` with `--a`, `--b`, `--rep-a`, `--rep-b` and `--windows` = the phase-1 windows, exactly as ACTRSP section 4.2 reads.
  * STRATEGIC cells come from the strategic-audit reader (`raw/2026-11-16-v8f-strategic-audit/reader/src/hwm_eval.py`, `strategic_cells.py`, `strategic_text.py`), as ACTRSP section 4.3 names them. ⚠️ **ESCALATED:** `contrast_l1full.py` imports `hwm_eval` from the code-of-record folder, whose `STRATEGIC` constant reads PAUSED (`hwm_eval.py:29`). The MM's freeze line names the one contrast script that imports the strategic-audit reader for both ACTRSP and LON-PLAN. UNVERIFIED which one does today.
* **Contrasts**, per band (0-2 s, 2-6 s, full 6 s), per family, never pooled:
  * A1 − A0 and A1 − HOLD (deciding);
  * A1 − S;
  * R − A0;
  * A0 − HOLD;
  * A0 − phase-1 HWM-ORACLE;
  * inference-seed replicates of A0 and A1.
* **Families** (ACTRSP section 4.3, unchanged):
  * **LONGITUDINAL:** `acc_mae` (deciding, 0-2 s) and `speed_err`. Distance-keeping: `hw_err`, `tg_err` and `ttc_err` on windows in state LEAD, with n per cell. The 0-2 s band uses the K = 10 lead block; 2-6 s and the full 6 s use K = 30. Signed `hw_d`, `tg_d` and `ttc_d` are printed on arm rows.
  * **LATERAL:** `yaw_mae` (the guard, 0-2 s), `head_err`, `curv_err` (steps with logged speed ≥ 1 m/s, n printed) and `xtrack`. `ade` is beside.
  * **TACTICAL:** plan-derived, `tac_man_dis`, `tac_lat_dis` and `tac_lon_dis` per band, with kappa (`hwm_eval.TACTICAL_SCOPE`). Goal and anchor selection: n = 0, NOT COMPUTABLE (the planner has no decision output).
  * **STRATEGIC, reported in full** (the item 25 pause ended by PI ruling 2026-11-16):
    * `s_adh` (with its signed-degree twin) and `s_false`, full 6 s band only, with the logged-path rate beside. The 0-2 s and 2-6 s bands are empty by construction.
    * Expected n, INHERITED from ACTRSP section 4.3 (measured there on the ACTSENS arms): 910 of the 994 windows carry a text row; S_ADH has 8-16 scorable windows in 3-6 episodes, which is not decision-grade; S_FALSE has about 850 windows in 69-70 episodes.
    * On HWM-ORACLE these cells measure the world model's use of an oracle goal, not a strategic decision. A1 and A0 differ only in knots 1-10.
    * `S_CKPT`, strategic intent 6-30 s, derived goals and detected info are NOT COMPUTABLE, each with its reason and would-have n, as ACTRSP section 4.3 lists them. That is a structural n/a, never a pass.
    * COMPLETE_M, theta and the 30 m margin are as ACTRSP fixed them.
    * **No LON-PLAN outcome is decided on a STRATEGIC cell.**
* **DIR-ACC**, beside with no bar, new.
  * Per window: x = mean over knots 1-10 of (plan acc − a0), and y = the same for the logged acceleration. DIR-ACC is the Pearson correlation of x and y over windows, with its CI from the episode-cluster bootstrap (B 2,000, seed 0).
  * It is computed for A0, A1, S, R and the banked phase-1 plans.
  * It asks whether a plan's departure from HOLD points the way the drive went, separately from its size. It is n/a for a plan with x ≡ 0 (HOLD).
* **Estimator.** Paired episode-cluster bootstrap (`taniteval/ci.py` through `hwm_eval.contrast`), B 2,000, seed 0, over the 994 SCORE windows in 71 episodes. It answers *another draw of episodes*, nothing else.
* **Inference floor** F_I,k(q) = max(|A0 seed 1 − A0 seed 0|, |A1 seed 1 − A1 seed 0|) on the row's point means, per chassis. It answers *another inference run* (the planner samples its candidates). For scale, phase 1's HWM-ORACLE 0-2 s `acc_mae` replicate is −0.0013 / −0.0136 (MEASURED, `RESULT_HWMPLAN_R35_s{0,1}.md`).
* **Training runs.** Nothing is trained, so no training floor is computed. A claim about the R35 recipe needs both chassis (two training runs) to agree. It is not claimed for further training runs. ACTSENS's `NULL_s2` / `NULL_s3` would add two more, named in section 6.

## 6 Committed outcomes (both chassis required; a label at one chassis only is SEEDS-DISAGREE, and the less favourable label's next step applies)

**Definitions, per chassis k, on the 0-2 s band, inference seed 0:**
* Δ_H = A1 − HOLD (`acc_mae`).
* Δ_R = A1 − A0 (`acc_mae`).
* Δ_S = A1 − S (`acc_mae`).
* G_yaw = A1 − A0 (`yaw_mae`), the guard.
* **"Separated-better beyond floor":** the CI lies wholly below 0, |Δ| > F_I,k(q), and the seed-1 point Δ is also below 0.
* **"WORSE":** the CI lies wholly above 0 and Δ > F_I,k.

| outcome | condition (both chassis) | next lever (Rule Zero; run in the same work package unless blocked) |
|---|---|---|
| **LON-PLAN-CEILING** (the primary criterion passes) | Δ_H separated-better beyond floor; A1 not HOLD-collapsed on SCORE (D ≥ 0.10); G_yaw not WORSE | (1) **The 2-6 s band, zero training:** FLAT-ORACLE with REG at its own λ\*, selected by section 3's procedure with C on the full 6 s band, plus its A0 / R / S. It runs under this amendment, and its λ\* line is added to the Log before its SCORE plans. (2) **The level-2 image constraint** (section 7) at λ\*_k. (3) **The T1-kin read:** HWM-PLAN with the predicted goal at λ\*_k. **Blocked** on a goal head trained under its own card and gate (box 2, PI go, `QUESTIONS_LONPLAN.md` Q2). (4) **MM:** SEED + REG(λ\*_k) becomes the solver of record for later HWM-PLAN arms, and ACTRSP Option B's acceleration term loses the reason it was written for |
| **LON-PLAN-NARROWS, INFORMATIVE** | not CEILING; Δ_R separated-better beyond floor; Δ_S separated-better beyond floor; G_yaw not WORSE | Near HOLD, REG's solve carries acceleration information, but not enough to beat holding. Next: **the level-2 image constraint** (section 7) at λ\*_k. Level 1's 0-2 s cost is the distance to level-2 rollout states, which are planned under off-image macros (ratio 6.77 / 6.79). That is the largest measured planner defect left (Rule Zero rule 5) |
| **LON-PLAN-NARROWS, SHRINKAGE-ONLY** | Δ_R separated-better beyond floor; Δ_S not separated-better | REG's gain is the pull toward HOLD. Near HOLD, the level-1 latent cost points in no usable acceleration direction. Next: **the level-2 image constraint** at λ = 0. After it: a level-1 cost read through RA, the 2 s readout with transition skill (Amendment 1 section 1.4, frozen weights `readout_K10_R35_s{k}.pt`). Zero training; its own amendment. Note that PT3M / PT3F spent the decoded-action cost at FLAT1 on M2b; RA is a different readout, measured on R35 |
| **LON-PLAN-NULL** | Δ_R not separated-better beyond floor at either chassis, and not WORSE | As SHRINKAGE-ONLY |
| **LON-PLAN-HARM** | Δ_R WORSE at either chassis, or G_yaw WORSE | REG stays off. First, C-L1 to C-L4 are re-run on the SCORE argv: a wiring defect is the first suspect, because M2b's TUNE curve fell with λ in every row. Then the image constraint at λ = 0 |
| **LAMBDA-COLLAPSE** | no λ > 0 admissible on TRAIN at a chassis, after the extension | The REG term outweighs the goal cost at every dose tried. Next: the image constraint at λ = 0 |
| **INVALID** | any control in section 8 fails | Nothing is read. The failing control is fixed and the arm is re-run |

* **Reported beside every outcome, no bar:**
  * the TRAIN-CEILING flag (C_k(λ\*) < C_k(HOLD) on TRAIN, the in-sample sign);
  * the SEED effect A0 − phase 1;
  * DIR-ACC;
  * R − A0;
  * the full family table.
* **A failed bar is reported as FAILED, then its next lever runs.** No dose, arm, window or criterion changes after any SCORE number is seen. No SCORE bracket of λ is run, because that would be dose selection on scored windows.
* **Done** means one of two things. (a) CEILING, with next steps (1) and (2) started. (b) The image-constraint amendment drafted and its build started, with each training-gated lever named with its blocker.
* **If both chassis pass close to the floor** (|Δ_H| < 2 F_I,k): before any solver-of-record change, the same A0 / A1 read runs on ACTSENS's `NULL_s2` / `NULL_s3` (R35-recipe training replicates, ACTRSP section 4.4) at their own TRAIN-selected λ\*.

## 7 "Then the level-2 image constraint": what it means in the frozen HWM-PLAN, and why it needs its own amendment

* **Its status in the frozen card.** It has **no card row.** AMEND_HWMPLAN_2 (last paragraph) says it "gets its own amendment before any of its windows is scored". The L1-FULL verdict names it as the precondition for the BAND-SPLIT re-plan.
* **What it is.** Level 2 plans only macro-actions in the image of the macro encoder over bounded primitive sequences: `macro = macro_enc(act_embed(p))`, with p inside the level-1 prior box. This is the property the decode check measured as missing (NN distance 7.16 / 7.21 against 1.06; ratio 6.77 / 6.79).
* **Two implementations; the MM picks one at that amendment's freeze:**
  * **(I) By construction.** Level 2 optimises primitives p: 10 macro steps × 3 primitives × 2 channels. It rolls the level-2 predictor on `macro_enc(act_embed(p))`. Knots 11-30 are then p itself.
    * The decode residual is exactly 0, and no dose is needed.
    * It keeps the direct 0.6 s steps that WMCEIL credits, unlike `l1_full`, which re-plans with level-1 dynamics.
    * It needs a new shadow on the level-2 optimiser variable, which is new code of record in `hwm_plan.py`.
  * **(II) Penalty.** A `cost_wrap` term at level 2 (the G7 hook pattern, `hwm_plan.py:470-474`): w · (distance of the macro to its nearest TRAIN macro)².
    * The code is smaller, but it brings a second dose and a TRAIN macro bank.
  * **Recommended: (I).** It has no dose, and its success criterion is an identity.
* **What that amendment must carry:**
  * a flag `--l2-image {off, prim}`, default `off` = today bit for bit;
  * controls: default bit-identical to phase 1 on the 16-window smoke; planted world recovers in-image macros; decode residual exactly 0.0 under `prim`; mutation (the shadow not installed) returns the residual to ≥ 0.8 and must go RED;
  * the same 994 windows, deciding on 0-2 s `acc_mae` as here, with 2-6 s beside;
  * **its λ fixed by this read's outcome** before its windows are scored: λ\*_k on CEILING or NARROWS-INFORMATIVE, 0 otherwise.
* **Schedule.** Its build is new code and needs no LON-PLAN result. It is the next lever in every LON-PLAN outcome. So the MM can start its build agent in parallel with LON-PLAN's CPU read.

## 8 Controls and reachability (the read is INVALID if one fails; each new check has a deliberate-regression arm that must go RED)

### 8.1 Reachability of `--lam`, traced (MEASURED, code of record at `45e6c130`, sha256-equal to the local `E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/hwm_plan/` and `.../plantest2/plantest2_lib.py` today)

| hop | file:line | what happens |
|---|---|---|
| 1 | `run_hwm.py:31` | `--lam` float, **default 0.0**; `--seed-mode` default `hold`. The parser is shared by `windows` / `plan` / `report` (`:219-220`) |
| 2 | `run_hwm.py:22-24` | `solver_from(a)` → `H.SolverCfg(..., lam=a.lam, ...)` |
| 3 | `run_hwm.py:75-83`, `:138-142`, `:169`, `:176` | `make_planner` → `HWMPlanner(wm, geom, solver_from(a), ...)`, built in each spawned worker from the argv dict. The report builds its own at `:195-196` and records `str(solver_from(a))` at `:209` |
| 4 | `hwm_plan.py:87`, `:403-404` | `SolverCfg.lam` (default 0.0); `self.solver`; `self.plan_fn = Q.plan_topdown2` |
| 5 | `hwm_plan.py:420-425` | `seed_mode prior` with λ ≠ 0 → **AssertionError** "REG needs the HOLD seed" (`:421`). Else `seeds = hold_seeds_g(...)` (`:424`) and `reg = dict(lam, ref=seeds[1], std=prior_std_orig) if lam != 0.0 else None` (`:425`) |
| 6 | `hwm_plan.py:438` (hier), `:441` (flat), `:490` | `plan_fn(..., seeds=seeds, reg=reg)` per batch, generator `make_gen(seed, batch0 + j)` |
| 7 | `plantest2_lib.py:129-135` | level 1 only: horizon check, then `cost_fn = base + reg_term(...)` |
| 8 | `plantest2_lib.py:155-158` | `reg_term = λ · mean(((a − ref) / std)²)` |
| 9 | `plantest2_lib.py:101`, `:137-138` | `st._optimize(cost_fn, …)` (AdamW, ≤ 90 iterations, patience 10), then `best_cost.argmin`: the term reaches the gradient and the candidate pick |

* **The default disables it.** `--lam 0.0` gives `reg = None` (`:425`). Every LON-PLAN argv must NAME `--seed-mode hold --lam <value>`, A0's included.
* **Would a mutation that turns it off be caught today?**
  * **At hop 8, yes.** PLANTEST2's `test_c2_mutation_reg_off_goes_red` (`test_plantest2_controls.py:60-64`) zeroes `reg_term`, and the limit check goes RED.
  * **At hops 1-6, NO.** `test_hwm_plan.py` has 0 tests with λ > 0: grep for `lam=`, `.lam`, `--lam` and `reg=` finds only `lam_sub` and `decode_lam`. PLANTEST2's tests enter at `plan_all2` / `run_planner2` and never pass through `run_hwm` or `HWMPlanner`. A mutation such as `solver_from` dropping `lam`, or `plan_batch` building `reg = None`, would pass every existing test and plan λ = 0 under a λ label.
* **And one more trap, MEASURED in `run_hwm.py:160-166`.** `plan` **skips every `plans_<arm>_s<seed>.npz` that already exists**, and the file name carries no λ. A λ run pointed at a λ = 0 directory would bank nothing new. Its report would then label the λ = 0 plans with the report's own `--lam` (`:209`). The plan files carry no solver record.

### 8.2 New controls (`test_lonplan.py` plus the run wrapper; written and green before any SCORE window is planned)
* **C-L1, reach.** On the toy stack and on the R35_s0 16-window smoke with real TRAIN priors:
  * λ = 1 changes level-1 knots against λ = 0 at the same seed (max |Δ| > 0);
  * the REG term's gradient on the level-1 actions is non-zero;
  * **exact-zero control:** `--lam 0` with `hold` equals a run whose `reg_term` is patched to 0.0, bit for bit;
  * **mutation:** `solver_from` with `lam` pinned to 0.0 must go RED.
* **C-L2, level-1-only identity.** In hier `decode` at the same seed:
  * knots 11-30, the level-2 and level-3 actions and the level-2 / level-3 costs are bit-equal between λ = 1 and λ = 0, while knots 1-10 differ;
  * on SCORE, A1 − A0 reads exactly 0.0 [0, 0] for 2-6 s `acc_mae` and `yaw_mae`;
  * **mutation:** REG also applied at level 2 must go RED.
* **C-L3, limit.** λ = `LAMBDA_LIMIT` (1.0e6, `plantest2_lib.py:15`) through `HWMPlanner`: knots 1-10 equal HOLD_c within 1e-4 prior std (PLANTEST2's tolerance), in hier and flat. **Mutation:** REG reference set to zeros must go RED.
* **C-L4, argv → built.**
  * The wrapper parses each argv with `run_hwm.add_common` and asserts `solver_from(a).lam` and `.seed_mode` equal the declared values.
  * It writes `SOLVER.json` (argv, `str(SolverCfg)`, code sha256s) into a **λ-specific out dir**.
  * It **refuses an out dir that already holds any `plans_*`** (the skip-existing trap).
  * **Mutations:** a declared/argv mismatch, and a pre-planted `plans_hwm_oracle_s0.npz`, must both refuse.
* **C-READBACK.** After banking, batch 0 of every SCORE plan file is re-planned from its `SOLVER.json` argv at `OMP_NUM_THREADS=2`, and plans and `cost1` must be bit-equal. **Deliberate regression:** the A1 file read back at λ = 0 must fail.
  * All LON-PLAN runs use the same thread count: dumps at other thread counts drift (programme memory).
* **C-REG, the regression arm.** R − A0 on 0-2 s `acc_mae` must be separated-worse at both chassis, and D(R) > D(A0). Else the λ path does not act in the declared direction, and the read is INVALID. On TRAIN, the sign check must read C_k(−1) > C_k(0) before any SCORE plan at λ ≠ 0.
* **C-SHRINK.** S at α = 0 equals HOLD, and its contrast against `hold` reads exactly 0.0 [0, 0]. S at α = 1 equals A0 exactly.
* **C-TRAIN-ISO.**
  * The selection script asserts 0 TRAIN clips among the eval clips.
  * It runs with `hwm_data.eval_windows_from_cache` and `wmact_lib.open_windows` patched to raise (as PLANTEST2's `test_c4_tuning_never_reads_score`).
  * Its windows equal the source 16-frame windows frame for frame (max |Δ| 0.0).
  * At λ = 0 its plans equal `run_hwm plan`'s on the same windows bit for bit.
* **C-IN, inputs.**
  * Checkpoints, TRAIN latents and code-of-record files are sha256-equal to the phase-1 record and the freeze blobs. MEASURED equal today; asserted again at run time.
  * `windows.npz` is pulled from `hwmplan/hwmplan_phase1_R35.tar` and sha256-equal to its per-file record: s0 `371c753b…`, s1 `8833e292…`.
* **Unchanged:** the card's controls 1-15. `test_hwm_plan.py` (50 passed in 522.89 s, `TEST_HWMPLAN.log`) and `test_plantest2_controls.py` (14 passed) are re-run green on the code of record.

## 9 Cost and approvals

**MEASURED rates** (dev box, 1 worker, `OMP_NUM_THREADS=2`, 994 windows; from the timestamps in `phase1_R35/phase1_R35.log` and `phase2_l1full/phase2_l1full.log`):

| run | wall per (chassis, inference seed) |
|---|---|
| HWM-ORACLE `decode` | 24.9 / 24.0 min (s0), 25.1 / 27.6 min (s1) ⇒ 1.45-1.67 s per window |
| FLAT-ORACLE | 33.2 / 37.1 min (s0), 40.0 / 38.8 min (s1) |
| HWM `l1_full` | 35.6 / 27.3 min (s0), 24.2 / 23.5 min (s1) |
| windows stage | 2.8 min (s1) |

* **The INHERITED "about 2.2 h per checkpoint and inference seed pair" (Amendment 1 section 2.2), re-derived.** The logs give **1.99 h (s0) and 2.19 h (s1) per checkpoint for HWM + FLAT, each at both inference seeds.** For HWM alone at both seeds it is **0.82 h / 0.88 h**. So the inherited figure holds for both planners together, and it overstates an HWM-only pair by about 2.5×.
* **LON-PLAN, ESTIMATED** from those rates, assuming the HOLD seed and REG keep the per-window rate within ±30 % (they can change the early-stop iteration count). The 16-window smoke (C-L1) times it first.

| stage | runs | wall |
|---|---|---|
| TRAIN selection | (7 grid values + sign check) × 2 chassis = 16 runs × 564 windows | 3.6-4.2 h |
| extension, worst case | 6 more runs | +1.4-1.6 h |
| SCORE | A0 ×2 seeds, A1 ×2, R ×1, per chassis = 10 runs | 4.0-4.6 h |
| C-READBACK | 10 files × one batch (phase-1 batch units 76-122 s) | 0.2-0.35 h |
| contrast, report, S, DIR-ACC | | about 0.3 h |

* **Total: about 8-9.5 h wall at 1 worker × 2 threads** (16-19 thread-hours), and about 11 h with the full extension.
* **Two workers** about halve the wall time if 4 cores are free (ESTIMATED; phase 1 ran 1 worker).
* **A0 may start at once:** the TRAIN stage and A0 do not depend on each other.
* **Build:** one Sonnet agent writes `lonplan_select.py`, `lonplan_shrink.py` (S and DIR-ACC), the run wrapper and `test_lonplan.py`. No change to the code of record.
* **Disk, ESTIMATED:** about 0.3 GB on F: (52 GB free, MEASURED today). It is banked to the phase-1 HF repo (private) with per-file sha256, then the local copies are deleted under the PI's delete rule.
* **Approval.**
  * The CPU read itself is of the kind the MM has run without a PI go (phase 1, G3).
  * It belongs to Amendment 1's **Option A, which is pending the PI's choice**: `QUESTIONS_LONPLAN.md` Q1.
  * The training-gated next levers need the PI: the goal head on CEILING, and world-model action-sensitivity training: Q2.
  * No box-2 time, GPU or spend is in this amendment.

## 10 Fills the MM sets at freeze
1. **Per-chassis λ\* (recommended, as drafted) or one λ\* shared by both chassis.** Shared means the λ that minimises the mean of C_0 and C_1 among values admissible at both. A shared λ\* is stricter for replication, but it ignores each chassis's own cost scale.
2. **The contrast script** that imports the strategic-audit reader, named once for ACTRSP and LON-PLAN (section 5).
3. **The image-constraint implementation**, (I) or (II), for AMEND_HWMPLAN_4 (section 7).

## 11 Unchanged
The card's bars, arms, outcomes and controls 1-15. FREEZE fills 1-5, as amended by AMEND_HWMPLAN_1 and AMEND_HWMPLAN_2. The solver of record of every arm other than LON-PLAN's (`seed_mode prior`, `lam 0`). The 994 SCORE windows. ACTRSP sections 4.3, 4.4, 8 and 9. Amendment 1's measurements. The ACTRSP arms stay held, as Amendment 1 holds them.

## 12 Register rows for the MM (to land with the freeze)
* **`H-LONPLAN-REG`**, HYPOTHESIS, OPEN. A HOLD-anchored REG(λ) at a TRAIN-selected dose lets HWM-ORACLE beat HOLD on 0-2 s acceleration at T0 on R35, both chassis. Read: this amendment.
* **`D-RUNHWM-LAM-UNTESTED`**, MEASURED, a defect class. In the HWM-PLAN code of record, `--lam > 0` is reachable but untested through `run_hwm` / `HWMPlanner` (hops 1-6), and `plan` silently skips existing plan files whose names carry no λ. Fixed for LON-PLAN by C-L1 to C-L4 and C-READBACK. This is the "built, tested, unreachable from its caller" family, here as "reachable, untested from its caller".

## Log
* 2026-11-16, agent draft.
  * Measurements made for this draft (CPU, no window planned or scored): the TRAIN latent census (frame table, overlap, window counts), sha256 of inputs and code, and phase-1 / G3 wall times from the banked logs.
  * Scripts: session scratchpad `insp.py`, `overlap.py`, deleted at hand-back.
  * Freeze line, λ\*_0 and λ\*_1: Master Mind.
* **2026-11-16 ~14:35Z, Master Mind: FROZEN** (the text above is the agent's draft unchanged, sha256 `08f69348…`). No SCORE or TRAIN window has been planned at any λ, and no λ has been selected. The section 10 fills:
  1. **λ\* is per chassis**, as drafted: λ\*_0 on R35_s0's TRAIN selection set and λ\*_1 on R35_s1's. Each chassis has its own cost scale. Replication is read from the outcome table (section 6), not forced through a shared dose.
  2. **The contrast script** for both ACTRSP and LON-PLAN is `contrast_l1full_strat.py`, a build deliverable of this amendment. It is `phase2_l1full/contrast_l1full.py` with `hwm_eval` resolved to the strategic-audit reader (`raw/2026-11-16-v8f-strategic-audit/reader/src/`, landed `7a571fc7`). Control **C-STRAT**:
     * on the phase-1 inputs, every non-STRATEGIC cell must be bit-equal to `contrast_l1full.py`'s;
     * the STRATEGIC cells must be computed (not PAUSED), or carry a reasoned n/a with n;
     * a deliberate-regression arm (the code-of-record import) must read PAUSED.
  3. **The image constraint (section 7):** the (I) / (II) choice belongs to AMEND_HWMPLAN_4's own draft. Nothing in LON-PLAN depends on it.
  * **Run order:** build (`test_lonplan.py` and the run wrapper with C-L1 to C-L4, C-READBACK, C-STRAT, each with its deliberate-regression arm, green), THEN the TRAIN selection (λ\*_0 and λ\*_1 written into this Log), THEN SCORE.
  * **Q1 (QUESTIONS_LONPLAN.md):** the PI is told that LON-PLAN runs on dev-box CPU after the build unless the PI says no. It has no box 2, GPU, training or spend. **Q2 stays with the PI.**
* **2026-11-16 ~18:55Z, Master Mind: λ\* SELECTED on TRAIN** (section 3). The text above this entry is the frozen file, sha256 `7bbdc9fe…`; `AMEND_HWMPLAN_3.sha256` now carries the hash of the file with this entry. No SCORE window has been planned at any λ ≠ 0.
  * **λ\*_0 = 300** (R35_s0). The base grid chose 100, the largest value run, with D 0.2196 ≥ 0.10, so the extension up ran: 300 is admissible with a lower C, and 1000 has D 0.0660 < 0.10, which stops it.
  * **λ\*_1 = 100** (R35_s1). The base grid chose 100; the extension up ran 300, whose D 0.0934 < 0.10 stops it.
  * Both selections ran on dev-box CPU, 1 worker × 2 threads, OMP 2: `lonplan_select.py` from the build working copy (landed `bd6f2276`). R35_s1 finished ~18:31Z, R35_s0 ~18:43Z. Records: `lonplan/selection/LAMBDA_SELECTION_R35_s0.json` (sha256 `06804864…`) and `LAMBDA_SELECTION_R35_s1.json` (`905c3c43…`), with every argv, input sha256 and code-of-record sha256. Inputs equal section 3.1: checkpoints `d920d62c…` / `a97e6017…`, TRAIN latents `76a2d4f5…` / `692eb002…`. Selection windows: 564 per chassis in 141 TRAIN clips, 0 of them in the eval set.
  * **Tier: T0 diagnostic on TRAIN windows. Evidence: MEASURED. A selection record, never a result.**

| λ | C s0 | D s0 | D_yaw s0 | C s1 | D s1 | D_yaw s1 |
|---|---|---|---|---|---|---|
| −1 (sign check) | 1.4848 | 1.7450 | 1.3767 | 1.4494 | 1.6917 | 1.4028 |
| 0 (A0, not admissible) | 1.2677 | 1.4884 | 1.1942 | 1.2304 | 1.4352 | 1.1982 |
| 0.3 | 1.1916 | 1.3950 | 1.1463 | 1.1568 | 1.3474 | 1.1341 |
| 1 | 1.0286 | 1.2067 | 1.0440 | 1.0045 | 1.1681 | 1.0215 |
| 3 | 0.8043 | 0.9340 | 0.8696 | 0.7796 | 0.8916 | 0.8109 |
| 10 | 0.5902 | 0.6515 | 0.6083 | 0.5713 | 0.6026 | 0.5048 |
| 30 | 0.4437 | 0.4181 | 0.3720 | 0.4202 | 0.3631 | 0.2667 |
| 100 | 0.3472 | 0.2196 | 0.1877 | **0.3299** | 0.1775 | 0.1443 |
| 300 | **0.3121** | 0.1167 | 0.1182 | 0.3056 (D < 0.10) | 0.0934 | 0.1048 |
| 1000 | 0.3011 (D < 0.10) | 0.0660 | 0.0943 | not run | | |

  * C is the mean 0-2 s `acc_mae` in m/s² over the 564 windows; D and D_yaw are in TRAIN prior-std units (section 3.3).
  * **Sign check (C-REG on TRAIN): ok at both.** C(−1) − C(0) = +0.2171 (s0) and +0.2190 (s1).
  * **Beside, no bar.** C(HOLD) 0.2965 at both chassis (HOLD does not depend on the world model); C(PRIOR) 0.5132.
    * **The TRAIN-CEILING flag is false at both.** C(λ\*) is 0.3121 (s0) and 0.3299 (s1), above C(HOLD) 0.2965. On TRAIN, every admissible λ reads above HOLD, and C falls monotonically toward HOLD as λ rises.
    * This is the in-sample sign that section 6 prints beside the outcome. It decides no label; SCORE does. HYPOTHESIS: the TRAIN curve has the shape of a pull toward HOLD, which is the question arm S answers on SCORE.
  * **DIR-ACC on TRAIN** (section 5, beside, no bar). Pearson r at λ\*: 0.332 [0.227, 0.493] (s0) and 0.356 [0.274, 0.482] (s1). Estimator: episode-cluster bootstrap over the 141 clips, B 2,000, seed 0. At λ 0 r is 0.301 (s0) and 0.279 (s1); at λ −1 it is 0.288 and 0.267.
  * **Arm S's α, fixed now on TRAIN** (section 4, α_k,c = D_TRAIN,k,c(λ\*_k) / D_TRAIN,k,c(0)), computed from the records above:
    * s0: α_acc = 0.078433, α_yaw = 0.098969.
    * s1: α_acc = 0.123680, α_yaw = 0.120455.
  * **Next, in the frozen order: SCORE.** Before the first SCORE plan at λ ≠ 0, three steps:
    * build `lonplan_shrink.py` (arm S, C-SHRINK) and the SCORE-side DIR-ACC, each with its deliberate-regression arm;
    * fetch or regenerate the R35_s1 SCORE windows;
    * re-run the code-of-record tests.
  * A0 (λ 0) does not depend on λ\*.
