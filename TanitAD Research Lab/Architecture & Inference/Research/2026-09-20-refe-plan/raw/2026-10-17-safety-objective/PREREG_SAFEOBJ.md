# PREREG SAFEOBJ: constrained EPO on the g5 set (TTC, NC), measured against arm A. The RL-S1 safety-objective A40 arm (R0d / R1d / R1o)

**Written 2026-10-17 by the Research Lab design agent. Nothing has run yet: no SAFEOBJ training step, no trainer edit, no SAFEOBJ number.**
- sha256 is in `PREREG_SAFEOBJ.sha256`.
- Why this arm exists: the PI chose it on 2026-10-17 to fill the idle A40 (Master Mind brief). The brief is design only.
- Companion files: `SPEC_SAFEOBJ.md` (math, flags, gate) and `IMPL_PLAN.md` (the build).
- Sources are the archive of commit `daa4568`. PK = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`.
- Evidence class on every number:
  - **MEASURED**: ours, with an artifact path.
  - **ESTIMATED**: an offline read or a model, not a training run.
  - **INHERITED**: from a register row or a prose document, not re-verified here.
  - **HYPOTHESIS**.

## 0. Headline

* **Lever (one): Lagrangian EPO.** Arm B maximises J subject to two constraints on every training batch, each with its own dual weight updated once per optimiser update:
  * `E_piB[TTC] >= E_piA[TTC] - eps`
  * `E_piB[NC] >= E_piA[NC] - eps`
  * The flag is `--epo-con g5`.
  * Option (b), the fixed penalty `R' = R - 0.25 (1 - ttc)`, is a special case of this lever: the dual frozen at lambda_TTC = 0.25 (SPEC identity I2). (b) is therefore not a separate arm. It is the committed next lever of outcome O2.
* **A cheaper point already exists and sets the bar.** Logit interpolation from C0-A to R0-B at alpha = 0.25 reads PDMS **+0.60 [+0.27, +0.95]** with TTC **-0.03 [-0.36, +0.27]** on RL-val.
  * Source: ESTIMATED, `PK/raw/2026-10-14-rls1b-attribution/dose.json`.
  * It is deployable at 0 A40-h: one extra 4.3 M-parameter scorer forward on the frozen trunk.
  * It has never been read on navtest, and its alpha was chosen on RL-val after the curve was seen.
  * So the trained arm must land **above** this free frontier (bar G), or it bought nothing.
  * Its navtest read should run as a parallel zero-A40 stream, under its own pre-registration (§6, O2/O3).
* **Arms:**
  * **R0d**: seed 0, the constraint on.
  * **R1d**: seed 1, the constraint on.
  * **R1o**: seed 1, the constraint OFF. It is R1d's matched control on identical trainer bytes, and the first seed replicate of the unconstrained R-family trade.
  * Order is R0d, then R1d, then R1o. A kill stops the chain.
* **Cost:** at most **10.5 A40-h**, i.e. 3 x 3.5 (§3). If R0d is killed at u500, the total is about 1.8 A40-h.
* **PI decisions needed:**
  1. Is Stage 2 (R1d + R1o, about 7 A40-h) covered by "fill the idle A40"?
  2. Optional R0o, the seed-0 lever-off arm on identical bytes: +3.5 A40-h, not recommended (§3).
  3. Registering the logit-mixture navtest read (Master Mind).

## 1. Why this lever: the measured attribution (Rule Zero point 5)

### 1.1 The trade to repair

R0 at u1000, B - A on RL-val (3,000 sets, 75 logs; paired log-cluster bootstrap, 10,000 resamples, seed 20260927). MEASURED, `PK/raw/2026-10-15-rls1-adam-dose/ab_vs_base.json`:

| sub-score | B - A |
|---|---|
| PDMS | +1.15 [+0.54, +1.75] |
| TTC | -1.00 [-1.73, -0.30] |
| NC | -0.33 [-0.69, -0.03] |
| EP | +3.44 |

The same trade shows up across the programme:
- In every EPO arm: R0b, R0c (INHERITED, D-REFE-RLS1B-R0B, D-REFE-R0C-READ).
- In P2's supervised seed replicate: TTC -0.49 [-0.97, -0.07] with EP +1.80, on navtest T1 (MEASURED, `PK/eval/RESULT_P2REP.md`).

### 1.2 Every lever tried so far, and how much of the TTC loss each recovered

| lever | read | TTC recovered | source |
|---|---|---|---|
| EPO dose 0.25 (R0b) | RL-val TTC -0.57 / -0.70 (u250 / u500), K3-killed | ~0 | D-REFE-RLS1B-R0B, INHERITED |
| KL anchor = arm A (R0c) | RL-val TTC B - A -0.53 / -1.10 | ~0 | D-REFE-R0C-READ, INHERITED |
| TTC/NC head calibration | calibrated TTC -0.70 to -0.87 at all 6 reads | ~0 | D-REFE-RLS1-O2-CALIB, INHERITED |
| head-only EPO | J fraction 0.164; TTC proxy worse per unit J | n/a (fails on J) | D-REFE-PHEAD-FEAS, ESTIMATED |
| separate safety head (A2; real R0 states; P2 TTC x100) | -1.53 to -0.78, -1.25 to -0.88, -1.07 to -0.75 | **30-49 %**; bar met at 0 of 3 states | `PK/raw/2026-10-16-safety-head/RESULT.md` §1.1, ESTIMATED |
| TTC weight 5 to 10 in the selection (SELTTC; diagnostic, controls failed) | -1.53 to -1.44, -1.25 to -1.28, -1.07 to -1.13 | at most **6 %** | same file, §10, ESTIMATED |

⇒ **51-70 % of the loss sits in arm B's PROGRESS rows** (EP, C, lane), which push the pick toward faster proposals (A2-PROGRESS-ROWS).
- A lever that acts on one channel is capped by that channel's share.
- A safety-side lever is capped at 49 %, and its sim says the true value is lower still.

### 1.3 Why (a) and not (b)

* **(b) acts on the reward's preference. That preference was measured to be already correct.**
  * On **58 of the 59** RL-val sets where R0's pick turned TTC-failing, R0's own reward prefers C0's pick (D-REFE-RLS1B-1 (1); INHERITED from the register).
  * No candidate reward, a TTC penalty or multiplier included, has a TTC-failing optimum in more than **0.07-0.11 %** of the 16,000 RL-train scenes.
  * So `R - 0.25(1 - ttc)` only reaches TTC through the gradient that (a) also uses. It does so at a weight fixed without evidence: TTC's effective reward weight rises from NC·DAC·5/12 ≈ 0.42 to ≈ 0.67, i.e. ×1.6.
  * The nearest measured analogue: doubling TTC's weight at selection bought at most +0.09 of 1.53 (SELTTC).
* **(a) acts on the OUTCOME.**
  * The constraint's gradient is -λ ∇ E_piB[ttc]. It reaches every row the selection score depends on, EP, C and lane included (SPEC §1.3). So it pushes down exactly the progress logits that A2 located.
  * The dual weight keeps growing until the in-sample outcome holds. It needs no prior estimate of the weight that is "enough", and the sim cannot supply one: it does not reproduce R0's TTC loss (V-NO, `k250T_read.json`).
* **(a) ⊇ (b) at equal A40 cost** (SPEC identity I2). (b) wins only if the dual's noise or lag hurts more than a fixed weight helps. That case is outcome O2's next lever, run at the dual's own **measured** time-averaged λ, not at 0.25.

### 1.4 The constraint will bind (reach), and a feasible point exists

* **Reach.** The trade is visible on R0's own **training** data, so an in-batch constraint is active. It is not a pure generalisation gap.
  * Data: R0's banked states read on the RL-train pool (302 sets / 30 logs; 296 of them sets the optimiser trains on; `adam_common.prepare_pool` side "train" = `rlfan_sets_train`).
  * P2 TTC B - A ×100: **-1.74 / -1.22 / -1.01** at u250 / u500 / u1000.
  * NC: -0.52 / -0.48 / -0.35.
  * ESTIMATED (cached context, CPU), `PK/raw/2026-10-16-safety-head/realstates.json`, `states.r0_u*.FULL.*.train`.
* **Feasibility.** The interpolation frontier (dose.json, ESTIMATED; RL-val; vs C0-A) contains points with both gain and safety:

  | alpha | PDMS | TTC |
  |---|---|---|
  | 0.25 | +0.60 | -0.03 |
  | 0.375 | +0.71 | -0.27 |
  | 0.5 | +0.84 | -0.47 |
  | 1.0 (R0 itself) | +1.16 | -1.00 |

  A parameter state with about half the gain and no TTC loss exists along that line. Dual ascent targets such a point directly; R0b's dose did not reach it (AdamW, D-REFE-RLS1-HADAM).

### 1.5 Alternatives considered and not chosen

| alternative | why not chosen | when it would come back |
|---|---|---|
| Per-set hinge constraint | stricter than g5, which is a MEAN, so it costs more J | logged as a diagnostic only |
| Projected-gradient / CPO-style hard constraint | AdamW's preconditioner voids the first-order guarantee; needs a separate backward per micro-batch | held for O5 |
| Progress-row anchor | the gain lives in the same rows (SH_A-sel keeps 0.72-0.75 of the gain with A's safety rows), so it cancels the gain | — |
| Held-out-split dual | only needed if the in-sample constraint generalises badly | held for O4 |

## 2. The lever: exact values, all registered before data (math in SPEC §1)

| item | value | derivation |
|---|---|---|
| constrained terms | TTC (col 3), NC (col 0) of the PDM labels | g5 = "TTC and NC may not fall by > 0.5 versus control" (`rl_val_eval.py:52`, `:306`). TTC alone invites migration into NC, which already reads -0.28 (R0c u500) to -0.33 (R0 u1000) |
| reference | arm A's policy on the SAME micro-batch, detached | g5's contrast is B vs the BCE-only control. A's own TTC gain (+0.83 vs base, D-REFE-RLS1-HADAM) is part of what B must match |
| policy | `epo_policy` at T = 0.1 under the base's rule (`scorer_finetune.py:188-191`) | the very pi of J; a second temperature would be a second lever |
| eps | **0.0** | val - train P2 TTC on R0's states is -0.21 (u500) and -0.35 (u1000). An in-sample slack of 0.005 would put RL-val near -0.7 to -0.85, a g5 breach by construction. eps 0 leaves about 0.35 points of margin. ESTIMATED; the val pool is 62 sets / 13 logs |
| dual step eta | **4.0** per unit violation (probability units) per update | a sustained violation at the g5 size (0.005) for 50 updates (5 % of the run, inside one 250-update check) lifts λ to 1.0, about 2.4× TTC's own effective reward weight (0.42) |
| λ_max | **10.0** | a cap, not a target. Reaching it is an event (O5) |
| λ0 | **0.0** | step 0 is R0 exactly (identity I1) |

## 3. Arms, order, cost

| arm | argv (SPEC §3) | seed | control for every delta | A40-h |
|---|---|---|---|---|
| **R0d** | R0's banked argv (sha256 `24451b62…`) + the five `--epo-con*` flags, own `--out` / `--dump-scores` | 0 | its own arm A (same pass); C0 is the second control | 3.5 |
| **R1d** | R0d with `--seed 1` | 1 | its own arm A; R1o is its matched lever-off control | 3.5 |
| **R1o** | R0's argv with `--seed 1` and `--epo-con off` NAMED | 1 | its own arm A | 3.5 |
| R0o (NOT registered) | R0's argv on the new bytes | 0 | — | +3.5, PI decision |

**Order and stop rules: R0d, then R1d, then R1o.**
- A K3 kill of R0d stops the chain. Bar S cannot then hold on both seeds, so no verdict can change; this saves about 7 A40-h.
- A K3 kill of R1d stops the chain: not replicated, saves 3.5 A40-h.
- R1o runs only if both constrained arms pass S, because only then is a lever claim on the table.

**Controls:**
- The seed-0 lever-off control is **banked R0**. Its EPO and loss segments are byte-pinned to the bytes the new trainer reverses to: test_rl_s1b P-pins, plus the reversal chain in IMPL step 2. Its lever-off path is proved identical by identity I1.
- `main.fwd`'s default-off aux path is the only residue, as already accepted for R0b and R0c (PREREG_RL_S1 §10.3, O5).
- The identical-bytes matched control required by the brief is R1o, at seed 1.
- R0o would add a same-seed rerun of R0. It is optional; the PI decides.

**Cost basis:**
- MEASURED: R0 ran 1,000 updates in 3.4 A40-h (`PK/raw/2026-10-14-rls1-r0-kill/README.md:44`).
- INHERITED: C0 and R0 took 3.4-3.5 h (`PK/eval/PREREG_RL_S1.md` §10.3).
- MEASURED: R0c ran at 10.63 s/update to u500, one evaluation included (`PK/raw/2026-10-16-r0c-read/sft.jsonl:26`).
- A K3 kill after the u500 check costs about 1.7-1.9 A40-h (ESTIMATED from R0's README).
- The constraint adds two log-softmaxes over [8, 64] per micro-batch: ESTIMATED < 1 %, MEASURED by the preflight speed probe before launch.
- K6 (6 h per arm) is unchanged.
- Navtest reads: on the dev box, 0 A40-h, following the `PREREG_RL_S1.md` §11.6 precedent. That cost is UNVERIFIED.

## 4. Reads, tiers, estimator

* **Selection tier (RL-val).**
  * The true NAVSIM v1.1 sub-scores of the scorer's argmax pick on the 3,000 RL-val sets / 75 logs.
  * Read at u250 / u500 / u750 / u1000 from the run's dumps (`rl_val_eval.py`; K3 monitor `rls1_checks.py`, refdir = the arm's own `scores/`, as for R0c).
  * A selection tier is never an endpoint.
* **Endpoint (T1, as `RESULT_P2REP.md` labels it).**
  * NAVSIM v1.1 PDMS on the full navtest: 12,146 tokens, 136 logs. It is a pseudo-simulation of a 4 s open-loop plan, not closed loop.
  * Read on the final u1000 B and A of each passing arm. There is no navtest early stopping.
* **Estimator, everywhere:** paired log-cluster bootstrap, **10,000 resamples, seed 20260927**, all sub-scores ×100.
  * It answers "another draw of LOGS".
  * It does not answer "another training run" (§7 does) or "another inference run". The argmax scorer is deterministic up to kernels, so the I-rep in §7 covers that.

## 5. Bars, written before data. Each has a pass meaning and a fail meaning (§6)

* **S, safety (g5 and g3 as registered, B vs own A, RL-val):**
  * The TTC and NC point deltas are both ≥ -0.5 at **every** check (u250, u500, u750, u1000).
  * g3 holds at every check: pick / fan-mean path ≥ 0.894 and shortest-bin share ≤ 0.55, the guard against a collapse to short plans (`rl_val_eval.py:50-51`).
  * K3 still kills on two consecutive breaches. A single non-consecutive breach does not kill, but it fails S.
* **G, gain above the free frontier (RL-val, u1000, B vs own A):**
  * The PDMS lower 95 % bound is > 0.
  * **AND** the PDMS point is ≥ G(t), where t is the arm's TTC point delta and G(t) = max(+0.60, F(t)).
  * F is the linear interpolation of dose.json's Pareto points (-0.03, +0.60), (-0.27, +0.71), (-0.47, +0.84), held flat beyond -0.47.
  * So the bar is +0.60 at TTC ≥ -0.03, +0.71 at -0.27, and +0.84 at the bound.
  * +0.60 is 52 % of R0's +1.15.
* **L, lever claim (after Stage 2; necessary, not sufficient, two pairs):**
  * The TTC effect of the constraint, (Rkd B-A) - (unconstrained B-A), is > 0 at u1000 on both seeds. The seed-0 pair uses banked R0; the seed-1 pair uses R1o.
  * Each pair's effect is > 2 × F_seed, where F_seed = max(|ΔTTC(R0 vs R1o)|, |ΔTTC(R0d vs R1d)|) on the B - A contrasts.
  * The same rule is applied to PDMS: the reported cost of the lever.
* **E, endpoint ADOPT (navtest T1). Both pairs R0d-B vs R0d-A and R1d-B vs R1d-A must meet all of:**
  1. each ≥ +0.25 and their mean ≥ +0.5;
  2. the pooled lower bound is > 0;
  3. the mean contrast is > 2 × max(seed floor, I-rep |Δ|);
  4. **no PDMS sub-score separates adversely**: the TTC and NC upper bounds are ≥ 0. This is the RR1 clause that P2-s1 failed;
  5. every sub-score is reported, EP included.
  * REFUTED iff the pooled upper bound is < +0.25. Otherwise NOT PROVEN.

## 6. Outcomes committed in advance, each with its next lever (Rule Zero)

| outcome | condition | meaning | next lever |
|---|---|---|---|
| **O1 SAFE + ABOVE FRONTIER** | R0d: S and G | the objective repairs the trade at a gain the free mixture cannot buy | Stage 2 (R1d, then R1o), then bar L, then navtest E. ADOPT means the constrained EPO enters P6's scorer loss |
| **O2 SAFE, BELOW FRONTIER** | S holds; the PDMS lower bound is > 0 but the point is below G(t) | the dual overpays; the free mixture is the better deployable | (i) 0 A40: the alpha-0.25 logit-mixture navtest read (own prereg); (ii) one arm of (b) at the dual's measured mean λ over u250-u1000 (fixed weight, no dual noise; PI) |
| **O3 SAFE, FLAT** | S holds; the PDMS lower bound is ≤ 0 | no TTC-neutral EPO direction is reachable on this scorer; the objective route is closed | the logit mixture (0 A40); then the representational lever (P6 trunk), as D-REFE-SELRULE-FIT ranks it |
| **O4 BREACHED, DUAL HELD IN-SAMPLE** | K3 kill or S fails, AND the in-sample violation averaged over the 100 updates before the first breached check is ≤ 0.001, AND λ_TTC or λ_NC > 0 on ≥ 50 % of post-ramp updates | the constraint holds on the training data but does not transfer to RL-val: a generalisation gap | dual fed from a log-disjoint carve of RL-train (never RL-val). First, 0 A40: measure the train-vs-carve gap on banked R0 states with the A2 harness; then one arm (PI) |
| **O5 BREACHED, DUAL SATURATED OR LAGGING** | K3 kill or S fails AND (λ reached λ_max OR the in-sample violation over that window is > 0.001) | the dual could not hold even in-sample | the projected-gradient (first-order hard) constraint, or a PID dual. CPU unit proof first; one arm (PI) |
| **O6 INERT** | λ_TTC = λ_NC = 0.0 at every update | the constraint never bound. This contradicts §1.4's in-sample reach, so it is a mechanism failure (the "built, tested, unreachable" class) | the run is read as an R0 replicate. Audit the in-batch c_A / c_B logging against the A2 harness on the run's own dumps before any new arm |
| **O7 OTHER KILL** | K2 / K4 / K6, or g3 alone | per PREREG_RL_S1 §5. A g3 breach means the dual bought TTC with short plans | g3's quantity as a third constrained term (PI) |

R1d's outcomes use the same table. If R0d is O1 and R1d is not, the result is **NOT REPLICATED**: no lever claim, R1o does not run, and the next step is O2's (i).

## 7. Replicate plan and noise floors (H-ESTIM-SEED-1)

A separated CI from one seed is necessary, not sufficient. Each floor below answers a different question, and each effect is quoted next to the floor it cleared.

| floor | measured by | status |
|---|---|---|
| **episodes / logs** | the bootstrap of §4 | — |
| **same run again** | arm A of R0d vs arm A of banked R0 (same seed and flags; A never sees the lever) | MEASURED precedent, R0-A vs C0-A: PDMS +0.01 [-0.13, +0.14], TTC 0.00 [-0.13, +0.13] (`PREREG_RL_S1.md` §10.6). Re-measured for free |
| **training seed** | |R0d - R1d| (constrained) and |R0 - R1o| (unconstrained), each on the B - A contrast | one draw each (two pairs). Note: R0, R0b and R0c all ran at **seed 0**, so R1o is the programme's first measurement of whether the R-family TTC trade survives a seed change |
| **inference** | the I-rep of `PREREG_RL_S1.md` §3 (eval batch 8 vs 32, shuffled order, non-deterministic kernels) on R0d-B u1000 | dev box |

**Decision rule:** an effect smaller than 2 × the largest applicable floor is not called an effect. The g5 / K3 decision stays one seed per arm, as registered.

## 8. The four metric families (binding; per family, never pooled)

* **LONGITUDINAL.**
  * RL-val: TTC, EP, the g3 path ratio and the shortest-bin share.
  * Navtest: the RR3 component table (speed, headway, time-gap and TTC-to-lead members), as `RESULT_P2REP.md` reports it.
* **LATERAL.**
  * RL-val: **NOT COMPUTED**. `rl_val_eval` has no lateral instrument (n = 0, as in R0c's read).
  * Navtest: heading, curvature, yaw-rate and cross-track through the RR3 reader.
  * ⚠️ UNVERIFIED that this reader accepts RL-S1 scorer checkpoints. That is IMPL step 9: a work item, not an excuse.
* **TACTICAL.**
  * RL-val: lane-change rate, precision and recall vs the label base rate (`rls1_checks.lane_change_stats`, all arms).
  * Navtest: lateral-decision kappa, which may not fall by > 0.03 vs the control (`PREREG_RL_S1.md` §4).
* **STRATEGIC.** **PAUSED** under the PI ruling of 2026-09-19 (PI_DECISION_QUEUE item 25). Reported as paused: not a pass, and not omitted.

## 9. Kills

K0-K6 are as registered (`PREREG_RL_S1.md` §5), with these changes:
- K3 and K4 read against the arm's **own A**, as for R0c.
- **R1o (the control) runs K3 in OBSERVE-ONLY mode.** Its g5 is logged and never kills.
  - Declared before data: the unconstrained arm is expected to breach, as R0 did.
  - A kill at u500 would delete the matched control at the checks where the lever is read.
  - K2 and K6 stay live on R1o.

## 10. Diagnostics (pre-registered; they decide nothing)

Logged every 25 updates:
- λ_TTC(t) and λ_NC(t), and the per-update in-sample violations.
- c_A and c_B for both terms.
- The in-sample P1 gap: the TTC of B's argmax pick vs A's on the batch.
- The share of sets with a per-set hinge violation.
- Hits on λ_max.

Read after the run:
- The time-averaged λ over u250-u1000. It is O2's (b) weight.
- The A2 decomposition of R0d's states: how much of the residual TTC delta sits in the safety rows vs the progress rows.

## 11. Disclosures

* **Numbers seen before writing:**
  - every row quoted in §1;
  - the realstates train / val split (computed by this agent from `realstates.json` while designing; it fixed eps);
  - the dose curve, which fixed bar G.
* **None of these is a SAFEOBJ number.** The values in §2 were chosen from them, before any constrained run existed.
* **RL-val is the hyper-parameter tier.** eps, eta and the G frontier come from RL-val-adjacent reads, which is why the decision for ADOPT is navtest (E).
