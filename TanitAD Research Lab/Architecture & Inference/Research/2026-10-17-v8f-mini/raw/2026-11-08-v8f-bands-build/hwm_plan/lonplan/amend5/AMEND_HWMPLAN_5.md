# AMEND_HWMPLAN_5 (FROZEN 2026-11-17 by the MM; fills in MM_RULINGS_AMEND5.md): RA-L1, level 1's 0-2 s acceleration cost read through RA, zero training, on the 994 SCORE windows

**Status.** FROZEN 2026-11-17 by the MM, as drafted the same day by the Arch+Inference agent; the fills are in `MM_RULINGS_AMEND5.md`. No SCORE window has been planned or scored for this draft, and no SCORE outcome was opened. The only RA passes made for it ran on TRAIN windows (section 3). The Master Mind (MM) freezes it and sets the fills of section 10; **the base solver is its one open fill**, and all three of its branches are written out.

**Frozen inputs this amends** (paths under `raw/2026-11-08-v8f-bands-build/hwm_plan/` unless named):
* `PREREG_V8F_HWMPLAN_DRAFT.md` `ac20d690`, `FREEZE_HWMPLAN.md` `ae2bbe40`, `AMEND_HWMPLAN_1.md` `dcbd55a9` (world model R35), `AMEND_HWMPLAN_2.md` `b9a7602a`, `AMEND_HWMPLAN_3.md` (LON-PLAN), `IMPL_CHOICES_HWMPLAN.md` `c478d6cf`.
* `lonplan/amend4/AMEND_HWMPLAN_4.md` (L2-IMG), frozen 2026-11-17, fills in `MM_RULINGS_AMEND4.md`. Its section 6 names this read as the next lever in every outcome row and sets its base solver mechanically.
* `lonplan/readout/RESULT_LONPLAN.md`: SEEDS-DISAGREE, CEILING FAILED at both chassis. Its section 4 names this read after L2-IMG.
* ACTRSP Amendment 1, `raw/2026-11-16-v8f-actrsp-prereg/amend1/AMENDMENT_1_BCAL_DRAFT.md` (frozen at `4e122aa5…`, commit `76a8012f`), sections 1.4 and 1.5: the readout RA and R35's response through it.
* Code of record: `hwm_plan.py` `5a072272…`, `run_hwm.py` `935b3422…`, `hwm_eval.py` `79392075…`, `hwm_data.py` `a2d3922e…` (sha256, MEASURED equal today in `E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/hwm_plan/`).

**Tier.** Every planner row is **T0**: the goal is HWM-ORACLE, the encoded true-future latent at frame 30, and nothing is trained. A T0 row is a world-model diagnostic and never a driving-performance number. RA-ORACLE (section 4) uses the true frame-10 latent as its target: a stronger oracle, a ceiling and never a plan. HOLD and PRIOR are T1-kin references. The TRAIN measurements of section 3 are in-sample feasibility records, never results.

## 0 What changes (the whole amendment, in one table)

| item | the base solver (fill B, section 4) | this amendment |
|---|---|---|
| level 1's acceleration knots 1-10 | the latent cost: squared distance of the lifted level-1 rollout to K1 = 4 level-2 rollout states (`plantest2_lib.py:117-127`) | **the RA pass:** re-planned from HOLD_c with cost (RA(F¹⁰(z₀, a) − z₀) − â\*)², one candidate, the library's optimiser at level-1 settings |
| the target â\* | none | RA's read of a source plan's knots 1-10 through the same world model: level 2's primitives (**RA-L2**) or the base's level-1 knots (**RA-L1**) |
| level 1's yaw knots, knots 11-30, levels 2 and 3 | the base's | **unchanged, bit for bit** (the base's banked plan) |
| arms | the base | **RA-L2** and/or **RA-L1** (lever arms), **MEAN-L2 / MEAN-L1** (matched flat mean), **RA-NEG** (deliberate regression), **RA-SHUF** (shuffled readout), **RA-OFF** (identity), **RA-ORACLE** (ceiling, beside) |
| deciding read | 0-2 s LONGITUDINAL `acc_mae`, CEILING = beat HOLD | the same cell and the same bar |
| everything else | | unchanged: the two R35 chassis, the 994 SCORE windows, the TRAIN priors, the TRAIN/SCORE split, λ = 0, the estimator, the families, controls 1-15, and LON-PLAN's / AMEND_4's controls on the base |

## 1 Why, from measurements

* **The planner keeps level 1's 0-2 s acceleration far above HOLD, and the latent cost does not point anywhere useful near HOLD.**
  * A0 − HOLD on 0-2 s `acc_mae` = +0.9651 [+0.8949, +1.0328] (s0) and +0.8685 [+0.7929, +0.9409] (s1).
  * REG at λ\*_k closes the gap only by pulling toward HOLD, and at s1 does not beat a matched pull toward HOLD: A1 − S = −0.0016 [−0.0134, +0.0112].
  * MEASURED, T0, paired episode-cluster bootstrap B 2,000 seed 0, 994 windows / 71 episodes: `lonplan/readout/RESULT_LONPLAN.md` sections 1-2.
* **Level 1's cost is a 128-dimensional distance in which acceleration is a small part.** Most of a level-1 latent change is not action-driven: the whole one-step change of R35 is 8.7-10.3 % action response by norm (ACTRSP card section 1, read in Amendment 1 section 1.5 as a wrong-scope quantity). A distance over all 128 dimensions can therefore be reduced without moving the 2 s acceleration the right way. MEASURED basis; the mechanism is HYPOTHESIS.
* **RA is the one level-1 readout with transition skill on acceleration.** Ridge on `z_{t+10} − z_t` → mean acceleration over the 10 transitions.
  * Clip-held-out R² 0.211 [0.148, 0.274] (s0) and 0.206 [0.142, 0.269] (s1); +0.163 / +0.158 over the best state-only model.
  * Transition-only slope with z_t exactly fixed: s_swap 0.2008 [0.1404, 0.2648] / 0.2032.
  * R35's own 2 s acceleration response read through RA is G_aa 1.244 [1.116, 1.376] / 1.202 [1.066, 1.333] of the real one: not deficient.
  * MEASURED on the 141 TRAIN clips, clip-cluster bootstrap B 1,000: Amendment 1 sections 1.4-1.5, `amend1/out/kstep_R35_s{0,1}.json`. No yaw-rate readout is calibratable at level 1 (RY's s_swap CI includes 0), so RA reads acceleration only.
* **Level 2's own primitives carry more 0-2 s acceleration than level 1's plan (TRAIN hint).** Under P, level 2's chosen primitives as knots 1-10 read 0-2 s `acc_mae` 1.0729 / 1.0422 m/s² against level 1's 1.2024 / 1.1913; under P23 1.0953 / 1.0799 against 1.1590 / 1.1670. MEASURED, TRAIN, in-sample, 128 windows / 32 clips, no interval: `amend4/probe_s{0,1}_p23.json`, AMEND_4 section 3. So level 1, planning toward level 2's sub-goals with the latent distance, loses acceleration information that level 2's own actions hold. Which level's actions RA reads is therefore a design question (section 2.2).
* **HYPOTHESIS H-RAL1.** Replacing level 1's latent distance on the acceleration channel by the squared difference of two RA reads (the plan's and a source plan's, through the same world model) lets level 1 carry the source's 2 s acceleration intent without the 127 other latent directions, and starting from HOLD_c it moves only as far as that intent asks. Then 0-2 s `acc_mae` falls against the base, and possibly below HOLD.
* **What is expected, stated before any SCORE number (ESTIMATED from TRAIN, section 3).** 
  * CEILING needs X to beat HOLD on 0-2 s `acc_mae`. On TRAIN, in-sample for R35 and for RA, **even the best target an upper level could hand level 1 (the TRUE frame-10 latent) reads far above HOLD through the RA pass:** RA-ORACLE − HOLD = +0.541 [+0.482, +0.601] (s0) and +0.586 [+0.525, +0.651] (s1).
  * No zero-training knob on this path closes it (section 3): a HOLD-anchored REG at any λ from 0.003 to 1000 stays above HOLD or collapses onto it; a gate on R35's own gain stays above HOLD at every quantile; one candidate beats sixteen.
  * So the expected labels are **RAL1-NULL or RAL1-HARM with CEIL-ORACLE failing, i.e. READ-LIMITED** (and the same for RA-L2 under B1/B2). CEILING is not the expected outcome.
  * The read is still worth running: about 1 h of CPU, held out for RA (RA never saw a SCORE window), and its RA-ORACLE row settles READ-LIMITED with the programme's estimator instead of an in-sample TRAIN record. Whether to run it or record the expectation is MM question M7.

## 2 The operation

### 2.1 What RA consumes and returns
* **RA** = `readout_K10_R35_s{k}.pt`, the `RA` entry: `ra(D) = ((D − mu) / sd) · W + ym`, with W of shape (128, 1), fit by ridge at λ_RA = 1.0 (s0) and 0.1 (s1), each chosen on clip-grouped folds and stored in the file (`amend1/amend1_kstep.py`, code `a13f6f16`).
  * **Input:** a level-1 displacement D = z₁₀ − z₀ (128 numbers). In the planner, z₀ = `z_now` and z₁₀ = **F¹⁰(z₀, a)**, R35's level-1 rollout after 10 actions (`st.rollout(1, z₀, a)[..., 9, :]`, context 1 = `plan-ctx 1`), so RA reads knots 1-10 and never knots 11-12.
  * **Output:** the mean scaled acceleration over the 10 transitions. Raw m/s² = 2.0 × scaled (`hwm_plan.ACC_SCALE`).
  * Only W's direction and R35's response along it enter the RA pass. Both sides of its cost are RA reads of R35 rollouts from the same z₀, so `mu`, `ym` and any state part of the read cancel.
* **The frozen weights (sha256):**
  * s0 `edab7dce7b57f9ae21d90e5cc317c914aed432751fde7a7d4e6cf6af0748a6cf` (git blob `7fcafdfb`),
  * s1 `a6d252aa405f624a7cea4129a7f7d65ba5c047045d88e89b340a8abb935bb495` (git blob `95a43146`),
  * at `raw/2026-11-16-v8f-actrsp-prereg/amend1/out/readout_K10_R35_s{0,1}.pt` on `agent/arch-inf-20260803` (landed `76a8012f`; tip `e5324710`). Every run asserts the sha256 and refuses on a mismatch.

### 2.2 Which level's actions RA reads
* **The plan side is always level 1's.** The variable is level 1's 12 acceleration knots, as now. RA reads its rollout after knots 1-10.
* **The target side is a source plan, read through the same world model:** â\* = ra(F¹⁰(z₀, [src_acc, yaw_b]) − z₀), where `src_acc` is the source's acceleration on knots 1-10 and `yaw_b` is the base's level-1 yaw on the same knots. With the same yaw on both sides, the target depends on the source's acceleration only.
  * **RA-L2:** the source is **level 2's own primitives**, knots 1-10 (`l2_prim_raw[:, :10]`). They exist under P and P23 only. The TRAIN hint of section 1 favours this source.
  * **RA-L1:** the source is **the base's level-1 plan**, knots 1-10 (`a1_raw[:, :10]`). It exists in every branch. This is the literal reading: level 1's own latent-cost plan, read at 2 s through RA, then realised from HOLD_c.
  * **Not a source: A0's level 2.** A0's decode starts at macro 3 (`hwm_plan.stitch`, j0 = 10 // 3), so A0 has no level-2 primitives for knots 1-10. Decoding macros 0-3 would read off-image macros whose decode hits a bound on 94-97 % of rows (`D-HWM-UPPER-OFFIMAGE`). A0's level-2 macros are also not banked (`score/A0_s0/plans_hwm_oracle_s0.npz` holds `plan_raw`, `cost1`, `iters1`, `idx`, `a1_raw`, `decode_resid`).
* **Why RA does not read level 2's primitives as the plan itself:** that is P2-0, already a beside row of AMEND_4. It reads level 2's per-knot profile, which on TRAIN sits at 1.04-1.10 m/s², about 3.7× HOLD. The RA pass keeps one number of the source (its 2 s acceleration as R35 sees it) and takes the profile from HOLD_c.

### 2.3 The RA pass (implementation: post-hoc on the base's banked plans, study-local)
* **Order.** The base plan is complete before the pass: levels 3, 2 and 1, knots 11-30, and every random number are the base's banked SCORE plan. The pass draws no random number.
* **Variable.** a_acc (B, 1, 12): level 1's acceleration channel, all 12 knots, scaled units. Knots 11-12 do not enter RA's read (structural zero gradient, control C-GRAD).
* **Start.** HOLD_c's acceleration, **one candidate (S = 1)**. RA's cost constrains one number of a 12-knot plan, so it has a manifold of minima. A pick among noisy candidates would then be decided by the noise orthogonal to RA's direction. MEASURED on TRAIN (section 3, RA_ORACLE_S16): RA-ORACLE with 16 candidates reads +0.313 [+0.278, +0.350] (s0) and +0.303 [+0.267, +0.340] (s1) worse than with one (`ra5_ceiling_s{0,1}.json`, RA_ORACLE_S16).
* **Cost.** (ra(F¹⁰(z₀, [a_acc, yaw_b]) − z₀) − â\*)², per window.
* **Optimiser.** The library's `st._optimize` at level-1 settings: AdamW, lr 0.1, at most 90 iterations, patience 10, rel_tol 1e-4, weight decay 0, clipped to the level-1 prior box B₁ after every step (`hjepa.py:948-990`, unchanged).
* **Output.** The best iterate's acceleration on knots 1-10 replaces the base plan's knots 1-10 acceleration. Yaw on knots 1-30 and acceleration on knots 11-30 are the base's, bit for bit.
* **What replaces the latent distance.** On the acceleration channel of knots 1-10, the squared 128-dimensional distance of level 1's lifted rollout to level 2's sub-goals is replaced by the squared difference of two RA reads. On the yaw channel nothing is replaced: the latent cost's yaw stays, because no level-1 yaw readout is calibratable (Amendment 1 section 1.4).
* **Code.** A study-local module `lonplan/amend5/ra_l1.py` with a run wrapper. It reads `windows.npz` (z_now, a0), the base plan files and the readout, and writes new plan files. The code of record is untouched (fill M2). `ra5_ceiling.py` (section 3) is a working prototype of the pass.

### 2.4 What stays fixed
* **λ = 0** (the base's), `--seed-mode hold`, T3 2, `plan-ctx 1`, `vocab off`, `constr-weight 0`, S 16 and n_iters 90 in the base, inference seeds 0 and 1, `OMP_NUM_THREADS=2`.
* **The two chassis:** R35_s0 `d920d62c…` and R35_s1 `a97e6017…` (sha256 of `ckpt_last.pt`), each with its own TRAIN priors from `latents_train_R35_s{k}.pt` (`76a2d4f5…` / `692eb002…`).
* **The 994 SCORE windows in 71 episodes:** `windows.npz` sha256 `371c753b…` (s0) and `8833e292…` (s1).
* **The TRAIN/SCORE split.** Priors and RA come from TRAIN latents only (2.5). No step reads a SCORE window before the freeze.
* **The readout weights, frozen:** sha256 above, no refit, no rescaling, no dose. RA's s_swap is not used: it cancels in the pass.

### 2.5 The leak check (RA was not fit on SCORE windows)
* **The training record.** `amend1_kstep.py` reads one data file, `latents_train_R35_s{k}.pt` (header: "141 TRAIN clips; no eval window"), builds every 10-step stretch with finite actions, and fits the frozen RA on all of them (lines 189-193). Its record `kstep_R35_s{k}.json` gives n_pairs 8,306 and n_clips 141 at both chassis. Amendment 1's status line: "No eval latent was read, and nothing was fit on a scored window."
* **Re-derived today (MEASURED, `ra5_ceiling_s{0,1}.json`, `C_RA_FIT`).** From the local `latents_train_R35_s{k}.pt` (the same sha256 as the phase-1 record), the script rebuilds RA's fit set exactly as `amend1_kstep.py` does and refits at the frozen λ_RA = 1.0:
  * **8,306 pairs in 141 clips at both chassis**, equal to the fit record;
  * the refit reproduces the frozen weights: max |ΔW| 2.4e-10 (s0, max |W| 0.016) and 1.8e-9 (s1, max |W| 0.028); max |Δmu| ≤ 3.8e-10, max |Δsd| ≤ 1.0e-9, |Δym| 0.0;
  * **0 of the 141 TRAIN clips are among the 71 SCORE episodes** (the eid list of the 994 SCORE windows, read for ids only).
  * ⇒ RA was fit on exactly these TRAIN latents and on no SCORE window.
* **At inference, the RA arms read** z_now, HOLD_c (a0, the measured acceleration at NOW, as HOLD and the base already read) and the base plan (state, goal and HOLD only). No logged future enters an RA arm. RA-ORACLE does read the true frame-10 latent, by design, and is a ceiling row, never an arm.
* **Goal and situation stay disjoint.** No situation classifier exists anywhere in this read. The goal is HWM-ORACLE, unchanged.

### 2.6 Alternatives considered and not recommended
* **RA added to the latent cost** (`cost1 + μ · RA term`). It brings a dose μ and a TRAIN selection of it, and it keeps the 128-dimensional distance on the acceleration channel, which is the term section 1 suspects.
* **RA as a selector** among the base's 16 level-1 candidates. Those candidates are not banked, and the pick would still be among latent-cost solutions.
* **RA on the goal** (z₃₀ − z₀). RA was fit on 2 s displacements; a 6 s displacement is outside its fit (a wrong-scope read).
* **RY for yaw.** Not calibratable (s_swap CI includes 0 at both chassis).

## 3 Measurements made for this draft (CPU, TRAIN windows only, zero training)

**Instruments**, banked next to this draft, each importing the code of record unchanged (from `E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/`):
* `amend5/ra5_ceiling.py` (`38902a20…`) → `ra5_ceiling_s{0,1}.json` (`7f8182cb…` / `df20aab8…`): C-RA-FIT and the rows at λ 0.
* `amend5/ra5_reg.py` (`302c816c…`) → `ra5_reg_s{0,1}.json` (λ 0-1000) and `ra5_reglow_s{0,1}.json` (λ 0.003-0.3). **Added after the first read** showed the pass amplifying; its constants and grids were fixed before each ran. The first grid ran before the optional grid argument was added (that one line differs).
* `amend5/ra5_gate.py` (`7ef0934b…`) → `ra5_gate_s{0,1}.json`: **added after parts 1-2**, constants fixed before it ran.
* **Windows:** LON-PLAN's TRAIN selection set, all 564 windows in 141 TRAIN clips, the same at both chassis. RA was FIT on these clips, so every RA row is in-sample for RA as well as for R35, and optimistic. **Estimator:** clip-cluster percentile bootstrap, B 2,000, seed 0, over the 141 clips; it answers another draw of TRAIN clips with R35 and RA fixed. No interval here is a result.
* Yaw was held at HOLD_c in these passes (the SCORE arms hold the base's yaw); yaw moved by exactly 0.0.

| row (0-2 s `acc_mae`, m/s², TRAIN in-sample) | s0 | s1 |
|---|---|---|
| HOLD (a0 held) | 0.2965 | 0.2965 |
| ORAC_MEAN: the logged 2 s mean, held (an oracle, never a plan) − HOLD | **−0.1046 [−0.1259, −0.0864]** | −0.1046 [−0.1259, −0.0864] |
| RA_READ: RA on the TRUE displacement, held, no world model − HOLD | +0.1942 [+0.1621, +0.2286] | +0.1972 [+0.1643, +0.2330] |
| **RA_ORACLE: the RA pass, target from the TRUE frame-10 latent − HOLD** | **+0.5410 [+0.4816, +0.6010]** | **+0.5856 [+0.5250, +0.6512]** |
| RA_PLANT: target formed through R35 from the logged actions − HOLD | −0.0097 [−0.0301, +0.0107] | +0.0049 [−0.0158, +0.0263] |
| RA_PLANT: mean abs(plan 2 s mean − logged 2 s mean); a0's is 0.2657 | 0.1441 | 0.1540 |
| RA_SHUF − RA_ORACLE (W permuted) | **+0.4549 [+0.3864, +0.5222]** | **+0.3442 [+0.2656, +0.4209]** |
| RA_NEG − RA_ORACLE (target mirrored) | +0.0381 [−0.0298, +0.1074] | +0.0321 [−0.0362, +0.0983] |
| RA_ORACLE_S16 − RA_ORACLE (16 candidates, argmin) | +0.3133 [+0.2782, +0.3497] | +0.3031 [+0.2672, +0.3399] |
| RA_ORACLE + REG at λ 0.3 (the largest λ with D ≥ 0.10) − HOLD; D | +0.0132 [+0.0040, +0.0226]; 0.131 | +0.0152 [+0.0055, +0.0250]; 0.137 |
| RA_ORACLE + REG at λ 0.1 / 0.03 / 0.01 − HOLD | +0.056 / +0.170 / +0.312 | +0.065 / +0.188 / +0.338 |
| RA_ORACLE + REG at λ ≥ 1 | HOLD-collapsed (D 0.036-0.069), +0.0001 to +0.0019 | HOLD-collapsed (D 0.037-0.071) |
| R35's gain along W, g (scaled per scaled): q10 / q50 / q90 | 0.037 / 0.224 / 0.510 | 0.010 / 0.227 / 0.516 |
| RA_ORACLE − HOLD by gain tercile: low / mid / high | +0.760 / +0.603 / +0.259 | +0.796 / +0.644 / +0.318 |
| gain gate (the pass where g ≥ the quantile, HOLD_c elsewhere) − HOLD: q50 / q75 / q90 | +0.175 / +0.065 / **+0.020 [+0.008, +0.035]** | +0.200 / +0.072 / **+0.021 [+0.009, +0.037]** |
| Pearson r with the logged 2 s mean: a0 / RA_READ | 0.776 / 0.465 | 0.776 / 0.480 |
| r of the 2 s departure from a0 against the logged one: RA_READ / RA_ORACLE / RA_PLANT | 0.381 / 0.288 / 0.678 | 0.394 / 0.284 / 0.695 |
| the pass's median final cost (scaled²): ORACLE / PLANT / SHUF | 0.0 / 0.0 / 3.4e-4 | 0.0 / 0.0 / 1.1e-3 |
| wall, s per window per S = 1 pass (2 threads, 2-3 processes side by side) | 0.046 | 0.042 |

**What the table says (TRAIN, in-sample):**
1. **The format is not the limit.** A flat 0-2 s plan at the true 2 s mean beats HOLD by 0.105 m/s². The RA pass produces exactly such a flat plan from HOLD_c.
2. **RA's read is weaker than a0.** Reading even the TRUE 2 s displacement, RA predicts the 2 s mean at r 0.47-0.48, against a0's 0.78. Held as a plan, it sits +0.19-0.20 above HOLD.
3. **Inverting through R35 makes it worse.** With the true frame-10 target, the pass lands +0.54-0.59 above HOLD. R35's gain along W is low and uneven (q10 0.01-0.04), and the inversion amplifies R35's error along W by about 1/g. Even the top gain tercile stays +0.26-0.32 above HOLD, and the strictest gate (q90) still reads +0.020-0.021, separated-worse.
4. **Even a target R35 can reproduce exactly reaches only HOLD** (RA_PLANT). The read keeps one number, weights the knots by R35's response rather than uniformly, and is clipped on low-gain windows. It recovers about half of a0's miss to the logged mean (0.144-0.154 against 0.266).
5. **No REG dose rescues it.** With LON-PLAN's HOLD-anchored REG, every admissible λ (D ≥ 0.10) stays separated above HOLD, and every larger λ collapses onto HOLD.
6. **The controls behave as designed.** The pass reaches its target read (median cost 0.0). The shuffled readout is separated-worse (+0.34-0.45). One candidate beats sixteen (+0.30-0.31). The mirror is not separated from RA-ORACLE, because most of the departure is amplified error rather than direction; that is why RA-NEG sits beside and RA-SHUF is the regression arm.

## 4 Arms (SCORE: the 994 windows in 71 episodes, R35_s0 and R35_s1)

All RA passes run on the base's banked SCORE plan files, at both chassis. The pass is deterministic, so an arm's inference seed is its base's.

**The base solver (the one open fill), set mechanically by AMEND_HWMPLAN_4's labels (its section 6) and written into the Log before any RA plan exists on SCORE:**
* **B1:** an AMEND_4 lever arm reads CEILING. The base is that arm (both: P23 if its Δ_H point is lower at both chassis, else P).
* **B2:** else an arm reads NARROWS. The base is that arm (both: the same tie rule on Δ_0).
* **B3:** else A0's upper levels: LON-PLAN's banked `score/A0_s{k}/plans_hwm_oracle_s{0,1}.npz`.

| arm | what it is | B1 / B2 (base P or P23) | B3 (base A0) | seeds | role |
|---|---|---|---|---|---|
| **BASE** | the base's banked plans | P or P23 (AMEND_4 files) | A0 (LON-PLAN files) | 0, 1 | control |
| **RA-OFF** | the wrapper with the pass disabled | — | — | 0 | identity: must equal BASE bit for bit (C-OFF5) |
| **RA-L2** | target from level 2's primitives | **committed lever** | not computable on A0 (2.2); computed on P's banked files **beside**, no outcome decided | 0, 1 | lever |
| **RA-L1** | target from the base's level-1 plan | second lever arm, its own label, AMEND_4's multiplicity guard | **committed lever** | 0, 1 | lever |
| **MEAN-L2 / MEAN-L1** | HOLD_c's acceleration + one uniform shift to the source's mean acceleration over knots 1-10; no world model, no RA | with RA-L2 / RA-L1 | with RA-L1 | 0, 1 | matched control: does the read through R35 add anything over the source's plain 2 s mean? |
| **RA-SHUF** | the committed arm with W permuted by `RandomState(0)` on both sides (target and plan) | yes | yes | 0 | **deliberate regression and shuffled-readout control; must read worse than the committed arm** (C-SHUF) |
| RA-NEG | the committed arm with its target mirrored about HOLD_c's own read: â\*_NEG = 2 ra(F¹⁰(z₀, [HOLD_c, yaw_b]) − z₀) − â\* | yes | yes | 0 | beside, no bar: does the target's sign reach the plan? |
| **RA-ORACLE** | the RA pass with target ra(z₁₀ − z₀), the TRUE frame-10 latent; yaw at HOLD_c | yes | yes | — | **beside, a ceiling** (never a plan): what the RA read could give with a perfect level-1 target |
| P2-0 | AMEND_4's beside row (level 2's primitives as knots 1-10) | yes | on P's files | 0 | beside |
| HOLD, PRIOR | references | | | — | T1-kin |

* **No new FLAT arm, no λ > 0 arm, no new upper-level plan.** A REG + RA arm would need its own TRAIN selection; it is named under NARROWS.

## 5 Read

* **TRAIN stage, before any SCORE plan (beside, in-sample, never a result).** RA-L1 (and RA-L2 on P / P23), MEAN, RA-NEG and RA-SHUF on the base's TRAIN plans (LON-PLAN's `sel_s{k}/lam_0` for A0; AMEND_4's TRAIN stage for P / P23; 564 windows in 141 clips). It reports each arm − BASE and − HOLD per family with the clip-cluster bootstrap, and it must read RA-SHUF − (committed arm) > 0 at both chassis (the sign check of C-SHUF). It changes nothing in this design.
* **Contrasts**, per band (0-2 s, 2-6 s, full 6 s) and per family, never pooled:
  * **X − HOLD and X − BASE** for each lever arm X (deciding);
  * **X − MEAN-src** (specificity, labels INFORMATIVE / MEAN-ONLY);
  * RA-SHUF − X (C-SHUF), RA-SHUF − BASE and RA-NEG − X (beside);
  * RA-L2 − RA-L1 where both exist; X − P2-0; X − A1 (LON-PLAN's REG, beside);
  * RA-ORACLE − HOLD and RA-ORACLE − X (the ceiling rows);
  * the inference-seed replicates of X and BASE.
* **Mechanism rows (beside, direction stated now, HYPOTHESIS; no outcome is decided on them):**
  * **DIR-ACC** as LON-PLAN: the Pearson r of the plan's 0-2 s departure from HOLD against the logged one, for BASE, X, MEAN, RA-NEG, RA-SHUF and RA-ORACLE. H-RAL1 predicts r(X) ≥ r(BASE).
  * **DEP:** the mean |2 s mean departure from HOLD| of X against BASE. The pass is expected to shrink it.
  * **HIT:** the median final RA cost of the pass (the target reached), and the share of windows whose pass hit B₁ on any knot.
* **Estimator.** Paired episode-cluster bootstrap (`taniteval/ci.py` through `hwm_eval.contrast`), B 2,000, seed 0, over the 994 SCORE windows in 71 episodes, through `lonplan/contrast_l1full_strat.py` (C-STRAT). It answers *another draw of EPISODES*, and nothing else.
* **Inference floor.** F_I,k(q) = max(|X seed 1 − X seed 0|, |BASE seed 1 − BASE seed 0|) on the row's point means, per chassis and lever arm, as AMEND_HWMPLAN_3 defines it. It answers *another INFERENCE run*. The pass itself draws no random number, so all of X's inference variance comes through its base. For scale, LON-PLAN's 0-2 s `acc_mae` floors were 0.0073 (s0) and 0.0041 (s1) (MEASURED, RESULT_LONPLAN section 1).
* **Training runs.** Nothing is trained. *Another TRAINING run* is answered only by the two chassis agreeing; a claim about the R35 recipe needs both. ACTSENS's `NULL_s2` / `NULL_s3` (R35-recipe replicates) carry a pass near the floor (section 6). RA itself is one fit per chassis; a second readout fit is not a training run of the world model and is not read here.

## 6 Committed outcomes (per lever arm X; both chassis required; a label at one chassis only is SEEDS-DISAGREE, and the less favourable label's next step applies)

**Definitions, per chassis k and lever arm X, on the 0-2 s band, inference seed 0:**
* Δ_H = X − HOLD (`acc_mae`), the CEILING cell.
* Δ_B = X − BASE (`acc_mae`), the lever's effect.
* Δ_M = X − MEAN-src (`acc_mae`), the specificity cell.
* G_x = X − BASE (`xtrack`, 0-2 s), the guard. (`yaw_mae` and `head_err` read exactly 0.0 against BASE by construction, control C-YAW-ID.)
* **"Separated-better beyond floor":** the CI lies wholly below 0, |Δ| > F_I,k(q), and the seed-1 point Δ is also below 0. **"WORSE":** the CI lies wholly above 0 and Δ > F_I,k(q).
* D(X): mean |plan acc − HOLD acc| over knots 1-10 in TRAIN prior std (LON-PLAN's D). X is HOLD-collapsed if D < 0.10.
* **The ceiling flag CEIL-ORACLE** (beside, decides the next lever only): RA-ORACLE − HOLD separated-better at both chassis.
* RA-L2's label is written RAL2-…, RA-L1's RAL1-….

| outcome | condition (both chassis) | next lever (Rule Zero; started in the same work package unless blocked) |
|---|---|---|
| **CEILING** (the primary criterion passes) | Δ_H separated-better beyond floor; G_x not WORSE; D(X) ≥ 0.10 | (1) **MM:** the RA pass becomes the candidate level-1 longitudinal cost for later HWM-PLAN arms, after the multiplicity guard below and, if \|Δ_H\| < 2 F_I,k at either chassis, the same read on `NULL_s2` / `NULL_s3`. (2) **The 2-6 s band:** a K = 30 acceleration readout fitted on the same 141 TRAIN clips by RA's procedure (CPU, zero training of the world model), read at level 2 under its own amendment. (3) **The T1-kin read,** HWM-PLAN with a predicted goal under base + RA: **blocked** on a goal head trained under its own card and gate (box 2, PI, `QUESTIONS_LONPLAN.md` Q2a) |
| **NARROWS** | not CEILING; Δ_B separated-better beyond floor; G_x not WORSE. Qualifier **INFORMATIVE** if Δ_M is separated-better beyond floor, else **MEAN-ONLY** | The read carries part of the 0-2 s defect, not enough to beat HOLD. Next, by CEIL-ORACLE: **if CEIL-ORACLE holds,** the source's 2 s intent is the limit: (a) X + REG with λ re-selected on TRAIN by AMEND_HWMPLAN_3 section 3's procedure on X; (b) world-model action-sensitivity training (box 2, PI, Q2b), named with that blocker. **If CEIL-ORACLE fails,** the read itself is the limit: the READ-LIMITED row's lever |
| **NULL** | Δ_B not separated-better beyond floor at either chassis, and not WORSE | Next, by CEIL-ORACLE. **If it holds:** a perfect level-1 target would beat HOLD through RA, and the hierarchy's 2 s intent is what fails. The lever is the source: RA-L2 under the other AMEND_4 arm (P23 if X was on P, and the reverse), then action-sensitivity training (PI, Q2b). **If it fails:** READ-LIMITED, below |
| **READ-LIMITED** (a NULL, NARROWS or HARM row with CEIL-ORACLE failed) | RA-ORACLE − HOLD not separated-better at either chassis | Even the true frame-10 latent, read through RA and inverted through R35, does not beat HOLD. The limit is RA's single direction or R35's error along it, not the hierarchy. Section 3 already says a flat 2 s plan at the true mean beats HOLD, so the 2 s format is not the limit, and every zero-training knob of this path (REG dose, gain gate, candidate count) was tried on TRAIN without effect. The next levers, ranked by section 3's terms (Rule Zero rule 5), are both blocked: (1) **R35's response along RA's direction**, the larger term (RA-ORACLE − RA_READ ≈ +0.35 / +0.39 on TRAIN; gain q10 0.01-0.04): ACTRSP Option B, the calibrated 2 s acceleration term whose predicted effect is to flatten the gain across states (Amendment 1 section 2.3), or world-model action-sensitivity training. **Blocked** on box 2 and the PI (Q2b), asked with this read's numbers. (2) **RA's precision** (RA_READ +0.19 / +0.20 above HOLD; r 0.47-0.48 against a0's 0.78): a 2 s readout fit on more TRAIN clips by RA's procedure, CPU only, **blocked** on the level-1 latents of more TRAIN clips from box 2's `cache_full/train/clips` (PI P3). ESTIMATED from the gate rows: neither term alone is enough, because even the top gain tercile stays +0.26-0.32 above HOLD |
| **HARM** | Δ_B WORSE at either chassis, or G_x WORSE | First, C-OFF5, C-YAW-ID, C-GRAD, C-PLANT and C-RA-FIT are re-run on the SCORE argv: a wiring defect is the first suspect. Then the row of CEIL-ORACLE, as NULL |
| **INVALID** | any control of section 8 fails, C-SHUF included | Nothing is read. The failing control is fixed and the arm re-run |

* **Multiplicity guard.** Under B1 and B2 two lever arms are read against the same bar. A CEILING at one arm only changes the solver of record after the same read on `NULL_s2` / `NULL_s3` agrees.
* **The beside RA-L2 on P under B3** decides no outcome. If it reads CEILING-like cells while RA-L1 does not, that is reported as a HYPOTHESIS for the MM, with its own amendment before any solver change.
* **The training-gated lever behind every non-CEILING row** is world-model action-sensitivity training on box 2. It is not pre-approved (`QUESTIONS_LONPLAN.md` Q2b, `MM_RULINGS_AMEND4.md` P2) and goes to the PI with this read's result. Amendment 1 section 1.5 (G_aa 1.2) weakens its case on acceleration, and this read's CEIL-ORACLE row is what sizes it.
* **Secondary label, 2-6 s band:** none. The pass changes no knot beyond 10, so per-knot 2-6 s metrics read exactly 0.0 against BASE (C-BAND-ID). Integrated cells carry the 0-2 s change forward and are reported.
* **A failed bar is reported as FAILED, then its next lever runs.** No arm, window, constant, readout or criterion changes after any SCORE number is seen.
* **Done** means one of two things: (a) CEILING, with next steps (1) and (2) started; or (b) the next row's lever drafted and started, or, where it is blocked, named with its blocker and what unblocks it.

## 7 Families (all four, ACTRSP section 4.3 as LON-PLAN applied it; T0)

* **LONGITUDINAL:** `acc_mae` (deciding, 0-2 s) and `speed_err`.
  * Distance-keeping: `hw_err`, `tg_err` and `ttc_err` on windows in state LEAD, with n per cell; signed `hw_d`, `tg_d`, `ttc_d` on arm rows. LON-PLAN's 0-2 s cells had 214 windows / 32 episodes, and `tg` 182 / 31 (MEASURED, `c_A1_A0_s{0,1}.json`). The 0-2 s band uses the K = 10 lead block; 2-6 s and the full 6 s use K = 30.
  * Under the pass the speed profile changes from knot 1 on, so every integrated LONGITUDINAL cell, distance-keeping included, can move in every band.
* **LATERAL:** `yaw_mae` (0-2 s) and `head_err` read **exactly 0.0 [0, 0] against BASE in every band** (yaw is the base's, and heading integrates yaw only): a control identity, not a result. `curv_err` (steps with logged speed ≥ 1 m/s, n printed; 950 windows / 71 episodes in LON-PLAN), `xtrack` (the guard) and `ade` move through speed.
* **TACTICAL:** plan-derived `tac_man_dis`, `tac_lat_dis` and `tac_lon_dis` per band, with kappa against the logged manoeuvre. The longitudinal class can move; the lateral class can move only through speed. Goal and anchor selection: n = 0, NOT COMPUTABLE (the planner has no decision output).
* **STRATEGIC, reported in full** (the item 25 pause ended by PI ruling 2026-11-16):
  * `s_adh` with its signed-degree twin, and `s_false`, full 6 s band only, with the logged-path rate beside. LON-PLAN's n: `s_adh` 12 windows / 4 episodes (s0) and 8 / 5 (s1), not decision-grade; `s_false` 853 / 69 and 855 / 70 (MEASURED, RESULT_LONPLAN section 3).
  * The pass changes speed and therefore positions, so these cells can move.
  * `S_CKPT`, strategic intent 6-30 s, derived goals and detected info are NOT COMPUTABLE on this planner, each with its reason and would-have n as ACTRSP section 4.3 lists them: a structural n/a, never a pass.
  * On HWM-ORACLE these cells measure the world model's use of an oracle goal, not a strategic decision. **No RA-L1 outcome is decided on a STRATEGIC cell.**

## 8 Controls and reachability (the read is INVALID if one fails; each new check has a deliberate-regression arm that must go RED)

### 8.1 Reachability of `--ra-l1`, from the actual caller
* The flag `--ra-l1 {off, l2, l1, neg, shuf, oracle}` is traced argparse → namespace → the wrapper's pass config → the readout loaded and sha-checked → `ra()` inside the cost → `st._optimize` → `backward` → the knots written into `plan_raw`. The build report records the `file:line` of each hop.
* ⚠️ **Its default, `off`, disables it.** Every arm's argv names its value, and BASE / RA-OFF name `--ra-l1 off`.

### 8.2 New controls (written and green before any SCORE window gets an RA pass)
* **C-RA-FIT, the leak check.** RA's fit set rebuilt from the local TRAIN latents has 8,306 pairs in 141 clips; the refit at λ_RA reproduces (mu, sd, W, ym) within 1e-8; 0 TRAIN clips are SCORE episodes. **MEASURED today** (2.5). **Mutation:** a refit that adds one SCORE window's displacement must change W beyond 1e-8 and go RED; a TRAIN latents file with one SCORE eid planted must refuse.
* **C-RA-SHA.** The readout's sha256 equals section 2.1's value. **Mutation:** one changed byte refuses.
* **C-OFF5, RA off = the base, bit for bit.** RA-OFF's `plan_raw`, `a1_raw` and `cost1` equal the base file's on every window. Before that, C-READBACK of the base itself: batch 0 of every base file used is re-planned from its `SOLVER.json` argv at `OMP_NUM_THREADS=2`, bit-equal. **Mutation:** a wrapper that runs the pass while the flag reads `off` must go RED.
* **C-YAW-ID / C-BAND-ID.** Under every RA arm: yaw knots 1-30 and acceleration knots 11-30 equal the base's exactly (max |Δ| = 0.0). On SCORE, X − BASE reads exactly 0.0 [0, 0] for `yaw_mae` and `head_err` in every band, and for per-knot `acc_mae` in 2-6 s. **Mutation:** the pass writing knots 1-12 into the trajectory must go RED.
* **C-GRAD.** On the toy stack and a 16-window TRAIN smoke: the cost's gradient on acceleration knots 1-10 is non-zero, and exactly 0.0 on knots 11-12 and on yaw. **Mutation:** RA reading after 12 actions must make knots 11-12 non-zero (RED).
* **C-PLANT, the inversion.** (i) **Analytic identity:** a linear toy world z_{t+1} = z_t + B a_t with a linear readout and a target formed from a known action sequence: the pass reaches the target read to 1e-6, and with a uniform per-knot response its plan's mean equals the source's mean to 1e-6. **Mutation:** the target formed with the plan's own actions (a pass that cannot move) must leave the plan at HOLD_c and go RED. (ii) **Real model, TRAIN stage:** the pass reaches its target read, median final cost ≤ 1e-4 (scaled²) at both chassis. MEASURED today (section 3): median final cost 0.0 at both chassis (ORACLE and PLANT targets). RA_PLANT's 2 s mean misses the logged mean by 0.144 / 0.154 against a0's 0.266. That miss is the read's structural loss, not a wiring fault, and it has no bar.
* **C-SHUF, the deliberate-regression arm.** RA-SHUF permutes W, which keeps the readout's scale and breaks its alignment with the transition. On SCORE, RA-SHUF − X on 0-2 s `acc_mae` must be separated-worse at both chassis. Before any SCORE pass, the TRAIN stage must read RA-SHUF − X > 0 at both chassis (sign check). MEASURED today with the oracle target (section 3): RA_SHUF − RA_ORACLE = +0.4549 [+0.3864, +0.5222] (s0) and +0.3442 [+0.2656, +0.4209] (s1).
* **RA-NEG, the mirrored target (beside, no bar).** It was drafted as the regression arm and moved beside after the first TRAIN read: with the oracle target, RA-NEG − RA-ORACLE is not separated (section 3), because most of the pass's departure is amplified error, not direction. It is reported with its DIR-ACC.
* **C-MEAN.** MEAN-src at a zero source departure equals HOLD_c exactly. **Mutation:** a shift taken from the logged accelerations must be refused by a guard that MEAN reads only the source file.
* **C-L4-RA, argv → built.** The wrapper parses each argv, asserts the built pass equals the declared one (source, readout sha, mirror, permutation), writes `SOLVER_RA.json` into a new or empty out dir, and stamps every plan file `ra_l1_planned`. **Mutations:** a declared/argv mismatch and a pre-planted `plans_*.npz` must both refuse.
* **C-IN, C-TRAIN-ISO, C-STRAT:** as in LON-PLAN. Inputs are sha256-equal to the record (now including both readouts and the base files); no SCORE window is read by a TRAIN step; the strategic-audit reader computes the STRATEGIC cells.
* **Unchanged:** the card's controls 1-15; `test_hwm_plan.py` (50), `test_plantest2_controls.py` (14), `test_lonplan.py` and AMEND_4's tests re-run green.

## 9 Cost and approvals

**MEASURED rates:** the S = 1 RA pass costs 0.042-0.046 s per window (2 threads, 2-3 processes side by side; `ra5_ceiling_s{0,1}.json`, `sec_per_window_S1`). Model load and C-RA-FIT take under 3 s per run. LON-PLAN's base planner costs 1.30 s per window, for comparison.

**ESTIMATED for the read** (1 process × 2 threads; ±30 %):

| stage | passes | wall |
|---|---|---|
| TRAIN stage | (RA-L1, RA-NEG, RA-SHUF, and RA-L2 under B1/B2) × 2 chassis × 564 windows | about 5 min |
| SCORE lever arms | (RA-L1, RA-L2) × 2 chassis × 2 seeds × 994 | about 6 min |
| SCORE RA-NEG, RA-SHUF, RA-ORACLE | 3 × 2 chassis × 994 | about 5 min |
| C-READBACK of the base files (batch 0, up to 8 files) | | about 0.2 h |
| MEAN rows, RA-OFF, contrasts, DIR-ACC and mechanism rows | | about 0.5 h |

* **Total: about 1 h, of which about 15 min is RA passes** of dev-box CPU wall time. No GPU, no box 2, no training, no spend. If P / P23 TRAIN-stage files lack `l2_prim_raw` (fill M4), add about 10 min per arm and chassis to re-plan them on TRAIN.
* **Build:** one Sonnet agent: `ra_l1.py` with the wrapper and flag, the MEAN assembly, the tests of section 8.2, the toy for C-PLANT (i). ESTIMATED 2-3 h of agent time. `ra5_ceiling.py` is a working prototype of the pass, the target forms, the mirror and the permutation.
* **Files the build needs:**
  * `readout_K10_R35_s{0,1}.pt` (mirror, `raw/2026-11-16-v8f-actrsp-prereg/amend1/out/`);
  * `ckpt_last.pt` of R35_s0 / s1 (`C:/Users/Admin/t2v8f_pulls/ladder_R35_s{k}/`) and `latents_train_R35_s{k}.pt` (`F:/TanitAD-scratch/hwm_plan/`);
  * the SCORE `windows.npz` per chassis (s1 local at `lonplan/score_windows_s1/`; s0 from `hwmplan/hwmplan_phase1_R35.tar` per C-IN);
  * the base files: LON-PLAN `score/A0_s{k}/plans_hwm_oracle_s{0,1}.npz` and `sel_s{k}/lam_0/` (TRAIN); under B1/B2 AMEND_4's P / P23 SCORE and TRAIN files with `l2_prim_raw`;
  * the code of record (unchanged) and `lonplan/contrast_l1full_strat.py`.
* **Disk:** under 0.1 GB on F:. Banked to the phase-1 HF repo (private) with per-file sha256, then the local copies are deleted under the PI's delete rule.
* **Approval.** CPU reads of this kind ran without a PI go (phase 1, G3, LON-PLAN, AMEND_4). `QUESTIONS_AMEND5.md` lists the MM's fills and the PI items.

## 10 Fills the MM sets at freeze
0. **The base solver** (B1 / B2 / B3), written into the Log from AMEND_4's readout before any RA pass on SCORE (QUESTIONS M0).
1. **Which level's actions RA reads** (M1): recommended RA-L2 committed under B1/B2 with RA-L1 as a second registered arm; RA-L1 committed under B3, with RA-L2 on P beside.
2. **Code location** (M2): recommended the study-local post-hoc module.
3. **RA-L2 on A0** (M3): recommended not run.
4. **The P / P23 TRAIN-stage files** (M4): confirm `l2_prim_raw` is banked.
5. **RA-ORACLE on SCORE** (M5): recommended registered, beside, as the ceiling that sorts NULL from READ-LIMITED.

## 11 Unchanged
* The card's bars, arms, outcomes and controls 1-15.
* FREEZE fills 1-5, as amended by AMEND_HWMPLAN_1 to 4.
* The solver of record of every arm other than RA-L1's.
* The 994 SCORE windows. LON-PLAN's and AMEND_4's results and banked plans.
* ACTRSP sections 4.3, 4.4, 8 and 9. The ACTRSP arms stay held, as Amendment 1 holds them. RA stays the frozen readout of Amendment 1; nothing here refits it.

## 12 Register rows for the MM (to land with the freeze)
* **`H-RAL1`**, HYPOTHESIS, OPEN. Replacing level 1's latent distance on the 0-2 s acceleration channel by the squared difference of two RA reads (the plan's and a source plan's, through R35) lowers HWM-ORACLE's 0-2 s `acc_mae` against the base at T0 on R35, at both chassis; the CEILING bar asks whether it beats HOLD. Read: this amendment. Expected (ESTIMATED, TRAIN in-sample): RAL1-NULL or HARM with CEIL-ORACLE failing (READ-LIMITED); TRAIN RA-ORACLE − HOLD +0.541 / +0.586.
* **`D-RA-CEIL-TRAIN`**, MEASURED (TRAIN, in-sample for R35 and for RA, 564 windows / 141 clips, T0), a feasibility record: through RA and R35, a perfect frame-10 target lands 0-2 s `acc_mae` +0.54 / +0.59 m/s² above HOLD. No REG dose (λ 0.003-1000) and no gain gate (q50-q90) brings it below HOLD. RA's direct read of the true displacement is +0.19 / +0.20 above HOLD (r 0.47-0.48 against a0's 0.78), while a flat plan at the true 2 s mean is 0.105 below. Artifact: `amend5/ra5_ceiling_s{0,1}.json`.

## Log
* 2026-11-17, agent draft.
  * Measurement for this draft (CPU, TRAIN windows only; no SCORE window planned or scored; the SCORE `windows.npz` of s1 read for its `eid` list only, for the split guard): `amend5/ra5_ceiling.py` → `ra5_ceiling_s{0,1}.json`, logs `ra5_ceiling_s{0,1}.log`. The readout files were exported from the mirror blob and sha256-checked inside the script.
  * Base solver (fill 0), freeze line and section 10 fills: Master Mind.
* 2026-11-17, MM freeze. Fills M0-M7 as in `MM_RULINGS_AMEND5.md`. The base solver (fill 0) is written into `BASE_SOLVER_AMEND5.md` from AMEND_HWMPLAN_4's result before any RA pass on SCORE, so this file stays byte-equal to its sha256.
