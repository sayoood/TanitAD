# AMEND_HWMPLAN_4 (DRAFT for the Master Mind's freeze): L2-IMG, the level-2 image constraint at λ = 0 on R35, by construction, read on the 994 SCORE windows

**Status.** DRAFT, 2026-11-17, Arch+Inference agent. Nothing here is frozen. No SCORE window has been planned under the constraint, and no SCORE outcome was opened for this draft. The only plans made for it are on TRAIN windows (section 3), with a scratch shadow that is not the code of record. The Master Mind (MM) freezes it and sets the fills of section 10.

**Frozen inputs this amends** (paths under `raw/2026-11-08-v8f-bands-build/hwm_plan/` unless named):
* `PREREG_V8F_HWMPLAN_DRAFT.md` `ac20d690`, `FREEZE_HWMPLAN.md` `ae2bbe40`, `AMEND_HWMPLAN_1.md` `dcbd55a9` (world model R35), `AMEND_HWMPLAN_2.md` `b9a7602a` (G3, the decode check), `IMPL_CHOICES_HWMPLAN.md` `c478d6cf`.
* `AMEND_HWMPLAN_3.md` (LON-PLAN), frozen 2026-11-16. Its section 7 names this lever, the flag, the controls and the λ rule it must carry.
* `lonplan/readout/RESULT_LONPLAN.md`, landed `869fae8b`: SEEDS-DISAGREE, CEILING FAILED at both chassis. Its section 4 names this lever as next.
* Code of record: `hwm_plan.py` `5a072272…`, `run_hwm.py` `935b3422…`, `hwm_eval.py` `79392075…`, `hwm_data.py` `a2d3922e…` (sha256). MEASURED equal today between the mirror tip `751f3ae7` and the local copy that the LON-PLAN runs imported.

**Tier.** Every planner row here is **T0**. The goal is HWM-ORACLE, the encoded true-future latent at frame 30, and the planner has no trained part. A T0 row is a world-model diagnostic and never a driving-performance number. HOLD and PRIOR are T1-kin references. The TRAIN measurements of section 3 are in-sample feasibility records, never results.

## 0 What changes (the whole amendment, in one table)

| item | today (LON-PLAN's A0) | this amendment |
|---|---|---|
| level 2's optimisation variable | macro-actions m (B, S, 10, 32) in the level-2 prior box | **arm P:** primitive actions p (B, S, 30, 2) in the level-1 prior box; the macro is `macro_enc_2(act_embed(p))` |
| level 3's optimisation variable | macro-actions (B, S, 2, E3) in the level-3 prior box | unchanged in P. **Arm P23:** the same 30 primitives, macro `macro_enc_3(macro_enc_2(act_embed(p)))` |
| knots 11-30 | `decode_macro` of level 2's chosen macros (300 AdamW steps from HOLD_c) | **the chosen level-2 p, knots 11-30, as they are**; no decode |
| level 1 | the library's: 12 knots against K1 = 4 level-2 rollout states | unchanged |
| REG | λ = 0 | **λ = 0** (AMEND_HWMPLAN_3 section 7: λ\*_k only on CEILING or NARROWS-INFORMATIVE; LON-PLAN read SEEDS-DISAGREE, less favourable branch SHRINKAGE-ONLY) |
| arms | A0, A1, R, S | **P** (the committed lever), **P23** (its extension, section 3), **RS** (deliberate regression), A0 reused (section 4) |
| deciding read | 0-2 s LONGITUDINAL `acc_mae`, CEILING = beat HOLD | the same cells and the same bar, per lever arm |
| everything else | | unchanged: the two R35 chassis, the 994 SCORE windows, the TRAIN priors, the TRAIN/SCORE split, the estimator, the families, controls 1-15, and LON-PLAN's C-L4 / C-READBACK / C-STRAT |

## 1 Why, from measurements

* **LON-PLAN closed the gap to HOLD only by pulling toward it.**
  * A0 (seed HOLD, λ 0) is far worse than HOLD on 0-2 s `acc_mae`: A0 − HOLD = +0.9651 [+0.8949, +1.0328] (s0) and +0.8685 [+0.7929, +0.9409] (s1).
  * REG at λ\*_k lands on HOLD and does not beat it: A1 − HOLD = +0.0065 [−0.0005, +0.0137] (s0) and +0.0303 [+0.0183, +0.0443] (s1).
  * At s1, REG does not beat a matched pull toward HOLD: A1 − S = −0.0016 [−0.0134, +0.0112].
  * MEASURED, T0, paired episode-cluster bootstrap B 2,000 seed 0, 994 windows / 71 episodes: `lonplan/readout/RESULT_LONPLAN.md` sections 1 and 2.
* **Level 1's 0-2 s cost is a distance to level-2 rollout states.** At level 1 the library's cost is the squared distance of the lifted level-1 rollout to `sub`, the first K1 = 4 states of level 2's rollout of its chosen macros. The last sub-goal counts in full and the earlier ones at 0.5 (`plantest2_lib.py:117-127`; `sub` at `:119`, the chosen rollout at `:142`). Level 1 plans toward whatever level 2 planned.
* **Level 2 plans macros the encoder cannot produce (phase 1).** On the phase-1 plans (seed prior), the planned macros' median nearest-neighbour distance to the TRAIN in-image macro cloud is 7.16 / 7.21, against 1.06 / 1.06 for held-out in-image macros: ratio 6.77 / 6.79, and every planned macro lies beyond the held-out q95. Re-decoded, their residual is median 0.916 / 0.918, and 94 % of the decodes hit the action bounds. MEASURED: `F:/TanitAD-scratch/hwm_plan/decode_check/result_s{0,1}.json` (`step3_nn`, `step2_planned_redecode_300it`), AMEND_HWMPLAN_2.
* **The same holds under A0's own solver, and at level 3 too (new, this draft).** On 128 TRAIN windows in 32 clips, A0's level-2 macros lie 7.25 / 7.08 times farther from the in-image cloud than the logged macros of the same windows (leave-own-clip-out), all of them beyond the logged q95. The decode residual is median 0.910 / 0.899, and 97 % / 94 % of the decoded rows hit a bound. Level 3 is off the image as well: ratio 2.23 / 2.23, with 100 % / 99.6 % beyond the logged q95. MEASURED, TRAIN, in-sample, section 3.
* **The prior boxes and the image barely coincide (new, this draft).** 53 % / 59 % of TRAIN in-image level-2 macros have a coordinate outside the level-2 prior box (TRAIN mean ± 2 TRAIN std per coordinate, `plantest_lib.py:37-43`), and 87 % / 89 % of level-3 macros lie outside the level-3 box. A0's planned level-2 macros are all inside the box (0 %). The box admits off-image points and excludes half of the image. MEASURED, section 3.
* **HYPOTHESIS H-L2IMG.** With level 2 restricted to the image of bounded primitive sequences, its rollout states are states the world model reaches under actions of the kind it was trained on. Level 1 then plans toward reachable sub-goals, and its 0-2 s plan moves toward the logged drive: 0-2 s `acc_mae` falls against A0, and possibly below HOLD.
* **What is expected, stated before any SCORE number (ESTIMATED from the TRAIN probe, in-sample).**
  * CEILING needs P to beat HOLD, so its 0-2 s `acc_mae` must fall by more than A0 − HOLD = 0.9651 / 0.8685 m/s², about 75 %.
  * On the 128 TRAIN windows, P moved 0-2 s `acc_mae` by −0.0225 / +0.0069 against A0, and P23 by −0.0659 / −0.0174, while level 1's sub-goal cost fell by 30 % / 21 % (P).
  * So the expected labels are **L2IMG-NULL and L23IMG-NULL**. The 2-6 s secondary label is expected to read HURTS (+0.15 to +0.23 on TRAIN). CEILING is not the expected outcome.
  * The read is still worth running. It settles H-L2IMG on held-out windows with the programme's estimator at about 5 h of CPU, and it fixes the base solver of the RA read that follows in every branch (section 6). Whether to run it, or to go straight to the RA read, is MM question M0 in `QUESTIONS_AMEND4.md`.

## 2 The constraint, operationally

### 2.1 The set
* **The level-2 image** is I₂ = { `macro_enc_2(act_embed(p))` : p ∈ B₁^30 }.
  * `act_embed` and `macro_encoders[0]` are R35's own frozen modules: the exact path `HJEPAStack.encode` uses to build level-2 actions (`hjepa.py:669-700`, embeddings grouped in strides of 3, window 1). The macro encoder ends in a LayerNorm (`hjepa.py:287-313`).
  * B₁ is the level-1 prior box `[prior_lo_1, prior_hi_1]` = TRAIN mean ± 2 TRAIN std per channel, fitted on R35's own TRAIN latents by `hwm_data.fit_priors` (`plantest_lib.py:37-43`). It is the box level 1 plans in.
  * 30 primitives = h2 · c2 = 10 macro steps × 3 level-1 steps (MEASURED from the built stack: strides (3, 5), K2 2, h2 10, T3 2).
* **The level-3 image (arm P23 only)** is I₃ = { `macro_enc_3(macro_enc_2(act_embed(p)))` : p ∈ B₁^30 }. T3 = 2 level-3 steps cover the same 30 primitives.
* **Nothing is trained, fitted or banked for it.** No macro bank, no decoder, no dose. The set is a parameterisation, not a projection.

### 2.2 How the planner stays in it (implementation (I), by construction)
* **Variable and cost.** Level 2's optimisation variable becomes p (B, S, 30, 2). Its cost is the library's level-2 cost evaluated at `M₂(p)`: the rollout of level 2 under `M₂(p)`, lifted to level 3, against level 3's sub-goals (`plantest2_lib.py:113-127`, unchanged). Under P23, level 3's cost is evaluated at `M₃(M₂(p))` against the goal in the same way.
* **Optimiser.** The library's `_optimize`: AdamW, at most 90 iterations, patience 10, rel_tol 1e-4, weight decay 0, with p clipped to B₁ after every step (`hjepa.py:948-990`, unchanged). Every iterate's macro is therefore in I₂ (and in I₃ under P23).
* **Candidates.** Candidate 0 = HOLD_c[:, :30], whose macro is A0's own level-2 seed (`hold_seeds_g` encodes the same sequence). Candidates 1-15 = HOLD_c + `prior_std_1` · N(0, 1), clipped to B₁, the way level 1 builds its HOLD-seeded candidates.
* **Random numbers.** N is drawn from a separate generator: `manual_seed(2⁴⁰ + seed · 100003 + batch)` at level 2, and `2⁴¹ + …` at level 3 under P23. The planner's own generator still draws the macro noise and discards it, so it is consumed exactly as under A0.
  * ⇒ Under P, level 3 is bit-identical to A0 at the same inference seed.
  * ⇒ Under P and P23, level 1 draws the same numbers as A0.
  * Both are MEASURED in section 3 (control C-L3ID).
* **Step size.** lr 0.1, the level-1 value, because the variable lives in level-1 units. It is fixed here and not tuned. (The upper levels' own lr, 0.3, is in macro units.)
* **No prior box on the macros.** The level-2 box (and the level-3 box under P23) is not applied to the macro. The image of B₁ replaces it, for the reason measured in section 1.
* **Knots 11-30** = the chosen level-2 p, knots 11-30, converted to raw units by `to_raw`. No decode runs, so the 2-6 s output is a primitive sequence whose macro is level 2's chosen macro (control C-PRIM-ID).
* **The pick.** The library's `argmin` over candidates of the tracked best level-2 cost; the chosen p is the same candidate's best iterate.

### 2.3 Alternatives considered and not recommended
* **(II) Penalty** `w · d(m, TRAIN macro bank)²` at level 2 (AMEND_HWMPLAN_3 section 7). It needs a dose w, a TRAIN selection of that dose, and a macro bank.
* **(III) Projection** by `decode_macro` after each level-2 step. That is 300 inner steps for each of up to 90 outer steps. And on in-image targets the decode itself leaves a median residual of 0.163 / 0.117 (MEASURED, decode check step 1), so the projected point is not in I₂ either.
* **The LayerNorm manifold.** Projecting m onto the macro encoder's LayerNorm output set (a 30-dimensional manifold in R³²) is cheap, but it is only a necessary condition. Each level-2 macro is a function of 6 numbers (3 primitives × 2 channels), so per macro step I₂ is at most a 6-dimensional subset of that manifold. The projection would not make a macro reachable.

### 2.4 What stays fixed
* **λ = 0** and `--seed-mode hold`, named in every argv.
* **The two chassis:** R35_s0 `d920d62c…` and R35_s1 `a97e6017…` (sha256 of `ckpt_last.pt`). Each uses its own TRAIN priors from `latents_train_R35_s{k}.pt` (`76a2d4f5…` / `692eb002…`).
* **The 994 SCORE windows in 71 episodes:** `windows.npz` sha256 `371c753b…` (s0) and `8833e292…` (s1), as in LON-PLAN.
* **The TRAIN/SCORE split.** Priors come from TRAIN latents only, and 0 of the 141 TRAIN clips are eval clips (MEASURED in LON-PLAN's selection record). No step reads a SCORE window before the freeze.
* **Dose-free.** No λ, no w, no grid, and nothing selected on SCORE data.
* **The other levers at phase 1's values:** T3 2, `plan-ctx 1`, `vocab off`, `constr-weight 0`, S 16, n_iters 90, inference seeds 0 and 1, and `OMP_NUM_THREADS=2` for every run.

## 3 Measurements made for this draft (CPU, TRAIN windows only, zero training)

**Instrument.** `amend4/l2img_probe.py` (sha256 `f1682600…`), banked next to this draft.
* `off`, `prim` and `rs` ran with the first version, `amend4/l2img_probe_v1.py` (`d3714854…`), into `probe_s{0,1}_off_prim_rs.json` (`e4a7f639…` / `f4f2e5a7…`).
* `p23` was **added after that first read** showed level 3 off the image. Its constants were fixed before it ran, and its code paths for the other arms are unchanged. It wrote `probe_s{0,1}_p23.json` (`21a09bae…` / `e0cb4872…`).
* The probe imports the code of record unchanged. It adds the section 2.2 shadow as instance attributes for one plan call, the way `shadow_solver` does.
* **Windows:** LON-PLAN's TRAIN selection set (`sel_s{k}/selection_windows.npz`, the same set at both chassis). It used batches 0 and 1 (128 windows, 32 TRAIN clips), generator `make_gen(0, batch)`, and 2 threads: the batches of LON-PLAN's own λ = 0 TRAIN run.
* **The in-image reference.** The level-2 and level-3 macros of every ok TRAIN 16-frame window (`st.encode`, as `fit_priors`; 9,165 and 1,833 macros). The nearest neighbour is taken **leaving out the query window's own clip**. The comparator is the macro of the same windows' logged TRAIN actions under the same rule.
* The design of sections 2 and 4 was written into the script header before it first ran. Only P23 was added after a number was read: TRAIN plays the role here that it played for LON-PLAN's λ selection, and SCORE stays untouched.

| quantity (TRAIN, in-sample, 128 windows / 32 clips per chassis; no interval: a feasibility record) | s0 | s1 |
|---|---|---|
| **C-OFF:** `off` against LON-PLAN's banked λ = 0 TRAIN plans (plans, `cost1`, level-1 actions, decode residual), max abs | **0.0** | **0.0** |
| **C-L3ID:** level-3 actions and cost, `prim` against `off`, max abs; planner generator state after the call, all four arms | **0.0**; equal | **0.0**; equal |
| logged in-image macros, median NN (leave-own-clip-out): level 2 / level 3 | 0.996 / 3.18 | 1.017 / 3.23 |
| level-2 NN ratio to the logged macros (median), share beyond the logged q95: `off` | **7.25**, 100 % | **7.08**, 100 % |
| … `prim` | 1.91, 67 % | 2.09, 72 % |
| … `p23` | 1.89, 65 % | 2.05, 72 % |
| level-3 NN ratio, share beyond q95: `off` (= `prim`) | **2.23**, 100 % | **2.23**, 99.6 % |
| … `p23` | 1.31, 45 % | 1.27, 37 % |
| TRAIN in-image macros outside the level-2 box / the level-3 box | 53.4 % / 87.5 % | 59.2 % / 88.7 % |
| planned level-2 macros outside the level-2 box: `off` / `prim` | 0 % / 90.3 % | 0 % / 93.0 % |
| `off` decode residual, median; rows with a bound hit (knots 11-30) | 0.910; 96.9 % | 0.899; 93.8 % |
| `prim` image residual, max; knots 11-30 − chosen p, max abs | 2.1e-7; 0.0 | 2.4e-7; 0.0 |
| level-1 cost, median: `off` / `prim` / `p23` / `rs` | 25.37 / 17.83 / 18.74 / 298.2 | 21.30 / 16.82 / 17.26 / 266.1 |
| level-2 cost, median: `off` / `prim` / `p23` | 2.90 / 6.84 / 6.37 | 2.65 / 6.40 / 5.97 |
| 0-2 s `acc_mae`, m/s²: HOLD / `off` / `prim` / `p23` / `rs` | 0.2850 / 1.2249 / 1.2024 / 1.1590 / 1.5494 | 0.2850 / 1.1844 / 1.1913 / 1.1670 / 1.4918 |
| 0-2 s `acc_mae` with level 2's own primitives as knots 1-10 (P2-0): `prim` / `p23` | 1.0729 / 1.0953 | 1.0422 / 1.0799 |
| 0-2 s `yaw_mae`, rad/s: HOLD / `off` / `prim` / `p23` | 0.0219 / 0.1068 / 0.1105 / 0.1119 | 0.0219 / 0.1138 / 0.1160 / 0.1212 |
| 2-6 s `acc_mae`, m/s²: HOLD / `off` / `prim` / `p23` | 0.5716 / 0.8676 / 1.0163 / 1.0282 | 0.5716 / 0.7807 / 0.9716 / 1.0109 |
| wall, s per window (two processes side by side, 2 threads each): `off` / `prim` / `p23` / `rs` | 1.01 / 1.07 / 1.05 / 0.83 | 1.05 / 1.08 / 1.03 / 0.79 |

**What the table says (TRAIN, in-sample):**
1. **LON-PLAN's control carries the defect.** Under A0's solver, level 2 plans off the image (ratio 7.25 / 7.08), and so does level 3 (2.23 / 2.23).
2. **The constraint works as built.**
   * The image residual is at most 2.4e-7, and knots 11-30 equal the chosen primitives exactly.
   * Level 3 is bit-identical under P, and level 1's random numbers are unchanged in every arm.
   * Level 1 reaches its sub-goals better: its cost falls by 30 % / 21 %.
   * Level 2 meets level 3's sub-goals worse (×2.4), because those sub-goals are themselves off the image. That is why P23 exists.
3. **In the image is not typical.** P's macros sit 1.9-2.1 times farther from the cloud than logged macros, and two thirds lie beyond the logged q95: bounded, but unusual, primitive sequences.
4. **Acceleration barely moves.**
   * 0-2 s: P − A0 is −0.0225 / +0.0069, and P23 − A0 is −0.0659 / −0.0174, against a gap to HOLD of about 0.9.
   * Level 2's own primitives as knots 1-10 read better than level 1's plan at 0-2 s: by 0.13 / 0.15 under P (P2-0) and by 0.06 / 0.09 under P23 (P23-2-0).
   * 2-6 s: the in-image primitives read worse than A0's bound-hitting decode (+0.149 / +0.191 for P).
5. **The regression arm acts as declared.** RS − A0 = +0.3245 / +0.3074 and RS − P = +0.3470 / +0.3005, with level 1's cost ×12. Level 1 does follow its sub-goals, so a null P is not a level 1 that ignores level 2.

## 4 Arms (SCORE: the 994 windows in 71 episodes, R35_s0 and R35_s1)

All arms: HWM-ORACLE, T3 2, `plan-ctx 1`, `vocab off`, `constr-weight 0`, `--seed-mode hold --lam 0`, and each chassis's own checkpoint and TRAIN priors.

| arm | argv difference | inference seeds | role |
|---|---|---|---|
| **A0** | `--l2-image off` | 0, 1 | control. LON-PLAN's banked `score/A0_s{k}` plans are reused **only if** C-OFF re-plans batch 0 of both seed files bit-equal under the new build; else A0 is re-run (MM fill 2) |
| **P** | `--l2-image prim` | 0, 1 | the committed lever (level 2 in the image) |
| **P23** | `--l2-image prim23` | 0, 1 | its extension (levels 2 and 3 in the image), read with its own label |
| **RS** | `--l2-image prim --l1-sub donor` | 0 | deliberate regression; **must read worse than A0** (C-RS) |
| P2-0, P23-2-0 | none: zero compute. Level 2's own primitives as knots 1-10, with the arm's knots 11-30 | 0 | beside, no bar: what level 1 adds over level 2's primitives at 0-2 s |
| A1 | LON-PLAN's banked REG at λ\*_k (300 / 100) | 0, 1 | beside, no bar: the image constraint against REG |
| HOLD, PRIOR | references | — | T1-kin |

* **How RS is built.** RS is P, except that level 1 plans against the level-2 rollout of a donor window.
  * The donor rule: per batch, one cyclic shift d\* = the smallest d ≥ 1 with clip(i + d mod B) ≠ clip(i) for every window i of the batch. It is a bijection within the batch. A batch with no such d is refused (d\* = 4 on every TRAIN batch of section 3).
  * Why this regression: it inverts the lever on the lever's own path. P makes level 1's sub-goals reachable from its own state, and RS hands level 1 sub-goals rolled out from another clip's state, which are not reachable. Level 2's actions and knots 11-30 are P's.
  * The hook: the one level-2 rollout call with 3-dimensional actions (`plantest2_lib.py:142`, after the level-2 pick at `:138`) returns the donor rows. Every other rollout is untouched.
* **Why P23 is an arm and not a later amendment.** It is the cheapest extension that could still make the image route work (Rule Zero rule 4): it removes the second off-image level that section 3 measured. It needs no dose, it adds about 1.2 h of CPU, and it shares P's code. It has its own label, so P's committed question is read on its own.
* **What the P / P23 plan files bank beyond today's keys:** `l2_prim_raw` (n, 30, 2), `cost2`, `cost3`, `iters2`, `iters3`, `image_resid` (n), `l2_image_planned`, and the generator-state hash per batch. The level-2 and level-3 macros of batch 0 are kept for the mechanism rows.
* **No FLAT arm and no λ > 0 arm.** FLAT has no level 2. A P + REG arm would need its own TRAIN selection; it is named under NARROWS in section 6.

## 5 Read

* **Instrument.** As LON-PLAN.
  * Each arm runs through the run wrapper in its own new out dir.
  * The paired contrasts use `lonplan/contrast_l1full_strat.py` (the strategic-audit reader; control C-STRAT as frozen in AMEND_HWMPLAN_3's Log), with `--a`, `--b`, `--rep-a`, `--rep-b` and `--windows` = the SCORE windows.
  * P2-0 and P23-2-0 are assembled by a zero-compute script from the arms' files.
* **TRAIN stage, before any SCORE plan (beside, in-sample, never a result).** P, P23 and RS run on LON-PLAN's full TRAIN selection set (564 windows in 141 clips) at both chassis. A0 is LON-PLAN's banked λ = 0 TRAIN run, bit-equal by C-OFF. It reports P − A0, P23 − A0 and RS − A0 per family, with the episode-cluster bootstrap over the 141 clips. It feeds C-RS's sign check, and it changes nothing in this design.
* **Contrasts**, per band (0-2 s, 2-6 s, full 6 s) and per family, never pooled:
  * **P − A0, P − HOLD, P23 − A0, P23 − HOLD** (deciding);
  * RS − A0 (C-RS) and RS − P (beside);
  * P23 − P, P − A1, P23 − A1 (beside);
  * P2-0 − P and P23-2-0 − P23 (beside, 0-2 s);
  * the inference-seed replicates of P, P23 and A0.
* **Mechanism rows: beside, with the direction stated now (HYPOTHESIS). No outcome is decided on them.**
  * **REACH:** level 1's achieved cost `cost1` of P (and of P23) against A0, per window. Reported as the median ratio and the paired episode-cluster bootstrap of the mean log-ratio. H-L2IMG predicts < 1 (TRAIN: 0.70 / 0.79).
  * **IMG:** the level-2 and level-3 nearest-neighbour ratios (section 3's rule) on the arms' batch 0 and on A0's batch-0 re-plan (C-OFF), with the image residual of P and P23 on all windows.
  * **COST2:** level 2's achieved cost, P and P23 against A0. It is expected to rise (TRAIN: ×2.4).
* **DIR-ACC** for A0, P, P23, RS, P2-0 and P23-2-0, as LON-PLAN defines it: the Pearson r of the plan's 0-2 s departure from HOLD against the logged one, with the episode-cluster bootstrap B 2,000 seed 0. Beside, no bar.
* **Estimator.** Paired episode-cluster bootstrap (`taniteval/ci.py` through `hwm_eval.contrast`), B 2,000, seed 0, over the 994 SCORE windows in 71 episodes. It answers *another draw of EPISODES*, and nothing else.
* **Inference floor.** F_I,k(q) = max(|X seed 1 − X seed 0|, |A0 seed 1 − A0 seed 0|) on the row's point means, per chassis and per lever arm X ∈ {P, P23}. It answers *another INFERENCE run*. Under P and P23, both generators move with the inference seed: the planner's, and the primitive-noise generator seeded from it. For scale, LON-PLAN's 0-2 s `acc_mae` floors were 0.0073 (s0) and 0.0041 (s1) (MEASURED, RESULT_LONPLAN section 1).
* **Training runs.** Nothing is trained. *Another TRAINING run* is answered only by the two chassis agreeing, and a claim about the R35 recipe needs both. ACTSENS's `NULL_s2` / `NULL_s3` (R35-recipe replicates) are named in section 6 for a pass near the floor.

## 6 Committed outcomes (per lever arm X ∈ {P, P23}; both chassis required; a label at one chassis only is SEEDS-DISAGREE, and the less favourable label's next step applies)

**Definitions, per chassis k and lever arm X, on the 0-2 s band, inference seed 0:**
* Δ_H = X − HOLD (`acc_mae`), the CEILING cell.
* Δ_0 = X − A0 (`acc_mae`), the lever's effect.
* G_yaw = X − A0 (`yaw_mae`), the guard.
* **"Separated-better beyond floor":** the CI lies wholly below 0, |Δ| > F_I,k(q), and the seed-1 point Δ is also below 0.
* **"WORSE":** the CI lies wholly above 0, and Δ > F_I,k(q).
* D(X): mean |plan acc − HOLD acc| over knots 1-10 in TRAIN prior std (LON-PLAN's D). X is HOLD-collapsed if D < 0.10.
* P's label is written L2IMG-…, and P23's L23IMG-….

| outcome | condition (both chassis) | next lever (Rule Zero; started in the same work package unless blocked) |
|---|---|---|
| **CEILING** (the primary criterion passes) | Δ_H separated-better beyond floor; G_yaw not WORSE; D(X) ≥ 0.10 | (1) **The level-1 cost read through RA** (frozen `readout_K10_R35_s{k}.pt`, zero training), drafted as AMEND_HWMPLAN_5 **on X's upper levels**, to push 0-2 s further. (2) **MM:** X becomes the candidate solver of record for later HWM-PLAN arms, after the multiplicity guard below. (3) **The T1-kin read,** HWM-PLAN with a predicted goal under X: **blocked** on a goal head trained under its own card and gate (box 2, PI, `QUESTIONS_LONPLAN.md` Q2a). (4) If \|Δ_H\| < 2 F_I,k at either chassis: the same X / A0 read on `NULL_s2` / `NULL_s3` before any solver-of-record change |
| **NARROWS** | not CEILING; Δ_0 separated-better beyond floor; G_yaw not WORSE | Level-2 (or level-3) reachability carries part of the 0-2 s defect, not enough to beat HOLD. Next: **the RA level-1 cost read (AMEND_HWMPLAN_5) on X's upper levels.** Named for the MM and not run here: X + REG with λ re-selected on TRAIN by AMEND_HWMPLAN_3 section 3's procedure on X (LON-PLAN's λ\*_k was selected on A0's planner and does not transfer) |
| **NULL** | Δ_0 not separated-better beyond floor at either chassis, and not WORSE | The off-image upper-level states are not what keeps level 1 above HOLD at 0-2 s. Next: **the RA level-1 cost read (AMEND_HWMPLAN_5).** If REACH reads < 1 (level 1 got closer to reachable sub-goals and acceleration did not move), level 1's latent cost points in no usable acceleration direction even toward reachable states. That is exactly RA's question |
| **HARM** | Δ_0 WORSE at either chassis, or G_yaw WORSE | First, C-OFF, C-PRIM-ID, C-L3ID and C-REACH are re-run on the SCORE argv: a wiring defect is the first suspect. Then the RA read on A0's upper levels |
| **INVALID** | any control in section 8 fails, C-RS included | Nothing is read. The failing control is fixed and the arm re-run |

* **The base solver of the RA read** is set by these labels, mechanically:
  * if either arm reads CEILING, that arm (both: P23 if its Δ_H point is lower at both chassis, else P);
  * else if either arm reads NARROWS, that arm (both: the same tie rule on Δ_0);
  * else A0's upper levels.
* **Multiplicity guard.** Two lever arms are read against the same bar. A CEILING at one arm only changes the solver of record after the same read on `NULL_s2` / `NULL_s3` agrees.
* **The training-gated lever behind every non-CEILING row:** world-model action-sensitivity training on box 2 (the L1-FULL verdict's lever). `QUESTIONS_LONPLAN.md` Q2b asks the PI to decide it **with this read's result**. It is not pre-approved, and it is named here with that blocker.
* **Secondary label, 2-6 s band (beside; no primary outcome is decided on it).** Under P and P23, knots 11-30 come from level 2's primitives, not from the decode that hits a bound on 94-97 % of rows.
  * **BAND-ADDS** if X − A0 on 2-6 s `acc_mae` is separated-better beyond floor at both chassis; **BAND-NEUTRAL** or **BAND-HURTS** otherwise, by the same rules.
  * On ADDS, the BAND-SPLIT re-plan that the L1-FULL verdict made conditional on this constraint gets its own amendment.
  * Expected (ESTIMATED, TRAIN): HURTS, +0.15 to +0.23.
* **Reported beside every outcome, no bar:** P23 − P, X − A1, the 2-0 rows, RS − P, DIR-ACC, REACH, IMG, COST2, the TRAIN stage, and the full family table.
* **A failed bar is reported as FAILED, then its next lever runs.** No arm, window, constant or criterion changes after any SCORE number is seen.
* **Done** means one of two things: (a) CEILING, with next steps (1) and (2) started; or (b) AMEND_HWMPLAN_5 (the RA read) drafted on the base solver above and its build started, with each training-gated lever named with its blocker.

## 7 Families (all four, ACTRSP section 4.3 as LON-PLAN applied it; T0)

* **LONGITUDINAL:** `acc_mae` (deciding, 0-2 s) and `speed_err`.
  * Distance-keeping: `hw_err`, `tg_err` and `ttc_err` on windows in state LEAD, with n per cell. LON-PLAN's 0-2 s cells had 214 windows / 32 episodes, and `tg` 182 / 31 (MEASURED, `c_A1_A0_s{0,1}.json`).
  * The 0-2 s band uses the K = 10 lead block; 2-6 s and the full 6 s use K = 30. Signed `hw_d`, `tg_d` and `ttc_d` are printed on arm rows.
* **LATERAL:** `yaw_mae` (the guard, 0-2 s), `head_err`, `curv_err` and `xtrack`, with `ade` beside. `curv_err` uses steps with logged speed ≥ 1 m/s, with n printed (950 windows / 71 episodes in LON-PLAN).
* **TACTICAL:** plan-derived `tac_man_dis`, `tac_lat_dis` and `tac_lon_dis` per band, with kappa against the logged manoeuvre. Goal and anchor selection: n = 0, NOT COMPUTABLE (the planner has no decision output).
* **STRATEGIC, reported in full** (the item 25 pause ended by PI ruling 2026-11-16):
  * `s_adh` with its signed-degree twin, and `s_false`, full 6 s band only, with the logged-path rate beside.
  * LON-PLAN's n: `s_adh` 12 windows / 4 episodes (s0) and 8 / 5 (s1), which is not decision-grade; `s_false` 853 / 69 and 855 / 70 (MEASURED, RESULT_LONPLAN section 3).
  * Unlike in LON-PLAN, P and P23 change knots 11-30, so the 2-6 s and full-6 s cells can move, `s_adh` and `s_false` included.
  * `S_CKPT`, strategic intent 6-30 s, derived goals and detected info are NOT COMPUTABLE on this planner, each with its reason and would-have n as ACTRSP section 4.3 lists them. That is a structural n/a, never a pass.
  * On HWM-ORACLE these cells measure the world model's use of an oracle goal, not a strategic decision. **No L2-IMG outcome is decided on a STRATEGIC cell.**

## 8 Controls and reachability (the read is INVALID if one fails; each new check has a deliberate-regression arm that must go RED)

### 8.1 Reachability of `--l2-image`, from the actual caller
* The flag must be traced argparse → namespace → the built solver → the shadow installed → the level-2 (and, under `prim23`, level-3) `_optimize` call → the cost → `backward`. The build report records the `file:line` of each hop.
* ⚠️ **Its default, `off`, disables it.** Every P, P23 and RS argv must name its value, and every A0 argv must name `--l2-image off`.

### 8.2 New controls (written and green before any SCORE window is planned under `prim` or `prim23`)
* **C-OFF, default identity.** `--l2-image off` is the code of record, bit for bit:
  * on the 16-window R35_s0 smoke: plans, `cost1` and `decode_resid` bit-equal to a run with nothing installed;
  * batch 0 of LON-PLAN's `score/A0_s{k}/plans_hwm_oracle_s{0,1}.npz` re-planned bit-equal (the reuse condition);
  * **MEASURED today on TRAIN:** the probe's `off` reproduces LON-PLAN's λ = 0 TRAIN plans on 128 windows at both chassis with max abs difference 0.0 (plans, `cost1`, level-1 actions, decode residual);
  * **mutation:** a build whose default is `prim` must go RED.
* **C-PRIM-ID, the identity.** Under `prim` and `prim23`:
  * knots 11-30 equal the chosen level-2 p, knots 11-30, exactly (max |Δ| = 0.0);
  * the image residual is ≤ 1e-6. It is ‖`M₂`(chosen p) − chosen macro‖ / ‖chosen macro‖, recomputed in a fresh call (float rounding of a different batch shape; MEASURED max 2.1e-7 / 2.4e-7, against A0's decode residual of median 0.90-0.91). Under `prim23` the same holds at level 3 for `M₃(M₂(·))`;
  * **mutations:** (a) the shadow not installed while the flag reads `prim`, which returns the residual to about 0.9, and (b) knots 11-30 taken from the decode, which breaks the identity. Both must go RED.
* **C-L3ID, the identities of the untouched levels.** At the same inference seed and batch:
  * under `prim`, level 3's chosen actions and cost are bit-equal to `off`;
  * under both `prim` and `prim23`, the planner generator's state after the call is bit-equal to `off`.
  * **MEASURED today on TRAIN:** 0.0 and equal, at both chassis, in all arms.
  * **Mutation:** the primitive noise drawn from the planner's generator must go RED (the generator state differs).
* **C-REACH, gradient.** On the toy stack and the 16-window smoke:
  * the level-2 cost's gradient on p is non-zero;
  * the zero control (p frozen, `requires_grad` off) reads exactly 0.0;
  * the level-2 iterates change between iteration 0 and the pick.
* **C-L4-IMG, argv → built.**
  * The wrapper parses each argv and asserts that the built `l2_image` equals the declared value. `SOLVER.json` records it.
  * The out dir must be new or empty (the skip-existing trap, AMEND_HWMPLAN_3 section 8.1).
  * Every plan file is stamped `l2_image_planned`.
  * **Mutations:** a declared/argv mismatch and a pre-planted `plans_hwm_oracle_s0.npz` must both refuse.
* **C-RS, the regression arm.**
  * On SCORE, RS − A0 on 0-2 s `acc_mae` must be separated-worse at both chassis.
  * Before any SCORE plan, the TRAIN stage must read RS − A0 > 0 at both chassis.
  * **MEASURED today on TRAIN (128 windows):** RS − A0 = +0.3245 / +0.3074 and RS − P = +0.3470 / +0.3005, with level 1's median cost 298 / 266 against 25 / 21.
* **C-READBACK.** As in LON-PLAN: batch 0 of every SCORE plan file is re-planned from its `SOLVER.json` argv at `OMP_NUM_THREADS=2`, with plans and `cost1` bit-equal. **Deliberate regression:** P's file read back at `--l2-image off` must fail.
* **C-IN, C-TRAIN-ISO, C-STRAT:** as in LON-PLAN. Inputs are sha256-equal to the phase-1 record; no SCORE window is read by a TRAIN step; the strategic-audit reader computes the STRATEGIC cells, and the code-of-record import reads PAUSED.
* **Unchanged:** the card's controls 1-15. `test_hwm_plan.py` (50), `test_plantest2_controls.py` (14) and `test_lonplan.py` are re-run green.

## 9 Cost and approvals

**MEASURED rates** (dev box, 2 threads per process):
* LON-PLAN SCORE A0 (seed HOLD, decode): 2,594 s for 2 inference seeds × 994 windows, so 1.30 s per window (`lonplan/score_A0_s0.log`, unit sums, 1 process).
* This draft's TRAIN probe, with two processes side by side: 1.01-1.08 s per window for `off`, `prim` and `p23`, and 0.79-0.83 for `rs` (section 3).

**ESTIMATED for the read** (1 process × 2 threads, sequential; rates ±30 %, because SCORE windows and a box shared with other jobs can move them):

| stage | runs | wall |
|---|---|---|
| TRAIN stage | (P, P23, RS) × 2 chassis × 564 windows | about 1.0 h |
| P | 2 chassis × 2 seeds × 994 windows | about 1.2 h |
| P23 | 2 chassis × 2 seeds × 994 windows | about 1.2 h |
| RS | 2 chassis × 1 seed × 994 windows | about 0.5 h |
| C-OFF on A0 (reuse) | batch 0 × 2 seeds × 2 chassis | about 0.1 h |
| C-READBACK | 10 files × batch 0 | about 0.2 h |
| contrasts, the 2-0 rows, DIR-ACC, mechanism rows | | about 0.5 h |

* **Total: about 4.7 h of wall time** at 1 process × 2 threads (about 3.3-6.1 h at ±30 %), or about half that with the two chassis side by side if commit headroom allows (≥ 5.5 GB free, as LON-PLAN's readout gated). No GPU, no box 2, no training, no spend.
* **Build:** one Sonnet agent. It writes the shadow module and the flag (section 10, fill 1), the wrapper extension, the 2-0 assembly, the mechanism-row script and the tests of section 8.2. ESTIMATED 3-4 h of agent time. `l2img_probe.py` is a working prototype of the shadow, the P23 extension and the regression hook.
* **Disk:** under 0.1 GB on F: (35.5 GB free today). It is banked to the phase-1 HF repo (private) with per-file sha256, then the local copies are deleted under the PI's delete rule.
* **Approval.** CPU reads of this kind ran without a PI go (phase 1, G3, LON-PLAN). `QUESTIONS_AMEND4.md` lists the MM's fills and the PI items.

## 10 Fills the MM sets at freeze
0. **Run this read, or go straight to the RA read** (QUESTIONS M0). The recommendation is to run it, and to draft AMEND_HWMPLAN_5 in parallel now.
1. **Where the new code lives.**
   * (a) In the code of record (`hwm_plan.py` `SolverCfg.l2_image`, `run_hwm.py --l2-image`), as AMEND_HWMPLAN_3 section 7 sketched.
   * (b) **Recommended:** a study-local module (`lonplan/amend4/l2img.py`) that installs the section 2.2 shadow around `HWMPlanner.plan_batch`, called by the run wrapper, with the code of record untouched. (b) keeps the code-of-record sha256 that LON-PLAN's `SOLVER.json` files and C-IN record. A0's banked plans then stay readable back with no new land, and `off` is the code of record by construction.
2. **A0:** reuse LON-PLAN's banked SCORE plans after C-OFF (recommended), or re-run them (+0.7 h per chassis).
3. **The RS rule:** the in-batch cyclic donor shift (recommended), or the donor list in `windows.npz` (`donor_indices`), which can point outside the batch and would need a cross-batch rollout.
4. **P23 as a registered arm** (recommended, section 4), or beside only.

## 11 Unchanged
* The card's bars, arms, outcomes and controls 1-15.
* FREEZE fills 1-5, as amended by AMEND_HWMPLAN_1, 2 and 3.
* The solver of record of every arm other than L2-IMG's.
* The 994 SCORE windows. LON-PLAN's result and its banked plans.
* ACTRSP sections 4.3, 4.4, 8 and 9. The ACTRSP arms stay held, as Amendment 1 holds them.

## 12 Register rows for the MM (to land with the freeze)
* **`H-L2IMG`**, HYPOTHESIS, OPEN. Planning level 2 (P), or levels 2 and 3 (P23), inside the image of the macro encoders over bounded primitive sequences lowers HWM-ORACLE's 0-2 s `acc_mae` against A0 at T0 on R35, at both chassis. The CEILING bar asks whether it beats HOLD. Read: this amendment. Expected (ESTIMATED, TRAIN in-sample): NULL.
* **`D-HWM-UPPER-OFFIMAGE`**, MEASURED (TRAIN, in-sample, 128 windows, T0), a defect. Under LON-PLAN's A0 solver, level-2 macros lie 7.25 / 7.08 times and level-3 macros 2.23 / 2.23 times farther from the in-image cloud than logged macros, and 97 % / 94 % of the decoded rows hit a bound. Artifact: `amend4/probe_s{0,1}_off_prim_rs.json`.
* **`D-HWM-PRIORBOX-IMAGE`**, MEASURED, a defect. The level-2 and level-3 prior boxes (TRAIN mean ± 2 std per coordinate) exclude 53-59 % and 87-89 % of TRAIN in-image macros, while every level-2 macro A0 plans lies inside the box. Artifact: the same files, `census_ref`.

## Log
* 2026-11-17, agent draft.
  * Measurements for this draft (CPU, TRAIN windows only; no SCORE window planned or opened): `amend4/l2img_probe_v1.py` → `probe_s{0,1}_off_prim_rs.json`; `amend4/l2img_probe.py` (P23 added) → `probe_s{0,1}_p23.json`; the logs `probe_*.log`. Also the code-of-record sha256 at the mirror tip and locally.
  * Freeze line and section 10 fills: Master Mind.
