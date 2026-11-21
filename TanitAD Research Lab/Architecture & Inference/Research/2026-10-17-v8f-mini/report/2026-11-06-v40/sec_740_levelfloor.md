### 7.40 Addendum A3, the level floor LF (2026-11-05): the heads' level, read outside O as a floor on the plan's progress, moves the crawl and does not close it, and addendum A4 measures the floor's ceiling

Every read here is T1-kin on `reindexed` (deciding) and on `logged` (LEAK-BORNE only), with the paired episode-cluster bootstrap (B = 2,000, seed 0, eval-141). The arms have zero training and ran on the dev-box CPU. O is C0's own checkpoint in every LF − C0 contrast, so O's training variance is absent from it; the floor heads come from one LLR training run per seed (`H-ESTIM-SEED-1`). All four metric families are reported for every arm. STRATEGIC is active for v8F (PI ruling 2026-10-17); the reader's STRATEGIC header note is the ITEM 25 boilerplate, and D3 is read over four families, as for CL, CLR and LLR.

**7.40.1 The operator F and its card (`R/raw/2026-11-05-levelfloor-a3-prereg/PREREG_LEVELFLOOR_A3.md`, sha256 `{{RES_A3P}}…`, landed `{{LAND_A3P}}` before any LF rollout) [M].**

* **Why a floor.** The head arms of §7.39 trained the heads, and three read-outs measured O ignoring them. The composite RILLRC fails D2 on the M1 crawl share alone (+0.0474 / +0.0504 against P10). The card therefore reads the trained heads' estimates directly into the composite, as a lower bound on the plan's along-track progress, so that O does not have to learn to read them. It is the decoupled longitudinal target-speed form, with the path still O's.
* **The operator, applied at every replan.** Inputs: the plan p (30 knots at DT = 0.2 s in the ego frame), the measured ego speed v0, and, from LLR_s{s}'s heads on the same tick batch, the level v_hat, the lead presence π and the gap g = 40 · gap_hat (m).
  1. v_f = κ · v_hat where v_hat ≥ 3 m/s (the crawl band's lower bound); otherwise there is no floor.
  2. If π ≥ 0.5, v_f is lowered to min(v_f, max(0, (g − 2 m) / 1.5 s)): the car-following equilibrium speed with a 2 m standstill gap and a 1.5 s time gap.
  3. s_f(t_k) = ∫₀^{t_k} min(v_f, max(v0, 0) + a_f t) dt with a_f = 1.0 m/s².
  4. s'_k = max(s_k, s_f(t_k)), with s_k the plan's cumulative arc length.
  5. The plan is resampled along its own polyline at s'_k.
* **Three properties by construction.**
  * F is the identity, bit for bit, on every replan where s_k ≥ s_f(t_k) at every knot.
  * s' is non-decreasing, because both of its terms are.
  * The path geometry stays O's, so F changes only the progress along it.
* **What F reads at inference.** No nav, route, limit, label or situation-classifier output: v_hat, π and g come from LLR's vision heads, and v0 is the measured ego speed. The arm is therefore vision plus measured ego history.
* **Doses and arms.** Both doses are fixed in the card: κ 0.50 is the crawl definition's own ratio, and κ 0.75 is halfway to the estimate itself. The planner is C0_s{s}, the base's own checkpoint, so LF − C0 isolates the floor. The bare arm is O's plan → S (sg72) → F; the composite is S → F → kbox.
* **A3_RULING_1** (`{{RES_AR31}}…`, written before any LF rollout) fixes three readings:
  * the bare arm runs the banked base's own flags (`--smooth sg72`, F after S), so its contrast with RIC0 differs in F only;
  * the M1 tick is the reader's definition;
  * "M1 moved" means RILF − RIC0 on M1 is separated-better at one or more seeds.
* **A3_RULING_2** (`{{RES_AR32}}…`, written after the 16 rollouts and before any rule was read) replaces one reader control.
  * The control "lift zero where unbound and never negative" refused on every rollout. On RILF050_s0, 52 of the 543 bound replans had a negative `lf_lift`, with a minimum of −0.45 m/s.
  * On those replans every output knot lay on F's input path, at a largest distance of 1e-6 m and at an arc position at or beyond s_k.
  * `lf_lift` is a chord-length difference over knots 1 to 15. Points moved forward along a bending path are joined by shorter chords, so the expectation was at fault, not the operator.
  * The control became `lift_zero_where_unbound`, which held on every rollout. The negative chord lift where F binds is reported as a result (§7.40.5).

**7.40.2 The build and its known answers (`R/raw/2026-11-05-levelfloor-lf-result/lf_build/` and `lf_read/`) [M].**

* **The tool.** F sits in the RI tool behind `--level-floor {off,on}`, `--floor-kappa` and `--floor-ckpt` (patched tool sha256 `6166178a…`), after S in every branch. The floor heads are a second model instance; on its first replan, its light path matched its own `forward_infer` with a maximum difference of 0.0.
* **Known answers.** With the floor off, and with κ 0, the tool reproduces three banked bases with a maximum difference of 0.0 m over the first 64 windows (KA-LF1 and KA-LF2): C0_s0 on `logged`, RIC0_s0 on `reindexed` with sg72, and RIC0C_s0 with sg72 and kbox.
* **Mutations.** 13 of 13 deliberate mutations of the operator and of the tool placement turn a test RED.
* **The reader.** `lf_read.py` reuses the LLR reader's P, G, F, D and LEAK rules through a sha-pinned path. After A3_RULING_2 it passed 22 tests, 25 of 25 mutations and its two known answers (29 of 29 and 45 of 45 checks). All 23 controls of the read passed.

**7.40.3 Dose κ 0.50 reads P-FAILS-M1-MOVED (`R/raw/2026-11-05-levelfloor-lf-result/lf_read/RESULTS.md`, sha256 `{{RES_LF}}…`, landed `{{LAND_LF}}`) [M].**

* **Reach.** F binds on 0.250 / 0.324 of the M1 crawl ticks (244 / 247 ticks), above the 1 % reach bar, so the rules are read.
* **P fails.** The share closed is 0.062 / 0.056 (bar 0.5). M1 RILF − RIC0 is −0.0033 [−0.0072, +0.0002] / −0.0033 [−0.0065, −0.0005]: separated-better at s1 only, so M1 moved.
* **G fails.** ADE6 is +0.1089 m at s0 (bar +0.0604). Cross-track MAE is separated-worse at both seeds, and DK-VIOL at s1.
* **F, LEAK-BORNE and D.**
  * F fails: 0.0033 against the C0REP floor of 0.0362 / 0.0038.
  * The arm is not LEAK-BORNE (logged share 0.041 / 0.057).
  * The composite RILFC fails D2 on the M1 crawl share alone (+0.0549 / +0.0601 against P10), with D1 and D3 true.

The rules:

@@T LF after=## Dose 050 (kappa 0.50)@@

* **Families, RILF − RIC0** (separated-better / separated-worse rows):
  * LONGITUDINAL: 0 / 0 (s0) and 0 / 1 (s1) of 6.
  * LATERAL: 4 / 1 at both seeds, of 6. Heading, yaw-rate, curvature and L2 curvature are better; cross-track is worse.
  * TACTICAL: 1 / 0 at both seeds (TAC-LAT).
  * STRATEGIC: 3 / 0 at both seeds (S-POS@80 −0.4956 / −0.6414).

All 19 rows:

@@T LF after=### 19 rows, dose 050: RILF - RIC0@@

* **The composite RILFC − P10** (read A), all 19 rows:

@@T LF after=### 19 rows, dose 050: RILFC - P10@@

**7.40.4 Dose κ 0.75 reads P-FAILS-M1-MOVED [M].**

* **Reach.** F binds on 0.455 / 0.462 of the M1 crawl ticks (200 / 199 ticks).
* **P fails.** The share closed is 0.395 / 0.392. M1 RILF − RIC0 is −0.0211 [−0.0334, −0.0112] / −0.0234 [−0.0363, −0.0131], separated-better at both seeds.
* **G fails broadly.** ADE6 is +0.3683 / +0.2172 m. Separated-worse at both seeds: along-track MAE, DK-VIOL, DK-HW and cross-track MAE; speed MAE at s0 as well.
* **F.** 0.0211 against 0.0362 at s0, inside the floor; 0.0234 against 0.0038 at s1, outside it.
* **LEAK-BORNE.** Not LEAK-BORNE (logged share 0.381 / 0.400).
* **D.** The composite RILFC fails D2 on M1 (+0.0380 / +0.0401) and, at s0, on DK-VIOL (+0.0387). D1 and D3 hold.

The rules:

@@T LF after=## Dose 075 (kappa 0.75)@@

* **Families, RILF − RIC0:**
  * LONGITUDINAL: 0 / 4 (s0) and 0 / 3 (s1) of 6.
  * LATERAL: 4 / 1 (s0) and 5 / 1 (s1).
  * TACTICAL: 1 / 0 (s0) and 2 / 0 (s1).
  * STRATEGIC: 3 / 0 at both seeds (S-POS@80 −1.2356 / −1.2768).
* **Against P10, the floor removes the bare arm's lateral deficit.** RILF − P10 reads LATERAL 4 / 0 at both seeds, where RIC0 − P10 reads 3 / 3.

All 19 rows:

@@T LF after=### 19 rows, dose 075: RILF - RIC0@@

* **The composite RILFC − P10** (read A), all 19 rows:

@@T LF after=### 19 rows, dose 075: RILFC - P10@@

**7.40.5 What the floor did: the instruments (reported beside the rules, never deciding) [M].**

@@T LF after=## Instruments (from the tick dumps@@

* **Where F binds.** F binds on 0.075 / 0.079 of all replans at κ 0.50 and on 0.187 at κ 0.75 (bare arm), concentrated in the crawl: 0.250 / 0.324 and 0.455 / 0.462 inside the M1 ticks. The lead cap binds on at most 0.005 of the M1 ticks. Where F binds, it lifts the 0 to 3 s plan speed by 0.563 / 0.509 m/s at κ 0.50 and by 0.666 / 0.662 m/s at κ 0.75, on average.
* **How far the floor speed is from the logged speed.**
  * The MAE is 6.322 / 6.467 m/s at κ 0.50 and 3.894 / 3.979 m/s at κ 0.75.
  * The floor exceeds the logged speed on 0.066 / 0.055 of the ticks with a floor at κ 0.50 and on 0.191 / 0.174 at κ 0.75.
  * **[H]** At κ 0.75 the floor pushes the plan above the human's own speed on about a fifth of the ticks where it is defined, which is where the along-track, speed and distance-keeping harm could come from. But at κ 0.50 it rarely does, and the ADE6 and cross-track harm remain. Which part of the harm is the estimate's and which is the operator's is therefore not settled by these instruments, and addendum A4 measures it (§7.40.6).
* **Negative chord lift** (A3_RULING_2), a result about the frozen operator.
  * On RILF050_s0 it occurs on 52 of the 543 bound replans. Across the four rollouts inspected for the ruling, the first percentile of the lift lies between −0.025 and −0.0007 m/s.
  * On those replans, the floor shortens the planned 0 to 3 s chord displacement while it advances the arc progress along O's bending path.
  * Per-rollout counts are in `LF_READ.json` (`lift_where_bound_overall`).

**7.40.6 What §7.40 establishes, and what it cannot.**

* **Established [M].**
  * A vision-only floor on the plan's progress binds where the crawl is and moves M1 the right way at both doses. It is built from LLR's trained level and lead heads and applied outside O, with O's path kept. At κ 0.75, M1 moves by −0.0211 / −0.0234, separated-better at both seeds; CL did not move M1 inside O (§7.39).
  * The floor does not close half the excess: 0.062 / 0.056 at κ 0.50 and 0.395 / 0.392 at κ 0.75. At κ 0.75 the guards fail broadly (ADE6 +0.3683 / +0.2172 m).
  * Its composite at κ 0.75 has the smallest M1 deficit against P10 of any composite read in §7.37 to §7.40: +0.0380 / +0.0401, against +0.0474 / +0.0504 for RILLRC. It also fails D2 on DK-VIOL at s0, and it is NOT DOMINANT.
  * The A3 card's committed outcome holds as written: *the floor binds but the head's estimate is too weak to lift the crawl*.
* **Not established [M].**
  * **Training variance.** The heads come from one LLR training run per seed (`H-ESTIM-SEED-1`). κ 0.75's M1 effect sits inside the C0REP floor at s0 (0.0211 against 0.0362) and outside it at s1 (0.0234 against 0.0038).
  * **The scene is not T2.** `reindexed` is a longitudinal-only approximation (§7.38.5). Under the floor, the ego runs ahead of the human on part of the ticks while the dynamic agents stay at the human's time, so the distance-keeping rows read a lead that does not react.
  * **Scope of the read.** Two doses were read, and the card names that multiplicity. `hold_now` was not read, because a frozen frame makes v_hat constant.
  * **The bottleneck is inferred, not measured.** The card's reading that the estimator is the bottleneck rests on the v_f error, not on a contrast.
* **Next (Rule Zero) [M].**
  * **The levers the A3 branch names.** Motion from vision (option (c): T2 re-perception, the A40) or a stronger estimator on more data. Each needs resources the PI has not allocated. **[E]** The A40 LoRA arm is ESTIMATED at about 470 h per seed on box 2.
  * **Run first, on the allocated CPU: addendum A4.** It is the cheapest discriminating experiment before asking for those levers (`PREREG_LEVELFLOOR_A4.md`, sha256 `{{RES_A4P}}…`, landed `{{LAND_A4P}}` before any LFO rollout). It re-runs LF075 with the level estimate replaced by the logged route speed, using the reader's own definition, and changes nothing else.
  * **What A4 is.** A CEILING diagnostic, inadmissible at inference by design. It is never a driving result and never a counted DOMINANT.
  * **A4's committed outcomes:**
    * **CEILING-CLEARS:** P and G hold with an exact level. The estimator is then the measured bottleneck: the estimator levers go to the PI with the ceiling as their measured value, and a noise sweep sets the accuracy they must reach.
    * **CEILING-HARMS:** P holds and G fails. The harm is then the operator's: the estimator levers are not signalled, and the next arm changes F.
    * **CEILING-SHORT:** P fails. A progress floor then does not close M1 at this dose.
  * A4's result is not in v40.

**7.40.7 Provenance [M].** Under `R/raw/` at the tip `{{TP}}`:
* `2026-11-05-levelfloor-a3-prereg/PREREG_LEVELFLOOR_A3.md` (sha256 `{{RES_A3P}}…`), with `FROZEN_SHA.txt`;
* `2026-11-05-levelfloor-lf-result/`:
  * `a3_prereg_additions/A3_RULING_1.md` (`{{RES_AR31}}…`) and `A3_RULING_2.md` (`{{RES_AR32}}…`);
  * `lf_build/`: the tool, the operator `code/level_floor.py`, the known answers and `PINS.txt`;
  * `lf_read/`: `RESULTS.md` (`{{RES_LF}}…`), `raw/LF_READ.json` (`{{RES_LFJ}}…`) and the 16 rollout arrays;
* `2026-11-06-levelfloor-a4-prereg/PREREG_LEVELFLOOR_A4.md` (`{{RES_A4P}}…`).

The claim rows are `H-LEVELFLOOR-1` and `H-LEVELFLOOR-1 (A4)` in `Project Steering/GOALS_AND_CLAIMS.md`. The build is `patch_v40.py`, checked by `checks_v40.py` and re-measured by `audit_v40.py`.
