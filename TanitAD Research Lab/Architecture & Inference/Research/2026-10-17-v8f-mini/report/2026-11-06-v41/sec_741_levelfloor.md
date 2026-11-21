### 7.41 Addenda A4 to A6, the level floor's ceiling, κ 1 and the bounded lift (2026-11-06): the progress-floor family is exhausted on CPU

Every read here is T1-kin on `reindexed` (deciding; `logged` only for the LEAK-BORNE rule of the admissible arms), with the paired episode-cluster bootstrap (B = 2,000, seed 0, eval-141), which answers another draw of episodes only. The arms have zero training and ran on the dev-box CPU. O is C0's own checkpoint in every contrast with C0, so O's training variance is absent from those contrasts, and the floor heads come from one LLR training run per seed (`H-ESTIM-SEED-1`). All four metric families are reported for every arm, in the tables below. The readers' header carries the ITEM 25 note on STRATEGIC (PAUSED, PI ITEM 25, 2026-09-19: reported beside the verdict, never scored as a pass of its own); §7.40 records the PI ruling of 2026-10-17 under which STRATEGIC is active for v8F, the STRATEGIC rows are in every table, and D3 is read over four families (Read A), as for LF.

* **Two kinds of arm, named in every label.**
  * A **CEILING** arm takes the human's logged route speed as its level input, exact or under a priori noise. That is a privileged channel and is inadmissible at inference by design. A CEILING arm is never a driving result, a capability claim or a counted DOMINANT; each ranks levers and nothing else.
  * An **ADMISSIBLE** arm reads the level from LLR's vision heads, as LF does. Only these can count.
* **Family counts in the tables.** The line under each 19-row table counts separated-better and separated-worse rows per family (the readers' sign convention: lower is better for every row except TAC-LAT and TAC-LON, where higher is better; ADE6 and the M1 crawl share are outside the family counts).
* **One family.** A4, A5 and A6 are successive addenda to the A3 card of §7.40. Each fixes its next arm in the committed outcomes of the card before it, and each card was frozen before any rollout of its own arms.

**7.41.1 A4, the ceiling at κ 0.75 (`R/raw/2026-11-06-levelfloor-a4-prereg/PREREG_LEVELFLOOR_A4.md`, sha256 `{{RES_A4P}}…`, landed `{{LAND_A4P}}`; read `R/raw/2026-11-06-levelfloor-a4-lfo-result/lfo_read/RESULTS.md`, sha256 `{{RES_LFO}}…`, landed `{{LAND_LFO}}`) [M].**

* **Why.** §7.40 read LF at κ 0.75 closing 0.395 / 0.392 of the M1 excess with ADE6 +0.3683 / +0.2172 m, and the A3 card named the level estimator as the bottleneck. That reading was an inference. Nothing measured separated two parts of the harm: the estimator's error, which a better estimator removes, and the operator's own harm (the ramp, the resampling along O's path), which no estimator removes. At κ 0.50 the floor exceeds the logged speed on only 0.066 / 0.055 of the ticks, yet ADE6 is +0.1089 m at s0 (§7.40.3).
* **The arm LFO075.** LF075 with the level estimate replaced by the logged route speed of the tick: the path length of knots 1 to 6 of the logged route in the ego frame, divided by 1.0 s, computed in the tool at the replan before F. Where the tick has no complete logged route there is no floor. Planner C0, lead cap from LLR's heads, κ 0.75, every constant of F as in A3. The bare arm is O's plan → S (sg72) → F and the composite is S → F → kbox, as in LF. Four rollouts (bare and composite, two seeds), on `reindexed` only. `logged` is not read: the level input is the human's speed by construction, so the LEAK-BORNE rule has no question left (disclosed in the card).
* **The hypothesis and the rules, written before any rollout.** **[H]** With an exact level, the floor at κ 0.75 would close at least half of the M1 excess inside the CL guards. The rules are A3's with LF := LFO: P_O (share closed at least 0.5 at both seeds, M1 separated-better), G_O (ADE6 at most +0.0604 m and no non-crawl row separated-worse), F_O (against the C0REP floor), the attribution RILFO − RILF075 (reported, never deciding), and the composite read as the CEILING of D, never counted.
* **Committed outcomes (card section 5).**
  * CEILING-CLEARS: the estimator is the measured bottleneck; the next CPU arm is a noise sweep.
  * CEILING-HARMS: the operator is the problem; the estimator levers are not signalled and the next arm changes F.
  * CEILING-SHORT: a floor on progress does not close M1 at this dose; the estimator levers are not signalled for this form; what remains is T2 re-perception or the operator at κ 1 in its own addendum.
* **The build and its known answers (`lfo_build/`, patched tool sha256 `6744d3e4…`, regenerable from the LF tool `6166178a…`) [M].** The flag is `--floor-vhat-source {heads,logged}`.
  * KA-O1: a full rollout with the heads source reproduces the banked RILF075_s0 byte for byte (46 common tick fields, 7,197 rows, 141 dump files, the analysis JSON and the 44 per-window metric arrays).
  * KA-O2: with the floor off, a full rollout reproduces the banked RIC0_s0 byte for byte.
  * KA-O3: the oracle input equals the reader's `logged_speed` to 1e-9 m/s, and a tick without a complete route gets no floor.
  * 12 of 12 tool mutations turn a test RED. The reader passed its two known answers (24 of 24 and 17 of 17 checks), 31 of 31 reader, pre-check and chain mutations turn a test RED, and all 13 controls of the read passed.
  * **Disclosed.** The LF chain's gate 0 loop over code and test files never ran, because a sed delimiter doubled as the alternation character. The LFO chain uses another delimiter and a test catches a changed pinned file. LF results are unaffected (`HANDOFF_LFO.md`).

**The read: CEILING-SHORT.** F binds on 0.536 / 0.621 of the M1 crawl ticks (237 / 240 ticks), above the 1 % reach bar, so the rules are read.

* **P_O fails.** The share closed is 0.126 / 0.132 (bar 0.5), against 0.395 / 0.392 with the heads at the same dose. M1 RILFO − RIC0 is −0.0067 / −0.0079, separated-better at both seeds.
* **G_O fails, narrowly.** ADE6 is −0.0233 / −0.0246 m, inside the guard and not separated. Accel MAE and cross-track MAE are separated-worse at both seeds, and L1 lateral-accel MAE at s1.
* **F_O fails (it needs both seeds).** 0.0067 against the C0REP floor of 0.0362 at s0 is inside the floor; 0.0079 against 0.0038 at s1 is outside it.
* **The ceiling of D (never counted).** RILFOC − P10 fails D2 on the M1 crawl share alone (+0.0527 / +0.0563), with D1 and D3 true.

The rules:

@@T LFO after=Reach control (F binds on >= 1 %@@

All 19 rows, RILFO − RIC0 (the deciding contrast):

@@T LFO after=### 19 rows: RILFO - RIC0 (reindexed@@

@@F LFO after=### 19 rows: RILFO - RIC0 (reindexed as=RILFO − RIC0@@

* **Attribution, RILFO − RILF075** (the level estimate's error at a fixed operator, dose, planner and lead cap; reported, never deciding), all 19 rows:

@@T LFO after=### 19 rows: ATTRIBUTION: RILFO - RILF075@@

* **The reading of the attribution [M].** The exact level removes most of LF075's harm: ADE6 −0.3916 / −0.2418 m, along-track MAE −0.3158 / −0.2105 and DK-VIOL −0.0269 / −0.0269, all separated-better. It also removes most of LF075's closure: the M1 crawl share is +0.0143 / +0.0155 higher, separated-worse at both seeds. LF075's closure therefore came largely from the estimate overshooting the human's speed, and the A3 reading that the estimator is the bottleneck is not supported at κ 0.75.
* **The ceiling of D, RILFOC − P10** (read A, all 19 rows, never counted):

@@T LFO after=### 19 rows: RILFOC - P10@@

* **Instruments (beside, never deciding).** No-route ticks are 987 (0.137), where there is no floor. The check v_log − v_f against (1 − κ) · v_log has a maximum of 9.50e-07 m/s, so the oracle input is the logged speed.

@@T LFO after=## Instruments (from the tick dumps@@

* **The heads-source arm, same instruments:**

@@T LFO after=Beside, the heads-source arm LF075@@

* **Why κ 0.75 could not close the crawl: the geometry (`beside/M1_GEOMETRY.md`, descriptive, computed by the Master Mind after the read, chooses nothing) [M].**
  * On 0.848 / 0.838 of the M1 ticks, 0.75 · v_log is below 5.2 m/s, so a κ 0.75 floor cannot lift the target above the M1 threshold. At κ 1 that share is 0.0 at both seeds.
  * On the M1 ticks the median v0 is 4.484 / 4.478 m/s and the median v_log is 5.968 / 6.023 m/s. The 1.0 m/s² ramp cannot reach 5.2 m/s within the executed 2 s on 0.122 / 0.133 of them.
  * κ 1 was already the card's committed next arm for CEILING-SHORT, so this geometry explains the outcome and chooses nothing.
* **Next (committed in the card, executed in §7.41.2).** The operator at κ 1 in its own addendum, A5.

**7.41.2 A5, κ 1 with the exact level, a priori noise and the heads (`R/raw/2026-11-06-levelfloor-a5-prereg/PREREG_LEVELFLOOR_A5.md`, sha256 `{{RES_A5P}}…`, landed `{{LAND_A5P}}`; read `R/raw/2026-11-06-levelfloor-a5-lf5-result/lf5_read/RESULTS.md`, sha256 `{{RES_LF5}}…`, landed `{{LAND_LF5}}`) [M].**

* **The card.** Frozen at 04:39:36 (Europe/Berlin) before any κ 1 rollout or tool change. The title says 04:5x; `A5_NOTE_TIME_LABEL.md` records that the label is the only difference. The hypotheses, written before any rollout:
  * **[H1]** With the exact level at κ 1, the floor closes at least half of the M1 excess inside the CL guards.
  * **[H2]** The closure survives some estimator error, and the noise level at which P or G first fails is the accuracy spec an estimator must meet.
* **The arms, all at κ 1 (planner C0, lead cap from LLR's heads, F and its constants as in A3, placement as in LF).**
  * **LFO100** (CEILING): the level is the logged speed, as in A4.
  * **LFN100x10 and LFN100x20** (CEILING): the level is v_log · exp(σ ε) with σ = 0.10 and σ = 0.20, fixed a priori and not fitted. ε is drawn from N(0, 1) independently at every replan by one generator per rollout, seeded 20261106, whether or not a floor applies, so the bare and composite rollouts of a seed see the same draws.
  * **LF100** (ADMISSIBLE, vision-only): the level comes from LLR's level head, as in LF. It is the only arm that can count.
  * 20 rollouts: 16 on `reindexed` (deciding), and LF100 also on `logged` (bare and composite) for its LEAK-BORNE rule.
* **The rules.** The ceiling arms use P_O, G_O and F_O as in A4, each with one label from CEILING-CLEARS, CEILING-HARMS, CEILING-SHORT and F-UNREACHED. LF100 uses the A3 rules exactly. The spec read is reported and never deciding: σ-star is the largest σ in {0, 0.10, 0.20} at which P_O and G_O both hold at both seeds, set beside the heads' realised relative error.
* **Multiplicity, named in the card.** The heads-floor family has now read three doses (κ 0.50, 0.75, 1.0); a counted DOMINANT would name its dose and say that three were read.
* **Committed outcomes (card section 5).**
  * LF100 D holds at both seeds with P and G and is not LEAK-BORNE: counted DOMINANT, a scale signal to the PI with the video evidence.
  * LFO100 CEILING-CLEARS with LF100 not DOMINANT: the form works given a good level and σ-star is the spec; a σ-star of at least 0.10 sends the estimator levers to the PI, and a σ-star of 0 means the form needs an exact level.
  * LFO100 CEILING-HARMS: matching the human's speed along O's path harms on the named rows, so the operator is the problem; the next arm changes F in its own addendum and no estimator lever is signalled.
  * LFO100 CEILING-SHORT: the progress-floor form is exhausted.
* **The build and its known answers (`lf5_build/`, patched tool sha256 `88b275a1…` = the LFO tool plus `apply_lf5_patch.py`) [M].** The flags are `--floor-vhat-noise` and `--floor-vhat-noise-seed`.
  * KA-N1: σ = 0 on the noise path equals the LFO tool at κ 1 on a 64-window smoke, bit for bit.
  * KA-N2: the bare and composite draws are identical (192 rows).
  * KA-N3: (a) κ 0.75 with no noise equals the LFO tool's own smoke bit for bit; (b) it equals the banked RILFO075_s0 on the 46 of 64 windows whose model output is batch-identical. The float32 output moves on 18 of 64 windows with or without the patch, so (b) cannot be bit-identical on all of them, and (a) is the strict comparison.
  * Tool tests 5 of 5 and 8 of 8 tool mutations RED. Reader known answer ka-a4: 6 of 6 (the banked RILFO075 arrays in the LFO100 slot reproduce `LFO_READ.json`). Reader tests 6 of 6 and 11 of 11 mutations RED. All 29 controls of the read passed.
* **What failed, and what was changed before any rule was read (`A5_NOTE_READER_FIX.md`) [M].** The 20 rollouts finished with every rollout check PASS. The reader's first production run then refused at the extraction of the four RILFO100 arrays: with σ = 0 the tool writes `floor_vhat_noise_seed = None` into the ticks file, numpy stores it as a 0-d object array, and the pinned loader opens ticks with `allow_pickle=False`. One script, run once before the rerun, moved each original file aside (kept, not deleted) and wrote a new file with every other member copied byte for byte and that key dropped; its record shows byte identity of every kept member and that the new files load. The card, the reader, the rules, the tool and the rollouts were not changed, and the reader was rerun unchanged.
* **The disclosure the card makes about the noise arms.** They carry one noise draw (one generator seed, the same draws at both seeds), so a difference between noise levels smaller than a second draw would show is not resolved here.

**The joint reading: OPERATOR-IS-THE-PROBLEM.** Matching the human's speed along O's path harms on the named rows. LF100 reads P-HOLDS-G-OR-D-FAILS, which counts as a result, and the heads-floor family has now read three doses.

**The exact level, LFO100: CEILING-HARMS.** The share closed is 0.738 / 0.739, separated-better at both seeds, so P_O holds. G_O fails on two rows at both seeds: DK-VIOL (+0.0235 / +0.0168) and cross-track MAE (+0.1232 / +0.1068) are separated-worse, while ADE6 (+0.0333 / +0.0068 m) stays inside the guard of +0.0604 m.

@@T LF5 after=## LFO100 (CEILING): **CEILING-HARMS**@@

All 19 rows, LFO100 − RIC0:

@@T LF5 after=## LFO100 (CEILING): **CEILING-HARMS** nth=2@@

@@F LF5 after=## LFO100 (CEILING): **CEILING-HARMS** nth=2 as=LFO100 − RIC0@@

**The exact level under a priori noise: CEILING-HARMS at both σ.** The share closed stays above 0.5 (σ 0.10: 0.710 / 0.660; σ 0.20: 0.712 / 0.638), but G_O fails broadly at both seeds. ADE6 is +0.6140 / +0.5848 m at σ 0.10 and +0.9543 / +0.9486 m at σ 0.20. The noise is symmetric in the log of the level, and the separated rows move one way: along-track MAE, speed MAE, DK-VIOL, DK-HW and DK-TG are separated-worse at both seeds. **σ-star is None**: G_O already fails at σ = 0 (LFO100), so no σ holds (reported, never deciding).

Rules, σ 0.10:

@@T LF5 after=## LFN100x10 (CEILING): **CEILING-HARMS**@@

All 19 rows, LFN100x10 − RIC0:

@@T LF5 after=## LFN100x10 (CEILING): **CEILING-HARMS** nth=2@@

@@F LF5 after=## LFN100x10 (CEILING): **CEILING-HARMS** nth=2 as=LFN100x10 − RIC0@@

Rules, σ 0.20:

@@T LF5 after=## LFN100x20 (CEILING): **CEILING-HARMS**@@

All 19 rows, LFN100x20 − RIC0:

@@T LF5 after=## LFN100x20 (CEILING): **CEILING-HARMS** nth=2@@

@@F LF5 after=## LFN100x20 (CEILING): **CEILING-HARMS** nth=2 as=LFN100x20 − RIC0@@

**The heads, LF100 (the only admissible arm): P-HOLDS-G-OR-D-FAILS [M].** P holds and F holds; G fails, D fails (read A, four families) and the arm is not LEAK-BORNE.

* **P.** The share of the M1 excess closed is 0.899 / 0.783.
* **G fails broadly.** ADE6 is +1.2559 / +1.0891 m, with along-track MAE, speed MAE, DK-VIOL, DK-HW and cross-track MAE separated-worse at both seeds.
* **D fails.** D1 fails (ADE6 is not separated-better), and D2 fails on DK-HW, DK-VIOL and along-track MAE at s0 and on DK-VIOL, the M1 crawl share and along-track MAE at s1. D3 holds over four families.
* **The heads' error on the same frames.** The log-ratio standard deviation of v_hat against the logged speed is 0.636 / 0.626 and the MAE is 2.596 / 2.607 m/s, against realised log-ratio standard deviations of 0.100 and 0.201 for the two noise arms. The heads' v_hat exceeds the logged speed on 0.516 / 0.495 of the ticks.

@@T LF5 after=## LF100 (admissible, vision-only): the A3 rules@@

@@F LF5 after=## LF100 (admissible, vision-only): the A3 rules as=LF100 − RIC0@@

* **Instruments (beside, never deciding), all 20 rollouts:**

@@T LF5 after=## Instruments (beside, never deciding)@@

* **Next (A5 section 5, executed in §7.41.4).** For CEILING-HARMS the card commits an arm that changes F, in its own addendum, with no estimator lever signalled. The A6 card was written after a descriptive diagnosis of these rollouts (§7.41.3).

**7.41.3 The diagnosis that chose A6 (`R/raw/2026-11-06-levelfloor-a6-prereg/beside/A6_DIAG.md`, sha256 `{{RES_A6D}}…`, landed `{{LAND_A6D}}`): descriptive, computed after the A5 read, never deciding [M].**

The inputs are the banked A5 rollouts and their tick dumps, all on eval-141 `reindexed`. Per-window differences against RIC0 are summed inside window classes and divided by the reader's n, so the class contributions add up to the reader's delta. The scripts and their JSON are beside the note.

1. **The heads' harm is overshoot.** In windows where the floor speed exceeded the logged speed by more than 0.25 m/s on a bound tick, ADE6 is +1.088 of +1.089 m (RILF100_s1).
2. **Most of the closure is preventive.** Only 0.113 / 0.106 of the bound ticks of the exact arm (0.211 / 0.208 for the heads) have v0 below 5.2 m/s. Windows with a bound tick at v0 of at least 5.2 m/s carry −0.0317 of the exact arm's −0.0394 M1 effect at s0, so a gate on the ego's own speed would drop most of the closure.
3. **The heads are a weak binary classifier where it matters.** On the M1 ticks, v_hat is at least 5.2 m/s on 0.572 / 0.458. On ticks with a logged speed below 3 m/s it is at least 5.2 m/s on 0.221 / 0.186.
4. **With the exact level, the harm sits where the floor holds speed against O's slowdown.** Windows where F asked for more than max(v0, 5.5 m/s) are where it helps (ADE6 −0.1975 / −0.1840). The other bound windows carry the DK-VIOL contribution (+0.0235 / +0.0135) and the cross-track contribution (+0.1108 / +0.1073). Those are class contributions; the reader's totals for LFO100 are +0.0235 / +0.0168 and +0.1232 / +0.1068 (§7.41.2). The note's reading: *the floor holds the first second's level constant over the 6 s plan while the human slows later in it.*
5. **A floor confined to the first 2 s is rejected as a design.** The plan executes its first 10 knots (2 s) per replan and is scored over 30 (6 s), so such a floor would leave the executed motion identical and change only the scored tail: it would change the score and not the drive.

* **The lever chosen for A6.** Bound the lift relative to O's own plan, v_f(τ) ≤ v_O(τ) + δ. This caps the one-sided overshoot that any level error produces (items 1 and 3, and the noise arms of A5), keeps O's own slowdown shape inside the hold windows (item 4), and leaves the ramp, the lead cap and κ 1 as in A5.
* **Adaptivity, disclosed.** The δ doses were chosen after looking at eval-141 rollouts, and the A6 read is on the same windows. The diagnosis is descriptive and never decides a rule.

**7.41.4 A6, the bounded lift F_B at κ 1 (`R/raw/2026-11-06-levelfloor-a6-prereg/a6_prereg/PREREG_LEVELFLOOR_A6.md`, sha256 `{{RES_A6P}}…`, landed `{{LAND_A6P}}` before any A6 rollout; read `R/raw/2026-11-06-levelfloor-a6-lf6-result/lf6_read/RESULTS.md`, sha256 `{{RES_LF6}}…`, landed `{{LAND_LF6}}`) [M].**

* **The operator F_B.** A5's F with one added term in the floor speed. v_O(τ) is O's planned speed after S, taken from consecutive knot distances along its polyline over DT = 0.2 s, with knot 0 at the origin:

$$
v_f(\tau)=\min\!\Big(\kappa\,\hat v,\;\; v_0+1.0\,\tau,\;\; v_O(\tau)+\delta\Big)
$$

  The lead cap on κ · v_hat is exactly as in A3. The rest is unchanged: s_f(t) is the integral of v_f, s' = max(s, s_f) is resampled along O's polyline as in A3, the floor applies only where v_hat is at least 3 m/s, and the placement is as in LF (bare = S → F_B, composite = S → F_B → kbox). Since v_f is at most v_O + δ, the lift is bounded: `s' − s ≤ δ · t` at every knot.
* **Doses.** δ ∈ {0.5, 1.0} m/s, fixed a priori. The M1 ticks need a median lift of about 0.7 m/s to leave the crawl (v0 median 4.48 m/s), so δ 1.0 covers that median and δ 0.5 is half of it.
* **The arms.** κ = 1.0 in every arm; the CEILING arms are LFOB05 and LFOB10, the only arms that can count are LFB05 and LFB10:

@@T A6P after=**Doses:**@@

* **Rollouts and bases.** 24 rollouts: 16 on `reindexed` (deciding), and LFB05 and LFB10 also on `logged`, bare and composite, for the LEAK-BORNE rule. The bases are the banked arrays, unchanged (RIC0, RIC0C, C0, C0C, C0REP, P10 and, from A5, RILF100, RILF100C, RILFO100, RILFO100C). The attributions are LFB{δ} − RILF100 and LFOB{δ} − RILFO100, per dose.
* **The rules.** The CEILING arms use P_O, G_O and F_O exactly as in A4 (one label each); the admissible arms use the A3 rules exactly (P, G, F, D read A over four families, LEAK-BORNE on `logged`, the reach control).
* **Multiplicity and adaptivity, named in the card.** The heads-floor family will have read five doses (κ 0.50, 0.75 and 1.0, then δ 0.5 and 1.0 at κ 1); a counted DOMINANT would name its δ and say that five were read. The form and the δ doses were chosen after looking at eval-141 rollouts and the A6 read is on the same windows, so a counted DOMINANT would carry that disclosure.
* **Committed outcomes (card section 5, read in this order).**
  1. LFB05 or LFB10 with D, P and G at both seeds and not LEAK-BORNE: counted DOMINANT; it names κ 1, its δ, the five doses and the adaptivity, and the PI is signalled for scale with the video evidence.
  2. Otherwise a CEILING-CLEARS at either LFOB dose: the bounded form works given a good level and the heads are the bottleneck; the estimator levers go to the PI and the next CPU arm is a temporal filter of the heads' level.
  3. Otherwise LFOB is CEILING-HARMS at both doses: bounding the lift does not remove the operator's harm; in the card's words, *The progress-floor family is exhausted on CPU*, and T2 re-perception (option (c)) and the A40 are reported to the PI.
  4. Otherwise (CEILING-SHORT at both doses, or SHORT at one and HARMS at the other): the bound removes the closure. The floor's closure needs a lift larger than O's plan supports, and no measured estimator supports such a lift; the family is exhausted on CPU, and T2 and the A40 are reported to the PI.
  5. F-UNREACHED at a dose: the instrument is checked first.
* **The build and its known answers (`lf6_build/`, patched tool sha256 `949c4b7a…` = the LF5 tool `88b275a1…` plus `apply_lf6_patch.py`; `level_floor.py` unchanged, `2781e23a…`) [M].** The flag is `--floor-lift-cap DELTA`; absent, the tool is the LF5 tool. F_B lives in the tool. The A5 defect is fixed there: the ticks omit `floor_vhat_noise_seed` when it is None, and the strict load passed in the pre-flight. Smoke against smoke, 64 windows, bit for bit:
  * K1: δ 0 equals the floor off (heads bare, heads composite, logged bare); F never binds. This is the analytic zero control.
  * K2: δ 10^9 equals the LF5 tool (three smokes).
  * K3: δ 0.5 is partial, read on the same inputs (A6_RULING_1 below).
  * The bound `s' − s ≤ δ · t` holds on every knot (largest excess 4e-16 m), and an independent recomputation from the dumped plan agrees to 3e-8 m.
  * Tool tests 12 of 12 and 14 of 14 tool mutations RED. Reader tests 10 of 10 and 19 of 19 reader mutations RED, including one-character changes of δ, κ, the smoothing and a tolerance. Reader known answer ka-a5: 12 of 12 (the banked A5 arrays in the A6 slots reproduce `LF5_READ.json`'s LFO100 and LF100 rules and labels exactly, and the attribution against itself is exactly zero). All 32 controls of the read passed.
  * The reader's production path (24-array extraction, the LFB reads, the LFOB reads, the joint reading) was exercised end to end by ka-a5 and the unit tests before the chain's first production run.
* **A6_RULING_1 (`R/raw/2026-11-06-levelfloor-a6-lf6-result/a6_prereg_additions/A6_RULING_1.md`, sha256 `{{RES_A6R1}}…`, landed `{{LAND_A6R1}}`; written 2026-11-06 07:06 after the known-answer runs and before any A6 rule was read) [M].** The card asks that δ 0.5's bind rate lie strictly between those of δ 0 and δ 10^9 on a 64-window smoke.
  * Read on each smoke rollout's own ticks the order fails: δ 0.5 binds 0.4844 against 0.4271 at δ 10^9 (heads bare), because a smaller lift leaves the car slower and a slower car meets more crawl ticks.
  * Read on the same inputs (the ticks F saw in the δ 0.5 smoke, run through δ 0 and δ 0.5 of the tool under test and δ 10^9 of A5's independent `apply_floor_batch`), the order holds: 0 < 0.484 < 0.490.
  * **The ruling.** Known answer 3 is read on the same inputs, because a known answer pins a property of the operator and on fixed inputs the bound set grows with δ by construction. The rollout-level rates are reported beside and do not gate, and the narrow margin (1 to 3 of 192 ticks) is disclosed. The card is not edited, and no P, G, F or D rule, outcome or dose changes. Two key names in `KA_LF6.json` were renamed after the run, and no value changed (`A6_NOTE_KA3_READING.md`).

**The joint reading: THE-BOUND-REMOVES-THE-CLOSURE (card outcome 4).** Both LFOB arms read CEILING-SHORT, so the bound removes the closure. Both admissible arms read P-FAILS-M1-MOVED, which counts as a result, and the heads-floor family has now read five doses (κ 0.50, 0.75 and 1.0, then δ 0.5 and 1.0 at κ 1).

**The exact level, bounded at δ 0.5 (LFOB05): CEILING-SHORT.** The share closed is 0.384 / 0.329, separated-better at both seeds, below the bar of 0.5. ADE6 is −0.1165 / −0.1027 m. G_O fails on cross-track MAE (+0.0189 / +0.0210, separated-worse at both seeds). F_O fails (0.0205 against 0.0362 at s0, inside the floor; 0.0196 against 0.0038 at s1, outside it).

@@T LF6 after=## LFOB05 (CEILING): **CEILING-SHORT**@@

All 19 rows, LFOB05 − RIC0:

@@T LF6 after=## LFOB05 (CEILING): **CEILING-SHORT** nth=2@@

@@F LF6 after=## LFOB05 (CEILING): **CEILING-SHORT** nth=2 as=LFOB05 − RIC0@@

* **Attribution, LFOB05 − RILFO100** (the same exact level at the unbounded lift; beside):

@@T LF6 after=Attribution (beside): LFOB05 - RILFO100@@

**The exact level, bounded at δ 1.0 (LFOB10): CEILING-SHORT.** The share closed is 0.544 / 0.488, separated-better at both seeds; s1 is below the bar of 0.5. ADE6 is −0.1163 / −0.0930 m. G_O fails on cross-track MAE (+0.0488 / +0.0582) at both seeds, with DK-VIOL (+0.0134) at s0 and accel MAE (+0.0458) at s1. F_O fails (0.0290 against 0.0362 at s0, inside the floor; 0.0291 against 0.0038 at s1, outside it).

@@T LF6 after=## LFOB10 (CEILING): **CEILING-SHORT**@@

All 19 rows, LFOB10 − RIC0:

@@T LF6 after=## LFOB10 (CEILING): **CEILING-SHORT** nth=2@@

@@F LF6 after=## LFOB10 (CEILING): **CEILING-SHORT** nth=2 as=LFOB10 − RIC0@@

* **Attribution, LFOB10 − RILFO100** (beside):

@@T LF6 after=Attribution (beside): LFOB10 - RILFO100@@

* **The ceiling of D at both doses (never counted).** D1 and D3 hold and D2 fails at both doses and both seeds, on the M1 crawl share alone.

**The heads, bounded at δ 0.5 (LFB05, admissible): P-FAILS-M1-MOVED [M].** P, G, F and D (read A, four families) all fail, and the arm is not LEAK-BORNE. The share of the M1 excess closed is 0.382 / 0.327 and ADE6 is +0.1588 / +0.1336 m (separated-worse at both seeds). Heading, yaw-rate and curvature MAE are separated-better at both seeds. D2 fails on the M1 crawl share alone.

@@T LF6 after=## LFB05 (admissible, vision-only): **P-FAILS-M1-MOVED**@@

@@F LF6 after=## LFB05 (admissible, vision-only): **P-FAILS-M1-MOVED** as=LFB05 − RIC0@@

* **Attribution, LFB05 − RILF100** (the unbounded heads floor of A5; beside):

@@T LF6 after=Attribution (beside): LFB05 - RILF100@@

**The heads, bounded at δ 1.0 (LFB10, admissible): P-FAILS-M1-MOVED [M].** P, G, F and D (read A) fail, and the arm is not LEAK-BORNE. The share of the M1 excess closed is 0.570 / 0.492 (s1 is below the bar of 0.5) and ADE6 is +0.4344 / +0.3956 m. Heading, yaw-rate and curvature MAE are separated-better at both seeds. D2 fails on DK-VIOL and the M1 crawl share at s0, and on the M1 crawl share alone at s1.

@@T LF6 after=## LFB10 (admissible, vision-only): **P-FAILS-M1-MOVED**@@

@@F LF6 after=## LFB10 (admissible, vision-only): **P-FAILS-M1-MOVED** as=LFB10 − RIC0@@

* **Attribution, LFB10 − RILF100** (beside):

@@T LF6 after=Attribution (beside): LFB10 - RILF100@@

* **Instruments (beside, never deciding), all 24 rollouts.** The bound `s' − s ≤ δ · t` holds on every bound tick of every rollout (the largest excess is 1.78e-15 m). The δ term is the binding minimum on 0.947 / 0.961 of the bound ticks at δ 0.5 and on 0.842 / 0.858 at δ 1.0 (heads, bare).

@@T LF6 after=## Instruments (beside, never deciding)@@

**7.41.5 What the family establishes, and what it cannot.**

* **Established [M].**
  1. **The max-floor turns level error into one-sided overshoot.** A floor of the form max(s, s_f) lifts the plan wherever the level exceeds O's progress and never lowers it. With the heads, ADE6 is +1.2559 / +1.0891 m at κ 1, and the windows where the floor speed exceeded the logged speed carry +1.088 of +1.089 m (§7.41.3). With the exact level under symmetric noise in the log of the level, ADE6 is already +0.6140 / +0.5848 m at σ 0.10. Bounding the lift against O's own plan removes most of the heads' harm: LFB05 − RILF100 is −1.0970 / −0.9555 m in ADE6.
  2. **With the exact level, the unbounded floor harms DK-VIOL and cross-track MAE.** LFO100 closes 0.738 / 0.739 of the M1 excess and is separated-worse on DK-VIOL (+0.0235 / +0.0168) and cross-track MAE (+0.1232 / +0.1068) at both seeds, with ADE6 inside the guard (+0.0333 / +0.0068 m). The diagnosis puts that harm where the floor holds the first second's level against O's slowdown (§7.41.3).
  3. **The bound removes the closure, and most of the harm goes with it.** At δ 0.5 the exact level closes 0.384 / 0.329 instead of 0.738 / 0.739, with DK-VIOL not separated (+0.0034 / +0.0067) and cross-track MAE reduced to +0.0189 / +0.0210, still separated-worse. At δ 1.0 the closure is 0.544 / 0.488, DK-VIOL is +0.0134 / +0.0050 and cross-track MAE +0.0488 / +0.0582.
  4. **Even the exact level, bounded, closes only part of the M1 excess:** 0.384 / 0.329 at δ 0.5 and 0.544 / 0.488 at δ 1.0 (bar 0.5 at both seeds). At δ 1.0 the heads close 0.570 / 0.492, more than the exact level at s0, at an ADE6 of +0.4344 / +0.3956 m against −0.1163 / −0.0930 m for the exact level.
  5. **The bounded exact-level arms improve ADE6 by about 0.1 m on RIC0** (−0.1165 / −0.1027 m at δ 0.5 and −0.1163 / −0.0930 m at δ 1.0, separated-better at both seeds) **and fail G_O on cross-track MAE.** They are CEILING diagnostics and never driving results.
  6. **At κ 0.75 the exact level closes less than the heads do** (0.126 / 0.132 against 0.395 / 0.392). The descriptive geometry (§7.41.1) is that 0.75 · v_log is below the M1 threshold on most M1 ticks, and the attribution puts the heads' closure there largely on overshooting the human's speed.
  7. **No counted DOMINANT exists.** Five heads-floor doses were read (κ 0.50, 0.75 and 1.0, then δ 0.5 and 1.0 at κ 1), none of the admissible arms satisfied D, and the reader's joint reading signals no estimator lever for this form.
* **Not established [M].**
  * **Training variance.** The heads come from one LLR training run per seed (`H-ESTIM-SEED-1`), and the C0REP floor on M1 is 0.0362 at s0 and 0.0038 at s1. The M1 effects of the bounded exact-level arms (0.0205 / 0.0196 at δ 0.5, 0.0290 / 0.0291 at δ 1.0) and of LFO075 (0.0067 / 0.0079) sit inside the floor at s0 and outside it at s1, so the seed-level evidence is split; LFO100's (0.0394 / 0.0440) sits outside it at both seeds. The reads do not measure a second LLR training run.
  * **What the interval answers.** The paired episode-cluster bootstrap answers another draw of episodes only. The planner, F, F_B and the oracle input are deterministic, so no inference seed applies; the noise arms carry one draw (§7.41.2).
  * **The scene is not T2.** `reindexed` is a longitudinal-only approximation (§7.38.5), and under the floor the ego runs ahead of or behind the human while the dynamic agents stay at the human's time, so the distance-keeping rows read a lead that does not react. The arms are T1-kin, and no T2 number exists.
  * **Adaptivity.** The form of F_B and its two δ doses were chosen after looking at eval-141 rollouts and the A6 read is on the same windows. Five doses were read in all.
  * **The CEILING arms are inadmissible at inference.** Their level input is the human's logged route speed, so they rank levers and say nothing about a drivable system.
  * **Scope of the operator.** The family tested a floor on the plan's along-track progress along O's own path, with a 1.0 m/s² ramp, a 3 m/s band and the A3 lead cap. A different operator (for example one that also moves the path, or a floor confined to some ticks) is not read here.

**7.41.6 The remaining levers and what unblocks each (Rule Zero) [M].** A refutation of the progress-floor form is a waypoint. The done condition is a result that clears its committed bar, or every remaining lever blocked with what would unblock it named. No counted DOMINANT exists, so there are no evidence videos and no scale signal.

* **(a) The A40 for the LoRA perception run.** The PI decision `D-V8F-SPEED-PERCEPTION`, option (a): the A40 go for the LoRA run. It waits on three PI decisions: the A40 provisioned, the frames on HF private or a disk provision (LORA §7, AR-4), and the go for about 200 GPU-h (2 seeds). **[E]** On box 2 the same LoRA arm is ESTIMATED at about 470 h per seed, which is why it needs the A40. The go is the PI's. §7.41 adds the measurement that the exact level, bounded or not, does not close the crawl under this operator.
* **(b) T2 re-perception.** Not provisioned as a decision-grade tier (AlpaSim, CARLA or a neural re-renderer; §7.38.5). It unblocks with a re-perception tier. A decision input landed after the A6 read (`D-V8F-CRAWL-T2-SIZING`, a sizing and not a claim; `R/raw/2026-11-06-t2-sizing-v8f/T2_SIZING_V8F.md`, sha256 `{{RES_T2S}}…`, landed `{{LAND_T2S}}`), with its own evidence classes. The T2-A kit on box 2 can run v8F on the 2 scenes on hand with a PI go per job, but only one scene holds the M1 situation, so that stage cannot decide the lever. A decision-grade read needs about 42 M1-bearing scenes (**[E]**), a download of about 26 GB (80 GB worst case, PI approval), about 3.7 box-2 hours (**[E]**) and 6 to 8 days of engineering; no A40 is needed for T2. Until that read exists, `reindexed` stays a longitudinal-only approximation and no T2 number exists.
* **(c) A stronger level estimator on more data.** The exact-level ceiling says this cannot clear the bar under this operator: even the exact level, bounded, closes 0.384 / 0.329 and 0.544 / 0.488 of the excess, and unbounded it fails the guards. The estimator levers are therefore not signalled by this evidence, and a different operator would need its own card.
* **Run next on the allocated CPU.** The progress-floor family has no further committed arm: outcome 4 of the A6 card commits only the report of T2 and the A40 to the PI. Further CPU items come from `Project Steering/BACKLOG.md`, the programme's pull-list.

**7.41.7 Provenance [M].** Under `R/raw/` at the tip `{{TP}}`:
* `2026-11-06-levelfloor-a4-prereg/PREREG_LEVELFLOOR_A4.md` (sha256 `{{RES_A4P}}…`, landed `{{LAND_A4P}}`), with `FROZEN_SHA.txt`;
* `2026-11-06-levelfloor-a4-lfo-result/`:
  * `lfo_build/`: the tool `tool/v8f_t1_eval_ri_lfo.py` (sha256 `6744d3e4…`), `HANDOFF_LFOBUILD.md` (`{{RES_LFOBH}}…`) and `PINS.txt`;
  * `lfo_read/`: `RESULTS.md` (`{{RES_LFO}}…`, landed `{{LAND_LFO}}`), `raw/LFO_READ.json` (`{{RES_LFOJ}}…`) and `HANDOFF_LFO.md` (`{{RES_LFOH}}…`);
  * `beside/M1_GEOMETRY.md` (`{{RES_M1G}}…`);
* `2026-11-06-levelfloor-a5-prereg/PREREG_LEVELFLOOR_A5.md` (`{{RES_A5P}}…`, landed `{{LAND_A5P}}`);
* `2026-11-06-levelfloor-a5-lf5-result/`: `lf5_build/` (tool sha256 `88b275a1…`), `lf5_read/RESULTS.md` (`{{RES_LF5}}…`, landed `{{LAND_LF5}}`), `lf5_read/raw/LF5_READ.json` (`{{RES_LF5J}}…`), `lf5_read/A5_NOTE_READER_FIX.md` (`{{RES_A5NF}}…`) and `lf5_read/A5_NOTE_TIME_LABEL.md` (`{{RES_A5NT}}…`);
* `2026-11-06-levelfloor-a6-prereg/`: `a6_prereg/PREREG_LEVELFLOOR_A6.md` (`{{RES_A6P}}…`, landed `{{LAND_A6P}}`) and `beside/A6_DIAG.md` (`{{RES_A6D}}…`) with its scripts and JSON;
* `2026-11-06-levelfloor-a6-lf6-result/`: `a6_prereg_additions/A6_RULING_1.md` (`{{RES_A6R1}}…`, landed `{{LAND_A6R1}}`), `lf6_build/` (tool sha256 `949c4b7a…`, `A6_NOTE_KA3_READING.md` `{{RES_A6NK}}…`), `lf6_read/RESULTS.md` (`{{RES_LF6}}…`, landed `{{LAND_LF6}}`) and `lf6_read/raw/LF6_READ.json` (`{{RES_LF6J}}…`);
* `2026-11-06-t2-sizing-v8f/T2_SIZING_V8F.md` (`{{RES_T2S}}…`, landed `{{LAND_T2S}}`), cited in §7.41.6 only.

The claim rows are `H-LEVELFLOOR-1 (A4)`, `H-LEVELFLOOR-1 (A5)` and `H-LEVELFLOOR-1 (A6)` in `Project Steering/GOALS_AND_CLAIMS.md`. The build is `patch_v41.py`, checked by `checks_v41.py` and re-measured by `audit_v41.py`.
