### 7.39 Option (a), the level-from-image head (2026-11-05): CL does not move the crawl and O ignores the level, CLR adds nothing inside the recovery recipe, and the lead head LLR is the arm that follows

* **Status of this section (v39) [M].** Three trained arm families on box 2 (CL, CLR; LLR designed and frozen) and one zero-training read-out check (`H-LEVELHEAD-1`, `H-LEVELHEAD-2`; `R/raw/2026-11-05-levelhead-cl-freeze/`, `-cl-result/`, `-cl-readout/`, `-a2-freeze/`, `-clr-result/`, `-llr-freeze/`, with the code under `-level-head-code/` and `-lead-head-code/`). Tier T1-kin for the arm reads (the RI tool, bit-identical to the frame-freeze tool on `logged` and `hold_now`; `reindexed` is the deciding scene, §7.38.1), T0 for the read-out check. Estimator: the paired episode-cluster bootstrap, B = 2,000, seed 0, eval-141 (141 episodes / 2,399 windows), which answers *another draw of EPISODES* only; the training-run question is answered only by the C0REP floor (rule F), so a separated interval is necessary and not sufficient (`H-ESTIM-SEED-1`); the planner is deterministic at eval, so no inference seed applies. Four families per seed, never pooled; STRATEGIC is active for v8F (PI ruling 2026-10-17). The reader's header for the CL and CLR reads still carries the ITEM 25 note (STRATEGIC PAUSED, three-family D3 beside); D3 is read here over four families as in §7.36, the S-POS rows are reported and counted, and the three-family D is False as well.

**7.39.1 The head, its loss and its wiring (`level-head-code/`, `lead-head-code/`).**

* **The level head [M].** `lvl_head` is an MLP(d, d, 1) on the mean of the CURRENT tick frame's adapted tokens (vision only) and outputs v_hat in m/s. Its loss is a masked Huber, L_level = Huber(v_hat − v_log) with delta 1.0 m/s, where v_log is the LOGGED speed at NOW (`kin3[:, 0]`, the cache `v`). O reads `e_level(v_hat.detach() / 10)`: `e_level` is Linear(1, d) with weight and bias zero-initialised and its output is ADDED to O's ego conditioning token, at every tick and at inference alike (no exposure gap). The detach keeps the plan's loss out of the head (the G-LIVE control reads exactly 0.0, and removing the detach reads above 0 and turns the check RED). Zero-init makes the step-0 forward equal the off model's bit for bit (measured on the real C0 init checkpoint: 276 tensors, mismatch 0, fresh keys the six declared ones, lever 66,561 parameters). An appended token was rejected because it changes the Reader's query mean and adds a softmax key (a test shows it). Vision only at inference: ego state, nav, limit and T's code are refused by a build guard.
* **The weight [M].** The total is L = L_rest + W L_level. W is fixed by formula before any run so that the level term is a tenth of the total at step 0: **W = (0.1 / 0.9) × L_rest / L_level**, with L_rest the total loss without the level term and L_level the unweighted level loss, both measured at step 0 of a real-data selfcheck on the C0 s0 init with the head on (box-2 probe key `level_head_probe.w_level_for_10pct_share`). Measured: L_rest 20.45931339263916, L_level 11.21243953704834, so W = 0.20274419640036745, frozen as **W = 0.2027** (`FROZEN_W.txt`).
* **The lead head [M].** `lead_head` is an MLP(d, d, 2) on the same input: a presence logit (lead within 40 m) and gap_hat = sigmoid(·), the gap over 40 m. Its loss is W_lead × (BCE on presence over the labelled frames + Huber(gap_hat − min(gap, 40) / 40) with delta 0.1, i.e. 4 m, on the frames with a lead within 40 m), labels from the `--lead-train` file (the gate refuses a file covering less than 50 % of the TRAIN clips). W_lead comes from the same formula with L_lead: **W_lead = (0.1 / 0.9) × L_rest / L_lead = 3.362** (probe 3.361547620908847 under the token wiring).
* **Reachability and tests [M].** The flags reach the loss through the launch gate (G-DVB, G-LIVE: non-zero gradient on the new parameters, exact-zero controls). Level lever: 27 tests, 15 of 15 source mutations RED. Lead lever: 45 tests, 35 of 35 mutations RED. The flag-off model equals the tip bit for bit (golden K-BUILD, 36 tests). The LLR launch package: 21 tests, 38 of 38 mutations RED.

**7.39.2 CL: the level head inside C0's recipe reads P-FAILS-M1-NOT-MOVED (`R/raw/2026-11-05-levelhead-cl-result/cl_read/RESULTS.md`, landed `{{LAND_CLR_}}`).**

* **Setup [M].** CL is C0's frozen argv (init `M2b_s{s}_dagoacc`, 3,000 steps, lr 1e-4, batch 64, pool 4x10, `--recover-p 0`) plus `--level-head on --w-level 0.2027`, nothing else (argv audit). Card `PREREG_LEVELHEAD_CL.md` (sha256 `875d8c19…`, written before any CL run, W frozen before launch). Rules, all on `reindexed`: P (M1 share closed >= 0.5 at both seeds with CL − C0 on M1 separated-better); G (no non-crawl row separated-worse, ADE6 <= +0.0604 m); F (|CL − C0| on M1 exceeds the C0REP floor); D (counted DOMINANT: composite D1, D2, D3 with P and G holding); LEAK-BORNE. The outcome and next arm were committed per branch before launch.
* **Reading [M].** Outcome **P-FAILS-M1-NOT-MOVED** at both seeds. The lever reached O (`CL_PRECHECK.json`: e_level |weights| sum 0.297 / 0.386, not zero), and the crawl did not move:

@@T CLRES after=## Rules (card section 4@@

  M1 RICL − RIC0 is −0.0016 [−0.0042, +0.0011] / −0.0003 [−0.0026, +0.0020] (not separated). ADE6 is +0.0007 / −0.0206 m. G fails at s0 on two LATERAL rows (heading MAE +0.2257, L1 lateral-accel MAE +0.2199, separated-worse); at s1 no row is separated-worse and along-track MAE, heading MAE, yaw-rate MAE and L1 lateral-accel MAE improve. The frame-freeze reading is PARTIAL (shares 0.043 / −0.025, not deciding). 15 controls PASS. All 19 rows of RICL − RIC0 (the deciding contrast), per family:

@@T CLRES after=## 19 rows: RICL - RIC0@@

* **The composite RICLC = CL + S (sg72) + kbox vs P10 [M].** NOT DOMINANT: D1 and D3 (four families) hold at both seeds and D2 fails on the M1 crawl share alone (+0.0583 [+0.0393, +0.0809] / +0.0586 [+0.0373, +0.0846]). All 19 rows of RICLC − P10 (Read A):

@@T CLRES after=## 19 rows: RICLC - P10@@

**7.39.3 The head read-out check: v_hat PARTIAL, O-INSENSITIVE (`R/raw/2026-11-05-levelhead-cl-readout/RESULT.md`, T0).**

* **Setup [M].** Addendum A1 (`PREREG_CL_READOUT_A1.md`, sha256 `1f96d833…`, written after the CL read and before any read-out number) fixes two zero-training measurements on the CL checkpoints at tick 0 of eval-141 (logged history and frames): tracking (R² of v_hat against the logged speed over all windows; MAE in the 3 to 8 m/s band) and O's sensitivity (replace v_hat/10 by (v_hat ± 3)/10 and read the change of the plan's 0 to 3 s mean speed). Labels: TRACKS if R² >= 0.5 and band MAE <= 1.5 m/s; DOES-NOT-TRACK if R² < 0.3 or band MAE > 2.5; else PARTIAL. O-SENSITIVE if the +3 m/s shift moves the plan's mean speed by >= 0.5 m/s; O-INSENSITIVE below 0.1. This is a head and world-model diagnostic (T0), not a driving result, so the four-family table is not engaged.
* **Reading [M].** Both seeds read **PARTIAL** (tracking) and **O-INSENSITIVE**. Tracking, then sensitivity:

@@T A1 after=## 1. Tracking@@

@@T A1 after=## 2. O's sensitivity@@

  The band error is a one-sided over-estimate (mean v_hat minus logged speed in the band +1.01 / +0.65 m/s). The sensitivity is three orders of magnitude below the 0.1 m/s floor. Controls: a shift of 0.0 changes the plan by exactly 0.0, and a reach control (a fixed random direction of norm 8 added to e_level's output) moves the plan's mean speed by 0.118 / 0.099 m/s, so the harness reads a change at that site and O answers it weakly even at that size. The committed branch is the lead-aware head (A1 row "DOES-NOT-TRACK or PARTIAL"). **[H]** (flagged by the reader, not a rule): a lead-aware head that enters O the same way, in a model trained the same way, would meet the same ignoring. Tracking was read at tick 0 only.

**7.39.4 CLR: the level head inside RAPR-P01's recovery recipe reads P-FAILS-ATTR-NOT-SEPARATED-BETTER (`R/raw/2026-11-05-levelhead-clr-result/clr_read/RESULTS.md`, landed `{{LAND_CLRR}}`).**

* **Setup [M].** Addendum A2 (`PREREG_LEVELHEAD_A2.md`, sha256 `9a117b20…`, written after the read-out check and before any CLR or LLR run) puts the heads inside RAPR-P01's recovery recipe (one deviation per window, a-priori law, p 0.1, rejoin target), where the history is perturbed and the target is the human's, so vision is the informative input. CLR is RAPR-P01's argv plus `--level-head on --w-level 0.2027`, no new code. Bases: C0 (P, G, D) and RAPR-P01 under `reindexed` (attribution). A2 ruling 1 (`A2_RULING_1.md`, sha256 `{{RES_AR1}}…`, before any CLR read) names the one cell the card left open.
* **Reading [M].** Outcome **P-FAILS-ATTR-NOT-SEPARATED-BETTER** at both seeds. P: M1 RICLR − RIC0 is separated-better (−0.0124 [−0.0173, −0.0076] / −0.0077 [−0.0132, −0.0028]), but the share closed is 0.233 / 0.130 (bar 0.5). G fails: along-track MAE and DK-VIOL are separated-worse (DK-VIOL at both seeds, DK-TG at s0, ADE6 and along-track MAE at s1; ADE6 +0.0323 / +0.0585 m). F: |RICLR − RIC0| on M1 is 0.0124 against the floor 0.0362 at s0 (inside) and 0.0077 against 0.0038 at s1 (outside). Not LEAK-BORNE (logged 0.229 / 0.127, reindexed 0.233 / 0.130). The frame-freeze reading is PARTIAL (0.235 / 0.110). Rules:

@@T CLRR after=## Rules (CL card section 4@@

  The attribution row decides: against RAPR-P01's own checkpoint the level adds nothing on M1 (RICLR − RIRAPRP01 −0.0015 [−0.0038, +0.0005] / +0.0003 [−0.0027, +0.0031]), and CLR's share (0.233 / 0.130) is of the size of RAPR-P01's (0.2147 / 0.1391, §7.36.4a), so the closure is the recovery recipe's and not the level's. Families, RICLR − RIC0: LONGITUDINAL 0 better / 2 worse (of 6) at both seeds; LATERAL 4 / 0 (s0) and 5 / 0 (s1); TACTICAL 1 / 0 at both; STRATEGIC 3 / 0 at both. All 19 rows of RICLR − RIC0:

@@T CLRR after=## 19 rows: RICLR - RIC0@@

  The attribution contrast, all 19 rows (CLR against RAPR-P01's own checkpoint, reindexed):

@@T CLRR after=## 19 rows: RICLR - RIRAPRP01@@

* **The composite RICLRC = CLR + S (sg72) + kbox vs P10 [M].** NOT DOMINANT: D1 and D3 hold, D2 fails on the M1 crawl share alone (+0.0479 [+0.0300, +0.0689] / +0.0527 [+0.0308, +0.0783]); DK-VIOL (+0.0269 / +0.0017) is not separated. All 19 rows of RICLRC − P10 (Read A):

@@T CLRR after=## 19 rows: RICLRC - P10@@

* **CLR's own O-sensitivity, which fixes LLR's wiring [M].** The A1 instrument on the CLR checkpoints reads O-INSENSITIVE at both seeds (`WIRING.md`; T0, not a driving result):

@@T WIR after=Note: both seeds.@@

  The A2 card maps O-INSENSITIVE to feeding both heads' outputs as one separate conditioning token of O (`--aux-cond token`, non-zero init).

**7.39.5 LLR: the lead head, designed and frozen before training (`R/raw/2026-11-05-levelhead-llr-freeze/`).**

* **Design [M].** LLR is CLR's argv plus `--lead-head on --w-lead 3.362 --aux-cond token` (everything else, including the recovery tokens, RAPR-P01's). The frozen record (`FROZEN_LLR.txt`, sha256 `{{RES_FL}}…`) reads `WIRING=token` and `W_LEAD=3.362`; both were written before training (repository commit `574dea8`). In the token wiring O reads one extra conditioning token `e_aux` = Linear(3, d), default (non-zero) init, of the three detached values (v_hat / 10, sigmoid(logit), gap_hat); it REPLACES the two additive paths (`e_level` and `e_lead` are not built), so it is a wiring O cannot average away. By construction the step-0 plan differs from the init's (3.1 to 3.6 m on a synthetic batch), and the level share is recorded and not windowed. The gate adds the lead lever to CLR's: fresh keys exactly the declared ten (lvl_head x4, lead_head x4, e_aux x2), vision-only proof, `aux_token_dim` 3, every probe control exactly 0.0, and the 50 % label-coverage refusal.
* **Reads and committed outcomes [M].** LLR is read as CLR was (T1-kin on the RI tool, bases C0 and RAPR-P01, `reindexed` deciding), with the attribution LLR − CLR on M1 beside. The card commits: *"D holds for an arm at both seeds: counted DOMINANT. Signal the PI for scale, with the video evidence."* *"LLR P fails: the vision-head family on frozen single-frame features is exhausted at these forms; the remaining levers are the named blockers of option (c) (T2 re-perception, the A40), reported to the PI with the numbers."* *"P holds, G or D fails: the arm's dose (W / 2) under the same card."*
* **UNVERIFIED before the run [M].** A real-data selfcheck with both heads and the recovery lever on had not run before box 2; the chain duration (about 2.5 h) is ESTIMATED from CLR's; the lead label file is not sha-pinned in the package (its sha256 is recorded in the gate token and compared with the run's).
