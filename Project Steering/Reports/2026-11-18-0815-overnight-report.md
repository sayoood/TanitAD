# TanitAD overnight report, 2026-11-17 evening to 2026-11-18 morning (Master Mind)

Every number carries its evidence class, tier and estimator. Intervals are paired episode- (or clip-) cluster bootstraps, B 2,000, seed 0, and answer "another draw of episodes / clips" only. Repo commits are on `agent/arch-inf-20260803` unless named.

## Headline

1. **STRAT1 CKPT120: the H2 rung follows the supplied route at 120 m at both seeds (S120-FOLLOWS).**
   * MEASURED, T0, zero training.
   * real − donor: +0.745 [+0.647, +0.824] at s0 and +0.695 [+0.599, +0.778] at s1.
   * It holds where H1 fails, i.e. where the route token does not reach S: +0.730 / +0.689, separated.
   * Driving (24 T1-kin cells): H1 KEEPS. H2 SEEDS-DISAGREE: one cell at one seed, which is inside the rig's measured noise.
   * Wording: route FOLLOWING given a route along the driven path. On PhysicalAI this is optimistic by construction; it does not measure route choice. Landed 6cfcdf6b.
2. **ACTRSP Option B is training on box 2.**
   * The first launch stopped at the pilot gate on a real defect: a G-DVB reader left its noise on the CPU while the model was on CUDA. Every CPU test had passed.
   * Fix 86da79cd, proved on the dev-box GPU. G-SUITE PASS twice:
     * 4c396aff against a13f6f16: 0 regressions; 23 new tests pass.
     * 86da79cd against 4c396aff: 0 regressions, 0 missing, 14,877 tests.
   * Pilot PASS: 0.548 s/step, peak GPU 3.4 GB. The 10 arms take about 13.5 h (ESTIMATED from MEASURED parts).
   * Control **C10' PASS** on R35, measured before any fine-tune exists: s_swap on acceleration over the 141 eval clips is 0.2280 [0.2020, 0.2520] / 0.2223 [0.1952, 0.2473].
   * The read side and read chain are built, dry-run PASS, and running. They read each arm as it banks, about 50 min of dev-box CPU per arm (MEASURED).
   * The first arm, FTRSP_s0, trained its 4,000 steps, then the post check stopped it on two **checker** specification defects. Both were MEASURED on the run; neither is a defect of the run:
     * **C5.** The card's "141 of 145" counts state-dict entries. 12 of those are prior buffers with no optimiser state. Every trainable tensor got a gradient: 129 of 129.
     * **C2.** A 1e-4 relative bar is beyond float32 once the term trains to 3e-4 of a level near 6. The residual was at most 1.67 float32 ulps.
     * The fixed checker keeps regression arms that go RED (weight 0, 0.5, 0.98; 128 of 129). No outcome bar or metric was read or changed. Landed 6eec95ae.
     * FTRSP_s0 retrained under the chain, passed the post check and was banked at 07:05Z. The failed run is kept beside and not read as an arm. FTNULL_s0 is training now.
     * The read chain then stopped on a second copy of the same C5 count (141), in its own evidence tool. I applied the same ruling there and bound it to the checker's constants, so the two cannot drift again. FTRSP_s0's evidence now holds on C2, C3 and C5, and its planner read is running.
3. **DSCALE10K: block 2 DATA-HURTS (H-V8F-DSCALE10K-1 FAILED as written); block 3 shows the tactical cost travels with the 10k clips, and that more steps on B's data is a lever.**
   * MEASURED, T1-kin plus T0. At matched steps, the union (4,512 + 10,000 clips) costs TACTICAL at both seeds:
     * TAC-LAT −0.0504 [−0.0738, −0.0267] at s0;
     * TAC-LON −0.0959 [−0.1379, −0.0567] at s1.
   * All four tactical points are negative at both seeds. LONGITUDINAL does not cost. STRATEGIC gains at s1.
   * The HYPOTHESIS is that the 10k clips' masked VLM goal bits dilute tactical supervision. The F1 diagnostic is NOT-DILUTION as committed but power-limited, so it neither confirms nor excludes this.
   * **Block 3** separates "more data" from "more epochs per clip". Both arms are trained and banked: UE_s0 (the union at 61,568 steps) and BL_s0 (B's data at 61,568 steps). Each checkpoint's sha256 matches across the pod, HF and the pulled file (MEASURED).
   * **Block 3 read: `H-DATA-SCALE-1` = NEITHER** (MEASURED, stage A, ONE seed, one-pair floor 2F_A = 0.1656 m on ADE6 T0). Both readings of the card's "U − BL" agree; I ruled before the read that they are reported together.
     * U_s0 − BL_s0 = −0.0090 [−0.1055, +0.0897].
     * UE_s0 − BL_s0 = −0.0958 [−0.183, −0.0053]: separated, but below the bar.
   * **The finding block 3 was built for (MEASURED, T1-kin):** the 10k clips cost TAC-LAT by the same amount at both training lengths:
     * at 19,164 steps (U − B_s0): −0.0871 [−0.1297, −0.0446];
     * at 61,568 steps (UE − BL): −0.0855 [−0.1216, −0.0492].
     * ⇒ The tactical cost travels with the 10k clips' content, not with fewer epochs per clip. That explanation is eliminated at this seed.
     * Dilution of tactical supervision (the labelled share falls to 31 %) remains the leading HYPOTHESIS.
   * **And a lever:** more steps on B's own data (BL − B_s0) gains LONGITUDINAL, LATERAL and TACTICAL, at the cost of STRATEGIC S2.
     * TAC-LAT +0.0817 [+0.0467, +0.1175]; TAC-LON +0.0550 [+0.0229, +0.0875].
     * B at 19,164 steps is under-trained.
     * The 10k clips gain STRATEGIC at matched steps (UE − BL).
   * **Next lever:** the mixture arm (the union at 61,568 steps with the tactical supervision rebalanced to the labelled clips). It is the discriminating experiment for dilution. If dilution holds, it keeps BL's tactical gain and UE's strategic gain. It needs a pre-registration and your go for pod time.
4. **Two readout levers eliminated, both zero-training:**
   * **AMEND_6 R6-SAME:** fit-set size raises the readout's R² (0.222→0.281, 0.208→0.279), and the planner's Δ_R does not separate. Landed 59f31da8.
   * **AMEND_7 R7-FIT-FAIL:** the nonlinear (MLP) readout ties the linear one on fit (ΔR² +0.0005 / −0.0104, CIs contain 0). It is WORSE on the pass contrast (Δr_wm −0.066 / −0.070, separated at both chassis). Landed 2b2c6eb8.
   * Both cards name Option B as the next lever, and it is running.
5. **Rig noise measured (C0REP, COST-HOLDS).** Two same-seed runs separate on 5 of 48 single cells (10.4 %). S1's TAC-LAT cost holds under the replicate floor at seeds 0, 10 and 11. ⇒ We read multi-seed, aggregate rules; single cells are not claims.

## Built and staged (waiting on a decision or on compute)
* **NAVRB2** (the roundabout nav channel you asked for): built, with G-SUITE 0 regressions. It started on the pod after block 3. Its first mint refused at the launch gate, on a real gap: the init-checkpoint check did not count the new channel's key (`e_nav_rb.weight`) as declared-new. No GPU time was spent past the refusal.
  * Fix 08aa20ce, with a test that fails on the old line. A GPU selfcheck on the refused mint's exact argv passes.
  * The fix's pod G-SUITE then read 0 regressions and 0 missing, but **my new test failed** inside its whole file.
    * Cause, MEASURED: an imported module-scoped fixture (`pack`, from the DAgger tests) patches the synthetic world until the module ends, and my test ran after it.
    * I had verified the test with `-k`, which skips that fixture.
    * Fix e71782a3 moves the test ahead of it; the trainer is unchanged. On the dev box, the whole file gives 29 passed in the new order and 1 failed in the old order, with the pod's error.
  * The e71782a3 suite PASSED: 0 regressions, 0 new-test failures, 0 missing, 14,994 tests. The chain restarted at 06:53Z. N1R_s0 is done (07:08Z), and S1R_s0 is next.
  * The pod GPU was idle from 05:32Z to 06:53Z, 81 min (MEASURED from the logs). That idle time is my verification miss.
* **TFAR** (T reads S's 120 m point as a fourth checkpoint token, the CKPT120 follow-up): built, regression 0 over 145 files. Waiting on your Q1–Q6.
* **X_s0** (DSCALE: the 10k clips alone, the card's conditional arm for DATA-HURTS): chain built and staged. Waiting on your go.

## Decisions I need from you
1. **X_s0 go: about 2.6 pod-h** (ESTIMATED from MEASURED block rates). It runs after NAVRB2. Without it, or a TFAR go, **the LoRA pod idles once NAVRB2 ends, about 08:00Z** (ESTIMATED from N1R_s0's MEASURED ~15 min per arm, mint included).
2. **The DSCALE mixture arm** (block 3's next lever, above): I am drafting its pre-registration now. It needs your go for pod time; the card will carry the estimate. Seeds of BL and UE are the alternative if you prefer the one-seed result firmed up first.
3. **TFAR Q1–Q6** (`QUESTIONS_TFAR_DGOAL.md`): the architecture, the go (about 1.6 pod-h), the TN arm, H2 s10/s11 (about 0.5 pod-h), the derived-goals definition, and D0.
4. The rain-sheet answers are still open from earlier.

## Operations
* **Disk:** F: fell to 2.4 GB free overnight. I swept my own temp data, logging each path.
  * Overnight: TFAR test basetemps and logs, about 18.6 GB.
  * This morning:
    * the dev box: 2.5 GB of pytest basetemps and tree exports;
    * the pod: 50.7 GB of finished-suite basetemps and tree copies;
    * an 8.8 GB zero-filled memmap left by a stopped 10-27 extract. The complete extract is kept.
  * F: is at 30 GB, above the DSCALE card's 25 GB bar, which I did not move. The pod has 160 GB free.
* **Defects found and fixed at launch, before any GPU time was spent:**
  * Option B's PowerShell launcher built broken command lines: PowerShell's comma binds tighter than `+`.
  * Bare WMI bash lacked coreutils.
  * The chain's "shipped" marker does not record the commit, so a plain resume would have trained the OLD tree. I relaunched from a fresh state.
  * The read chain waited for a marker the planner never writes.
  * Each is recorded in its package.
* **Defects of my own verification found this morning, each fixed with evidence:**
  * NAVRB2's new test, checked with `-k` only (above).
  * The C5 ruling applied to one of its two copies.
  * Both are filed as working rules: run a new test's whole file; when a ruling changes a constant, grep every copy of it.
