# Verdict reading: PREREG_V8F_N1DW. The roundabout-window weight gains on D1-HO at seed 0 only, and the eval-141 guard carries a cost at both weights, so no committed outcome row is reached (2026-11-20)

**Status:** the verdict reading of the card, drafted by an MM agent and reviewed by the MM against the card's section 7 and the result files. Section 3.3 was added by the MM. It is landed in the repo under `raw/2026-11-19-n1dw/read/verdict/`, beside the eval-141 code and outputs in `raw/2026-11-19-n1dw/read/eval141/`.
* Card `PREREG_V8F_N1DW.md` (sha256 59f2b7f9, re-hashed from the remote tip c9757d7e) with `AMENDMENT_N1DW_1.md` (sha256 bff20b4d). The amendment touches only the PROBE-N1D-FEAT regression arm and the tree seeds; no N1DW bar, row or control.
* Machine-readable labels and every number below: `RESULT_N1DW_VERDICT.json` (sha256 4d74f91e), written by `build_verdict.py`, which applies section 7 row by row and asserts its gain and guard flags against the two readers' own flags.

**Evidence, tier, estimator (every number here unless marked)**
* MEASURED, ours, zero training. Artifacts: D1-HO `RESULT_N1DW_D1HO.json` (sha256 6473f154, equal to the landed copy at the tip); eval-141 `eval141/out/RESULT_N1DW_EVAL141.json`, `eval141/out/N1DW_STRATEGIC_contrast.json`; STRATEGIC labels `verdict/N1DW{2,4}_STRATEGIC_S1slot.json`, reference `verdict/N1R_STRATEGIC_S1slot.json`.
* Tier: **T1-kin/old** (the card's frozen executor) for D1-HO and the eval-141 guard. STRATEGIC: **T0** at the window's first tick, fresh ages, as STRAT1.
* Estimator: `taniteval.ci.paired_episode_cluster_bootstrap`, B 2,000, seed 0. It answers "another draw of EPISODES" only. Training-run variance is carried by F_HO and the two seeds; inference variance does not apply (the planner does not sample).
* Limit: optimistic by construction, with the route supplied along the driven path.

## 1. The verdict per weight

| weight | D1-HO gain (bar 0.0424) | eval-141 guard | rows of section 7 | label |
|---|---|---|---|---|
| **w = 2** | s0 yes (+0.0448), s1 no (+0.0246) | cost at s0 (distance-keeping time gap 2-6 s); none at s1; no ROUTE-COST | none fires | **no committed row reached** |
| **w = 4** | s0 yes (+0.0453), s1 no (+0.0143) | cost at s0 (time gap 2-6 s) and at s1 (TAC-LAT); no cell at both seeds; no ROUTE-COST | none fires | **no committed row reached** |

**The clauses that decide it, quoted from section 7 and section 5.2:**
* Guard, 5.2: "The guard holds at a weight iff no cell is a cost at either seed and there is no ROUTE-COST". A cost at s0 exists at both weights, so **the guard does not hold at either weight**.
* **RBW-ONE-SEED**: "at a weight, a gain at exactly one seed, guard holds". Its D1-HO part holds at both weights; **its guard clause fails at both**. The failure is the same under the narrower reading "the guard at the gaining seed", since the cost sits at s0, the gaining seed, at both weights.
* **RBW-GAIN** and **RBW-GAIN-ONE-COST** need "D1-HO a gain at BOTH seeds": not met.
* **RBW-COSTS**: "a guard cell a cost at both seeds, or D1-HO a cost at both seeds". No single cell is a cost at both seeds, and D1-HO is not a cost at any arm. w = 4 carries costs at both seeds, but on two different cells (s0 time gap, s1 TAC-LAT). That is not this row as written.
* **RBW-SPLIT**: both weights gain at the same seed (s0), not at "the other". Not met.
* **RBW-DOSE** and **RBW-NO-GAIN** need "no gain at either weight"; each weight has a gain at s0. Not met. RBW-DOSE also fails on its own: N1DW4 - N1D at s1 is -0.0028 [-0.0142, +0.0082], not separated.
* **RBW-FLOOR-HIGH**: the bar 0.0424 is not above every observed N1DW - N1R (+0.0448 at w2_s0, +0.0453 at w4_s0). Not met.

**What this is.** The case of a gain at one seed together with a guard cost at one seed has no row. Coordinator item R3 handled a one-seed cost only together with a gain at both seeds (RBW-GAIN-ONE-COST). ⇒ **The card names no committed next step for this read.** Choosing one is a PI decision (section 6). Nothing enters the recipe.

**Printed with the label, as the card asks:**
* Attribution: at s0, N1DW - N1D is separated at both weights (+0.0157 [+0.0061, +0.0252] w2, +0.0162 [+0.0061, +0.0264] w4). So the attribution caveat ("not attributed to the weight over the channel") does not apply at the gaining seed. At s1, N1DW - N1D is not separated at either weight.
* Margins: the s0 gains clear the bar by 0.0024 (w2) and 0.0029 (w4).
* K-CLIP: 60 of 60 logged steps clipped at every N1DW arm, as for N1D.
* Mean two-seed N1DW - N1R: w2 +0.0347, w4 +0.0298, a difference of 0.0049.

## 2. STRATEGIC: why it read NOT-YET, and what it reads now

**Diagnosis, hop by hop**
1. `eval141/n1dw_view.py:24-28` copies the N1DW passes into the **N1 slot**: `eval_N1DW-N1DW{w}_s{s}_<v>.npz` becomes `t0/eval_STRAT1-N1_s{s}_<v>.npz`. The S1 slot stays empty. This was inherited from `navrbd_view.py`, whose docstring says so.
2. Frozen `strat1/read/strat1_strategic.py`:
   * `:252-255`: S1_s0 / S1_s1 are set to "NOT-YET, a T0 pass is missing".
   * `:267-269`: route-following blocks are still computed for the N1 slot.
   * `:278-280`: the RC-A rows against C0 are built for S1 only.
   * `:284`: `per = {s: done[f"S1_s{s}"]["block"] ...}` feeds the label from S1 only.
   * `outcome()` (`:145-154`) then returns "NOT-YET | the card needs both seeds" for an empty `per`.
   * `:300`: the T1 per-tick row is built only for tags that start with "S1".
3. ⇒ **The NOT-YET is an instrument wiring defect.** It says the S1 slot is empty and says nothing about N1DW. The N1DW passes were all present and read; no seed file was missing.
4. Frozen `strat1/read/strat1_ctx_read.py`:
   * `:264` reads only `eval_STRAT1-S1_s{s}_real.npz` and its `ctx_logits`.
   * `:6` says "C0 and N1 have no head: n/a". The run's own output agrees: `not_in_arm: {"C0": "no output", "N1": "no output"}`.
   * Second probe: the N1DW T0 pass carries no `ctx_logits` key (keys checked on `eval_N1DW-N1DW4_s0_real.npz`).
   * ⇒ The CTX NOT-YET is structural: **N1DW has no context head**. It is NOT-COMPUTABLE, not pending.
5. `n1dw_read141.py:77-78,131-135,153` adds to `not_yet` only a missing contrast file or a missing zero dump. It never reads the view's label, and it never records its own N1D beside rows. That is why it printed "not-yet 0" while two items were NOT-YET (the S-label, and N1DW - N1D on eval-141, section 3).

**Fix and rerun (CPU, 2 threads, one process, each run behind `wait_commit.py 5.5 60`, which passed at 26.4-26.6 GB free)**
* New wrapper `verdict/n1dw_view_s1slot.py` puts the arm in the **S1 slot** of the frozen `strat1_strategic.py`. It is admissible because that reader reads no ctx output and no `nav_source` field.
* Runner `verdict/run_s1slot.sh`, log `verdict/s1slot.log`: three runs, N1DW4, N1DW2, and N1R as the reference. The views were deleted afterwards (`DELETIONS_VERDICT.log`).
* **Identity control:** for all four arms, the route-following blocks and sensitivity rows from the N1 slot (the chain's own outputs) equal the S1-slot blocks exactly (dict equality). The fix adds the label, the RC-A rows against C0 and the T1 row attempt. It changes no number.
* **The CTX and the ROUTE-COST contrast were not rerun.** CTX cannot run (no head). The contrast was complete; see below.

**ROUTE-COST was decided on complete input.**
* All 24 RC-A rows (2 weights x 2 seeds x 6 cells) of `N1DW_STRATEGIC_contrast.json` read status OK, with n 2,294 to 2,322 windows and 136 to 137 episodes.
* None is a cost at any seed. 23 of 24 deltas are negative (lower error than N1R). The one positive delta, +0.0010 m (w2_s1, position at 30 m), is not separated.
* ⇒ `route_cost = false` at both weights rests on measured input, and the guard label does not rest on a missing input.

**STRATEGIC in full (T0; arcs 30 / 50 / 80 m; n at 30 / 50 / 80 m = 75 / 149 / 244 windows and 19 / 23 / 25 episodes, for every arm)**

| | N1DW2_s0 | N1DW2_s1 | N1DW4_s0 | N1DW4_s1 |
|---|---|---|---|---|
| S_CKPT level, real / donor / zero, 80 m | 0.840 / 0.205 / 0.193 | 0.869 / 0.225 / 0.217 | 0.840 / 0.193 / 0.172 | 0.877 / 0.213 / 0.193 |
| real - donor, 80 m | +0.635 [+0.540, +0.728] | +0.643 [+0.539, +0.729] | +0.647 [+0.553, +0.737] | +0.664 [+0.559, +0.749] |
| real - best constant (0.766), 80 m | +0.074 [-0.116, +0.294] | +0.102 [-0.070, +0.300] | +0.074 [-0.117, +0.297] | +0.111 [-0.061, +0.301] |
| real - C0 (real nav), 80 m | +0.107 [+0.004, +0.244] | +0.119 [+0.015, +0.263] | +0.107 [+0.004, +0.244] | +0.127 [+0.020, +0.270] |
| N1DW - N1R, S_CKPT separated cells (9 per seed) | none | none | none | donor 50 m -0.027 [-0.057, -0.006]; zero 80 m -0.029 [-0.052, -0.005] |
| N1DW - N1R, route-following skill (real - donor) | none separated | none separated | none separated | none separated |
| RC-A N1DW - N1R (6 cells, floor C0_rep - C0_s0) | gains at 80 m: position -0.79 m [-1.91, -0.01], heading -1.16 deg [-2.62, -0.07]; no cost | no cell separated; no cost | no cell separated; no cost | no cell separated; no cost |
| RC-A N1DW - C0 | no cell separated | no cell separated | no cell separated | no cell separated |

* **Route-following label (frozen STRAT1 rule, deciding arc 80 m): S-PARTIAL at both weights.** Real - donor is separated above 0 at both seeds; real - best constant is not separated at either seed. **N1R reads S-PARTIAL too**, so the weight does not change the label.
* The frozen rule's next step for S-PARTIAL is the 120 m checkpoint card. That card is outside this card; it is printed for the record only.
* On the roundabout windows (235 / 16 episodes), the RB-beside route-following rows are n/a at every arc, n = 0 ("below 30 windows or 10 episodes").
* **Detected information** (country, weather, surface, illumination, period of day): **NOT-COMPUTABLE at all four arms, n = 0 heads**, because N1DW has no context head, as N1 and N1D before it. Fog is unscored (PI 2026-11-17): 390 eval steps / 15 clips. The no-fog weather constant is 0.8703 over 13,841 steps.
* Controls: K7 hold 0.0 / announced x40 deg 1.0 / mirrored 0.0. K8 donor 0.10656 equals its expected value.
* **The T1 per-tick S_CKPT row is not quoted.** Its gate K9 (T1 tick-0 `s_ckpt` against the T0 real pass, atol 1e-6) fails at every arm, the banked C0 arms included:
  * max abs diff 11.7 to 72.9 over 2,399 windows;
  * the same in the NAVRBD view.
  * This is a pre-existing instrument item, not an N1DW finding. STRATEGIC here is T0 only.

## 3. The four families per weight and seed (never pooled)

### 3.1 On eval-141, the guard: N1DW - N1R, 2,399 windows / 141 episodes unless marked, STRAT1 cost / gain rule, floor F2
Each cell reads delta [95% CI]. Marks: **C** cost, **G** gain, s separated inside the floor, blank not separated.

| family | cell | N1DW2_s0 | N1DW2_s1 | N1DW4_s0 | N1DW4_s1 |
|---|---|---|---|---|---|
| LONG | speed 0-2 s | +0.0002 | -0.0012 | -0.0010 | -0.0036 G |
| LONG | along 0-2 s | +0.0005 | -0.0012 | -0.0018 G | -0.0031 G |
| LONG | accel 0-2 s | -0.0037 s | +0.0049 s | +0.0003 | +0.0054 s |
| LONG | headway 0-2 s (698 / 72) | +0.0060 | +0.0176 | +0.0006 | +0.0077 |
| LONG | time gap 0-2 s (646 / 72) | -0.0016 | +0.0129 | -0.0013 | +0.0121 |
| LONG | speed 2-6 s | -0.0092 | -0.0393 s | -0.0524 s | -0.0469 s |
| LONG | along 2-6 s | +0.0168 | +0.0099 | -0.0066 | -0.0004 |
| LONG | accel 2-6 s | -0.0946 | -0.2194 G | -0.2170 G | -0.2531 G |
| LONG | headway 2-6 s (554 / 61) | +0.1081 | +0.1121 | +0.1998 | +0.0078 |
| LONG | **time gap 2-6 s** (522 / 60) | **+0.0450 [+0.0136, +0.0793] C** | +0.0093 | **+0.0456 [+0.0130, +0.0852] C** | +0.0141 |
| LAT | cross-track 0-2 s | -0.0015 G | -0.0023 G | -0.0026 G | -0.0031 G |
| LAT | heading 0-2 s | -0.0203 G | -0.0147 | -0.0332 G | -0.0265 G |
| LAT | yaw rate 0-2 s | -0.0766 s | +0.0122 | -0.0943 s | -0.0279 |
| LAT | curvature 0-2 s | -0.0023 G | +0.0002 | -0.0025 G | -0.0014 |
| LAT | cross-track 2-6 s | -0.0279 | -0.0050 | -0.0283 | -0.0002 |
| LAT | heading 2-6 s | -0.5555 s | +0.1313 | -0.7729 s | +0.0298 |
| LAT | yaw rate 2-6 s | -2.3159 s | -3.2537 s | -3.9184 s | -0.8145 |
| LAT | curvature 2-6 s | -0.0140 | -0.0061 | -0.0293 s | +0.0307 |
| TAC | **TAC-LAT** | -0.0008 | -0.0092 | +0.0004 | **-0.0163 [-0.0325, -0.0017] C** |
| TAC | TAC-LON | +0.0000 | +0.0058 | -0.0025 | +0.0017 |
| ADE/FDE | ADE / FDE 0-2 s | -0.0006 / -0.0008 | -0.0026 G / -0.0071 G | -0.0033 G / -0.0038 | -0.0044 G / -0.0093 G |
| ADE/FDE | ADE / FDE 2-6 s | -0.0051 / -0.0422 | +0.0034 / -0.0239 | -0.0217 / -0.1176 | -0.0057 / -0.0303 |

* Floors of the two cost cells (F2): time gap 2-6 s 0.0219 s; TAC-LAT 0.0067.
* Beside, MEASURED from `strat1/navrbd/read/out/RESULT_NAVRBD.json`: the parent channel arm already carries the time-gap cost at s0. N1D_s0 - N1R_s0 reads +0.0316 [+0.0061, +0.0617], a cost (522 / 58). N1D_s1 - N1R_s1 is not a cost.
* ⚠️ **N1DW - N1D on eval-141 was not computed** (`beside_eval141_N1DW-N1D` reads "NOT-YET (no --n1d-141 dumps)").
  * `run_eval141.sh:64` does not pass `--n1d-141`.
  * The N1D eval-141 dumps were deleted locally on 2026-11-19 (`strat1/navrbd/read/DELETIONS_MM.log`). They are banked in HF `navrbd_read/navrbd_read_outputs.tar` (sha256 143f3b2a).
  * Card 5.2 says "Every cell also prints N1DW - N1D". This is a work item.
  * It is also the one read that tells whether the s0 time-gap cost belongs to the weight or to the channel. N1DW reads +0.045 against N1D's +0.032, but that comparison is unpaired.
* RB windows of eval-141 (235 / 16), beside:
  * D1 N1DW - N1R: +0.0596 [+0.0047, +0.1224] w2_s0; +0.0426 [-0.0128, +0.0960] w2_s1; +0.0596 [+0.0201, +0.1013] w4_s0; +0.0255 [-0.0329, +0.0860] w4_s1.
  * Z-USE TAC-LAT real - zero: +0.0553 [+0.0091, +0.1024] w2_s0; +0.0383 [-0.0050, +0.0861] w2_s1, not separated; +0.0596 [+0.0132, +0.1107] w4_s0; +0.0468 [+0.0092, +0.0873] w4_s1.

### 3.2 On D1-HO (5,736 windows / 497 held-out episodes), beside, no bar
* **TACTICAL** (the deciding family):
  * TAC-LAT N1DW - N1R is in section 1.
  * TAC-LON N1DW - N1R: +0.0265, +0.0258, +0.0253, +0.0249 (w2_s0, w2_s1, w4_s0, w4_s1), all separated. N1DW - N1D is separated at s0 only (+0.0188 w2, +0.0176 w4).
  * D2-HO, N1DW4_s0 vs N1R: turn recall -0.0205 and false-turn share -0.0491, both separated.
  * D2-HO, N1DW4_s1 vs N1R: turn recall +0.0142 and false-turn share +0.0202, both separated.
  * D2-HO at w2: only the s1 turn recall is separated, +0.0179.
* **LATERAL**, N1DW - N1D, every lateral cell at both horizons and all four arms is lower or not separated:
  * heading 2-6 s: -1.18 / -1.03 / -1.19 / -1.09 deg;
  * yaw rate 2-6 s: -3.65 / -4.54 / -5.56 / -6.12 deg/s;
  * curvature 2-6 s: separated at three arms, not at w2_s1 (-0.0141);
  * cross-track 2-6 s: separated at three arms, not at w4_s0 (-0.0426).
* **LONGITUDINAL**, N1DW - N1D:
  * speed, along-track and accel at 2-6 s are lower, separated, at all four arms.
  * Along-track at 0-2 s is higher, separated, at w2_s0 (+0.0038 m), w2_s1 (+0.0060 m) and w4_s1 (+0.0025 m).
  * ADE 0-2 s at w2_s1 is +0.0036 m, separated.
  * **Headway and time gap: NOT-COMPUTABLE at every arm and both horizons, n = 0** (no lead block for held-out clips).
* **STRATEGIC**: read on eval-141 (section 2). The held-out cell has no route-setting reference.

### 3.3 MM addition: N1DW - N1D on eval-141 (computed 2026-11-20 after this reading was drafted), beside, no bar

The N1D eval-141 dumps were pulled from HF `navrbd_read/navrbd_read_outputs.tar` (sha256 143f3b2a, equal to its bank record). `n1dw_read141.py --n1d-141` was rerun on them, giving `verdict/RESULT_N1DW_EVAL141_withN1D.json`. Its guard labels equal the guard read of record. MEASURED, T1-kin/old, eval-141 2,399 windows / 141 episodes, paired episode-cluster bootstrap B 2,000 seed 0 (episodes only).

| N1DW - N1D | time gap 2-6 s (s) | headway 2-6 s (m) | TAC-LAT | ADE 2-6 s (m) | speed 2-6 s (m/s) | cross-track 2-6 s (m) |
|---|---|---|---|---|---|---|
| w2_s0 | +0.0064 [-0.0117, +0.0246] | +0.0351 [-0.0916, +0.1668] | +0.0008 [-0.0104, +0.0129] | +0.0087 [-0.0229, +0.0437] | +0.0166 [-0.0243, +0.0662] | -0.0102 [-0.0332, +0.0147] |
| w2_s1 | +0.0067 [-0.0149, +0.0284] | +0.0654 [-0.0379, +0.1706] | -0.0142 [-0.0287, +0.0000] | +0.0395 [-0.0060, +0.0878] | -0.0069 [-0.0503, +0.0384] | +0.0291 [-0.0052, +0.0680] |
| w4_s0 | +0.0126 [-0.0099, +0.0402] | +0.1102 [-0.0373, +0.2680] | +0.0021 [-0.0108, +0.0167] | -0.0079 [-0.0486, +0.0343] | -0.0266 [-0.0628, +0.0069] | -0.0105 [-0.0452, +0.0290] |
| w4_s1 | +0.0097 [-0.0159, +0.0384] | -0.0204 [-0.2129, +0.1792] | **-0.0213 [-0.0387, -0.0046], a cost** | +0.0304 [-0.0205, +0.0821] | -0.0146 [-0.0550, +0.0263] | +0.0339 [-0.0038, +0.0748] |

* **The seed-0 time-gap cost belongs to the channel, not to the weight.** N1DW - N1D on that cell is not separated at either weight. The parent N1D_s0 already carries +0.0316 against N1R (`RESULT_NAVRBD.json`). The weight adds no separated time-gap cost over its parent at either seed.
* **The w4 seed-1 TAC-LAT cost belongs to the weight.** N1DW4_s1 - N1D_s1 is -0.0213 [-0.0387, -0.0046]. At w2_s1 the same cell is -0.0142 [-0.0287, +0.0000], not separated.
* This is a beside read. It does not change the verdict of record, which stays as section 1 reads it: no committed row is reached. It is information for the PI's choice in section 6.

## 4. Controls

| control | value | status |
|---|---|---|
| K-RE, D1-HO (N1R, N1D and N1D-zero recomputed against `RESULT_NAVRBD.json`) | 6 of 6 equal | PASS |
| K-RE, eval-141 | 48 rows compared, 0 mismatches | PASS |
| K1, N1R_s0 self-pair | 0.0 [0.0, 0.0] | PASS |
| K-CONST, majority class | 0.338215 | PASS |
| **Z-EVAL**, eval-141 real - zero on the zero-history windows | 2,160 windows / 130 episodes; max abs window difference 0.0 on every cell, 0 NaN mismatches, at all four arms | **PASS at all four** |
| Z-USE, D1-HO channel real - zero | +0.0481 / +0.0145 / +0.0453 / +0.0228 (w2_s0, w2_s1, w4_s0, w4_s1), all separated | reported |
| eval-141 problems | [] | PASS |

* **Why the brief saw "Z-EVAL None":** the reader stores Z-EVAL per arm (`controls["Z-EVAL"][<arm>]["pass"]`), with no top-level `pass` key. A top-level lookup returns None. Every per-arm value is True.
* **Z-USE, what it says:** against N1D's own real - zero (+0.0464 at s0, +0.0202 at s1, from `D1_HO_each_acc`), the weight does not raise the arm's use of the channel. Every arm is within 0.006 of its seed's N1D value. That comparison is unpaired (ESTIMATED as a difference of two paired deltas).

## 5. Executor
Every number here is **T1-kin/old**, the card's frozen executor. A pre-registered overlap read under the new executor of record (**T1-kin/hb0.2**, PI ruling D16-D, card `raw/2026-11-19-n1dw/overlap_hb/PREREG_N1DW_HB_OVERLAP.md`) is pending. Its result will be reported beside this one.

## 6. Decisions for the PI
1. **The outcome gap.** Gain at one seed with a guard cost at one seed has no committed row, at either weight. The card does not say what follows. The options, each with its cost:
   * **(a) s10 / s11 at w = 2** (2 arms, about 1 to 1.5 LoRA pod-h, ESTIMATED from card section 8). They would be read under the same bar, plus the carry test that RBW-GAIN-ONE-COST commits: the time-gap 2-6 s cell must not be a cost at both new seeds. Why w = 2: the only committed cross-weight rule (the RBW-SPLIT tie rule) gives w = 2, because the two-seed means differ by 0.0049 < 0.005. w = 2 also carries one cost seed against w = 4's two.
   * **(b) The lateral-only weight**: the extra term on the cross-track component only, at w = 2 and 4. This is the next lever RBW-COSTS names for a LONGITUDINAL cost, and both s0 costs are LONGITUDINAL (distance-keeping time gap). 4 arms, about 2 to 3 LoRA pod-h (ESTIMATED).
   * **(c) Park the weight line and take N1DC** (item 3).
   * The MM's reading, for the PI to weigh:
     * The weight's own gain over N1D is about +0.016 at s0 and not separated at s1.
     * It clears the bar by under 0.003.
     * The N1DC draft aims at a room of about +0.10 at T0 (ESTIMATED, from PREREG_N1D_CEIL).
     * RULE ZERO item 5 (largest measured effect first) favours (c), with (a) only if the PI wants the weight line resolved.
2. **Seeds s10 / s11:** the card asks for them only under RBW-ONE-SEED, RBW-GAIN-ONE-COST or RBW-SPLIT. None of these holds, so **s10 / s11 are not the card's committed next step here**. They are option 1(a) and need the PI's choice.
3. **N1DC go** (draft `F:/TanitAD-scratch/n1dc/PREREG_V8F_N1DC_DRAFT.md`, base N1D, about 1.5 LoRA pod-h ESTIMATED):
   * **This label does not make N1DC the committed next card.** N1DW's route to the probe-picked lever runs through RBW-NO-GAIN, which is not reached.
   * The draft's own conditional arm (N1DC at the carried weight) needs RBW-GAIN, which is not reached either.
   * N1DC does not depend on this label. It stands on its own go.

**For the MM (no PI decision):**
* DONE (section 3.3): N1DW - N1D on eval-141. The s0 time-gap cost belongs to the channel; the w4 s1 TAC-LAT cost belongs to the weight. At w = 2 the weight adds no separated cost over N1D on these cells at either seed, which bears on option 1(a).
* Change the view so the label slot is wired: use `verdict/n1dw_view_s1slot.py`, or an equivalent, in the next chain.
* Add the view label and the beside status to `n1dw_read141.py`'s `not_yet`.
* Look at K9's failure at every arm, the banked C0 included.

## Deliverable manifest (all under `F:/TanitAD-scratch/n1dw/read/verdict/`, dev box F:, staged, not in the repo)
| file | what |
|---|---|
| `MM_READING_N1DW_VERDICT.md` | this reading |
| `RESULT_N1DW_VERDICT.json` (4d74f91e) | labels, the rows they rest on, four families per arm, STRATEGIC, controls, input sha256s |
| `build_verdict.py` (1baf796f) | applies section 7 mechanically; asserts against the readers' flags |
| `n1dw_view_s1slot.py` (ab2fec25; docstring line reference fixed after the run, code unchanged) | wrapper: the arm in the S1 slot of the frozen strategic reader |
| `run_s1slot.sh` (f7624e6a), `s1slot.log` | the gated CPU rerun and its log |
| `N1DW2_STRATEGIC_S1slot.json` (c4cf59f2), `N1DW4_STRATEGIC_S1slot.json` (92f64a90), `N1R_STRATEGIC_S1slot.json` (b0d1c53b) | frozen-reader outputs with the label wired |
| `row_verdict.txt` | draft GOALS_AND_CLAIMS row H-V8F-NAVRB2-1 |
| `DELETIONS_VERDICT.log` | three view folders of copies, 117,876 + 117,876 + 26,284 KB, deleted after their runs |
