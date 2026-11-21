# PREREG_V8F_DSCALE_STAGEB (FROZEN 2026-11-19 by the MM after the coordinator's review, NOTHING RUN): does the union base survive stage B? UE - BL at two seeds

Status: drafted by a design agent for the MM, per coordinator direction D9. The coordinator reviewed it, and R1 to R5 are applied: UNION-TRADE (section 5 and the conditional arm in section 2), C2b inside the outcome logic (sections 5 and 6), the two-by-two with B and the B-pair floor draws (sections 4 and 4a), UNION-GAIN-ONE-COST (section 5), and the arm order (section 2). The MM froze the card before any stage-B number existed. Nothing trained, launched, staged, committed or pushed. The pod was read only. The cost go is the PI's. Every fact is in `FACTS_STAGEB.md` (ids F-xx below are its ids, e.g. B5, C5).
Evidence classes: M measured (artifact named), I inherited, E estimated, H hypothesis, U unverified.
Parents: `CARD_DATASCALE_TRAIN.md` (stage-B recipe, cell rule, headline cells), `CARD_DSCALE_MIX.md` + `AMENDMENT_UM_A2.md` (UM / UE seeds), `MM_READING_REPS1.md` (the stage-A two-seed read that triggers this card).

## 1. Question

At stage A the union (UE: 14,512 clips, 61,568 steps) costs TAC-LAT against the corpus-only baseline at equal steps (BL: 4,512 clips, 61,568 steps): UE_s0 - BL_s0 -0.0855 [-0.1216, -0.0492], UE_s1 - BL_s0 -0.0513 [-0.0808, -0.0246] (M, E2). The stage-A reading also shows BL has one seed only (E7). The deployed recipe finishes with stage B (3,000 steps of DAgger fine-tune on labelled windows). Does the union's TAC-LAT cost survive that finish, and does the union's gain elsewhere (STRATEGIC at stage A, `MM_READING_REPS1.md`) survive it, at two seeds, against a floor measured from same-arm seed pairs?

ONE lever: unique clips (14,512 against 4,512) at identical steps, identical stage-B recipe, identical code. UM - BL and UM - UE are read beside and decide nothing.

## 2. Arms, cost, order

All arms: LoRA pod, one job at a time under the GPU queue lock, RTX 4000 Ada. Costs are chain-log walls from this pod (FACTS section B); the cost go is the PI's.

| prio | arm | what | measured basis | pod-h (E) |
|---|---|---|---|---|
| 1 | **UE_s0.B** | stage B from the banked UE_s0.A (sha256 `2fd4b747`), `--seed 0 --dagger-seed 0`, argv = `U_s0.B.tmpl` with the arm's own init, bank and bank sha | union stage B: U_s1 2.11 h end to end (`block2.log:21-27`), U_s0 about 2.07 h (`block1.log:20-23`); bank4 6,603 s / 6,567 s | **2.11** |
| 2 | **BL_s0.B** | stage B from the banked BL_s0.A (`0b374ded`), `--seed 0 --dagger-seed 0`, argv = `B_s0.B.tmpl` | corpus stage B: 0.75 h (B_s0), 0.76 h (B_rep), 1.12 h (B_s1) (`block1.log:3-9`, `block2.log:7-13`) | **1.12** (0.75 fast) |
| 3 | **UE_s1.B** | stage B from the banked UE_s1.A (`f13df8ce`), `--seed 1 --dagger-seed 1`, argv = `U_s1.B.tmpl` | as prio 1 | **2.11** |
| 4 | **BL_s1.A** | stage A at seed 1: BL_s0.A argv with `--seed 1`, `--out .../BL_s1/A` | BL_s0.A 3.58 h (`block3.log:19-20`); UE 3.29 to 3.91 h | **3.58** |
| 5 | **BL_s1.B** | stage B from BL_s1.A, `--seed 1 --dagger-seed 1`, argv = `B_s1.B.tmpl` | as prio 2 | **1.12** |
| | **Deciding set** | prio 1-5 | | **10.04** (9.30 with the fast BL bank) |
| conditional (R1) | **UE_s0.A->BLB, UE_s1.A->BLB** | the union stage A finished by a **corpus-only stage B**: UE_s.A as init, BL_s.B's stage-B data flags (corpus cache, labels, join, `--exclude`, its own DAgger bank on the corpus windows), seed s. **Trigger: UNION-TRADE** (section 5) | as BL stage B | **about 1.1 each, 2.2 total**; asked of the PI now with its trigger |
| beside | UM_s0.B, UM_s1.B | stage B of the mixture arm from UM_s0.A / UM_s1.A (checkpoints on the pod and banked before the post check); union windows | as prio 1 | +4.22, not needed for the deciding read; **not asked** |

Notes on the table.
* Cost per arm: UE stage B **2.11 pod-h**, BL stage B **1.12 pod-h** (upper), BL_s1 stage A **3.58 pod-h**. The three UE / BL stage-A checkpoints at s0 and UE_s1 already exist (FACTS A5), so only BL_s1.A is a stage-A cost.
* Stage-B step cost is 3,000 steps, about 600 to 850 s of the walls above (B6, B8); the rest is the DAgger bank, which scales with windows (306,566 against 985,048; B10).
* Priority is an order, not a chain (R5). After prio 1-2 the seed-0 UE - BL read exists (labelled one-seed). Prio 3 gives F_UE and the two-seed UE.B - B.B read of section 4a after about 5.3 h; it does not depend on BL_s1.A. Prio 4-5 add the seed-1 baseline and F_BL.
* No new trainer flag, no new loss term, no code change. G-DVB therefore adds no entry; each arm still needs its own launch-gate PASS token bound to commit, argv sha256 and data-manifest sha256s (project rule, 2026-09-26). Tree: all arms run from `cdbff971`, the commit both stage-A gate tokens bind (MEASURED, section 9a item 6).
* Disk: the gate wants 60 GB free on `/root` (116 GB read by `df`, not authoritative for the volume; dd-probe before launch, memory note).

## 3. The deciding comparison

**UE - BL at stage B, two seeds, all four metric families.** Pair (s) = UE_s.B - BL_s.B for s = 0, 1, same 141 episodes and 2,399 windows as every DSCALE read, T1-kin for LONGITUDINAL, LATERAL, TACTICAL (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene), STRATEGIC at T0 first tick with the T1 tick-0 identity asserted. Fog windows are unscored (PI 2026-11-17). STRATEGIC is reported in full (PI 2026-11-16): S1 checkpoint position error at 30 / 50 / 80 m, S2 S_CKPT skill at 80 m under real / donor / zero nav, each with n.

Cells per family and the eight headline cells that decide the label are those of the parent card (E9): L1 op speed MAE, L2 op headway abs error; A1 op yaw-rate MAE, A2 op curvature MAE; T1 TAC-LAT acc, T2 TAC-LON acc; S1 position error at 80 m, S2 skill at 80 m. ADE and FDE are reported beside and decide nothing. Per-family, never pooled. A cell that cannot be computed is listed with its reason and n.

**Estimator.** Paired episode-cluster bootstrap, B 2,000, seed 0, 141 episodes (`taniteval.ci.paired_episode_cluster_bootstrap`). It answers "would another draw of EPISODES say this?" only. It does not answer "another TRAINING RUN" (that is the floor, section 4) and does not answer "another INFERENCE RUN" (the planner is deterministic, and all dumps are made in one environment at one thread count: pod CPU, 2 threads, as `MM_READING_REPS1.md`).

**Cell rule** (the parent card's, with the stage-B floor of section 4). At one seed, a headline cell is a **gain** if its interval is separated on the better side and abs(delta) is above the cell's floor F_c; a **cost** if separated on the worse side and abs(delta) above F_c. A family gain at a seed needs at least one headline gain and no headline cost in that family; a family cost needs at least one headline cost.

## 4. The replicate floor at stage B (same-arm seed pairs)

* F_UE,c = abs(UE_s1.B - UE_s0.B) on cell c; F_BL,c = abs(BL_s1.B - BL_s0.B) on cell c; F_B,c = the largest pairwise abs difference among B_s0.B, B_rep.B and B_s1.B (R3: more draws, already banked; it can only raise the bar). **F_c = max(F_UE,c, F_BL,c, F_B,c)**, per cell, at stage B. Both pairs are measured on the same windows and the same estimator (point difference of the cell value). This is the floor the stage-A reading lacked: there the floor was the B-set pair at 19,164 steps applied at 61,568 (E1; `MM_READING_REPS1.md`).
* It answers "would another TRAINING RUN say this?". One pair per arm is a weak floor, and the card says so. A seed-disagree outcome (section 5) adds the third seed pair.
* Beside, at stage A with the new BL_s1.A: F_BL,A = abs(BL_s1.A - BL_s0.A) per cell, and the stage-A UE_s1 - BL_s1 pair, which removes the mixed-seed contrast (E7).
* Replicate control for the rule itself: report how many of the stage-B cells read "separated" in UE_s1 - UE_s0 and in BL_s1 - BL_s0 (two arms, different seeds, zero lever moved). The v7-tiny rig read 4 of 42 (9.5 %) (E6, I). If these pairs separate more cells than about 10 %, the interval of section 3 is read as optimistic and every "separated" label is reported with that rate beside it.
* Not a floor: the same-seed re-run of one arm. On this device it is not bit-identical (E8) but is a different question (inference and device nondeterminism, not training-seed variance).

## 4a. Beside, no new arm (R3): the two-by-two of data and steps at stage B
With the B arms already banked (B_s0, B_rep, B_s1 at stage B: 4,512 clips at 19,164 stage-A steps), the stage-B arms of this card fill a two-by-two:
* **UE.B - B.B** = the union recipe (more clips, more steps) against the recipe of record, two seeds on both sides as soon as UE_s0.B and UE_s1.B exist;
* **BL.B - B.B** = steps alone (same clips, 61,568 against 19,164 stage-A steps);
* UE.B - BL.B = clips alone at equal steps (the deciding pair, section 3).
Four families, same estimator and windows, floor F_c. Worded "plus commit" wherever the B arms' tree differs from `cdbff971` (the read prints each arm's gate-bound commit). Beside only: it decides no label.

## 5. Three committed outcomes, each with its next lever

Let a **family** be LONGITUDINAL, LATERAL, TACTICAL or STRATEGIC. The label is computed from the per-seed family statuses of pair (0) and pair (1).

**Cells entering the logic (R2).** Family statuses are computed per evaluation cell:
* eval-141 always;
* C2b (section 6) if the PI approves it.

A family gain at both seeds **on either cell** counts as a gain, and a family cost at both seeds **on either cell** counts as a cost; each status prints the cell it came from.

Why both cells: BL trains at equal steps on 4,512 clips, about 13 passes over the distribution eval-141 is drawn from, so eval-141 alone cannot reward what the lever is for.

Without C2b, every label carries **"corpus-distribution cell only; says nothing about breadth"**. C2a never enters the logic: it is corpus-distribution.

**The five rows are mutually exclusive by construction**: ABOVE-FLOOR has no cost at either seed; TRADE and COST each have a both-seed cost, and differ on whether a both-seed gain exists; GAIN-ONE-COST has a both-seed gain with a one-seed cost; INSIDE-FLOOR is the rest. No precedence rule is needed.

| outcome | rule (committed before any stage-B number) | meaning | next lever, to be executed in the same run (RULE ZERO) |
|---|---|---|---|
| **UNION-ABOVE-FLOOR** | at least one family has a family gain at BOTH seeds, and no family has a family cost at either seed | the union beats the corpus-only baseline above the replicate floor at both seeds: **the union is the candidate base** | (a) the union base goes to the MM's integration decision (coordinator D7, base lineage `648d4ca3`); (b) cheapest next experiment: the parent card's committed lever, LoRA on DINOv3 on the U set plus the VLM-CoT pass over the 10k clips (its masked goal bits), starting from the union base; (c) UM stage B (section 2, beside) is read for whether a corpus-only tactical term adds to the union. |
| **UNION-TRADE** (R1) | some family has a family gain at BOTH seeds AND another family has a family cost at BOTH seeds (on any cell that enters the logic). An eval-141 cost with a C2b gain is this row, worded **"in-distribution cost, breadth gain"** | a gain and a cost that a rule cannot net (for example a lateral-decision cost and a route-skill gain): **the card does not decide the base**. It goes to the coordinator and the PI with the per-family table, both seeds, both cells | the conditional corpus-only finish (section 2, R1): UE_s0.A and UE_s1.A finished by BL's stage-B data, about 2.2 pod-h. It tests whether the cost sits in the stage-A representation (it stays) or in the distribution of the finish (it goes). Read as UE.A->BLB - BL.B on the same cells and floor. The cost's cell map per country, platform and tactical class is printed beside it |
| **UNION-COST** | at least one family has a family cost at BOTH seeds, and **no family has a family gain at both seeds** | the union costs above the floor at both seeds and gains nowhere at both seeds: **the candidate base is corpus-only (the BL / B-set recipe)** | the data-scale line moves from how much to **what is sampled**: (a) read the stage-B cost map per country and platform (the 10k is stratified over 25 countries, eval-141 is not; parent card section on country shift) and per tactical class, to name which clips carry the cost; (b) cheapest experiment: corpus + a selected subset of the 10k clips at the same steps (selection by the cost map, for example the strata whose geometry TAC-LAT labels the corpus under-represents), with its own card and G-DVB entry; (c) the sampler-weight arm of the parent card's DATA-HURTS row. |
| **UNION-GAIN-ONE-COST** (R4) | some family has a family gain at BOTH seeds, and some family has a family cost at EXACTLY ONE seed (none at both) | a gain above the floor with a cost that did not replicate: neither "inside the floor" nor a clean gain. **The base is not decided by the card** | that cost cell is read on C2b (if approved) and at a third seed pair, BL_s2 / UE_s2 (stage A + stage B, about 12.4 pod-h, PI go). The smaller base is not chosen by default |
| **UNION-INSIDE-FLOOR** | any other result: no family gain or cost at both seeds, or a gain or cost at one seed only (and no both-seed gain), or opposite signs across seeds | the union is not separated from the baseline by more than the replicate floor: **the candidate stays on the smaller base**, and data scale is judged on the second cell (section 6) | (a) read the second evaluation cell of section 6 for the three arms that exist (BL, UE, UM) at zero training cost; (b) if the two seeds disagree in sign (SEEDS-DISAGREE), add the third pair BL_s2 / UE_s2 (stage A + stage B, about 12.4 pod-h, E from the section 2 walls) rather than a third UM seed; (c) otherwise the lever moves to the encoder / adapter (LoRA) at the corpus base. |

Prediction (H, written so it can be wrong): TAC-LAT UE - BL at stage B lies in [-0.09, -0.02] at both seeds (the stage-A cost shrinks but remains), STRATEGIC shows a gain at both seeds (stage-A read, `RESULT_REPS1.json` `family_status`: UE_s0 - BL_s0 gains on S1 position error and S2 skill at 80 m; UE_s1 - BL_s0 gains on S2 skill at 80 m only), LONGITUDINAL and LATERAL show none. That prediction reads UNION-TRADE. It fails if the stage-B finish removes the TAC-LAT cost (UNION-INSIDE-FLOOR or UNION-ABOVE-FLOOR) or enlarges it beyond -0.09.

## 6. A second evaluation cell that can reward broader data (requirement 3)

**Why.** Eval-141 holds corpus-distribution episodes only. A union gain from breadth cannot show on it. A held-out slice of the pool would.

**What was measured (FACTS section C).**
1. **No encoded slice of the scale pool exists that UE or UM did not train on.** The pool has 133,929 eligible clips; 10,000 are encoded, and every one of them is in the union training list (set arithmetic on the pod: union 14,512 = B 4,512 + 10,000; 10k and union overlap 10,000; B and 10k overlap 0) (C1, C8). The same list is in both UE's and UM's argv (`--cache-dir /root/data/cache_union_dscale`); UM only zeroes the tactical row weight on the 10k rows and still trains every other term on them (C2).
2. **The NAVRBD held-out grid is not held out for UE or UM.** `eval_grid_x_feasible.json` (497 episodes, 5,736 windows) is held out from N1R / N1D only. Checked against the union list: **497 of 497 clips are in the union list and in the 10k, 0 in the B set** (C5). Using it for UE / UM would score training data.
3. **Other encoded caches on the pod add nothing**: `cache_pod` eval (eval-141, already used), `cache_pod_smoke` (1 clip each), `cache_x_dscale` (a view of the 10k) (C3). The dev box does not hold the 10k cache (C4). Box 2 and a full dev-box archive search were not done here (UNVERIFIED, FACTS F2).
4. **One encoded, untrained slice exists, and it is not broad:** the **60 RA-1 holdout clips**, in `cache_pod/train`, absent from the union by id, absent from eval-141, with labels in the corpus join (60 clips, 6,056 steps, 5,996 labelled; C6). They are corpus-distribution clips, so they add n for the same distribution as eval-141 and cannot reward breadth. About 1,020 eval windows by the eval-141 window rate (E, C7); the grid, eval join and nav-text for them are not built.

**The card's decision.**
* Cell **C2a** (optional, zero encode cost): the 60 RA-1 clips, four families with n, all arms of this card plus UM. It buys extra corpus-distribution n and a check that the eval-141 result is not an artifact of its 141 episodes. It is reported as corpus-distribution and never as the breadth cell. It needs a grid, eval join and nav-text build on the pod (CPU, cost U).
* Cell **C2b** (needed for breadth; does not exist): held-out pool clips encoded for this purpose. **Separate cost line for the PI:**
  * Slice: 500 clips = 20 per country across the 25 countries of the 10k stratification, drawn from the 123,929 eligible pool clips that are not encoded (E, C8), after the census exclusions (our train, eval-141, val40, parity digests, challenging and OOD sets). Disjoint from the union list by id and by stable sid (C-DISJ); any hit is dropped.
  * Encode: the 10k pipeline ran at 1,364 clips/h in the last hour and 1,279 clips/h end to end (M, D1); the pilot split is stage 3,856, build 1,816 (13 workers) and encode 1,354 clips/h (D2). **500 clips: about 0.37 pod-GPU-h of encoding, about 0.39 h end-to-end wall (E, D5).** It uses the pod GPU, so it is queued under the GPU lock between arms and needs the PI's go.
  * Labels, join, tracks and nav-text for the slice: the GEO10K build exists and is CPU (D3); its per-clip time for 500 clips was not measured (U); budget under 1 h CPU wall by analogy to the pilot's per-clip build (E, weak).
  * Eval dumps: 6 arms (BL s0 s1, UE s0 s1, UM s0 s1) at about 8,500 windows each, CPU only: about 6 min per arm at the pod rate, about 20 min per arm at the dev-box rate (E, D6). Zero GPU.
  * Total new cost for C2b: about 0.4 pod-GPU-h plus under 1 h CPU build, plus dumps. Two sizes for the PI: 500 clips as above, or 250 clips at about half (n of the cell shrinks, floor widens).
* **Reading rule for C2b** (committed): the four families on C2b with n windows and episodes, paired episode-cluster bootstrap, B 2,000, seed 0, UE - BL and UM - BL at stage A today (arms exist) and at stage B when the arms of section 2 exist. A union gain above the section-4 floor on C2b and no gain on eval-141 reads "the union's benefit is breadth"; the reverse reads "no breadth benefit at this scale". **When the PI approves C2b, its stage-B family statuses enter the outcome logic of section 5 (R2)**, so a both-seed family gain on C2b counts as a gain. The stage-A C2b read stays beside.
* The estimator on C2b answers "another draw of EPISODES" drawn from the pool stratification. It does not carry training-run variance either.

## 7. Beside, zero training: BL against UE at T0 on eval-141 (requirement 4)

Purpose: separate a worse representation from rollout drift. Tier T0 is a world-model diagnostic and never driving performance.

**Already measured at stage A** (M, `RESULT_REPS1.json` `ade6_stage_A_T0`, `mechanism_stage_A_T0`; FACTS E3-E5):

| quantity (T0, eval-141) | BL_s0 | UE_s0 | UE_s1 | reading |
|---|---|---|---|---|
| policy-rolled ADE6 (m) | 2.0629 | 1.9671 | 2.0376 | UE_s0 - BL_s0 -0.0958 [-0.183, -0.0053], separated, below 2F_A = 0.1656 |
| GT-oracle ADE6 (m) | 0.7712 | 0.8281 | 0.8327 | UE - BL +0.0569 and +0.0615, both separated, UE worse; UE1 - UE0 +0.0046 not separated |
| T's latcons RMSE | 0.6822 | 0.6864 | 0.6851 | r = +0.6 % and +0.4 %, inside the stage-A T floor (abs_r 0.0308) |

**What it says (H, a reading, not a label):** UE's T1-kin TAC-LAT cost sits beside a UE policy-rolled ADE6 that is better and a GT-oracle ADE6 that is worse by about 0.06 m with a latcons RMSE that is flat. On those three numbers the representation gap is small on T's latent and visible on the GT-oracle metric. The GT-oracle definition was not re-read here (FACTS F4), so this stays a pointer until the run below.

**The run (no GPU, CPU dumps on the pod, existing checkpoints):**
1. Re-dump BL_s0, UE_s0, UE_s1 at stage A in one environment (pod CPU, 2 threads) and read the T0 set: ADE6, GT-oracle and absmask evals, T's latcons RMSE with the paired interval, goal macro-F1 split into VLM-class and geometry-class bits. About 2 min per arm (B9).
2. Add UE_s1 - BL_s1 once BL_s1.A exists (prio 3 of section 2 produces the checkpoint at no extra cost).
3. Repeat the same T0 set on the stage-B checkpoints after section 2.
4. Committed reading, per seed:
   * **REPRESENTATION-WORSE** if the GT-oracle ADE6 difference UE - BL is separated worse and above max(abs(UE_s1 - UE_s0), abs(BL_s1 - BL_s0)) on the same metric, or T's latcons r is at least +5 % and separated;
   * **ROLLOUT-DRIFT** if the T0 metrics are inside those floors or favour UE while the T1-kin TAC-LAT is a cost;
   * the label is printed per seed and the two are never pooled.
   REPRESENTATION-WORSE feeds the UNION-COST next lever (the encoder or the labels, not the sampler); ROLLOUT-DRIFT feeds it toward a stage-B / DAgger lever on the union.

## 8. Gate, controls and deliberate-regression arms

The launch gate (`stack/scripts/launch_gate.py`) must PASS for each arm. Each control states the value it must read.

| control | what it checks | must read |
|---|---|---|
| **G-PAIR** (argv pairing, CPU) | UE_s1.B vs UE_s0.B differ in exactly `--seed`, `--dagger-seed`, `--init-ckpt`, `--dagger-bank`, `--dagger-bank-sha256`, `--out`; BL_s1.B vs BL_s0.B likewise; UE_sX.B vs BL_sX.B differ only in the seven data flags, `--exclude` (BL only) and `--fit-join` (UE only); BL_s1.A vs BL_s0.A differ in exactly `--seed` and `--out`. | 0 other differences. **Regression arms:** (i) a changed `--lr` on UE_s1.B, (ii) a swapped `--dagger-seed`, (iii) `--exclude` added to UE, (iv) `--steps 2999`, (v) a BL argv pointing at `cache_union_dscale`, (vi) an init checkpoint of the other seed: each must be refused by the same function (as `PAIRING_PROOF.json`, 8 of 8 at UM-A2). |
| **G-INIT-CKPT** | each stage-B argv's `--init-ckpt` sha256 equals the banked stage-A sha: UE_s0 `2fd4b747`, UE_s1 `f13df8ce`, BL_s0 `0b374ded`, BL_s1 the sha written by its own stage A. | equal; a tampered copy (one byte flipped) is refused |
| **G-REACH** (zero-grad gate, post-training) | every registered parameter of a stage-B arm has Adam state and the 18,472-parameter class of untrained heads (`offset_head`, `tac_goal_tok_head`, `scorer.goal_point`) is absent from the trained set the same way in UE and BL; the parameter list equals the stage-A arm's. | 0 parameters without state beyond the stage-A list; **control** a parameter group built with `requires_grad` off must read exactly 0.0 gradient |
| **G-LIVE** | every declared loss term appears with a finite value in the stage-B smoke on real windows, DAgger swap rows present | all terms finite; the DAgger bank coverage at least 0.5 (parent card) or the arm is BLOCKED, never dropped |
| **C-DISJ** | the union list and the B list are disjoint from eval-141 by id and by stable sid; the C2b slice, once drawn, is disjoint from the union list | 0 and 0 and 0. **Control:** plant one union clip id in the C2b manifest; the check must refuse. |
| **C-HOLD known value** | the NAVRBD grid `eval_grid_x_feasible.json` checked against the union list | **497 of 497 in the union list**, so it is refused as a held-out cell for UE and UM (FACTS C5). A reading of 0 means the probe is broken. |
| **K-ENV** (read side) | the stage-A cells of BL_s0 / UE_s0 / UE_s1 re-dumped in the stage-B read environment reproduce the banked rows | the 238-cell check of `MM_READING_REPS1.md`: max drift at most 0.1 x F_cell, 0 separated-flag changes |
| **Self-pair** (read side) | the reader run on UE_s0.B - UE_s0.B | **exactly 0.0 on every cell, interval [0, 0], 0 separated** |
| **Swap control** (read side) | UE - BL and BL - UE through the same reader | delta equals minus delta to 1e-12; separated flags equal; the family gain and cost labels exchange |
| **Floor non-zero control** | F_UE,c and F_BL,c on the stage-B pairs | not all exactly 0.0: a floor that reads 0.0 on every cell means the two seeds produced one run and the floor is read as UNMEASURED |
| **Reach of the lever** | R_s0 (10k labels deranged, stage A, exists) vs UE_s0 on TACTICAL | already measured at stage A; this card adds no new regression arm at stage B because the lever it moves (the clip set) does not change |
| **Gradient-through-the-flag** | no new flag is added by this card | a flag whose default disables it is named in the argv; the checker's refusal of any argv key outside the pairing list |

DELIBERATE-REGRESSION ARM at the read: feed the reader a pair where UE_s0.B's TAC-LAT cell is shifted by 2 x F_c (planted). The label must move from UNION-INSIDE-FLOOR to the planted outcome, and the shifted cell must read separated and above F_c. A planted shift of 0.5 x F_c must NOT clear the floor.

## 9. Open decisions for the MM (not decided by this draft)

1. **Spend:** 10.04 pod-h (deciding set), or 14.26 with UM beside. Cost go is the PI's.
2. **Second cell:** C2a only (free, not breadth), or C2a + C2b (about 0.4 pod-GPU-h plus CPU), or neither.
3. **BL_s1.A first or last:** this draft puts the seed-0 pair first (largest decision value per hour); BL_s1.A first would give the stage-A UE_s1 - BL_s1 pair after 3.6 h.
4. **Outcome table precedence:** UNION-COST is defined to override gains in other families at both seeds, and it names the trade-off. The MM may prefer a fourth label for a trade-off.
5. **Coordinator D7:** whether the union base is carried into integration at all is decided there; this card answers only whether it earns that.
6. **Tree and commit for UE / BL stage B:** `cdbff971` is the proposed single tree; the BL stage-A commit field was not read (UNVERIFIED).

## 9a. MM rulings on section 9 (2026-11-19, before any stage-B number)
1. **Spend: three lines go to the PI.** (i) The deciding set, prio 1-5, about 10.0 pod-h (E). (ii) C2b, about 0.4 pod-GPU-h plus CPU. (iii) The conditional corpus-only finish of R1, about 2.2 pod-h, triggered by UNION-TRADE only. UM stage B is not asked: UM - UE is not a quotable lever effect (DILUTION-ONE-SEED), and both UM arms read BLOCKED (reach).
2. **Second cell.** C2a (60 RA-1 clips; CPU build only) goes in, labelled corpus-distribution. C2b (500 held-out pool clips, about 0.4 pod-GPU-h of encoding plus CPU) is asked of the PI as its own cost line. It is the only cell that can reward breadth.
3. **Order (R5).** UE_s0.B, BL_s0.B, UE_s1.B, BL_s1.A, BL_s1.B.
4. **Fourth label (coordinator R1, accepted; my earlier "no fourth label" ruling is replaced).** UNION-TRADE is its own row. Under the draft's precedence, a TAC-LAT cost with a STRATEGIC gain would have resolved to "base = corpus-only" by rule order alone, and a lateral-decision cost and a route-skill gain cannot be netted by a rule. R4 adds UNION-GAIN-ONE-COST for the same reason.
5. **Queue.** On the LoRA pod, after the four N1DW arms (PI-approved, in build). Where it sits against TanitADRuled's GPU stage (PI: after N1DW) is the PI's call.
6. **Tree and commit: MEASURED, one commit.** The launch-gate tokens of both stage-A arms bind the same commit: `/root/dscale10k/gate/tokens/BL_s0.A/gate_token.json` and `.../UE_s0.A/gate_token.json` both read `"commit": "cdbff971fa6dad4d20b937fe8abda6cb9f622adb"`. All stage-B arms and BL_s1.A therefore run from tree `cdbff971`, and the UE - BL contrast has no commit difference. G-INIT-CKPT and C-INIT still decide at launch.
7. **Absence probe.** The "no held-out encoded slice" finding is MEASURED on the pod manifests and the dev box. Box 2 and the full archive were not searched (U). The C2b line makes the answer moot for the breadth cell, so it is not re-probed before freeze.

## 10. Register row (proposal)

`H-V8F-DSCALE-STAGEB-1` (2026-11-19, PRE-REGISTERED): the union's stage-A TAC-LAT cost (UE - BL, two seeds, T1-kin, M) survives stage B. Claim read by the outcome table of section 5 on four families at two seeds against the stage-B replicate floor. Status: PRE-REGISTERED (DRAFT), NOT RUN. Next lever by outcome: section 5. It never ends the stream.
