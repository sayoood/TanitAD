# PREREG_V8F_NAVRBD (DRAFT 2026-11-18, NOT FROZEN, nothing built, nothing trained): exit number and countdown distance as roundabout channels (RB-DIST)

Status: drafted by a design agent for the Master Mind (MM), who rules on section 0 and freezes it (sha256 file) before any arm is minted. No arm is minted, launched or trained by this card's author. Evidence classes: MEASURED (M, artifact named), PUBLISHED (P), INHERITED (I), ESTIMATED (E), HYPOTHESIS (H), UNVERIFIED (U).
Artifacts: `FEASIBILITY_NAVRBD.md` (facts, counts, file:line), `BUILD_PLAN_NAVRBD.md` (files, gates, chain), `out/RB10K_COUNTS.json`.

## 0. For the MM before freeze

1. **Code base `4dedac45` + navrb2.patch (= `e71782a3`)**, the code the N1R arms trained from. The new lever is added on top of it; the pairing arms N1R_s0 / N1R_s1 are banked on private HF `datasets/Sayood/tanitad-v8f-strat1-runs` (rev 48098c45, MM_READING_NAVRB2).
2. **Order of operations (binding on the card):** (a) the HO floor `F_HO` is computed from the banked N1R pair on the held-out cell and its sha256 is written into this card BEFORE any N1D checkpoint exists; (b) the build and gate; (c) the N1D arms. A floor read after an N1D number is not admissible.
3. **PI go needed:** pod time for 3 arms (about 0.75 pod-h, section 6) and the held-out T1-kin dumps on the pod (CPU, about 1.5 h, E).
4. **Where the lever's width comes from:** the ED feature set of RB-JOINT (state + exit number + countdown distance), the one set that decoded the roundabout lateral decision above the state alone (+0.0581 [+0.0287, +0.0857], 5-fold episode-grouped CV over 12,201 strided windows / 426 episodes, T0 decodability, MM_READING_RBJOINT). Vision alone did not reach the 0.0383 bar there.

## 1. Question

NAVRB2 showed the model reads a one-column roundabout state (Z-USE separated at both seeds, both pairs) and that TAC-LAT on the 235 roundabout windows of eval-141 did not move beyond the floor (F2_rb 0.0383; RB-READ-NO-GAIN). RB-DIST-ZT could not see a text effect on 16 episodes; RB-JOINT, powered on 426 train episodes, found one (ED over S +0.0581). Does a trained N1-recipe arm that is also fed the **exit number and the countdown distance** decide the roundabout lateral manoeuvre better than N1R, on a held-out set large enough to see an effect of that size?

## 2. Channel (the variant)

* New lever `--nav-rb-feat state|exitdist` (default `state` = N1R bit for bit). With `exitdist`, `nav_rb` is a **15-column** vector per cache step, `e_nav_rb = Linear(15, d, bias=False)`, zero-initialised, added to the nav token exactly where N1R adds its 1-column version (S, T, O).
* Columns, all from route-text table rows at or before the step (the ED set of RB-JOINT, binned):
  * c0 `state` = N1R's channel (1 iff flag 0 and a roundabout class at the NOW row).
  * c1..c6 `exit_cur` one-hot over {1, 2, 3, 4, 5, 6+} from the text at the NOW row (all zero when the text has no exit number).
  * c7..c12 `exit_last` one-hot over the same bins from the latest row at or before NOW with flag 0 and an exit number (all zero if none).
  * c13 `dist` = router countdown at the row, **dead-reckoned** like channel 3: `max(0, dist_text - arc travelled since the row's time) / 100`, clipped to [0, 3]; 0 when absent.
  * c14 `dist_valid` = 1 iff the router text carries a distance.
* All columns are 0 whenever c0 is 0 (the channel is silent off roundabouts).
* Train: dead-reckoned from the logged track (the `clip_nav` rule). Eval, T1: columns c0..c12 and c14 held from t0; **c13 reduced by the own rollout arc** per tick (the pinned tool does this for channel 3 at `v8f_t1_eval.py:215-219`; the study wrapper does it for c13). DAgger swap: held as logged (`nav_rb` has no swap rule; the logged and own arcs differ by the DAgger deviation only; named risk, section 9).
* Why 15 columns and not a token in the existing vocabulary: same parity argument as NAVRB2 section 2 (a fourth token value has no slot; a separate zero-initialised module keeps the step-0 model identical to the flag-off model).

## 3. Admissibility (binding)

* **Computed from the route input only, at training and inference.** Reads `cls_id`, `flag`, `dist_m`, `text_i`, `age_s` of the navtok table and the parsed texts (`exit_n`) and the own arc for the countdown (the same quantity channel 3 already uses). No situation module (AST guard in the test, as NAVRB2), no join label, no v9 nav, no ego state beyond the arc that N1R's channel 3 already reads. **Nav is an input at training and inference.**
* **Not the output of the situation classifier**, in any form: the exit number and the distance are parsed text; the classifier plays no part.
* **No future row:** replacing table rows after the step by garbage leaves c0..c14 unchanged (test); a mutant reading row `j + 5` goes RED. The exit-last rule reads backward only.
* **Optimistic by construction (stated):** on PhysicalAI the route text is GPS-matched to OSM routing along the drive's own trace (I, NAVTOK_REPORT). Every number is worded "with the route supplied along the driven path". The held-out cell inherits the same property; it measures whether a trained arm uses the supplied text, not whether it plans a route.
* Label use: TAC-LAT labels (`lat_cls`) are built offline and may use ego; inference is vision plus the route input.

## 4. Arms (LoRA pod, `flock /root/gpu_queue.lock`, one job at a time; the N1 recipe, 3,000 steps, fine-tune from M2b)

| priority | arm | = | paired with |
|---|---|---|---|
| 1 | **N1D_s0** | N1R_s0 argv + `--nav-rb-feat exitdist --nav-rb-npz <navtext_nav_rbd.npz> --nav-rb-sha256 <s>` (variant real, lr-mult 10) | N1R_s0 (banked) |
| 2 | **N1D_s1** | N1R_s1 argv + the same | N1R_s1 (banked) |
| 3 | **N1Dx_s0** (deliberate regression) | N1D_s0 argv with `--nav-rb-variant shuffle`: the 15-column rows are **deranged across clips** in train (a clip reads another clip's channel rows, resampled to its length), eval variant likewise | N1R_s0 |

* Every arm names the new flag, defaults included. N1R's argv sha256 is unchanged (the checker's N1R render is not touched).
* Conditional (each needs the PI's go on cost, runs only on its trigger): S1D at s0 / s1 (4 arms, E4 of NAVRB2 style) if D1-HO is a gain; N1D at s10 / s11 (2 arms) if the two seeds disagree.

## 5. Read

**Tiers.** D1-HO and the eval-141 cells: **T1-kin** (self-action open loop, m = 10, causal bucket, dead-reckoned nav with `nav_rb` held from t0 and c13 dead-reckoned), PRIMARY. All arms of the held-out comparison dumped in one environment (the pod, CPU, one thread count). STRATEGIC: T0 at the window's first tick as STRAT1.
**Estimator.** `taniteval.ci.paired_episode_cluster_bootstrap`, B 2,000, seed 0, over the episodes of the stratum. It answers "another draw of EPISODES" only; "another TRAINING RUN" is the floor (5.3); "another INFERENCE RUN" does not apply (deterministic planner, one environment).

### 5.1 Deciding cells
* **D1-HO (deciding, powered):** TAC-LAT accuracy (the 24-cell reader's per-window TAC-LAT indicator, restricted to roundabout windows) on the **held-out 10k RB grid windows: 7,594 windows / 545 episodes** (M, `out/RB10K_COUNTS.json`; grid = stride 4, t = 3 mod 4, state at NOW, label present). The 10k clips are disjoint from train4572 and eval-141 (0 overlap, M), so no N1R or N1D arm has seen them. Delta = N1D - N1R, paired, per seed. Labels from the 10k geometry release (`lat_cls`, three classes populated: LANE_KEEP, TURN_L, TURN_R).
* **D1 (beside, as NAVRB2):** TAC-LAT on eval-141 RB windows (235 / 16 episodes; 202 scored by rbdist). Reported with its interval and no bar; it cannot resolve an effect of 0.058 (intervals about 0.3 wide, MM_READING_RBJOINT).
* **D2-HO (beside):** TURN recall and side-given-turn accuracy on the held-out RB TURN windows (12,024 t>=3 windows / 344 episodes over all steps; the grid subset printed), and the false-turn share on non-turn RB windows.
* **Power (E):** RB-JOINT's ED - S interval half-width was 0.0285 on 426 episodes; scaled to 545 episodes about 0.025 (scaling by the root of the episode ratio; U for two trained arms, whose pairing is noisier). The bar 0.0383 sits above that half-width, so a true effect of about 0.058 is resolved on this cell; a true effect near 0.0383 is not guaranteed to be.

### 5.2 Guard: the four families on eval-141, no cost
* The STRAT1 24 cells per seed (LONGITUDINAL, LATERAL, TACTICAL, ADE / FDE) for N1D - N1R, STRAT1 cost / gain rule, floor F2.
* **STRATEGIC in full** (unpaused): route setting (checkpoint position / heading error at 30 / 50 / 80 m vs RC-A, 6 cells), route following S_CKPT at 80 m under real, donor and zero nav, detected-information cells as in the N1R read (no S head in N1, so the cells print the N1 reasons). **Fog is unscored** (PI ruling 2026-11-17, `whead/nofog/AMEND_WH_NOFOG.md`); weather is scored on clear / rain / snow; fog n windows / episodes printed.
* Each cell also prints N1D - N1R.

### 5.3 Floor (fixed before any N1D number)
* **F_HO = |N1R_s1 - N1R_s0|** on D1-HO, the same windows, from the banked checkpoints dumped in the pod environment. Written to `HO_FLOOR.json` with its sha256 and recorded in this card before N1D trains.
* Eval-141 cells: F2 = max(F, F_rep) as NAVRB2 section 5.3.
* **GAIN on D1-HO at a seed:** N1D - N1R >= max(0.0383, F_HO) and the paired CI lower bound above 0. 0.0383 is the NAVRB2 rig floor on roundabout windows; F_HO is this cell's own training-run noise.

## 6. Controls (each with the value it must read)

| id | control | must read |
|---|---|---|
| Z0 | `--nav-rb-feat state` build vs `e71782a3` (every state-dict key, value, loss term, output) | bit-identical |
| Z1 | exitdist build at step 0 vs state build / flag-off, same init and batch | outputs, loss terms and RNG stream bit-identical |
| Z2 | gradient on `e_nav_rb.weight` **per column group** on a real batch (c0, c1..c6, c7..c12, c13, c14), real channel / channel zero | > 0 on every group that has a 1 or a nonzero value in the batch (the batch's own counts printed) / exactly 0.0 |
| Z-EVAL | each N1D checkpoint dumped with the channel zeroed (zero-training) on windows whose channel rows are all 0 | exactly 0.0 on every cell |
| Z-USE | real - zero on RB windows (held-out and eval-141) | reported; the model reads the channel iff separated |
| K1 | reader self-pair (N1R_s0 vs N1R_s0) on D1-HO | exactly 0.0 |
| K-CONST | the majority-class predictor on the held-out RB windows | accuracy equals the majority share exactly |
| K-RE | recomputed banked pairs (N1R - N1 at s0 / s1) equal `RESULT_NAVRB2.json` | exact |
| G-CLOCK-RBD | table builder: per-step columns equal an independent recompute; age >= 0 on every row used; **regression arm (rows read 5 steps = 1 s ahead) must FAIL** on both splits; the 10k table reproduces the counts 7,594 / 545 at the grid | PASS; regression FAILS |
| **N1Dx_s0** | channels deranged across clips | D1-HO N1Dx - N1R < max(0.0383, F_HO), i.e. not a gain; if it is a gain the arm's rule, not the channel, produced it and no N1D label stands |
| K-MUT | delete the `e_nav_rb` term; delete the join line; delete the exit-last read; delete the c13 dead-reckoning; read row `j + 5` | each named test RED |
| G-GEN | the evaluator wrapper with a 15-column checkpoint and the flag dropped from its cfg | REFUSED |

## 7. Outcomes, committed before any number (each names its next lever, RULE ZERO)

Read per seed pair (N1D - N1R at s0 and at s1); "guard" = no cost on any of the 24 eval-141 cells at both seeds and no ROUTE-COST.

| outcome | condition | next |
|---|---|---|
| **RBD-GAIN** | D1-HO a gain at BOTH seeds (section 5.3), N1Dx not a gain, Z-USE separated, guard holds | the channel enters the recipe. Seeds s10 / s11 (PI go) and **S1D**: the route-text arm with the exit + distance channel (4 arms); then the vision-geometry lever on the remaining RB gap vs the v9 recall |
| **RBD-ONE-SEED** | gain at exactly one seed | N1D at s10 / s11 (2 arms), COST4 / GAIN4 under F2 |
| **RBD-NO-GAIN** | Z-USE separated at both seeds, D1-HO not a gain at either | the **stacking-combiner check** (CPU, zero-training): the RB-JOINT vision probe's out-of-fold probabilities stacked into the ED tree, on the same CV and on the held-out windows, to test whether vision and text together carry more than either (the caveat 1 of MM_READING_RBJOINT); then the **vision lever**: an RB window weight in the training loss and/or the 8x20 grid input (RAW8 - S was +0.0266 separated, below the bar) |
| **RBD-UNREAD** | Z-USE not separated at either seed | dose: `--nav-rb-lr-mult` 30 at s0 (PI go); first the Adam-state reachability read on `e_nav_rb.weight`, per column group |
| **RBD-COSTS** | a guard cell a cost at both seeds, or D1-HO a cost at both seeds | not adopted; N1 replicate with the new code (the zeroed-channel arm, 1 arm) to separate the channel from training noise |
| **RBD-FLOOR-HIGH** | F_HO above 0.0383 and above every observed N1D - N1R | the cell cannot resolve this lever at these seeds: more seeds of the pair (PI go), never a lower bar |

* Both pairs are printed; if two labels hold, both are printed. Bars, controls, arms and outcomes do not change after any number is seen. A bar that fails is reported as written, then the triggered arm is built (RULE ZERO).

## 8. Why a replicate of the question is built into the cell
The paired bootstrap on D1-HO answers "another draw of episodes" and is blind to training variance; the floor F_HO and the two-seed rule carry the training-run question. The held-out set removes the thinness of eval-141 (16 episodes) but not the single-seed limit: a one-seed gain is RBD-ONE-SEED, never a gain.

## 9. Known risks
* 3,000 steps at lr 1e-4 (x10 on the new group) may under-train a 15-column zero-initialised weight (RBD-UNREAD covers it).
* **Driving side:** the exit an exit number names depends on left- or right-hand traffic, which the route text does not carry; the 10k spans 25 countries. If a text channel cannot resolve it, the effect is capped below RB-JOINT's CV figure, which was fit on a mix of the same kind.
* The RB-JOINT result is a gradient-boosted tree on these features (T0 decodability). A neural nav token may not use them the way a tree does (VED8 - RAW8 read -0.0031, the head ignored the concatenated features). A zero effect here says the trained path, not the text, failed.
* Dead-reckoned c13 in DAgger swap rows is held as logged (arc differences only).
* The D1-HO windows are RB windows by the logged path's route text; N1R and N1D read the same table, so the contrast is clean, the absolute level is optimistic.
* Held-out label classes: three populated (as train).
* Pod preconditions (the 10k cache view, join_X, tracks) are UNVERIFIED until the dump's own checks run.

## Log
* 2026-11-18, design agent: drafted before any N1D artifact exists. Counts and controls of section 5 are the measured ones in `out/RB10K_COUNTS.json`.
