# PREREG_V8F_STRAT1 (FROZEN 2026-11-16 by the MM, NOT RUN): the strategic layer into v8f-mini, with route text in and detected strategic information out

Status: drafted by the Architecture agent, FROZEN by the MM on 2026-11-16 (about 12:15 local) before any arm trained and before any window was scored. Build of record: branch `agent/v8f-strat1-20261116` = `b9250158` (base `cdbff971`, 10 files; G-SUITE on the pod pending). Checker additions after `b9250158` (token minting, runner) may not change any trainer, model, data or evaluator path; the token binds the exact commit. The launch needs a `strat1-gate-token-v1` PASS (`BUILD_SPEC.md` section 8). PI: "Build the arm" (2026-11-16); retrain authorised 2026-11-08 (I, register row `PI: horizon bands`). STRATEGIC is unpaused (PI 2026-11-16, I, register row `PI RULINGS, 2026-11-16`).
Evidence classes: MEASURED (M, artifact named), PUBLISHED (P), INHERITED (I), ESTIMATED (E), HYPOTHESIS (H), UNVERIFIED (U).

## 0. Decisions for the MM / PI (escalated)

1. **Checkpoint loss: no change to its target or weight** (section 3.3). S's route is made consistent with the route text through its INPUT; a text-side loss on S would train on the deciding STRATEGIC cell's own rule. The 6-30 s band (a 120 m checkpoint) is the next card and an architecture change, so it needs a PI go.
2. **Base commit `cdbff971`** (OBD-CTL's code of record), not `81cee6c1` (`BUILD_SPEC.md` section 0).
3. **Nav distance counts down** from logged odometry between text rows. This differs from `PREREG_NAVTEXT_DRAFT` N1, which used the raw text distance: row age median 2.09 s, p90 15.45 s (M, `navtok_table.npz`).
4. **Gate route**: a study checker (`strat1_check.py`) around the trainer's selfcheck, mapped to all seven G-checks, as every v8f-mini arm so far; `launch_gate.py` has no v8f-mini profile (M).
5. **Pod time**: about 3.8 h (core) to 5.3 h (all seven arms) of the LoRA pod (E, section 7). The pod also carries the 10k-corpus work. The PI's go on pod time is needed if "Build the arm" does not cover it.
6. **Cost basis**: the brief's 0.48-0.50 s/step is the WM-only rate (ACTSENS). A policy fine-tune on box 2 runs at 0.722-0.769 s/step (M, section 7).

**MM rulings on section 0 (2026-11-16, at freeze):** (1) checkpoint loss unchanged: accepted. (2) base `cdbff971`: accepted. (3) countdown distance: accepted; G-CLOCK-NAV passed on real data (train 6.481 vs 6.963 m, eval 5.347 vs 5.716 m) and its `+ travelled` regression arm failed as required (M, `F:/TanitAD-scratch/strat1/package/BUILD_REPORT.md`). (4) gate route: a study checker that mints `strat1-gate-token-v1` bound to commit, argv sha256 and every data sha256, with a runner that refuses without a matching token, meets the PI rule of 2026-09-26 in the same way as the OBD and ACTSENS tokens; accepted. (5) pod time: the PI's "Build the arm" (2026-11-16) is read as covering the 7 arms within the section 7 stop (a projected chain above 10 h goes back to the PI). (6) cost basis: the pilot's measured s/step replaces the estimate.

## 1. Question

Does v8f-mini, fine-tuned on the recipe of record with (a) NVIDIA's route text as its nav input and (b) vision-only heads that detect country, weather, road surface, illumination and period of day, keep its driving, make S's route follow the announced route, and detect the strategic context better than a constant?

## 2. Admissibility

* **Goal-input rule (CLAUDE.md 2026-08-03).** The detected-info outputs feed neither the planner nor any goal path: `ctx_logits` goes to the loss and the eval only, pinned by test T-ISO (perturbing the head leaves every planner output bit-identical). They share the adapter with S, T and O during training (a declared coupling, measured by arm N1). Goals derived from detected info are a later card.
* **The route text is admissible as a goal/route input.** It is GPS matched to OSM routing (I, NAVTOK_REPORT headline 1) and carries no situation-classifier output. It is computed along the drive's own trace, so it is **optimistic by construction**: every STRATEGIC number is worded "with the route supplied along the driven path".
* **Vision-only inference for the heads (CLAUDE.md 2026-08-03).** The heads read `feats` and `frame_ok` only (test T-VIS: AST source guard, key-guarded read set, bit-identity under ego/nav/limit changes). Labels use metadata (country and clock hour) and VLM/probe outputs, which is allowed for labels.
* **Nav is an input at training and inference** (PI). The control's nav is the v9 ego-future token (`allow_oracle_nav (ego-future)`, I, NAVTOK_REPORT section 1). It shares its source with the RC-A labels and the lateral labels, so the control's lateral, tactical and route rows carry an echo advantage. Section 5 controls for it.
* Parity: corpus `train4572` with `--exclude 540:600` (4,512 clips, `train_sha256` `37f4f9ee...`, M, `summary.json`), eval-141, unchanged. The 60 excluded clips are exactly the 60 the label release lacks (M, LABEL_SPEC section 4).

## 3. Arms (LoRA pod, one job at a time, 3,000 steps each, fine-tune from M2b)

Recipe of record = OBD-CTL: DAG-OACC (3,000 steps, lr 1e-4, warmup 100, p_gt 0.2, DAgger p 0.1, `--w-acc 49.6969`) + `--detach-upper` (M, `obd_OBD-CTL_s0/argv.txt`; I, register `H-V8F-OBD-1`). Argvs: `BUILD_SPEC.md` section 10.

| priority | arm | nav | ctx heads | init / DAgger bank / seed |
|---|---|---|---|---|
| 1 | **C0_s0** | v9 | off | M2b_s0 / s0 / 0 |
| 2 | **S1_s0** | text (countdown) | on, `w_ctx 0.5`, head lr x10 | M2b_s0 / s0 / 0 |
| 3 | **C0_rep** | v9 | off | M2b_s0 / s0 / **2** (the training-run floor) |
| 4 | **C0_s1** | v9 | off | M2b_s1 / s1 / 1 |
| 5 | **S1_s1** | text | on | M2b_s1 / s1 / 1 |
| 6 | N1_s0 | text | off | M2b_s0 / s0 / 0 |
| 7 | N1_s1 | text | off | M2b_s1 / s1 / 1 |

* Priority order, not a dependency chain: a chain stopped after arm 3 still yields a seed-0 read with its floor.
* N1 splits the nav source from the heads: N1 - C0 is the nav effect; S1 - N1 is the heads' effect on driving through the shared adapter.
* The banked box-2 OBD-CTL arms (s0, s1, s10, s11 at `cdbff971`; M, `t2v8f_pulls/obd_OBD-CTL_*`) are a different GPU. They are read beside C0 as a cross-venue replicate, never as the control.

### 3.1 Nav text (S1, N1)
Rule in `BUILD_SPEC.md` section 5: TURN-class side; distance counted down from the logged track since the row started; token on within `d_ann = max(30 m, 6 s x v)`; the same five channels as v9 (drop-in, no architecture change). Text and v9 agree on side where both announce: of 96,346 train v9-turn windows, 56.1 % have a same-side text turn, 38.1 % a FOLLOW-type text (bends, roundabouts), 4.7 % the opposite side (I, NAVTOK section 4).

### 3.2 Detected-info heads (S1)
One MLP `Linear(256,256) GELU Linear(256,41)` (76,329 params) on the mean of the NOW frame's adapted tokens. Per-variable CE against `strat_labels_table.npz` (sha256 `8055a321...`). The variables are country 25, weather 4, surface 3 (half weight), illumination 4 (twilight masked) and period of day 5. Visibility is not built: card audit 25/40, below always-good 31/40 (I, CONTEXT_LABELS_CARD).

### 3.3 Checkpoint loss: what exists and why it stays
* **Exists**: `T["ckpt"]` = L1 of S's (x, y, psi) at 30/50/80 m against v9 RC-A, the ego's own future path at those arcs, masked by `rc_ok & has`, weight `w_ckpt 1.0` (`v8f_mini.py:1163-1166` in `loss` at `cdbff971`; total term `w_ckpt * T["ckpt"]`; RC-A from `_rc_arrays`, `v8f_mini_data.py:151`). Under `--detach-upper`, S gets no gradient from T or O.
* **It stays**:
  * RC-A is the driven route, and the text is that route's router description (same drive, I). Where the cell is scorable, the logged path takes the announced side on 10 of 10 windows (I, STRATEGIC_PLAN section 3).
  * A text-side hinge on S's heading would only add signal where RC-A is invalid (3.2-4.4 % of windows, M via I, STRATEGIC_PLAN `rc_ok_rate`).
  * It would also train S on the S_CKPT rule itself, so the deciding STRATEGIC cell would measure its own loss.
* **What changes for S**: its nav input. The read then asks whether S follows the text (section 4.4).

## 4. Read

**Tiers.**
* LONGITUDINAL, LATERAL and TACTICAL: **T1-kin** (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene), eval-141, 141 episodes / 2,399 windows, per seed, as the OBD read (I, `PREREG_V8F_OBD.md` "Read").
* STRATEGIC route cells: **T0 at the window's first tick**. No own action has run there, and the reader asserts the T1 tick-0 dump equals the T0 output. A T1 per-tick row is reported beside.
* Detected info: **T0**. It is a per-frame perception output, and the camera stream is logged in every tier we run.

**Estimator.** `taniteval.ci.paired_episode_cluster_bootstrap`, B 2,000, seed 0, over the 141 eval episodes. It answers "another draw of EPISODES". "Another TRAINING RUN" is the floor F. "Another INFERENCE RUN" does not apply: the planner is deterministic, and every arm is dumped in one environment and one thread count (memory: thread drift moves dumps by mm).

**Floor F (training-run variance).** For each cell, F = |C0_rep - C0_s0| at seed 0, applied at both seeds (OBD rule r4). It is one replicate pair. A stronger floor needs more replicates; SEEDS-DISAGREE (section 6) adds them.

**Cost / gain.** A cost on a cell at a seed: Δ = S1 − C0 is separated on the worse side and |Δ| > F. A gain: separated on the better side and |Δ| > F.

### 4.1 LONGITUDINAL, LATERAL, TACTICAL (unchanged readers; 24 cells per seed)
Per band (operative knots 1-10, tactical knots 11-30), as the OBD card:
* ADE and FDE;
* LONGITUDINAL: speed MAE, acceleration MAE, along-track MAE, headway abs error, time-gap abs error, DK-VIOL;
* LATERAL: heading, curvature, yaw-rate and cross-track MAE;
* TACTICAL: TAC-LAT and TAC-LON accuracy.

Beside, with no bar: T's own decisions (T-LAT-F1, T-LON-F1, goal macro-F1), top-3 reads, and the nav-echo split of 4.3.

### 4.2 STRATEGIC, route-setting (6 cells per seed)
S's checkpoint position error and heading error against RC-A at 30, 50 and 80 m (`checkpoint_metrics`), S1 − C0, with the same cost/gain rule. The C0 side is echo-prone (its v9 nav shares the ego future with RC-A); a cost here is read as "the deployable nav sets the route less precisely than the oracle nav".

### 4.3 TACTICAL nav-echo split (beside, no bar)
Windows are split by the text at NOW (announced = a text turn within `d_ann`) and read per arm under real, donor and zero nav. Echo flags are as `PREREG_NAVTEXT_DRAFT` section 5. An echo-flagged gain is reported as echo.

### 4.4 STRATEGIC, route following: S_CKPT (deciding arc 80 m)
* S_CKPT is 1 when S's checkpoint heading at arc a is on the announced side by >= theta (25 deg; SLIGHT 10 deg), scored where `d_text + 20 m <= a` (`strategic_cells.py ckpt_cell`, I). COMPLETE_M 20 and the thetas are fixed here.
* Sensitivity rows at COMPLETE_M 10 and 30 are reported and do not decide.
* n on the eval grid: 214 windows / 24 episodes at 80 m (left 44, right 170); 120 / 22 at 50 m; 46 / 18 at 30 m (M, `proto_eval141_counts.json`, I by this agent).
* Per arm and seed: S_CKPT under real nav, donor nav (another episode's window, `donor_indices` seed 0) and zero nav.
* **Skill = real − donor (paired).** Real alone is never quoted for a text-fed arm.
* Best constant: always the majority side among scorable windows (right at 80 m: 170/214 = 0.794, I).

### 4.5 STRATEGIC, detected information (S1; C0 and N1 have no head: n/a, reason "no output")
* Per variable: accuracy and balanced accuracy at the window's NOW frame, against the eval constant baseline. The eval constants (I, LABELS_REPORT) are country 0.092, weather 0.846, surface 0.785, illumination 0.595 (twilight steps excluded, reported n/a with n = 240 steps / 1 clip), period of day 0.241.
* Per-class recall is reported for every variable.
* **Country rig control:** the rig baseline is the most common train country per `platform_class`, applied to eval clips. Platform per clip comes from the requested sidecar (LABEL_SPEC section 5). Country accuracy is also read within each platform stratum. Cramer's V country x platform is 0.288 train, 0.167 bias-corrected eval (I, LABELS_REPORT).
* Label ceiling: weather 36/40, surface 32/40 and illumination 30/39 against the PI's own frames (I, card). A head at the label's agreement level is reading the label pipeline as well as the PI would; no number here is called "accuracy against the world".

### 4.6 STRATEGIC members not in this arm (reported, not omitted)
* Derived goals: **not in this arm** (later card; goal-input rule).
* Strategic intent 6-30 s: **not computable**, no output beyond 80 m of arc. The would-have n from the ACTSENS text-turn windows is 32 at 6-12 s, 69 at 12-30 s and 20 above 30 s (I, STRATEGIC_PLAN section 2).
* Visibility: **not built** (label below its constant).

## 5. Controls (each with the value it must read)

| id | control | must read |
|---|---|---|
| K1 | reader self-pair (C0_s0 vs C0_s0) | exactly 0.0 on every cell |
| K2 | constant predictor through the detected-info reader | exactly the constant baseline |
| K3 | planted perfect predictor (the labels themselves) | 1.0 on every variable, and DETECTS |
| K4 | **C-RAND**: S1's checkpoint with `ctx_head` re-initialised (seed 0), eval only | NOT DETECTS on every variable; if any passes, the read is broken and nothing is quoted |
| K5 | clip-shuffled labels (seed 0) scored against S1's predictions | NOT DETECTS |
| K6 | rig-only country predictor | exactly the rig baseline; NOT DETECTS-BEYOND-RIG |
| K7 | S_CKPT on HOLD (straight) checkpoints / mirrored side / a checkpoint at the announced side x 40 deg | 0.0 / flips to 0 / 1.0 |
| K8 | S_CKPT donor on a synthetic arm that copies its nav input | moves to the donor's value |
| K9 | T1 tick-0 S output vs T0 output, same arm | equal (atol 1e-6) |
| K10 | trainer mutations (BUILD_SPEC section 9): ctx total line deleted, nav overwrite deleted, head wired into O, detach removed | each named test RED |
| K11 | G-CLOCK-NAV with the countdown sign flipped | FAIL |

## 6. Outcomes, committed before any number (each names what it triggers)

**Driving (24 family cells, both seeds; S1 vs C0, floor F)**

| outcome | condition | triggers |
|---|---|---|
| **KEEPS** | no cost on any of the 24 cells at either seed | the route text replaces the oracle nav in the recipe; N2 card (44-class text embedding + distance, an `e_nav` change) to recover roundabouts, bends and ramps |
| **ADDS** | KEEPS and the same cell a gain at both seeds | as KEEPS; the gain is named with its echo-split row |
| **COSTS-NAV** | a cell costs at both seeds, and N1 − C0 costs on that cell at both seeds | N2 card next (the 38.1 % FOLLOW-type text on v9-turn windows is the named suspect); no adoption |
| **COSTS-HEADS** | a cell costs at both seeds in S1 − C0, and N1 − C0 does not cost there | arm S1d = S1 with `--ctx-detach on` (built, default off), same seeds and reads |
| **SEEDS-DISAGREE** | a cost at exactly one seed | C0 and S1 at seeds 10 (init 0) and 11 (init 1); a cell is a cost at >= 3 of 4 seeds with one per init |

**STRATEGIC route-setting (4.2):** a cost at both seeds on an RC-A cell with KEEPS on the 24 is **ROUTE-COST**. It is reported as written and triggers the N2 card.

**STRATEGIC route following (4.4, S1 at both seeds)**

| outcome | condition | triggers |
|---|---|---|
| **S-FOLLOWS** | at 80 m, real − donor > 0 separated AND real − best-constant > 0 separated | the 120 m checkpoint card (6-30 s band; PI go) |
| **S-PARTIAL** | real − donor separated, not above best constant | the same card. Reason: inside 80 m the scene already shows the turn, so the nav's value lies beyond the vision horizon |
| **S-IGNORES** | real − donor not separated at >= 1 seed | zero-training check first: S1 with nav zeroed vs donor on RC-A error (does S read nav at all?); then the 120 m card |

**Detected info (4.5, S1, per variable, both seeds)**

| outcome | condition | triggers |
|---|---|---|
| **DETECTS(v)** | accuracy − constant > 0 separated, and K4 NOT DETECTS | v is a reported model output; country also needs DETECTS-BEYOND-RIG |
| **DETECTS-BEYOND-RIG** (country) | accuracy − rig baseline > 0 separated | country is a reported output; otherwise reported as "rig-level" |
| **NOT-DETECTED(v)** | otherwise | zero-training probe on S1's frozen adapter (attention pooling over all 40 tokens and the 4 history frames, fit on train once, scored on eval once). If the probe detects, the next arm is that head; if not, the representation lever (the 8x20 grid or the 416x1024 encode) |

Bars, controls and arms do not change after any number is seen. A bar that fails is reported as written, then the triggered arm runs (RULE ZERO).

## 7. Cost

* **Training (pod).**
  * MEASURED policy rate on box 2 (RTX 4060): 0.722-0.769 s/step, 2,165-2,306 s per 3,000-step DAgger fine-tune, over 30 banked runs (`t2v8f_pulls/{obd,stopbal,stopperc}_*/summary.json`).
  * The brief's 0.48-0.50 s/step is ACTSENS, which trains the WM only with a frozen planner and a latent cache (M, same folder, `wm_only: true`); it is not this workload.
  * RTX 4000 Ada: ESTIMATED 0.43-0.77 s/step. The spec-sheet throughput ratio (about 1.8x FP32, 1.3x bandwidth vs the RTX 4060; P, not banked, U) bounds only the GPU share; the data path and the DAgger swap run on the CPU.
  * Budget at 0.77 s/step: 38.5 min per run, about 45 min with gate and pilot. Core arms 1-5: **about 3.8 h**; all 7: **about 5.3 h**; SEEDS-DISAGREE adds 4 runs, about 3 h (all E).
  * The pilot measures the real rate; a projection above 10 h for the chain stops it and goes back to the MM.
* **Reads (dev box CPU).**
  * T1 dumps: about 10 min per arm (I, OBD README, E).
  * T0 STRATEGIC passes: 3 nav variants x 7 arms on 2,399 windows, about 2 h (E).
  * Reads: under 30 min (E).
* **Build:** about one Sonnet agent-day (E): one builder, the head, two evaluator flags, a checker, about 19 tests.
* **Disk:** at most 1.5 GB per banked arm (OBD's stated bound, E); 13 MB of label and nav tables.

## 8. Known risks

* A 3,000-step fine-tune may under-adapt S, T and O to the new nav semantics. A cost from that reads as COSTS-NAV. The T0 train curves are beside evidence, not a result.
* The DAgger banks were rolled out by M2b under the v9 nav. They are the same for C0 and S1, so they do not confound the comparison, but they are not on-policy for a text-fed model.
* Pod preconditions are UNVERIFIED: the cache format at `/root/data/cache_pod`, shipped inputs, memory cap. The checker asserts each before arm 1.
* Thin STRATEGIC n: 24 episodes at 80 m. A non-separated real − donor may be thinness; S-IGNORES then names the window count.
* The time base of the context labels is ASSUMED (at most 1.6 % of eval steps move at 0.4 s, I).

## Log
* 2026-11-16, MM: frozen. Bars, controls, arms and outcomes above do not change after any number is seen.
