# PRE-REGISTRATION -- measure 5r ("v4r"): the measure-5 slow-plan labels computed on the Amendment-7-REPAIRED plans

Written 2026-09-28 ~06:45 Berlin, BEFORE any v4r label exists. Both outcomes are committed below. PI decision (chat
~02:20, relayed by the coordinator): M5 deployment stays on HOLD, and the lever named in `eval/RESULT_M5_EFFECTIVENESS.md`
§4 starts now. `eval/PREREG_MEASURE5.md` (blob c668b5b6) is unchanged; its rig is reused, and every departure from it is
listed here.

## 0. Why
M5 passed its registered test (ADOPT, E1 V4 - V3 +0.1534 [+0.1037, +0.2104]) against a truth scored BEFORE SPEC
Amendment 7 (the executed plan takes heading[18] at the last native pose; `refe/planner.py repair_last_heading`).

Re-read POST HOC against the repaired truth, E1 was -0.0388 [-0.0791, -0.0009], inside the seed floor 0.0517. Two
findings explain why:
* A 0.75x copy's advantage over its own source fell from +21.0 to +1.43 PDMS.
* The v4 comfort label rewarded copies almost only where the source carried the heading defect: +0.686 vs +0.006, over
  15,200 pairs (`raw/m5_effectiveness/label_mechanism.json`).

The v4 labels describe plans that the deployed planner no longer executes. v4r labels the plan that WILL be executed.

## 1. The configuration under test (fixed now)
* **Labeller:** `refe/onpolicy_label_v4.py --slow-copies --slow-factors 0.75 --slow-n-src 8` (STOP on, sources
  scorer_top), which is M5's C2, **plus the new flag `--repair-last-heading` (default OFF)**.
* **What the flag does:** before EVERY labelling function, every REFe candidate passes through
  `planner.repair_last_heading` (imported, never copied). That covers the 64 originals, the 8 copies and STOP, and both
  labelling functions: the teacher's `onpolicy_label.Scorer.score` and NAVSIM's `navsim_dac.navsim_dac_and_comfort`.
  On STOP the repair is the identity.
* **NOT repaired:**
  * the reference candidate (the teacher plan on navtrain; the human future on navtest);
  * the SERVED set (`traj`, `yaw`: what the scorer reads).
  At inference the scorer scores the unrepaired proposals and the repair acts after selection, so a slot's label becomes
  the outcome of EXECUTING that slot's plan if it is picked.
* **Line format:**
  * lines carry `label_version` 5 and `"repair": "last_heading_hold"`;
  * the served arrays equal v4's bit for bit;
  * the side file (`slowaux_*`) carries the dropped originals' REPAIRED labels.
* **Labelling run:** every sample is labelled with its copies (`--slow-frac 1.0`), as in M5. The arms choose per sample.

## 2. Rig (as PREREG_MEASURE5 §3, unchanged unless listed)
* **Starting point:** `snap_epoch015.pt` (md5 d7c59f4f...).
* **Data:**
  * the SAME 1,600 train + 300 val navtrain samples;
  * the SAME shared trunk cache (`refe_trunk_cache/d7c59f4f`, built 2026-09-27/28, every shard gate PASS);
  * the SAME 1,900 props chunks, which are M5's queue. The v4r labeller runs on copies of them in a new queue, so
    nothing is re-forwarded.
* **Training:** §3d exactly (scorer-only, AdamW, lr 1.0322678e-4 as recorded for M5, wd 0.01, score weight 0.1,
  batch 16, 4 epochs, bf16 autocast on the GPU, 3 seeds), with the SAME batch orders.

## 3. Arms (identical samples, order, steps and seeds; only the label content differs)
* **V3 (CONTROL):** pure 64 with v3 labels, unrepaired: the live labeller's content.
  * The M5 model files are REUSED (sha256 in `raw/m5_effectiveness/ft_models_sha256.json`) iff the v4r data's key list
    equals M5's key list exactly.
  * Otherwise V3 is retrained in lockstep.
* **V4r (GATING):**
  * samples carrying at frac 0.5 (the labeller's own hash, M5's split: 803 of 1,600) use the served v4r set;
  * the others use the V3r set.
* **V3r (reported):** pure 64 with REPAIRED labels, i.e. the v4r line's kept slots plus the side file's dropped slots.
  This isolates the repair of the labels from the copies.
* **V4 (reported):** M5's V4, reused under the same rule as V3.

## 4. Truth, tokens, metrics
**Truth = REPAIRED:** the unchanged NAVSIM v1 harness on every candidate as the deployed planner would execute it
(`repair_last_heading` then the seam's `to_navsim`):
* the 64 originals;
* the 0.75x copies of snapshot 015's own top-8;
* STOP.

**Stage 1 -- W3's 200 tokens (93 logs).** The truth is already banked, and it was built before this document:
* originals: `D:/Projects/TanitAD/data/refe_navtest/proptable/sub200_ep015_repaired/table.npz` (like_for_like_016,
  64/64 harness runs PASS, all gates);
* copies: `score/refe_sub200_ep015_f075rep_r00..r07` (`eval/m5_repaired_extras.py`, 8/8 PASS, controls in
  `raw/m5_effectiveness/repaired_extras.json`);
* STOP: `score/refe_sub200_ep015_stopzeros`.

⚠️ **EXPOSURE, declared:** V3 and V4 were read on these tokens against this truth POST HOC (RESULT_M5 §3). V3r and V4r
never were. The bars below are M5's, fixed before any data, plus one added non-inferiority bar (E2).

**Stage 2 -- CONFIRMATION on fresh tokens: Amendment 5's 923 tokens (43 logs, disjoint from W3's 200;
`raw/a5_confirm/a5_confirm_tokens.json`).**
* Scorer context: M6's `D:/Projects/TanitAD/data/refe_proxy/cache_eval_ep015`, which carries visual_ctx, scene_ctx,
  native proposals and logits through the planner's own input path, fp32 (its build gates C1 and C2 read 0.0).
* That context is admissible only if its NAVSIM-grid proposals equal `proptable/a7confirm_ep015/proposals.npz` bit for
  bit, and A0's logits on it reproduce that file's logits to <= 1e-3 with identical picks.
* Truth: 73 new harness runs on the repaired seams.

**Metrics:** exactly PREREG_MEASURE5 §4, with the repaired truth:
* **E1 (PRIMARY):** masked slow-plan pair concordance;
* **E3a:** pair concordance among the 64;
* **E3b:** PDMS of the argmax among the 64;
* **E2:** PDMS of the pick among the 64 + 9 extras, masked.
* **Estimator:** per-token V4r - V3 of seed means; paired log-cluster bootstrap over the logs (93, then 43), 10,000
  resamples, 95 % percentile, seed 20260927.
* **Seed floor:** the largest |seed_i - seed_j| of the token-mean E1 within V3 and within V4r.
* Inference is deterministic.
* **Reported with every readout (the binding four families):** EP and the NAVSIM sub-scores of each arm's E3b and E2
  pick (same estimator); `families6.py` blocks (longitudinal, lateral, tactical; strategic UNAVAILABLE in NAVSIM); E4;
  the E2 selection mix; the navtrain-val BCE; V3r and V4 contrasts.

## 5. Gates (all must pass before a statistic is read; a failing gate is reported, never waived)
* **G-R1 REPAIR MECHANISM (label-free: the labelling functions are stubbed; run before any v4r label).** At the input of
  `Scorer.score` AND of `navsim_dac_and_comfort`, with the flag ON, for every REFe candidate (originals, copies, STOP):
  * x, y bit-identical to the unrepaired candidate on all 20 poses;
  * heading[0..18] bit-identical;
  * heading[19] == the unrepaired heading[18], exactly;
  * the reference candidate bit-identical;
  * the served traj/yaw bit-identical to flag OFF.

  With the flag OFF the inputs are the unrepaired candidates exactly. Deliberate-regression arms, each of which must turn
  G-R1 RED:
  * m1: heading[18] := heading[17] (the wrong index);
  * m2: the repair applied to the served set, not the labelled candidates;
  * m3: the repair skipped on the copies.
* **G-R2 FLAG OFF == v4.** With the flag OFF, the modified labeller reproduces M5's v4 set and side lines byte for byte
  (timing fields excluded) on the first 2 chunks of M5's queue (32 samples).
* **G-R3 NAVSIM drivable area + comfort on repaired plans == the harness on the repaired seams.**
  * Tokens and slots: W3's 200 tokens, 73 slots each, 14,600 slots.
  * Method: the real CLI with the flag ON, read back through `train.OnPolicyBank`.
  * Bar: **100 %** for both. This is the pattern that made DAC and comfort 100 % in M5.
* **G-R4 collision label on the copies (PREREG_MEASURE5 §2's rule), against the repaired harness NC on the same slots.**
  * Bar: copy agreement >= the originals' and >= 90 %.
  * Bar: NOT one-directional (neither disagreement direction > 3x the other).
  * If G-R4 fails, v4r is NOT tested. The design is BLOCKED on the collision label, and the next lever is the
    NAVSIM-faithful collision label on the simulated states (PREREG_MEASURE5 §1b).
* **G-R5 coverage.**
  * The v4r labels cover the same 1,900 keys, with 0 failed and 0 selfcheck_failed.
  * The self-check is extended so that the LABELLED copy == repair(copy) and the SERVED copy == copy.
  * `m5_finetune_eval.py data` finds 0 kept-slot mismatches.
* **G-R6 eval (M5's gates on the repaired truth).**
  * repair(the model's copies) -> to_navsim equals the harness-scored repaired copies bit for bit;
  * the harness is complete;
  * A0 reproduces the table logits (<= 1e-3) and the picks;
  * the masked route leaves the 64 unchanged (<= 1e-4);
  * the estimator self-test reproduces snapshot_pair_under_rule's published interval.

## 6. Decision -- both outcomes committed now
**Stage 1 (W3's 200):**
* **STAGE-1 PASS** iff ALL of these hold:
  * E1 V4r - V3 >= **+0.03**, with CI lower bound **> 0**, and the effect larger than the seed floor;
  * E3a CI lower bound **> -0.01**;
  * E3b CI lower bound **> -2.0** PDMS;
  * E2 CI lower bound **> -2.0** PDMS.

  Then stage 2 runs.
* **REFUTED** iff E1's CI upper bound < 0. v4r is not deployed. The next lever is a NAVSIM-faithful label for every
  component on the simulated states (collision first, §1b), then a re-test.
* **Otherwise NOT PROVEN.** Not deployed, and stage 2 is not run. The same next lever applies.

**Stage 2 (A5's 923, fresh):** the same four bars.
* **ADOPT** iff met. The live change is then the ON-POLICY labeller on the pod labelling the repaired plans (flags
  `--slow-copies --slow-factors 0.75 --slow-frac 0.5 --repair-last-heading`, label_version 5). That is a pod-side recipe
  change; a kit is drafted, and it is shipped only on the PI's go.
* **REFUTED** iff E1's CI upper bound < 0.
* **Otherwise NOT PROVEN.**

**Reported, never gating:** V3r - V3 (the repair of the labels alone), V4r - V3r (the copies given the repair),
V4r - V4, EP, the four families, E4, and the label-ranking ceiling of the v4r labels against the repaired truth
(G-R3's data).

## 7. Compute (MEASURED on 2026-09-27/28 unless marked)
* **Labelling (CPU):** 1,900 samples at ~0.93 GB and ~1.0-1.2 sets/min per worker. Workers are capped by the dev box's
  free-RAM rule (>= 5 GB free).
* **G-R3:** 200 navtest samples.
* **GPU:**
  * the fine-tune of V3r + V4r (plus V3/V4 if retraining is required) takes 12-24 min (M5: 18.1 min for 3 arms);
  * the eval takes ~2 min per 10 models on 200 tokens.
  * GPU start: coordinator's rule.
  * Hard stop: 12:30.
* **Stage 2:** 73 harness runs x 923 tokens, ~9 min each (ESTIMATED from 115 s per 200 tokens): ~11 CPU-h. It runs
  only after STAGE-1 PASS.


## ADDENDUM 1 -- 2026-09-28 ~14:05 Berlin: V3r becomes a SECOND GATING arm in Stage 2 (registered BEFORE any Stage-2 output)
Written after the STAGE-1 PASS (`eval/RESULT_M5R_STAGE1.md`), at the coordinator's instruction ("M5r Stage 2: GO", 2026-09-28).
**No Stage-2 input or output exists at the time of writing:** no context gate, no seam, no harness run, no eval on the
923 tokens. Everything above this addendum is unchanged; the Stage-1 verdict stands as read.

**Why.** Stage 1's REPORTED contrasts (not gating, on the exposed 200 tokens) put the effect in the repair of the labels:
* V3r - V3 E1 +0.0778 [+0.0488, +0.1069];
* V4r - V3r E1 -0.0188 [-0.0508, +0.0093].

V3r (the pure 64 with REPAIRED labels, no slowed copies) is the cheaper live change: it needs no set composition and
no copies. Only a fresh-token test can decide between V3r and V4r, so V3r is added here as a gating arm.

**Arms read in Stage 2.** The models are exactly Stage 1's (no retraining):
* **V3** (control): `refe_m5/ft/V3_s{0,1,2}.pt`;
* **V4r**: `refe_m5r/ft/V4_s{0,1,2}.pt`, trained on the v5 data's served sets at frac 0.5;
* **V3r**: `refe_m5r/ft/V3_s{0,1,2}.pt`, trained on the v5 data's pure repaired sets;
* **V4** (reported): `refe_m5/ft/V4_s{0,1,2}.pt`.

The sha256 of every file is in `eval/raw/m5_effectiveness/ft_models_sha256.json` and `eval/raw/m5r/ft_models_sha256.json`.

**Per-arm test.** Each of V4r and V3r is tested against V3 with the SAME four bars and the SAME estimator as §6
Stage 2 (paired log-cluster bootstrap over the 43 logs, 10,000 resamples, 95 % percentile, seed 20260927, the
repaired truth, the gates of §4-§5). Per arm X in {V4r, V3r}:
* **PASS(X)** iff ALL of these hold:
  * E1 X - V3 >= +0.03, with CI lower bound > 0, and the effect larger than the seed floor;
  * E3a lower bound > -0.01;
  * E3b lower bound > -2.0 PDMS;
  * E2 lower bound > -2.0 PDMS.

  The seed floor (X) is the largest |seed_i - seed_j| of the token-mean E1 within V3 and within X, on the 923 tokens.
* **REFUTED(X)** iff E1 X - V3 has CI upper bound < 0.
* Otherwise **NOT PROVEN(X)**.
* **Multiplicity, declared:** two arms, each at 95 %, with no correction. The adoption rule below never adopts an
  arm that failed its own four bars.

**Adoption decision (committed now):**
* **Both pass:** adopt **V3r** (the cheaper change), UNLESS V4r beats V3r on the named bar. The named bar is E1
  V4r - V3r (paired, same estimator), which must have its CI lower bound > 0 AND be larger than the seed floor of V3r
  vs V4r (the largest within-arm seed pair of V3r and of V4r). If it does, adopt **V4r**.
* **Only V3r passes:** adopt **V3r**.
* **Only V4r passes:** adopt **V4r**.
* **Neither passes:** nothing is adopted. Each arm's verdict (REFUTED / NOT PROVEN) is reported as read. The next
  lever is §6's: a NAVSIM-faithful label for every component on the simulated states.

**What an ADOPT means.** An ADOPT of either arm leads to a DRAFTED pod kit, shipped only on the PI's go:
* V3r: the on-policy labeller with `--repair-last-heading` and NO `--slow-copies` (label_version 5, pure sets);
* V4r: `--slow-copies --slow-factors 0.75 --slow-frac 0.5 --repair-last-heading`.

**Reported, never gating:** V4r - V3r on every metric, V4 - V3, EP and the NAVSIM sub-scores of each arm's picks,
the four families (strategic UNAVAILABLE in NAVSIM), and the E2 selection mix.

**Scope note (PI ruling 2026-09-28: no HD-map geometry for planning; navigation commands are OK).**
* The labels are offline ground truth, so NAVSIM's scorer inside the labeller is admissible.
* The arms under test change ONLY the scorer (`score_q_mlp`, `score_dec`, `score_head`), whose inputs are the
  candidate trajectories and the visual context (camera tokens + calibration).
* However, the REFe planner's trajectory decoder receives a `goal` built at inference from the route polyline on the
  HD map (`planner._goal_for` -> `_route_with_lane_rank(map_api, ...)` -> `route_goal_positions`). This path is SHARED
  by every arm, including A0 and V3. It is not introduced or changed by this measure, and this test does not decide it.
* Any live deployment must therefore reconcile it with the ruling (a PI item, stated in the Stage-2 RESULT).

**CORRECTION to Addendum 1's scope note -- 2026-09-28 13:57 Berlin (scope prose only; NO arm, bar, estimator or decision changed; the registered text above, blob 2f32042b, is left unedited).** The PI ruled at ~13:50 on 2026-09-28, asked about exactly this path, that REFe's registered route goal (`planner._goal_for` -> route polyline -> `route_goal_positions`) STAYS as navigation input. The ruling blocks only NEW map-geometry uses (Amendment 8's lane fallback, map DAC, route-progress EP). The route-goal path is therefore NOT an open PI item, and M5r adds no map-geometry path at inference.
