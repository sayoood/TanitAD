# RESULTS -- trigger probe 2: what the nav encodes, the trunk without the oracle nav, and motion cues (H-CRAWL-06 precursor, continued)

Zero training of the planner, dev-box CPU. Card frozen first: `PREREG_FROZEN.md` (sha256 `d9af82be053463e37f295f5dc0b774a103e5858a7bddf2fd03a77a20739a40cb`, 2026-10-28T23:11:02+0100, `FROZEN_SHA.txt`).
**Evidence class: MEASURED** (artifacts in `raw/`; every number below is printed by `tables2.py` -> `raw/tables2.txt`, or sits in `raw/VERDICT.json`). **Tier: T0-class** probe of representations on logged inputs; not driving performance.
**Estimator:** clip(episode)-cluster bootstrap, B = 2000, seed 0, on the **mean of per-fold AUCs ("fold")**, pooled out-of-fold AUC beside. It answers "another draw of EPISODES"; it is blind to training variance (H-ESTIM-SEED-1) and to the draw of a label permutation (section 3); every Z arm rests on ONE checkpoint (DAG-OACC s0, sha256 `67265ec1...8bc45f`).
Same 12,000-window samples, folds and bootstrap resamples as the previous probe (P-STOP 849 clips, P-MOVE 2,140 clips; GO = the registered launch mask). The old arms E / V+E / Z+E are re-scored on the fold-mean metric from their banked OOF scores (no refit): **E 0.626 / 0.732, V+E 0.680 / 0.785, Z+E 0.800 / 0.881** (P-STOP / P-MOVE).

## 1. Verdicts by the committed rules (TRAIN, linear, fold-mean, per population)

| | P-STOP | P-MOVE |
|---|---|---|
| **Z_nn:** dAUC(Z_nn+E - E) | **+0.134 [0.110, 0.155]** (E 0.626 -> Z_nn+E 0.760; Z+E 0.800) | **+0.117 [0.105, 0.129]** (E 0.732 -> 0.849; Z+E 0.881) |
| rule | **FAILED-CONTROL as frozen** (C2, section 3). Its arithmetic reads VISION-THROUGH-TRUNK; the post-freeze null band says the C2 miss is a draw artifact | **VISION-THROUGH-TRUNK** (all frozen controls pass) |
| dAUC(Z+E - Z_nn+E) | +0.040 [0.027, 0.055] | +0.032 [0.027, 0.037] |
| flags | F1 (MLP +0.126 [0.103, 0.150]): robust. F2 (eval-141 +0.123 [-0.083, +0.290]): not triggered | F1 (MLP +0.073 [0.064, 0.083]): robust. F2 (eval-141 +0.075 [+0.005, +0.148]): not triggered |
| **V4:** best arm = V4 | V4+E 0.675 vs V+E 0.680: **-0.005 [-0.011, +0.001]** | V4+E 0.784 vs V+E 0.785: **-0.000 [-0.003, +0.002]** |
| rule | **NO-MOTION** | **NO-MOTION** |

**Read the Z_nn row with section 4 before quoting it.** The committed label says "the trunk holds go/stay information that is not the nav's". That is literally true, and it is not vision: a **second ego-future input** (the "speed limit", section 2) is still in the trunk, and when it is scrambled too the trunk falls to the level of the frozen features V+E and the eval-141 gain disappears.

## 2. ARM 1 -- what the nav encodes

**How the nav input is built (tip `c1c30fb7`, line numbers of the tip copies `scratch/tip_*.py`).**
* The model reads `nav` as a 5-vector `e_nav = Linear(5, d)` applied to `b["nav"]` at the tick frame: `v8f_mini.py:650` (definition) and `:708` (run_s), `:719` (run_t), `:735` (run_o); there is no null-nav / nav-dropout path anywhere in `v8f_mini.py`, `v8f_mini_data.py`, `v8f_mini_train.py`, `v8f_t1_eval.py`. At inference a_t = a_s = 0, so S, T and O all read NOW.
* `v8f_mini_data.py:199-208` (`build_label_join`): nav5 = (FOLLOW, TURN_L, TURN_R one-hot, d_next/100 clipped to [0, 3], args_valid); `:510` (`item`) gathers it for the hmax = 16 window frames. `v9_labels.py:39-42`: the nav INPUT fields; `nav_t_next_s` / `nav_token_ttime` (future time) are forbidden as inputs.
* `build_v9_labels.py:233-260` (`turn_table`): the turns are every sustained-yaw segment classified by `is_turn` (turn-band literal 30 deg at `:43`; its exact use inside `is_turn` is UNVERIFIED: that function lives in `s2_geom_emit_v7.py`, not read) of **the WHOLE recording** (`manoeuvre_sequence`, `:243`), i.e. of the ego's own FUTURE path; `:263-283` (`nav_fields`): `ann` = announced turns with `s_end > s_now`; `args_valid = 1` iff one exists; `nav_token` = TURN_L/R if the turn is in progress or `d_next <= d_ann = max(30 m, 6 s * v_now)`, else FOLLOW; `d_next` = arc distance from now to the turn start. `:794` calls it per frame; `:1257` records the policy `allow_oracle_nav (ego-future)`.
* **Can it encode stay vs go? Only weakly, and never the timing.** A turn can lie ahead only if the ego's path continues, so `args_valid = 1` says "the ego will move at some point in the remaining recording". It says nothing about WHEN (the 6 s horizon of the GO label), and a straight departure (no turn) is encoded exactly like a stay (FOLLOW, args_valid 0).
  MEASURED (full populations, `raw/results_nav2_desc.json`): P-STOP GO rate 57.4 %: **63.3 % when a turn lies ahead (70 % of windows) vs 43.8 % when none (30 %)**; by token FOLLOW 52.4 %, TURN_L 60.9 %, TURN_R 69.7 %; d_next <= 5 m: 70.9 %; P(turn ahead | GO) 77.0 % vs P(turn ahead | not GO) **60.1 %** (most "stay" windows still have a turn ahead because the ego departs later in the clip). P-MOVE: 56.5 % vs 42.6 %, d_next <= 5 m (in a turn) 75.0 %.

**What the nav is worth as a feature (fold-mean AUC; `raw/results_nav2_*.json`, linear / MLP agree):**

| | P-STOP | P-MOVE |
|---|---|---|
| **N5** (the nav token alone) | 0.581 [0.544, 0.620] | 0.584 [0.564, 0.604] |
| **N5+E - E** | **+0.036 [0.010, 0.062]** (MLP +0.031 [0.005, 0.058]) | **+0.014 [0.007, 0.022]** (MLP +0.011 [0.005, 0.017]) |
| args_valid alone (NV) | 0.579 (N5: 0.581) | 0.547 (N5: 0.584) |
| **L** (the "speed limit" input) alone / **L+E - E** | 0.609 / **+0.072 [0.057, 0.087]** | 0.624 / **+0.053 [0.044, 0.062]** |
| **NL+E** (the previous probe's "N": nav ++ limit) | 0.727 (reproduces the previous 0.727 fold-mean / 0.719 pooled) | 0.794 |
| dAUC(Z+E - N5+E) / (Z+E - NL+E) | +0.138 / +0.073 | +0.135 / +0.086 |

**Finding that changes the reading of the previous probe (post-freeze, `raw/POSTFREEZE_LOG.md` item 1).** The previous diagnostic attributed Z's edge to "the oracle nav". Most of that diagnostic's N was the **`limit` / `limit_ok` input, which is NOT a posted speed limit**:
tip `build_v8_speed_max_labels.py:16-19, 138-146` and `refc_v3_train.py:270-273` state it is `g_tac.goals.SPEED_BAND.v_hi_ms` = "the maximum of the ego's OWN REALISED speed over [t0+2 s, +6 s]" (provenance ego-future, oracle; quantised to 4 bins, "bin + v0 recovers R^2 0.9702 of the raw ego future"), one value per clip.
So the trunk has TWO ego-future channels; the speed ceiling carries about twice the go/stay information of the nav token (L+E - E +0.072 vs N5+E - E +0.036 on P-STOP). Neither is a vision signal.

## 3. Controls (`raw/VERDICT.json`, `raw/control_*.json`, `raw/pytest_output.txt`: 14 passed)
* **C1 label shuffle** (fold-mean, CI covers 0.5): PASS for E, N5+E, Z_nn+E, Z_nl+E, V4+E, Vdiff+E in both populations (0.492-0.507).  **C3 leak** (GT label appended): PASS, 1.0000, for N5+E, Z_nn+E, Z_nl+E, V4+E, Vdiff+E.
* **C2 episode-shuffled labels (seed 2):** PASS for E (both), Z_nn+E P-MOVE (0.5065 [0.490, 0.524]), V4+E (0.496 [0.450, 0.539] / 0.493 [0.472, 0.515]).
  **FAIL, as frozen: Z_nn+E on P-STOP, fold-mean 0.4543 [0.418, 0.491]: the CI misses 0.5.** By the frozen text that voids the arm ("FAILED-CONTROL, no verdict drawn") and that is the committed reading in `VERDICT.json` (the test file pins the failure).
  It is below chance, not above: the pipeline cannot manufacture signal. POST-FREEZE diagnosis (`c2_band.py`, items 5-6 of the log): the bootstrap resamples episodes with the labels fixed, so it cannot see the spread over donor-label DRAWS. Ten more draws (seeds 3-12) give
  Z_nn+E P-STOP 0.5105 sd 0.0226 [0.470, 0.544]; with seed 2 the 11-draw mean is 0.505, seed 2 is the minimum of the 11 (a 1/11 event), and the true arm (0.760) exceeds the band maximum by 0.216. All other bands are centred within 0.02 of chance (Z+E P-STOP 0.521 is the farthest). Reported beside, never replacing, the frozen reading.
* **N1 identity:** a fresh forward with the TRUE nav reproduces the banked Z (max |diff| 1.9e-6 over 128 windows), and the swap harness with donor := self reproduces it identically (1.9e-6; the emitted plan changes by exactly 0.0). Eval-141 true-forward vs banked Z: 0.0. This also shows the exported tree used to execute (`dagt1_tree_b43c7a9a`) has the tip's forward path (the two differ only in training-side code).
* **N2 reachability:** of 128 windows 109 get a donor nav that differs from their own; for all of them h_O changes (relative L2, 100 %; median 0.141) and the plan changes (median max |d traj60| 0.59 m); windows whose donor nav equals their own are bit-for-bit unchanged (max rel L2 0.0). Nav+limit swapped: median rel L2 0.244, median plan change 2.37 m.
  Over the full samples the donor nav equals the window's own nav in 13.6 % (P-STOP) / 10.0 % (P-MOVE) of windows, the donor limit in 37.0 % / 37.6 %.
* **V1 frames:** frame t of the old extraction equals the cache's frame t (max |diff| 0.0), every stored lag frame equals the cache's frame t - lag, frames differ from frame t in 100 % of 200 checked rows per population, no unfilled row (`extract_v4_fill.json`). **EV0:** the re-fitted comparators E, V+E, Z+E reproduce the previous probe's eval-141 AUC with |diff| 0.0 (both families).

## 4. ARM 2 -- Z_nn: the trunk with the nav scrambled (donor nav; the model has no null-nav path)
TRAIN fold-mean AUC (linear [CI]; MLP in brackets):

| | P-STOP | P-MOVE |
|---|---|---|
| E | 0.626 [0.613, 0.640] (0.628) | 0.732 [0.719, 0.745] (0.771) |
| V+E (frozen current frame) | 0.680 (0.681) | 0.785 (0.786) |
| **Z_nn+E** (nav scrambled) | **0.760 [0.736, 0.782]** (0.754) | **0.849 [0.838, 0.860]** (0.844) |
| Z_nn alone | 0.760 | 0.848 |
| Z_nl+E (nav AND limit scrambled; POST-FREEZE) | 0.685 [0.660, 0.709] (0.681) | 0.813 [0.802, 0.825] (0.813) |
| Z+E (true nav) | 0.800 | 0.881 |
| dAUC(Z_nn+E - V+E) | +0.080 [0.057, 0.102] | +0.064 [0.055, 0.075] |
| **dAUC(Z_nl+E - E)** (post-freeze, "rule reading" VISION-THROUGH-TRUNK) | **+0.060 [0.035, 0.083]** | **+0.082 [0.070, 0.094]** |
| dAUC(Z_nl+E - V+E) | **+0.006 [-0.016, +0.027]** (MLP +0.000) | +0.029 [0.021, 0.037] (MLP +0.027) |
| dAUC(Z+E - Z_nl+E) | +0.115 [0.093, 0.137] | +0.067 [0.060, 0.075] |

* Scrambling the nav alone removes only 0.040 (P-STOP) / 0.032 (P-MOVE) of Z+E: **the nav is not where the trunk's go/stay information mostly lives.** Scrambling the speed ceiling as well removes a further 0.074 / 0.035. The residual of the P-STOP trunk (Z_nl+E - E +0.060) is **the same size as the frozen features' own gain (V+E - E +0.054)**: Z_nl+E - V+E = +0.006 [-0.016, +0.027].
  The contributions are not additive (interactions), so these are two readings of one table, not a decomposition.
* **Eval-141 (unseen, no refit; single fit, episode bootstrap; P-STOP n = 101 / 19 episodes, P-MOVE n = 507 / 67 episodes):** Z_nn+E - E **+0.123 [-0.083, +0.290]** (E 0.517, Z_nn+E 0.640) / **+0.075 [+0.005, +0.148]** (E 0.708, Z_nn+E 0.783); Z+E - Z_nn+E +0.015 / +0.021 (CI reach 0).
  **Z_nl+E - E +0.072 [-0.114, +0.227] / +0.019 [-0.040, +0.080]; Z+E - Z_nl+E +0.066 [-0.033, +0.151] / +0.076 [+0.014, +0.140].** So on unseen windows the scrambled-nav trunk keeps its edge on P-MOVE, and the nav-and-ceiling-scrambled trunk does NOT clear E: the TRAIN residual (+0.06 / +0.08) is unconfirmed (and may be memorisation: the checkpoint trained on these windows and their GT).
* **Why the committed label over-reads.** (i) The ceiling input still in Z_nn is an ego-future oracle. (ii) The donor nav makes the input inconsistent for a trunk trained only on consistent navs, so Z_nn can under-state a nav-free trunk (committed caveat ii). (iii) TRAIN windows were seen in training (caveat i); eval-141 is wide on P-STOP.
  What is established: with the nav scrambled the trunk keeps 0.76 / 0.85, so the nav is not the carrier; what the residual consists of (vision, memorisation, or the ceiling) is separated only by Z_nl, whose TRAIN residual equals V+E on P-STOP and whose eval residual is not distinguishable from 0.

## 5. ARM 3 -- V4: motion cues from the model's own 4 frames (t-3 .. t)
TRAIN fold-mean AUC, linear (MLP): 

| | P-STOP | P-MOVE |
|---|---|---|
| E / V+E | 0.626 / 0.680 | 0.732 / 0.785 |
| **V4+E** (4 frames concatenated, PCA-64) | **0.675 [0.646, 0.705]** (0.665) | **0.784 [0.771, 0.798]** (0.786) |
| **Vdiff+E** (V_t - V_{t-3}, PCA-64) | 0.637 [0.625, 0.650] (0.635) | 0.734 [0.722, 0.748] (0.771) |
| dAUC(V4+E - V+E) | **-0.005 [-0.011, +0.001]** (MLP -0.015 [-0.025, -0.005]) | **-0.000 [-0.003, +0.002]** (MLP +0.000) |
| dAUC(Vdiff+E - V+E) | -0.043 [-0.071, -0.013] | -0.050 [-0.062, -0.038] |
| dAUC(Vdiff+E - E) | +0.011 [0.001, 0.022] | +0.003 [-0.001, 0.006] |

* Both arms are below 0.01 against V+E: **motion in the model's own 4 cached frames carries nothing the current frame does not**, on TRAIN, in both populations, with the nonlinear probe, and on eval-141 (V4+E - V+E -0.003 [-0.028, +0.015] / +0.006 [-0.009, +0.022]). The motion-only arm (Vdiff) is a small gain over E on P-STOP (+0.011) and none on P-MOVE.
* Caveats: the best-of-two selection is a winner's curse (not covered by the CI; V4 won by 0.038 / 0.050 over Vdiff, so it is immaterial here); the frames are 0.2 s apart, so t-3 is 0.6 s back; PCA-64 of the concatenation is dominated by the static content (the Vdiff arm is the motion-pure test and it is also null against V+E).
* Strata (below): no stratum in which V4 beats V+E. Exploratory pointer only (no rule, CIs reach 0): in P-STOP, where V+E has no gain, Vdiff+E - V+E is **+0.116 [-0.004, +0.200]** on TL_GREEN clips and +0.049 [-0.016, +0.133] on lead-departs, and negative elsewhere (TL_RED -0.077, NO_TL -0.057).

## 6. Strata (TRAIN, linear fold-mean; TL_* are CLIP-level and VLM-derived, coarse; none enters a rule; `raw/results_strata2.json`)
P-STOP: GO rate (full pop.) TL_RED 0.445, TL_GREEN 0.886, lead-departs 0.817, NO_TL 0.607. dAUC against E:

| P-STOP stratum | V+E - E | Z_nn+E - E | Z_nl+E - E | V4+E - V+E |
|---|---|---|---|---|
| TL_RED | +0.099 [0.041, 0.147] | +0.142 [0.098, 0.181] | +0.076 [0.032, 0.120] | -0.006 |
| TL_GREEN | -0.058 [-0.125, +0.048] | +0.053 [-0.054, +0.134] | -0.011 [-0.072, +0.077] | +0.011 |
| lead departs >= 2 m in 2 s | -0.019 [-0.095, +0.046] | +0.035 [-0.033, +0.095] | -0.028 [-0.088, +0.024] | -0.007 |
| NO_TL | +0.061 [0.020, 0.099] | +0.134 [0.100, 0.163] | +0.062 [0.028, 0.092] | -0.001 |

Where the frozen features add nothing (green light, lead departing), the trunk without the nav AND the ceiling adds nothing either; the trunk's gain there (Z_nn) comes through the oracle channels. P-MOVE: Z_nn+E - E +0.039 (TL_RED) to +0.135 (NO_TL); Z_nl+E - E +0.027 to +0.096; V4+E - V+E within +-0.009 everywhere.

## 7. What it implies for the next v8F lever
1. **The trunk's trigger information beyond ego is mostly the two ego-future inputs, not vision.** Nav alone is a small, time-agnostic cue (+0.036 / +0.014 over E); the speed ceiling is the larger one; with both scrambled the P-STOP trunk equals the frozen-feature level and the eval-141 gain is not distinguishable from 0. A loss-only stop/go head on this trunk (the previous probe's D1 lever) would mostly learn to read those oracle channels: it would score on the logged-input T0 probe and buy no deployable skill. The head is not the lever until the oracle inputs are out of the loop.
2. **Cheapest next experiment (zero training, PI-gated only as to the tier):** evaluate the deployed planner's GO windows with nav AND ceiling at their donor values (the Z_nl scramble, at the planner output: plan error on the 327 GT-launch windows) to get the launch skill without ego-future inputs; this probe cannot give that number (it measures representations, not plans). It decides whether the crawl is a perception defect or an oracle-dependence defect.
3. **Motion is not the lever** (V4/Vdiff null). The perception-side signal that exists is the TL_RED stratum (V+E +0.099); green-light and lead-departure windows show no gain from frozen current-frame OR 4-frame features. The live perception lever is therefore targeted (light state, lead gap/TTC at higher resolution or an ROI crop), not temporal pooling.
4. PI decision still open from the previous probe, now sharper: both the nav and the speed ceiling are ego-future channels on this corpus; any GO-skill number measured with them in the input is optimistic by construction (per the admissibility rule on goal inputs).

## 8. Post-freeze choices (`raw/POSTFREEZE_LOG.md`; none moved a verdict)
1. The `limit` input is an ego-future ceiling (section 2) -> Z_nl added beside Z_nn. 2. RAM sequencing: concurrent jobs (this package's own V4 extraction/PCAs) pushed free RAM to 1.3-1.8 GB; the 3 GB floor was kept for the V4/Vdiff PCA readers, extraction and eval; the light lin/mlp/pair stages (64-d scores, < 0.4 GB) ran at 1-1.5 GB; the first eval extraction (started at 3.5 GB free) was killed at 1.6 GB and re-run alone; the V4 PCA chain was stopped and resumed (per-fold files banked).
3. C2 null band (seeds 3-12) and its criterion (item 5-6): reported beside the frozen failure. 4. Unit-test file pins the one frozen failure as a failure. 5. `decide2.py` applies the thresholds to point estimates plus the CI-excludes-0 clause exactly as written; the best-of-two V4 selection is by linear fold-mean AUC on TRAIN.

## 9. Provenance and what is UNVERIFIED
* Reused: `tp_lib.py` copied byte-identical (cmp + sha256 in the test); `extract_z.load_model` imported unmodified from the previous package; windows, samples, banked OOF scores, PCA scores, rebuilt TRAIN join, eval features: read-only inputs, sha256 in `raw/INPUTS_SHA256.json` (29 files, checkpoint sha256 re-checked).
* Executed code: the exported tree `dagt1_tree_b43c7a9a` (as in the previous probe); the tip `c1c30fb7` was read by `git show` for the code citations. Their diff in `v8f_mini.py` / `v8f_mini_data.py` is training-side only (`launch_intent_w`, `launch_mask`, ego-perturb stream); N1 reproduces the banked Z to 1.9e-6.
* UNVERIFIED (inherited from the previous probe): the cache `F:/TanitAD-scratch/speedread/cache/train/clips` byte-for-byte against box 2's `D:\tanitad\cache_full`; the rebuilt TRAIN join against box 2's `join_train.npz`. UNVERIFIED: that the Z_nn gain on TRAIN is not memorisation (eval-141 is wide on P-STOP); whether a nav-free trunk (trained without the oracle inputs) would behave like Z_nn/Z_nl.
* One checkpoint, one seed: every Z statement is blind to training variance (H-ESTIM-SEED-1). The register row H-CRAWL-06-PROBE is the Master Mind's to update; nothing was staged, committed or pushed.
