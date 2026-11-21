# RESULT_V8F_SPEEDSUB: does the banked ego-only read-out, put into the five speed cells O reads, cut O's ADE6? Outcome **PASS** (a screening result at T1-class open loop)

Claim row `H-V8F-SPEEDSUB-1`. Pre-registration `Project Steering/PREREG_V8F_SPEEDSUB.md` (sha256 `1951a240…` at tip `ea5e3d3b`; no amendment; promoted by the Master Mind before any SPEEDSUB number). Instrument `stack/scripts/v8f_mini_eval_speedread.py`, sha256 `a95f4fb1…` (the landed blob and the run tree's copy hash to it, compared here). Run 2026-10-28 07:59:08 to 08:17:43 (18 min 35 s; dev-box CPU, 4 threads, `CUDA_VISIBLE_DEVICES=-1`, `logs/run.out`); the pre-registered reader `subread` ran 08:05:36–08:05:38 and printed `ZZSPEEDSUB-DONE PASS`; all **27 control flags PASS** in `SUBREAD.json` (sha256 `1fe7fcb3…`). The outcome word was not copied: section 1 recomputes it from the reader's rows, and `read/REDERIVE.json` recomputes the rows themselves from the per-window dumps. Every number below is MEASURED in this run unless marked INHERITED, ESTIMATED or DERIVED (arithmetic shown in `audit/NUMBERS_AUDIT.md`).

* **Tier.** SUBREAD.json `tier`: `T1-class open loop (X_s's own held S / T outputs; only the five speed cells O reads are replaced)`. `tier_gt`: `T0 (reads the eval GT): ceiling / control only, never a result`. `tier_regression`: `deliberate-regression arm (the read-out's cells shuffled across windows): a control, never a result`. So this is **T1-class open loop**, never closed loop: no recorded future action, and the `gt` arm is the T0 ceiling and a control, never a result. The unmodified harness stamps each dump `raw.tier` = `T0-open-loop-own-held-outputs (trunk-probe grid)`; the tier of this read is the instrument's stamp above, the tier of SPEEDCAL's and SPEEDREAD's ADE6 rows. **ADE6 at this tier is not a capability claim** (PREREG section 6): no ADE6 here is a driving-performance number.
* **Estimator.** SUBREAD.json `estimator`: `taniteval.ci.paired_episode_cluster_bootstrap, B=2000, seed 0; answers 'another draw of EPISODES' only (H-ESTIM-SEED-1)`. 141 episodes / 2,399 windows per seed, paired against `own`. It answers *"would another draw of EPISODES say this?"* and nothing else. It does not answer *"would another TRAINING RUN say this?"* (`H-ESTIM-SEED-1`): **there is no replicate arm in this experiment**, so the two seeds (O's two training runs) carry that question at n = 2. It does not answer *"would another READ-OUT FIT say this?"*: one fit (`readout.pt` sha256 `a205609a…`) serves both seeds. The planner is deterministic, so there is no inference-seed term.
* **Base and arms.** X_s = `M2b-mixo_s<s>` (the SPEEDCAL / KINDX / SPEEDREAD base), **not** DAG-OACC or DAG-ORC. s0 sha256 `c6a62876…`, s1 `68a0f335…`, each recorded in every dump (`speedread.x_sha256`). No model is trained. Modes per seed: `own` (X_s's cells), `identity` (own through the hook; control S1), `gt` (the GT cells where the window has a v9 row, else own; T0 ceiling), `ego` (**the arm**: `mlp_ego(ego block at the tick)`), `egoshuf` (the same cells shuffled across the 2,399 windows by `numpy.random.default_rng(20261029).permutation(2399)`; the deliberate-regression arm).
* **Declared open channel (`D-SPEEDBAND-EGO-ECHO-1`).** The PASS was read with the **logged `limit` lane**. The logged speed-limit bucket in the ego block is the bucket of the ego's own realised future max speed: the T1 tool (`stack/scripts/v8f_t1_eval.py`) lists it as an OPEN channel under `--limit logged`, and bin plus `v0` recovers it at R² 0.9702 (`Project Steering/GOALS_AND_CLAIMS.md:11518` at tip `ea5e3d3b`, read here; INHERITED claim). By PI ruling it is admissible as a DECLARED input, never a training signal, and T reads it too (PREREG section 2). `nav` is the other declared input in the block (a route input, admissible at training and inference by PI ruling). Section 4.5 says how much of the read-out's skill rests on each.
* **Coverage.** The four families are in section 3, per family and per seed, never pooled. Gaps are named there, per family, with their reason and n: distance keeping and the anti-echo block (LONGITUDINAL), anchor selection and selected-vs-executed manoeuvre (TACTICAL) are UNAVAILABLE in the harness path (not computed, not passed), and the `egoshuf` control has no family run. The outcome above uses none of them.

## 1. The outcome, as the pre-registered table decides it

| | s0 | s1 |
|---|---|---|
| **ARM** ADE6, `ego` − `own` (m) | **−0.1290 [−0.2107, −0.0548]** | **−0.0686 [−0.1151, −0.0232]** |
| ADE6 mean, `own` → `ego` (m) | 2.1977 → 2.0687 | 2.1201 → 2.0515 |
| separated-better (CI upper end < 0) | yes: upper end −0.0548 | yes: upper end −0.0232 |
| bar: point ≤ −F/2 = −0.06035 m | yes: 0.0687 m beyond the bar | yes: 0.0083 m beyond the bar |
| HARMS test: separated-worse (CI lower end > 0) | no: lower end −0.2107 | no: lower end −0.1151 |
| **REGRESSION ARM** `egoshuf` − `own` (must be separated-worse) | +6.7181 [+5.6709, +7.8141], separated-worse | +6.4923 [+5.3813, +7.6678], separated-worse |
| ADE6 mean, `own` → `egoshuf` (m) | 2.1977 → 8.9157 | 2.1201 → 8.6124 |
| **CEILING** `gt` − `own` (T0, control only) | −0.7164 [−0.8182, −0.6234]; mean 2.1977 → 1.4813 | −0.6676 [−0.7613, −0.5753]; mean 2.1201 → 1.4526 |
| CAPTURE = `ego` / (`gt` − `own`) | 0.1801 | 0.1028 |
| F/2 as a share of the `gt` ceiling (derived) | 8.4 % | 9.0 % |

Keys: `SUBREAD.json` `rows.<s>.ego | egoshuf | gt | capture | regression_arm_separated_worse`. The means are the dumps' `raw.traj.ade6_m.mean` (4 decimals), so a delta can differ from the difference of two printed means by 0.0001 (`egoshuf` − `own` at s0: 6.7180 from the means, +6.7181 from the paired cells). "Separated-better" is the CI's upper end below 0; "separated-worse" is its lower end above 0 (PREREG section 5). F = 0.1207 m is the cross-seed spread of M2b (INHERITED, `RESULT_V8F_GRID.md`; the instrument's `F_M`, read back in SUBREAD.json `F_m`).

The pre-registered conditions (PREREG section 5, REFUSED first, then HARMS, then PASS), restated against the rows. The reader's word in SUBREAD.json is the decision of record.

| outcome | condition | this run |
|---|---|---|
| REFUSED | a control failed, a seed is missing, or `egoshuf` is not separated-worse at both seeds | no: all 27 flags true, both seeds present, `egoshuf` lower ends +5.6709 / +5.3813 |
| HARMS | `ego` separated-worse at either seed | no: lower ends are below 0 at both seeds |
| **PASS** | at **both** seeds: separated-better **and** `ego_s ≤ −0.06035` m | **yes**: upper ends −0.0548 / −0.0232 and points −0.1290 / −0.0686 |
| NULL | anything else | not reached |

**PASS.** The script that built this document coded the table above from PREREG section 5 and applied it to the reader's rows: PASS, equal to SUBREAD.json `outcome` (`detail` null). `read/REDERIVE.json` (`read/rederive_sub.py`) recomputed the three paired rows per seed (`ego`, `egoshuf`, `gt`, each against `own`) from the per-window ADE6 of the dumps with `taniteval.ci.paired_episode_cluster_bootstrap` called directly (B = 2,000, seed 0), applied the same table, and got PASS with identical delta, lo and hi to 4 dp on all 18 values; the decision is the same whether the point is taken rounded or unrounded (unrounded points −0.1290 / −0.0686 m).

**Committed next, quoted from SUBREAD.json `next`:** *"T1-kin zero-training substitution of the banked ego read-out on the DAG-OACC / DAG-ORC base, both seeds, under its own PREREG (PREREG_V8F_SPEEDSUB section 6)"*. Section 6 says what that requires and what is in progress.

### How to read this PASS (reported as written; none of it changes the word)

1. **It is a screening result, by the PREREG's own terms.** The arm was commissioned after SPEEDREAD's beside table (which holds eval-141 RMSE) was read; the selection rule was restated on TRAIN fold 0 only before any SPEEDSUB number (control S13, section 2: mean fold-0 hold-out RMSE `mlp_ego` 0.2475 against `lin` 0.2724, `lin_ego` 0.2807, `mlp` 0.2920). It is still the third read-out examined on these 141 episodes (`lin` and `mlp` were run in SPEEDREAD; `lin_ego` was not), so PREREG section 0 and section 11 apply: *a PASS is a statement about two O training runs and one read-out fit at T1-class open loop; it is not a lever claim for any trained arm and not a driving result.*
2. **s1 clears the bar by 0.0083 m only** (point −0.0686 against −0.06035; 1.14 times the bar). At both seeds the CI's upper end (−0.0548 / −0.0232) lies **above** −0.06035, so the intervals do not exclude an effect smaller than the bar. The registered table tests the point estimate against the bar and the interval's sign against 0, not the interval against the bar; it is reported as registered. s0 is 2.14 times the bar.
3. **The PREREG's expectation about which condition binds did not hold.** Section 1 of the PREREG expected the separation condition to bind before the point bar, because SPEEDREAD's CI half-widths were 0.075 and 0.055. Here the CI half-widths are 0.078 (s0) and 0.046 (s1); at s1 the point bar binds (0.0083 m of margin) before separation (upper end −0.0232, 0.0232 m of margin). That sentence was an ESTIMATED prior, not a criterion.
4. **The two seeds' effects differ by 0.0604 m** (−0.1290 against −0.0686), about half of F. That is the only look this experiment has at training-run variance (n = 2, no replicate arm); it is a description, not an estimate of it. CAPTURE of the T0 ceiling is 0.1801 at s0 and 0.1028 at s1: `ego` recovers 18 % and 10 % of what the GT cells would give, and O still has most of the 0.7 m left on the table.
5. **What `ego` reads is wider than 'ego motion'** (PREREG section 2): the 22-dim ego block at the tick frame holds the 8 P10 kinematic features, the 5 `nav` channels (a route input; optimistic on PhysicalAI, where the only route supplier is the ego's own future path), the 4-wide speed-limit bucket and its validity (the logged `limit` lane, a declared open channel, see the header; in v8F's own `INFERENCE_INPUT_KEYS`) and the 4 history-validity flags. No ablation was fitted in this experiment (no re-fit is allowed in it), so **which input carries the ADE6 cut is not measured here**. The read-out's own skill was attributed afterwards by the prerequisites agent (section 4.5): it survives without `nav` and does not survive without `limit`.

## 2. Controls: each must read its known value before any row is read

The reader's flags are in `SUBREAD.json` `controls`; the right-hand column of each row is the value `read/REDERIVE.json` recomputed from the arrays (or the dump's own field), so the flag and its reading are both on the page. All 27 flags are true (11 per seed, then S11, S12, S13 once, and S6b per seed), and `egoshuf` reads separated-worse at both seeds (section 1), which is the regression arm's committed reading.

| id | must read (PREREG section 4) | s0 | s1 | measured detail (s0 / s1) |
|---|---|---|---|---|
| S1 | `identity` − `own` ADE6 max \|Δ\| = 0.0 and plans equal | yes | yes | ADE6 means `own` / `identity`: 2.1977 / 2.1977 ; 2.1201 / 2.1201; max \|Δ\| and plan equality recomputed: true |
| S2 | non-speed cells O reads = X_s's own, max \|Δ\| = 0.0, in `identity`, `gt`, `ego`, `egoshuf` | yes | yes | `speedread.hook.nonspeed_max_abs_dev` = 0.0 in all four hooked dumps at both seeds |
| S3 | the hook's tick slice = the model's own `_scene` pad mask (first batch) | yes | yes | `speedread.hook.tick_mirror_ok` true in all four hooked dumps at both seeds; 38 batches each |
| S4 | episode ids equal across all five modes | yes | yes | recomputed from the dumps' `eid`: true |
| S5 | `own` ADE6 = SPEEDREAD's banked `own`, per window max \|Δ\| ≤ 1e-5 m | yes | yes | max \|Δ\| 6.63e-06 / 7.79e-06 m (bar 1e-5; the run used 4 threads against SPEEDREAD's 8, which the bar allows) |
| S6 | `gt` ADE6 = SPEEDREAD's banked `gt`, per window ≤ 1e-4 m | yes | yes | max \|Δ\| 4.00e-06 / 4.57e-06 m |
| S6b | `gt` − `own` (delta, lo, hi) reproduces SPEEDREAD `READ.json` `rows.<s>.gt` within 5e-4 | yes | yes | −0.7164 [−0.8182, −0.6234] / −0.6676 [−0.7613, −0.5753], equal to the tip's SPEEDREAD READ.json |
| S7 | what O reads in `ego` = this run's identity `pred_mlp_ego`, standardised, ≤ 1e-5 | yes | yes | max \|Δ\| 0.0 / 0.0 (std units); the preflight after s0 read the same (S7 dev 0) |
| S7b | this run's identity `pred_mlp_ego` = SPEEDREAD's banked one, ≤ 1e-5 m/s | yes | yes | max \|Δ\| 0.0 / 0.0 m/s: the same read-out, checkpoint and windows SPEEDREAD scored |
| S8 | `readout_sha256` in every hooked dump = `fit.json` = the pin `a205609a…` | yes | yes | all ten dumps carry the pin; `readout.pt`, `fit.json` and the pin agree (recomputed) |
| S9 | `egoshuf`: permutation sha equals the registered one, alignment max \|Δ\| ≤ 1e-6, `n_seen` = 2,399, fixed points ≤ max(3, 1 %) | yes | yes | perm sha256 `64785f8a…` recomputed from the seed; alignment 0.0; `n_seen` 2,399; fixed points 0 |
| S10 | `ego` moves ≥ 99 % of windows' cells away from `own`'s; `egoshuf` differs from `ego` on ≥ 95 % | yes | yes | fraction moved 1.0000 / 1.0000; `egoshuf` differs from `ego` on 1.0000 / 1.0000 |
| S11 | train parity (4,512 / 306,566 / `f6a1dee0…`) | yes (one flag for both seeds) | | `fit.json` `train_parity_ok` true |
| S12 | eval-141 ∩ fit clips = ∅ | yes (one flag) | | `fit.json` `eval_train_disjoint` true |
| S13 | the section 0 selection rule applied to `fit.json` picks `mlp_ego` | yes (one flag) | | argmin of the mean fold-0 RMSE over the four read-outs = `mlp_ego` (values in section 1, point 1) |

Beside the S-series, the family tools' own controls (section 3) all hold at both seeds: `longlat` LL-1 (identity reads exactly 0.0 on every row), LL-2 (its ADE6 row equals `SUBREAD.json`, for `ego`, `egoshuf` and `gt`) and LL-3 (per-arm values equal the dump's own, to 1e-4), and `pairfam` (identity on 45 cells, reproduction of 119 point values per arm, ADE6 recompute max deviation 0.0).
LL-3's largest deviation is 4.99e-05 (bar 1e-4), the 4-decimal rounding of the dumps' stored values. The stop rule worked as designed: the preflight after s0 (08:02:24) read S7b dev 0, S7 dev 0, moved 1.0000 before seed 1 started.

## 3. The four families (beside the decision; ADE is one row of four; per family, never pooled)

The reader's table decides the outcome from ADE6 only. The binding eval standard asks for four families beside it, each with its estimator, so they were computed on the same dumps by the family tools named in the PREREG (section 9): `v8f_paired_families.py` (unmodified, from the run tree) for TACTICAL and STRATEGIC, and `families_longlat_sub.py` (a parametrised copy of SPEEDREAD's `families_longlat.py`, sha256 `f981ea7a…`) for LONGITUDINAL and LATERAL. They changed nothing in section 1.

* **Tier.** T1-class open loop (own held outputs; no recorded future action; the tools stamp the same words). The `gt` − `own` table is the T0 ceiling and a control, never a result.
* **Estimator.** Paired episode-cluster bootstrap, B = 2,000, seed 0, 141 episodes: another draw of EPISODES only. One training run per seed and no replicate arm, so a separated cell is a pointer (`H-ESTIM-SEED-1`).
* **STRATEGIC is reported, not paused, for v8F** (`strategic_family`: ACTIVE for v8F-mini).
* **Markers.** B = separated-better, W = separated-worse, – = not separated, S+ / S− = separated on a row with no better direction (a bias or a ratio). Bold = separated. Error rows are lower-better; accuracy and F1 rows are higher-better.

### `ego` − `own` (the arm; point [95 % CI])

| family | row | s0 | s1 | n windows s0 / s1 |
|---|---|---|---|---|
| PRIMARY | ADE6 (m) | **−0.1290 [−0.2107, −0.0548] B** | **−0.0686 [−0.1151, −0.0232] B** | 2,399 / 2,399 |
|  | ADE2 (m) | +0.0080 [−0.0033, +0.0181] – | +0.0039 [−0.0013, +0.0092] – | 2,399 / 2,399 |
| LONGITUDINAL | speed MAE (m/s) | **−0.0662 [−0.1122, −0.0245] B** | **−0.0379 [−0.0638, −0.0130] B** | 2,399 / 2,399 |
|  | along-track MAE (m) | **−0.1403 [−0.2266, −0.0641] B** | **−0.0793 [−0.1269, −0.0337] B** | 2,399 / 2,399 |
|  | accel MAE (m/s²) | **−0.0497 [−0.0866, −0.0194] B** | **−0.0295 [−0.0456, −0.0131] B** | 2,399 / 2,399 |
|  | DK-VIOL / DK-HW / DK-TG | UNAVAILABLE | UNAVAILABLE | n = 0 / 0; the dump's reason: "no lead-agent track supplied" |
|  | anti-echo (hold-v0 baseline) | UNAVAILABLE | UNAVAILABLE | n = 0 / 0; 2,399 windows available; the dump's reason: "no ego speed at t0 in the window dict" |
| LATERAL | heading MAE (deg) | **−0.1515 [−0.2900, −0.0303] B** | −0.0429 [−0.1509, +0.0491] – | 2,350 / 2,373 |
|  | yaw-rate MAE (deg/s) | −0.2434 [−0.6091, +0.0725] – | **−0.2440 [−0.5149, −0.0133] B** | 2,328 / 2,364 |
|  | curvature MAE (1/m) | −0.0041 [−0.0107, +0.0018] – | −0.0047 [−0.0105, +0.0005] – | 2,328 / 2,364 |
|  | cross-track MAE (m) | +0.0020 [−0.0112, +0.0143] – | +0.0021 [−0.0134, +0.0161] – | 2,399 / 2,399 |
|  | L1 lateral-accel MAE (m/s²) | **−0.0096 [−0.0165, −0.0028] B** | −0.0013 [−0.0048, +0.0020] – | 2,328 / 2,364 |
|  | L2 curvature MAE, GT ≥ 5.2 m/s (1/m) | −0.0107 [−0.0319, +0.0003] – | **−0.0065 [−0.0136, −0.0006] B** | 2,096 / 2,102 |
|  | L3 yaw-rate MAE, GT ≥ 5.2 m/s (deg/s) | **−0.5435 [−1.4547, −0.0667] B** | **−0.3193 [−0.6480, −0.0600] B** | 2,096 / 2,102 |
| TACTICAL | TAC-LAT (lateral-decision accuracy) | +0.0013 [−0.0054, +0.0084] – | +0.0038 [−0.0021, +0.0092] – | 2,399 / 2,399 |
|  | TAC-LON (longitudinal-decision accuracy) | −0.0042 [−0.0246, +0.0163] – | +0.0104 [−0.0100, +0.0313] – | 2,399 / 2,399 |
|  | 5-way manoeuvre accuracy | +0.0017 [−0.0183, +0.0221] – | +0.0108 [−0.0075, +0.0300] – | 2,399 / 2,399 |
|  | goal-point error (m) | **−0.3943 [−0.6422, −0.1651] B** | **−0.1950 [−0.3323, −0.0558] B** | 2,399 / 2,399 |
|  | goal longitudinal bias (m) | **+0.6398 [+0.3316, +0.9665] S+** | **+0.3587 [+0.1257, +0.6058] S+** | 2,399 / 2,399 |
|  | goal lateral bias (m) | **−0.0732 [−0.1386, −0.0091] S−** | +0.0109 [−0.0603, +0.0785] – | 2,399 / 2,399 |
|  | goal bearing MAE (deg) | −0.0133 [−0.2417, +0.2332] – | −0.0199 [−0.1127, +0.0583] – | 2,367 / 2,367 |
|  | goal range ratio | **+0.0153 [+0.0050, +0.0255] S+** | +0.0052 [−0.0100, +0.0208] – | 2,367 / 2,367 |
|  | lateral head macro-F1 | +0.0000 [+0.0000, +0.0000] – | +0.0000 [+0.0000, +0.0000] – | 2,107 / 2,107 |
|  | longitudinal head macro-F1 | +0.0000 [+0.0000, +0.0000] – | +0.0000 [+0.0000, +0.0000] – | 2,240 / 2,240 |
|  | goal head macro-F1 | +0.0000 [+0.0000, +0.0000] – | +0.0000 [+0.0000, +0.0000] – | 2,365 / 2,365 |
| STRATEGIC | S-POS@30 (m) | +0.0000 [+0.0000, +0.0000] – | +0.0000 [+0.0000, +0.0000] – | 2,322 / 2,322 |
|  | S-POS@50 (m) | +0.0000 [+0.0000, +0.0000] – | +0.0000 [+0.0000, +0.0000] – | 2,319 / 2,319 |
|  | S-POS@80 (m) | +0.0000 [+0.0000, +0.0000] – | +0.0000 [+0.0000, +0.0000] – | 2,294 / 2,294 |
|  | S heading@30 (deg) | +0.0000 [+0.0000, +0.0000] – | +0.0000 [+0.0000, +0.0000] – | 2,322 / 2,322 |
|  | S heading@50 (deg) | +0.0000 [+0.0000, +0.0000] – | +0.0000 [+0.0000, +0.0000] – | 2,319 / 2,319 |
|  | S heading@80 (deg) | +0.0000 [+0.0000, +0.0000] – | +0.0000 [+0.0000, +0.0000] – | 2,294 / 2,294 |

### `gt` − `own` (T0 ceiling, a control only; point [95 % CI])

| family | row | s0 | s1 | n windows s0 / s1 |
|---|---|---|---|---|
| PRIMARY | ADE6 (m) | **−0.7164 [−0.8182, −0.6234] B** | **−0.6676 [−0.7613, −0.5753] B** | 2,399 / 2,399 |
|  | ADE2 (m) | −0.0109 [−0.0235, +0.0011] – | **−0.0119 [−0.0196, −0.0041] B** | 2,399 / 2,399 |
| LONGITUDINAL | speed MAE (m/s) | **−0.3544 [−0.4059, −0.3064] B** | **−0.3424 [−0.3878, −0.2990] B** | 2,399 / 2,399 |
|  | along-track MAE (m) | **−0.7644 [−0.8685, −0.6692] B** | **−0.7122 [−0.8054, −0.6194] B** | 2,399 / 2,399 |
|  | accel MAE (m/s²) | **−0.0536 [−0.0970, −0.0140] B** | **−0.0303 [−0.0543, −0.0084] B** | 2,399 / 2,399 |
|  | DK-VIOL / DK-HW / DK-TG | UNAVAILABLE | UNAVAILABLE | n = 0 / 0; the dump's reason: "no lead-agent track supplied" |
|  | anti-echo (hold-v0 baseline) | UNAVAILABLE | UNAVAILABLE | n = 0 / 0; 2,399 windows available; the dump's reason: "no ego speed at t0 in the window dict" |
| LATERAL | heading MAE (deg) | +0.0808 [−0.1576, +0.3139] – | −0.0839 [−0.3452, +0.0962] – | 2,349 / 2,373 |
|  | yaw-rate MAE (deg/s) | +0.0570 [−0.4042, +0.4881] – | −0.0760 [−0.3837, +0.1851] – | 2,329 / 2,364 |
|  | curvature MAE (1/m) | +0.0005 [−0.0122, +0.0122] – | −0.0032 [−0.0099, +0.0027] – | 2,329 / 2,364 |
|  | cross-track MAE (m) | **−0.0378 [−0.0601, −0.0183] B** | **−0.0430 [−0.0657, −0.0231] B** | 2,399 / 2,399 |
|  | L1 lateral-accel MAE (m/s²) | −0.0044 [−0.0120, +0.0027] – | +0.0022 [−0.0029, +0.0067] – | 2,329 / 2,364 |
|  | L2 curvature MAE, GT ≥ 5.2 m/s (1/m) | **−0.0158 [−0.0381, −0.0042] B** | **−0.0108 [−0.0220, −0.0026] B** | 2,097 / 2,104 |
|  | L3 yaw-rate MAE, GT ≥ 5.2 m/s (deg/s) | **−0.7495 [−1.6944, −0.2279] B** | **−0.5456 [−1.0354, −0.1744] B** | 2,097 / 2,104 |
| TACTICAL | TAC-LAT (lateral-decision accuracy) | −0.0071 [−0.0192, +0.0042] – | −0.0042 [−0.0142, +0.0054] – | 2,399 / 2,399 |
|  | TAC-LON (longitudinal-decision accuracy) | **+0.1159 [+0.0875, +0.1450] B** | **+0.1446 [+0.1188, +0.1706] B** | 2,399 / 2,399 |
|  | 5-way manoeuvre accuracy | **+0.0805 [+0.0513, +0.1108] B** | **+0.0988 [+0.0738, +0.1254] B** | 2,399 / 2,399 |
|  | goal-point error (m) | **−2.2239 [−2.5417, −1.9337] B** | **−2.0637 [−2.3514, −1.7739] B** | 2,399 / 2,399 |
|  | goal longitudinal bias (m) | **+0.6815 [+0.1831, +1.2044] S+** | +0.2837 [−0.1308, +0.7242] – | 2,399 / 2,399 |
|  | goal lateral bias (m) | **−0.1294 [−0.2446, −0.0261] S−** | −0.0302 [−0.1403, +0.0734] – | 2,399 / 2,399 |
|  | goal bearing MAE (deg) | −0.1541 [−0.6931, +0.3614] – | −0.2168 [−0.5473, +0.0024] – | 2,367 / 2,367 |
|  | goal range ratio | −0.0080 [−0.0402, +0.0214] – | −0.0233 [−0.0522, +0.0034] – | 2,367 / 2,367 |
|  | lateral head macro-F1 | +0.0000 [+0.0000, +0.0000] – | +0.0000 [+0.0000, +0.0000] – | 2,107 / 2,107 |
|  | longitudinal head macro-F1 | +0.0000 [+0.0000, +0.0000] – | +0.0000 [+0.0000, +0.0000] – | 2,240 / 2,240 |
|  | goal head macro-F1 | +0.0000 [+0.0000, +0.0000] – | +0.0000 [+0.0000, +0.0000] – | 2,365 / 2,365 |
| STRATEGIC | S-POS@30 (m) | +0.0000 [+0.0000, +0.0000] – | +0.0000 [+0.0000, +0.0000] – | 2,322 / 2,322 |
|  | S-POS@50 (m) | +0.0000 [+0.0000, +0.0000] – | +0.0000 [+0.0000, +0.0000] – | 2,319 / 2,319 |
|  | S-POS@80 (m) | +0.0000 [+0.0000, +0.0000] – | +0.0000 [+0.0000, +0.0000] – | 2,294 / 2,294 |
|  | S heading@30 (deg) | +0.0000 [+0.0000, +0.0000] – | +0.0000 [+0.0000, +0.0000] – | 2,322 / 2,322 |
|  | S heading@50 (deg) | +0.0000 [+0.0000, +0.0000] – | +0.0000 [+0.0000, +0.0000] – | 2,319 / 2,319 |
|  | S heading@80 (deg) | +0.0000 [+0.0000, +0.0000] – | +0.0000 [+0.0000, +0.0000] – | 2,294 / 2,294 |

### What the rows say (beside; none of this moves the outcome)

Separated cells of `ego` − `own`, by seed (a 49-cell read: the 12 longlat rows plus the 37 family-tool cells that carry an interval; at a nominal 5 % about 2 to 3 would separate with no effect, PREREG section 9):

* **s0:** 11 of 49 cells separated: `ade6_m` −0.1290 [−0.2107, −0.0548] B; `speed_mae_mps` −0.0662 [−0.1122, −0.0245] B; `along_mae_m` −0.1403 [−0.2266, −0.0641] B; `accel_mae_mps2` −0.0497 [−0.0866, −0.0194] B; `heading_mae_deg` −0.1515 [−0.2900, −0.0303] B; `L1_lat_accel_mae_mps2` −0.0096 [−0.0165, −0.0028] B; `L3_yaw_rate_mae_fast_degps` −0.5435 [−1.4547, −0.0667] B; `tactical.goal_setting.goal_point_error_m` −0.3943 [−0.6422, −0.1651] B; `tactical.goal_setting.goal_long_bias_m` +0.6398 [+0.3316, +0.9665] S+; `tactical.goal_setting.goal_lat_bias_m` −0.0732 [−0.1386, −0.0091] S−; `tactical.goal_setting.goal_range_ratio` +0.0153 [+0.0050, +0.0255] S+.
* **s1:** 9 of 49 cells separated: `ade6_m` −0.0686 [−0.1151, −0.0232] B; `speed_mae_mps` −0.0379 [−0.0638, −0.0130] B; `along_mae_m` −0.0793 [−0.1269, −0.0337] B; `accel_mae_mps2` −0.0295 [−0.0456, −0.0131] B; `yaw_rate_mae_degps` −0.2440 [−0.5149, −0.0133] B; `L2_curvature_mae_fast_1pm` −0.0065 [−0.0136, −0.0006] B; `L3_yaw_rate_mae_fast_degps` −0.3193 [−0.6480, −0.0600] B; `tactical.goal_setting.goal_point_error_m` −0.1950 [−0.3323, −0.0558] B; `tactical.goal_setting.goal_long_bias_m` +0.3587 [+0.1257, +0.6058] S+.

1. **LONGITUDINAL: the three registered rows move, at both seeds.** Speed MAE, along-track MAE and accel MAE are separated-better at s0 and at s1 (SPEEDREAD's `lin` read moved none of them, INHERITED, `RESULT_V8F_SPEEDREAD.md` section 4). The `ego` change is 18.7 % (s0) and 11.1 % (s1) of the `gt` ceiling on speed MAE, 18.4 % / 11.1 % on along-track MAE (both in line with the ADE6 CAPTURE of section 1), but 92.7 % / 97.4 % on accel MAE: `ego` takes nearly the whole of the ceiling's accel gain, which is itself small (the ceiling moves accel MAE by only 0.05 and 0.03 m/s²). **Distance keeping (DK-VIOL / DK-HW / DK-TG) and the anti-echo block are not computed**: status UNAVAILABLE, n = 0, in every dump of both seeds (the scoring call was given no lead-agent track and no t0 speed). That is a work item, not a pass; until the adapter exists (about 60 lines, PREREG section 9) half of the LONGITUDINAL family is unmeasured for this arm.
2. **LATERAL: pointers only, none worse.** s0 separates heading MAE, L1 lateral-accel MAE and L3 yaw-rate MAE (GT ≥ 5.2 m/s), all separated-better; s1 separates yaw-rate MAE, L2 curvature MAE and L3 yaw-rate MAE, all separated-better. L3 is separated-better at both seeds (−0.5435 / −0.3193 deg/s). No lateral row is separated-worse at either seed; cross-track MAE is not separated (point +0.0020 / +0.0021 m).
3. **TACTICAL: the decisions do not move; the goal point does.** TAC-LAT, TAC-LON and the 5-way manoeuvre accuracy are not separated at either seed. The goal-point error is separated-better at both seeds (−0.3943 / −0.1950 m) and the goal longitudinal bias moves by +0.6398 / +0.3587 m (S+: a bias has no better direction; the base's own bias is −2.4016 m at s0 and −0.9872 m at s1, and `ego` reduces its magnitude). The three head macro-F1 rows read exactly +0.0000 [+0.0000, +0.0000] at both seeds: the substitution acts on the code O reads, after T has run, so T's heads cannot move. The goal-point and bias cells are read from O's plan, so they move with ADE6 and are not an independent confirmation of it. The per-token goal F1 rows are in `families/pairfam_ego_s0.md`; this family's gaps are in the table below.
4. **STRATEGIC: exactly zero, by construction.** All six S rows (S-POS@30/50/80, S heading@30/50/80) read +0.0000 [+0.0000, +0.0000] at both seeds, for `ego` and for `gt`. The substitution acts on the code O reads after S and T have run, so this is a property of the substitution site and is **not evidence about strategic quality**.
5. **The 2 s horizon does not improve.** ADE2 is not separated at either seed and its point estimate is positive (worse): +0.0080 [−0.0033, +0.0181] / +0.0039 [−0.0013, +0.0092] m. The gain is a long-horizon gain; the `gt` ceiling moves ADE2 by less than 0.02 m (table above).
6. **Read the separated cells as pointers.** They come from a 49-cell read at one nominal level, with no replicate arm. Several rows share one mechanism (speed MAE, along-track MAE, accel MAE, ADE6 and the goal-point error all follow the plan's longitudinal profile), so they are not independent confirmations of each other.
7. **Not run:** the `egoshuf` control has no family tables (the PREREG asks for its ADE6 row only, which section 1 reports); this is stated so its absence is not read as a pass.

### Gaps, per family: what is not computed, with its reason and n

The same gaps hold in all four family JSONs (`ego` and `gt`, both seeds).

| family | row not computed | status | n | reason |
|---|---|---|---|---|
| LONGITUDINAL | DK-VIOL / DK-HW / DK-TG | UNAVAILABLE | 0 | the scoring call was given no lead-agent track ("no lead-agent track supplied"). The lead block exists, but the reading tool's CLI is hard-wired to other arms: an adapter of about 60 lines (PREREG section 9, INHERITED); a work item, not built |
| LONGITUDINAL | anti-echo (hold-v0 baseline) | UNAVAILABLE | 0 of 2,399 windows | no ego speed at t0 in the window dict ("no ego speed at t0 in the window dict") |
| LONGITUDINAL | target-speed accuracy with an interval | point values only (section 4.4) | 2,399 windows | the family tools have no paired row for it |
| TACTICAL | anchor selection | UNAVAILABLE | 2,399 | each dump's arm commits to ONE path per window: there is no candidate fan and no selection to score |
| TACTICAL | selected vs executed manoeuvre | UNAVAILABLE | 2,399 | both label streams are executed manoeuvres; scoring a declared decision needs a tactical head's selection, which a trajectory dump does not carry |
| TACTICAL | goal per-token F1: `SPEED_BAND`, `TAKE_EXIT_L`, `TRAFFIC_LIGHT_REACT` | UNAVAILABLE | 0 / 695 / 695 labelled windows, 0 positives each | no positive label for the token on this grid, so its F1 is undefined |
| TACTICAL | goal per-token F1, 8 further cells | point values only, no interval | see `families/pairfam_ego_s0.md` | F1 undefined on at least one resample (no positive label in the drawn episodes) |
| TACTICAL | confusion over the manoeuvre classes | not in the tool's output (it reports accuracy and kappa) | 2,399 | a work item: the dumps carry the predictions, the tool does not print the matrix |
| LATERAL, STRATEGIC | none | computed | 2,096 to 2,399 (LATERAL); 2,294 to 2,322 (STRATEGIC) | STRATEGIC reads exactly 0.0000 by construction |
| all four | the `egoshuf` control's family tables | not run | | the PREREG asks for its ADE6 row only |
| all four | the T1-kin tier | not run for this arm | | it is the next arm (section 6) |

## 4. Beside (deciding nothing)

### 4.1 The read-out's own accuracy, cell by cell (eval RMSE to the GT, m/s)

Per-cell RMSE on the eval-141 windows where the GT cell is present (`cells.<s>.n_present`), `T_own` = X_s's own speed cells as O reads them, `mlp_ego` = the arm's cells (identical at the two seeds: the read-out does not depend on the seed). Recomputed from the arrays in `REDERIVE.json` to 1e-4; the values are SUBREAD.json `cells`.

| cell | n present | `T_own` s0 | `T_own` s1 | `mlp_ego` | cut vs `T_own` s0 | cut vs `T_own` s1 |
|---|---|---|---|---|---|---|
| `lon_v_target_ms` | 2,334 | 2.8940 | 2.7470 | 2.5404 | 12.2 % | 7.5 % |
| `v_a_ms` | 2,351 | 1.9190 | 1.3020 | 0.7517 | 60.8 % | 42.3 % |
| `v_end_ms` | 2,317 | 2.7953 | 2.6916 | 2.5570 | 8.5 % | 5.0 % |
| `v_lo_ms` | 2,351 | 2.2571 | 2.1425 | 1.8607 | 17.6 % | 13.2 % |
| `v_hi_ms` | 2,351 | 2.0927 | 1.8020 | 1.7143 | 18.1 % | 4.9 % |

`mlp_ego` is below `T_own` in all five cells at both seeds, as the PREREG's section 0 said it would be (that table is re-derived here from the arrays, not copied). **That did not hold for SPEEDREAD's ridge, which cut `v_a_ms` by 40–59 % and moved ADE6 by about 0** (INHERITED, PREREG section 0), so a lower cell RMSE was not a sufficient condition; here it coincides with an ADE6 cut at both seeds. Which of the five cells carries the cut is not known from this experiment (that is the question of the unrun PREREG_V8F_SPEEDCELL, section 6).

### 4.2 Post-hoc, not registered: the plan's rule-violation rates (point values and paired intervals)

This block is **not in the PREREG**. The harness scores every plan against constraints built from the GT (`consistency_a` in each dump, label CONS(a) in the stage logs: speed band, stop pass, turn, lead time-gap; `taniteval.v8f_consistency.consistency`, called in `v8f_mini_eval.py` `evaluate`), and the per-window violation flags are in the dumps. They were read after section 1 with the same paired estimator (B = 2,000, seed 0, 141 episodes), rate = share of the 2,399 windows that break the rule, **lower is better**. They decide nothing and are one more pointer family (about 5 rows per contrast, post-hoc), but they bear on how far an ADE6 gain can be trusted, so they are reported. The rates equal the dumps' own `raw.consistency_a.*.rate_all` (asserted).

| rule | rate `own` s0 / s1 | rate `ego` s0 / s1 | rate `gt` s0 / s1 | `ego` − `own` s0 | `ego` − `own` s1 |
|---|---|---|---|---|---|
| any rule | 0.1642 / 0.1926 | 0.1859 / 0.2118 | 0.0825 / 0.1238 | **+0.0217 [+0.0050, +0.0404] W** | **+0.0192 [+0.0038, +0.0346] W** |
| SPEED | 0.0975 / 0.1113 | 0.1075 / 0.1280 | 0.0375 / 0.0446 | +0.0100 [−0.0067, +0.0275] – | **+0.0167 [+0.0025, +0.0296] W** |
| STOP | 0.0196 / 0.0279 | 0.0217 / 0.0292 | 0.0104 / 0.0263 | +0.0021 [+0.0000, +0.0046] – | +0.0013 [+0.0000, +0.0029] – |
| TURN | 0.0067 / 0.0079 | 0.0075 / 0.0079 | 0.0067 / 0.0075 | +0.0008 [−0.0008, +0.0025] – | +0.0000 [−0.0017, +0.0025] – |
| LEAD | 0.0775 / 0.0959 | 0.0938 / 0.1121 | 0.0379 / 0.0654 | **+0.0163 [+0.0063, +0.0263] W** | **+0.0163 [+0.0054, +0.0283] W** |

**Reading.** The rate of plans that break at least one rule rises under `ego` at both seeds and is separated-worse at both (+0.0217 [+0.0050, +0.0404] / +0.0192 [+0.0038, +0.0346]), carried by the LEAD rule at both seeds (+0.0163 / +0.0163) and by SPEED at s1; the `gt` cells move the same rate the other way, down by 0.0817 / 0.0688 (T0 ceiling, a control). So the `ego` cells cut the mean trajectory error while making the plan break a GT-derived rule more often. This block does not alter the PASS (ADE6 is the registered row). It is the strongest reason for the next arm to carry a rule-violation guard beside ADE6 (section 6).

### 4.3 Post-hoc, not registered: ADE6 by event stratum (point values, no interval)

The harness reports the mean ADE6 inside three event strata (`raw.strata_ade6`: turn, stop, lead-brake windows; the PREREG section 7 'cheapest first, no run needed' item). The dumps keep the stratum **mean** only, not the window list, so there is **no interval** here: these are point values on small, possibly overlapping sets (the stop stratum is 81 windows), to be read as a direction only. Reported because on the stop windows `ego` and `gt` pull in opposite directions.

| stratum | n windows | `own` s0 / s1 | `ego` s0 / s1 | `gt` s0 / s1 | `ego` − `own` s0 / s1 | `gt` − `own` s0 / s1 |
|---|---|---|---|---|---|---|
| turn | 158 | 2.7870 / 3.0110 | 2.6156 / 2.7995 | 2.1449 / 2.3435 | −0.1714 / −0.2114 | −0.6421 / −0.6675 |
| stop | 81 | 2.1385 / 3.2723 | 2.4759 / 3.4485 | 1.4244 / 1.5420 | +0.3374 / +0.1762 | −0.7142 / −1.7303 |
| lead_brake | 319 | 2.4948 / 2.5914 | 2.3683 / 2.6309 | 1.7772 / 1.6645 | −0.1265 / +0.0395 | −0.7176 / −0.9268 |

**Reading.** On the 81 stop windows `ego` is *worse* than `own` at both seeds (+0.3374 / +0.1762 m, point values) while `gt` is better by 0.7142 / 1.7303 m; turn windows improve at both seeds, and lead-brake windows improve at s0 and read +0.0395 m at s1. If this survives a paired interval, the `ego` cells help on the turn windows and hurt on the stop windows, where the GT cells help. Getting those intervals needs the strata's window lists, which the harness builds inside `evaluate` and does not store: a small adapter, named in section 6, not built here.

### 4.4 Point values from the dumps: target-speed accuracy (no interval)

The binding LONGITUDINAL family names target-speed accuracy. The dumps carry it as the fraction of horizon **steps** (not windows) whose speed error is inside a band (`raw.families.longitudinal.target_speed_acc`; the bands are proposed reporting tolerances, not a gate). Point values only; no paired interval was computed.

| band | `own` s0 / s1 | `ego` s0 / s1 | `gt` s0 / s1 |
|---|---|---|---|
| within 0.5 m/s | 0.4547 / 0.4895 | 0.4779 / 0.5138 | 0.5820 / 0.6283 |
| within 1.0 m/s | 0.6557 / 0.6952 | 0.6773 / 0.7101 | 0.7786 / 0.8269 |
| within 2.0 m/s | 0.8398 / 0.8684 | 0.8546 / 0.8766 | 0.9240 / 0.9499 |

### 4.5 Attribution of the read-out's skill (FINDING_1; beside, changes no outcome)

The PREREG section 6 'required first' attribution was run by the prerequisites agent and is cited here, not re-run: `E:/Projects/TanitAD-artifacts/dagland/speedsub_t1_prep/FINDING_1_ATTRIBUTION.md` (sha256 `e89af4a0…`); artifact `F:/TanitAD-scratch/speedsub_t1_prep/refit_all.json` (sha256 `0f830665…`). MEASURED on TRAIN fold 0 (the hold-out of `fit.json`, 846 clips), not on eval-141 and not on ADE6. The `mlp_ego` family was refitted on subsets of the 22-dim block; the first row is a control refit that reproduces the banked `mlp_ego` (the script asserts the mean equals `fit.json`'s). Mean hold-out RMSE over the five cells, train-SD units, lower is better:

| design | dim | mean fold-0 RMSE (SD units) | cost against the full block |
|---|---|---|---|
| full block = `mlp_ego` as banked (control refit) | 22 | 0.24751 | none |
| no `nav`: `p10f + limit + limit_ok + frame_ok` | 17 | 0.26313 | +0.0156 |
| no `limit`: `p10f + nav + frame_ok` | 17 | 0.28032 | +0.0328 |
| neither: `p10f + frame_ok` | 12 | 0.30849 | +0.0610 |

**Reading.** The read-out's skill survives without `nav` and does not survive without `limit`. Dropping `nav` costs 0.0156 SD and the no-`nav` read-out (0.26313) is still below the best other banked read-out on this fold (`lin`, 0.2724); dropping `limit` costs 0.0328 SD and leaves the read-out (0.28032) above `lin`. Starting from `p10f + frame_ok` alone, adding `limit` closes 74 % of the gap to the full block and adding `nav` closes 46 % (derived from the four values; FINDING_1's three-seed figures are 73 % and 46 %). The MLP's own seed floor on these means is 0.0003 to 0.0013 over three seeds (FINDING_1 section 1, INHERITED), far below the gaps, so the ordering is not a seed effect; it is one hold-out fold, not an eval.

**What this does and does not say about the PASS.** It answers the route-supply worry of section 1 at the read-out level: `nav` is not the main carrier of the read-out's skill. The main carrier is `limit`, the logged bucket of the ego's own realised future max speed, the channel declared in the header. It does **not** say which input carries the ADE6 cut: no reduced read-out was substituted (they have narrower `xm / xs` than the 22 columns `subevals` hands a read-out; a column-map field is about 5 lines of instrument change, not built, FINDING_1 section 5). If the ADE6 cut follows the read-out's skill it rests mostly on a declared open channel; that is a hypothesis, not a measurement.

## 5. What this read cannot claim

* **Not a driving result, not closed loop.** T1-class open loop: X_s's own held S / T outputs, five cells replaced; ADE6 at this tier is not a capability claim (PREREG section 6). The T1-kin read (self-action open loop) is the first tier at which one can be made, and it has not been run for this arm.
* **Not a lever claim for any trained arm.** Two O training runs, one read-out fit, zero training. The episode-cluster bootstrap is blind to training variance (`H-ESTIM-SEED-1`): a separated interval from a one-seed arm is necessary, not sufficient, and this experiment has **no replicate arm**. The two seeds agree in sign and both clear the bar, which is the replication this design has; it does not measure the run-to-run spread.
* **Not a statement about DAG-OACC / DAG-ORC.** The base is M2b-mixo. DAG-trained O has seen its own rollouts, so its response to the five cells may differ (PREREG section 6, infeasible-as-written item 1).
* **Not an attribution of the ADE6 cut.** The 22-dim ego block carries `nav` (a route input, optimistic on PhysicalAI) and the logged `limit` lane beside the P10 kinematics. The read-out's skill was attributed (section 4.5: it rests mostly on `limit`), but no reduced read-out was substituted, so which input carries the ADE6 cut is unmeasured.
* **Read with an open channel.** The PASS was read with the logged `limit` lane, a channel declared under `D-SPEEDBAND-EGO-ECHO-1` (the bucket of the ego's own realised future max speed; admissible by PI ruling as a declared input, never a training signal; T reads it too). It is not a causal-bucket result: the T1-kin lane uses `--limit causal` (section 6).
* **Not a complete longitudinal read.** Distance keeping and the anti-echo block are UNAVAILABLE (n = 0), so the LONGITUDINAL family is half measured (section 3). The `egoshuf` control has no family run. TACTICAL cannot move by construction on the three head rows, and STRATEGIC cannot move at all.
* **A selection disclosure stands.** `mlp_ego` was commissioned after the eval-141 beside table was read; the TRAIN-only rule (S13) picks the same read-out, and this is the third read-out examined on these windows.
* **Disclosed design limits that still apply** (PREREG section 11): tick frame only, no ego history; `nav` and the logged `limit` bucket ride in the ego block; the fit used the eval's age cycle, not the trainer's random ages; the `egoshuf` permutation is global, so it tests harm against far windows, not near neighbours; `lin_ego` was fitted and is untested here.

## 6. Next (Rule Zero): the PASS leaves a committed next; the attribution is measured and the follow-up is being pre-registered

**Committed next, verbatim from SUBREAD.json `next`:** *"T1-kin zero-training substitution of the banked ego read-out on the DAG-OACC / DAG-ORC base, both seeds, under its own PREREG (PREREG_V8F_SPEEDSUB section 6)"*.

**PREREG_V8F_SPEEDSUB section 6, verbatim** (the Master Mind's design; the file at tip `ea5e3d3b`, sha256 `1951a240…`):

> ## 6. PASS: the T1-kin arm (the Master Mind's design; two parts are infeasible as written, flagged)
>
> Zero-training substitution of the banked `mlp_ego` on the **DAG-OACC / DAG-ORC** base, re-read at **T1-kin** with FLATDAG's rollout (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene; about 260 s per rollout, INHERITED from SPEEDREAD §6 citing FLATDAG). It needs its own PREREG with its own bar, both seeds and a replicate. What this draft can and cannot say:
>
> * **Infeasible as written, 1:** the base checkpoints' paths, shas and the rollout command are **not read in this draft** (UNVERIFIED). SPEEDSUB's PASS says nothing about them: DAG-trained O has seen its own rollouts, so the response to the five cells may differ from M2b-mixo's.
> * **Infeasible as written, 2:** the banked read-out was fitted with the eval's age cycle on M2b's `pool4x10` inputs and T's `--kin-mode yaw`. A DAG base must share that kin mode and cell standardisation (`cons_mu / cons_sd`); that is checkable and must be checked first (pair-compatibility, as SPEEDAVG's `pair_violations`).
> * **Required first:** the attribution the 22-dim block hides. The read-out can be re-fitted on `train/train_arrays.npz` (ego, targets and masks are stored; no extraction is needed, ESTIMATED minutes of CPU) as `p10f + frame_ok` only, and as `p10f + limit + frame_ok` without `nav`. A PASS carried by `nav` is a route-supply result, not a speed-estimation result.
> * **Deployable form** (as SPEEDREAD §6 describes): the read-out as a `forward_infer` speed sub-head, buffers absent by default so every earlier checkpoint stays bit-identical.
> * ⛔ ADE6 at T1-class open loop is not a capability claim; the T1-kin read is the first tier at which one can be made.

**Status of that section's prerequisites at the time of this document (done by the prerequisites agent, not by this document):**

| prerequisite (PREREG section 6) | status |
|---|---|
| the attribution the 22-dim block hides: re-fit the read-out as `p10f + frame_ok` only, and as `p10f + limit + frame_ok` without `nav`, on `train/train_arrays.npz` | **done: FINDING_1 (MEASURED; section 4.5).** The skill survives without `nav`, not without `limit` |
| pair-compatibility of a DAG base with the banked read-out (the same `--kin-mode yaw` and the same `cons_mu / cons_sd` cell standardisation; checkable, as SPEEDAVG's `pair_violations`) | reported finished by the prerequisites agent; **its result is not cited in this RESULT** |
| the base checkpoints' paths, shas and the rollout command (UNVERIFIED in the PREREG) | reported finished by the prerequisites agent; **its result is not cited in this RESULT** |

**The T1-kin lane uses `--limit causal`, so the follow-up refits the read-out on the causal bucket.** The T1-kin rollouts of the DAG line feed `limit` as a causal bucket, `speed_max_bin` of the max measured ego speed over a trailing window of rows (20 rows, FINDING_1 section 3), computed by `causal_bucket` in `stack/scripts/v8f_t1_eval.py`, which reads nothing after the current row (the function and its docstring are read at tip; the lane assignment is INHERITED from FINDING_1 and the coordinator). It is not the logged sidecar the banked read-out was fitted on. The banked weights on `limit` were learned against the logged bucket, the future-speed oracle, so under `--limit causal` the read-out's strongest input changes meaning; FINDING_1 estimates the skill would sit between the neither design and the full block (ESTIMATED, not measured). The follow-up is therefore a new TRAIN fit of the read-out on the causal bucket, recomputed per window from `native_tracks.pt` (an extract-like step over the 306,566 train windows, not done; whether the tracks cover every train clip is not checked, FINDING_1 section 3). The driving model stays untrained, so the substitution stays training-free for O; what PREREG section 6 (quoted above) calls the banked read-out becomes a refitted one. **That follow-up is pre-registered as `PREREG_V8F_SPEEDSUB_T1` (landed `842ec2ac`, before its read-out was fitted and before any T1-kin substitution rollout).**

What is not blocked and can proceed in the meantime, zero GPU, in the order of the largest expected information per hour (these are suggestions of this document for the Master Mind; none is a registered criterion, and the committed next above is unchanged):

1. **Carry a rule-violation guard into the T1-kin PREREG.** The plan's rule-violation rate rose under `ego` at both seeds (section 4.2), separated-worse, while ADE6 fell. A T1-kin PASS that cut ADE6 and raised this rate would be a different result from one that did not; pre-register the guard and its direction before the arm runs.
2. **Paired intervals on the stop windows.** Section 4.3's point values put the `ego` loss on the stop windows and the `gt` gain there too. The strata's window lists are not in the dumps; an adapter that stores them (or recomputes `event_strata` from the same inputs) turns a pointer into a paired row. Cheap, CPU, minutes of compute after the adapter.
3. **The distance-keeping adapter** (about 60 lines, PREREG section 9). Until it exists, DK-VIOL / DK-HW / DK-TG are not measured at this tier (the T1-class harness path). The T1-kin readers of the DAG line carry DK rows, so the T1-kin arm's DK half may come from those; whether they accept this arm is not checked here.
4. **PREREG_V8F_SPEEDCELL is not triggered by a PASS** (PREREG section 7 names it for NULL or HARMS). It is still the cheapest way to learn which of the five cells carries the cut and whether `ego` helps or hurts per cell (ESTIMATED 1.0 to 1.5 CPU-h at 4 threads, PREREG section 7). Whether to run it before T1-kin is the Master Mind's call.

**What blocks the committed next:** its own PREREG (own bar, both seeds, a replicate; PREREG section 6), which that section says needs the attribution (done) and the pair-compatibility check first; it is pre-registered as `PREREG_V8F_SPEEDSUB_T1` (landed `842ec2ac`), and the causal-bucket refit above is part of it; its instrument is being built. No compute, spend or PI decision is named as a blocker by the PREREG for this zero-training step. The rollout cost is INHERITED (about 260 s per rollout, SPEEDREAD section 6 citing FLATDAG).

## 7. Run record and provenance (artifact path, sha256)

**Run record** (`logs/run.out`, `logs/speedsub_after_t1.log`). No stop, no refusal, no re-run: every stage ran once, in order.

| stage | start | end | wall |
|---|---|---|---|
| waiter | 04:34:06 | 07:59:08 | the waiter held `run_speedsub.sh` until the DAG-ORC T1 chain finished and 7,138 MB were free (`logs/speedsub_after_t1.log`) |
| `evals_s0` (5 modes) | 07:59:11 | 08:02:24 | 3 min 13 s |
| `preflight_s0` | 08:02:24 | 08:02:24 | S7b, S7 and the moved fraction, before seed 1 |
| `evals_s1` (5 modes) | 08:02:24 | 08:05:36 | 3 min 12 s |
| `subread` | 08:05:36 | 08:05:38 | 2 s |
| families (`pairfam` x 4, `longlat` x 4) | 08:05:38 | 08:17:43 | 12 min 05 s |

No SPEEDSUB stage stopped. The waiter log also carries a stop of the DAG-ORC T1 chain at 06:05:21 (a box-2 run-record control; that is not this run) and the waiter's re-arm at 06:11:48.

The run script is `raw/2026-10-28-speedsub-draft/run/run_speedsub.sh` (sha256 `8255f69f…`, already in the repo and byte-identical to the package copy); the executed copy of the waiter's invocation is not separately hashed in the logs (UNVERIFIED which byte-copy the waiter called, the stage outputs match the script's stages). The run tree is `F:/TanitAD-scratch/speedsub_tmp/tree` (an overlay of SPEEDREAD's `f3971165` tree with the two staged scripts; `run/make_tree.sh`).

Repo paths are under `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/` and are staged by the hold (`landing/mk_hold_speedsub_result.py`). Source paths are where the bytes were read from.

| role | source path | repo path | sha256 |
|---|---|---|---|
| reader output (decision of record) | `F:/TanitAD-scratch/speedsub/subread/SUBREAD.json` | `raw/2026-10-28-speedsub-result/read/SUBREAD.json` | `1fe7fcb366e8123721d82891408c6d3c15948745173deb87f1ddeb5a4c53826d` |
| reader outcome | `F:/TanitAD-scratch/speedsub/subread/OUTCOME.txt` | `raw/2026-10-28-speedsub-result/read/OUTCOME.txt` | `178778b3294c13ac753756bc16043ac86115b43b4c08e33da5d8ec1701ba1ecc` |
| independent re-derivation (this package) | `E:/Projects/TanitAD-artifacts/dagland/speedsub_result/REDERIVE.json` | `raw/2026-10-28-speedsub-result/read/REDERIVE.json` | `f72573fc7eecff2d8b79a283761cc2a284d166c7e24415fa0917221f61cf8ee6` |
| re-derivation script | `E:/Projects/TanitAD-artifacts/dagland/speedsub_result/rederive_sub.py` | `raw/2026-10-28-speedsub-result/read/rederive_sub.py` | `3d2cd6a0661adf038a6b6f51af77f2b2f82995046f60e51edb9159e06d2e4c3e` |
| re-derivation console output | `E:/Projects/TanitAD-artifacts/dagland/speedsub_result/rederive_sub.out` | `raw/2026-10-28-speedsub-result/read/rederive_sub.out` | `a3fc0a3636ec6f2c87224be972542bd2d6a72790551909ec4ad48e054da76836` |
| eval dump s0 own | `F:/TanitAD-scratch/speedsub/eval_s0/eval_M2b-mixo_s0-speedsub-own.json` | `raw/2026-10-28-speedsub-result/eval_s0/eval_M2b-mixo_s0-speedsub-own.json` | `ee7a8aa247ce81372101dbb660788d03174c13143935192474861e3480ed4f89` |
| eval dump s0 identity | `F:/TanitAD-scratch/speedsub/eval_s0/eval_M2b-mixo_s0-speedsub-identity.json` | `raw/2026-10-28-speedsub-result/eval_s0/eval_M2b-mixo_s0-speedsub-identity.json` | `da62621905421d9e70972c00de3da846b424a1aae96e5c457e8565db0294e36a` |
| eval dump s0 gt | `F:/TanitAD-scratch/speedsub/eval_s0/eval_M2b-mixo_s0-speedsub-gt.json` | `raw/2026-10-28-speedsub-result/eval_s0/eval_M2b-mixo_s0-speedsub-gt.json` | `18fbb0c4130eb4d7b0193a007c02a35d8053fd28498c243ea42f4dab2457d402` |
| eval dump s0 ego | `F:/TanitAD-scratch/speedsub/eval_s0/eval_M2b-mixo_s0-speedsub-ego.json` | `raw/2026-10-28-speedsub-result/eval_s0/eval_M2b-mixo_s0-speedsub-ego.json` | `f55a661a2ef90aa9686fa3c39898ddd79ad210cbe12532a0c6b6af83e58760c3` |
| eval dump s0 egoshuf | `F:/TanitAD-scratch/speedsub/eval_s0/eval_M2b-mixo_s0-speedsub-egoshuf.json` | `raw/2026-10-28-speedsub-result/eval_s0/eval_M2b-mixo_s0-speedsub-egoshuf.json` | `8a6c265c717a403d2045bdd8f082b0355870a619da61767999ed56a4be1947c3` |
| eval dump s1 own | `F:/TanitAD-scratch/speedsub/eval_s1/eval_M2b-mixo_s1-speedsub-own.json` | `raw/2026-10-28-speedsub-result/eval_s1/eval_M2b-mixo_s1-speedsub-own.json` | `09af27166296fdeaf3235601d42a3f637430b9152c1f361d181c889b7c7585e7` |
| eval dump s1 identity | `F:/TanitAD-scratch/speedsub/eval_s1/eval_M2b-mixo_s1-speedsub-identity.json` | `raw/2026-10-28-speedsub-result/eval_s1/eval_M2b-mixo_s1-speedsub-identity.json` | `fde634c3cc829d1a05156008e25a718c65b40cb3c4aaaa1ca4d2dd57e6d4de2f` |
| eval dump s1 gt | `F:/TanitAD-scratch/speedsub/eval_s1/eval_M2b-mixo_s1-speedsub-gt.json` | `raw/2026-10-28-speedsub-result/eval_s1/eval_M2b-mixo_s1-speedsub-gt.json` | `6c3469f45277ae7dee8778d04c7612a1ed4ba940a76186c6f9c03128c140884d` |
| eval dump s1 ego | `F:/TanitAD-scratch/speedsub/eval_s1/eval_M2b-mixo_s1-speedsub-ego.json` | `raw/2026-10-28-speedsub-result/eval_s1/eval_M2b-mixo_s1-speedsub-ego.json` | `b27a301974771109bdd65b420ddf1bbd22ff602f5d7879779ebc96b08dc3de4c` |
| eval dump s1 egoshuf | `F:/TanitAD-scratch/speedsub/eval_s1/eval_M2b-mixo_s1-speedsub-egoshuf.json` | `raw/2026-10-28-speedsub-result/eval_s1/eval_M2b-mixo_s1-speedsub-egoshuf.json` | `ae20c525aa9359fe69a83933540bf1ffbb867e9e7a6bb72e1af3a3d951957721` |
| longlat s0 (json) | `F:/TanitAD-scratch/speedsub/families/longlat_s0.json` | `raw/2026-10-28-speedsub-result/families/longlat_s0.json` | `9f53abbe29ed2d0941ccddd28b87311a63791c6ee2c91363f438480c92eda059` |
| longlat s0 (out) | `F:/TanitAD-scratch/speedsub/families/longlat_s0.out` | `raw/2026-10-28-speedsub-result/families/longlat_s0.out` | `724b56f83b342aa3d23cfd97c97d7ce649c75318e009ce569ef19191ab4792d6` |
| pairfam ego s0 (json) | `F:/TanitAD-scratch/speedsub/families/pairfam_ego_s0.json` | `raw/2026-10-28-speedsub-result/families/pairfam_ego_s0.json` | `ad327b04d4a1dfe2162aad1cc02f732eb1157e6a4929917eb4fecd219e61d1c9` |
| pairfam ego s0 (md) | `F:/TanitAD-scratch/speedsub/families/pairfam_ego_s0.md` | `raw/2026-10-28-speedsub-result/families/pairfam_ego_s0.md` | `f4191829c9ddd32339eac3a80f2219179e7c29c73740157f1ef968fa3bf0b7a5` |
| pairfam ego s0 (out) | `F:/TanitAD-scratch/speedsub/families/pairfam_ego_s0.out` | `raw/2026-10-28-speedsub-result/families/pairfam_ego_s0.out` | `711ba2224f1e7adb24fcbee0842aac7316f79cc2847ef719899eab7e73446ba1` |
| pairfam gt s0 (json) | `F:/TanitAD-scratch/speedsub/families/pairfam_gt_s0.json` | `raw/2026-10-28-speedsub-result/families/pairfam_gt_s0.json` | `548cc5f5cd04a46e05005f3b38d9915a6cad21eaae85ed25ca2ede6cdc1d7f8a` |
| pairfam gt s0 (md) | `F:/TanitAD-scratch/speedsub/families/pairfam_gt_s0.md` | `raw/2026-10-28-speedsub-result/families/pairfam_gt_s0.md` | `f8cca34af1602d8914891987f0d2e4cab07b96104111d021ef0e675c03a7975a` |
| pairfam gt s0 (out) | `F:/TanitAD-scratch/speedsub/families/pairfam_gt_s0.out` | `raw/2026-10-28-speedsub-result/families/pairfam_gt_s0.out` | `51710927620f746e2782f798c8dcf81a1fa5df3265fdba5627408b44b1c43da5` |
| longlat s1 (json) | `F:/TanitAD-scratch/speedsub/families/longlat_s1.json` | `raw/2026-10-28-speedsub-result/families/longlat_s1.json` | `805e772dd9f5ca99509578eb9968e7563a5ad3a57afd5e9c3f9153f6356ebc62` |
| longlat s1 (out) | `F:/TanitAD-scratch/speedsub/families/longlat_s1.out` | `raw/2026-10-28-speedsub-result/families/longlat_s1.out` | `9def974e7c89301ee1708e55044ab71187f6bb83df1b562a8de331c906bcf8a1` |
| pairfam ego s1 (json) | `F:/TanitAD-scratch/speedsub/families/pairfam_ego_s1.json` | `raw/2026-10-28-speedsub-result/families/pairfam_ego_s1.json` | `fac7cb3452030e17dcc4c9ad4a5dd26639b2815d3fbb60dc86736dab94d97ccc` |
| pairfam ego s1 (md) | `F:/TanitAD-scratch/speedsub/families/pairfam_ego_s1.md` | `raw/2026-10-28-speedsub-result/families/pairfam_ego_s1.md` | `cd59202f503b0f7583059925cb747facb8e8984e54452499fdcb9b7140316b73` |
| pairfam ego s1 (out) | `F:/TanitAD-scratch/speedsub/families/pairfam_ego_s1.out` | `raw/2026-10-28-speedsub-result/families/pairfam_ego_s1.out` | `0dd2bf032f5d5f1d08e448f66fca912705d97d07495b556adec1a9a9e675df7d` |
| pairfam gt s1 (json) | `F:/TanitAD-scratch/speedsub/families/pairfam_gt_s1.json` | `raw/2026-10-28-speedsub-result/families/pairfam_gt_s1.json` | `3228ea21a29d7593e59be7ab137afed62883b8c1c4dbea39b28c23fcff1b1292` |
| pairfam gt s1 (md) | `F:/TanitAD-scratch/speedsub/families/pairfam_gt_s1.md` | `raw/2026-10-28-speedsub-result/families/pairfam_gt_s1.md` | `9a4b6df80b5832282312ff2f25cc8d063b11866c630108977d75d5d22d488e2b` |
| pairfam gt s1 (out) | `F:/TanitAD-scratch/speedsub/families/pairfam_gt_s1.out` | `raw/2026-10-28-speedsub-result/families/pairfam_gt_s1.out` | `6616bd7755d17f7108043751a082b0245532932dc513b9d40cedfe45c60346db` |
| log run.out | `F:/TanitAD-scratch/speedsub/run.out` | `raw/2026-10-28-speedsub-result/logs/run.out` | `af63ed7acc294ddbb82db65a3e76171bf14cb34fe42a2dc677ae02a3089d0a93` |
| log evals_s0.log | `F:/TanitAD-scratch/speedsub/evals_s0.log` | `raw/2026-10-28-speedsub-result/logs/evals_s0.log` | `5a9a29f83ea518bf2469aa79a5fdc2b815cfe991e0b87e62726f8e326c879908` |
| log evals_s1.log | `F:/TanitAD-scratch/speedsub/evals_s1.log` | `raw/2026-10-28-speedsub-result/logs/evals_s1.log` | `3c0da1673f399de8dfe1d6d9e77c23ef2c70a04917695db9bf2426db78a5f026` |
| log preflight_s0.log | `F:/TanitAD-scratch/speedsub/preflight_s0.log` | `raw/2026-10-28-speedsub-result/logs/preflight_s0.log` | `1f924e5b03a4e33dcb3fd0ccd34479275864d62306b9da6407e71843e467e521` |
| log subread.log | `F:/TanitAD-scratch/speedsub/subread.log` | `raw/2026-10-28-speedsub-result/logs/subread.log` | `ec15fefbecb038aeb7d60dd136f5c3b9e1cc74352fcc5ed02268bf75fd59821c` |
| waiter log | `F:/TanitAD-scratch/speedsub_after_t1.log` | `raw/2026-10-28-speedsub-result/logs/speedsub_after_t1.log` | `3b494349d8279431491096633921d5b2ef533aae639a259711274eef1e9da973` |
| run wrapper generated by the run script | `F:/TanitAD-scratch/speedsub/run_with_taniteval.py` | `raw/2026-10-28-speedsub-result/run/run_with_taniteval.py` | `6f5bcf10390230085bcdb8bbe97f9d97032bd9d0b5a61184b23b6c18d9e4e81e` |
| PREREG at the tip (already in the repo) | `git show ea5e3d3b:Project Steering/PREREG_V8F_SPEEDSUB.md` | `(repo root) Project Steering/PREREG_V8F_SPEEDSUB.md` | `1951a240196330401abe54416c5c224026f114cb3605d8be0dce7e97d127c432` |
| instrument (already in the repo) | `F:/TanitAD-scratch/speedsub_tmp/tree/stack/scripts/v8f_mini_eval_speedread.py` | `(repo root) stack/scripts/v8f_mini_eval_speedread.py` | `a95f4fb14da3fd8f6f89e55dfcb23d5d26a81b09dd24285aba394f36546634ca` |
| family tool, LONGITUDINAL / LATERAL (already in the repo) | `F:/TanitAD-scratch/speedsub_tmp/tree/stack/scripts/families_longlat_sub.py` | `raw/2026-10-28-speedsub-draft/families/families_longlat_sub.py` | `f981ea7a44d86ee12528e29ca2e0ea67947c557e2a8aee0bf1cd16d935a316a8` |
| SPEEDREAD READ.json (already in the repo; S5 to S7b inputs) | `F:/TanitAD-scratch/speedread/read/READ.json` | `raw/2026-10-28-speedread/read/READ.json` | `46543c392ee412c76edab48ec95c993e11ba621cc43188ee8e25f7b966d4579f` |
| SPEEDREAD fit.json (already in the repo) | `F:/TanitAD-scratch/speedread/fit/fit.json` | `raw/2026-10-28-speedread/fit/fit.json` | `c2e51ac204e5284dac5546ed80f03a3762f569f482a3e498c9c9ae12b4f04ae7` |
| FINDING_1 attribution (the prerequisites agent's file; not staged by this hold) | `E:/Projects/TanitAD-artifacts/dagland/speedsub_t1_prep/FINDING_1_ATTRIBUTION.md` | none | `e89af4a0a24642682c51f4fad593c7a56681948897071a9cc6b49cd3ffc8e8f0` |
| attribution refits (not staged by this hold) | `F:/TanitAD-scratch/speedsub_t1_prep/refit_all.json` | none | `0f830665eb616909c400265e7e6658d2094c2840fc98c7ef9d36f4bbf9baa381` |
| read-out `readout.pt` (not staged; single copy on F:) | `F:/TanitAD-scratch/speedread/fit/readout.pt` | none | `a205609ab47325a0ad9c0141be8ae3a0fec653d59c6db17ec1cf0d460bf2316d` |
| checkpoint X_0 `M2b-mixo_s0` (not staged) | `C:/Users/Admin/box2_ship/logs/v8f_mixo/runs/M2b_s0_mixo/ckpt_last.pt` | none | `c6a628762790b4fc3f7634867fc15b8167b035692ff518203c9fc0b3c126250d` |
| checkpoint X_1 `M2b-mixo_s1` (not staged) | `C:/Users/Admin/box2_ship/logs/v8f_kindx_s1/runs/M2b_s1_mixo/ckpt_last.pt` | none | `68a0f3356c37b25ce15b5f860f8dc919e23efc8b6bf200a2997f9ad6b521e915` |

**Not in the repo, single copy on F: (dev-box scratch).** The per-mode `eval_*.npz` and `speedread_cells_*.npz` dumps under `F:/TanitAD-scratch/speedsub/eval_s{0,1}/` (1.8 to 2.3 MB and 0.1 to 0.2 MB each; `REDERIVE.json` was computed from them and the family JSONs record the `ego` / `own` / `gt` npz sha256 of their inputs), SPEEDREAD's banked `eval_*.npz`, `speedread_cells_*-identity.npz` and `fit/readout.pt` (controls S5, S6, S7b read them), the run tree, and the checkpoints. If F: is lost, controls S5, S6 and S7b and the family tools cannot be reproduced; the JSONs, logs and this audit remain.

The in-repo instrument tests are INHERITED from the draft package (`raw/2026-10-28-speedsub-draft/`, PREREG section 10): 66 tests pass (39 new, 27 SPEEDREAD), 31 of the 39 new tests fail on the unmodified instrument, 28 of 28 mutants killed. They were not re-run for this document.
