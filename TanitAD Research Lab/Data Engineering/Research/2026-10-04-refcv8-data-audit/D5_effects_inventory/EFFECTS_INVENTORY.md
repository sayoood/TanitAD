# EFFECTS_INVENTORY — every measured effect and defect of refcv7-r101-s0 (final step 50,400, launch tree fec3a0d)

Stream D5 of the refcv8 data audit · Data Engineering · 2026-10-04 · read-only, zero GPU, nothing staged or committed.
PI request: *"understand all effects, define measures and validate them … retrain without losing a lot of time."*

## 0. How to read this file

**Stamp on every row unless the row says otherwise.** Evidence class: **M** = MEASURED (ours, artifact path given) ·
**M\*** = MEASURED by D5 in this package, re-derived from the raw artifact (`raw/d5_verify.json`, 109/109 checks, 0 unreadable
files) · **I** = INHERITED (a document's claim, not re-verified) · **E** = ESTIMATED · **H** = HYPOTHESIS.
Tier: **OL** = open loop on logged frames (single-shot plan, held-out eval139 clips, one DDIM draw). **OL-perc** = open-loop
perception diagnostic. **NS-OL** = NAVSIM open-loop benchmark (T1-family; zero-shot PhysicalAI-AV B1 -> nuPlan cameras; non-parity;
stage-1 loop OPEN). **Nothing in this file is closed loop; T2 is not provisioned.**
Estimator: **EC** = episode-cluster bootstrap (B stated; paired where stated) — answers *"another draw of EPISODES"* only.
**LC** = NAVSIM log-cluster bootstrap, B = 2000. Every interval is blind to **training-run variance** (one training seed, `H-ESTIM-SEED-1`);
only the inference-sampler floor is measured (INS-3). A separated interval is necessary, not sufficient, for a lever claim.
Every planner row is from the launch tree, where **the speed ceiling does not reach the emitted plan** (SPEC_REFCV7 §26.1, SPD-3).

**Path aliases (all repo-relative unless marked):**
`RF` = `TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv7-route-following/` ·
`MB` = `TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv7-map-box-diagnostics/` ·
`BAT` = `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/` ·
`NAV` = `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/` (`NAV/5k`, `NAV/30k` = `raw/milestones/step5000`, `step30000`) ·
`REG` / `CLM` / `RET` = `Project Steering/MODEL_REGISTRY.md` (refcv7 section, lines 5278–5293) / `GOALS_AND_CLAIMS.md` (block `REFCV7-2026-10-04-FINAL`, lines 16235–16250) / `RETRACTION_LOG.md` (R27, R28), **read from branch tip 0796180** ·
`REEL` = `taniteval/tools/RENDER_REFCV7_VIDEO.md` §7 · `RPL` = `taniteval/tools/RENDER_REFCV7_REPLAY.md` §8 ·
`CFG` = `D:/refcv7_eval_kit/ckpt/config.json` (md5 e6512a01…, the launch record) ·
`LBL` = label file `s2_labels_v8_train.jsonl.gz` md5 `b45377a1…` (= the md5 `CFG` records) · `D5V` = `D5_effects_inventory/raw/d5_verify.json`.

**Status codes.** `FIXED-INF` fixes at inference/readout with no training · `RETRAIN` needs a training change · `OPEN` cause or lever not yet identified ·
`PENDING` the measurement is still running.
**Rungs (PLAN_REFCV8 §6).** `R0` zero-GPU (CPU on banked artifacts/labels) · `R0-inf` inference-only forward, no training · `R1` frozen-trunk head-only probe on cached
decoder inputs (decides branch vs full retrain) · `R2` v7-tiny ladder arm with a deliberate-regression arm · `R4` branch fine-tune from 50.4k (~1.4 d for 12k steps at 9.9 s/step, E) · `R4'` full run (~6 d).
**Bars.** A bar marked *(PLAN)* is written in `PLAN_REFCV8.md`; *(PROPOSED)* is D5's suggestion and needs pre-registration. No bar is binding until SPEC_REFCV8 exists.

## 1. Sources

**Read and used:** `RF/{RESULT,SPEC_ADDENDUM_A1–A4}.md`, `RF/raw/{route_analysis,box_nms}.json` (bodies), `RF/raw/{a2_rescorer,a3_rescorer}.json` (key lists only; their CV figures come from `RF/RESULT.md`) · `MB/{RESULT,LANDING_READY_FIXES}`, `MB/raw/{M_c,B_box,inrun_curve}.json` (bodies; `M_a`, `M_b`, `M_d`, `M_e`, `M_f`, `fit_map` through `RESULT.md` only) ·
`BAT/{RESULT,SPEC,README}.md`, `BAT/raw/step30000/{battery_summary,RESULT_step30000}.json` (heads), `BAT/raw/PREREG_SEED_GROUP_CHECK_50400.md` · `NAV/RESULT.md`, `NAV/raw/milestones/{step5000,step30000}/{BARS,summary_*,decomposition_*,families_*}.json`,
`NAV/raw/milestones/step30000/scores_navtest/*/*.counts.json`, `step50400/runner.log`, `step30000/complete.log`, `epdms_waiter.json` · `REG`, `CLM`, `RET` (tip 0796180) ·
`SPEC_REFCV7.md` (§1–3, §26, §26.1) · `PLAN_REFCV8.md` (D: working tree, 11:59) · `REEL`, `RPL` · `CFG` · `LBL` (D5 re-derived 15 facts from it) ·
packages `2026-09-26-refcv7-launch-gate` (headline), `…-residual-prior` (headline), `…-step-cost` (TL;DR), `…-restart-options` (headline).

**Not read (and why):** `G:` (forbidden) · Thor (not touched; in-run `metrics.jsonl` is read only through `MB/raw/inrun_curve.json` and the registry) · the reel/replay media (`Media/…`, 260 MB) — the two tool documents' own MEASURED
sections are used · build-time packages in full (`…-map-hires/BUILD.md`, `…-box-head/BUILD.md`, `…-eval-loader`, `…-training-watch`, `…-watch-pace`, `2026-09-19-refcv7-build`): pre-launch construction records, no post-training effect;
their outcomes enter through `REG`/`CLM` · `BAT/raw/*/g0.json` (not opened; verdicts come from `BAT/RESULT.md`) · `NAV/raw/milestones/*/decomposition_navhard.json` bodies · D1–D4 outputs (not yet banked at 12:03).

**Pending at the time of reading (12:00–12:15 Berlin):** NavSim **50,400**: navtest and navhard bridges ran on CUDA (navtest seams complete 09:40Z, navhard bridge running since 09:40Z); **no scorer has run, no `summary_*.json`, no `BARS.json`** ·
NavSim **30,000**: navtest R7_A1 scored (counts PASS); R7_A1_s1 scoring launched 09:43Z; `summary_navtest.json`, `BARS.json`, `compare_vs_step5000_*.json` not yet written · navtest **EPDMS (v2) secondary read**: waiter `GATING`, 0/15 done ·
**four-family battery at 50,400**: waits for the GPU lock (held by the NavSim runner); **no battery number exists for ANY refcv7 checkpoint** (INS-1).

**Where two sources disagree (both given; the one that holds is named):**
1. *A6 status.* `BAT/RESULT.md` (written ~08:15) says SPEC amendment A6 is "DRAFTED, NOT REGISTERED"; `BAT/SPEC.md` and `BAT/raw/SPEC_SHA256_AMENDMENT_A6.txt` say **REGISTERED 2026-10-04T08:17:13+02:00** by the Master Mind, before any 50,400 number. **The SPEC and the sha file hold** (later, and the artifact the judge reads). `CLM` mentions A5 only.
2. *30k NavSim state.* `REG` and `CLM` (tip 0796180) say 30k navtest/navhard are RUNNING. Raw: **navhard 30k is complete** (`NAV/30k/summary_navhard.json`, 10:23) and **navtest R7_A1 30k is scored** (`…/r7s30000_R7_A1.counts.json`, PDMS 0.71885). **The raw artifacts hold; `REG`/`CLM` are stale on this point.**
3. *Duplicates after NMS.* `RF/RESULT.md` §4 headline "2.12 → 1.07 boxes per object (≥ 2: 66.5 % → 6.5 %)" is read at the **old** gate 0.2589 for both arms. The fix ships with its **re-fitted** gate 0.2145, where the same file reads **1.105 boxes/object, 10.4 % ≥ 2**
   (`RF/raw/box_nms.json` box3d `centre_NMS(2.5)` `dups_at_fit_gate`; D5V). Both are in BOX-3. No registry value exists.
4. *Box P = R gate.* `MB` quotes two fits: 0.2589 / 0.2357 (the run's own 256 TRAIN windows) and 0.2567 / 0.2307 (TRAIN-DIAG, 1,112 windows). The shipped F4 file uses the TRAIN-DIAG values; `REEL` and the replay draw boxes at 0.2589. Not a conflict of fact; a choice of fit set.
5. *Turn ratio.* `RF/RESULT.md` §5 says "V0 median 0.88 on turns"; `RF/raw/route_analysis.json` carries the **mean 0.774 [0.668, 0.872]**. Different statistics; both are in SEL-2.
6. *Goal-token positives.* `PLAN_REFCV8.md` K7 says TAKE_EXIT_L 20; `CFG` `tac_goal_stats.census` and `LBL` both read **21**. **`CFG` holds.**
7. *Stale checkout.* `D:/Projects/TanitAD/Project Steering/MODEL_REGISTRY.md` has **0** occurrences of "refcv7" (checkout HEAD 37645fc, Sep 19); the refcv7 section exists only at the tip. `MB/LANDING_READY_FIXES.txt` already warns that D: is stale. Every `REG`/`CLM`/`RET` quote here is from `git show 0796180:<path>`.

## 2. The inventory

### 2.1 Perception — 10 cm map (MAP)

**Table A — what was measured**

| id | effect (one sentence) | size [CI] (estimator) | n | tier | evidence + path | root-cause class | status |
|---|---|---|---|---|---|---|---|
| MAP-1 | The declared `prior_corrected` decision rule elects almost no thin-class cell, so the map product shows no edge/hatched at all; TRAIN-fitted per-class thresholds (F1) recover every class to the EVAL-oracle ceiling. | IoU declared -> F1: lane 0.068 [0.056, 0.078] -> 0.164 [0.145, 0.181] (paired Δ +0.097 [0.086, 0.107]); crosswalk 0.048 -> 0.105 (+0.058 [0.044, 0.070]); arrow 0.027 -> 0.059 (+0.032 [0.022, 0.042]); edge 0.000 -> 0.042 (+0.042 [0.036, 0.047]); hatched 0.000 -> 0.062 [0.005, 0.106]. Only 1.2e-6 of GT-edge cells have p̂ >= 0.5. EC, B = 1000 | 1,112 windows / 139 episodes, 5 bands pooled | OL-perc | M `MB/RESULT.md` §1.2, `MB/raw/M_c.json` (D5V verified, 12 cells); edge never > 0.0008 in 101 in-run rows `MB/raw/inrun_curve.json` | decision rule | FIXED-INF (opt-in `class_threshold` rule + monitor keys, landed b60cba6 per `CLM`; thresholds are checkpoint-specific and bound to the run's class weights) |
| MAP-2 | After the rule is fixed the ceiling is line **localisation × range**: the edge class is predicted but not placed. | edge IoU 0.042 -> 0.113 at 0.2 m tolerance (all bands); 0–20 m 0.062 -> 0.235 (0.484 at 1 m); **80–100 m 0.003 -> 0.006** (a 20× near->far drop); lane 0.164 -> 0.264 (0–20 m 0.283 -> 0.494); AUROC edge 0.850 (near 0.889, far 0.787), lane 0.909. The *predicted* drivable boundary is within 0.2 m of the GT edge for 37.5 % (0–20 m) -> 9.5 % (80–100 m), while 94.4 % of GT-edge cells lie within 2 cells of the GT boundary (labels registered). EC | same | OL-perc | M `MB/RESULT.md` §1.4–1.5, `MB/raw/M_b.json`, `M_d.json`, `M_f.json`; geometry E: one stride-8 feature = 0.33 m × 4.5 m at 20 m (cylindrical, f_ref 488.92) | architecture (stride-8 tap, depth footprint) + loss (hard-label CE on 1–3-cell lines, no tolerance) — post-hoc reading; the pre-registered D3 localisation rule read FALSE by its letter | RETRAIN |
| MAP-3 | It is **not** over-fitting and **not** a train/eval shift: the head does not localise the thin classes on its own TRAINING windows either. | thr_phat IoU TRAIN − EVAL gap CIs all contain 0 (lane −0.003 [−0.031, 0.026]; edge +0.001 [−0.006, 0.008]); TRAIN τ\* IoU < 0.15 for crosswalk 0.123, arrow 0.071, edge 0.043, hatched 0.054. Contrast: the binding G-MAP-OVERFIT reached edge 0.547 / lane 0.773 on a small set at 3,000 steps. EC (unpaired) | TRAIN-DIAG 139 episodes × 8 windows vs EVAL-DIAG | OL-perc | M `MB/RESULT.md` §1.3, `MB/raw/M_a.json`; overfit record I `REG` row "binding overfit records" | loss / architecture (not data coverage, not shift) | RETRAIN (answers "is the map data the problem?": the data are registered and consistent, MAP-2) |
| MAP-4 | The area classes are mediocre and the head's own drivable boundary is off by more than 0.5 m for most edge cells beyond 20 m. | drivable IoU 0.574 [0.551, 0.595] (declared) / 0.576 (F1) -> 0.635 at 0.2 m -> 0.724 at 1 m tolerance; sidewalk 0.467 [0.429, 0.500] -> 0.500 [0.468, 0.530] (F1). Per-band drivable IoU is not tabulated in `RESULT.md`. EC | same | OL-perc | M `MB/RESULT.md` §1.2, §1.4 | architecture (shares MAP-2) | RETRAIN |

**Table B — lever, measure, validation**

| id | refcv8 lever | measure that shows it worked (metric · split · bar) | cheapest validation rung |
|---|---|---|---|
| MAP-1 | **F1** ship the class-threshold rule; **refit τ\* on every new checkpoint's TRAIN-DIAG** (the file is bound to one checkpoint); **F2** log thresholded + 0.2 m-tolerant IoU so the monitor sees the head, not the rule | thresholded IoU per class and band on EVAL-DIAG 1,112 windows; bar: within 0.005 of the EVAL-oracle threshold for every class (the F1 standard) *(PROPOSED)*; declared-rule IoU reported beside it | `R0-inf` (one ~10 min forward pass + CPU refit; no training) |
| MAP-2 | **F3** tolerance-aware thin-class target (±2-cell dilated positives, or distance-to-line regression) — branch-compatible; stride-4 near-range tap — trunk change, defer (PLAN §8-4) | edge/lane IoU at k = 0 and k = 2, per band 0–20 … 80–100 m, paired vs refcv7+F1 with separated CI; magnitude bar to be pre-registered after the ladder shows effect size *(PROPOSED)*; a second branch seed for the replicate floor | `R2` v7-tiny map-head ladder (the G-MAP-OVERFIT harness + its must-fail arms), then `R4` |
| MAP-3 | none (diagnostic); informs the lever choice: change the target/capacity, not the data | none new; keep the TRAIN-DIAG vs EVAL-DIAG gap panel in the refcv8 eval | `R0` |
| MAP-4 | F3 plus a drivable-boundary term; **link measure first** (see gap G2) | drivable IoU per band + % of predicted boundary within 0.2 m of GT (37.5 % -> 9.5 % today); NAVSIM DAC-zero rate as the downstream read | `R0-inf` for the link ablation; `R4` for the lever |

### 2.2 Perception — boxes (BOX)

**Table A**

| id | effect | size [CI] (estimator) | n | tier | evidence + path | root cause | status |
|---|---|---|---|---|---|---|---|
| BOX-1 | The declared 0.5 detection gate fires on almost nothing; the Watch alarm read a gate/loss mismatch, not a detector failure. A per-head TRAIN P = R gate puts conf_ratio in band. | conf_ratio 0.0059 (2/339, in-run, 128 windows); 0.008 [0.004, 0.014] (EVAL-DIAG); agent 0.000. Max presence p: box3d 0.688 (29 of 318,300 slots >= 0.5), agent 0.378 (0 slots). F4 gate 0.2567 / 0.2307: conf_ratio **0.973 [0.850, 1.090]** / **0.995 [0.807, 1.187]**; box3d F1 0.016 -> 0.303 [0.283, 0.323] (paired Δ [0.266, 0.308]); AP unchanged. Under focal α 0.25 the 0.5 gate means "match belief >= 0.75". EC, B = 1000 | 1,061 labelled windows / 137 episodes; n_pos 3,390 | OL-perc | M `MB/RESULT.md` §2, `MB/raw/B_box.json` (D5V verified); reel: detections at 0.5 on 3/12 clips only (`REEL` §7) | decision rule (gate not invariant to the presence loss) + loss (flat score) | FIXED-INF (opt-in `--det-presence-gates`, b60cba6) |
| BOX-2 | The detector's real limit is ranking quality: precision ≈ recall ≈ 0.31 (box3d) / 0.21 (agent) at the P = R point; the presence score is flat. | AP@2 m box3d **0.248 [0.226, 0.272]**, agent **0.131 [0.119, 0.145]**; AP@4 m 0.330 / 0.178; F1 at P = R 0.303 / 0.207 [0.185, 0.224]; Σp / n_pos 6.42 (agent 7.23): bulk over-confident, top under-confident; TRAIN BCE temperature T\* = 0.46 (sharpen). TRAIN reads alike. Only pooled classes are published. EC | same | OL-perc | M `MB/RESULT.md` §2.2–2.3 | loss (focal presence -> flat score) + architecture (decoder refinement) | RETRAIN |
| BOX-3 | The "3–6 overlapping boxes" the PI saw are real duplicates, and a centre-distance NMS removes them. | At gate 0.2589: **2.12** boxes per detected object, 66.5 % have >= 2, 31.6 % >= 3; 46.8 % of 2,165 FPs lie within 2 m of a GT; cluster score spread median 0.033 (agent: 2.60 boxes, 68.7 % >= 2, 63 % of FPs). TRAIN reads alike (2.20; 68 %). **NMS 2.5 m** (TRAIN-chosen): AP@2 m 0.248 -> **0.349 [0.314, 0.389]** (Δ +0.101 [+0.083, +0.121]); F1 0.303 -> 0.392 (Δ +0.089 [+0.075, +0.105]) at the re-fitted gate 0.2145 (conf_ratio 0.995); agent AP 0.131 -> 0.300 (Δ +0.168 [+0.141, +0.193]), F1 Δ +0.161 [+0.144, +0.178]. Boxes/object after NMS: 1.07 (6.5 % >= 2) at 0.2589 **but 1.105 (10.4 % >= 2) at the shipped gate 0.2145**. BEV-IoU NMS is worse (AP 0.327). EC, B = 1000, paired | same | OL-perc | M `RF/RESULT.md` §4, `RF/raw/box_nms.json` (D5V verified incl. the 1.105) | architecture (slot decoder without duplicate suppression) + decision rule | FIXED-INF (NMS function and fit are in `RF/code/route_metrics.py`, `analyze_box_nms.py` — **package only, not wired into `stack/`**; the replay viewer carries an NMS toggle) |
| BOX-4 | Interface constraint: the planner reads the **raw** `sigmoid(presence_logit)` as a feature and a soft token scale, so re-gated or NMS'd presence is a distribution shift; whether duplicates and flat scores hurt the planner (NC, TTC) is untested. | none measured | – | – | I `MB/RESULT.md` F5 (from source `refc_agents.py:345,347,360`; `presence_hard` False in `CFG`) | architecture (interface) | OPEN |

**Table B**

| id | refcv8 lever | measure (metric · split · bar) | cheapest rung |
|---|---|---|---|
| BOX-1 | ship per-head P = R gates **refit on the new checkpoint** (F4); evaluate the A10 alarm at that gate | conf_ratio in [0.5, 1.5] at the TRAIN P = R gate (EVAL-DIAG); F1 at the gate; AP reported unchanged *(PROPOSED)* | `R0-inf` |
| BOX-2 | F6: sharper presence objective / more decoder refinement; temperature is learnable (T\* 0.46) | AP@2 m and AP@4 m, pooled AND per class (VRU separately) and per range band, paired vs refcv7 after NMS (0.349 / 0.300) — a bar after NMS, not before | `R2` v7-tiny box ladder (G-BOX-OVERFIT + one-frame ladder exist), then `R4` |
| BOX-3 | inference NMS 2.5 m / 3.0 m **with** its re-fitted gate; wire it into the stack and the eval | boxes per detected object <= 1.15 and share >= 2 boxes <= 0.15 at the shipped gate; AP@2 m and F1 paired vs no-NMS *(PROPOSED)* | `R0` (CPU on banked packs; refit per checkpoint) |
| BOX-4 | planner ablation: feed NMS'd presence vs raw to the frozen refcv7 planner on the NAVSIM seam | NAVSIM NC and TTC means, paired LC, vs the raw-presence arm; plus a presence-OFF arm *(PROPOSED)* | `R0-inf` (bridge arm; frozen weights, so a shift shows as a degradation) then `R4` if it matters |

### 2.3 Tactical decision, goals and their labels (TAC)

**Table A**

| id | effect | size [CI] (estimator) | n | tier | evidence + path | root cause | status |
|---|---|---|---|---|---|---|---|
| TAC-1 | Tactical lat/lon and the 22 goal tokens are supervised on **one 4-s band per clip**: every record is anchored at raw t0 = 8.0 s with band 2–6 s, and a window counts only if its NOW is within ±2.0 s of t0 (`window_in_band`), i.e. NOW in [6, 10] raw seconds of a 20 s clip. | 4,572 / 4,572 records `t0_s` = 8.0 and `tactical_s` = [2, 6] (M\*); ≈ 40 of ≈ 171 windows labelled (≈ 23 %, **E**, D1 measures it exactly). On the 107 EVAL GT-turn windows `lat_v7` is **IGNORE on 72 (67 %)**. | train 4,369 clips / 746,946 windows (`CFG`) | label census | M\* `LBL` (D5V) and the rule itself, read at the launch blob (`git show fec3a0d:stack/tanitad/data/v7_labels.py`, `window_in_band` L678–698: `abs(t_now − t0) <= (hi − lo) / 2`); M `RF/RESULT.md` §1.1; E ≈ 23 % `PLAN_REFCV8.md` §1 | data coverage | RETRAIN (labels) |
| TAC-2 | The tactical lateral head does not carry the turn: it is far worse than the plan's pick. | lat head side-correct on GT-turn windows **39.3 % [27, 52]** (L 32.5 %, R 43.3 %; says LANE_KEEP on 26/40 left and 36/67 right turns); strict TURN 24.3 % [12, 38]; goal tokens TURN_L/R side 37.4 % [22, 54]; straight windows 82.8 %. Where `lat_v7` exists it matches the geometric 6-s side on 49 % (n = 35 of 107). Reel (12 busy clips, descriptive, no CI): decoder argmax = v7 GT on lat 288/479 and lon 283/479 labelled windows, with whole-clip failures (lat 0/40 on three clips). EC | 107 GT-turn windows / 38 episodes; reel 479 labelled windows | OL | M `RF/RESULT.md` §1.1–1.2, `RF/raw/route_analysis.json` `chain_eval_s0`; `RPL` §8 | data coverage (TAC-1) + data semantics (the label is not "a turn within 6 s") | RETRAIN |
| TAC-3 | The tactical labels' meaning is uncertain: the VLM and our geometric label disagree on a third of records, and `lat_peak_m` is not a plausible lateral offset. | Alpamayo vs geometry **lateral 1,537 / 4,275 (36.0 %), longitudinal 1,655 / 4,567 (36.2 %)** disagree (M\*); `lat_peak_m` median 19.67 m, p90 110.8, max 305.3, 70.2 % > 5 m (M\*) — meaning/units UNVERIFIED; which side is right is unknown | 4,572 records | label census | M\* `LBL` (D5V); first measured by MM 2026-10-04 (`CONTEXT.md`) | data semantics | OPEN (D2 decides) |
| TAC-4 | The 22-token goal vocabulary is under-supported by the corpus: 5 tokens cannot be trained (no supervised negative) and 10 sit under the 200-positive scoreability floor. | no negatives: YIELD, SPEED_BAND, CORRIDOR_OFFSET, GAP_TARGET, REACT_ON_ONCOMING. < 200 positive records: LANE_CHANGE_R 15, TRAFFIC_LIGHT_REACT 18, YIELD_FOR_TURN_R 20, OVERTAKE_VEHICLE 20, YIELD_FOR_TURN_L 21, TAKE_EXIT_L 21, TRAFFIC_LIGHT_REACT_YELLOW 22, LANE_CHANGE_L 23, MERGE 79, TAKE_EXIT_R 128. No banked goal-token AP/AUROC result exists for any refcv7 checkpoint (the battery cell is pending). | 4,572 records | label census | M `CFG` `tac_goal_stats.census` (n_under_scoreability_floor = 10), confirmed M\* | data coverage | OPEN (nothing in this corpus can supply them; L4) |
| TAC-5 | The **longitudinal** decision is the weakest tactical read; lateral is good. | open-loop 0–2 s (trajectory-derived): lon accuracy **0.827, κ 0.540** (ORACLE-117 0.942 / 0.853), lat 0.957 / κ 0.823 (0.964 / 0.851). NAVSIM navtest 5k, 4-s horizon: lon acc 0.6913 [0.676, 0.705], κ 0.5386 [0.516, 0.559]; lat acc 0.852, κ 0.714 [0.683, 0.744]; brake_stop recall 0.627, turn_right recall 0.621. EC | 1,112 windows; 12,146 tokens / 136 logs | OL; NS-OL | M `RF/RESULT.md` §2.1, `raw/route_analysis.json`; `NAV/5k/families_navtest_R7_A1.json` | loss / decision rule (selection by sampler confidence, SEL-1, SPD-1) | RETRAIN |

**Table B**

| id | refcv8 lever | measure | cheapest rung |
|---|---|---|---|
| TAC-1 | **L1** dense tactical labels for every window from the ego's own next 6 s (labels may use ego, PI 2026-08-03) | coverage >= 95 % of windows; agreement with the geometric turn on EVAL turn windows >= 0.95; a synthetic circle reads TURN, a straight reads LANE_KEEP; a time-shuffled arm must FAIL *(PLAN §4)* | `R0` (label build + checks), then `R1` |
| TAC-2 | L1 + the frozen-trunk head-only retrain | tactical lat side-correct on GT-turn windows 0.39 -> >= 0.90 (PROPOSED), strict TURN recall; the pick-level gate is in SEL-1 | `R1` |
| TAC-3 | **L5**: D2's verdict on `lat_peak_m` and on VLM-vs-geometry; geometry stays the label of record unless D2 shows otherwise | per-field semantics table with a known-value control (a circle must read "turn"); label-noise bound on tactical accuracy | `R0` |
| TAC-4 | **L4**: supervised negatives where geometry can supply them; drop tokens < 50 positives from the loss; add an eval surface with >= 200 positives per scored token | per-token positives/negatives and AP vs prevalence, tokens under the floor reported UNSCOREABLE | `R0` (counts), `R1` (AP) |
| TAC-5 | L1 plus a longitudinal selector target (SPD-1) | lon accuracy and κ at 0–2 s and on NAVSIM 4 s, paired vs refcv7; target lon κ toward the oracle's 0.853 *(PROPOSED)* | `R1` |

### 2.4 Planner selection and route following (SEL)

**Table A**

| id | effect | size [CI] (estimator) | n | tier | evidence + path | root cause | status |
|---|---|---|---|---|---|---|---|
| SEL-1 | **Route following breaks at SELECTION inside the decoder**, not in generation and not in E9: the fan always holds a turn-correct candidate, the pick loses it. | fan contains a correct candidate **100 %**; decoder pick turns the right way **84.1 % [74, 92]** (random 38.5 %); E9 pick 84.1 % (E9 changes the direction class on none); **D_core = C − K = +0.159 [+0.080, +0.258]**; replicates on sampler seed 1 (+0.150 [+0.076, +0.245]) and on the 12 reel clips (+0.373 [+0.014, +0.886], 3 clips with turns). Wrong-direction picks are mostly "straight" (15 of 17), not the opposite side (2 of 17). The sampler's own confidence favours the pick over the oracle on 81.5 % of 81 disagreements. EC, B = 2000, paired | 107 GT-turn windows / 38 episodes (40 L, 67 R) | OL | M `RF/RESULT.md` §1.1–1.4, `RF/raw/route_analysis.json` `decision_eval_s0/s1/reel` (D5V verified) | loss / decision rule (rank by sampler confidence; flat top) | RETRAIN |
| SEL-2 | The pick **under-turns**: heading magnitude is lost more than direction. | heading within 15° of GT: fan best 100 %, oracle 75.7 %, **pick 51.4 %** (decoder 53.3 %); turn ADE **3.19 m [2.74, 3.72]** vs oracle 1.19 m; terminal-heading error 22.4° [17.3, 28.5] on turns vs 2.35° on straights; turn ratio θ_plan / θ_gt mean **0.774 [0.668, 0.872]** (median 0.88, post-hoc); bound B2 (heading within 15°): turn ΔADE −0.534 [−1.066, +0.028] (CI includes 0). EC | 107 turn windows | OL | M `RF/RESULT.md` §1.1, §2.2; `raw/route_analysis.json` | loss / decision rule | RETRAIN |
| SEL-3 | The "messy fan" is the **flat top** of an otherwise informative ranking: the top-8 spans opposite directions on a quarter of windows. | top-8 terminal-heading range **54.3° [47.8, 61.5]** on turns, 27.4° on straights; top-8 holds both a left and a right candidate on **27.0 % [22, 32]** of classified windows (30.1 % [25, 36] of GT-straight); Spearman(E9 score, −ADE) 0.753 [0.731, 0.773], random control −0.005 [−0.011, +0.002]; oracle's median E9 rank on turns is 3. Full 117-fan endpoint spread 35.7 m median (reel, descriptive). EC | 800 classified windows | OL | M `RF/RESULT.md` §3, `raw/route_analysis.json` `fan_eval_s0`; `RPL` §8 | loss / decision rule (same defect as SEL-1, viewed geometrically) | RETRAIN |
| SEL-4 | The residual prior `ha0_ext_pose` pulls the pick to the side the car is **currently yawing**, which breaks entry into roundabouts and turns. | 14 of 17 wrong-direction picks (82 %) carry the prior's direction (reel 46 of 59, 78 %); the prior itself is direction-correct on only 57 % of EVAL turn windows (reel 35 %). **Post-hoc attribution, n small, not pre-registered.** | 17 wrong picks | OL | M `RF/RESULT.md` §1.5, `raw/route_analysis.json` `posthoc_prior_diag_eval_s0` (D5V verified) | architecture (prior design) | RETRAIN (low evidence) |
| SEL-5 | The E9 goal-graft re-ranks the decoder's pick on 10.5 % of reel windows and has **no measured value** on open-loop ADE. | reel: E9 changes the pick on 217 / 2,059 windows; removing it (W3) all-window ΔADE −0.071 [−0.145, −0.015] but **not replicated** on seed 1 (−0.024 [−0.084, +0.026]); decoder pick without E9 (V5) +0.028 [−0.074, +0.123]. EC, paired | 800 windows; 2,059 reel | OL | M `RF/RESULT.md` §2, `RPL` §8; D5V (W3) | architecture (E9 rank bypasses the ceiling mask, SPD-3, and adds no accuracy) | RETRAIN or remove |
| SEL-6 | **No inference-only fix exists:** 16 pre-registered selection arms, including a linear and a gradient-boosted re-scorer over the model's own 16–19 outputs, all missed their bar. | best all-window arm W3 −0.071 (not replicated); best turn arm V3 (nav-compliance ×10) turn-direction +4.7 pp [0.0, +10.8], turn ΔADE −0.064 [−0.194, +0.079] (seed 1 −0.097 [−0.197, −0.013]), straight +0.024; A2 linear re-scorer TRAIN 5-fold CV ADE 1.4693 vs shipped 1.4838 m, EVAL +0.031 [−0.028, +0.086]; A3 trees CV 1.6179 (in-sample 1.4853), EVAL +0.493 [+0.332, +0.669]; shuffled-target control worse (+0.148 [+0.046, +0.291]) as it must be. The statement is about these two function classes and this feature set, not about learnability in general. EC | EVAL 800 classified windows; TRAIN 1,112 | OL | M `RF/RESULT.md` §2 (W3, V3, V0c re-read from `raw/route_analysis.json`; the A2/A3 figures are I from `RESULT.md`, not re-read from `raw/a2_rescorer.json` / `a3_rescorer.json`) | information not present in inference-time outputs (decision rule exhausted) | RETRAIN |

**Table B**

| id | refcv8 lever | measure (metric · split · bar) | cheapest rung |
|---|---|---|---|
| SEL-1 | selector retrain on the emitted fan with a **listwise** target (or build the disentangled scorer `refcv7_wta` / `refcv7_select`, in the launch code but not built); with L1 labels and L2 nav | turn-direction-correct pick on GT-turn windows **0.84 -> >= 0.95** and D_core -> 0 (CI covers 0), paired, on the same 107 windows + a replicate sampler seed *(PLAN R1 gate)*; random chance reported beside (0.385) | **`R1`**: cache refcv7-50.4k decoder inputs, retrain only the selector/tactical decoder (minutes per arm). The R1 outcome picks branch vs full run |
| SEL-2 | same as SEL-1, plus time-localised nav (NAV-1) | heading within 15° at 6 s on GT-turn windows **0.514 -> >= 0.70** (PLAN R1 gate; fan best 0.757); turn ADE vs oracle 3.19 / 1.19 m; terminal-heading error | `R1` |
| SEL-3 | same defect; add a candidate-diversity-aware or MBR-aware selection only if R1 shows the listwise target does not close it | share of windows whose top-8 holds both directions (27 % today; <= 10 % with the GT straight share reported) *(PROPOSED)*; Spearman vs −ADE >= 0.753 | `R1` |
| SEL-4 | prior dropout in training or a route-aware prior | wrong-direction picks that carry the prior's direction (14/17) on a larger turn set; needs more than 17 events | `R2` (v7-tiny arm with a deliberate-regression arm) |
| SEL-5 | drop the E9 graft, or retrain it with the ceiling mask in its rank (§26.1 fix) | all-window ΔADE paired, replicated on seed 1; picks changed | `R0` for the decoder-pick-only arm (done: V5); `R1` for a retrained graft |
| SEL-6 | training (above) | the SEL-1 gate on EVAL **and** the TRAIN-fitted re-scorers re-run on the new fan (they must still gain nothing) | `R1` |

### 2.5 Planner speed and longitudinal (SPD)

**Table A**

| id | effect | size [CI] (estimator) | n | tier | evidence + path | root cause | status |
|---|---|---|---|---|---|---|---|
| SPD-1 | **The speed profile of the pick is the largest open-loop planner error.** Restricting the pick to candidates whose 6-s progress is within 10 % of GT removes more error than direction and heading together. | B3 ΔADE: **all windows −0.606 [−0.728, −0.493] m** (51.6 % of the ORACLE-117 gap of −1.175 [−1.343, −1.007]), turns −0.892 [−1.300, −0.523], straights −0.511 [−0.653, −0.379]; B4 (heading ∧ speed) −0.674 [−0.815, −0.541]. Slot-mean along-track error on turns 2.17 -> 0.71 m for the oracle. Attribution bound (label-side), not a lever. EC, paired | 800 classified windows (107 turn, 588 straight) | OL | M `RF/RESULT.md` §2.2, `raw/route_analysis.json` `A4_bounds` (D5V verified) | loss / decision rule (the rank is dominated by sampler confidence, A2 weight 3.13) | RETRAIN |
| SPD-2 | The plan **runs ahead of GT on turns**. | signed along-track error at 6 s: turns **+3.272 [+1.380, +5.047] m**, all +0.951 [+0.166, +1.730], straights +0.426 [−0.438, +1.341] (n.s.); \|along\| 6 s on turns 7.42 [6.25, 8.75] m; speed MAE 0–2 s 0.312 [0.260, 0.370] turns / 0.262 [0.242, 0.286] all m/s. EC | same | OL | M `RF/raw/route_analysis.json` `levers_eval_s0` V0 (D5V verified) | loss / decision rule; **H**: speed is not coupled to curvature in the rank (the label `ADAPT_SPEED_FOR_CURVE` exists on 995 of 4,572 records but only inside the 4-s band, TAC-1) | RETRAIN |
| SPD-3 | **The speed ceiling does not reach the emitted plan** (E9 re-ranks with the reach mask only). | filter ON vs OFF bit-identical on 204/204 warmup scenes at 1,500; reel: plan exceeds the fed ceiling on **110 / 2,059 windows (5.3 %)** (50 on clip 7, 39 on clip 8; separately, 31 windows, all on clip 7, have NO candidate under the ceiling, so the filter keeps the whole fan); NAVSIM: declared-ceiling arm − A1 = +0.07 PDMS [−0.00, +0.14] (5k), n.s. **The emulated §26.1 fix costs open-loop ADE: all-window +0.086 [+0.018, +0.158] m (seed 1 +0.087 [+0.017, +0.160]), straights signed along 6 s +0.43 -> −0.11 m** (plan slower than GT). EC, paired | 800 windows; 2,059 reel; 12,146 tokens | OL; NS-OL | M `SPEC_REFCV7` §26.1; `NAV/RESULT.md` §0, §5; `REEL` §7; `RF/RESULT.md` §1.6; `RPL` §8; D5V (V0c) | decision rule (G-DVB/G-LIVE checked the filter is built, not that the emitted plan obeys it) | FIXED in package (`ceil_keep` / `e9_rank`, 52/52 tests), **not live**; rides the next closure change |
| SPD-4 | The max-speed input is a **per-clip oracle** (ego's realised max over [anchor+2, anchor+6] s, snapped up to the posted ladder) fed on **every** window, so it is stale away from the anchor and wrong at stops. The label file itself calls this a "TRAIN/DEPLOY MISMATCH, not a leak": at deployment the user supplies a coarser limit; NAVSIM is fed a per-token speed-limit table (`NAV/raw/inputs/speed_limits_navtest.json`, 12,146 tokens; provenance I). | the label file's own audit fields (M\* read): v_hi←v0 R² 0.8789, ←(bin, v0) 0.9702, the bin recovers 75.4 % of the missing future information; "75 % of intersection clips get <= 30 km/h where a real map would say 50" (an artefact of the ego standing still); **1,741 of 4,572 clips (38 %) are fed a <= 30 km/h ceiling** (M\*). Effect on driving: A1 − VMAXOFF on NAVSIM warmup S2-EPDMS-u = −0.0040 (5k) / −0.0041 (30k), i.e. withholding the ceiling scores slightly HIGHER, both inside the seed floors 0.0084 / 0.0069 (the input does not help there); navtest 200-token VMAXOFF − A1 +0.42 [−0.01, +1.14] n.s. | 4,572 records; 204 warmup tokens; 200 navtest tokens | NS-OL | M\* `LBL`; M `NAV/5k`, `NAV/30k` `summary_warmup.json` pairs | data semantics + input usage (train/deploy mismatch) | RETRAIN (L3) |
| SPD-5 | On NAVSIM plans the learned longitudinal profile is **no better than its own kinematic prior** and commits further ahead of the human, while the learned lateral part improves; between 5k and 30k the lateral family improved while the longitudinal family shows no resolved improvement. | navtest 5k (12,146 tokens / 136 logs): A1 vs PRIOR_ha0p — speed MAE 0.7634 [0.7303, 0.7988] vs 0.7069 [0.6622, 0.7561]; target-speed within 0.5 m/s **0.4897 [0.4689, 0.5090] vs 0.5954 [0.5738, 0.6158]**; along-final bias **+1.647 [1.535, 1.762] vs +0.218 [0.026, 0.398] m**; progress ratio vs human 1.184 [1.161, 1.209] vs 0.989 [0.974, 1.004]; lateral better (heading MAE 4.03° vs 5.69°). navhard stage-1 (450 windows / 76 logs) 5k -> 30k: heading MAE 7.09° -> 4.72°, lat κ 0.492 -> 0.748, but speed MAE 0.748 -> 0.758, lon κ 0.415 -> 0.366, along-final bias +1.77 -> +1.42 m. Intervals are **unpaired** episode-cluster CIs; a paired read is a zero-GPU item. | see left | NS-OL | M `NAV/5k/families_navtest_R7_A1.json`, `families_navtest_PRIOR_ha0p.json`; `NAV/5k`, `NAV/30k` `families_navhard_*.json` | loss (selection by sampler confidence) + decision rule | RETRAIN |

**Table B**

| id | refcv8 lever | measure | cheapest rung |
|---|---|---|---|
| SPD-1 | selector retrain with a progress-aware listwise target; **L3** per-window speed input | speed-profile ΔADE bound re-run on the new fan; speed MAE 0–2 s; 0–2 s lon κ 0.540 -> toward 0.85; target-speed accuracy @0.5 m/s and along-final bias on NAVSIM (0.49, +1.65 m today) *(PROPOSED magnitudes)* | `R1` |
| SPD-2 | same, with curvature-conditioned speed targets | signed along-track at 6 s on GT-turn windows +3.27 -> within ±1 m *(PROPOSED)* | `R1` |
| SPD-3 | apply the §26.1 fix **and** calibrate the ceiling bins (the fix is correct but exposes SPD-4: honouring an oracle bin costs ADE) | plan-over-ceiling windows 110/2,059 -> 0; all-window ADE reported with and without the fix; ceiling-bin calibration curve (bin vs realised) | `R0` (emulation done, V0c); `R4` ride-along |
| SPD-4 | **L3**: window-local realised max snapped to the posted ladder (leak measured and stated), or a coarser posted-limit proxy (PI decision §8-2) | the file's OOF R² of the future max from v0 vs from (bin, v0) per option; a shuffled-ceiling arm must lose the gain; VMAXOFF arm on NAVSIM must stay n.s. or improve | `R0` (leak test), `R2` (shuffled arm) |
| SPD-5 | same as SPD-1; add the NAVSIM longitudinal family to every refcv8 readout | paired A1 − PRIOR on speed MAE, target-speed accuracy, along-final bias, progress ratio (navtest and navhard) | `R0` (paired re-read from the banked seams) |

### 2.6 Navigation input (NAV)

**Table A**

| id | effect | size [CI] (estimator) | n | tier | evidence + path | root cause | status |
|---|---|---|---|---|---|---|---|
| NAV-1 | The nav input is **one token per clip, fed on every window**, so it cannot say *when* to turn. | **193 of 291** left/right-nav EVAL windows are GT-straight (66 %); nav side == GT on GT-turn windows 0.439 [0.28, 0.61] overall, 0.940 on L/R-clip turns (n 50), 0.000 on 193 straights. Turn starts a median **7.3 s** after the anchor, 52.7 % > 6 s, 42.0 % > 10 s, p90 26.2 s (1,675 L/R records, M\*). The file carries `nav_30s.entries` with `t_start_s`, `t_end_s`, arc-length `distance_m` for all 1,675 L/R clips (M\*); the run feeds only the token (`nav_args` false). | EVAL 291 windows; train 1,675 L/R records (779 L / 838 R clips in the 4,369-clip cache, `CFG`) | OL; label census | M `RF/RESULT.md` §1.1, `raw/route_analysis.json`; M\* `LBL` (D5V) | input usage (information present in the file, unused) | RETRAIN (L2 is a loader change; the model must retrain to use it) |
| NAV-2 | The nav-compliance graft is **inert**. | learned gate 0.1625 = 0.042–0.045 × the per-window score std; removing the term changes **1** decoder pick and **2** E9 picks of 291 informative windows; as a hard filter it forces turns on 193 straight windows (V2 straight ΔADE +0.326 [+0.197, +0.459]). The oracle complies on 93.6 % of nav-right turn windows, the pick on 76.6 %. EC | 291 windows | OL | M `RF/RESULT.md` §1.3, §2; D5V (gate) | input usage (NAV-1) + loss | RETRAIN |
| NAV-3 | On NAVSIM the nav input has **no measurable effect**, and its provenance differs from training: the nav token is an ego-future oracle on PhysicalAI, an official route on NAVSIM. | R7_A1 − NAVOFF on warmup S2-EPDMS-u: −0.0002 (5k), +0.0034 (30k), both inside the seed floors 0.0084 / 0.0069; vision matters a lot (A1 − BLIND +0.172 at 5k, +0.303 at 30k). No navhard NAVOFF arm exists. | 204 warmup tokens | NS-OL | M `NAV/5k`, `NAV/30k` `summary_warmup.json` pairs; provenance `CFG` `nav_from_v7_stats` | input usage + data semantics (oracle) | OPEN |

**Table B**

| id | refcv8 lever | measure | cheapest rung |
|---|---|---|---|
| NAV-1 | **L2** per-window next-turn token + time/distance to it; FOLLOW once no turn is ahead (supplied route, same ego-future oracle caveat as today, now time-correct — PI decision §8-3) | windows with a turn token and no turn inside the stated horizon -> 0; the 193/291 straight-under-L/R count falls to the expected residual (turns beyond 6 s, reported separately); pick-turns-the-commanded-way on windows with a turn starting within 6 s *(PLAN §4)* | `R0` (label build + count), `R1` |
| NAV-2 | drop the soft graft or make it hard-and-timed once L2 exists | gate / score-std ratio; picks changed without it; V2-style hard filter must no longer cost straight ADE | `R1` |
| NAV-3 | add NAVOFF/NAVSHUF arms to navhard; state the provenance on every NAVSIM row | NAVOFF paired deltas on navhard S2-EPDMS (not only warmup) | `R0-inf` |

### 2.7 NAVSIM (NS) — the open-loop benchmark, with the sub-scores that decide it

NavSim references: STOP, CV, HUMAN, PRIOR_ha0p are banked floors; REFe (trained on navtrain, not like for like) reads full-navtest PDMS 83.24 [81.77, 84.63] (`REG`).

**Table A**

| id | effect | size [CI] (estimator) | n | tier | evidence + path | root cause | status |
|---|---|---|---|---|---|---|---|
| NS-1 | navtest PDMS: beats standing still at 5k, clearly at 30k; **50,400 pending**. | **5k** (CUDA bf16): A1 65.5976, STOP 61.8202, HUMAN 94.5514, CV 20.6517, PRIOR 58.6613; **A1 − STOP +3.7774 [+1.58, +5.86]** (LC, paired; separated on log_name AND nuplan_drive); seed floor 0.0849; BAR-R7-N1 **PASS**. **30k** (CPU fp32): A1 **71.8849** (scorer summary, **no interval yet**) = +10.06 over STOP (point); NC 91.28, DAC 87.98, EP 69.54, TTC 80.84, DDC 94.86. 5k -> 30k straddles a device change (KP floor: same selection 88.2 %, median endpoint 0.40 m). | 12,146 tokens / 136 logs | NS-OL | M `NAV/5k/BARS.json`, `summary_navtest.json` (D5V verified); M\* `NAV/30k/scores_navtest/r7s30000_R7_A1/…counts.json` | – | PENDING (30k interval, 50.4k) |
| NS-2 | **navhard official two-stage EPDMS is below standing still**, at 5k and at 30k. | 5k: A1 0.1624 vs STOP 0.2985, Δ **−0.1361 [−0.1713, −0.1027]** (BAR-R7-NH1 FAILED); 30k: A1 0.2269 vs STOP 0.2985, Δ **−0.0716 [−0.1112, −0.0346]**, separated worse; A1 − PRIOR +0.0969 [+0.0605, +0.1315] (30k), +0.0324 [+0.0019, +0.0627] (5k, SPEC conjunction not separated). Inference-seed floor 0.0036 (30k). LC | 5,462 stage-2 + 450 stage-1 scenes | NS-OL | M `NAV/5k`, `NAV/30k` `summary_navhard.json` (D5V verified) | composite of NS-4 and NS-5 | PENDING (50.4k) |
| NS-3 | warmup S2-EPDMS-u: FAILED at 5k, NOT PROVEN at 30k. | 5k: A1 0.4487 vs STOP 0.5212 (margin −0.0726, floor 0.0084) FAILED; 30k: 0.5224 vs 0.5212, margin +0.0011 inside 2 × floor 0.0139 -> NOT PROVEN. No interval (7 logs < 8). | 204 tokens | NS-OL | M `NAV/5k/BARS.json`; `NAV/RESULT.md` §6.3 | composite | PENDING (50.4k scoring) |
| NS-4 | **Drivable-area compliance (DAC) is the largest sub-score deficit and ceiling.** | navtest 5k: DAC 84.55 vs STOP 96.53; **single-term ceiling +7.296 PDMS** (25.2 % of the 28.95 gap to HUMAN); DAC = 0 zeroes 1,876 tokens (1,770 where it is the only zero term). 30k navtest: DAC 87.98 (zero on 12.0 % of tokens). navhard stage-2 DAC-zero rate **35.5 % (5k) -> 26.0 % (30k)** vs STOP 13.7 %; DAC mean 0.645 -> 0.740 vs STOP 0.863. **Cause not attributed.** | as NS-1 / NS-2 | NS-OL | M `NAV/5k/decomposition_navtest.json` (D5V verified), `summary_navhard.json`; M\* `…counts.json` | NOT ATTRIBUTED (candidates: map MAP-2/4, selection SEL-1, lateral, speed) | OPEN |
| NS-5 | **Collision (NC) and time-to-collision (TTC)** are the second safety deficit; navhard NC barely moved from 5k to 30k. | navtest 5k: NC 88.09 vs STOP 97.40, TTC 75.96 vs 96.40; single-term ceilings TTC **+4.324**, NC **+1.930**; NC = 0 zeroes 1,354 tokens. 30k navtest NC 91.28 (zero on 8.3 %), TTC 80.84. navhard stage-2 NC-zero **16.1 % (5k) -> 15.6 % (30k)** vs STOP 4.0 %; NC mean 0.827 -> 0.833 (vs DAC +0.095 over the same interval). Duplicate/flat boxes (BOX-3/4) are one untested candidate. | as above | NS-OL | M `NAV/5k/decomposition_navtest.json`, `summary_navhard.json`, `NAV/30k/summary_navhard.json` | NOT ATTRIBUTED | OPEN |
| NS-6 | refcv7 buys **progress at the expense of safety** relative to standing still, and has pockets worse than STOP. | EP 64.2 (5k) / 69.5 (30k) vs STOP 31.0, HUMAN 87.0 (EP ceiling +4.162); by speed band (5k, A1 − STOP): **2–5 m/s −11.57** (A1 58.37 vs 69.94, n 3,458), 0–2 m/s +9.29, 8–12 m/s +21.34, >= 12 m/s +36.26; by command: **RIGHT −3.73** (n 1,575), LEFT +0.34, STRAIGHT +6.31. 30k decomposition not yet written. LC not applied to the strata | 12,146 tokens | NS-OL | M `NAV/5k/decomposition_navtest.json` (D5V verified) | not attributed (low-speed and right-turn behaviour, SPD/SEL) | OPEN (30k/50.4k PENDING) |
| NS-7 | **Positive control:** the learned residual adds over its own kinematic prior, and vision is essential. | A1 − PRIOR_ha0p: navtest +6.9364 [+5.01, +8.75] (5k); navhard 30k +0.0969 [+0.0605, +0.1315]; A1 − BLIND warmup +0.172 (5k) / +0.303 (30k). LC | as above | NS-OL | M `NAV/5k`, `NAV/30k` summaries | – | keep as a control row in every refcv8 table |

**Table B**

| id | refcv8 lever | measure | cheapest rung |
|---|---|---|---|
| NS-1 | none (scoreboard) | navtest PDMS, paired vs STOP and vs refcv7 at the same device, with the inference-seed floor; bar 2 × floor *(BAR-R7-N1 form)* | `R5` |
| NS-2 | scoreboard; moves with NS-4/5 | navhard EPDMS paired vs STOP; **target: separated >= 0 vs STOP** (the bar refcv7 failed) *(PROPOSED as the refcv8 NH bar)* | `R5` |
| NS-3 | scoreboard | warmup S2-EPDMS-u beyond 2 × floor | `R5` |
| NS-4 | **first an attribution probe (PROPOSED, not in PLAN):** for every DAC-zero token, does the 117-fan hold a DAC-clean candidate (the D_core analogue for DAC)? if yes the fix is selection (SEL-1); if no it is the map/lateral | DAC-zero rate on navhard S2 (26.0 %) and navtest (12.0 %); fan-contains-clean rate and pick-picks-clean rate on DAC-zero tokens; DAC single-term ceiling | `R0-inf` (export the fan in the bridge, score 117 hypotheses per token on CPU) |
| NS-5 | the same probe for NC/TTC; plus BOX-4 ablation | NC-zero and TTC; fan-contains-clean and pick rates | `R0-inf` |
| NS-6 | follows from the probe; add stratified (command × speed band) rows to every NAVSIM readout | stratum deltas vs STOP with LC intervals | `R0` |
| NS-7 | keep | – | – |

### 2.8 Instruments and process (INS)

**Table A**

| id | effect | size [CI] (estimator) | n | tier | evidence + path | root cause | status |
|---|---|---|---|---|---|---|---|
| INS-1 | The four-family battery has produced **no number for any refcv7 checkpoint**: its loader-reproduction gate G0 failed at 5,000 and 30,000. | 5k: `eval_traj` in-run 0.64714 vs 8-seed replay 0.673517 (sd 0.00212); 24 seeds show sd 0.01072, in-run z = −2.28 => the K = 8 estimator was the defect (A5, K = 24). 30k: `eval_tacv6_goal_conf_bce` 1.52 % off (bar 1 %): a hard-threshold target over 271 supervised cells flips on numerics; the replay drops `--trunk-compile` and its drift grows 3.6e-5 -> 5.1e-4 -> 1.64e-3 (SMOOTH median). A5 and A6 (registered 08:17:13) are in force. **50,400 pending**; the registered seed-group check (F test, step 50,400) decides tail-event vs structured instrument. | 128 fixed in-run windows, 24 seeds | instrument | M `BAT/RESULT.md` (which cites `BAT/raw/step{5000,30000}/g0.json`, not opened by D5), `BAT/SPEC.md` A5/A6; `CLM` REFCV7-G0-A5 | instrument | PENDING (G0 at 50.4k) |
| INS-2 | The thin-class alarm measured the **decision rule**, not the model ("collapsed at 5,000" is retracted: that is the step the alarm arms). The 128-window in-run monitor is also noisy per class (crosswalk 0.021 at 50,400 after 0.042 at 17,500). | edge IoU under the declared rule never > 0.0008 in any of 101 in-run eval rows; crosswalk 0.021 vs 0.042 on 128 windows | 101 rows × 128 windows | OL-perc | M `RET` R27, `MB/RESULT.md` §1.1 | instrument | FIXED (F2 monitor keys) |
| INS-3 | **Inference-sampler variance** is real and large for single windows; **training-seed variance is unmeasured.** | E9 pick identical on **91.1 %** of windows between sampler seeds 0 and 1; emitted plans differ by a median **0.64 m** (max 12.83 m); mean ADE 1.723 vs 1.695 m over all valid-GT windows; reel plan move 5.26 m on one clip. NAVSIM floors: PDMS 0.0849 (5k), S2-EPDMS-u 0.0084 / 0.0069 (5k / 30k). Perception heads are seed-invariant (603/603 terms spread 0 at the step-1,500 validation checkpoint; bit-identical determinism at 50,400, `MB` C7). | 1,112 windows | OL | M `RF/raw/route_analysis.json` `seed_floor` (D5V); `REEL` §7; `BAT/RESULT.md` §3 | decision rule (sampled planner) | OPEN for training seed (`H-ESTIM-SEED-1`) |
| INS-4 | NAVSIM numbers are not directly comparable across steps because device and precision differ per split: **5k** all three splits CUDA bf16 (native trunk, exact dedup dropped as KD failed); **30k** warmup and navtest CPU fp32 (the GPU lock was held > 3 h, designed fallback), navhard CUDA; **50,400** warmup CPU fp32 (first launch: K0 misread under a pytest-rootdir error, since pinned by `pytest.ini`), navtest/navhard relaunched on CUDA (K0 True, KD False). | KP floor (step 1,500): same selection 88.2 %, 4-s endpoint median 0.40 m, p90 2.88 m | 204 scenes (KP) | NS-OL | M `NAV/RESULT.md` §3, §6.2, §7; `NAV/raw/milestones/step50400/runner.log`, `step30000/complete.log` | instrument | PENDING (`compare_vs_step5000_*.json` flags the device change) |
| INS-5 | Cost: refcv7 runs at **9.876 s/step = 1.54 × refcv6** (6.41); the 10 cm map branch is 83–90 % of the +3.5 s delta (dev-box ablation; Thor INHERITED); the R4 bundle (conflict cadence 50, logged-steps-only map/box census, freeze, fixes) would save an **E 0.76–1.40 s/step** and was never run. | see left | – | compute | M `…-step-cost/RESULT.md`; E `…-restart-options/RESTART_OPTIONS.md`; `REG` | architecture / compute | open; sets the price of R4 (12k steps ≈ 1.4 d) and R4' (≈ 6 d) |
| INS-6 | In-run eval **plateaued from ~35,000** (in-run `eval_traj` 0.515–0.529 at 35k / 40k / 45k / 50.4k); ≈ 15,400 steps (≈ 41 h at 9.9 s/step) show no in-run gain. NavSim gained 5k -> 30k (PDMS 65.60 -> 71.88 point, device changed), so the plateau is an in-run L1 statement, not a driving one. | – | in-run 8 batches | OL | I `CLM` REFCV7-DONE | – | informs branch-first; **50.4k NavSim (PENDING) tests whether driving also plateaued** |
| INS-7 | Ten G-LIVE-admitted modules (8 strategic + the replaced lat3/lon3 prior pair) take **zero gradient by construction** (strategic layer OFF, PI R5); their model-side freeze was owed at the next restart. `CFG` also records "42/138 optimizer tensors took no gradient (5,305,667 params = 52.2 % of a declared trainable budget)"; the denominator is not defined there and D5 did not re-derive it. | – | – | architecture | I `REG` quoting stamp; `CFG` `effective_weights` | architecture | OPEN (low priority; saves memory, not accuracy) |

**Table B**

| id | refcv8 lever | measure | cheapest rung |
|---|---|---|---|
| INS-1 | run G0-A6 at 50,400; for refcv8 keep `--trunk-compile` in the replay (Thor loader exists) or classify `goal_conf_bce` as THRESHOLD by name | G0-A6 PASS; mutations M1/M2/M4 detected; the seed-group F test (p >= 0.05 supports the tail-event reading) | `R0-inf` (waiting for the GPU lock) |
| INS-2 | F2 monitor keys on | thresholded + 0.2 m-tolerant IoU in the Watch | `R0` |
| INS-3 | add a **training-seed replicate** to the refcv8 ladder (two branch seeds, or a replicate arm with unchanged flags) and report the sampler floor beside every lever | a lever effect must exceed the replicate arm's difference from its base, not only the bootstrap interval | `R2` (v7-tiny replicate, cheap) |
| INS-4 | one device and precision per comparison; re-run 30k on CUDA if a step-to-step claim is made | paired navtest/navhard on one device | `R0-inf` |
| INS-5 | price every lever in s/step before choosing branch vs full | – | `R0` |
| INS-6 | branch-first (PLAN §8-1) | NavSim 5k vs 30k vs 50.4k on one device | `R0-inf` |
| INS-7 | freeze or drop the dead modules in the branch config | G-LIVE: no unreachable-by-accident group | `R2` |

## 3. Ranking by measured size

**Rule.** Effects are ranked **within a driving family** by the measured size of the available gain (the lever's upper bound or the defect's magnitude, in the family's own unit, with its CI), and **across families by the share of the layer's own reference gap**
(ADE: the ORACLE-117 gap 1.175 [1.007, 1.343] m; NAVSIM: the 28.95 PDMS gap between A1 at 5k (65.60) and HUMAN (94.55); perception: absolute metric points). Perception sits after planner/NAVSIM rows because **no measurement links a perception fix to a driving score** (gap G2); the planner reads the map as encoder features (`--bev-source map_hires_pool`) and boxes as raw presence, not through the decision rules F1/F4 change.

### 3.1 By family

| family | order (largest first), each with size and CI |
|---|---|
| **Route following / heading** (turn windows, EC paired) | SEL-1 direction 84.1 % -> 100 % (D_core +0.159 [+0.080, +0.258]; B1t turn ΔADE **−0.566 [−0.986, −0.239] m** = 28 % of the 1.995 m turn oracle gap; all-window only −0.076 [−0.132, −0.029]) · SEL-2 heading within 15° **51.4 % vs 75.7 %** (B2 turn −0.534 [−1.066, +0.028], all −0.125 [−0.208, −0.040]) · SEL-3 flat top (27 % of windows hold both directions) · SEL-4 prior pull (14/17) · NAV-1 (66 % of L/R-nav windows are straight) · NAV-2 (gate 0.045 × std) · TAC-2 (39.3 % side-correct) |
| **Speed profile ADE** | SPD-1 B3 **−0.606 [−0.728, −0.493]** all (51.6 % of the oracle gap), −0.892 turns · SPD-2 over-travel on turns +3.27 m at 6 s · SPD-5 NAVSIM: progress ratio 1.184, along-final bias +1.65 m, target-speed accuracy 0.49 vs prior 0.60 · SPD-3 ceiling: emulated fix **+0.086 [+0.018, +0.158]** ADE cost, 110/2,059 windows over the ceiling · SPD-4 oracle bin on every window (38 % of clips <= 30 km/h) |
| **NAVSIM DAC / NC** (5k navtest single-term ceilings, PDMS points; 30k navhard zero-rates) | NS-4 DAC **+7.296** (25.2 %; navhard DAC-zero 35.5 % -> 26.0 % vs STOP 13.7 %) · NS-5 TTC **+4.324** (14.9 %) · EP +4.162 (14.4 %, NS-6) · NC **+1.930** (6.7 %; navhard NC-zero 16.1 % -> 15.6 % vs 4.0 %). The terms trade; the ceilings do not add up. · pockets: 2–5 m/s −11.57 vs STOP, RIGHT −3.73 |
| **Map thin-class IoU** (F1 gain; headroom) | MAP-1 gains lane **+0.097**, hatched +0.062, crosswalk +0.058, edge +0.042, arrow +0.032 (all paired CIs exclude 0; fixed at inference) · MAP-2 headroom: edge x2.7 at 0.2 m tolerance, 20× near->far collapse (0.062 vs 0.003) |
| **Box AP and duplicates** | BOX-3 agent AP **+0.168 [+0.141, +0.193]**, box3d **+0.101 [+0.083, +0.121]** (+41 % relative), F1 +0.089 · BOX-1 F1 0.016 -> 0.303 at the right gate · BOX-2 the remaining ceiling: AP 0.349 / 0.300 after NMS, P = R precision 0.39 |

### 3.2 The top 10 by measured size, with their levers

| # | id | effect and size | lever | rung |
|---|---|---|---|---|
| 1 | SPD-1 | speed-profile selection: **−0.606 [−0.728, −0.493] m** ADE = 51.6 % of the oracle gap (−0.892 on turns) | listwise selector on the emitted fan + L3 per-window speed | `R1` |
| 2 | NS-4 | DAC: **+7.30 PDMS** single-term ceiling (25 % of the gap to HUMAN); navhard DAC-zero 26.0 % vs STOP 13.7 % | **attribution probe first** (fan-contains-clean-candidate on DAC-zero tokens), then selection/map | `R0-inf` |
| 3 | NS-5 | TTC **+4.32** and NC **+1.93** ceilings; navhard NC-zero 15.6 % vs 4.0 %, barely moved 5k -> 30k | same probe; BOX-4 ablation | `R0-inf` |
| 4 | SPD-5 / NS-6 | learned longitudinal no better than its kinematic prior (target-speed 0.49 vs 0.60; along-final bias +1.65 vs +0.22 m); EP is bought with safety | selector + L3; paired A1 − PRIOR family read | `R0`, `R1` |
| 5 | SEL-1 / SEL-2 | turn selection: direction 84.1 % vs fan 100 %; heading within 15° 51.4 % vs 75.7 %; turn ADE 3.19 vs 1.19 m | selector + L1 + L2 | `R1` (PLAN gate: >= 0.95 / >= 0.70) |
| 6 | BOX-3 | duplicates: AP@2 m **+0.101** (box3d), **+0.168** (agent) with centre NMS and a re-fitted gate | inference NMS + gate; wire into the stack | `R0` |
| 7 | MAP-1 | thin classes hidden by the rule: lane **+0.097**, hatched +0.062, crosswalk +0.058, edge +0.042 | F1 thresholds refit per checkpoint | `R0-inf` |
| 8 | MAP-2 | placement × range: edge IoU 0.062 at 0–20 m vs 0.003 at 80–100 m; x2.7 headroom at 0.2 m tolerance | F3 tolerance-aware target (stride-4 tap deferred) | `R2` -> `R4` |
| 9 | TAC-1 / TAC-2 | tactical GT on ≈ 23 % of windows; `lat_v7` IGNORE on 67 % of turn windows; head 39.3 % side-correct | L1 dense labels | `R0` -> `R1` |
| 10 | NAV-1 | 66 % of L/R-nav windows are straight; gate inert (1–2 of 291 picks) | L2 time-localised nav | `R0` -> `R1` |

Next by size: SPD-3 ceiling (5.3 % of reel windows over the ceiling; the fix costs +0.086 ADE until the bins are calibrated) · BOX-2 (AP 0.35 / 0.30 after NMS) · SEL-4 prior pull (n = 17) · INS-3 (the training-seed floor).
**Rule Zero note:** every row above is a lever with a validation rung, not a verdict. Rows 2–3 are the only top-10 items whose lever is **not yet identified**; their first action is a zero-training probe, named in NS-4 Table B.

## 4. Effects with NO measure yet (gaps the refcv8 plan must fill)

| gap | what is missing | why it matters | proposed measure | rung |
|---|---|---|---|---|
| G1 | **Attribution of NAVSIM DAC/NC/TTC failures** to map, selection, lateral or speed. The 117-fan is never scored on NAVSIM. | NS-4/5 are the largest NAVSIM ceilings and have no cause | per DAC-zero / NC-zero token: does the fan hold a clean candidate, and does the pick take it (the D_core analogue); `REFe`'s LANE-1 did this for its own fan (88–97 % of violating picks had a clean hypothesis) | `R0-inf` |
| G2 | **Perception -> planning dependence.** No arm ablates the 10 cm map features, or box presence/NMS, in the planner. | decides whether F1/F3/F4/NMS can move driving at all | MAPOFF / BOXOFF / NMS'd-presence arms on the NAVSIM seam and on the route set | `R0-inf` |
| G3 | **Distance keeping** (headway, time gap, min TTC to the lead). `UNAVAILABLE` in the route package and in every NAVSIM family block (no lead track supplied); the battery computes it from the B1 lead block at {1.0, 2.0} s only. | one of the two longitudinal components the binding four-family rule names (with target-speed accuracy) | lead-track export for eval139 and for navtest; or a NAVSIM-native TTC read as proxy (done: TTC sub-score) | `R0` |
| G4 | **Battery numbers at any checkpoint** (BAR-R7-1/2/3, BAR-M7-1..4, BAR-B7-1/2, yaw-rate cell, tactical κ on in-band windows, goal selection AUROC/AP per token). | the official four-family instrument; G0 failed twice, 50.4k pending | finish G0-A6 at 50,400, then the S2 rolls | `R0-inf` |
| G5 | **Per-class and VRU box AP, range-binned AP, size/yaw/depth error.** Only pooled AP@2 m and AP@4 m exist; per-class keys exist in the in-run row on 128 windows with n < 30 per cell. | pedestrians and cyclists drive different failures than vehicles | AP by class and range band on EVAL-DIAG 1,061 windows | `R0-inf` |
| G6 | **Goal-token AP** (22 tokens, 10 under the scoreability floor) and **g_tac goal-point quality** against GT, with CIs. Only the reel (descriptive) and the trajectory-derived goal FDE exist. | tactical goal setting is a named family | scoreable tokens only; the rest reported UNSCOREABLE with n | `R0-inf` / `R1` |
| G7 | **Training-seed variance** for any lever or effect (`H-ESTIM-SEED-1`). Every interval here answers "another draw of episodes". | a separated CI from a one-seed arm is necessary, not sufficient | replicate arm with unchanged flags on the v7-tiny ladder | `R2` |
| G8 | **Strata**: results by day/night (2,120 night clips), road class (urban 2,965 / intersection 861 / highway 746) and country (~25). None of the effects above is cut by stratum. | a defect concentrated in night or intersections would be hidden by pooling | stratified rows for SEL-1, SPD-1, MAP, BOX | `R0` |
| G9 | **Closed loop (T2).** Every number is open loop; NAVSIM is an open-loop benchmark; the PI's "not following the route" is judged on logged frames. | route and speed effects compound in closed loop | not provisioned; state the limit on every row | – |
| G10 | **Traffic-light and stop behaviour.** 779 labelled records carry a light state; no refcv7 measure of reacting to red/green exists. RED (376 positive records) and GREEN (363) are scoreable; YELLOW (22) and the colourless token (18) are under the 200-positive floor. | a safety-relevant behaviour with labels but no metric | RED/GREEN token AP + an action-at-red probe on a scored subset | `R1` |
| G11 | **Box train-vs-eval AP gap**, and whether the box head's memorisation on a small set (G-BOX-OVERFIT AP 1.000) transfers. Only the duplicate and calibration statistics are shown TRAIN vs EVAL. | separates capacity from generalisation as for the map (MAP-3) | AP@2 m on TRAIN-DIAG packs (already banked) vs EVAL-DIAG | `R0` |
| G12 | **NAVSIM at 15k and 20k**, and a NAVSIM-trained control for refcv7. Milestones exist at 5k / 30k / 50.4k only; REFe is trained on navtrain. | cannot separate a plateau from a domain gap | run the bridges on the banked 15k / 20k checkpoints | `R0-inf` |
| G13 | **Exact label coverage** (% windows with tactical GT, % windows where nav says turn but none follows within 6 s / 10 s, % windows whose own speed exceeds the fed ceiling). Currently E ≈ 23 %. | sizes TAC-1, NAV-1, SPD-4 | D1 (running) | `R0` |
| G14 | **Lane discipline** (oncoming / off-lane) for refcv7. LANE-1 measured it for REFe (2.83 % oncoming); NAVSIM navhard stage-2 means DDC 0.808 -> 0.839 and LK 0.477 -> 0.491 (5k -> 30k) are the only refcv7 reads. | the PI's video review flagged lane behaviour | LANE-1 harness on the refcv7 seam | `R0-inf` |
| G15 | **30k NAVSIM decomposition and families on navtest** (strata, single-term ceilings), and the 50.4k set. | NS-4/5/6 are 5k-only for the strata | run `decompose7.py` / `families7.py` on the finished seams | `R0` |

## 5. The four metric families — where they are missing for refcv7

| family | what exists for refcv7 (tier, n) | what is missing |
|---|---|---|
| **LONGITUDINAL** | open loop (RF, 800–1,112 windows, EC): speed MAE 0–2 s 0.262 [0.242, 0.286] m/s, along-track \|err\| 6 s, signed bias, ADE/FDE per class. NAVSIM navtest 5k (12,146 tokens, EC): speed MAE 0.7634, target-speed accuracy 0.4897 / 0.7421 / 0.926 (within 0.5 / 1 / 2 m/s), along bias +0.765, final +1.647 m, accel MAE 0.4251 m/s², progress ratio 1.184, hold-v0 BEATEN (Δ −0.5212 [−0.5782, −0.4652]), echo CLEAN. TTC/NC/EP sub-scores. | **distance keeping (headway, time gap, min TTC)**: UNAVAILABLE in RF and NAVSIM, pending in the battery (G3) · target-speed accuracy on the PhysicalAI surface (battery pending) · ceiling compliance with a CI (reel only) · anything at 15k/20k · 30k/50.4k navtest family blocks |
| **LATERAL** | RF: heading MAE 0–2 s 1.50° [0.70, 2.97] all / 4.01° turns, curvature MAE, cross-track \|6 s\| 2.21 [1.82, 2.60] m, terminal-heading error. NAVSIM 5k: heading MAE 4.031° [3.61, 4.47], **yaw-rate MAE 3.156 °/s [2.83, 3.47]**, curvature MAE 0.0147, cross MAE 0.467 m, cross-final 1.188 m; navhard 5k -> 30k heading 7.09° -> 4.72°. | **yaw-rate on the PhysicalAI surface** (battery A3 cell pending) · lane discipline / oncoming (G14) · lateral by stratum (G8) |
| **TACTICAL** | RF trajectory-derived 0–2 s (1,112 windows): lat acc 0.957 κ 0.823, lon acc 0.827 κ 0.540, goal-point FDE 0.617 m, bearing MAE 1.43°. NAVSIM 5k trajectory-derived 4 s: lat 0.852 / 0.714, lon 0.691 / 0.539, 5-way κ 0.602, goal FDE 3.55 m [3.40, 3.72]. Decoder-head side-correct on turns 39.3 % [27, 52]. Reel: decoder vs GT 288/479 lat, 283/479 lon. | **decision-grade decoder accuracy/κ on in-band windows** (battery) · **22-token goal AP/AUROC** (G6) · `g_tac` vs GT with CIs · **anchor/candidate selection quality as a tactical read** (RF has it as SEL-1, not through the family block) · tactical on dense windows (needs L1 labels) |
| **STRATEGIC** | `--no-strategic` (PI R5, 2026-09-27): **N/A, n = 0**, stated. Route-setting appears only as nav compliance against the oracle nav: turns 0.740 [0.580, 0.878], straights 0.021 [0.005, 0.042] (RF). | on NAVSIM the family is `UNAVAILABLE` and the package states it is **blocked on eval engineering, not on the corpus** (NAVSIM carries a lane graph and a route): a NAVSIM strategic label builder is the work item · nothing on PhysicalAI (no map/route in the published corpus) |

## 6. What D5 changed or verified

* `code/d5_verify.py` re-derives **109 numbers** from the raw JSON, scorer files and the label file: **109 OK, 0 fail, 0 unreadable files** (`raw/d5_verify.json`).
  Controls: **C-MUT** the comparison rejects a deliberately wrong expectation (read True); **C-MD5** the label file md5 equals the md5 the launch `config.json` records (`b45377a1f25263b5c0f3d318c126b1ac`); **C-N** 4,572 records;
  **C-UNREAD** 0 unreadable files. Independently verified from `LBL`: t0 = 8.0 s on all 4,572 records, L/R nav timing (median 7.3 s, 52.7 % > 6 s, 42.0 % > 10 s, p90 26.2 s), VLM disagreement 1,537 / 4,275 and 1,655 / 4,567, `lat_peak_m`, ceiling bins.
* One NavSim value is **new**: step-30,000 navtest R7_A1 (PDMS 71.8849 and sub-scores) is read from the scorer's own `counts.json` summary, which `NAV/RESULT.md` and the registry do not yet carry; D5 recomputed the same mean from the CSV (12,146 token rows, average row excluded) and it matches.
* Not re-verified (INHERITED into this file): the lever table of `RF/RESULT.md` §2 beyond W3 / V3 / V0c; A2/A3 CV figures; map `M_b/M_d/M_f` beyond what `M_c` and the paired lane CI check; G0 internals; registry binding-overfit records; step-cost shares.
* No raw clip UUID appears in this file or its raw output (the label file was read; only aggregates left the process).

## 7. Deliverable manifest

| artifact | where it lives |
|---|---|
| `EFFECTS_INVENTORY.md` (this file) | repo: `TanitAD Research Lab/Data Engineering/Research/2026-10-04-refcv8-data-audit/D5_effects_inventory/EFFECTS_INVENTORY.md` |
| `code/d5_verify.py` | repo: same folder, `code/` |
| `raw/d5_verify.json` (109 checks, controls) | repo: same folder, `raw/` |
| `LANDING_READY_D5.txt` | repo: `…/2026-10-04-refcv8-data-audit/LANDING_READY_D5.txt` |
| Thor state | none (Thor not touched; no process left running) |
