# PLAN_REFCV8 — understand every effect, measure it, validate the fix cheaply, then retrain once

*Master Mind, 2026-10-04. Status: **DRAFT v2 for the PI** — Phase A (the data audit, D1–D5) is DONE; the PI's binding
requirements R8-1…R8-7 (§0) are added and their work packages WP-A…WP-D are RUNNING; the open decisions are in §8.*
*PI, 2026-10-04: "I need a big plan for refcv8, I need to understand all effects define measures and validate them to
retrain without loosing a lot of time. ... is our training data ok? are any thing missing or uplausible, are we using
them wrongly? Why the video is saying, no gt label in this frame?"*

Every number below carries its evidence class. Planner numbers are open-loop on held-out eval139 from the launch tree
`fec3a0d` (one training seed), where the speed ceiling does not reach the emitted plan (SPEC_REFCV7 §26.1).

---

## 0. ⭐ BINDING PI REQUIREMENTS FOR refcv8 (PI, 2026-10-04 afternoon) and the work packages that deliver them

*PI, verbatim (abridged): "the refcv8 plan should contain highly quality proven measures to finally boost the results
of refc to frontier, this includes: the correction of our labels — each camera frame should have corresponding
tactical goals and actions in the future time frame (2 to 8 seconds from the current reference frame) with
corresponding constraints like distance and time (not only one per clip); each frame should have a corresponding nav
command; ... a route checkpoint extracted from the ego trajectory defining the next reference route point in the
vehicle coordinate system — not a label, the route waypoint goal as training and inference waypoint in addition to
the nav command, simulating the nav system of the car. You can assign this job to the data fly wheel agent. — We
should assure and validate that the tactical layer is learning tactical goals, actions and their constraints like
distance and time, and validate that they are constraining the trajectory planner and that the planner and its fan
are consistent to the different tactical labels. We should condition the fan generation and trajectory selection
with the tactical goals/actions, so the most probable actions are influencing the planning process. — We should
implement all your identified fixes for map and boxes. — We should optimize architecture, training workflow to
dramatically improve the map and box heads (do we need an initialization of the heads? or other tricks?) — Any other
fixes for identified problems of refcv7 from you?"*

| req | requirement | work package · owner | the measure that proves it (bars committed in SPEC_REFCV8 before any refcv8 number) |
|---|---|---|---|
| **R8-1** | **Per-frame tactical goals + actions over [NOW+2 s, NOW+8 s]** with constraints (distance, time, target speed, …), not one record per clip | **WP-A · Data FlyWheel** (`FlyWheels/TanitAD_DataFlyWheel/incoming/2026-10-04-v9-labels/`) | coverage ≥ 95 % of windows (refcv7: **23.22 %**, D1, n = 746,946); every turn window labelled (refcv7: 35.1 % of 88,238); agreement with an independently written geometry derivation ≥ 0.95 on TURN / STOP, with analytic controls + a mutation that must go red; constraint error (distance, time) vs the 100 Hz egomotion ≤ stated tolerance |
| **R8-2** | **Per-frame nav command** (announced junction turns, time-localised, with distance/time to the turn) | WP-A | nav says turn but no turn starts within 6 s: refcv7 **74.3 % of L/R windows** (205,292, D1) → target ≤ 5 %; realised turns fed a matching command: refcv7 61.0 % → target ≥ 90 % within the announcement horizon |
| **R8-3** | **Route checkpoint INPUT**: the next reference route point in the vehicle frame, from the ego trajectory, at training AND inference, beside the nav command (simulates the car's navigation system) | WP-A (data) + WP-B (model input) + EvalFlyWheel (NavSim bridge derives the same point from the scene's route) | ECHO controls: a trivial planner that drives to the checkpoint at v0 sets a floor the model must beat; a shuffled-checkpoint arm must lose the gain; the checkpoint must carry no speed information (R² of future speed from checkpoint ≈ from v0); lateral leak quantified (the ego's within-lane offset at the checkpoint vs a smoothed route). ⚠️ Optimistic on PhysicalAI by construction (ego-future-derived), exactly like nav — stated on every result |
| **R8-4** | **The tactical layer learns goals, actions and constraints; it CONDITIONS fan generation AND trajectory selection; planner + fan are consistent with the tactical decision** | **WP-B · Architecture & Inference** (`TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-tactical-conditioning/`) | (i) tactical accuracy on per-frame labels: lat/lon macro-F1, goal AP vs prevalence, constraint MAE (distance, time, target speed) vs a prior/constant control; (ii) **controllability**: forcing the tactical condition to TURN_L / TURN_R / LANE_KEEP / STOP-at-d on the same scene must move the fan and the pick accordingly (direction-consistent ≥ 0.95; stop within tolerance of d); a shuffled-condition arm must lose it; (iii) **consistency**: share of fan candidates and of the pick consistent with the conditioning action; (iv) route following: turn direction-correct pick ≥ 0.95, heading within 15° ≥ 0.70 (refcv7 0.84 / 0.51) |
| **R8-5** | **All identified map + box fixes** | **WP-C · Perception fixes** | F1 per-class map thresholds (lane 0.068 → 0.164, edge 0.000 → 0.042, MEASURED); F4 presence gates (conf_ratio 0.97 / 1.00 in band); **F4b centre-distance NMS** (AP@2 m 0.248 → 0.349, agent 0.131 → 0.300; boxes per object 2.12 → ~1.1) with the re-fitted gates; mask the ego as an agent box (18 clips, D3); clip track-id-switch rate targets (6,410 frames, D3); z/h claims near-field only; drop duplicate videos + recording-shared clips (D3) |
| **R8-6** | **Architecture + training workflow that DRAMATICALLY improves map and box** (head initialisation? other tricks?) | **WP-D · Perception architecture** (literature banked + frozen-BEV-feature probes + v7-tiny) | per-class AP@{0.5, 1, 2} m by range bin, boxes per object, conf_ratio; map IoU per class × range with 0.2 m tolerance; each trick pre-registered with a regression arm; the remaining localisation × range ceiling (edge IoU 0.062 at 0–20 m → 0.003 at 80–100 m) is the target |
| **R8-7** | **Every other refcv7 fix the Master Mind identified** | WP-B / WP-C / WP-E (Master Mind) | listed in §0.2 with its own measure |

### 0.1 What changed in the strategy because of R8-1…R8-6

R8-3 adds an INPUT, R8-4 changes the decoder (conditioning) and R8-6 may change the lift and the heads. A head-only
branch from 50,400 can no longer carry all of it. **New default: a WARM-STARTED full run** — every module whose shape
is unchanged starts from refcv7 step 50,400 (trunk, lift, BEV, existing heads), every NEW path (route checkpoint
token, tactical conditioning, new head parts) starts zero-gated so step 0 reproduces refcv7, then ~30k steps on the
v9 labels (≈ 3.5–4 days at ~10–11 s/step on Thor). R1 (head-only on a frozen trunk, building now) still runs first:
it measures how much of R8-4 the existing trunk already supports, and it is the fastest place to debug the
conditioning before the trainer change. A from-scratch run (~6 days) only if the warm start is shown to block a fix.

### 0.2 R8-7 — the Master Mind's additional fixes (each with its measure)

| # | defect (evidence) | fix | measure |
|---|---|---|---|
| X1 | The pick loses the turn the fan contains (fan 1.00 → pick 0.84; heading-15 0.76 → 0.51) and speed-profile choice is the largest planner lever (B3 −0.606 m) | the selector is TRAINED on the emitted fan with a listwise target over direction + speed-profile error (not only the decoder score) | pick regret vs oracle and random on the same windows; B3 bound captured |
| X2 | Residual prior: 14 of 17 wrong-direction turn picks follow its side (post hoc, route package) | residual-prior dropout + conditioning the prior on nav/route | wrong-direction picks that follow the prior |
| X3 | The speed ceiling never reached the emitted plan (SPEC_REFCV7 §26.1); max-speed input was a future oracle (R² 0.988 in band) and its "unknown" row was never trained, while NavSim feeds "unknown" on 45 % of navtest tokens | past-only max-speed input N2 (leak 3.5 %) + a trained "unknown" row + the ceiling applied to the EMITTED plan | input-removed / input-shuffled arms; plan > fed ceiling (refcv7 110 / 2,059 reel windows) |
| X4 | Tactical loss is 0.29 % of the total loss (D4); `LANE_CHANGE_L` was trained as a negative on 100 % of tactical windows because the eval label load overwrote module state the DataLoader workers read (D1, new) | a loss budget sized from measured gradient shares; label-state isolation (no module-level state across loads) + a test that pins it | per-family gradient share logged; the declared-vs-built census equals what the workers see |
| X5 | NUDGE labels curves (10–12 % agree, D2); 3 of 8 lateral and 1 of 8 longitudinal classes have ZERO windows (D1) | the v9 vocabulary drops dead classes and redefines NUDGE (or the 3-way lateral head) | per-class support printed with every tactical number |
| X6 | NavSim navhard FAILS (EPDMS 0.227 vs STOP 0.299 at 30k; DAC-zero 26.0 %, NC-zero 15.6 %); cause unattributed | D6 failure anatomy (running) maps each failure class to a lever; the NavSim bridge feeds announced junction turns + the route checkpoint from the scene route | navhard EPDMS vs STOP, DAC/NC-zero rates, per failure class |
| X7 | No distance-keeping metric anywhere (G3); the box store's x ≤ 61 m scope excludes 84 % of joined agents (D4) | extend the agent store range; implement headway / time-gap / TTC on the lead | the longitudinal family complete |
| X8 | The four-family battery has produced no number for any refcv7 checkpoint (G0 estimator issues, G4) | A5/A6 battery amendments carried; the 50,400 battery runs before refcv8 launches | four families present for refcv7-50.4k as the baseline row |
| X9 | One training seed everywhere; the bootstrap answers "another draw of episodes" only (H-ESTIM-SEED-1) | a second seed of the refcv8 run if compute allows; otherwise every lever claim is read against the v7-tiny replicate floor | replicate arm |
| X10 | Poses lead the image by 0–34 ms (mean 0.19 m, D3); day/night is a clock label (D3); left turns on eval rest on 13 clips (D3) | interpolate poses to the camera timestamp; stratify night by brightness; report left-turn n | — |

**X6 answered by D6 (NavSim failure anatomy, 2026-10-04, `…/2026-10-04-refcv8-data-audit/D6_navsim_failure_anatomy/RESULT.md`;
navhard step 30,000, all 5,912 tokens, failing plans re-scored exactly in the local metric cache, max diff 0.0):**
* **DAC-zero 1,563 scenes — the plan's failure, not the scene's:** a clean path exists (PDM-Closed reference clean) in
  **1,136 = 72.7 % [68.2, 77.4]**; 0 leave the drivable area at t = 0; the second inference seed is clean on only 7.7 %.
  Classes: LATERAL-DRIFT 30.9 %, ROUTE-FOLLOWING (under-turn) 23.2 %, ON-ROUTE-still-fails 18.4 %, OVER-STEER 13.0 %,
  SPEED 9.4 %, NO-RECOVERY 2.7 %, WRONG-SIDE 2.2 %. 5k → 30k: drift and route-following shrank, SPEED doubled (67 → 147).
* **NC-zero 889 scenes — LONGITUDINAL:** **843 = 94.8 %** are front collisions into a vehicle ahead (moving 56.7 %,
  stopped 34.5 %); the plan's speed at 4 s exceeds the reference by a median **+3.87 m/s** in those scenes (−0.48
  overall); the learned planner does not beat its own prior on NC (15.6 % vs 16.3 %). ⇒ distance keeping / braking for a
  lead is a refcv8 lever in its own right: selector speed (X1), dense longitudinal labels with FOLLOW / BRAKE_TO / HOLD
  and lead constraints (R8-1), lead perception (WP-C/WP-D), and the distance-keeping metric (X7).
* **Premature turns (association):** on 724 scenes with a LEFT/RIGHT command but a straight route inside the horizon the
  plan turns anyway in ~59 % — NavSim's command carries no distance; refcv8's nav carries distance/time (R8-2) and the
  NavSim bridge must derive the same announced turn + route checkpoint from the scene's route (EvalFlyWheel).
* The max-speed "unknown" row (45 % of tokens) is NOT a disproportionate failure source (navhard DiD vs STOP −1.7 pp
  [−5.7, +1.7]); no cheap train/deploy fix is indicated there.
* Counterfactual (ESTIMATED upper bound): matching STOP's DAC where STOP is clean lifts official EPDMS 0.2269 → 0.3504
  (above STOP's 0.2985).
* **P3 lever ceilings (MEASURED, exact devkit re-score of EDITED plans — upper bounds, not selectable candidates; controls
  reproduce the banked scores exactly):** a perfect SPEED fix (best of 0.8 / 0.6 / 0.4 × the planned speed) clears
  **46.0 % [40.4, 51.2]** of navhard NC-zero (official EPDMS ceiling 0.2269 → 0.2880, still below STOP) and **93.7 %** of
  navtest NC-zero (PDMS ceiling 71.88 → 78.82); the cleared plans still travel 13.2 m (reference 11.3 m) — not "stop".
  A ROUTE-FOLLOW oracle (the plan's own progress laid on the route centreline) clears **51.4 % [46.4, 57.4]** of navhard
  DAC-zero (official EPDMS ceiling **0.3403, above STOP**) and 85.6 % on navtest (PDMS 76.95); a bounded ≤ 1.5 m lateral
  shift clears only 17.3 % ⇒ the errors are gross route errors, not boundary errors. **Ranking on navhard: the route /
  lateral lever (+0.113) exceeds the whole deficit to STOP (−0.072); speed alone (+0.061) does not.** On navtest the
  collisions are almost purely a speed-commitment failure. Next: P1 (does the 117-fan hold a clean candidate on these tokens, and does the pick take it?)
  and P2 (NAVOFF on the premature-turn scenes) — pre-registered, queued behind the battery on the dev-box GPU.

**R0b CLOSED (A6 under A7, 2026-10-04 ~14:10):** the DEPLOYABLE time-localised nav (announced turns only, soft rule
T3a) **PASSED all four criteria** on a dense capture of the 139 held-out episodes (4,634 windows per sampler seed):
turn ΔADE −0.131 [−0.207, −0.062], direction +0.064, straight +0.002, replicated on seed 1 (−0.127); the derangement
control failed as required. It is SMALL — 0.270 of the perfectly-timed-nav bound, −0.015 m all-window at natural
window frequency — and 56 % of turn windows carry no nav signal. ⇒ it ships as an opt-in inference rule for refcv7, and
L2 is confirmed for refcv8; the bigger levers stay the tactical constraints (D0: progress −0.484 m) and the selector.

### 0.3 WP-RL — DiffusionDriveV2 RL post-training of refcv7 as a parallel extension (PI, 2026-10-04 afternoon)

*PI: "At parallel we should plan a post training with RL exactly as stated in the DiffusionDrive paper 2 as extension
to see the effect. Run it on the Thor and evaluate the results."* Package:
`TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv7-ddv2-rl-posttrain/`.

* **Object:** refcv7-r101-s0 at 50,400. **Method:** the paper-exact spec already banked (arXiv 2512.07745 + released
  code, `…/2026-09-15-ddv2-rl-prep/SPEC_DDV2_RL_PAPER.md`): truncated-diffusion policy, η = 1 / 0, multiplicative
  exploration noise, intra-anchor GRPO + inter-anchor truncation, L_RL + 0.1·L_IL, AdamW 2e-4, reward = PDMS
  (NC × DAC × (5·EP + 5·TTC + 2·C)/12). Every deviation our setting forces (117 anchors, PhysicalAI, DAC from SAM3
  drivable area, one Thor) is tabulated.
* **Why it can work now when it did not before:** the programme's earlier ports FAILED / HARMED (H-DDV2RL-1,
  H-DDV2RL-2: ADE 0.2994 → 0.5502 at T1), and the attribution showed the RL-OFF control moving as much as RL. The
  reward was broken: no DAC (no map on PhysicalAI) and a saturating ego-progress term. SAM3 maps now cover all 4,369
  train clips (D3), which unblocks the DAC term — the condition the repaired-reward pre-registration (H-DDV2RL-3)
  was waiting for.
* **Arms:** RL (seed 0), RL-s1 (replicate, if the budget allows), **RLOFF** (same steps, RL weight 0 — the control that
  decided H-DDV2RL-2), BASE, RL-SHUF (rewards shuffled — must not beat RLOFF). Primary endpoint T1 on held-out eval139,
  four families, RL vs RLOFF (the RL effect) and RL vs BASE; secondary: route set, NavSim navtest / navhard zero-shot.
* **Compute:** Thor, ≤ ~24 GPU-h for the RL arm, checkpointed every ~45 min and stoppable/resumable, so the refcv8
  critical path (A7 capture, R1, v7-tiny ladder) takes Thor's GPU in short slots; launch gate binding before training.

---

## 1. The answer to "why does the replay say *GT goals: not labelled at this instant*?"

MEASURED (label file `s2_labels_v8_train.jsonl.gz`, md5 `b45377a1…`, 4,572 records; trainer `refc_v3_train.py:3884-3916`,
`v7_labels.py:678-710`):

* The tactical labels (8 lateral + 8 longitudinal actions, 22 goal tokens) come from **ONE record per clip**. Every one of
  the 4,572 records is anchored at the **same instant, raw t0 = 8.0 s**, with a tactical band of 2–6 s ahead.
* The trainer accepts a window only when its NOW is within **±2.0 s of t0** (`window_in_band`). The clip cache covers
  raw ≈ 0.1–20.1 s (`grid_start_s` median 0.113 s, dt 0.1007 s), so windows whose NOW is outside **6.0–10.0 s** get
  IGNORE: no tactical loss in training, and "not labelled at this instant" in the replay.
* ESTIMATED ≈ 23 % of windows labelled (≈ 40 of ≈ 171 per clip); stream D1 measures it exactly. Route package, MEASURED:
  the lateral label is IGNORE on **72 of 107** eval turn windows.

The replay is reporting the truth: the tactical head of refcv7 was trained on about a quarter of the windows, and on
the minority of turns that fall inside that 4-s band.

## 2. What we already know about the data (before the audit finishes)

| # | finding | evidence | consequence |
|---|---|---|---|
| K1 | Tactical + goal GT only within ±2 s of one anchor per clip (§1) | MEASURED | tactical head barely sees turns; selection cannot lean on it (lateral head right side 0.39 on turns) |
| K2 | The **nav input is one token per clip, fed on every window**. For L/R clips the turn starts a median **7.3 s** after the anchor; 52.7 % > 6 s, 42.0 % > 10 s (p90 26.2 s) | MEASURED, label file | nav says "turn" while the car drives straight: 193/291 L/R-nav eval windows are straight; the nav-compliance gate learned 0.163 (near inert) |
| K3 | The time-localised route **is already in the file** (`nav_30s.entries`: every turn's start/end time and arc-length distance) — refcv7 did not use it | MEASURED | the cheapest nav fix is a loader change, not new data |
| K4 | The **max-speed input is one value per clip** = the ego's realised max speed over [t0+2, t0+6] s, fed on every window. The label file itself records that the bin recovers **75.4 %** of the future-speed information v0 lacks, and that **75 % of intersection clips** get ≤ 30 km/h because the ego was stopped | MEASURED (quoted from the file's own audit fields) | an oracle hint that is stale away from the anchor and wrong at stops; speed profile is the largest measured planner lever (−0.606 m ADE if right) |
| K5 | The VLM (Alpamayo) disagrees with our geometric tactical label on **36 %** of lateral (1,537/4,275) and **36 %** of longitudinal (1,655/4,567) records | MEASURED | label noise of unknown direction — D2 decides which side is right |
| K6 | `a_tac.lat_args.lat_peak_m`: median 19.7 m, max 305 m | MEASURED; **SETTLED by D2**: mis-NAMED, not broken — the peak lateral displacement in the first-sample frame over the clip's first 20 s (reproduced at corr 0.9995, n = 4,369); refcv7 never read it | docstring fix only |
| K10 | **NUDGE labels curves, not nudges**: on train only 10.3 % (NUDGE_L) / 12.1 % (NUDGE_R) agree with an independent ego geometry; the heading never returns on 89.7 % of 1,028 NUDGE records; NUDGE is 23.5 % of the lateral mass. TURN agrees 85–87 %, LANE_KEEP 92 %; lateral overall 72.1 % (3,149/4,369) | MEASURED (D2, analytic controls 17/17) | the tactical head is trained to call a road curve a "nudge" |
| K11 | Longitudinal labels are computed over [0, 6] s, not the declared [2, 6] band (HOLD/CREEP excepted); the effective ACCELERATE bar is +1.0 m/s, not the documented +1.5. As the trainer applies them, agreement decays 80 % → 64 % across the ±2 s band | MEASURED (D2) | a band-vs-computation mismatch; fix the window and the bar |
| K12 | In-band, the fed speed ceiling equals the window's own realised future max at **R² 0.988** (mean abs error 0.44 m/s): a near-perfect future-speed oracle. Outside the band it is stale (over all windows v_now beats it, R² 0.899 vs 0.889); 11.9 % of windows exceed their bin; slow ego ⇒ low bin (intersections ≤ 30 km/h 74 %, 69.8 % even when moving) | MEASURED (D2, 100 Hz egomotion) | the largest measured planner lever (speed profile) rides on an oracle input; L3 must be redesigned, see §8-2 |
| K13 | Nav files: `build_v8_nav30s.py:83` reads `suppressed` while the emitter writes `applied`, so 66 suppressed turns stay in `nav_30s`; 68 `nav_command` turns (> 30 s ahead) have no `nav_30s` entry. Only 24.7 % of TURN-token windows see a same-side turn begin in their own next 6 s | MEASURED (D2) | L2 must use announced entries only and fix the cap |
| K7 | 5 of 22 goal tokens are untrainable (no supervised negatives); several have ≤ 23 positive clips in 4,572 (LANE_CHANGE_L 23, LANE_CHANGE_R 15, OVERTAKE 20, TAKE_EXIT_L 20, YIELD_FOR_TURN_L/R 21/20) | MEASURED (config.json `tac_goal_stats`; label file) | those tokens cannot be learned from this corpus as labelled |
| K8 | Lateral classes: LANE_KEEP 65 %, NUDGE 24 %, TURN 12 % of records | MEASURED | turns are rare AND mostly outside the band |
| K9 | No map topology, lane graph, posted speed limit or traffic-light state exists in the published PhysicalAI-AV corpus; our only route and speed-limit suppliers are the ego's own future | PUBLISHED (dataset card) + programme record | nav and max-speed inputs are optimistic by construction; say so on every result |

| K14 | **Raw data is sound (D3)**: 897 k pose rows with 0 non-finite and no physically implausible yaw-rate / lateral-acceleration rows; 0 black / frozen / over-exposed frames in 88 k decoded; 27.6 M boxes all carry z and h; the ego's next-6-s path is road-like on 99.92 % of the SAM3 map GT; train/eval balance within ~1 pp on turns, stops, speed, night. **The GT boxes do NOT carry duplicates** (0.34 % of boxes) — the reel's ~2.1 boxes per object is prediction-side | MEASURED (D3, controls read known values) | no raw-data blocker for refcv8 |
| K15 | **Split leakage across recordings**: no shared clip id, but 3 (image-confirmed) to 5 eval clips share a ~140 s source recording with a train clip; 135 train recording groups cover 296 clips (6.8 %); 2 train pairs are the same video under two ids. **The ego appears as an agent box in 18 train clips (693 frames)** | MEASURED (D3) | eval mildly optimistic on ~2–4 % of clips; fix both in the refcv8 corpus manifest |
| K16 | **Eval caveats**: left-turn route following rests on **13** eval clips (eval nav mix skewed, p = 0.0062); day/night is a clock label (only ~8 % of clips are actually dark vs 46 % "night"); box z is reliable near-field only; 13.4 % of train clips have no lane line (real, by country) | MEASURED (D3) | report left turns with that n; cut night by brightness; no z claims beyond ~30 m |

| K17 | **Usage audit (D4)** — three channels WRONG, each with a measured cost: nav (one token per clip; the nav-compliance graft's own training signal is right on only **35.7 %** of informative windows, which is why its gate stuck at 0.163), tactical labels (23.4 % of windows supervised; left/right-nav records carry LANE_KEEP on 49–52 %; all v7-label tactical terms together are **0.29 % of the loss**, no class weights), max speed (future oracle; the "unknown" row NavSim feeds on 45 % of navtest tokens was never trained). SUBOPTIMAL: the residual prior (14 of 17 wrong-direction turn picks follow its side), goal-token negatives (the PI's 2026-09-16 caption-absence ruling never reached the run — the sidecar is bound to another label blob), ego dropout 0.5 withholds the speed input on half of training, the box store's x ≤ 61 m scope excludes 84 % of joined agents (distance keeping) | MEASURED, label-level (D4; controls: eval grid rebuilt bit-exactly, A5 reproduced, label file's leak 75.3 % vs 75.4 %, analytic tracks, two mutations red) | the design in `REFCV8_LABEL_DESIGN.md`: L1 labelled windows **23.4 % → 88.9 %**, turn windows 36 → 107 / 107; L2 wrongly commanded straight windows **193 → 14 / 588**, graft signal 35.7 % → 80.0 % right |

**Verdict so far:** the images, poses, boxes and maps are sound (K14) apart from two small defects (K15). The **labels are used wrongly in time**: per-clip
records are applied **too narrowly** for tactical supervision (±2 s) and **too broadly** for the nav and speed inputs
(the whole 20 s clip). This is consistent with, and explains, the route-following diagnosis: the 117-candidate fan holds
a correct turn on 100 % of turn windows, but the pick turns the right way on only 84 %, and its heading is within 15° on
only 51 % (best available 76 %).

## 3. Phase A — data audit (RUNNING since 2026-10-04 ~12:10 Berlin; ETA ~15:00)

Package: `TanitAD Research Lab/Data Engineering/Research/2026-10-04-refcv8-data-audit/` (shared `CONTEXT.md`).

| stream | question | measures it produces |
|---|---|---|
| **D1** coverage census | which channel carries a value on which window? | the per-window label truth table (746,946 train + 23,772 eval windows, built through the trainer's own dataset code) with ego-future geometry; % windows with tactical GT, % turn windows labelled, % windows where nav says turn but no turn comes within 6 s / 10 s, % windows whose own speed exceeds the fed ceiling. Controls: reproduce 72/107 and 719,739/746,946 |
| **D2** label validity | do labels mean what we assume, and agree with the ego's own motion? | field-by-field semantics with file:line; confusion matrices of a_tac lat/lon vs an independent geometric derivation (analytic control: a circle must read "turn"); VLM-vs-geometry verdict; lat_peak_m verdict; speed-ceiling validity |
| **D3** raw plausibility | are poses, boxes, maps, frames and the split sane? | train/eval recording overlap; pose jumps/accel/yaw-rate outliers; GT box duplicates and implausible sizes; SAM3 GT drivable-on-own-path; frozen/black frames; rig mix; corpus balance |
| **D4** usage + design | are we using the data wrongly; what replaces it? | per-input row: per-clip vs per-window, oracle at inference, train/deploy mismatch (incl. what NavSim feeds), loss share, verdict; `REFCV8_LABEL_DESIGN.md` with each replacement's admissibility check, validation test and targeted number |
| **D5** effects inventory | every measured refcv7 effect in one table | effect · size + CI · tier · root-cause class · lever · measure · cheapest validation rung; gaps with no measure; missing metric families |

## 4. Phase B — the v9 label set (ZERO GPU; ~1 day)

Built from data we already hold. Every change ships with an independent cross-check, an analytic or known-value
control, and a mutation that must go red.

| change | definition (final form from D4's design) | validation (must pass before any GPU) |
|---|---|---|
| **L1 dense tactical labels** | lat/lon action for **every** window from the ego's own next 6 s (labels may use ego — PI 2026-08-03); NUDGE redefined (it labels curves today, K10) or replaced by a 3-way lateral head LANE_KEEP / TURN_L / TURN_R as the first arm; longitudinal on one declared window with the documented +1.5 m/s bar (K11); VLM-only semantics stay anchor-bound unless D4 gives a propagation rule | coverage ≥ 95 % of windows; on eval turn windows agreement with the turn direction ≥ 0.95; a synthetic circle reads TURN, a straight reads LANE_KEEP; time-shuffled labels must FAIL the agreement bar |
| **L2 time-localised nav** | per window: the next upcoming `nav_30s` entry → token + distance/time to the turn; FOLLOW once no turn is ahead | 0 windows with a turn token and no turn inside the stated horizon; the 193/291 straight-under-L/R count must fall to the expected residual (turns beyond 6 s, reported separately) |
| **L3 per-window speed input** | DECISION §8-2: (a) window-local realised max snapped UP to the posted ladder, or (b) a coarser posted-limit proxy | the file's own OOF test (R² of the future max from v0 vs from bin+v0) reported per option — the leak is measured, not assumed; a shuffled-ceiling arm must lose the speed gain |
| **L4 goal negatives** | supervised negatives for the 5 masked tokens where geometry can supply them; drop tokens with < 50 positives from the loss | per-token positive/negative counts; AP floor vs prevalence |
| **L5 fixes from D2/D3** | whatever the audits mark DEFECT (e.g. lat_peak_m, duplicate GT boxes, misregistered maps) | each with its own known-value control |

## 5. Measures — the pre-registered metric set for refcv8 (all four families + the route set)

Bars are committed in SPEC_REFCV8 before any refcv8 number exists; the baseline row is refcv7 at 50,400 on the same
windows. Estimator: paired episode-cluster bootstrap over eval139; a **replicate** (second sampler seed AND, for the
final run, the question "would another training run say this?" stated) for every lever claim.

| family | measure | refcv7 baseline (MEASURED) |
|---|---|---|
| route / LATERAL | turn-direction-correct pick on GT-turn windows; heading within 15° at 6 s; curvature and yaw-rate error; cross-track | 0.84; 0.51 (fan best 0.76); turn ADE 3.19 m (fan best 1.19) |
| selection | fan→pick drop (+ random and oracle on the same windows); pick regret | +0.159 [+0.080, +0.258]; random 0.39 |
| LONGITUDINAL | speed MAE 0–2 s / 2–6 s; speed-profile ADE attribution; distance keeping (headway, TTC to lead); ceiling compliance | speed lever −0.606 m; plan > fed ceiling on 110/2,059 reel windows |
| TACTICAL | lat/lon accuracy + macro-F1 on DENSE v9 labels and on the anchor band (continuity); goal-token AP vs prevalence | lateral head right side 0.39 on turns |
| STRATEGIC | N/A — strategic layer OFF by PI ruling R5 (2026-09-27); reported as absent with the reason, never silently dropped | — |
| nav compliance | on windows with a turn starting within 6 s: pick turns the commanded way | to be measured on v9 nav (D1 gives the denominator) |
| perception | map thin-class IoU (F1 thresholds), box AP@2 m, boxes/object, conf_ratio | lane 0.164, edge 0.042; AP 0.248 / 0.131 → 0.349 / 0.300 with NMS; 2.12 boxes/object |
| closed loop (NavSim) | navtest PDMS vs STOP; navhard two-stage EPDMS incl. DAC-zero and NC-zero rates; warmup S2-EPDMS-u | 5k: PDMS 65.60 vs STOP 61.82 PASS; 30k navhard 0.227 vs STOP 0.299 FAIL (DAC-zero 26.0 %, NC-zero 15.6 %); 50.4k RUNNING |

Controls on every panel: STOP / hold-action, refcv7-50.4k itself (paired), a deliberate-regression arm per lever (e.g.
time-shuffled dense labels, permuted nav), and the identity controls the replay already carries.

## 6. Phase C — the validation ladder (this is where the time is saved)

Each lever must clear its rung before it may enter the run. Rungs run in parallel where they do not share a GPU.

| rung | what | cost | gate to the next rung |
|---|---|---|---|
| **R0** zero-GPU | Phase B label checks; leak tests | hours, CPU | every L-change passes its control and mutation |
| **R0b** time-localised nav at inference — **A5 DONE** | route package addendum **A5** (sha256 `5f102584…`, registered 10:01:55Z before any number; A4's bound for a perfectly timed nav: −0.566 m on turns): **reported arm T2 FAILED** on criterion 4 only (seed 0: turn ΔADE −0.264 [−0.568, −0.0005], turn direction +0.093, straight +0.009 [+0.001, +0.020], all −0.043; seed 1 turn −0.204 [−0.447, +0.028] — not separated). Straight damage vs the clip token: **+0.326 → +0.009**. Control T2c failed as required. Secondary T3 (soft ×10) cleared all four criteria (turn −0.159 [−0.312, −0.040], seed 1 −0.143 [−0.286, −0.032]) — secondary, not promotable. Post hoc: 54 % of T2's turn gain came from turns `nav_command` had SUPPRESSED (curves, obstacle passes) — information a real nav would not give; nav is silent on 55 of 107 GT-turn windows (that half needs a vision turn cue, L1) | done | ⇒ **A6** (sha256 `b4099578…`, 10:32:25Z): T3 with ANNOUNCED turns only (the deployable nav) as the reported arm, on a denser capture of the same 139 held-out episodes (Thor GPU, ~30–40 min). RUNNING |
| **R1** head-only, frozen trunk | cache refcv7-50.4k's decoder inputs (trunk + BEV features) for a seeded train sample (~1,000 clips × 20 windows) and all eval139 windows on Thor; retrain ONLY the tactical decoder and the selector on v9 labels | ~1–2 h caching + minutes per arm | ⭐ **decides branch vs full retrain**: if dense labels + time-localised nav lift turn-correct pick ≥ 0.95 and heading-15 ≥ 0.70 with the trunk frozen, the information is in the trunk ⇒ branch fine-tune; if not ⇒ the trunk must learn it ⇒ full run |
| **R2** v7-tiny ladder | the real trainer at ~19 M params (~17 min/arm on Thor): each L-change ON + its deliberate regression; declared-vs-built and gradient-reach checks | ~3–4 h | each lever moves its metric and its regression arm does not |
| **R3** launch gate | `run_gate.py` PASS token (binding since 2026-09-26) | ~3 h | PASS |
| **R4** branch run `refcv8-b` | fine-tune from refcv7 50,400 with v9 labels + the selector fix + ride-alongs (R4 speed bundle, F1 map thresholds, F4 gates, box NMS at inference); 12k steps | ~1.4 days at 9.9 s/step | — |
| **R5** eval | the §5 metric set, G0, NavSim navtest/navhard/warmup, route set, replay + tactical video | ~1 day | SPEC_REFCV8 bars |
| (R4′) full run `refcv8` | only if R1 says the trunk lacks the information, or §8-4 approves trunk changes (F3 stride-4 map tap) | ~6 days (refcv7: 2026-09-28 00:03 → 10-03 20:38) | — |

Why branch-first: refcv7's in-run eval plateaued from ~35,000 (MEASURED, registry block REFCV7-2026-10-04-FINAL), the
measured defects sit in labels, inputs and selection, not in the trunk, and R1 tests that assumption directly in hours
instead of assuming it for six days.

## 7. Timeline (Berlin)

| when | what |
|---|---|
| Sun 2026-10-04 (today) | Phase A audit lands (ETA ~15:00); D1–D5 results → this plan's §2 updated; register rows |
| Mon 10-05 | Phase B v9 labels + R0; R1 feature caching on Thor in parallel; SPEC_REFCV8 pre-registration drafted |
| Tue 10-06 | R1 head-only arms (hours); R2 v7-tiny ladder; R3 launch gate; **PI go** → R4 launches Tue evening |
| Wed–Thu 10-07/08 | R4 branch run (~1.4 days) |
| Fri 10-09 | R5 evals; report with the four families and NavSim |

≈ **5 days to a refcv8 result**. The full-run path (R4′) adds ~4.5 days. NavSim leaderboard submission stays OFF until
the leaderboard is examined with the PI (PI 2026-10-04).

**⇒ REVISED for the PI's R8-1…R8-7 (§0), ESTIMATED:**

| when | what |
|---|---|
| Sun 10-04 (running since ~14:00) | WP-A v9 label SPEC; WP-B design (tactical conditioning, route checkpoint, selector, X1–X4); WP-C perception fixes; WP-D perception design + literature; R1 harness + identity; A7 nav capture; D6 NavSim failure anatomy |
| Mon 10-05 | WP-A eval139 build + echo/leak study → train build; R1 head-only arms incl. tactical conditioning H5 (hours); WP-D frozen-BEV probes; WP-C fixes landed |
| Tue 10-06 | WP-B implementation + warm-start identity test; v7-tiny ladder for every new path with its regression arm (~17 min/arm on Thor); the refcv7-50,400 four-family battery as the baseline row |
| Wed 10-07 | SPEC_REFCV8 registered (every bar of §0 and §5); launch gate; **PI go** → refcv8 launch (warm-started from 50,400) |
| Wed 10-07 → Sun 10-11 | refcv8 run, ~30k steps (~3.5–4 days); Training Watch live |
| Mon 10-12 | evals: §5 metric set + the R8-4 controllability/consistency suite + NavSim + replay/tactical video |

≈ **8 days to a refcv8 result** under the wider scope; ~10 if the warm start has to be dropped. The label build (WP-A)
and the conditioning implementation (WP-B) are the critical path.

## 8. Decisions for the PI

1. **Branch-first** (fine-tune from 50,400, full run only on R1 evidence) — recommended.
2. **Speed input L3** (REVISED after D2, K12): the in-band ceiling is the window's own future max at R² 0.988, so a
   per-window 4-s realised max would put a speed-profile ORACLE on every window — and speed profile is the largest
   planner lever we measured, so a gain could be the leak. The input stays (PI requirement R1, 2026-09-27; the PI's
   2026-09-16 authorisation of a future-ego value inside its posted-limit window still governs), but closer
   to what a posted limit is: per window, the max realised speed over a long stretch of road ([NOW−10 s, NOW+10 s])
   snapped UP to the 8-step road-law ladder {20, 30, 50, 70, 80, 100, 120, 130} km/h (already in the v8 file), with
   the leak (OOF R² of the 6-s future max from v0 vs from bin + v0) and a shuffled-input arm reported beside every
   speed result. Default: this; alternative: the 4-s per-window max (stronger signal, larger leak).
   **⇒ SUPERSEDED by D4's measurement (2026-10-04 ~13:00):** ANY value derived from the ego's future is an oracle — a
   per-window realised max leaks **57.0 %** of the future-speed information v0 lacks (71.7 % on [NOW+2, NOW+6]). D4
   measured NON-oracle options that read only PAST ego speed (admissible at inference: the car knows its own history):
   **N2** past-20-s max snapped up to the road-law ladder with a 50 km/h urban floor — leak **3.5 %**, the human
   exceeds it on 6.5 % of windows; **N3** coarse urban / rural / motorway — leak 1.9 %, exceeded 4.4 %; no input — 0 %.
   **New default: N2**, trained with an "unknown" row on a share of windows (NavSim feeds the never-trained all-zero
   "unknown" row on **45 %** of navtest tokens — a train/deploy mismatch refcv7 carries), and every speed result
   reported beside the input-REMOVED and input-SHUFFLED arms. The PI's 2026-09-16 future-ego authorisation is then not
   needed; R1 (input + cap) is satisfied by a past-only limit.
3. **Nav L2** is a supplied route ("turn left in X m"), optimistic on PhysicalAI because it comes from the ego's own
   future — the same caveat as today's token, now time-correct. Default: adopt.
4. **Trunk changes** (map F3 stride-4 tap; anything that needs R4′): defer to after refcv8-b unless R1 forces a full run.
   **⇒ SUPERSEDED by R8-6 (PI 2026-10-04):** perception architecture changes are IN scope; WP-D ranks them by banked
   evidence and probes; those that fit a warm start (zero-gated new paths) go into refcv8.
5. (new, R8-3) **Route-checkpoint variant** — fixed arc-length lookahead (RC-A, L ∈ {30, 50, 80} m) vs next decision
   point (RC-B): WP-A's echo/leak study recommends; the PI confirms before SPEC_REFCV8.
6. (new, §0.1) **Warm-started full run** (~30k steps from 50,400, new paths zero-gated) instead of a head-only branch
   — recommended; from scratch only if the warm start is shown to block a fix.
7. (new) **Tactical loss budget**: refcv7 pinned it at 0.1 (0.29 % of the realised loss, D4). Default: size it from
   measured gradient shares on the v7-tiny ladder.
8. (new, from WP-C/D3) **Train corpus re-selection.** Dropping the 6 train clips that share a recording with eval clips
   and one clip of each of the 2 duplicate-video pairs RE-SELECTS episodes (the programme's parity invariant; refcv7's
   own corpus is not the parity corpus, but the rule is general). Default: **keep the refcv7 train corpus unchanged**
   (comparability with refcv7), mask only the 693 ego-as-agent-box frames (a label fix, not a re-selection), and report
   every eval139 number also on the 134 clips that share no recording with train (a leak-free sensitivity row).
   Alternative: drop the 8 clips (≈ 0.2 % of train). PI's call.
9. (new, WP-D B4) **LiDAR depth target.** PhysicalAI-AV ships `lidar_top_360fov`, but no projected depth target exists;
   building one streams ~342 MB per clip (≈ 1.5 TB over 4,369 clips, ESTIMATED) — a Data FlyWheel job and a download
   decision. Published evidence is two-sided (BEVDepth +2.2 mAP, MapTRv2 +5.1; DualPathOcc −0.96 mIoU with a hard depth
   target). Default: defer to after refcv8's first result unless P-BOX/P-MAP show a longitudinal-placement ceiling the
   other levers cannot move.
10. (new, WP-D P-TEMP) **Temporal BEV fusion** warps past BEV features with the ego's PAST motion. refcv7 already feeds
    the observed window's ego track (`--ego-history`, PI ruling 2026-09-02 on measured state at cycle time). Default:
    treat past ego-motion for warping as admissible under the same ruling; the probe stays deferred until the PI confirms.
11. (new, WP-A Stage 2) **Route checkpoint and road geometry.** MEASURED (eval139 + a 600-clip train sample,
    `FlyWheels/TanitAD_DataFlyWheel/incoming/2026-10-04-v9-labels/RESULT.md` §S2): a trivial planner that only aims at
    the checkpoint already beats refcv7 on direction (RC-B 0.972, RC-A50 0.916 vs 0.841) and heading-within-15°
    (RC-A50 0.654 vs 0.514); every variant leaks future-speed information beyond NAV under a nonlinear model (trees
    +0.052…+0.131 vs the 0.03 bar; the linear model passes at +0.006…+0.010; the trees are not yet certified), and the
    leak is ROAD CURVATURE AHEAD — a heavily smoothed route with no lane-level detail carries 80–90 % of RC-A50's
    increment. **Master Mind's provisional ruling:** road-geometry speed information is admissible inside a route input
    (a deployed navigation system supplies the route over a map); lane-level ego choice is not. ⇒ tentatively RC-A50
    with a noised training input + ≥ 0.3 dropout, the leak re-measured against the heavy route (E2′ ≤ 0.01 under a
    certified instrument, pre-registered). **The PI confirms or overrules** (if overruled: no route checkpoint; nav
    token + spatial args stay the route input). On NavSim the legal row carries no route either way.

**WP-A status (2026-10-04 ~16:15): v9 release BUILT and VALIDATED** (R8-1/R8-2/R8-3; `FlyWheels/TanitAD_DataFlyWheel/
incoming/2026-10-04-v9-labels/`): per-frame tactical labels on **95.09 %** of train windows (refcv7 23.22 %; eval 94.57 %,
FAIL as registered by 0.4 pp); TURN side 0.980 / 0.988, STOP 0.991 / 0.994 against an independent derivation; realised
announced turns get the matching per-frame nav token on 99.6 %; the route checkpoint RC-A50 with training noise passes
the registered leak test against a road-level route (conditional on the PI confirming decision 11); lane change ships
never-positive (SAM3 lane lines could not measure it against the VLM text, 14 / 101). Reader + INTEGRATION.md → WP-B.

**WP-D status (2026-10-04 ~14:30): design DONE, probes REGISTERED** (`…/2026-10-04-refcv8-perception-architecture/`
RESULT.md, PERCEPTION_DESIGN.md, PREREG_WPD_PROBES.md sha256 `c054190b…`). MEASURED from refcv7's own log + PROBE-0:
* **The box heads were schedule-limited, not plateaued**: cosine took the LR to ~0 at 50,400 (~1.08 epochs) while every
  band still rose — box3d AP@2 m 0.199 → 0.248 (+0.050 [+0.041, +0.058]) from 30k to 50.4k, AP@1 m +55 % relative.
* **The 300 "learned" query reference points NEVER MOVED** (median 0.031 m over 50,400 steps; 91 of 300 at |y| > 12 m;
  only 125 ever produce a true positive). **The score is placement-blind** (the top-scored slot on an object is the
  nearest one only 34.5 % / 23.0 %); re-ranking each object's own slots by placement + NMS (ORACLE) lifts box3d AP@1 m
  0.130 → 0.263. Localisation error is longitudinal (median |Δx| 0.96 m vs |Δy| 0.38 m at 0–20 m).
* **The planner reads the WEAKER detector** (agent AP@2 m 0.131 vs box3d 0.248).
* **The map is image-bound beyond ~40 m** (ANALYTIC: at 80–100 m one stride-8 feature row explains 903 label rows; a
  0.15 m line is 0.8 px) — **R8-6's "edge IoU 0.003 at 80–100 m" is a resolution bound, not a head defect**; the R8-6
  measure is AMENDED to a range-adaptive metric beyond 40 m (M5) and the IoU bars apply to 0–40 m. Near range (0–20 m,
  edge recall within 1 m 0.60) is a representation / target problem.
* **REFUTED: "the planner starves the perception trunk"** — box3d + map + tactical carry 99.2–99.5 % of the trunk
  gradient norm vs the trajectory loss.
* **"Do we need an initialization of the heads?" — not for the biases** (the focal prior washed out); **YES for the
  query anchors and memory positions** (per-layer anchor refinement + a zero-initialised camera-ray embedding); the most
  valuable initialisation is refcv7's own heads, which were still learning.
* **Design, ranked:** B1 keep training the heads (warm start, re-warmed LR, EMA; ESTIMATED +0.03…+0.07 AP@2 m over 30k);
  B2 modern set prediction (hybrid one-to-many groups, contrastive denoising, quality-aware presence target, per-layer
  anchor refinement); B3 one detector for the planner (box3d → AgentTokenEmbed, zero-gated); M1 stride-4 near lift +
  10 cm decoding on 0–40 m only + a placement-tolerant line target; B4 camera-ray embedding + LiDAR depth as an auxiliary
  target (two-sided published evidence). Probes P-GRAD → P-BOX → P-MAP decide which enter SPEC_REFCV8.

**WP-C status (2026-10-04 ~15:30): DONE** — all six fixes as opt-in modules with defaults bit-identical: F4b NMS
reproduced (box3d AP@2 m 0.2483 → 0.3494, agent 0.1314 → 0.2996, boxes per object 2.12 → 1.07 / 2.60 → 1.01); F1 / F4
reproduced with 0.0 difference + a TRAIN re-fit tool for any checkpoint; ego-box mask (693 → 0 footprint boxes);
track-id-switch rate masking (6,227 events; max |v_rel| 764 → 78 m/s); z/h by range with a low-trust flag; eval map
masking of the 2 clips without GT confirmed. 86 tests; 29/29 reintroduced defects caught. Trainer wiring (I1–I4) → WP-B.
