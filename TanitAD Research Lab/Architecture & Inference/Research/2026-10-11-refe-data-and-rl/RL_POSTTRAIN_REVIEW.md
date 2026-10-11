# RL post-training for NAVSIM planners, and a draft reward + minimal experiment for REFe

> 2026-10-11. Research Lab agent, one deliverable. Package: `TanitAD Research Lab/Architecture & Inference/Research/2026-10-11-refe-data-and-rl/`.
> **Design only. No training ran, no reward code exists yet, nothing was launched.** Two CPU measurements were run (pick telemetry, section 3.4; label activation, section 4.2).
> **Evidence classes** (every number carries one): **[M]** MEASURED (ours, artifact path) | **[P]** PUBLISHED (banked primary PDF, arXiv key) |
> **[I]** INHERITED (our doc, not re-verified) | **[E]** ESTIMATED | **[H]** HYPOTHESIS | **[U]** UNVERIFIED.
> Every [P] number below was read from a PDF in `TanitAD Research Lab/Library/papers/` (text extracted with `pdftotext`). Where a table's
> check-mark columns were lost in extraction, the row/column assignment was reconstructed from column order and **cross-checked against
> the paper's own prose**; any row I could not cross-check is marked [U].

## 0 · Headline (read this, then decide)

1. **RL is not the main lever for the 6.30-point gap, and the published record says so.** Controlled RL / preference steps on NAVSIM
   (same checkpoint, one change) gain **+0.35 to +2.7 PDMS** on bases of 87.6-94.1 [P, section 1]. The big headlines (+4.3 ReCogDrive,
   +8.6 AutoVLA) start from weak bases (80.5-86.5). In REFe's own architecture family (CLOVER, a DrivoR-style 64-proposal generator-scorer)
   the closed-loop refinement stage is **+0.35** (94.13 to 94.48) [P, 2605.15120 Table 5]. Our base is **88.25** [M, `eval/RESULT_SFT4.md`], inside
   the 87.6-89.1 band where full-pipeline RL gave +1.4 to +2.7 and selector-side methods gave +0.2 to +1.2 [P, section 1].
   **Prior for a scorer-only RL arm: +0.2 to +1.2 PDMS [E], i.e. 3-19 % of the gap.** P2 / P6 stay the main path.
2. **EP is the hard part, and no published controlled RL step reproduces a +11.5 EP move.** The largest controlled EP moves are
   ReflectDrive-2 **+7.1** (but TTC **-4.7**, NC **-1.5**) and ReCogDrive **+6.4** [P]; the selector-matched DDv2 step moves EP only **+0.7**,
   and 4.6 of its 5.3 headline EP points come from the selector, not from RL [P, 2512.07745 Table 9]. DriveDPO (selector-side preference) moves
   EP **+0.3** [P]. Safety-only reward collapses EP to **63.3** [P, 2510.24108 Table 8].
3. **What is REFe-specific and favourable:** the 64 hypotheses are already all scored by NAVSIM's own scorer, so the RL reward for **every** candidate is
   exact and known (no sampling). That is the "exhaustive" regime where ZTRS found policy gradient stable and GRPO(k=4, 8) / REINFORCE unstable
   [P, 2510.24108 Fig. 3, Table 6]. The selection loss is large on a frozen fan: random 72.19 / pick 83.24 / best-of-64 97.07 on the same 12,146 tokens
   at the pre-SFT-4 scorer [M, `raw/2026-10-04-gap-review/gap_selection.json`]; on clean fans (44 % of tokens) **7.57 of the 8.74** weighted-part loss points are EP [M, `gap_fan_quality.json`].
4. **Recommendation:** build and pre-register the **scorer-only, frozen-fan exhaustive policy-gradient arm** (sections 5-6). It fits **about 1.0-1.1 A40-days** [E],
   leaves the generator untouched, and has a free replicate and control structure. Do **not** start generator-side GRPO on REFe: it needs a stochastic
   policy REFe does not have, our own DDv2 port collapsed the fan and damaged selection [M], and the only published generator+scorer evidence says
   joint updates diverge (PDMS < 10) unless alternated [P, 2605.15120 Table 7].
5. **The PI's three requested ingredients, measured:** drivable-area distance as a graded training penalty is cheap, dense and has published support
   (ReflectDrive-2 DAC +3.1 from a map distance-field loss [P]). Lane-consistency has an official counterpart (EPDMS lane keeping, 0.5 m / 2 s [M, local devkit]) but **has no direct term in
   v1 PDMS** (it can only act through DAC / EP), so do not expect it to show in the 94.55 bar. **Lane-change is the weak one: the label fires on 2.7 % of frames [1.4, 5.0] in a 336-frame pilot, and 0 of 14 slow-lead frames
   had a human lane change [M, `raw/label_activation.json`].** At that rate it is very unlikely to move a 64-way ranking [E], and PDMS v1 pays nothing beyond EP = 1. Treat it as a TACTICAL-family aim, curated
   and measured separately, not a PDMS lever.

**Decisions needed (PI):** (D1) sequencing against P6, which needs the same single A40: run this ~1.1-day experiment before P6, after P2's verdict, or never.
(D2) Confirm map-derived reward terms (drivable-area distance, lane geometry) are inside the 2026-10-11 ruling: labels and rewards only, never inference inputs.
(D3) Whether the lane-change shaping arm (R2) is worth a screening slot given a 2.7 % activation.

**Integration (Master Mind):** the reward module, the on-policy proposal dumper and the EPO loss do **not exist**; they are specified here, not built. Landing this package stages two measurement
scripts + raw JSON + 4 newly banked PDFs. Launch needs a launch-gate PASS token with a G-DVB entry for every new flag (CLAUDE.md, 2026-09-26).

## 1 · Which published RL / preference methods have a CONTROLLED gain on NAVSIM?

"Controlled" = the same checkpoint with one change (RL step on / off) read on the same benchmark. All rows are navtest unless stated. **No row below reports a training
replicate except CLOVER (three seeds, "< 0.02 PDMS") and ZTRS Table 6 (init-seed variance of EPO vs REINFORCE)**; a single-run delta below about +1 is not a lever by our own
replicate rule (H-ESTIM-SEED-1: a zero-lever replicate cleared "separated" on 9.5 % of cells [M, CLAUDE.md]).

| method (library key) | controlled step | Delta PDMS | sub-scores moved (EP in bold) | reward | map: reward / input | compute |
|---|---|---|---|---|---|---|
| **DiffusionDriveV2** (2512.07745) | DiffusionDrive + same selector 89.1 -> DDv2 91.2; Table 9 | **+2.1** (headline +3.1 vs 88.1 w/o selector) [P] | NC +1.1, DAC +0.9, TTC +2.6, C -0.1, **EP +0.7** (86.8 -> 87.5). Headline EP +5.3, of which selector alone +4.6 (82.2 -> 86.8, with TTC -2.5, NC -1.0) [P] | PDMS-based group-relative; collision -1; truncated advantage (no negative) [P]. DAC also -1 and a GT-anchored bar in the release [I, register D-DDV2-SHARPENINGS] | reward only; input = camera + LiDAR BEV (ResNet-34) | RL 10 ep navtrain, batch 512, 8 x L20; selector 20 ep [P]. Single run |
| **ReCogDrive** (2506.08052) | IL planner (Table 3 row 3) 86.5 -> +DiffGRPO 90.8 | **+4.3** [P] | NC -0.2, DAC +2.6, TTC +0.7, C 0, **EP +6.4** (80.9 -> 87.3) [P] | PDMS from the NAVSIM simulator, single-step decision, G samples [P] | reward only; input camera + text | RL 10 ep, batch 128 [P]; whole pipeline 32 x H20; RL-stage GPUs [U]. Text says base 83.3, table row 1 says 82.4 [P, internal inconsistency] |
| **ReflectDrive-2** (2605.04647) | SFT chain 87.7 -> +full-rollout RL 89.1 (Table 6); with inference-time AutoEdit the same step reads 88.0 -> 91.0 (AutoEdit adds <= 0.3 before RL, +1.9 after) | **+1.4** (+3.0 with AutoEdit) [P] | NC **-1.5**, DAC +1.2, TTC **-4.7**, C -0.2, **EP +7.1** (82.2 -> 89.3) [P] | terminal closed-loop planning score on the final post-edit plan; group 6 [P] | reward only; **plus a training-time drivable-area distance-field loss**: DAC 93.9 -> 97.0, PDMS 84.8 -> 87.2 (+2.4) [P]. Source of the indicator map [U]. Inputs: cameras + nav + ego | compute [U] |
| **DriveDPO** (2509.17940) | w/o DPO 88.8 -> safety-DPO 90.0 (Table 4) | **+1.2** [P] | NC +0.6, DAC +0.8, TTC +1.2, C -0.1, **EP +0.3** [P] | PDMS-ranked preference pairs over a fixed candidate set (rejected-selection ablation: vanilla +0.5, distance +0.9, imitation-based +1.2) [P] | reward only | [U] (not read) |
| **Curious-VLA** (2603.06049) | IL final 87.6 (Table 5) -> RL 90.3 (Table 6) | **+2.7** [P] | NC +0.1, DAC +1.8, TTC +0.3, C -0.2, **EP +2.0** (86.5 -> 88.5) [P] | PDMS + "spanning" focal reward (+0.2); sampling matters: human-difficulty sampling (and random / full-set) **collapsed, 35.2 PDMS**; reward-diversity filters 88.8-90.1 [P] | reward only | [U] (config table not read) |
| **HAD** (2604.03581), v2 EPDMS | local-refinement stage w/o RL 87.3 -> +RL 88.6; coupled PDMS reward 87.8 -> metric-decoupled 88.6 | **+1.3**; decoupling **+0.8** [P] | decoupling: DAC +0.6, TTC +0.4, **EP -0.7**, EC +1.7 [P]. RL added at the coarse stage instead: 87.2 (worse) [P] | offline cached reward retrieval: 0.2449 s -> 0.0042 s per trajectory, 64.4 h -> 13.6 h [P] | reward only | 8 x A100, 85 epochs total [P] |
| **ZTRS** (2510.24108), EPDMS | no-human RL (EPO) vs IL, navtest | **-0.9** (85.3 vs 86.2); navhard +3.8 / +2.8 over GTRS-Dense [P] | **EP 84.1 vs 88.5 (-4.4)** [P] | rule-based EPDMS, all 16,384 vocabulary entries scored every step. Ablation: safety-only EP **63.3**, + progress 83.8, + comfort 84.1 (EPDMS 72.0 / 82.4 / 85.3) [P] | reward only | 24 x A100, 15 ep, batch 528 [P] |
| **CLOVER** (2605.15120), same family as REFe | Stage 1 94.13 -> Stage 2 closed-loop 94.48 (Table 5, 12,146 scenes); vs DrivoR baseline 93.69 | **+0.35** (full pipeline +0.8) [P] | vs DrivoR (Table 1): NC +0.1, DAC +0.1, TTC +0.2, **EP +1.7** (90.0 -> 91.7) [P]. Joint non-iterative update: **< 10 PDMS**; scalar top-k targets 93.9 vs vector-Pareto 94.5 [P] | scorer fitted to true evaluator sub-scores; generator refined to teacher-selected top-k / vector-Pareto targets with stability regularisation, alternating [P] | reward only (evaluator labels); input 4 cameras, DINOv2-S + LoRA, no LiDAR | Stage 2: **60-80 h on 4 x A100** (~240-320 A100-h), includes live NAVSIM scoring [P]. Seeds "< 0.02" [P, claimed] |
| **DriveZero** (2609.06055) | privileged PPO teacher -> camera student; ViT-S navtest: human traj 93.92 -> DriveRL traj 93.61 -> + goal aug 94.41 (Table 7) | RL-teacher distillation **-0.31** alone, **+0.49** with goal aug [P] | EP 91.99 -> 91.64 -> 91.79 [P] | teacher reward (A.4, section 3) | **map in teacher input AND reward**; student camera-only | teacher: 96 GPUs x ~21 h (~2,016 GPU-h), 922,703 nuPlan scenes [P] |
| **AutoVLA** (2506.13757) | SFT (CoT) 80.54 -> RFT 89.11 | **+8.57** [P] | sub-scores [U]; "best-performing checkpoint selected" (set [U]); best-of-6 oracle 92.12 [P] | PDMS - CoT length penalty, GRPO, LoRA, 6,000 steps | reward only | SFT 8 x L40S, 5 ep [P]; RFT GPUs [U] |
| **DIVER** (2507.04049) | final 88.3, EP 82.6 [P] | controlled RL step [U] (check-mark columns lost) | - | PPO / GRPO ablation rows exist | **map elements are a conditional INPUT** (PADG): inadmissible under the input ruling | 8 x RTX 4090 [P] |
| **EvaDrive** (2508.09158) | roadmap S0 83.1 -> S5 94.9 | +11.8 **confounded** (two-stage generator +5.0, multi-turn inference refinement) [P] | "Aggressive" preference weights: EP **93.9** vs "Conservative" 90.5, PDMS 94.9 vs 93.5 [P]. Unreplicated outlier | multi-objective reward + adversarial critic | reward only | 4 x H20 [P] |
| **TrajHF** (2503.10434) | NAVSIM: single sample 86.4, EM-15 87.6, **PDMS-selector (oracle) 94.3** [P] | no controlled NAVSIM RL gain; RLHF gains are on an internal set. EvaDrive's comparison row "TrajHF 94.0" is an **oracle** figure [P] | - | - | - | - |
| **Gigapixel** (2606.19641) | self-play-distilled vs BC, DrivoR, navhard | EPDMS +1.8 (50.1 vs 48.3) [P] | navhard, not navtest | simulator RL teacher + DAgger | - | - |
| **AD-R1** (2511.20325) | DiffusionDrive 88.1 -> naive optimistic world-model RL **85.3** -> + counterfactual synthesis 89.1 | final +1.7 abs on two baselines [P] | - | learned world-model critic | - | - |

**What the table says**

- **Gain shrinks with base strength.** Bases 80.5-86.5: +4.3 to +8.6. Bases 87.6-89.1: +1.2 to +2.7. Bases 93.7-94.1: +0.35 to +0.8 [P, rows above]. REFe is at 88.25.
- **Where EP moves, safety margin pays for it.** EP +7.1 costs TTC -4.7 and NC -1.5 (ReflectDrive-2); the PDMS-trained selector in DDv2 buys EP +4.6 with TTC -2.5 and NC -1.0 [P]. A reward that sums the PDMS
  terms accepts that trade by construction. It is the same trade REFe's scorer already makes.
- **Selector-side preference (DriveDPO, DDv2's rank loss +0.2) buys safety, not EP** (+0.3 / flat) [P].
- **Oracle numbers circulate as results** (TrajHF 94.3 / 94.0, AutoVLA best-of-6 92.12). None is used as a gain here (retraction class: best-of-N read as a skill gap, `H-RL-OBJECTIVE-1`).
- **Map information:** in every NAVSIM-v1 RL method above, the map reaches the policy only through the NAVSIM scorer (reward). Exceptions: DIVER (input, inadmissible), ReflectDrive-2 (a training loss),
  DriveZero teacher (privileged input; the student is camera-only). **Nothing here needs a map at inference.**
- **Compute:** published RL stages cost 8-32 GPUs for 1-3 days, or CLOVER's ~240-320 A100-h; our ceiling is 48 A40-h. Only a scorer-only arm on a frozen fan fits.

### 1.1 The teacher already carries the PI's reward terms (so RL on the student is not where they first enter)

DriveZero's PPO teacher reward (A.4) is `r_t = h_t + (1 - d_t) g_t + (1/110) prod_k q_{t,k}`, K = {cross-lane, centerline, curb, comfort, TTC, overspeed} [P, 2609.06055].
Hard: new collisions -1.0; off-road terminates, speed-dependent base -2.0; goal +0.5 within 1.5 m. Release runtime values (`driverl_teacher.yaml`, domain randomisation off): cross-lane
weight 0.4, wrong-way 0.2, centerline-deviation floor 0.75 (capped at 4 m, normalised by the free lateral room), curb clearance 0.7 m (weight 0.5, speed-scaled), comfort 0.5, TTC 0.5 [M, read from `C:/Users/Admin/dz/DriveZero/DriveRL/release/configs/driverl_teacher.yaml`].
In the release, crossing a **non-solid** boundary is a reserved free class (lane change), a **solid** crossing is a cross-lane event and can terminate the rollout [M, `off_road.py:100-160`, `engine.py:123-136`].
Our REFe student regresses a teacher trajectory (WTA L1) and never sees these terms directly. Our reproduction of the teacher scores PDMS 89.81 raw and **94.10 tracker-compensated** (EP 85.18 -> 88.76), against the paper's 95.8 / EP 91.5
[M, `eval/RESULT_P4_TEACHER_NAVTEST.md`; paper value P]. **The deployed (pre-SFT-4) student's EP is 8.5 points below its own teacher's (76.70 vs 85.18) on the P4 token set [M, same file]**: that is an imitation-plus-selection gap, not a teacher gap. (SFT-4's EP of 81.20 is on the full navtest, a different set; the same-set student-teacher gap after SFT-4 is not measured [U].)

## 2 · What our own RL history forces on the design [M unless marked]

| finding (artifact) | consequence for REFe |
|---|---|
| DDv2 port on refcv5-v2: seed 1 diverged (grad norm 15,712 vs 147) under the no-clipping recipe; the imitation term **collapsed the 117-anchor fan by 93 %** (endpoint spread 37.52 -> 2.45 m); RL damaged **selection** (exact picks -17.0 pp), not the generator (`H-DDV2RL-1/2`, register) | freeze the generator; clip gradients; anchor the new loss to the SFT scorer (KL) and keep the BCE sub-score term |
| GRPO advantage identically zero in 92 % of windows for the collision channel (`D-RL-COLL-SPARSE-1`, PhysicalAI rig) | measure each term's activation **before** weighting it (section 4.2). REFe's PDMS reward is dense on the fan (mean-of-64 72.19 vs best 97.07 [M]); the lane-change label is not |
| the same-seed rerun of `ddv2_rl_refcv5.py` diverges 3e-7 -> O(1) in 80 steps (`D-DDV2RL-L1-T1-REGRESSION`) | a single run is one draw: **training replicate and same-code control are mandatory**; no verdict from one arm |
| DAC term was algebraically absent on that rig (no maps: DAC == 1) | REFe's scorer has the nuPlan maps via `metric_cache_parts` (drivable-area map, centerline, route lanes) [M, `refe/navsim_pdm_targets.py:87`]; the term exists, so the REFe reward can use it |
| `ego_progress` saturates: 10 / 12 / 14 / 20 m/s all score 38.5390; the two-sided EP repair failed its own test (`D9`) | NAVSIM's own EP is also a capped ratio (section 4.4): EP gives no reward for over-speed, so the "go faster" hack is closed by construction and the open hack is the opposite one |
| `H-RL-OBJECTIVE-1` retracted: a best-of-N statistic read as a skill gap | every oracle number below is quoted with random and actual on the same tokens |
| earlier RL arm failed its committed bar (separated `ade_m` +0.0362) | the bar below is reported as written; a miss is FAILED, then the next lever runs |
| four "built, tested, unreachable from the caller" instances (CLAUDE.md, 2026-09-10/11) | the experiment's gate includes a non-zero gradient on the new loss, a control that reads exactly 0.0, and a mutation arm |

## 3 · Reward design for REFe (draft, for pre-registration)

### 3.0 Principles

- **P1 identity at the boundary.** Every graded term is computed from the *same objects* the NAVSIM scorer uses (`PDMDrivableMap`, the PDM-Closed centerline, the 41-step LQR-executed states), so a graded term is
  zero exactly when the official metric is satisfied. Tested, not asserted (section 6.5, G-REWARD-ID).
- **P2 maps are labels only.** Every geometry source is a training-time cache; nothing enters the model input (PI ruling 2026-10-11). The route-needs-a-lane-change label uses the registered route roadblocks, which are already an admissible input.
- **P3 graded terms break zero-advantage ties.** Where all 64 proposals share NC = DAC = 0 or = 1, PDMS is flat; a graded distance is not.
- **P4 small weights.** Shaping terms may reorder near-ties, never override PDMS.
- **P5 evaluate on the executed plan.** The LQR-tracked 41-step states, not the 8-pose plan. Only 29 % of the deployed student's DAC failures are tracker-made (188 of 643 repairable) [M, P4 F2], so most of the signal is real path error.

### 3.1 Composition

```
R_i  =  PDMS_i  -  lam_d * D_i  -  lam_k * K_i  +  lam_l * L_i          i = 1..64 proposals of one scene
PDMS_i  = NC_i * DAC_i * (5 EP_i + 5 TTC_i + 2 C_i) / 12   (NAVSIM v1.1, the `pdms` array of refe/navsim_pdm_targets.score_plans)   [M, REVIEW_7 l.146]
lam_d = 0.15,  lam_k = 0.05,  lam_l = 0.05          (initial values, all [H]; a weight-sensitivity study is outside the 1.1-day budget, section 7)
```

### 3.2 Term-by-term

| term | exact computation | data source (field verified) | expected effect on the four families | hacking risk and guard |
|---|---|---|---|---|
| **(a) base** `PDMS_i` | `score_plans(parts, executed_plans8(props, repair=True))["pdms"]`; the six sub-scores NC, DAC, EP, TTC, C, DDC are the BCE targets as today | `metric_cache_parts(scenario)` returns `(pdm_traj, ego, obs, centerline, route lanes, drivable_area_map)` [M, `refe/navsim_pdm_targets.py:87`]; GT agents interpolated 0.5 s -> 0.1 s over 5 s [M, same file docstring]. Cost 1.78 CPU-s per 64-proposal set [I, `eval/PREREG_SFT4.md` Amend. 3] | LONG: EP, TTC; LAT: via DAC; TACTICAL: indirect; STRATEGIC: PAUSED (PI item 25) | PDMS is the evaluation metric: optimising it is the goal and the risk (benchmark-specific tracker exploitation). Guard: report tracker-vs-plan deviation (teacher median 0.45 m, student 0.53 m [M, P4 F2]) and kinematic feasibility; no tracker-specific reward |
| **(b) drivable-area excess** `D_i` | `g_i(t) = max over the 4 bbox corners of dist(corner, union of drivable polygons)`, 0 when inside; `D_i = mean_t min(g_i(t)/0.5 m, 1)` on the 41 executed states. Optional inner-margin term: `(0.3 m - c_i(t))+/0.3 m` where `c_i(t)` is the corner clearance to the boundary, active only when 0 <= c < 0.3 m | `PDMDrivableMap` built from roadblock, roadblock-connector, intersection and carpark-area polygons [M, `navsim/.../pdm_occupancy_map.py:130-180`]; map radius 100 m [M, `metric_cache_processor.py:54`, local devkit; v1.1 pin `3e8291b` [U]]. nuPlan maps for the 4 cities are on the dev box (2.3 GB) [M]. Published precedent: distance-field loss, DAC +3.1 [P, 2605.04647 Eq. 13-14] | LAT: DAC up; LONG: neutral | Graded penalty could bias against speed if the polygon coverage truncates. Guard: coverage radius 100 m covers 4 s at 25 m/s; test `g > 0 iff DAC = 0` on 10,000 proposals (G-REWARD-ID). Margin term is speed-independent so it cannot favour slowness |
| **(c) lane consistency** `K_i` | outside intersections (`drivable_area_map.is_in_layer(INTERSECTION)`, the scorer's own switch): `k1 = mean_t clip((o(t) - 0.25)/0.75, 0, 1)`, `o` = distance of the ego centre to the PDM-Closed centerline; `k2 = mean_t clip((|dpsi(t)| - 10 deg)/20 deg, 0, 1)` only where `o(t) < 1.5 m` (so a deliberate lane change is not a heading violation); `K = 0.6 k1 + 0.4 k2`. Straddle `k3 = fraction of time the ego bbox crosses a lane boundary` with no lane-change label, added at weight 0.2 | `centerline` is `PDMPath` of PDM-Closed [M, `pdm_scorer.py:615-650`]; EPDMS lane keeping is `dist > 0.5 m for >= 2 s` outside intersections [M, local devkit `pdm_scorer.py:59-60`]; `Lane.polygon`, `left_boundary`, `right_boundary`, `baseline_path` exist [M, nuplan `abstract_map_objects.py:32,106-124`]. Teacher precedent: cross-lane 0.4, centerline 0.75, wrong-way 0.2 [M, `driverl_teacher.yaml`] | LAT: cross-track, heading, curvature, yaw-rate MAE (the families6 rows); **not scored by v1 PDMS** | A centreline pull biases against necessary offsets (parked cars, narrow roads). Guard: cap at 0.05, deadband 0.25 m, suppress in intersections and when a lane-change label fires. **Expected PDMS effect: about 0.** It is a lateral-family term, and v1 PDMS has no LK |
| **(d) lane change** `L_i` | see 3.3 | `scenario.get_route_roadblock_ids()`, `Lane.adjacent_edges`, `outgoing_edges`, `get_roadblock_id` [M, nuplan source + `refe/navtrain_scenarios.py`]; tracked vehicles from the scenario | TACTICAL: lateral decision accuracy / kappa; LAT | gratuitous lane changes; see 3.3 |
| **(e) progress guard** | see 3.4 | NAVSIM `_calculate_progress`: projection of start and end onto the centerline [M, `pdm_scorer.py:476-490`] | LONG: EP, speed bias, along-track bias | the slow-pick hack; see 3.4 |

### 3.3 Lane-change term: label and guards

Labels (registered here, before the rate was read; script `code/measure_label_activation.py`):

- **(i) human lane change**: the human's next 4 s (8 poses at 0.5 s) map-match to two different LANE objects of the same roadblock, and the final pose is > 2.0 m from the start lane's baseline path. Lane connectors (intersections) never count.
- **(ii) slow lead**: nearest tracked vehicle at ego-frame x in [2, 40] m, |y| < 1.6 m, with speed < 0.7 x ego speed, ego speed > 4 m/s. **Adjacent free**: an adjacent lane exists and no vehicle centre lies inside its polygon with x in [-12, +35] m.
- **(iii) route needs it**: the start lane has no outgoing edge into the next on-route roadblock while an adjacent lane has (not measured; definition only [U]).

Shaping: `L_i = +1[LC_i executed AND NC=DAC=TTC=1 AND (label (i) OR (iii))] * progress_gain_i  -  1[LC_i executed AND no label fires] * 1  -  dither_i`, clipped to +/-1, weight 0.05.
`progress_gain_i = clip((EP_i - EP_stay)/0.2, 0, 1)` where `EP_stay` is the best safe non-lane-changing proposal's EP. Label (ii) is a **permission** (not penalised), **never a reward**: in the pilot 0 of 14 slow-lead frames had a human lane change.

How gratuitous changes are avoided: (1) positive credit only when a *human-validated* label fires; (2) full safety gate; (3) the reward is bounded by `progress_gain`, so a lane change that does not beat staying earns nothing; (4) an unlabelled change is penalised; (5) dither (>= 2 sign changes of lateral velocity in 4 s) is penalised; (6) lane-change precision / recall against the labels is a reported TACTICAL metric, and a rate above the label base rate by > 2x is a kill condition.

### 3.4 Progress, and the "slow down to stay safe" hack

- **How EP is defined [M, REVIEW_7 l.149-150; `refe/navsim_pdm_targets.py:score_plans`]:** progress along the PDM-Closed centreline, multiplied by NC and DAC, divided by `max(agent, PDM-Closed)`, forced to 1 when that maximum is <= 5 m. So EP is capped at 1, is zero after a collision or off-road, and gives **no** reward for exceeding PDM-Closed: over-speeding is not rewarded.
- **The open hack is the opposite one, and it is the baseline behaviour.** At the pre-SFT-4 scorer the pick sat at median length rank 49 of 64, 75 % of picks were shorter than the fan median, the shortest quintile took 47.5 % of picks [M, REVIEW_7 §2.3], while the fan's safe proposals averaged human-level EP (87.5 vs 87.0) [M]. Giving the selector NAVSIM's true EP alone made it worse (-1.18 [-2.07, -0.39]) because faster plans fail NC / TTC that the scorer cannot judge [M, REVIEW_7 §2.3].
- **Re-measured on SFT-4 u400 (new, [M], `raw/pick_length_telemetry.json`, 10,070 tokens whose executed pick is a member of the stored fan; the other 2,076 are re-planned by the route fix and excluded, never nearest-matched).** Controls first: the stored final-model pick reproduces REVIEW_7 (pick path / fan mean 0.8868, median length rank 49.0, 75.15 % shorter than the fan median on all 12,146 tokens), and the deployed seam reproduces the stored pick on 99.99 % of common tokens.

  | pick statistic (10,070 common tokens) | deployed (84.46) | SFT-4 u400 (88.25) |
  |---|---|---|
  | pick path / fan mean path | 0.885 | **0.914** |
  | median length rank (0 = longest, of 64) | 49 | 52 |
  | share of picks shorter than the fan median | 0.754 | **0.797** |
  | shortest-bin / longest-bin share of picks (13-rank bins) | 0.458 / 0.070 | **0.506 / 0.093** |
  | mean pick path (m) vs mean fan path 21.67 m | 19.16 | 19.55 |

  SFT-4 picks are longer on average but **more concentrated at both extremes**, and the slow-pick share did not fall. The slow-pick pathology persists after the best scorer we have. This is the baseline the guard is written against.
- **Published evidence that the progress term must be explicit:** safety-only reward gives EP 63.3 vs 84.1 with progress + comfort [P, 2510.24108 Table 8].

Guards, in order of strength: (g1) PDMS (EP weight 5/12) stays the base reward; (g2) a **progress advantage among safe proposals only**, `A_prog_i = clip((min(p_i, p_ref) - min(p_best_safe, p_ref))/p_ref, -1, 0)` (shortfall relative to the best *safe* proposal, with progress capped at PDM-Closed's `p_ref` exactly as EP caps it, so
it never rewards exceeding the reference), added to the group advantage with weight 0.5 [H]; this is the leak-free form, since REVIEW_7 shows EP only helps when applied to safe proposals (cross-fitted tilt +0.18 [+0.04, +0.30] [M]); (g3) **telemetry gates**, evaluated every 250 updates on held-out scenes: pick / fan-mean path must not fall below the baseline 0.914 - 0.02, the shortest-bin share must not exceed 0.55; (g4) the longitudinal bias rows of families6 (speed bias, along-track bias) must not move more negative than the same-step control by > 0.15 m/s / 0.25 m; (g5) the scorer's own margin: NC and TTC sub-scores may not fall by > 0.5 each versus control (ReflectDrive-2 paid TTC -4.7 for EP +7.1 [P]).

## 4 · Data checks behind the reward terms

### 4.1 Fields verified (read in source, not recalled)

| need | field / call | where verified |
|---|---|---|
| drivable polygons for the signed distance | `PDMDrivableMap.from_simulation(map_api, ego_state, radius)` over ROADBLOCK, ROADBLOCK_CONNECTOR, INTERSECTION, CARPARK_AREA | `C:/Users/Admin/navsim-crun/devkit/navsim/planning/simulation/planner/pdm_planner/observation/pdm_occupancy_map.py:130-180` [M] |
| centreline, route lanes, drivable map per scene | `metric_cache_parts(scenario)` | `refe/navsim_pdm_targets.py:87` [M] |
| lane geometry and adjacency | `Lane.polygon / baseline_path / left_boundary / right_boundary / adjacent_edges / outgoing_edges / get_roadblock_id / speed_limit_mps` | nuplan-devkit `abstract_map_objects.py:32,63,97,106-141,210` [M] |
| lane boundary types | gpkg layer `boundaries` has `boundary_type_fid` with values {0, 2, 3} (Boston); **semantic decoding of the ids [U]** | `E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0/us-ma-boston/9.12.1817/map.gpkg` [M] |
| route requirement | `scenario.get_route_roadblock_ids()` (keyed on the initial token, so one scenario per token) | `refe/navtrain_scenarios.py` docstring [M] |
| lane-keeping reference | `lane_keeping_deviation_limit 0.5 m`, `lane_keeping_horizon_window 2.0 s`, skipped inside intersections | local devkit `pdm_scorer.py:59-60, 615-650` [M] |
| EP and DAC definitions | `_calculate_progress` (projection on the PDM centreline), `_calculate_drivable_area_compliance` (any non-drivable ego area) | local devkit `pdm_scorer.py:422-429, 476-490` [M] |

Caveat on the devkit: the local `navsim-crun/devkit` is the v2 code; our harness is v1.1 (`navsim @ 3e8291b`). Parity of the quoted functions with the v1.1 pin is assumed, not diffed [U].

### 4.2 Label activation pilot (CPU only)

`code/measure_label_activation.py`, 14 local `trainval` logs, 336 frames, 0 errors, seed 20261011; raw `raw/label_activation.json`, log `raw/label_activation.log`. Wilson 95 % intervals.

| label | count | rate [95 % CI] |
|---|---|---|
| human lane change within 4 s | 9 / 336 | **2.7 % [1.4, 5.0]** |
| slow lead (as defined) | 14 / 336 | 4.2 % [2.5, 6.9] |
| human lane change given slow lead | **0 / 14** | - |
| slow lead AND adjacent lane free | 3 / 336 | **0.9 % [0.3, 2.6]** |
| adjacent lane exists / free | 138 / 90 | 41.1 % / 26.8 % |
| human path crosses a lane connector within 4 s | 201 / 336 | 59.8 % [54.5, 64.9] |
| lateral deviation > 0.5 m from the start lane path within 4 s | 115 / 336 | 34.2 % [29.4, 39.5] |
| pure lane-following windows (start on a lane, no connector) | 135 / 336 | 40.2 %; lane change 7/135 = 5.2 %, slow lead 7/135 = 5.2 % |

**Read with these limits.** (1) The 336 frames come from local `trainval` logs, **not** the navtrain token list, and 28.9 % have ego speed < 1 m/s, so the sample is not NAVSIM-navtrain-representative [U]. (2) n = 9 lane changes: the interval is wide. (3) The pilot defines the label; it does not validate it against human judgement.
A navtrain-token run of about 3,000 frames is a ~1.5 h CPU follow-up (the script takes a token list with a ten-line change).

**Implications [E].** Even at the upper CI bound (5.0 %), a lane-change term is active in at most one scene in twenty. At the pilot's 2.7 % on 24,000 training scenes that is about 650 scenes, of which the slow-lead trigger covers a handful. The term cannot move PDMS and will be statistically invisible in a navtest read.
Treat it as a curated TACTICAL objective: oversample the labelled scenes, measure lateral-decision kappa and lane-change precision / recall, and expect no PDMS credit (EP is capped at 1, so a change that beats PDM-Closed's progress earns nothing more in v1 PDMS).

## 5 · Mechanism: what RL should update, and how a GRPO group maps onto 64 hypotheses

### 5.1 Verdict: scorer first, generator frozen, alternation later

| option | evidence | verdict |
|---|---|---|
| **scorer only, frozen fan** | selection-bound: random 72.19 / pick 83.24 / best-of-64 97.07; oracle NC+DAC+TTC 95.16 (pre-SFT-4 scorer, 12,146 tokens) [M, REVIEW_7 §2.4]. A good selector sits far from the oracle too (CLOVER: selected 93.69 / mean 79.72 / oracle 99.33 [P]), so the oracle gap is a ceiling, not a target. All 64 rewards are exact, so exhaustive policy gradient has zero sampling variance; GRPO(k=4, 8) and REINFORCE were unstable in the scorer setting [P, 2510.24108] | **do this** |
| generator only | needs a stochastic policy with a log-probability; REFe's 64 queries are deterministic. Our DDv2 port collapsed the fan and hurt selection [M] | not now |
| both, jointly | CLOVER joint non-iterative update: **< 10 PDMS** [P, Table 7]; self-distillation without scorer fitting 93.8-94.0 vs full alternating 94.5 [P] | forbidden |
| both, alternating | CLOVER Stage 2: scorer fit to true sub-scores, then generator toward teacher-selected vector-Pareto targets with a stability term, 60-80 h on 4 x A100 [P] | a **second experiment**, only if the scorer arm passes, and at P6 scale |

### 5.2 The scorer objective (EPO-style, exact)

REFe's scorer emits six logits per proposal; selection is `argmax_i A_i`, `A_i = NC * DAC * DDC * (5 EP + 5 TTC + 4 C)/14` on the sigmoids [M, `refe/train.py:611-614`]. Define the selection policy
`pi_i = softmax_i( log(A_i + eps) / T )` with T = 0.1 and the exact expected reward `J = sum_i pi_i R_i`:

```
L  =  BCE(six sub-scores, true labels)                       # unchanged; keeps the heads calibrated and diagnosable
   +  lam_pg(t) * [ -J  +  beta * KL(pi_SFT || pi) ]          # exact over all 64 candidates; no sampling
   ;  lam_pg ramps 0 -> 1 over 200 updates; beta = 0.1; grad-clip 1.0   (all [H])
```

- **Group mapping.** The group is the 64 hypotheses of one scene (G = 64, the whole action set), the exhaustive limit of GRPO. We use the raw-reward expected gradient, not std-normalised advantages: a scene with no reward spread contributes exactly zero, which is correct, and std-normalisation would up-weight such noise scenes.
- **Hypothesis-level intents.** DDv2's intra-anchor grouping exists to stop modes being ranked against each other [P, 2512.07745 section 4.4]. REFe's queries are not anchored, so no per-anchor grouping is needed for the scorer arm. For the *generator* stage, group by endpoint cluster (within 2 m), as CLOVER measures cluster counts [P, Table 26].
- **Trainable set = SFT-4's:** `score_q_mlp`, `score_dec`, `score_head` only; the 64 proposals stay bit-identical [M, `eval/PREREG_SFT4.md` l.32]. Context stays detached, so the new loss cannot touch the trunk.
- **Why a decision-focused loss can reach EP where the decomposed one cannot [H].** Replacing the predicted EP head by true EP made things worse (-1.18) because the safety heads cannot rank fast plans; the *joint* argmax outcome is what `J` optimises, so the safety-versus-progress trade is learned
  end to end instead of head by head. Predicted TTC mean 0.56 vs true 0.84 and predicted DAC 0.71 vs 0.87 [M, REVIEW_7 §2.3] are the calibration defects a ranking loss may partly absorb. **If this is wrong, EP will not move and the arm FAILS its secondary check.**
- **Scale of the plausible win [E].** CLOVER's fixed-proposal scorer diagnostic reaches 92.4 test / 95.0 train on a fan with oracle 99.4 [P, Table 20]. Our fan's oracle is 97.07, so a strong scorer on it might reach roughly 91-93; the room above 88.25 is about 3-5 points, of which a scorer-only RL arm may take a fraction. This is arithmetic from one published diagnostic, not a forecast.

### 5.3 Cost realism

All [E] unless marked. SFT-4 ran scorer-only at **14.4 s per update of 64 sets** (8 x 8) [I, `STOP_CONSEQUENCES_2026-10-11.md`]. The labelling cost is 1.78 CPU-s per set [I]; whether it includes `metric_cache_parts` is [U].

## 6 · Pre-registerable minimal experiment: "RL-S1" (draft, not in force)

> Status: DRAFT. Becomes binding only when landed in git before the first training step, with the PI's D1-D3 answered, a launch-gate PASS token, and the code in 6.5 built. Amendments are dated and written before any read.

### 6.1 Design

Base: **SFT-4 arm B u400** (`model_sft4_B_u400.pt`, md5 `8e382eeb...`, full-navtest 88.25 [M]) on the frozen `model_final.pt` fan. Generator, trunk and LoRA frozen; only the three scorer modules train. Everything below uses the SFT-4 optimiser recipe (AdamW, lr 1e-4, 50 warm-up, cosine, wd 0.01, batch 8 x accum 8, bf16).

| arm | what changes | seeds | purpose |
|---|---|---|---|
| **A0** anchor | SFT-4 u400, not trained further | - | re-read on the same harness (drift check, expected 88.25) |
| **C0 / C1** | **SFT-continued control**: BCE only, identical sets, order, steps (1,000 updates) | 0, 1 | what "keep training" gives; C0 - C1 is the **training floor** of the control |
| **R0 / R1** | **RL-core**: BCE + EPO term with reward `R = PDMS - 0.15 D` + progress guard (g2) | 0, 1 | the lever; R0 - R1 is the **training floor** of the lever |
| **R2** | R0 + lane-consistency `K` + lane-change shaping `L` (screening, single seed) | 0 | the PI's lane ingredients; R2 - R0 is read against the R0 / R1 floor (n = 1 pair: necessary, not sufficient) |
| **I-rep** | **inference-seed replicate**: the C0 and R0 checkpoints evaluated a second time with a different eval batch size (8 vs 32), shuffled token order and non-deterministic kernels on | - | the inference floor: picks flipped and delta PDMS for an unchanged checkpoint. REFe selection is an argmax, so the "seed" is bf16 reduction order, not a sampler |

All arms see the same sets in the same order for the same seed, so C vs R is a within-seed pair. The RL-train scenes are chosen from navtrain scenes **the SFT-4 scorer did not train on** (the 28,727 labels not written at arming plus unlabelled scenes [I, SFT-4 Amend. 4]; exact list [U], to be frozen from the SFT-4 `data` event), 48,000 scenes, log-disjoint from an RL-val set of 3,000 scenes (>= 60 logs). The PhysicalAI parity rule is not involved (`SPEC_P6_PAPER_FAITHFUL.md` section 1); this is a navtrain partition, not a re-selection of any programme corpus. The data-selection package (`DATA_SELECTION_REVIEW.md`, other owner) is the place to change *what* is in the bank; this arm consumes whatever it yields.

Because the fan is frozen, **labels are computed once and shared by all arms**, and the RL-val read is a **table lookup** (the true PDMS of any scorer's pick), costing nothing.

### 6.2 Budget

| item | where | cost | class |
|---|---|---|---|
| launch-gate smoke, identity, mutation arms | A40 | 0.1 day | E |
| dump the 64 proposals for ~51,000 scenes (forward only) | A40 | 51,000 x 0.07-0.22 s = 1.0-3.1 h (0.22 s is the measured on-policy dump rate [I, `raw/2026-09-26-onpolicy-switch/dump_cost_pause_ab.txt`]) | E |
| label the 64 proposals plus `D`, `K`, `L` terms | pod CPU, concurrent with the dump | 51,000 x (1.78-6) CPU-s = 25-85 CPU-h, 1-4 h on 24 vCPU | E |
| 5 training runs x 1,000 updates x 14.4 s | A40 | 5 x 4.0 h = 20 h | E (14.4 s is [I]) |
| 8 full-navtest reads (A0 drift check, C0, C1, R0, R1, R2, plus 2 inference replicates) | dev box, never the pod | 8 x ~5.2 h = ~42 h serial (seam 17,497 s + harness 1,336 s per read [M, `RESULT_SFT4.md`]); runs behind the training | M |
| **A40 total** | | **about 23-26 h = 1.0-1.1 A40-days** (margin to the 2-day cap: 0.9) | E |

### 6.3 Endpoints and the bar (written before any read)

- **Primary:** full-navtest PDMS, 12,146 tokens, 136 logs, NAVSIM v1.1, paired log-cluster bootstrap (10,000 resamples, seed 20260927). It answers *another draw of logs*, **not** another training run or another inference run. Tier: pseudo-simulation of a 4 s open-loop plan against logged agents; it is not closed loop.
- **Contrasts:** `R0 - C0` and `R1 - C1` (within-seed pairs).
- **ADOPT-FOR-NEXT-STAGE** iff all of:
  1. both contrasts have point Delta >= +0.25 and their mean is **>= +0.5**;
  2. the pooled paired CI lower bound is > 0;
  3. the mean contrast exceeds **2 x max(|R0 - R1|, |C0 - C1|, inference-replicate |delta|)** (with two replicate pairs the floor estimate is crude; this is a necessary condition, not a sufficient one);
  4. no longitudinal or lateral families6 component separates adversely (the SFT-4 rule [M]), and the g3-g5 telemetry bounds hold;
  5. every sub-score paired delta is reported, including EP.
- **REFUTED** iff the pooled upper bound < +0.25. **Otherwise NOT PROVEN.** At published gains of +0.2 to +1.2 [P], NOT PROVEN is a likely outcome; it is a registered outcome, not a failure to decide.
- **Detectable effect [E]:** SFT-4 vs deployed was +3.79 [+3.05, +4.54] (half-width 0.75) [M]. A paired delta of +0.5 with a similar token-level spread would have a half-width of roughly 0.3-0.5, so the bar is near the resolution limit. Nothing is claimed below it.
- **Secondary mechanism checks (not gating):** EP delta >= +1.0 if the decision-focused hypothesis (5.2) is right; pick / fan-mean path up; shortest-bin share down; the clean-fan EP loss (7.57 of 8.74 weighted points at the old scorer [M]) re-measured on the arm.
- **Four-family guards:** LONGITUDINAL (speed MAE, along-track MAE, speed bias, along-track bias), LATERAL (heading, cross-track, curvature, yaw-rate MAE) by the adverse-separation rule; TACTICAL lateral / longitudinal decision kappa, reported, may not fall by > 0.03 versus C0 (no interval exists, so a threshold); **STRATEGIC: PAUSED** (PI ruling 2026-09-19, `PI_DECISION_QUEUE.md` item 25), reported as paused, not as a pass, not omitted.
- **Selection of the checkpoint:** the final update of each arm. No navtest-based early stopping. Hyper-parameter and early-stop decisions use RL-val lookups only.

### 6.4 What kills it

| code | condition | action |
|---|---|---|
| K0 | no launch-gate PASS token; a new flag without a G-DVB entry | do not start |
| K1 | `lam_pg = 0` run does not reproduce C0 at 50 updates (bf16 tolerance); the `-J` gradient on the scorer reads 0 (control must read exactly 0.0 with the term off, non-zero with it on); the mutation arm (delete the line emitting the term) stays green | gate fails, fix, no GPU-day spent |
| K2 | grad norm > 20 x the control's median, or NaN | stop that arm, hand off (our DDv2 port reached 15,712 vs 147) |
| K3 | any g3 / g4 / g5 telemetry bound breached at two consecutive 250-update checks on RL-val | stop that arm |
| K4 | RL-val true PDMS of R below C by > 0.5 at two consecutive checks | stop that arm |
| K5 | lane-change rate of R2 above 2 x the label base rate | stop R2 |
| K6 | any arm exceeds 6 h on the A40 | stop, write HANDOFF, fresh agent continues |

### 6.5 What has to be built (none of it exists)

`rl_reward.py` (terms b, c, d, e, with a G-REWARD-ID test: `g > 0 iff DAC = 0` on 10,000 labelled proposals, and the analytic identities `lam = 0 => R = PDMS`); `rl_dump_fan.py` (frozen-generator proposal dumper); an extended labeller (adds `D`, `K`, `L`); the EPO term in a copy of the SFT-4 scorer trainer with G-DVB entries; the RL-val lookup evaluator. A fresh agent can start from sections 3-6.

### 6.6 Outcomes committed in advance, and the next lever (RULE ZERO)

- **ADOPT:** add the EPO term to P6's scorer loss. The 64 labels are computed every step anyway, so the cost is near zero. Then run the CLOVER-style alternating generator stage at P6 scale.
- **NOT PROVEN with EP flat:** the decision-focused hypothesis is unsupported; the next lever is the calibration of the TTC / NC / DAC heads (REVIEW_7 l.209-218), which a BCE-only control already tests. Re-read C0 against A0 before concluding anything about RL.
- **REFUTED:** RL on a frozen fan does not help at this base. Remaining levers, with blockers named: (a) the paper-faithful run P6 (blocked on P2's verdict, a pod with >= 24 vCPU, and the PI's P6-T/C/N/Y decisions); (b) generator-side alternation (blocked on a stochastic-policy design and A100-class compute that we do not have); (c) tracker-aware targets (P5), a zero-training EP lever of +3.58 EP on the teacher [M] and +0.41 on the student [M, P4 F2].

## 7 · Not done, and what is uncertain

- No RL, GPU or pod work ran. The reward is a draft with initial weights [H]; the sensitivity arm is not in the 1.1-day budget.
- The lane-change pilot is 336 frames from `trainval`, not navtrain tokens, 28.9 % near-stationary: representativeness [U]. Boundary-type ids {0, 2, 3} are not decoded [U].
- DIVER's controlled RL row, AutoVLA's sub-scores and checkpoint-selection set, ReCogDrive's RL-stage GPU count, and DriveDPO / Curious-VLA / ReflectDrive-2 compute were not read [U].
- DDv2's reward composition is not in the paper text I read; the "DAC -1, GT bar" detail comes from our own code audit [I].
- 14.4 s per update and 1.78 CPU-s per set are inherited handoff numbers; neither has a banked log [I].
- CLOVER's "three seeds, < 0.02 PDMS" is a claim in an appendix; our own replicate arm produced 9.5 % false-positive "separated" cells on another rig [M], so I do not treat it as a floor for REFe.
- The SFT-4 pick telemetry uses 10,070 of 12,146 tokens; 2,076 are re-planned by the route fix and are not comparable.

## 8 · Proposed register rows (for the Master Mind to land; not edited here)

| id | claim | class / status |
|---|---|---|
| `D-RL-NAVSIM-PRIOR-1` | controlled RL / preference steps on NAVSIM gain +0.35 (same-family generator-scorer, 94.1 base) to +2.7 (87.6 base); +4.3 and +8.6 start from weak bases; no controlled step reproduces an EP gap of 11.5 | PUBLISHED (15 rows, banked primaries); one row confounded (EvaDrive) |
| `D-RL-EP-TRADE-1` | RL moves EP mostly by trading TTC / NC (ReflectDrive-2 EP +7.1 / TTC -4.7; DDv2 selector EP +4.6 / TTC -2.5); selector-side preference moves EP +0.3 | PUBLISHED |
| `D-REFE-SLOWPICK-SFT4-1` | after SFT-4 u400 the pick is still shorter than the fan median on 79.7 % of 10,070 comparable tokens (pre-SFT-4: 75.4 %); pick / fan-mean path 0.914 (0.885); shortest-bin share 0.506 (0.458) | MEASURED, `raw/pick_length_telemetry.json`, controls reproduce REVIEW_7 |
| `D-REFE-LCLABEL-SPARSE-1` | human lane change 2.7 % [1.4, 5.0] of 336 pilot frames; slow-lead AND adjacent-free 0.9 %; 0/14 slow-lead frames had a human lane change | MEASURED pilot, representativeness UNVERIFIED |
| `H-REFE-RL-S1` | scorer-only exhaustive policy gradient on the frozen fan beats the SFT-continued control by >= +0.5 PDMS (full navtest, paired, above the replicate floor) | PRE-REGISTRATION DRAFT, not in force |

## 9 · Deliverable manifest

| artifact | where | state |
|---|---|---|
| this review | `repo: TanitAD Research Lab/Architecture & Inference/Research/2026-10-11-refe-data-and-rl/RL_POSTTRAIN_REVIEW.md` | staged |
| label-activation measurement script | `repo: .../2026-10-11-refe-data-and-rl/code/measure_label_activation.py` | staged |
| label-activation raw output (336 frames) + log | `repo: .../raw/label_activation.json`, `raw/label_activation.log` | staged |
| pick-length telemetry script + raw JSON | `repo: .../code/pick_length_telemetry.py`, `raw/pick_length_telemetry.json` | staged |
| 4 newly banked PDFs: 2508.09158 EvaDrive, 2509.17940 DriveDPO, 2506.13757 AutoVLA, 2603.06049 Curious-VLA (2605.15120 CLOVER was already banked, citation added) | `repo: TanitAD Research Lab/Library/papers/`, indexed in `library.json`, `LIBRARY.md` | staged (the two index files also carry other agents' entries) |
| extracted PDF text used for reading | scratchpad only (`C:/Users/Admin/AppData/Local/Temp/claude/.../scratchpad/txt/`) | **exists in ONE place, deliberately not staged** (re-derivable with `pdftotext`) |
| anything on a pod | none | - |
