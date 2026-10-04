# RESULT — WP-D (refcv8 R8-6): why refcv7's map and box heads stop where they stop, and what that says about the fix

Architecture & Inference, WP-D, 2026-10-04 (Berlin). Plan: `Project Steering/PLAN_REFCV8.md` §0 R8-6 (and R8-5,
owned by WP-C). Model: refcv7-r101-s0 (launch tree `fec3a0d`, `D:/refcv7_eval_kit/ckpt/config.json`, checkpoints md5 in
`…/ckpt/MD5SUMS`). The design that follows from this file is `PERCEPTION_DESIGN.md`; the probes are
`PREREG_DRAFT_WPD_PROBES.md`.

**Tier of every number: OPEN-LOOP PERCEPTION DIAGNOSTIC** (teacher-forced forward or the run's own log; no planner in
the loop; nothing here is a T1 or driving number). **The four metric families** (longitudinal / lateral / tactical /
strategic) are **N/A for this file, by family, for one reason: no plan is produced or scored** — perception quality
feeds the longitudinal family (distance keeping needs the lead box) and the planner's BEV, but no family number exists
here. **Estimators:** in-run curves are window means over a FIXED 128-window / 80-episode set (spread = checkpoint-to-
checkpoint model variation, not episode sampling); late slopes are OLS with an OLS SE that is OPTIMISTIC (rows are
serially correlated); decision-grade intervals are episode-cluster bootstraps over EVAL-DIAG (137 labelled episodes,
B = 1,000, paired where arms share windows). They answer *"another draw of EPISODES"* only — one training seed, no
replicate (H-ESTIM-SEED-1).

## 0. Headline

1. **The BOX heads were NOT plateaued — they were still improving when the schedule ended.** The cosine schedule ran
   ~1.08 epochs and annealed both groups to LR ≈ 0 at 50,400 (MEASURED, `lr` and `data_epoch` in the run's
   `metrics.jsonl`). Box3d AP@2 m (class-agnostic, in-run) rose **0.181 → 0.229** from 30k to 48k with a late slope of
   **+0.023 per 10k steps** (OLS t ≈ 12) *while the LR fell from 1.9e-5 to 0*; the agent head identically (+0.025 /
   10k). **On the decision-grade 1,061 EVAL-DIAG windows (PROBE-0, §2.1), paired 50,400 − 30,000: AP@2 m 0.199 →
   0.248, +0.050 [+0.041, +0.058]; AP@1 m 0.084 → 0.130, +0.046 [+0.037, +0.056]; agent AP@2 m +0.043 [+0.034,
   +0.054].** The relative late gain is largest at the tightest tolerances (AP@1 m +55 %, AP@2 m +25 %, AP@4 m +13 %):
   the head was still learning to PLACE boxes.
2. **The MAP head IS near its plateau, and its ceiling is set by IMAGE SUPPORT, not by training length.** Late slopes
   ≤ +0.004 IoU per 10k (lane), +0.0014 (edge), ≈ 0 (crosswalk since ~17.5k); 90 % of every thin class's gain was in
   by 14.5k–29k. ANALYTIC (`raw/geom_bound.json`): **the whole 40–100 m half of the 10 cm map — 60 % of its depth —
   is decoded from ~1.3 stride-8 feature rows (≈ 11 image rows)**; one stride-8 row in 80–100 m must explain **903**
   label rows, a 0.15 m painted line is **0.8 px** wide there. MEASURED (diagnostics M-b): edge **precision** at 1 m
   tolerance is flat over range (0.72 → 0.69) while **recall** at 1 m collapses 0.60 → 0.23 → 0.11 → 0.03 → 0.01 —
   the far fall-off is the head not firing where the image has < 1 feature row, not lines placed wrongly.
3. **The trunk is owned by perception, so "the planner starves perception" is FALSE.** The detector's "aux" side
   (box3d + 10 cm map + tactical decoder, each at its loss weight) carries **99.2–99.5 %** of the trunk gradient norm
   against the trajectory loss at every phase, with cosine ≈ 0 between them (|cos| < 0.015; MEASURED, the run's
   conflict detector, 5,039 rows; the agent-head loss is in neither side, `refc_v3_train.py` CONFLICT_*_TERMS). Rebalancing against the planner is not a
   perception lever. What is NOT logged is the split *inside* aux (box vs map vs tactical) — a 5-minute probe (P-GRAD).
4. **Head initialisation — MEASURED: the box head's 300 "learned" reference points never moved.** Median displacement
   from their seed-0 uniform init **0.031 m** after 50,400 steps (max 0.18 m) — the magnitude of a random walk
   (lr·√N ≈ 0.02 m), i.e. ~zero mean gradient (`raw/refpts_by_step.json`, control: the init is REPRODUCED from the
   module's rule, not read from a file). The anchors ARE their random init: **91 / 300 sit at |y| > 12 m**. The
   presence prior, by contrast, is irrelevant after training (bias −4.595 → −4.529; the diagnostics already showed
   undoing it overshoots conf_ratio to 88.9). Details and the design answer in `PERCEPTION_DESIGN.md` §1.
5. **The planner reads the WEAKER of the two detectors.** The planner's agent head decodes 416 stride-32 tokens with
   learned queries and no BEV (`refc.py` agent seam); box3d reads stride-16 tokens + BEV with anchors. EVAL-DIAG
   AP@2 m **0.131 (agent) vs 0.248 (box3d)** (MEASURED, diagnostics `B_box.json` T0).
6. **The boxes are better placed than they are RANKED.** At the TRAIN P = R gate there are **1.97 (box3d) / 3.12
   (agent) boxes per detected object**; among the slots on one object the top-scored one is the nearest only **34.5 % /
   23.0 %** of the time (median 1.22 m away while a 0.67 m slot exists). Re-ranking each object's OWN slots by
   placement and de-duplicating (an ORACLE ceiling with the same boxes) takes box3d **AP@1 m 0.130 → 0.263** and
   AP@2 m → 0.349; admissible NMS alone gives AP@2 m 0.339 but LOSES AP@1 m (−0.019). The residual error is
   **longitudinal** (median |Δx| 0.96 / 1.56 m vs |Δy| 0.38 / 0.46 m at 0–20 / 20–40 m). The IGNORE-radius presence
   exemption is ELIMINATED as a duplicate cause (objects next to an IGNORE row have fewer duplicates). §2.

## 1. Training dynamics from the run's own log (priority 1)

Source: `TanitAD Research Lab/Data Engineering/Research/2026-10-04-refcv8-data-audit/D1_label_census/raw/
refcv7_run_records/metrics.jsonl` (38,608,246 B; 6,150 rows = 1,008 train rows every 50 steps, 101 in-run eval rows
every 500 steps on a fixed 128-window set, 5,039 conflict-detector rows every 10 steps). Script `code/td_analyze.py`
→ `raw/td_summary.json`, `raw/td_curves.json`. Row kinds are identified by key content, never position.

### 1.1 The schedule the heads saw

| step | encoder-group lr (logged) | head-group lr (= 2 × encoder, `--opt dd`) | data_epoch |
|---|---|---|---|
| 2,000 | 5.0e-5 (peak) | 1.0e-4 | 0 |
| 20,000 | 3.5e-5 | 7.0e-5 | 0 |
| 30,000 | 1.9e-5 | 3.8e-5 | 0 |
| 40,000 | 5.5e-6 | 1.1e-5 | 0 |
| 50,400 | 5.3e-14 | ~1e-13 | 1 |

MEASURED (`lr`, `data_epoch`; the head group is the encoder group × 2 by `param_groups_dd`, `timm_trunk.py:1316-1356`,
encoder_lr_mult 0.5). 50,400 × 16 = 806,400 window samples over 746,946 train windows ≈ **1.08 epochs**. Step time
**10.03 s/step** over the run (MEASURED, `elapsed_s`).

### 1.2 Box heads: monotone, decelerating, NOT converged

In-run class-agnostic AP (`eval_*_ap2m`) and the per-threshold class-mean mAP, window means ± SD over 5 rows centred
on the step (MEASURED, `raw/td_summary.json`):

| head | metric | 5k | 10k | 20k | 30k | 40k | 47.9k | late slope /10k (30–50.4k) | step at 90 % of gain |
|---|---|---|---|---|---|---|---|---|---|
| box3d | AP@2 m (class-agnostic) | 0.047 | 0.062 | 0.123 | 0.181 | 0.221 | **0.229** | **+0.023 ± 0.002 (t 12.0)** | 36,000 |
| box3d | mAP@1 m | 0.002 | 0.003 | 0.007 | 0.015 | 0.021 | 0.028 | +0.007 (t 6.5) | 46,500 |
| box3d | mAP@4 m | 0.023 | 0.028 | 0.048 | 0.092 | 0.080 | 0.090 | −0.001 (t −0.1) | 28,500 |
| box3d | mAP@2 m, 0–20 m | 0.020 | 0.022 | 0.046 | 0.084 | 0.170 | **0.240** | +0.083 (t 7.3) | 44,000 |
| box3d | objectness AUROC | 0.767 | 0.808 | 0.864 | 0.893 | 0.897 | 0.902 | +0.003 (t 5.0) | 25,500 |
| box3d | centre err p50 (m) | 1.045 | 1.011 | 0.921 | 0.813 | 0.754 | 0.743 | −0.030 (t −5.3) | 32,000 |
| agent | AP@2 m (class-agnostic) | 0.022 | 0.031 | 0.053 | 0.088 | 0.120 | **0.133** | **+0.025 ± 0.002 (t 14.5)** | 40,500 |
| agent | centre err p50 (m) | 1.142 | 1.013 | 0.781 | 0.650 | 0.619 | 0.597 | −0.028 (t −5.8) | 28,500 |
| box3d | precision at the P = R gate (256 TRAIN calib windows) | 0.041 | 0.039 | 0.186 | 0.262 | 0.291 | 0.295 | +0.016 (t 8.2) | 30,000 |
| agent | precision at the P = R gate (same) | 0.000 | 0.006 | 0.047 | 0.149 | 0.205 | 0.212 | +0.045 (t 3.6) | 37,500 |
| box3d | the P = R gate itself | 0.259 | 0.262 | 0.244 | 0.244 | 0.255 | 0.259 | +0.005 | — |
| box3d | conf_ratio at the declared 0.5 gate | ≤ 0.006 at every one of the 101 rows (diagnostics `inrun_curve.json`) | | | | | | | |

Reading (MEASURED, with the estimator caveats in the header):
* **Still rising at LR ≈ 0.** AP@2 m gained ~20 % of its final value after step 30,000 (in-run 0.181 → 0.229;
  1,061 windows 0.199 → 0.248) while the LR fell from 1.9e-5 (encoder group) to zero. A head whose metric is still climbing while the step size goes to zero was stopped by the schedule,
  not by its capacity. This is the DETR slow-convergence signature (DETR needs 500 epochs, Deformable DETR 50
  [LIB:2010.04159]; DN-DETR reaches its baseline in half the epochs [LIB:2203.01305]) on ~1 epoch.
* **What improves late is mostly placement.** On the 128 in-run windows the class-mean mAP@4 m and objectness AUROC
  were at 90 % of their gain by ~25–28k, and the near band (0–20 m class-mean AP@2 m 0.084 → 0.240) carried the late
  gain. ⚠️ The class-agnostic AP@4 m on 1,061 windows was NOT flat (+0.038 [+0.029, +0.047], 30k → 50.4k, PROBE-0):
  the in-run "AP@4 m saturated" reading is a class-mean / 128-window artefact; the correct statement is the relative
  one in the headline (+55 % at 1 m vs +13 % at 4 m).
* **No oscillation, no over-fit.** 5-row SDs shrink with the LR (0.010 at 20k → 0.0007 at 48k); train presence loss
  0.478 → 0.381 and eval 0.505 → 0.405 move together.
* **The decoder's layers barely refine.** Deep-supervision layer losses at 47.9k: eval 5.79 / 5.61 / 5.55 (layer 0 →
  2), **−4 %**; train 5.13 / 4.73 (layer 0 → 2), −8 %. DETR's own 6-layer decoder gains +8.2 AP from the first to the
  last layer and its first layer is the only one NMS helps [LIB:2005.12872]: our 3-layer decoder behaves like an
  early DETR layer — consistent with the duplicates (§2.2).
* **Presence never became confident:** `presence_frac_confident` ≤ 1e-4 at every eval row (the 0.5 gate under focal
  α 0.25 = match belief 0.75; diagnostics §2.4).
* **The presence-exemption zone grows.** Unmatched slots within 2 m of an IGNORE row get presence weight 0 (A9 R3).
  Eval: **41 → 49 exempt slots per 16-window batch vs 42 matched** (rising, late slope t 4.7); train at the end: 68
  (box3d) / 88 (agent) exempt vs 48.7 matched per batch. The head increasingly parks slots where no negative gradient
  reaches them — but PROBE-0 (§2.2) shows this does NOT drive the duplicates or the false positives (exempt slots are
  3.8 % / 6.4 % of FPs); eliminated as a lever.
* **Loss composition of one box3d decoder layer at the end** (train, last 40 rows; weights from `SLOT_LOSS_W` /
  `slot_refine`): presence 0.38 × 2.0 = 0.76 · centre 0.98 · cls 0.64 · size 0.46 · yaw 0.39 · **rates 2.44 × 0.5 =
  1.22 (the largest term, ~26 %)** · z 0.17 · h 0.12. The rate targets are unclipped finite differences that D3
  found carry track-id-switch jumps up to 276 m (6,410 frames) — WP-C owns the clip.

### 1.3 Map head: near plateau

Raw-argmax IoU on the in-run set (pooled inter/union over bands; the declared-rule IoU is the decision artefact the
diagnostics retired):

| class | 5k | 10k | 20k | 30k | 40k | 47.9k | late slope /10k | 90 % of gain |
|---|---|---|---|---|---|---|---|---|
| drivable | 0.529 | 0.544 | 0.561 | 0.564 | 0.571 | 0.571 | +0.003 (t 4.3) | 17,000 |
| lane | 0.111 | 0.124 | 0.134 | 0.144 | 0.152 | 0.154 | +0.004 (t 7.8) | 29,000 |
| crosswalk | 0.039 | 0.042 | 0.051 | 0.050 | 0.050 | 0.050 | −0.001 (t −1.1) | 11,500 |
| edge | 0.014 | 0.020 | 0.026 | 0.028 | 0.029 | 0.030 | +0.0014 (t 5.9) | 14,500 |
| arrow | 0.013 | 0.032 | 0.049 | 0.056 | 0.059 | 0.062 | +0.003 (t 4.7) | 29,000 |
| lane 0–20 m | 0.278 | 0.284 | 0.293 | 0.292 | 0.300 | 0.298 | +0.001 (t 0.9) | 4,000 |
| lane 60–80 / 80–100 m | 0.031 / 0.006 | 0.035 / 0.006 | 0.022 / 0.003 | 0.029 / 0.005 | 0.043 / 0.017 | 0.050 / 0.024 | +0.011 / +0.012 | 40–41.5k |
| edge 0–20 m | 0.042 | 0.050 | 0.059 | 0.062 | 0.064 | 0.064 | +0.0007 (t 3.0) | 16,000 |

MEASURED. The decision-grade 1,112-window version (diagnostics M-e, ORACLE-threshold IoU 5k → 50.4k: lane 0.138 →
0.164, edge 0.029 → 0.042) says the same. **More steps buy ≤ +0.004 lane IoU per 10k** — the map needs a different
input or target, not a longer run. (The far-band lane creep under annealing is real but tiny: 60–80 m 0.029 → 0.050.)

### 1.4 Who owns the trunk gradient

Conflict-detector rows (`cd_*`, every 10 steps). aux = every LIVE term of `CONFLICT_PERCEPTION_TERMS` at its loss
weight — on this arm box3d (1.0), map_hires (1.0) and tac_v6 (1.0); r7_wta / r7_scorer / map / bev are 0 and absent
(config `effective_weights`). traj = `CONFLICT_PLAN_TERMS` (traj × TRAJ_WEIGHT). ⚠️ The agent-head loss (w 1.0) is in
NEITHER side. Aux share of the gradient norm = gn_aux / (gn_aux + gn_traj).

| trunk part | 5k | 20k | 30k | 47.9k | cos(aux, traj) |
|---|---|---|---|---|---|
| whole trunk | 0.992 | 0.995 | 0.994 | 0.994 | −0.008 … +0.009 |
| stem / layer1 / layer2 | 0.991–0.992 | 0.995 | 0.995 | 0.994 | |
| layer3 | 0.993 | 0.996 | 0.995 | 0.994 | |
| **layer4** | **0.216** | 0.345 | 0.365 | 0.373 | |
| temporal fuse | 0.991 | 0.994 | 0.993 | 0.992 | |

MEASURED. Only layer4 (read by the planner's stride-32 map and by the agent head, whose loss this detector does not
count) is trajectory-dominated. The map's own
share inside "aux" is not logged; its loss is 0.91 of a 36 total (~2.5 %) against box3d 14.7 (~41 %) — so the map's
trunk gradient is plausibly small, UNVERIFIED (P-GRAD).

## 2. PROBE-0 — the box heads' own EVAL-DIAG packs, zero GPU

Inputs: the diagnostics package's trainer-made `window_packs` for box3d and agent on EVAL-DIAG (1,061 labelled
windows / 137 episodes; last decoder layer's 300 slots, VIS-1 POS / IGNORE targets) at 5k / 15k / 20k / 30k / 50.4k
— copied from Thor `refcv7_post/diag/out/*.packs.pkl` with md5 equal on both ends (recorded in
`raw/box_probe0.json` "inputs"; NOT in the repo, they carry a per-clip integer id) — and `ckpt_50400.pt` for the
anchors. Script `code/box_probe0.py` (CPU, 29 min). **Controls, all PASS:** C1 the vectorised greedy matcher ==
the trainer's `detection_metrics.greedy_rows` (kind / slot / GT identical on 120 window × threshold cases; the
confidence differs at the 7th decimal because theirs is a float32 sigmoid — the control was first written as an
exact match and FAILED on that digit, so it now asserts identity of the assignment and |Δconf| < 1e-6); C2 AP@2 m at
50,400 = the diagnostics' T0 (box3d 0.24828 vs 0.2483, agent 0.13140 vs 0.1314); C3 a paired bootstrap of an arm
against itself reads exactly 0; C4 the identity oracle and NMS radius 0 reproduce T0 exactly.

### 2.1 The AP curve on the decision-grade windows (class-agnostic, all ranges unless stated)

| head | metric | 5k | 15k | 20k | 30k | 50.4k | 50.4k − 30k, paired [95 % CI] |
|---|---|---|---|---|---|---|---|
| box3d | AP@0.5 m | 0.001 | 0.004 | 0.005 | 0.014 | 0.030 | +0.016 [+0.010, +0.021] |
| box3d | AP@1 m | 0.014 | 0.029 | 0.054 | 0.084 | 0.130 | +0.046 [+0.037, +0.056] |
| box3d | **AP@2 m** | 0.066 | 0.102 | 0.138 | 0.199 | **0.248** | **+0.050 [+0.041, +0.058]** |
| box3d | AP@4 m | 0.128 | 0.187 | 0.218 | 0.292 | 0.330 | +0.038 [+0.029, +0.047] |
| box3d | AP@2 m, 0–20 m | 0.114 | 0.171 | 0.239 | 0.333 | 0.415 | +0.082 [+0.061, +0.101] |
| box3d | AP@2 m, 20–40 m | 0.039 | 0.069 | 0.085 | 0.130 | 0.166 | +0.036 [+0.026, +0.050] |
| box3d | AP@2 m, 40–60 m | 0.020 | 0.031 | 0.037 | 0.057 | 0.070 | +0.014 [+0.007, +0.021] |
| agent | **AP@2 m** | 0.024 | 0.054 | 0.062 | 0.088 | **0.131** | **+0.043 [+0.034, +0.054]** |
| agent | AP@1 m | 0.006 | 0.023 | 0.027 | 0.044 | 0.083 | +0.039 [+0.032, +0.047] |

MEASURED (`raw/box_probe0.json` Q1). Every band and threshold was still rising over the last 20k steps (the estimator
answers "another draw of episodes"; this is one run's trajectory, not a lever comparison, so no replicate is needed
for the statement "this run had not converged").

### 2.2 Duplicates at the TRAIN P = R gate (box3d 0.2567, agent 0.2307)

Greedy matching at 2 m on the slots above the gate; a false positive is a DUPLICATE when its nearest positive (within
2 m) was already detected by a higher-scored slot.

| head | detected objects | duplicates | **boxes per detected object** | duplicates / all FPs | FPs that are presence-exempt slots | by band 0–20 / 20–40 / 40–60 m |
|---|---|---|---|---|---|---|
| box3d | 1,015 | 982 | **1.97** | 43 % | 3.8 % | 2.06 / 1.72 / 1.11 |
| agent | 699 | 1,484 | **3.12** | 56 % | 6.4 % | 3.28 / 2.32 / 1.67 |

MEASURED (Q2). The plan's 2.12 (R8-5, INHERITED) is the same phenomenon on a different count. **Eliminated
candidate:** "the IGNORE-radius presence exemption breeds duplicates" — objects WITH an IGNORE row within 2 m have
FEWER duplicates (box3d 1.22 vs 1.99 boxes per object; agent 2.09 vs 3.18), and exempt slots are only 3.8 % / 6.4 % of
false positives. The exemption is not the mechanism; it is dropped as a lever (arm B5 of the first prereg draft).
The duplicates sit in the NEAR band, where ~9 anchors reach each object (§2.6) and a 3-layer decoder does not
suppress them — DETR's first-layer behaviour (§1.2).

### 2.3 Is the presence score aware of localisation?

Among the slots within 2 m of a positive (3,072 objects with ≥ 2 such slots, box3d): **the top-scored slot is the
nearest one for only 34.5 % (agent 23.0 %)**; the top-scored slot sits a median **1.22 m** from the object while the
nearest slot sits **0.67 m** away (agent 1.31 vs 0.54 m); the mean within-object rank correlation between score and
nearness is **+0.16** (agent +0.11). MEASURED (Q3). ⇒ the head usually HAS a well-placed slot and ranks a worse one
first — the ranking defect a quality-aware presence target exists to fix (VFL / Stable-DINO).

### 2.4 Same-box ceilings: NMS (admissible) and quality re-ranking (ORACLE)

| head | arm | AP@0.5 m | AP@1 m | AP@2 m | AP@4 m |
|---|---|---|---|---|---|
| box3d | T0 (as trained) | 0.030 | 0.130 | 0.248 | 0.330 |
| box3d | NMS r 1 m | 0.025 | 0.134 | 0.271 (+0.023 [+0.020, +0.028]) | 0.362 |
| box3d | **NMS r 2 m** | 0.019 | 0.111 (−0.019) | **0.339 (+0.090 [+0.076, +0.107])** | 0.483 (+0.153) |
| box3d | NMS r 3 m | 0.017 | 0.091 | 0.336 | 0.561 |
| agent | T0 | 0.027 | 0.083 | 0.131 | 0.178 |
| agent | **NMS r 2 m** | 0.012 | 0.085 | **0.279 (+0.148 [+0.127, +0.171])** | 0.409 |

MEASURED (Q4). NMS buys AP@2–4 m and COSTS AP@0.5–1 m (it keeps the higher-scored, worse-placed duplicate — §2.3).
WP-C's F4b numbers (AP@2 m 0.349 / 0.300, INHERITED via PLAN R8-5) are the same lever at a slightly different
setting.

**The quality re-ranking ceiling (ORACLE; `code/box_probe0b.py` → `raw/box_probe0b.json`).** Within each object's
cluster of slots (≤ 2 m, each slot assigned to its nearest GT), the cluster's OWN scores are re-assigned so that the
nearest slot holds the highest: the multiset of scores and every cluster's rank against the rest of the list are
unchanged (controls C5 score multiset preserved, C6 identity permutation == T0: PASS).

| head | arm (ORACLE = uses GT; a ceiling, not a result) | AP@0.5 m | AP@1 m | AP@2 m | AP@4 m |
|---|---|---|---|---|---|
| box3d | ORACLE re-rank | 0.059 (+0.029 [+0.023, +0.035]) | 0.192 (+0.062 [+0.051, +0.073]) | 0.248 (±0) | 0.329 (±0) |
| box3d | **ORACLE re-rank + NMS r 2 m** | 0.077 (+0.048) | **0.263 (+0.134 [+0.109, +0.159])** | **0.349 (+0.101 [+0.085, +0.118])** | 0.473 |
| agent | ORACLE re-rank + NMS r 2 m | 0.105 (+0.078) | 0.240 (+0.157 [+0.125, +0.192]) | 0.293 (+0.161 [+0.136, +0.188]) | 0.406 |

⇒ **The same boxes, ranked by their own placement and de-duplicated, would DOUBLE box3d AP@1 m (0.130 → 0.263).**
The head's localisation is better than its ranking lets AP show; NMS alone loses AP@1 m (−0.019) because it keeps the
higher-scored, worse-placed duplicate. A quality-aware presence target and a de-duplicating decoder (design B2) aim at
exactly this gap.
⚠️ **Disclosure (post-hoc):** PROBE-0's own Q4 oracle (score × q for slots within 2 m of a GT, all other slots
untouched) demoted true positives against untouched far false positives and read AP@2 m 0.105 — an ORACLE DESIGN
ERROR, not a property of the head. It stays in `raw/box_probe0.json` as run; the permutation oracle above was defined
after seeing it and supersedes it.

### 2.5 Where the localisation error lives: LONGITUDINAL

Greedy TPs at 4 m among slots above the P = R gate, |Δx| (along the camera axis) vs |Δy| (lateral), median (p75):

| band | box3d \|Δx\| | box3d \|Δy\| | agent \|Δx\| | agent \|Δy\| | ANALYTIC ground depth per image row | lateral m / px |
|---|---|---|---|---|---|---|
| 0–20 m (n 843 / 662) | **0.96 (1.79)** | 0.38 (0.85) | 1.11 (2.18) | 0.35 (0.87) | 0.14 m | 0.020 |
| 20–40 m (n 331 / 152) | **1.56 (2.42)** | 0.46 (0.93) | 1.74 (2.56) | 0.45 (1.00) | 1.27 m | 0.061 |
| 40–60 m (n 12 / 7) | 1.14 (2.48) | 0.36 (0.52) | 1.49 (1.95) | 1.02 (2.10) | 3.53 m | 0.102 |

MEASURED (Q5) beside ANALYTIC (`raw/geom_bound.json`, band centres). The error is **2.5–3.4× larger along range than
across it**. In 0–20 m it is ~7 image rows of depth and ~19 px laterally — far above what the pixels resolve, so near
range is a learning / representation limit (depth cues, geometry in the memory), not an optics limit; in 20–40 m the
longitudinal error is ~1.2 image rows, near the single-frame ground-contact bound — there, depth cues beyond the
contact row (size, temporal parallax, LiDAR-supervised depth) are what can help. The 40–60 m band has too few TPs to
read (n 12).

### 2.6 The box3d anchors

| quantity | value |
|---|---|
| anchors within the ±4 m tanh reach of a positive | median **9** (p10 6); 0.03 % of positives unreachable |
| distinct slots that produce a TP at the P = R gate | **125 of 300 (42 %)** |
| TP slots whose anchor is at \|y\| > 12 m | 16 of 125 |
| TP offset from its anchor, mean \|Δx\|, \|Δy\| | 1.57 m, 2.16 m |
| **TPs whose offset saturates the tanh (> 3.6 m on an axis)** | **16.5 %** — where d tanh / d raw ≤ 0.19 |

MEASURED (Q6) + `raw/refpts_by_step.json` (anchors moved a median 0.031 m in 50,400 steps). Reach is not the
problem; the static uniform layout is: 58 % of the queries never produce a confident TP, ~9 anchors compete for each
object (duplicates, §2.2), and one TP in six regresses at the flat end of its tanh.

## 3. The geometry of the map's range fall-off (ANALYTIC)

`code/geom_bound.py` → `raw/geom_bound.json`: the cylindrical projection the lift itself uses (`bev_lift.py`,
`rig_projection.project_cam_to_frame`), f_ref 488.92 px, 416 × 1024, camera height 1.45 m (census range 1.20–1.69 m
also stored), zero pitch (±1° moves the horizon 8.5 px and changes band spans only where the image bottom clips the
0–20 m band). Controls in-line: a point placed at a known offset reads it back; 1° = 8.53 px; W/2/f = 60° half-FOV.

| band | image rows | stride-4 rows | **stride-8 rows** | label rows (10 cm) per s8 row | ground depth per image row | lateral m / px | lateral m / s8 column | 0.15 m line, px wide |
|---|---|---|---|---|---|---|---|---|
| 0–20 m | 172.1 | 43.0 | **21.5** | 9 | 0.14 m | 0.020 | 0.16 | 7.3 |
| 20–40 m | 17.7 | 4.4 | **2.2** | 90 | 1.27 m | 0.061 | 0.49 | 2.4 |
| 40–60 m | 5.9 | 1.5 | **0.74** | 271 | 3.53 m | 0.102 | 0.82 | 1.5 |
| 60–80 m | 3.0 | 0.74 | **0.37** | 542 | 6.91 m | 0.143 | 1.15 | 1.05 |
| 80–100 m | 1.8 | 0.44 | **0.22** | 903 | 11.4 m | 0.184 | 1.47 | 0.81 |

Nearest visible ground: 3.4 m (full frame, h 1.45 m), 4.3 m with the rig-B bottom strip the trainer zeroes (43 rows).

**Against the MEASURED recall collapse** (diagnostics M-b, thr_phat, EVAL): edge R@1 m 0.60 / 0.23 / 0.11 / 0.03 /
0.01 and lane R@1 m 0.82 / 0.50 / 0.35 / 0.23 / 0.16, with edge P@1 m 0.72 / 0.73 / 0.68 / 0.66 / 0.69. From 20–40 m
to 80–100 m the image support falls **10×** (17.7 → 1.8 rows) and edge R@1 m falls **21×**. ⇒ **Beyond ~40 m the 10 cm
map is image-resolution-bound** (consistent with Far3Det's point that fixed tolerances are "too harsh for the
far-field" [LIB:2211.13858]); the SAM3 GT there is a multi-frame / multi-camera product (lane GT fraction is 1.0–1.5 %
in EVERY band, `M_f.json`), i.e. supervision the current frame cannot see at 10 cm. **0–20 m is NOT
resolution-bound** (21.5 stride-8 rows, a lane line is 7 px wide), yet edge R@1 m is only 0.60 and the predicted
drivable boundary sits within 0.2 m of the GT edge on only 37.5 % (diagnostics M-f): **that band is a
representation / target problem.** 20–40 m sits between: 2.2 stride-8 rows, 0.49 m per stride-8 column — a stride-4
tap doubles both.

## 4. What this establishes for the design (pointers into `PERCEPTION_DESIGN.md`)

| finding | evidence class | design consequence |
|---|---|---|
| box heads still improving at LR 0 after 1.08 epochs | MEASURED (§1.2, §2.1) | warm-start continuation with a re-warmed head LR is lever #1 for boxes, at zero params |
| late box gains are relatively largest at 1 m (+55 %) vs 4 m (+13 %); the error is longitudinal (§2.5) | MEASURED | localisation levers (refinement, geometry, depth) rank above recall levers |
| decoder layers refine 4 % | MEASURED | iterative anchor refinement + more supervision per object (hybrid / DN) |
| anchors frozen at their random init | MEASURED | the head DOES need a better (data / per-frame) anchor init; prior bias does not |
| 1.97 / 3.12 boxes per object; top score on the nearest slot 34.5 % / 23.0 %; same-box ORACLE re-rank + NMS doubles AP@1 m | MEASURED (+ ORACLE ceiling) | quality-aware presence + de-duplicating supervision (hybrid, CDN) before any capacity change |
| IGNORE-radius exemption does not breed duplicates | MEASURED (eliminated) | not a lever |
| planner reads the weaker detector | MEASURED | one detector: box3d → planner, zero-gated |
| map plateau; far band image-bound | MEASURED + ANALYTIC | spend 10 cm where the image supports it (0–40 m); stride-4 tap for 20–40 m; target/representation work for 0–20 m; range-adaptive far metric |
| aux (box3d + map + tactical) carries 99.4 % of the trunk gradient norm vs the trajectory loss | MEASURED | no planner rebalancing; a perception-first stage is NOT needed to protect perception's share of the trunk |

## 5. Deviations, limits, UNVERIFIED

* The in-run eval set is 128 fixed windows; its AP curve is noisy for rare classes (class-mean mAP rows) — the
  class-agnostic AP and PROBE-0's 1,061-window curve are the quotable ones.
* Late slopes are linear fits over a window where the LR is falling; they are NOT forecasts and none is extrapolated
  beyond 1.5× the fitted window. No learning-curve exponent is quoted.
* The per-term split of the aux gradient (box vs map vs tactical) is not logged → P-GRAD.
* The geometry table assumes flat ground and zero pitch; per-clip pitch shifts row positions, not band spans.

## 6. For the register (proposed rows — the Master Mind owns `GOALS_AND_CLAIMS.md` / `RETRACTION_LOG.md`)

* **SUPPORTED (open-loop perception diagnostic, one seed):** "refcv7's box heads were schedule-limited, not
  capacity-limited, at 50,400" — AP@2 m still rising +0.023 / 10k at LR → 0 after 1.08 epochs (§1.2, §2.1).
* **SUPPORTED:** "refcv7's 10 cm map is near its plateau and its range fall-off is image-support-bound beyond ~40 m"
  (§1.3, §3; ANALYTIC + MEASURED). ⇒ R8-6's measure should not bar 10 cm IoU beyond 40 m.
* **NEW FINDING (MEASURED):** "the box3d head's learned reference points never moved from their random init"
  (median 0.031 m over 50,400 steps).
* **NEW FINDING (MEASURED):** "the planner consumes the weaker detector" (agent AP@2 m 0.131 vs box3d 0.248).
* **NEW FINDING (MEASURED + ORACLE ceiling):** "refcv7's boxes are better placed than ranked" — top score on the
  nearest slot 34.5 % (box3d); re-ranking each object's own slots + NMS r 2 m: AP@1 m 0.130 → 0.263 (§2.4).
* **ELIMINATED (MEASURED):** "the IGNORE-radius presence exemption breeds duplicates" (§2.2).
* **REFUTED (as a perception lever):** "the planner's gradient starves the perception trunk" — the trajectory loss is
  < 1 % of the trunk gradient norm at every phase (§1.4).
* **STALE ABSENCE (flag for the Lab backlog):** `LAB_BACKLOG.md` P-11 (2026-09-02) says PhysicalAI-AV has no
  LiDAR-projected depth; the corpus ships `lidar_top_360fov` (MEASURED 2026-09-11, `…/2026-09-11-lidar-bev-gt/`).
  The projected depth TARGET is unbuilt — buildable, not absent. (The B13 ledger already noted the unblocking on
  2026-09-13; the backlog row was not revisited.)

