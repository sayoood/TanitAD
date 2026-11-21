# RESULT R1 — head-only retraining on a frozen refcv7 trunk: the cut, the cache, the identity controls (arms: §4)

**Stamp on every number.** Evidence class MEASURED unless marked. Model refcv7-r101-s0 step 50,400 (Thor `ckpt.pt`, strict
load, the launch tree `fec3a0dccf`, byte-identical to `git show fec3a0d`). OPEN-LOOP single-shot planning on logged
frames of held-out eval139 (never closed loop). ONE training seed of the trunk. Nav = ORACLE (ego-future provenance).
SPEC: `SPEC_R1.md`, REGISTERED by the Master Mind, sha256 `e9873e12…`, 2026-10-04T11:15:41Z (`raw/SPEC_SHA256.txt`),
before any arm was trained or scored. Harness: `DESIGN.md`. Raw artifacts: `raw/`.

## 1. The cut point (DESIGN.md §1)

Frozen and cached: the trunk (pooled_seq [8, 2048] and kv = `decoder.feat_proj(stride-32 map)` [416, 384]), the
perception branch (`bev_feats` [96, 120, 64]) and the 300 agent slots. Re-run from the cache, unchanged tree code: the
measurement encoder (nav), the hierarchy hook (nav → z_tac, g_tac, target_latent), the tactical behaviour decoder (nav,
max-speed), agent_embed, ego history + residual prior, the anchored-diffusion decoder with its DDIM sampler (the fan is
nav-conditioned, so it is regenerated, never cached), every selection graft and E9. 20.15 M trainable / 80.3 M frozen.

## 2. Identity controls — ALL PASS (codec v2); I-D after the H0 eval

GPU, sampler seed 0 before every phase, batch 1, 5,528 eval windows (`raw/identity_report_eval.json`,
`raw/record_eval_capture.json`). Tolerances were fixed in DESIGN.md §5 before the GPU run.

| control | what is compared | measured | bar | verdict |
|---|---|---|---|---|
| **I-0** | the capture's FULL forward vs the route package's banked `eval_s0.npz` | 1,112 / 1,112 bank windows joined; fan, s_core, s_e9, p_lat, p_lon max \|Δ\| **0.0**; differing sel_idx / sel_idx_base **0 / 0** | ≤ 1e-5, 0 picks | **PASS** |
| **I-X** | head-only path fed the full-precision cached tensors vs the FULL forward | 5,528 windows: fan, traj, s_core, s_e9, p_lat, p_lon, p_goal, g_tac, z_tac lat/lon all **0.0**; differing E9 / decoder picks **0 / 0** | ≤ 1e-5, 0 picks | **PASS — the cut is exact** |
| **I-S** (codec v2) | head-only path fed the STORED tensors (kv fp16, BEV int8 per channel × row, slots fp16) vs the FULL forward | differing E9 picks **2 / 5,528 (0.04 %)** (their E9 top-2 margins 8e-5 and 1.8e-4; median margin 0.45), decoder picks 3; fan \|Δ\| **p99 0.035 m** (max 0.107); s_e9 p99 0.009; p_lat max 1.1e-3 | picks ≤ 1 %, fan p99 ≤ 0.05 m | **PASS** |
| I-S (codec v1, superseded) | same, BEV int8 per CHANNEL (first 2,304 windows) | 5 picks (0.2 %); fan p99 **0.074 m** | same | **FAILED → codec changed by the pre-stated rule** |
| **I-D** | H0 replayed from DISK by the arms harness vs the capture's in-process STORE outputs | 4,634 scored windows: differing sel / sel_base **0 / 0**, max \|Δ s_e9\| **0.0**, max \|Δ p_lat\| 0.0 (`raw/I_D.json`) | bit-identical | **PASS** |
| **K-NAV** (known value) | the arms' announced time-localised nav on the A5 1,112-window grid vs A6's own window counts | active **117 / 1,112**; on GT-turn grid windows **47 / 107** (A6: 117 / 1,112 and 47 / 107) | equal | **PASS** |

**Codec study** (`raw/codec_study_gpu.json`, `code/r1_codec.py`, GPU, 192 scored windows, each tensor varied alone,
decision rule written before it ran): fan p99 — kv fp16 0.003 m, slots fp16 0.002 m, BEV int8/channel **0.059 m**,
BEV int8/(channel, row) 0.028 m, BEV int16 0.002 m, BEV fp16 0.002 m. Only the per-row int8 BEV fits 15 GB (13.9 GB
for 11,634 stored windows; int16 / fp16 would need 21.4 GB) → codec v2.

**Defects found before any arm (all fixed, all MEASURED):** (1) the disk replay differed from the in-process STORE phase
by s_e9 5.7e-6 because the STORE feed held strided slot views — contiguous feeds read 0.0 (CPU smoke); (2) a codec-study
forward hook RETURNED a value and so REPLACED the strategic module's output (crashed, caught, fixed); (3) the eval loader
returns every tensor frozen, so the trainable set must be re-enabled explicitly (`r1_heads.trainable_head_params`).

**Window sets.** EVAL: 5,528 = every GT-turn window 2,317 (849 L / 1,468 R) + an equal seeded sample 2,317 + the
1,112-window grid (`raw/selection_eval.json`); scored = 4,634. The turn set equals A6's dense capture exactly; the random
half is a different draw (2,575 / 4,634 windows in common), so R1's H0 is not A6's V0 on the sample half. TRAIN: 7,000
windows / 700 clips (1,727 GT-turn) (`raw/train_subset_record.json`). CPU window selection vs the tree's own GT: max
\|Δ\| 7.7e-6 m (eval) / 9.9e-7 m (train).

**Label controls (SPEC_R1 §6 as amended by A1).** The registered control "dense-label coverage ≥ 0.95 of scored windows"
read **0.8677** — exactly the share of scored windows whose 6-s future lies inside the clip (slot 60 valid 0.8677; **0**
IGNORE windows with a valid slot 60): it measured clip-end truncation, not labelling. The Master Mind registered
`SPEC_R1_A1.md` (sha256 `cf3caf0b…`, 2026-10-04T17:27:57Z) **before any arm was read against a bar**: coverage ≥ 0.95 on
EACH classified class. MEASURED: turn-left 1.000 (n 849), turn-right 1.000 (1,468), straight 1.000 (1,385), gentle 1.000
(248) → **PASS**; unclassified 684 windows, coverage 0.104 (reported, not a bar). Label validity (dense lateral side vs
the GT turn class on the 2,317 GT-turn windows) **0.9991 → PASS**. H2s regression labels: agreement with the GT turn
class on TRAIN turn windows **0.479** (< 0.95 required → PASS; unshuffled 0.991; 90.1 % of windows moved).

## 3. Cache (Thor `/home/nvidia/refcv8_r1/cache/`)

Eval 5.65 GB (23 shards: 19 trunk-carrying + 4 grid-only, 5,528 windows, wall 41 min incl. two head-path identity
forwards per window); train 28 shards, 7,000 windows with box3d slots (wall 28 min at batch 8). Everything under
`/home/nvidia/refcv8_r1/`: **14.32 GB** (MEASURED `du -sb`, before the arms). Format: DESIGN.md §3.

**Timing smoke** (`raw/timing_smoke.json`, 30 H1 steps, nothing saved or scored): 0.686 s per step at batch 32, peak
6.35 GiB ⇒ the registration rule gives **1,320 steps per arm** (≈ 15 min of GPU).

**H0 (the paired baseline), scored set** (seed 0 [95 % CI] / seed 1): turn direction-correct **0.815 [0.730, 0.885]** /
0.813; heading within 15° **0.505 [0.432, 0.579]** / 0.510; the fan holds a direction-correct candidate on 1.000; turn
ADE 3.30 m vs oracle-117 1.22 m (random-in-reach 11.17 m). n = 2,317 GT-turn windows in **43** episodes (the route
package's 38 was the 1,112-window grid). Consistent with RESULT.md's grid reading (0.841 / 0.514).

## 4. Registered arms (SPEC_R1 + A1) — NO ARM CLEARS THE R1 GATE; H5 FAILS CONTROLLABILITY

Source `raw/score_registered_arms.json` (`code/r1_score.py`, md5 in LANDING_READY_R1.txt). Every trained arm: 1,320
steps (timing rule), batch 32, lr 5e-5, AdamW wd 0, training seed 0, from the 50,400 weights, frozen trunk. Eval: 4,634
scored eval139 windows (2,317 GT-turn / 43 episodes; 1,385 GT-straight), batch 1, sampler seeds 0 and 1. Paired
episode-cluster bootstrap vs H0 (B 2,000) — it answers "another draw of EPISODES"; the seed-1 column answers "another
INFERENCE draw"; "another TRAINING run" is NOT measured (one training seed per arm; no arm cleared, so SPEC §6's
replicate rule never triggered). Bars: dir ≥ 0.95 with separated gain, heading-15 ≥ 0.70, straight ΔADE ≤ +0.05 m
(CI upper ≤ +0.10) — all on BOTH seeds.

| arm | turn dir-correct s0 [CI] / s1 | gain vs H0 s0 [CI] | heading-15 s0 / s1 | straight ΔADE s0 [CI] / s1 (m) | turn ΔADE s0 [CI] (m) | bar |
|---|---|---|---|---|---|---|
| H0 (no training) | 0.815 [0.730, 0.885] / 0.813 | — | 0.505 / 0.510 | — | — | baseline |
| H0n (announced nav at inference only) | 0.820 / 0.817 | +0.004 [0.000, +0.010] | 0.509 / 0.513 | −0.020 [−0.049, +0.008] / −0.009 | −0.045 [−0.116, 0.000] | FAILED (dir, h15) |
| H1 (old labels) | 0.854 [0.777, 0.917] / 0.853 | +0.038 [+0.023, +0.055] | 0.548 / 0.539 | +0.076 [+0.024, +0.124] / +0.097 | −0.135 [−0.274, −0.003] | FAILED |
| H2 (dense labels) | 0.854 / 0.852 | +0.038 [+0.023, +0.056] | 0.547 / 0.539 | +0.069 [+0.021, +0.114] / +0.112 | −0.099 [−0.234, +0.026] | FAILED |
| H2s (time-shuffled dense) | 0.852 / 0.851 | +0.036 [+0.020, +0.054] | 0.552 / 0.544 | +0.097 [+0.041, +0.149] / +0.117 | −0.112 | FAILED (required) |
| H3 (+ announced nav) | 0.891 [0.816, 0.944] / 0.891 | +0.076 [+0.040, +0.119] | 0.566 / 0.558 | +0.068 [+0.014, +0.123] / +0.080 | **−0.267 [−0.488, −0.068]** | FAILED |
| H4 (+ listwise selector) | 0.905 [0.841, 0.956] / 0.906 | +0.090 [+0.047, +0.141] | 0.548 / 0.551 | +0.159 [+0.068, +0.270] / +0.199 | −0.105 [−0.400, +0.170] | FAILED |
| H5 (+ tactical condition) | **0.909 [0.846, 0.957]** / 0.907 | +0.093 [+0.049, +0.145] | 0.549 / 0.550 | +0.169 [+0.065, +0.305] / +0.179 | −0.114 [−0.401, +0.168] | FAILED |
| H5c (deranged condition) | 0.906 / 0.905 | +0.091 | 0.552 / 0.554 | +0.158 / +0.196 | −0.117 | FAILED |

**H5 controllability (SPEC_R1 §6; forced on the 3,950 classified scored windows; pick follows the forced direction):**
TURN_L 0.274 / 0.273, TURN_R 0.437 / 0.430, LANE_KEEP 0.428 / 0.433 (seed 0 / 1) — **FAILED** (bar 0.95 each).
The control H5c reads the SAME (0.268 / 0.431 / 0.428): **the learned decoder condition contributes nothing measurable**
at w = 1 — what forcing moves (27–44 %) is the forced tactical posterior acting through the existing tac8 selection
prior, present in both arms. Training-side tell: H5's condition weight norm grew 0 → 0.376 (H5c 0.050).

**TACTICAL family (behaviour decoder vs the dense labels, every scored window; z_tac heads in the JSON):** lateral
accuracy / macro-F1 H0 0.470 / 0.452 → H1 0.488 / 0.482 → **H2 0.818 / 0.809 → H3 0.851 / 0.841**; H2s 0.697 / 0.666;
longitudinal accuracy 0.238 → 0.545 (H2) → 0.582 (H3). Support: LANE_KEEP 1,699, TURN_L 850, TURN_R 1,472; HOLD 21,
CREEP 120, ACCELERATE 1,273, BRAKE_TO 1,014, CRUISE 1,593. **LONGITUDINAL / LATERAL point families** (ADE, FDE,
along/cross at 6 s, speed / curvature / yaw-rate MAE 0–2 s, per class) are in the JSON per arm and seed.
LONGITUDINAL distance keeping: UNAVAILABLE (no lead tracks in this harness). STRATEGIC: N/A (OFF, PI ruling R5).

**What the registered arms say (reading rule SPEC_R1 §7: "no arm clears" ⇒ gaps + TRAIN-FIT diagnostic → A2):**
1. **The trunk carries the TACTICAL decision.** Retraining only the heads on frozen features takes held-out lateral
   accuracy against the dense labels from 0.47 to 0.85 (H3) — the turn IS decodable from the frozen trunk.
2. **It does not reach the pick.** H2 (dense labels) moves the pick exactly as much as H1 (old labels) and H2s
   (shuffled labels): +3.8 pp. The pick is not driven by the tactical layer's decision.
3. **The levers that moved the pick:** the time-localised announced nav (H3 vs H2: +3.7 pp, turn ADE −0.27 m) and
   the listwise selector (H4 vs H3: +1.4 pp direction, but straight ADE +0.09 m worse and turn ADE gain lost).
4. **Every TRAINED arm worsens straight driving** (+0.07 to +0.20 m), H1 included: a fine-tune cost, not a label cost.
5. **The gap to the gate:** best turn direction 0.909 vs 0.95; best heading-15 0.566 vs 0.70; no arm meets the
   straight bar. Heading-15 barely moves in any arm (0.505 → ≤ 0.566): the turn MAGNITUDE is the larger miss.
