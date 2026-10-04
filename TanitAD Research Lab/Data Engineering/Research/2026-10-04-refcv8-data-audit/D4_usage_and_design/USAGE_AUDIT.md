# D4 — USAGE AUDIT: how refcv7-r101-s0 consumed its data ("are we using the data wrongly?")

Stream D4 of the refcv8 data audit · 2026-10-04 · revision 2: D2's label-validity findings and the route package's A5
folded in, on the Master Mind's instruction. Nothing in D2 or A5 was re-measured.

## Stamps on every number below, unless a line says otherwise

**Evidence classes.**
* `MEASURED (D4)`: computed here. Sources: `raw/d4_measure.json`, `raw/d4_nav_announced.json`,
  `raw/d4_vmax_nonoracle.json`, `raw/d4_lanechange.json`, `raw/loss_shares.json`, `raw/controls.json`; code in
  `code/`.
* `MEASURED (D2)`: `../D2_label_validity/RESULT.md` and `SEMANTICS.md`.
* `MEASURED (A&I)`: the route-following `RESULT.md`, its `RESULT_A5.md`, or the map/box RESULT of 2026-10-04.
* `INHERITED`: another package, not re-run here.
* `UNVERIFIED`: marked inline.

**Code.** Launch tree `fec3a0d`. I read the copies in `C:/Users/Admin/ev7` only after checking their blobs against
`git rev-parse fec3a0d:<path>` in `C:/Users/Admin/tanitad-push`. Every check read VERIFIED (40-char blobs on both
sides): `refc_v3_train.py`, `v7_labels.py`, `clip_clock.py`, `refcv6_max_speed.py`, `refc_v3.py`,
`refcv6_tactical.py`, `nav_conditioning.py`, `refc_tactical.py`, `refcv6_selection.py`, `refc_select.py`.
`refb_labels.py` is under `stack/scripts/` and was read but not blob-verified.

**Run record.** `D:/refcv7_eval_kit/ckpt/config.json` and `thor_reads/metrics_final_50400.jsonl`
(md5 `cfcbba41…`).

**Data.** Labels `s2_labels_v8_train.jsonl.gz` (md5 `b45377a1…`) and `…_eval.jsonl.gz` (md5 `eefc38d1…`). The v2ep
manifests of the exact caches refcv7 read:
* train, copied from Thor, md5 `3c9f8bc8…`, identical to D2's copy: 4,369 clips, 746,946 windows;
* eval139, md5 `33433232…`: 139 clips, 23,772 windows.

The clock is the trainer's own `--clip-clock-sidecar` (md5 `78466f99…`).

**Tier.** These are label and data-usage measurements. No model ran and nothing here is a driving number.

**Split.** TRAIN unless the line says EVAL. EVAL-DIAG is the route RESULT's 1,112-window grid, reproduced
bit-exactly here (digest `92e36a1a…`).

## 0. Headline — the WRONG / SUBOPTIMAL rows, ranked by measured consequence

### 1. Nav input — `--nav-from-v7`, one ego-future token per CLIP on all ~171 windows — **WRONG** (time-localisation)

**Timing (D4, D2).**
* The token names a turn that starts a median 7.3 s after the anchor (D2).
* Only **24.7 %** of TURN-token windows see a same-side turn begin in their own next 6 s (D2).
* 55.2 % of the classified windows of L/R clips are GT-straight (D4: 112,948/204,470).
* On EVAL, 193 of 291 L/R-nav windows are straight (A&I).

**The graft learned to ignore it.**
* The nav-compliance predicate is right on only **35.7 %** of informative windows (D4).
* The gate learned 0.163, and removing the term changes 1–2 picks of 291 (A&I).

**Time-localising fixes the damage (A5).**
* The straight-window cost of the nav filter drops from +0.326 to **+0.009 m**.
* The soft time-localised term (T3) clears its bar.
* The hard rule (T2) fails replication.

**Builder defects in the source (D2).**
* `nav_30s` never reads the suppression flag (`suppressed` vs `applied`; 66 records).
* 68 tokens name a turn more than 30 s ahead and have no `nav_30s` entry.
* 76 clips hide a junction turn behind a leading curve.

### 2. Tactical lat/lon (+ the 22 goal tokens), from one anchor record, only where |NOW − t0| ≤ 2 s — **WRONG**

**Coverage.**
* Only **23.4 %** of training windows are supervised (D2: 173,409/741,483; metrics: 4.0 of 16 rows per batch).
* On EVAL-DIAG GT-turn windows the label is IGNORE on 71–72/107.

**Definitions (D2).**
* **NUDGE is semantically wrong:** corroborated on only 10.3 % / 12.1 %; in 89.7 % the heading never returns; only
  21 of 1,028 records are real nudges. NUDGE is **23.5 % of the lateral mass**.
* The longitudinal label is computed over [0, 6] s, not the declared [2, 6]. ACCELERATE uses an effective
  **+1.0 m/s** bar instead of the documented +1.5.

**As the trainer applies it, longitudinal accuracy decays with distance from the anchor (D2).**
* Across the ±2 s band it falls **80.5 → 63.8 %**, while lateral stays flat at ~71 %.

**The labels contradict the nav input.**
* Records with `NAV_TURN_L` carry lat `LANE_KEEP` on 398/811; for `NAV_TURN_R` it is 446/864 (D4).
* The head therefore learns that nav does not predict lat. It is side-correct on only 0.393 of GT turns (A&I).

**Loss weight.**
* All v7-label tactical terms together are **0.29 %** of the loss value, and the CE is unweighted (D4).

### 3. Max-speed input — `--max-speed-input-v6`, the anchor record's `SPEED_BAND.v_hi`, one per CLIP, 4-way — **WRONG** (oracle in-band, stale out of band, deploy mismatch)

**Oracle in-band, stale elsewhere.**
* Inside the band (23.4 % of windows) the fed value equals the window's own future max over [NOW+2, NOW+6] at
  **R² 0.988**, which makes it a future-speed oracle (D2).
* Over all windows, the current speed v_now predicts that future max **better** than the fed value does
  (R² 0.899 vs 0.889) (D2).
* The value is stale on 17.8 % of windows (D4).

**Leak.**
* It recovers 23.8 % of the future information beyond v0 overall, 34.1 % before the band, and **44.5 %** at the
  anchor (D4).

**Violations.**
* The human's own speed exceeds the fed limit on 11.9 % of windows (D2, 100 Hz) / 13.4 % (D4, 10 Hz, [NOW, NOW+6]).
* So enforcing the ceiling (§26.1) costs **+0.086 [+0.018, +0.158] m** ADE (A&I).

**Ladder and semantics.**
* The 4-way ladder lumps 50–100 km/h into one input (21.7 % of clips).
* A slow ego reads as a low limit: 74.0 % of intersection clips get ≤ 30 km/h, 69.8 % even when moving (D2).

**At deployment** (INHERITED, NavSim).
* 45 % of navtest tokens have no map limit: the all-zero row, never trained.
* Where a limit exists, its bin is above the oracle bin on 52 %.

### Rows 4–9

| rank | channel | verdict | measured consequence |
|---|---|---|---|
| 4 | **residual prior** `ha0_ext_pose` (past-only, from a 1-step finite difference) | SUBOPTIMAL | **14 of 17** wrong-direction turn picks follow the prior's side. The prior is direction-correct on only 57 % of turn windows (A&I, post-hoc). |
| 5 | **22-token goal negatives** (`measured`, no cot-absence sidecar) | SUBOPTIMAL (PI ruling not applied) | 5/22 tokens are masked. RED gets negatives only by entailment: 403 negatives vs 3,793 ignored records. The PI's 2026-09-16 ruling is not in the run: the only sidecar is bound to blob `fa89ea55…`, not `b45377a1…`. EVADE_IN_CORRIDOR and the LANE_CHANGE goal tokens require a NUDGE (D2 SEMANTICS §8), so they inherit NUDGE's defect. Goal tokens are side-correct on only 0.374 of GT turns (A&I). |
| 6 | **ego dropout 0.5** | SUBOPTIMAL — UNVERIFIED | Half of training runs a no-v0 regime that deployment never uses. The speed profile is the dominant oracle-gap term (−0.606 m, A&I B3); dropout's share of it is not isolated. |
| 7 | **box/agent store scope** x ≤ 61 m, \|y\| ≤ 17 m | SUBOPTIMAL for distance keeping — UNVERIFIED | 414 of the 491 joined agents per batch (84 %) fall outside the scope, so a lead vehicle beyond 61 m is never a target. The distance-keeping family is UNAVAILABLE in the harness. |
| 8 | **10 cm map** thin classes | SUBOPTIMAL (recipe F3); the decision rule is an inference issue (F1) | Edge IoU is 0.042 exact vs 0.113 at 0.2 m tolerance, with a 20× near-to-far drop (A&I). The target itself is registered (94 % of GT-edge cells lie within 0.2 m of the GT drivable boundary). |
| 9 | **truncated 6 s futures** | CORRECT masking, minor | 22.8 % of windows lack a full 6 s future (170,391/746,946). The trajectory, `a_star`, E9 and E8 terms are all masked. |

**Settled, and not a usage defect:** `lat_peak_m` is mis-NAMED, not broken. It is the peak lateral displacement over
the first 20 s, in the clip-start heading frame, and refcv7 never read it (D2 §1).

**Rows judged CORRECT:** frames, ego history, v0, clock, the E8/E9/trajectory/cascade targets, the kin3 2 s core aux,
the content of the agent and box3d targets, the content of the map target, and the anchors (§2).

## 1. Method and controls

Every control must read a known value, and all PASS.

| control | known value | measured |
|---|---|---|
| EVAL-DIAG grid rebuilt from the eval manifest | digest `92e36a1a…` (A&I C4) | `92e36a1a74b8b699` — **PASS** |
| GT 6 s class on that grid (route SPEC §3 literals, re-typed rather than imported) | 40 L / 67 R / 588 S / 105 gentle / 312 unclassified | **identical** |
| per-clip nav side correct on the 107 GT-turn windows | 0.439 | 47/107 |
| A5's nav_tl (all entries, H = 6 s) on the same grid | GT-turn L/F/R 12/55/40; straight 4/572/12 (RESULT_A5 §2) | **identical** (`raw/d4_nav_announced.json`) |
| `lat_v7` IGNORE on GT-turn windows | 72/107 | 71/107 (the 3 eval clips without a measured clock are tactical-excluded by the trainer; I clocked them on the fallback) |
| anchor-level leak of the label file's own `speed_max_input` | R² 0.8789 / 0.9702 / 75.4 % recovered | 0.8735 / 0.9688 / **75.3 %** (4,369 clips) |
| `v_hi` recomputed from manifest poses at the anchor row | equals the label | within 0.2 m/s on 4,363/4,369 (D2 independently: 0.005 m/s median) |
| leak test: shuffled candidate / the target itself | ≈ 0 / 1 | −0.0005 / 1.0000 |
| analytic tracks (straight, R 15 m turns, 3.5 m lane change, 1.2 m nudge, stop, accelerate) | literal classes | 7/7 |
| NavSim-equivalent 20 m / 2 m rule on an R 15 m left turn | first LEFT at NOW 7.479 s | 7.4 s |
| **mutations (must go RED):** horizon moved 6 s earlier; nav arc origin at the clip start | lose the TURN; shift the trigger | RED; RED (6.5 → 14.6 s) |

## 2. The channel table (walks `config.json` argv)

**Legend.**
* **scope:** C = one value per clip, W = per window.
* **oracle@inf:** information the deployed car would not have.
* **share:** weighted loss value / total loss, mean over 109 rows at step ≥ 45,000 (`raw/loss_shares.json`). A
  loss-value share, not a gradient share.

### 2a. Inputs

| # | channel (flag) | file → loader → tensor → consumer | scope | windows carrying it | oracle@inf | train / deploy | verdict |
|---|---|---|---|---|---|---|---|
| I1 | frames (`--v2-cache`, 416×1024 cylindrical, window 8 × 3-frame stacks, `--equalize-bottom-rows 43`) | v2ep → `V3Dataset` → trunk | W | all 746,946 | no | NavSim stitch; camera ~0.57 m higher than the training median (INHERITED) | CORRECT (D3 audits plausibility) |
| I2 | ego history (`--ego-history`) | `ep.poses[t:t+w]` (`refc_v3_train.py:3827-3828`) → GRU | W, past | all | no | NavSim 2 Hz, interpolated | CORRECT |
| I3 | v0 / `ego_state`, `ego_dropout 0.5` | one Bernoulli keep per row, shared by goal / core / scene (`refc_v3.py:2090-2096`) | W | kept on ~50 % | no (PI 2026-09-02) | deployment always has v0 | v0 CORRECT; dropout SUBOPTIMAL / UNVERIFIED |
| I4 | **nav token** (`--nav-from-v7`) | `oracle_nav` → `enable_nav_from_v7` (`:2737-2812`, *"Per-clip constant … window-independent"*) → `item["nav_cmd"]` (`:3880-3888`) → 4 consumers: core one-hot (`refc.py:4548`), `nav_inj` → z_tac/ctx (`refc_v3.py:1053-1062`), v6 decoder condition (`:1766-1776`), compliance graft (`refc.py:3559`) | **C** | all; L/R clips 37.0 % of windows | **yes** (ego-future) | NavSim feeds a **per-frame** `driving_command` = f(pose NOW, route, map), firing at 20 m ahead / \|y\| ≥ 2 m on the route centreline, so it also fires on curves (INHERITED) | **WRONG** |
| I5 | nav args (`--nav-args`) | not fed. The existing reader is **per-clip and anchor-relative** (`:2779-2783`) | C | 0 | yes | — | not used; **WRONG if switched on as-is** |
| I6 | **max speed** (`--max-speed-input-v6`) | sidecar → `item["v_max_ms"]` (`:3914-3918`) → `MaxSpeedOneHotEncoder` {30, 50, 100, 120} → v6 decoder condition + the decoder ceiling filter (`refc_v3.py:1777-1791`; E9 bypasses it, §26.1) | **C** | all | **yes**: realised max over [t0+2, t0+6] | NavSim: map limit on 54.8 % of tokens, all-zero (untrained) otherwise | **WRONG** |
| I7 | residual prior (`ha0_ext_pose`) | `a0 = Δv/dt`, `κ0 = clamp(ω0/max(v0, 2), ±0.30)`; zeroed when `ego_keep` is False | W, past | ~50 % | no | NavSim reproduces it bit-exactly (KPR) | SUBOPTIMAL |
| I8 | anchors (117, v0-conditioned, `alat`) | units declared in the file | — | — | no | same | CORRECT |
| I9 | perception → planner (`map_hires_pool`, soft agent presence) | encoder features | W | all | no | same | CORRECT |
| I10 | clock (`--clip-clock-sidecar`) | `t_now = g0 + (t+w−1+2)·dt` (`:3753-3759`) | W | 4,347/4,369 from the sidecar | — | — | CORRECT (G3 worst error 0.0 s; D2: speed reproduced to 0.0017 m/s) |

### 2b. Targets

| # | target | source → mask | scope | rows supervised | weight → share | verdict |
|---|---|---|---|---|---|---|
| T1 | trajectory L1 (winner anchor) + focal anchor CE + cascade (`:4539-4570`) | future poses, masked by `future_valid_ext`; `a_star` on valid slots (`:4552-4555`) | W | all (22.8 % truncated) | 1 / 1 / 1 → 1.7 % / 0.0 % / 5.3 % | CORRECT |
| T2 | E9 selection CE vs `fan_err` (`:4755-4760`) | masked | W | all | 1.0 → 5.0 % | CORRECT |
| T3 | E8 goal regression at τ 2/4/6 s (`refb_labels.py:1628-1662`) | future poses; valid = False past the clip end | W | all | 0.5 → **7.4 %** | CORRECT |
| T4 | core kin3 lat/lon CE (`:4592-4602`; `refc_tactical.py:148-155`: \|Δyaw\| ≥ 0.15 rad, ±1 m/s over 2 s) | the window's own 2 s future | W, **dense** | all | 0.025 × 2 → 0.06 % | CORRECT. The precedent for the dense refcv8 labels. |
| T5 | z_tac lat/lon CE on v7 labels (`:4610-4644`) | `tactical_class_ids` → −100 outside ±2 s (`v7_labels.py:678-710`) | C on in-band W | **23.4 %** | 0.025 each → 0.13 % | **WRONG** (row 2) |
| T6 | v6 behaviour decoder: 22-token BCE + lat/lon CE + confidence (`:5239-5279`; `refcv6_tactical.py:700-848`) | same band; **no class weights** | C on in-band W | 23.4 % | budget 0.1 → 0.15 % | **WRONG** (rows 2, 5) |
| T7 | 22-token negatives | geometry tokens + entailment (`v7_labels.py:1025-1035`) | C | — | inside T6 | SUBOPTIMAL (row 5) |
| T8 / T9 | agent / box3d set losses (VIS-1, 2 m IGNORE radius) | join / join3d | W | 48 positives, 29 IGNORE, 414 out of scope per batch | 1.0 / 1.0 → 37.4 % / 40.3 % | targets CORRECT; scope SUBOPTIMAL |
| T10 | 10 cm SAM3 map CE with class weights | SAM3 GT | W | supervised cells | 1.0 → 2.5 % | CORRECT; thin-class resolution SUBOPTIMAL |
| T11 | route aux / g_str / LAN | OFF (`--no-strategic`) | — | 0 | 0 | CORRECT for this arm; the STRATEGIC family is UNAVAILABLE |

**Loss composition** (D4, total 36.12): box3d 40.3 % · agent 37.4 % · goal_tac 7.4 % · cascade 5.3 % · sel_v3 5.0 % ·
map 2.5 % · traj 1.7 % · **all v7-label tactical terms 0.29 %**. The tactical budget is pinned at 0.1 by spec
(`refcv6_tactical.py:703-707`), so raising it is a PI / MM decision.

## 3. Detail of the three WRONG rows

### Row 1 — nav

**What I measured on TRAIN (D4).**
* Of the classified windows of L/R clips:
  * 55.2 % are GT-straight;
  * 24.2 % turn to the commanded side.
* On follow clips, 7.6 % of windows are GT turns.

**Why the graft cannot learn.** The compliance predicate (`refcv6_selection.py:79-107`) is right on 35.7 % of
informative windows. Per-window tokens fix this (`raw/d4_nav_announced.json`):

| token | predicate right, informative windows | informative windows / all classified |
|---|---|---|
| announced, time-localised (refcv8 design) | **80.0 %** (64,704/80,909) | 14.7 % |
| A5's all-entry time-localised | 80.3 % | 15.8 % |

**The NavSim parity trap.** A token computed from the ego's own driven path at 20 m / 2 m is the closest proxy of
NavSim's command. It fires on **6.7 %** of all windows (50,349) where nothing is announced: curves, lane changes,
obstacle passes. On TRAIN, those manoeuvres are the ego's own future decisions, so this path rule is not admissible
as the training token. The time-localised design must use announced entries only (A6; see the design §1).

**Builder defects (D2 SEMANTICS rows 6–8).** These decide what "announced" can be built from:
* the `suppressed` / `applied` key mismatch;
* the 30 s cap (68 tokens beyond it);
* the `seq[0]`-only rule, which hides 76 junction turns behind a curve.

**A5 attribution.** 54 % of T2's turn gain sits on 7 windows where `nav_command` had suppressed the turn, i.e.
information a real nav would not give. Nav is silent on 55/107 GT-turn windows, and the other half needs a vision
cue: the dense tactical label.

### Row 2 — tactical labels

**What D2 measured.**
* Lateral, per class: TURN 87.0 / 85.4 %, LANE_KEEP 91.6 %, NUDGE 10.3 / 12.1 %. Overall 72.1 % (train).
* Longitudinal: 80.3 % on the declared [2, 6] window vs 89.5 % on the [0, 6] window the builder actually used.
* ADAPT_SPEED_FOR_CURVE hides the speed class underneath it.
* As applied per window, longitudinal accuracy decays 80.5 → 63.8 % across the band.

**What D4 adds.**
* The tactical head is CONDITIONED on the per-clip nav one-hot, yet the in-band label says LANE_KEEP on half of
  the L/R records.
* The CE is unweighted while LANE_KEEP is 64.7 % of lat labels.
* The terms are 0.29 % of the loss value.

**Expected change from the dense design.** Read on in-band GT-turn windows: label side-correct 65.4 % → 85.0 %
(24,679/29,047).

### Row 3 — max speed

* **Leak test (D4).** 5-fold clip-grouped OOF; the target is the window's own max over [NOW+2, NOW+6];
  n = 576,555 windows. v0 alone gives R² 0.8926; v0 plus the fed 4-way bin gives 0.9181, i.e. **23.8 %**
  recovered.
* **Oracle where it is fresh (D2).** In-band, the fed value IS the window's future max (R² 0.988).
* **A per-window version would be an oracle everywhere.** The containing step of [NOW, NOW+6] recovers
  **57.0 %**; the [NOW+2, NOW+6] variant recovers 71.7 %. The non-oracle options are in the design §2.

## 4. Slots for D1 / D2

**D1.**
* The exact per-token window coverage.
* The share of L/R-clip windows inside their turn interval. D2's 24.7 % is the TURN-token version of this.

**D2: answered.**
* `lat_peak_m`;
* class validity;
* the longitudinal window;
* the speed oracle.

**Still open with D2.** The in-band override rule (design §3) needs D2's per-window conflict count between the
record class and the dense TURN / not-TURN decision.

## 5. Register proposals (the Master Mind owns `GOALS_AND_CLAIMS.md`)

* **MEASURED.**
  * refcv7's nav input is per-clip and wrong for 55.2 % of the classified L/R-clip windows.
  * The compliance predicate is right on 35.7 % of informative windows with that token, and on 80.0 % with an
    announced time-localised one.
* **MEASURED.**
  * Per-window realised-max ceilings are future oracles: 57.0 % (8-step) / 71.7 % ([+2, +6]) of the missing future
    information recovered.
  * The refcv7 per-clip value recovers 23.8 % (44.5 % at the anchor).
  * Non-oracle past-only proxies recover 1.9–3.5 %.
* **MEASURED.**
  * Records with an L/R nav token carry LANE_KEEP on 49–52 %.
  * The v7-label tactical terms are 0.29 % of the loss value.
* **NOT APPLIED.** The PI's 2026-09-16 negatives ruling: the sidecar is bound to blob `fa89ea55…`.
