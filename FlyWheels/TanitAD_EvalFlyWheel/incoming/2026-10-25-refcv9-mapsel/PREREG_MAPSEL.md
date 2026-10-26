# PREREG_MAPSEL -- refcv8 EMA-48,852: select the 149-candidate fan under the PI's three-tier MAP rule, 0 training

*Research Lab agent for the Master Mind, 2026-10-25 ~23:30 Europe/Berlin. Sealed (sha256 in `PREREG_MAPSEL.sha256`) BEFORE any per-candidate map feature, any roll, any arm pick or any
human-trajectory control value exists on the eval windows. Unit tests (`code/test_mapsel_lib.py`) and a PLANTED synthetic world (`smoke/make_synth_bank.py`) were run before the seal; they touch no
eval map data. Literature input folded in: `lit/MAP_IN_PLANNER_DESIGN.md` (Master Mind message 2026-10-25 ~23:20).*

**PI directive (chat, 2026-10-25):** *"the planner should use the map head ... the trajectory must be in the drivable area, should avoid red edges and preferably within the lane."*

## 1. Question and hypotheses

Does selecting refcv8's own 149-candidate fan under that three-tier rule, at 0 training, improve MAP-COMPLIANCE without costing accuracy or progress?
* **H-MAPSEL-GT** (bound): with the TRUE map (SAM3 GT raster) the rule lowers off-drivable, edge-contact and lane-crossing rates. This is the **T0-ORACLE bound for the learned-selector lever**
  (a DrivableCritic / Hydra-MDP++ / DDv2-style selector with map-derived labels); it is NOT a deployable result.
* **H-MAPSEL-OWN** (deployable): the same rule driven by refcv8's OWN 10 cm map head (vision only, zero training) lowers off-drivable separated below BASE, at no separated four-family cost.
* **H-MAPSEL-GAP**: the GT-minus-OWN gap per tier is the size of the perception-quality lever (is P3's thin-class recall worth compute?).

## 2. Setup (all fixed)

| item | value |
|---|---|
| model | refcv8 EMA-48,852, `E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt`, md5 `b4364029b68f29a33eeb4aebbbbd002b`; code tree `C:/Users/Admin/refcv8_pf/gtmap_oracle/tree_fb47a58` (fb47a58) |
| windows | eval139 = 4,754 windows / 139 episodes, `windows_s0.json` md5 `a3a78dba889e70aa69a05271559794ad`, PER-FRAME nav, os-only, inference seed 0, ONE native roll (no hook / force / calibrator) |
| tier | **T1 self-action OPEN LOOP on logged eval139 frames** (not driving performance). GT-* arms are a **T0-ORACLE BOUND** (privileged map). |
| estimator | paired episode-cluster bootstrap over the 139 episodes (`pair_read_r8.paired_cell` -> `taniteval.ci`, B 2000, seed 0). It answers **"another draw of EPISODES" only**; one training seed, one inference seed. A separated CI is NECESSARY, not sufficient (`H-ESTIM-SEED-1`); the L1 BASE-os seed-1-vs-seed-0 difference is printed beside every four-family row as the inference-draw FLOOR. |
| evidence class | MEASURED (artifacts under `result/`, `F:/refcv9_mapsel/`) |
| GT map | `E:/refcv6_eval_kit/data/sam3_gt_v3_eval` (137 of 139 clips have a file; windows of the other two have NO GT map: GT arms = BASE there, excluded from every map-compliance rate), n_stack 3, the dataset item's own `map_fine` |
| populations | S6-compliant (3,668 S6 windows, `E["compliant"]`; the MAP-COMPLIANCE headline = these windows with a GT map) and S2-compliant (4,159 of 4,754) -- the battery's compliant populations |

## 3. Definitions

### 3.1 Frame, rasters, classes
Ego frame at t0, x forward, y LEFT. Raster [1000 rows along x = 0..100 m, 600 cols along y = -30 (RIGHT) .. +30 m], 10 cm cells; cell (i, j) = x in [0.1 i, 0.1 (i+1)), y in [-30 + 0.1 j, ...).
Codes: 0 nocls, 1 drivable, 2 lane (painted lane marking), 3 crosswalk, 4 arrow, 5 edge (the "red edges"), 6 hatched, 7 sidewalk, 255 not seen. **DRIVABLE = {1,2,3,4,6}** (`refcv8_conditioning.DRIVABLE_CODES`:
markings lie ON the road surface). A point outside the raster is UNCHECKED (unknown passes). `255` cells never raise a violation.
* **GT planes**: the code raster of the window.
* **OWN planes** (from `out["perception"]["map_hires_logits"]`, the declared `prior_corrected` rule with the model's own frozen class weights `w`): `s = z - log w`; `p^ = softmax(s)`;
  `pdrv_u8 = round(255 * sum_{c in DRIVABLE} p^_c)`; `pedge_u8 = round(255 * p^_edge)`; `code_u8 = argmax s` (asserted equal to `map_head_hires.decide('prior_corrected', w)`).
  OWN drivable at a point := `pdrv_u8 >= 128` (P >= 0.502; D6-P4's 0.5 gate). OWN edge := `code_u8 == 5`. OWN lane := `code_u8 == 2`.

### 3.2 Path, footprint, per-candidate flags
* **Plan** = 8 REAR-AXLE slots at t = 0.5, 1, 1.5, 2, 3, 4, 5, 6 s. The path is the polyline origin -> slot 0 -> ... -> slot 7, sampled CONTINUOUSLY every **0.25 m** of path length from the 0.5 s slot to the 6 s slot
  (the t in [0.5, 6] s rule; the final slot is always sampled; cap 800 samples). Heading = the segment direction (< 5 cm keeps the previous heading; the first falls back to 0).
* **Footprint** = nuPlan box **5.176 x 2.297 m, centre 1.461 m ahead of the rear axle** (`pdm_proxy.PROXY`, pinned by the unit test); check points **FP5 = 4 corners + centre** (= `drivable_feature`'s set), at every path sample.
* **off_drv(c)** = any FP5 point in range, SEEN, on a non-DRIVABLE cell (GT: code not in DRIVABLE; OWN: `pdrv_u8 < 128`).
* **edge_contact(c)** = any FP5 point in range within **0.3 m** of an edge cell (the edge mask dilated by 3 cells, square). GT: code 5; OWN: decided code 5.
* **lane_cross(c)** = the **centre** point (the centre line) in range inside the lane mask dilated by **2 cells (+-0.2 m, a 0.5 m band >= the 0.25 m step)**. GT: code 2; OWN: decided code 2.
* **Lane-change exception (T3 only)**: a window is a lane-change window iff its v9 row (clip, raw frame = ws + 2) has a finite `lc_t_cross_s` in (0, 6] s (48 of 4,754 windows, MEASURED). The row join is validated against the FED nav
  (`r8_nav_fed[:,0]` vs `log1p(nav_d_next_m/50)`; 11 of 12 checkable clips agree at 100 %; the 12th disagrees at every offset = a nav-feed difference); if < 80 % of the checked clips agree the exception is DISABLED and stated.
  The exception is **label-derived** (v9, GT-side) in every arm, so OWN-T123 carries an oracle exception (flagged); the headline deployable arms are T1-based.
* **E9 eligibility** = the exact replica of `refc_v3.e9_rank`'s finite mask (reach / ceil / emit masks with the dead-row fallbacks); every arm selects ONLY among E9-eligible candidates and ranks by the model's own `sel_score_v3`.
* **Shipped stage**: after the pick, `retime_stage.retime_plan(plan8, banked prior8, ALL_T, DEPLOY8 table, cast=True)` exactly as SHIP-R8 does; every compliance metric is measured on the **EMITTED (staged) 8-slot plan**.

### 3.3 Arms (all share the SAME fan, scores, eligibility and stage)
Tiers are lexicographic filters over the eligible set, each **skipped (fallback counted) if it would empty the set**; the pick is then `argmax sel_score_v3` over what is kept.
* `T1` keep not-off_drv; `T12` then keep not-edge_contact; `T123` then keep not-lane_cross (skipped in lane-change windows).

| arm | source of the map | tiers |
|---|---|---|
| BASE | -- | native pick (**must equal the banked emitted plan exactly**) |
| GT-T1, GT-T12, GT-T123 | GT raster (T0-ORACLE bound) | 1 / 1-2 / 1-3 |
| OWN-T1, OWN-T12, OWN-T123 | the model's OWN planes (deployable, zero training; T123 carries the oracle lane-change exception) | 1 / 1-2 / 1-3 |
| **OWN-CAL-T1** | OWN `pmin` score (below) | 1 |
| DER-T1, DER-T12, DER-T123 | OWN planes of ANOTHER episode (cyclic next episode by clip index, position-matched window: `min(p, n_donor-1)`) -- the deranged control | 1 / 1-2 / 1-3 |
| DER-CAL-T1 | the deranged `pmin` | 1 |
| HUMAN (reference, not an arm) | the logged future path at the 8 slots, ego frame at t0 | -- |

**CALIBRATED arm (`OWN-CAL-T1`)** -- per the Master Mind's literature input: reject a candidate if its own-map score `pmin(c)` = the MINIMUM over its in-range FP5 samples of `pdrv_u8 / 255` is **below tau**, unless none survive.
**tau is fixed a priori from the HUMAN paths only**: `tau = quantile_0.01(method='lower')` of `pmin(human path)` over the S6-valid human windows (own raster of the same window), so that **>= 99 % of human trajectories pass**.
It is computed once, before any candidate is scored, from privileged GT trajectories and the model's own map, with **no outcome metric**. **It is calibrated ON THE EVAL human paths; a train-side calibration is not available in this roll** --
the arm is therefore labelled *privileged-calibrated* and is a co-headline only with that label. The hard rule's plan-level false-alarm rate on human paths (share of human paths with `pmin < 0.502`) is reported beside it.

### 3.4 Predictions committed BEFORE the roll (not bars)
Own-map IoU on this head (perc8 RESULT sec. 3, MEASURED, 4,686 windows): drivable **0.605 / 0.613 / 0.548** (0-20 / 20-40 / 40-60 m), lane **0.199 / 0.060 / 0.026**, **edge 0.001 (effectively empty)**.
=> (a) **OWN-T12 and OWN-T123 are PREDICTED near-identical to OWN-T1** (< 2 % of windows pick differently); tiers 2-3 are reachable only with GT. (b) A hard filter on a 0.6-IoU drivable mask is expected to be limited by
plan-level false alarms (D6-P4: per-point recall ~0.95 compounds over 40 points), which `OWN-CAL-T1` addresses. (c) Literature: no reference planner hard-filters on its OWN predicted map (only PDM-Closed, with the GT map); the
strongest published effect is a learned selector with map-derived labels (Hydra-MDP++ DAC 92 -> 96, DDv2 96.2 -> 97.0) -- **GT-T1 is our bound for that lever** and is reported as such.
(d) DER caveat: on D6/navhard the fan's pick was cleaner (0.065) than a random fan member (0.320) ... in the SAME direction a pick change alone can improve compliance; C3 below is therefore a conservative gate and its failure is
reported as INCONCLUSIVE-DER, not FAIL.

## 4. Metrics (the four families are ADDED, ADE is not alone)

1. **MAP-COMPLIANCE (new family, vs the GT map; per-window 0/1 of the EMITTED staged plan, then the episode-cluster mean):** off-drivable rate; edge-contact rate; lane-marking crossing rate **outside lane-change windows**.
   Population: S6-compliant with a GT map (headline) and S2-compliant with a GT map. Paired delta vs BASE with CI.
2. **STOP-trap guard (literature: on NAVSIM a STOP plan scores DAC 96.5; a compliance filter can win by picking slow/short candidates):** the emitted plan's path length to 6 s and the share of plans whose 6 s displacement is < 1 m,
   per arm and vs BASE and vs HUMAN; plus the longitudinal progress/speed four-family rows (`LON_along_mae_m`, `LON_speed_mae_mps`, `LON_along_abs_final_m`, `LON_speed_within_0p5`, `LON_accel_mae_mps2`,
   signed along-track at 6 s and signed speed error on GT-cruise, false-/missed-decel).
3. **Four families via the battery code** (ADE/FDE, LONGITUDINAL incl. speed & along-track, LATERAL incl. heading / curvature / yaw-rate / cross-track, TACTICAL traj-lat/lon correct) on **S6 (+stage)** and **S2 (+stage)**, compliant windows,
   `l2_score.arm_cells` / `l2_oracle_s2_rows.cells`, paired vs BASE, with the inference-draw FLOOR beside. Not computed (stated, n = 0): distance-keeping and STRATEGIC (no lead track in the banked dumps / `--no-strategic`).

## 5. Controls (a failing hard control makes the run VOID; nothing is read)

* **K_HUMAN (0 GPU, run BEFORE the roll; STOP the experiment if it fails):** the HUMAN path scored by the SAME footprint / sampler / GT raster must read **off-drivable <= 2.0 %** on S6-compliant GT windows, and its path must equal the banked S6 `g`
  (parity < 1e-3 m). > 2 % means the footprint, raster or frame geometry is wrong. The chain computes it before taking the GPU lock and aborts without a GPU if it fails.
* K_G (fan 149, 8 slots; `out["perception"]["map_hires_logits"]` shape [1,8,1000,600]; declared rule `prior_corrected`; decide-parity mismatch cells = 0; n_stack 3; batch size 1), K_B (emitted = fan[sel]), K_C (sel_idx = argmax over the E9-eligible set),
  K_A (the roll's plan8 is **bit-equal** to L1's un-hooked BASE-os bank on every window: attaching the GT map changed nothing), K_RAST (a raster for every window), K_BASE (the BASE arm reproduces the emitted pick on every window),
  K_STAGE (staged BASE == banked S2 stage, bit-equal), K_S6_alignment, K_DERANGED (every donor is another episode), TAU_CAL (>= 99 % of human paths pass at tau). K_LC_JOIN is NOT a VOID control: if it fails the lane-change exception is DISABLED and the run says so.
* K_LIVE: the own drivable decision agrees with the GT drivable on SEEN cells better than the deranged donor's by > 0.02 (the map is informative about position).
* K_GTLIVE: GT-T1 lowers the off-drivable rate separated below BASE (else the instrument is broken).
* Unit tests with ANALYTIC literal targets and two deliberate-regression arms (`test_mapsel_lib.py`: wrong box-centre offset and an 8 m path step must be detected); the roll's capture is unit-tested against PLANTED logits (prior-correction detected).

## 6. Bars, verdict, next lever (both outcomes committed)

### 6.1 Clauses for an OWN arm X (on S6-compliant windows with a GT map unless stated)
* **C1** off-drivable rate of X separated BELOW BASE (paired CI upper bound < 0).
* **C2** NO four-family row (S6c and S2c, +stage) separated WORSE than BASE (CI excludes 0 on the bad side).
* **C3** the DERANGED control does not beat BASE: neither the tier-matched DER arm nor DER-T123 is separated below BASE on off-drivable.
* **C4** STOP guard: no progress/speed row listed in sec. 4.2 separated worse; the path length not separated SHORTER than BASE; the stop share not separated HIGHER than BASE.
* **PASS(X) = C1 and C2 and C3 and C4.**

### 6.2 Verdict word (mechanical)
* **VOID** if a hard control (sec. 5) fails.
* **PASS (arm list)** if OWN-T1 or OWN-CAL-T1 (the two headline deployable arms; OWN-CAL-T1 labelled privileged-calibrated) passes; OWN-T12 / OWN-T123 passing alone is **PASS-SECONDARY** (4 pre-declared OWN arms, multiplicity noted).
* **INCONCLUSIVE-DER** if some OWN arm clears C1, C2, C4 but the deranged control also beats BASE (the pick change alone improves compliance; OWN-minus-DER is then the evidence and is printed).
* **FAIL** otherwise. A FAIL is a waypoint, not a deliverable (sec. 6.3).
* The GT-vs-OWN gap per tier (BASE, GT, OWN, DER rates; `own_captures_share_of_gt_gain`) is reported for every tier regardless of the word.

### 6.3 Next lever (Rule Zero; mechanical template in `mapsel_score.next_lever`)
* PASS -> register as an opt-in inference row (T1 stamp); train the DrivableCritic on GT-map labels (GT-T1 = its bound; GT-minus-OWN = its headroom); re-read on navtest.
* GT-T1 fails C1/C2/C4 -> the map channel is not the bottleneck: a learned listwise selector over the same fan + a fan-availability audit.
* GT-T1 clears C1/C2/C4 and OWN captures < 50 % of its gain -> PERCEPTION is the bottleneck: learned critic on own-map features with GT labels + P3 thin-class / drivable false-alarm work.
* GT-T1 clears them and OWN captures >= 50 % but no OWN arm passed -> fix the failing clause (stop-gaming / accuracy regression / selector problem).

## 7. Caveats stated up front
* The footprint is nuPlan's box in the rear-axle convention (the trained critic's); the PhysicalAI vehicle's own dimensions differ; K_HUMAN is the check that the geometry is usable.
* tau is calibrated on the eval human paths (privileged). The lane-change exception is label-derived. GT-* are oracle bounds. The rule reads lane markings as a THIN class (own lane IoU 0.2) and edge as an empty class.
* One training seed and one inference seed: the CI answers "another draw of episodes" only. The planner is a stochastic sampler; a gain below the inference-seed floor is not an effect.
* Open loop (T1): compliance of the logged-frame plan, not driving performance. No leaderboard claim.

## 8. Compute
ONE dev-box roll (4,754 windows + a 9-clip smoke) on the shared `gpu_lock` (queued behind the probe-gate, L2-B and L1C chains), ESTIMATED 2,200-3,600 s (L2 / L1 logged rolls; this roll adds one softmax over [8,1000,600] and ~1 MB of raster writes per window);
the chain projects the total after 150 windows and stops (exit 7) above 5,400 s. Scoring is CPU (~20 min). Disk: rasters + trace on `F:/refcv9_mapsel/`, text on C: (< 200 MB).

## 9. Pins
Code under `code/` (sha256 list written to `CODE_SHA256_AT_START.txt` by the chain and `CODE_SHA256_AT_SEAL.txt` at the seal): `mapsel_lib.py`, `mapsel_data.py`, `mapsel_roll.py`, `mapsel_score.py`, `mapsel_human_control.py`, `test_mapsel_lib.py`, `chain_mapsel.sh`,
`refcv8_eval_remap_devbox_mapsel.json` (= the P4 / L2 dev-box remap, byte-identical). Imported read-only: `refcv9_l2/code/{l2_common,l2_score,l2_oracle_s2_rows}.py`, the battery `pair_read_r8.py`, `taniteval.ci`.
