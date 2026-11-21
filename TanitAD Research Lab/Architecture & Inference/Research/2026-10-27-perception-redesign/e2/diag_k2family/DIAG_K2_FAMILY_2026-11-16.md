# Why the K2 family fails E2: root-cause diagnosis, no training (Opus agent for MM 4a3d05, 2026-11-16)

Brief: `dagland/r9_briefs/BRIEF_K2_FAMILY_DIAG_2026-11-16.md`. Read-only. No GPU used. No Thor process started.
Scratch, scripts and derived tables: `F:/refcv9_e3/k2diag/`. Tip read: mirror `581715b4`.

**Headline.** The pre-registered E2 read takes **`model_ema.pt` at 6,000 steps with EMA decay 0.9998**. Under that read,
**every module K2 builds from scratch is in effect a 1,500 to 2,500-step model**, while K0's map head has 54,852 steps of training.
This flaw is in the instrument, it affects only K2, and it does not depend on the map dose. On top of it, the default-cut primary
compares two different heads at different operating points. The next arm costs **zero training**: re-read both arms on their
**raw** end-of-WSD weights at FIT levels, about 6 Thor-h. The pre-registration is in §4.

**Sources.**
* [TR]: Thor `r8ns/r9e2/runs/r9e2-{K0,K2M1N20,K2DC}/metrics.jsonl` (sha256 `fabca10f…`, `63a01553…`, `aaf34833…`, pulled
  equal), read by `k2diag/traj.py` (`traj.out`) and `plan_traj.py`.
* [E1TR]: Thor `r8ns/r9e1/runs/r9e1-{C0,M1N20}/metrics.jsonl`, read by `traj_e1.py` (`traj_e1.out`).
* [FIT]: repo `…/e2/read_K2M1N20/diag_fitthr/EVAL_{K0,K2}_fitthr.json`.
* [DC]: `C:/Users/Admin/r9e2_read_K2DC/pull/read/`.
* [DT]: repo `…/read_K2M1N20/reread_dtfix/read/`.
* [KO]: repo `…/read_K2M1N20/knockouts/README.md`.
* [E1V]: `dagland/e1/verdicts/M1N20/read_v3/arm_M1N20/thr_44f7c569_7e89334f/VERDICT_M1N20.md`.

## 1. Every argv difference (MEASURED, `k2diag/argv_diff.out`, from the `stack/ops/runs.d/*.argv.json` files)

* **K0 → K2M1N20 (9 flags):**
  * the map head: `--r9-map-banded` and `--r9-map-banded-near-m 20`;
  * the map weight: `--w-map-hires` 1.0 → **25.0**;
  * T1v: `--r9-hist-supervise-geom`;
  * W-MAP-2: `--r9-plan-map-channels`;
  * W-BOX: `--r9-plan-box-tokens`;
  * the critic: `--r8-critic-drivable` with `--w-r8-drivable 0.06472`;
  * `--out`.
* **Identical in K0, K2M1N20 and K2DC:**
  * init `ckpt_ema_48852.pt`, `--opt dd` (AdamW), lr 1e-4, 6,000 steps;
  * WSD schedule with warm-up 1,000;
  * **`--ema-decay 0.9998`**, seed 0, batch 16.
* **K2M1N20 → K2DC:** only the four `--map-hires-share-*` and `-w-bounds` controller flags.
* **E1 (C0, M1N20) → E2 (69 to 75 flags):**
  * E1 used the perception rig (`--r9-perc-rig`, `--arm flat`, no planner), 4,000 steps, cosine schedule, warm-up 200;
  * E1 used **`--ema-decay 0.999`**.
* **The heads differ.** K0 runs refcv8's **8-way softmax hires head**, trained since init. K2's banded head reuses the refine trunk,
  but its readouts `head_near`, `mid` and `far` are **new and randomly initialised** (`stack/tanitad/models/map_banded.py:156-168`).

## 2. Q1: why the map primary loses, ranked by measured evidence

**R1. Instrument: a precision-only primary at a fixed cut compares two different heads.** MEASURED, T0.
* K2M1N20's gap falls from **−0.337 to −0.077** when each arm uses its own FIT thresholds [FIT].
* At the default cut, K0 flags 3.3 % of 5–10 m cells at sidewalk recall 0.533. Its 8-way edge plane adds almost nothing:
  swedge precision 0.739 against sidewalk-only 0.712.
* K2DC flags 5.9 % at recall 0.769. Its edge plane is trained on dilated 0.5 m targets and cuts precision from 0.564 to **0.359**
  [DC `EVAL_K*.json` nearfield table].
* E1 already used the FIT protocol, so R1 cannot explain why E1 and E2 disagree.

**R2. EMA immaturity of the new modules. This affects only K2 and is the same at any dose.** MEASURED mechanism, T0. Effect on the
DEV primary: HYPOTHESIS.
* **Why the read lags.** `WeightEMA` has a fixed decay, no warm-up, and starts its shadow from the step-0 weights
  (`stack/tanitad/train/weight_ema.py`; `refc_v3_train.py:11703-11730`).
* **How much of the init survives** (`k2diag/ema_math.json`):

  | read | share of the shadow that is still the init |
  |---|---|
  | E2, 0.9998 over 6,000 steps | **0.301** |
  | E1, 0.999 over 4,000 steps | 0.018 |
  | E3 D1 read, 36,000 steps | 0.0007 |
  | E3, 73,280 steps | about 4e-7 |

* **What that does to a new module in E2:**
  * a readout that converges with a time constant of 1,000 steps keeps **37.6 %** of its initial error in the EMA, against
    0.25 % in the raw weights;
  * a zero-init gate that ramps linearly reads **0.42×** its final value.
* **In-run comparison at step 6,000** (eval139, 8 batches, T0, no CI, not the DEV split) [TR]:

  | arm | edge_h IoU, near (raw / EMA) | drv loss, near (raw / EMA) | sdw IoU, near (raw / EMA) | critic BCE `r8_drv` (raw / EMA) |
  |---|---|---|---|---|
  | K2M1N20 | 0.225 / **0.177** | 0.282 / **0.369** | 0.575 / 0.543 | 0.407 / **0.494** |
  | K2DC | 0.219 / **0.166** | 0.299 / **0.389** | 0.562 / 0.534 | 0.402 / **0.504** |
  | K0 (8-way head) | — | — | sidewalk IoU 0–20 m: 0.608 / 0.615 (EMA ≥ raw) | — |
  | E1 M1N20, step 4,000 [E1TR] | 0.211 / 0.209 (converged) | — | — | — |

  The EMA at step 6,000 sits where the raw weights were at steps **~1,500 to 2,500**.
* **This fits the E1-to-E2 swing, on DEV at FIT levels** (MEASURED, [E1V], [FIT]):

  | arm | E1 | E2 | change |
  |---|---|---|---|
  | control (C0 → K0) | 0.4911 | 0.4776 | −0.014 |
  | banded (M1N20 → K2M1N20) | 0.5132 | 0.4008 | **−0.112** |

  So the swing specific to K2 is about −0.10, and only K2 builds new modules.

**R3. Map dose is eliminated as a lever for (a).**
* MEASURED, in-run: cutting the dose 4–7× (K2M1N20 → K2DC) moved the raw IoUs by ≤ 0.013 [TR], and the DEV primary by −0.0055
  [DC].
* ANALYTIC (an Adam property, not separately measured):
  * only the map loss reaches the banded head, because W-MAP-2 and the critic read the map **detached**
    (`stack/tanitad/train/plan_wires_flag.py:4,88`);
  * AdamW (`refc_v3_train.py:9307`) is invariant to a constant loss scale on those parameters, so the dose acts only on the shared
    trunk and lift;
  * [KO] already shows that the trunk share was 0.77–0.83 while the map still lost.
* ⇒ **K2DS (a smoother dose) is predicted not to move (a).**

**R4. Operating point at FIT.** MEASURED [FIT]. K2 still flags more cells:
* 5–10 m: sidewalk recall 0.765 against 0.674, precision 0.570 against 0.601;
* 10–20 m: recall 0.816 against 0.742, precision 0.573 against 0.626.

A primary that scores precision alone penalises this.

**R5. Full-config competition (planner, K = 10).** HYPOTHESIS, weak. K2's raw in-run drv and sdw IoUs are 0.03 below E1 M1N20's,
but the two runs used different eval windows.

## 3. Q2: the residual lateral regression in K2DC

**What the rows show.** T1, paired episode-cluster bootstrap over 70 DEV episodes, K − K0. MEASURED [DC] and [DT]
(`k2diag/fam_cmp.out`).

| row | K2M1N20, s0 | K2DC, s0 | K2DC, s1 |
|---|---|---|---|
| curvature | +7.5 %, separated | **+12.5 %, separated** | +4.9 %, not separated |
| yaw rate | +5.8 %, separated | **+6.4 %, separated** | +2.6 %, not separated |
| heading | +2.8 %, separated | **+2.2 %, separated** | +0.5 %, not separated |
| cross-track | +7.3 %, separated | +3.2 %, not separated | +1.1 %, not separated |
| LON accel | — | — | +2.1 %, separated |

So cross-track roughly halved when the dose came down. The regression is jitter, not bias: |Δe| from frame to frame is
+12–13 mm [DT `lat_attrib.json`, KO].

**Carriers, ranked:**
1. **The EMA-immature planner-side modules.**
   * The critic damps jitter: zeroing it made the jitter **worse** [KO]. Its EMA BCE (0.504) is 25 % worse than its raw BCE
     (0.402) [TR, T0].
   * The zero-init W-MAP-2 projection and W-BOX gates read about 0.42× in the EMA (analytic).
   * The knockouts zero these modules **inside the EMA model**, so they cannot see this effect.
   * HYPOTHESIS.
2. **The map term still reshapes the trunk.** K2DC's trunk share is about 0.11, against 0.0004–0.0036 for K0 [KO]. This is
   consistent with cross-track halving when the dose fell. MEASURED, one run per arm.
3. **T1v.** It only widens the map loss's supervision mask (`refc_v3_train.py:13744`), so it is confounded with carrier 2.
4. **Training-run variance.** There is no replicate (`H-ESTIM-SEED-1`). The rows separate at s0 and not at s1, which also looks
   like inference-seed noise.

**Cheapest discriminating experiment:** the raw re-read in §4, stage B. If K2DCraw − K0raw has no separated lateral row, the EMA
carried it. If the rows persist, the regression is trained in (carrier 2 or 3).

## 4. Q3: one next arm, E2-RAW (zero training)

**What it is.** Re-read **K0 and K2DC at their raw step-6,000 weights**, the WSD decay end in `ckpt.pt`. K2M1N20 raw is
optional, as a second dose point.
* Decode (a) at each arm's own FIT levels, as design 2.1 specifies and the fitthr recipe implements.
* Everything else follows the E2 rule (REFCV9_DESIGN 4.4), at inference seeds 0 and 1.

**Argv delta.** No change to the training argv. Two read-side changes, both MM integration items:
* the read driver hard-codes `model_ema.pt` (`stack/ops/r9e2_read/r9e2_read_env.thor.sh:27,38`) and needs a raw switch;
* drop `--require-ema` from the dump and pass `--ckpt <run>/ckpt.pt` (`r9_e1_dump.py:398` records it as raw).

**Cost.** ESTIMATED from K2DC's MEASURED read stages (`E2_READ_K2DC/ROLL_STATUS.txt`, diag_fitthr `STATUS.txt`); compare 20.5
Thor-h for any training arm.

| stage | content | cost |
|---|---|---|
| A | map only: FIT dump 0.19 h + DEV dump 0.36 h + EVAL 0.22 h per arm, for 2 arms | **≈ 1.5 Thor-h** |
| B | roll, camblind and mapsel for (c), (d) and (e): ≈ 2.2 h per arm | ≈ 4.4 Thor-h |
| | S6c on the dev box | CPU only |

Thor is currently running MM's K2DC fit dump on the EMA weights, which supplies the EMA comparison point.

**Pre-registerable criterion.** Freeze it before any raw number exists:
* **GO** iff all of these hold at both seeds:
  * (a) K2DCraw > K0raw on `nf_swedge_prec_lv_5_20` at FIT levels (point);
  * (b) box AP@2 m: K2DCraw > K0raw (point);
  * (c) S6c sidewalk rate: K2DCraw < K0raw (point);
  * (d) no four-family row is separated worse;
  * (e) K2DCraw is separated better than its camera-blind arm.
* **Mechanism prediction.** It decides the diagnosis, not the GO. Fitted primary, EMA → raw: K2DC gains **≥ +0.03** and K0 moves
  **≤ 0.01** in either direction. The STRATEGIC family is reported PAUSED.

**Committed outcomes:**
1. **GO.** The E2 screen is read at raw weights, and E3 runs as approved. Its EMA carries 4e-7 of the init at 73,280 steps, so
   nothing changes. Every future screen of ≤ 10k steps that adds modules must read raw weights or add an EMA warm-up (a code lever
   with a G-DVB entry).
2. **(a) fails on raw.** EMA is eliminated as the map's carrier, and dose already is. The next lever is map capacity: M3N20L64,
   which the PI approved at ~13 Thor-h.
3. **(d) fails on raw.** The lateral regression is trained in. The next arm is K2DC with share target **0.02**, the design's
   whole-trunk floor, at about 20.5 Thor-h. It is not K2DS: smoothing does not lower the mean dose.

**UNVERIFIED.**
* No raw number on DEV exists yet.
* The in-run IoUs come from 128 windows with no CI.
* The R2 and R5 split of the E1-to-E2 swing is not measured.
* There is one training run per arm.

## Manifest

All of the following are in `F:/refcv9_e3/k2diag/` (scratch, one copy, not staged; MM to bank if wanted):
* the scripts `argv_diff.py`, `traj.py`, `traj_e1.py`, `plan_traj.py`, `fam_cmp.py`, `nf_dump.py`, `ema_math.py` and
  `metrics_keys.py`;
* their outputs (`*.out`, `traj*.json`, `plan_traj.json`, `fam_cmp.json`, `ema_math.json`).

This report is the only file outside scratch. Deletions are listed in the hand-back.
