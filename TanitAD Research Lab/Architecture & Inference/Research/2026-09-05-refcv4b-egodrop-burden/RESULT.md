# The ego-dropout burden on refcv4b is real, longitudinal, shrinking — and has a vision-only fix

`Stream: Architecture & Inference · 2026-09-05 · branch agent/arch-inf-20260803.`
`Dumps and GPU forwards by the FlyWheel agent that died at the API limit (steps 5,000 and
9,500, 4,823 windows / 141 B1-v7.2 EVAL episodes each, paired kept/withheld forwards on the
dev-box RTX 4060, 1,074 s per dump); analysis (zero GPU) and this write-up by the Master Mind.`

**Tier: T1** (self-action open loop) on a checkpoint at **step 9,500 of 40,284 (23.6 %)** —
an **EARLY-TRAINING DIAGNOSTIC**. Nothing here is a capability claim for the arm; the
open-loop suite and the echo gate on the final checkpoint decide that.
**Estimator:** paired episode-cluster bootstrap over the 141 episodes, n_boot 2,000
(`taniteval/ci.py`); never `overlapping_holdout_se`.
**Controls read their known values:** `ha` **0.29962** and `ha0` **0.67229** ADE 0–2 s on these
windows vs the banked refcv3-40284 surface **0.2996 / 0.6723** (Δ < 2e-05); `ha0` jerk reads
**exactly 0** at slot resolution (the constant-a/constant-κ bank paths are the discretisation
floor); the withheld regime is bit-identical to a zeroed-ego forward (**0.0 m** max abs
difference), so "withheld" is genuinely the training-time dropout regime.

## The question

`D-REFCV4B-EGODROP1` (MODEL-FREE) showed that under `ego_dropout 0.5` the withheld rows' anchor
bank is rolled at a **fixed 10 m/s** (`refc.py:1341-1344`), giving a geometric ceiling of
**7.59 m vs 1.11 m** for kept rows. The hypothesis raised by the Master Mind: that is the same
*shape* as refcv3's failure — a mis-geometried fan forcing the free offset head to override it,
the override being what destroyed smoothness — and here the burden is ~13× larger, on half the
training distribution. It was a hypothesis; the offset head is unclamped and might simply cover it.

## Verdict: REAL BURDEN — longitudinal, shrinking with training, not an abort reason

### 1. The offset head works ~10× harder on withheld rows, and does not fully cover the gap

| MEASURED, step 9,500, mean over the 8 slots, m | kept | withheld |
|---|---|---|
| selected-anchor offset norm | **0.641** [0.613, 0.673] | **6.179** [4.801, 7.813] |
| … of which along-track / lateral | 0.449 / 0.355 | **6.030** / 0.614 |
| geometric need (assigned anchor → GT) | 1.125 | 7.610 |
| residual after the offset | **1.073** [1.006, 1.142] | **2.994** [2.530, 3.577] |

The offset recovers **61 %** of the withheld need (7.61 → 2.99 m) against **5 %** on kept rows
(1.125 → 1.073) — but the withheld residual stays **2.8×** the kept one, and the burden is
**almost entirely along-track** (6.03 of 6.18 m), which is exactly the axis the fixed-speed
roll gets wrong.

### 2. The four families say the same thing, and only the longitudinal one moves

| family · 2 s · T1 · n = 4,823 | `os` kept | `os` withheld | `ha` | `ha0` |
|---|---|---|---|---|
| LONGITUDINAL speed MAE (m/s) | 0.500 | **1.859** | 0.254 | 0.488 |
| LONGITUDINAL along-track MAE (m) | 0.384 | **2.155** | 0.235 | 0.471 |
| LONGITUDINAL accel MAE (m/s²) | 1.046 | 1.464 | 0.317 | 0.479 |
| LONGITUDINAL ego-progress ratio | 0.995 | 0.965 | 0.975 | 1.008 |
| LATERAL cross-track MAE (m) | 0.183 | 0.215 | 0.123 | 0.313 |
| LATERAL heading MAE (°) | 2.38 | 2.68 | 1.55 | 2.78 |
| LATERAL curvature MAE (1/m) | 0.022 | 0.023 | 0.004 | 0.007 |
| LATERAL yaw-rate MAE (°/s) | 5.10 | 6.32 | 1.45 | 2.37 |
| TACTICAL | present, n = 4,823 (raw JSON) | present | — | — |
| STRATEGIC | **UNAVAILABLE with reason** (no route channel in this surface; the compliance metric is being built) | | | |
| distance-keeping | PRESENT, n = 1,153 windows with a lead agent (raw JSON) | | | |

Withheld vs kept: speed MAE **×3.7**, along-track **×5.6**, cross-track ×1.2, curvature ×1.05.
⚠️ **The early read nobody should soften:** at step 9,500 the *kept* selected trajectory does not
beat the hold-action control `ha` on any family at 2 s (speed 0.50 vs 0.25; along 0.38 vs 0.23;
cross 0.18 vs 0.12); it sits near `ha0` longitudinally and beats `ha0` laterally. That is 24 % of
training and is the diagnostic, not the result — refcv3 never beat `ha` at 40 k either, which is
the bar this arm exists to clear.

### 3. Smoothness — the PI's stated concern — in the programme's units

Slot-resolution finite differences on the emitted waypoints at their native instants
(0.5, 1, 1.5, 2, 3, 4, 5, 6 s); *not* a 10 Hz measure; `ha0` reads exactly 0 by construction.

| jerk, mean |·| (m/s³) | kept `os` | withheld `os` | GT (human) | `ha` |
|---|---|---|---|---|---|
| total | **2.42** [2.28, 2.58] | **3.65** [3.33, 4.00] | **0.86** [0.80, 0.93] | 0.03 |
| lateral (d(v²κ)/dt) | 2.11 | 2.18 | — | — |
| max |·| | 5.79 | 8.55 | — | — |

Withheld rows are **51 % rougher** than kept rows (non-overlapping intervals) and the excess is
**longitudinal** (lateral jerk 2.18 vs 2.11 — identical). So the PI's smoothness concern has two
parts: ego-dropout adds roughness on the withheld half, and — separately — **even the kept
trajectory is ~2.8× rougher than the human** (2.42 vs 0.86). The second part is not an
ego-dropout finding; it is the selected-trajectory smoothness problem the DiffusionDrive audit
(`D-REFC-DDAUDIT-2`, non-convergent refinement passes) and the low-speed flyability finding
(`D-REFCV4B-FLYLOW1`) bear on.

### 4. It is shrinking fast — the model is learning to infer speed from vision

| step | withheld offset (m) | withheld residual (m) | own 2 s speed prediction, withheld, MAE vs GT (m/s) | oracle − selected ADE, withheld (m) |
|---|---|---|---|---|
| 5,000 | 13.39 [11.43, 15.42] | 5.59 [4.68, 6.58] | 4.91 | −11.07 [−12.67, −9.44] |
| 9,500 | **6.18** [4.80, 7.81] | **2.99** [2.53, 3.58] | **1.84** | **−3.86** [−4.36, −3.35] |

Selection is also de-collapsing on both regimes: distinct anchors selected 32 → 43 (kept), 35 →
44 (withheld); straight-ahead share 74.5 → 60.1 % (kept), 45.5 → 20.4 % (withheld); no regime is
degenerate. This is the vision-only rule working as intended — a model that cannot read v0 is
learning to read speed from the frames — and it is why this is **not an abort reason**.

### 5. The fix, priced — option (c) is now viable, and it is vision-only

MODEL-FREE ceiling on the anchor PRIOR (oracle-in-vocabulary ADE, withheld rows), if the
withheld bank were rolled at:

| roll speed for withheld rows | ceiling (m) |
|---|---|
| fixed 10 m/s (shipped) | **7.61** |
| **the model's own 2 s speed prediction** (`g_tac`, the only speed the live forward emits) | **2.00** at step 9,500 (5.03 at step 5,000) |
| the true v0 — ⛔ the LEAK bound, refused as a design | 1.125 |

Rolling at the model's own prediction recovers **86 %** of the gap to the leak bound with no ego
input — it uses a quantity derived from vision. **Admissibility:** it contains no label and no
future, so it is not an echo of GT; it *is* a self-conditioning loop (the bank depends on a
prediction from the same trunk), so it must carry an intervention control at eval (shuffle the
predicted speed across windows: the gain must vanish) and be pre-registered on the v7-tiny rig
before it earns a launch. The other options: (a) keep 10 m/s — costs the 2.99 m residual and the
51 % roughness on half the rows; (b) speed-bucketed reference — ⛔ inadmissible if the bucket key
is the true v0 on a withheld row (that is the leak in disguise); (d) lower `ego_dropout` — weakens
the one lever measured to make an arm read the scene (`H-ECHO-8`), so it is the last resort, not
the first.

**Decision:** no change to the live run. Pre-register option (c) for refcv5 (`H-EGODROP-PRED`
below) and re-read this burden on the final checkpoint, where the trend in §4 predicts it will
have shrunk further.

## What this does NOT claim

- Not a driving result and not a comparison to refcv3: T1, step 9,500, one checkpoint pair.
- The kept-vs-withheld contrast is between two regimes of the SAME model on the SAME windows;
  the paired deltas in the raw JSON are the decision-grade form, the level tables above are
  the readable form, and their intervals do not overlap.
- Strategic family: absent with reason, not omitted.

## Deliverable manifest

| artifact | where |
|---|---|
| this write-up | `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refcv4b-egodrop-burden/RESULT.md` |
| analysis outputs (full, incl. paired deltas, by-v0-band, four families 2 s and 6 s, smoothness on three grids) | `…/raw/EGODROP_9500.json`, `…/raw/EGODROP_5000.json` |
| dump manifests (window/episode identity, checkpoint md5s, model checks) | `…/raw/manifest_9500.json`, `…/raw/manifest_5000.json` |
| instruments | `…/raw/scripts/egodrop_dump.py`, `egodrop_analyze.py`, `egodrop_compare.py`, `bank_copy.py`, `run_5000.log`, `run_9500.log`, `chain.log` |
| the dumps themselves (283 files × 2, 15 MB each) and the pulled checkpoints (428 MB, 1.29 GB) | **ONE PLACE ONLY:** `devbox:C:\Users\Admin\refcv4b_egodrop\out_{5000,9500}\dump8\`, `…\pull\` — re-derivable from the pod's checkpoints, not banked |
| register | `D-REFCV4B-EGODROP2` (supports and sharpens `D-REFCV4B-EGODROP1`), `H-EGODROP-PRED` (open, pre-registered) |
