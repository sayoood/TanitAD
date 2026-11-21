# PRE-REGISTRATION — is refcv8's sharp-turn under-rotation a DESIGN effect or a LEARNING-RATE-PHASE effect?

**Registered by the Master Mind 2026-10-18 ~03:55 Berlin, BEFORE any refcv7 RAW-15k/20k/30k battery number exists.**
Claim it serves: `H-R8-TURN-UNDERCORRECT-1` (GOALS_AND_CLAIMS, d113bb3). Tier of every number below: **T1 self-action OPEN LOOP on
logged eval139 frames** (not driving performance). Estimator: paired episode-cluster bootstrap over the 139 eval episodes,
B 2,000 — it answers *"another draw of EPISODES"* only, NOT another training run (H-ESTIM-SEED-1) and NOT another inference run
(inference seeds 0 and 1 are reported separately and must agree).

## 1. Why this exists (MEASURED, the confound)

The registered step-30,000 reading of the turn defect compares refcv8 milestones with **refcv7-50,400, which is fully
annealed** (logged lr 5.27e-14 at 50,400). refcv8 is mid-schedule: lr 4.02e-5 at 15,000 and 3.30e-5 at 20,000. refcv7 was at
**4.16e-5 / 3.48e-5 at the same steps** (both runs' trainer telemetry, read by
`FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/raw/turn_lever/train_telemetry_read.out`), so
**matched step ≈ matched learning-rate phase (within 6 %)**. Banked slopes (through-origin slope of turn-aligned heading error
on GT heading, sharp windows GT |yaw| ≥ 0.15 rad/s, last 2-s chord, `raw/heading_drift/`): refcv7-50,400 **−0.092**; refcv8
EMA **−0.130 / −0.155 / −0.141 / −0.141** and raw **−0.234 / −0.137 / −0.217 / −0.112** at 5k / 10.5k / 15k / 20k. refcv7 has
**no** mid-schedule number on this instrument, so "refcv8 under-rotates more than refcv7" is so far a comparison across two
learning-rate phases.

## 2. The experiment (eval only; no training — the PI's "don't retrain refcv7" is untouched)

refcv7's own milestone checkpoints already exist on the dev box (`E:/refcv7_eval_kit/ckpt/ckpt_{15000,20000,30000}.pt`,
raw weights; refcv7 trained without an EMA). Each goes through the same four-family battery as the refcv8 milestones
(standalone pose clock, inference seeds 0 and 1, the 4,754 S2 windows / 139 episodes), then `heading_drift.py` exactly as at
refcv8 steps 15,000 / 20,000. Order: 20,000 → 30,000 → 15,000. Dev-box RTX 4060 only, under the shared GPU lock; the
refcv8 step-30,000 queue keeps its place in the lock order.

**Known-value control first (VOID rule).** The code tree that rolled the banked refcv7-50,400 dumps (`C:/Users/Admin/ev7`)
no longer exists, so the re-rolls use the refcv8 launch tree. Before any milestone roll, refcv7-50,400 itself is re-rolled
on a small clip subset and compared with the banked `step50400_a6text` dumps on the same windows: the `os` plans must agree
to **≤ 1e-3 m max abs** (and the GT / model-free arms bit-identically). If they do not, **every refcv7 milestone number from
this run is VOID** and is reported as VOID with the measured difference — not used, not "close enough".

## 3. Quantities

* `s7(N)` refcv7 RAW-N slope; `s8r(N)` / `s8e(N)` refcv8 RAW-N / EMA-N slope; `s7f` = refcv7-50,400 slope (−0.092).
* Primary contrast: **`D_ema(N) = s8e(N) − s7(N)`** (the PI scores bars on EMA weights, 2026-10-16). Secondary:
  `D_raw(N) = s8r(N) − s7(N)` (raw vs raw, both un-averaged). Phase term: **`P7(N) = s7(N) − s7f`**.
  Negative = more under-rotation.
* Read at N = 20,000 first (both refcv8 numbers are banked); N = 30,000 when refcv8's step-30,000 battery lands; 15,000 as the
  trajectory point.

## 4. Outcomes — committed now, before the data

| outcome | condition (both inference seeds must agree; "separated" = 95 % CI excludes 0) | consequence |
|---|---|---|
| **PHASE** | `P7(20k)` separated negative **and** `D_ema(20k)` NOT separated negative | the gap to refcv7-50,400 is a learning-rate-phase effect. The turn-loss arm (A1, `raw/turn_lever/PREREG_DRAFT_R8_TURN_REFINE_LOSS.md`) is **NOT recommended on this evidence**; the turn claim is decided at refcv8's FINAL (annealed) EMA against refcv7-50,400. |
| **DESIGN** | `D_ema(20k)` separated negative **and** `P7(20k)` NOT separated negative | the under-rotation is refcv8's own. A1 stays the recommended next lever (PI decision: compute). |
| **MIXED** | anything else (incl. seeds disagreeing) | report the phase fraction `P7(N) / (s8e(N) − s7f)` with its CI; no recommendation change until the N = 30,000 matched pair, read by the SAME table. |

`D_raw` is reported beside every row and **cannot override** the table: raw refcv8 slopes moved by up to 0.12 between
consecutive milestones, more than the gap itself, which is exactly the training-trajectory variance this bootstrap is blind to.

## 5. What this does NOT answer

Another training run (one checkpoint per arm per step). Closed-loop driving (T1 open loop only). Whether refinement recovery
(`refine_shrink.py`) follows the slope — reported if the sidecars exist, never used to pick the outcome.
