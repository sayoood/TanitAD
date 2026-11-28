# PREREG_V8F_LATSEM2: does each world-model level encode and imagine the SCENE, not only ego motion? (Master Mind, 2026-11-07, written before any LATSEM2 code or statistic exists)

## Why
* The PI, 2026-11-07, verbatim: *"try to decode the internal imagination not only by ego motion semantics"*.
* LATSEM read speed, acceleration, yaw rate and their 2 s change. Every target was the car's own motion. Whether a level knows about the lead vehicle, the other agents or the manoeuvre the situation calls for was never read.

## Data and targets (eval-141, the LATSEM frame bank and windows)
* M2b s0 / s1, zero training, the same 139 episodes, frame bank and windows as LATSEM.
* Targets per frame, from labels only (privileged signals are fine for labels; inference is untouched):
  1. **lead present** (binary), **lead gap** (m, frames with a lead only) and **lead closing speed** (m/s, the gap's rate over 0.4 s), from the v8F loader's lead fields;
  2. **agents within 30 m** (count) and **nearest-agent distance** (m), from the obstacle join (`build_obstacle_join.py`, 10 dynamic classes);
  3. **longitudinal class** (7 classes, HOLD, STOP, CREEP, DECELERATE, FOLLOW, KEEP, ACCELERATE, as `PREREG_V8F_TOWN.md`'s log derives it);
  4. **turn ahead** (binary: the absolute heading change over the next 4 s exceeds 0.3 rad).
* The exact fields, units and NaN rules are fixed in `IMPL_CHOICES_LATSEM2.md`, frozen with its sha256 before any statistic.

## Statistics (LATSEM's estimator: episode-cluster bootstrap, B 2,000, seed 0; probes cross-fitted by the same two episode folds; ridge lambda 10 or logistic with C 0.1, fixed now)
* **S1, what a level ENCODES:**
  * R² for continuous targets;
  * for binary and class targets, balanced accuracy minus the fit-fold majority baseline's.
  * ENCODES: > 0, separated, at both seeds.
* **S2, what a level IMAGINES:** the target at the future frame, decoded from the rollout (true actions, L1 shrunk as `AMENDMENT_WMACT_1` commits), against the same probe applied to the start state.
  * SKILL = 1 - err(rollout) / err(start), with squared error, or 0-1 error for the binary targets.
  * IMAGINES q at (l, k): > 0, separated, at both seeds.
* **S3, does the imagined scene FOLLOW the action:**
  * the imagined lead gap under constant "accelerate" (+1.5 m/s^2) minus "brake" (-3.0 m/s^2), at the longest horizon, on windows with a lead;
  * ACTION-SEMANTIC-LEAD if < 0, separated, at both seeds (accelerating closes the gap).

## Controls (each must read its known value, or the read is INVALID)
1. The start-state query reads SKILL exactly 0.0.
2. Two identical action sequences give S3 exactly 0.0.
3. A probe fitted on labels permuted across episodes (a cyclic shift, no fixed points) is not separated above 0 on any row.
4. A label-shuffled regression arm of S3 (lead gaps permuted across windows before decoding) is not separated below 0.

## Outcomes (per level and target, as defined above; no single verdict)
* **SCENE-AWARE at level l:** ENCODES lead gap or agent count at both seeds.
* **SCENE-IMAGINING at level l:** IMAGINES lead gap at the longest horizon at both seeds.
* **ACTION-SEMANTIC-LEAD at level l:** as S3.
* If no level is SCENE-AWARE, the next lever is MRATE's W1 (`PREREG_V8F_MRATE.md`) and then a scene-target auxiliary, under its own card.
* If a level is SCENE-AWARE but not SCENE-IMAGINING, the next lever is the reliance arm on that level.
* Every outcome is re-read on MRATE's W0 / W1 when they exist, as MRATE's beside rows.

## Pictures (decide nothing)
* Nearest-frame retrieval for lead-present windows under accelerate and brake, as in LATSEM. Per-level PCA maps coloured by lead gap and agent count. Raw frames stay off the public repo.

## Log (dated, append-only after the freeze)
* 2026-11-07 12:57:27 (dev-box clock, from `date`), Master Mind. **Frozen:** the first 3769 bytes above this log, sha256 `1325f21c902967f3c86fdd2f9b56acf3ee28de01648d97496545cf6a130b1751`, before any code, run or statistic of this card exists.
