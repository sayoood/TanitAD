# PREREG_V8F_N1DREP (DRAFT, MM 2026-11-22; to be frozen before the replicate trains): a same-seed replicate of N1D gives the operative guard cells a measured floor at the executor of record

Status: DRAFT. This is N1DM-COST's committed next lever (`PREREG_V8F_N1DM.md` 958d143d, section 3; `MM_VERDICT_N1DM.md`).

## 0. Why (MEASURED, quoted)
* N1DM's eval-141 guard read GUARD-COST-BOTH-SEEDS on operative 0-2 s cells: ade +0.0122 / +0.0124 m, along_mae +0.0123 / +0.0124 m, speed_mae +0.0084 / +0.0081 m/s (`eval141/out/RESULT_N1DM_EVAL141.json`).
* The floor was F2: a same-seed replicate of C0, from the OLD executor (0.0005 m on ade). No replicate of N1D exists, at any executor.
* The heading bound does not bind in 0-2 s, so the old and hb0.2 executors give these cells to the last digit; the floor's gap is the ARM (C0 against N1D, and N1D's DAgger swap), not the executor.
* N1D's trainer runs the DAgger swap (`--dagger-p 0.1`). A same-seed run of that trainer is not bit-identical (memory, MEASURED on an earlier pair), so the replicate measures run-to-run noise rather than reading an identity.

## 1. Arm
* **N1DREP_s0** = N1D_s0's argv (N1DM_s0's argv without `--aux-route modes --w-aux-route 0.5`; the exact form read from the pod's N1D record) with only `--out` changed. Seed 0, init M2b_s0, bank `strat1/inputs/dagger_bank/s0`, code of record 573ad839.
* One seed: the floor is per cell |N1DREP_s0 - N1D_s0|, on the same 2,399 windows.

## 2. Read (T1-kin/hb0.2, eval-141, 141 episodes)
* The same cells and estimator as the N1DM guard (paired episode-cluster bootstrap, B 2,000, seed 0). The estimator answers "another draw of EPISODES"; the replicate answers "another TRAINING run" at fixed seed and init; inference is deterministic.
* F_REP per cell = abs(mean N1DREP_s0 - mean N1D_s0). The N1DM guard is re-read beside its verdict with floor max(F2, F_REP). The verdict of record stays N1DM-COST, as written.

## 3. Outcomes, committed in advance
| name | condition | reading | next |
|---|---|---|---|
| **N1DREP-COST-HOLDS** | on ade, along_mae and speed_mae, N1DM - N1D exceeds max(F2, F_REP) at both seeds | the operative cost is above run-to-run noise | F_REP becomes the floor of record for operative cells on the N1 line at hb0.2 |
| **N1DREP-COST-IN-NOISE** | on at least two of the three, N1DM - N1D is within max(F2, F_REP) at either seed | the cost is run-to-run noise; N1DM is still not adopted, because D1-HO shows no gain | F_REP as above; the decoder-path family reads no gain and no cost |
| **N1DREP-IDENTICAL** | every dump array equals N1D_s0's | the trainer is deterministic at this seed; F_REP = 0 is an identity, not a floor | a second-seed replicate (N1DREP_s1 with a fresh dagger seed) is the floor: back to the PI |

## 4. Cost (ESTIMATED)
* Pod: about 0.2 h training (N1DM's measured 701 to 705 s per arm) plus the gate, about 0.3 pod-h.
* Dev box: one eval-141 hb0.2 dump (about 6 min, as the N1DM chain) and the read.
