# Note N9 to PREREG_V8F_PROOF_ADD12 (SPATIAL): stage 0e missed control e3 and is not quoted; stage 0f, one change (how the tokens are reduced), written before its data (2026-11-13)

## 1. Stage 0e as it ended (15:21, `spatial/s0e/VERDICT_S0E.md`, controls block)

* e1 PASS (largest value under permuted labels -0.002), e2 position PASS (AUROC 1.000 from the grid state, 0.497 from the mean state), e2 motion PASS (1.000 from two grid frames, 0.491 from two mean frames, 0.493 from one grid frame), e4 PASS (1,200 clips, no eval clip, 30 of 30 label rows and 30 of 30 kinematics rows exact, 12 shard hashes).
* **e3 FAIL.** On the eval bank the projected grid state reads the lead gap at R2 0.216 where stage 0b read 0.3981 (tolerance 0.03), and the projected mean state at -0.077 where stage 0b read 0.2577 (tolerance 0.05).
* By note N7 section 3 no outcome name is printed and no read of stage 0e is quoted. The lead has read the controls block only; the read values in `RESULT_S0E.json` were not opened.
* Cause: the one Gaussian projection from 1,024 to 64 numbers per cell does not keep what the read needs. That reduction was my choice in N7; the control did its work.

## 2. Stage 0f: one change, the reduction of the tokens

The states are built at full token width (1,024) on the pod, CPU only, idle priority, one clip in memory at a time.
* Bases from clips that are in no read: the train clips at positions 1,800 to 2,399 of the ascending hash order (600 clips; not in the probe set, not eval, outside the block 540 to 600). Per cell: the cell's mean and its top 16 principal directions. For the mean token: its mean and its top 128 principal directions. No label enters.
* With these fixed bases, the grid state (640 numbers) and the mean state (128 numbers) of every frame of the probe clips (positions 600 to 1,799) and of the eval clips, float16, in the state-file layout of `pod/probe/extract_states.py`.
* The bases are fixed and come from other clips, so no state is fitted on any read row. The reads run in the state-file mode of note N8 (`spatial/s1/s1_read.py`, train probe set: r-a and the change read with arms H, N, E; grid against mean through its `pair` mode) and, on the eval bank, with `spatial/s1/s1_eval_read.py`.
* Everything else is stage 0e unchanged: labels, the 0 to 60 m rule, folds by clip, probe and subsample rule, continuous-lead pairs, bootstrap.
* Departure from stage 0b named: there the bases were fitted on the training folds of the eval bank; here they come from train clips. Control f3 measures whether that matters.

## 3. Controls (each must read its known value, or no outcome name is assigned)

* f1 alignment (control t1 of the tool): the rows of a state file are the rows of the probe set, exactly.
* f2 labels permuted within fold (t3): every R2 and skill at or below 0.02.
* f3, the e3 rule of note N7 unchanged: on the eval bank the grid principal state reads the lead gap within 0.03 of 0.3981 and the mean principal state within 0.05 of 0.2577.
* f4 identity of the two code paths (t2): the state-file mode fed with the mean over the 40 projected cells reproduces, within 0.005, the value the stage 0e code path computed for that same state. The two stage 0e numbers serve as an identity reference only: this compares two code paths on one state and does not depend on whether that state keeps the lead. For that reason the tool's precondition "stage 0e controls pass" is removed (one line, recorded in `TIMES_NOTE.md`); nothing else of the tool changes.
* Planted position and motion are not repeated: a state file cannot be re-planted on the dev box. They passed on real tokens in stages 0b, 0c, 0d and 0e.

## 4. Outcome names

Those of note N7, quoted as "stage 0f (train corpus, principal states)", assigned by the lead from the tool's numbers with rule E11 of `S0E_DEFS.md` (a difference qualifies when its point value and the lower end of its 95 % interval are above 0):
`S0-LEAD-KEPT` / `S0-LEAD-LOST` per state (gap R2, state minus ego vector); `GRID-ADDS` / `GRID-EQUAL` (gap R2, grid minus mean); the change read as `s0c_read.outcome_names` defines it (`S0-CHANGE-READ` / `-EGO` / `-FLAT` per state, `GRID-ADDS-CHANGE` / `GRID-EQUAL-CHANGE`, `HISTORY-ADDS` / `HISTORY-EQUAL`).

## 5. Both outcomes

As note N7 section 4, unchanged.

## 6. If f3 fails again

Then principal states with bases from train clips do not reproduce the stage 0b read on eval frames. No third reduction is tried. The untrained reference on the train corpus is dropped, and the trained-state reads of note N8 stand with the eval-bank values of stage 0b beside them.

## 7. Limits

T0, a world-model diagnostic, never driving performance. Bases from 600 other train clips; 1,200 probe clips; untrained linear states; linear and kernel reads.
LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25).
