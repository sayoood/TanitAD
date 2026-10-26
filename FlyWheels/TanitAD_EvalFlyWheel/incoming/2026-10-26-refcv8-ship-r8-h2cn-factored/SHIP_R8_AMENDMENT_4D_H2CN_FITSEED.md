# SHIP-R8 AMENDMENT 4d -- the H2cn stage re-trained at fit seeds 1 and 2: is Amendment 4's PASS a lever effect or one draw of the stage fit? (Master Mind, 2026-10-26 ~05:36 CET)

Sealed BEFORE any fit-seed-1/2 fit, stage, position, cell or clause exists. Amendments 1-4, 4b, 4c and 5 stand; nothing registered is edited.

## 0. Why (disclosed)
Amendment 4 read **PASS** on every clause for `H2CN_DEPLOY8` from ONE fitted stage (`H2CN_STAGE_FROZEN.pt` sha256 `f293b7bd...6f32`, registered
04:24:43; `F.h2n_fit(..., seed=0)`). Its interval is the paired episode-cluster bootstrap, which resamples EPISODES with the stage held fixed: it
never answers *"would another fit of the stage say this?"* (`H-ESTIM-SEED-1`). In `l1_fit_score.h2n_fit` the seed sets BOTH the episode-grouped
CV folds (`folds_of(eid, seed)`) and the MLP initialisation / order (`mlp_train(..., seed)`); a different seed is therefore a second, complete
training run of the stage on the same FRESH 300-clip data. The PI is about to choose the shipped stage (H2CN / H2CN_CAP8 / H2CN-L / FACTORED);
that choice should rest on the stage's run-to-run floor, not on one draw.

## 1. Arms
* `H2CN_FS1_DEPLOY8`, `H2CN_FS2_DEPLOY8`: Amendment 4's recipe VERBATIM (`h2cn_freeze.py` -> `h2n_fit`, S2 and S6 heads, same inputs, same CV
  grid and selection rule, same FRESH split, compliant rows, disjointness checks; forward `h2cn_progress` + running max +
  `eval_polyline(polyline_geometry(plan8), s8)`) with the fit seed set to 1 and 2. The CV choices (weight decay, epochs) are RE-SELECTED at
  each seed by the registered rule and reported; they are not pinned to seed 0's.
* Each replicate stage is frozen to its own file; its sha256 is written to its own `REGISTERED_H2CN_FS{1,2}_STAGE.sha256` by the Master Mind
  BEFORE any eval-window output of that stage. Work in a separate copy of the sub-read code (`C:/Users/Admin/refcv8_pf/ship_r8_sub_fs/`);
  the registered directory `ship_r8_sub/` is read, never written.

## 2. Controls (a failed control voids the reads)
* **K-REPRO0:** the copied code at fit seed 0 reproduces the registered stage: sha256 of the re-frozen stage == `f293b7bd...6f32` (or, if the
  container bytes differ, every tensor bit-equal AND the predictions on all FRESH rows bit-equal -- stated which).
* **K-SEED-MOVES:** the fit-seed-1 and -2 stages differ from seed 0 (some parameter tensor differs). Without it the replicate is a copy.
* Amendment 4's K-DISJ, K-RELOAD, K-DETERM, K-RETIME identity and RETIME-TAC identity, re-run per replicate; `H2CN_DEPLOY8` (seed 0) reproduces
  its landed verdict bit-for-bit from the copied forward.

## 3. Reads and the replicate bar
* **R1:** SHIP-R8 clauses (1)-(7)+(1b) on each replicate (both inference seeds), verbatim from Amendment 4 with the arm swapped -> OVERALL.
* **R2:** the four-family deltas of each replicate vs `RETIME_DEPLOY8` on the same draw (E1, E3, FD, MD, S6 compliant), beside seed 0's.
* **R3 (the floor):** each replicate PAIRED vs `H2CN_DEPLOY8` (seed 0) on the same draw: S2/S6 ADE, FD, MD with CI. A separated difference
  between two fits of the same recipe is the fit-seed floor made visible; it is reported as such, never as a lever.
* **REPLICATED** iff R1 OVERALL PASS at BOTH fit seeds AND, at both fit seeds and both inference seeds, FD vs `RETIME_DEPLOY8` is separated
  below and E3 vs `ha0_ext_pose` is separated below. Otherwise **NOT REPLICATED**, naming the clause and the seed.

## 4. Tier / estimator
T1 self-action OPEN LOOP on logged eval139 frames (not driving performance); paired episode-cluster bootstrap B 2000 seed 0 over EPISODES; ONE
refcv8 training seed; inference seeds 0 and 1 never pooled; fit seeds 0/1/2 never pooled (each read separately).

## 5. Committed next steps
* **REPLICATED** -> Amendment 4's PASS is a lever effect across stage fits; the stage choice goes to the PI with the fit-seed spread printed
  beside every row. The same replicate is owed by 4b (the clamp is deterministic given the stage: it inherits this read) and by 4c if it passes.
* **NOT REPLICATED** -> the shipped stage is a 3-fit ENSEMBLE (mean progress of the fit-seed 0/1/2 heads, running max, same positions),
  pre-registered as its own amendment before it is read; H2CN_DEPLOY8 is not recommended alone.
