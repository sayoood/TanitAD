# PREREG_FRESH2 -- a second fresh PhysicalAI-AV pull: confirm the decel-decision levers on windows nobody has seen (Master Mind, 2026-10-27)

Seal: sha256 of this file + wall-clock time in `PREREG_FRESH2.sha256`, written BEFORE `l1c_select.py --n-eligible 1020` runs; no FRESH2 clip id,
file, label, feature or score exists before the seal. PI go: `D-R9-FRESH2`, 600 clips (2026-10-27, in chat, "600 clips (Recommended)").

## 0. Why (MEASURED, disclosed)
`H-R9-R9PF-LH-1`: the R9 profiles transfer a correct decel decision (GT regime: S6 ADE -0.40 m) and amplify a wrong one (LH: +0.29 m) -- decision
quality binds. Two candidate fixes exist and neither can be confirmed with what we hold: the equal-weight D2 + LH fusion read +0.0450 [+0.0185, +0.0716]
over LH on eval139 (POST-HOC, beside `H-R9-LHA0-D2-1`), and a LH refit on more clean clips has no clips to grow on. eval139 is barred for further
decode variants (`PREREG_R9PF_LH` s5) and eval141 = eval139 + 2 clips.

## 1. Selection and fetch (mechanical; code of the L1C package, `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-25-refcv9-l1c/code/`)
1. Back up `F:/l1c/select/` before anything runs. `l1c_select.py --n-eligible 1020` (eligibility E1-E5 and the sha256(clip_id) order unchanged).
   **STOP unless rows 1-420 of the new `candidates.jsonl` are byte-identical to the backed-up file.** FRESH2 = rows 421-1020 (600 clips).
2. `l1c_fetch.py --first 420 --count 600` (front-wide mp4 + egomotion + calibration, as L1C); obstacle.offline parquet for the same clips by
   `fetch_fresh_obstacle.py`'s method. Raw ids stay under `F:/l1c/`; the repo carries sha12 only.
3. **Usable** = fetch ok AND every L1C chain step (episode build, clock sidecar) accepts the clip. A refused clip is logged by sha12 + step + reason
   and EXCLUDED; there is no replacement (the pool is fixed at ranks 421-1020).

## 2. Split (fixed now, before any label exists)
Usable FRESH2 clips in candidate order (sha256 ascending): **CONFIRM** = the first ceil(2/3 x n_usable); **GROW** = the rest. CONFIRM is never used to
fit, select or tune anything. GROW is used only to refit LH (arm `LH_G`).

## 3. Stage 1 -- the T0 label-level read on CONFIRM (run ONCE)
* **Population and labels:** exactly PREREG_LHA0_D2 s3's K-HEAD definitions (S6-valid, nav-compliant, GT in {decel = BRAKE_TO/CREEP/HOLD, cruise =
  FOLLOW/CRUISE}), LON6 labels built WITH the lead pass (the `H-R9-FRESHLABEL-1` fix, LONHEAD_C's label code), z_tac from refcv8 EMA-48,852
  (`ckpt_ema_48852.pt`, md5 b4364029...), a0 / psi_dot0 by `refcv9_lha0`'s feature code.
* **Arms (same windows):** `LH` (`LH_FROZEN.pt`, sha256 6fd3ea1e...d8f6); `D2` (`D2_FROZEN.json` 69135b91...); `FUSE` = z(D2 logit) + z(LH logit),
  z-scored over the evaluated windows without labels, no fitted weight (the eval139 rule verbatim); `LH_G` = LH's recipe (`lh_fit.py`: features,
  lambda grid by out-of-fold CE, 5 folds by episode, seed 0) refit on the L1C FIT rows + GROW rows; `constant`; `FUSE_perm` = FUSE with a0 and psi_dot0
  permuted within each CONFIRM episode (`np.random.default_rng(0)`, the D2 permutation function).
* **Estimator:** Mann-Whitney AUC (average-rank ties); paired episode-cluster bootstrap over CONFIRM episodes, B 2000, seed 0, every arm on the same
  draws; CI = 2.5 / 97.5 percentiles of the paired difference. Printed: n windows, n decel, n episodes, d per arm.
* **Bars (unrounded):**
  * **F-1 (fusion confirmed):** AUC(FUSE) - AUC(LH) >= +0.020 AND the CI lower bound > 0.
  * **G-1 (grown head):** AUC(LH_G) - AUC(LH) >= +0.020 AND the CI lower bound > 0.
* **Controls:** `constant` must read 0.5000 (else INVALID); `FUSE_perm` must NOT clear F-1's bar (else F-1 is VOID: the gain is not per-window a0).
  Power (ESTIMATED from the eval139 SE scaled 1/sqrt(n)): ~360 CONFIRM episodes detect ~+0.023 at 80 %; the bar is set below that on purpose and
  needs the CI to exclude 0 as well.

## 4. Committed outcomes
* **F-1 and/or G-1 PASS:** the passing decision (FUSE preferred if both) is the decel decision for refcv9's progress profiles. Stage 2 = a T1 roll
  pre-registration (R9 profiles fed that decision vs the shipped H2CN stage) on CONFIRM windows, drafted and sealed BEFORE any roll.
* **Both FAIL:** the post-hoc decision line is closed on unseen data. The decision moves into refcv9 training (an in-model decision head trained on
  the train corpus with the clean-split calibration of `REFCV9_INMODEL_DESIGN.md`) -- named as the next lever, not run here.
* **VOID / INVALID:** reported; no claim.

## 5. What the interval answers
Another draw of EPISODES with the scores held fixed. Not another training run (one refcv8 trunk, one fit per head) and not another inference run
(heads deterministic; z_tac bit-equal across inference seeds, `K_HEAD.json`). Tier T0: a label-level diagnostic, never driving performance.
