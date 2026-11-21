# HANDOFF -- combo_ens_retime (ENS-K -> RETIME stacking), EvalFlyWheel analyst, 2026-10-19 ~08:10 Berlin

PKG = `C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery` (unlanded qland hold; nothing git-added, nothing written to G:).

## State: DONE for the 30k eval-gains read; two clean reads are blocked on upstream artifacts
* PREREG_COMBO.md sha256 `a888e7cc99c3346dcbe71244f11c1d6f6ce654bccec9aeb7ec764b7050701287` (pinned in code; never edit; amend in a new file).
* `RESULTS_COMBO_30k.md` + `COMBO_step30000ema_unc_eval.{json,md}`: label **STACKS (EXPLORATORY, eval-fitted gains)**; all 12 controls PASS; run twice, 0 differing leaves. Marginal ENS-on-top-of-RETIME = -0.0030 m (1.2 % ADE) for up to x4 draws; combo = 79-81 % of additive.
* Code: `code/combo_ens_retime_r8.py` (sha 83ff0a56c162...), `code/test_combo_ens_retime_r8.py` (17 passed; log `pytest_test_combo_ens_retime_r8.log`), `code/run_combo_ens_retime_r8.sh` (memory-gated wrapper).

## Exact next steps (each is ONE command; assert on the .json, not the exit code)
1. Clean-gains 30k read, when `raw/retime_trainfit/FROZEN_GAINS_RETIME_train.json` + `.sha256` exist (train-fit chain: `grep ZZTF /e/refcv8_eval_kit/chain/trainfit_chain.log | tail`; done = `ZZTFPOSTENDZZ rc=0` and `RESULTS_TRAINFIT.md` exists):
   `sh code/run_combo_ens_retime_r8.sh step30000ema_unc train`  -> `raw/combo_ens_retime/step30000ema_unc_train/COMBO_step30000ema_unc_train.{json,md}`
2. REGISTERED read, when (a) `KD_STEP=48852 sh code/kdraw_queue_r8.sh` has produced `E:/refcv8_eval_kit/kdraw/step48852ema_unc/dump_s{2..7}/KDRAW_VERIFY.json ok`, (b) the final battery trees exist under `C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_unc/`, (c) the train gains exist:
   `sh code/run_combo_ens_retime_r8.sh step48852ema_unc train`
   The script refuses (names the missing precondition) rather than substituting. The mask sha must again be `4ebc6b86...` or the read is VOID.
3. Copy the label + the ADDITIVITY block into the hand-back; report the three transfers (single-draw -> averaged, 30k -> 48,852, train -> eval) and "S6 K=4 UNTESTED".

## Do not
Edit PREREG_COMBO.md; refit gains on ENS plans (that is a different arm needing its own addendum); run with `COMBO_TEST_MUTATION` set into the production dir (the script refuses); run two computes at once (needs FreeVirtualMemory >= 6 GB, ~3-4 GB peak, 5 min CPU).
