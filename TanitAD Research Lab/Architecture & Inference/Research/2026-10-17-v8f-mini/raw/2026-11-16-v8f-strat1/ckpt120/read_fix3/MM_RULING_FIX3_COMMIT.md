# MM ruling: CKPT120 read, per-arm training commit (2026-11-17)

**What changed.** The arms after H1 (H2_s0, H2_s1, H1_rep, R1_s0, R2_s0) train at fix3, commit `aedf8716` on branch `agent/v8f-ckpt120-20261117`. H1_s0 and H1_s1 were trained and banked at `213250ce`. The read held every bank manifest to `213250ce`, so it would have refused the five fix3 arms.

**Why the read's code of record stays `213250ce`.** MEASURED in the C: mirror:

- `git diff --stat 213250ce aedf8716` shows two files only: `stack/scripts/v8f_mini_train.py` (+43/-1) and the new test `stack/tests/test_v8f_ckpt120_dagger_nav_s.py`.
- The trainer change is in `dagger_probe` alone: a new `_dagger_nav_s_check`, and `kept_moved` leaves out `M.NAV_S_KEYS`. It is a launch-time check on the DAgger swap. It changes no model, loss or optimiser code.
- The evaluator blobs are equal at both commits: `v8f_mini_eval.py` is `323b7d28` and `v8f_t1_eval.py` is `3715c326` at each. These are the files `TREE_PINS` checks.

**The change.**

- `c120_lib.py` adds `FIX3_COMMIT` and `ARM_COMMIT`. `bank_refusal(arm, ckd, commit=None)` holds each arm to its own training commit, and an explicit `commit=` still wins.
- `after_ckpt120.sh` polls `/root/strat1/logs/ckpt120_fix3.log` and stops on `ZZ CKPT120 FIX3 QUEUE FAIL`. Its local manifest check uses the per-arm commit.
- `test_c120_read.py` adds `test_bank_refusal_per_arm_commit`: an H2 bank at `213250ce` is refused and one at `aedf8716` passes. 29 of 29 pass on CPU.
- Backups `*.before_mm_fix3` sit next to each file. They are deleted once this lands.

**What it does not change.** The pre-registration, every bar, the arms and the evaluators all stay as they were.
