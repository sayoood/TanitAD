# HANDOFF: the fan-equivalence tolerance (written 2026-10-12; state of round 1, exact next step)

## State
* `rl_fan_dump.py --compare-atol X` (+ `--compare-min-matched N`) is built and tested (`refe/test_rl_fan_dump.py` ZZTEST_PASS 44). Stage `fancmp` in `refe/rls1_pod.sh`
  is wired and tested (`refe/test_rls1_chain.py` ZZTEST_PASS 297) but REFUSES until `RLS1_FANCMP_ATOL` is set: round 1 (`README.md`, `verdict.json`) chose no tolerance.
* Round 1 pre-registered rule failed on 5 of 6 decision arms (max delta only 3.2x-8.9x of tol 0.5, needed 10x) and another vehicle's rig sits AT the 0.25 floor.
* Post-hoc lead (`posthoc_mean_abs.json`): per-set MEAN |delta| over x/y has a floor of 0.0122 (1,365 sets) and separates every detectable arm by 14x-135x. Not pre-registered.

## Next step (cheapest experiment that could still make it work; needs the dev-box GPU, ~25 min, so it is a FRESH agent's job)
1. Write the pre-registration FIRST (append to `run_calibration.py`'s docstring, or a new `run_calibration2.py`): statistic = per-set mean |delta| over x/y values
   (call it m); bound M = 2 x (max m over the NULL arms incl. the 1,365-set N_full), rounded up to a multiple of 0.005; ACCEPT iff 0 null sets above M AND every
   decision arm has >= 25 % of its sets above M AND its worst set's m >= 5 x M. Decide in the text whether another vehicle's rig (A1) stays a decision arm: it will
   fail again (its worst m is 1.7x the floor); if it is demoted, say so and why (indistinguishable from the null by direct measurement), never silently.
2. DISJOINT keys: `pick_keys` uses `rows[::28][:48]`; use `rows[14::28][:48]` (check zero overlap with `keys48.tsv`). Rebuild with `--phase prep`
   (new `A` dir, e.g. `.../atol_cal2`), then `--phase dump` in the background (each arm waits on `wait_idle.py`, ~80 s per arm, 16 arms), then `--phase analyse`.
3. If it passes: add `--compare-mean-atol` to `rl_fan_dump.py` (a set fails when its mean |delta| over x/y exceeds it; same exit 5, same markers, tests + RED
   mutations as for `--compare-atol`), keep `--compare-atol 0.5` as the catastrophe bound, pin both in the chain (`CHOSEN_*` in a verdict.json that the chain test already
   cross-checks: extend `check_fancmp`), update runbook D-RLS1-5. If it fails: report the arm that does not separate.
4. The pod's own floor is the other unknown: the first `fancmp` run on the A40 writes `delta_quantiles` into `$R/fancmp/rl_fan_dump_fancmp_s0of1.manifest.json`.
   Bank it and tighten.

## How to run / check
`C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe run_calibration.py --phase prep|dump|analyse` (TMP/TEMP on D:, never touch `D:/Projects/TanitAD/data/refe_proxy/pixels`).
Tests: `refe/test_rl_fan_dump.py`, `refe/test_rls1_chain.py`, `refe/test_launch_gate.py --static`. Work dir: `D:/Projects/TanitAD-artifacts/_rlfan_work/` (35 GB free on D:).
