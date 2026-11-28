# HANDOFF -- r8_wrapper_diag (2026-10-16)
Running (CPU only, OMP_NUM_THREADS=6, detached): `python r8_wrapper_diag.py` in this dir -> `raw/diag_arms.json`
(banked after every arm), log `raw/diag.log`, per-arm key tensors `raw/arms/*.pt`, 4-row batch cache `cache/batch4.pt`.
Hypothesis under test: the G0 wrapper-clause FAIL is a NOISE-ASSIGNMENT artifact of `wrapper_probe_r7.py`
(it zeroes only `torch.randn_like`; refcv8 draws base DDIM eps with `torch.randn` at refc.py:3036 and the extras /
source-anchor choice from R8Generator.randn_eval/rand_eval), not a merge defect and not batch coupling.
Discriminator: zero-noise arms (`zero_*`) must show micro13/micro31/perm/row-alone == plain to float tolerance;
control `zero_seed1` == `zero_plain` bit-for-bit, `live_repeat` == `live_plain` bit-for-bit.
If it died: check `errors` in raw/diag_arms.json; rerun reuses cache/batch4.pt.

## 2026-10-16 03:40 -- run 1 DONE (raw/diag_arms.json, raw/diag_summary.json). Run 2 launched:
`python r8_wrapper_diag2.py` -> raw/diag2_arms.json (log raw/diag2.log): LIVE noise recorded on the 4-row plain arm and
REPLAYED per row into micro13 / micro31 / permuted / row-alone arms; plus plain at seeds 1..7 (redraw spread).

## 03:50 run 2 DONE (raw/diag2_arms.json). Run 3: validate_fix.py -> raw/validate_fix.json (real probe, P3, CPU, unfixed vs fixed).

## 04:00 ALL RUNS DONE. RESULT.md written. Nothing running.
