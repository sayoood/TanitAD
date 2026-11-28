# HANDOFF: mutation lists and pins for ACTCONS, LEADSUP, DISTRULE, QHEAD N1 (2026-11-19)

Folder `F:/TanitAD-scratch/v8proof/mutlists/` (one subfolder per lever; `common/` holds the runner parts). Pod work folder `/workspace/v8proof/mutlists/`.
Method: each mutation is ONE exact string replacement (old text must occur exactly once) in a COPY of the lever tree on the pod; the lever's tests run on CPU in a fresh python child
(`common/minipytest.py` is a plain-python stand-in for pytest, the pod has no pytest; `common/pod_mut_driver.py` does the apply / run / restore). A mutation counts RED when at least one test fails.
`expected_all_red` says whether every test named in advance as the catcher failed (RED by a different test is reported as it is). No mutation was dropped or tuned.

## Counts
| lever | mutations | red | still green | control before / after | check run |
|---|---|---|---|---|---|
| ACTCONS | 12 | 12 | 0 | green / green | `test_v8f_act_cons.py` (14 tests) on `/workspace/v8proof/tree_actcons` copy, pins verified equal to `actcons/PINS.txt` |
| LEADSUP | 14 | 12 | 2 | green / green | `test_v8f_lead_sup.py` (16 tests) on `tree_leadsup` copy |
| DISTRULE | 32 | 24 | 8 | green / green | `test_dist_rule.py` (22 tests) + `test_margin_qcal.py` (13 tests, written here: the lever has none for the margin and quantile-map forms) |
| QHEAD N1 | 7 (existing) | 7 | 0 | green / green | not re-run; existing json copied with `COUNTS_QHEAD_N1.txt`; pins only |

ACTCONS: m6 (`adapt.detach()` dropped) and m12 (gate refusal of `--act-cons-max` without weight) are RED but through `test_gradient_zero_mutations_go_red` and
`test_default_off_is_a_lever_that_declares_nothing_when_max_is_changed` instead of the named tests (`expected_all_red` false).
DISTRULE control: 2 of the 35 tests are environment-bound on the pod and are excluded from every verdict (listed in `ignore_failed` of the spec and in the control record):
`test_geometry_copy_equals_headway_read` (needs the dev-box `headway_read`) and `test_pod_script_refusals_layout_and_one_rollout` (fake python and cgroup files). Both are untested by this list.

## STILL-GREEN mutations (what each means; the reasons are readings of the tests, not separately proven)
* LEADSUP m2 (window mask `all` to `any`): probably no test has a window with some frames not real inside the hist span; the "frame_ok all hist frames" rule is untested.
* LEADSUP m5 (gap part divided by the presence count): the all-masked test reads 0 either way; a part-wise denominator is not pinned by a value test.
* DISTRULE d8 (unchanged steps not copied back): the sampled path reproduces the knots to float32 after the cast, so the bit-for-bit copy is redundant in the tests (near-equivalent mutant).
* DISTRULE w1 (progress offset `a_l - a_m` dropped from the source gap): probably the end to end tests use ticks where the offset is 0 or the effect is not asserted; the offset is not pinned by a test.
* DISTRULE w5 (head presence threshold 0.5 to 0.0): probably the head-source test does not separate probabilities below 0.5; the threshold is not pinned.
* DISTRULE w8 (source-bad ticks stay present): probably no test has a present tick with a non-finite source value through the wrapper.
* DISTRULE w9 (refusal of `--dist-rule` with a non-run mode removed): `test_cli_refusals` apparently does not cover that refusal.
* DISTRULE q2 (map not capped by `min(g, Q)`): equivalent on every valid map (check_map forces ys <= xs, so Q <= g); the min only matters for an invalid map.
* DISTRULE q5 (PAV removed from the fit): the synthetic quantiles are already monotone, so pooling never fires; a non-monotone fit is untested.
* DISTRULE q8 (p95 read as p90 in margin_calib): my test tolerance (0.5 m) sits exactly at the 0.5 m difference of the synthetic world; the p95 value is not pinned tightly.
These need a stronger test before the port relies on them; the list does not hide them.

## What was NOT run
* QHEAD N1 mutations (not re-run, existing file as banked 2026-11-19T03:57:50Z). `margin_calib.main` / `qcal_fit.main` file paths and real-data fits: only `calib_arm`, `fit_arm`, `up_half`, `pav`, `apply_map` on synthetic data.
* Nothing needing a GPU, the training data or the dev-box torch (all pod CPU, one process at a time, 33 to 34 GB cgroup headroom at each start; `CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4`).
* The trainer loop of ACTCONS and the real-data selfcheck (a4 smoke scripts of `actcons/checks`): the pytest file stands for the check; a1 to a4 scripts were not re-run.
* Note: ACTCONS and LEADSUP mutations ran with the first version of `pod_mut_driver.py`/`minipytest.py`; the later edits (setitem, per-mutation test files, env, ignore list) do not change behaviour for those two levers; PINS name the final files. A control-only rerun of ACTCONS with the final driver was green.

## Files
`<lever>/MUTATIONS_<LEVER>.json` (counts line first), `run_mutations_<lever>.py` (steps; the pod runner is `common/pod_mut_driver.py`), `make_spec_<lever>.py` + `spec_<lever>.json`, `PINS_<LEVER>.txt`, `RUN.log`, `pod_out_<lever>.json` (raw driver output).
QHEAD: `qhead_n1/MUTATIONS_QHEAD_N1.json`, `COUNTS_QHEAD_N1.txt`, `PINS_QHEAD_N1.txt`. PINS cover the lever tool, test and my runner files (sha256, path relative to `F:/TanitAD-scratch/v8proof`).

## Temp files left (deletes refused / not attempted)
Dev box: `mutlists/actcons/_p.txt`, `mutlists/leadsup/_p.txt`, `mutlists/actcons/podrun.sh` (copy), `mutlists/common/_p.py _p2.py _p3.py _p4.py`, `mutlists/distrule/_e.py _e2.py _e3.py`, `spec_control.json`, `src.tar`, `src/` (copy of the distrule files); all small.
Pod `/workspace/v8proof/mutlists/`: `actcons`, `leadsup`, `distrule` each with `tree_pristine`, `tree_work` (copies of stack and taniteval, about 0.45 GB each), `_tmp`, out and spec json; `common/`. Nothing else touched; no tree edited; nothing committed or staged.
