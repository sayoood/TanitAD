# PRE-REGISTRATION, ROUND 2: the RL-S1 fan-equivalence bound on the per-set MEAN |delta| statistic, on DISJOINT keys

Written 2026-10-12 21:18 (dev-box local clock, from `date`), BEFORE any round-2 dump, prep output or number exists. Rulings of the Master Mind (2026-10-12) are written in as given.
Evidence class of everything this file will later report: MEASURED (ours). This file contains the rule only; the results section at the bottom is filled in afterwards and never edits the rule.

## 1. Why a second round, and what it can and cannot confirm

Round 1 (`../README.md`, `../verdict.json`) pre-registered a rule on the per-set MAX delta and chose no tolerance: 5 of 6 decision arms missed the 10x bar and the floor is one bf16 ulp (0.25) in every null.
A POST-HOC look (`../posthoc_mean_abs.py`, `../posthoc_mean_abs.json`) found that the per-set MEAN |delta| over the x/y values has a floor of 0.012193 over N_full (1,365 sets) and separates the detectable arms.

* The statistic (mean |delta|), the factor 2, the 25 % and the 5x below were chosen AFTER seeing round 1's decision-arm distributions. Round 1's data therefore cannot confirm them; only keys that round 1 never saw can.
* Round 2 changes the KEYS and nothing else: every arm is the same defect, the same seeds, the same variants, the same reference sets and the same dump path as round 1 (`run_calibration.py` is imported unchanged; see section 10).
* A round-2 PASS is evidence about ONE statistic on ONE 48-key sample per arm on the dev-box RTX 4060 against the A40's banked sets. It is not an equivalence proof (section 9).

## 2. Sample (frozen)

* Keys: `rows[14::28][:48]` of `D:/Projects/TanitAD-artifacts/_rlfan_work/identity/keys.tsv` (header row excluded; 1,365 rows). Round 1 used `rows[::28][:48]`.
* Disjointness is asserted by code before the dump starts (`run_calibration2.py --phase prep` aborts on any shared (token, log) pair with round 1's `keys48.tsv`). The measured overlap count is written into section 12.
* Reference: `sets_10075_proxy.jsonl` (the A40's banked step-10,075 sets); its sha256 must equal round 1's `d2c4e47468ee3ba7b6c07c30ed68c496e26dfdd483387572323a7161106832d7` (`../prep_manifest.json`).
* Statistic of a set: **m = (1 / n) * sum |new - ref| over every x and every y value of the set (yaw excluded)**; this is element 3 of `run_calibration.arm_deltas()` and the quantity `posthoc_mean_abs.py` tabulated.

## 3. Arms (definitions as in round 1, `../run_calibration.py` `ARMS`)

| arm | role | defect |
|---|---|---|
| N0_b4 / N1_b1 / N2_b2 | NULL | correct dumper, batch 4 / 1 / 2 (round-2 keys) |
| N_full | NULL | round 1's identity-run dump, ALL 1,365 keys, CPU re-analysis (no new GPU work): max m 0.012193 |
| A2_calib_half_scale | DECISION | intrinsics fx fy cx cy x 0.5 |
| B1_img_hflip | DECISION | all four cameras mirrored left-right |
| B2_img_bgr_rgb | DECISION | channel order swapped on all four cameras |
| C1_head_noise10 | DECISION | `traj_head.2` weight + N(0, (0.1 std(W))^2), seed 20261012 (the same noise draw as round 1) |
| C2_snap_epoch021 | DECISION | another banked checkpoint, step 7,348 (judged offline with the step relabelled to 10,075, as in round 1; the CLI refuses it with exit 4) |
| A1_calib_other_vehicle | INFORMATION (demoted, section 6) | each log's rig replaced by another vehicle's rig |
| N3_jpeg | INFORMATION | every image re-encoded at JPEG q95, nothing else |
| S_focal105, S_yaw3, S_head_noise1, S_snap027, S_wrong_scene, S_cam_reversed | INFORMATION | the round-1 sensitivity arms (S_wrong_scene cannot resolve frames and is expected to exit 4 with no sets, as in round 1) |

The identity run's five 24-key invariance dumps (N_b1, N_b2, N_b4, N_b4shift, N_b4w2) are NOT null arms of round 2. The analysis prints their max m for the record; they cannot move M or the verdict.

## 4. The bound (computed by code, from nulls only)

* F = the largest m over every set of the four NULL arms (N0_b4, N1_b1, N2_b2 on the round-2 keys, and N_full): "computed from round-2 nulls AND N_full together".
* **M = 2 x F, rounded UP to a multiple of 0.005** (`math.ceil(2F / 0.005 - 1e-9) * 0.005`, then `round(., 6)`).
* "Above M" means strictly m > M. M depends on the null arms only, never on a decision arm.

## 5. Decision rule (applied by `run_calibration2.py --phase analyse`, not by hand)

ACCEPT M (verdict PASS) iff ALL of:

1. **0 null sets above M** (every one of the four null arms). This is true by construction of M (M >= 2F > F), so it is an integrity check on the code, not evidence.
2. **Every DECISION arm has >= 25 % of its sets above M** (at least ceil(0.25 x 48) = 12 of 48).
3. **Every DECISION arm's worst set has m >= 5 x M.**

Decision arms: A2, B1, B2, C1, C2. Otherwise the verdict is FAIL and `CHOSEN_MEAN_ATOL` stays null.

## 6. Master Mind ruling on A1 (2026-10-12, decided BEFORE round 2; stated verbatim)

> A1 (another vehicle's calibration) is DEMOTED to an information arm, decided before round 2: in round 1 its per-set max delta equalled the 0.25 floor and its worst m was 0.021 vs the 0.012 null floor, so it does not move the fan beyond run-to-run noise; RL-S1's labels are computed on the dumped fan, so a fault that does not move the fan cannot corrupt RL-S1's data. It is still run and reported; it does not decide.

Scope note (mine, not part of the ruling): PASS therefore certifies a gate on the fan's movement. It does not certify that the calibration table is the right vehicle's.

## 7. Information arms

A1, N3_jpeg and the S_* arms are run and reported (sets above M, worst m / M, quantiles). They decide nothing, in either direction. Any sentence of the verdict that leans on one of them for a pass or a fail is out of bounds.

## 8. Verdict states and what voids a run

* **PASS** / **FAIL** as in section 5.
* **INCOMPLETE** (no M, no verdict): a NULL or DECISION arm is missing; an arm has other than 48 matched sets (N_full other than 1,365); or the reference sha256 differs from round 1's. The remedy is `--phase dump --only <arm>` and a re-analysis; no rule value changes. Nothing else voids a run.
* The CLI's own `--compare-atol 0.5` exit codes (kept exactly as in round 1, `--compare-min-matched 40`) and `cli_vs_analysis_max_delta_equal` are recorded and reported, but they do not decide. A `cli_vs_analysis_max_delta_equal` of False must be named in the first line of the report.
* GPU, torch and checkpoint hashes are recorded per arm and compared with round 1's; a difference is reported, not voiding.
* The regenerated perturbed checkpoints (`ckpt_head_noise10.pt`, `ckpt_head_noise1.pt`) are compared by sha256 with round 1's `prep_manifest.json` (`1bc91b07...`, `4860499f...`); equal or not is reported, not voiding.

## 9. Outcomes committed in advance

* **PASS ->** build `--compare-mean-atol M` in `refe/rl_fan_dump.py` (a set fails when its mean |delta| over x/y exceeds it; same exit 5, same markers, tests plus RED mutations as for `--compare-atol`), keep `--compare-atol 0.5` as the catastrophe bound, pin BOTH in the chain (`CHOSEN_*` in a verdict.json that the chain test cross-checks; extend `check_fancmp`), update runbook D-RLS1-5. The pod's own floor (A40 vs the A40's banked sets) is still unmeasured; the first `fancmp` run on the A40 writes `delta_quantiles` and M is to be tightened from it. PASS is not "the fan is calibration-identical" (section 6) and not "the pod is equivalent".
* **FAIL ->** report each decision arm that does not separate, with its count above M, its worst m / M, and the largest M' that would still separate it (the smaller of the arm's 12th-largest m and its worst m / 5), and whether any M' exists above 2F. No tolerance is chosen. The numbers 25 %, 5x, 2x and 0.005 are NOT re-tuned on round-2 data. A third round needs its own pre-registration on a third key set (`rows[7::28]` or `rows[21::28]`).

## 10. Instrument

`run_calibration2.py` (this directory, frozen before the dump; its sha256 is in section 12) imports `../run_calibration.py` unchanged and overrides only `pick_keys` (`rows[14::28][:48]`), the work directory (`D:/Projects/TanitAD-artifacts/_rlfan_work/atol_cal2`, never `atol_cal`) and the analysis (section 5). Prep, the per-arm dumps and the CLI invocation are round 1's own functions.
Each arm blocks on `D:/Projects/TanitAD-artifacts/_rlfan_work/wait_idle.py` and the dump aborts (exit 7) on a non-zero exit from it.
Python for GPU work: `C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe`, `PYTHONIOENCODING=utf-8`, `TMP`/`TEMP` on D:. `D:/Projects/TanitAD/data/refe_proxy/pixels` is read-only throughout.

## 11. Honest limits (known before the run)

* One checkpoint, one noise draw (the same one as round 1), 48 sets per arm: round 2 tests new KEYS, not new training runs or new noise draws.
* The dev-box floor is a cross-GPU floor (RTX 4060 vs the A40's banked sets); the pod's own floor may be far lower, in which case M is loose for the pod.
* The statistic and thresholds came from round 1's decision-arm data (section 1); round 1 is not evidence for them.
* N_full contains the round-2 keys among its 1,365 sets, so the floor is partly in-sample for those keys. That can only raise F and make the rule harder to pass.

## 12. Results section (filled in AFTER the fact; it never edits sections 1-11)

Filled 2026-10-12 21:25, after `--phase prep` and BEFORE the dump started (no round-2 delta or m value existed):

* **Key overlap with round 1: 0 (token, log) pairs** (0 by token alone as well), asserted by code in `--phase prep`, which aborts otherwise. Caveat that the assertion does not cover: **3 log names (scenes) are shared** with round 1, always with different tokens, so the two samples are disjoint in keys but not fully disjoint in scenes.
* Round-2 keys: `keys48_round2.tsv` sha256 `26a009ddbce6b4ef00a03d89c8f1557b5b30b5ac6e1d83109e10a7f2f37b97ec` (round 1: `ea206daab13b8d37fa72df1bb303973db502a504861b50cd741b346990d4319a`).
* `run_calibration2.py` sha256 at freeze: `68395086552ef87fc22ac69850dae5863ec7707281d72676e7ab0346011f4083`; `../run_calibration.py` (imported unchanged) sha256 `9db9ca1e0e4f1d2c5201f8c0139533419ea7c1d9961ecb21f55c35ae9df4b712`.
* Reference sets sha256 equals round 1's (asserted in prep). The regenerated perturbed checkpoints are byte-identical to round 1's (`ckpt_head_noise10.pt` 1bc91b07..., `ckpt_head_noise1.pt` 4860499f...: `variants_equal_round1` true for both) and the four calibration variants are byte-identical too (`calib_variants_equal_round1` true for all four). So the arms are the same defects as round 1; only the keys changed.
* Code self-tests, run before the dump and decided nothing: (1) the wait_idle guard (`--phase selftest`): a waiter exiting 3 aborts the dump with exit 7 and the dumper is never reached; a waiter exiting 0 reaches the dumper (control). (2) The analysis code was exercised on ROUND 1's dumps (`--a-dir .../atol_cal`, output outside the repo, labelled "CODE SELF-TEST"); it reproduces `../posthoc_mean_abs.json`'s counts. Its PASS on round 1's data is in-sample and carries no weight.
* Dump start / marker: see `HANDOFF.md`.
* Verdict: NOT YET RUN.
