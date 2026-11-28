# BUILD_AMEND5: the read of AMEND_HWMPLAN_5 (RA-L1), built 2026-11-17

Spec: `AMEND_HWMPLAN_5.md` (FROZEN, sha256 `de6f7477…`, checked by the launcher). Tier T0. CPU only, zero training. **No SCORE window has had an RA pass.** The only SCORE-file reads are the identity checks on base files (C-READBACK re-plans batch 0 under the base's own solver; C-OFF5 copies).
Folder: `F:/TanitAD-scratch/actrsp/lonplan/amend5/build/` (code, tests, launcher). Outputs so far: `../controls/`, `../train/`, `../readout/TRAIN_STAGE_s{0,1}.json`, `../logs/`.

## Pieces (file:line)
| piece | where |
|---|---|
| stage table, chassis / base / argv builder, `BASE_SOLVER_AMEND5.md` reader (lines `BASE_SOLVER: A0\|P\|P23`, `BRANCH: B1\|B2\|B3`; B3 with A0, B1 / B2 with P or P23) | `amend5_stage.py` (`read_base_solver`, `build`) |
| RA pass, targets (`off l2 l1 neg shuf oracle`), MEAN-src, wrapper, stamps, SOLVER_RA.json | `ra_l1.py` (`ra_pass` 167, `plan_arrays` 186, `mean_assembly` 259, `check_declared` 282, `check_built` 303, `run_arm` 357) |
| controls (pure checks, toy world, per-chassis runner, merge) | `amend5_controls.py` |
| driver: every arm of a stage for one chassis, one process | `amend5_run.py` |
| TRAIN-stage report, SCORE contrasts, labels, four families, `RESULT_AMEND5.json` + `.md` | `amend5_read.py` |
| tests (34), launcher | `test_amend5.py`, `run_tests_amend5.cmd`, `run_amend5.sh`, `amend5.cmd`, `launch_amend5.ps1`, `env_amend5.sh` |
| readout smoke on synthetic arm folders (no model, no RA pass) | `smoke_readout.py` |

## Reachability of `--ra-l1` (section 8.1), file:line in `build/`
argparse `ra_l1.py:276` (`--ra-l1`, choices `off l2 l1 neg shuf oracle`; the stage table spells it into every argv, `amend5_stage.py:111`) → namespace `ra_l1.py:287` (`check_declared`: every flag NAMED, parsed arm equals the declared arm; an unnamed `--ra-l1` is refused) → `run_arm` hands `a.ra_l1` to `plan_arrays` `ra_l1.py:398` → readout loaded and sha-checked `ra_l1.py:64-72` (`load_readout`, bytes hashed then parsed), held in the context `ra_l1.py:155` → `ra_fn` / target `ra_l1.py:230-242` → `ra()` inside the cost `ra_l1.py:174` → `st._optimize` `ra_l1.py:180` (library, `autograd.grad` at `tanitad/models/hjepa.py:978`) → best iterate into `plan_raw[:, :10, 0]` `ra_l1.py:250` → written `ra_l1.py:405`. `check_built` `ra_l1.py:399` then compares the EXECUTED pass (source key actually read, readout sha, mirror, permutation) with the declared one. Default `off` disables the pass: every argv names its value; BASE and RA-OFF name `--ra-l1 off`.

## Controls (section 8.2): results, each with its regression arm. A control passes only if the green arm reads ok AND the regression arm reads RED
Run on both chassis, bases A0, `controls/controls_s{0,1}.json`: **RED: none** (s0 256 s, s1 255 s wall). Merged `controls/CONTROLS.json` ok.
| control | green | regression arm (read RED) |
|---|---|---|
| C-RA-SHA | readout sha256 equals section 2.1 | one flipped byte refuses |
| C-RA-FIT | 8,306 pairs / 141 clips, max dW 2.4e-10 (s0), 1.75e-9 (s1), 0 TRAIN clips in SCORE episodes | refit with one SCORE window's displacement added goes RED; a planted SCORE eid goes RED |
| C-READBACK | batch 0 of A0's four SCORE plan files re-planned from SOLVER.json, bit-equal (s0, s1 files x chassis) | lambda override 10 on A0 s0 reads not equal |
| C-OFF5 | RA-OFF equals every base file on disk (A0 TRAIN s0, SCORE s0 and s1, both chassis) and the wrapper's written file, bit for bit on plan_raw, a1_raw, cost1, iters1, idx, decode_resid | a wrapper that runs the pass while the flag reads off goes RED |
| C-YAW-ID / C-BAND-ID | modes l1, neg, shuf, oracle: yaw knots 1-30 and acceleration knots 11-30 max abs delta exactly 0.0, a1_raw and cost1 untouched | writing knots 1-12 RED; writing yaw RED (each alone) |
| C-GRAD | toy and 16 TRAIN windows: non-zero on acceleration knots 1-10, exactly 0.0 on knots 11-12 and on yaw | RA read after 12 actions: knots 11-12 non-zero, RED |
| C-PLANT (i) | linear toy: the cost is 0 at the source's actions (<= 1e-6), the closed-form flat solution reads the source to 3.6e-15, the library optimiser reaches cost <= 1e-4 and the plan's mean within 2e-3 of the source's | a target formed with the plan's own actions: plan stays at HOLD_c, RED |
| C-PLANT (ii) | 64 TRAIN windows: median final cost 8.3e-10; full TRAIN stage below | a 1-iteration pass: median 2.5e-3, RED |
| C-SHUF | TRAIN-stage sign (below); SCORE check in the readout | a mirrored (sign-flipped) row reads RED (unit test) |
| C-MEAN | zero source departure equals HOLD_c exactly (bit-equal); the guard accepts the source-derived shift | a shift taken from the logged accelerations raises |
| C-L4-RA | argv equals declared, built equals declared (source key, readout sha, mirror, permutation), fresh folder | declared / argv mismatch, unnamed `--ra-l1`, wrong built source, a pre-planted `plans_*.npz`: all refuse |
| C-IN | ckpt, windows, TRAIN latents, readout, spec, base plan files sha256-equal to the record; code of record equal to A0's SOLVER.json at the FULL 64-hex values | a wrong record sha RED |
| C-TRAIN-ISO | TRAIN windows are LON-PLAN's selection set, 0 clips in SCORE episodes | a planted SCORE eid RED |
| C-STRAT | `lonplan_cstrat.py` (B 200 here): non-STRATEGIC cells bit-equal to the code of record, STRATEGIC computed | the record's eval module reads PAUSED and is flagged |
On SCORE, in the readout (INVALID if red): X - BASE exactly 0.0 [0, 0] on `yaw_mae`, `head_err`, per-knot 2-6 s `acc_mae` (`check_score_identity`, mutation tested); RA-SHUF - committed arm separated-worse; RA-OFF equals the base file; the TRAIN-stage checks.
Tests: `test_amend5.py` **34 passed** (every label rule reached, each rule's condition dropped by a mutation hook changes the label; toy controls; stage rules; end-to-end wrapper on the toy). The launcher also runs `test_lonplan`, `test_hwm_plan`, `test_plantest2_controls`, `test_amend4` if present (not re-run by me).

## TRAIN stage (ran now; A0's TRAIN plans, both chassis; 564 windows, 141 clips; IN-SAMPLE for R35 and for RA; decides nothing)
0-2 s LONGITUDINAL `acc_mae`, m/s^2, clip-cluster bootstrap B 2,000 seed 0 (`readout/TRAIN_STAGE_s{0,1}.json`, all families and bands there; headway and STRATEGIC not computable on TRAIN, stated in the file):
| row | s0 | s1 |
|---|---|---|
| BASE (A0) - HOLD | +0.9712 [+0.9171, +1.0244] | +0.9339 [+0.8791, +0.9854] |
| RA-L1 - BASE | -0.4916 [-0.5407, -0.4436] | -0.4247 [-0.4672, -0.3809] |
| RA-L1 - HOLD | +0.4795 [+0.4262, +0.5324] | +0.5092 [+0.4577, +0.5636] |
| MEAN-L1 - HOLD | +0.4511 [+0.3973, +0.5008] | +0.5090 [+0.4571, +0.5609] |
| RA-L1 - MEAN-L1 | +0.0285 [+0.0001, +0.0581] | +0.0003 [-0.0279, +0.0272] |
| RA-SHUF - RA-L1 (C-SHUF sign, must be > 0) | **+0.0769 [+0.0348, +0.1193]** | **+0.0744 [+0.0296, +0.1180]** |
| RA-NEG - RA-L1 | +0.0131 [-0.0503, +0.0778] | -0.0372 [-0.1003, +0.0261] |
| RA-ORACLE - HOLD | +0.5410 [+0.4817, +0.6037] | +0.5856 [+0.5234, +0.6503] |
Median final pass cost RA-L1 7.0e-10 / 8.4e-10, RA-ORACLE 1.5e-7 / 3.6e-7 (bar 1e-4): C-PLANT (ii) holds. Share of windows touching B1: RA-L1 0.115 / 0.121. RA-ORACLE - HOLD reproduces the draft's section 3 row (+0.541 / +0.586) with this build's code. The C-SHUF sign and C-PLANT (ii) hold at both chassis, so the SCORE read is not blocked by them. These are in-sample TRAIN records, never results; MEAN-L1 - HOLD sits near RA-L1 - HOLD, as the draft's expectation of RAL1-NULL / READ-LIMITED leaves open.

## Readout smoke
`smoke_readout.py` ran the whole SCORE readout (contrasts through `contrast_l1full_strat.py`, labels, DIR-ACC, mechanism rows, md) on SYNTHETIC arm folders (A0's banked plans with a constant nudge, no model, no RA pass, B 100) in a throwaway root: it ran end to end and reached the INVALID path (C-SHUF not separated on the nudged copies). Numbers meaningless and not kept; the 207 MB root was deleted.

## Launcher
`launch_amend5.ps1` (WMI, CreateFlags 0x08000200, `cmd /c amend5.cmd < NUL`) -> `amend5.cmd` (`C:\Program Files\Git\usr\bin\bash.exe`) -> `run_amend5.sh` (`export PATH="/usr/bin:/bin:$PATH"`). Stages: 0 spec sha, `BASE_SOLVER_AMEND5.md` present and consistent, tests; 1 controls (pair of chassis when commit >= 11 GB else serial, each gated at >= 5.5 GB) with every base whose SCORE files exist; 2 TRAIN stage (arms already planned are skipped) and its checks; 3 SCORE arms (`amend5_run.py plan --stage score`); 4 readout. Last line of `amend5/read.log`: `ZZ AMEND5 READ DONE ZZ` or `ZZ AMEND5 READ FAIL ZZ <why>`. **The MM must write `BASE_SOLVER_AMEND5.md` with the two machine lines first.**
Wall-time estimate (ESTIMATED, +-40 %): controls 4.5 min per chassis for A0, plus about 2 min per extra P / P23 file; TRAIN stage about 4 min per chassis (measured: 47 s for the first arm with load, 10-30 s for the rest); SCORE arms about 8 min per chassis (the draft's 0.042-0.046 s per window x 994 x about 10 arm-seeds, plus 15 s load per run); readout contrasts about 9-12 per chassis at B 2,000, ESTIMATED 25-45 min in all (smoke: 40 s each at B 100). About 1-1.3 h in total.

## Stated deviations and choices where the spec is silent
1. **C-PLANT (i) tolerance.** The spec asks the pass to reach the target read "to 1e-6". The library optimiser at level-1 settings stops on its patience rule before that, even on a linear toy (read error 3e-3 to 4e-3, cost 1.8e-5). Measured, not tuned. The toy check therefore demands cost <= 1e-4 (the real-model bar of C-PLANT ii) and plan mean within 2e-3 of the source's, and demands **1e-6 only of the analytic identities** (cost exactly 0 at the source's actions; the closed-form flat solution reads the source to 3.6e-15).
2. **C-IN pin of `hwm_plan.py`** (MM instruction): pinned to the FULL 64-hex values from LON-PLAN's `score/A0_s0/SOLVER.json` (`amend5_stage.CODE_SHA_FULL`, `5a0722719f7ce06c94c91213db6ef542b2fe16d346622ba506ce8808bd6d714b`), not the card's `5a072272…`, whose 8th hex digit is a slip. Stricter than AMEND_4's 7-digit prefix.
3. **RA-NEG / RA-SHUF source.** The spec says "the committed arm"; they take the committed arm's source (l1 under B3, l2 under B1/B2), both in the stage table.
4. **RA-ORACLE on the base's seed-1 file too** (spec lists no seed for it): needed to apply the seed-1 clause of "separated-better beyond floor"; its floor is max(|ORACLE seed 1 - seed 0|, |BASE seed 1 - seed 0|). **CEIL-ORACLE holds iff RA-ORACLE - HOLD is separated-better at both chassis**; section 6's READ-LIMITED row says "not separated-better at either chassis", read here as: the flag fails.
5. **F_I** for the X - HOLD, X - BASE and X - MEAN cells is the one of the X - BASE contrast (max of X's and BASE's seed 1 - seed 0 on that metric); the guard G_x uses `xtrack`'s own floor.
6. **Yaw source of the pass:** the base's level-1 yaw (a1_raw / ch1_scale) on both sides, as section 2.2; RA-ORACLE uses HOLD_c's yaw as section 4 says. HOLD_c clipped to B1 is the start; a start knot sitting on the B1 edge counts in the HIT share.
7. **MEAN-src** is assembled in raw units: HOLD_c acceleration + (source mean - HOLD_c mean over knots 1-10), clipped to B1. At a zero departure it equals HOLD_c bit for bit. The guard refuses a shift that is not the source-derived one.
8. **Pass batching:** 64-window batches as the prototype; the optimiser's early stop reads the batch mean, so a window's iterate depends on its batch (deterministic for a fixed file and batch layout).
9. **Seeds:** lever arms, MEAN and ORACLE on the base's seed 0 and 1 files; OFF, NEG, SHUF seed 0. TRAIN: seed 0 (the base's TRAIN files have one).
10. **Readout names:** `BASE_SOLVER_AMEND5.md` format (above) is my choice; the MM writes it. P / P23 TRAIN-stage arms and RA-L2 need `l2_prim_raw` in the files (`ra_l1.run_arm` refuses without it; AMEND_4's wrapper writes it and checks it). RA-L2 beside on P under B3 is planned only when P's SCORE files exist, with `--beside` (refused for any other combination).
11. Not done: no P / P23 file was available (AMEND_4's arms were not yet planned), so C-READBACK, C-OFF5 and the TRAIN stage ran on A0 only; the launcher repeats the controls for every base whose files exist. The readout `score` path has been run on synthetic folders only.

## Deliverable manifest
| artifact | where |
|---|---|
| code, tests, launcher, this report, `MANIFEST_sha256.txt` | `F:/TanitAD-scratch/actrsp/lonplan/amend5/build/` (scratch only; **not yet in the repo, not staged**) |
| controls JSON | `F:/TanitAD-scratch/actrsp/lonplan/amend5/controls/` |
| TRAIN-stage plan files (10 folders) and reports | `.../amend5/train/`, `.../amend5/readout/TRAIN_STAGE_s{0,1}.json` |
| readout blobs used by the build (blob `7fcafdfb` / `95a43146`, sha256-equal to section 2.1) | `.../amend5/readout_in/` |
Escalation: the build needs to be copied into the repo (suggest `raw/2026-11-17-ra-l1-build/`) and staged by the MM; it exists on one disk only.

## Deletions (own temporary data, literal paths)
`.../amend5/smoke_root` (207 MB, synthetic, re-creatable), `.../amend5/build/tmp/smoke1` (9.4 MB), `.../amend5/build/tmp/bt` (0.8 MB), `.../amend5/build/tmp/toy_try.py` (4 KB): all deleted. `.../amend5/build/tmp/torchinductor_Admin` (0 bytes) left.
