# v8F rung alpha, arm alpha-1 (pose-only P10) -- RESULT (pre-check + build + gate; nothing launched)

2026-10-17, CPU only (`CUDA_VISIBLE_DEVICES=-1`), base commit `c415f1b2` (the mirror tip `f6f96d6a` differs from it in none of the files
this hold modifies). Design, decisions and the reachability trace: `DESIGN.md`. Argv pair: `argv_alpha0_alpha1.md`.

## HEADLINE -- the pre-check: alpha-1's premise HOLDS for the pose-only P10 at 2 s; at 6 s refcv7 is better

> **P10-pose beats refcv7 at 2 s.** On the S1 eval139 grid (T1 self-action open loop; n 4,754 windows / 139 episodes; paired
> episode-cluster bootstrap B 2000, seed 0), `os - P10pose` ADE 0-2 s is **+0.0221 [+0.0105, +0.0346] SEPARATED** at inference seed 0
> and **+0.0227 [+0.0106, +0.0350] SEPARATED** at seed 1 (S1's steer P10 premise was +0.0276 [+0.0163, +0.0398]; the pose-only
> floor keeps **80 %** of it). The **shipped 30-knot artifact** itself reads **+0.0178 [+0.0056, +0.0308] SEP** (seed 0) / +0.0184
> [+0.0060, +0.0310] SEP (seed 1) -- **64 %** of the steer figure. The premise is **longitudinal**: LON speed MAE **+0.0351 [+0.0246,
> +0.0473] SEP**, along-track MAE **+0.0265 [+0.0179, +0.0366] SEP**, accel MAE +0.0696 [+0.0546, +0.0863] SEP (shipped artifact: speed
> +0.0280 SEP, along +0.0253 SEP, accel +0.0334 SEP). Positive = P10-pose has the smaller error.
>
> **It does not carry to 6 s.** `os - P10pose` ADE 0-6 s is **-0.4390 [-0.6359, -0.2441] SEP** (shipped artifact -0.4898 SEP): refcv7 is
> better by ~0.44-0.49 m, and the gap is LATERAL (cross-track -0.63 m SEP, heading -2.06 deg SEP; LON speed/along-track not separated).
> The pose-only P10 still beats alpha-0's own prior (`ha0_ext_pose`) at both horizons: **-0.0378 [-0.0477, -0.0281] SEP @ 2 s,
> -0.5699 [-0.7719, -0.3713] SEP @ 6 s** (shipped artifact -0.0335 SEP / -0.5192 SEP).
>
> **What this is and is not.** It is a statement about P10-pose as a *floor* (the prior alone, Delta = 0) against refcv7's *trained*
> model. It licenses spending Thor time on the swap (the prior is a better start than alpha-0's, by a margin that is separated
> at both horizons and is longitudinal at 2 s). It does **not** say alpha-1 will beat alpha-0 once trained: that is read on the
> trained arms against the replicate floor. The intervals answer *"another draw of EPISODES"* only (models fixed). "Seed 1" re-scores
> the same deterministic P10 floor against refcv7's SECOND banked inference-seed dump (the two agree to 0.0006 m); neither interval
> says anything about TRAINING-seed variance.

**Verdict in one paragraph.** The refcv8 chassis now builds, trains (CPU smoke: forward, loss, backward, stamp, checkpoint, resume
guard) and evaluates with its residual prior swapped to the frozen **pose-only** P10 (`--residual-prior p10_pose`), `Delta == 0`
rolls **exactly** to P10u, and `ADE6(P10u, P10) = 0.0204 m` at S2 / 0.0198 m at S6 (bar 0.05 m, **met**). The launch gate has a `v8f`
profile (alpha-1) and a `v8f_steer` profile (the optional alpha-1s) with clauses on five checks, each with deliberate-regression
arms; **alpha-1: 21 of 21 arm rows go RED, 6 of 6 controls GREEN; alpha-1s: 19 / 19 RED, 6 / 6 controls GREEN.** `v8f` REFUSES
`--residual-prior p10` by name ("steer at inference: no PI ruling (SPEC_REFCV7 A5)"). The token **cannot PASS yet**: one open item
(`V8F-RUNG-GEOMETRY`, Master Mind, set at launch). No trained arm exists, so there is **no driving result**.

## 1. What was measured (evidence class MEASURED unless stated)

### 1.1 The pre-check (tier T1, self-action open loop; estimator: paired episode-cluster bootstrap, B 2000, seed 0, `taniteval/ci.py`)

Arms: `os` = refcv7-50,400 (banked T1 dump, `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-17-p10-floors-refcv/`); `P10pose` = the
pose-only P10 with the **S1 recipe** (per-horizon fit on the S1 refcv grid, refit on the 4,572-clip train view: S2 = 4 steps x 0.5 s,
S6 = 6 steps x 1.0 s); `P10pose_artifact` = **the shipped 30-knot artifact** scored on the same grid; `ha0_ext_pose` = alpha-0's prior;
`P10` = S1's steer P10 (control: reproduces S1's +0.0276). Windows: `t0 = 7 + (e mod 5) + 5j`, the eval139 dump `ws`. Raw:
`raw/precheck/` (`eval_S{2,6}_s{0,1}[_art].json`, `analysis_S{2,6}_s0[_art].json`, `tables.md`, `summary.json`; code `code/precheck_*.py`).

Mean ADE (m): S2 -- `os` 0.2784, `ha0_ext_pose` 0.2941, steer `P10` 0.2508, `P10pose` **0.2564**, `P10pose_artifact` 0.2606.
S6 -- `os` 2.5675, `ha0_ext_pose` 3.5764, steer `P10` 2.9736, `P10pose` **3.0065**, `P10pose_artifact` 3.0572.

Paired ADE / FDE, 0-H s (delta = x - y; **positive = y has the smaller error**):

| H | pair | seed | ADE [95 % CI] | FDE [95 % CI] |
|---|---|---|---|---|
| 2 s | os - P10pose | 0 | **+0.0221 [+0.0105, +0.0346] SEP** | +0.0144 [-0.0168, +0.0466] |
| 2 s | os - P10pose | 1 | **+0.0227 [+0.0106, +0.0350] SEP** | |
| 2 s | os - P10pose_artifact | 0 / 1 | +0.0178 [+0.0056, +0.0308] SEP / +0.0184 [+0.0060, +0.0310] SEP | +0.0093 [-0.0232, +0.0428] |
| 2 s | os - P10 (steer, control) | 0 | +0.0276 [+0.0163, +0.0398] SEP | |
| 2 s | P10pose - ha0_ext_pose | 0 | -0.0378 [-0.0477, -0.0281] SEP | -0.0621 [-0.0843, -0.0414] SEP |
| 2 s | P10pose_artifact - ha0_ext_pose | 0 | -0.0335 [-0.0425, -0.0246] SEP | -0.0570 [-0.0772, -0.0386] SEP |
| 2 s | os - ha0_ext_pose | 0 | -0.0157 [-0.0321, +0.0021] not separated | -0.0477 [-0.0894, -0.0053] SEP |
| 6 s | os - P10pose | 0 | **-0.4390 [-0.6359, -0.2441] SEP** | -1.2442 [-1.7666, -0.7381] SEP |
| 6 s | os - P10pose_artifact | 0 | -0.4898 [-0.6921, -0.2894] SEP | -1.4166 [-1.9602, -0.8846] SEP |
| 6 s | P10pose - ha0_ext_pose | 0 | -0.5699 [-0.7719, -0.3713] SEP | -1.6952 [-2.2611, -1.1276] SEP |
| 6 s | P10pose_artifact - ha0_ext_pose | 0 | -0.5192 [-0.7093, -0.3306] SEP | -1.5227 [-2.0521, -0.9928] SEP |
| 2 s | P10pose_artifact - P10pose (refit) | 0 | +0.0043 [+0.0024, +0.0061] SEP | |

The shipped artifact (30 knots, `v8f_mini` recipe) is slightly worse than the per-horizon S1-recipe refit (+0.0043 m @ 2 s, +0.0507 m @
6 s), as expected of a single fit serving every horizon; it is the artifact alpha-1 uses, so the artifact rows are the operative ones.

Four families, `os - P10pose` (S1-recipe refit, seed 0; for error metrics positive = P10pose better; **for TAC `*_correct` positive =
refcv7 better**). STRATEGIC: **PAUSED** (PI ruling 2026-09-19, `PI_DECISION_QUEUE.md` ITEM 25) -- not a pass, not omitted.

| family | metric | 2 s | 6 s |
|---|---|---|---|
| LONGITUDINAL | speed MAE (m/s) | **+0.0351 [+0.0246, +0.0473] SEP** | -0.0255 [-0.0798, +0.0344] |
| LONGITUDINAL | along-track MAE (m) | **+0.0265 [+0.0179, +0.0366] SEP** | +0.0075 [-0.1183, +0.1437] |
| LONGITUDINAL | accel MAE (m/s^2) | +0.0696 [+0.0546, +0.0863] SEP | -0.0429 [-0.0689, -0.0176] SEP |
| LONGITUDINAL | distance keeping: headway / time-gap / min-TTC | +0.0012 / -0.0179 / -0.144, none separated (n 1,250 / 1,124 / 1,250) | +0.0081 / -0.0139 / -0.258, none separated (n 927 / 876 / 927) |
| LATERAL | cross-track MAE (m) | -0.0035 [-0.0136, +0.0058] | -0.6309 [-0.8008, -0.4746] SEP |
| LATERAL | heading MAE (deg) | -0.1148 [-0.2090, -0.0192] SEP | -2.0612 [-2.5737, -1.6100] SEP |
| LATERAL | yaw-rate MAE (rad/s) | -0.0063 [-0.0083, -0.0044] SEP | -0.0198 [-0.0241, -0.0158] SEP |
| TACTICAL | lateral-class correct | +0.0263 [+0.0164, +0.0364] SEP (refcv7 better) | +0.0984 [+0.0788, +0.1187] SEP |
| TACTICAL | longitudinal-class correct | -0.0278 [-0.0422, -0.0141] SEP (P10pose better) | +0.0772 [+0.0403, +0.1130] SEP |

Reading: at 2 s P10-pose wins the LONGITUDINAL family (speed, along-track, accel; distance keeping is a tie) and loses lateral
heading / yaw-rate; at 6 s refcv7 wins the lateral family by a wide margin. That is the S1 picture for the steer P10, with the pose
floor weaker laterally than it is longitudinally -- its curvature is a noisy pose-derived estimate (`POSE_KAPPA_CAP` 0.3). The same
rows for the shipped artifact are in `raw/precheck/tables.md` (S2 speed +0.0280, along +0.0253, accel +0.0334, all SEP).
`P10pose - ha0_ext_pose` (the swap itself): 2 s speed -0.0311 SEP, along -0.0352 SEP, cross-track -0.0084 SEP, yaw-rate +0.0000 n.s.;
6 s speed -0.1373 SEP, along -0.4654 SEP, cross-track -0.2270 SEP.

### 1.2 The build

| quantity | value | estimator / n | artifact |
|---|---|---|---|
| `ADE6(P10u_pose, P10_pose)`, eval139 **S2** windows | **0.0204 m** [0.0159, 0.0252]; median 0.0089, p95 0.0722, p99 0.1915, max 1.82; 7.6 % of windows > 0.05 | per-window mean, episode-cluster bootstrap B 2000 over 139 episodes; n 4,754 | `raw/measure_alpha1_pose_floor.json` |
| the same, **S6** windows | **0.0198 m** [0.0147, 0.0259]; median 0.0087, p95 0.0722, max 1.82; 6.2 % > 0.05 | same; n 3,668 | same |
| bar (brief) | `<= 0.05 m`; **met on both**; no escalation | | |
| (alpha-1s, steer P10-600) `ADE6(P10u, P10)` | 0.0238 m on both grids (S2 [0.0180, 0.0305], S6 [0.0169, 0.0320]); bar met | same | `raw/measure_alpha1_floor.json` |
| in-decoder P10-pose vs the fit script's own predictions (known value) | max error **1.5e-5 m** on 565 real eval141 windows (bar 1e-4) | max over windows | `raw/mutation_matrix_pose.json` (CONTROL G-CLOCK) |
| the same, window ending ONE frame early | beyond 1e-4 m on **98.05 %** of windows | | same |
| `p10_clock_shift` / `p10_pose_a0_stride2` arms | RED: up to 21.2 m / 10.9 m on 554 of 565 windows | | same |
| `Delta == 0` vs P10u, through `roll_plan` | max **0.0 m** (565 windows) | | same |
| conversion: curvature cap 0.3 / 0.5 / **1.0** / 2.0 (pose P10, S2) | 0.0385 / 0.0290 / **0.0204** / 0.0163 m | n 4,754 | `raw/measure_alpha1_pose_floor.json` |
| conversion (steer P10): open-loop vs closed-loop inverse | `ADE6` 0.389 m vs 0.0238 m | S2 | `raw/interp_variants_final.json`, `sens_cap_floor_final.json`, `smooth_variants_final.json` |
| refit: P10-pose, 4,572 clips, stride 8, 41,139 rows | fit 129.7 s CPU; `ADE6` on eval141 (n 2,399) 2.4871 m vs `ha0_ext_pose` 2.8519 m, `kd_x_pose` 2.9444 m, steer P10-600 2.5744 m | | `raw/p10_pose/p10_fit.json` |
| fit clips vs eval139 / eval141 | **overlap 0 / 0** by clip id and by stable episode id (4,572 fit clips); asserted BEFORE the fit | exact set tests | `p10_fit.json`; `test_p10_pose_prior.py`; G-CLOCK |
| artifact identity | `p10_pose_prior.pt` 74,604 B, sha256 `e70a9ae30b4418705442bffdeb62d9810a43b0a6ccce0c177796c72295fe693d`, content sha256 `7ba84874...ff20` | | `raw/p10_pose/`, `stack/tanitad/configs/v8f_p10/` |
| deliberate-regression matrix, alpha-1 (`v8f`) | **21 / 21 arm rows RED, 6 / 6 controls GREEN, 0 regex mismatches** | CPU, tiny model on real windows | `raw/mutation_matrix_pose.json`, `.log` |
| the same, alpha-1s (`v8f_steer`) | 19 / 19 RED, 6 / 6 GREEN, 0 mismatches | | `raw/mutation_matrix_steer.json`, `.log` |

Interval estimators: the `ADE6(P10u, P10)` intervals answer *"another draw of EPISODES"* for a **deterministic** function of the
window (there is no training or inference seed in it); they are not a model comparison.

## 2. Tests (CPU)

* `tests/test_p10_pose_prior.py` **23 passed** (new): the artifact sha literal, variant refusals by name, disjointness, the port equals
  the banked fixture (<= 1e-4 m) and an independent numpy reference, the triple IS `ha0_ext_pose`'s derivation, a one-frame-early
  window FAILS, past-window-only (no steer, supplied actions refused), frozen, `ADE(P10u, P10) <= 0.05` and `Delta = 0` exact, the mode
  registry, stamp / variant, the model build + reachability + mutations, the G-DVB variant clause.
* `tests/test_p10_prior.py` (steer + shared; helpers made variant-aware), `tests/test_launch_gate_v8f.py` (**56**, rewritten as a
  parametrisation over both variants through a `Var` class): the profile, the argv pair, G-DVB / G-LIVE / G-CKPT / G-EVAL / G-CLOCK
  with arms, the loader remap, the arms runner. Counts per file from the merged full-suite junit are in sec. 6.
* `tests/test_residual_prior.py` (the `ACTION_MODES` pin is unchanged: `p10_pose` is a POSE mode) and `tests/test_declared_vs_built.py`
  (registry size 276 -> 278, commented) are the only pre-existing tests touched.

## 3. Reachability (CLAUDE.md "built, tested, unreachable")

All hops are traced in DESIGN.md sec. 6. MEASURED outcomes: the P10 module is called on every training forward; perturbing the live
P10 buffers changes the loss by > 1e-4; the residual heads take a non-zero gradient while every P10 tensor has `grad is None`
(exactly 0); the *unfrozen* mutation reads non-zero (so the zero is a reading); the fallback-to-`ha0_ext_pose` mutation is caught
(the module is called 0 times, and the emitted prior disagrees with the independent recompute); a P10 of the **other variant** is
caught at G-DVB (build) and at G-EVAL (loader) by name. **Defaults checked:** `--residual-prior` defaults to `off` and `--p10-prior*`
to `None`, so the gate's `required_flags` and `required_values` make the launch argv NAME all three (a launch that omits them is
refused, with an arm).

## 4. DIAGNOSTIC (ADE only -- **not an eval result**; no trained arm; T0 plan against ground truth, eval139)

What alpha-1's plan *is* before the residual heads move it (`raw/measure_alpha1_pose_floor.json`). **ADE alone: a build diagnostic;
the four-family read of the prior is sec. 1.1.**

| mean ADE vs GT (m) | S2 (4,754 w; 0.5-2.0 s) | S6 (3,668 w; 1-6 s) |
|---|---|---|
| **P10u (pose; shipped artifact) = alpha-1 at Delta = 0** | 0.2614 | 3.0598 |
| `ha0_ext_pose` prior = alpha-0 at Delta = 0 | 0.2941 | 3.5764 |
| `ha0_ext` (echo) | 0.2886 | 3.5479 |
| S1's steer P10 (horizon-specific; **not loadable as a per-tick prior**) | 0.2508 | 2.9736 |
| `os` = refcv7-50,400 (banked T1 dump, inference seed 0) | 0.2784 | 2.5675 |
| *(alpha-1s start, the steer P10-600's P10u)* | *0.2670* | *3.1739* |

Paired (arm minus P10u_pose; positive = P10u better; episode-cluster bootstrap B 2000, seed 0):

| arm | S2 | S6 |
|---|---|---|
| `ha0_ext_pose` prior | +0.0327 [+0.0235, +0.0420] SEP | +0.5166 [+0.3263, +0.7074] SEP |
| `ha0_ext` | +0.0272 [+0.0181, +0.0362] SEP | +0.4880 [+0.3013, +0.6706] SEP |
| S1's steer P10 | -0.0106 [-0.0135, -0.0077] SEP | -0.0862 [-0.1210, -0.0521] SEP |
| `os` | +0.0170 [+0.0046, +0.0299] SEP | -0.4924 [-0.6942, -0.2914] SEP |

Reading (HYPOTHESIS until a trained arm is read): alpha-1 starts from a plan that is separated-better than alpha-0's at both horizons
and separated-better than `os` at 2 s; it trails `os` by 0.49 m at 6 s, which is what the residual heads are for. The two priors differ
as plans by 0.143 m (S2) and 2.09 m (S6) mean displacement and as slot-mean controls by 0.378 m/s^2 (accel) and 0.0119 1/m (curvature)
on average: the warm-started residual heads (D4) meet a visibly different `P` at step 0.

## 5. Open items and the cheapest next lever for each (RULE ZERO: none of these is a finished refutation; there is no refuted arm)

| item | owner | what closes it | cheapest next |
|---|---|---|---|
| V8F-RUNG-GEOMETRY (D2) | Master Mind | the tiny rung's `--steps/--batch/--eval-*` + the B1 guard, written into `required_values` of `v8f` and the same values in both argvs; the token is issued then | the pair is runnable as shipped at the full-run placeholder |
| V8F-P10-ACTION-INPUT | PI | a ruling that steer[t0] is admissible at inference | none for alpha-1 (pose-only). Opens optional arm alpha-1s only |
| (D4) continuation tree | Master Mind | if alpha-1 does not clear its bar: (i) cold-start the heads, (ii) `--r8-lat-prior-dropout 0`, (iii) both | each is one flag off alpha-1; launch in the same turn that alpha-1 reads short |
| 6 s lateral gap vs refcv7 (-0.44 m) | -- | the residual heads must close it; the pose curvature (cap 0.3, noisy) is the likely limit | read the trained alpha-1's lateral family; if it trails alpha-0 laterally, test a smoothed pose curvature in `pose_kin3` |

Closed by the amendment: **D1** (steer: not an alpha-1 input, by construction), **D3** (refit banked and sha-pinned), **D5** (repo-relative
path, sha is the identity).

## 6. Full-suite verdict (`stack/`, CPU, final tree vs pristine base `c415f1b2`)

**0 new failures.** MEASURED, CPU, `raw/suite/` (`raw_compare.json`, `rerun_resolution.json`, `suite_summaries.txt`; instrument
`code/shard_suite.py` + `code/compare_suites.py`). The FINAL tree (`stack/` + `taniteval/`, the files of this hold) was run as 12 parallel
pytest shards (no `pytest-xdist` in the venv; 11 min wall instead of ~40) and compared, test id by test id, with the **serial** run of the
pristine base `c415f1b2` (`--continue-on-collection-errors`; `tools/` is absent from both snapshots, so `test_secret_scan.py` errors at
collection in both).

| | final tree (sharded) | base `c415f1b2` (serial) |
|---|---|---|
| test cases | 10,493 | 10,381 |
| failed or errored | 118 | 115 |
| skipped | 199 | 199 |
| failed in BOTH (same ids) | 115 | 115 |
| tests only in the hold | 112 (= 33 `test_p10_prior` + 23 `test_p10_pose_prior` + 56 `test_launch_gate_v8f`) | -- |
| fixed by the hold | 0 | |

The 115 are the base's pre-existing failures (missing `tools/`, the kit-bind md5 pair, the G-SUITE env list, ...), identical in both runs.
**Three ids failed in the sharded run and not in the base; none is a regression**, and each passes when re-run serially
(`tests/test_amp_and_threads.py` + `tests/test_p10_pose_prior.py`: 33 passed in 11.5 s): `test_amp_and_threads::
test_the_reordering_ACTUALLY_changes_the_thread_pool` (the test says its environment must be scrubbed of `OMP_NUM_THREADS`; the
shard harness sets it to 2) and the two `test_p10_pose_prior` tests that read the banked fit record in this package's `raw/p10_pose/`
(absent from a `stack/`-only snapshot; present in any repo checkout). A first sharded run whose snapshot lacked `taniteval/` entirely showed 219 failures, mostly `FileNotFoundError` on
`taniteval/tools/*`; it was discarded and the run above replaced it (`scratch/full_suite2_run1_no_taniteval`, not shipped). Per-file counts on the final tree:
`test_p10_prior` 33 passed, `test_p10_pose_prior` 23, `test_launch_gate_v8f` 56, `test_residual_prior` 33 + 1 skipped (the same skip
in the base), `test_declared_vs_built` 40.

## 7. Register rows proposed for `GOALS_AND_CLAIMS.md` (the Master Mind appends; not edited here -- edit from the tip's copy)

* `H-V8F-A1-PREMISE-1` -- **MEASURED, premise holds at 2 s:** the pose-only P10 beats refcv7-50,400 on the S1 eval139 grid at 2 s
  (`os - P10pose` ADE +0.0221 [+0.0105, +0.0346] SEP seed 0, +0.0227 SEP seed 1; shipped artifact +0.0178 [+0.0056, +0.0308] SEP;
  LON speed +0.0351 SEP, along-track +0.0265 SEP; T1 open loop, floor vs trained model, episode-variance interval only). **Refcv7 is
  better at 6 s by 0.44-0.49 m (lateral).** This is the premise of alpha-1, not its bar: the bar is read on the trained arms against
  the replicate floor.
* `D-V8F-A1-INPUT-SET` -- alpha-1 differs from alpha-0 in the prior only: `p10_pose` reads past poses (the same channels as
  `ha0_ext_pose`); the steer-reading P10 is the optional arm alpha-1s and the `v8f` profile refuses it by name (SPEC_REFCV7 A5; PI
  ruling on steer at inference pending, `V8F-P10-ACTION-INPUT`).
* `D-V8F-A1-P10-ARTIFACT` -- **CLOSED:** `p10_pose_prior.pt` sha256 `e70a9ae3...693d`, 4,572 train clips, disjoint from eval139/eval141
  (overlap 0/0 by clip and by episode id), banked in `raw/p10_pose/` and `stack/tanitad/configs/v8f_p10/`.
* `H-V8F-A1-PRIOR-1` -- the pose P10 prior beats the `ha0_ext_pose` prior as a start plan by 0.517 m @ 6 s / 0.033 m @ 2 s (separated,
  T0, ADE only) -- a **build property**, not the alpha-1 bar.
* `D-V8F-A1-CAP` -- `P10_KAPPA_CAP` 1.0 (not 0.3) because 0.3 doubles the conversion error through a few sharp-turn windows
  (`raw/sens_cap_floor_final.json`; pose: 0.0385 vs 0.0204 m).

## 8. Deliverable manifest (everything is in the qland hold; NOTHING is staged in git, committed or pushed)

Hold: `C:/Users/Admin/qland/hold_v8f_a1/` (`stage/` new files, `mods-rw/` full edited files built from the tip's blobs, `msg.txt`,
`build_report.json`). Working tree it was built in: `C:/Users/Admin/v8fa1/tree` (scratch; `C:/Users/Admin/v8fa1/scratch` holds the build
scripts). Suite artifacts: `C:/Users/Admin/v8fa1/scratch/full_suite2/`.

| kind | repo path | |
|---|---|---|
| new (stage) | `stack/tanitad/models/p10_prior.py` | frozen P10 (both variants) + loader + provenance |
| new (stage) | `stack/tanitad/configs/v8f_p10/{p10_pose_prior.pt, p10_pose_fixture.npz, p10_prior.pt, p10_fixture.npz, MANIFEST.json}` | the artifacts (pose sha256 `e70a9ae3...693d`; steer `af5a29dd...d7b3`) + known-value fixtures |
| new (stage) | `stack/tests/test_p10_prior.py`, `stack/tests/test_p10_pose_prior.py`, `stack/tests/test_launch_gate_v8f.py` | steer + shared, pose (23), gate (56) |
| new (stage) | `TanitAD Research Lab/Architecture & Inference/Research/2026-10-20-v8f-rung-alpha/` | this package: `DESIGN.md`, `RESULT.md`, `argv_alpha0_alpha1.md`, `argv_v8f-a{0,1,1s}-s0.json`, `argv_pair_check.json`, `code/`, `raw/` (incl. `raw/p10_pose/`, `raw/precheck/`, `raw/suite/`, mutation matrices) |
| modified (mods-rw) | `stack/tanitad/models/kinematic_prior.py`, `stack/tanitad/refs/refc.py`, `stack/scripts/refc_v3_train.py`, `stack/tanitad/train/declared_vs_built.py`, `stack/scripts/launch_gate.py`, `stack/scripts/launch_gate_arms.py`, `stack/tanitad/eval/refcv7_loader.py`, `stack/tests/test_residual_prior.py`, `stack/tests/test_declared_vs_built.py` | every edit is additive except the `ACTION_MODES` pin and the registry-size pin in the two tests |
