# refcv8 step-20,000 G0 wrapper clause: is the `lat_tac` miss fp32 reduction-order noise or a micro-batch coupling?

EvalFlyWheel, 2026-10-17, dev box, CPU only (`CUDA_VISIBLE_DEVICES=-1`, `OMP_NUM_THREADS=6`, GPU and its lock untouched; NB `CUDA_VISIBLE_DEVICES=""` does not hide the card on Windows, an empty value deletes the variable and `torch.cuda.is_available()` stayed True).

## PRE-REGISTERED READING (copied verbatim from the brief, written BEFORE any result was produced)

Run the fixed probe on CPU on the same 4 rows and the same micro-batch splits, in two arms:
- **(F32)** fp32, which must approximately reproduce the GPU's `lat_tac` wrapper rel. Report the CPU value; it need not be bit-equal to the GPU.
- **(F64)** the model and inputs cast to float64, with every draw site still neutralised.

For each arm report, per term: the wrapper rel (micro-batched vs plain), and `lat_tac`'s absolute value and absolute difference.

- **R-NUMERICS:** F64's `lat_tac` wrapper rel is <= 1e-9 (or >= 1000x below F32's) AND F32 shows a rel of the same order as 1e-5. Then the miss is fp32 reduction order. Also report whether ANY term in F64 exceeds 1e-9.
- **R-COUPLING:** F64's `lat_tac` rel stays >= 1e-7. Then localise which op mixes rows: run each row alone vs in a batch, find which row differs, and name the module if you can.
- Otherwise: INCONCLUSIVE, with the numbers.

If float64 cannot run through some module (an op without double support), cast only what works and say exactly which modules stayed fp32. That weakens the reading, so state the weakening.

Do not write or propose a change to the bar.

### Operationalisation fixed BEFORE running (the brief's words left open; I fix them here, not after seeing data)
* "same order as 1e-5" = the CPU-F32 `lat_tac` wrapper rel lies in [1e-6, 1e-4] (a factor of 10 either side of 1e-5).
* "wrapper rel" = exactly the probe's statistic: `max(rel(plain, micro13), rel(plain, micro31))`, `rel(a,b) = |a-b| / max(|a|, 1e-12)`, on the scalar loss term that `compute_losses_v3` returns, unchanged `wrapper_probe_r7._analyse`.
* The probe is the unmodified `wrapper_probe_r8fix.probe` (P3: fp32 trunk, NCHW, deterministic, micro sizes [1,3] and [3,1], batch = the cached 4 rows of the G0 in-run batch, perm sha256 `99adbc7e...`, windows 20658, 368, 10637, 7314). F64 differs ONLY by `model.double()`, `torch.set_default_dtype(float64)`, every floating batch tensor and the ingested frames cast to float64.
* F64 is only admissible as a fp64 reading if the audit (a TorchFunctionMode that logs every op that returns a non-float64 float tensor, plus the dtype of the `lat_tac` scalar and of `lat_logits_tac`) shows the `lat_tac` path itself ran in float64. If it did not, the weakening is stated and the reading is downgraded to INCONCLUSIVE for R-NUMERICS.
* Secondary, not part of the registered reading: each row alone vs in the 4-row batch (logit level), and an independent float64 cross-entropy recomputed from the captured `lat_logits_tac` and the labels.

---
# RESULT (written after the runs; evidence class MEASURED = this directory's artifacts, CPU)

## Reading: **R-NUMERICS** (every registered criterion met; the `lat_tac` path was verified float64 in the F64 arm)

| criterion (as operationalised above) | value | met |
|---|---|---|
| F32 CPU `lat_tac` wrapper rel in [1e-6, 1e-4] ("same order as 1e-5") | **1.0647e-5** (GPU: 1.0647e-5, `../wrapper_clause_r8fix_gpu.json`) | yes |
| F64 `lat_tac` wrapper rel <= 1e-9 | **1.98e-14** | yes |
| F64 >= 1000x below F32 | **5.4e8 x** (F32/F64 = 536,873,020) | yes |
| F64 `lat_tac` rel >= 1e-7 (R-COUPLING) | no | -- |
| ANY term in F64 above 1e-9? | **none: 0 of 369 terms; the maximum over all terms is 1.98e-14 (it is `lat_tac` itself)**; W1 and W2 are still detected in F64 | -- |

Source: `result.json` (+ `raw/arm_F32.json`, `raw/arm_F64.json`). Both arms: ckpt `E:/refcv8_eval_kit/ckpt/ckpt_20000.pt` (strict load, missing 0 / unexpected 0, step 20000), the 4 cached rows (perm sha256 `99adbc7e...`, windows 20658, 368, 10637, 7314), P3 (fp32 trunk, NCHW, deterministic), micro [1,3] / [3,1], the unmodified `wrapper_probe_r8fix.probe`, `OMP_NUM_THREADS=6`, torch 2.11.0+cu128, `cuda_available False`. Wall: F32 probe 241 s, F64 probe 660 s.

### `lat_tac`, per arm (MEASURED)
| | plain | micro13 | micro31 | abs diff (13 / 31) | wrapper rel | floor |
|---|---|---|---|---|---|---|
| **F32 (CPU)** | 0.0027991077 | 0.0027991077 (bit-equal) | 0.0027991375 | 0 / **2.98e-8** | **1.0647e-5** | 0.0 |
| **F64 (CPU)** | 0.0027991187 | 0.0027991187 | 0.0027991187 | 5.55e-17 / 5.55e-17 | **1.98e-14** | 1.5e-16 |

* The abs difference is **exactly 128 ulps of the result in BOTH precisions** (F32: 2^-25 = 128 x 2^-32; F64: 5.55e-17 = 128 x 4.34e-19). The F32/F64 ratio is 536,873,020 against 2^29 = 536,870,912, the fp32/fp64 epsilon ratio. A difference that shrinks by exactly the machine-epsilon ratio is rounding noise; a cross-row coupling has no reason to scale with the datatype.
* **Why a 1-2 ulp wobble reads 1e-5 (ESTIMATED, order of magnitude):** the 4-row CE is only 0.0028 (true-class prob 0.9948 to 0.9995) while the logits are large (max |logit| per row 23.9 / 37.0 / 60.9 / 34.2). One fp32 ulp at |logit| in [32,64) is 3.8e-6, and the observed micro-batch logit moves are 1-2 ulps at each row's largest logit (<= 7.6e-6). CE moves by about (1 - p_true) x dlogit ~ 1e-8 per row; dividing by a CE of 0.0028 gives ~1e-5. The same absolute noise on a term of magnitude ~1 would read ~1e-8.
* Independent check (MEASURED): CE recomputed in float64 from the **captured** `lat_logits_tac` and the labels (all four = class 0, none ignored) equals the F64 scalar to 4.3e-19, so the scalar is a pure function of the logits and the noise lives in the logits. The F32 scalar differs from the float64 CE of its own logits by 8.7e-9 (3.1e-6 rel): the fp32 loss evaluation's own rounding.
* `lat_tac` plain F32 vs F64: rel 3.9e-6 (same input values: the fp32-ingested frames are upcast), so `lat_tac` is the same function in both arms to fp32 accuracy.
* Logit level, per-row max |dlogit| vs plain (`lat_logits_tac`): F32 micro13 [3.8e-6, 1.9e-6, 7.6e-6, 3.8e-6], micro31 [1.9e-6, 1.9e-6, 7.6e-6, 7.6e-6], permuted [3.8e-6, 0, 0, 1.9e-6]; F64 micro13 = micro31 = [1.1e-14, 1.4e-14, 2.1e-14, 2.1e-14]. Every row moves, none stands out.
* **Magnitude at other steps.** The GPU probe json banks `plain` for six terms only, **not `lat_tac`**, so the 4-row `lat_tac` at 15,000 is not carried (and I did not measure it; that checkpoint is archived at `C:/Users/Admin/refcv8_eval_archive/ckpt/ckpt_15000.pt`). Carried: the GPU `lat_tac` wrapper rel is outside the banked top 8 at 5,000 / 10,500 / 15,000, hence bounded by the 8th-ranked term, **<= 9.2e-7 / 5.5e-7 / 5.3e-7**, and was **1.0647e-5 at 20,000**. The replay's full-eval-set `eval_lat_tac` (`probe/g0_probe.json`, 3 significant digits) reads 0.00333 / 0.00562 / **0.00311** / **0.004** at 5,000 / 10,500 / 15,000 / 20,000. The 20,000 4-row value is 0.0027991 (this run). So the term was about as small at 15,000 as at 20,000; the 20,000 miss is this 4-row batch's logits putting a 1-2 ulp wobble on a 0.0028 CE, not an obvious change in the model (inference from the numbers above, not a separate test).

### Per term, all 369 terms, both arms: `per_term_wrapper_rel.csv`
| arm | terms > 1e-5 | max over terms other than `lat_tac` | max over all terms | W1 / W2 detected |
|---|---|---|---|---|
| F32 CPU | 1 (`lat_tac`) | 2.2e-6 (`r8_v9_mae_lat_c`) | 1.0647e-5 | yes / yes |
| F64 CPU | 0 | 3.7e-15 (`r8_cons`) | 1.98e-14 | yes / yes |

Both arms ran the unamended 369-term analysis including `r8_spd_crc`; its floor (0.064 F32, 0.18 F64) is the by-design hash change under row reordering that W-R8-1 item 2 already removes, and its wrapper rel is not among the maxima. The CPU F32 arm reproduces the GPU outcome in the respect that matters: exactly one term above the 1e-5 bar, and it is `lat_tac`.

### Each row alone vs in the 4-row batch, under P3 (`raw/arm_*_rows_P3.json`; secondary)
| | max abs dlogit (`lat_logits_tac`), rows 0..3 | max rel over all 137 row-batched output tensors, rows 0..3 |
|---|---|---|
| F32 | 3.8e-6, 1.1e-5, 1.5e-5, 7.6e-6 | 1.3e-6, 1.2e-5, 4.0e-6, 4.4e-6 |
| F64 | 1.1e-14, 7.1e-15, 2.8e-14, 2.1e-14 | 1.1e-14, 8.7e-14, 1.4e-14, 1.2e-14 |

No row is anomalous in either precision, and F64 reduces every row by ~1e8 to 1e9, so there is no row-mixing op to name (the R-COUPLING localisation was not needed). The F64 single-row `lat_tac` equals the batch's float64 per-row CE to the 10+ digits printed.

## What the F64 arm actually was (the weakening, stated exactly)
* `model.double()`: **848 of 848 parameters float64, all 216 float buffers float64** (the other 105 buffers are int64); no module stayed fp32 by dtype. `torch.set_default_dtype(float64)`. All floating batch tensors cast to float64. The uint8 frames still go through the original fp32 ingest (`/255` in fp32) and are then upcast, so **F64 sees the same input values as F32**.
* **The model has explicit fp32 casts, and the first four F64 attempts crashed on them** (dtype errors in `F.conv2d`, `F.linear`, `F.grid_sample`: `timm_trunk._s8_backbone` ends in `out.float()`, `refc.py:2809` `.to(torch.float32)`, `map_head_hires.py:648` `arange(dtype=torch.float32)`, `valid.float()` on a bool). To run at all, the F64 arm redirects them **from this script; no repo file was edited**: `Tensor.float()` -> float64 (except uint8), `Tensor.to(torch.float32)` and factory calls with `dtype=torch.float32` -> float64. Counts over the whole F64 process: 517 `.float()` calls and 2,267 `.to` / factory redirects (top sites in `result.json` `F64_patches`, including `kinematic_prior.py` `absolute_controls` / `compose_ticks` / `roll_plan` and `refc.py` `_state_to_path`). So F64 is **"the same graph with its own fp32 casts removed"**, not a replay of the shipped fp32 numerics. That is the intent here (does the discrepancy scale with epsilon?), but it is a modification of the model's arithmetic.
* **Dtype audit** (a `TorchFunctionMode` over the F64 plain forward, 33,059 ops): only **6 events** returned a non-float64 float, at 5 sites: the frame ingest (`frames_to_device` `.float()` and `div_`, by design) and two `detach` calls in `map_head_hires.py:987` / `:1131` (fp32 class-weight buffers in the map-hires loss, off the `lat_tac` path; those terms read <= 6.4e-16). The `lat_tac` scalar and `lat_logits_tac` were `torch.float64` in every F64 arm. The audit ran on the plain arm only; the other arms run the same code.
* **Open observation, cause UNVERIFIED:** 22 of the 369 terms have plain values that differ between F32 and F64 by more than 1e-3 (19 by more than 1e-2, up to 68 %: `r8_cons` 0.0076 vs 0.0236, `r8_v9_mae_lat_c`, `r8_sat`, `r8_v9_cons`, `tacv6_lat_ce` / `tacv6_lon_ce`, ...), median 0 (csv column `plain_rel_F32_vs_F64`). `r8_spd_crc` is a checksum of the batch's bytes and is expected to change with dtype (2.87e9 vs 4.15e9). The rest are probably discrete decisions (class labels, anchor selection) or differences of near-equal quantities on 4 rows that flip between fp32 and fp64, and/or the redirected `kinematic_prior` integration; I did not separate these. It does not touch the reading (`lat_tac` agrees to 3.9e-6; each arm is judged against its own plain forward), but it means **the F64 statement "0 of 369 terms" describes the float64 function, and for those 22 terms that function is not the fp32 one**. In F32 all 22 are within the bar (<= 2.2e-6).

## Limits (what this does not show)
* One checkpoint (20,000), one batch (4 rows), one micro-split pair, one CPU thread layout (`OMP_NUM_THREADS=6`; two earlier step-5,000 CPU runs gave 5.2e-6 and 3.4e-6 for the same wrapper statistic, cause UNVERIFIED). The F32 CPU value matching the GPU to 4 digits (1.06471e-5 vs 1.06470e-5) means both landed on the same fp32 lattice step; CPU and CUDA reduction orders are different code, so this is agreement of outcome, not of mechanism.
* F64 rules out a row-coupling larger than ~1e-13 in this graph; it cannot rule out one that exists only in a CUDA kernel path absent on CPU. The CPU F32 run reproducing the CUDA miss on the same single term is the evidence that CPU is a fair proxy here.
* The trunk's `dedup_frames` lever (shares frames across rows, exact-equality verified) and `chunk_ckpt 8` were on in P3 in every arm (`result.json` `rows_alone_P3`, `levers`); the rows-alone test covers them.
* Not part of this experiment and not touched: the bar. The numbers above are what the Master Mind and the PI decide from.

## Corrections to my own intermediate artifacts (kept for transparency)
* The `rows_alone` block inside `raw/arm_F32.json` / `raw/arm_F64.json` ran **after the probe had restored the as-built bf16/NHWC trunk levers**, i.e. not under P3 (F32 showed 1 % logit moves and yaw branch-cut jumps). **Invalid as a coupling test; superseded by `raw/arm_*_rows_P3.json`**, which set P3 explicitly and record the levers. The probe numbers in those two jsons are unaffected (the probe sets P3 itself). `raw/wrapper_fp64_probe.py.as_run_for_arm_jsons` is the script exactly as it ran for them; the delivered `wrapper_fp64_probe.py` adds only the `--rows-only` path, the `rows_alone_condition` stamp, and the extras in `--combine`.
* `CUDA_VISIBLE_DEVICES=""` does **not** hide the card on this Windows box (an empty value deletes the variable; `torch.cuda.is_available()` stayed True in my first attempt, which stopped at an assertion before any CUDA call). All reported runs use `-1` and record `cuda_available False`. The earlier diag harnesses set `""`; they built the model on device `"cpu"`, so their compute was CPU regardless.
* No GPU, no GPU lock, no git, nothing written outside this directory. Free commit: 21 GB at F64 start, lowest sample I read 11.4 GB (F64 working set ~10 GB), above the 4 GB abort line; the in-script watchdog never fired.

## DELIVERABLE MANIFEST (all under `C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/raw/step20000/wrapper_fp64/`, unstaged; the Master Mind lands it)
* `RESULT.md` (this file; pre-registration first), `result.json` (verdict, criteria, ulp analysis, F64 patch counts, rows-alone P3, audit)
* `per_term_wrapper_rel.csv` (369 terms x both arms), `make_per_term_csv.py`
* `wrapper_fp64_probe.py` (the experiment: `--arm F32|F64`, `--rows-only`, `--combine`)
* `raw/arm_F32.json`, `raw/arm_F64.json`, `raw/arm_F32_rows_P3.json`, `raw/arm_F64_rows_P3.json`, `raw/*.log`
* `raw/wrapper_fp64_probe.py.as_run_for_arm_jsons`
* Inputs read, not modified: the package `code/`, the battery `code/` of launch tree `C:/lgs/Lfb47a58376`, `C:/Users/Admin/qland/work/r8_wrapper_diag/cache/batch4.pt` (perm sha asserted equal to G0's), `E:/refcv8_eval_kit/ckpt/ckpt_20000.pt`, `config.json`.
