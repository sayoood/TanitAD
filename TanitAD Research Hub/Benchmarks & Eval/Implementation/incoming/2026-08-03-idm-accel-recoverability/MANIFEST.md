# STREAM D — deliverable manifest

**Date** 2026-08-03 · **Substrate** dev box (RTX 4060, `C:\Users\Admin\venvs\tanitad`),
**0 pod GPU-h** · **Question** is `long_accel` UNRECOVERABLE from the frozen v1 latents, or merely
UNRECOVERED by the head tried?

Everything below is in the working tree and **staged with `git add`. Nothing is committed and
nothing is pushed** (agent contract). Nothing lives only on a pod or only in a worktree.

## Instrument — reusable, in `stack/`, unit-tested

| path | what | staged |
|---|---|---|
| `stack/tanitad/eval/accel_probe.py` | the capacity/architecture probe instrument: `DualRidge` (exact kernel ridge, whole alpha path per eigendecomposition, linear + rbf, GPU Gram), `window_features` (centre / window / **diff** / centre+diff bases), `inject_signal` (detection-sensitivity control with shared gain and a choosable carrier direction), `MLPHead`, `GRUHead`, `probe_loss` (= the shipped `idm_head.idm_loss` **plus a per-scalar mask**), `fit_probe_head`, `predict_head`, `r2_score` (0.0 not NaN on a constant target) | ✅ |
| `stack/tests/test_accel_probe.py` | 12 contract tests: dual ridge == an independently solved primal ridge; the fp32 Gram path matches the fp64 reference; the alpha path is monotone in shrinkage; rbf beats linear on a nonlinear target; **`probe_loss` is numerically identical to the shipped loss when unmasked**; the mask isolates one channel's gradient; the injection control fires when planted and is silent when not; shared injection gain across two blocks; head output contracts; the trainer learns a recoverable scalar | ✅ |

## Experiment — this results directory

`TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-08-03-idm-accel-recoverability/`

| path | what | staged |
|---|---|---|
| `verify_substrate.py` | step 0: re-derive the labels from the raw episodes, re-encode two episodes through the frozen encoder, re-measure the ORACLE CEILING, characterise the target | ✅ |
| `raw/substrate_verification.json` | its output | ✅ |
| `run_accel_recoverability.py` | the sweep: closed-form capacity ladder, neural ladder, detection-sensitivity, oracle-input capacity control, context length; four families + paired episode-cluster bootstrap for every arm | ✅ |
| `results_accel_recoverability.json` | the primary artifact (B=2000) | ✅ |
| `oracle_input_capacity.py` | the CAPACITY CONTROL done properly — the same heads and a closed-form ridge on the TRUE speed window, raw and standardised input | ✅ |
| `raw/oracle_input_capacity.json` | its output — the decisive row: closed-form ridge on the true speed window reaches `long_accel` **R² 0.9262 [0.8876, 0.9507]** through the identical protocol | ✅ |
| `probe_standstill_filtered.py` | kills the last alternative explanation: refit with every stationary train window removed, scored on the full AND the moving-only held-out set | ✅ |
| `raw/standstill_filtered.json` | its output | ✅ |
| `analyze_speed_error.py` | mechanism: speed-error autocorrelation, the derived-accel route end to end, an error budget, and the scalar-vs-trajectory **self-consistency control** | ✅ |
| `raw/speed_error_mechanism.json` | its output | ✅ |
| `raw/preds_speedarm.npz` | per-window predictions + GT + episode ids — a 0-GPU re-analysis surface | ✅ |
| `summarize.py` | renders the result JSON as the report's tables, so no number is transcribed by hand | ✅ |
| `raw/summary_tables.txt` | its output, the exact source of every table in §4 | ✅ |
| `ACCEL_RECOVERABILITY.md` | the report | ✅ |
| `raw/run_log.txt` | the run log of the shipped run | ✅ |
| `raw/results_v1_BADEPOCHSELECTION.json` | the FIRST full run, kept deliberately: its epoch budget was selected on `long_accel` inner R² (noise), which destroyed the positive control. Report §7 | ✅ |
| `raw/smoke_*.json` | the smoke runs, kept because two instrument defects were caught in them (report §7) | ✅ |

## Inputs (NOT in the repo — off-Drive data, unchanged by this work)

| path | what |
|---|---|
| `C:/Users/Admin/tanitad-data/eval/idm_derived_accel_latents.pt` | the banked latent cache built by the P9 panel; **re-verified here**, not inherited |
| `C:/Users/Admin/tanitad-data/eval/comma2k19-val-61c46fca8f7f/ep_000{40..89}.pt` | the raw episodes the labels were re-derived from |
| `C:/Users/Admin/tanitad-data/eval/v1_speedjerk_ckpt.pt` | the frozen flagship-v1 encoder (step 29999), used only to re-encode two episodes for verification |

## Escalations — these need someone else to act

1. **`stack/scripts/idm_head.py:104-127`** states as settled that *"the channel carries no recoverable
   information from the frozen v1 latents at this scale"*, and attributes the failure to a **5× error
   amplification**. The conclusion is now much better supported (17 arms + a decisive capacity
   control); the **mechanism is wrong** (the speed error is autocorrelated at 0.9265, so differencing
   cancels ~93 % of it — the killer is the target's 0.587 m/s² dynamic range), and the claim needs
   its **sensitivity floor** attached. See the report §6.
2. **`BACKLOG.md` A7 (Delta-JEPA: displacement instead of endpoints)** is partly answered here at
   probe level by the `diff` feature basis. See the report §4.
3. **Label defect** — `ep_00045`'s CAN `long_accel` is identically 0.0 and `ep_00080`'s is 0.0 except
   its last two windows; **7.57 % of all windows have `long_accel` exactly 0.0**, and TRAIN `yaw_rate`
   carries heading-repair outliers to ±15.3 rad/s while the held-out episodes top out at 0.48 rad/s.
   Both bear on every IDM number ever quoted on this cache.
4. **Instrument follow-up (cheap):** `DualRidge.predict` recomputes the kernel per alpha. Caching the
   test Gram per kernel would cut the rbf-on-18,432-features arms by roughly an order of magnitude.
   Deliberately NOT changed after the run started, so the staged file is byte-for-byte the one that
   produced `results_accel_recoverability.json`.
