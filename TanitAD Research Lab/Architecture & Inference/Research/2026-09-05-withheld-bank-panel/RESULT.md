# Withheld-bank panel on the v7-tiny rig (H-EGO-LIT-4) — RESULT

status: ALL SEVEN ARMS COMPLETE (2026-09-05, 10:09:34 → 11:43:15 Europe/Berlin; every arm rc0, 2,000/2,000 steps, 40 metric rows each). The one-variable audit **PASSES** on the actual launch commands (`raw/ONEVAR_AUDIT.txt`). `panel_score.py 40 640` is RUNNING. ⛔ No result number is quoted in this file yet.

Owner: Architecture & Inference FlyWheel agent (Claude Opus 5), resumed 2026-09-05 after the predecessor died at a model rate limit while waiting for the GPU.
Pre-registration: `H-EGO-LIT-4` in `Project Steering/GOALS_AND_CLAIMS.md`; reading rules in `SPEC.md`. Both outcomes were committed before this panel ran.
Compute: dev-box RTX 4060 only, one arm at a time, `OMP_NUM_THREADS=6`. The live run `refcv4b-b1-v72-40k` on pod `tanitad-refcv3` is untouched; Thor is not used.

## Scope stamp (binding, and it applies to every number in this file)

**RIG scope** — v7-tiny rung, ~19 M params, **48 non-parity training episodes**, 2,000 steps, **one seed per arm**. The corpus `physicalai-train-14231cd29c74` references **no registered parity key**, so nothing here is cross-arm comparable with the parity arms and **nothing here may enter `MODEL_REGISTRY.md`** (H-SCALE-2). This panel validates a **DESIGN** and a **GATE**, never a model claim. Scoring is on the episode-disjoint val cache `physicalai-val-bb543bdf7836` (the scorer globs `*val*`), so training and scoring episodes do not overlap.

## Arms — all complete, every lever verified ENGAGED (not merely passed)

| arm | lever vs A0 | wallclock | lever verified by |
|---|---|---|---|
| A0_fixed | — (control; refcv4b's shipped policy) | 792.7 s | `withheld_bank.mode = fixed` |
| A1_pred | `--withheld-bank pred --withheld-bank-warmup 450` | 766.4 s | `withheld_bank_active` = 0.0 on every row ≤ step 450 and 1.0 from step 500 — the bank went live exactly where the pre-registration said it would |
| A2_random | `--withheld-bank random --withheld-bank-warmup 450` | 896.5 s | pool stamp n 9,543, mean 5.7998 m/s, p10/p50/p90 = 0.0 / 5.763 / 10.835, drawn from the TRAINING marginal of v0, **independent of the row** |
| A3_drop25 | `--ego-dropout 0.25` | 792.8 s | mean `ego_keep_frac` **0.7375** over 40 rows vs A0's **0.4813** |
| A4_none | `--withheld-bank none` | 778.7 s | runtime stamp `withheld bank: mode=none` |
| A5_regress | `--ablate-frames` (GATE CONTROL, never a model) | 788.2 s | `ablate_frames: true` in its own config |
| **A0b_replicate** | **nothing at all — A0 run a SECOND time** | 757.3 s | audited as moving zero levers; see below |

**Warm-up N = 450**, read by the pre-registered rule (the first step at which the 5-row running mean of `withheld_speed_mae` falls below 2.5 m/s) from A0's own 40-row log. A1 and A2 SHARE it, so A1-vs-A2 isolates exactly one variable: the speed SOURCE.

**One-variable audit — `ONEVAR_VERDICT=PASS`**, computed from `config.json["argv"]` (what each trainer actually received), not from the launch script's intent. The held-constant set — arm, size, data-root, episodes, steps, batch, lr, warmup, seed, log-every, anchors, n-anchors, control-units, sel-accel-max, device — is identical across all seven runs, and A1 vs A2 differ in `--withheld-bank` and nothing else.

## ⭐ The finding that changes how every headline number must be read

**Training on this rig is NOT deterministic, and the panel's own estimator cannot see it.**

Because A1 and A2 carry `--withheld-bank-warmup 450`, they roll the FIXED bank for steps 1–450 — over that stretch they *are* A0's configuration, same seed, same data order. Their divergence there is pure run-to-run nondeterminism, measured free from the panel's own logs (`raw/noise_floor.py`):

| A1 vs A0, 9 identical-config rows | mean abs Δ | max abs Δ |
|---|---|---|
| `withheld_speed_mae` | 0.194 m/s | 0.701 m/s |
| `loss` | 0.495 | 1.728 |
| `goal2s_err_m` | 0.464 | 2.042 |

66/72 logged differences are non-zero, and the divergence **amplifies**: 0.00007 m/s at step 50 → 0.701 by step 300. Tiny floating-point nondeterminism (non-deterministic cuDNN kernels / atomic reductions) grown by training dynamics.

⛔ **Why it binds.** The paired episode-cluster bootstrap resamples **episodes at eval time with the trained models held fixed**. It answers *"would this delta survive on other windows?"* — not *"would it survive on another training run?"*. With one seed per arm those are different questions, and only the second licenses "the lever caused it". **A CI excluding zero is therefore not sufficient evidence that a lever did anything**, if two runs of the same configuration differ by more than the delta. This is the programme's own rule about quoting an interval without the variance component that dominates it — here that component is not in the interval at all.

⇒ **A0b_replicate** was added mid-panel for exactly this: A0's flags, A0's seed, run again, scored through the identical instrument. Its paired delta vs A0 is the **eval-level noise floor**, and every headline delta will be printed beside it. ⚠️ This was **not** in the original pre-registration; it is recorded as an addition, not a criterion retrofitted to a result that had already been seen.

## Ops findings (MEASURED; each is a false-failure generator)

1. ⛔ **The GPU-wait launcher could never have fired.** `wait_and_launch.sh` counted busy jobs with a `Get-CimInstance … -match 'refav1_arm.py|refcv3_arm.py|t1_eval.py'` filter whose own PowerShell command line **contains the pattern**, so the query matched **itself**. `procs` read a constant 4 for over an hour — including while GPU utilisation fell to 5 % — of which: the querying `powershell.exe`, two shell wrappers, and an `ssh.exe` to **Thor** (not a local GPU consumer at all). `n` could never reach 0. The documented self-match trap in a `Win32_Process` costume; the fix is to enumerate processes **without** filtering on the pattern and read the rows.
2. ⚠️ **`wc -c < file` is a METADATA probe, not a content probe.** GNU `wc -c` takes the size from `fstat()` and never reads the bytes, so during a mount outage it returned the correct 802,715 bytes for a file whose every content read failed — yielding a "the mount is up for the shell but down for Python" reading that was **wrong**. True content reads (`cat` / `head -c`) failed on target **and** control 15/15, which is what actually justified the Drive-client restart.
3. ⛔ **A freshly created directory refuses ALL writes** — MSYS first, then the native Windows API too (12 retries × 4 s, all four files) — while `mkdir` returns 0, `test -d` says EXISTS, reads keep working, and a tiny `echo >` into the *same* directory succeeds. **Restarting the Drive client clears it immediately and completely**, on all four occurrences. The dangerous part is the *shape* of the lie: the banking script reports *"banked metrics.jsonl has 0 rows — do not quote this arm"*, which reads as a claim about the **arm** when the fault is entirely in the **access path**. An agent trusting that line would discard a good 2,000-step run.

## Numbers

None yet — scoring is in flight. Every number will carry its evidence class, its T-tier, its arm and its artifact path under `raw/<arm>/`. ⛔ ADE is one row of the four families and never the gate (`H-ECHO-4`).
