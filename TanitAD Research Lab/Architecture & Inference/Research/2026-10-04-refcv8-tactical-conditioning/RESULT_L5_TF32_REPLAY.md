# WP-B L5: TF32 off inside the gradient-share fp32 replay (2026-10-05)

## What L5 changes, and what it does NOT

* **Changed:** `stack/tanitad/train/grad_share.py::fp32_replay` (+10 lines). Inside the replay of a gs-due step,
  `torch.backends.cuda.matmul.allow_tf32` and `torch.backends.cudnn.allow_tf32` are set to False and the fp32 matmul
  precision is set to `"highest"`. All three are restored in `finally`. The row gains `gs_fp32_replay_tf32_off = 1`.
* **Not changed:** nothing the training step executes. The replay runs before the training forward, only on gs-due
  steps. It already forks the global RNG, restores the dedicated generators, buffers and levers, and never touches `.grad`.
* **Evidence that the training step is bit-identical (MEASURED, CPU, pinned):**
  `tests/test_refcv8_grad_share_fp32.py::test_the_replay_leaves_the_training_step_bit_identical`. The loss and every
  `.grad` after a replay equal those of a step taken without it. The NEW test
  `test_the_replay_runs_with_TF32_OFF_and_restores_every_switch` asserts that, inside the replay, the switches read
  (False, False, `"highest"`), and that afterwards they are back to the configured (True, True, `"high"`).
* **Consequence for N** (MM condition, 2026-10-05): S0 timed on cb8289b fixes N. L5 does not touch the timed path.

## Why (MEASURED, Thor, the L4 G-SMOKE)

Under `--trunk-bf16`, the L4 replay switched bf16 off. cuDNN's TF32 default was still ON, and the trainer never sets it.
The trunk linearity control read **4.2e-4** against the 1e-4 bar; the bf16 direct reading had been 5.9e-3. P-GRAD's
bar was measured with both TF32 switches off (`…/2026-10-04-refcv8-perception-architecture/code/pgrad.py:68-69`).

## The two-halved GPU diagnosis (MM ruling): `code/tf32_microcheck.py`

The test net is built like the trunk: timm resnet34, 9 input channels, batch 2 × 9 × 208 × 512, on the Thor GPU, all readings on the same batch.
* **TF32 ON** must reproduce the >1e-4 order. This is the control that proves TF32 is the cause.
* **TF32 OFF** must read ≤ 1e-4. It is read twice; the repeat is reported, not gated.
* **L5 replay**, with the global switches ON and a bf16 lever ON, must read ≤ 1e-4 and restore every switch.

### Measured (Thor GPU, torch 2.13.0+cu130, `raw/tf32_microcheck.json`): **PASS on all three**

| reading | trunk `lin_rel_err` | bar 1e-4 |
|---|---|---|
| TF32 ON (cuDNN default) | **5.26e-4** | violates: the cause is reproduced, the same order as the G-SMOKE's 4.2e-4 |
| TF32 OFF | **8.85e-7** | meets it, ~600× below the ON reading |
| TF32 OFF, repeat | 8.85e-7 | the two OFF readings are NOT bit-identical (cuDNN's backward is non-deterministic) but agree to 2e-10 |
| **L5 replay**, global switches ON + bf16 lever ON | **8.85e-7** | meets it; every switch and the lever are restored afterwards |

TF32 alone accounts for the violation. With TF32 off inside the replay, the control reads at fp32 rounding, which is the
same order as P-GRAD's 6.6e-6 on the full model and better.

## Full-size S3 re-read (MM ruling)

`code/r8_smoke_thor_L5_s3.sh`: 10 steps from refcv7-50,400 on the L5 tree, with the gs reading at step 10 and no eval.
The same MemAvailable ≥ 40 GB guard applies, inside the lock. No arm launches before this passes.

## Gate (dev box, the MM's rule)

Tip 26dcb6a: 7 failed / 2,564 passed / 11 skipped. Candidate (tip + the 2 L5 files): 7 failed / 2,565 passed / 11 skipped.
**Candidate-only failures: none.** The shared 7 also fail on the tip.
