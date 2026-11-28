# Head-path control: the driver's own commands on an UNTRAINED head (2026-10-15)

Purpose: prove the head path (`aux_navtest_forward` on the cached u400 ctx -> `aux_s1_readout perception` with `--pred-init`) runs on real data with the driver's command builders, **before any H arm exists**. This is not an arm and not a read: the head is the A2.4 initial head of seed 0, and the numbers mean nothing about training.

- Forward: `cmd_forward` on `init_heads/seed0/aux_drv.pt`, ctx cache `c3_full_ctx` (1,058 tokens), `--device cpu --no-gpu-lock`, venv `driverl-eval` (torch 2.7.1). `rc 0`, **325.3 s** (`forward.log`, `heads/run.json`), identity PASS 1,058 / 1,058 (`heads/identity.json`).
- **Bit-identical to the banked A2.4 floor** `p3_floor_full/heads/head_pred_H2-s0.npz`: `logit`, `sdf_m`, `token`, `log` all `np.array_equal` (MEASURED here). So the driver's venv and device reproduce the floor exactly: a trained head and its floor are read on the same device and build. (The npz FILE sha256 differs because `meta_json` carries run metadata; the arrays do not.)
- Perception: `cmd_perception_navtest` with `--pred H2-s0=<new> --pred-init H2-s0=<banked floor>`, `rc 0`, **10.0 s**. `perception_navtest.json`: `p3.floor.trained_minus_init` = est 0.0, CI [0.0, 0.0] (trained == init, so the paired difference is exactly zero, as the A2 construction requires). The decision reads `H2_KILLED` because this head's pooled AUC (0.659) is below the scorer's (0.787); the floor lower bound is 0.0. K2 reads `K2_FIRES` (IoU [0,40) 0.184), matching `raw/2026-10-13-aux-p3-floor/full_1058/FLOOR_1058.md`.
- The estimator pin `check_harness_json` passed on the output.

The A2 `P3_AT_FLOOR` outcome itself (a head above the scorer whose paired lower bound is <= 0) is covered by `chk_p3_floor` and the real-harness grid `chk_p3_matches_harness`, not by this control.
