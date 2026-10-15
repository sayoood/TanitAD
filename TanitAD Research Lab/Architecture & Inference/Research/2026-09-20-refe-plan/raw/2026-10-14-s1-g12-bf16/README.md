# S1 G12 under CUDA bf16: two instrument defects in the preflight probe, and the pod's torch build (package)

The full write-up is **`eval/PREREG_AUX_DRIVABLE_S1.md` Amendment A6**. The probes in this folder call it "A5": it was renumbered on landing, because the
execution-order amendment had landed as A5 first.
- **Probe fixes:** the G12 fix agent (Sonnet), 2026-10-14/15.
- **The torch 2.7.1 addendum and the landing:** the Master Mind, 2026-10-15.

| what | where |
|---|---|
| H2 root cause: the autocast weight-cast cache. (a) rounding and (b) degenerate step-0 scalars are refuted | `probe_b_rootcause.py` / `.json`, `probe_c_g12_values.*` |
| H1 root cause: `warn_only=True` keeps the non-deterministic memory-efficient SDPA backward. bf16 and the late cuBLAS setdefault are refuted | `probe_d_h1_determinism.py`, `probe_d_*.json`, `probe_d_summary.json` |
| the fixed probe, proven on CUDA bf16 (RTX 4060, torch 2.11) | `refe/test_aux_g12_bf16.py`, output `test_aux_g12_bf16.out` (`ZZTEST_PASS 7`: 30 checks plus 6 RED mutations) |
| **the pod's torch build, 2.7.1+cu128** (same build in the local `driverl-eval` venv), as written | **FAILS**: strict mode raises torch's cuBLAS determinism error (`torch271_no_env.log`) |
| **the same, with `CUBLAS_WORKSPACE_CONFIG=:4096:8` exported at process start** | **`ZZTEST_PASS 7`** (`torch271_cublas_env.log`) |
| ⇒ `refe/s1aux_pod.sh` exports it on the **preflight line only**; the trainers are unchanged | pinned as `PREFLIGHT_NO_CUBLAS_ENV` in `refe/test_s1aux_chain.py`, with a RED mutation |
| the agent's suite outputs | `*.out`, `run_all_tests.sh` |

## On the A40 (Master Mind, 2026-10-15)

**S1's G-SUITE pre-run on the pod (d257647, the gate's environment, the real data) all PASS:** `rl_reward` 31, `rl_epo` 42, `aux_drivable_head` 68,
`aux_s1_wiring` 20, `launch_gate --static --no-banked` 645.

**H2's fix: VERIFIED on the A40 (MEASURED, `a40_H2-s0_PREFLIGHT_PASS.json`, `a40_H2-s0_preflight.log`).**
- **The arm.** H2-s0's preflight passed at 2026-10-15T05:12:30Z, with `verdict PASS` and `ZZSFT_PREFLIGHT_PASS`. The chain then logged `ZZS1AUX_ARM_GATE_OK H2-s0` at 05:22:33Z and `ARM_TRAIN_START` at
  05:24:53Z.
- **G12 `ok: true`.** This is the gate that refused under bf16 before the fix:
  - `feed_probe_reads_difference: true` and `feed_probe_known_field_reads_difference: true`;
  - `feed_probe_max_abs_diff` 0.6875, with **3,092 / 3,584** elements differing;
  - `feed_zero_init` and `feed_step0_equal_base` both true;
  - `score_to_head_grads_all_none: true`;
  - the head-group gradients are all non-zero when declared (`blocks.0` 272.2, `blocks.1` 269.4) and **exactly 0.0 at `w_aux` 0**;
  - `adam_head_params_with_state` 43 / 43.
- **Resources.** Preflight measured 2.0 s per microbatch and 5.51 GB of GPU memory.

**H1's fix: NOT YET VERIFIED on the A40.** The strict-determinism path and the cuBLAS env at process start are what this arm adds. Its first A40 preflight is
**H1-s0**, the second arm in the queue. If it refuses, the chain logs `ZZS1AUX_REFUSED` and the pod watcher wakes the Master Mind.
