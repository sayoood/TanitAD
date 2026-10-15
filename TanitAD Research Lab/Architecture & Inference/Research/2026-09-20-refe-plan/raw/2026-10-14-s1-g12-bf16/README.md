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

**UNVERIFIED until it runs there:** G12 on the A40 itself. The first H-arm preflight is preceded by a pod pre-run of S1's G-SUITE.
