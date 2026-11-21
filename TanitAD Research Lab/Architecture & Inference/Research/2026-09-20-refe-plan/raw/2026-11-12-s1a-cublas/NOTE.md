# S1a preflight refusal on the pod: cuBLAS determinism config (2026-11-12, Master Mind)

## What happened
- The S1a pause ran as designed. P7 was stopped at its 07:51:23Z checkpoint and the GPU emptied. HA-s0 then got through its EGR and DD stages.
- The HA-s0 preflight was then **REFUSED** (`stage=preflight_HA-s0`, 07:57:15Z), and P7 RESUMED at once (pid 196679). HA-SHUF was not run.

## Cause (MEASURED, `ha-s0/preflight.log` on the pod)
- The error was `RuntimeError: Deterministic behavior was enabled ... this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2`.
- It was raised inside `g12_probe` (`scorer_finetune.py:2476` → `model.py:708/436`, `nn.MultiheadAttention` in-projection).
- The pod runs torch 2.7.1+cu128. G12's identity arms run under STRICT determinism.
- On this torch, the late `os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ...)` at `scorer_finetune.py:2473` does not satisfy the check. The variable must exist at **process start**.
- This was already known and fixed for S1aux: PREREG_AUX_DRIVABLE_S1.md A6 addendum, `s1aux_pod.sh:417-419`, guarded by `test_s1aux_chain.py` `PREFLIGHT_NO_CUBLAS_ENV`.
- `vdac_pod.sh` was written from a parent script that never got the A6 fix. Its three `--preflight` modes launched without the variable.
- Not caught earlier because:
  - G-SUITE is CPU-only;
  - the S1a dry run stops after the inputs stage, before any GPU preflight.

## Fix (landed with this note)
- **`refe/vdac_pod.sh`:** `env CUBLAS_WORKSPACE_CONFIG=:4096:8` at process start on `preflight`, `mutate` and `mutate_ego`. The trainer (`run`) runs without it, as in S1aux and the N arms.
- **`refe/test_vdac_gate.py`:** a `PREFLIGHT_NO_CUBLAS_ENV` guard over all three modes, plus a guard that `run` stays without the variable.
  - In-test mutations: dropping the variable from any one mode turns the guard RED.
  - A real-file mutation (the variable removed from `preflight`) made the test FAIL with `PREFLIGHT_NO_CUBLAS_ENV`.

## Tests on the patched tree (dev box, CPU)
| test | result |
|---|---|
| `test_vdac_gate.py` | ZZTEST_PASS (V1-V11) |
| `test_suite_nesting.py` | ZZTEST_PASS 250 |
| `test_gate_suite_prerun.py` | ZZTEST_PASS 144 |

## Next
1. Re-ship to `/workspace/refe-plan-vdac` only (never refe-plan-p7).
2. Rerun `s1a_go.sh`; stage A rebuilds the suite pre-run on the new tree.
3. The pause then repeats.
