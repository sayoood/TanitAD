# Launch-gate defects that only a real GPU exposes (2026-10-12)

The P2 pod's launch gate refused a correct trainer twice on 2026-10-12. Each defect was invisible on the dev box, because the gate's rehearsal fixtures run on the CPU (`gate_fixture.rehearsal_argv` passes `--cpu`).

| # | check | symptom on the pod | root cause | fix |
|---|---|---|---|---|
| 1 | G-DVB `--amp` | declared bf16 != built float32 | the probe hooked the backbone's OUTPUT, which ends in a LayerNorm; CUDA autocast always runs layer_norm in float32 | `gate_checks.amp_probe_module` = first `nn.Linear` (ad580b6) |
| 2 | G-CKPT | U and H1 rc 1, CUDA OOM | the gate process still held the smoke model plus 24.11 GiB of CUDA cache | `_free(s, deep=True)` before `check_ckpt` (ad580b6); the pod then read 0.14 GiB |
| 3 | G-CKPT | resumed != uninterrupted on 276/643 tensors | two plain CUDA runs are not bit-identical. U and H1 differ at step 1 (`traj_L1` 0.4384 vs 0.4359), BEFORE the halt at step 2, so nothing had been resumed yet | G-CKPT's trainings run through `DET_BOOT`: deterministic algorithms, cuDNN deterministic, `CUBLAS_WORKSPACE_CONFIG=:4096:8` |

## Reproduction of defect 3 on the dev-box GPU (RTX 4060, MEASURED 2026-10-12)

Same tiny fixture as the rehearsal (`gate_fixture.tiny_p2_fixture`), with `--cpu` removed and `--amp bf16` added, `ckpt_steps` 4. Scripts in this directory.

| G-CKPT launch mode | verdict | detail |
|---|---|---|
| plain CUDA (the shipped gate) | **FAIL**, as on the pod | 228/415 tensors differ; `traj_L1` [1.8114, 1.5643, 1.4895, 1.2225] vs [1.8114, 1.5643, 1.4886, 1.2239] |
| deterministic + math SDPA forced (`ckpt_gpu_repro.py`) | **PASS** | model, optimiser state and loss trajectory bit-identical |
| deterministic, kernels chosen by PyTorch (`ckpt_gpu_repro2.py`, **the shipped fix**) | **PASS** | bit-identical; no extra attention memory |
| the shipped fix + deliberate regression `resume_drops_opt` | **FAIL** (as required) | 228/415 tensors differ and the optimiser state differs: the check keeps its power |

Not established: whether two deterministic invocations with DIFFERENT `--out` paths match each other. With `--synthetic` images they did not at step 0. The gate never compares runs across invocations; U, H1 and H2 run within one invocation and match bit for bit.

Test: `refe/test_gate_gpu_probes.py` (static checks of all three fixes, mutations RED, and the CUDA probe for defect 1 when a GPU is present).
