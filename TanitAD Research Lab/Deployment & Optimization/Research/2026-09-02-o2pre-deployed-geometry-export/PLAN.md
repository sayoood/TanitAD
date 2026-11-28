# PLAN — E-DEPLOY-O2PRE-E2E

`2026-09-02 · Deployment & Optimization · compute budget: 0 GPU, ~90 s CPU`

## Priority order (a killed run still yields value at every step)

| # | step | yields on its own | state |
|---|---|---|---|
| 1 | Read the O2-pre row and the stream's STATE.md directly | F1 — the row is stale and STATE.md never recorded it | ✅ done |
| 2 | Confirm in-process which `tanitad` tree is imported, and read `SpatialGridReadout.forward` | the fix under test is the live repo's, not a stale mirror | ✅ done |
| 3 | Read what the shipped test actually asserts | F2 — it pins the submodule at token level, not the row's ask | ✅ done |
| 4 | Export the composed `image → encoder → readout` at 176×624 | F3 — the end-to-end assertion the row asked for | ✅ done |
| 5 | Run the tiling control (256×640) and the **negative control** | what makes F3 evidence rather than a green light | ✅ done |
| 6 | Escalate the stale row, the missing geometry flag, the silent skip | integration, per the operating standard | ✅ done |

## Compute

* **0 GPU.** CPU-only ONNX export; `nvidia-smi` showed no Python compute on the
  dev-box 4060 before or after.
* ⛔ **Thor: not touched.** ⛔ **A40 pod (`tanitad-a40`, REF-C v3 H-arm, ~35 h
  remaining): not touched.**
* ⛔ **No write into `stack/`**, per the Deployment stream's intake boundary rule.

## Owners / integration

* **Owner:** Research Lab (daily pass 2026-09-02).
* **Escalated to the DeployFlyWheel** — the stale row, promoting the end-to-end
  script into `stack/tests/` through the `Implementation/incoming/` intake, the
  missing geometry flag on the 2026-07-08 export script, and the silent
  `importorskip`. See `COMMS.md`.

## Reproduce

```bash
python code/export_encoder_readout_deployed.py --out raw/o2pre_e2e.json
```

Requires `onnx` + `onnxruntime` (both present in the dev venv: 1.22.0 / 1.27.0).
The script asserts each arm's outcome against a **pre-declared `must_export`**, so
the negative control failing is a PASS and the negative control *succeeding* would
be a FAIL — the run cannot be read as green unless the regression arm really
regresses.

## Next, if the DeployFlyWheel picks it up

1. Promote to `stack/tests/` via intake (three parametrised functions).
2. Add a **dynamic-batch** arm at 176×624 — untested here, batch is pinned at 1.
3. Only then O2 proper (the TRT engine), which remains gated behind
   `quant_gate.py` (backlog row 1) and the `#4590` FP32-fallback finding.
