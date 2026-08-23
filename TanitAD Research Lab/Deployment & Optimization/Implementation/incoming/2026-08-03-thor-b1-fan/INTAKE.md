# INTAKE — Thor B1: the candidate fan, the engine's graph, and an encoder export blocker

**Date:** 2026-08-03 · **Stream:** Production & Optimization · **Class:** measurement + one
`stack/`-changing finding (O2-pre) · **Cost:** $0, owned hardware, ~25 min wall-clock.

## What

Three artifacts, all produced on `thor6` (NVIDIA Thor, torch 2.13.0+cu130, TRT 10.13.3.9,
`~/venvs/tanitad-edge`), at the v5f deployed geometry 176×624/117°:

| file | what it is |
|---|---|
| `thor_b1_fan_and_fastpath.py` / `.json` | the B1 harness + raw result: eager fan by batch, K=20 rolls (1 / 9-batched / 9-serialised), three fp16 TRT engines built and timed (fastpath ON b1, OFF b1, OFF b9), tick reconstruction |
| `thor_b1b_fastpath_probe.py` / `.json` | the second-location probe: predictor × opset {17,18} × fastpath {ON,OFF} + encoder × fastpath, with an `nn.MultiheadAttention` module census and ORT-CPU parity per cell |

⚠️ **`thor_b1_fan_and_fastpath.json` carries an appended `ZZ_verdict_correction` key.** All measured
numbers are untouched and bit-identical to the file on Thor; only the harness's own prose verdict for
`FAN_AMORTISATION` was mis-specified (it tested "is the fan FREE", printed "does not amortise"). The
correct reading — the fan amortises 3.41× per candidate in eager and ~9× in TensorRT — is in the
correction key and in the research note.

⛔ **Random weights. LATENCY ONLY.** No accuracy claim is made or implied. The four-family gate
(plan B2/B3) is untouched by this package and still owed.

## Why it matters

1. ⭐⭐ **A deployment requirement, not an optimisation:** the shipped `predictor_fp16.plan` is
   **batch-1 static** while the deployed `TacticalSelector` fans **9** candidates
   (`config.py:95`, `fourbrain.py:562-585`). Serialised = **243.84 ms (244 % of budget)**;
   through a batch-9 engine = **56.13 ms (56 %)**. **4.3×.**
2. ✅ **The published 5.33× survives its own correction** — the corrected (fastpath-OFF) engine is
   **1.187 ms** vs the published **1.168 ms** (1.6 %). It had never been re-timed after the fix.
3. ⚠️ **The MHA-fastpath mechanism did not reproduce at 6 cells** (10 MHA modules present; opset 18
   fastpath-ON exports clean). The 0.726 that drove the retraction has the near-identical-across-
   precisions signature the runbook itself calls a wiring bug. Annotated in the runbook, **not**
   silently corrected — its owner should revisit it.

## The `stack/`-changing finding — O2-pre

⛔ **The encoder does not export to ONNX at the deployed 176×624 geometry**, at either fastpath
setting:

```
SymbolicValueError: Unsupported: ONNX export of operator adaptive_avg_pool2d,
output size that are not factor of input size
```

* **Blocks O2 entirely** (the TRT encoder engine — the designated fallback if bf16 fails B2).
* **Retracts an inherited claim on geometry:** our 2026-07-08 "encoder+readout exports clean at
  opset 17 and 18" was measured at **256×256**, where the adaptive-pool output size *is* a factor of
  the input. At 176×624 it is not. Root-cause class: *a passing export check re-asserted on a changed
  configuration without re-running it* — the same class as the torch-version lesson.
* **Proposed fix (next package, not this one):** replace the shape-derived `adaptive_avg_pool2d` with
  an explicit-kernel `avg_pool2d` (or pad to a factor) in the encoder's pooling path, plus a
  failing-then-passing export test **parameterised on the deployed geometry** — the test running at
  256×256 only is exactly what let this through.

**Proposed target location:** `stack/tanitad/models/` (encoder pooling) + `stack/tests/`
(export test). **Risk:** low — a pooling shape change is numerically equivalent when the sizes
divide, and the test pins both geometries. **Rollback:** revert the pooling call.

## Tests run

* Harness self-validation: the 1-candidate K=20 roll reproduces the published 81.5–81.7 ms to
  **0.02 %**, and the batch-1 fp16 engine reproduces the published 1.168 ms to **1.2 %** ⇒ this
  harness measures the same thing the profile run measured.
* Per-cell ORT-CPU parity vs eager accompanies every export before any timing is read.
* Every timing: warmup 10, `cuda.synchronize()` per rep, p50 **and** p99 (trtexec: 200 iters,
  500 ms warmup, median GPU compute).

## Known limits

* Encoder bf16 measured **30.24 ms** here vs **27.78 ms** published (+8.8 %) — the board was running
  a desktop session with a browser. Ratios are unaffected; absolute encoder time should be re-taken
  on an idle box.
* The tick is **encoder + imagination fan**. It excludes `step_readout` decode, candidate scoring and
  the tactical/strategic head decode. A wired end-to-end selector measurement is the follow-up.
* K=20 per candidate is an upper bound; the selector rolls the maneuver-primitive length. The roll is
  linear in K, so scale proportionally.
* The script behind the original `thor_trt_gate.json` is not on disk, so its export target cannot be
  inspected — the non-reproduction is scoped to what was re-run.
