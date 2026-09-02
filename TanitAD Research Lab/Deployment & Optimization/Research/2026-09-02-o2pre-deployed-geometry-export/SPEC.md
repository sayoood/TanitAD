# E-DEPLOY-O2PRE-E2E — pin the encoder's export at the DEPLOYED geometry

`Research Lab · Deployment & Optimization · daily pass 2026-09-02 · 0 GPU`

## Why this exists

`TanitAD Research Lab/Deployment & Optimization/BACKLOG.md:18`, verbatim:

> 🔴 **O2-pre** (NEW) fix the encoder's shape-derived pooling so it exports at
> 176×624 … `adaptive_avg_pool2d, output size that are not factor of input size`
> blocks **O2 entirely**; the 2026-07-08 "encoder exports clean" claim was 256×256
> and is **false at the shipping geometry**. Intake + **export test at the
> deployed geometry**

The failure class is the programme's most-repeated one: **a claim that is true at
one scope, quoted as if it were general.** "Exports clean" was measured at
256×256 square; the shipping frame is 176×624 cylindrical, whose 11×39 token grid
does not tile the 4×4 readout grid.

## Hypotheses, both outcomes committed in advance

| id | hypothesis | outcome A | outcome B |
|---|---|---|---|
| **H-O2-1** | The row is still open — the shape-derived pooling still blocks export. | The fix must be written. | ⚠️ If the fix is already in the tree, the finding is that **the row and STATE.md are stale**, and the work becomes verifying the fix rather than making it. |
| **H-O2-2** | The test shipped with the fix satisfies the row's ask (*"export test at the deployed geometry"*). | Nothing further is needed; close the row. | ⛔ If it tests a submodule at token level rather than the encoder at the deployed geometry, the row's ask is **unmet** and the end-to-end assertion must be written. |
| **H-O2-3** | The composed `image → encoder → readout` path exports at 176×624. | ✅ O2's named ONNX blocker is cleared. | ⛔ It still fails ⇒ the repair is incomplete and O2 stays blocked. |

## ⛔ Success criteria — an export test that only ever passes proves nothing

1. **Deployed arm** (176×624 → 11×39 → 4×4, non-tiling): must **export**, and the
   graph must contain **no** `AveragePool`/`Adaptive*` node.
2. **Tiling control** (256×640 → 16×40 → 4×4, the geometry every v7 arm trains
   at): must **also export**, proving the repair did not disturb the live path.
3. ⛔ **NEGATIVE CONTROL** — the same deployed geometry with the **pre-repair
   operator spliced back onto the forward path** — must **FAIL to export**.
   Without it a PASS cannot be distinguished from *"this torch/onnx version
   exports adaptive pooling anyway"*, and the whole test would be uninformative.
   This is the deliberate-regression rule applied to an export gate: *if the gate
   does not fail the regression arm, a PASS means nothing.*
4. **Numeric floor** — ONNX Runtime vs eager max relative error reported per arm.
   An export that produces a different function is not a passing export.
5. Every arm reports its token grid and whether it tiles, so a PASS cannot come
   from accidentally landing on a tiling geometry.

## Method

`code/export_encoder_readout_deployed.py` builds the composed module
(`in_channels 9`, `patch 16`, `d_model 128`, `depth 2`, readout `grid 4`,
`d_readout 32`), exports at opset 17 with `dynamo=False`, inspects the graph's op
types, and runs ORT against eager. The negative control rebinds the readout's
`forward` to the pre-repair `adaptive_avg_pool2d` implementation.

Imports resolve to the **live G: tree** (verified in-process via
`tanitad.__file__`), so the fix under test is the one the repo holds — not a
possibly-stale off-Drive mirror.

**Tier:** N/A — an export/graph property, not a model capability read. The
four-metric-families rule binds *evals* and does not bind here; stated explicitly
so the omission is not read as one.

## ⛔ Scope

ONNX only. **No TensorRT engine is built and no Thor hardware is touched.** The
`#4590` silent-FP32-fallback finding and the `quant_gate.py` requirement (backlog
row 1) are untouched and still gate every precision claim.
