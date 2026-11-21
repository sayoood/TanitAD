# RESULT — E-DEPLOY-O2PRE-E2E

`2026-09-02 · Research Lab · Deployment & Optimization · 0 GPU (CPU export) · raw/o2pre_e2e.json`

**Tier: N/A** — an export/graph property, not a model capability read. No T0/T1
stamp applies and the four-families rule (which binds *evals*) does not bind here.
Stated so the omission is not mistaken for one.

**Headline.** Backlog row **O2-pre is already fixed in the working tree** and both
the row and the stream's STATE.md are stale — but the row's *actual ask*, an
export test **at the deployed geometry**, was **not** satisfied by the test that
shipped with the fix. This package supplies it: **encoder → readout exports
end-to-end at 176×624** (ORT vs eager `6.87e-07`), while a **negative control
reproduces the original Thor `SymbolicValueError` verbatim** on the dev box.

---

## F1 — ⚠️ THE ROW IS STALE: THE REPAIR IS IN THE TREE, THE BACKLOG STILL SHOWS IT 🔴 OPEN

**MEASURED** (ours; direct reads of the live G: tree, this pass):

| artifact | state |
|---|---|
| `TanitAD Research Lab/Deployment & Optimization/BACKLOG.md:18` | still **`🔴 O2-pre (NEW) … 🔵 next, 0 deps`** |
| `TanitAD Research Lab/Deployment & Optimization/Research/STATE.md` | exists; **`O2-pre` 0 hits, `176x624` 0 hits** — the stream's state file has no entry for its own next item |
| `stack/tanitad/models/readout.py` | ⭐ **repaired** — `_adaptive_avg_matrix()` builds the constant averaging matrix; `pool_mh`/`pool_mw` are **non-persistent** buffers (strict checkpoint loads still pass); `forward` takes a two-matmul route on the non-tiling path |
| `stack/tests/test_readout_onnx_pool.py` | ⭐ **exists** (7,089 B), docstring names O2-pre |

Verified in-process: `tanitad.__file__` resolves to the **live G: tree**
(`…/TanitAD/stack/tanitad/__init__.py`), and `SpatialGridReadout.forward`'s
non-tiling branch is the matmul route — so the fix under test is the one the repo
holds, not a mirror.

⇒ The row and STATE.md should be updated. *(Root-cause class: the same
finish-before-you-start gap the operating standard names — the work landed, the
tracking did not.)*

## F2 — ⛔ BUT THE SHIPPED TEST DOES NOT TEST WHAT THE ROW ASKS FOR

The row asks for *"Intake + export test **at the deployed geometry**"*, and exists
to falsify a **2026-07-08 "encoder exports clean"** claim. What shipped tests
neither:

* `test_readout_onnx_pool.py` exports a **bare `SpatialGridReadout`** against a
  `[1, 429, 32]` **token** tensor with a toy `d_model = 32`. It never constructs
  `ViTEncoder`, and never exports from an **image** input.
* The claim it must falsify is an **encoder** claim, made by
  `…/Implementation/onnx_export/export_encoder_predictor.py`, whose input shape is
  hard-coded `[1, 9, 256, 256]` and whose argparse exposes **no geometry flag at
  all**.

⇒ **Pinning the submodule that held the defect is not the same as pinning the
claim that was false.** The end-to-end assertion at 176×624 did not exist.

⚠️ A second gap: the shipped test is guarded by `pytest.importorskip("onnx")` and
`importorskip("onnxruntime")`. That is deliberate (onnx is Thor-only in the
project's assumption) — but it means the single export assertion **skips
silently** wherever those are absent, and a skipped guard protects nothing.
MEASURED this pass: the dev venv **has** `onnx 1.22.0` and `onnxruntime 1.27.0`,
so it can and should run here.

## F3 — ⭐⭐ THE END-TO-END EXPORT NOW PASSES AT THE DEPLOYED GEOMETRY, WITH A NEGATIVE CONTROL THAT FAILS

**MEASURED** (ours; dev-box CPU, torch 2.11.0+cu128, opset 17, `dynamo=False`;
`raw/o2pre_e2e.json`). Composed module: `image → ViTEncoder → SpatialGridReadout`,
`in_channels 9`, `patch 16`, `d_model 128`, readout grid 4×4.

| arm | geometry | tokens | route | export | graph | ORT vs eager |
|---|---|---|---|---|---|---|
| ⭐ **deployed** | **176×624** | **11×39 = 429** | constant-matrix matmul (non-tiling) | ✅ **exports** | **no pooling op**, `MatMul` present | **6.868e-07** |
| **tiling control** | 256×640 | 16×40 | `AvgPool2d` (exact) | ✅ exports | `AveragePool` present | 6.161e-07 |
| ⛔ **negative control** | 176×624 | 11×39 | **pre-repair `adaptive_avg_pool2d` spliced back onto the forward path** | ⛔ **FAILS, as required** | — | — |

`ALL_PASS = True` (each arm's `must_export` matched its outcome).

⭐ **The negative control is what makes the positive result evidence.** It fails
with the *original* error, reproduced verbatim on the dev box:

```
SymbolicValueError: Unsupported: ONNX export of operator adaptive_avg_pool2d,
output size that are not factor of input size.
```

Without it, a PASS could not be distinguished from *"this torch/onnx version
happens to export adaptive pooling anyway"* — the deliberate-regression rule from
the validation standard, applied to an export gate. **11 % 4 ≠ 0 and 39 % 4 ≠ 0**,
so the deployed arm genuinely takes the repaired route; it is not passing by
accidentally tiling.

⭐ **And the tiling control matters as much.** 256×640 → 16×40 tiles exactly onto
4×4, keeps the `AvgPool2d` path, and still exports with `6.16e-07` agreement — so
the repair **did not disturb the path every v7 arm actually trains on**.

## F4 — ⚠️ ONE RESIDUAL, HARMLESS BUT WORTH REMOVING

`readout.py:89` still *constructs* `nn.AdaptiveAvgPool2d` on the non-exact path,
as `self.pool`. `forward` never calls it there (the matmul branch runs instead),
so it is inert for `torch.onnx.export`, which traces only what `forward` executes
— confirmed by F3's deployed arm exporting with **no pooling op in the graph at
all**. But a dead submodule sitting on exactly the path where it used to be the
blocker is an invitation to a future refactor re-attaching it.

⇒ Proposed as backlog row **L-14**: delete the dead `self.pool` construction on
the non-exact path.

---

## What this changes

1. **`Deployment & Optimization/BACKLOG.md:18`** — O2-pre's *code* is done; the
   row's *test* requirement is satisfied only now, by this package. The row should
   move to ✅ with both links.
2. **`Deployment & Optimization/Research/STATE.md`** — carries no O2-pre entry;
   its export section still describes the pre-repair world.
3. **`…/Implementation/onnx_export/export_encoder_predictor.py`** — the 2026-07-08
   "exports clean" script is hard-coded to `[1, 9, 256, 256]` with no geometry
   flag. Whatever else is decided, **a geometry argument belongs on it**, or the
   next "exports clean" claim will be true at one geometry again.
4. **O2 (the TRT encoder engine)** — its named blocker is measured cleared at the
   ONNX level. ⛔ That is *not* the same as an engine building on Thor; see below.

## Limits, stated

* ⛔ **This is an ONNX-export result, not a TensorRT result.** No engine was built
  and no Thor hardware was touched. The `#4590` silent-FP32-fallback finding and
  the `quant_gate.py` requirement (backlog row 1) are untouched and still gate any
  precision claim.
* `d_model = 128`, `depth = 2` were used to keep the export small. The defect is
  **geometric** — it is the token grid failing to tile the readout grid — so width
  and depth are not implicated. ⚠️ Stated rather than assumed: a width-dependent
  export failure would not be caught by this test.
* Batch is fixed at 1 and no dynamic axes were declared. A dynamic-batch export at
  this geometry is untested here.
* F1's "already fixed" is a statement about the **working tree**. Whether the
  change is committed, and whether it passed through the
  `Implementation/incoming/` intake the stream's STATE.md mandates, was **not**
  established — `git log` on this path timed out repeatedly against the G: mount,
  and per the environment rule an empty or failed git result is not evidence.
