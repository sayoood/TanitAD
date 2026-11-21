# 2026-07-08 — ONNX export of the operative path + data-cluster compliance review #1

**Agent:** production-optimization-agent (Saturday, run #1) · **Phase:** 0 · **Quality:** full
**Budget:** 1 iteration, ~1.5 h wall-clock, 2 web searches (of 25). Hardware: RTX 4060 dev machine,
CPU export path (fp32, strict numerics). Cost: $0 (local).

This is the first run of the Production & Optimization stream (D-020 §3). Two measured deliverables:
an ONNX export + parity experiment (backlog P0.2, G-H/G-P2) and compliance review #1 of
`stack/tanitad/data/` (backlog P0.3, one intake package with failing-then-passing tests, G-P1).

---

## 1. Measured experiment — ONNX export of the operative path (backlog P0.2)

**Goal.** Export the two operative-tick graphs (encoder+readout; multi-horizon predictor) to ONNX,
prove numeric parity vs PyTorch under pinned fp32 numerics, and enumerate any unexportable ops —
the prerequisite for the TensorRT-on-Orin path (backlog P1.4). Real weights: `ckpt_full.pt`
(step from checkpoint), TanitAD-4B-M 0.263 B params.

**Method.** `Implementation/onnx_export/export_encoder_predictor.py`. Two graphs matching the deploy
shape: encoder+readout `[1,9,256,256] → [1,2048]` (runs every frame); predictor
`states[1,8,2048], actions[1,8,2] → (z_h1,z_h2,z_h4)` (runs on the causal window; dict output
wrapped to a tuple — ONNX has no dict output). `eval()` disables the F-5 grad-checkpoint lever.
Parity = max/mean |Δ| torch-fp32 vs ONNXRuntime-CPU over 5 random inputs; speed = same-device
Torch-CPU vs ORT-CPU (1 thread each). Opset 17, legacy exporter (`dynamo=False`).

**Results (measured).**

| Graph | Export | Parity max\|Δ\| (tol 1e-4) | Torch-CPU p50 | ORT-CPU p50 | ONNX size |
|---|---|---|---|---|---|
| encoder+readout | **OK** (opset 17) | **8.8e-6** ✅ | 103.2 ms | 455.0 ms | 405 MB |
| predictor (3 heads) | **OK** (opset 17) | **1.2e-5** ✅ | 17.1 ms | 24.6 ms | 425 MB |

- **No unexportable ops.** `nn.MultiheadAttention`, FiLM (`Linear`+`chunk`), the causal `torch.triu`
  bool mask, `AvgPool2d` readout, and the residual add all export cleanly at opset 17. The
  "FiLM/attention op unsupported" falsifier (backlog P0.2) **did not fire** — the model needs no
  plugin or rewrite for ONNX.
- **Parity holds with ~5× margin** under the 1e-4 fp32 target (worst case 1.2e-5).
- **G-P2 speed delta (the honest part): ORT-CPU is SLOWER than Torch-CPU** — 4.4× on the encoder,
  1.4× on the predictor. This is expected: torch-CPU uses tuned MKL + MHA fast paths; naive ORT-CPU
  decomposes MHA. **The value of ONNX here is the portable IR feeding TensorRT on the Orin target —
  NOT a CPU speedup.** Reporting speed alone (without this caveat) would misrepresent the result.
  The deployment latency story stays the GPU decision-tick (15.07 ms p50, 2026-07-08 baseline);
  ONNX/TRT is the next lever, measured on-target in P1.

**Forward-compatibility check (added after the literature sweep).** torch made `dynamo=True` the
default in 2.9 and **removed the legacy fallback in 2.11** (we run 2.11); the literature warns the
dynamo path throws `UnsupportedOperatorError` on the fused `_native_multi_head_attention` kernel.
**Tested directly: the `dynamo=True` exporter ALSO exports our encoder cleanly** (parity max|Δ|
6.7e-6 vs torch). So our export path is safe on both the legacy and the modern exporter — the
fused-MHA trap does not bite our `need_weights=False` blocks under `torch.export`. **Ops gotcha
(real, dev-machine):** the dynamo exporter prints progress with emoji (`✅`) and crashes with
`UnicodeEncodeError` under the default Windows cp1252 console — export must run with
`PYTHONUTF8=1` (or `PYTHONIOENCODING=utf-8`). Documented in PRODUCTION_READINESS.

**Artifacts.** Script + `parity.json` in `Implementation/onnx_export/`; the ~0.4 GB `.onnx` files are
written off-Drive (scratchpad) and intentionally not committed. New venv deps: `onnx 1.22`,
`onnxruntime 1.27`, `onnxscript 0.7.1` (dev/export-only; not a stack runtime dep — flagged for the
inference-wheel dependency audit, backlog P2.10).

---

## 2. Compliance review #1 — `stack/tanitad/data/` (backlog P0.3)

Cluster reviewed: `epcache.py`, `mixing.py` (save/load + `MixedWindowDataset`), `_contract.py`
(contract + `EpisodeWindowDataset`), against the PRODUCTION_READINESS checklist. Two test-proven
defects shipped as one small intake package; two lower-priority findings logged for review #2.

### Finding 1 (HEADLINE) — cache-key collision (`epcache.py:30-34`)

Per-source identity is `getattr(s,"name",None) or (s.get("clip_id") if dict else str(s))`. For a
`Path`, `.name` is the **basename**, so two clips sharing a filename across directories collide.
**Reproduced live** against a faithful copy of the current logic:

- `chunk_0/scene_000.hevc, chunk_0/scene_001.hevc` and `chunk_1/scene_000.hevc,
  chunk_1/scene_001.hevc` → **identical** key `37215f6f5632`. This is exactly the **cosmos
  chunk-pairing failure class** (28/60 clips got the wrong chunk's actions, fixed in the loader
  2026-07-08) — still latent in the cache key. A relaunch silently loads the WRONG episodes.
- dict sources missing `clip_id` → every id is `None` → unrelated sets collide (`6f33b4d44cec`).

**Fix.** `_source_id` keys paths by **full** path, keys dicts by `clip_id` and **raises** when
absent (no silent `None`), stable `repr` fallback otherwise. Trade-off flagged: full-path keys are
machine-local (acceptable — the cache already lives under `cache_root`).

### Finding 2 — silent persistence of mis-shaped episodes (`mixing.save_episode`)

`save_episode` writes whatever it is handed; a build item with actions/poses length ≠ frames `T`
(or non-4D frames) is persisted silently and detonates later inside a training window. Added the
cheap shape guard at the write boundary — the same "fail here, not deep in training" doctrine that
`_contract.assert_contract` already encodes.

### Intake package

`Implementation/incoming/2026-07-08-data-cluster-compliance/` — fixed self-contained `epcache` +
`save_episode` copies, `INTAKE.md`, and `tests/` (7 + 5). **12 passed in 1.62 s.** Failing-then-
passing is explicit: `*_legacy_collides` tests embed the current logic and assert the bug exists
today; paired `*_fixed_*` tests assert resolution. Pending orchestrator triage.

### Lower-priority (logged, not in this package — kept small & mergeable)

- `epcache` `DONE` marker is **written but never read** (`grep` confirms one writer, zero readers);
  resume is per-file. Docstring corrected in the package copy; behavior unchanged.
- `EpisodeWindowDataset.__init__` (`_contract.py:120`): an episode shorter than
  `window+max_horizon` contributes **0 windows silently** (no log/counter) — observability gap,
  candidate for review #2 (`tanitad/models/` cluster) or a data-review #1b.

---

## 3. Literature (my discipline's SEARCH duty, 2 searches)

- **NVIDIA TensorRT Model Optimizer (ModelOpt)** is the confirmed INT8/FP8/FP4 PTQ route: in-place
  quantization on a loaded model with a calibration dataloader, emits QDQ nodes TensorRT fuses into
  INT8/FP8 kernels; multi-modal stacks keep the **vision tower at FP16 by default** — reinforces the
  2026-07-14 Architecture note (native-TRT ViT INT8 is a trap; quantize the predictor/heads first,
  keep the ViT encoder higher-precision, measure probe-fit delta as the accuracy metric). Sources:
  [Spheron ModelOpt guide](https://www.spheron.network/blog/tensorrt-model-optimizer-modelopt-quantization-guide/),
  [NVIDIA FP8→TensorRT blog](https://developer.nvidia.com/blog/model-quantization-turn-fp8-checkpoints-into-high-performance-inference-engines-with-nvidia-tensorrt/).
- **torch ONNX export (2.11):** dynamo default since 2.9, fallback removed in 2.11; the fused-MHA
  `UnsupportedOperatorError` is a known trap — but **verified not to affect our model** (§1). If a
  future torch removes the legacy exporter and a fused path DOES appear, the drop-in workaround is a
  plain-ops MHA (`Linear`/`bmm`/`softmax`). Source:
  [torch.onnx docs](https://docs.pytorch.org/docs/2.12/onnx.html).

---

## 4. Actionable recommendations

1. **[MVP orchestrator]** Triage `2026-07-08-data-cluster-compliance` — the cache-key fix closes a
   silent-wrong-data class already observed once (cosmos). Low blast radius (one cache rebuild).
2. **[Production, next run]** TensorRT fp16 engine from these ONNX graphs on the 4060 (backlog P1.4);
   report GPU latency + accuracy delta on 100 held-out windows. ONNX parity is now cleared.
3. **[Architecture, informational]** ONNX exports without op changes at opset 17 AND opset 18/dynamo
   — no architecture constraint needed for exportability; keep `need_weights=False` in attention.
4. **[Dependency audit, backlog P2.10]** onnx/onnxruntime/onnxscript are export-time only — keep them
   out of the inference-only runtime wheel.

## 5. Backlog delta
- P0.2 ONNX export → **DONE** (parity cleared, both exporters, no plugin need).
- P1.4 TensorRT fp16 → **unblocked**, promoted to next-run top.
- New P1.x: plain-ops MHA drop-in kept on the shelf (only if a future torch drops the legacy exporter
  AND the fused path reappears — not needed today).
- New P2.x: Windows export must set `PYTHONUTF8=1` (dynamo emoji-print crash) — captured in readiness.
