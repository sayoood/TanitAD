# Per-layer FP16-vs-INT8 benchmark + on-device runbook

*Companion to `DEPLOYMENT_PLAN.md`. This is the harness the recipe mandates before ANY INT8 commit (§2.4), plus the on-device build runbook. Split explicitly into measurable-now (A40 proxy) and on-device (Orin/Thor).*

---

## 0. Why this exists — the trap it defends against

**INT8 ≠ speedup on a ViT on Orin.** Non-optimal INT8 kernels (reformat overhead, INT8 attention paths that fall back to or run slower than the FP16 fused path) can make an INT8 ViT run **~2.7× slower than FP16** on Orin-class hardware. And our accuracy margin over the kinematic floor is only ~0.05 m, so an INT8 accuracy regression invisible in a "<1 % loss" framing erases the model's whole claim. **Therefore INT8 is adopted per-layer, only where BOTH a latency benchmark AND an accuracy gate favour it — never model-wide, never assumed.**

---

## 1. The two gates every INT8 candidate must pass

### Gate A — per-layer latency (does INT8 actually run faster, layer by layer)

- **A40 proxy (measurable now):** build three engines from the exported ONNX (`encoder_readout_f4b.onnx`, `predictor_f4b.onnx`): **FP16**, **INT8 (weight+activation, entropy-calibrated)**, and **best-typed** (`--best`, TRT picks per layer). Dump the per-layer profile:
  ```
  trtexec --onnx=encoder_readout_f4b.onnx --fp16   --dumpProfile --dumpLayerInfo --separateProfileRun --exportProfile=enc_fp16.json
  trtexec --onnx=encoder_readout_f4b.onnx --int8 --fp16 --calib=calib.cache --dumpProfile --exportProfile=enc_int8.json
  trtexec --onnx=encoder_readout_f4b.onnx --best --dumpProfile --exportProfile=enc_best.json
  ```
  (No `trtexec` in the pip `tensorrt` wheel → use the builder-inspector path: `trt_build.py` builds the engine and `create_engine_inspector().get_engine_information(JSON)` gives per-layer info; extend it with `--int8` + a calibrator to emit the INT8 profile.)
- **Compare per layer, not per model.** Any layer where INT8 ≥ FP16 latency stays FP16 (TRT's `--best` or an explicit per-layer precision does this automatically; the point is to *read the profile* and confirm INT8 won where it was applied).
- ⚠️ **Kernel selection is per-architecture.** The A40 (SM 8.6) proxy proves the harness and gives the FP16 baseline, but the INT8-vs-FP16 *winner can flip on Orin (SM 8.7)* — the 2.7× trap is an Orin kernel-selection property. **Gate A is only decision-grade when re-run on the target.**

### Gate B — horizon-stratified accuracy (does INT8 keep the trajectory)

- Run the deployed checkpoint open-loop (no replanning) on the fixed val set at fp32, fp16, and each INT8 candidate.
- **Plot per-step Δpose error AND SE(2)-accumulated error at every one of the 20 steps** — not just ADE@2s.
- **Statistic:** paired episode-cluster bootstrap over the 40 val episodes (`taniteval/ci.py`). **Never** `overlapping_holdout_se` (1.28–2.06× too narrow; CLAUDE.md).
- **Falsifiers (any one kills that precision for the rollout):**
  - ADE@2s degradation whose paired CI excludes 0, or
  - point degradation > 0.02 m, or
  - a degradation ratio that GROWS with horizon (ADE@0.5/1/1.5/2 s) — the compounding signature ⇒ keep the state-carrying path at higher precision.
- Flat-across-horizon degradation ⇒ a per-step bias ⇒ correctable by recalibrating the `step_readout`, not by abandoning INT8.

**INT8 ships for a block only if it passes A (faster, on the target) AND B (no accuracy falsifier). Default = FP16.**

---

## 2. INT8 calibration specifics (so the run is reproducible)

- **Weight-only INT8/FP8 on the predictor blocks first** — weights are 100 % of the rollout's binding traffic and the weight/activation-safety asymmetry is the best-evidenced result in the search. Keep activations, `step_readout`, and the ViT at FP16.
- **Calibrator:** entropy (`IInt8EntropyCalibrator2`) or, better, TRT's **INT8 + per-channel** for transformer weights — never per-tensor.
- **Calibration set:** a few hundred real val windows (the same `physicalai-val-0c5f7dac3b11` the accuracy panel uses), fed as the network's natural inputs (encoder: 9-ch frames; predictor: real (state, action) windows harvested from a v1 rollout). Cache to `calib.cache`.
- **If weight-only INT8 fails Gate B → QAT**, not more PTQ tricks (PTQ-collapses→QAT-recovers repeats across TAO/CILRS/D4RL).
- **Never** joint W4A4 or 3-bit (documented collapse on DINO-WM).

---

## 3. On-device runbook (hardware-blocked until an Orin/Thor is on hand)

```
# 1. Flash: Orin -> JetPack 6.2 (L4T 36.4.3, CUDA 12.6, TensorRT 10.3) or 7.x
#          Thor -> JetPack 7.0 (CUDA 13.0, TRT 10.13) or 7.2 (CUDA 13.2.1, TRT 10.16.2)
# 2. Copy the static-shape ONNX (encoder_readout_f4b.onnx, predictor_f4b.onnx) to the device.
# 3. Build + profile ON THE DEVICE (engines are NOT portable across GPU arch):
trtexec --onnx=encoder_readout_f4b.onnx --fp16 --dumpLayerInfo --dumpProfile --exportProfile=enc_fp16_orin.json --saveEngine=enc_fp16.plan
#    verify fused MHA: grep the layer info for a fused attention / Myelin block; a standalone Softmax => NOT fused (#4537)
# 4. Gate A per-layer FP16-vs-INT8 on the device (§1). 5. Gate B accuracy (§1) via taniteval at the engine's precision.
# 6. Wrap enqueueV3 in cudaStreamBeginCapture/EndCapture (one CUDA graph). Call enqueueV3 once first to flush deferred shape updates.
# 7. (Thor) repeat with --fp8 and NVFP4 once the TRT build supports the flag on the flashed JetPack.
```

**Precision order on Jetson: FP16 engine FIRST (the 7–16× arithmetic lever on Orin), CUDA graph SECOND** — the reverse of the A40 order (`DEPLOYMENT_PLAN.md` §5). Re-measure; do not assume.

---

## 4. What is measurable now vs on-device

| check | A40 proxy (now) | Orin/Thor (blocked) |
|---|:--:|:--:|
| ONNX→TRT-FP16 build succeeds | ✅ | rebuild on device |
| FP16 engine latency | ✅ (proxy) | ✅ target truth |
| MHA fused? (#4537) | ✅ (SM86 tactic) | ✅ (SM87/Blackwell tactic may differ) |
| per-layer FP16 profile | ✅ | ✅ |
| INT8 calibration + accuracy Gate B | ✅ (accuracy transfers) | latency does not transfer |
| INT8-vs-FP16 per-layer *winner* (2.7× trap) | ❌ arch-specific | ✅ only here |
| NVFP4 anything | ❌ needs Blackwell | ✅ Thor only |
| real vehicle tick @ nvpmodel mode | ❌ | ✅ (name the power mode, as we name precision) |
