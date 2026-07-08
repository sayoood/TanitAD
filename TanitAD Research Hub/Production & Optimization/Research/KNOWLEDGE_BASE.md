# Production & Optimization — Knowledge Base

Deltas only, deduplicated, newest first. Each entry: fact + source (repo path or URL).

## 2026-07-08 (run #1)

- **The TanitAD-4B operative path is ONNX-exportable with no op changes.** encoder+readout
  `[1,9,256,256]→[1,2048]` and predictor `states[1,8,2048],actions[1,8,2]→(z1,z2,z4)` export clean
  at **opset 17 (legacy exporter)** and **opset 18 (`dynamo=True`, torch-2.11 default)**. Parity vs
  PyTorch (fp32, strict numerics, ORT-CPU): max|Δz| 8.8e-6 (encoder) / 1.2e-5 (predictor), tol 1e-4.
  `nn.MultiheadAttention` (`need_weights=False`), FiLM, the causal `torch.triu` bool mask, and the
  `AvgPool2d` readout all have stable symbolics. Predictor dict output must be wrapped to a tuple.
  Source: `Implementation/onnx_export/{export_encoder_predictor.py,parity.json}`.
- **ONNXRuntime-CPU is 1.4–4.4× SLOWER than PyTorch-CPU** on our graphs (encoder 455 vs 103 ms;
  predictor 24.6 vs 17.1 ms, 1 thread each). ONNX's value is the portable IR for TensorRT-on-Orin,
  NOT a CPU speedup — never quote ORT-CPU latency as an optimization win (G-P2). Source: parity.json.
- **torch 2.11 removed the legacy-exporter fallback; `dynamo=True` is default since 2.9.** The
  literature warns of a fused `_native_multi_head_attention` `UnsupportedOperatorError` trap — but it
  does NOT fire for our `need_weights=False` blocks under `torch.export` (verified, parity 6.7e-6).
  Drop-in fallback if a future torch regresses: replace MHA forward with plain `Linear`/`bmm`/
  `softmax`. Source: `Research/2026-07-08-...md` §1/§3, https://docs.pytorch.org/docs/2.12/onnx.html
- **Windows ONNX-export gotcha:** the dynamo exporter prints emoji (`✅`) progress and crashes with
  `UnicodeEncodeError` under the default cp1252 console. Run exports with `PYTHONUTF8=1` /
  `PYTHONIOENCODING=utf-8`. Source: measured this run.
- **NVIDIA ModelOpt is the INT8/FP8/FP4 PTQ route** (in-place quantize + calibration dataloader →
  QDQ nodes TensorRT fuses to INT8/FP8 kernels). Multi-modal stacks keep the **vision tower FP16 by
  default** — reinforces "native-TRT ViT INT8 is a trap; quantize predictor/heads first, ViT stays
  higher precision, accuracy metric = probe-fit delta." Sources:
  https://www.spheron.network/blog/tensorrt-model-optimizer-modelopt-quantization-guide/ ,
  https://developer.nvidia.com/blog/model-quantization-turn-fp8-checkpoints-into-high-performance-inference-engines-with-nvidia-tensorrt/
- **`epcache` cache key was collision-prone** (basename-only path id; `None` for dict-without-
  `clip_id`) — same silent-wrong-data class as the cosmos chunk-pairing bug. Fixed via full-path /
  clip_id-or-raise identity. Source: `Implementation/incoming/2026-07-08-data-cluster-compliance/`.
- **`eval()` disables the F-5 grad-checkpoint lever** (`encoder.py:60-61` gates it on
  `self.training and t.requires_grad`) — so no export-specific flag is needed to turn checkpointing
  off; just export in eval mode. Source: `stack/tanitad/models/encoder.py`.
