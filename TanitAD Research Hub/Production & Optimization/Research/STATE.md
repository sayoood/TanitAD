# Production & Optimization — STATE

- **LAST_RUN:** 2026-07-10 (run #3, Saturday agent) — branch `worktree-prod-opt-20260710`
  (worktree `.claude/worktrees/prod-opt-20260710`, D-026 isolation; orchestrator merges to main).
- **QUALITY:** full (one measured experiment with decision-grade numbers — INT8 weight-quant curve
  + clean-GPU latency — and one compliance review with a 10-test standalone-green intake package;
  G-H, G-P1, G-P2 met. INT8 *latency* not claimable without TRT int8 kernels, so the measured INT8
  delta is the 4× weight-memory footprint reported next to the decision-space accuracy — not a skip.)
- **Phase:** 0

## Where this stream stands (one paragraph)

Run #3. **INT8 weight-quant curve localizes the sensitivity — it is the READOUT, not the ViT.**
Per-output-channel symmetric int8 weight-only fake-quant, scored in decision space on the same 64
windows (step-6500, 4060): ViT-encoder INT8 **98.4 %** agreement (1 flip, 1.6 cm), predictor INT8
**95.3 %** (3 flips, 10.9 cm) → both SAFE; but heads/whole-model INT8 collapse to **48.4 %** (33
flips, **1.67 m** wp-shift), and `int8_all == int8_heads` while encoder- and predictor-alone are
safe → the damage is entirely in **{readout, inv_dyn, imagination}**, tell = encoder-state rel-err
7e-4→1.67e-2 (only the readout produces state). **Deploy rule: INT8 the ViT encoder + predictor
weights (−552 MB of 825 MB at 4×), keep the readout ≥fp16.** This **refines the KB "ViT INT8 trap"**
— that trap is an *activation*-quant phenomenon; ViT *weights* round cleanly. Also **closed P1.4b**:
clean idle-GPU fp32 tick **15.76 ms / 63.5 Hz** (≈ the 15.07 ms baseline → the 33.5 ms 07-09 read
was contention, falsifier confirmed), fp16 **13.40 ms / 74.6 Hz / 1.18×**. Compliance **review #3**
(data-window index) shipped a fail-loud intake: silent short-episode drop across `_contract.py:120`
/ `toy_driving.py:131` / `comma2k19.py:278` → count + warn + `ValueError` on empty, parity preserved,
10 standalone tests. INT8 *latency* + the joint fp16-act × int8-weight bar still need the TRT engine
(P1.4a).

## Next actions (checkboxes)

- [ ] **Next run — P1.4a (top):** build the mixed engine once the TRT/ModelOpt toolchain is in
      (dev-box `tensorrt`+`onnxruntime-gpu`, or an idle pod): INT8 weights on encoder+predictor,
      readout ≥fp16, fp16 activations. Verify the **pre-registered joint bar** — agreement ≥95 % /
      wp-shift ≤~4 cm on the 64 windows; if the two ~5 % error terms compound below it, drop the
      predictor to fp16 and INT8 only the encoder (98.4 %, −228 MB, the safest single win).
- [ ] **Clean per-precision VRAM:** re-measure fp16/int8 with a single model resident (this run's
      fp16 1.65 GB double-counts the fp32 reference) → publish the true fp16-vs-fp32 VRAM delta.
- [ ] Carry logged review-#2 findings into a numerics package: `tactical_pred` fail-fast +
      `imagination_nll` `exp(-logvar)` clamp (backlog P1.7).
- [ ] Compliance **review #4** (`stack/scripts/` + training loop): resume/atomic-write/log hygiene,
      cgroup awareness (F-5/F-6/F-7) — fold in the `epcache` DONE-marker-unused finding.

## Standing facts / gotchas (this stream)

- RTX 4060 is the declared Orin latency proxy (I8). Clean decision-tick baseline: **15.07 ms p50 /
  1.08 GB VRAM fp32** (2026-07-08 MVP loop, exclusive GPU). **Absolute latency needs an exclusive,
  clock-pinned GPU** — the 2026-07-09 run was CarlaUE4-contended (99 % util) and read 33.5 ms; only
  the accuracy deltas and speedup *ratios* survive contention, not absolute Hz or per-precision VRAM.
- **Precision policy: fp16 on the decision path, never bf16** (measured 2026-07-09). TRT-fp16
  acceptance bar pre-registered (≥95 % agreement, ≤~4 cm wp-shift on 64 windows).
- **INT8 weight-quant policy (measured 2026-07-10): INT8-safe = ViT encoder (98.4 %) + predictor
  (95.3 %); INT8-RED-LINE = the readout / state-projection** (heads INT8 → 48.4 %, 1.67 m). The
  "keep ViT FP16" heuristic is an *activation*-quant rule and does NOT apply to weight-quant — ViT
  weights round cleanly. Clean idle-GPU latency baseline: fp32 **15.76 ms/63.5 Hz**, fp16
  **13.40 ms/74.6 Hz** (per-precision VRAM still needs a single-model-resident run).
- **TensorRT not installed on the dev box** (`import tensorrt`→missing; ORT CPU EP only). The ONNX
  IR is exported + parity-clean, so only the toolchain/engine step remains.
- ONNX export deps (`onnx`, `onnxruntime`, `onnxscript`) installed in the venv — **export/dev only**,
  never in the inference runtime wheel.
- Windows dynamo ONNX export crashes on emoji progress under cp1252 → run with `PYTHONUTF8=1`.
- Boundary: NEVER write `stack/` directly. Experiments live in `Implementation/onnx_export/` (off-
  Drive for large `.onnx`); stack-changing fixes go through `Implementation/incoming/` intake.

## HANDOFF

Run completed cleanly on branch `worktree-prod-opt-20260710` (worktree isolation, D-026) —
orchestrator to merge to main. Note: the experiment artifacts (`int8_quant/`, the windowing intake)
were first written into the MAIN working tree by an interrupted earlier start of this run; they were
copied into the worktree and committed there. The stray untracked copies in the main tree are
removed at session end (they arrive on main via the branch merge). Main advanced 859caa8→52bb39e
mid-session (loop's step-21k gate preview) — worktree was ff'd to 52bb39e before committing.
