# D-B1-GATE as an implementable spec — engine inspection, fallback detection, calibration format

**Package** `Deployment & Optimization/Research/2026-08-29-quant-gate-spec` · **Author**
Research Lab agent (daily run 003, 2026-08-29) · docs/primary verification, 0 GPU. Serves
register row **D-B1-GATE** (OPEN, blocks B1) and continues `2026-08-28-vit-int8-fp8-thor`.
Vendor-doc facts are PUBLISHED-DOC (NVIDIA docs, retrieved 2026-08-29, version-drift caveat);
the gate records runtime truth regardless.

## Verified API facts (the spec's load-bearing surface)

1. **Per-layer precision dump exists in two forms.** [PUBLISHED-DOC]
   - Python: `inspector = engine.create_engine_inspector()` →
     `inspector.get_layer_information(i, trt.LayerInformationFormat.JSON)` /
     `get_engine_information(...)`.
   - CLI: `trtexec --profilingVerbosity=detailed --exportLayerInfo=layers.json`
     (+ `--exportProfile=profile.json` for per-layer measured latency).
   - Layer JSON fields (exact): `Name`, `LayerType`, `Inputs`/`Constants`/`Outputs` (each
     tensor with `Dimensions`, `Datatype` — e.g. `Int8`, `Float` — and `Format`),
     `TacticName`, `StreamId`, `Metadata`.
   - ⛔ **Detail is gated at BUILD time**: default `ProfilingVerbosity` is
     `kLAYER_NAMES_ONLY` — an engine built without `kDETAILED` is UNINSPECTABLE for
     precision. The gate must treat that as REFUSE, never PASS.
2. **TRT #4590 re-verified from the primary (GitHub, fetched today): OPEN, no workaround.**
   Jetson Thor devkit, **SM 110, TensorRT 10.13.3.9, CUDA 13.0**: *"TensorRT accepts
   `BuilderFlag.FP8`/`BuilderFlag.FP4` but silently builds FP32 engines"* — no error, no
   warning; reporter's evidence = engine size ~FP32 (382 MB) + outputs `DataType.FLOAT` +
   full-precision weights memory. Opened 2025-10-04. (Incidentally pins the Thor TRT/CUDA
   pair from a primary — one report; gate still records its own at runtime.)
3. **Strong typing is the structural fix, and the legacy path is a dead end.** [PUBLISHED-DOC]
   With a strongly-typed network (explicit Q/DQ from the ONNX), *"builder flags are neither
   required nor allowed"* — precision is taken from the network, types are not auto-tuned, so
   the hint-ignored fallback class cannot occur. **TRT 11.x REMOVES the weak-typing precision
   flags (`BuilderFlag.FP16`/`INT8`, `setPrecision`, `setDynamicRange`, implicit
   quantization) entirely** — a gate built around builder flags would die at the next major.
4. **INT8 calibration cache format (legacy path, if used).** [PUBLISHED-COMMUNITY — NVIDIA
   declined to document it officially (empty search below)]: text file, header
   `TRT-<version>-EntropyCalibration2`, then `<tensor_name>: <hex>` per line, hex = raw
   float32 bits of the scale; written by `IInt8EntropyCalibrator2.write_calibration_cache`.
   Parse defensively; never hand-edit. For the ViT trunk: **32 calib images suffice**
   [lib `2111.12293`]; per-channel-calibrate → per-tensor-deploy reparam [lib `2212.08254`].

## The gate script — `quant_gate.py` (DeployFlyWheel implements; stages are the spec)

**Inputs:** engine path, FP16 reference engine, probe-set path, mandatory-layer name
patterns (the trunk's QKV/proj/MLP GEMMs), thresholds file. **Output:** verdict JSON under
`raw/` (registry-quotable): environment (TRT/CUDA/GPU/sm), engine sha256s, every measurement,
per-check PASS/FAIL/REFUSE, overall verdict.

- **Stage 0 — preconditions (fail-closed).** Record `tensorrt.__version__`, device, sm.
  Probe inspectability: if layer JSON lacks per-tensor `Datatype` fields ⇒ **REFUSE
  ("rebuild with ProfilingVerbosity.DETAILED")**. A gate that cannot see must say so — the
  decode-into-memmap family: the success path must be unable to lie.
- **Stage 1 — precision census (anti-#4590).** Parse every layer; classify compute layers
  (GEMM/Conv/MatMul patterns) by the `Datatype` of inputs/outputs. Rules: (a) every
  mandatory-pattern layer at target precision (Int8/FP8) — offenders listed BY NAME;
  (b) attributable-census fraction ≥ 0.9 at target among compute layers; (c) fused/mangled
  names (TRT concatenates fused layers) matched by substring; **the unattributed fraction is
  REPORTED, and unattributed > 10 % ⇒ INCONCLUSIVE, not PASS.** LayerNorm/Softmax/GELU are
  expected FP16 — they are NOT in the mandatory set (standard ViT PTQ keeps them high-precision
  [lib `2111.12293`]); encoding this prevents spurious FAILs.
- **Stage 2 — behavioural cross-checks (what #4590's reporter used).** (a) engine size:
  target ≤ 0.6× the FP16 reference (weights dominate; threshold in config, both sizes
  recorded); (b) measured latency (100 warm iters, in-process on Thor — only in-process
  probes are admissible there): target ≤ 0.8× FP16, else SUSPECT-FALLBACK ⇒ FAIL; (c) I/O
  dtypes from the engine bindings recorded. Flags claim nothing; bytes and milliseconds do.
- **Stage 3 — dense-head four-family accuracy.** Quantised vs FP16 on the fixed probe set:
  per-family deltas (LON/LAT/TAC/STR) with paired episode-cluster bootstrap CIs, plus the
  DENSE head outputs explicitly — the ≤6-bit cliff (W4A4 box AP −23.2; SAM-L 36.4→5.4
  [lib `2405.03144`]) is invisible to scalar heads. Bands pre-registered; tier stamp
  T0-DIAGNOSTIC printed in the verdict.
- **Stage 4 — verdict.** Overall PASS only if 1–3 pass; any REFUSE/INCONCLUSIVE propagates
  (never "green by blindness"). JSON banked; registry row cites the JSON.

**Build-recipe demands the gate enforces:** build with `kDETAILED` always (a few % build-time
cost, zero runtime cost); prefer **strongly-typed + modelopt Q/DQ ONNX** over builder flags
(fact 3); FP8 arms on sm_110 additionally require the census to show FP8 datatypes actually
present (fact 2 — flags alone silently produce FP32, and FP8 is ≥50 % less efficient than
INT8 on this class anyway [lib `2303.17951`] — INT8 W8A8 first, per predecessor).

## What this changes for TanitAD (≤3)

1. **DeployFlyWheel implements `quant_gate.py` from the stage list above** — every API call
   named here is doc-verified today; the REFUSE state is first-class, and D-B1-GATE's
   "engine inspection + latency + size + four-family accuracy" becomes four concrete stages
   with fail-closed semantics.
2. **Move the B1 build recipe to strongly-typed + explicit Q/DQ now** — it eliminates the
   silent-fallback CLASS structurally (not just detects it), and the weak-typing flags are
   already removed in TRT 11, so this is also the only forward-compatible path.
3. **Do not attempt FP8/FP4 on Thor via builder flags at all** — #4590 re-verified OPEN with
   no workaround; any FP8 claim must come from a strongly-typed build PLUS a census showing
   FP8 datatypes in the layer JSON, else it is presumed FP32.

## Named empty searches

- **Official NVIDIA documentation of the calibration-cache format**: NONE — NVIDIA declined
  on their own forum ("could it be officially documented"); format is community-established.
  Fragility named: parse defensively, regenerate rather than edit.
- **An existing open-source PASS/FAIL precision-verification gate**: NOT FOUND —
  `trt-engine-explorer` (trex) visualizes engine JSON but does not gate; `quant_gate.py` is
  net-new.
- **NVIDIA response/workaround on #4590**: NONE in the issue as of today.

## Sources

Banked (cited-by updated): 2111.12293 (PTQ4ViT) · 2212.08254 (RepQ-ViT) · 2303.17951
(FP8-vs-INT8) · 2405.03144 (dense-head cliff). Vendor docs (PUBLISHED-DOC, retrieved
2026-08-29, not bankable as PDFs): TRT engine-tools page (IEngineInspector, JSON fields),
TRT command-line-programs page (trtexec flags), TRT working-with-quantized-types +
10x→11x migration pages (strong typing, flag removal), GitHub NVIDIA/TensorRT#4590.
