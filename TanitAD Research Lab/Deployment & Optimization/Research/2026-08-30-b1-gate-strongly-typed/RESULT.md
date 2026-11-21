# D-B1-GATE: TRT #4590 re-checked, and a reframe that changes what the gate must assert

**Package** `Deployment & Optimization/Research/2026-08-30-b1-gate-strongly-typed` · **Author**
Research Lab agent (daily run 004, 2026-08-30) · literature + primary-doc reading, 0 GPU,
Thor untouched. Seed: `D-B1-GATE` — has TRT #4590 moved, and if not, what has landed on
strongly-typed + explicit Q/DQ?

**Continues** `…/2026-08-29-quant-gate-spec/RESULT.md` (the 4-stage fail-closed `quant_gate.py`
design). That package already re-verified #4590 OPEN and already named strongly-typed + explicit
Q/DQ as the structural fix. ⇒ **This package adds what was NOT in it** (F2–F7): the reframe of
what #4590 actually is, the reason a numerical check can never catch it, the JetPack version
wall, and three verification caveats that would each have produced a wrong gate.

---

## F1 — #4590 IS STILL OPEN. No NVIDIA response, no fix version, no workaround.

[PUBLISHED — verified twice, HTML page **and** GitHub REST API, which agree]
`https://github.com/NVIDIA/TensorRT/issues/4590`

| field | value |
|---|---|
| title | *"Jetson Thor (SM 110) advertises FP8/FP4 throughput but TensorRT silently falls back to FP32 when FP8/FP4 flags are enabled"* |
| state | **open** · `state_reason: null` · `closed_at: null` |
| created / updated | 2025-10-04 / 2026-01-09 |
| comments | **1** — from `bowCine89`, **`author_association: NONE`** (a fellow user, not NVIDIA) |
| labels | `Module:Quantization`, `Module:Runtime` |

Reported on TensorRT 10.13.3.9 / CUDA 13.0 / SM 110. ⛔ **Do not record this as fixed.**

## F2 — ⭐⭐ THE REFRAME: #4590 IS A DIAGNOSTICS BUG, NOT "FP8 IS BROKEN ON THOR"

[PUBLISHED — TensorRT docs, `inference-library/work-quantized-types.html`] Verbatim:

> *"FP8 quantization exclusively supports explicit quantization via Q/DQ layers. There is no
> calibration-based implicit quantization path for FP8 — implicit quantization is limited to
> INT8 only."*

Supported types: **INT8** (implicit *and* explicit); **FP8E4M3, INT4, FP4E2M1 — explicit
only**. And for Q/DQ networks, *"precision-control build flags are not required and should not
be specified."*

[INFERRED — well-supported] The reporter fed a plain ONNX with **no Q/DQ nodes** and set
`BuilderFlag.FP8`/`FP4`. That flag only *permits* FP8 kernels for layers carrying FP8 Q/DQ —
with none present, **nothing was ever going to be quantised**, on Thor or anywhere. The genuine
defect is that TensorRT **emitted no warning**.

[PUBLISHED, corroboration on our exact hardware] Issue **#4599** shows FP8 **did engage on
Thor** once the user did ONNX Q/DQ surgery — layer info confirmed FP8 present (~20 % latency win
on ViT-Base). **Failure and success track the presence of Q/DQ, not the platform.**

⇒ ⭐ **Our gate's exposure is not "Thor cannot do FP8". It is "we built via builder flags at
all."** That is a build-path change, not a measurement change — and it is available today.

## F3 — ⛔ THE REASON THIS GATE CANNOT BE FIXED BY MEASURING HARDER

[INFERRED from F2 + the failure's structure] **A numerical-equivalence check cannot detect a
silent fallback**: comparing an FP32 reference against a *secretly-FP32* engine yields **zero
error**. The failure makes the gate score **better**, not worse. ⇒ **Detection must be
STRUCTURAL — plan size, dtypes, layer info — never numerical alone.**

*(Same family as the `jpeg_buf`-is-png trap, where the poisoned artifact was the FLOOR arm: a
defect whose signature is indistinguishable from success is a false-positive generator, and only
a structural assertion catches it.)*

## F4 — Strongly-typed: the claim is HALF right, and the failing half is benign

[PUBLISHED — TensorRT dev guide, `advanced.html#strongly-typed-networks`] Verbatim:
*"When you specify to TensorRT that a network is strongly typed, it infers a type for each
intermediate and output tensor… **Inferred types are adhered to while building the engine.**"*
*"As types are not autotuned, an engine built from a strongly typed network can be **slower**."*
*"For strongly typed networks, the layer APIs `setPrecision` and `setOutputType` are not
permitted, nor are the builder precision flags `kFP16`, `kBF16`, `kFP8`, `kINT8`, `kINT4`, and
`kFP4`."* (`kTF32` excepted.)

⚠️ **But a documented FP32/FP16 fallback SURVIVES strong typing** [PUBLISHED — TRT 10.16.1
release notes]: *"FP8 Convolutions on GPUs with SM89/90/120/121 do not support kernel sizes
larger than 32… **FP16 or FP32 fallback kernels will be used with suboptimal performance**"*;
and *"There are no optimized FP8 Convolutions for Group Convolutions."* (⚠️ **SM110 is not in
that list** — scope it before quoting.)

⭐ **THE LOAD-BEARING DISTINCTION — the two fallbacks are different animals:**

| build path | what falls back | does the accuracy gate LIE? |
|---|---|---|
| builder flags, no Q/DQ (**#4590**) | the **whole quantisation** — the graph is pure FP32 | ⛔ **YES.** The gate scores an FP32 engine and passes. |
| strongly-typed + explicit Q/DQ | only the **kernel/tactic**; Q/DQ nodes remain and still round the values | ✅ **NO.** Quantisation error is in the numerics regardless of kernel choice. |

⇒ Strong typing does not make fallback impossible; it **converts a silent accuracy lie into a
visible performance loss.** That is exactly what un-gates the read. Explicit quantisation also
fails *loudly*: *"you can encounter a `could not find any implementation` error."*

## F5 — ⛔ THE VERSION WALL: TRT 11 removes the bug class, and JetPack forbids TRT 11

[PUBLISHED — TRT 11.0.0 release notes + 10.x→11.x migration guide] TensorRT **11.0.0 removes
weak typing entirely**: `setDynamicRange`, `ILayer::setPrecision`, `setOutputType` and **all**
per-precision `BuilderFlag`s are gone; `createNetworkV2()` is strongly typed by default;
`trtexec` loses `--fp16/--int8/--fp8/--best`. ⇒ **The #4590 bug class is structurally impossible
in 11.x — the flags do not exist.** OSS timeline: 11.0 GA 2026-06-02 · 11.1 2026-06-24 ·
**11.2 2026-08-04**.

⛔ **…and we cannot have it** [PUBLISHED — TRT 11.2.1 release notes, verbatim]:
> *"**NVIDIA JetPack is not supported in TensorRT 11.2.1. Jetson deployments must remain on a
> TensorRT 10.x release supported by their JetPack version.**"*

**JetPack 7.2.1** (2026-08-11): Jetson Linux 39.2.1, CUDA 13.2.1, **TensorRT 10.16.2**,
cuDNN 9.20.0 — AGX Thor DevKit. ⇒ Thor is pinned to **10.x**, so our fix is the **opt-in
`kSTRONGLY_TYPED` flag**, which exists in 10.x (`trtexec --stronglyTyped`).

[PUBLISHED — NVIDIA staff, developer forum, 2026-07-24] FP8/FP4 *are* reachable on Thor tensor
cores: *"TensorRT 11's NVFP4 GEMM kernels (on Thor SM 11.0) only ship fp16/bf16 input/output
tactics, so the autotuner found 'no tactics' for the fp32 linear."* ⭐ **Note the shape of that
failure — `Autotuner: no tactics to implement operation` is LOUD, not silent.** Workaround: cast
to fp16 before quantising. NVIDIA also reproduced poor Thor INT8 internally (+37 % on Orin vs
**+2.7 % on Thor**) and their recommendation was **ModelOpt explicit quantization +
`--stronglyTyped`** — i.e. **NVIDIA prescribes exactly this migration, on our hardware.**

## F6 — Verifying achieved precision: the instrument, and three caveats that would each break the gate

[PUBLISHED — `inference-library/engine-tools.html`] `IEngineInspector` →
`getLayerInformation()`; requires `ProfilingVerbosity=kDETAILED` (default prints names only).
CLI: `trtexec --profilingVerbosity=detailed --dumpLayerInfo --exportLayerInfo=<file>`.
NVIDIA's own canonical recipe [PUBLISHED — NVIDIA developer blog, Ruixiang Wang, 2026-06-09]:
`trtexec --onnx=… --stronglyTyped --saveEngine=…`, where `--stronglyTyped` *"forces TensorRT to
respect the precision annotations that ModelOpt baked into the ONNX graph, ensuring our FP8
weights and activations actually execute in FP8."*

⛔ **Three caveats — settle these before building the gate on layer info:**
1. **Documented:** with **dynamic shapes**, dims show as `-1` and *"the tensor format information
   will not be shown."*
2. ⚠️ **UNSETTLED:** the layer-info JSON exposes `"Datatype": "Int8"` — that is a **tensor
   dtype**, not provably the layer's **compute precision**, and a second reader of the same page
   saw no precision field at all. ⇒ **MEASURE this on our actual TRT 10.16.2 Thor build before
   trusting it.**
3. ⛔ **NVIDIA (forum 71064): *"The network layers will always report FP32. The engine layers are
   the ones which will have different precisions."*** Never read `getPrecision()` on the
   **network** — it is meaningless, and it would return FP32 for a correctly-quantised engine.

**Tools that do NOT solve it:**
- **Polygraphy** — `--trt-outputs mark all` carries a documented Heisenberg problem: it *"can
  perturb the generated engine due to differences in timing, layer fusion choices, and format
  constraints, which can hide the failure"* — it breaks the very Q/DQ fusions being validated.
  Good for localising numeric divergence; **not a precision-attestation tool.**
- **ModelOpt** — `print_quant_summary()` reports **inserted quantizers** (pre-build *intent*),
  not achieved precision; its eval harness emits only top-1/top-5 and latency.
- **Nsight DL Designer** "precision donut" is NVIDIA's own visual check, ⚠️ but it ships TRT 11.1
  and is a **host GUI** — **could not confirm it runs on Jetson aarch64**; treat as x86-host-side
  until measured.
- **TREx** (experimental) parses layer-info JSON to pandas; a precision column is plausible but
  **UNCONFIRMED** from its README.

## F7 — Literature: nothing published audits whether a compiled engine ran in its claimed precision

[PUBLISHED, banked] The literature covers *whether quantised math degrades accuracy* and
*formal bit-exact verification* — **not** whether an accelerator engine executed in the precision
it claims. That gap is filled only by vendor tooling and bug reports. Nearest neighbours:
`2204.04220` (quantised-vs-FP32 **disagreement** analysis; `Margin` the best indicator;
disagreements concentrate under distribution shift) · `2207.06282` (metamorphic search *hunting*
inputs where precisions disagree) · `2012.08185` (bit-exact QNN verification is **PSPACE-hard**
vs NP for real-valued — verifying what the hardware ran is categorically harder) ·
`2601.12638` (**most relevant: AV + TensorRT** — one-layer-at-a-time INT8 → AP → sensitivity
rank; a reusable per-layer audit loop, but it does **not** verify achieved precision) ·
`2603.08747` (NVFP4/MXFP4 layer-wise sensitivity — a *prior on where fallback would hurt*) ·
`2606.06527` · `2510.25602` · `2605.20868` (runtime per-head error bounds vs an FP16 reference).

⛔ **A caution worth propagating beyond this package:** during this pass a search summariser
produced a **confident, concrete per-layer SQNR example** (ResNet-50, *"48 layers > 38 dB,
conv4_3 at 18–24 dB"*) attributed to a real NVIDIA blog. The blog was opened and **those numbers
are not in it.** The *technique* (per-layer SQNR vs an FP32 reference) is standard and right for
us, but **no primary NVIDIA document specifies it for TensorRT engines.** Same family as our own
scope-error traps, and it nearly entered this report as a MEASURED-looking figure.

---

## What this changes for TanitAD (≤3)

1. ⭐ **Re-scope `D-B1-GATE` from a platform blocker to a build-path defect.** The row currently
   reads as though Thor's FP8 path is unusable pending #4590. F2 + #4599 say FP8 **works on Thor
   with explicit Q/DQ**; the false-positive risk comes from **builder-flag builds**. ⇒ the gate is
   un-blockable **today** on TRT 10.16.2 by migrating to **ModelOpt explicit Q/DQ +
   `--stronglyTyped`** — which is NVIDIA's own recommendation for this hardware (F5).
2. ⛔ **Make the gate's first stage STRUCTURAL, not numerical** (F3). Add three near-free
   assertions ahead of any accuracy stage: **plan-file size vs the FP16 baseline** (#4590's own
   tell: 199 MB FP16 vs 382 MB "FP8"), **output `DataType` enumeration**, and **`Total Weights
   Memory`** from the build log. These catch whole-engine fallback immediately; the accuracy
   comparison cannot, by construction.
3. **Before depending on `IEngineInspector` for per-layer precision, MEASURE that a usable
   precision field is emitted on our TRT 10.16.2 Thor build** (F6 caveat 2), and hard-code the
   rule that `getPrecision()` is read on the **engine, never the network** (caveat 3). Budget the
   documented FP8-conv fallbacks as **latency**, not as an accuracy risk (F4).

## Named empty searches

- **A published methodology for detecting silent precision fallback in a compiled inference
  engine**: NOT FOUND (two probes: quantisation-validation literature, and formal-verification
  literature). F7.
- **Confirmation that Nsight DL Designer runs on Jetson aarch64**: NOT FOUND — UNVERIFIED.
- **A TREx precision column**: UNCONFIRMED from its README.
- ⚠️ **`2603.08747` ID confirmed** (abstract page loaded twice: Cim, Topcu, Kandemir; 2026-03-05;
  cs.AR; ICLR 2026). Unverified and **not quotable**: `2106.05997`, `2306.13793`, `2603.04308`,
  `2602.05902`, `2411.09909`.

## Banked primaries

NEW this package: `2204.04220` · `2207.06282` · `2012.08185` · `2601.12638` · `2603.08747` ·
`2606.06527` · `2510.25602` · `2605.20868`. **Non-arXiv primaries are URLs, not banked PDFs** —
TRT issues #4590 / #4599 / #4020 / #3737, the TensorRT dev guide, the 10.16.1 and 11.0.0/11.2.1
release notes, the JetPack 7.2.1 notes, NVIDIA developer-forum threads, and the 2026-06-09
NVIDIA blog. ⚠️ Their evidence class is **PUBLISHED (cited URL)**; a GitHub issue and a release
note are **moving targets** — re-verify at gate-build time rather than inheriting F1's state.
