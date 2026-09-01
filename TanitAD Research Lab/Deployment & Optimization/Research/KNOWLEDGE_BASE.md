# KNOWLEDGE_BASE — Production & Optimization

> **Curated, deduplicated, newest first.** Format:
> `- [YYYY-MM-DD] [source] finding (1-3 lines) — impact: H_x / WP_y — link`
>
> This is the **Production & Optimization** agent's findings log. The router across all areas is
> [`../../KNOWLEDGE_BASE.md`](../../KNOWLEDGE_BASE.md).
>
> ⛔ **Three layers, and they are not interchangeable.**
> **PAPER** (`Paper/TANITAD_PAPER.md`) = the scientific account of the frontier work — derivations,
> argument, results in narrative form.
> **KNOWLEDGE_BASE** (this file) = what we learned, written for an agent about to make a decision.
> **LIBRARY** (`../../Library/`) = the evidence. Every `[PUBLISHED]` entry cites a **library key**,
> not only a URL — bank it with `python tools/kb_add.py <arxiv-id> --tag <topic> --cited-by <report>`.

⚠️ **This area carried no entries through the 2026-08-18 restructure.** That is an emptiness to be
filled, not a statement that nothing was learned here — deployment, latency and quantisation
findings have been living inside other areas' reports. File them here.

- [2026-08-30] [PUBLISHED/github+api] **TRT #4590 is STILL OPEN** — verified twice (HTML + REST API): state open,
  created 2025-10-04, updated 2026-01-09, **1 comment from `author_association: NONE`** (a user, not NVIDIA), no
  fix version, no workaround; labels `Module:Quantization`/`Module:Runtime`; reported on TRT 10.13.3.9 / SM 110 —
  impact: D-B1-GATE — `2026-08-30-b1-gate-strongly-typed/RESULT.md` F1
- [2026-08-30] [PUBLISHED/nvidia-docs] ⭐⭐ **REFRAME: #4590 is a DIAGNOSTICS bug, not "FP8 is broken on Thor".**
  TRT docs: *"FP8 quantization exclusively supports explicit quantization via Q/DQ layers… implicit quantization
  is limited to INT8 only."* The reporter set builder flags on a graph with NO Q/DQ nodes ⇒ nothing was ever
  going to be quantised; the real defect is the MISSING WARNING. Issue #4599 shows FP8 DID engage on Thor with
  ONNX Q/DQ surgery (~20 % latency win, ViT-Base). ⇒ our exposure is "we built via builder flags", not the
  platform — impact: D-B1-GATE re-scope, un-blockable today — same RESULT.md F2
- [2026-08-30] [derived] ⛔ **A NUMERICAL CHECK CAN NEVER CATCH THIS: comparing FP32 against a secretly-FP32
  engine yields ZERO error — the failure makes the gate score BETTER.** Detection must be STRUCTURAL: plan-file
  size vs the FP16 baseline (#4590's tell: 199 MB FP16 vs 382 MB "FP8"), output `DataType` enumeration, `Total
  Weights Memory` from the build log. Same family as the jpeg_buf-is-png all-zero FLOOR arm — impact: gate stage
  order — same RESULT.md F3
- [2026-08-30] [PUBLISHED/nvidia-docs] **Strongly-typed does NOT abolish fallback — it converts a silent ACCURACY
  lie into a visible SPEED loss.** Builder-flag+no-Q/DQ ⇒ the whole quantisation is lost and the gate lies;
  strongly-typed+Q/DQ ⇒ only the kernel/tactic falls back while Q/DQ nodes still round the values, so the
  accuracy number stays valid. Documented residual fallbacks: FP8 convs with kernel > 32 (SM89/90/120/121 —
  ⚠️ SM110 NOT listed) and FP8 group convs — impact: budget as latency, not accuracy risk — same RESULT.md F4
- [2026-08-30] [PUBLISHED/nvidia-docs] ⛔ **VERSION WALL: TRT 11 removes weak typing (bug class impossible) but
  "JetPack is not supported in TensorRT 11.2.1 — Jetson deployments must remain on a TensorRT 10.x release".**
  JetPack 7.2.1 (2026-08-11) ships **TRT 10.16.2** on Thor ⇒ our fix is the opt-in `kSTRONGLY_TYPED` /
  `trtexec --stronglyTyped`, available now. NVIDIA staff prescribe **ModelOpt explicit quantization +
  `--stronglyTyped`** for Thor specifically (they measured +37 % Orin vs +2.7 % Thor INT8) — impact: D-B1-GATE
  migration path — same RESULT.md F5
- [2026-08-30] [PUBLISHED/nvidia-docs] ⚠️ **Three caveats before trusting `IEngineInspector` for per-layer
  precision:** (1) with dynamic shapes dims read `-1` and *"tensor format information will not be shown"*;
  (2) the JSON's `"Datatype"` is a TENSOR dtype, not provably the layer's COMPUTE precision — MEASURE it on our
  TRT 10.16.2 Thor build first; (3) ⛔ NVIDIA: *"The network layers will always report FP32. The engine layers
  are the ones which will have different precisions"* — never read `getPrecision()` on the NETWORK. Polygraphy's
  `--trt-outputs mark all` is NOT an attestation tool (documented Heisenberg: it perturbs the very Q/DQ fusions
  under test) — impact: gate implementation — same RESULT.md F6
- [2026-08-30] [PUBLISHED/library] **No published work audits whether a compiled engine RAN in its claimed
  precision** (two probes). Nearest: `2204.04220` (quantised-vs-FP32 disagreement; `Margin` best indicator),
  `2012.08185` (bit-exact QNN verification is PSPACE-hard), `2601.12638` (AV+TensorRT per-layer sensitivity
  loop), `2603.08747` (NVFP4/MXFP4 layer-wise sensitivity = a prior on where fallback hurts). ⛔ CAUTION: a
  search summariser fabricated a concrete per-layer SQNR example attributed to a real NVIDIA blog — the blog does
  NOT contain those numbers — impact: quant-gate literature basis — same RESULT.md F7

- [2026-08-31] [RELAYED lib `2601.22156` BANKED-BUT-UNREAD] Transformer->hybrid linear-attention DISTILLATION is claimed at ~25% of from-scratch training budget; an SSM/linear-attention predictor carries O(1) state rather than a growing KV cache.
  impact: **the only architectural family that could make the K~300 strategic rollout fit the budget today's rollout-depth measurement says nothing else fits; RELAYED, so it may NOT decide a GPU-day until the primary is read** - `Frontier Scan/Daily/2026-08-31/RESULT.md` F5

- [2026-08-31] [RELAYED/UNECE-GRVA trade summary 2026-07] **GRVA reportedly discussed 'the ban of online learning' in vehicles - this lands directly on INJECTED BACKLOG ROW I-3 (online encoder updates from T1 rollout disagreement at deployment).** If adopted, I-3 survives only as an OFFLINE/fleet-loop method, never an in-vehicle one.
  impact: **constraint-to-VERIFY, not a constraint - it is a relayed report of a DISCUSSION and no GRVA primary was read. Read the primary BEFORE designing I-3** - `Frontier Scan/LEDGER_C3_regulatory.md`

- [2026-09-01] [MEASURED ours, RTX 4060, 120 reps] **AT EQUAL ROLLOUT BUDGET, BREADTH COSTS 5.94x LESS THAN DEPTH** - B=960 predictor-steps spent as (N=16, K=60) costs **73.33 ms**; as (N=320, K=3) costs **12.34 ms**. Breadth grows only **11.1x for 960x the candidates** (flat to N~32 - extra candidates are FREE), while depth is linear in K. The terminal value head that buys a small K costs **0.150 ms = 1.2 %**, flat in N.
  impact: **"equal budget" means two different things (steps vs latency) and they differ ~6x - a step-parity comparison silently handicaps the deep arm; TD-MPC2's H=3 is also the cheapest point on OUR measured budget surface. ⛔ COST half of injected I-1 only - the QUALITY half is untouched and is the real experiment** - `Deployment & Optimization/Research/2026-09-01-terminal-value-vs-fan-scoring/RESULT.md`
