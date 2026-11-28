# LATENCY-BUDGET: decision memo for the PI (2026-10-20; mirror tip `abd4dd81`)

> **PI_DECISION**
> **Q:** what absolute planning-latency budget (cycle, p99 deadline, per-stage share) do REFe/P2/P6 and the CEM search answer to on Thor? (`LATENCY-BUDGET`; `MM_RULING_C2AMEND` DQ3/DQ4)
> **Options:** (1) strict, one cycle 10 Hz / p99 100 ms; (2) two-rate, pick path 10 Hz / p99 100 ms plus a search stage at 2 Hz / p99 300 ms on cached features; (3) relaxed, one cycle 5 Hz / p99 200 ms.
> **Recommend (2)**, fallback pre-committed: if the Thor trunk number says the pick path cannot make 100 ms, it runs at 5 Hz / 200 ms (REFe's native rate) and nothing else reopens. P6 is not gated; its verdict gains a latency line.
> **Known:** MEASURED on the REFe stack on Thor = nothing; every REFe-on-Thor figure below is ESTIMATED. **Needed:** one Thor slot for "M1" (trunk + pick tick), which `SEARCH-THOR-LATENCY` does not cover and which decides which option is reachable.

## 1. What we hold (repo, tip `abd4dd81`)

| item | number | class / tier | artifact |
|---|---|---|---|
| A working budget exists, never ruled | 100 ms / 10 Hz; Mission Plan: brains at "100 ms vs 500 ms vs 1 second (subject of investigation)". The BACKLOG row's "exists nowhere" is too strong | INHERITED design text; old flagship line only | `Mission Plan.md` l.44; `THOR_PROFILE.md`; `PROGRAM_OVERVIEW.md` 5.0.2; `MPC_WM_DESIGN.md` E3 (never run) |
| Thor, old flagship WM (263.6 M, 1 cam) | tick 98.63 / p99 102.57 ms (bf16 encoder + graph predictor); flagship step-29999 full tick (encoder, heads, 9-candidate fan) 60.3 / 63.1 / 65.3 ms (p50/p95/p99) | MEASURED (Thor), latency only, **different model** | `thor_combined_tick.json`; `thor_d6_tick_intent_K20.json` |
| Thor, REFe/P2 trunk, scorer, search | **none**; absent from the registry, goals and REFE plan docs (C2 s7 probe) and from a `git grep` of the tip | UNMEASURED | `AMENDMENT_C2.md` s7 |
| Dev-box CPU (not Thor), fp32, n=5 | pick-only 61.1; C8 870.3 (14 calls); O4 2,703.3 ms (38 calls). Host numpy inside: 55.5 / 214.5 ms. O4/C8 = 3.11x | MEASURED (CPU) | `2026-10-20-search-thor-latency/raw/devbox5/` |
| `SPEC_P6` | **no inference-latency text**; its only latency is the training labeller (0.9 s/micro-batch). P6 keeps P2's graph, so it changes no inference cost, and is already training | MEASURED (training) | `SPEC_P6_PAPER_FAITHFUL.md` 3.2 |
| Value at stake | C8 search +0.5575 [+0.1058, +0.9253] PDMS points over P2's pick (C1, O1_CONFIRMED), selection tier, label-only, 14 scorer calls/token | MEASURED (C1; corrected by the MM: it was on the tip at `0d5cef5`, before this memo's base) | `raw/2026-10-19-search-p2-confirm/c1/RESULT_SPC.json`; `H-SEARCH-P2-1` |

## 2. Literature (PUBLISHED; primaries banked, keys in `raw/EVIDENCE_LEDGER.md`)

- **100 ms is a norm, not a law.** Alpamayo-R1 reports 99 ms end to end on an RTX 6000 Pro Blackwell and calls ~100 ms typical (3.4 vision + 16.5 prefill + 70 reasoning + 8.8 trajectory; trajectory-only 29 ms; 2511.00088). Apollo's EM planner targets a 100 ms emergency reaction, planner under 100 ms on average (1807.08048). A real L4 Autoware.Universe vehicle: planning stage mean 119 / p99 267 ms, perception-to-decision mean 522 ms (2510.11448).
- **NAVSIM-class planners: none on Thor, and they disagree.** DiffusionDrive 45 FPS on a 4090 but 7 FPS with 20-step sampling (2411.15139); Hydra-MDP++ 206-271 ms on a V100 (2503.12820); GTRS's ViT-L scorer 400 ms/forward on an A100, batch 1 (2601.05083). Orders of magnitude only. **DriveZero (2609.06055) publishes no inference latency**: no target to match.
- **Parallel candidates are nearly free, sequential calls are not** (WoTE: 64 -> 256 trajectories +1.5 ms on an L20, 2504.01941). Closed loop, rate and tails both matter: UniAD +20 % driving score from 1 to 20 FPS; VAD with realistic frame drops lost 24.6 % (2601.07393; CARLA, not NAVSIM).

## 3. Options (ESTIMATED; arithmetic in `raw/budget_arithmetic.py`)

The trunk is the first-order term. REFe encodes 4 cameras x 960x512 through a ViT-L: **6.11 TFLOP** per cycle (E). At Thor's bf16 GEMM rate (126.6 TFLOPS, INHERITED) that is **48 ms at 100 % utilisation, 80 ms at 60 %, 121 ms at 40 %**; the A40 training step (MEASURED 0.2227 s) implies 74-111 ms forward-only. The central ~80-110 ms already sits on the 100 ms line before any scorer or search.

| | **(1) strict** | **(2) two-rate (rec.)** | **(3) relaxed** |
|---|---|---|---|
| cycle / p99 | 10 Hz / 100 ms | pick 10 Hz / 100 ms; search 2 Hz / 500 ms | 5 Hz / 200 ms |
| shares, ms | trunk 65, proposals+scorer 10, search 10, margin 15 | pick 65 / 10 / margin 25; search 300 + 200 hand-off | trunk 100, proposals+scorer 10, search 60, margin 30 |
| search as pinned (C8 14 / O4 38 dependent calls) | **neither fits** (best case 28 / 76 ms vs 10; with the final stage batched, 6 calls, still 12). Search becomes a distillation teacher | **C8 fits all 4 cost regimes** (28-112 vs 300); **O4 fits 3 of 4** (76-304) | C8 only in the best regime (28; others 63-112 vs 60); O4 never; with 6 calls both fit (12-48) |
| P6 (geometry locked) | trunk gate <= 65 ms: likely fails at eager bf16 (E 48-121); needs FP8/TensorRT or fewer cameras | pick path as (1); search separate | trunk gate <= 100 ms: passes the optimistic half only |
| main risk | pick path may not fit | plan applied up to a cycle late: **gain UNMEASURED** | below the 10 Hz norm; closed-loop cost |

Cost regimes (U knobs): scorer call 1.5 ms (CUDA graph) or 4 ms (eager), plus host numpy 0.5 ms (ported to GPU) or 4 ms (dev-box rate; ARM factor UNVERIFIED). On the dev-box CPU the host is 6-8 % of the search; on a GPU the scorer shrinks from ~59 ms to a few and the numpy stays, so **porting `composite`/`aggregate` is a precondition of every option**. Reading `tts_cem.py` (sha `9dea9603`) shows the n_elite final re-scorings are independent single calls, so the dependent depth is 6 (K 5 + 1), not 14 or 38; batching them is a code change (new pin, identity control).

## 4. Recommendation, and what would change it

**Adopt (2):** it is the Mission Plan's own multi-rate design, leaves the always-on path alone, and keeps both searches deployable in principle while charging the staleness openly. Wording rule: a search config is "deployable on Thor" only if a Thor JSON with `controls_ok` shows its stage p99 <= 300 ms, the pick path meets its own deadline, and the stale-plan read (M3) keeps the gain. Until then DQ3's wording stands.

**Changes it:** (a) M1 trunk+pick p99 in (100, 200] ms: pick path at 5 Hz, structure unchanged; above 200 ms nothing is deployable at 5 Hz and "keep our encoder" (PI, 2026-10-11) must reopen first. (b) Graph-captured scorer call above ~6 ms, or numpy not portable: option (1). (c) M3 puts the gain's lower CI at or below 0: option (1). (d) A sense-to-act requirement under ~150 ms (COLA, secondary): option (1). (e) Orin as target (named in the Mission Plan, nothing measured): re-set.

## 5. To MEASURE on Thor

`SEARCH-THOR-LATENCY` (built; 117 tests, 50/50 mutations; **not run on Thor**) covers the scorer-only pick, C8 and O4 at p50/p95/p99, call/row counts, scorer-vs-host split, identity/determinism controls, device state, one seed. It **excludes the trunk and proposal generator by construction** (so `ratio_to_a` overstates; quote `added_ms_vs_a`). Not covered:

- **M1 (decides the option):** trunk + proposals + pick on real P2 weights, 4x960x512 incl. decode/undistort/H2D, bf16 eager and graph/TensorRT, p50/p95/p99 over >= 200 windows plus 180 s sustained (`THOR_PROFILE.md` addendum A).
- **M2:** the depth-6 batched search vs the pinned path, with the identity control. **M3:** the searched plan applied 1-5 frames late (selection tier first).
- **M4/M5:** contention with the rest of the Thor stack; Orin; sense-to-act. Not measurable now; margins carry them.

**Split.** MEASURED: old-line Thor ticks, dev-box CPU search, A40 step. PUBLISHED: section 2. INHERITED: the 100 ms norm, Thor GEMM rate, +0.56. ESTIMATED: every REFe-on-Thor number. No navtest label or pre-registered arm was touched; P2's run config was not re-checked (geometry = `refe/model.py` default, UNVERIFIED).
