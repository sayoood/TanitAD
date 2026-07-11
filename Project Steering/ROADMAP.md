# TanitAD — Roadmap

> Cross-backlog, production-readiness roadmap (D-029, Sayed commission, authored 2026-07-12).
> Synthesis only — every entry below traces to a real backlog line, gate, or directive (cited
> inline). No invented items. Source snapshot: `HEAD=1787c3b` (driving-diagnostic harness shipped,
> results pending), REF-A pool+grid both at 30k, REF-B not yet launched, pod1 main run 90% and
> crawling at the 62 GB cgroup cap. Maintained by: Project Steering / orchestrator (weekly refresh
> alongside the W-report; re-sequence whenever a Now item lands or a diagnostic result changes the
> routing).

**Legend (file abbreviations used in "traces to" columns):**
`T&D`=`TanitAD Research Hub/Tools&DevEnv/BACKLOG.md` · `DE`=`.../Data Engineering/BACKLOG.md` ·
`AI`=`.../Architecture & Inference/BACKLOG.md` · `BE`=`.../Benchmarks & Eval/BACKLOG.md` ·
`OA`=`.../Opponent Analyzer/BACKLOG.md` · `PO`=`.../Production & Optimization/BACKLOG.md` ·
`DIAG`=`Benchmarks & Eval/DRIVING_DIAGNOSTIC_FRAMEWORK.md` · `LB`=`Benchmarks & Eval/LEADERBOARD.md` ·
`REFARCH`=`Project Steering/REFERENCE_ARCHITECTURES.md` · `HL`=`.../HYPOTHESIS_LEDGER.md` ·
`P0P`=`Project Steering/Phase 0 Plan.md` · `D-0xx`=`DECISIONS.md` entry.

---

## 1. North star + Phase-0 exit gate

**North star** (D-008, `HL` H0, `LB` competitor-efficiency block, `OA/SCENARIO_DATABASE.md` intro):
a sub-300 M hierarchical latent world model — "40× smaller than Alpamayo-1-class VLAs" — that wins
not on scale but on **hierarchy + imagination + self-monitoring + per-scenario excellence** against
opponents 15–120× larger (Alpamayo-2 32 B, GAIA-3 15 B). H0 ("E2E superiority") is already `confirmed`
in the ledger; every other hypothesis (H1–H16) is instrumented evidence toward that claim. The
ultimate falsifiable target is stated in the Opponent Analyzer's own charter: *prove TanitAD excels
at every scenario in `SCENARIO_DATABASE.md`.*

**But the program has one binding constraint right now, and this roadmap orbits it** (D-029
rationale, verbatim: *"The single-camera driving gap (D1 6.44 m) is the top risk every agent's
experiments should now orbit"*). Quoting the revised exit criteria in full, because every Now/Next
item below is sequenced against it (`DIAG` §3):

> **Phase 0 is NOT done at "gates measured."** It is done when, on single-camera + history:
> 1. **Open-loop:** model beats constant-velocity AND go-straight at ADE@{1,2}s on BOTH straight and
>    curve strata (not just aggregate) — i.e. it demonstrably tracks the road, curves included.
> 2. **Closed-loop (the real test, per 2605.00066):** MetaDrive/CARLA route completion above a
>    defined floor with imagine-and-select planning — proves the D2 ranking translates to driving.
> 3. **Decodability:** held-out ADE within a defined factor of the oracle ceiling (readout
>    generalizes).
>
> **Only then do more cameras / sensors / the H-stack proceed.**

Where we stand against it today: D1 = **6.44 ± 0.55 m FAIL** (structurally near the sub-metre bar,
camera-frame unit — `LB` flagship, step 27,000/90%-trained); D2 = **PASS** (dir-acc 0.864, P4 0.971 —
ranking works); D3 = **FAIL, K-step-improved** (imagined 1.97 m vs oracle 1.52 m, ratio 1.30). None of
criteria 1–3 above are met yet. Section 2 is this roadmap's answer to "what closes that gap."

---

## 2. Now / Next / Later

Counts: **9 Now** (this week) · **6 Next** (2–4 weeks) · **5 Later** (Phase 1). Every row cites its
trace; "Falsifier" is the pre-registered condition that kills the claim, not just a success metric.

### NOW — this week (driving-gap-critical or already in flight)

| # | Discipline | Deliverable | Traces to | Target / falsifier | Readiness target | Cost |
|---|---|---|---|---|---|---|
| N1 | Benchmarks&Eval + Architecture | **Diagnostic §A — baseline-relative read.** Run CV / go-straight / constant-yaw-rate baselines through the shipped harness against the 27k ckpt. | `DIAG` §1.A ("FIRST, decisive"); `AI` P0 #3e2 (CV kinematic-floor row, "7.51 is bad too" — Sayed) | If model beats all 3 baselines at every horizon → genuinely broken; if CV also ≈5–7 m → D1's sub-metre bar is harsh, not the model. Falsifier stated inline. | Prototype→validated (harness `stack/scripts/driving_diagnostic.py` shipped `1787c3b`, 694 lines+tests; **not yet run** — §Results still empty) | Local/pod1, hours, $0 |
| N2 | Benchmarks&Eval + Architecture | **Diagnostic §B — decode ladder** (ridge/MLP/model-head/oracle-ceiling, fit=eval vs held-out). | `DIAG` §1.B — *"routes the entire rest of the program"* | Oracle-ceiling small (~1–2 m) + held-out 6.44 m → readout/generalization bug; oracle-ceiling also large (~5 m+) → representation bug. | Same harness, same gap (not yet run) | Local/pod1, hours, $0 |
| N3 | Benchmarks&Eval + Architecture | **Diagnostic §C+D — error localization + step-curve** (ADE by curvature/speed/corpus stratum; ADE vs step-count on 8.5k/14k/27k ckpts, all already on hand). | `DIAG` §1.C, §1.D ("cheap — run on existing checkpoints") | `straight`≈1–2 m + `sharp`≫ → capability gap (data/planning fix); `straight`≈6 m too → fundamental (representation/training) fix. | Same harness (not yet run) | Local, hours, $0 |
| N4 | Architecture | **REF-A full-realmix retrain** — comma+PhysicalAI mixed corpus (PhysicalAI pod2 build COMPLETE, 401 train/100 val), pool- and grid-adapter both already at 30k comma-only (pool ridge_a10 **17.01** vs grid **20.22** — grid REFUTES the mean-pool-confound hypothesis directionally); full-mix training is the de-confounded, decisive run. | `AI` P1 #3d1/#3d2; program report `2026-07-12-0130` §2–3 | Closes/confirms the 14.2-vs-7.5-m (confounded, W31) gap at matched protocol; falsifier: full-mix grid-REF-A still ≫ main → frozen-encoder deficit is real (H4 evidence), not artifact | Validated experiment, prototype result (comma-only leg done; full-mix leg in flight) | Pod2 A40, ~4.5 h, $0 incremental (rides approved pod) |
| N5 | Architecture | **REF-B first full training launch** (rev2: strategic transformer d384×4 + route-nav commands, budget-matched 260.7 M, built+merged) — blocked on pod3 comma-extraction completing. | `REFARCH` REF-B section; `PS` §4 ("training GO now only awaits Sayed's rev2 look"); `RESOURCE_LEDGER.md` row `refb-post30k` ($40 confirmed in chat) | D4 arbitration: does imagine-and-select beat a matched-budget hierarchical BC stack? Falsifier: REF-B matches us on D8/LOPS/SC-01 (structurally shouldn't) → WM premise wounded | Prototype (code built/tested; 0 training steps run at scale yet) | Pod1/pod3, ~2 days, $40 envelope (pre-confirmed) |
| N6 | Tools&DevEnv | **TanitResim continuous development** — dual-sink `.rrd` empty-file bug, live-proxy gRPC path; scenario-cards + side-by-side arm panels + single-port serving just shipped (`e3d9757`). | `tools-devenv-agent.md` P1 (D-029); `T&D` P0 #2 | One clean replay bundle per eval episode (predicted-vs-actual + BEV); falsifier: dual-sink still drops frames post-fix | Validated, advancing (real app; single-port sidesteps the proxy variant of the bug, root case still open) | Local/pod2 deploy, hours, $0 (push-to-main, no intake) |
| N7 | Tools&DevEnv | **TanitScena v1** — parse `SCENARIO_DATABASE.md` (14 entries) into a structured store + local vector index (semantic search) + TanitResim-language UI. | D-029 (2); `tools-devenv-agent.md` P2 | Top-1 retrieval correct on ≥8/10 held-out natural-language queries; falsifier: worse than keyword grep | **Prototype, pre-code** (`tanitscena` branch touched only `DECISIONS.md`+agent docs; `stack/tanitad/scena/` does not exist yet) | Local, days, $0 |
| N8 | Benchmarks&Eval | **nuScenes-mini bench harness** — loader + metric trajectory head + community L2@{1,2,3}s + collision rate, reported WITHOUT the AD-MLP ego-status shortcut. | `DIAG` §2 ("Build: nuScenes-mini loader... Phase-0.5 build"); `LB` open-loop table (TanitAD row "—", "Phase 1 target: first entry"); D-002 (nuScenes-mini named for OOD probes) | First honestly-placed community-standard number with the shortcut caveat attached; no pass/fail yet — this is placement, not a claim | **Prototype, not started** (no nuScenes loader exists in `stack/` yet) | Local/4060, days, $0 |
| N9 | Data Engineering | **Realmix HF recipe-dataset + `DATA_MANIFEST.json`/`rebuild_cache.py`** — kills the pod1 single-point-of-failure exposed by the 2026-07-11 rsync stall. | `DE` P0 #-2/#-1 (Sayed 2026-07-11 night) | Any pod self-provisions the exact corpus, verified by sampled-episode SHA256 vs pod1; **already proven twice** (pod2 + pod3 rebuilt from origin, program report `0130`) — remaining: HF recipe-dataset v1 (402-ep, PRIVATE) still needs Sayed's public flip | Validated (rebuild proven 2×); recipe-dataset itself still prototype | Local+pod, hours, $0 |

### NEXT — 2–4 weeks

| # | Discipline | Deliverable | Traces to | Target / falsifier | Readiness target | Cost |
|---|---|---|---|---|---|---|
| X1 | Architecture (routed by N2) | **Readout-vs-representation remedy** the diagnostic selects: nonlinear/learned trajectory head + route-diverse probe training + calibration (if readout) OR architecture/training-signal/model-size change (if representation). | `DIAG` §1.B, §1.F ("model's own head" arm) | Whichever arm is selected must move held-out ADE beyond N1–N3's noise band; falsifier: no movement | Prototype (not buildable until N2 routes it) | Pod (A40/A6000), days, $0–40 |
| X2 | Data Engineering + Architecture | **Data-mix ablation + curve-oversampling** — comma-only vs pai-only vs mixes (does the 0.6-physicalai realmix dilute highway?); + semantic/strategic-label dataset survey (nuPlan/DriveLM/CoVLA/L2D/Talk2Car/AUTOPILOT-VQA/Bench2Drive) for the curve/maneuver-scarcity remedy. | `DIAG` §1.D; `DE` P1 #2d (Sayed directive, from the REF-B strategic-layer `follow`-only defect) | Comma-only improves straight-highway ADE if dilution is real; curvature-bucket population measured (is `sharp` <5% of windows, i.e. genuinely starved) | Prototype (survey not started; ablation needs N3's stratum counts first) | Pod (training arms) + local (survey), days |
| X3 | Architecture | **Resolution-sensitivity probe** — encode val at 128/256/384 px (ViT pos-emb interpolated, no retrain), re-probe ADE per curvature stratum + far-hazard LAL slice. | `AI` P1 #3e0 (Sayed question, 2026-07-11 night); `DIAG` §1.G | 384 helps `sharp`/far-hazard strata but not `straight` → resolution caps maneuver/long-range acuity specifically; falsifier: flat deltas → 256 is not binding | Prototype, not started | Local/4060 or idle pod, hours, $0 |
| X4 | Tools&DevEnv + Benchmarks&Eval | **Closed-loop route-completion** with imagine-and-select planning — CARLA-on-pod is the actual substrate (D-014); **note:** `DIAG` §1.F/§3.2 names "MetaDrive/CARLA" together even though D-014 retired MetaDrive for training/sim-arm use — flagging this tension rather than silently picking one. | `DIAG` §1.F, §3.2 (exit criterion 2); D-014; `T&D` P1 #3 (graphics-pod dry-run, "NOT urgent") | Route completion above a defined floor; the open-loop⊥closed-loop footnote (2605.00066) means D1 may be pessimistic — this is the actual test | Prototype (nullrhi physics validated; camera-driven ego needs the graphics-pod recipe, not fired yet) | Pod2 (existing, nullrhi) + graphics-pod only if pixels become critical-path, $0 baseline |
| X5 | Benchmarks&Eval | **Community-benchmark placement maturing** — N8's nuScenes harness → NAVSIM v2 EPDMS / Bench2Drive entry-requirements readiness checklist (formats, licenses, compute budgets). | `BE` P2 #6; `DIAG` §2 | Readiness checklist complete before the post-D4–D6 entry; open-loop⊥closed-loop caveat stays attached to any number | Prototype, not started | Local, days, $0 |
| X6 | Architecture | **Operative-scale K-step confirmation** (K∈{1,2,4} from pod2 ckpt, `imag_rel`/horizon, NOT dir-acc — proven to saturate) + extend imagination horizons past 0.4 s to cover D3's 2 s target. | `AI` P0 #2b/#2c (escalate before touching trained config, D-018) | K=4 (already adopted, D-027) confirmed at decision-grade scale with no regression at ≤4-step horizon; couples directly with D3 | Validated at reduced scale (−64%/−87%/−92% imag_rel, matched-compute); decision-grade sweep not yet run | Pod2 (existing), hours, $0 |

### LATER — Phase 1 (only after the Phase-0 exit gate above is met)

| # | Discipline | Deliverable | Traces to | Target / falsifier | Readiness target | Cost |
|---|---|---|---|---|---|---|
| L1 | Architecture | **Multi-camera encoder scaling** — weight-shared batched encode (N∈{1,4,7}), static token pruning (sky/hood masks), triplane/temporal efficient encoding. | `AI` P1 #3d; `ENCODER_MULTICAM_OPTIMIZATION.md` attack order; `DIAG` §1.H (explicitly **gated on §G/X3**) | 7-cam ≈2–2.5× single-cam latency, not 7×; falsifier: batching doesn't amortize (VRAM-bound) | Investigation doc only; 0 measurements yet | Local/4060 first pass, then pod-class |
| L2 | Architecture | **H2 attention-based camera/modality steering** — tactical layer selects {front/+side/+rear}, routed on H15 epistemic σ. | `HL` H2 ("supported, race is on"); `P0P` G0.7 (Pareto quality-vs-FLOPs demo) | G0.7 Pareto plot on multi-view clips beats fixed-camera baseline at matched FLOPs | Open (interface ready per HL H8) | Pod-class, Phase 1 budget |
| L3 | Architecture | **H16 active depth interrogation** — tactical-commanded, σ-triggered ROI depth queries (ZipDepth-class specialist). | `H16_ACTIVE_DEPTH_INTERROGATION.md` F1–F3; `HL` H16 ("Phase-1 H2 window ~Sep") | F1: trigger fires ahead of reveal on ≥70% of occlusion events at <1 query/10s false-positive; falsifier stated inline | Open (dossier only); F1 is cheap enough to pull earlier (pure replay+logs, one forward pass, 4060) | Local first (F1), then pod for F2/F3 |
| L4 | Data Engineering + Architecture | **Sensor fusion** — multi-view (front+L+R+rear, 500 clips reserved) + radar/lidar from PhysicalAI-AV. | `P0P` §2.2 row B2 (500 multi-view clips reserved for the G0.7 demo); D-002 (PhysicalAI multi-sensor feeds H2) | G0.7 modality-steering demo is the Phase-0 preview; full fusion is Phase 1 | Data reserved, not yet used | Pod-class, Phase 1 budget |
| L5 | Architecture + Project Steering | **Reference-arch three-way paper §7 section** — main vs REF-A vs REF-B, matched gate tables, judged by the architecture-design workflow panel (D-026). | `REFARCH` "Sequencing & ownership" §4 — *"the three-way comparison IS the H1/H4 evidence section"*; `Paper/TANITAD_PAPER.md` (D-020 §2) | Gate deltas outside seed noise at matched steps AND matched val routes (I3/I7) | Depends entirely on N4/N5 landing first | Local (writing), reuses N4/N5 compute |

---

## 3. Per-discipline depth pass

Each subsection sequences that discipline's **own** `BACKLOG.md` by priority (as the discipline
itself ranked it), calls out the production-readiness gap, and seeds a `GOALS.md` (D-029 §2):
1–3 measurable objectives with target numbers, for the discipline to adopt verbatim or refine.

### Tools & DevEnv
**Priority order (from `T&D` BACKLOG.md):** P0.1 `ci.ps1` (pytest+I2 tripwire, budget ≤4 s warm
overhead/≤6 s per test) → P0.2 episode→Rerun `.rrd` replay (feeds N6/TanitResim) → P1.3 CARLA
graphics-pod dry-run (blocked on a graphics-capable pod, not urgent) → P1.4 pod bootstrap v2 →
P1.5 verify Drive "Available offline" fix (~30 s/run G-E win, needs Sayed's 1 click) → P2.6
Windows/Linux path+encoding audit → P2.7 AlpaSim clone-and-inspect (Phase-1 watch only).
Plus the two standing D-029 product duties (N6 TanitResim, N7 TanitScena) which now outrank the
classic backlog in practice.

**Production-readiness gap:** TanitResim is *validated, advancing* (real code, shipped features,
one open bug); TanitScena is *prototype, pre-code* — the actual gap-closer this discipline needs is
simply starting the `stack/tanitad/scena/` implementation, not more design.

**GOALS.md seed:**
1. `ci.ps1` gates every commit at <15 s warm wall-clock; a newly-added slow fixture must make it
   exit nonzero (falsifier built into the goal).
2. TanitResim: 0 open dual-sink `.rrd` bugs + a shipped 3-arm (main/REF-A/REF-B) comparison view
   within one week of REF-B's first checkpoint.
3. TanitScena v1: parse all 14 `SCENARIO_DATABASE.md` entries, ≥8/10 correct top-1 semantic-search
   retrieval on a held-out query set.

### Data Engineering
**Priority order (from `DE` BACKLOG.md):** P0.-2 realmix HF recipe-dataset → P0.-1 data-mix-as-recipe
(manifest+rebuild, **both already substantially delivered** — rebuild proven on pod2+pod3 per the
`0130` report) → P0.0 R1 top-up to 2,000 (currently 1,926, 74 short — 2 more egomotion chunks) →
P0.1 WorldModel-Synthetic-Scenarios pose probe (decides cosmos-mirror vs IDM/H7 loader path) →
P0.2 SCENARIO_DATABASE data-sourcing (SC-04/SC-11) → P1.2d semantic/strategic-label survey (=X2) →
P1.2c Y-pilot-50 dashcam → P1.2b NuRec feasibility → P1.3 Zenseact ZOD pilot → P1.4
WorldModel-Synthetic license+pilot → P1.5 `data:physicalai` tag audit → P1.6 comma2k19 Chunk 2–10
streaming plan.

**Production-readiness gap:** the reproducibility asset (manifest+rebuild) jumped straight to
*validated* (proven twice under real incident pressure) — ahead of its own backlog's expectations.
The lagging piece is the **pose-field verdict** on WorldModel-Synthetic-Scenarios (P0.1): the loader
path (near-zero cosmos-mirror vs IDM/H7 vs video-only) is undecided and blocks three downstream
scenario rows (SC-02/05/06).

**GOALS.md seed:**
1. R1 corpus = 2,000 urban clips (from 1,926) this week; then camera-fetch ≤32 chunks (~64 GB)
   extracting all gate-passing clips per chunk.
2. WorldModel-Synthetic-Scenarios pose-field verdict (yes/no) within one `huggingface_hub`
   file-listing call — unblocks SC-02/05/06 data-sourcing.
3. Semantic/strategic-label dataset survey delivered with exactly one ranked Phase-1 ingest
   recommendation (curve/maneuver-scarcity remedy for X2).

### Architecture & Inference
**Priority order (from `AI` BACKLOG.md):** P0 #2b decision-grade K-step sweep (=X6) → P0 #2c extend
imagination horizons (=X6, couples) → P1 #3c REF-A/REF-B builds (=N4/N5) → #3d1 REF-A/REF-B on full
realmix (=N4) → #3d2 REF-A grid adapter (**done directionally**, refutes the mean-pool-confound
hypothesis — see the protocol-sensitivity note below) → #3e2 CV kinematic floor (=N1) → #3e0
resolution probe (=X3) → #3 RoPE+AdaLN bake-off (small expected Δ, cheap smoke-test first) → #3b
orthogonality instrument for `spectral.py` → #4 H4 arm-B DINOv3 WM (=REF-A) → #4b E2E BC reference
(=REF-B) → #4c pixel-prediction WM reference (Phase 1) → #5 tactical horizon ablation (8 vs 16).

**Honest note on REF-A numbers (protocol-sensitivity, not a contradiction to paper over):** three
different REF-A ADE@1s figures exist across recent runs — 14.2 m (W31 head-to-head, confounded
mean-pool adapter), 17.01 m / 20.22 m pool/grid (`d1_probe` protocol, `0130` report), 7.60 m
(replay-app live session, same report). This spread is itself Exhibit A for why N2's decode-ladder
must land before any H4 claim is made — protocol choice is currently moving the number more than
the architecture is.

**Production-readiness gap:** the gate runner (D1–D3+spectral) is *validated* — real route-resampled
code (`e9b2491`), CI-integrated. No architecture lever has gone from `planned` to a trained-config
change yet (D-018 discipline: everything escalates first) — that is by design, not a gap, but it
means Next-bucket items (X1, X3, X6) are the first real tests of the bake-off harness's promise.

**GOALS.md seed:**
1. Decision-grade K∈{1,2,4} sweep at operative scale (pod2 ckpt): `imag_rel` per horizon, target no
   regression at the ≤4-step horizon vs the already-adopted K=4 default (D-027).
2. Resolution probe (X3): ADE delta at 128/256/384 px on the `sharp`/far-hazard strata specifically,
   reported against the `straight` stratum as the control.
3. Spectral-sizing decision-grade rerun at the final Stage-0 checkpoint (rank was 35→43 and still
   climbing at 6.5k/30k) — closes D-021's open proposal one way or the other.

### Benchmarks & Eval
**Priority order (from `BE` BACKLOG.md):** P0.1 LAL-v2 integration into `metrics.py` → P0.2 ≥3-seed
SC-01 CARLA re-run (measure OKRI per-seed SD) → P0.3 closure-incursion detector fix (H9, currently
reads 0) → P1.0 AUTOPILOT-VQA probe-transfer (D-028 seam) → P1.3 WP.29 paragraph extraction →
P1.4 scenario-metric wiring dry-run → P1.5 Metis deep-read → P2.6 NAVSIM/Bench2Drive entry audit
(=X5) → P2.7 per-scenario excellence leaderboard section. The driving-diagnostic harness (N1–N3) and
the nuScenes bench build (N8) are this discipline's actual top-of-week items, ahead of the numbered
backlog, per D-029's "orbit the driving gap" mandate.

**Production-readiness gap:** the metric suite (LAL/TMS/OKRI/CNCE/LOPS) is *validated* (22
analytic-ground-truth tests, verified live against the gate runner). The gap is entirely on the
**scenario side**: SC-01 is still single-seed/scripted-policy, and the closure-incursion detector is
a known-broken instrument sitting unfixed since it was flagged.

**GOALS.md seed:**
1. ≥3-seed SC-01 re-run: OKRI/LOPS/TMS as mean±CI; target non-overlapping CIs vs the reactive
   baseline (falsifier: CIs overlap → no "beats baseline" claim, per the pre-registered rule).
2. nuScenes-mini L2@{1,2,3}s harness live, first number posted to `LEADERBOARD.md` WITHOUT the
   ego-status shortcut, within the Next window.
3. Closure-incursion detector reads nonzero on the reactive run (currently a hard 0 — instrument
   defect, not a policy result).

### Opponent Analyzer
**Priority order (from `OA` BACKLOG.md):** P0.1 SC-13 stationary-object/same-lead authoring (Avride,
cheapest high-value item, reuses comma2k19 + LAL-v2/OKRI) → P0.2 SC-14 red-light spec (near-free off
SC-04) → P0.3 W-04 degraded-visibility matched-pairs weather re-test (D8, falsifier: still ~0.5 AUROC
at 30k → escalate to the H15 σ-head) → P1 scenario-DB expansion sweep (standing duty) → P1 Ghost
Cut-Through/Blind Creep specs → P1 watch-list deep-reads (Autobrains→L4, Metis param-count watch,
AlpaSim, SkyJEPA) → P2 W-06 unit-economics dossier → P2 per-opponent counter-scenario coverage matrix.

**Production-readiness gap:** of 14 cataloged scenarios (SC-01…SC-14), only SC-01 has reached
`live-measured` and none have reached `excellence-proven` (the lifecycle's terminal, public-claim
stage per `SCENARIO_DATABASE.md`'s own definition). SC-04 is `spec-drafted` with offline oracle tests;
most others sit at `catalogued`. The gap is entirely pipeline-stage advancement, not scenario
discovery (discovery is keeping pace with opponent recalls week over week).

**GOALS.md seed:**
1. SC-13 → `spec-drafted` this week (Avride/W-08 competence gap, cheapest high-value item per the
   backlog's own ranking).
2. D8 matched-pairs weather-axis AUROC > 0.6 by the 30k gate (currently ~0.5 unpaired at 6.5k);
   falsifier already pre-registered in the backlog item.
3. Advance ≥2 scenarios one full lifecycle stage (e.g. `spec-drafted`→`oracle-tested`) — currently
   SC-01 is the only entry past `oracle-tested`.

### Production & Optimization
**Priority order (from `PO` BACKLOG.md):** P0.1–3 (latency baseline, ONNX export, compliance review
#1) are all **DONE**. P1.3b ZipDepth safety-envelope eval (explicitly "SECOND-STEP, not Phase 0" per
Sayed) → P1.4a TRT toolchain install (blocked: `tensorrt` not importable on the dev box) → P1.4b
clean-GPU latency (**DONE** — fp32 15.76 ms/63.5 Hz, fp16 13.40 ms/74.6 Hz, both clock-pinned) →
P1.5 compliance review #2 (**DONE**, fail-fast intake shipped) → P1.6 INT8/FP8 quantization curves
(**in flight, uncommitted** — `Implementation/int8_quant/`: encoder/predictor quantize safely,
agreement 0.984/0.953; **heads do NOT**, agreement 0.484, 33/64 decision flips, 1.67 m mean
waypoint shift) → P1.7 `tactical_pred` fail-fast + `imagination_nll` logvar clamp (**in flight,
uncommitted** — `Implementation/incoming/2026-07-10-contract-windowing-failloud/` ships the sibling
fix: fail-loud `EpisodeWindowDataset` windowing) → P1.8 compliance review #3 (scripts/training loop,
same intake) → P2.9–11 DSSAD/ISMR scaffold, memory-envelope profile, dependency hygiene.

**Production-readiness gap — and a live instance of it:** the INT8 curve and the windowing fail-loud
fix are both **real, measured, and sitting uncommitted** in `Implementation/` right now (confirmed via
`git status` at time of writing) — exactly the intake-hygiene debt the orchestrator has flagged three
weeks running (see §6). The fix here is procedural (commit the intake), not technical.

**GOALS.md seed:**
1. Ship the INT8 selective-quantization recipe as an intake: quantize encoder+predictor (safe,
   ≥0.95 agreement) but explicitly NOT the heads (0.484 agreement, 33 flips/64) — a decision, not
   just a measurement.
2. TRT-fp16 engine built once the toolchain lands (idle-pod or dev-box install), verified against
   the pre-registered bar (≥95% agreement, ≤~4 cm wp-shift).
3. Commit the fail-loud windowing fix + `tactical_pred`/`imagination_nll` numerics package this
   week — it has been measured and written, only the commit is missing.

---

## 4. Dependency graph

**Chain 1 — the diagnostic routes everything (N1–N3 → X1–X2):**
`DIAG §A (N1)` decides whether D1=6.44 m means *broken* or *harsh-metric-relative* → reframes how
every future D1 number is read. `DIAG §B (N2)` — **"routes the entire rest of the program"**: readout
verdict ⟶ nonlinear trajectory head / probe calibration (X1a); representation verdict ⟶
architecture / training-signal / resolution / model-size (X1b, X3). `DIAG §C (N3)` — curve-specific
failure ⟶ data-mix + curve-oversampling (X2); uniform failure ⟶ representation fix (X1b). `DIAG §D
(N3)` — still-descending loss at 27k ⟶ undertrained (more data/steps); flat ⟶ capacity-bound.

**Chain 2 — resolution gates the entire Phase-1 encoder budget (X3 → L1):**
X3's resolution ablation is the ONLY thing that unlocks L1 (multi-cam/triplane/temporal efficient
encoding) — stated explicitly in the source: *"[§H] Not a cause of the current failure... Gated on
§G: only relevant once resolution is shown to matter"* (`DIAG` §1.H). If X3 comes back flat, L1 is
descoped, not merely deprioritized.

**Chain 3 — the reference arms feed H1/H4 and the paper (N4/N5 → L5):**
N4 (REF-A full-mix, de-confounded) + N5 (REF-B first launch) → matched-protocol comparison at
main@30k → H4/H1 evidence → *"the three-way comparison IS the H1/H4 evidence section"* (`REFARCH`
§4) → L5 (paper §7). Nothing in L5 can start before both N4 and N5 produce matched-protocol numbers.

**Chain 4 — closed-loop is the real arbiter and gates ALL of Later (X4 → L1–L4):**
D2 ranking already passes (0.864) but is unproven as *driving* — X4 (closed-loop route-completion)
is the translation step `DIAG` §3.2 names as the actual Phase-0 exit test, and that section is
explicit: **"Only then do more cameras / sensors / the H-stack proceed."** X4 itself is gated on the
CARLA graphics-pod recipe only if camera-driven (not scripted) ego becomes critical-path — currently
"not urgent" (`T&D` P1 #3) because nullrhi physics-only is sufficient for the scripted-policy
scenario suite.

**Chain 5 — data reproducibility de-risks everything, blocks nothing (N9, parallel):**
`DATA_MANIFEST.json` + `rebuild_cache.py` (proven on pod2+pod3) remove the pod1 single-point-of-
failure and feed the paper's reproducibility statement (`DE` P0 #-1) — foundational, not on the
critical path, but its ABSENCE would have blocked N4 (REF-A full-mix needed a second self-sufficient
data node, which N9 is what provided it).

**Chain 6 — TanitScena is bounded by Opponent Analyzer's authoring pace (OA → N7):**
N7's semantic search is only as useful as `SCENARIO_DATABASE.md`'s content — SC-13/SC-14 authoring
(Opponent Analyzer P0.1/P0.2) directly grows what N7 has to index; a stale DB makes a perfect search
index pointless.

---

## 5. Production-readiness scorecard

| Deliverable | Grade | Evidence | Single next step |
|---|---|---|---|
| **Train pipeline** | Validated | 27k/30k (90%) real-recipe steps on pod1; mmap-OOM root-caused+fixed once (`7b5faa6`, data_s 73%→nominal); OOM-guard auto-restarted 20× one night with 0 crash-loss, but throughput now craters at the 62 GB cgroup cap (near-zero steps/h at 90%) | `PO` P1 #8 ops-fragility review #3 — LRU-aware cgroup eviction instead of crash-avoidance-only |
| **Gate runner** (D1–D3+spectral) | Validated | Route-resampled D1 (mean±CI, 8 splits) is real code (`e9b2491`); integrated with the D-017 rework; runs in CI | Fold the diagnostic's baseline/decode-ladder verdict into the runner so D1's threshold can be redefined baseline-relative if N1 calls for it |
| **Replay / TanitResim** | Validated, advancing | Real app (`stack/tanitad/resim/`); scenario cards + side-by-side arm panels + single-port serving shipped (`e3d9757`); dual-sink `.rrd` bug documented, single-port sidesteps its proxy variant | Fix the remaining dual-sink empty-file case; ship the 3-arm (main/REF-A/REF-B) view once N5 produces a checkpoint |
| **TanitScena** | Prototype, pre-code | D-029 mandate + agent-file scaffolding only (`0ab6060`); `stack/tanitad/scena/` does not exist; the `tanitscena` branch has touched only docs | Build the markdown→structured-store parser over the 14-entry `SCENARIO_DATABASE.md` as the MVP slice |
| **Data recipe** (realmix + manifest) | Validated (proven 2×) | `rebuild_cache.py` self-provisioned the full corpus from origin on BOTH pod2 and pod3 without touching pod1 (program report `0130`); HF `Sayood/tanitad-realmix` v1 (402-ep) in progress, PRIVATE | Verify rebuilt-cache sampled-episode SHA256 matches pod1 byte-for-byte; then Sayed's explicit public flip |
| **CARLA harness** | Validated, narrow | Live-physics SC-01 measured (OKRI 12.83 vs 32.37, `-nullrhi`); camera-rendering blocker root-caused (RunPod compute-only driver caps + UE4.24 Vulkan-offscreen bug); single-seed/scripted-policy only | ≥3-seed re-run emitting LAL-v2 (`BE` P0 #2); graphics-pod recreation only when camera-driven ego is critical-path |
| **Eval suite** (LAL/TMS/OKRI/CNCE/LOPS) | Validated | 22 analytic-ground-truth tests; verified live against the gate runner; LAL-v2 superseded LAL-v1 (blind to smooth anticipation) | Integrate the LAL-v2 intake into `metrics.py` (`BE` P0 #1); fix the closure-incursion detector (hard 0, `BE` P0 #3) |

---

## 6. Cross-cutting risks (honest)

| Risk | Evidence (real, dated) | Mitigation in flight |
|---|---|---|
| **OOM / pod memory cap** | mmap dataloader's unbounded working set root-caused once (`7b5faa6`); the SAME class of pressure now stalls the 30k run at 90% inside the 61–62 GB cgroup (near-zero steps/h, 20 auto-restarts one night — program report `0130`) | OOM-guard prevents silent kill but not the throughput collapse; the real fix (`PO` P1 #8, LRU-aware eviction) is not yet built |
| **Data fragility** | Throttled rsync stalled the 2026-07-11 ~21:30 record run (`DE` P0 #-1 trigger incident); HF free-tier commit quota killed a cache upload; comma2k19 re-download silently stalled at 44 GB **twice** (`0130` report) | `DATA_MANIFEST.json`+`rebuild_cache.py` now proven on two independent pods without touching pod1 — the fragile path (direct copy off the training pod) has a working alternative |
| **Open-loop ⊥ closed-loop** | Standing `LEADERBOARD.md` footnote (arXiv 2605.00066, gate G-B1): ADE/FDE has no reliable correlation with closed-loop Driving Score; NAVSIM PDMS ranks non-monotonically vs Bench2Drive DS (paired n=8) | `DIAG` §3 makes closed-loop route-completion a **hard** Phase-0 exit criterion, not an optional extra; D2's ranking pass (0.864) is the unproven-closed-loop hope X4 exists to test |
| **Agent-health regression** | `PROJECT_STATE.md` W29 "only 3/6 agents committed"; W30 "Data-Eng no-show" + "0/5 clean worktree branches"; W31 "branches still empty 3rd cycle" (de-escalated, 0 work lost — loop carried it). **Live right now:** `Implementation/int8_quant/` and `Implementation/incoming/2026-07-10-contract-windowing-failloud/` sit **untracked in `git status` at the time of writing**, despite W31 reporting hygiene debt "CLEARED" — real, measured work (§3 Prod&Opt) is not yet committed | Loop/orchestrator has reliably carried real work onto `main` even when discipline branches stay empty; the session-end commit guardrail flagged in W29/W30/W31 is still unbuilt — this roadmap's own commit is one more manual instance of the same gap |
| **License firewall** | D-012: PhysicalAI-AV is internal-dev-only/confidential, excluded from public claims; D-022: WorldModel-Synthetic-Scenarios (OpenMDW-1.1) held at conservative default (HOLD) pending an **explicit** Sayed/legal accept — not auto-adopted by closing the numbering gap | The realmix HF dataset (N9) is designed by construction to ship comma-by-reference + SHA256 + rebuild instructions — never raw PhysicalAI bytes — so the firewall survives even the reproducibility push |

---

## 7. Top-3 dependency-critical items on the driving-gap critical path

1. **N2 — Diagnostic §B (decode ladder).** Explicitly "routes the entire rest of the program" —
   every Next-bucket remedy (X1, and half of X2/X3) is a downstream branch of this one experiment's
   verdict.
2. **N1 — Diagnostic §A (baseline-relative read).** Decides whether D1=6.44 m is a real driving
   failure or a metric-protocol artifact — changes how every gate number in `LEADERBOARD.md` gets
   read from here forward, including REF-A/REF-B's own numbers.
3. **X4 — Closed-loop route-completion.** `DIAG` §3 names it the actual Phase-0 exit arbiter and
   states outright that more cameras/sensors/the H-stack (all of Later, L1–L4) do not proceed until
   it passes.
