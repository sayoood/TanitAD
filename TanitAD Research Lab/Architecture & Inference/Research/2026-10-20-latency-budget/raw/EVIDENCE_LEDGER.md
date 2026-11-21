# EVIDENCE LEDGER for LATENCY_BUDGET_PROPOSAL.md (2026-10-20; tip `abd4dd819211a2d5382ce4926e38c9fa25201cd2`)

Classes: MEASURED (ours, artifact) / PUBLISHED (banked primary, library key + location) / INHERITED (our doc, not re-verified) / ESTIMATED (arithmetic in `budget_arithmetic.py`) / UNVERIFIED.
Tier: latency numbers carry no driving tier (they are not a driving claim); the only driving number here (+0.56 PDMS) is a selection-tier read.
Repo paths are relative to the repo root; `RA` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`.

## A. Repo

| id | claim | class | artifact / location |
|---|---|---|---|
| A1 | PI design intent: the three brains run at different rates, "e.g. 100 ms vs 500 ms vs. 1 second (subject of investigation)"; targets Orin and Thor; efficiency is a moat hypothesis; "we should not use all of the compute they are offering" | INHERITED (PI text) | `Project Steering/Mission Plan.md` l.44, l.97 |
| A2 | Working budget 100 ms (10 Hz planning) used by the old flagship line | INHERITED | `TanitAD Research Lab/Architecture & Inference/Implementation/incoming/2026-08-02-thor-deployment-profile/THOR_PROFILE.md` (header); `Project Steering/PROGRAM_OVERVIEW.md` 5.0.2; registry text "12 % of the 100 ms budget" (REF-C plan tick 11.5 ms) |
| A3 | MPC planner design replans at 10 Hz; its E3 "Thor latency budget: candidates x batch schedule vs the 10 Hz replan deadline" was never run | INHERITED (design) | `.../Implementation/incoming/2026-08-06-mpc-planner-design/MPC_WM_DESIGN.md` s2, s6 |
| A4 | Old flagship WorldModel (263.58 M, 176x624, 1 cam), Thor, bf16 encoder + CUDA-graph predictor, K 20: tick 98.63 ms p50 / 102.57 p99; encoder bf16 28.82 / 31.08; fp32 tick 272.56 | MEASURED (Thor) | `.../2026-08-02-thor-deployment-profile/thor_combined_tick.json` |
| A5 | Flagship step-29999, full tick (encoder + strategic + tactical heads + 9-candidate fan + decode + scoring), K 20: 60.296 p50 / 63.076 p95 / 65.277 p99 ms; budget_ms 100 | MEASURED (Thor) | `TanitAD Research Lab/Deployment & Optimization/Implementation/incoming/2026-08-03-thor-batch9-engine/thor_d6_tick_intent_K20.json` (arm C); scope: latency + tactical decision, not four-family accuracy |
| A6 | v7-tiny Thor tick fits t(B) = 7.45 + 5.05 B ms; CUDA graph removes 70 % of the fixed term; TF32 under graph 22 % | MEASURED (Thor) | `products/P6-TanitDeploy/2026-08-23-v7tiny-baseline-profile/raw/thor_v7tiny_plan_and_cudagraph.json`; `.../2026-08-23-thor-lever-ladder/raw/thor_graph_tf32_paired_AB.json` (H-DEPLOY-3/7) |
| A7 | Thor bf16 4096^3 GEMM 1.0856 ms (= 126.6 TFLOPS); fp8 0.5510 ms; the speedup is a function of GEMM size | INHERITED (the table is in `THOR_PROFILE.md` addendum 4; no JSON for it found in that package) | `THOR_PROFILE.md` addendum 4 |
| A8 | No Thor latency of REFe/P2 (trunk, proposals, scorer, search) at three probes: (i) C2 amendment s7 (registry, goals, REFE plan docs, at `c3a5c329`); (ii) `git grep -iE` of (refe\|dinov3\|vit-l) near (thor\|jetson\|orin) near (ms\|latency) over `*.md *.json *.py` at the tip: only the NAVSIM-scoring backend and the instrument's own files; (iii) `REFE_PLAN.md` lists "REFe's measured latency on Thor" as a to-do | MEASURED by search (absence); UNVERIFIED for files outside the repo | `RA/raw/2026-10-19-search-p2-confirm/c2/AMENDMENT_C2.md` s7; `RA/REFE_PLAN.md` |
| A9 | Dev-box CPU search timing (fp32, 4 threads, n 5, 1 repeat, 1 warm-up): pick-only p50 61.06 (scorer 60.99, host 0.08; 1 call, 64 rows); C8 870.29 (scorer 823.75, host 55.51; 14 calls, 896 rows); O4 2,703.30 (scorer 2,479.76, host 214.53; 38 calls, 3,392 rows); O4/C8 3.11x; `is_thor_number` false | MEASURED (dev-box CPU) | `RA/raw/2026-10-20-search-thor-latency/raw/devbox5/THOR_SEARCH_LATENCY_devbox-cpu_none_20261020T023743Z.json` (read from the tip; copy at `thorlat/raw/devbox5`) |
| A10 | The instrument: 117 tests, 50/50 guard mutations caught; excludes trunk and proposal generator by construction; refuses on a busy Thor | MEASURED | `RA/raw/2026-10-20-search-thor-latency/{RESULT,HANDOFF,RUN_ON_THOR}.md` |
| A11 | Search structure: K 5 dependent population calls (each ONE complete M-set); then the base fan call and one `score_replaced` call PER elite (8 for C8, 32 for O4) issued one by one; dependent depth if the final stage were batched = 6. Calls: C8 5+1+8 = 14, O4 5+1+32 = 38 | MEASURED by reading source | `TanitAD Research Lab/Architecture & Inference/Research/2026-10-18-ep-rl-lit/l1_stepA/tts_cem.py` (sha256 `9dea96038d18ae0b...`, identical to the `l1run` copy I read) |
| A12 | C2 amendment: UNMEASURED with the s7 protocol; PASS worded "a selection-tier gain on navtest; deployability UNMEASURED"; no pass/fail on latency; an absolute budget is an open architecture decision | RULED | `RA/raw/2026-10-19-search-p2-confirm/c2/MM_RULING_C2AMEND.md` DQ3, DQ4 (sha in `.sha256`) |
| A13 | SPEC_P6 has no inference-latency text. Its latency lines are the training labeller (one micro-batch period, about 0.9 s at micro-batch 4; MEASURED 57.0 s / 64) | MEASURED (training) | `RA/SPEC_P6_PAPER_FAITHFUL.md` 3.2, 3.3 |
| A14 | REFe geometry: 4 cameras at 960x512, ViT-L (width 1024, depth 24, patch 16), 16 registers per camera, 64 proposals, 20 steps at 5 Hz; DINOv3 cls + 4 registers inside the trunk | INHERITED (config defaults); P2's own run config not found (`summary.json` carries no geometry) | `RA/refe/model.py` l.9, 68-93; `RA/REFE_MODEL.md` |
| A15 | A40 training step: 0.2227 s per sample-pass, bf16 + TF32 + compile, batch 4 x accum 64 (57.0 s/step / 256) | MEASURED (training) | `RA/SPEC_P6_PAPER_FAITHFUL.md` 0.1 (source `refe-final-2026-10-01/summary.json`, `metrics.jsonl`) |
| A16 | C1 read O1_CONFIRMED, +0.56 PDMS for the C8 search over P2's pick | INHERITED (MM brief, 2026-10-20 ~08:45Z; the tip's register still reads C1 as stopped at its GPU stage) | `Project Steering/GOALS_AND_CLAIMS.md` `H-SEARCH-P2-1`; selection tier, SEEN-SCENE |
| A17 | P6 is training on the A40 (pod runs `bcc04b5`); state not re-checked | INHERITED | `Project Steering/BACKLOG.md` `P6-RESUME-BANK`; `par18/common.md` |

## B. Literature (all primaries banked; `kb_add.py --verify` clean; page/table located in the extracted text)

| id | library key | claim | class |
|---|---|---|---|
| B1 | 2511.00088 (Alpamayo-R1) | Tab.14, RTX 6000 Pro Blackwell: vision 3.43 + prefill 16.54 + reasoning 70 (40 tokens) + trajectory 8.75 (5 steps) = 99 ms; trajectory-only baseline 29 ms; auto-regressive trajectory variant 312 ms; the text calls the real-time requirement typically 100 ms | PUBLISHED |
| B2 | 2411.15139 (DiffusionDrive) | 45 FPS on an NVIDIA 4090 (abstract); Tab.2: TransfuserDP with 20 DDIM steps 7 FPS (6.5 ms x 20 = 130 ms plan module), the 2-step truncated variant 45 FPS (3.8 ms x 2) | PUBLISHED |
| B3 | 2503.12820 (Hydra-MDP++) | Tab.1 latency: ResNet34 206.2 ms, V2-99 271.0 ms, TransFuser 221.2 ms (V100); UniAD 555.6 ms (A100) | PUBLISHED |
| B4 | 2506.06659 (DriveSuprim) | Tab.14: ViT-L 12.5 FPS (345 M), Hydra-MDP++ ViT-L 14.4 FPS (337 M); V2-99 22.1 / 24.0 FPS. Hardware not stated in the text I extracted (training on 8 A100). The same architecture reads 271 ms in B3 and about 42 ms here: the two tables are not comparable | PUBLISHED; hardware UNVERIFIED |
| B5 | 2601.05083 (DrivoR) | single-element batch, A100, no quantisation or acceleration: GTRS ViT-L baseline 400 ms/forward, DrivoR 110 ms/forward | PUBLISHED |
| B6 | 2303.12077 (VAD) | Tab.1: VAD-Tiny 59.5 ms (16.8 FPS), VAD-Base 224.3 ms (4.5 FPS), RTX 3090; UniAD 555.6 ms (A100) | PUBLISHED |
| B7 | 2504.01941 (WoTE) | Tab.6, NVIDIA L20: 64 / 128 / 256 trajectories 17.2 / 17.9 / 18.7 ms | PUBLISHED |
| B8 | 2609.06055 (DriveZero) | the extracted text contains no inference latency, FPS or ms for the camera student (grep: latency, FPS, "ms", real-time, onboard); data rate 5 Hz | PUBLISHED (absence in one document) |
| B9 | 1807.08048 (Apollo EM planner) | introduction: in an emergency the system could react within 100 ms against 300 ms for a normal human driver; results section: on a Nuvo-6108GC-GTX1080 IPC the planner takes less than 100 ms on average | PUBLISHED |
| B10 | 2510.11448 (Autoware.Universe, SIM) | Tab.V, production-ready L4 vehicle, actuation excluded: perception-to-decision mean 521.905 / p95 788.794 / p99 1036.798 ms; planning stage mean 119.242 / p95 198.372 / p99 266.568 / max 392.435; LiDAR 10 Hz; localisation max 10 Hz; 100 ms of delay adds 1.79 m at 17.9 m/s | PUBLISHED |
| B11 | 2305.07147 (COLA) | AV systems expected to react in under about 150 ms vs humans 700-1500 ms (this figure cites its ref. [45]); planning module average latency at the 100 ms scale; Apollo fuses at 10 Hz to match the LiDAR | PUBLISHED; the <150 ms is SECONDARY |
| B12 | 2601.07393 (SW-HW co-optimisation) | CARLA/Bench2Drive closed loop, UniAD: driving score +20.33 % from 1 to 20 FPS, plateau near 20-24 FPS; VAD with measured frame drops -24.62 % vs its baseline 42.35. Table 2's per-FPS rows extracted garbled and are not used | PUBLISHED (closed loop, not NAVSIM) |
| B13 | 2104.07830 (Pylot) | planner P99 runtimes vary 30 vs 550 ms (FOT), 15 vs 76 (RRT*), 25 vs 760 (Hybrid A*) with collision-avoidance outcomes depending on them; detection AP50 halves at 35 ms runtime at 20 m/s | PUBLISHED; banked, cited as background only |
| B14 | 2609.34085 (AD-E2E-JEPA) | 0.8 s per scene for an 8-frame world-model rollout over 256 candidates on an A100: world-model candidate evaluation sits far above 100 ms | PUBLISHED; not used in the memo's table |

New library entries written by `kb_add.py`: 1807.08048, 2104.07830, 2510.11448 (see `library_entries_added.json`).

## C. Estimates (budget_arithmetic.py)

| id | quantity | value | inputs |
|---|---|---|---|
| C1 | trunk FLOPs | 1.527 TFLOP per camera (1,925 tokens: linear 1.163 + attention 0.364), 6.11 TFLOP for 4 cameras | A14; formula 2 x 12 d^2 L x N + 4 N^2 d L |
| C2 | trunk time, Thor bf16, by utilisation of the A7 GEMM rate | 48.2 / 80.4 / 120.6 ms at 100 / 60 / 40 % | A7, C1 |
| C3 | trunk cross-check from the A40 step | 74.2-111.4 ms forward-only (step / 3 to step / 2: frozen trunk, LoRA, backward between 1x and 2x forward) | A15 |
| C4 | search ms on Thor by dependent depth and per-call cost (scorer call + host numpy: 2.0 graph+ported, 5.5 graph+unported, 4.5 eager+ported, 8.0 eager+unported) | C8 (14): 28 / 77 / 63 / 112; O4 (38): 76 / 209 / 171 / 304; batched (6): 12 / 33 / 27 / 48 | A9 (host 3.97 and 5.65 ms per call), A11; the scorer-call knobs and the ARM factor are UNVERIFIED |
| C5 | stage fit | see `budget_arithmetic.out.txt` `which_per_call_regimes_fit_the_stage` | C4 and the share tables |
| C6 | full cycle, pick-only / +C8 / +O4 (trunk 48-121 ms, pick 2-7 ms, search ranges above with a x3 ARM factor on the host) | 50-128 / 78-352 / 126-736 ms | C2, C4 |

Limits: no utilisation, per-call or ARM figure above has a Thor measurement behind it; they are the unknowns M1 and `SEARCH-THOR-LATENCY` exist to replace. The rows effect on the scorer (more rows per call) is assumed small (B7 supports it for a different model); it is not measured for this scorer on a GPU.
