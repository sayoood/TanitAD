# TanitAD — PROJECT STATE

> **This is the single entry point for every new working session (human or agent).**
> Read this file first. It tells you where the project stands, what is decided, and what to do next.
> Update it at the end of every working session (see `Project Steering/CONTINUATION_PROTOCOL.md`).

- **Last update:** 2026-07-08 evening (autonomous loop: D-020 extensions + cosmos chunk fix + D8 paired preview + I8 latency baseline + scenario-suite wiring; training step ~7.6k/30k)
- **Current phase:** Phase 0 — foundation & first edge proofs
- **Constitution:** `Project Steering/Mission Plan.md` (owned by Sayed, never edited by agents)
- **Final evaluation date (P7):** 2026-10-05

---

## 1. Where we are (one paragraph)

Project kicked off 2026-07-05. Prior-experiment repos (ALPS-4B, 4B-HRM, ACRE, RSRA-4B) analyzed; initial
deep research across H0–H15 done and documented; master plan and Phase 0 plan written; the `stack/`
Python package scaffolded with the 4B world-model skeleton, instrument checks (I1–I4), and a smoke test
that runs on the RTX 4060; research hub concept written and weekly agents defined; repo pushed to
https://github.com/sayoood/TanitAD.

## 2. Key documents (read in this order)

| Priority | Document | Purpose |
|---|---|---|
| 1 | `Project Steering/Mission Plan.md` | Constitution: vision, goals, hypotheses H0–H15, principles P1–P8 |
| 2 | `PROJECT_STATE.md` (this file) | Current truth |
| 3 | `DECISIONS.md` | Decision log (ADR style) — what is decided and why |
| 4 | `Project Steering/Master Plan.md` | Refined execution plan, phases 0/1/2, gates |
| 5 | `Project Steering/Phase 0 Plan.md` | Detailed current-phase plan, week by week |
| 6 | `TanitAD Research Hub/INITIAL_RESEARCH_SYNTHESIS.md` | Research baseline across all hypotheses |
| 7 | `Project Steering/CONTINUATION_PROTOCOL.md` | How to resume/continue work across sessions |
| 8 | `stack/README.md` | The implementation: how to run, test, train |

## 3. Current focus (Phase 0)

The active week-by-week schedule lives in `Project Steering/Phase 0 Plan.md` §6.
Summary of the immediate next actions:

- [x] W1: smoke training runs on RTX 4060 (`stack/experiments/p0-s000-kickoff-smoke`: loss 3.34→1.89,
      I2 pass 3.6e-7, I4=6.17 — untrained baseline, correctly blocking predictive claims)
- [x] W1: MetaDrive wrapper (WP2) — contract-identical adapter merged + CI-green (`stack/tanitad/data/metadrive_env.py`);
      live rollout needs a supervised source-install of MetaDrive (PyPI pkg no-go on py3.13; see Tools&DevEnv STATE)
- [x] W1: I1–I4 instrument checks implemented + in test suite (10/10 tests pass)
- [x] W1: D-008 executed — TanitAD-4B-M at **261.1 M params** instantiated (budget enforced by test);
      H15 ImaginationField (advection + refine + epistemic σ) wired into training; D9 gate defined;
      exact Phase-0 data spec in Phase 0 Plan §2.2; 24 tests green + 1 sim-skip
- [x] W1: D-009 executed — real camera data first: comma2k19 loader (`stack/tanitad/data/comma2k19.py`,
      HF mirror, route-level splits, real CAN actions), `base250cam` config, Windows-safe extractor;
      Chunk_1 (8.7 GB) downloading locally; 36 tests green
- [x] W1: comma2k19 loader (D-009) **real-data-validated** (DataEng agent): `av` decodes real HEVC →
      [200,3,256,256]@~105 fps, A8 real≈0.053; contract reconciled into shared `_contract` (channels=1|6|None);
      comma2k19 exported + `test_comma2k19_contract.py`; data card + H7 note; 40 tests green
- [x] W1: p0-sB00 real-camera pipe-proof DONE on 4060 — pipeline validated (I2 9.5e-7 pinned; bf16 +
      horizon-subset encoding fixed the 22 GB OOM; Chunk_1 = 188 segments local). Learning at batch 2
      invalid by design: **F-2 SigReg starvation** (n=32/step → erank 23/2048; I4=98.9 caught it);
      live collapse-health rows (erank/dim_std/step_ratio) now in every training log.
- [ ] W1: **Sayed**: start the A40 pod per `stack/RUNPOD_RUNBOOK.md` — p0-sB01 at batch 64 (SigReg
      n=1024) is the first run whose learning signal counts (planned $25)
- [x] W2: **Data strategy v1.0** (`DataEng/DATA_STRATEGY.md`, D-012): PhysicalAI-AV = first RICH
      corpus (access verified, clip_index enables urban filtering; staged R0 500 clips → R1 2 000 →
      R2 multi-view; usage tagged, license resolved at Phase-0 exit); comma2k19 = license-clean
      public anchor; composition ~60/25/15; research synthesis updated to v1.1 (theory grounding)
- [ ] W2: PhysicalAI-AV Stage R0 ingestion (DataEng agent top duty Tuesday; MVP assists)
- [x] W2: D-010 **sim arm unblocked** (Tools&DevEnv): MetaDrive front-camera RGB path renders 6ch/256
      2-frame stacks (comma2k19-identical) + perturbation policy + occluder/blocked-route scenarios —
      intake pkg `Tools&DevEnv/Implementation/incoming/2026-07-13-metadrive-frontcam-perturbation/`
      (17 tests, 0 new deps), pending orchestrator triage. The merged 1ch BEV adapter is (correctly)
      rejected by `MixedWindowDataset` — proven by test.
- [x] W2: **D-014 — MetaDrive retired** (Sayed: too old / Python pinning). Sim arm split: training
      mix ← NVIDIA synthetic corpora (WorldModel-Synthetic-Scenarios: emergency/lanechange/nudging/
      pedestrian/weather + Cosmos-Drive-Dreams CC-BY-4.0, both ungated); closed loop (D5/D6, G0.5)
      ← CARLA-in-Docker on RunPod, W31–32 (Tools&DevEnv top backlog). Training never blocked on sim.
- [x] W2: **Cosmos-Drive-Dreams loader** (DataEng, D-014 sim arm): CC-BY-4.0 → first *publicly-claimable*
      RICH AV corpus (the license review excluded real PhysicalAI-AV from public claims). Intake pkg
      `Data Engineering/Implementation/incoming/2026-07-14-cosmos-drive-dreams-loader/` — derives
      steer/accel from 4×4 `vehicle_pose` geometry, D-015 9-ch, `CORPUS_META`≡comma2k19 (D-017 I7 →
      admissible in D-010 mix), CLIP-split (I3); 9 tests, stack 73✓/1s. `DATASET_LANDSCAPE.md` created
      (D-012 duty). Pending orchestrator triage.
- [x] W2 (autonomous loop iter 1–2): **gate runner + spectral-sizing packages INTEGRATED** with the
      D-017 rework (P4 path, imag-rel→diagnostic incl. A13 test, I7 rows); 97 tests green.
      **Spectral DIAGNOSTIC on the live ckpt (step 3000, pod GPU): fit R²=0.997, operator effective
      rank ≈35, energy knee ≈22, k*≈11 → task-relevant dynamics rank is LOW (tens) — NOT a collapse
      signature.** Reframes the step-10k erank question: low live-erank may reflect intrinsic
      highway-dynamics rank; judge the checkpoint by ACTION DISCRIMINATION (D2/P1-P4), not raw rank.
      Decision-grade sizing still awaits the trained ckpt. Local R0 fetch complete: 500/500 urban clips.
- [ ] W2: Stage-0 bake-off (residual+change-weighted vs MSE; grid readout vs pooling; probe_imag vs probe_real)
- [ ] W2–3: D1–D3 gates measured (see Phase 0 Plan §4) — **step-15k preview due ~2026-07-09 midday**
      (ETA at ~300 steps/h). Turnkey commands: pod
      `python scripts/evaluate_checkpoint.py --ckpt /workspace/experiments/p0-sB01-realmix/ckpt.pt
      --cache-dirs /workspace/data/comma2k19/_epcache /workspace/data/physicalai/_epcache
      --out /workspace/experiments/p0-sB01-realmix`; then local D8 paired re-run
      `python stack/scripts/d8_preview.py --ckpt <pulled ckpt> --comma-cache
      C:/Users/Admin/tanitad-data/eval/comma2k19-val-61c46fca8f7f --cosmos-root
      C:/Users/Admin/tanitad-data/eval/extracted --pairs-root C:/Users/Admin/tanitad-data/eval/pairs
      --out <json>`. Watch: D2 must not regress (P1 0.872 / P4 0.940 at 5k); D1 ADE trend; paired
      D8 fraction should rise from 16/23.

## 4. Open questions / blocked items

- ~~NVIDIA PhysicalAI-AV license review~~ **RESOLVED 2026-07-07 (DataEng):** real AV sets = NVIDIA
  internal-dev-only/confidential/12-mo → **excluded from public claims**; comma2k19 (MIT) is the public
  corpus; Cosmos-Drive-Dreams (CC-BY-4.0) is the publicly-safe AV asset. Internal use needs Sayed+legal
  sign-off. See `TanitAD Research Hub/Data Engineering/Research/2026-07-07-physicalai-av-license-review.md`.
- RunPod budget approval per training stage (owner: Sayed; default budget in Master Plan §7)
- **DECISION NEEDED (D-021, proposed 2026-07-08):** latent dim k as a *measured* design variable
  (spectral knee) — keep 2048 for Phase 0, but Phase-1 resizing only with a trained-checkpoint
  spectral result + gate impact. | default: keep 2048, keep measuring (nothing blocked).
- **Sayed task (~10 min, supervised):** MetaDrive source install on the dev machine (PyPI no-go on
  py3.13) — unblocks D-010 sim-mix live rollout, D5/D6 blocked-route and D9 occluder scenarios.
  Command sequence prepared by Tools&DevEnv (their BACKLOG P1.3).

## 5. Session log (newest first, keep last ~15 entries)

| Date | Session | What happened | Artifacts |
|---|---|---|---|
| 2026-07-08 (loop, afternoon) | Autonomous loop iterations (MVP) | Four measured results on top of D-020: (1) **cosmos chunk-pairing bug found+fixed** — 28/60 extracted clips are chunk-1 and were getting chunk-0 actions; temporal semantics settled (121-frame chunks @30 Hz, 10 s/300-pose clips; T=39 exact); verified on real bytes; cosmos cleared for D-010 mix. (2) **SC-05/D8 preview** (step-6500 ckpt, 4060): naive relative imag-error inverted (AUROC 0.34) — confounds identified; **matched-pairs redesign measured: 16/23 same-scene clear→degraded pairs show higher imagination error (p≈0.047)** — first directional positive for H11; pre-registered re-runs at 15k/30k. (3) **I8 batch-1 latency baseline**: decision tick **15.07 ms p50 / 1.08 GB VRAM** fp32 un-optimized (~66 Hz vs 10–20 Hz req.); first real CNCE (median 2.02×10⁵) + TMS expert band → LEADERBOARD efficiency block + paper §7. (4) **Scenario→telemetry→metric wiring PASSED** end-to-end on the work-zone oracle (all 5 discriminative checks). Stack: **149 tests green**. Training healthy all day (single trainer, ~300 steps/h). | `stack/scripts/{d8_preview,cosmos_pairs,latency_cnce_baseline,scenario_suite_dryrun}.py`, `stack/experiments/{p0-d8-preview,p0-latency-baseline,p0-scenario-wiring}/`, LEADERBOARD, paper §7 |
| 2026-07-08 | D-020 program extensions (MVP orchestrator) | All five Sayed proposals executed. (1) Hub **deep-screen digest**: all 6 intake packages verified LIVE; findings routed (K-step/RoPE bake-offs → Architecture backlog, ZOD/synthetic-scenarios → DataEng backlog, D-021 proposed). (2) **`Paper/TANITAD_PAPER.md` v0.1** — living postdoc-level paper (4B architecture + SigReg/Epps–Pulley math, imagine-and-select + calibrated decode, H15 NLL, spectral theory, instrument doctrine, step-5000 results honestly bounded). (3) **Production & Optimization stream**: Master Plan §3 row, Saturday agent (06:44) + `PRODUCTION_READINESS.md` + backlog (latency baseline I8, ONNX parity, compliance review #1 = data/). (4) **Agent depth upgrade**: `_common-protocol.md` — mandatory measured experiment per run (gate G-H), burst-compute section (Colab CLI/idle-pod etiquette/4060), budgets 3→4 iters/2→4 h; per-discipline `BACKLOG.md` seeded ×6. (5) **`SCENARIO_DATABASE.md`**: SC-01…SC-12 from documented opponent failures (FACT/CLAIM/INFER), lifecycle → excellence-proven, joint Opponent↔DataEng↔Benchmarks duty wired into agent files. D-020 recorded; D-021 proposed (§4). | `Paper/TANITAD_PAPER.md`, `TanitAD Research Hub/2026-07-08-screening-digest.md`, `.../agents/production-optimization-agent.md`, `.../Production & Optimization/{BACKLOG,PRODUCTION_READINESS}.md`, `.../Opponent Analyzer/SCENARIO_DATABASE.md`, 6× `BACKLOG.md`, `DECISIONS.md` D-020/D-021 |
| 2026-07-14 | Data Engineering agent | **Cosmos-Drive-Dreams loader** (intake, D-014 sim arm): the license review (D-002) excluded real PhysicalAI-AV from public claims, so shipped a loader for the **CC-BY-4.0** synthetic corpus — the one *publicly-claimable* rich AV asset. No CAN → `poses_to_signals` derives steer (bicycle `κ=yaw_rate/v`, low-speed-clipped)/accel/yaw/v from per-frame 4×4 `vehicle_pose`; D-015 9-ch, D-016 focal (120°=PhysicalAI), **`CORPUS_META`≡comma2k19 (D-017 I7)** → admissible in the D-010 real+sim mix (proven by test); CLIP-level split (I3), per-weather episode_id. **9 pkg tests + stack 73✓/1s**, 0 new deps. Created **`DATASET_LANDSCAPE.md`** (D-012 standing duty, was missing): 3 tiers, license/size/actions/urban-richness/cost-to-batch. Lit: LAWM/Drive-JEPA/HiLAM/CLAW/DeFI H7 surge (external support, no upgrade, P8). P8: pose axis-order pod-verified via `verify_real_clip` before any trained claim. | `.../Data Engineering/Implementation/incoming/2026-07-14-cosmos-drive-dreams-loader/`, `.../Data Engineering/Research/2026-07-14-cosmos-drive-dreams-loader-and-landscape.md`, `.../Data Engineering/Research/DATASET_LANDSCAPE.md` |
| 2026-07-17 | Opponent Analyzer agent (run #1) | First live sweep. Built `WEAKNESS_CATALOG.md` v1 (W-01…W-07) + `OPPONENT_PROFILES.md` v1 (Wayve/Waymo/Pony/Momenta/Autobrains/Alpamayo/Tesla), all FACT/CLAIM/INFER-labeled (G-O1). Headline deltas: Waymo **3,871-vehicle construction-zone recall** (freeway autonomy suspended, 20+-city expansion frozen) → new **Work-Zone Phantom** scenario (W-01→H15/H9/H1, intake pkg, **9/9 offline tests**); NTSB school-zone VRU + school-bus stop-arm probes (W-02/W-03); Tesla open NHTSA **degraded-visibility** case (W-04→H11/H15/H2); Wayve Series D corrected to **$1.2 B/$8.6 B**, GAIA-3=15 B offline WM; NVIDIA **Alpamayo-2=32 B** on-car VLA (W-05 foil, still our supply chain); arXiv **latent-WM/JEPA surge** → "world model" no longer differentiating, moat=hierarchy+efficiency+imagination+self-monitoring. Recs to Thu agent: wire Work-Zone Phantom, add degraded-visibility D8 stressor, competitor param counts→CNCE leaderboard. Ledger H0/H6 evidence row (no upgrade, P8). Loop: 12/25 searches. | `.../Opponent Analyzer/Research/2026-07-17-opponent-sweep-w2.md`, `WEAKNESS_CATALOG.md`, `OPPONENT_PROFILES.md`, `.../Opponent Analyzer/Implementation/incoming/2026-07-17-work-zone-phantom-scenario/` |
| 2026-07-16 | Benchmarks & Eval agent | Custom metric suite shipped (intake, D-011): **LAL/TMS/OKRI/CNCE/LOPS** (Deep Think 14) + trajectory `extra_metrics` seam that plugs into Wednesday's D1–D3 gate runner — **verified live** against `run_d1`; 22 tests all on **analytic ground truth** (G-B2). Research: **open-loop ADE/FDE ⊥ closed-loop DS** (arXiv 2605.00066, ranking inversions) → validates decode-gates-necessary-not-sufficient + the custom-metric thesis; NAVSIM-v2 EPDMS 51.3 navhard (thin on occlusion = our OKRI/LOPS niche); Bench2Drive SOTA ctx (TF++ 86.97/71.97); CARLA ~5 DS seed variance → **≥3-seed mean±CI** rule adopted; WP.29 June-2026 ADS GTR (ISMR/DSSAD sub-asks). LEADERBOARD gains a separate closed-loop block + weak-claim footnote; REGULATION_TRACE enriched. Consumed **D-014** mid-run (MetaDrive retired) → suite is sim-agnostic, scenarios re-target CARLA-on-pod. Note: auto-commit swept my files into `5940129`/`47a89c4`; `hub(bench-eval)` reconcile = `51b432b`. | `.../Benchmarks & Eval/Implementation/incoming/2026-07-16-eval-metric-suite/`, `.../Benchmarks & Eval/Research/2026-07-16-*.md`, `Benchmarks & Eval/LEADERBOARD.md`, `Benchmarks & Eval/REGULATION_TRACE.md` |
| 2026-07-14 | Architecture & Inference agent | Two intake pkgs (D-011): **#0 `p0-spectral-sizing`** — latent-dim sizing from the action-conditioned transition spectrum (L2/2606.27014; knee vs 2048 readout; 8 tests) and **#1 D1–D3 gate runner** — instrument-doctrine gating (BLOCKED≠FAIL; I1–I4 first; vs-pool & probe A3 ablations; extra_metrics seam for Thu; 13 tests). Research: decode ≠ planning (2512.24497 → D1–D3 are necessary-not-sufficient, D4–D6 arbitrate); V-JEPA-2-AC 300 M ≈ our envelope; LeWM supports SIGReg-only (H3); DriveMoE/GEMINUS → route MoE on H15 σ (H2); native-TRT ViT INT8 trap → OwLite/ModelOpt (H5). Consumed mid-session D-012/D-013 + JEPA-theory note. Stack suite 65✓/1s | `.../Architecture & Inference/Implementation/incoming/2026-07-14-{spectral-sizing-p0,gate-runner-d1-d3}/`, `.../Architecture & Inference/Research/2026-07-14-*.md` |
| 2026-07-13 | Tools&DevEnv agent | D-010 sim arm unblocked: MetaDrive front-camera RGB path (6ch/256 2-frame stacks, comma2k19-identical geometry/alignment) + scripted perturbation policy (off-expert coverage) + occluder(H15/D9)/blocked-route(D5/D6) scenario configs; proved the old 1ch BEV adapter is rejected by `MixedWindowDataset`. Intake pkg (17 tests, 0 new deps, import 1.38 s); live rollout still gated on supervised MetaDrive source-install. KB: Alpamayo 2 Super (32B) + OmniDreams = Phase-1 watch, reinforce P5/C2 | `.../Tools&DevEnv/Implementation/incoming/2026-07-13-metadrive-frontcam-perturbation/`, `.../Tools&DevEnv/Research/2026-07-13-*.md` |
| 2026-07-07 | Data Engineering agent | comma2k19 (D-009) real-data-validated: `av` decodes real HEVC @~105 fps, A8 real≈0.053 → change-weighting justified; forked contract reconciled into shared `_contract` (channels param) + `test_comma2k19_contract.py`; comma2k19 exported; Windows `\|` confirmed (handled by `extract_comma2k19.py`); H7 deltas (LAOF, Sensorimotor-WM); data card + note | `.../Data Engineering/Research/2026-07-07-*.md`, `stack/tanitad/data/_contract.py`, `stack/tests/test_comma2k19_contract.py` |
| 2026-07-06 | D-008 scale-up | Model scaled to 261 M (measured per-component budget in Phase 0 Plan §2.1); H15 imagination in Phase 0 (module + losses + D9); exact data spec; RunPod runbook + ledger row; local 261 M pipe-check run | `stack/tanitad/models/imagination.py`, `stack/RUNPOD_RUNBOOK.md`, DECISIONS D-008 |
| 2026-07-06 | Tools&DevEnv agent | WP2 MetaDrive→toy-contract wrapper (17✓/1skip); MetaDrive install verdict (PyPI no-go py3.13, source is GO); AlpaSim/AlpaGym = Phase-1 cloud (40–60 GB VRAM); Rerun.io picked for viz | `stack/tanitad/data/metadrive_env.py`, `TanitAD Research Hub/Tools&DevEnv/Research/2026-07-06-*.md` |
| 2026-07-05 | Kickoff | Repo analysis, initial research, plans, stack scaffold, hub setup, first push | see §2 table |
