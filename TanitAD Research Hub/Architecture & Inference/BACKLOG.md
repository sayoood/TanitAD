# Architecture & Inference — Experiment Backlog

Prioritized roadmap (D-020 §4). Each run: execute ≥1 item, report measured numbers, re-prioritize.
Format per item: goal / method / resource / expected number / falsifier.

## P0 — next run

0a. **[✅ DONE 2026-07-15 — H15 imagination edge liveness (resolved the `h15=0.0` WATCH)]** GPU
   diagnostic on the exact flagship code path: imagination BUILT (22.06 M), grad reaches it (L1 44.6) +
   encoder (L1 36.7), fire rate 0.4525≈`mask_prob`, mean-when-fired 0.611. `h15=0.0` = LOGGING ARTIFACT
   (last-micro sample; 46.3 % of rows false-zero while training; true idle 6.3 %). **Edge healthy, no
   config change (D-018).** Shipped observability fix (intake `2026-07-15-h15-logging-fidelity`, 6✓).
   Artifact: `Implementation/h15_logging_diagnostic/` + `Research/2026-07-15-*.md`.

0b. **Multi-step belief-rollout imagination (UWM-JEPA 2605.25313-motivated) — NEW top item.**
   Goal: does training H15 on a **K-step** masked rollout (advect+refine over k∈{1,2,4}, NLL each step)
   beat the current **1-step-only** imagination, and does the per-cell **epistemic σ persist or dissipate
   over the operative K-step rollout**? Method: prototype the K-step masked-rollout loss in
   `Implementation/` (reuse `ImaginationField` + `sector_mask`); on a held-out ckpt measure (i) σ-retention
   vs horizon, (ii) blind-rollout probe-R² retention (UWM-JEPA's metric) vs a no-imagination baseline.
   Resource: 4060 / idle-pod, hours, $0. Expected: σ persists and R²-retention beats baseline at horizons
   ≤4 → multi-step training earns its place; **Falsifier:** σ collapses toward 0 with horizon OR R² drops
   as fast as no-imagination → 1-step is sufficient, close the item. **Gate:** D9 (calibration) + D8
   (self-monitor AUROC). **D-018:** escalate before any trained-config change. Cross-ref: the σ-dissipation
   risk directly threatens the H11/D8 self-monitor trigger at long horizons.

1. **[✅ DONE 2026-07-08 — spectral-sizing on real trained latents]** Ran `run_spectral.py` on the
   step-6500 `ckpt_full.pt` (24 val eps, 7,176 pairs, 4060): **fit R²=0.99, rank ≈43, knee 31, k*=21 →
   OVER-PROVISIONED** (knee ≪ 512, well inside the predicted 20–60 band → efficiency-moat evidence for H3).
   Feeds D-021. **Remaining:** re-run at the FINAL Stage-0 ckpt (rank still climbing 35→43) before any resize.
   Artifact: `Research/2026-07-08-spectral_step6500.json` + note §5b.
2. **[✅ DONE 2026-07-09 — K-step rollout bake-off, first measured arm]** Matched-compute K=2 vs K=1
   (2×2000 steps, real comma2k19, 4060, 11.74 M reduced-but-real probe, OFAT-verified). Rollout is
   **nearly free (+0.5 % wall-clock, 0 params)**. **Metric lesson:** the planned falsifier (D2 dir-acc)
   **saturated at 1.0** → non-discriminative; the discriminative signal is **`imag_rel`**, on which K=2
   cuts 1-step latent-pred error vs persistence **2.914→1.049 (−64 %)** but does NOT help the 4-step
   horizon → **K must match the decode horizon**. D1 FAIL + D3 BLOCKED at this scale ⇒ no decision-grade
   claim (D-004). Artifact: `Research/2026-07-09-...md` + `Implementation/kstep_bakeoff_probe/`.
   **Superseded by P0 #2b + #2c below.**

2b. **Decision-grade K-step sweep K∈{1,2,4} at OPERATIVE scale** — the real bake-off arm.
   Method: matched-compute trained arms from the pod2 step-8k `ckpt_full.pt` (Phase C, idle-pod or
   Colab L4/A100-with-ledger-row); primary metric **`imag_rel` per horizon** (NOT dir-acc — it
   saturates) + D3 ratio once imagination horizons are extended. Expected: K=4 lowers `imag_rel` at
   horizons ≤4 without D2 regression; falsifier: no `imag_rel` improvement at matched steps → drop
   K-step. **D-018 Tactic → escalate to Sayed before it touches the trained config.**

2c. **Extend imagination horizons past 0.4 s** (predictor imagines k∈{1,2,4}=0.4 s; the plan's D3 is 2 s).
   Couples with 2b (K must cover the horizon). Method: add longer horizons to `predictor.horizons` in a
   prototype; measure `imag_rel`/D3 at 1 s/2 s. Falsifier gate: D3. Escalate before trained-config change.

## P1

3c. **REF-A/REF-B reference builds (Sayed 2026-07-09)** — full build plans in
   `Project Steering/REFERENCE_ARCHITECTURES.md`: REF-A frozen-DINOv3 WM (feature-cache training,
   stability measures specified) behind the K-step arms; REF-B 4B vision-action E2E (no WM,
   budget-matched) post-30k on pod1. Pre-registered decision rules included.
3d. **Multi-cam encoder optimization — experiment #1 (batched multi-cam encode)** — extend
   `latency_cnce_baseline.py` with N in {1,4,7} batched encodes on the 4060; expected ~2-2.5x
   single-cam latency at N=7 (not 7x). Investigation doc:
   `Research/ENCODER_MULTICAM_OPTIMIZATION.md` (attack order + production-readiness table).
3d1. **REF-A/REF-B on the FULL realmix (Sayed directive 2026-07-11 evening):** both reference
   arms train on the SAME data mix as the main run (comma + PhysicalAI R1), not comma-only.
   For REF-A this requires extending stage 1: DINO-precompute the PhysicalAI epcache episodes
   (same `dino_precompute.py`, pod3→pod2 cache copy after the post-30k pod1→pod3 rsync), then
   the flagship grid-REF-A retrains on the mixed feature corpus. The comma-only grid run
   (in flight tonight) is kept as the pool-vs-grid ADAPTER ABLATION row, not the flagship.
   Mix ratio: reproduce the main run's realmix ratio; record any residual loader difference.
3d2. **REF-A stage-2b: GRID adapter (Sayed review 2026-07-11 — "expected better from DINO").**
   The v1 adapter MEAN-POOLS the 256 DINO tokens → destroys the spatial structure the DINO-WM
   lineage relies on (DINO-WM/EponaV2/DINO-world all keep patch tokens). Build the literature-
   faithful variant: tokens → SpatialGridReadout-style adapter → state; retrain 30k feature-level
   (~4.5 h on the A40, features already cached); re-run refa_probe. Expected: closes a large part
   of the 14.2-vs-7.5 ADE gap; falsifier: if grid-REF-A still ≫ main at matched protocol, the
   frozen-encoder deficit is real (H4 evidence, not adapter artifact). ALSO: comma-only main-model
   probe re-run (pod1, post-30k) for the apples-to-apples row.
3e2. **D1 kinematic floor: constant-velocity baseline row (Sayed 2026-07-11 — "7.51 is bad too").**
   Add CV-extrapolation (velocity from pose history, extrapolate 1 s) as a baseline row next to
   D1 in `evaluate_checkpoint.py`. Purpose: split the 7.5 m into (a) irreducible action-uncertainty
   + protocol hardness vs (b) true encoder deficit. If CV lands ~1–2 m on comma highway, the
   linear-probe gap is real work; if CV is also ~5–7 m, D1's threshold needs redefining as
   baseline-relative (gate honesty). Plus unit audit ("camera" unit semantics vs metres).
3e0. **Resolution-sensitivity probe (Sayed question 2026-07-11 night: does 256px cap the latents?).**
   Mechanisms at risk: sub-patch far objects (vehicle@80m ≈ 10-15px < patch 16 → long-range
   anticipation cap: LAL-v2/LOPS/SC-01 at range), far lane geometry (D1@2s contributor), signage
   content (Phase-2 only). Experiment: encode val episodes at 128 / 256 / interpolated-384
   (ViT pos-emb interpolation, no retrain for the directional read), compare D1-probe ADE per
   horizon + D2 dir-acc + a far-hazard LAL slice. Expected: 384 helps mostly at ≥2s horizons and
   far-hazard onset; falsifier: flat deltas → 256 is not the binding constraint, close the
   question. Roadmap position: 256 global → Phase-1 multi-cam 256-320 → H16 native-res ROI
   channel (source frames are 1164×874 = 4.5× linear detail unseen by the encoder) → uniform
   bump ONLY if measured gaps survive. Resource: 4060/idle pod, hours, no retrain.
   Goal: cheapest possible falsification pass on F1 BEFORE any build. Method: replay SC-01 3-seed
   telemetry + the D8 degraded-pairs episodes; compute the H15 per-sector σ trace from an existing
   checkpoint; ask "would a σ-threshold trigger have fired on the critical sector BEFORE the
   reveal/onset?" (pure logs + one forward pass per window, 4060). Expected: trigger fires ahead of
   reveal on ≥70 % of occlusion events at <1 query/10 s false-positive rate in free cruise.
   Falsifier (F1): fires uniformly / misses onsets → the trigger premise dies before we build
   anything. Full dossier + F1–F3: `Research/H16_ACTIVE_DEPTH_INTERROGATION.md`. Cross-refs:
   Prod-Opt backlog 3b (ZipDepth ROI cost curve), H2 scheduler window (~Sep) for the online version.


3. **RoPE + AdaLN conditioning** — one-lever bake-off vs FiLM/learned positional embedding
   (2512.24497 "AdaLN+RoPE best"). Smoke-scale first (d256 on 4060, 1k steps, probe fit — reuse the
   `kstep_bakeoff_probe` harness), promote to Colab arm only if smoke shows ≥ +2% probe fit. Falsifier:
   Δ within noise → close. **Prior lowered (arXiv 2605.08567):** AdaLN vs cross-attn is a wash for
   LOW-dim actions and our actions are 2-D → expect small Δ; keep AdaLN (not cross-attn), test cheap.
3b. **Orthogonality instrument for `spectral.py`** — from arXiv 2605.26379: LeJEPA's optimal-planning
   guarantee needs the identified latent to be linear+**orthogonal**. Add a check that the trained
   readout covariance is ~isotropic/diagonal (an I-row, gates the D-021 sizing claim's admissibility,
   not an architecture change). Ship as an intake with a test. Cheap, makes the theorem falsifiable
   on our own checkpoint.
4. **H4 arm-B: frozen DINOv3 world model — PROMOTED (Sayed ask 2026-07-09), design fixed:**
   (a) precompute DINOv3-**B/16** features once over the comma epcaches (16×16 grid @256px matches
   our readout geometry; pod2 post-arms or Colab T4 — embarrassingly parallel); (b) train
   predictor+readout on frozen features (`--data cached` path, ~110M trainable, low VRAM, fast);
   (c) same D1/D2 probe protocol vs our 15k ckpt (preview) and 30k (decision-grade), same held-out
   routes, I7-checked. B/16 = fair-compute match to our 99.5M encoder; DINOv3-L later as the
   foundation-scale upper bound. Caveat stands (2512.24497: DINO > V-JEPA for planning readouts —
   the arm most likely to beat us; be honest). Owner: loop or Wed agent, whoever reaches it first;
   queued behind the K-step arms on pod2.
4b. **E2E behavior-cloning reference arm (Sayed ask 2026-07-09 — the D4 opponent):** same encoder
   trunk + direct action-regression head (camera→steer/accel), trained on the same comma caches at
   matched steps via `--data cached`. Purpose: gate D4's learned baseline — does imagine-and-select
   beat direct regression at matched data/compute? Closed-loop comparison in CARLA once the
   camera path exists; open-loop preview = action-prediction error + D2-style ranking. Cheap
   (~100M trainable, reuses everything).
4c. **Pixel-prediction WM reference (Phase 1, small-scale, budget line):** generative-decoder
   variant to make "latent beats pixels on sample efficiency" (H3) measured, not cited. Explicitly
   NOT Phase 0.
5. **Tactical horizon ablation** — measure D2 at horizons {8, 16} separately (gate runner already
   emits per-horizon rows); decide if the 16-horizon head earns its 26.5M params.

## P2 / theory watch

6. **σ-gated tactical MoE** (route on ImaginationField.logvar) — WP4/Phase-1; needs epistemic
   interface; design note first.
7. **SIGReg-vs-spectral-contrastive theory gap** (2606.27014 constants don't transfer to SIGReg) —
   watch Balestriero/Klindt/PKU-Yisen-Wang lineages for a bridging result; escalate if a paper
   directly bounds SIGReg-trained planning regret.

## Done / retired
- (2026-07-08) Gate runner D1–D3 + spectral module shipped via intake; integrated with D-017 rework.
