# Driving-Capability Diagnostic Framework (Sayed-directed, 2026-07-12)

**Mandate (Sayed, verbatim intent):** the model is NOT driving in highway (6.44 m ADE@1s, overlay
confirms). Phase 0 is finished ONLY when single-camera + history yields *solid* driving. Derive the
improvement/failure sources across **data, data-mix, data-size, architecture, model-size, decoding,
planning, resolution, efficient-encoding** — from **proofs and experiments, not guesses**. This file
is the pre-registered plan: every hypothesis has a falsifiable test, a pre-registered prediction, and
a cost. No causal claim is adopted until its experiment returns a number. Results append to §Results.

## 0. What we ALREADY know (real numbers, not guesses)
- **D1 held-out ADE@1s = 6.44 ± 0.55 m** (27k, 8 route splits, stable — the earlier 5.18→11.52 swing
  was split-luck, proven by the probe-capacity discriminator `d1_probe_capacity.py`).
- **D3 oracle in-distribution decode ADE@2s = 1.52 m** vs imagined 1.97 m. The oracle (perfect-latent,
  in-distribution) decode being ~4× better than the held-out D1 probe is a **preliminary signal toward
  a readout/generalization bottleneck** — NOT yet a conclusion; the diagnostic's decode-ladder at
  matched horizon/split confirms or kills it.
- **D2 PASS (dir-acc 0.864):** within-window imagination RANKING works even though absolute waypoint
  regression fails. Ranking-good + regression-bad is itself a clue (points at readout/calibration, not
  a blind encoder).
- **Frozen DINO is worse** (REF-A pool 17.0 / grid 20.2 ADE@1s comma-val) — from-scratch encoder
  carries more ego-geometry than open-web pretraining at this scale.
- **I1 oracle-probe fit R² ≈ 0.98** — the harness is not broken.

## 1. The causal taxonomy — each an isolated, falsifiable experiment

### A. Is 6.44 m even bad? (baseline-relative interpretation) — **FIRST, decisive**
- **Test:** constant-velocity / go-straight / constant-yaw-rate baselines, same val, same splits
  (`driving_diagnostic.py` §1). **Pre-registered:** on comma highway (mostly straight, ~30 m/s), CV
  travels ~30 m in 1 s; if CV ADE@1s ≈ 1–2 m and model = 6.44 m → **model is genuinely broken**
  (worse than trivial). If CV ≈ 5–7 m too → the metric/protocol is harsh and 6.44 is less alarming.
  **Falsifier for "model broken":** model beats all three baselines at every horizon.

### B. Representation vs Readout (where does the info die?)
- **Test:** decode ladder — ridge(α), MLP, model's own head, and the **oracle in-distribution
  ceiling** (fit=eval) vs held-out (§2). **Pre-registered:** if oracle-ceiling ADE@1s is small
  (~1–2 m) but held-out is 6.44 → **READOUT/generalization** problem (the latent HAS the info; the
  linear probe fails to generalize across routes) → remedies: nonlinear/learned trajectory head,
  more decode capacity, route-diverse probe training, calibration. If oracle-ceiling is ALSO large
  (~5 m+) → **REPRESENTATION** problem (the info isn't linearly present) → remedies: architecture,
  training signal, resolution, model size. **This experiment routes the entire rest of the program.**

### C. Error localization (can it drive at all, or only fail on curves?)
- **Test:** stratify ADE by future-path curvature (straight/gentle/sharp), ego-speed, corpus (§3).
  **Pre-registered:** if `straight` ADE ≈ 1–2 m and `sharp` ADE ≫ → the model DOES track straight
  driving and fails maneuvers (a *capability gap*, not total failure) → data (curve/maneuver scarcity
  in comma highway) + planning. If `straight` ADE is also ~6 m → the model cannot even hold a lane
  (a *fundamental* failure) → representation/training. Comma-vs-pai split tells us if highway
  specifically is the problem or it's uniform.

### D. Data — mix, size, distribution
- **D1 mix ratio:** the 0.6-physicalai realmix may under-weight highway. **Test (post-diagnostic):**
  train comma-only vs pai-only vs mixes, compare straight-highway ADE. **Prediction:** comma-only
  improves highway ADE if mix dilution is the cause.
- **D2 curve/maneuver scarcity:** comma2k19 is highway-dominated (nav-command derivation showed
  ~all `follow`). **Test:** measure the curvature-bucket POPULATION (from §3 stratum counts) — if
  `sharp` is <5% of windows, the model is starved of the exact events it fails. **Remedy path:**
  the semantic/behavior-label dataset survey (nuPlan/DriveLM/CoVLA — Data-Eng backlog 2d) + curve
  over-sampling.
- **D3 data size:** 810 episodes / ~27k steps is small. **Test:** ADE vs training-step curve from
  the 8.5k/14k/27k checkpoints (we have all three) — is ADE still descending at 27k? **Prediction:**
  if the slope is steep at 27k → undertrained (more steps/data helps); if flat → capacity/architecture
  bound. **This is cheap — run on existing checkpoints.**

### E. Model architecture & size
- **Encoder capacity:** run the diagnostic on REF-A (frozen DINO, 86M frozen + adapter) and REF-B
  (deeper 25-block encoder) once trained — three-encoder comparison at matched readout. **Already
  underway** (REF-A done, REF-B training). If a bigger/different encoder closes the straight-ADE gap
  → architecture/size; if all three fail identically → the bottleneck is downstream (readout/data).
- **Model size:** 261M. **Test (Phase-1):** a 2× predictor width ablation IF §B says representation.
  Deferred until §B routes here — no point scaling if it's a readout bug.

### F. Decoding & planning
- **Trajectory head vs linear probe:** §B's "model's own head" arm. If a trained trajectory head
  >> linear probe → the linear-probe gate (D1) understates the model; the DECODER is the lever.
- **Imagine-and-select planning:** D2 ranking works (0.864) — so a *planner* that selects among
  imagined rollouts may drive far better than the raw regression suggests. **Test (Phase-1):**
  closed-loop MetaDrive with imagine-and-select vs open-loop ADE — the open-loop⊥closed-loop
  footnote (2605.00066) means D1 may be pessimistic about actual driving.

### G. Resolution (Sayed's topic)
- **Test:** encode val at 128/256/384(pos-emb-interpolated), re-probe ADE per curvature stratum
  (Arch backlog 3e0). **Pre-registered:** if 384 helps `sharp`/far-hazard strata but not `straight`
  → resolution caps maneuver/long-range acuity specifically. If flat → 256 is not the binding
  constraint. No retrain needed for the directional read.

### H. Efficient encoding (Sayed's topic — Alpamayo triplane/temporal)
- Not a *cause* of the current failure (single-cam) but the Phase-1 scaling path. **Gated on §G:**
  only relevant once resolution is shown to matter; then triplane-fixed-budget lets us raise
  resolution without token blow-up (`ENCODER_MULTICAM_OPTIMIZATION.md` addendum).

## 2. Benchmarking — community-accepted + ours (leaderboard placement)
Honest placement even with bad numbers. Unit discipline: our D1 is **camera-frame**; community L2 is
**metric BEV** — not directly comparable until we run their protocol.
- **nuScenes open-loop planning L2 @1/2/3s + collision rate** — THE community standard (UniAD/VAD/LAW).
  **Build:** nuScenes-mini loader (we have the OOD probe path) + a metric trajectory head + their L2
  protocol. **Caveat (must report):** AD-MLP (2305.10430) showed ego-status alone scores ~0.4 m — we
  report WITHOUT ego-status shortcut, and attach the open-loop⊥closed-loop footnote.
- **NAVSIM v2 EPDMS (navhard)** — the current standard; Phase-1 target (already contextualized in
  LEADERBOARD).
- **Bench2Drive / MetaDrive closed-loop** — the real arbiter; Phase-1.
- **Ours:** D1–D9, LAL-v1/v2, OKRI/LOPS/TMS, CNCE, SC-01..14, D8 OOD.
- **Leaderboard action NOW:** the honest 27k gate row is posted (LEADERBOARD flagship block) with
  full caveats; the nuScenes row is a Phase-0.5 build (below).

## 3. Revised Phase-0 exit criteria (Sayed: solid single-cam driving before scaling)
Phase 0 is NOT done at "gates measured." It is done when, on single-camera + history:
1. **Open-loop:** model beats constant-velocity AND go-straight at ADE@{1,2}s on BOTH straight and
   curve strata (not just aggregate) — i.e. it demonstrably tracks the road, curves included.
2. **Closed-loop (the real test, per 2605.00066):** MetaDrive/CARLA route completion above a defined
   floor with imagine-and-select planning — proves the D2 ranking translates to driving.
3. **Decodability:** held-out ADE within a defined factor of the oracle ceiling (readout generalizes).
Only then do more cameras / sensors / the H-stack proceed. **Proposal to formalize as a constitution
addendum** (Project Steering/Proposals) — flagged for Sayed.

## Results (append as experiments land)
- **2026-07-12 driving_diagnostic (27k):** _running on pod1 — baselines, decode ladder, error
  localization. Numbers appended here on completion._
- **Step-curve (8.5k/14k/27k ADE):** _queued — cheap, existing checkpoints._
