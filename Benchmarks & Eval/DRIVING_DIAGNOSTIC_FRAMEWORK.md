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

## Results

### 2026-07-12 driving_diagnostic (27k, 8 route splits, 2240 windows) — DECISIVE
Instruments trustworthy: **I1 oracle-probe fit R²=0.91 (PASS)**, **I2 batch-consistency 3.5e-7 (PASS)**.

**Decode ladder — ADE@1s (m):**
| | ridge α1 | ridge α10 | MLP |
|---|---|---|---|
| **held-out** (route split) | 5.10 | 5.32 | **3.89** |
| **oracle ceiling** (in-dist fit=eval) | 1.87 | 2.19 | **1.65** |
| **constant-velocity baseline** | — | — | **0.28** |

**Error localization — model vs CV, ADE@1s (m):**
| stratum | model | CV | model/CV |
|---|---|---|---|
| **straight** (n=323) | **2.75** | **0.18** | **15×** |
| gentle (n=75) | 4.52 | 0.41 | 11× |
| sharp (n=38) | 2.84 | 0.79 | 3.6× |
| comma/highway (n=238) | 2.21 | 0.20 | 11× |
| physicalai/urban (n=198) | 4.08 | 0.36 | 11× |
| high-speed (n=204) | 2.26 | 0.19 | 12× |

**Proof-based verdict (what the numbers DO support):**
1. **The model is ~10–15× worse than constant-velocity EVERYWHERE, including straight highway
   (2.75 vs 0.18 m).** It fails the *easiest* case. This is NOT a "handles straights, fails curves"
   capability gap — it is a **fundamental failure to encode/decode ego-trajectory** even trivially.
   Sayed's read confirmed with proof.
2. **The failure is BOTH representation AND readout, quantified:**
   - **Readout/generalization gap** = held-out MLP 3.89 vs oracle-ceiling MLP 1.65 = **2.4× overfit**
     across routes. Fixable with a nonlinear trajectory head + route-diverse decode training.
     MLP < ridge at both levels → **D1's ridge gate UNDERSTATES decodability** (3.89 not 6.44).
   - **Representation floor** = even the ORACLE in-distribution ceiling is **1.65 m**, ~9× worse than
     CV on straights. **Even with perfect decode, the visual latent does not carry metric
     ego-displacement at driving precision.** This is the deeper, primary problem.
3. **HONEST CAVEAT (do not overclaim):** CV uses ground-truth pose history (privileged ego-state the
   vision-only probe never sees). The fair question is the oracle ceiling — "from vision alone,
   best in-distribution, how well can trajectory be recovered?" = **1.65 m@1s** — not driving-grade
   (sub-metre needed) but not catastrophic. **D2 ranking PASSES (0.864)** → the latent carries
   selection-relevant info even though absolute regression is poor.

**Root cause — evidence-backed hypothesis (was a guess, now supported):** the model was trained to
predict + RANK latent futures (JEPA/SigReg + imagination — D2 passes), but has **no explicit metric
ego-trajectory training target** beyond the small inverse-dynamics head. Nothing forces the latent to
linearly encode metric ego-displacement → oracle ceiling stuck at 1.65 m. **#1 lever: add explicit
ego-motion / trajectory-prediction supervision** (or strongly upweight inverse-dynamics) so the
representation is ego-grounded; the 1.65 m ceiling is the target to break. **#2 lever: nonlinear
route-generalizing decode head** (closes 3.89→1.65). Resolution/model-size/data-mix are SECONDARY
until the objective encodes ego-motion (straight-road failure rules resolution out as primary).
Artifact: `stack/scripts/driving_diagnostic.py`, `/workspace/experiments/driving_diagnostic.json`.

- **Step-curve (8.5k/14k/27k ADE):** queued — tests whether held-out ADE still descends at 27k
  (readout undertraining) vs is representation-bound. Cheap; existing checkpoints.
- **NEXT experiments this routes:** (1) explicit-ego-supervision training arm (fine-tune 27k with a
  trajectory-prediction head + upweighted inv-dyn, measure oracle-ceiling shift); (2) route-diverse
  MLP decode head; (3) resolution ablation (secondary); (4) REF-A/REF-B same diagnostic when trained.

### 2026-07-15 baseline_floor (Benchmarks & Eval) — §A denominator corrected + speed-gate fix
Data-only, $0, local CPU; 26 132 anchors (comma2k19-val 25 110 + Cosmos-DD 1 022), 10 Hz.
Metric shipped tested (8 analytic tests): `Implementation/incoming/2026-07-15-baseline-floor/`.
The §0/§A single-CV floor (CV ADE@1s ≈ 0.28 m) is **not** the honest denominator — CV is the
weakest kinematic null on curves. Best-of-3 (CV / go-straight / **CTRV**), per-stratum, median ADE@1s:

| stratum (comma-hwy, v=25) | n | **best floor** | CV | CTRV | CV/CTRV |
|---|---:|---:|---:|---:|---:|
| straight | 18 785 | **0.056** | 0.088 | 0.062 | 1.4× |
| gentle | 3 008 | **0.059** | 0.275 | 0.060 | **4.6×** |
| sharp (genuine, speed-gated) | 212 | **0.164** | 0.404 | 0.167 | 2.4× |

- **Denominator correction:** the honest floor is **~0.056 m@1s** (CTRV-dominated, wins 55–58 %),
  not 0.28 m. → model held-out 6.44 m = **~115× the floor** (not 10–15×); oracle-ceiling 1.65 m =
  **~29× floor**. Verdict *direction unchanged and reinforced*; **use `skill_score` = model_ADE ÷
  per-stratum best-of-3 floor** in the D1 gate, not a single CV scalar.
- **Protocol fix (adopt before the next `driving_diagnostic` run):** speed-gate the curvature strata
  (v ≥ 2 m/s). 12.4 % of comma anchors are near-standstill (median 0.01 m/s); ungated, GNSS yaw-jitter
  mislabels them "sharp" (κ = yaw_rate/v singular at v→0) with a spurious 0.003 m floor. §C strata are
  otherwise standstill-polluted.
- **§D2 update:** the ungated Cosmos-DD sample is **not** a curve/maneuver source (0 % genuine sharp,
  95.8 % straight); comma-highway carries more real curve content. The curve-scarcity remedy needs the
  semantic-label survey (nuPlan/DriveLM/CoVLA), not more Cosmos-DD.
- Caveat (P8): baselines use privileged GT ego-state → this is the denominator, not a model competitor.
  Full note: `Research/2026-07-15-baseline-floor-honest-denominator.md`.
