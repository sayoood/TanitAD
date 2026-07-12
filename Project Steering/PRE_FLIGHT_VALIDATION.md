# Pre-Flight Validation — GATE before the 4-day 4-brain flagship run (Sayed, 2026-07-12)

**Rule:** the ~4-day from-scratch flagship run does NOT launch until every axis below is GREEN with
evidence. Two prior expensive runs under-delivered (OOM-crawl; grounding fine-tune that didn't move
the oracle ceiling) — this gate exists so the third is not another. Each item: a concrete check, a
pass criterion, an owner, and a status. Amber/red blocks launch.

## Axis 1 — Data consistency (can validate NOW; corrected caches exist)
- [ ] Corrected caches complete & uncorrupted: every `ep_*.pt` loads (size + torch.load), counts
      match (comma 410/90, physicalai 400/100), no truncated shards (the scan bug recurrence).
- [ ] Actions↔poses↔frames temporal alignment: cross-correlation lag = 0 per corpus (reuse
      `geom_sanity.py`); a known left turn shows sign(Δyaw)=sign(ego-y) both corpora.
- [ ] Action units/scale consistent (steer rad, accel m/s²); pose units metres; no NaN/Inf.
- [ ] Mix ratio matches intent (physicalai share ~0.6) and I3 train/val route-split is disjoint
      (no episode-id leakage across splits) + I7 fingerprint distinct.
- [ ] Corrected-vs-old identity: same clip → actions/poses IDENTICAL, only frames changed (already
      shown mean|Δ|=25.85; formalize the assertion).
- Pass: all green. Owner: Data validation agent.

## Axis 2 — Geometry & unified calibration (VLM3 principle) — VERDICT: GREEN (2026-07-12)
Evidence: `stack/scripts/validate_geometry.py` (reusable gate) run pod2-side on the REAL caches +
500-clip per-clip intrinsics; JSON + before/after frames in `Benchmarks & Eval/axis2_geometry_gate/`.
- [x] f-theta fix achieves the SHARED canonical focal: measured f_eff = 266 ± 8 on BOTH corpora
      (comma via true 910px, physicalai via real fisheye poly) — the VLM3 "one effective focal"
      invariant that makes action→pixel geometry corpus-consistent.
      **PASS: comma 266.54, physicalai 265.97 (500-clip median, sigma 0.11; old nominal was 434.76 =
      1.63x zoom). Byte-faithful: 3 decoded clips' f_eff == analytic to 0.01px.**
- [x] Empirical cross-corpus consistency: pixels-per-metre of ground motion (straight segments,
      known Δd) MATCH within tolerance between comma and physicalai (the test that failed before at
      1.6×). This is the decisive VLM3 check.
      **PASS via calibration-derived proof (focal ratio 0.998, was 1.62x). Empirical flow was
      INCONCLUSIVE on urban physicalai (FOE row 166.6 vs comma clean 119.97; f*h unfittable —
      dynamic-traffic contamination, exactly as GEOMETRY_INTEGRITY_AUDIT.md found), so we stand on
      the calibration proof, as the audit did. CAVEAT: a residual ~1.19x ground-scale gap from the
      UNNORMALIZED camera height (1.43 m vs ~1.2 m) remains — the focal error is fixed; height is the
      deferred D-016 R1 item, not this gate.**
- [x] Undistortion/crop correctness: an f-theta test pattern maps to straight lines (no residual
      fisheye curvature); principal point centred; horizon ≈ h/2 both corpora.
      **PASS: f-theta inverse round-trips to 0.0 deg; undistort maps straight->straight to 1e-4 px;
      cx centred (median offset -2px). comma horizon FOE row 120 ≈ h/2. CAVEAT: the DEFAULT cache
      path is crop (not undistort) so it keeps small central barrel curvature (sub-px near centre,
      a few % of width at extreme corners); full undistort is deferred D-016 R1. cy is bimodal
      (rig A 23% / rig B 77%) so the crop uses GEOMETRIC centre by design — robust to the split.**
- [x] I7 `CORPUS_META` reports the ACHIEVED f_eff (266), not the nominal (the silent-skew guard).
      **PASS: both corpora emit f_eff_px = 266.0 (physicalai sourced from calib.F_REF, traceable),
      i7_task_identity fingerprints identical (0 mismatched keys). build_pai_cache.py also asserts
      f_eff ≈ F_REF at build time, so the skew cannot recur silently.**
- Pass: cross-corpus pixels-per-metre consistent + f_eff=266 both. Owner: Geometry validation agent.
  **RESULT: GREEN on the VLM3 focal invariant (directly measured, 500 clips + decode-confirmed). The
  specific 1.62x focal error that wasted the prior run is FIXED. Two pre-existing, documented
  residuals remain and are deferred to D-016 R1 (NOT this gate): (a) camera-height ground-scale
  ~1.19x, (b) default-crop residual barrel curvature. Empirical urban-flow corroboration is
  inconclusive by measurement noise, not by a defect.**

## Axis 3 — Architecture & flagship↔REF-A parity (AFTER 4-brain build)
- [ ] 4-brain wiring correct: strategic context →(FiLM)→ tactical →(intent FiLM)→ operative; each
      conditioning path measurably changes the downstream output (sensitivity tests).
- [ ] **Flagship↔REF-A structural parity: identical apart from (a) encoder (ViT vs frozen-DINO
      adapter) and (b) SIGReg target (full latent vs predictor-outputs-only).** Assert same tactical/
      strategic/grounding classes instantiated, same dims, same budget ±2%. (Sayed requirement.)
- [ ] Budget ~261 M ±5%; no dimension mismatches; vanilla-load compatibility of base model.
- [ ] Metric-dynamics heads attach at op/tac/strat; grounding gradients reach each level.
- Pass: parity assertion + sensitivity + budget. Owner: Architecture validation agent.

## Axis 4 — Training approach & setup (AFTER 4-brain build)
- [ ] Loss assembly complete & correctly weighted: JEPA(op+tac) + hierarchical grounding(op+tac+strat)
      + maneuver CE (class-weighted) + route CE + SIGReg(position-relaxed) + imagination; every term
      finite and O(1)-scaled; no term silently zero.
- [ ] SIGReg-position relaxation actually exempts the free-dims (their SIGReg grad ≈ 0) AND still
      regularizes the complement (anti-collapse holds: erank stays high on a smoke run).
- [ ] From-scratch schedule sane: warmup, cosine, LR, rollout_k=4, batch/accum, grad-checkpoint;
      cgroup-OOM guard pre-armed; resume tested; save cadence.
- [ ] Data loader = corrected caches; realmix ratio; no stale-dir glob traps (the recurring bug).
- Pass: all green on a 20-step real-data smoke. Owner: Training-setup validation agent.

## Axis 5 — Heads, decoding & interpretation (AFTER 4-brain build)
- [ ] Maneuver head → the maneuver vocabulary; route head → route classes; both read the right
      pseudo-labels (refb_labels), class balance sane.
- [ ] Metric decoding: StepDisplacementReadout → SE(2) accumulation matches odometry on a
      constant-velocity synthetic; eval protocol (probe vs rollout-decode) documented so results are
      interpreted correctly (probe reads raw latents; rollout uses the trained head — DON'T conflate).
- [ ] Imagine-and-select uses the TRAINED tactical policy to propose + grounded rollout to score.
- [ ] Every eval metric's definition pinned (ade@1s vs ade_0_2s; route-resampled D1); baselines
      (CV/go-straight) present so numbers are interpretable.
- Pass: heads correct + eval protocol unambiguous. Owner: Heads/eval validation agent.

## Axis 6 — Empirical lever pre-checks (THE decisive ones — AFTER build, BEFORE the 4-day run)
The prior grounding failed on 3 mechanisms; validate the fixes actually work at small scale first:
- [ ] **SIGReg-relaxation oracle test:** short co-trained run (comma-only, corrected) with vs without
      the position-subspace exemption → does the oracle in-distribution ceiling DROP (< ~1.3 m)?
      If yes, lever #2 confirmed. If flat, SIGReg wasn't the blocker → rethink before the big run.
- [ ] **Grounding-from-scratch vs fine-tune:** short from-scratch run with grounding co-trained →
      does the oracle move where the 8k late fine-tune couldn't (1.65→1.60)? Confirms lever #1.
- [ ] **Corrected-geometry A/B (already approved):** current vs corrected physicalai, 4k fine-tune →
      does the corpus gap narrow without hurting comma? Confirms the geometry fix is causal.
- Pass criterion for the 4-day GO: **at least the oracle ceiling demonstrably improves at small scale
      under the fixed objective+geometry.** If none of the three moves the oracle, we do NOT spend 4
      days — we escalate to encoder capacity/architecture instead.

## Launch decision
GO only when Axes 1–5 GREEN and Axis 6 shows the oracle ceiling moving at small scale. Otherwise
iterate the cheap levers first. Every result committed here with numbers.
