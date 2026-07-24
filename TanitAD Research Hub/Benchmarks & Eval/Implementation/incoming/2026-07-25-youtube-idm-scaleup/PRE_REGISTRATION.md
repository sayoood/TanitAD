# P4 PRE-REGISTRATION — YouTube-IDM non-CC SCALE-UP (committed BEFORE the downstream read)

**Written 2026-07-25, before `run_youtube_pilot_downstream.py` is run on the scaled corpus.**
Both outcomes are committed in advance (operating-standard rule 5). Evidence class of every
number below: the pilot references are **MEASURED** (`…/2026-07-24-youtube-idm-pilot/pod_artifacts/`);
the scale-up numbers are **PENDING**.

## The question
The 80-clip CC pilot WON *directionally*: FLOOR speed_r2 **−0.520 ± 0.201** → PSEUDO_YT **+0.563 ± 0.047**
(3 seeds, clip-cluster bootstrap 95% CI of the gap excludes 0 every seed), **≈0.92 of the real-label
parity ceiling**, ADE 12.8→6.3 m. Its named caveats were: **80 clips (vs ~300 for decision-grade),
3 seeds, fixed-HFOV geometry.** Non-CC licensing removes the CC-scarcity ceiling. **Does the win HOLD
at decision-grade scale (500–1000 clips, ≥4 seeds) — and does the CI tighten?**

## Identical protocol (parity firewall — unchanged from pilot & `run_idm_parity_validation.py`)
- Substrate: frozen flagship-v1 encoder (state_dim 2048, ckpt md5 `b5f07d9e…585`).
- Labeler L: v1 + IDMHead trained on parity real labels {rigA[:60]+rigB[:60]+comma[:40]} — identical recipe.
- Pretrain corpus D: **the scaled non-CC YouTube pseudo-labeled latents** (the only change vs pilot).
- Downstream: physicalai-VAL, both rigs, finetune 15 / test 65 — the SAME split parity-validation uses.
  Creates NO WM parity arm; does NOT re-select the canonical WM episodes. The FLOOR-vs-PSEUDO_YT gap is
  **within-split** (both arms finetune on the same 15 val clips, test on the same 65), so the known
  val↔parity-train leak — which only threatens a WM arm *trained on parity-train and tested on val* —
  does **not** confound this pretraining-benefit gap.
- Metric: parity-val **test speed_r2** (primary) + traj ADE@2s + yaw_r2 (caveat). ≥4 seeds. Per-seed
  clip-cluster bootstrap 95% CI of the (PSEUDO_YT − FLOOR) speed_r2 gap.
- Fraction-of-ceiling = (PSEUDO_YT − FLOOR)/(CEILING − FLOOR). CEILING = parity real-label pretrain
  = **0.6507**, common FLOOR = **−0.4387** (cited from `results_idm_parity_validation.json`, same split);
  computed **in-run** as a secondary confirmation if the 300 parity-pretrain latents encode in time.

## Committed decision rule (pre-registered)

**① HOLDS — DECISION-GRADE WIN** (the scale-up confirms the pilot) requires ALL of:
  - (a) PSEUDO_YT beats FLOOR on speed_r2 for **every** seed (≥4/4), AND
  - (b) the per-seed clip-cluster bootstrap **95% CI of the gap excludes 0 for every seed**, AND
  - (c) **fraction-of-ceiling ≥ 0.80** (holds the pilot's substantive ≈0.92 "near real-label ceiling"
        claim — this, not the raw gap, is the non-trivial quantity; a broken negative FLOOR is rescued
        by *any* competent pretraining, so the raw gap alone is not the bar), AND
  - (d) **the CI tightens vs the pilot**: across-seed std(PSEUDO_YT speed_r2) **≤ 0.047** (the pilot's)
        AND the pooled bootstrap gap-CI half-width ≤ the pilot's — i.e. scale did not add noise.
  → **VERDICT GO, decision-grade:** the non-CC YouTube domain transfers at scale; the full
    multi-thousand-hour harvest/commitment is justified beyond directional.

**② PARTIAL / BOUND** (win survives but "scale improves it" does not): (a)+(b) hold, but
   fraction-of-ceiling **< 0.80** OR the CI does **not** tighten (d fails). → the directional GO stands,
   but name the cause — **non-CC domain heterogeneity** (mixed channels/resolutions/mounts widen the
   distribution), **label noise at scale**, or **fixed-HFOV geometry** (re-run with GeoCalib). Report
   which, and whether GeoCalib is the indicated fix.

**③ FAIL / REVERSAL** (the pilot did not survive rigor): (a) fails (some seed does not beat FLOOR)
   OR (b) fails (any seed's gap CI includes 0). → the 80-clip directional win was optimistic
   (small-n / favorable CC sub-domain); **the full harvest is NOT justified** on this evidence.
   Escalate with the measured reversal.

## Standing caveats (carried from the pilot; do not overclaim)
1. speed + longitudinal-trajectory are the trustworthy channels (MEASURED zero-shot cross-domain R² 0.60–0.66);
   yaw's downstream lift rides on the 15-clip real finetune, not zero-shot yaw quality.
2. Geometry is fixed-HFOV unless GeoCalib lands; recorded per-pointer, so any BOUND-on-geometry
   verdict is re-runnable with per-video intrinsics without re-harvesting.
3. Non-CC broadens the domain (more channels/mounts/resolutions) — this is expected to *help* transfer
   breadth but *could* widen label noise; outcome ② is the pre-committed home for that.
