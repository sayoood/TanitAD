# Operative flagship-speed @19k: blind-rollout σ-dissipation REPRODUCES; readout isotropy is converging (E1+E2, decision-quality)

**Agent:** Architecture & Inference (Wednesday weekly). **Date:** 2026-07-18 (real wall-clock).
**Backlog item:** P0.1 (fleet directive 2026-07-17) — *"E1+E2 on the operative flagship@30k → drops the
pre-reset caveat → decision-grade."*
**Resource (G-I):** eval pod `tanitad-eval` (A40 48 GB, idle 0 % on entry); 2× blind-rollout seeds
(36.4 s + 34.2 s) + 1 orthogonality pass (~40 s), ~2 min GPU + setup; **cost $0** (standing pod).
Why not bigger: this is exactly the model-scale eval the mandate reserves for the pod; the 4060 cannot
hold the 263 M flagship + PhysicalAI val comfortably and the ckpt lives on the pod. LOCK.arch-inf held.
**Readiness:** *validated* diagnostic on the operative model (2 seeds, stable); becomes *decision-grade*
on the identical re-run at flagship @30k (ETA ~Jul-19–23) — the only remaining gap is the last 11k steps.

---

## 0. TL;DR

The flagship crossed to the **operative post-reset speed recipe** (`flagship-speed`, step **19000**, the
first CV-beater — 0.628 ade@2s @19k per TanitEval). I re-ran my two 2026-07-17 instruments on it, on the
pod's canonical held-out PhysicalAI val, **dropping the pre-reset step-6500 base250cam caveat**:

1. **E1 — blind K-step belief rollout: the σ-dissipation + attractor collapse REPRODUCES, and slightly
   sharpens.** Backlog P0.1's falsifier ("it does NOT reproduce → the speed+jerk recipe fixed it") is
   **NOT met**: the operative flagship's 1-step imagination field still dissipates its epistemic σ and
   collapses to a common attractor under autoregressive rollout. **Freeze-1 (parallel-horizon) again holds
   flat and beats persistence 7×.** → the *cap the operative H15 self-monitor at 1-step / parallel-horizon*
   constraint is confirmed on the model we will actually ship.
2. **E2 — readout orthogonality is CONVERGING toward admissibility, as predicted.** `iso_ratio_active`
   rose **0.254 → 0.546** (crossed the 0.5 line), `cond_number_active` **218 → 61** — SIGReg is doing its
   job as the run matures. Still **NOT-YET-ADMISSIBLE** (off-diagonal coordinate correlation 0.32 > 0.1),
   so the LeJEPA (2605.26379) optimal-planning guarantee is not yet licensed — but the trajectory is
   favorable and the 07-17 prediction ("expect iso to rise with SIGReg training") is verified.
   `active_k ≈ 19`, `cov_effective_rank ≈ 30` ≪ 2048 → **readout capacity is not the D1 bottleneck (G1).**

Both results are on the operative model. Neither triggers a config change (D-018 restraint); both carry a
turnkey re-run for the @30k confirmation.

---

## 1. E1 — blind K-step belief rollout on the operative flagship (P0.1 falsifier)

**Setup.** `WorldModel(flagship4b_config, action_dim=3)` = the exact `taniteval.loaders` build for
`flagship-speed`; strict state-dict load of `/root/models/flagship-speed/ckpt.pt` (step 19000). 40 val
episodes × 8 windows = **320 windows**, K=8 horizons, stride 6 frames. Sector-mask the current frame,
encode the masked belief, roll the ImaginationField forward **fully blind** (no re-observation), compare
per-horizon hidden-cell tokens against the true future encodes. Metrics on **centered** tokens (the raw
ViT DC component saturates cosine ~1 for everything, incl. chance). Script:
`Implementation/belief_rollout_diagnostic/blind_rollout_flagship.py`; results
`results/2026-07-18-blind_rollout-flagship-speed-seed{0,1}.json`. 2 seeds, tight agreement.

**Measured (seed 0; seed 1 within ±0.01 on every cell):**

| horizon | cos_rollout | cos_freeze1 | cos_persist | cos_chance | σ_hidden (logvar) | σ_visible | calib_gap | attractor cos |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| k1 | **0.232** | 0.232 | 0.033 | 0.016 | −9.461 | −9.836 | +0.374 | 0.219 |
| k2 | 0.116 | 0.227 | 0.035 | 0.028 | −9.312 | −9.676 | +0.364 | 0.452 |
| k3 | 0.016 | 0.224 | 0.032 | 0.004 | −9.399 | −9.819 | +0.420 | 0.716 |
| k4 | **−0.035** | 0.224 | 0.033 | −0.001 | −9.437 | −9.846 | +0.409 | 0.756 |
| k6 | −0.027 | 0.222 | 0.036 | −0.001 | −9.503 | −9.887 | +0.384 | 0.768 |
| k8 | −0.006 | **0.213** | 0.035 | −0.003 | −9.564 | −9.938 | +0.373 | **0.805** |

**Reading.**
- **Fidelity dies by k3.** The recursively fed-back belief matches the true future at k1 (0.232, 7×
  persistence) but decays to **chance by k3** (0.016) and goes *negative* at k4 — even faster than the
  pre-reset ckpt (which reached chance at k4). Blind autoregression is worthless past ~1 step.
- **σ dissipates → false confidence (falsifier NOT met).** Hidden-cell σ nets **−9.461 → −9.564** across
  the horizon (a small k2 bump, then monotone decay): the field grows *more* confident exactly as its
  predictions become worthless. The absolute σ is **lower than the pre-reset ckpt** (~−9.5 vs −7.8) → the
  speed recipe made the field *more* confident overall, which makes the temporal miscalibration **worse**,
  not better. The backlog P0.1 falsifier ("speed+jerk already fixed it") is refuted.
- **Attractor collapse is sharper.** Inter-sample centered cosine climbs **0.219 → 0.805** (vs 0.21→0.57
  pre-reset): by k8 every sample's belief has drifted to ~one common attractor. Belief centered-energy
  collapses to ~0 while true-token energy stays flat — this is the model, not the scene ("Biased Dreams"
  2604.25416, measured on the operative model).
- **Freeze-1 is the safe operative mode, confirmed.** Running the imagination *once* at k=1 and holding it
  retains **0.232 → 0.213 cosine flat across all 8 horizons** and beats persistence (~0.033) 7× throughout.
  The pathology is the **recursion**, not the 1-step prediction — the same verdict as 07-17, now on the
  operative model.
- **Refinement vs 07-17: the σ is *spatially* calibrated but *temporally* anti-calibrated.** Two calibration
  properties are actually GOOD on the operative model: (a) `calib_gap` is **positive ~+0.37–0.42** — hidden
  cells get more σ than visible ones (it knows the masked sector is uncertain); (b) per-cell error↔variance
  correlation is **positive 0.29–0.43** within each horizon (σ points at the right cells *spatially*). The
  failure is precisely one axis: σ does not **grow with horizon** while fidelity dies. This sharpens the
  design target — we do not need to rebuild spatial σ, only to make it *horizon-aware*.

**Falsifier verdict (P0.1):** **REPRODUCES** (σ-dissipation + attractor collapse persist on the operative
speed flagship). The speed+jerk recipe did not fix the recursion pathology; the "which ingredient fixed it"
branch does not apply.

**This is the third leg of the fleet's one story** (FLEET_REVIEW §3): imagination is real but modest
(panel: vision_use 12.9 %, imagination 8.7 %), planning brains are speed-starved (tactical wp 3.38 m),
and **recursion + decode waste the good 1-step imagination** — E1 is the mechanism for the third clause.

---

## 2. E2 — readout orthogonality / isotropy on the operative flagship (D-021 admissibility)

**Setup.** The verified 2026-07-10 instrument (`spectral_orthogonality.orthogonality_report`), unchanged,
pointed at the same operative ckpt. 40 val episodes → **7,964 readout latents** (dim 2048, `world.encode`).
Script `Implementation/orthogonality_verification/run_orthogonality_flagship.py`; result
`.../2026-07-18-orth-flagship-speed.json`.

| metric | step-6500 base250cam (07-17) | **flagship-speed @19k (07-18)** | read |
|---|--:|--:|---|
| active_k (energy-knee subspace) | 23 | **19** | task-relevant readout dims ≪ 2048 |
| iso_ratio_active | 0.254 | **0.546** | ↑ crossed 0.5 — SIGReg converging |
| cond_number_active | 217.9 | **61.2** | ↓ 3.6× better conditioned |
| rms_offdiag_corr | 0.424 | **0.321** | ↓ but still > 0.1 → not orthogonal yet |
| cov_effective_rank | 26.0 | **29.7** | ~tens, over-provisioned |
| iso_ratio_global | ~0 | ~0 (5.1e-8) | over-provisioning **by design**, not failure |
| participation_ratio_global | 5.0 | 9.3 | — |
| **verdict** | NOT-YET-ADMISSIBLE | **NOT-YET-ADMISSIBLE** | gated on off-diagonal 0.32 > 0.1 |

**Reading.**
- **The 07-17 prediction is verified.** "Expect iso to rise with SIGReg training" — it did: active-subspace
  isotropy 0.254 → 0.546, conditioning 218 → 61 as the run matured 6.5k → 19k. The favorable trajectory is
  the point; the readout is *converging* toward the LeJEPA identifiability condition.
- **Still not licensed.** The verdict trips on residual **coordinate correlation** (rms_offdiag 0.32 > 0.1):
  SIGReg's slice-Gaussianity target is not the same as active-subspace *coordinate* isotropy (a known gap).
  So D-021's "optimal-planning" corollary (2605.26379) remains **withheld** on this ckpt — the `spectral.py`
  spectrum stays descriptive (rank/knee), not a planning-optimality certificate.
- **G1 evidence, reaffirmed on the operative model.** active_k ≈ 19, cov_eff_rank ≈ 30 ≪ 2048 → the readout
  is **over-provisioned**; **latent capacity is not the D1-failure cause** (rules out "too small a latent").
  Combined with E1, the architecture-side story for G1 is: capacity is fine, recursion + decode are the leaks.

---

## 3. Actionable recommendations (G-B, each tied to a gate + falsifier per G-AI1)

1. **[operative, cheap, D-018 escalate] Run the operative H15 self-monitor in parallel-horizon mode, not
   autoregressive.** freeze-1 holds 0.213–0.232 flat vs rollout's collapse-to-chance. Predict each horizon
   direct-from-last-observation (like the predictor's parallel heads). **Gate: D8** (AUROC on
   degraded-visibility). **Falsifier:** parallel-horizon D8 AUROC ≤ autoregressive → recursion isn't the
   operative bottleneck at ≤8 steps (contradicted by this data → unlikely). *Changes self-monitor semantics
   → escalate before it becomes the operative default.* (Backlog 0b-B.)
2. **[build, D-018 escalate] Multi-step belief-rollout training with a horizon-aware σ term + anti-attractor
   penalty.** The σ is spatially calibrated but temporally flat; target = σ that **grows monotonically with
   blind horizon** (NLL at k∈{1,2,4} on the *recursive* path + penalize belief-energy collapse /
   inter-sample-cosine growth). **Gate: D9 + D8. Falsifier:** σ still dissipates after multi-step training →
   the ImaginationField architecture (not the recipe) is the ceiling → adopt parallel-horizon permanently.
   (Backlog 0b-A; this is flagship-v2 lever (a)/(f)-adjacent.)
3. **[watch, no action] Readout whitening / orthogonality penalty is a *latent* lever, not urgent.** Iso is
   converging on its own; a whitening lever would only be worth a one-lever bake-off IF the @30k ckpt still
   reads NOT-YET-ADMISSIBLE **and** D1/D2 want the optimal-planning corollary. **Falsifier for building it:**
   @30k iso_ratio_active ≥ 0.7 and rms_offdiag < 0.1 → SIGReg gets there alone, drop the lever. (Backlog 3b.)
4. **[decision-grade re-run, queued] Re-run E1+E2 verbatim on flagship @30k the moment it lands** (turnkey:
   the two scripts above; ~2 min pod time). This is the only step from validated → decision-grade, and it
   couples directly to the flagship-vs-CV verdict (G1). Expected: σ-dissipation persists (architecture
   property); iso_ratio_active rises further toward (but likely not past) admissibility.

**No architecture change is executed this run** (G-AI1: every lever names its gate + falsifier and is
D-018-escalated; nothing touches the trained config).

---

## 4. Provenance & honesty (P8)

- **Val distribution differs from 07-17.** 07-17 used comma2k19 val; this run used the pod's canonical
  PhysicalAI held-out val (the set TanitEval's panels use). Absolute cosine levels are therefore not
  directly comparable to 07-17 (0.232 vs 0.357 at k1); the **qualitative pathology** (dissipation, attractor,
  freeze-1-safe) is a model-internal blind-rollout property and transfers — it is what the falsifier tests,
  and it reproduced. The @30k re-run inherits the same val, keeping the comparison internally consistent.
- **Instrument identity guaranteed by test.** `tests/test_flagship_parity.py` (5 tests) pins the
  flagship variant's metric primitives bit-for-bit against the 07-17 `blind_rollout` ones, so the two runs
  are read on the same instrument. Full package: **15/15 tests green** (tanitad venv).
- **@19k, not @30k.** This is the operative model 11k steps short of the Stage-0 verdict; labeled
  *validated*, not *decision-grade*. No hypothesis status change (H15, H11, H3 unchanged; evidence deltas
  only).
- Process: the 2026-07-10 orthogonality instrument is **still unmerged** (now 3rd+ week). Re-flagged for
  orchestrator merge in STATE.
