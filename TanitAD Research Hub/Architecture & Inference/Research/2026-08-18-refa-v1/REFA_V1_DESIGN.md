# REF-A **v1** — design, validation, and pre-registration

**Date:** 2026-08-18 · **Status:** ⭐ **TRAINING-READY** (code landed, 39/39 tests green,
CPU + GPU smokes measured) · **Not yet launched** — the 30k v6F S-W run owns Thor until ~2026-08-22.

**Mandate (PI, 2026-08-18):** *redesign REF-A into v1 combining findings 1–7 from our work and the
literature review; adopt test-time planning; DINOv3 instead of DINOv2; DINO-WM's own CEM/MPC
configuration; adapt the width of the visual interface; repair the action search; implement and
validate; keep the predictive character to 6 s in operative/tactical; keep the multi-hierarchical
layers and goal conditioning.*

---

## 1. WHAT REF-A WAS, AND THE TWO FINDINGS THAT REDESIGNED IT

REF-A: frozen **DINOv2-B/14**, 224×224 → **256 tokens / 51.39°**, adapter into a **compact state**,
**supervised head**, **no planning**. ADE@2s **2.1675 m** (T0), plateaued.

* **Ours** (`…/2026-08-18-encoder-localisation/`, verdict `P2-PRESERVED`): the information is
  **present and preserved at five measured stages**, arriving intact at the exact latent the eval
  decoded; the trained adapter did **not** collapse (per-dim std 0.8011 vs 0.220 random). ⇒ the
  deficit is on the **consumption** side.
* **The literature** (`…/2026-08-18-frozen-encoder-literature/`): frozen encoders succeed in
  configuration **A** (huge frozen VLM + wide interface + supervised head — FROST-Drive: frozen 14 B
  **8.17 RFS / 1.04 m** beats the *same encoder fine-tuned* **8.13 / 1.47**, while a frozen ImageNet
  ViT is the worst arm at **7.39 / 2.28**) or configuration **B** (moderate frozen encoder + future-
  feature prediction + test-time planning — DINO-WM, V-JEPA 2-AC, DeepSight, LAW).
  **REF-A had A's consumer with B's encoder class and neither one's compensating strength.**

⇒ **v1 commits to configuration B in full, and fixes the interface defects that were
configuration-independent.**

## 2. THE NINE CHANGES, EACH WITH ITS SOURCE AND ITS TEST

| # | change | source | pinned by |
|---|---|---|---|
| 1 | **DINOv3** ViT-L/16 (d=1024), still frozen, still cached | DeepSight's world-state target; PI | `test_change_1_…` |
| 2 | **640 tokens / 120° / 256×640** (was 256 / 51.39° / 224²) | DINOv2's H/14 map documented-insufficient for small+distant objects | `test_change_2_…` (×2) |
| 3 | ⛔ **no bottleneck**: `d_state ≥ d_enc` | FROST-Drive width ablation **8.17 → 7.68** | `test_change_3_…` (×3) |
| 4 | primary loss = **predict future patch features** (L2) | DINO-WM: latent L2, *"no auxiliary reconstruction, reward, or terminal losses"* | `test_the_primary_loss_…` |
| 5 | **patch tokens only, never CLS/pooled** | DINO-WM ablation: global R3M/ResNet18/**CLS** "significantly degrades" | `test_change_5_…` |
| 6 | behaviour from **iCEM + MPC** at test time | DINO-WM / V-JEPA 2-AC / GPC; repairs C101 | 8 planner tests |
| 7 | hierarchy kept: strategic —FiLM→ tactical —FiLM→ operative | our `fourbrain.run_hierarchy`; PI | `test_the_tactical_intent_actually_reaches_…` |
| 8 | goals enter the **planning COST**, not only a head | v3 direction: *"target-speed and mode-switching become the PLANNING COST not a head"* | `test_plan_runs_end_to_end_…` |
| 9 | **6 s** at three rates; strategic on its **own** predictor over a strategy-only subspace | PI + three-planner directive | `test_change_9_…` (×2), `test_the_strategic_predictor_is_SEPARATE_…` |

## 3. THE ARCHITECTURE, MEASURED

```
cached DINOv3 fields [T, 640, 1024]      (frozen; 0 trainable encoder params)
   └─ FeatureStandardizer (fit ONCE, refit raises)
      └─ WideAdapter          640×1024 → 640×1024   ⛔ no compression
         ├─ operative  TokenFieldPredictor  Δ0.2 s × 30 = 6.0 s   (640 tokens)
         ├─ tactical   TokenFieldPredictor  Δ0.6 s × 10 = 6.0 s   ( 64 queries)
         ├─ strategic  SubspacePredictor    Δ1.5 s ×  4 = 6.0 s   (256-d subspace)
         ├─ StrategicPolicy → ctx ─FiLM→ TacticalPolicy → intent ─FiLM→ operative
         └─ proposal head (AUXILIARY — seeds the planner, never in the loss)
```

**MEASURED parameter budget** (real config, `RefAV1Config()` with both brains):

| component | params | share |
|---|---:|---:|
| operative field predictor | 80,043,008 | 45.99 % |
| tactical field predictor | 54,850,560 | 31.52 % |
| tactical policy | 21,684,493 | 12.46 % |
| strategic policy | 7,990,275 | 4.59 % |
| adapter | 2,762,752 | 1.59 % |
| strategic subspace predictor | 1,911,040 | 1.10 % |
| proposal (auxiliary) | 537,108 | 0.31 % |
| **TRAINABLE TOTAL** | **174,043,172** | **sub-300M ✅** |
| frozen encoder | **0** | by construction |

## 4. ⭐ THE ACTION-SEARCH REPAIR (C101), AND WHAT IT DOES *NOT* FIX

C101: our CEM planner was **35.8 % worse than constant velocity at T1**. Configuration B puts that
component on the critical path, so it is repaired first, four ways:

1. **Kinematic action space** — `(a, κ)` at 10 Hz through the programme's single unicycle
   integrator. Every candidate is feasible; jerk is a parameterisation, not a penalty.
2. **iCEM, not vanilla CEM** — coloured noise `S(f) ∝ f^-β`, β = 2.5, plus elite memory and
   shift-init. MEASURED: lag-1 autocorrelation **> 0.60** (β=2.5) vs **< 0.15** (β=0), and
   step-to-step control change **< 0.5×** white noise. White-noise plans demand accelerations that
   reverse every step — the most likely mechanical cause of C101.
3. ⭐ **Baseline injection — a structural floor.** CV, hold-`v0`, `decel_1.5` and the imitation
   proposal are injected into every iteration *and* into the final argmin. **The returned plan's
   modelled cost is ≤ min(baseline cost) by construction.** Proven by an adversarial cost function
   that makes the CEM optimum arbitrarily bad (`test_THE_FLOOR_…`); the guard is switchable so its
   contribution is a measurable ablation, not an always-on comfort.
4. ⛔ **AND THE LIMIT, STATED IN THE CODE:** the floor is a floor in **modelled** cost. A
   miscalibrated cost model still loses on realised metrics — that is *how C101 happened*, and
   mechanism 3 alone would not have caught it. `cost_fidelity()` measures Spearman ρ between
   modelled cost and realised outcome; the admission gate below refuses to quote any planner number
   until it passes.

⚠️ **A REAL BUG THIS CAUGHT ON FIRST RUN.** Because the baselines also compete *inside* the CEM
loop, the CEM's own best sample is frequently the baseline itself. With a strict `<` in the final
comparison the result was labelled `source="cem"` while the returned controls were byte-identical to
constant velocity — **the floor held but the provenance lied**, and a planner report would have
credited the search for a plan it did not find. Fixed to `<=` with the baseline's own controls
returned; the test that found it is now the regression guard.

## 5. ⭐ THE COMPUTE FINDING THAT CHANGED THE PLANNING DESIGN

MEASURED on the RTX 4060 at the real geometry: a 10-step rollout of the **640×1024** operative field
costs **160 ms per candidate**, scaling linearly (2553 / 5129 / 10519 ms at n = 16 / 32 / 64 — the
GPU is saturated already at n=16), i.e. **~6 candidates/s**.

| planner configuration | rollouts | RTX 4060 | A40-class (÷6) |
|---|---:|---:|---:|
| DINO-WM published (300 × 30) **on the operative field** | 1,975 | **325 s / tick** | 54 s / tick |
| DINO-WM published (300 × 30) **on the tactical field** (v1 default) | 2,099 | ⭐ **21.5 s / tick** | ~3.5 s / tick |

⇒ **Coarse-to-fine, using the hierarchy we already have:** search on the **64-query tactical field**
(10× cheaper), then **re-score the winner and the baselines on the full 640-token operative field**.
The fine re-score costs a handful of rollouts, not a population, and it makes a coarse-level mistake
appear as a **reported rank flip** (`coarse_fine_agree=False`) instead of vanishing. `plan_level=
"operative"` stays reachable so the deviation from DINO-WM is an ablation, not an unstated
compromise. GPU smoke at the full published config: `source=cem`, coarse cost 1.1058, **fine
re-score agrees** (plan 1.0922 < cv 1.1009), peak CUDA **0.95 GB**.

Also fixed here: the naive rollout stored every intermediate field — **300 × 10 × 1.31 MB = 3.9 GB**
for a cost that reads only the terminal field. `last_only` + chunked evaluation; peak measured
**0.95 GB**.

## 6. VALIDATION PERFORMED (all MEASURED today)

* **39/39 tests green** — `stack/tests/test_refa_v1.py`, every change in §2 pinned by name.
* **Real-scale build**: 174,043,172 trainable, 0 encoder params, shapes
  `op_pred (1,30,640,1024)` / `tac_pred (1,10,64,1024)` / `str_pred (1,4,256)` — all three at 6.0 s.
* **CPU trainer smoke** (6 steps, hierarchy ON): loss 89.48 → 45.67, all three feature losses
  present and finite, gnorm finite, **adapter per-dim std 0.490 → 0.520** (the REF-A collapse
  monitor, healthy).
* **GPU planner smoke** at DINO-WM's full published configuration: 21.5 s/tick, floor holds,
  coarse/fine agree, 0.95 GB peak.

⛔ **What is NOT validated:** anything about driving quality. No number here is a result about
performance; there is no trained checkpoint. §7 exists so that cannot be forgotten later.

## 7. PRE-REGISTRATION — written before any REF-A v1 training run

**Primary question.** Does configuration B (frozen encoder + feature prediction + test-time
planning) beat REF-A's configuration-A recipe *on the same frozen-encoder class*?

**Primary comparison.** REF-A v1 at **5 k steps** vs **REF-A's banked 5 k milestone**, paired
episode-cluster bootstrap (`taniteval/ci.py`; ⛔ never `overlapping_holdout_se`), **all four metric
families reported separately, never pooled**. Tier: **T0** for anything teacher-forced; a driving
claim requires **T1**.

**Gates, both outcomes committed in advance:**

| gate | pass | fail |
|---|---|---|
| **G1 — planner admissible at all** | `cost_fidelity` ρ ≥ 0.5 on ≥ 200 banked windows | ⛔ **no planner number may be quoted**, and v1 is scored head-only until fixed |
| **G2 — the floor is real, not just modelled** | realised metric of the planner ≤ realised CV on the same windows | report the gap; the floor was modelled-only, exactly C101's shape |
| **G3 — the redesign beats the old recipe** | paired Δ vs REF-A@5k separated and favourable on ≥ 2 of 4 families | configuration B does not rescue a small frozen encoder here — a real, publishable negative |
| **G4 — coarse-to-fine is honest** | `coarse_fine_agree` ≥ 0.8 across eval windows | the tactical search is too coarse; fall back to `plan_level="operative"` and re-cost |
| **G5 — no collapse** | adapter per-dim std stays > 0.4 | dead run, stop |

**Ablations pre-registered (each one flag):** `--no-hierarchy` (change #7 control) ·
`inject_baselines=False` (the floor's contribution) · `beta=0` (vanilla CEM vs iCEM — the C101
mechanism) · `plan_level="operative"` (the DINO-WM-faithful cost) · `d_state=512` (the bottleneck
FROST-Drive measured).

**Declared confounds.** v1 changes **nine** things at once. It is therefore a *recipe* test, not an
attribution test: a win does not tell us which change earned it, and the ablation list above is the
only path to attribution. ⇒ **E-RECON-2 remains the attribution experiment and is not replaced by
this arm.**

## 8. LAUNCH READINESS

**Ready:** model, planner, trainer, tests, geometry contract, gates.
**Blocked on two inputs, both named:**

1. **Stage-1 cache does not exist yet** — DINOv3 ViT-L/16 patch tokens at 256×640/120° for the
   canonical 2,376-episode corpus (parity key `physicalai-train-e438721ae894`, skip-hash
   `f09e44db`). The trainer **refuses** any cache whose geometry disagrees, so this cannot be
   silently substituted. Estimated: one encode pass, no gradients, resumable.
2. **DINOv3 licence** — still an open PI decision in the backlog.

**Command once both land:**

```bash
python stack/scripts/refa_v1_train.py --cache <dinov3_w120_cache> --steps 5000 --bs 8 --out ~/experiments/refa-v1-5k
```

Thor is occupied by v6F S-W until ~2026-08-22; this arm does not contend for it before then.
