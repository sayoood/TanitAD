# H2 — Phase 1 plan (consolidated, decision-ready)

**Date:** 2026-07-25 · **Status:** all three commissioned inputs delivered; this is the synthesis.
**Inputs:** `H2_SUBSTRATE_AND_LABELING.md` (substrate/label/counts) · `Research/2026-07-25-h2-sensor-attention/H2_RESEARCH_AND_SOTA.md` (prior art/novelty/architecture) · `Data Engineering/Research/2026-07-25-h2-multicam-data-survey/H2_EXTERNAL_DATA_SURVEY.md` (corpora) · `H2_DESIGN_FRAMING.md` (design + PI constraints D1–D4).

---

## 1. Verdict: **BUILD**, phase 1 in-corpus, with the claim re-framed

Every gating fact came back favourable except one, and that one changes the *claim*, not the *feasibility*.

| Gate | Result | Verdict |
|---|---|---|
| Multi-camera substrate | **7 cameras, 360°, 100.00 % presence**, per-camera intrinsics + 6-DoF extrinsics on 100 %, `obstacle.offline` 96.90 % | ✅ better than assumed |
| Non-circular label | `L1_gate` touches **no model input**; ~624/2,376 parity episodes = **15× the power bar** | ✅ |
| Situation power | intersections **846 eps**, lane changes **1,172 eps** | ✅ powered |
| Roundabouts | **19 strict / 105 loose** — corroborated absent by an independent prior | ❌ **descope from phase 1** |
| Efficiency (PI primary) | **1.83 %** of frames need a 2nd camera → **84–85 % of multi-camera encoder compute saved**, and *not* threshold-fragile | ✅ strong |
| Front-camera decidability | ROC-AUC **0.650/0.685** vs a 0.46–0.56 shuffle band — above chance but **weak**, and a *lower bound* | ⚠️ **re-scopes the claim** |
| Novelty | mechanism **already published** (DriveMoE, CVPR 2026) — **but its supervision is circular** | ⚠️ **re-frames the contribution** |

**The honest claim, as it now stands:**

> Not *"we invented learned camera selection."* Instead: **the field's camera-selection supervision is circular — labels are hand-written rules scored against themselves, so "learned the need" and "learned the rule" are indistinguishable. We construct the label counterfactually, evaluate by withholding, and report what changes.** Plus a measured deployment number: **84–85 % of surround-camera compute is avoidable.**

That is a defensible, novel, and *falsifiable* contribution. It is smaller than "we invented this" and considerably more likely to survive review.

---

## 2. ⭐ The first experiment is NOT the one we set out to run

**MEASURED:** the encoder's actual input is **51.4°**, while `front_wide_120` sees **120.5°**. **We discard 57 % of the front camera's own field before asking for a second camera.**

`cross_left_120` spans ≈ +6.5° … +127.7°. Our crop ends at ≈ +26°. So the band **+26° … +60°** is **already captured, already paid for, and thrown away.**

> **E0 — "are we solving a problem we created?"** Recompute `L1_gate` against the *full* 120.5° front field instead of the 51.4° crop. If a large share of "needs cross-left" events are actually visible in discarded front pixels, then the first and cheapest fix is **widen the crop**, not **activate a second camera** — and H2's real scope narrows to the genuinely off-front residual.

**This must run before any model work.** It is pure geometry over existing annotations (CPU, hours). It cannot fail informatively: either it shrinks the problem (a win — cheaper capability) or it confirms the residual is genuinely off-front (a win — the workstream is well-posed). Not running it risks building a router whose job is to undo our own preprocessing.

---

## 3. Sequence — each step cheap, each gated

| # | Step | Cost | Gate to proceed |
|---|---|---|---|
| **E0** | **Widened-crop recompute** (§2) | CPU, hours | Report the off-front **residual** share; H2 scope = that residual |
| **E1** | **Held-out 3 m confirmation.** Decision-relevance lift is 2.22× [1.30, 3.14] at 3 m but **0.43× [0.24, 0.71] at 6 m** — and **3.0 m was chosen after seeing the sweep.** Post-hoc threshold selection; the substrate agent flagged it on itself | **2 CPU-hours** | Lift holds on held-out episodes at 3.0 m only. **If it does not, the label is not decision-relevant and H2 stops here** |
| **E2** | **Camera axis in `LakeRecord`** — `frames` is single-view `[T,C,S,S]`; H2 needs a camera dimension. Unavoidable for *every* option incl. staying in-corpus | 1.5–2 eng-days | — |
| **E3** | **Encoder-feature decidability probe.** Replaces geometric proxies with real frozen-WM features; settles whether 0.65 is the substrate's ceiling or an artifact of the proxy | **~30 GPU-min, no training** | AUC materially > 0.685 ⇒ *anticipation* claim survives; ≈0.65 ⇒ re-scope to *reaction* |
| **E4** | **The discriminating probe (4 arms).** **A** head on frozen front-cam latents · **B** *no-image control* (symbolic context only) · **C** majority · **D** best-fixed-camera + random. PR-AUC per class, paired episode-cluster bootstrap B=2000, ≥40 clusters | hours, no pod | See falsifiers below |
| **E5** | Only if E4 passes: the situation→option→sensor head, hard-gated, with router-collapse guards | pod-days | — |

### Pre-registered falsifiers (both outcomes committed, per the operating standard)
- **F-A** — A's CI overlaps C (majority) ⇒ **V1 refuted at the substrate**; branch to V2, do **not** build the head.
- **F-B (decisive)** — **B matches A** ⇒ need is predictable **without the image**; the claim is **vacuous however high A scores** ⇒ back to label design. *(Precedent this is not paranoia: DriveBench's text-only ablation had GPT-4o at 35.37 % clean vs **36.48 % text-only** — removing the image helped.)*
- **F-C** — ≥2 cameras needed on >50 % of frames ⇒ the efficiency claim is dead. *(Already measured at **1.83 %** — F-C is provisionally cleared, to be re-checked post-E0.)*
- **Publishable null, committed in advance:** *"off-front sensor need is not predictable from the forward view above the majority baseline."* Since **nobody has tested this non-circularly**, that negative is arguably more interesting than a modest positive.

---

## 4. Corpus decision — and the one that is not about counts

**Phase 1 runs in-corpus on PhysicalAI-AV.** Intersections (846 eps) and lane changes (1,172 eps) clear both bars — `N_mech` = 40 and `N_train` = 200 — so **no acquisition is needed to start.** Roundabouts (19 strict) are descoped.

⚠️ **But a separate condition binds regardless of any count.** PhysicalAI-AV is **`gated-confidential`**, and derivatives inherit the strictest tier. **An H2 result built only on it can never be shown publicly** — no paper figures, no demo, no USP. If H2 is meant to be a differentiator, a **`ship`-tier** corpus is required *independently of coverage*.

**And we already own one.** **Cosmos-Drive-Dreams** (`cosmos_dd`): **CC-BY-4.0, `owned-safe`, commercially usable**, loaded since D-014 with an existing loader — **7 views including cross-left/right 120°**, per-view intrinsics, 30 fps ego/camera poses, **4D object tracking with IDs**, HD map. **~2–3 eng-days**, cheapest option by 2×, zero new license risk.

**Recommendation:** develop on PhysicalAI (powered, zero acquisition) **and, for $0, count Cosmos-DD's intersection/roundabout content from already-cached metadata.** If it is adequate, it becomes the *publishable* twin of every phase-1 result. That is a free option we should exercise immediately.

**Do NOT pursue:** A2D2 (3D boxes only inside the *front* FOV — it never annotates the agents our label needs — plus CC-BY-ND forbids derivatives) · Waymo/Waymax (`refuse`) · ZOD, BDD100K, comma2k19 (single camera) · rounD/inD/exiD/highD/INTERACTION (**drone/BEV — no ego camera at all**; their roundabout counts are the richest in the survey and are precisely the numbers that must not masquerade as a fit).

---

## 5. Guards wired in from the start (each with a cited precedent)

1. **Label circularity** → the **no-image control arm** (F-B). Our own precedent: `route_target = _NAV_TO_ROUTE[nav_cmd]` ⇒ `route_skill = 0.0` *by construction*.
2. **Router collapse** → per-class recall + **PR-AUC vs majority, never accuracy**; routing entropy logged from step 0; **global-batch (not micro-batch) load balancing** — micro-batch LBL *prevents* specialization (arXiv:2501.11873). Precedent: GEMINUS's VanillaMoE (59.23 DS) is **worse than its own single-expert baseline** (60.73), and its router scores **2.87 % on "Give Way"** behind a 68.06 % aggregate — the same accuracy-hides-zero-skill shape as our `route_acc 1.0 / route_skill 0.0`.
3. **The Soft-MoE trap** → Soft-MoE is differentiable and collapse-free but **combines all tokens**, which would **silently delete the efficiency claim**. Every metric is reported on the **hard-gated** path.
4. **Two-rig geometry** → frustum membership must use **per-clip `cy`** (rig A ≈ 543 / rig B ≈ 755), or the label is systematically wrong for rig B and merely looks like noise.
5. **Estimator discipline** → paired episode-cluster bootstrap, estimator named on every interval, ≥40 clusters. **Never `overlapping_holdout_se`** (measured 2026-07-25 to bias intervals *and* point estimates, up to ×4.29 with sign flips).
6. **Asymmetric costs stated, not buried in an F1** → a missed activation is a potential safety failure; a spurious one costs only compute. The operating point is chosen on that asymmetry and reported.

---

## 6. Open items for the PI

1. **Cosmos-DD as the publishable twin** — approve the $0 metadata count? *(Recommended: yes. It is free and it is the only route to H2 being a public USP.)*
2. **Novelty framing** — accept "the field's supervision is circular; ours is not" as the contribution, given DriveMoE (CVPR 2026) already published the mechanism? The efficiency lever is also crowded (adaptive perception publishes 70.20 % energy reduction at ~2 % accuracy cost), so **C-CAP, not C-EFF, is where the defensible novelty sits** — while efficiency remains the motivation and a strong measured number.
3. **If E3 says ~0.65** — accept the re-scope from *anticipate* to *detect-on-arrival*, or invest in a longer-horizon input (the WM predictor's imagination, capped by σ→chance at k=4)?

## 7. Escalations (not H2-blocking, but owed)

- 🔴 **License, PROVISIONAL:** `schema.py` registers `nuscenes` as `CC-BY-NC-4.0, share_alike=False`; **nuscenes.org states CC BY-NC-**SA**-4.0**. If confirmed, `share_alike` must flip to `True`, routing nuScenes into the segregated copyleft shard. Same root-cause class as the 2026-07-13 ZOD correction. **Needs a terms-of-use confirmation.**
- **`DATASET_LANDSCAPE.md` has no camera-count column** — that omission is what let a single-camera corpus (ZOD) be briefed as a multi-camera candidate. Add the column; it is a two-minute structural fix for a class of error that has now bitten twice.
- **`calib_r1.pinhole_rectify`** (D-016 R1, 9/9 green) is **still not folded into `calib.py`** — required if nuScenes is ever ingested (f_eff ≈ 360 vs our 266).
