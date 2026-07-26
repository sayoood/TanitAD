# H2 — Phase 1 plan (consolidated, decision-ready)

> # 🔴 SUPERSEDED IN PART — READ THIS FIRST (2026-07-26)
>
> **E1, the pre-registered stop gate, FAILED. Two claims in §2 and §3 below are now measured FALSE.**
> Results: `.../incoming/2026-07-25-h2-e0-e1/H2_E0_E1_RESULTS.md`.
>
> **1. `L1_gate` has NO decision-relevance. The label is refuted as a capability target.**
> Held-out lift at 3.0 m = **1.16× [0.9975, 1.3272]** (paired episode-cluster bootstrap, B=2000,
> **2,159 episode-clusters**, zero clip overlap with the sweep) — the CI **includes 1.0** and sits far
> below the 1.5× bar. **Both pre-registered PASS criteria fail.**
> The root cause is MEASURED, not inferred: on the sweep's *own two chunks*, the 105 clips it did **not**
> draw give **0.99× [0.53, 1.53]** — same geography, same rig, same code, **zero effect**. 80-clip
> subsamples at a *fixed* 3.0 m span **0.42–2.14**, and **P(lift ≥ 2.22) = 2.0 %**. **The 2.22× was an
> 80-episode fluctuation read at a sweep's argmax** — the winner's-curse/forking-paths class. The
> mechanism's *shape* survives (monotone decay crossing 1.0 at ≈3.5–4.0 m, the lane-width prediction),
> but out of sample the peak sits at **≤1.5 m**, not 3.0 m — and it was correctly **not** re-scoped onto.
>
> **2. §2's "widen the crop is the cheaper fix" is measured FALSE — I had the arithmetic backwards.**
> The split is **63.6 % [60.5, 66.3] recoverable-by-crop / 36.4 % genuine off-front residual** (uniform
> across strata: junction 63.5 %, lane-change 62.0 %), which *looked* like it favoured widening. But
> covering the full front field costs **2.30× the native pixels — always on** — versus **1.007
> cameras/frame** for selective activation. **Selective activation is ~2.2× CHEAPER than widening the
> crop.** §2's recommendation is withdrawn; the E0 experiment was still worth running, because it is what
> produced the refutation.
>
> **What SURVIVES and is quotable:** the **need-RATE**. The residual rate is **0.67 %** ⇒ **1.007–1.064
> cameras/frame ⇒ 84.8–85.6 % saved vs always-on-7**, and the gate rate **reproduced out-of-sample to
> three digits** (1.832 % vs 1.83 %) on **27× the episodes**.
>
> ⚠️ **But state C-EFF precisely, or it over-claims.** The rate is robust; what it is a rate *of* is a
> **geometric-presence** trigger whose **safety-relevance is exactly what E1 just refuted**. So the
> honest form is: *"an off-front agent is geometrically proximate to the ego's path on 0.67 % of frames"*
> — **not** *"the ego needed another camera on 0.67 % of frames."* The efficiency arithmetic is sound;
> the semantics of the trigger are open until a decision-relevant label exists.
>
> **Net:** *the label's frequency generalises perfectly; its decision-relevance does not exist.*
> **C-EFF (rate) stands. C-CAP (capability) needs a new label — §3's E-sequence restarts at the label.**
>
> ---
>
> ## ✅ RESOLVED 2026-07-26 — a replacement label PASSED. H2 is GO.
> `…/incoming/2026-07-26-h2-label-v2/` — **`L2` replaced BOTH halves of the refuted label**, not just the
> trigger. Trigger = the minimum braking effort `a_req` the ego would need, along its **own realised path**,
> to stay clear of an off-front agent — **AND** no agent inside the crop already requires it (that second
> clause **is** the agent-removal counterfactual: delete the off-front agents and nothing the encoder sees
> demands the brake, so that agent is the *binding* constraint). Response = an actual brake application.
>
> **Held-out lift 2.41× [1.3998, 3.7041]**, paired episode-cluster bootstrap B=2000, **1,415 clusters**,
> zero chunk overlap with DEV. **τ\* was set by a power rule that never reads the lift, and 0.5 m/s² is NOT
> the curve's maximum (1.0 gives 2.77×) — there is no argmax anywhere in the construction.**
> **Leave-one-chunk-out: 16/16 exclude 1.0** (2.15–2.77×) — precisely the test `L1_gate` failed.
> Coverage: trigger 0.555 % of frames, label 0.0712 %; left 0.316 % / right 0.239 %, never both.
>
> ### ⚠️ Three qualifications that travel WITH the GO — do not drop them
> 1. **Quote 2.09× [1.19, 3.38] (speed-adjusted), not 2.41×.** Braking-state adjustment removes separation
>    (1.35× [0.82, 2.05]); a lead-time test favours mediation (trigger precedes the brake in 65.6 % of
>    cases, median 0.55 s) but does **not** settle it.
> 2. **Junctions are NULL — 0.45× [0.00, 1.40].** That is H2's *headline* situation. All the signal is
>    **off-junction (2.86×)**. The capability we can currently evidence is not the intersection story.
> 3. **The genuine off-front residual is NOT separated — 1.66× [0.78, 3.06].** ⭐ **So today's demonstrable
>    capability is FRONT-PERIPHERY attention, not cross-camera switching.** This converges with two
>    independent results: E0's **63.6 % recoverable-by-crop**, and the Orin measurement that **widening the
>    front field costs only ~+8.6 % of the tick** because the encoder is 4.5 % of DRAM bytes. **The
>    defensible H2 v1 is adaptive front-periphery attention — cheaper, measurable now, and honestly
>    scoped.** Cross-camera switching stays a hypothesis, not a claim.
>
> **C-EFF's condition is now satisfied for the primary scope**, and **the E-sequence resumes at E3**
> (encoder-feature decidability, ~30 GPU-min) — **run E3 BEFORE building the head**, per the original plan.
>
> *Discipline note worth preserving:* the L2 agent refuted **two of its own hypotheses** and reported them
> as measured — the path-preserved ego route buys nothing (1.78 vs 1.89), and agent-constant-velocity is
> **backwards** (realised-future agents give **3.93× [1.79, 6.78]**). It deliberately did **not** adopt the
> better-looking variant, flagging it for a fresh pre-registration instead. That is the correct handling of
> a post-hoc improvement.

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

## 5b. ✅ CORPUS DECISION — PI, 2026-07-25 (binding)

> *"I think we can use PhysicalAI data set for research, so let's do it with this data as proof of concept."*

**Phase 1 = PhysicalAI-AV, research tier, proof-of-concept. No external acquisition. Execution starts
now.** The powered strata (intersections **846 eps**, lane changes **1,172 eps**) clear both `N_mech`=40
and `N_train`=200, so nothing blocks the PoC. Roundabouts stay descoped.

**What this decision explicitly accepts** (recorded so it is not rediscovered as a surprise later):
PhysicalAI-AV is `gated-confidential`, so **every phase-1 artifact — numbers, figures, videos — is
research-tier and internal.** It cannot appear in a paper, a demo, or a public USP claim in this form.
That is a deliberate, sensible PoC trade: *prove the capability first on the corpus that is powered and
free, decide the publication vehicle afterwards.*

**The deferred option remains open and cheap:** if the PoC works, **Cosmos-Drive-Dreams** (CC-BY-4.0,
`owned-safe`, commercially usable, already loaded, 7 views incl. cross-left/right 120°, 4D tracking with
IDs) is the **ship-tier twin** that would make the same result publishable — ~2–3 eng-days, and counting
its content costs **$0** from cached metadata. Not needed to start; **re-raise at the phase-1 verdict.**

---

## 6. Open items for the PI

1. ~~**Cosmos-DD as the publishable twin** — approve the $0 metadata count?~~ **DEFERRED by §5b** — not
   needed for the PoC; re-raise at the phase-1 verdict as the publication vehicle.
2. **Novelty framing** — accept "the field's supervision is circular; ours is not" as the contribution, given DriveMoE (CVPR 2026) already published the mechanism? The efficiency lever is also crowded (adaptive perception publishes 70.20 % energy reduction at ~2 % accuracy cost), so **C-CAP, not C-EFF, is where the defensible novelty sits** — while efficiency remains the motivation and a strong measured number.
3. **If E3 says ~0.65** — accept the re-scope from *anticipate* to *detect-on-arrival*, or invest in a longer-horizon input (the WM predictor's imagination, capped by σ→chance at k=4)?

## 7. Escalations (not H2-blocking, but owed)

- 🔴 **License, PROVISIONAL:** `schema.py` registers `nuscenes` as `CC-BY-NC-4.0, share_alike=False`; **nuscenes.org states CC BY-NC-**SA**-4.0**. If confirmed, `share_alike` must flip to `True`, routing nuScenes into the segregated copyleft shard. Same root-cause class as the 2026-07-13 ZOD correction. **Needs a terms-of-use confirmation.**
- **`DATASET_LANDSCAPE.md` has no camera-count column** — that omission is what let a single-camera corpus (ZOD) be briefed as a multi-camera candidate. Add the column; it is a two-minute structural fix for a class of error that has now bitten twice.
- **`calib_r1.pinhole_rectify`** (D-016 R1, 9/9 green) is **still not folded into `calib.py`** — required if nuScenes is ever ingested (f_eff ≈ 360 vs our 266).
