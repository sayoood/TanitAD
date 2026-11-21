# H2 · E1 + E0 — RESULTS

**Date:** 2026-07-26 (local, Europe/Berlin; started 2026-07-25). **Author:** research engineer (H2 E0/E1 stream).
**Pre-registration:** `PRE_REGISTRATION.md`, this folder, written before any held-out number existed.
**Status:** decision doc. **CPU only — no GPU, no training, no model inference, no pod touched.**

**Evidence classes:** `MEASURED` (ours + path/command) · `PUBLISHED` (cited) · `INHERITED` (another
agent/doc, NOT re-verified) · `ESTIMATED` · `HYPOTHESIS`.

---

# 0. VERDICT IN ONE BOX

> ## E1 — **FAIL**
> Held-out lift at the pre-registered 3.0 m gate: **1.16× · 95 % CI [1.00, 1.33]**
> (paired episode-cluster bootstrap, B = 2000, **2,159 episode-clusters**, zero overlap with the sweep).
> The CI **includes 1.0** and the point estimate is **far below the 1.5× bar**. Both PASS criteria fail.
>
> **`L1_gate` is not decision-relevant. Per the pre-registration, H2 stops at the label.**
> The in-sample **2.22×** was an **80-episode sampling fluctuation read at the argmax of a 6-point
> threshold sweep** — proven below on the sweep's *own two chunks*.
>
> ## E0 — the split (run as a flagged deviation, §5)
> **63.6 % [60.5 %, 66.3 %] of `L1_gate` positives are RECOVERABLE-BY-CROP.** Only **36.4 %** are a
> genuine off-front residual. Uniform across strata.
>
> ## …but "just widen the crop" is **NOT** the cheap fix — and this is the surprise
> Covering the full front field costs **2.30× the native pixels** (MEASURED). Widening therefore costs
> **2.30× front-encoder tokens always-on**, versus **1.007–1.064 camera-equivalents** for selectively
> activating a second camera. **Selective activation is ~2.2× cheaper than widening the crop.**
>
> ## The efficiency claim (C-EFF, the PI's primary) **SURVIVES and improves**
> Residual need-rate **0.67 %** of frames ⇒ **1.007 cameras/frame** ⇒ **85.6 % saved vs always-on-7**
> (84.8 % under a conservative ±2 s hysteresis). Previously 1.02–1.13 cams/frame, 84–85 %.
> **The gate *rate* reproduced out-of-sample to three digits (1.832 % vs 1.83 %) on 27× the episodes.**

**The one-line reading:** *the label's **frequency** generalises perfectly; its **decision-relevance**
does not exist. C-EFF is real, C-CAP's label is not.*

---

# 1. What was run, and on what

| | |
|---|---|
| Held-out episodes | **2,159 clips / 24 chunks / 11,054,224 agent-frames / 325,614 scored frames** |
| Overlap with the sweep | **0 clips** (asserted in code, `h2e_e1.py:33`) |
| Episode-clusters | **2,159** — 54× the n ≥ 40 decision-grade bar |
| Estimator | **paired episode-cluster bootstrap**, B = 2000, seed 0, resampling clips; both arms recomputed inside the same draw. Machinery imported from `taniteval/taniteval/ci.py` (`episode_index`, `_draws`). **`overlapping_holdout_se` used nowhere.** |
| Geometry | per-clip `(cx, cy)` **and** per-clip 6-DoF extrinsics on every projection (two-rig corpus: rig A cy ≈ 543 / rig B cy ≈ 755) |
| Cost | ~7 CPU-minutes of compute + a **2 MB** HF calibration pull. No GPU. pod1/pod2/pod3 untouched. |

**How the held-out set got 27× bigger than planned.** `L1_gate` needs `obstacle.offline` +
calibration + `egomotion` on the same chunk. Locally only 4 chunks had all three, 2 of which the
sweep used. The missing artifact was **calibration, which is ~60 KB per chunk** — so all 22 remaining
chunks were pulled for **2 MB total** (`scripts/h2e_pull_calib.py`, 22/22 succeeded). The held-out set
was pre-committed as *"all local obstacle chunks except 0036/0170"* **before** the pull, so its size is
not a post-hoc choice. (MEASURED)

### 1.1 The rewrite is validated against the thing it audits

The vectorised builder (`scripts/h2e_build.py`) reproduces `H2_SUBSTRATE_AND_LABELING.md` §6.3/§6.4
**exactly**, on the substrate audit's own 80 clips (`fidelity_check.json`):

| quantity | substrate audit (published) | this rewrite | |
|---|---|---|---|
| `L1_gate` positive frames | 1.83 % | **1.83 %** | ✅ |
| `L1_label` positive frames | 0.91 % | **0.91 %** | ✅ |
| lift @ 3.0 m | 2.22× [1.30, 3.14] | **2.22× [1.28, 3.16]** | ✅ (n⁺ = 228 identical) |
| lift @ 6.0 m | 0.43× [0.24, 0.71] | **0.43× [0.24, 0.70]** | ✅ |

*A rewrite that cannot reproduce the number it is auditing is not admissible. This one can.* (MEASURED)

---

# 2. E1 — THE STOP GATE: **FAIL**

**Pre-registered, d = 3.0 m only, no sweep, no re-tuning.**

| quantity | held-out (2,159 eps) | sweep (80 eps, INHERITED) |
|---|---|---|
| **lift** | **1.16× · CI [0.9975, 1.3272]** | 2.22× [1.30, 3.14] |
| P(response \| gate⁺) | 26.86 % (n = 5,964 frames) | 50.00 % (n = 228) |
| P(response \| gate⁻) | 23.15 % (n = 319,650 frames) | 22.57 % |
| risk difference | **+0.0371, CI [−0.0006, +0.0759] — not separated** | — |
| gate rate | **1.832 %** | 1.83 % |
| episodes gate-positive | 503 / 2,159 | 21 / 80 |
| bootstrap draws used / skipped | 2000 / **0** | — |

**Both PASS criteria fail:** the CI includes 1.0 (lower bound **0.9975**), and 1.16 is far under the
1.5× bar the substrate agent itself set. **Excess lift attenuates 1.22 → 0.16, a factor of 7.6.**
(MEASURED, `e1_heldout.json`, `scripts/h2e_e1.py`)

### 2.1 Robustness — the FAIL is not an artifact of any single choice

| check | result | reading |
|---|---|---|
| response threshold −0.5 / −1.0 / −1.5 / −2.0 m/s | 1.10× / **1.16×** / 1.17× / 1.12× | flat — not a response-definition artifact |
| phase-0-selected held-out clips only (n = 548) | 1.17× [0.85, 1.52] | unchanged |
| **speed-matched pooled lift** (MH-weighted over 6 speed bins) | **1.18× [1.02, 1.34]** | **the obvious confound does not hide an effect** |
| per-chunk (24 chunks) | 16/24 above 1.0, median ≈ 1.18× | no single chunk drives it |

The speed check mattered: gate⁺ frames sit at **5.87 m/s** mean ego speed vs **12.31 m/s** for gate⁻,
so a deceleration response is mechanically easier to *miss* in one arm. Matching on speed moves the
estimate by **+0.02** — nothing. (MEASURED, `e1_confound_check.json`)

---

# 3. WHY it failed — root cause, measured, not inferred

Three nested samples, **identical code, identical d = 3.0 m** (`e1_attenuation.json`):

| sample | clips | n⁺ | lift | 95 % CI |
|---|---|---|---|---|
| (a) the sweep's own 80 clips | 80 | 228 | **2.22×** | [1.28, 3.16] |
| (b) **all 185 clips of the *same two chunks*** | 185 | 567 | 1.44× | [0.95, 1.96] |
| (b′) **the 105 clips of those chunks the sweep did NOT draw** | 105 | 339 | **0.99×** | [0.53, 1.53] |
| (c) held-out | 2,159 | 5,964 | **1.16×** | [1.00, 1.33] |

> **(b′) is the decisive line. Same two chunks, same countries, same rig, same code — and the effect
> is exactly nothing (0.99×).** The attenuation is **not geography, not distribution shift, not a
> code difference.** It is the 80-episode draw itself.

**How lucky was that draw?** Drawing 80 clips at a time from the held-out pool with the threshold
**held fixed at 3.0 m**: the 2.5/50/97.5 percentiles of the lift are **0.42 / 1.14 / 2.14**, and
**P(lift ≥ 2.22) = 2.0 %**. An 80-episode sample can move this statistic by 5× on its own — and the
sweep additionally *selected* 3.0 m as the argmax of six candidates after seeing them, which is a
second multiplier on top. (MEASURED)

**Corroboration from the other direction:** among the 24 held-out chunks (~90 clips each),
**2 individually produce a "2.2×-with-CI-excluding-1" result** (chunk 0906: 2.52× [1.40, 3.58];
chunk 0928: 2.28× [1.34, 3.23]) — while the pooled answer is 1.16×. **A chunk-scale sample reproduces
the original headline by chance roughly one time in twelve.**

### 3.1 Root-cause class, for `RETRACTION_LOG.md`

> **Class: a point estimate quoted at the argmax of a sweep on a small sample, as though it had been
> pre-specified.** The correction is not "use a different threshold" — it is that *the argmax of a
> sweep is not an estimate of anything* until it is confirmed out of sample. This is the same class
> `CLAUDE.md` already records for learning-curve exponents (*"the same log gives −0.387 … −0.738
> depending on the window"*) and it has now cost the program a second time in a different instrument.
>
> **What the substrate agent got right, and it should be said:** it flagged this on itself, named the
> exact probe, pre-committed both outcomes, and refused to headline 2.22×. The process worked. The
> number did not.

---

# 4. The mechanism story — **corroborated in shape, refuted in magnitude**

Held-out lift as a continuous function of the conflict radius. **Descriptive only — it cannot and
does not move the §2 verdict.**

| d (m) | 1.0 | 1.5 | 2.0 | 2.5 | **3.0** | 3.5 | 4.0 | 4.5 | 5.0 | 6.0 | 8.0 | 10.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **lift** | **1.90×** | 1.73× | 1.12× | 1.12× | **1.16×** | 1.02× | 0.91× | 0.81× | 0.77× | 0.71× | 0.66× | 0.67× |
| CI lo | 1.16 | 1.13 | 0.78 | 0.91 | 1.00 | 0.89 | 0.81 | 0.72 | 0.69 | 0.64 | 0.60 | 0.61 |
| CI hi | 2.64 | 2.35 | 1.52 | 1.36 | 1.35 | 1.16 | 1.02 | 0.90 | 0.86 | 0.79 | 0.73 | 0.74 |
| gate rate | 0.07 % | 0.12 % | 0.30 % | 0.71 % | 1.83 % | 3.37 % | 5.55 % | 6.89 % | 7.86 % | 9.65 % | 12.4 % | 12.8 % |

**The profile is smooth and monotone, and it crosses 1.0 at ≈ 3.5–4.0 m — exactly where a lane width
(3.0–3.5 m) puts two vehicle centres in adjacent lanes.** The physical mechanism the substrate agent
proposed is therefore **corroborated**: proximity really does grade into free-flow adjacency, and the
sign flip at 6 m is real, not an artifact.

**But the mechanism does not rescue 3.0 m.** Out of sample the peak sits at **≤ 1.5 m**, and 3.0 m is
already on the shoulder of the crossing. The sweep placed the peak at 3.0 m; held-out places it at
half that. *A real monotone trend plus an 80-episode sample is exactly the recipe for an argmax that
lands in the wrong place with a confident-looking CI.*

⚠️ **`d = 1.0 m` is NOT a replacement threshold and must not be adopted as one.** It is
1.90× [1.16, 2.64] on **220 frames = 0.07 % of frames**, and at 1.0 m *centre-to-centre* with agents
measuring 1.9–2.3 m wide, the boxes are **deeply overlapping** — that is an imminent-collision event
class, not a yield-anticipation one, and it is ~26× rarer. Re-scoping onto it would be precisely the
re-sweep the pre-registration forbids. It is recorded as a **lead for a differently-designed label**,
nothing more.

**One further lead, same status.** The only speed regime with held-out decision-relevance is
**< 3 m/s: 2.50× [1.69, 3.51]** (1,548 gate⁺ frames) — but there P(response | gate⁻) is just 7.74 %,
i.e. the ego is already creeping or stopped, and "yields" and "is already yielding" are not
separable. Suggestive, confounded, not actionable as-is.

---

# 5. E0 — the real scope

> ⚠️ **DOCUMENTED DEVIATION FROM THE PRE-REGISTRATION.** E0 was pre-registered to run **only if E1
> passed**. E1 failed, and it is run anyway, for two reasons stated before the numbers were read:
> **(1)** C-EFF — the PI's *primary* claim per D3 — depends only on the gate **rate**, which is
> independent of the lift and which reproduced exactly; **(2)** if H2 needs a *different* label, the
> off-front geometry decides what that label can even be scoped to, and it is already computed.
> **Nothing in §5 can resurrect `L1_gate`; the §2 verdict stands unchanged.** The deviation is
> recorded here rather than by editing the pre-registration.

### 5.1 The split — 63.6 % is recoverable by widening the crop

| level | RECOVERABLE-BY-CROP | GENUINE OFF-FRONT RESIDUAL |
|---|---|---|
| **agent-frames** (15,202 triggers) | **9,661 = 63.6 % [62.1 %, 64.9 %]** | 5,541 = **36.4 %** |
| **frames** (5,964 gate⁺) | **3,792 = 63.6 % [60.5 %, 66.3 %]** | 2,172 = **36.4 %** |
| **episodes** (of 2,159) | 503 gate⁺ under the crop | **398** still gate⁺ on the full field |

Frame-level decomposition of the recoverable share: **3,410** because the triggering agent is simply
visible in the discarded front pixels, **382** because the wide field also reveals a *conflicting*
agent, which retires the request under clause (iv).

**Azimuth confirms the mechanism.** Recoverable triggers sit at **|az| p50 = 26.4°** — immediately
outside the 25.7° crop edge, i.e. in the thrown-away band. Residual triggers sit at **p50 = 49.6°**
(p10 = 39.2°). The two populations are cleanly separated by the crop boundary, which is what a
correct geometry should produce.

**Gate rate falls 1.832 % → 0.667 %** on the full front field: **a 2.75× smaller problem.**

### 5.2 Strata — the split is a property of geometry, not of situation

| stratum | frames | clips | gate⁺ | gate rate | recoverable share |
|---|---|---|---|---|---|
| **junction** (in) | 24,886 | 279 | 663 | **2.66 %** | **63.5 % [52.4 %, 74.2 %]** |
| junction (out) | 300,728 | 2,159 | 5,301 | 1.76 % | 63.6 % [60.6 %, 66.4 %] |
| **lane change** (in) | 9,040 | 133 | 129 | 1.43 % | **62.0 % [54.4 %, 71.3 %]** |
| lane change (out) | 316,574 | 2,159 | 5,835 | 1.84 % | 63.6 % [60.7 %, 66.3 %] |

Detectors are `situ_full.py`'s, thresholds verbatim, evaluated on the label grid. **The recoverable
share is ~63 % everywhere** — it does not vary by situation, so no stratum-specific scoping helps.
Junctions do enrich the need itself (2.66 % vs 1.76 %, **1.5×**); lane changes slightly deplete it.

### 5.3 ⭐ The counter-intuitive result: widening the crop is the **expensive** option

The plan doc states (INHERITED, `H2_PHASE1_PLAN.md` §2): *"the first and cheapest fix is widen the
crop, not activate a second camera."* **On compute, that is measured to be false.**

**MEASURED** from the real f-theta polynomials over **4,965** clip-camera rows: covering the full
120.5° front field takes **2.30× the native pixels** of the 51.4° crop (p05 2.28, p95 2.33 — tight,
because f-theta is near-equiangular).

| option | what it costs | duty cycle |
|---|---|---|
| **A** — keep 256 px, widen the field | **2.30× coarser angular resolution** on the *primary driving task* | always |
| **B** — keep angular resolution, widen horizontally (588×256) | **~2.30× front-encoder tokens** | **always on** |
| **C** — keep angular resolution, square (588²) | **~5.3× front-encoder tokens** | **always on** |
| **selective 2nd camera on the residual** | **1.007 cams/frame** (1.064 at ±2 s hysteresis) | 0.67 % of frames |

> **Selective activation is ~2.2× cheaper than the cheapest resolution-preserving widening, and ~5×
> cheaper than the square one.** Option A is free in FLOPs but pays in angular resolution on the task
> the model actually has to do — and `calib.py`'s `F_REF = 266` canonicalization (D-016) exists
> precisely to hold that resolution fixed across corpora.
>
> **This cuts against the direction the brief anticipated.** The recoverable share *is* large — 63.6 %,
> comfortably "outcome A" — but the inference "therefore widen the crop" does **not** follow. Reported
> as measured.

A fourth option the numbers permit, offered without a recommendation: a **second wide low-resolution
tap of the same front camera feeding only the H2 head**, leaving the driving encoder's 51.4° crop
untouched. No new sensor, no resolution loss on the primary task. Its cost is a small extra encoder
pass and it is not priced here.

### 5.4 C-EFF re-derived on the residual — the claim survives and improves

| definition | policy | left | right | either | **cams/frame** | vs 7 | vs 3 |
|---|---|---|---|---|---|---|---|
| cropped 51.4° | instantaneous | 0.78 % | 1.12 % | 1.83 % | 1.0190 | 85.4 % | 66.0 % |
| cropped 51.4° | ±1 s hysteresis | 2.72 % | 3.49 % | 5.91 % | 1.0621 | 84.8 % | 64.6 % |
| cropped 51.4° | ±2 s hysteresis | 4.42 % | 5.42 % | 9.24 % | 1.0983 | 84.3 % | 63.4 % |
| **RESIDUAL 120.5°** | **instantaneous** | 0.27 % | 0.41 % | **0.67 %** | **1.0068** | **85.6 %** | 66.4 % |
| **RESIDUAL 120.5°** | ±1 s hysteresis | 1.68 % | 2.04 % | 3.63 % | 1.0373 | 85.2 % | 65.4 % |
| **RESIDUAL 120.5°** | ±2 s hysteresis | 2.97 % | 3.45 % | 6.23 % | **1.0642** | **84.8 %** | 64.5 % |

**Re-derived headline:** **1.007–1.064 cameras/frame ⇒ 84.8–85.6 % of surround-camera encoder compute
avoidable**, replacing the previous 1.02–1.13 ⇒ 84–85 %. The claim **strengthens** slightly, because
the residual need is rarer than the cropped need. (MEASURED)

**And it is now confirmed out of sample**, which the original was not: the gate rate reproduced at
**1.832 %** on 2,159 held-out episodes against **1.83 %** on the sweep's 80 — 27× the data, three
matching digits. **C-EFF is the robust half of H2.** F-C (the ">50 % of frames need ≥2 cameras"
falsifier) is cleared by a factor of ~75.

### 5.5 The residual gate is not decision-relevant either

Full-front `L1_gate` at 3.0 m: **1.09× [0.89, 1.30]** (n⁺ = 2,172, 2,159 clusters). Consistent with
§2 — narrowing to the genuine off-front residual does not recover decision-relevance.

---

# 6. What H2 phase 1 should now target — recommendation

**1. Do not build the `L1_gate` head. Do not re-sweep the threshold.** The pre-registered stop
condition fired. Adopting `d = 1.0 m`, or `< 3 m/s`, or a hysteresis variant, would be the same
error a second time on the same data. (Binding, per `PRE_REGISTRATION.md`.)

**2. Split the workstream, because the two claims have now diverged sharply.**

| | status | action |
|---|---|---|
| **C-EFF** (efficiency, PI-primary per D3) | ✅ **MEASURED, held-out-confirmed, 85 % saving** | **Proceed.** It never depended on the lift. It is quotable today at **1.007–1.064 cams/frame**. |
| **C-CAP** (capability / decision-relevance) | ❌ **label refuted at the substrate** | **Back to label design.** No GPU until a label passes a held-out gate. |

**3. C-EFF needs one honest caveat attached, not removed.** A need-rate is only an efficiency claim
if the need is *correctly identified*. Right now we have a rate whose decision-relevance is null, so
the defensible sentence is *"under a geometric visibility-plus-conflict criterion, ≥ 2 cameras are
required on 0.67 % of frames"* — **not** *"85 % of compute is avoidable without losing capability."*
The second sentence needs a working C-CAP label. **State it the first way until then.**

**4. For the replacement label, the ranking has changed** — and E0 changes it, which is the value E0
delivered despite the FAIL:
   - **`L1-occlusion` moves to first** (substrate §6.7). E0 shows **63.6 %** of current positives are
     agents in the *discarded front pixels* — i.e. clause (i) is dominated by our own crop, not by
     the world. Occlusion is the version of "the model cannot see it" that is a property of the
     scene. ~1 CPU-day, no new data.
   - **`L1-lateral`** (aborted/deferred lane change instead of deceleration) — the deceleration
     response is measured flat across four thresholds, so the response variable, not just the gate,
     is a suspect. Lane changes are 133 held-out episodes here and 1,172 corpus-wide.
   - **Any new label must be pre-registered with a held-out gate before it is believed.** The cost is
     ~7 CPU-minutes on the now-cached 24-chunk table. There is no excuse for skipping it.

**5. Scope H2 to the 36.4 % residual, not the 63.6 %** — but **not** by widening the crop (§5.3). The
residual is 398 episodes here, ≈ 0.67 % of frames, and it is where a second camera is genuinely the
only remedy.

**6. Two items to escalate rather than write into a doc** (the "10-day README" failure mode):
   - `H2_PHASE1_PLAN.md` §2 and §3 carry **2.22×** and *"the cheapest fix is widen the crop"*. Both
     are now measured false. That doc is the synthesis a PI decision would be taken from — **it needs
     the correction applied, not appended.**
   - `RETRACTION_LOG.md` needs the §3.1 root-cause class.

---

# 7. Limitations, stated plainly

1. **`L1_gate` is refuted; the *idea* of learned camera selection is not.** E1 tested one label. A
   negative on it says nothing about whether a better-posed label is predictable — and the
   pre-registered publishable null (*"off-front sensor need is not predictable from the forward
   view"*) is **not** established by this result, because we never got to test predictability.
2. **`obstacle.offline` is `scene:obstacles:autolabels:v2` — machine labels, not human GT.** Stamp
   `prov: "autolabel"`. Systematic misses of small/distant agents would attenuate any lift, and that
   possibility is not excluded here.
3. **Occlusion is still not computed.** Clause (i) remains out-of-FOV/out-of-crop only (substrate U-3
   is still open, and §6.4 above raises its priority).
4. **The response is longitudinal only.** A yield expressed as a lateral abort is invisible to
   `Δv ≤ −1 m/s`. Four thresholds were checked and all are flat, which argues the response variable
   is under-powered rather than mis-tuned.
5. **Frames with no annotated agent at all are not in the denominator** — inherited from the
   substrate audit's aggregation and kept deliberately so the 1.83 % is comparable. It makes the
   efficiency rates **conservative** (the true fraction of *all* frames needing a 2nd camera is
   lower).
6. **Strata are kinematic proxies**, not map labels — PhysicalAI-AV ships no HD map (three-leg
   absence, substrate §B.3).
7. **2,159 held-out clips are not the 2,376-episode parity corpus.** 548 are phase-0-selected; the
   result is unchanged on that subset (1.17× [0.85, 1.52]). Parity is not broken — **nothing here
   re-selects training episodes.**

---

# 8. Deliverable manifest

**All artifacts are in the repo working tree, staged by the orchestrator. Nothing is on a pod or in a
worktree. Per instruction: no `git add`, no commit, no push was performed by this agent.**

Repo path: `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-25-h2-e0-e1/`

| file | what it is |
|---|---|
| `H2_E0_E1_RESULTS.md` | this document |
| `PRE_REGISTRATION.md` | written before any held-out number existed |
| `e1_heldout.json` | **the E1 verdict** + per-chunk + distance profile + sensitivities |
| `e1_attenuation.json` | the nested-sample root cause, incl. the 105-clip (b′) line |
| `e1_confound_check.json` | speed-matched lift (exploratory) |
| `e0_split.json` | recoverable/residual split, strata, C-EFF, widened-crop cost |
| `fidelity_check.json` | proof the rewrite reproduces the substrate audit |
| `scripts/h2e_probe_split.py` | establishes the held-out split; proves zero sweep overlap |
| `scripts/h2e_pull_calib.py` | the 2 MB calibration pull (22/22 chunks) |
| `scripts/h2e_build.py` | the label-table builder (vectorised `crux3` + `in_front_full` + strata) |
| `scripts/h2e_stats.py` | paired episode-cluster bootstrap, ratio/difference/rate/share forms |
| `scripts/h2e_fidelity.py` · `h2e_e1.py` · `h2e_attenuation.py` · `h2e_e0.py` · `h2e_confound.py` | the five analyses, in run order |
| `scripts/_vendored_crux.py` | the substrate audit's `crux.py`, vendored so the pipeline runs from the repo alone |

**Intermediate parquets are NOT in the repo** (14 M rows, ~1 GB) and live at
`…\scratchpad\h2e_heldout.parquet` and `h2e_sweepchunks.parquet`. They rebuild from the repo scripts
in **~4.5 CPU-minutes**:

```
python scripts/h2e_pull_calib.py
python scripts/h2e_build.py <out>\h2e_sweepchunks.parquet 0036 0170
python scripts/h2e_build.py <out>\h2e_heldout.parquet 0174 0181 0617 0834 0840 0852 0868 0906 \
       0919 0928 0931 1573 1852 1860 1864 1870 1880 1900 2433 2498 2500 2503 2820 2838
python scripts/h2e_fidelity.py && python scripts/h2e_e1.py && python scripts/h2e_attenuation.py \
       && python scripts/h2e_e0.py && python scripts/h2e_confound.py
```

**Data read (read-only, dev box):** `C:\Users\Admin\tanitad-data\physicalai\` —
`labels/obstacle.offline/*.zip` (24 chunks), `labels/egomotion/*.zip`,
`calibration/camera_intrinsics/*.parquet` + `sensor_extrinsics/*.parquet` (52 chunks after the pull),
`r0/phase0_selection.parquet`. **No pod touched. No GPU used. No training job perturbed.**
