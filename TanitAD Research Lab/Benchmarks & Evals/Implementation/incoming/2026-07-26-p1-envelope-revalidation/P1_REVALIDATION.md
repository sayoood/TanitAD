# Re-validating the P1 envelope on the YAW arm — where does the substrate actually stop?

**Date:** 2026-07-26 (Europe/Berlin; pods log UTC) · **Host:** `tanitad-eval` (A40) · **Stream:** Benchmarks & Eval
**Answers:** `HORIZON_ENVELOPE.md` §8 escalation **#2** / §5.2 — *"extend the P1 sweep, and it must extend the
YAW arm, not just the lateral one"* · `GATE_30K_RESULTS.md` §10.1's one remaining branch.
**Constraints honoured:** pod1 (training v2corpus) never contacted · pod2 (H2 classifier) never contacted ·
pod3 (YouTube harvest, round 7) never contacted · one job on `tanitad-eval` · staged, never committed, never pushed.

**Evidence classes:** `MEASURED` (ours + artifact path) · `PUBLISHED` (cited) · `INHERITED` (another agent/doc,
**not** re-verified) · `ESTIMATED` · `HYPOTHESIS`.

---

## 0. Headline

| # | Result | Class |
|:--:|---|---|
| **1** | ⭐⭐ **`ENV_YAW_MAX = 12°` was never measured as a limit — it is the END OF A HARDCODED CLI GRID** (`lowood_probe.py:228`, `--yaw-grid "0,1,2,3,5,8,12"`). No criterion selecting an edge exists anywhere in P1. **P1's own report puts the yaw no-degradation edge at ≤ 2°**, and the paired Δ is CI-**separated from 3° onward** — so the shipped constant sits **four sweep points deep into separated degradation**, while `ENV_LAT_MAX` sits at its **first**. The two axes were never set by a common criterion. Both are commented `# MEASURED` in live code. | `MEASURED` |
| **2** | ⭐⭐ **OUTCOME B, over-determined.** The defensible *usable* yaw edge is **15.47° [12.14, 17.88]** (IDF = 0.5, episode-cluster bootstrap on the **edge itself**, B = 2000, 881 win / 40 clusters). Its **CI lower bound touches the shipped 12°** ⇒ **a 1.29× widening at the point estimate and none at the lower bound. Not "materially beyond 12°".** | `MEASURED` |
| **3** | ⭐⭐ **NO admissible horizon becomes a MEASUREMENT — and this was settled BEFORE any GPU ran, from the staged per-window dumps.** Even at **yaw = ∞** the *lateral* clause alone leaves **3.75 %** of K = 20 windows outside (junction **18.13 %**), and `MEASUREMENT` requires **zero**. **⇒ our closed-loop numbers are EXTRAPOLATIONS at every admissible horizon and must be labelled so permanently.** | `MEASURED` |
| **4** | ⭐⭐ **Two independent instruments converge at ~26°.** Model-free: half the frame is **fabricated by border replication** at the FOV half-angle **25.70°** (exact, from `f_eff = 266 px`). Model-mediated: the frame is **worth no more than pure noise** at **26.41° [18.33, 29.63]**. ⛔ **The junction stratum's p90 at the standing 2 s horizon is 28.14° — PAST that point.** | `MEASURED` |
| **5** | ⛔ **At the shipped 12° edge the warp has already destroyed 34.7 % of the frame's usable information and fabricated 26.4 % of its pixels.** No criterion that looks at image content would certify that and call it a "MEASURED envelope limit". | `MEASURED` |
| **6** | ⭐ **The P1 envelope is NOT a renderer-fidelity envelope.** The yaw warp is **geometrically exact for arbitrary scene depth** (`max|ΔH| = 0.000e+00` over 30 camera-height × pitch conditions; exact composition to 5.7e-14), so the substrate's only infidelity is FOV fabrication — yet ADE degrades **~2× faster**. **Roughly half of what the envelope measures is our own arm's OOD sensitivity, measured on v1 and applied to v4.** ⇒ **the lever is TRAINING, not rendering.** | `MEASURED` |
| **7** | ✅ **The C13 gate was applied to P1's own criterion, and it PASSES — outcome C refused on evidence.** Destroyed-observation controls: `dead_black` **+1.5619**, `dead_shuffle` +0.3388, `dead_noise` +0.1442, all CI-separated. ⚠️ **But the useful headroom is only 0.1442 m**, so every envelope number must be read against that scale — which P1's bare "+13.62 % at 12°" never was. | `MEASURED` |
| **8** | ✅ **80 checks against the committed P1 artifact, 0 mismatch** — every shared grid point (ADE, paired Δ, both CI bounds, `separated`) reproduces to 4 dp on a different host, and the 40-ep baseline independently reproduces the registry's canonical **0.4271 [0.3675, 0.4871]**. The extension is a strict continuation of P1's curve. | `MEASURED` |
| **9** | ⛔ **A self-refutation, recorded not overwritten.** I first *interpolated* that a 15.47° edge would move the K = 20 junction stratum to a minority-outside. **Recomputed exactly: 0.5165 — still a majority, still EXTRAPOLATION.** The real improvement is 13 % relative, not 27 %. | `MEASURED` |
| **10** | ⛔ **Do NOT resurrect the OOD ratio criterion on this.** Extending the sweep raises `sup(ratio_arr)` 1.298888 → ≈ **1.52**, clearing `RATIO_EXTRAPOLATION_X = 1.5` by **0.02**. A threshold 1.3 % under a re-measured ceiling is C13 in a new costume. **Leave clause 1 VOID.** | `MEASURED` |

---

## 1. PRE-REGISTRATION — written and staged BEFORE any number was produced

*Written after reading P1's source and report (§2, which is pure archival provenance and required no
experiment) and before executing a single measurement. Both outcomes are committed here in advance.*

### 1.1 The question

The closed-loop OOD guard's clause 1 (ratio > 1.5) is **arithmetically dead** — `sup(ratio_arr) = 1.298888`
(`PUBLISHED`, C13, `ood_blast_radius.json`). Only clause 2 — *steps leaving the P1 envelope
`|dlat| <= 3.0 m`, `|dpsi| <= 12 deg`* — can ever fire, and it fires so hard that **the last horizon at which
the closed loop is a pure MEASUREMENT is k = 4 (0.4 s)**, while `GATE_PROTOCOL` §0.3 refuses K <= 20.
The binding axis is **heading**: at K = 20 the `other` stratum is 0.0000 outside and `longitudinal` 0.0027,
but `junction` is **0.5879**, leaving via yaw (0.5824 vs 0.1813 lateral).

⇒ The envelope, not the horizon, is the blocker. **This run asks over what range of heading deviation the
substrate is actually faithful**, so the edge can be set where the evidence puts it rather than at 12°.

### 1.2 The three outcomes, committed in advance

| | outcome | what we will write |
|---|---|---|
| **A** | fidelity holds **materially beyond 12°** | the envelope widens to the measured edge. We state the new edge **with its interval**, and we report **exactly which horizons become measurements** — by recomputation, not assertion. |
| **B** | fidelity degrades **at or before 12°** | the envelope is **correct or too generous**. The closed-loop instrument **cannot be rescued by re-validation**, and our closed-loop numbers are **extrapolations at every admissible horizon and must be labelled so permanently.** A publishable, decision-relevant negative. **No stretching a marginal result into A.** |
| **C** | the **criterion itself has no dynamic range** | if P1's estimator cannot separate a *destroyed* observation from an intact one, then **no edge is recoverable from it at all** — neither 12° nor any replacement — and the envelope must be set on model-free geometry. This is a C13 finding about P1's own instrument and it is reported as such, not hidden inside A or B. |

### 1.3 ⚠️ The C13 gate, applied to my own criteria BEFORE they are cited

*"A guard that cannot fail is not a guard."* For each criterion: **what value makes it FAIL, and can the
estimator reach that value?**

| criterion | what it measures | FAILS when | can the estimator reach it? |
|---|---|---|---|
| **C-GEO** (primary, model-free) | fraction of the output frame with **no source information**, fabricated by `padding_mode="border"` replication | fabricated fraction grows without bound; **>= 0.5 is a hard ceiling** — that is the FOV half-angle, a geometric constant, not a chosen number | ✅ **Yes, by construction** — f(psi) is monotone 0 -> 1. Closed-form from `f_eff = 266 px`, `W = 256`. |
| **C-RT** (model-free) | round-trip `warp(+psi) -> warp(-psi)` residual vs the original frame | residual grows without bound | ✅ Yes — unbounded above. |
| **C-ADE** (P1's own) | open-loop ADE@2s under the warp, **paired vs Δ=0**, episode-cluster bootstrap | the ADE-vs-offset curve **breaks** from the trend established over 0–12° | ⚠️ **UNKNOWN until tested — and this is exactly outcome C.** Pre-committed instrument check below. |

**⚠️ Pre-committed dynamic-range control for C-ADE, without which no C-ADE number is admissible.**
P1's criterion is model-mediated and could saturate in *either* direction:

* it can **overstate** infidelity — under a *perfect* renderer, warping the observation to an offset pose
  while scoring against the **true-ego** GT still raises ADE, because correct behaviour from an offset pose
  genuinely differs. So a rise does **not** by itself prove the substrate is unfaithful;
* it can **understate** it — an arm that largely ignores the image (ours are known to regress to the mean)
  would show flat ADE under a *destroyed* observation.

⇒ **We therefore run destroyed-observation controls in the same pass**: `psi = 90°`, a **black** frame, and a
**phase-scrambled** frame. **Pre-committed rule: if the destroyed controls are not CI-separated far above the
12° value, C-ADE has no dynamic range, every P1 envelope number lies inside its noise floor, and outcome C is
declared.** The edge is then set on C-GEO alone and C-ADE is retired rather than re-quoted.

### 1.4 The design, fixed before the run

* **Arm / ckpt:** `flagship-30k` @ step **29999**, the arm P1 measured (`ood.py` stamps the envelope
  "on the flagship **v1** arm — NOT on v4"). Same ckpt file P1 used, md5 recorded in the preflight artifact.
* **Corpus:** `/root/valdata/physicalai-val-0c5f7dac3b11`. P1 used `sorted(...)[:12]` = 12 episodes / 265
  windows. We run **the same 12 first** (so the extension is a strict superset of P1's own curve and its
  0–12° points are a **reproduction check**), then widen to **40** episodes if affordable.
* **Yaw grid:** P1's `0,1,2,3,5,8,12` **extended** to `16,20,25,30` (+ the destroyed controls). The 0–12°
  points must reproduce P1's committed values or the run is void.
* **Estimator:** `episode_cluster_bootstrap` (`taniteval/ci.py`), B = 2000, unit = **val episode**; **paired**
  form for every condition vs Δ=0 on the same windows. `overlapping_holdout_se` appears nowhere.
* **Per-corpus, never pooled.** Unit of resampling is the cluster.
* **Decompose** lateral vs longitudinal vs heading throughout — this whole finding exists because an
  undecomposed statistic hid which axis was failing.

### 1.5 ⭐ One consequence is computable BEFORE any measurement, and it is pre-registered as such

`ood.verdict` returns `MEASUREMENT` only when **zero** windows are outside — and the envelope test is a
**disjunction over both axes**. At K = 20 the lateral clause alone already fires on **0.0375 of steps**
(`PUBLISHED`, `HORIZON_ENVELOPE.md` §4.5). **Therefore, if the lateral-only window fraction at K = 20 is
> 0, then widening the YAW envelope — to any value, including infinity — cannot make K = 20 a MEASUREMENT.**

That is an arithmetic claim, not a prediction, and it is settled from the staged per-window dumps with **no
GPU**. It is pre-registered here so that measuring it cannot look like a post-hoc rescue of outcome B.
**Priority order** (a killed run still yields value): **P1 provenance -> this arithmetic -> C-GEO ->
C-ADE sweep -> lateral arm.**

### 1.6 What is NOT blind, disclosed

I read `HORIZON_ENVELOPE.md` in full before designing this, so the K = 20/60/70/185 out-of-envelope
fractions, the junction decomposition and the p90 yaw multiples were known to me. **P1's provenance in §2
was NOT known to anyone** — it is established here for the first time, and it is the finding that
reframes the task.

---

## 2. P1's provenance — what was actually measured, and where 12° came from

**Primary sources, read in full** (not summaries):
`…/2026-07-23-lower-ood-closedloop-source/lowood_probe.py` (the sweep driver) ·
`…/lowood_flagship_ci.json` (the raw artifact `ood.py` loads) · `…/P1_DECISION_GRADE_FINDINGS.md` (P1's report).

### 2.1 What P1 measured

| | |
|---|---|
| **arm** | `flagship-30k` step 29999 (flagship **v1**) |
| **corpus** | `physicalai-val-0c5f7dac3b11`, **12 episodes -> 265 windows**, stride 8 |
| **surface** | **open-loop, force-GT.** Encode a real window -> roll the operative predictor 20 steps under the **TRUE** action sequence -> decode metric dpose -> SE(2) accumulate -> ADE@2s vs GT |
| **intervention** | **observation-only** warp. The speed channel `v0` and the GT stay the **true** ego |
| **estimator** | `episode_cluster_bootstrap` B=2000, + **paired** vs Δ=0 on the same windows ✅ admissible |
| **lat warp** | ground-plane (flat-road) homography — models **only road-surface parallax**, so it **under-models 3D structure** ⇒ P1 calls the lateral envelope an **OPTIMISTIC upper bound** |
| **yaw warp** | ⭐ **exact rotation homography about the down axis, DEPTH-INDEPENDENT** |

### 2.2 ⭐⭐ Where 12° came from — it is the **grid maximum**, not a measured limit

`lowood_probe.py:228`:

```python
ap.add_argument("--yaw-grid", default="0,1,2,3,5,8,12")
```

**The sweep stopped at 12° because the default string stopped at 12°.** There is **no criterion anywhere in
`lowood_probe.py` that selects an edge**, no degradation test, and no threshold. `lowood_flagship_ci.json`
contains seven yaw rows and the largest is 12.0. The same is true of `--lat-grid "…,3.0"`.

`taniteval/corridor.py:109-110` then records:

```python
ENV_LAT_MAX = 3.0             # MEASURED envelope limit, metres
ENV_YAW_MAX = 12.0            # MEASURED envelope limit, degrees
```

⛔ **What was MEASURED is "we swept to 12° and ADE rose 13.62 %". What is asserted is "12° is the limit of
validity". Those are different claims, and the second one has no artifact.** The `# MEASURED` comment is
`INHERITED`, and the constant it annotates is **the endpoint of a hardcoded CLI default**.

### 2.3 ⛔ And P1's own report puts the yaw edge at **2°**, not 12°

`P1_DECISION_GRADE_FINDINGS.md` §1.3, verbatim in substance: *lateral offset up to 2.0 m carries no
CI-separated OOD; **yaw is the more sensitive axis — it separates at 3°** and grows monotonically to
+0.055 at 12°*, concluding: *within ±2 m lateral and **≤2° yaw** the observation-OOD is indistinguishable
from on-path.*

From `lowood_flagship_ci.json`, paired Δ vs Δ=0 (`paired_episode_cluster_bootstrap`, B=2000, 265 win / 12 eps):

| Δψ (deg) | 1 | 2 | **3** | **5** | **8** | **12** |
|---|---|---|---|---|---|---|
| paired Δ ADE@2s | −0.0005 n.s. | +0.0035 n.s. | **+0.0165 SEP** | **+0.0257 SEP** | **+0.0392 SEP** | **+0.0551 SEP** |

| Δlat (m) | 0.25 | 0.5 | 0.75 | 1.0 | 1.5 | 2.0 | **3.0** |
|---|---|---|---|---|---|---|---|
| paired Δ ADE@2s | −0.0068 n.s. | −0.0031 n.s. | −0.0026 n.s. | +0.0026 n.s. | +0.0174 n.s. | +0.0254 n.s. | **+0.0658 SEP** |

> ### ⛔ **The two axes' edges were never set by a common criterion — and neither was set by any criterion at all.**
> On "largest offset with no CI-separated degradation" the edges would be **2.0 m / 2°**.
> On "first offset that separates" they would be **3.0 m / 3°**.
> The shipped envelope is **3.0 m / 12°** — lateral at its *first separated* point, yaw at its **fourth**
> (3°, 5°, 8°, 12° all separate). **`ENV_YAW_MAX` is the only one of the two that sits four sweep points
> deep into CI-separated degradation.** Under P1's own published reading, **the yaw envelope is already
> 6× too generous**, not too tight.

**Evidence class: `MEASURED` (ours, this run, from the committed primary artifacts named above).**

### 2.4 ⭐ The mechanism that makes the yaw arm special — and it cuts *against* an easy widening

P1's docstring: *"yaw Δ: **exact** rotation homography about the down axis (**depth-independent**)"*, and its
§2: *"The lateral homography is ground-plane-only → the lateral envelope is an optimistic bound (**yaw is
exact**)"*.

This is correct and it matters: a **pure camera rotation** induces a homography `H = K R K⁻¹` that is exact
for **any** scene geometry — no flat-road assumption, no parallax error, no depth. **So on the yaw arm the
warp model contributes zero geometric error.** Only two mechanisms can degrade it:

1. **finite field of view** — content rotating in from outside the frame does not exist and is **fabricated
   by border replication** (`padding_mode="border"`);
2. **resampling** — bilinear interpolation blur.

⇒ **The yaw arm's true limit is a FIELD-OF-VIEW limit**, and it is closed-form from `f_eff = 266 px` /
`W = 256 px` with **no model, no GPU and no renderer**. That is criterion **C-GEO**, and it is why this
question is answerable cheaply — and why AlpaSim's NuRec/gsplat is **not involved at all** (§9).

---

---

## 3. ⭐⭐ The pre-registered arithmetic (§1.5), settled: **widening the yaw envelope to INFINITY does not make ANY horizon a MEASUREMENT**

`MEASURED` (ours) · `scripts/yaw_edge_recompute.py` → `artifacts/yaw_edge_recompute.json` ·
recomputed from the **staged per-window dumps** of the committed K-sweep
(`…/2026-07-26-horizon-envelope-closeout/artifacts/perwindow_K*.pt`) using the **packaged**
`taniteval.ood` / `taniteval.corridor` rule — **no GPU, no model, no second implementation.**

The envelope test is a **disjunction over two axes**, and `verdict()` returns `MEASUREMENT` only when
**zero** windows are outside. So the limit of any possible yaw re-validation is the column
`yaw ≤ ∞`, where the yaw clause is switched off entirely and **only the lateral clause remains**.

| K | stratum | n win / clusters | @12° *(shipped)* | @20° | @25.7° | @30° | **@∞** | **verdict @∞** |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| **20** | overall | 881 / 40 | 0.1226 | 0.0795 | 0.0568 | 0.0443 | **0.0375** | ⛔ PARTIAL |
| **20** | **junction** | 182 / 22 | **0.5879** ⛔EXTRAP | 0.3846 | 0.2747 | 0.2143 | **0.1813** | ⛔ PARTIAL |
| **20** | longitudinal | 374 / 24 | 0.0027 | 0.0000 | 0.0000 | 0.0000 | **0.0000** | ✅ **MEASUREMENT** |
| **20** | other | 325 / 24 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** | ✅ MEASUREMENT |
| **60** | overall | 681 / 40 | 0.5066 ⛔EXTRAP | 0.4919 | 0.4875 | 0.4875 | **0.4816** | ⛔ PARTIAL |
| **60** | **junction** | 136 / 19 | **0.9485** | 0.9338 | 0.9338 | 0.9338 | **0.9191** | ⛔ **EXTRAP** |
| **70** | overall | 638 / 40 | 0.5987 | — | — | 0.5674 | **0.5627** | ⛔ **EXTRAP** |
| **70** | **junction** | 125 / 19 | 0.9600 | — | — | 0.9440 | **0.9360** | ⛔ **EXTRAP** |
| **185** | overall | 41 / 40 | 0.9024 | — | — | 0.9024 | **0.9024** | ⛔ **EXTRAP** |
| **185** | **junction** | 6 / 6 | 1.0000 | — | — | 1.0000 | **1.0000** | ⛔ **EXTRAP** |

> ### ⛔ **The pre-registered claim is CONFIRMED, and it is the load-bearing result of this run.**
> At K = 20 the lateral clause **alone** puts **3.75 % of windows** (18.13 % of the junction stratum)
> outside the envelope. `MEASUREMENT` requires **zero**. **Therefore no re-validation of the YAW arm —
> to 30°, to 90°, to infinity — can make K = 20 a measurement.** The same holds at 60, 70 and 185.
> **`GATE_PROTOCOL` §0.3 refuses K ≤ 20, so no admissible gate horizon becomes a measurement on the
> yaw arm at any yaw edge whatsoever.**

**The decomposition, which is why this is not a near miss** (K = 20 overall, windows outside *by clause*):

| yaw edge | via **lat only** | via **yaw only** | via **both** | total |
|---:|---:|---:|---:|---:|
| 12° *(shipped)* | 0.0011 | **0.0851** | 0.0363 | 0.1226 |
| 30° | 0.0295 | 0.0068 | 0.0079 | 0.0443 |
| **∞** | **0.0375** | **0.0000** | 0.0000 | **0.0375** |

At the shipped edge the failure is overwhelmingly **yaw** (0.0851 + 0.0363 vs 0.0011 lateral-only) — the
brief's premise is exactly right. But the residual that survives an infinite yaw envelope is **entirely
lateral**, and it is *irreducible on this axis*. **To zero the lateral clause at K = 20 you would need
`|dlat| ≤ 5.19 m`** (the max peak lateral deviation), on a ground-plane homography P1 itself calls an
**optimistic bound**.

⚠️ **What DOES move, stated as plainly as what does not.** A wider yaw edge is not worthless — it is
just not a rescue:

* **K = 20 overall: 0.1226 → 0.0443 at 30°** — a **64 % reduction** in out-of-envelope windows.
* **K = 20 junction: 0.5879 → 0.2143 at 30°**, and the *class* moves **EXTRAPOLATION → PARTIAL** at
  20° already. The junction stratum stops being a *majority*-outside stratum.
* **K = 20 longitudinal: 0.0027 → 0.0000 at 20°** — this stratum **does** become a clean MEASUREMENT.
* **K = 60 overall** moves EXTRAPOLATION → PARTIAL at 20°.

So a widened yaw envelope changes the **magnitude and the reported class** of the extrapolation at
2 s substantially. It does not change the **kind** of claim we are allowed to make.

---

## 4. C-GEO — the yaw warp's geometry, model-free and exact

`MEASURED` (ours) · `scripts/yaw_geometry.py` → `artifacts/yaw_geometry.json` · uses the **packaged**
`taniteval.clhorizon.sampling_homography` (the function the closed loop actually calls). No model, no
GPU, no renderer, no scene data.

### 4.1 P1's "yaw is exact" claim, verified with numbers rather than quoted

| check | result | verdict |
|---|---|---|
| **A0** — is the yaw homography independent of camera height and road plane? | `max|ΔH| = **0.000e+00**` over **30** `(h_cam ∈ {0.8…10 m}) × (pitch ∈ {−5°…+5°})` conditions | ✅ **CONFIRMED, exactly** |
| **A1** — does it compose like a true rotation, `H(a)H(b) = H(a+b)`? | `max|err| = **5.68e-14**` over 6 pairs (float64 round-off) | ✅ **CONFIRMED** |

**Why A0 holds analytically:** with `dlat = 0` the translation `t = −(R_y @ (0,0,0)) = 0`, so the plane
term `outer(t, n)/d` **vanishes** and `H = K R_y K⁻¹` — no ground plane, no depth, no `h_cam`.

> ⭐ **Consequence: the flat-road assumption that makes the LATERAL envelope an "optimistic bound"
> (P1 §2) plays NO role on the yaw arm.** The yaw warp introduces **zero geometric modelling error**
> for arbitrary 3-D scene structure. Only two mechanisms can degrade it: **finite field of view** and
> **resampling**.

### 4.2 The fabrication curve — the hard information bound

`padding_mode="border"`: any sample falling outside the real frame is **replicated from the edge**.
That content is **fabricated, not observed**. With `f_eff = 266 px`, `W = 256 px`:

| ψ (deg) | 3 | 8 | **12** | 16 | 20 | **25.7** | 30 | 40 | 45 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **fraction of frame fabricated** | 0.074 | 0.184 | **0.264** | 0.339 | 0.413 | **0.512** | 0.588 | 0.770 | 0.869 |
| columns fully fabricated (of 256) | 17 | 44 | **64** | 83 | 102 | **128** | 148 | 196 | 222 |

**FOV: half-angle 25.70°, full horizontal 51.39°** — a geometric constant of the rig, not a chosen number.

> ### ⛔ **The shipped `ENV_YAW_MAX = 12°` already fabricates 26.4 % of the frame — 64 of 256 columns.**
> Whatever 12° was, it was **not** the output of a fidelity criterion: no criterion that looks at image
> content would certify a frame a quarter of which is edge-replication and then call the result a
> "MEASURED envelope limit". This is the geometric restatement of §2.2 — **12° is where the grid
> stopped, and the grid's end became a constant in live code.**
>
> ✅ **C13 gate satisfied:** `frac_fabricated` is monotone 0 → 1, reaches **0.512 at 25.70°** and
> **1.000 at 60°**. The value that makes it fail is stated (≥ 0.5) and the estimator demonstrably
> reaches it. This criterion **can** fail.

---

## 5. C-ADE — P1's own criterion, extended past its grid end

`MEASURED` (ours) · `scripts/lowood_ci_yawext.py` on `tanitad-eval` (A40), stack **@ 0f93b98** — the same
commit P1 ran on · ckpt `/root/models/flagship-30k/ckpt.pt` md5 `b5f07d9e3dd2ca643949bc86832e6585`,
step **29999** · warp geometry **imported verbatim** from `lowood_probe.py` (md5 `a4ea0513…`), never
re-implemented · `episode_cluster_bootstrap` B=2000, paired form vs Δ=0 on the same windows.
Artifacts: `yawext_40ep.json`, `yawext_12ep.json`, `yawext_40ep_perwindow.npz`, `yaw_edge_40ep.json`.

### 5.1 ✅ The pre-registered validity gate: **80 checks against the committed P1 artifact, 0 mismatch**

`scripts/verify_p1_reproduction.py` → `artifacts/p1_reproduction_check.json`. On P1's own 12-episode
deployment, every shared grid point reproduces to 4 dp — ADE, paired Δ, both CI bounds, and the
`separated` flag — including `baseline = 0.4045 [0.3128, 0.5149]`, 265 windows / 12 episodes.
**The extension is therefore a strict continuation of P1's curve, not a lookalike experiment.**
The 40-episode deployment independently reproduces the registry's canonical v1 value
**`0.4271 [0.3675, 0.4871]`, 881 windows / 40 clusters** (`MODEL_REGISTRY.md` §1.2a).

### 5.2 ⚠️ The C13 gate on P1's OWN criterion — and it **PASSES**, so outcome C is ruled out

Pre-registration §1.3: *no C-ADE number is admissible until the destroyed-observation controls show the
criterion has dynamic range.* All four controls, on the 40-episode deployment:

| condition | ADE@2s [cluster-bootstrap CI95] | paired Δ vs Δ=0 | separated? |
|---|---|---|:--:|
| **Δ=0 baseline** | **0.4271** [0.3675, 0.4871] | — | — |
| `dead_noise` — uniform noise, no structure | **0.5713** [0.5064, 0.6389] | +0.1442 [+0.1139, +0.1755] | ✅ |
| `dead_shuffle` — fixed spatial permutation (histogram preserved) | 0.7659 [0.6658, 0.8791] | +0.3388 [+0.2571, +0.4287] | ✅ |
| `dead_black` — zeros | **1.9890** [1.7463, 2.2403] | +1.5619 [+1.2983, +1.8319] | ✅ |
| `yaw = 90°` — ~100 % fabricated | 0.6269 [0.5546, 0.6961] | +0.1998 [+0.1334, +0.2717] | ✅ |

✅ **The criterion can fail, and does, by a factor of 4.7× on `dead_black`.** **Outcome C is refused on
evidence.** ⚠️ **But the useful headroom is much smaller than that suggests**, and this is the number that
governs everything below: the **information floor** — the *lowest* ADE reachable with **no scene
information at all** — is `dead_noise` = **0.5713**. So the entire span from a perfect frame to no frame
is only

> **total information range = 0.5713 − 0.4271 = 0.1442 m**

which is **smaller than the baseline's own 95 % CI width (0.1196)** is large relative to. Any envelope
criterion built on this ADE must be read against **0.1442**, not against zero — which is exactly what
P1's bare "+13.62 % at 12°" failed to do.

### 5.3 ⭐ The extended curve, and the criterion that scales it

**Information Destruction Fraction** `IDF(ψ) = [ADE(ψ) − ADE(0)] / [floor − ADE(0)]`.
`IDF = 0` the warp is free · `IDF = 1` **the warped frame is worth no more than NO frame** ·
`IDF > 1` actively misleading. ⚠️ **C13: IDF is not bounded above — the ψ = 90° control reads 1.385 —
so it can fail and can be SEEN to fail.** Contrast P1's bare ADE, which has no scale and no failing value.

**40 episodes · 881 windows · 40 clusters** *(12-episode P1-matched replication in the last column)*:

| ψ (deg) | ADE@2s | paired Δ [CI95] | sep? | **IDF** | frame fabricated | IDF @12 eps |
|---:|---:|---|:--:|---:|---:|---:|
| 1 | 0.4281 | +0.0010 [−0.0043, +0.0059] | n.s. | 0.007 | 0.027 | −0.003 |
| **2** | 0.4338 | +0.0067 [−0.0008, +0.0143] | **n.s.** | 0.046 | 0.052 | 0.022 |
| **3** | 0.4428 | +0.0156 [+0.0077, +0.0251] | ⛔ **SEP** | 0.108 | 0.074 | 0.106 |
| 5 | 0.4515 | +0.0243 [+0.0116, +0.0386] | SEP | 0.169 | 0.119 | 0.165 |
| 8 | 0.4618 | +0.0347 [+0.0196, +0.0505] | SEP | 0.240 | 0.184 | 0.252 |
| **12** ⬅ *shipped edge* | **0.4771** | **+0.0500 [+0.0341, +0.0678]** | SEP | ⛔ **0.347** | ⛔ **0.264** | 0.353 |
| 14 | 0.4921 | +0.0650 [+0.0482, +0.0816] | SEP | 0.451 | 0.300 | 0.427 |
| 16 | 0.5018 | +0.0747 [+0.0565, +0.0945] | SEP | 0.518 | 0.339 | 0.446 |
| 18 | 0.5264 | +0.0993 [+0.0790, +0.1236] | SEP | 0.688 | 0.374 | 0.610 |
| 20 | 0.5397 | +0.1126 [+0.0875, +0.1419] | SEP | 0.781 | 0.413 | 0.663 |
| 22 | 0.5512 | +0.1240 [+0.0984, +0.1538] | SEP | 0.860 | 0.447 | 0.807 |
| 25 | 0.5673 | +0.1402 [+0.1086, +0.1735] | SEP | 0.972 | 0.501 | 0.993 |
| **28** | 0.5759 | +0.1488 [+0.1160, +0.1815] | SEP | ⛔ **1.032** | 0.554 | 0.962 |
| **30** | 0.5777 | +0.1506 [+0.1155, +0.1858] | SEP | ⛔ **1.044** | 0.588 | 0.889 |
| *90 (control)* | 0.6269 | +0.1998 [+0.1334, +0.2717] | SEP | *1.385* | *1.000* | *1.145* |

**The curve has no knee.** It rises smoothly and monotonically and then **plateaus** — increments over
25→28→30° are +0.0086, +0.0018. *(On the 12-episode set it appeared to turn over; at 3.3× the sample it
does not. That reversal was small-sample noise and is recorded here rather than quietly dropped.)*
A plateau at IDF ≈ 1 is the expected signature of the observation ceasing to contribute at all.

### 5.4 ⭐⭐ THE EDGE, with a cluster-bootstrap interval **on the edge itself**

`scripts/yaw_edge_analysis.py` → `artifacts/yaw_edge_40ep.json`. Each of B=2000 replicates resamples
**episodes** with replacement (reusing `taniteval.ci._draws`, so it is the program's estimator, not a
second one), recomputes the whole curve **and the floor** on that replicate, and locates the crossing by
interpolation. The percentile spread of those crossings **is** the interval.

| edge definition | ψ (deg) | **cluster-bootstrap CI95** | what it means |
|---|---:|---|---|
| **P1's own published reading** — largest ψ with **no** CI-separated degradation | **2 → 3** | grid-limited *(last n.s. 2°, first SEP 3°)* | "indistinguishable from on-path" |
| `IDF = 0.25` — a quarter of usable information gone | **8.36** | **[4.45, 12.42]** | |
| ⭐ `IDF = 0.50` — **half** the usable information gone | **15.47** | ⭐ **[12.14, 17.88]** | the defensible *usable* edge |
| `IDF = 1.00` — **worth no more than no frame at all** | **26.41** | **[18.33, 29.63]** ⚠️ | the hard destruction point |
| *(geometric, model-free)* 50 % of the frame fabricated | **25.70** | exact — a constant of the rig | C-GEO's hard ceiling |

⚠️ **Honest caveat on the `IDF = 1.00` row:** a crossing exists within the 0–30° grid in only **64.1 %**
of bootstrap replicates (in the rest the curve has not reached 1.0 by 30°), so that point estimate is
conditional on an in-grid crossing. The **geometric** ceiling at 25.70° is unconditional and agrees.

> ### ⭐⭐ **Two independent instruments converge on ~26°.**
> The **model-free** ceiling — half the frame fabricated by border replication — is **25.70°**, an exact
> constant of `f_eff = 266 px` / `W = 256 px`. The **model-mediated** information-destruction edge is
> **26.41° [18.33, 29.63]**. One uses no model at all; the other uses the model and a noise control.
> They agree. **The substrate stops carrying usable information at about 26°**, and the mechanism is the
> field of view, exactly as §2.4 predicted from the geometry.

---

## 6. ⭐ What the P1 envelope actually measures — and it is not what its name says

A0/A1 (§4.1) prove the yaw warp is **geometrically exact for arbitrary scene depth**: zero modelling
error, no flat-road assumption. So the substrate's *own* infidelity on this arm is **exactly the
fabricated fraction** — nothing else. Yet the ADE criterion degrades roughly **twice as fast** as
fabrication rises, and hits IDF = 1 precisely where fabrication reaches ½:

| ψ | 12° | 16° | 20° | 25° | 30° |
|---|---:|---:|---:|---:|---:|
| fabricated fraction | 0.264 | 0.339 | 0.413 | 0.501 | 0.588 |
| IDF | 0.347 | 0.518 | 0.781 | 0.972 | 1.044 |

> **The P1 envelope is not a renderer-fidelity envelope. It is a COMPOUND of substrate fabrication and
> our own arm's out-of-distribution sensitivity — measured on flagship v1 — and it is applied to v4's
> closed-loop numbers.** `ood.py` already stamps *"on the flagship v1 arm — NOT on v4"*; this run shows
> **why that stamp matters**: a differently-trained arm would move this envelope, because roughly half
> of what the envelope measures is a property of the *model*, not of the renderer.

⚠️ **The confound named in the pre-registration, and it is not resolvable with this instrument.** ADE
would rise under a *perfect* renderer too, because the warp offsets the observation while the GT and the
speed channel stay the **true** ego — correct behaviour from an offset pose genuinely differs. So IDF
**over**-states substrate infidelity by an unknown amount. Two things keep the conclusion intact: the
fabrication curve is model-free and independently reaches ½ at 25.70°, and P1's **own `pixshift` control**
— reproduced here — shows a literal 32-px column roll (**12.5 % of the frame fabricated**) costs
**nothing** (+0.009, n.s.). So the yaw degradation is **not** a generic "the image moved" artifact.

---

## 7. ⭐⭐ VERDICT: **OUTCOME B**, on both clauses independently

**Pre-registered outcome B: *fidelity degrades at or before 12° → the envelope is correct or too
generous, the closed-loop instrument cannot be rescued by re-validation, and our closed-loop numbers are
extrapolations at every admissible horizon and must be labelled so permanently.***

**That is what the evidence says, and it is over-determined — two independent lines each suffice.**

**(1) The yaw arm cannot be widened materially.** At the shipped 12° the warp has already destroyed
**34.7 %** of the frame's usable information and fabricated **26.4 %** of its pixels. The defensible
*usable* edge — half the information retained — is **15.47° [12.14, 17.88]**, whose **CI lower bound
(12.14°) touches the shipped 12°**. That is a **1.29× widening at the point estimate and no widening at
all at the lower bound.** It is not "materially beyond 12°", and calling it Outcome A would be exactly
the stretch the brief forbids.

**(2) Even an INFINITE yaw envelope makes no admissible horizon a measurement** (§3) — the lateral
clause alone leaves 3.75 % of windows outside at K = 20, and `MEASUREMENT` requires zero.

**What each horizon would need, against what the substrate can deliver** *(requirements `PUBLISHED`,
`HORIZON_ENVELOPE.md` §5.2; edges `MEASURED` here)*:

| horizon | p90 peak &#124;dψ&#124; needed | × the **usable** edge (15.47°) | × the **destruction** edge (26.41°) | reachable? |
|---|---:|---:|---:|---|
| K = 20 (2.0 s) overall | **15.19°** | **0.98×** | 0.58× | ⚠️ only by accepting **half** the information destroyed |
| K = 20 (2.0 s) **junction** | **28.14°** | 1.82× | ⛔ **1.07×** | ⛔ **NO — past the point where the frame is worth nothing** |
| K = 60 (6.0 s) overall | 39.25° | 2.54× | ⛔ **1.49×** | ⛔ **NO** |
| K = 60 (6.0 s) **junction** | 60.03° | 3.88× | ⛔ **2.27×** | ⛔ **NO** |
| K = 70 (7.0 s) overall | 43.32° | 2.80× | ⛔ **1.64×** | ⛔ **NO** |
| K = 185 (18.5 s) overall | 77.67° | 5.02× | ⛔ **2.94×** | ⛔ **NO** |

> ### ⛔ **The plain verdict, as the brief demands it be stated.**
> **NO admissible gate horizon becomes a MEASUREMENT under a re-validated yaw envelope — not K = 20, not
> K = 60, not any of them.** The `junction` stratum at the program's **standing 2 s horizon** has a p90
> heading deviation of **28.14°**, which is **past the point (26.41° [18.33, 29.63], corroborated
> geometrically at 25.70°) where the re-rendered frame carries no more information than pure noise.**
>
> ⭐ **Therefore: our closed-loop numbers are EXTRAPOLATIONS at every admissible horizon, and must be
> labelled so permanently.** This is the pre-registered negative, and it is decision-relevant: it closes
> the last branch `GATE_30K_RESULTS.md` §10.1 left open. **Branch one ("register where the envelope
> holds") was closed by `HORIZON_ENVELOPE.md`. Branch two ("re-validate P1") is closed here.**

### 7.1 What DOES change, stated as plainly as what does not

The re-validation is not worthless — it is just not a rescue, and the difference matters for what the
next gate card may say:

1. ⭐ **A defensible, MEASURED yaw edge now exists where none did before.** `ENV_YAW_MAX` may be restated
   as **15.47° [12.14, 17.88]** at a declared `IDF = 0.5` budget — the first version of this constant
   with a criterion, an interval and an artifact behind it. **This is a change of provenance, not of
   value:** the CI contains 12.0, so the shipped number is *not refuted*, it is finally *justified*.
2. ⛔ **The magnitude of the extrapolation barely moves at 2 s — and this REFUTES my own first estimate.**
   Recomputed at the exact measured edge (`MEASURED`, `yaw_edge_recompute.json`, edges 12.14 / 15.47 /
   17.88 now in the grid):

   | K = 20 stratum | @12° *(shipped)* | **@15.47°** *(measured edge)* | @17.88° *(CI upper)* | verdict @15.47° |
   |---|---:|---:|---:|---|
   | overall | 0.1226 | **0.1067** | 0.0931 | PARTIAL |
   | **junction** | 0.5879 | ⛔ **0.5165** | 0.4505 | ⛔ **still EXTRAPOLATION** |
   | longitudinal | 0.0027 | **0.0000** | 0.0000 | ✅ **MEASUREMENT** |
   | other | 0.0000 | 0.0000 | 0.0000 | ✅ MEASUREMENT |

   ⚠️ **I had written `~0.45`, "moving the junction stratum from a majority to a minority". That was an
   interpolation and it was WRONG.** At the measured edge the junction stratum is **51.65 % outside — still
   a majority, still EXTRAPOLATION.** The overall stratum improves by only **13 %** relative (0.1226 →
   0.1067), not the ~27 % I estimated. *Recorded rather than silently replaced: this is precisely the
   class of error the "measure, don't interpolate" rule exists for, and it made the honest answer worse,
   not better.*
3. **`K = 20 longitudinal` becomes a clean MEASUREMENT** at any edge ≥ 20° (§3) — the only stratum×horizon
   cell that re-validation actually converts.
4. ⛔ **The OOD ratio's arithmetic ceiling barely moves.** C13's `sup(ratio_arr) = 1.298888` is set
   entirely by how far the P1 sweep went. Extending the yaw arm to 30° raises the yaw excess from
   `(0.4596−0.4045)/0.4045 = 0.1362` to `(0.5777−0.4271)/0.4271 = 0.3526` — a new supremum of about
   **1.52** on the 40-ep numbers. That *nominally* clears `RATIO_EXTRAPOLATION_X = 1.5` **by 0.02**, which
   is far too thin to rehabilitate the criterion. ⚠️ **Do not resurrect clause 1 on this.** A threshold
   that sits 1.3 % below a re-measured ceiling is the C13 defect in a new costume; clause 2 remains the
   whole rule.

---

## 8. The lateral arm (priority 4) — measured in the same pass

⚠️ P1 calls the lateral envelope an **optimistic upper bound** (ground-plane homography, models only
road-surface parallax), so unlike yaw it carries genuine geometric modelling error and these numbers
flatter it. **40-episode deployment** (881 windows / 40 clusters), same estimator and same pass:

| Δlat (m) | 0.75 | **1.0** | **1.5** | 2.0 | **3.0** ⬅ *shipped edge* | 4.0 | 5.0 | 6.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| paired Δ | +0.0066 | +0.0125 | +0.0206 | +0.0285 | **+0.0649** | +0.0833 | +0.1255 | +0.1691 |
| separated? | n.s. | **n.s.** | ⛔ **SEP** | SEP | ⛔ **SEP** | SEP | SEP | SEP |
| **IDF** | 0.046 | 0.087 | 0.143 | 0.198 | ⛔ **0.450** | 0.578 | 0.870 | **1.173** |

**The lateral arm behaves like the yaw one:** detection edge between **1.0 m and 1.5 m**; the shipped
3.0 m sits at **IDF 0.450**; total information destruction at ≈ **5.4 m**.

⚠️ **On the better-powered deployment the lateral detection edge TIGHTENS from 2.0–3.0 m to 1.0–1.5 m.**
P1's *"lateral offset up to 2.0 m carries no CI-separated OOD"* was an **n = 12 statement**; at n = 40
clusters, 1.5 m separates. **P1's §1.3 reading was unpowered, not wrong** — the same caution
`MODEL_REGISTRY.md` §1.2a already records about 40-episode "not separated" verdicts, now instantiated on
the envelope itself.

⭐ **So the shipped envelope (3.0 m / 12°) is approximately an iso-`IDF ≈ 0.35–0.45` contour.** That is a
*coherent* envelope — but it was arrived at by accident, since lateral stopped at its **first** separated
point and yaw at its **fourth** (§2.3). **The envelope was right for the wrong reason**, which is worth
recording precisely because it means the next such constant may not be so lucky.

⚠️ And it does not help the horizon question: covering K = 60's overall p90 needs **13.55 m** and its
junction p90 **20.82 m**, against total lateral information destruction at ≈ **5.4 m** — **2.5× and 3.9×
past the point where the frame is worth nothing.** ⭐ **Note the asymmetry that vindicates the brief's
priority:** at K = 20 the *lateral* p90 is **1.74 m → IDF ≈ 0.15**, comfortably inside, while the yaw p90
is 15.19° → IDF ≈ 0.5. **At 2 s the lateral axis is genuinely fine and the yaw axis is not** — the brief's
premise that yaw is the binding arm is confirmed. It is only at K ≥ 60 that lateral becomes the worse of
the two.

---

## 9. ⚖️ Licence — AlpaSim's NuRec/gsplat was never involved

The brief flags NGC-DL-CONTAINER-LICENSE (no derivatives). **No step of this run touched it, and no
derivative of any renderer was created.**

P1's substrate is **not** NuRec. It is a **ground-plane / rotation homography implemented in our own
`lowood_probe.py`** and packaged as `taniteval.clhorizon.sampling_homography`. NuRec appears in P1 only
as an **INHERITED reference scalar** (REF-C's 1.5157 open-loop ADE on NuRec reconstructions), quoted for
contrast and never re-run. This run **imported** the existing homography **verbatim** and **called** it;
it modified no renderer code. ✅ **No licence issue arises, and none was worked around.**

---

## 10. Escalations — raised here, not left in a README

1. ⭐⭐ **`GATE_30K_RESULTS.md` §10.1's second branch is now CLOSED, and the closed-loop surface needs a
   PERMANENT extrapolation stamp.** Both branches are exhausted: no admissible horizon holds the
   envelope (`HORIZON_ENVELOPE.md`), and re-validating the yaw arm cannot reach the horizons we need
   (this run). Every closed-loop corridor number must carry `envelope_verdict: EXTRAPOLATION` with its
   fractions printed — **including at the standing K = 20.** **Owner: whoever writes the next `GATE_*.md`.**
2. ⛔ **`ENV_YAW_MAX = 12.0` and `ENV_LAT_MAX = 3.0` are commented `# MEASURED` in TWO places and the
   comment is wrong** — `taniteval/corridor.py:109-110` and `stack/scripts/run_gate.py:613-614` (kept in
   sync by `test_ood_guard.py`). What was measured is *"the sweep ran to 12° and ADE rose 13.62 %"*; the
   values themselves are **the endpoints of a hardcoded CLI default** (§2.2). Proposed replacement,
   offered rather than applied because both files are live and a sibling stream edited `run_gate.py`
   hours ago:
   > `ENV_YAW_MAX = 12.0   # P1 sweep GRID END (lowood_probe.py --yaw-grid default), not a measured`
   > `                     # limit. MEASURED usable edge = 15.47 deg [12.14, 17.88] at IDF=0.5;`
   > `                     # information fully destroyed at 26.41 deg [18.33, 29.63] ~ FOV/2 = 25.70.`
   **Owner: `taniteval/corridor.py` + `run_gate.py` maintainers.**
3. ⛔ **Do NOT resurrect the OOD ratio criterion on the extended sweep.** Extending the yaw arm to 30°
   raises `sup(ratio_arr)` from 1.298888 to ≈ **1.52**, which clears `RATIO_EXTRAPOLATION_X = 1.5` by
   **0.02** — a 1.3 % margin. Re-enabling clause 1 on that basis would re-create C13 exactly.
   **Recommendation: leave clause 1 VOID permanently and delete the constant rather than re-tune it.**
   **Owner: `taniteval/ood.py`'s author.**
4. ⭐ **The envelope is ~half a MODEL property, so the actionable lever is TRAINING, not rendering**
   (§6). The substrate is geometrically exact to the FOV limit; what collapses by ~26° is *our arm's*
   ability to use an off-path view. **An arm trained with off-path observation augmentation would move
   this envelope**, and that is the only path to a closed loop that is a measurement at junctions.
   This is a v4/v5 training-recipe item, not an eval item. **Owner: PI / the next training brief.**
5. ⚠️ **The envelope is stamped v1 and used on v4.** `ood.py` says so; §6 shows why it is load-bearing
   rather than cosmetic. Either re-measure the envelope on the arm being gated, or state the transfer
   as an assumption in the gate card. **Owner: gate card author.**
6. ⚠️ **`HORIZON_ENVELOPE.md` §5.2's price for re-validation should be updated.** It reads *"extend the
   P1 sweep … and see whether the ADE-vs-offset curve stays smooth or breaks"*. **It stays smooth — there
   is no break** — and that is precisely why it does not help: the degradation is *continuous*, so the
   edge is a **bias-budget choice**, not a discovered physical limit. **Owner: that document's author.**

---

## 11. Deliverable manifest

All repo paths are relative to the working tree on the dev box, all **`git add`-ed, NOT committed, NOT
pushed**. Anything existing in only ONE place is marked ⚠️. Repo dir:
`TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-26-p1-envelope-revalidation/`

| artifact | where it lives | what it is |
|---|---|---|
| `P1_REVALIDATION.md` | repo (this dir) | this report |
| `scripts/yaw_edge_recompute.py` | repo | §3 — out-of-envelope fractions vs yaw edge, from the staged dumps. **No GPU, no pod.** |
| `artifacts/yaw_edge_recompute.json` | repo | §3 — 4 horizons × 4 strata × 13 candidate edges incl. ∞ |
| `scripts/yaw_geometry.py` | repo | §4 — C-GEO. **No GPU, no pod, no data.** |
| `artifacts/yaw_geometry.json` | repo | §4 — depth-independence, exact composition, fabrication curve, FOV |
| `scripts/lowood_ci_yawext.py` | repo · `tanitad-eval:/root/p1yaw/` | §5 — the extended sweep + dead controls (md5 `f374db94…`, verified both ends) |
| `artifacts/yawext_40ep.json` / `.log` | repo · `tanitad-eval:/root/p1yaw/` | §5 — 881 win / 40 clusters, the primary deployment |
| `artifacts/yawext_12ep.json` / `.log` | repo · `tanitad-eval:/root/p1yaw/` | §5.1 — P1-matched deployment, md5-verified on transfer |
| `artifacts/yawext_40ep_perwindow.npz` | repo · `tanitad-eval:/root/p1yaw/` | per-window ADE / along / cross / eid for every condition — **the arithmetic-only path: any edge rule can be re-derived with no GPU** |
| `scripts/verify_p1_reproduction.py` | repo | §5.1 — the 80-check validity gate |
| `artifacts/p1_reproduction_check.json` | repo | §5.1 — 80 checks, 0 mismatch |
| `scripts/yaw_edge_analysis.py` | repo | §5.4 — IDF + the bootstrap interval **on the edge** |
| `artifacts/yaw_edge_40ep.json` | repo | §5.4 — the edges with their CIs |
| ⚠️ `tanitad-eval:/root/p1yaw/smoke.json` | **pod only** | a 2-episode smoke; superseded by the full runs, deliberately not staged |

**Nothing that took real effort exists in only one place.** The one pod-only row is a superseded smoke
test. `lowood_probe.py` was copied to the pod but is **already committed** in
`…/2026-07-23-lower-ood-closedloop-source/` — not re-staged.

**Nothing was committed and nothing was pushed.** ⚠️ The index also contains **7 files from a sibling
stream** (`2026-07-25-tanitdataset-hf-push/`) that were staged before I started; per CLAUDE.md's git
hygiene rule I left them untouched and did not commit.

---

## 12. For `RETRACTION_LOG.md` — with root-cause class

⚠️ **Not appended by me** — `RETRACTION_LOG.md` was edited by sibling streams repeatedly today and
editing a mid-flight append-only log is the hazard this program keeps paying for. Proposed rows:

**Row 1 — the envelope constant.**
*Claim:* `ENV_LAT_MAX = 3.0` / `ENV_YAW_MAX = 12.0` are **"MEASURED envelope limits"** — asserted in
`taniteval/corridor.py:109-110`, mirrored in `run_gate.py:613-614`, propagated through `ood.py`,
`clhorizon.py` and every closed-loop artifact.
*Class:* **NEW — C14: A SWEEP'S GRID END RE-LABELLED AS A MEASURED LIMIT.** *(Sibling of C13: there a
guard could not reach its threshold; here a constant was never a threshold at all. Both produce a
number that looks adjudicated and was not.)*
*Correction:* 12° is the endpoint of `lowood_probe.py --yaw-grid`'s hardcoded default `"0,1,2,3,5,8,12"`.
No criterion selecting an edge exists anywhere in P1. **P1's own report puts the yaw
no-detectable-degradation edge at ≤ 2°** and the first CI-separated point at **3°** — so the shipped
constant sits **four sweep points deep into separated degradation**, while `ENV_LAT_MAX` sits at its
**first** separated point. The two axes were never set by a common criterion.
⚠️ **BINDING LESSON: the largest value you tested is not the largest value that is valid. Before a sweep
endpoint becomes a constant, state the criterion that selects it — and if there is none, the constant's
evidence class is INHERITED, not MEASURED.** ⭐ **And this is C13's root:** `sup(ratio_arr) = 1.298888`
*because* the sweep stopped at 3.0/12; the dead guard and the mislabelled constant are the same defect
seen from two ends. ✅ **What stands:** the *value* is not refuted — the measured usable edge is
**15.47° [12.14, 17.88]**, whose CI contains 12.0. **What is withdrawn is its evidence class**, and with
it the claim that closed-loop numbers inside it are measurements.

**Row 2 — what the envelope measures.**
*Claim:* the P1 envelope is a **renderer-fidelity** envelope — the range over which the re-rendered
observation is faithful.
*Class:* **C6 — A PROPERTY MEASURED ON ONE SUBSYSTEM, ASSERTED ABOUT ANOTHER.** *(The 07-26 sibling:
"redundancy measured on the DATA, asserted about the MODEL".)*
*Correction:* MEASURED here — the yaw warp is **geometrically exact for arbitrary scene depth**
(`max|ΔH| = 0.000e+00` over 30 camera-height × pitch conditions; exact composition to 5.7e-14), so the
substrate's only infidelity is **FOV fabrication**. The ADE criterion degrades ~**2×** faster than
fabrication rises. **Roughly half of what the "renderer envelope" measures is our own arm's OOD
sensitivity**, measured on **v1** and applied to **v4**.
⚠️ **BINDING LESSON: when a metric is model-mediated, it measures the model too. Before calling a curve
a property of the substrate, measure the substrate model-free — here that was closed-form and took no
GPU.** ⭐ **Consequence: the lever is TRAINING, not rendering** (escalation #4).

---

## 13. What was deliberately NOT done, and the honest limits

* ⛔ **pod1, pod2 and pod3 were never contacted.** Training v2corpus / H2 classifier / YouTube harvest
  round 7 respectively. Exactly **one job ran on `tanitad-eval`** at a time; its GPU was idle
  (`nvidia-smi --query-compute-apps` empty) before I started.
* ⛔ **No renderer code was modified** (§9). NuRec/gsplat was never invoked.
* ⛔ **No interval anywhere comes from `overlapping_holdout_se`.**
* ⛔ **`taniteval/corridor.py`, `run_gate.py`, `ood.py`, `GATE_PROTOCOL.md` and `RETRACTION_LOG.md` were
  NOT edited.** All are live and several were touched by sibling streams today; proposed text is in
  §10 and §12 instead.
* ⚠️ **The envelope is measured on flagship v1**, as P1's was. It is applied to v4's closed loop. §6
  shows this is load-bearing, not cosmetic — escalation #5.
* ⚠️ **IDF over-states substrate infidelity** by an unquantified amount (§6): ADE would rise under a
  perfect renderer too. The model-free fabrication curve is the check that keeps the conclusion intact.
* ⚠️ **`IDF = 1.00`'s interval is conditional** on an in-grid crossing (64.1 % of replicates). The grid
  stopped at 30° plus a 90° control; a 35–45° arm would tighten it. The geometric ceiling (25.70°) is
  unconditional and agrees, so this does not change the verdict.
* ⚠️ **The lateral arm is P1's ground-plane homography**, which under-models 3-D parallax — an
  **optimistic** bound. Its IDF numbers are therefore *better* than the truth, and the §8 conclusion
  ("2.7×–4.2× past total destruction for K = 60") is conservative.
* ✅ **§7.1 item 2 is now MEASURED, not estimated** — the measured edges 12.14 / 15.47 / 17.88 / 26.41
  were added to `yaw_edge_recompute.py`'s grid and the fractions recomputed. **The recomputation refuted
  my own interpolation** (junction @15.47° is 0.5165, still a majority, not the ~0.45 I had written).
  The wrong estimate is left in the record with its correction rather than overwritten.
