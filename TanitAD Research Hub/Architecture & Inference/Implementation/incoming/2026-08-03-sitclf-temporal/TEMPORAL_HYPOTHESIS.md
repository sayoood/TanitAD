# Is the situation classifier limited by MISSING TEMPORAL CONTENT?

**Date** 2026-08-03 · **Stream** sitclf temporal · **Substrate** dev box (RTX 4060), **0 pod GPU-h** —
no pod was touched. **Pre-registration** `./PRE_REGISTRATION.md`, written and staged **before** any
held-out number from this run was read. **Run directory** `TanitAD Research Hub/Architecture &
Inference/Implementation/incoming/2026-08-03-sitclf-temporal/`.

⛔ PI ruling 2026-08-03 honoured throughout: **labels may use ego; INFERENCE IS VISION-ONLY.** Every
deployable arm below reads the frozen v1 camera latents and nothing else. The two `CPOS_*` arms are
**privileged power controls, not deployables**, and are excluded from every ranking.

---

## 0. The headline

**The temporal hypothesis is REFUTED in its actionable form.** Not because the classifier is blind to
motion — it is not — but because **all the motion this task can use is already inside the frozen
trunk's 200 ms 3-frame stack, and nothing added downstream reaches it.**

| # | finding | evidence class |
|---|---|---|
| 1 | **The brief's premise was false.** The sitclf features are not single-instant: the encoder eats a **3-frame stack** (`config.py:17,360`; `refc.py:248`) and the head stacks 8 of them — **0.9 s** of motion-bearing evidence, not 0. | MEASURED, 2 probes |
| 2 | **H-T2 (motion subspace) refuted MECHANISTICALLY, before any AP.** The rank-16 *appearance* basis already retains **88.1 %** of the frame-to-frame difference variance against the purpose-built motion basis's 89.5 % — a **1.4 pp** gap — at mean principal cosine **0.9803**. There is no discarded motion subspace. | MEASURED, `results_subspace.json` |
| 3 | **H-T1 (window length) refuted on the decision-grade situation.** On `intersection` (216 clusters) **not one** of 9 longer-window arms separates above the deployed 0.7 s window; several separate below. **A single latent — 17 parameters, zero history — ties the deployed window at +0.009 [−0.049, +0.054].** | MEASURED, `results_fast.json` |
| 4 | **H-T3 (parameterisation) refuted.** The exactly-invertible remap control moves `lane_change` by **+0.003 [−0.052, +0.057]**, and on `intersection` the appearance+motion arm's deficit (−0.089) is *identical* to the pure reparameterisation penalty (−0.089) — the motion block contributes exactly nothing. | MEASURED |
| 5 | ⭐ **But motion IS load-bearing — it is just already captured.** Deleting the 200 ms inside the encoder's own stack costs **~70 % of the skill** (recovery 0.297 / 0.303 / 0.316 on `intersection`, all separated). So the classifier is **not** an appearance shortcut. | MEASURED, `results_stillframe.json` ⚠️ off-distribution caveat in §4c |
| 6 | ⭐ **The finding nobody was looking for: the two situations have OPPOSITE temporal signatures.** `intersection` skill decays monotonically with the anticipation horizon (+0.982 → +0.378 over 1→5 s, precision-lift 2.713 → 1.794); `lane_change` **rises** with it and separates **only** at 5 s. The programme forces one window and one horizon onto both. | MEASURED, `results_horizon.json` |
| 7 | **The label pipeline's causality fix is landed and tested — and I sized its blast radius**, which nobody had: `alon_pre` moves **4.70 %** of its own scale on **72.4 %** of frames, `omega_pre` **3.24 %** on **50.7 %**, with **every detector and every label bit-identical**. | MEASURED, `causality_blast_radius.json` |
| 8 | ⚠️ **The brief's source citation is wrong.** `refc.py:1112-1117` is the anchor-endpoint prior, not the frame-selection code. The claim is true at **`refc.py:1688, 1691`**. | MEASURED |

**What this redirects.** Stop buying temporal context for the situation head — window, motion
features and motion bases are all measured dead. The two live levers this study exposes are
**(a) per-situation horizons and windows** (finding 6), and **(b) a trunk that learns motion over a
longer span**, i.e. BACKLOG **B5** — because the only place motion has ever helped here is *inside*
the encoder, never downstream of it.

---

## 1. ⭐ THE BRIEF'S PREMISE IS FALSE AS STATED — measured at two probes

The hypothesis was handed to me as: *our models keep only the LAST frame's feature map, so the
cross-attended tokens are single-instant; a single RGB frame carries no relative velocity, no closing
rate, no TTC; that would explain a capacity curve peaking at 129 parameters.*

Two independent facts, both MEASURED here, break that chain **before** any classifier is fitted.

**(a) The per-frame latent is not a single frame.** MEASURED — a real episode-cache tensor is
`frames_u8 [199, 9, 256, 256]` (probe 1: `C:/Users/Admin/tanitad-data/physicalai/_epcache/
physicalai-val-bb543bdf7836/ep_*.pt`), and `stack/tanitad/config.py:17` reads
`in_channels … 9 = camera (3-frame stack, D-015)` with `config.py:360` *"3 RGB frames at 100 ms
spacing channel-stacked"* (probe 2). **Every v1 latent already integrates 0.2 s of motion.**

**(b) The head does not read one latent.** `stack/tanitad/eval/sitclf.py:causal_window` stacks
`WIN = 8` latents at offsets −7..0. **0.7 s of latent history is already in the design matrix**, and
combined with (a) the deployed head already sees **0.9 s of motion-bearing evidence**.

⇒ The situation classifier is **not motion-blind by construction**. Whatever is true of REF-C's
single-instant cross-attention (`stack/tanitad/refs/refc.py:1112-1117`, a sibling stream's read,
INHERITED and not re-verified here) does **not** transfer to this classifier, which has a different
input path. The real questions are narrower, and they are what this study tests:

| id | mechanism | verdict |
|---|---|---|
| **H-T1** | the 0.9 s window is too short for a ~3 s manoeuvre | §4 |
| **H-T2** | rank-16 appearance PCA truncates the motion subspace away | **§2 — REFUTED, mechanistically** |
| **H-T3** | the stack spans the differences but the optimiser cannot find them | §4 + §3 |

---

## 2. ⭐ H-T2 IS REFUTED WITHOUT A SINGLE LABEL — the appearance basis keeps the motion

H-T2 is a claim about **subspaces**, so it is checkable with no classifier, no label and no AP: how
much of the frame-to-frame difference survives projection onto the appearance basis the deployed arm
actually uses? Artifact `results_subspace.json`, script `subspace_diag.py`, fold-0 fit rows
(39,793), same seeds as the main run.

| Δ lag | rank | Δ-variance kept by the **appearance** basis | by the purpose-built **motion** basis | motion advantage | mean principal cos |
|---|---:|---:|---:|---:|---:|
| 0.1 s | 16 | **0.8808** | 0.8952 | **+0.0144** | 0.9803 |
| 0.1 s | 64 | 0.9908 | 0.9931 | +0.0023 | 0.8815 |
| 0.1 s | 256 | 1.0000 | 1.0000 | +0.0000 | 0.9507 |
| 0.3 s | 16 | **0.8936** | 0.9081 | **+0.0145** | 0.9568 |
| 0.3 s | 64 | 0.9952 | 0.9960 | +0.0008 | 0.9102 |
| 0.3 s | 256 | 1.0000 | 1.0000 | +0.0000 | 0.9590 |

**At the deployed rank 16 the appearance basis already retains 88.1 % of the frame-to-frame
difference variance**, against 89.5 % for a basis fitted directly on that difference — an advantage
of **1.4 percentage points** — and the two subspaces have a mean principal cosine of **0.9803**, i.e.
they are very nearly the same subspace. By rank 64 the gap is 0.2 pp and by rank 256 it is zero.

⇒ **There is no discarded motion subspace to recover.** A motion-basis arm has essentially nothing
to find that the appearance basis has not already kept, and H-T2's mechanism does not exist in this
substrate. This is measured, label-free and independent of every modelling choice downstream.

### ⚠️ A defect in this very diagnostic, caught before publication

The first version of the calculation measured the difference's variance **about the appearance
mean** instead of about its own mean. Δ has mean ≈ 0, so subtracting a large appearance mean makes
the total dominated by ‖μ_appearance‖² — a constant offset the appearance basis reproduces almost
perfectly by construction. It reported the appearance basis holding **0.9520** of the Δ variance at
rank 16, which measured the centring and not the subspace.

⭐ **The correction moved the number AGAINST the conclusion** (0.9520 → 0.8808, making the motion
basis look relatively better) **and the conclusion survived it.** The corrected figures are the ones
above.

⛔ **`results_temporal.json` → `controls.H_T2_SUBSPACE_DIAGNOSTIC` contains the SUPERSEDED, wrongly
centred block**, because the hour-long fit had already passed that stage when the defect was found.
It is left untouched rather than edited after the fact; `results_subspace.json` carries a
`_supersedes` pointer, and **`results_subspace.json` is the quotable artifact**. `run_temporal.py`
has been fixed so a re-run is correct.

---

## 3. ⭐ The study reproduces B4's banked row BIT-IDENTICALLY — an unplanned end-to-end validation

`run_horizon.py` rebuilds the situation events from the episode caches, refits the PCA, refits the
ridge and reruns the bootstrap in a **separate process from a separate script**, and at
`lead_s = 3.0` it lands on the banked B4 `ridge_pca16_w8` row exactly:

| situation | this study (`results_horizon.json`, lead 3.0 s) | B4 (`…/2026-08-03-sitclf-matched-capacity/results_matched_capacity.json`) |
|---|---|---|
| `lane_change` | AP 0.02841 · lift **1.269 [1.075, 1.571]** · 1,749 pos | AP 0.02841 · lift **1.269 [1.075, 1.571]** · 1,749 pos |
| `roundabout` | AP 0.03822 · lift **2.619 [1.893, 3.944]** · 1,142 pos | AP 0.03822 · lift **2.619 [1.893, 3.944]** · 1,142 pos |
| `intersection` | AP 0.16607 · lift **1.677 [1.454, 1.996]** · 7,032 pos | AP 0.16607 · lift **1.677 [1.454, 1.996]** · 7,032 pos |

Agreement to 5 decimal places on the point estimate **and both interval bounds**, on all three
situations, is a C-FID-class check that the label rebuild, the fold machinery, the PCA, the ridge
and the estimator in this stream are the same ones that produced the banked table. Every number
below therefore sits on the same footing as B4's.

*(The B4 comparison is a REPRODUCTION, not a shared computation: `run_horizon.py` never reads
`results_matched_capacity.json`, and its own C-FID assertion — rebuilt frame count vs substrate
frame count — must pass before it produces anything.)*

---

## 3b. What the controls establish before any arm is quoted

| control | result | what it licenses |
|---|---|---|
| **NEG_FEAT**, fitted and scored FIRST | AP-lift **0.715 – 1.327** over **57** vision-arm × situation cells with the image features permuted across clips. Per situation: `intersection` **0.956 – 1.042**, `lane_change` 0.899 – 1.327, `roundabout` 0.715 – 1.054 | the protocol does not manufacture signal; 1.0 is chance, and on the decision-grade situation the null sits within ±4.4 % of it. Same band as B4's 0.756–1.331 |
| **NEG_LABEL** (labels permuted across whole clusters) | **0.974 – 1.265** (9 cells) | ditto from the label side |
| ⭐ **PARAMETERISATION INVARIANCE** — `ridge_app16_w8_diffparam`, an exactly invertible remap of the reference's own window | `lane_change` **+0.003 [−0.052, +0.057]**, not separated | **the control behaves exactly as designed**: a pure change of basis moves nothing. Without this, any "explicit motion channels help" reading would be unattributable — and it is what makes H-T3 testable rather than rhetorical |
| ⭐ **C-POS ORACLE** — ego over the FUTURE 3 s, the label's own evidence window ⛔ not a deployable | `lane_change` **+1.098 [+0.610, +1.942] SEPARATED** | the rows CAN be separated. **A null from a vision arm is therefore a fact about vision, not about the instrument** — the distinction the pre-registration makes mandatory |
| **C-POW** — positive clusters counted before any score was read | `intersection` **216** · `lane_change` **55** · `roundabout` **37** | roundabout is **below the bar of 40 ⇒ no verdict**; the other two are eligible |
| **SLOW ≡ FAST** | 260 values compared, **0 differing, max abs diff 0.000e+00** | the shared-draw re-analysis is bit-identical to the per-arm one, so both artifacts are the same measurement |

⚠️ **One control column in this run is degenerate and must not be read.** `build_features` did not
apply the clip permutation to the ego block, so `NEG_FEAT__CPOS_*` was byte-identical to its real
arm and its `paired_vs_own_null` is exactly `+0.000` by construction. It never touched a **vision**
arm's null and never entered the C-POS predicate (which reads `paired_vs_reference`).
`run_temporal.py` is fixed; see `MANIFEST.md` §3.

---

## 4. The ladder — 21 arms, every one against its own permuted-feature null

Full tables: `tables.md` (generated from the JSONs by `render_tables.py`, never retyped).
Ladder numbers: `results_fast.json`, **bit-identical** to `results_temporal.json` where the latter
completed (260 values, max abs diff 0.000e+00).

### 4.1 `intersection` — 216 positive clusters, DECISION-GRADE. MDE 0.323 (widest) / 0.247 (median)

| group | arm | history | flat | params | AP-lift | vs reference |
|---|---|---:|---:|---:|---:|---|
| **A** matched capacity | `ridge_app128_w1` | 0.0 s | 128 | 129 | 1.525 | −0.199 [−0.473, +0.031] |
| | `ridge_app64_w2` | 0.1 s | 128 | 129 | 1.510 | −0.214 **SEP WORSE** |
| | `ridge_app32_w4` | 0.3 s | 128 | 129 | 1.644 | −0.081 [−0.308, +0.112] |
| | **`ridge_app16_w8` ⬅ REF** | 0.7 s | 128 | 129 | **1.724** | — |
| | `ridge_app8_w16` | 1.5 s | 128 | 129 | 1.184 | **−0.541 [−0.868, −0.276] SEP WORSE** |
| | `ridge_app4_w32` | 3.1 s | 128 | 129 | 1.120 | **−0.605 [−0.959, −0.323] SEP WORSE** |
| **B** fixed rank 16 | `ridge_app16_w1` | 0.0 s | 16 | 17 | 1.734 | **+0.009 [−0.049, +0.054]** |
| | `ridge_app16_w4` | 0.3 s | 64 | 65 | 1.746 | +0.022 [−0.013, +0.051] |
| | `ridge_app16_w16` | 1.5 s | 256 | 257 | 1.576 | −0.149 **SEP WORSE** |
| | `ridge_app16_w32` | 3.1 s | 512 | 513 | 1.601 | −0.123 [−0.321, +0.058] |
| **C** motion basis | `ridge_mot16_w8` | 0.7 s | 128 | 129 | 1.063 | **−0.662 SEP WORSE** |
| | `ridge_app8mot8_w8` | 0.7 s | 128 | 129 | 1.202 | **−0.522 SEP WORSE** |
| | `ridge_app16mot16_w8` | 0.7 s | 256 | 257 | 1.635 | −0.089 **SEP WORSE** |
| **D** transformer | `tf_app16_w8_d128` | 0.7 s | 128 | 417,028 | 1.470 | **−0.254 SEP WORSE** |
| | `tf_app16_w32` | 3.1 s | 512 | 420,100 | 1.291 | **−0.434 SEP WORSE** |
| **CTRL** invariance | `ridge_app16_w8_diffparam` | 0.7 s | 128 | 129 | 1.635 | −0.089 [−0.174, −0.037] SEP |

⭐ **NOT ONE longer-window or motion arm separates ABOVE the reference. Nine separate BELOW it.**
And the sharpest row in the whole study: **`ridge_app16_w1` — a SINGLE latent, 17 parameters, zero
frames of history — is statistically indistinguishable from the deployed 0.7 s window,
+0.009 [−0.049, +0.054]**, an interval ~5× tighter than the study's own MDE.

⚠️ **`ridge_app16mot16_w8` (−0.089) and the pure reparameterisation control (−0.089) are the SAME
number.** The motion block therefore contributes *exactly* the reparameterisation penalty and
nothing else — H-T2 and H-T3 refuted in one row.

### 4.2 `lane_change` — 55 positive clusters. MDE 0.449 (widest) / 0.278 (median)

| group | arm | history | AP-lift | vs reference |
|---|---|---:|---:|---|
| **A** matched capacity | `ridge_app128_w1` | 0.0 s | 1.442 | +0.182 [−0.120, +0.679] |
| | **`ridge_app16_w8` ⬅ REF** | 0.7 s | **1.259** | — |
| | `ridge_app8_w16` | 1.5 s | 1.536 | **+0.276 [+0.034, +0.574] SEP** |
| | `ridge_app4_w32` | 3.1 s | 1.599 | **+0.340 [+0.026, +0.702] SEP** |
| **B** fixed rank 16 | `ridge_app16_w16` | 1.5 s | 1.299 | +0.040 [−0.061, +0.162] |
| | `ridge_app16_w32` | 3.1 s | 1.348 | +0.089 [−0.126, +0.429] |
| **C** motion basis | `ridge_mot16_w8` | 0.7 s | 0.987 | **−0.273 SEP WORSE** |
| **D** transformer | `tf_app16_w8_d128` ⬅ **PEAK** | 0.7 s | **1.723** | **+0.463 [+0.162, +0.962] SEP** |
| **CTRL** invariance | `ridge_app16_w8_diffparam` | 0.7 s | 1.262 | +0.003 [−0.052, +0.057] |

The pre-registered CONFIRMED predicate **fires**: two matched-capacity long-window arms separate
above the reference. ⚠️ **But it does not survive its own replication.** Group A trades window
length against PCA rank at fixed capacity, so `app8_w16` and `app4_w32` change **two** things at
once. **Group B holds rank at 16 and moves the window alone — and nothing separates at any window.**
Both separating intervals also *barely* exclude zero (+0.034 and +0.026 lower bounds). And the
largest lever on `lane_change` is not temporal at all: it is the 417 k-parameter **head**, at +0.463.

⇒ On `lane_change` the honest statement is **"window length is not shown to be the operative
variable"**, not "longer windows work".

### 4.3 `roundabout` — 37 positive clusters ⇒ **UNDERPOWERED_C_POW, no verdict** (bar is 40)

### 4.4 ⚠️ The pre-registered verdict on `intersection` fired INDETERMINATE — and why the null still stands

My pre-registration required the C-POS oracle to **separate above the reference**. On `intersection`
it did not: `CPOS_ORACLE_egofuture30` reaches lift 1.746 against the reference's 1.724,
**+0.022 [−0.353, +0.369]**. By my own rule that is `INDETERMINATE_C_POS_FAILED`, and it is reported
as such in `results_fast.json`. ⛔ I have **not** weakened the threshold.

**But the predicate was mis-specified, and I am saying so rather than quietly re-reading it.** The
job of C-POS is to prove that a null comes from the arm and not from a dead instrument. Requiring
the oracle to beat *the reference* tests something else entirely — and on this substrate the oracle
fails it for a substantive reason: the ego channels are **trailing** 0.5 s means, so even reading
(t, t+3 s] they lag the turn's onset. The right predicate was "the oracle separates above **chance**",
and it does: **1.746 [1.547, 1.996]**, lower bound clear of 1.0.

The instrument's power is in any case demonstrated **directly, without the oracle**:

* the reference separates strongly from its own null, **+0.738 SEP**;
* **13 arms separate from the reference** on `intersection`, and the **smallest difference the
  contrast actually resolved was 0.0891** — far below any effect the temporal hypothesis predicts;
* the key null, `ridge_app16_w1` vs the reference, has an interval of **±0.05**.

An instrument that resolves −0.089 would have resolved +0.089. It did not, in 21 arms.
⇒ **This is a well-powered null, not an unpowered one**, and it is reported as *no effect above the
stated MDE* — never as an unbounded refutation, which the parent pre-registration forbids.

---

## 4b. ⭐ THE ANTICIPATION HORIZON — and the finding nobody was looking for

`lead_s = 3.0` has been a frozen constant since the original pre-registration and had **never been
characterised**. Artifact `results_horizon.json`, script `run_horizon.py`, reference recipe pinned
(appearance PCA-16, WIN 8, 129 params/head), PCA basis fitted once per fold and reused at every
horizon so the only thing moving is the label's lead.

⛔ **No constant is re-selected.** Every horizon is reported, none is chosen, the detectors and
their frozen thresholds are untouched, and the deployed value stays 3.0 (pre-registration §5b).

⚠️ **The row set and the base rate both move with the lead** (a longer lead creates more positive
frames and masks more end-of-clip rows), so `precision_lift` = P@5 % ÷ base rate is reported beside
AP-lift: it is normalised at the operating point and does not inherit AP-lift's `1/base` ceiling.
**Both move the same way, which is what makes the reading safe.**

### `intersection` — 230 positive clusters, decision-grade — **HORIZON-LIMITED**

| lead | rows | pos | base | AP | AP-lift | P@5 % | fires / true | **precision lift** | Δ vs null |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 s | 79,930 | 2,455 | 0.0307 | 0.06242 | **2.032** | 0.0833 | 3,996 / 2,455 | **2.713** | **+0.982** SEP |
| 2 s | 75,406 | 4,832 | 0.0641 | 0.11445 | 1.786 | 0.1546 | 3,770 / 4,832 | 2.413 | +0.785 SEP |
| 3 s ⬅ deployed | 71,010 | 7,032 | 0.0990 | 0.16607 | 1.677 | 0.2490 | 3,550 / 7,032 | 2.515 | +0.685 SEP |
| 4 s | 66,645 | 8,996 | 0.1350 | 0.20460 | 1.516 | 0.2875 | 3,332 / 8,996 | 2.130 | +0.530 SEP |
| 5 s | 62,277 | 10,678 | 0.1715 | 0.23350 | **1.362** | 0.3076 | 3,114 / 10,678 | **1.794** | **+0.378** SEP |

Skill over the permuted-feature null falls **monotonically, +0.982 → +0.378 (2.60×)**, and the
base-rate-normalised precision lift falls with it, **2.713 → 1.794**. Two differently-constructed
metrics agreeing in direction is what rules out the base-rate artefact I was worried about.

### `lane_change` — 64 positive clusters — **HORIZON-FLAT, and it leans the OTHER WAY**

| lead | pos | base | AP-lift | **precision lift** | Δ vs null |
|---:|---:|---:|---:|---:|---|
| 1 s | 633 | 0.0072 | 1.195 | 1.169 | −0.136 |
| 3 s ⬅ deployed | 1,749 | 0.0224 | 1.269 | 0.812 | +0.084 |
| 5 s | 2,692 | 0.0392 | **1.408** | 1.174 | **+0.436** SEP |

The **only** lead at which `lane_change` separates from its own null at all is **5 s** — the longest
one tested. Its skill *rises* with the horizon, the exact opposite of `intersection`.

### `roundabout` — 39 positive clusters ⇒ **UNDERPOWERED_C_POW at every lead, no verdict**

Its lifts are large (9.919 at 1 s) and every lead separates, but 39 clusters is **below the
pre-registered bar of 40** and the parent study's own roundabout label control fails. ⛔ **No
roundabout number here decides anything**, and it is reported only so its absence is not mistaken
for an oversight.

### ⭐ THE SYNTHESIS — the two situations have OPPOSITE temporal signatures

| | `lane_change` | `intersection` |
|---|---|---|
| longer **lead** | **helps** (only 5 s separates) | **hurts**, monotonically (+0.982 → +0.378) |
| what it is | a 4 s ego manoeuvre (`LC_W_S = 4.0`) whose precursors build up over seconds | a scene-recognition problem — the junction is either visible now or it is not |

**The programme has been forcing ONE window and ONE horizon onto two phenomena with opposite
timescales.** That is a concrete, mechanistic reason why "one head for three situations" underperforms,
and it is independent of encoder capacity, head capacity and feature richness.

---

## 4c. ⭐ THE APPEARANCE-SHORTCUT EXPOSURE — the still-frame control

**Raised mid-flight by the coordinator**, from the latent-bottleneck stream's OUTCOME V: a single
32×32 grayscale **still frame** reads ego `speed` at **93 % of the 800 ms learned latent** and
**1.75× the best motion-only arm**; all ten linear pure-difference arms sit at the null. Since the
situation labels are pure functions of the ego pose track, a "vision-only" situation classifier may
be riding **appearance → speed → the ego-derived label** rather than perceiving the situation. That
is the indirect form of the PI's binding leak test.

### Am I exposed? — **YES, and I can measure it rather than guess**

| what I had already measured | what it says about the exposure |
|---|---|
| `ridge_app16_w1` (a **single** latent, 17 params, **no** multi-frame window) vs the 0.7 s reference: `intersection` **+0.009 [−0.049, +0.054]**, `lane_change` +0.030 [−0.020, +0.079] | one latent is **statistically indistinguishable** from eight. Whatever the classifier reads, it is available in a **single instant** |
| every motion-basis arm | separated **WORSE** on `intersection`, never better anywhere |
| the H-T2 subspace diagnostic | the appearance basis already retains 88 % of the Δ variance |

⇒ Three independent lines already point the same way, and they are **exactly what the coordinator's
finding predicts**. ⚠️ But `win=1` is **not** a clean still-frame control: that single latent is
itself encoded from a D-015 **3-frame stack**, so it still carries 200 ms of motion. To close the
gap I built the control that removes the last of it.

### The control: same encoder, motion deleted from the input

`build_stillframe_substrate.py` re-encodes all 500 clips with the 9-channel input replaced by the
**last RGB frame replicated three times** — identical shape, dtype and preprocessing, zero
inter-frame motion. VERIFIED at build time: the three slots are bit-identical, the last frame is
preserved exactly, and the *real* stack's mean |rgb(t−2) − rgb(t)| is **0.02642**, i.e. the motion
that was removed was really there. Rows, labels, clip ids and folds are identical by construction
(asserted as C-FID), so the two substrates are comparable with the **paired** episode-cluster
bootstrap.

The reported statistic is **recovery** = (still_lift − still_null) / (real_lift − real_null): skill
over each substrate's **own** permuted-feature null, so the two are on one scale.

### The result — the classifier is **NOT** an appearance shortcut

| situation | arm | real skill over null | still-frame skill | **recovery** | still − real |
|---|---|---:|---:|---:|---|
| `intersection` | `ridge_app16_w8` | +0.685 | +0.204 | **0.297** | −0.490 [−0.731, −0.312] **SEP** |
| `intersection` | `ridge_app16_w1` | +0.687 | +0.209 | **0.303** | −0.491 [−0.707, −0.307] **SEP** |
| `intersection` | `tf_app16_w8_d128` | +0.443 | +0.140 | **0.316** | −0.292 [−0.489, −0.124] **SEP** |
| `lane_change` | `ridge_app16_w8` | +0.084 | +0.034 | 0.405 | −0.304 [−0.459, −0.141] **SEP** |
| `lane_change` | `tf_app16_w8_d128` | +0.326 | +0.104 | 0.318 | −0.434 [−0.768, −0.132] **SEP** |
| `roundabout` ⛔ unpowered | `ridge_app16_w8` | +1.863 | +0.966 | 0.518 | −0.820 **SEP** |

**Deleting the 200 ms of motion inside the encoder's own 3-frame stack costs ~70 % of the skill**,
consistently across three arms and all three situations, every contrast separated. So on the
question the coordinator asked: **no, the sitclf numbers are not a pure appearance shortcut** —
appearance alone retains roughly 30 % of the skill, not most of it.

⚠️ **The honest caveat, and it cuts against my own conclusion.** The frozen v1 trunk was trained
**only** on real 3-frame stacks, so a stack of three identical frames is **off-distribution** for it.
The measured drop therefore conflates "motion removed" with "input off-distribution" and is an
**UPPER bound** on motion's contribution. The true appearance-only share is ≥ 30 %. A cleaner
control (a within-stack temporal shuffle, or a trunk retrained on still stacks) is a follow-up, not
something this substrate can settle.

⚠️ `lane_change / ridge_app16_w1` shows `recovery` 7.11 in the raw JSON. It is **uninterpretable** —
the denominator (real skill +0.004) is essentially zero. `render_tables.py` suppresses it; it must
not be quoted.

### ⭐ WHAT THIS COMBINES TO — the useful motion window is ~200 ms wide, and we already have it

| direction | evidence | result |
|---|---|---|
| **remove** the encoder's 200 ms motion | still-frame control | skill falls to **~30 %** — motion is load-bearing |
| **add** motion downstream — window 0.7 s → 3.1 s | Groups A and B, 9 arms | nothing separates above the reference; several separate **below** |
| **add** motion downstream — explicit motion bases | Group C, 4 arms | separated **WORSE** everywhere it is powered |
| **add** motion downstream — one latent vs eight | `ridge_app16_w1` | **+0.009 [−0.049, +0.054]** — indistinguishable |

⇒ **All of the motion this task can use is already inside the frozen trunk's 3-frame stack, and
nothing bolted on downstream adds to it.** That is the actionable form of the temporal hypothesis,
and it is refuted.

---

## 5. ⚠️ The brief's citation is wrong — and the underlying claim is true at other lines

The brief attributes "our models keep only the LAST frame's feature map" to
`stack/tanitad/refs/refc.py:1112-1117`. **Those lines say no such thing.** MEASURED by reading them:
they are the body of `_lan_anchor_prior`, computing a z-scored anchor endpoint —
`end_x = self.anchors[..., -1, 0]`, `z = (end_x - end_x.mean()) / end_x.std()` — and the `-1` there
indexes the last **waypoint of an anchor**, not the last frame of a sequence.

⭐ **The claim itself is nevertheless TRUE, and verifiable at two independent locations:**

* **implementation** — `refc.py:1688` `fmap = fmap_all.reshape(b, w, *fmap_all.shape[1:])[:, -1]`
  and `refc.py:1691` `fmap, pooled = self.encoder(frames[:, -1])`;
* **documentation** — `refc.py:722` *"REF-C is structurally single-instant: `RefCModel.forward`
  cross-attends the LAST frame's feature map only"*, echoed at `:705`, `:1013`, `:1506`, `:1545`.

The correction matters because a wrong line reference is how an INHERITED claim survives audit
without ever being checked — the exact failure class `RETRACTION_LOG.md` exists to log. **Anyone
quoting this should cite `refc.py:1688,1691`.**

### ⛔ THE BOUNDARY THAT MUST TRAVEL WITH THIS RESULT

REF-C and the situation classifier **do not share an input path**:

| | REF-C | situation classifier |
|---|---|---|
| frames reaching the encoder | **last frame only** (`refc.py:1691`) | a **3-frame stack**, 100 ms spacing (`config.py:17,360`) |
| latents reaching the head | **one** | **8**, offsets −7..0 (`sitclf.causal_window`) |
| motion-bearing span | **0 s** | **0.9 s** |

⇒ Whatever this study concludes about the situation classifier **does not transfer to REF-C**, whose
S6 arm is registered at `refc.py:726-727` as *"conditional on the sibling temporal-feature stream"*.
A null here must **not** be read as cancelling that arm: REF-C really is single-instant, this
classifier never was, and they need separate evidence.

---

## 6. The label pipeline's causality break — verified, and its blast radius SIZED

**Status: already fixed, by a sibling stream, earlier the same day. I verified rather than redid it.**

`stack/tanitad/data/situations.py` built `alon_pre` / `omega_pre` as a trailing mean of
`np.gradient` — a **centred** difference — under a comment reading `STRICTLY CAUSAL`, so both
channels read one frame (0.1 s) past `t` on every interior frame. The fix (`backward_diff`, with
`causal_pre=True` the default and `causal_pre=False` reproducing the legacy channels bit-for-bit)
is at HEAD, and `stack/tests/test_label_causality_and_nav.py` covers it —
`test_backward_diff_is_strictly_causal`, `test_causal_pre_is_the_default_and_legacy_is_reproducible`
and a detector-channel invariance test. **VERIFIED BY RUNNING: 18 passed in 8.59 s.**

⭐ **What was still missing was a NUMBER.** The module's blast-radius note names the consumers but
never says how far the leaky channels actually are from the causal ones — and "a defect exists"
versus "the defect is 4.7 % of the channel" license very different decisions about rebuilding banked
artifacts. MEASURED here over **100 val clips / 19,900 frames**
(`causality_blast_radius.py` → `causality_blast_radius.json`):

| channel | mean abs change | p99 | max | **relative to the channel's own scale** | **frames changed > 1 % of scale** |
|---|---:|---:|---:|---:|---:|
| `alon_pre` (m/s²) | 0.0317 | 0.1884 | 0.4618 | **4.70 %** | **72.4 %** |
| `omega_pre` (rad/s) | 0.00286 | 0.0200 | 0.0593 | **3.24 %** | **50.7 %** |

**`LABEL_SIDE_IDENTICAL: true`** — `omega`, `kappa`, `alon` and **all three detectors** return
bit-identical events under both modes (12 lane changes / 7 roundabouts / 49 intersections over those
clips). So the fix could not have silently re-derived a single situation label, which would have
retro-fitted a pre-registered study. That is the load-bearing invariant and it now has a test *and*
a measurement behind it.

### ⚠️ One banked artifact is still on the LEAKY channels

MEASURED by rebuilding clip 0's ego block both ways and comparing against the bank:
`C:/Users/Admin/tanitad-data/eval/sitclf_b4_substrate.npz`'s `E` block matches
`causal_pre=False` **exactly** (max abs diff 0.000e+00) and differs from the causal version by
7.17e-2. It was built before the fix landed.

Consequences, stated precisely:

* ⛔ **No deployable arm is affected** — ego is not a legal inference input, so no arm in this study
  or in B4 reads `E`.
* ⚠️ **`regime_strata` does** — the LONGITUDINAL/LATERAL family strata in `four_family_report` are
  defined by `[v, alon_pre, omega_pre]`, so those stratum boundaries are drawn with a channel that
  peeks 0.1 s ahead. A **stratification** variable is not a model input and a paired within-stratum
  contrast stays valid, but the boundaries are not exactly the causal ones and that is disclosed
  rather than assumed away.
* ⚠️ My two `CPOS_ego_*` power controls read `E` and therefore inherit the same 0.1 s peek. It makes
  them, if anything, slightly **stronger** than a causal ego arm — which is conservative for a
  control whose job is to prove the rows are separable.

---

## 7. Manifest and status

See `MANIFEST.md`.
