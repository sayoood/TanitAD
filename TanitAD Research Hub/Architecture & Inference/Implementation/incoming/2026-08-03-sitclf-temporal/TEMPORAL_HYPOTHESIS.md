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

*(filled from `results_temporal.json` — see §4)*

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

*(filled from `results_temporal.json` / `results_fast.json` — see §4)*

---

## 4. The ladder

*(filled from `results_temporal.json` via `render_tables.py` — see `tables.md`)*

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
