# Situation classifier v2 — camera-only anticipation is REAL at 2.1× the null

**MEASURED 2026-07-29 on pod3.** The PI's directive: *"the classifier should detect the situation of
lane change not necessarily based on objects — only on situational labels… you need the detection of
lane changes and an intersection label"*, and *"at inference the ego does not know the future, so the
main scene information is the front camera."*

**Labels:** emitted by `stack/scripts/emit_situation_labels.py` from the promoted
`stack/tanitad/data/situations.py` over the **canonical parity corpus**
`physicalai-train-e438721ae894` (2,376 episodes, 472,627 frames), `lead_s = 3.0`.
Roundabout **computed but not emitted** — the PI deferred it ("skip roundabouts for later"), and the
26 held-out clusters were unpowered anyway.

| situation | positives | scorable | **base rate** |
|---|---|---|---|
| lane_change | 7,120 | 389,663 | **0.01726** |
| intersection | 11,006 | 385,100 | **0.02816** |

## Result — 5-fold CV, best mean CV-AP per head

| head | inputs | **best CV-AP** | ÷ shuffle null |
|---|---|---|---|
| **head_priv** | privileged (upper bound) | **0.23765** | **10.2×** |
| **head_ego** | ego kinematics only | **0.08858** | **3.8×** |
| head_img_ego | camera + ego | 0.07929 | 3.4× |
| head_img_ego_concat | camera ⊕ ego | 0.05272 | 2.3× |
| ⭐ **head_img** | **CAMERA ONLY** | **0.04869** | **2.1×** |
| **head_img_shuf** | camera, **shuffled** (control) | **0.02342** | — |

Per-config detail: `head_priv` pw20 0.23765 / pw50 0.20155 · `head_ego` pw20 0.08858 / pw50 0.08768 ·
`head_img_ego` pw50 0.07929 / pw20 0.06879 / r64 0.06271 · `head_img_ego_concat` pw50 0.05272 /
pw20 0.04983 / r64 0.04983 · `head_img` pw20 0.04869 / pw50 0.04825 / r64 0.04694 ·
`head_img_shuf` pw50 0.02342 / pw20 0.02294 / r64 0.02251.

## Reading it

⭐ **The shuffle control lands at 0.02342 against a pooled base rate of ~0.0227** — i.e. the null
behaves exactly like a null. That is what makes the rest of the column interpretable, and it is the
reason the control was run at all.

⭐ **Camera-only anticipation is REAL: 0.04869 ≈ 2.1× the shuffle null.** The front camera alone
carries genuine anticipatory signal for lane changes and intersections at a 3 s lead. This is the
claim the PI asked for and it holds.

⚠️ **Ego-only still wins (0.08858), and adding camera to ego HURTS (0.08858 → 0.07929).**
**Per C60 the ego baseline is optimistic BY CONSTRUCTION on human-driven logs**: the logged ego trace
already encodes the human driver's *already-executed* reaction to the situation, which is not
available in that form at inference. So `head_ego` is **not** a deployable baseline — it is an upper
bound contaminated by hindsight. ⛔ **Do not conclude "ego beats camera, so drop the camera."**

⚠️ The camera+ego degradation is worth a follow-up. **See the ridge section below** — under a linear
probe the camera nearly matches ego (0.836 of it) where the neural head puts it at 0.549, which bears
directly on whether this is an optimisation shortcut or a representation limit. **Both readings
remain open; neither is established.** A fair camera-vs-ego contrast also needs ego features that are
**causally available at inference** (ego state at t only, no future-derived terms).

## ⛔ Scope limits

- **Ego kinematics are barred as a model INPUT** by the PI's rule. `head_ego` / `head_img_ego` /
  `head_priv` exist as **references**, not as candidate deployables. The deployable arm is `head_img`.
- Future ego motion **is** legitimate for GENERATING and VALIDATING the labels — that is ground
  truth, not an input. This distinction is written into the emitter's docstring.
- **Two situations, not three.** The trainer was restricted to `SITS = ("lane_change",
  "intersection")` rather than carrying gen-1's roundabout arrays through, which would have mixed two
  label generations inside one bundle.
- ⛔ **NOT comparable to the gen-1 numbers** (`ridge_img` 0.0352 / `ridge_ego` 0.0404 / shuffle
  0.0166). Gen-1 scored **three** situations with a **different detector generation**. Compare within
  this table only.

## ⭐ THE RIDGE BASELINES — and they change the reading

**Re-run completed `rc=0` at 03:33 UTC** after the width fix (below). Closed-form ridge on ±1
targets, standardised on train rows — *"no optimiser to blame."*

| ridge | params | **CV-AP** | ÷ ridge null |
|---|---|---|---|
| ridge_img_ego | 153 | **0.06279** | 3.13× |
| **ridge_ego** | 25 | **0.05408** | 2.70× |
| ⭐ **ridge_img** (camera only) | 129 | **0.04522** | **2.26×** |
| ridge_img_shuf (control) | 129 | **0.02005** | — |

⭐⭐ **THE EGO ADVANTAGE IS LARGELY A CAPACITY ADVANTAGE, NOT AN INFORMATION ADVANTAGE.**

| regime | camera | ego | **camera / ego** |
|---|---|---|---|
| **linear (ridge)** | 0.04522 | 0.05408 | **0.836** |
| **neural (attn head)** | 0.04869 | 0.08858 | **0.549** |

Under a **linear** probe the two channels carry **comparable** predictive information (2.26× vs
2.70× their null). The large ego lead appears **only when the model has capacity to exploit it** —
ego gains **+64 %** going linear → neural (0.05408 → 0.08858) while camera gains **+8 %**
(0.04522 → 0.04869).

This **supports the shortcut hypothesis** flagged below: with ego present the optimiser has an easy
route and the vision pathway is under-trained. It is *not* that the camera lacks the information.

⚠️ **HYPOTHESIS, not established.** An equally consistent reading is that our image features
(frozen, PCA-reduced to r=16/64) are simply harder for a small attention head to exploit than a 25-
parameter ego vector — a *representation* limit rather than an optimisation shortcut. Discriminating
them needs a camera-only head with matched capacity/compute, or an ego-dropout schedule. **Neither
was run.** Do not cite "the vision pathway is under-trained" as a finding.

⭐ Independent confirmation worth noting: **camera-only clears its null in a completely different
model class** — 2.26× under closed-form linear ridge, 2.1× under the attention head. The camera
result does not depend on the optimiser.

## ⚠️ The ridge stage originally failed — a FIFTH hardcoded situation count

The FIRST pass exited `rc=1` after all six neural heads finished, at
`ridge_fit_predict`: `ValueError: shape mismatch: value array of shape (26130,3) could not be
broadcast to indexing result of shape (26130,2)`. **Another hardcoded `3`**, in the ridge path this
time — the same class as the `torch.full((3,))` that my earlier sweep missed, because I generalised
`n_out`, `range(3)` and two array shapes but never enumerated *every* literal.

✅ **FIXED and re-run 2026-07-29 03:05–03:33 UTC (`rc=0`)** — the ridge table above is from that pass.

⚠️ **THE PATTERN MATTERS MORE THAN THE FIX.** This was the **FIFTH distinct spelling** of the same
hardcoded situation count, each found only after the previous fix:
`n_out=3` · `range(3)` · `(N,3)` array shapes · `torch.full((3,))` · `torch.zeros(N, 3)`.
The last one survived every sweep because it is a **two-argument** form, not a tuple. Enumerating
spellings does not converge — each sweep only covers what you have already seen. `sc_train_v2.py`
now carries a **width assertion** (`out/ytr/vtr` must all equal `len(SITS)`) whose message names all
five, so the next reader changes the strategy rather than fixing a sixth. A full-spelling sweep of
the file now returns nothing.

⚠️ **No `train_summary.json` is written even on success** — all numbers here come from the log
lines, banked verbatim as `cv_log_lines.txt`.

## Deliverable manifest

| artifact | where |
|---|---|
| labels (wide, per-clip) | `pod3:/workspace/sitclf/bundle_v2m/sc_labels.npz` + `sc_meta.json` |
| labels (long, as emitted) | `pod3:/workspace/sitclf/bundle_v2/sc_labels.npz` + `sc_labels_summary.json` |
| 6 trained heads | `pod3:/workspace/sitclf/run_v2/head_{priv,ego,img,img_ego,img_ego_concat,img_shuf}.pt` |
| trainer (2-situation) | `pod3:/workspace/sitclf/scripts/sc_train_v2.py` |
| reshape/merge | `pod3:/workspace/sitclf/merge_v2.py` |
| log | `pod3:/workspace/sitclf/v2b_train.log` |
| emitter + detectors (IN REPO) | `stack/scripts/emit_situation_labels.py`, `stack/tanitad/data/situations.py`, `stack/tests/test_situations.py` |

✅ **PIPELINE FULLY BANKED 2026-07-29** — `gen1_sc_features.py` (114 lines, the extractor that
produced the 2,377 clip feature files everything downstream reads) and `gen1_sc_train.py` (396, the
3-situation ancestor that produced the gen-1 reference numbers) are now here too, AST-verified on
transfer. Nothing in the chain is single-disk except the DERIVED artifacts below.

⚠️ **The trained heads and label bundles remain POD-ONLY.** They are reproducible from the in-repo emitter plus
the parity cache, so this is a rebuild cost rather than a loss — but it is single-disk and should be
pulled or re-pushed if any of it becomes load-bearing.

## Evidence class

| claim | class |
|---|---|
| the CV-AP table | **MEASURED (ours)** — 5-fold CV, best mean CV-AP; **no confidence intervals computed** |
| base rates | **MEASURED** — reproduce exactly through the reshape (independent check) |
| "camera-only is real at 2.1× the null" | **MEASURED**, directional; ⚠️ no interval, so not decision-grade for a close call |
| "ego is optimistic by construction" | **INHERITED from C60** (PI's correction, verified) — not re-measured here |
