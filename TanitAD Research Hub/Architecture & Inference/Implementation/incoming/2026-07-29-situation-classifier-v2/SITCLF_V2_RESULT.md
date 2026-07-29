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

⚠️ The camera+ego degradation is worth a follow-up: with ego present, the optimiser has a shortcut
and the vision pathway is under-trained. A fair camera-vs-ego contrast needs ego features that are
**causally available at inference** (e.g. ego state at t only, no future-derived terms).

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

## ⚠️ The ridge stage did not run — and no summary JSON exists

`sc_train_v2.py` exited `rc=1` after all six neural heads finished, at
`ridge_fit_predict`: `ValueError: shape mismatch: value array of shape (26130,3) could not be
broadcast to indexing result of shape (26130,2)`. **Another hardcoded `3`**, in the ridge path this
time — the same class as the `torch.full((3,))` that my earlier sweep missed, because I generalised
`n_out`, `range(3)` and two array shapes but never enumerated *every* literal.

⇒ Consequences: **no `train_summary.json` was written**, so the numbers above come from the log
lines, and the linear-ridge baselines are absent. The six neural heads are the primary result and are
unaffected. Fix is one more `len(SITS)` substitution; it is a **secondary baseline** and was
deliberately not given priority over E-CR when pod3 freed.

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

⚠️ **The heads and label bundles are POD-ONLY.** They are reproducible from the in-repo emitter plus
the parity cache, so this is a rebuild cost rather than a loss — but it is single-disk and should be
pulled or re-pushed if any of it becomes load-bearing.

## Evidence class

| claim | class |
|---|---|
| the CV-AP table | **MEASURED (ours)** — 5-fold CV, best mean CV-AP; **no confidence intervals computed** |
| base rates | **MEASURED** — reproduce exactly through the reshape (independent check) |
| "camera-only is real at 2.1× the null" | **MEASURED**, directional; ⚠️ no interval, so not decision-grade for a close call |
| "ego is optimistic by construction" | **INHERITED from C60** (PI's correction, verified) — not re-measured here |
