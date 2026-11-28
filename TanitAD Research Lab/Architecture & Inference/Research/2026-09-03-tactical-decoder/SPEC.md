# SPEC — the refav1 tactical-decoder diagnosis panel

**Date:** 2026-09-03 · **FlyWheel:** Architecture & Inference · **Tier:** T0 ·
**Compute:** dev-box RTX 4060, 2 × 140 forward passes. ⛔ No pod, no Thor, no training, no model edit.

> ⚠️ **HONEST ORDERING.** `TanitAD_ValidateAIDesign` §1 requires a SPEC *before compute*. This
> package is a **DIAGNOSIS**, not a design arm: it re-reads artifacts already on disk and spends
> ~6 GPU-minutes of forward passes. The SPEC-before-compute obligation is discharged for the FIX by
> `Project Steering/PREREG_TACTICAL_DECODER.md`, which is written before any arm. This file records
> the diagnosis panel's own contract so its numbers are auditable, and it was written **after** the
> panel ran. Saying so here rather than back-dating it is the point.

---

## What is being measured, and why each panel exists

| panel | question it answers | tool | GPU |
|---|---|---|---|
| **P1 decode audit** | (e) the class-recall table on both banked checkpoints; (d) does the decision reach the goal | `tools/decode_audit.py` | 0 |
| **P2 label census** | (c) what distribution is the head fit to | `tools/decode_audit.py` (`label_census_*`) | 0 |
| **P3 window/band census** | how many rows ever carry a lateral gradient, train shape vs eval shape | `tools/window_band_census.py` | 0 |
| **P4 intent / logit probe** | (b) the decoder's real share of the loss; RANK vs ARGMAX; head geometry; the latent | `tools/intent_logit_probe.py` | 2 × 140 fwd |
| **P5 turn decomposition** | on a curved goal, does the cost prefer the turn — and what charges it | `tools/turn_decomposition.py` | 0 |
| **P6 seed/goal identity** | is the control the planner seeds the control the goal was rolled from | `tools/seed_goal_mismatch.py` | 0 |

## Splits

```yaml
splits:
  fit:  none — no parameter is fit on the banked checkpoints
  val:  none
  test: the 140 banked windows / 20 episodes (stride 10) that both banked T1 reads used,
        and the 147-clip v7.2 eval label set. Scored, never tuned on.
```
The one panel that fits parameters (P4's linear probe) fits them **leave-one-EPISODE-out**, with the
PCA basis and the standardiser fit on the FIT fold ONLY.

## Controls — every panel carries them

| control | must read | where |
|---|---|---|
| **constant-only** (majority-class predictor) | accuracy = the base rate **EXACTLY**; macro-recall = 1/K **EXACTLY** | `decode_audit.py::constant_only` — **raises `SystemExit` if it does not** |
| **shuffled-label**, TWO forms | `free` (row-wise) and `clustered` (episode blocks permuted whole). ⚠️ **clustered is quoted** — the v7.2 label is per-clip, so a free permutation is anti-conservative | `decode_audit.py::shuffled_label_null` |
| **nav conditions** | every decoder number under `nav_true`, `nav_shuffled`, `nav_zero` (the C6 / nav-echo obligation, `refa_v1.py:395`) | P1, P4 |
| **nav-only ceiling** | the accuracy a predictor reading NOTHING but `nav_cmd` attains on the same rows | P4 |
| **no-information value** | printed beside every AUC (0.500) and every recall | P1, P4 |
| **printed n and d** | `n ≪ d` named as underpowered BY CONSTRUCTION, not reported as a negative | P4 |
| **ULP** | every fp32 cost difference carries the ULP it was taken against and a `resolvable_in_fp32` flag | P5 |
| **deliberate regression** | ⚠️ **NOT APPLICABLE to a diagnosis** — no gate is being claimed. It is required, and specified, for every arm in `PREREG_TACTICAL_DECODER.md` §4 (`C-REG`, `D-REG`) | — |

## Estimator

Episode-cluster bootstrap, percentile interval, **full-set** point estimate (the construction in
`taniteval/taniteval/ci.py::episode_cluster_bootstrap`), 10 000 draws over the 20 episodes.
`overlapping_holdout_se` is never called — it biases the point estimate as well as the interval.

## Verify by CONTENT, not exit code

* the vocabulary is imported from `tanitad.models.v6` at run time and the tool **refuses** on any
  drift from its own hard-copy (`decode_audit.py::_check_vocab`);
* the constant-only control **raises** unless it reads the base rate and 1/K exactly;
* `intent_logit_probe.py` reuses `refav1_arm.load_model` / `build_loader` / the stride-10 window
  selection, so its 140 rows are the SAME 140 windows as the banked dumps — asserted by the loader's
  own `v0` and `nav_cmd` drift checks;
* every artifact under `raw/` is non-trivial and byte-sized in the manifest.

## Refusals recorded

* ⛔ Thor and every pod were **not contacted** (refav1's speed epoch runs to ≈ 2026-09-04 02:00Z).
* ⛔ No model or trainer file was edited. This package is read-only on `stack/`.
* ⛔ The retired `participation ≥ 8.56` floor is not used, quoted, or reintroduced anywhere.
* The training-stream loss ratio is left **UNVERIFIED** rather than estimated from a proxy log.
