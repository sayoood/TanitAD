# RESULT — P-RC21 ANCHOR SWEEP: the trust region works; the REWARD is the limit

**Date:** 2026-08-29 · **Owner:** TanitAD_TrainingFlyWheel · **Tier: T0 training-side**
**Pre-registration:** `PREREG_P_RC21.md` **AMENDMENT 1**, landed before any
anchored arm existed. **Evidence class:** MEASURED (ours).
⚠️ **NON-PARITY corpus** — no number here enters cross-arm comparisons.
✅ **Readout is now CORRECT**: eval mode (`"eval_mode": true`), exactly
deterministic (run-to-run spread 0.000), per-episode values stored ⇒ the
**PAIRED** episode-cluster bootstrap the house rule requires is computed here for
the first time. This SUPERSEDES `RESULT_P_RC21.md` §1 entirely.

---

## 0. Verdict

⭐ **THE ANCHOR WORKS — monotonically, and it is the first thing in this campaign
that moved the metric it was built to move.** ⛔ **AND THE REWARD IS THE LIMIT:
fan collision never falls at ANY anchor strength.** Pre-registered branch 2 of 5,
matched exactly: *"R3 held at every `w_anchor` but R2 never falls ⇒ the anchor
fixes drift and the reward still cannot prune collisions ⇒ do NOT scale; the next
lever is reward composition."*

## 1. The anchor-strength curve — the thing DDv2 never published

Paired episode-cluster bootstrap, 4000 reps, 15 clusters, same windows before/after.

| `w_anchor` | ΔR3 sel-ADE | % of the 0.654 m baseline | paired 95 % CI | |
|---|---|---|---|---|
| **0.0** | +0.2880 m | **+44.0 %** | [+0.1889, +0.3943] | **SEPARATED** — drift is real |
| **0.1** | +0.2207 m | +33.7 % | [+0.1382, +0.3032] | **SEPARATED** |
| **1.0** | +0.1115 m | +17.0 % | [−0.0007, +0.2085] | not separated |
| **10.0** | +0.0294 m | **+4.5 %** | [−0.0618, +0.1160] | not separated |

⭐ **Monotone across two decades**, and by `w_anchor ≥ 1` the drift is no longer
distinguishable from zero. The unanchored arm's drift IS separated, so this is a
real effect being suppressed, not noise being averaged.

## 2. ⛔ Fan collision never moves — at any anchor strength

| `w_anchor` | ΔR2 collision | paired 95 % CI (pp) | |
|---|---|---|---|
| 0.0 | +0.73 pp | [−0.18, +1.88] | not separated |
| 0.1 | −0.05 pp | [−0.61, +0.53] | not separated |
| 1.0 | +0.17 pp | [−0.10, +0.48] | not separated |
| 10.0 | −0.05 pp | [−0.23, +0.08] | not separated |

No trend, no separation, no sign consistency. The truncated advantage is not
pruning colliding candidates on this model and this reward.

## 3. ⭐ THE SHARPER FINDING: the composed reward REWARDS DRIFT

Read ΔR1 (the trained objective, on the deterministic fan) beside ΔR3:

| `w_anchor` | ΔR1 | ΔR3 |
|---|---|---|
| 0.0 | **+0.1272** [+0.0817, +0.1675] SEP | **+0.2880 m** SEP |
| 0.1 | +0.0694 [+0.0407, +0.0981] SEP | +0.2207 m SEP |
| 1.0 | **−0.0660** [−0.0770, −0.0553] SEP | +0.1115 m |
| 10.0 | **−0.0380** [−0.0468, −0.0292] SEP | +0.0294 m |

The only arm that **gains** composed reward is the one that drifts most, and the
gain shrinks in lockstep with the drift as the anchor tightens. ⇒ **The reward
gain and the ADE degradation are the same movement.** The policy is not learning
to drive better and drifting as a side effect; *the drift is what the reward is
paying for.*

⚠️ **And at `w_anchor ≥ 1` ΔR1 is NEGATIVE and separated** — under a strong trust
region the residual motion makes the composed reward *worse*. So there is no
window in this sweep where the policy improves its own objective without
drifting. That is a statement about the REWARD, not about the anchor.

## 4. ✅ The regression control validates the readout

| arm | ΔR1 | ΔR3 |
|---|---|---|
| `sreg` progress-only, 300 steps | **−0.2712** [−0.3146, −0.2289] **SEPARATED** | **+0.4279 m** [+0.2774, +0.5739] **SEPARATED** |

Degrades as predicted, on both axes, with paired separation. ⇒ the readout can
see the failure it must see, so §§1–3's nulls are measurements rather than
blindness. *(Note R2 for `sreg` reads −0.24 pp — collisions slightly DOWN while
ADE explodes: the milder form of the "left the road, so nothing to collide with"
signature that the broken estimator produced in extremis.)*

## 5. What this establishes, and what it does not

✅ **Established**
1. The reference-policy anchor is **correct and effective**: step-0 divergence
   exactly 0.00000000 m² against a frozen copy of itself, and drift suppressed
   monotonically 44 % → 4.5 % across two decades of `w_anchor`.
2. The instrument chain is **sound**: deterministic eval-mode readout, paired
   bootstrap, a regression control that separates on both axes.
3. The Master Mind's ruling was right on the mechanism — GRPO's stabiliser is a
   trust region, and supplying it fixed exactly the failure it was predicted to fix.

⛔ **NOT established, and not to be claimed**
1. **That the DDv2 mechanism transfers.** It did not prune collisions here.
2. Any driving claim. T0 only, NON-PARITY corpus, 15 clusters.
3. Any refcv3 number. v2.1 ≠ refcv3.

## 6. Next lever — the reward, per the committed branch

⛔ **Do NOT scale, do NOT add steps, do NOT sweep the anchor further.** The
committed consequence is *"the next lever is reward composition"*, and §3
sharpens why: the reward currently pays for drift. Ranked:

| # | item |
|---|---|
| 1 | **Diagnose WHICH component pays for drift.** R4 per-component means are stored per arm — decompose ΔR1 by component across the sweep. If `progress` supplies the gain, the v0-referenced form is still rewarding "go further than the reference" in a way the anchor has to fight. |
| 2 | **The collision term cannot rank what it cannot see.** It fires on ~50 % of windows (base rate) and is binary per candidate. A graded proximity term (distance-to-nearest-obstacle, not a hit/no-hit flag) would give the advantage something continuous to work with — the same fix class as graded headway. |
| 3 | Only after 1–2: re-run the sweep's `w_anchor ∈ {1, 10}` arms with the revised reward. The anchor is settled; it does not need re-establishing. |
| 4 | Still deferred: M-B9 (true diffusion-step density). §3 does not implicate the estimator — the policy moves and the reward tracks it; the objective is what is mis-specified. |
