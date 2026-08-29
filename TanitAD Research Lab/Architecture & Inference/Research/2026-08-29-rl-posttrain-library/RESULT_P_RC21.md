# RESULT — P-RC21: RL post-training of the REF-C v2.1 planner

**Date:** 2026-08-29 · **Owner:** TanitAD_TrainingFlyWheel · **Tier: T0 training-side**
**Pre-registration:** `PREREG_P_RC21.md`, committed `06c715d8b` **while P1 was
running and before any `readout_after` existed** — the precedence is in the
history, not in a claim about it.
**Evidence class:** MEASURED (ours). ⚠️ **NON-PARITY corpus** — nothing here
enters cross-arm comparisons or the leaderboard.

---

## 0. Verdict in one line

⛔ **P1 does NOT meet its committed outcome.** The fan-collision rate did not
move; the deployed trajectory drifted badly. ✅ **The instrument chain IS
validated** — the deliberate-regression arm degraded as predicted, CI-separated,
so P1's null is a measurement rather than a blindness.
⭐ **And the failure is interpretable, not a dead end**: this pilot ran with **no
policy anchor of any kind**, which is exactly what GRPO's trust region provides
and exactly what the observed drift predicts.

## 1. Numbers (15 val episodes / 120 windows, episode-cluster bootstrap ×2000)

### P1 — grpo, full composed reward, 2,000 steps
| metric | before | after | Δ | CIs |
|---|---|---|---|---|
| R1 fan reward | +0.9410 [0.828, 1.042] | +0.9933 [0.897, 1.082] | **+0.052** | overlap → not separated |
| **R2 fan collision** | **11.068 %** [5.03, 17.97] | **10.749 %** [4.93, 17.08] | **−0.319 pp** | overlap → **FLAT** |
| R3 sel-ADE | 1.797 m [1.245, 2.646] | 3.046 m [2.113, 4.600] | **+69.5 %** | overlap, but a large adverse point move |

**Committed PASS** required R2 to fall **CI-separated** *and* R3 to stay within
**+10 %**. Neither holds ⇒ **FAIL**. The result sits across two pre-registered
branches at once: *"R1 rises but R2 flat"* (composition imbalance — do NOT ship)
and *"R3 degrades > 10 %"* (anchor too weak).

### P2-reg — progress-only (deliberate regression), 300 steps
| metric | before | after | Δ | |
|---|---|---|---|---|
| R1 composed reward | +0.9586 [0.849, 1.057] | +0.6592 [0.549, 0.762] | **−0.299** | ⭐ **CIs DO NOT OVERLAP — separated** |
| R2 fan collision | 10.983 % | 11.569 % | +0.586 pp (worse) | direction correct |
| R3 sel-ADE | 2.105 m | 2.263 m | +7.5 % | |

⇒ **The pass condition for the AUDIT, not for the model.** The readout sees the
failure it must see, which is what makes P1's null trustworthy.

### Cost — MEASURED, and an order of magnitude below my own estimate
P1: 2,000 steps + both readouts in **176.4 s (0.088 s/step)**; P2-reg: 300 steps
in **29.7 s**. Whole campaign ≈ **3.5 minutes** on the RTX 4060.
⚠️ I estimated 15–60 min. **10–20× too pessimistic** — the LRU fix (whole corpus
pinned in RAM) removed the data-boundness I had priced in, and the run held
84–86 % GPU utilisation. Recorded so the next estimate starts from the measurement.
Trainable **8.63 M** / frozen **95.56 M**, `decoder.conf_head` excluded — verified
in both run records. Components fired at their A0 base rates (collision 53.7 %,
headway 33.3 %).

## 2. ⭐ Diagnosis: the pilot ran with NO policy anchor

`w_imitation = 0.0`, and no reference-policy term existed in the library at all.
Stated at launch (no IL loss wired for v2.1; the intended anchor was "the frozen
96 % of the model") — and **a frozen trunk is evidently not a substitute for a
constraint on the trainable head**. RL moved the decoder toward the rule-based
reward (R1 rose) while the deployed trajectory drifted 69 % (R3): the classic
unconstrained-policy-gradient failure.

⇒ **This SUPPORTS the published design rather than refuting the method.**

⭐ **AND THE DIAGNOSIS WAS SHARPENED BY THE MASTER MIND (2026-08-29), correcting
my framing:** I proposed the remedy as an *imitation* loss against labels.
**GRPO's actual stabiliser is a TRUST REGION against the REFERENCE POLICY** — a
divergence penalty between the training decoder and a frozen copy of the cold
start — not a label-supervised term. `w_imitation` was standing in for it. So
the correct fix is not "wire IL", it is **close the gap to the published
method**. See §6.

⚠️ **AND THE CONFUSION IS NOT PRIVATE — which is why this is a permanent line
rather than a footnote.** *"Imitation anchor"* and *"trust region"* are used
interchangeably in secondary summaries of the RLHF/GRPO family, so substituting
one for the other reads as a paraphrase rather than a design change. It is not:
an imitation loss pulls toward LABELS (needs a GT join, corpus-specific, must be
redone per dataset), a trust region pulls toward the REFERENCE POLICY (label-free,
corpus-agnostic, transfers unchanged). **A design detail becomes a silent
substitution exactly when the two names are treated as synonyms** — the same
family as quoting a number without its estimator. ⇒ In this programme the two
terms are now distinct vocabulary, and a document that says "anchor" must say
WHICH.

## 3. ⛔ Two defects this run exposed in my own instruments

### 3.1 The estimator was wrong — caught by the first P1 arm (TRAIN-C3)
The first chain COLLAPSED the planner: R2 11.113 % → **0.000 %** because R3
exploded **1.969 m → 347.227 m** — every candidate left the road, so nothing
remained to collide with. *A metric moving to its best value for the worst
reason.* Root cause, MEASURED by A/B (`code/estimator_ab.py`): `logp` was written
from the **raw noise draw**, which cancels the `(sample − mean)` dependence, so
the gradient could only shrink/inflate offsets and never move them toward good
samples. Fixed to `eps_eff = (sample.detach() − mean)/scale`; pinned by a
**DIRECTIONAL** test that requires actual convergence on a known optimum. Broken
artifacts preserved at `raw/p_rc21/p1-grpo-BROKEN-ESTIMATOR__*`.

### 3.2 ⚠️ The static audit went BLIND at the pilot's dt (TRAIN-C4)
P2-reg's own `reward_audit` reads **`clean`** for the hackable reward — it should
read **FLAGGED**. Cause MEASURED: at **dt = 0.5** the fixed 30 m progress
reference saturates *both* the sane reference *and* `bullet_straight` at the
clamp (both exactly **1.5000**); they TIE, and "strictly beats" finds no winner.
At dt = 0.1 the identical audit returns **FLAGGED** with winners
`[bullet_straight, teleport]`. **My test pinned dt = 0.1 only.**

⇒ The guard stopped being able to fail because a *sampling-rate* parameter
changed. **The BEHAVIOURAL regression arm caught what the static audit missed**,
which is the argument for two independent guards rather than a better single one.

## 4. ⚠️ Limits of this readout (fixes, not excuses)

1. **CIs are UNPAIRED.** House rule is the *paired* episode-cluster bootstrap for
   two arms on the same windows. My readout stores only aggregates, so paired
   deltas cannot be computed post-hoc — **R2's small move could be separated
   under a paired test and this readout cannot tell.**
2. **The readout's decoder noise is UNSEEDED.** The same checkpoint's
   before-readout differs run to run (R3 **1.797** vs **2.105** = 17 %). ⇒ the
   **+10 % R3 guard I pre-registered sits BELOW the instrument's own noise
   floor** and was mis-specified — it could not have failed honestly in either
   direction. +69.5 % clears it comfortably, so the adverse move stands, but the
   threshold must be re-derived from measured seeded noise.
3. **15 val episodes** ⇒ CIs ~13 pp wide on R2 by construction. A real effect
   smaller than that is undetectable here.
4. Standing: surrogate Gaussian-on-offset ≠ the true reverse-diffusion density;
   static-lead approximation; NON-PARITY corpus; 4 waypoints ⇒ coarse jerk.

## 5. What transfers to refcv3 (and what does not)

⛔ **No number here transfers.** v2.1 has no PhiTac hierarchy, no v7 vocabulary,
no goal graft.
✅ **These do:** the library runs end-to-end on a real trained planner at
**0.088 s/step**; the reward fires at its expected base rates on real scene
context; the veto/advantage plumbing is exercised; the readout can detect a
deliberately broken objective; and **an RL objective with no policy anchor
drifts the deployed trajectory** — which makes the anchor a prerequisite for A1,
not an option.

## 6. ⭐ DECIDED (Master Mind, 2026-08-29): build the REFERENCE-POLICY ANCHOR

A third option, chosen over both of mine (wire IL for v2.1 / wait for refcv3):

> a divergence penalty (L2 on emitted trajectories, or KL in density form)
> between the training decoder and a **FROZEN COPY of the cold-start decoder**
> on the same input.

Three reasons it dominates:
1. **It is what the method actually specifies** — GRPO/DDv2's stabiliser is a
   trust region against the *reference policy*, not an imitation loss against
   labels. This closes the gap to the published method rather than inventing an
   anchor.
2. **It is label-free** — no v2.1 data plumbing, no GT join, nothing to redo for
   a different corpus. That is precisely why the IL route was the expensive one.
3. **It transfers unchanged to refcv3** — same flag, same semantics; refcv3's own
   IL loss then becomes an *additional* option rather than a prerequisite.
   ⭐ Reuse the frozen-copy machinery that landed in the trainer today
   (MM-E4 `--o5-target frozen`: `_EmaCopy(τ=1)`, no update wired,
   resync-after-load) — proven and test-pinned; do not re-derive it.

### Sequence (agreed)
1. ⛔ **Instrument fixes FIRST — mandatory, a re-run is unreadable without them:**
   (a) **seed** the readout's decoder noise, then **re-specify the R3 guard**
   against the measured seeded noise; (b) store **per-window** values so the
   paired bootstrap is computable.
2. Build the reference-policy anchor.
3. **Re-run P1 with an anchor sweep (0 / low / high)** — which also yields the
   **anchor-strength curve DDv2 never published**.
4. The same P2-reg control.
5. ⛔ **Prereg amendment before the re-run**, per this evening's precedent, and
   the amendment must state that **the anchor sweep REPLACES the "raise
   `w_imitation`" remedy branch**.

Still deferred: M-B9 (pathwise gradients); `elapsed_s` in `run_posttrain`.
