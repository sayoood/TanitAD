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

---

## ⭐ ADDENDUM — the ΔR1 DECOMPOSITION (2026-08-29, Master Mind's sharper question)

The question asked was not *"which term pays for drift"* but the stronger
**"at `w_anchor`=10, does ANY component have a positive ΔR1?"** — because if none
does, the finding is not *one badly-shaped term* but **an objective with NO local
improvement direction inside the trust region**.

### Weighted contribution to ΔR1, per component, all four anchor strengths

| `w_anchor` | collision | comfort | **feasibility** | headway | progress | SUM |
|---|---|---|---|---|---|---|
| 0.0 | −0.0073 | +0.0115 | **+0.1101** | −0.0023 | +0.0151 | **+0.1272** |
| 0.1 | +0.0005 | −0.0050 | **+0.0727** | −0.0007 | +0.0020 | +0.0694 |
| 1.0 | −0.0017 | −0.0278 | −0.0355 | −0.0003 | −0.0007 | −0.0660 |
| 10.0 | +0.0005 | −0.0220 | −0.0170 | −0.0003 | +0.0007 | −0.0380 |
| REG (progress-only) | +0.0024 | −0.0702 | **−0.2213** | −0.0054 | **+0.0232** | −0.2712 |

### Answers

⛔ **At `w_anchor` = 1.0: NO component is positive.** The objective has no local
improvement direction inside that trust region — every direction it can reward
requires leaving the reference.

⚠️ **At `w_anchor` = 10.0 the answer is NOT a clean "none":** `collision`
(+0.0005) and `progress` (+0.0007) are positive. But they are **~3 % of the
magnitude** of the negative terms (`comfort` −0.0220, `feasibility` −0.0170), so
the strong claim holds *effectively* while the literal binary answer is "two,
negligibly". Stating both rather than rounding to the cleaner story.

### ⭐ THE DIAGNOSIS — and it refutes my own prediction

I predicted `progress` would supply the drift-paying gain. **It does not.**
`feasibility` supplies **+0.1101 of the +0.1272 total at w=0 — 87 %.** Progress
contributes 12 %.

**Mechanism.** `feasibility` is continuous, weighted 0.50, and the RAW ANCHOR
VOCABULARY is full of headroom (it samples 0–30 m/s and ±0.35 rad/s uniformly, so
many candidates sit outside the (a, κ) envelope). The cheapest way to raise it is
to emit **blander trajectories** — lower curvature, gentler acceleration. That
costs ADE (+44 %) and does nothing for collisions. ⇒ **the reward gain and the
drift are the same movement**, and the movement is "make the fan tamer".

⚠️ **The weights are NOT the problem.** Scene-grounded terms carry *more* weight
(collision 1.00 + headway 0.30 = 1.30) than scene-free ones (0.50 + 0.20 + 0.30 =
1.00). ⭐ **The gradient followed the EASIEST term, not the heaviest.**
`collision` is BINARY per candidate and only improves by actually avoiding an
obstacle; `feasibility` is continuous with slack everywhere. An advantage
estimator ranks what varies.

**Cross-check from the regression arm:** training on progress-only drives
`progress` to **+0.0232** (its largest value anywhere, as it must) while
`feasibility` **collapses −0.2213**. The two are in direct tension, which is what
the composed reward is supposed to balance and currently does not.

### What this changes about the fix
A graded proximity **barrier** (approved, implemented) gives the scene-grounded
direction something continuous to rank. But this decomposition says it is **not
sufficient alone**: as long as `feasibility` offers the largest easy gain on a
raw fan, the policy will keep buying it with drift. The re-run must therefore
report the decomposition again, and a `feasibility`-still-dominates result would
mean the term needs a reference (feasibility RELATIVE to the cold start's fan)
rather than an absolute one.

---

## ⭐ THREE FINDINGS THAT OUTLIVE THIS CAMPAIGN

Recorded here because they are design rules for every group-relative objective
the programme builds, not facts about REF-C v2.1.

### 1. ⛔ AN ADVANTAGE ESTIMATOR RANKS WHAT **VARIES**, NOT WHAT IS **WEIGHTED**

MEASURED, and the cross-check is what makes it a rule rather than an anecdote:
the scene-grounded terms carried **more** weight than the scene-free ones
(`collision` 1.00 + `headway` 0.30 = **1.30** vs `feasibility` 0.50 + `comfort`
0.20 + `progress` 0.30 = **1.00**) — and the gradient went to the scene-free
side anyway, with `feasibility` alone supplying **87 %** of the gain.

⇒ **A BINARY term inside a group-relative advantage is nearly INERT BY
CONSTRUCTION.** Centring a reward across candidates removes anything constant,
and a hit/no-hit flag is constant across the large majority of fans. Weighting it
higher does not help: zero variance times any weight is still zero.

**The design rule:** every term intended to *rank* must be continuous over the
range the candidates actually occupy. Terms that are genuinely binary belong in
the **constraint/veto channel**, not the reward — the same separation already
forced for TTC (`advantage.truncated_inter_anchor_advantage`) and for headway.

### 2. ⭐ THE REFUTED PREDICTION IS WORTH AS MUCH AS THE RESULT

The prediction (`progress` pays for the drift) was recorded **before** the
decomposition ran, and the measurement refuted it (`feasibility`, 87 %). Stating
the refutation alongside the number is what makes the 87 % credible rather than a
story fitted to data already seen — the same reason a pre-registration must land
before its outcome. A diagnosis that merely *confirms* what its author expected
should be trusted less, not more, unless the expectation was written down first.

### 3. ⚠️ "TAME THE FAN" IS THE SAME CLASS AS MM-E4 L1 — TWO LANES, ONE FAILURE, ONE EVENING

| lane | cheap continuous direction | what it satisfied | what it destroyed |
|---|---|---|---|
| MM-E4 L1 (drift ladder) | a regulariser aimed at the statistic its own read measures | drift −25 % | prediction |
| **P-RC21 (this lane)** | make the emitted fan blander | `feasibility` +87 % of the gain | ADE +44 % |

⇒ **THE CLASS: an objective admits a cheap continuous direction that satisfies
the metric while destroying the thing the metric was a proxy for.** Both were
found the same evening in independent lanes, which is what makes it a class
rather than a coincidence.

**The check, now structural rather than remembered:** before adopting any new
reward term, ask *what is the cheapest continuous way to increase this, and would
we deploy the behaviour that results?* The `proximity` barrier is built to fail
that check safely — it **caps at 0**, so its optimum is "keep a normal margin",
which sits inside the driving envelope rather than outside it. A term whose
optimum lies outside the envelope (an unbounded distance-maximiser, a saturating
headway) is this class waiting to happen.
