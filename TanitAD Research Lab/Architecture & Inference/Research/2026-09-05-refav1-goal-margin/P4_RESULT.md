# P4 — refav1 DOES turn. It turns EVERYWHERE, at the clip bound, half of it the wrong way.

**Row:** `D-REFAV1-P4-COS-THRASH` · **Class:** MEASURED, T1, dev-box 4060
**Arm:** `refav1-21109-p4-cos_argmax` — `--cost-metric cos` (the SHIPPED default) + plain
argmax, `--cost-weights 0.0,0.0,64.29715042415070`, `--window-stride 16`, ckpt step 21,109.
**Panel:** 8 turn-dense episodes, **40 windows**, 31 scored (9 near-stationary excluded).
**Raw:** `raw/gate_cos_argmax.json`, `raw/rec_cos_argmax.json`
**Instrument:** `tools/p4_turn_gate.py` · **Pre-registered:** `SPEC.md`, `raw/p4_episodes.json`

---

## The measurement

| | n | **executed curvature non-zero** | **direction correct** | \|κ\| p50 | \|κ\| max | head decoded a turn |
|---|---|---|---|---|---|---|
| **GT TURN** (`|κ| > 0.04`) | 11 | **0.9091** | **0.4545** | 0.1190 | **0.2000** | 0.7273 |
| **GT STRAIGHT** | 20 | **0.9000** | — | — | **0.2000** | 0.3000 |

⛔ **Every control passed:**
* `ha0_ext` — the non-planner integrator in the same npz — reads **1.0000 non-zero, absmax
  0.6031**, which matches the predecessor's independently banked **0.603** exactly. The
  curvature column is LIVE, so these are measurements.
* the **name-remapped bank join** agrees with the from-`g` curvature to **< 1e-4** — two
  independent routes to `gt_kappa`.
* the banked **zero-GPU decode prediction** for this exact window set is reproduced.
* 9 near-stationary windows excluded and counted, never silently dropped.

## What it means

⭐ **The PI's question has a yes: refav1 emits curvature on 91 % of the windows where the road
actually turns.** The gate is not shut on this configuration.

⛔ **But it is not driving.** It emits curvature on **90 % of STRAIGHT windows too**, both
classes saturate the planner's **`kappa_max = 0.2` clip bound**, and the direction is correct on
only **45 %** of turns — a coin flip. This is the failure `SPEC.md` named in advance:
*"a bias that buys turns by turning everywhere is not a fix."* Here nothing bought it; the arm
does it by default.

### The four families, T1 (`taniteval.ci` episode-cluster bootstrap; ⛔ never `overlapping_holdout_se`)

| arm | tier | ADE (m) | 95 % CI |
|---|---|---|---|
| **`cl` — the planner** | T1 | **1.8944** | [1.0547, 2.9504] |
| `ha` — hold the last observed (a, κ) | T1 | 0.8888 | [0.6026, 1.1928] |
| `ha0` — constant velocity | T1 | 0.9251 | [0.7024, 1.1783] |
| ⭐ **`ha0_ext` — hold the MEASURED (a₀, κ₀)** | T1 | **0.8772** | [0.6195, 1.1437] |
| `ol` — teacher-forced | **T0** | 0.8052 | [0.5787, 1.0577] |

⚠️ `ha0_ext` is the canonical integrator (`refav1_arm.hold_ext_controls`), decision **M11** /
`D-MM-ADJ-1` — never `echo_gate.ha0_ext`.
⚠️ **STRATEGIC is UNAVAILABLE** on this surface and is reported as such with its reason, not
omitted. **LONGITUDINAL distance-keeping is REFUSED** — no `--lead-block` was passed — so the
longitudinal family here is speed-side only. Both are declared, per the binding rule's clause 5.

⇒ **The planner is 2.16× worse than holding the measured state**, and it *chose* those plans:
`baseline_won_frac = 0.075`, i.e. **the CEM plan beat the baselines on 92.5 % of windows**. The
search is not failing to find what the cost wants — **the cost wants the wrong thing.**

## ⛔⛔ The variable nobody was controlling: `W_KAPPA`

This arm runs `--cost-weights 0.0,0.0,64.297` — **`W_KAPPA`, the curvature penalty, is exactly
zero.** With no curvature penalty the iCEM population's colored noise is free to wander to the
clip bound, and the goal term alone cannot discipline it. That is exactly what the table shows.

⚠️ **And that triple is not mine — it is the one the predecessor's own `run_ab.sh` and
`run_ab_targeted.sh` both pass.** ⇒ **every lateral conclusion drawn from those A/B arms is
drawn in this regime**, on a planner that curves at the clip bound on 90 % of straight road.

⚠️ **It also does not reproduce `D-REFAV1-DRIVE-GATE2`'s "`cos` executes 0/38".** Here `cos`
executes on 90 %. The two arms differ in at least the weights — and the banked `dump_cos_ext`
**records no cost block at all** (`D-REFAV1-COS-WEIGHTS-UNRECORDED`), so they cannot be shown
weight-matched. ⛔ **This does NOT retract gate 2**, which is independently supported by
`assert_metric_gate.py`'s within-script comparison at four search sizes. It does say the
**dump-level** contrast was confounded, and that **`W_KAPPA` — not the metric — is the dominant
lateral variable on this surface.**

## ⇒ What to do, and it is cheap

⭐ **Restore a non-zero `W_KAPPA` and re-measure.** The shipped triple is `{0.02, 0.05, 0.1}`;
the Stage-B fitted triple is `{12.859, 32.149, 64.297}`. The zeroed triple was adopted because
the weights were believed **inert** — a claim measured under `ccos` and not, on this evidence,
transferable to `cos`.

⛔ **Committed in advance, before that arm runs:**
* **If a non-zero `W_KAPPA` cuts the straight-window curvature rate** well below 0.90 while
  keeping the turn rate high, then the curvature penalty is the binding lateral term, the
  "weights are inert" claim does not hold under `cos`, and every arm on the zeroed triple needs
  re-reading.
* **If it does not move**, the weights really are inert here, the thrashing is the goal term
  or the search itself, and the vocabulary work (`--goal-kappa-turn`) becomes the live lever
  again.

⚠️ **And note what this does to the vocabulary escalation.** A planner that saturates `κ = 0.2`
on straight road is not limited by a goal vocabulary that tops out at `κ = 0.08`. ⇒ **the
vocabulary is NOT the binding constraint on THIS configuration** — it is the binding constraint
on the *goal field*, which is a different object, and `D-REFAV1-VOCAB-REALISED` already scoped
the vocabulary claim to the GOAL curvature rather than the EXECUTED one. The ordering changes:
**curvature discipline first, vocabulary second.**

⚠️ **Scope:** one arm, one checkpoint, **n = 11 GT-turn windows** on a panel deliberately
selected for turn density — these are CONDITIONAL rates, not corpus marginals, and `n = 11` is
thin. The paired arms (`ccos`, the seed replicate, and `--goal-kappa-turn 0.02`) were still
running at hand-off; `tools/analyze_p4.sh` re-runs the whole analysis on whatever has landed.
