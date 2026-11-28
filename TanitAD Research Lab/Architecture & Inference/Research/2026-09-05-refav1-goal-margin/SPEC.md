# SPEC — refav1 goal-margin: make it turn

**Pre-registration. Written and banked BEFORE any number in this package was read.**
**Agent:** TanitAD Architecture & Inference FlyWheel · **Date:** 2026-09-05
**Branch:** `agent/arch-inf-20260803`
**Predecessor:** `…/2026-09-05-refav1-make-it-drive/` (gates, lever, ceiling — all MEASURED)
**Register rows opened:** `D-REFAV1-MARGIN-ANATOMY`, `D-REFAV1-MARGIN-REFIT`,
`D-REFAV1-MARGIN-EXEC`

---

## The objective (the PI's, verbatim)

> *"our goal is a to let refav1 drive and proove the perfromqance of wm based architectures"*

⇒ The deliverable is **refav1 executing turns**, not another refutation. The diagnosis is
finished and is not re-derived here.

## What is INHERITED (not re-measured, cited)

| fact | value | source |
|---|---|---|
| gate 1: `LANE_KEEP` decode ⇒ curvature exactly 0.0 | 244/244 | `GATE_ON_REAL_MODEL.md` |
| gate 2: `cos` refuses a decoded turn | 0/38 (`ccos` 63 %) | `TWO_GATES.md` |
| the head's turn-vs-hold score separates | AUC **0.712** vs shuffled **0.506 ± 0.034** | `raw/decision_ceiling.json` |
| the margin gap | **0.297 logits** | same |
| the lever exists and is parity-pinned | zero bias ≡ no flag, bit-identical | `raw/lat_bias_parity_realckpt.json` |

## The architecture fact that sets the cost of the fix

`stack/tanitad/refs/refa_v1.py:1218` — `lat_head = nn.Sequential(LayerNorm(d_int),
Linear(d_int, n_lat))`, applied to `intent` (`:1956`). **The goal head is one linear map.**
⇒ A re-fit is a linear-head fit on banked `intent` features: CPU-minutes, not GPU-hours,
and the trunk and world model are untouched **by construction** (they are not in the
optimisation at all — asserted by parameter count, not assumed).

---

## P1 — the margin anatomy. Both outcomes committed.

**Instrument:** `tools/margin_anatomy.py` · **Raw:** `raw/margin_anatomy_s40.json` (+ `_s5`)
**Panel:** the banked logits, 282 windows / 141 episodes (stride 40) and 1974 (stride 5).

### The decision rule, committed in advance

> **CALIBRATION** iff `bal_acc_at_best_threshold >= 0.80` — the score separates and only the
> operating threshold is wrong, so a single scalar (`lat_logit_bias`) fixes it and **no
> training is needed**.
> **SEPARABILITY** iff `bal_acc_at_best_threshold < 0.80` — no threshold on this score
> operates acceptably, and the score itself must be re-fit (P2).

`bal_acc = (recall + specificity) / 2`, maximised over every threshold on
`s = max(TURN_/NUDGE_/LANE_CHANGE_ logits) − LANE_KEEP logit`. The 0.80 bar is chosen as the
point at which a lateral decision is better than four-in-five reliable — below that a
threshold rule is not a fix, it is a trade.

### ⛔ The control that can void a proposed objective

The predecessor proposes `median_margin_gap_logits` (today 0.297) as the P2 success
criterion. **A raw-logit margin is not scale-free.** `lat_head` ends in a `Linear`; scaling
its weight and bias by `c` multiplies every logit — and therefore the gap — by `c`, while
every argmax, every ROC point and the AUC are unchanged. **PREDICTION, committed:** with
`c = 3` the decode is bit-identical, the AUC is unchanged to 1e-12, and the gap is exactly
`3 × 0.297`. If that prediction holds, the raw-logit-margin objective is **gameable by weight
norm and inadmissible as stated**, and P2's objective must be a scale-free quantity
(balanced accuracy / AUC / Cohen's d) instead. If it fails, my reasoning about the head is
wrong and P2 must be re-specified.

### ⛔ The operating point, committed BEFORE the curve is read

* **PRIMARY — max Youden's J (= max balanced accuracy).** Justified by **symmetry of harm**:
  a missed turn departs the road at a curve; a false turn steers off a straight road. Neither
  is the cheap error, so the symmetric criterion is the honest one.
* **SECONDARY — max decode recall subject to false-turn-on-straight ≤ 0.10** (argmax's own
  0.073 plus headroom), reported because gate 2 refuses most decoded turns and so makes a
  decode false turn cheaper at the plan level than a decode miss.

⚠️ `correct_turn_rate` is **inadmissible** as an objective — the predecessor MEASURED that a
uniform-random control scores 0.280 on it, above every real setting. Recall is only ever read
beside the false-turn rate.

---

## P2 — the goal-head re-fit. Both outcomes committed.

**Runs only if P1 returns SEPARABILITY.** If P1 returns CALIBRATION, the deliverable is the
scalar and P2 is skipped — and that is a better outcome, not a lesser one.

**What is optimised:** `lat_head` alone (`LayerNorm` + `Linear`, `d_int → 8`).
**What is frozen:** everything else — trunk, encoder, world model, strategic and tactical
policies, `lon_head`, `route_head`. ⛔ **Asserted by parameter count and by a bit-identical
`intent` check**, never assumed.

**Inputs — the P3 admissibility check, run before training:**

| | |
|---|---|
| the head's input | `intent` = `tactical_policy(pooled_win, ctx, ego)` + `nav_to_intent(nav)` |
| ⛔ the rule | *could this have been computed from something the label was derived from?* |
| answer, to be MEASURED not assumed | `pooled_win` is vision. `ego` is `v0` at `t0` — PI-ruled admissible (`velocity-at-cycle-time`, 2026-09-02). `nav` is a conditioning input, and its provenance is **probed in this package** (`nav_valid`, `nav_cmd` distribution, and mutual information with the label) rather than asserted. |
| ⛔ the situation classifier | its output enters **nowhere** — the head sees `intent` only, and no arm in this package computes a situation posterior, argmax or embedding. |

⚠️ **I am not changing the inference-time input set.** The re-fit replaces the last linear
map on top of the *same* `intent` the shipped arm already computes. Any leak in `intent` is
pre-existing and INHERITED; this package must therefore **report** the nav provenance, and
must not claim to have removed a leak it did not touch.

**Objective, committed:** class-balanced margin-widening on a **scale-free** criterion —
balanced accuracy / AUC of the turn-vs-hold score — against the 86.5 % `LANE_KEEP` prior.
⛔ **Accuracy is not the headline and will barely move.** The pre-registered success
criterion is:

> **P2 SUCCEEDS** iff `bal_acc_at_best_threshold` rises **and** the decode operating point at
> the committed criterion beats the shipped argmax on Youden's J, **on a HELD-OUT split that
> the fit never saw**, with the fit's hyper-parameters chosen on the FIT split only.
> **P2 FAILS** iff the held-out J does not improve — in which case the honest conclusion is
> that `intent` does not linearly carry more than the shipped head already extracts, and the
> next lever is the candidate set (`±κ` baselines), not the head.

⛔ **Split by EPISODE, never by window.** Windows from one episode are near-duplicates; a
window-level split would leak and manufacture a success. n printed per split.
⛔ **A constant-predictor control and a shuffled-label control** both run, and must read the
no-information value, or the panel is void (CLAUDE.md probe-trap rule).

---

## P4 — does gate 1 actually open? Both outcomes committed.

**Claim to be tested:** on GT-turn windows the EXECUTED plan carries non-zero curvature.

* ⛔ **Every "exactly 0.0" claim carries a same-breath control column that must read
  NON-ZERO** — the `ha0_ext` curvature column (predecessor: 98.9 % non-zero, absmax 0.603).
  A zero with no live control beside it is a read error until proven otherwise.
* Reported: **executed-turn rate on GT-turn windows** AND **false-turn rate on GT-straight
  windows**, never recall alone.
* **GATE 1 OPENS** iff executed curvature is non-zero on a materially non-zero fraction of
  GT-turn windows where the control column reads non-zero in the same breath.
* **GATE 1 STAYS SHUT** iff the executed curvature remains 0.0 — in which case gate 2
  (`cos` vs `ccos`) or the candidate set is binding, and that is the reported result.

## P5 — the number the PI asked for

T1 (self-action open loop, per `EVAL_DOCTRINE.md` and the 2026-09-02 open/closed ruling),
**four metric families reported PER FAMILY, never pooled** — LONGITUDINAL, LATERAL,
TACTICAL, STRATEGIC — beside ADE. Paired episode-cluster bootstrap (`taniteval/ci.py`).
⛔ **NEVER `overlapping_holdout_se`.**
⛔⛔ **A SEED REPLICATE IS CARRIED** (`H-ESTIM-SEED-1`): a separated CI from a one-seed arm is
necessary, not sufficient — the bootstrap resamples EPISODES with the models held fixed and
is structurally blind to run-to-run variance; a zero-lever replicate produced "separated" on
3 of 18 metrics. Any lever claim is read against that noise floor or it is not quotable.
⚠️ `ha0_ext` is canonical as the INTEGRATOR (`refav1_arm.hold_ext_controls`), never
`echo_gate.ha0_ext` — decision **M11** / `D-MM-ADJ-1`; the two disagree by 1.86 m at 15 s.

## Compute constraints (binding)

⛔ Never touch `tanitad-refcv3` (refcv4b training) or Thor (refav1 Stage B). P1 is zero-GPU.
`OMP_NUM_THREADS=6` before any multi-arm job. The stack cannot run from the G: mount
(`Errno 22`); the off-Drive clone is used and the imported tree is verified.

## Falsifiability summary — what would make me wrong

1. P1 returns CALIBRATION ⇒ my expectation that a re-fit is needed is wrong, and the scalar ships.
2. The `logit_scale` control fails ⇒ my reading of the head architecture is wrong.
3. P2's held-out J does not improve ⇒ `intent` carries no more linearly extractable turn
   evidence, and the head is not the binding constraint after all.
4. P4's executed curvature stays 0.0 with a live control ⇒ gate 1 is not the binding gate at
   plan level and the candidate set is.
