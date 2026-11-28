# refav1 GOAL MARGIN — the answer is the VOCABULARY, and the knob is now in the repo

**Agent:** TanitAD Architecture & Inference FlyWheel · **Date:** 2026-09-05
**Branch:** `agent/arch-inf-20260803` · **Pre-registration:** `SPEC.md` (banked first)
**Predecessor:** `…/2026-09-05-refav1-make-it-drive/` (gates, lever, ceiling)
**Register rows:** `D-REFAV1-VOCAB-QUANT`, `D-REFAV1-MARGIN-SCALE`,
`D-REFAV1-TURN-THRESHOLD`, `D-REFAV1-NAV-MAGNITUDE`, `D-REFAV1-SOFT-KAPPA`,
`D-REFAV1-HEAD-SURVIVES`, `D-REFAV1-VOCAB-DESIGN`

---

## The one-paragraph answer

The margin was the wrong object. **refav1 does not turn because the lateral goal vocabulary
has no gentle sustained curve** — `canonical_controls` can command exactly two *sustained*
curvatures, `0` and `±0.08` (R 12.5 m), because `NUDGE_*` and `LANE_CHANGE_*` are S-curves with
zero net heading change, while this corpus curves at **R 100–1000 m**. MEASURED: on **90.6 %**
of GT-turn windows `LANE_KEEP` is the **vocabulary-optimal** token, and the head emits it on
**85.2 %** of windows against a vocabulary-optimal **95.6 %** — **the head already turns more
than its vocabulary justifies.** ⇒ no decision rule and no re-fit of `lat_head` can reach a
normal road curve, and both re-fit attempts duly failed their pre-registered controls (one is a
**nav echo**, one is **shrink-to-zero**). What the head *does* do is detect real turns
(**AUC 0.88** at R ≤ 33 m) and choose L vs R (**AUC 0.90**), neither of which is a nav echo.
⇒ **the fix is a magnitude the goal is ALLOWED to command**, and it is now implemented,
flag-gated and parity-pinned as **`--goal-kappa-turn`** (7 tests, 285 passed, zero-flag
bit-identical). Plan-level arms were on the GPU at hand-off.

---

## P1 — the margin: CALIBRATION or SEPARABILITY?

**Answer: neither question was well posed, because the verdict flips with the turn definition —
and my own pre-registration did not pin it.** I name that rather than exploit it.

| turn threshold | radius | n turn | AUC | best bal-acc | verdict under `SPEC.md`'s 0.80 bar |
|---|---|---|---|---|---|
| 1e-3 (the inherited default) | 1000 m | 2247 | 0.7235 | 0.6886 | **SEPARABILITY** |
| 1e-2 | 100 m | 703 | 0.7970 | 0.7672 | SEPARABILITY |
| **4e-2 = the vocabulary crossover** | **25 m** | **211** | **0.8806** | **0.8200** | **CALIBRATION** |
| 8e-2 = `GOAL_KAPPA_TURN` | 12.5 m | 84 | 0.8457 | 0.7904 | CALIBRATION |

⛔ Shuffled-label AUC is flat at **0.498–0.504 at every threshold**, so the rise is not an
artefact of the shrinking positive class. The `1e-3` row reproduces the banked
0.7120 / 0.2045 / 0.0733 exactly on the stride-40 panel.

**The principled resolution is a fact about the code, not a choice made after seeing results:**
the turn definition should be the curvature the goal token actually commands
(`GOAL_KAPPA_TURN = 0.08`, `refa_v1.py:119`), whose L2 crossover against `LANE_KEEP` is the
**analytic 0.040** and the **measured 0.04101**.

### The margin distribution, and the control that voids the proposed objective

| scale-free quantity (n = 282, the banked panel) | value |
|---|---|
| AUC | 0.7120 (shuffled **0.5055 ± 0.0336**) |
| Cohen's d | 0.5902 |
| **overlap coefficient of the two class-conditional densities** | **0.5845** |
| median gap as a fraction of pooled SD | **0.329** |

⛔⛔ **`median_margin_gap_logits` — the criterion `MUST_WE_RETRAIN.md` proposed for Step 2 — is
not a property of the decision.** `lat_head` is `LayerNorm → Linear`; scaling that `Linear` by
`c` multiplies the gap by `c`. **MEASURED at `c = 3`: decode bit-identical 282/282, AUC
unchanged to < 1e-12, Cohen's d unchanged, gap × 3.0000 exactly.** ⇒ an objective phrased
*"widen the margin gap in logits"* is satisfiable by scaling the head and changing nothing.

### The `lat_logit_bias` operating curve, with the committed criterion

| `commit b` | decode turn recall | false turn on straight | balanced acc | Youden J |
|---|---|---|---|---|
| **0.0 (argmax, CONTROL)** | 0.2045 | 0.0733 | 0.5656 | 0.1312 |
| 0.75 | 0.2576 | 0.1000 | 0.5788 | 0.1576 |
| 1.25 | 0.5909 | 0.2600 | 0.6655 | 0.3309 |
| **1.5 — PRIMARY (max Youden J)** | 0.7955 | **0.4600** | 0.6677 | **0.3355** |
| ≥ 2.25 | 1.0000 | 1.0000 | 0.5000 | 0.0000 |

⚠️ **My PRIMARY pre-committed criterion selects a FLOODING point** (46 % of straight roads get
a turn). That is not a criterion failure — it is what a symmetric criterion says when the score
is weak, and it is itself the evidence for SEPARABILITY at the 1e-3 definition. The
**SECONDARY** committed point (max recall at false-turn ≤ 0.10) is `b = 0.75`: recall 0.2576 at
false 0.1000 — a real but small move.

⇒ **Neither operating point is the deliverable, because the whole family is bounded by a
vocabulary that cannot express the road.** That is P1's real finding.

## P2 — was a re-fit needed? **The re-fit was re-specified, attempted twice, and FAILED both times.**

`ROOT_CAUSE.md` forced an amendment: a classifier over a vocabulary that cannot express a 200 m
curve cannot fix a 200 m curve. Step 2 became **predict a curvature MAGNITUDE** on the banked
`intent` (4786 windows / 141 episodes; extraction control C1 reproduces the banked `lat_logits`
with **max abs diff 0.0** and argmax **282/282**, so these are the same forward pass).

| attempt | outcome | the control that decided it |
|---|---|---|
| **soft posterior** `κ̂ = K·(p[TURN_L] − p[TURN_R])`, zero new parameters | ⛔ **FAILS** | shuffled-logit control **r = +0.0068 ± 0.0691** vs a real **r = −0.0999** — on the WRONG side; RMSE beats the ZERO floor only by shrinking (mean\|κ\| 0.0108 → 0.0031, R² 0.034) and it turns on **100 % of straight windows** |
| **trained ridge** `Linear(256→1)` on the frozen `LayerNorm(intent)`, episode-disjoint 85/28/28 | ⛔ **FAILS** | beats a CONSTANT on the decision that matters by only **+0.019**; and **R² 0.1165 → 0.0064 under nav shuffle — 94.5 % of its skill is nav**, and nav is supplied from the ego's own future path |

⚠️ **In both cases my verdict function said SUCCEEDS and the controls said FAILS. The controls
win, and that is recorded** — a verdict rule that can be wrong is itself worth banking.

⚠️ **Scoped honestly:** permuting nav *randomises* rather than *removes* a channel, so the nav
result is a necessary-condition failure, **not** a licence to say "the vision carries nothing".

## P3 — admissibility, MEASURED not assumed

* `plan()` builds the goal head's input at `refa_v1.py:2156` as `_run_brains(pooled_win,
  nav_cmd)` — **`ego=None`**. At inference the goal path sees **vision + nav only**.
* The **situation classifier's output enters nowhere**; no arm in this package computes one.
* The **target** (`gt_kappa`) is derived from the ego's future poses — a LABEL, and labels may
  use ego (PI 2026-08-03).
* ⚠️ **nav is weakly route-informative and that is INHERITED, not introduced here:**
  `P(GT turn | nav)` 0.411 / 0.500 / 0.592 against a base rate 0.468; **nav-only AUC 0.5967**
  against the head's **0.7120**. The bank carries `intent` twice (true and shuffled nav,
  2455/4786 rows changed) so any fit can be scored under both.

## P4 — does gate 1 open? **RUNNING at hand-off.**

⭐ **The zero-GPU half is already decisive, and it inverts the premise.** At the vocabulary
crossover (`|κ| > 4e-2`, R ≤ 25 m, **n = 211**) the **SHIPPED** head already decodes a
curvature-carrying token on **74.4 %** of windows at a **12.0 %** false-turn rate. **Gate 1 is
largely OPEN where a turn is the right goal.** The 20.5 % "turn recall" that motivated the
decision-rule programme is a statistic about 1000 m curves the vocabulary cannot express.

⇒ the binding constraint on *"does refav1 turn"* is **gate 2**, and P4 measures it directly:
3 arms on a turn-dense 8-episode panel (40 windows, 17 with `|κ| > 0.04`), stride 16,
paired window-for-window —

| arm | prediction, committed before the run |
|---|---|
| **A** `cos` + argmax (the SHIPPED DEFAULT) | executed curvature **exactly 0** |
| **B** `ccos` + argmax | **non-zero** on GT-turn windows |
| **C** `ccos` + argmax, `--plan-seed 1` | the **SEED REPLICATE** noise floor (`H-ESTIM-SEED-1`) |
| **D** `ccos` + `--goal-kappa-turn 0.02` (queued) | executed \|κ\| moves toward **0.02** if the vocabulary is the plan-level constraint |

⛔ **Every "exactly 0.0" claim carries `ha0_ext` — the non-planner integrator banked in the
same npz — as a same-breath control that must read NON-ZERO** (predecessor: 98.9 %, absmax
0.603). ⚠️ `ha0_ext` here is the arm's own `hold_ext_controls`, the canonical integrator
(decision **M11** / `D-MM-ADJ-1`), never `echo_gate.ha0_ext`. Analyser: `tools/p4_turn_gate.py`,
which **refuses** (`INCONCLUSIVE`, not a zero) if the control column is dead or the GT join is
empty. The banked zero-GPU decode prediction for this exact window set (**0.8235** on real
turns, **0.3478** on straights) is carried in `raw/p4_episodes.json` and re-checked by the
analyser, so the run must be on the windows it claims.

## P5 — the four families at T1

⛔ **NOT CLAIMED. No ADE and no four-family number is quoted from this package**, because the
arms had not landed at hand-off and the lookup shortcut does **not** hold on turn windows
(`GATE_ON_REAL_MODEL.md` §3), so nothing can be computed without re-planning. The seed
replicate (arm C) is in the run set precisely so that any lever claim can be read against the
rig's own run-to-run noise floor rather than a one-seed separated CI.

⚠️ Also banked for whoever picks this up: **the predecessor's 30-episode `ab` run is DEAD** —
its log stopped at 15:23 local at episode 10/30 and no `refav1_arm.py` process matches it. Its
partial dumps are at `C:/Users/Admin/refav1_drive/ab/dump_argmax` and are recoverable with
`--analyze-only`; the GPU time already spent need not be re-paid.

---

## ⭐ What was BUILT, not just measured

| deliverable | where | state |
|---|---|---|
| **`--goal-kappa-turn`** — the vocabulary's sustained curvature made a knob | `stack/tanitad/refs/refa_v1.py`, `taniteval/tools/refav1_arm.py` | committed, **zero-flag bit-identical** |
| parity pin, 7 tests incl. a forced-TURN control that stops the file being vacuous | `stack/tests/test_refa_v1_goal_kappa.py` | **285 passed, 1 skipped** (278 before) |
| a bug I shipped and then fixed: a bare `%` in argparse help kills `--help` | `stack/tests/test_argparse_help_percent.py` | class-level static guard, 3 tests |
| the `intent` bank (goal-head input, 4786 × 256, + nav-shuffled twin) | `C:/Users/Admin/refav1_margin/intent_stride2.npz` | ⚠️ **dev box only** |

## ⛔ Escalations

1. ⭐ **Three sustained curvature levels, sited at the corpus's own quantiles.** PRICED at zero
   GPU (`ESCALATION.md`): **100 % of real turns expressible instead of 38.7 %**, median
   curvature error on a turn **−3.7×**, at the **same RMSE** as today. The head's 8 slots
   already contain 3 never-emitted tokens, so it needs no new parameters.
2. ⛔ **A finer vocabulary without `ccos` changes nothing that reaches the wheels** — under the
   shipped `cos` a correctly decoded turn is refused on 38/38 windows. **The two must land
   together**, and the `ccos` hold-branch fix (predecessor escalation 2) is a prerequisite.
3. ⚠️ **`gt_kappa` is not a curvature below ~1 m/s.** 13 windows (0.27 %) read `|κ| > 0.2`,
   **92.3 % of them at v0 < 1 m/s**, max **3.565 1/m = R 0.28 m** — narrower than the car — and
   they carried the **entire** RMS (0.15827 raw vs 0.02192 clipped). **Any `yaw_rate/speed`
   curvature metric in the programme needs a speed floor on the WINDOW**, not just inside the
   ratio. My first run of `vocab_design.py` reported the raw figure and its table was
   meaningless; blast radius checked, every other result here is rank-based or already clipped.
4. ⛔ **`D-REFAV1-DRIVE-CEILING` needs two corrections applied**: its Step-2 prescription
   ("re-fit the goal head") is superseded, and its success criterion
   (`median_margin_gap_logits`) is retracted as scale-dependent. Both rows are already in
   `GOALS_AND_CLAIMS.md`.

## Deliverable manifest

| artifact | location |
|---|---|
| `SPEC.md`, `ROOT_CAUSE.md`, `STEP2_RESULT.md`, `ESCALATION.md`, `RESULT.md` | `repo:TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-goal-margin/` |
| `tools/` — `margin_anatomy.py`, `vocab_fit.py`, `threshold_sweep.py`, `vocab_design.py`, `extract_intent.py`, `kappa_head.py`, `soft_kappa.py`, `p4_turn_gate.py`, `run_p4.sh` | same package |
| `raw/` — `margin_anatomy_{s40,s5}.json`, `vocab_fit_{s40,dense}.json`, `threshold_sweep_{s40,dense}.json`, `vocab_design.json`, `kappa_head_fit.json`, `soft_kappa_s40.json`, `intent_stride2_prov.json`, `p4_episodes.json` | same package |
| the code: `refa_v1.py`, `refav1_arm.py`, `test_refa_v1_goal_kappa.py`, `test_argparse_help_percent.py` | `repo:stack/`, `repo:taniteval/` |
| six register rows | `repo:Project Steering/GOALS_AND_CLAIMS.md` |
| ⚠️ **`intent_stride2.npz` (48 MB) and the P4 dumps** | **`C:/Users/Admin/refav1_margin/` — DEV BOX ONLY, in ONE place** |
| ⛔ `kappa_head.pt` | dev box; **NOT deployable**, retained only as the artefact the FAILS verdict refers to |

---

## ⚠️ ADDENDUM — the banked `cos` dump does not record its cost weights

**Row:** `D-REFAV1-COS-WEIGHTS-UNRECORDED` · **Class:** MEASURED (a provenance fact)

`D-REFAV1-DRIVE-GATE2`'s headline — *"under `--cost-metric cos` a correctly decoded TURN is
executed on **0 of 38** windows, while `ccos` executes 63.2 %"* — compares two banked dumps.
Reading their manifests:

| dump | `cost` block in `manifest.json` |
|---|---|
| `refav1_sweep/dumps/dump_cos_ext` | ⛔ **absent — there is no `cost` key at all.** Its top-level keys are `action_units, arm_meaning, arms, corpus, episodes, first_plan_s, floors_added_post_hoc, goal, grid, hold_action_rule, hold_v0_rule, model, nav_shuffle, plan_cfg, plan_source_names, speed_channel, tiers, tool, wallclock_s` |
| `refav1_sweep/dumps/dump_ccos_comp` | `metric ccos`, weights **`{W_JERK 12.859, W_KAPPA 32.149, W_VEND 64.297}`**, `weights_source "CLI override"`, `shipped_weights {0.02, 0.05, 0.1}` |
| `refav1_sweep/dumps/dump_ccos_naive` | `metric ccos`, weights **`{0.02, 0.05, 0.1}`** — the shipped triple |

⇒ **The `cos` arm predates the cost-provenance stamping and cannot be shown, from the record,
to have run the same weights as the `ccos` arm it is compared against.** And the two `ccos`
dumps differ in `W_KAPPA` by a factor of **643** (32.149 vs 0.05), so the weights on this
surface are demonstrably not a constant across banked dumps.

⚠️ **`W_KAPPA` is the curvature penalty.** A comparison that concludes *"the metric refuses
turns"* between one arm with an unrecorded curvature penalty and one with a recorded one is,
on the record alone, **confounded between the metric and the penalty**.

### ⛔ What this does and does NOT retract

* It does **NOT** retract gate 2. `D-REFAV1-DRIVE-GATE2` is **also** supported by
  `tools/assert_metric_gate.py`, a controlled comparison inside ONE script on a tiny real
  `RefAV1` at four search sizes (`cos` 0/12, `ccos` 8/12 at every size). That arm is
  weight-matched by construction and is unaffected by this.
* It **does** attach a caveat to the specific **0/38 dump number**: its cost configuration is
  not in the record, so it should not be quoted as a weight-matched `cos`-vs-`ccos` contrast.
* ⚠️ **And it is a live question, not a bookkeeping one.** This package's own `cos` arm runs
  with the weights **explicitly recorded** as `{0.0, 0.0, 64.297}` — i.e. **`W_KAPPA` exactly
  zero** — and on the first 5 windows executed non-zero curvature on 2/2 GT-turn windows with
  `|κ|` up to **0.149**. ⛔ **n = 2: nothing is claimed.** The full 40-window arm is the
  measurement, and it is the first `cos` arm whose weights are in its own manifest.

⭐ **The durable fix is already in the code** — `refav1_arm.py` now stamps
`manifest["cost"]` with metric, weights, weights_source and the shipped triple, and
`manifest["goal_rule"]` with `lat_logit_bias` and `goal_kappa_turn`. Every arm banked from here
carries its own configuration. *Same family as the anchor-units trap: an artifact opened in
isolation must state what it was made under, because it is opened in isolation far more often
than its run record is.*
