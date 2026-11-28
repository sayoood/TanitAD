# ⛔ THE PANEL IS BLOCKED — and the fact it was built to explain does not survive a representative panel

**Row:** `D-VOCAB-L3-POWER` · **Class:** MEASURED, **zero GPU** · **Date:** 2026-09-05
**Executing:** `PREREG_D-VOCAB-L3_FLOOR_GAP` **ERRATUM-1 §E2** (the blocking power check)
**Instruments:** `tools/power_check.py`, `tools/floor_gap_anatomy.py`
**Raw:** `raw/power_check.json`, `raw/floor_gap_anatomy.json`
**Panels:** `dump_ccos_comp` (282 windows / 141 episodes, stride 40) · `abt` (28 windows /
14 episodes, stride 40) — both step 21,109, both `ccos`, both `ha0_ext` = the **INTEGRATOR**
form (M11). ⛔ **They do NOT share a cost triple** — `W_KAPPA` **32.149** vs **0.0** — and that is
carried through every comparison below rather than assumed away.

---

## The answer ERRATUM §E2 asked for

> *"compute the L=1 turning-window deficit to `ha0_ext` and its 50 % target first; if that
> target does not clear the ≈0.30 m inference-seed floor, the panel is UNDERPOWERED BY
> CONSTRUCTION and must be redesigned … before any GPU is spent."*

On the **representative** panel, at the **MEASURED** crossover |gt_κ| > 4e-2:

| quantity | value |
|---|---|
| turning windows | **10** of 266 valid (3.76 %) |
| `cl` ADE on turning | 0.8885 m |
| `ha0_ext` ADE on turning | 0.9513 m |
| **deficit `cl − ha0_ext` on turning** | **−0.0628 m** |
| 50 % target | **−0.0314 m** |
| inference-seed floor (`argmax@seed0` vs `@seed1`, same windows) | **0.1907 m** |
| **runnable as pre-registered** | ⛔ **NO** |

⛔ **The target is 6.1× SMALLER than the planner's own run-to-run noise, and it has the wrong
sign: there is no deficit on turning windows to shrink.** `cl` is **6.6 % BETTER** than the
`ha0_ext` floor there (ratio 0.934). A committed criterion of *"the deficit shrinks by ≥ 50 %"*
is not merely underpowered against this number — it is **undefined**, because a 50 % shrink of a
negative deficit is not a success condition.

⚠️ And **n = 10** windows over at most 10 episode clusters cannot carry an episode-cluster
bootstrap. Two independent reasons to block, either of which is sufficient.

## ⭐⭐ WHY — AND IT IS NOT THE THRESHOLD. IT IS THE PANEL **AND** A ZERO CURVATURE PENALTY

The prereg's §1 fact is *"both A/B arms sit **3.6–5.9×** BELOW the trivial `ha` / `ha0_ext`
floors on TURNING windows"*, and it comes from the 28-window `abt` A/B. `floor_gap_anatomy.py`
crosses PANEL × THRESHOLD, with the straight complement printed beside every turning cell:

| panel | thr | n_t | `cl` (turn) | `ha0_ext` (turn) | **ratio (turn)** | n_s | `cl` (straight) | `ha0_ext` (straight) | **ratio (straight)** |
|---|---|---|---|---|---|---|---|---|---|
| repr 282 | 1e-3 | 124 | 0.9786 | 0.8355 | **1.171** | 142 | 0.5218 | 0.2762 | 1.890 |
| repr 282 | 1e-2 | 34 | 1.3080 | 1.3211 | **0.990** | 232 | 0.6507 | 0.4220 | 1.542 |
| ⭐ **repr 282** | **4e-2** | **10** | 0.8885 | 0.9513 | **0.934** | 256 | 0.7287 | 0.5207 | **1.399** |
| abt 28 (argmax) | 1e-3 | 18 | 1.5628 | 0.4803 | **3.254** | 9 | 1.8510 | 0.3609 | **5.128** |
| abt 28 (argmax) | 1e-2 | 4 | 1.0886 | 0.3457 | **3.149** | 23 | 1.7580 | 0.4570 | **3.847** |
| abt 28 (argmax) | 4e-2 | 1 | 0.9358 | 0.2498 | **3.746** | 26 | 1.6867 | 0.4479 | **3.766** |
| abt 28 (prior050) | 1e-3 | 18 | 2.7581 | 0.4803 | **5.742** | 9 | 2.2375 | 0.3609 | **6.199** |

⛔⛔ **THE TWO PANELS DIFFER IN THE COST TRIPLE AS WELL AS IN THE WINDOWS, AND I AM NOT
ENTITLED TO ATTRIBUTE THE DIFFERENCE TO EITHER ALONE.** Read from each dump's own
`manifest["cost"]` (the instrument now prints it, so a reader cannot miss it):

| panel | metric | `W_JERK` | **`W_KAPPA`** | `W_VEND` |
|---|---|---|---|---|
| repr 282 (`dump_ccos_comp`) | `ccos` | 12.859 | **32.149** | 64.297 |
| abt 28 (both arms) | `ccos` | 0.0 | **0.0** | 64.297 |

⭐ **And the zero is now known to be a live defect, MEASURED the same day by another stream on a
third panel:** `D-REFAV1-P4-COS-THRASH` finds that at `(0, 0, 64.297)` the planner executes
curvature at the **`kappa_max = 0.2` clip bound on 90 % of STRAIGHT windows**, `cl` ADE **1.8944**
against `ha0_ext` **0.8772** (2.16×), and it **chooses** those plans (`baseline_won_frac 0.075`).
⇒ **the `abt` panel's uniform 3.2–6.2× is exactly what a planner with no curvature penalty
produces**, and that explanation covers the straights, which a lateral-vocabulary explanation
cannot.

**Three readings, all MEASURED, and the second and third are the ones that matter.**

1. The prereg's range reproduces on the `abt` panel — **3.15–5.74×** against `ha0_ext` across the
   three thresholds (the published 3.6–5.9 is within reach of any reasonable floor/threshold
   choice on that panel), so the number was not miscopied.
2. ⛔⛔ **ON THAT PANEL THE STRAIGHTS ARE JUST AS BAD — 3.77–5.13× for `argmax` and 5.81–6.20×
   for `prior050`.** A claim that says *"the planner is 3.6–5.9× below the floor **on turning
   windows**"* is, on its own panel, **equally true of straights**. ⇒ whatever produces it is not
   specific to turning, and **a sustained-curvature vocabulary cannot be it** — the vocabulary is
   not in play on a straight. This is control #2 of the instrument firing exactly as designed,
   and it is the reading that does NOT depend on comparing the two panels.
3. ⚠️ **The panel-to-panel difference itself is confounded** between window selection and
   `W_KAPPA`, and is reported as such. It is *consistent* with the thrash explanation and it does
   not isolate a cause. ⛔ An earlier draft of this document said *"the vector was the panel"*;
   that was more than the evidence supports and it is corrected here rather than quietly
   softened.

**Why the `abt` panel is not a sample of the road either:** its 14 episodes were selected because the
`prior τ=0.5` decode **changed** there (`.../make-it-drive/raw/ab_power_analysis.json`: 18 of 282
windows change; the targeted view keeps the episodes that carry them). It is selected on the
**LEVER**, and it is uniformly harder — `cl` ADE **1.61 m** there against **0.75 m** on the
representative grid.

## ⇒ Consequences for `PREREG_D-VOCAB-L3_FLOOR_GAP`

| the prereg says | status |
|---|---|
| §1 *"both arms sit 3.6–5.9× below the floors on TURNING windows"* | ⛔ **DOES NOT GENERALISE, AND IS NOT ABOUT TURNING.** True on the targeted `abt` panel (`W_KAPPA = 0`), where it is **equally true of straights** (3.77–5.13×); on the representative panel (`W_KAPPA = 32.149`) the turning ratio at the measured crossover is **0.934** — the planner **beats** the floor there. |
| §2 the mechanism: *a two-level sustained-curvature vocabulary must lose to a continuous ego-extrapolation floor on turns* | ⚠️ **NOT SUPPORTED BY THE FLOOR GAP.** The gap that exists on a representative panel is on **STRAIGHT** windows (+0.208 m, 1.399×), where no sustained-curvature vocabulary is in play at all. |
| §3 SUPPORTED ⇔ *"the deficit shrinks by ≥ 50 %, separated"* | ⛔ **NOT RUNNABLE.** The deficit is **−0.0628 m**; half of it is **−0.0314 m** against an inference-seed floor of **0.1907 m**, on **10** windows. |
| §4 *"it does not claim refav1 will drive"* | unchanged, and now doubly so. |

⭐ **This is the ERRATUM working.** §E2 was added precisely so a null from an underpowered panel
could not be produced and then quoted; the check it mandated fired before any GPU was spent, and
it found something stronger than low power — a motivating fact that does not generalise off the
panel it was measured on, and whose most likely cause (a **zero curvature penalty**) is not the
mechanism the prereg named. ⚠️ **`D-MM-VOCAB-1`'s approval of L=3 is untouched**: it rests on the oracle-chooser
design table (`vocab_design.json`), not on the floor gap. What is withdrawn is this document's
parent's *explanation* of the floor gap.

## ⛔ What must NOT be concluded

* **Not** *"turning is fine"*. Ten windows is ten windows; the point estimate 0.934 carries no
  interval here and none is claimed. The honest statement is that **the evidence for a turning
  floor gap is absent on a representative panel**, not that its absence is established.
* **Not** *"the vocabulary does not matter"*. The M15 oracle table stands on its own evidence,
  and the goal-margin stream's own self-correction (`ESCALATION.md` §4) already prices the
  realised gain from the magnitude alone at **2.3 %**, with the binding term being the head's
  recall.
* **Not** *"the 1e-3 threshold was the error here"*. It is still forbidden (R 1000 m), but on the
  `abt` panel the ratio barely moves with it (3.25 → 3.15 → 3.75). **The threshold is not the
  vector; the panel and its zero curvature penalty are the candidates, and they are not
  separated here.**
* **Not** *"the `abt` A/B is void"*. It is a PAIRED contrast in which both arms share the triple,
  so `D-MM-VOCAB-3`'s lever conclusion is unaffected. What is void is reading its ABSOLUTE level
  as a fact about the road.

## The redesign §E2 demands

1. **A turn-enriched panel.** `refav1_margin/p4` is one: 8 episodes, 40 windows at stride 16,
   **17 real turns at |gt_κ| > 4e-2** — a 4.5× higher turn density than the representative grid,
   built by the goal-margin stream for exactly this reason.
2. **Both inference seeds**, per §E1 — the floor must be built the same way for both arms.
3. **The saturation readout** of §E3 in both arms, so a shrink can be attributed.
4. ⛔ **And a CHOOSER for the level set** — see `L3_IMPLEMENTATION.md` §3. Without one there is
   no L=3 *arm* to run, only an L=3 *vocabulary*.

## Evidence class and scope

**MEASURED** throughout, from banked dumps, **zero GPU**. Controls: the floors read no plan seed
and are **BIT-IDENTICAL** between the seed-0 and seed-1 dumps (`max_abs_diff` **0.0** for `ha`,
`ha0` and `ha0_ext`, same grid asserted) — so a moving floor is excluded rather than assumed;
`n` is printed for every cell; the straight complement accompanies every turning cell; windows
with **v0 < 1 m/s are excluded and counted** (16 of 282, 1 of 28), because `yaw_rate/speed` is
not a curvature below the speed floor — the defect that swamped an RMS by 7× in the goal-margin
package. One checkpoint (21,109), one corpus.
