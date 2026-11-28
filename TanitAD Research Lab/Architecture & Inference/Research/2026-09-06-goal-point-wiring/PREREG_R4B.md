# PRE-REGISTRATION — `H-R4B-1`: the shape defect is in the DECODER OUTPUT PARAMETERISATION, and a curvature-penalised refinement removes it at no ADE cost

**Written AFTER the diagnostic and BEFORE the confirmation, and that is stated rather than
hidden.** The mechanism (§2.1 of `RESULT.md`) was confirmed against numbers published by a
different probe. The FIX (§2.2–2.3) is a **post-hoc finding on the same dump**, held out
internally by an episode split but never confirmed on a **different arm**. This file commits the
confirmation before it runs.

**Author:** goal-point wiring + R4b agent, 2026-09-06. **Branch:** `agent/arch-inf-20260803`.
**Predecessor:** `…/2026-09-06-goal-point/PREREG.md` §7, which named `R4b` as the pre-registered
next lever *if P3 (curvature) fails* — and predicted, correctly, that it would.

---

## 0. THE CLAIM

**`H-R4B-1`:** the `os − ha0_ext` curvature gap is generated in the decoder's **free-waypoint
output parameterisation**, not in routing, selection, or the anchor vocabulary; and constraining
the refinement to a smooth basis removes most of that gap **without** giving back the ADE the
refinement buys.

MEASURED support already banked (`raw/R4B_PARAM.json`, T1, 4,823 windows / 141 episodes, paired
episode-cluster bootstrap B = 2000):

* `emitted − anchor`: ADE **−0.131669** and curvature **+0.004132**, both separated. The
  refinement halves ADE and doubles curvature error.
* the picked anchor is at the model-free floor (0.004019 vs `ha0_ext` 0.003712, `ha` 0.004030).
* held-out (70 eps / 2,392 windows, λ chosen on a disjoint 71-episode half): curvature
  **−0.001903 [−0.003359, −0.000587] SEPARATED**, ADE **−0.000211 [−0.000690, +0.000266] ns**.

---

## 1. THE ARMS — one variable each, ZERO GPU

The seam is an **inference-time projection** of the offset the decoder already emits, so no arm
here needs a training run.

| arm | the ONE variable |
|---|---|
| `emitted` | — the incumbent, `plan_full` as shipped |
| `r4b_ridge` | vs `emitted`: the refinement is replaced by `argmin_u ‖u − off‖² + λ‖D²u‖²` |
| `r4b_poly2` | vs `emitted`: the refinement is projected onto a degree-2 polynomial in time |
| ⛔ `ROUGHEN` | vs `emitted`: the removed component **doubled** — the DELIBERATE-REGRESSION arm |
| `FEASIBLE` | vs `emitted`: the SHIPPED Stage-0 kinematic projection, to price what we already have |

Trivial controls, mandatory, and the panel is VOID without them: `ha0`, `ha0_ext`, `ha` must
reproduce **0.6723 / 0.2874 / 0.2996** to < 5e-4, `emitted` must reproduce **0.2965 / 0.008150**,
and the picked anchor **0.4281 / 0.004019**.

---

## 2. ⛔ THE CONFIRMATION SET — a DIFFERENT arm, because this is a post-hoc finding

⛔ **The banked refcv4b navflip dump is SPENT for this claim.** λ was selected on half of it and
the effect was discovered on the other. The confirmation must run on a dump this hypothesis has
never seen. In priority order, and each is a re-analysis with **zero GPU**:

1. **refcv5**, when it finishes (ETA ≈ 2026-09-08 07:33 UTC) — a different checkpoint, a different
   training run, the same harness.
2. **`refcv3-40284-openloop`** (`taniteval/results/`) — already banked, a different arm.
3. the **6 s grid** of the same dump — the 8-slot plans are banked while `g` is the 2 s grid, so
   this needs the GT extension and is a different *statistic*, not a different arm. Reported as a
   robustness read, never as the confirmation.

---

## 3. COMMITTED ENDPOINTS — the bar, written before the confirmation runs

λ is selected on a **FIT half of the confirmation dump's episodes** by the rule already used:
*argmin curvature MAE on FIT subject to ADE(FIT) ≤ ADE_emitted(FIT) + 0.005 m.* The SCORE half is
scored, never tuned on.

| # | endpoint | SUCCESS criterion |
|---|---|---|
| **C1** | curvature MAE, `r4b − emitted`, SCORE half | **separated NEGATIVE** |
| **C2** | ADE 2 s, `r4b − emitted`, SCORE half | **NOT separated POSITIVE** (a tie or better; the ADE halving must be kept) |
| **C3** | turn-window lateral accuracy, `r4b − emitted` | **NOT separated NEGATIVE** |
| **C4** | ⛔ the `ROUGHEN` arm's curvature | **separated POSITIVE** — if the deliberate regression does not fail, C1 means nothing |

⛔ **FAILURE** is any of: C1 not separated; C2 separated positive; C3 separated negative; C4 not
separated. **A failure is reported as a failure.**

⚠️ **The four families travel with every row**, never ADE alone: LONGITUDINAL (step-wise speed
MAE, along-track MAE — and the reason the lead-agent block is unavailable on a dump that carries
no lead), LATERAL (curvature MAE **with `ha0`'s 0.006802 straight-line floor beside it**, heading,
cross-track), TACTICAL (turn-window lateral accuracy and per-class recall, trajectory-derived via
`refc_tactical.factor_from_kinematics` — ⛔ **never** the `|dyaw| > 0.15` gate, which the human
fails on 3 of 9), STRATEGIC (⛔ per-family "not computable, and why", never silently dropped).

---

## 4. ⚠️ ADVERSE PRIORS, COMMITTED IN ADVANCE

1. **The held-out margin will SHRINK again on a different arm.** It already shrank from −0.0030
   (full set, tuned) to −0.0019 (held out). A confirmation margin of ~−0.001 would still clear C1
   and should not be read as a weakening of the mechanism.
2. **C3 is the one most likely to fail.** `ridge_3`'s turn accuracy is −0.0139 with CI
   [−0.0343, +0.0027] — not separated, but the interval's mass is on the wrong side. If a
   confirmation makes it separated, the honest read is that the high-frequency component **does**
   carry some turn signal, and the next lever is a basis that is smooth in **curvature** rather
   than in **position** — not a smaller λ chosen after seeing C3.
3. **A trained-in-basis decoder is a DIFFERENT experiment** and is not pre-registered here. The
   projection arm is a **lower bound** on it (a trained decoder would spend the basis better than
   a projection of a free offset does); the ORACLE-in-basis arms bound it from above and are
   inadmissible as capability claims. If the projection seam ships and the trained arm is later
   wanted, it gets its own pre-registration.

---

## 5. WHICH VARIANCE THE INTERVALS ANSWER

Every arm is **one checkpoint's own output re-parameterised**, so both operands of every margin
come from the **same forward pass**. The episode-cluster bootstrap therefore answers exactly
*"would another draw of EPISODES say this?"* — and there is no training-run variance
(`H-ESTIM-SEED-1`) and no inference-sampling variance (`D-REFCV4B-SEED-SCOPE`) in the comparison
to answer for. ⛔ It carries no licence to compare one trained model to another.

---

## 6. WHAT WOULD RETIRE THIS HYPOTHESIS

* C1 not separated on two independent confirmation dumps ⇒ the full-set effect was the tuning, and
  `H-R4B-1` is REFUTED for the projection form.
* C2 separated positive ⇒ it **is** a Pareto trade after all, `shrink_α` is the honest family, and
  the finding is reported as a trade-off rather than a fix.
* C4 not separated ⇒ the curvature instrument is not measuring what the arms claim to move and the
  whole panel is VOID, confirmation included.
