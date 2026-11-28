# PRE-REGISTRATION — E1d: WiSE-FT α-frontier over the E1c CL-SFT

**Written and committed BEFORE the run produced any number.** Launched 2026-07-28 on pod3
(idle; pod1 and pod2 untouched). Standing directive **D-A** ("loop until significant closed-loop
performance"), acting under the PI's standing authorization.

## 1. Why this experiment, and why NOT "train longer"

E1c's verdict is **BOUND**: 0 of 17 frontier points satisfy all six conditions.
- **P1 & P2** (corridor-departure@K185, overall **and** junction, paired, CI-separated **LOWER**)
  fire at **15/17** points and are multiplicity-robust at all 15. At step 2750 departure falls
  **0.5877 → 0.147 (Δ −0.4407)** and junction **0.8414 → ~0.40 (Δ −0.4414)**.
- **Guardrail Ga** (open-loop ADE@2s not separated-higher) holds at **0/17**. So does Gb1
  (anchor accuracy) and Gb2 (anchor-trajectory L1).

⚠️ **The obvious next step is ruled out by the data, not by opinion.** The open-loop cost does
**not** keep shrinking with training: it falls 0.5048 → 0.2197 over steps 500–2250 and then
**PLATEAUS** — 0.2083, 0.2158, 0.2133, 0.1893, 0.2026, 0.1969, 0.1947. Eight consecutive points
inside ±0.02 with no trend. **More steps cannot close Ga.** *(A first reading of this table as
"still declining" was wrong and is corrected here before it could motivate a GPU commitment.)*

Weight-space interpolation traces a **different curve** than early stopping. **PUBLISHED
precedent:** WiSE-FT (Wortsman et al., 2022) — interpolating a fine-tuned model with its base
dominates early stopping on the robustness/accuracy frontier. Our application to a closed-loop /
open-loop trade is **MEASURED here, not inherited**.

## 2. Design

`w(α) = (1−α)·w_base + α·w_ft` over the 91 trainable keys of
`delta_step04000.pt` (90 actually differ from base), fp64 accumulate, cast back to source dtype.
Non-floating tensors are copied verbatim from the FT and that is stated rather than silently rounded.

α ∈ {0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.85, 1.00}.

⭐ **The evaluator is NOT modified.** α is encoded in the step number (α·100), so
`e1c_eval.py` adjudicates these points with **literally the same code**, the same
`e1c_common.evaluate_point` / `render_verdict`, the same strata and the same multiplicity
correction as E1c's own frontier.

**Estimator:** paired episode-cluster bootstrap over the **44 held-out episodes**
(`taniteval/ci.py`, B=2000), resampling EPISODES. `overlapping_holdout_se` is used nowhere.
Val: `physicalai-val-heldout-79d4e3d2d4c6`; corridor half-width 1.75 m; junction 10°; stride 8.

**CONTROL (must pass or nothing else is quotable):** α=1.00 is bit-identical to the source delta
(asserted at build time) and must reproduce E1c frontier row 4000 —
`dep_overall −0.4274`, `dep_junction −0.4270`, `ade2s +0.1947`.

## 3. Both outcomes, committed in advance

**OUTCOME A — SUCCESS POINT EXISTS.** At least one α satisfies **P1 ∧ P2 ∧ Ga ∧ Gb1 ∧ Gb2 ∧ Gc**
on held-out data, multiplicity-robust.
⇒ That α is the **D-A deliverable**: the first materially-better closed-loop number in the program
with no open-loop regression. It becomes the REF-C closed-loop arm, is registered with its α, and
the next CL step builds on it. **Interpretation:** the CL/OL trade was a property of the *training
path*, and a better point on the weight-space segment escapes it.

**OUTCOME B — BOUND AGAIN.** No α satisfies all six.
⇒ **This is a real result, not a null.** It says the trade is a property of the **direction in
weight space**, not of how far along it we stopped — the two objectives genuinely conflict along
this segment. ⇒ **The lever is then the OBJECTIVE, which is exactly where E2a already pointed**
("the lever is the objective, not the encoder or the denoise steps"). Next step becomes a
*different loss* (e.g. supervising only failure-adjacent anchors while constraining the open-loop
head, or an explicit multi-objective constraint), **not** another point on this path.
**We do not re-run this segment with a finer α grid** — a denser grid on a monotone segment cannot
change the verdict, and pretending otherwise is how a null becomes an endless sweep.

**Reported either way**, with the estimator named, per corpus, no pooling.

## 4. What this does NOT claim

- It does not touch the **K=20** instrument, which is reported and **NON-DECIDING**.
- It does not revisit E1a's horizon finding (C6) or E2a's perceivability result.
- α is a *deployment* knob here, not a training result; a winning α still owes a from-scratch
  confirmation before it enters any headline.

## 5. Artifacts

`pod3:/workspace/e1c/alpha_sweep/` (deltas + `ALPHA_MANIFEST.json`),
`pod3:/workspace/e1c/make_alpha_deltas.py`, `pod3:/workspace/e1c/run_e1d_alpha.sh`,
result → `pod3:/workspace/e1c/e1d_alpha_result.json`. Terminal marker `E1D_ALPHA_RUN_DONE`.
