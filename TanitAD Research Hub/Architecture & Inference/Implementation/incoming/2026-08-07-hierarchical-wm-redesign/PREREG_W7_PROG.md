# Pre-registration — W7-PROG: does the anti-degeneracy progress term rescue selection?

**Written 2026-08-12 ~02:15Z, BEFORE the run. Both outcomes are bound below.**
**Evidence class of everything quoted here: MEASURED (ours), artifact paths given.**

---

## 1. Why this experiment exists

W7-FULL failed the pre-registered gate: **3.3348 m against a 0.4505 m threshold**, with **no
shortlist** (256/256 candidates, so the true best was always available) on the repaired trunk with
a refit head whose fan oracle is **0.1273** (`w7_full_gate.json`). The follow-up 0-GPU sweep
(`w7_selection_rules.json`) found:

- the argmin's **error-rank is 132.3 of 256** — the median, i.e. selection is no better than
  picking at random from the fan;
- the **mean error inside the top-m is FLAT at ~5.32 for every m** — which is the fan's own mean,
  so the roll-consistency cost carries essentially no information about candidate quality;
- but **top-m ceilings do fall** (0.356 at m=32), so a good rule over a top-m set exists.

And one configuration fact: **the anti-degeneracy progress term `--w-prog` has been weight 0.0 in
every W7 run to date** (`w7_roll_rerank.py:195`, `W_PROG_DEFAULT = 0.0`).

**The theory that makes this the first test.** The roll cost measures self-consistency: how well
the predictor's rollout agrees with itself under the candidate's controls. A self-consistency
objective has a **trivial minimiser — the near-stationary candidate**, because a plan that barely
moves is easy to be consistent about. `progress_arc_length` exists precisely to penalise that, and
it has been switched off. The observed signature (argmin at the median error-rank, top-m mean flat
at the fan mean) is exactly what a cost dominated by a degenerate minimiser looks like.

This also converges with tonight's T1 finding from the opposite direction: at T1 the closed loop
**over-**shoots (progress ratio 1.7279, speed bias +9.3892 m/s). A selector biased toward
**under-**progress and a rollout biased toward over-progress are two different failures, and this
run separates them rather than assuming they are one.

## 2. Hypothesis

**H-PROG:** the roll-consistency cost's argmin is dominated by low-arc-length (near-stationary)
candidates. Enabling `--w-prog` will move the argmin's error-rank materially below 132/256.

## 3. The run

Same fan, same 881-window grid, same repaired trunk + W4r head, selector-free (`--topk 256`) —
**only `--w-prog` changes**, so the contrast is attributable. Arms:

| arm | `--w-roll` | `--w-kin` | `--w-prog` |
|---|---|---|---|
| `w7-prog-0` (control, = W7-FULL) | 1.0 | 0.2 | **0.0** |
| `w7-prog-lo` | 1.0 | 0.2 | **0.1** |
| `w7-prog-hi` | 1.0 | 0.2 | **0.5** |

Two non-zero weights because a single one cannot distinguish "the term does nothing" from "the
weight was wrong", and three arms is still one short GPU job.

## 4. ⛔ Both outcomes, bound in advance

**PRIMARY endpoint — the argmin's mean ERROR-RANK over the 256-candidate fan** (not ADE). This is
deliberate: ADE can move for reasons that have nothing to do with whether the cost identifies good
candidates, and the mechanism under test is a *ranking* claim.

| outcome | condition | what we conclude, and what we do |
|---|---|---|
| ⭐ **CONFIRM** | error-rank falls **below 100/256** on at least one non-zero arm, AND that arm's ADE improves on the 3.3348 control | The degenerate-minimiser account is supported. Re-measure on a **fresh grid** before any deployment (the sweep that motivated this is stamped EXPLORATORY and re-used the W7 scoring windows), then re-run the gate. |
| **PARTIAL** | error-rank falls below 128 but not below 100, or falls without ADE improving | The term is real but not sufficient. The cost needs a goal-conditioned component, not a bigger anti-degeneracy weight — the V-JEPA-2-AC / DINO-WM goal-cost contrast in the paper §3.12 becomes the next lever, and W7-style self-consistency selection is retired as a headline route. |
| ⛔ **REFUTE** | error-rank stays at or above 128/256 on **both** non-zero arms | The degeneracy account is WRONG and I retract it. The cost is then uninformative for a reason not yet identified, and no further weight-tuning on this cost is admissible without a new mechanism hypothesis. Logged to `RETRACTION_LOG.md` with root-cause class "plausible mechanism, unfalsified before use". |

**Secondary, reported but never the gate:** selected ADE, the P7 calibration block
(Spearman with episode-cluster bootstrap CI), the mean within-window rank correlation, and the
realised **arc-length distribution of the selected candidates** — the last is the direct check on
whether the term did what it is supposed to do, independent of whether that helped.

⚠️ **What this run may NOT be used for.** It re-uses the W7 scoring windows, so it is
**EXPLORATORY by construction**: no arm here may be quoted as a v5.8f number, and a winning weight
must be re-measured on a fresh grid before deployment. Any registry row from this run carries that
stamp.

## 5. Falsifiability note

The REFUTE row is not decoration. The degenerate-minimiser story is *my* reading of the flat top-m
curve, and this programme has had three root-cause readings falsified in one session before
(CLAUDE.md, the `git commit` segfault). If both non-zero arms leave the error-rank at the median,
the correct response is to say the mechanism is unknown — not to try `--w-prog 2.0`.
