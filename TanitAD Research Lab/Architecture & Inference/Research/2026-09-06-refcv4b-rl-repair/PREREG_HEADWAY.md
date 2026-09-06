# PRE-REGISTRATION 2 — `H-RL-HEADWAY-QUANTILE-1`: the next lever, named in advance and now run

**status: PRE-REGISTERED. ⛔ WRITTEN AND COMMITTED BEFORE `_headway` IS TOUCHED.**
**date:** 2026-09-06 · **owner:** RL reward-repair stream · **GPU: 0.**

⭐ **THIS EXPERIMENT WAS NAMED BEFORE ITS TRIGGER FIRED.** `PREREG.md` §3.4 committed, in
advance, that if `P1-PRED-A` held while `P1-PRED-B` failed then *"`progress` was NOT the
mechanism"* and the next candidate is **`headway`** — *"the cheapest next experiment is to
re-read `headway` at a low QUANTILE of the per-step time gap instead of the single worst
step."* That is exactly what happened (`RESULT.md` §2), so this is the pre-registered
continuation and not a new hypothesis chosen after seeing which one looks promising.

---

## 1. The mechanism, and why it is now the LARGEST measured term

MEASURED this turn on the same 6,089 windows / 73 episodes, after the `progress` repair
(`cap=lead`), per-term weighted mean gap `hold_v0 − human`:

| rung | `progress` | **`headway`** | `collision` |
|---|---|---|---|
| all | −0.007737 | **−0.004818** | −0.000821 |
| ≤3.0 s | −0.001264 | **−0.011256** | −0.001602 |
| ≤2.0 s | +0.000112 | **−0.015899** | −0.003173 |

⇒ With `progress` repaired, **`headway` carries essentially the whole gap at the conflict
rungs** — 142× `progress` at ≤2.0 s — while the RATE still reads 0.5546. That is the
rate/mean divergence with one term left holding it.

**THE CLAIM.** `_headway` reduces the per-step time gaps with **`amin` — a MIN-OVER-N
ORDER STATISTIC over the horizon's steps**. A minimum fires on the single worst step, so
the term is **RARE AND LARGE**: it says nothing in most windows and a great deal in a few.
That is the shape that makes a mean and a rate disagree. ⭐ It is also the *same family of
defect* as `fan_floor@k` read alone — a min/max-over-N quantity answering a narrower
question than the one it is quoted for — which is why the instrument built on 2026-09-06
carries `rand_best@k` beside it.

⭐ **MEASURED, in a unit test rather than argued** (`test_rl_progress_leadcap.py::
test_headway_is_blind_to_this_perturbation_and_that_is_the_next_candidate`): moving three
of five lead samples by up to 5 m leaves `_headway` **bit-identical**, because the worst
step is elsewhere — while a **quantile over the same per-step gaps does move**.

## 2. ⛔ THE REPAIR — one variable

`ctx["headway_reduce"]`:

* `"min"` — **the default and the legacy behaviour, bit-identical**;
* `"q<Q>"` — the Q-quantile of the per-step time gap instead of its minimum, e.g. `"q0.25"`.

Nothing else changes: the same `t_star`, the same asymmetric shape around it, the same
`lead_len_m`, the same veto (TTC stays a VETO, untouched). ⛔ The reduction is over the
horizon's **STEPS**, all of which are POSITION queries on the lead's recorded track — no
closing rate enters the ranking, exactly as `PREREG.md` §2 requires.

**Q is fixed at 0.25 before any number is computed**, chosen as the standard lower-quartile
reading and NOT swept. ⛔ A Q chosen after seeing which one passes would be the goalpost
move this file exists to avoid; if 0.25 fails, the finding is that it fails.

## 3. ⛔ THE PREDICTION — both outcomes, same statistics as `PREREG.md`

Same ladder, same rows, same weights, same episode-cluster bootstrap (`n_boot` 4,000).
Baselines are **this turn's own `cap=off` column** (⚠️ not the 2026-09-05 bank — see §5):

| | all | ≤3.0 s | ≤2.5 s | ≤2.0 s |
|---|---|---|---|---|
| rate | 0.4418 | 0.4854 | 0.4866 | **0.5482** |
| SKEWSPLIT = median − mean | 0.009328 | 0.009561 | 0.011710 | **0.014943** |
| Σ DIV over 8 rungs | **6** | | | |

> **P2-PRED (`headway_reduce="q0.25"`, `progress_lead_cap` OFF — ONE variable):**
> 1. **`Σ DIV` falls below 6**, and
> 2. **`SKEWSPLIT` falls in magnitude at ≥ 2 of the three conflict rungs** (≤3.0 / ≤2.5 /
>    ≤2.0 s), and
> 3. **`rate(≤2.0 s)` falls** from 0.5482.
>
> ⛔ **THE OTHER OUTCOME, written now:** if these fail, then **neither ranking term is the
> generator** and the asymmetry is a property of the *population* — the informative windows
> are a skewed minority whatever the terms do — in which case the honest next step is a
> **magnitude-aware statistic pre-registered on its own** (the Master Mind's third option),
> and I say so rather than trying a third term.
> ⛔ A third combined arm (`lead` + `q0.25`) is scored and reported for completeness but is
> **NOT** the pre-registered test: it moves two variables.

## 4. ⛔ CONTROLS THAT MUST READ KNOWN VALUES

1. **`headway_reduce="min"` must reproduce the legacy term EXACTLY** (≤ 1e-12, all three
   sides, all 6,089 windows) — the channel control.
2. **DEGENERACY** — the `headway` spread `|hold − human|` must not collapse. A term made
   inert would move the rate for the wrong reason and is reported as a **DEGENERATE
   REPAIR**, not a pass.
3. **`frozen` stays far below the human** at every rung.
4. **The quantile must actually bind**: report the fraction of windows where
   `q0.25 != min`. If that fraction is ~0 the lever did nothing and no verdict is
   admissible.

## 5. ⚠️ A BASELINE CORRECTION THIS EXPERIMENT INHERITS — stated, not smoothed over

`PREREG.md` §4.1 required the cap-OFF re-run to reproduce the banked all-window mean gap
**−0.011239379601720929** to ≤ 1e-12. **It did not: the re-run reads −0.011732071340923,
err 4.927e-04.** The rate control reproduced **EXACTLY (0.000e+00)**.

⭐ **The cause is identified to the last digit, not waved at.** A per-window comparison of
all five components across all three sides (**18,267 evaluations**) finds
`progress`, `headway`, `feasibility`, `comfort` identical to **exactly 0.0**, and
`collision` differing on **13 of 18,267** — each by a full 1.0, all in the same direction
(the term now FIRES where it did not), in 2 episodes. **3 of those 13 are on `hold_v0`
alone**, and `3 / 6089 = 4.927e-04` — **the discrepancy exactly**.
⇒ `rewards._collision` changed after the 2026-09-05 bank was written. The banked mean is a
**stale baseline**, the banked *rate* is not affected, and the correct comparator for every
statistic here is **this turn's own cap-OFF column**, computed with the same code on the
same day. That is what §3's baseline table uses.

## 6. Estimator, tier, scope

Episode-cluster bootstrap, `n_boot` 4,000 — ⛔ it answers **only** *"would another draw of
EPISODES say this?"*. No arm is trained and no planner samples, so training variance
(`H-ESTIM-SEED-1`) and inference variance do not enter. Tier **T0** instrument probe,
NON-PARITY RL-fit windows. Evidence class MEASURED (ours).
⛔ **`G-REWARD` remains FAILED on its committed statistic.** Nothing here re-scores it.
