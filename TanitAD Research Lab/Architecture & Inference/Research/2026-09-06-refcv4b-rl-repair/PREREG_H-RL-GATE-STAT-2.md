# PRE-REGISTRATION 5 — `H-RL-GATE-STAT-2`: the same statistic, the same bar, on the REWARD AN ARM WOULD ACTUALLY TRAIN ON

**status: PRE-REGISTERED. ⛔ WRITTEN AND COMMITTED BEFORE THE CELL WAS COMPUTED ON ANY
SPLIT.**
**date:** 2026-09-06 · **owner:** RL gate stream (Arch + Inference FlyWheel) ·
**branch:** `agent/arch-inf-20260803` · **GPU: 0.**

    PREREG_HYPOTHESIS_ID: H-RL-GATE-STAT-2
    PREREG_STATISTIC: S6_wmr
    PREREG_BAR: S6_wmr <= 0.30
    PREREG_RUNG: all
    PREREG_CELL: ship

---

## 0. ⛔ WHAT THIS IS NOT

⛔ **`G-REWARD` REMAINS FAILED ON ITS COMMITTED STATISTIC (THE RATE).**
⛔ **`H-RL-GATE-STAT-1` FAILED on its committed bar** — `S6_wmr` **0.301183**, CI
[0.201040, 0.433903] against `<= 0.30` on the held-out split — and that verdict stands,
unchanged and unmoved. **Nothing here re-scores either of them.**

## 1. Why this is the next lever and not a goalpost move

`H-RL-GATE-STAT-1` deliberately held the REWARD at the **pre-repair** configuration
(`progress_lead_cap = off`, `headway_reduce = min`), because that is the reward
`G-REWARD` was scored on, so exactly ONE thing differed: the statistic. That was the
right first test and it is now answered: **FAILED, by 0.0012.**

⇒ The question that experiment could not ask is the one the PI actually needs answered:
**does the reward an RL arm would train on TODAY clear the bar?** That reward is not the
pre-repair one — since 2026-09-06 the shipped default is
`progress_lead_cap = "achievable"` (`_progress_cap_mode`: `True` -> `"achievable"`) with
`headway_reduce = "min"` (the default reduction). ⛔ **That combination was never
computed in this package**, so it is a genuinely new measurement and not a re-reading.

⚠️ **AND I WRITE THIS KNOWING IT WILL PROBABLY FAIL.** On the SELECT split the
`q0.25` cell reads `S6_wmr` **0.3596** — well above the bar — and the pre-repair cell
read **0.3012** on the held-out split. Both are above 0.30. ⛔ **The bar is NOT being
lowered, the rung is NOT being changed, and no cell is being chosen for its number: the
cell is fixed by what the code SHIPS, and the bar is the same transposed 0.30.** If it
fails, the finding is that it fails.

## 2. ⛔ THE OBJECT UNDER TEST

> `S6_wmr` = `sum(max(g, 0)) / sum(abs(g))` on `g_i = composed(hold_v0)_i −
> composed(human)_i`, cell **`ship`** = `progress_lead_cap="achievable"`,
> `headway_reduce="min"` — **the reward's shipped defaults**, read from
> `rewards._progress_cap_mode` and `rewards._reduce_time_gap`.

Same weights `{progress 0.3, collision 1.0, headway 0.3}`, same windows, same ladder,
same episode-cluster bootstrap, same SELECT/SCORE split rule, same `all` rung.

## 3. ⛔ THE BAR — unchanged, and unchangeable

> **PASS iff `S6_wmr <= 0.30` at the `all` rung with the WHOLE episode-cluster bootstrap
> interval below the bar (`ci_hi <= 0.30`).**

Identical to `H-RL-GATE-STAT-1`, which is identical to `G-REWARD`'s own committed 0.30
transposed onto the same `[0, 1]` scale. ⛔ **A third statistic, a looser bar or a
tighter rung after this result is inadmissible and is named here so it cannot be
attempted quietly.**

## 4. ⛔ BOTH OUTCOMES, WRITTEN NOW

* **PASS** ⇒ ⭐ **the repaired reward clears a magnitude-aware bar under a statistic that
  demonstrably moves with the lead's position** — and the honest reading is that the
  reward's two 2026-09-06 repairs did the work the RATE could not see. It still does not
  revive `G-REWARD`; it hands the PI a scoreable gate that the current reward passes.
* **FAIL** ⇒ reported as **FAILED**, with the point estimate and interval, and ⛔ **the
  conclusion is that the object to fix is the REWARD, not the way it is read.** Three
  statistics and two term repairs will then have failed to put the trivial
  constant-velocity path below 30 % of the gap mass, which is a statement about the
  reward's geometry and belongs to the PI / Master Mind as a design decision.

## 5. ⛔ CONTROLS THAT MUST READ KNOWN VALUES

1. ⭐ **RANKING REPLICATION.** The re-run recomputes the mutation ranking on the SELECT
   split from scratch. Every candidate's `RNR`, and in particular `S1_rate` **0.23** and
   `S6_wmr` **1.21**, must reproduce **exactly**. A ranking that does not replicate voids
   the selection that produced this hypothesis.
2. ⭐ **CHANNEL.** `S1_rate` on the **`nopert_min`** cell over all 6,089 windows must
   still read the banked **0.441780259484316** to 0.000e+00. It anchors the new rows file
   to the old one.
3. **CAP BINDS.** The `ship` cell must actually differ from `nopert_min` — report the
   fraction of windows where `progress` differs. ⛔ If that fraction is ~0 the cap is
   inert on this corpus and no verdict is admissible, because the two cells would be the
   same object.
4. **SPLIT DISJOINTNESS** — overlap 0, union == panel.
5. **`frozen` stays far below the human.**

## 6. Estimator, tier, evidence class

Episode-cluster bootstrap over the SCORE split's episodes, `n_boot` 4,000, alpha 0.05.
⛔ It answers **ONE** question: *"would another draw of EPISODES say this?"* — not another
training run (`H-ESTIM-SEED-1`, OPEN) and not another inference run; neither enters,
because **no arm is trained and no planner samples**. Tier **T0** instrument probe,
NON-PARITY RL-fit windows, **not a driving number**. Evidence class MEASURED (ours).
⛔ Four families: this scores a REWARD-GATE STATISTIC, not a policy.
