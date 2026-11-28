# PRE-REGISTRATION 4 — `H-RL-GATE-STAT-1`: the win-magnitude ratio, and its own bar

**status: PRE-REGISTERED. ⛔ WRITTEN AND COMMITTED BEFORE THE SCORE SPLIT WAS READ, AND
BEFORE THE PRIMARY CELL WAS COMPUTED ON ANY SPLIT.**
**date:** 2026-09-06 · **owner:** RL gate stream (Arch + Inference FlyWheel) ·
**branch:** `agent/arch-inf-20260803` · **GPU: 0.**

<!-- machine-readable declaration read by stack/scripts/rl_statistic_score.py.
     ⛔ The tool REFUSES to run unless these four lines are present, so the bar it
     applies is the bar that was COMMITTED and cannot be supplied at the command line. -->

    PREREG_HYPOTHESIS_ID: H-RL-GATE-STAT-1
    PREREG_STATISTIC: S6_wmr
    PREREG_BAR: S6_wmr <= 0.30
    PREREG_RUNG: all
    PREREG_CELL: nopert_min

---

## 0. ⛔ WHAT THIS IS NOT

⛔ **This is NOT `G-REWARD` repaired, and it may never be presented as such.**
`G-REWARD` **FAILED on its committed statistic (the RATE)** and stays failed. This is a
**NEW hypothesis with a NEW ID and its own bar**, as ruled. Whatever it scores, the
`G-REWARD` row is unchanged.

## 1. Why THIS statistic — the ranking that selected it, and the ground it stands on

`PREREG_STATISTIC_RANKING.md` (committed before any candidate was computed) enumerated
nine candidates and ranked them by a mutation test **before scoring any of them**. On the
SELECT split (3,672 windows / 40 episodes), with every control reading its known value:

| rank | candidate | family | RNR = \|dS(5 m)\| / SD_boot | verdict |
|---|---|---|---|---|
| 1 | `S9_skewsplit` | MIXED | 1.57 | survives — ⛔ but **declared a DIAGNOSTIC in the candidate table**, not a preference statistic, and therefore not selectable |
| **2** | **`S6_wmr`** | **MASS** | **1.21** | ⭐ **the top PREFERENCE statistic** |
| 3 | `S8_logmassratio` | MASS | 1.15 | survives |
| 4 | `S5_mean` | MASS | 1.13 | survives |
| 5 | `S4_q75` | ORDER | 0.96 | UNDERPOWERED |
| 6 | `S3_median` | ORDER | 0.52 | UNDERPOWERED |
| 7 | `S7_signedrank` | RANK | 0.50 | UNDERPOWERED |
| 8 | `S2_cliffs` | SIGN | 0.27 | UNDERPOWERED |
| 9 | ⛔ `S1_rate` — **the retired statistic** | SIGN | **0.23** | **UNDERPOWERED, and WRONG-SIGNED** |

⭐ **Three reasons `S6_wmr` is the selection, all of them pre-registered grounds:**

1. **It is the top-ranked PREFERENCE statistic by the pre-registered RNR criterion**
   (1.21). `S9_skewsplit` ranks above it but was declared a **diagnostic** in the
   candidate table before any number existed — an asymmetry measure cannot answer *"is
   the reward paying the trivial path?"*
2. **Its response is RIGHT-SIGNED and MONOTONE.** Moving the lead away relaxes
   distance-keeping, so the constant-velocity path must be paid **more**: `S6_wmr` moves
   **+0.073936** at 5 m and **+0.104582** at 15 m. ⛔ `S1_rate` moves **−0.011983** and
   `S2_cliffs` **−0.023965** — *backwards*.
3. ⭐ **It transposes `G-REWARD`'s bar without inventing a number.** `S6_wmr` lives on
   `[0, 1]` with 0.5 neutral, exactly like the rate, so the committed 0.30 carries over
   unchanged (§3).

## 2. ⛔ THE OBJECT UNDER TEST — one variable moved, and only one

> `S6_wmr` = `sum(max(g, 0)) / sum(abs(g))`, where `g_i = composed(hold_v0)_i −
> composed(human)_i` — **the share of the TOTAL reward-gap mass that is paid to the
> trivial constant-velocity path.**

⛔ **The REWARD IS HELD EXACTLY WHERE `G-REWARD` SCORED IT.** The primary cell is
`nopert_min` — `progress_lead_cap = off` and `headway_reduce = min`, i.e. the
**bit-identical pre-repair reward** on which the banked all-window rate
**0.441780259484316** was computed. ⇒ **Exactly one thing differs from `G-REWARD`: the
statistic.** The repaired-`progress` and quantile-`headway` variants are reported as
SECONDARY columns and are never the verdict.

⚠️ **The primary cell has NOT been computed on either split at the time of writing.** The
ranking ran on the `q0.25` cell; `nopert_min` is untouched. The bar below is therefore set
without having seen the number it will judge.

## 3. ⛔ THE BAR, and why this number and not another

> **PASS iff `S6_wmr <= 0.30` at the `all` rung, with the WHOLE episode-cluster bootstrap
> interval below the bar (`ci_hi <= 0.30`).** The point estimate is reported beside it.

* **0.30 is `G-REWARD`'s OWN committed ceiling, transposed, not chosen.** `G-REWARD`
  asked: *does the trivial path win at most 30 % of the **windows**?* The magnitude-aware
  form asks: *does it take at most 30 % of the **mass**?* Same number, same direction,
  same [0, 1] scale. ⛔ **No tuning, and none is admissible later.**
* **`all` is the transposition-faithful rung** — `G-REWARD`'s headline was the all-window
  rate (banked 0.441780). ⛔ **A tighter rung is NOT selectable**, even though the SELECT
  split's conflict rungs read lower on the secondary cell; picking a rung after seeing it
  is the goalpost move this file exists to prevent.
* **Whole-interval, not point**, matching the `DIV` discipline already used in this
  package: a claim whose interval straddles its bar has not cleared it.

## 4. ⛔ BOTH OUTCOMES, WRITTEN NOW

* **PASS** (`ci_hi <= 0.30` on the SCORE split) ⇒ **there is a statistic that can measure
  distance-keeping and the reward clears the transposed bar under it.** ⛔ It still does
  **not** revive `G-REWARD`; it establishes `H-RL-GATE-STAT-1` and hands the PI / Master
  Mind a scoreable gate to adopt or reject.
* **FAIL** ⇒ reported as **FAILED**, with the point estimate and interval. ⭐ **The
  instrument finding stands either way** — the mutation ranking is independent of the
  score, and the rate's RNR 0.23 with a wrong-signed response is a property of the
  statistic, not of this panel's outcome.
* ⛔ **Neither outcome re-opens `G-REWARD`, and neither licenses a third statistic chosen
  after seeing this one's number.** If `S6_wmr` fails, the honest next object is the
  reward itself, not another way of reading it.

## 5. ⛔ CONTROLS THAT MUST READ KNOWN VALUES

1. ⭐ **CHANNEL — the cell must BE the cell `G-REWARD` scored.** `S1_rate` on
   `nopert_min` over **all 6,089 windows** must reproduce the banked
   **0.441780259484316**. If it does not, the primary cell is not the pre-repair reward
   and no verdict here is admissible.
   ⚠️ The banked composed **MEAN** (`−0.011239379601720929`) is a **STALE baseline** —
   `rewards._collision` changed after the 2026-09-05 bank (13 of 18,267 evaluations,
   3 of them on `hold_v0`, `3/6089 = 4.927e-04` exactly). ⛔ **The mean is NOT a channel
   control and must not be quoted as a current value.** The rate is unaffected and IS the
   control.
2. **SPLIT DISJOINTNESS** — the SELECT and SCORE window sets must be disjoint and their
   union must be the full panel; reported as counts.
3. **`frozen` stays far below the human** on the scored split.
4. **The full ladder is reported**, marked SECONDARY, so a reader can see what a rung
   choice would have bought — and see that it was not taken.

## 6. Estimator, tier, evidence class

* **Estimator:** episode-cluster bootstrap over the SCORE split's episodes, `n_boot`
  4,000, alpha 0.05. ⛔ It answers **ONE** question: *"would another draw of EPISODES say
  this?"* — **not** another training run (`H-ESTIM-SEED-1`, OPEN) and **not** another
  inference run. Neither enters here: **no arm is trained and no planner samples**; this
  is arithmetic over fixed geometry.
* **Tier:** **T0** instrument probe, NON-PARITY RL-fit windows. ⛔ Not a driving number,
  and no capability claim.
* **Evidence class:** MEASURED (ours).
* ⛔ **Four families:** this scores a REWARD-GATE STATISTIC, not a policy. No four-family
  table is asserted or implied.

## 7. What is NOT changed by this document

No reward term, no weight, no arm, no guard, no banked result, no registry row. The
perturbation used to rank lives only inside the ranking tool and never reaches a reward
path. `G-REWARD` stays FAILED.
