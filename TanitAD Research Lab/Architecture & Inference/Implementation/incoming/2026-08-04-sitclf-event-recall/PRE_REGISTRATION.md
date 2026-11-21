# PRE-REGISTRATION — is the SHARED sitclf re-tune a DEPLOYABLE change, and what does it cost?

**Written 2026-08-04, BEFORE any number below was read.** Content-pinned with `git hash-object`;
the hash is re-verified at the end of the run and printed into `MANIFEST.md`. If the file changed
between pin and verify, every result in this stream is void.

**Stream** `2026-08-04-sitclf-event-recall` · **Substrate** dev box, **0 pod GPU-h** — no pod is
touched. **Parent** `../2026-08-03-sitclf-horizon/` (banked scores + report) and
`../2026-08-03-sitclf-temporal/`.

⛔ **PI ruling 2026-08-03 honoured throughout: labels may use ego; INFERENCE IS VISION-ONLY.** Every
arm reads frozen v1 camera latents and nothing else. The ego block appears ONLY as the
LONGITUDINAL / LATERAL **stratification** variable, and only in its **CAUSAL** rebuild
(`sitclf_b4_substrate.ego_causal.npz`), never the quarantined legacy block, which is run beside it
so the difference is measured rather than asserted.

---

## 0. THE ONE QUESTION

The parent stream banked, and did not promote, this row on `lane_change`'s **event yardstick**:

| arm | event recall | onsets warned | n_alarm | alarm prec. @5 s |
|---|---:|---:|---:|---:|
| `FROZEN` (W8, L3.0) | 0.1404 | 8 / 57 | 2,849 | 0.0470 |
| `C-GLOBAL` | **0.4737** | **27 / 57** | 2,849 | **0.0653** |

**⛔ `C-GLOBAL` IS NOT A DEPLOYABLE SETTING.** It is a *selection rule* evaluated out-of-fold, and
the parent's own `arm_configurations` show it chose **two different windows in the two folds** —
`(W32, L1.0)` in fold 0 and `(W1, L1.0)` in fold 1, the two opposite extremes of the grid. A
deployed classifier has ONE `(W, lead_s)`. So the banked `+0.3333` answers *"does re-tuning help?"*
and **not** *"what do we ship?"*

**This stream asks the shipping question:** *is there a **single fixed** `(W, lead_s)` that beats
the deployed `(8, 3.0)` on the event yardstick, out-of-fold, after multiplicity control — and what
does it cost on the decision-grade situation?*

---

## 1. WHAT IS FIXED BEFORE ANY NUMBER IS READ

Inherited verbatim from the parent so the two streams are comparable; **nothing here moves after a
number is read.**

| constant | value |
|---|---|
| grid | `WINS = (1, 4, 8, 16, 32)` × `LEADS = (1.0, 2.0, 3.0, 4.0, 5.0)` — 25 cells |
| frozen cell | `(W = 8, lead_s = 3.0)` |
| eval rows | `valid(lead = 5.0) ∧ hist_ok(WIN_MAX = 32)` |
| alarm budget | `TOP_FRAC = 0.05` by rank ⇒ `n_alarm` identical for every arm by construction |
| event look-back | `H_MAX_S = 5.0 s`, never crossing a cluster |
| deployed horizon for the precision column | `3.0 s` |
| folds | `cluster_folds(cc, 2, seed=0)`; inner `SEL_FRAC = 0.20` of the training clusters, `rng(0+f)` |
| ridge | `RANK = 16` PCA, `LAMBDAS = (1, 10, 100, 1e3, 1e4)`, λ chosen on SEL by AP-lift on the arm's OWN training label |
| estimator | **paired episode-cluster bootstrap**, `B = 2000`, α = 0.05, cluster draws from `taniteval`'s `_draws`. ⛔ **`overlapping_holdout_se` is not used anywhere.** |
| power bar | `C_POW ≥ 40` positive clusters |

**Powered situations (from the parent's pre-committed `c_pow_precommit.json`, INHERITED, not
re-derived):** `intersection` **216** ✅ · `lane_change` **55** ✅ · `roundabout` **37** ⛔
`UNDERPOWERED_C_POW`. ⛔ **The bar is not lowered. `roundabout` gets no verdict in this stream and
is reported with its `n`.**

---

## 2. THE ARMS — fixed now, and no arm may be added after a number is read

Every arm is a **single fixed `(W, L)`** unless marked otherwise. All are scored out-of-fold
(2-fold cluster CV; every row is scored by a ridge fit on the *other* fold).

| arm | definition | why it is here |
|---|---|---|
| `FROZEN` | `(8, 3.0)` | what is deployed |
| **`FIXED_L1`** ⭐ | `(8, 1.0)` — **the minimal single-knob change**: the deployed window, only the lead moved | the cheapest thing that could ship; **pre-specified, so tested WITHOUT multiplicity correction** |
| `FIXED_W32_L1` | `(32, 1.0)` — fold 0's global pick | is either half of `C-GLOBAL` deployable alone? |
| `FIXED_W1_L1` | `(1, 1.0)` — fold 1's global pick | same |
| `RETUNE_SEL` | **procedure, not a cell**: one shared `(W, L)` selected per fold on the **SEL split** by the **event criterion**, evaluated on the test fold | the honest estimate of what running a re-tune actually buys. The parent's §10.1 names this as the missing arm |
| `GRID` (25) | every cell | the **existence** test, with the correction of §4 |
| `C_ORACLE_POOLED` | `argmax` over the 25 cells of the **pooled** event-recall delta — i.e. **the same statistic the arms are judged by** | ⭐ a **true** upper bound on any fixed-cell arm by construction. This is the fix for the parent's `C-ORACLE-PS` defect, which maximised a *per-fold* statistic while the headline was *pooled*, was therefore not an upper bound, and fired under the null |

⚠️ **`FIXED_L1` is pre-specified but NOT blind, and I say so before measuring.** Its value comes
from the parent's banked per-lead sweep **on this same 500-clip substrate** (`W8, L1.0`,
`+0.0877 [+0.0196, +0.1637]` on `lane_change`), and both folds' independent global selection chose
`L = 1.0`. Testing it here is **confirmatory of a hypothesis generated on the same data**, which is
weaker than an out-of-sample confirmation and is *not* upgraded by the absence of a multiplicity
penalty. A genuinely independent confirmation needs a corpus this substrate does not contain, and
that is stated as a limitation in the report whatever the outcome.

---

## 3. CONTROLS — each one CAN fail, and what its failure would mean

⛔ **No `C-shuffled`-style leg is admitted**: permuting a score column and taking an argmax is a
uniform random pick and the control is vacuous by construction.

| control | construction | **fails if** |
|---|---|---|
| `C-FID-GRID` | recompute all 25 real cells from the substrate and compare to the parent's banked `CELL_w*_L*` columns | **any** `max abs diff > 0`. Then my pipeline is not the parent's and no comparison is admissible |
| `C-BUDGET` | assert `n_alarm` is exactly equal for every arm within a situation | any inequality — then the "same alarm budget" claim is false |
| **`C-CHANCE`** ⭐ | 200 uniform-random score columns at the same 5 % budget ⇒ the **chance floor** for event recall | ⚠️ this control exists because `FROZEN`'s frame-level precision-**lift** on `lane_change` is **0.883 — below 1.0, worse than chance at its own operating point**. If `FROZEN`'s *event* recall is also at or below the chance floor, the headline is not "a re-tune helps", it is "**the deployed setting is broken on this situation and nearly anything beats it**", and the report must lead with that instead |
| `C-NULL-*` | every arm re-run on the **clip-permuted feature** substrate (parent's `run_temporal.py` permutation, verbatim) | a NULL twin separating above its own NULL frozen ⇒ `SELECTION_ARTEFACT` for that arm |
| `C-ORACLE-NULL` | `C_ORACLE_POOLED` on the permuted substrate | it is *expected* to be > 0 — that margin **is** the winner's curse, and any grid-selected arm's gain must exceed it to mean anything |
| `C-FOLD` | the candidate's event-recall delta computed **within each fold separately** | opposite signs in the two folds ⇒ `NOT_DEPLOYABLE_UNSTABLE`, whatever the pooled number says |

---

## 4. MULTIPLICITY — registered in advance, because the parent had none

The parent disclosed this as a design defect: 25 cells tested, 2 separated, expected false
positives **1.25**, no correction registered.

**Registered here:** the family is the **25 cells within one situation**, tested one-sided
(`δ > 0`) on the bootstrap tail probability `p = 1 − p(δ > 0)`, corrected by **Holm–Bonferroni at
α = 0.05**. With `B = 2000` the smallest attainable `p` is `5.0e-4` and Holm's most stringent
threshold is `0.05 / 25 = 2.0e-3`, so the correction is attainable rather than decorative.

The two **pre-specified** arms — `FIXED_L1` and `RETUNE_SEL` — are **outside that family** and are
tested uncorrected, because they are named in §2 before any number is read. `C_ORACLE_POOLED` is a
control, not a test, and carries no α.

---

## 5. THE DECISION RULE — all five, or it is not deployable

A cell is **DEPLOYABLE** iff:

- **D1** it is a **single fixed `(W, L)`** — one setting for every situation, because the head is
  one model with S outputs;
- **D2** its `lane_change` event-recall delta vs `FROZEN` is **separated above 0** (Holm-corrected
  if it comes from the grid; uncorrected if pre-specified in §2);
- **D3** its `intersection` event-recall delta is **NOT separated below 0** — no regression on the
  decision-grade situation;
- **D4** its `lane_change` alarm precision @5 s is **NOT separated below** `FROZEN`'s — the gain is
  not bought with a worse operating point;
- **D5** the sign of its `lane_change` event-recall delta is **positive in both folds** evaluated
  separately.

### Verdict labels, fixed now

| label | condition |
|---|---|
| `DEPLOYABLE_SHARED_RETUNE` | D1–D5 all hold for ≥ 1 cell |
| `DEPLOYABLE_WITH_COST` | D1, D2, D5 hold; D3 or D4 fails ⇒ there is a gain **and a named cost** |
| `NOT_DEPLOYABLE_UNSTABLE` | D2 holds pooled, D5 fails |
| `NOT_DEPLOYABLE_NO_FIXED_CELL` | no cell satisfies D2 |
| `SELECTION_ARTEFACT` | the winning arm's NULL twin also separates above its NULL frozen |
| `FROZEN_IS_BELOW_CHANCE` | `C-CHANCE` shows `FROZEN`'s `lane_change` event recall at or below the chance floor ⇒ the framing changes, and this label is reported **in addition to** whichever of the above applies |

---

## 6. PREDICTIONS — committed in advance, both outcomes written

| # | prediction | confidence | what the opposite would mean |
|---|---|---|---|
| **P-1** | `FIXED_L1` separates above `FROZEN` on `lane_change` event recall, in the band **+0.05 … +0.12** | 70 % | if not separated ⇒ `NOT_DEPLOYABLE_NO_FIXED_CELL` for the cheap change, and the parent's `+0.0877` was a single unreplicated cell |
| **P-2** | the best **single fixed cell** buys **less than `C-GLOBAL`'s +0.3333**, because `C-GLOBAL` combines two *different* cells across the two folds | 70 % | if a single cell matches or beats +0.3333, the fold-disagreement is irrelevant and the change is simpler than I think |
| **P-3** | `C_ORACLE_POOLED` on the **permuted** substrate earns **> +0.05** — grid-max selection has a winner's curse on this n | 75 % | if it earns ≈ 0, the 25-cell max is unbiased here and the Holm correction is over-conservative |
| **P-4** | `FROZEN`'s `lane_change` event recall (0.1404) is **above** the `C-CHANCE` floor | 60 % — genuinely uncertain, given its frame-level precision-lift of **0.883 < 1.0** | if it is at or below the floor, the deployed setting has **no event-level skill at all** on `lane_change`, and that is the headline, not the re-tune |
| **P-5** | `RETUNE_SEL` (the honest procedure) lands **below** the best fixed cell and may not separate, because the SEL split holds only ~20 % of ~50 % of 500 clusters and `lane_change` has only 57 onsets in total | 65 % | if it separates, a re-tune procedure is itself shippable, not just a lucky cell |

---

## 7. THE FOUR BINDING FAMILIES (PI, 2026-08-02) — how each is handled here

Reported **per family, never pooled**, each on the paired episode-cluster bootstrap, on the same
rows as the event yardstick it accompanies.

| family | treatment in this stream |
|---|---|
| **TACTICAL** | ✅ computed — situation anticipation **is** the tactical decision. Event recall, alarm precision at two horizons, median lead, and the frame-level operating point with **precision AND recall AND both denominators** (`n_alarm` for precision, `n_pos` for recall) |
| **LONGITUDINAL** | ⚠️ *not computable as stated* for this arm class — target-speed accuracy and headway/time-gap/TTC need a **predicted path**; this arm emits a per-frame probability. Reported instead as decision quality **stratified by longitudinal regime** (decelerating / steady / accelerating / low-speed / cruise), with `n` per stratum. The instrument that closes this for trajectory arms (`taniteval.lead_metrics.distance_keeping`) exists and does not apply here |
| **LATERAL** | ⚠️ same — stratified by lateral regime (straight / turning), with `n` |
| **STRATEGIC** | ⛔ **UNAVAILABLE** — no route/goal/map label exists on PhysicalAI-AV (settled at five probes: no map, lane graph, junction annotation or route signal; `egomotion` is clip-local metres with no GNSS). Reported with the reason and the `n`, not silently dropped |

---

## 8. WHAT THIS STREAM WILL NOT DO

1. ⛔ **It will not re-litigate the per-situation null.** `NO_PER_SITUATION_GAIN_EXISTS` on
   `intersection` stands as the parent recorded it.
2. ⛔ It will not lower `C_POW`'s bar for `roundabout`.
3. ⛔ It will not add an arm, a threshold or an outcome after a number is read.
4. ⛔ It will not re-read the parent's Q-A verdicts.
5. ⛔ It will not touch any pod, and it will not push.
6. It will not claim an out-of-sample confirmation. Every number is a re-analysis of the **same 500
   clips**; the parent's absolute APs are not comparable to the banked parity table and neither are
   these.
