# ⛔ THE 165-ROW RE-READ AT THREE SEEDS — C100's LAST SUBSTANTIVE ROW IS A SEED ARTEFACT, AND ITS DIRECTION WAS UNDERSTATED

**Date:** 2026-08-18 · **Branch:** `agent/arch-inf-20260803` · **Agent:** ladder-3seed
**Eval tier:** ⛔ **T0-DIAGNOSTIC.** A frozen-latent linear readout is a world-model diagnostic and is
**never** driving performance. No number here is an ADE, a closed-loop result, or a claim about how
the car drives.
**Estimator:** `taniteval.ci.paired_episode_cluster_bootstrap`, n_boot 2000, clustered on the 70 eval
episodes. ⛔ `overlapping_holdout_se` is never imported.
**GPU: none.** CPU ridge on the dev box, from banked local caches. ⛔ **Thor was not touched** — the
S-W run holds PID 25477.

---

## ⭐ THE ANSWER, IN ONE PARAGRAPH

`MEASURED` (`raw/reread3_table.json`, route A). Re-running C100's 165-row re-read at **three inner-split
seeds** does not overturn C100 — **it strengthens it, and removes its one exception.** C100 reported
*"of 87 banked separated-FAILs, exactly ONE is a substantive finding: `ll_s09000 lead_gap`, K1B +0.748
[+0.002, +1.624]"*. ⛔ **That row is SEED-UNSTABLE.** Its bucket across seeds 0/1/2 is
`survive_both / die_at_repair / die_at_repair`, its K1B is **+0.748 / −0.022 / +0.145**, its CI covers
zero on two of three seeds, and its alpha moves **1e3 / 1e7 / 1e5**. ⇒ ⭐⭐ **AT THREE SEEDS THE
SUBSTANTIVE COUNT IS ZERO. Not one of the 87 banked separated-FAILs is a quotable finding about a
latent.** And the instrument that says so reproduces C100 exactly where it should: my route-A **seed-0**
rows are **bit-identical to the banked route-A seed-0 rows on 3 465 of 3 465 field comparisons**, so
this is an extension of C100's run, not a different one.

⭐ **AND THE SECOND RESULT IS THE ONE THAT GENERALISES.** C103's mechanism — *the C92 repair
un-truncates an alpha sweep the defect had frozen* — is now measured on the whole 165, not on 4 arms:
**the DEFECTIVE instrument picks the same alpha on all three seeds for 132 of 165 rows; the REPAIRED
one for 42.** ⇒ **The repair cut seed-stability of the alpha choice by 3.1×.** That is the direct
measurement of *"a stability claim measured under a defect is not inherited by the repaired
instrument"*, and it is why **22 of the 87 rows have no verdict at all** — their bucket flips across
seeds.

⛔ **AND THE TRIVIAL-PROXY CONTROL, NOW ON EVERY ROW.** Of **154** rows with a paired single-ego-speed
scalar on the **same window set**, the **scalar matches or beats the 2 048-dimension latent on 120**.
The v6 latent's **only** 3-seed-stable guarded PASSes in the entire ladder are `n_agents_all` at four
checkpoints — **and the scalar wins all four on the 3-seed mean.**

---

## 0. ⛔ THE STAMPS THAT BIND EVERY NUMBER BELOW

1. ⛔ **EVERY TABLE NAMES ITS REPAIR ROUTE AND THE TWO ARE NEVER POOLED (C100/C103).**
   **Route A = `--fit-mode unpen`** — `ridge_fit(…, intercept_col=-1)`, the repair taken from the
   module. **This is the route C100's 165-row inventory used**, so it is the route this re-derivation
   must use to be commensurable with it. **Route B = `--fit-mode centred`** — the locally re-derived
   repair, the route `LATENT_LINEAR_LADDER` §5.3/§7/§8 was rendered from. Both were run at 3 seeds
   here, and §6 places them side by side for the only purpose that is admissible: showing they differ.
2. **Everything else is byte-identical to the banked re-read** — same caches, same
   `p3_selection.json`, same join file, same widened alpha grid `[1e-2 … 1e7]`, same `n_boot 2000`,
   same 11 targets. The only parameterised changes are the seed set, the fit mode and the output
   directory (`code/chain_3seed.sh`, derived from `…/2026-08-18-k1-degeneracy-guard/code/chain_reread.sh`
   rather than editing it — that directory belongs to another stream).
3. ⚠️ **LEAD-ENRICHED, NOT PARITY** — inherited unchanged. 130 clips, 60 probe-train / **70 eval**,
   clip-disjoint. ⛔ **No episode is selected, added, removed, reordered or re-hashed by this work.**
4. **`v6F-SW-30k@11250` unless another step is named**, and ⚠️ **every v6 point is an EARLY READ at
   11 250 / 30 000 = 37.5 %.**
5. ⛔ **THE SCRATCH COPIES WERE SYNCED FROM THE REPO AND VERIFIED BY md5 + A REAL IMPORT BEFORE ANY
   RUN**, because C100 found the scratch `pc6_linear_readout.py` **pre-C92 with no `intercept_col` at
   all**. `MEASURED`: `pc6_linear_readout.py` md5 `871bc69e…`, `ll1_ladder.py` md5 `4b57f4a8…`, both
   equal to the repo copies, `ridge_fit` carries `intercept_col`, and `taniteval.degeneracy.k1_guard`
   imports. **Staleness is a property of the target; the repo being correct proves nothing about the
   file on the path you execute.**

---

## 1. ⭐ THE REPRODUCTION GATE — run first, because a 3-seed number that silently moved seed 0 would be a different experiment

`fit_one` draws its own `np.random.default_rng(seed)` per call and `taniteval.ci` bootstraps at a
**fixed** `seed=0`, so adding seeds 1 and 2 **must not perturb seed 0**. That is an argument; the gate
is the measurement.

**`MEASURED` · `raw/reread3_table.json` → `R0_reproduction_gate_vs_banked_seed0`**

| | |
|---|---|
| rows compared | 15 arms × 11 targets = **165** |
| fields compared per row | 21 (15 fit fields + 6 guard fields) |
| **fields identical** | ⭐ **3 465 / 3 465** |
| **GATE** | ⭐ **PASS** |

⇒ **My route-A seed-0 rows ARE C100's route-A seed-0 rows, to the last banked digit.** Everything that
differs below differs because seeds 1 and 2 exist, not because the instrument moved.
⚠️ **`n_eval` is asserted equal between the incumbent and the re-read on every row before any delta is
reported** — seeds change the SOLVE, never the window set, so an `n` that moved would invalidate the
pairing.

---

## 2. ⛔⛔ THE C100 INVENTORY, RE-DERIVED AT THREE SEEDS

The population is **exactly C100's**: the 87 rows whose **incumbent (pc6) verdict at seed 0** is
`FAIL-separated`. The bucket rule is `corrected_tables.py`'s, verbatim. The only change is that it is
applied **per seed**, and ⛔ **a row whose bucket is not the same on all three seeds is reported as
`SEED-UNSTABLE`, not as its mean. A verdict that flips across seeds is not a verdict.**

**`MEASURED` · route A (`unpen`) · seeds 0/1/2 · `R2_fail_inventory_3seed`**

| of the 87 banked separated-FAILs | **3-seed unanimous** | *C100 (seed 0)* | seed 0 / 1 / 2 separately |
|---|---|---|---|
| **die at the C92 repair** | **12** | *23* | 23 / 21 / 23 |
| ⛔ **killed by the C97 guard** | **33** | *42* | 42 / 42 / 39 |
| **flip to PASS** | **11** | *11* | 11 / 11 / 11 |
| **survive both** | **9** | *11* | 11 / 13 / 14 |
| ⛔ **SEED-UNSTABLE — no verdict** | ⛔ **22** | *(not measurable at 1 seed)* | — |
| ⭐ **of the survivors, SUBSTANTIVE** (`\|K1B_mean\|/gt_sd ≥ 0.02`) | ⛔ **0** | *1* | — |

⇒ ⛔ **C100's DIRECTION IS CONFIRMED AND ITS ONE EXCEPTION IS GONE.** 65 of 87 died to two independent
mechanisms at seed 0; at three seeds **45 die unanimously, 22 have no stable verdict, 11 are the
positive controls flipping to PASS, and the 9 that survive are all arithmetic** (§3).

### 2.1 ⭐ THE ROW C100 SINGLED OUT, AND WHY IT DOES NOT SURVIVE

C100's one substantive survivor, quoted verbatim from its own table: *"`ll_s09000` `lead_gap` — K1
+1.811 → +1.291, K1B +0.748 [+0.002, +1.624], guard OK, `|K1B|/gt_sd` 0.121."*

**`MEASURED` at three seeds:**

| seed | alpha | K1 | **K1B [CI]** | guard | bucket |
|---|---|---|---|---|---|
| **0** | 1e3 | +1.291 | **+0.748 [+0.002, +1.624]** | OK | `survive_both` |
| **1** | **1e7** | +0.022 | **−0.022 [−0.048, +0.002]** | NO-VERDICT-TO-GUARD | `die_at_repair` |
| **2** | 1e5 | +0.301 | **+0.145 [−0.239, +0.602]** | NO-VERDICT-TO-GUARD | `die_at_repair` |

⇒ ⛔ **It separates on ONE seed of three, its CI covers zero on the other two, and its alpha spans four
orders of magnitude.** The 3-seed mean K1B is **+0.290** (`|K1B|/gt_sd` 0.047) — but **the mean is not
the quotable object here; the instability is.** C100 already called this row *"fragile … its CI's lower
bound is +0.002 — it separates by a hair"* and said *"no trend is claimed and this survivor should not
be built on"*. **Three seeds convert that warning into a measurement.**

### 2.2 THE 22 ROWS WITH NO VERDICT — and they are exactly the rungs that carry signal

`MEASURED`. The 22 `SEED-UNSTABLE` rows fall on **five** targets:
**`nearest_any` 7 · `lead_gap` 5 · `ego_yawrate` 5 · `ego_v0` 4 · `lead_closing` 1.**

⇒ ⭐ **The instability is not spread evenly over the ladder — it concentrates on the near-tie rungs.**
`ego_curv` (13 rows in the 87) and `lead_present` (15 rows) produce **zero** unstable rows: they are
degenerate on every seed and stay that way. The rungs where a verdict would have been *interesting*
are precisely the rungs where a single seed cannot supply one.

⚠️ **A pattern in the alpha vectors, reported because it bears on the instrument rather than the
model.** On most unstable rows **seed 1 selects the grid's top alpha (1e7)** while seeds 0 and 2 pick
1e3–1e5. `MEASURED` over all 176 rows: the chosen alpha sits at a **grid edge** on **78 / 94 / 82** rows
for seeds 0 / 1 / 2, of which **64 / 80 / 68** are at the TOP (1e7) and a constant **14** at the bottom
(1e-2). ⇒ **The widened `[1e-2, 1e7]` grid is still binding on roughly half the rows.** For the
no-signal rows that is the *correct* repaired behaviour (shrink to the train mean — which the C97 guard
then classifies as degenerate), but ⚠️ **it means "alpha selection" on those rows is a boundary hit, not
an optimum, and a wider grid should be part of any future run.** `alpha_at_grid_edge` is emitted per row
so this is checkable rather than assumed.

---

## 3. ⛔ THE 9 SURVIVORS ARE ARITHMETIC — INCLUDING ONE ON A RANDOM-LATENT NULL

**`MEASURED` · route A · all 9 rows that are `survive_both` on all three seeds.**

| arm | target | **K1B (3-seed mean)** | `\|K1B\|/gt_sd` | guard | margin vs its C-V0 |
|---|---|---|---|---|---|
| `v6F@11250` | `ego_yawrate` | **+0.000486** | 0.0070 | OK | +0.000201 |
| `v6F@9000` | `ego_yawrate` | +0.000560 | 0.0081 | OK | +0.000275 |
| `v6F@9250` | `ego_yawrate` | +0.000543 | 0.0078 | OK | +0.000258 |
| `v6F@10000` | `ego_yawrate` | +0.000539 | 0.0078 | OK | +0.000254 |
| `v6F@2000` | `ego_yawrate` | +0.000444 | 0.0064 | OK | +0.000159 |
| `EGO-ORACLE n0.1` | `ego_yawrate` | +0.000344 | 0.0050 | OK | +0.000059 |
| `EGO-ORACLE n1` | `ego_yawrate` | +0.000252 | 0.0036 | OK | −0.000033 |
| `TOKENS-MEAN @11250` | `ego_yawrate` | +0.000925 | 0.0133 | OK | +0.000656 |
| ⛔ **`TOKENS-MEAN MATCHED-RANDOM NULL`** | `ego_yawrate` | ⛔ **+0.000002** | 0.00003 | OK | −0.000267 |

⇒ ⛔ **All nine are the same rung, all at `|K1B|/gt_sd` ≤ 0.013, and one of them is a `torch.randn`
cache.** A verdict a random-latent null reproduces is not a finding about a latent. This is C100's
reading (*"the other ten survivors are arithmetic, not findings"*) with the count moved 10 → 9 and one
of the two nulls now landing in the `SEED-UNSTABLE` bucket instead.

⭐ **The 11 flips-to-PASS are seed-stable and are the instrument validating itself** — unanimous on all
three seeds: **9 of 11 are EGO-ORACLE arms** (`n_agents_grid`, `n_agents_all`, `lead_gap` at noise
1×/3×/10×), all `guard OK` and monotone in injected noise; **the other 2 are the two RANDOM-LATENT
NULLS on `n_agents_all`, and the guard catches BOTH as `DEGENERATE-CONSTANT`.** ⇒ **The repair promotes
arms that carry signal and the guard rejects the two that carry none — on every seed.**

---

## 4. ⭐⭐ THE MECHANISM, MEASURED ON 165 ROWS INSTEAD OF 4 ARMS

C103's root-cause class is *"a stability claim measured under a defect is not inherited by the repaired
instrument"*, with the mechanism *"the C92 intercept defect had FROZEN the alpha sweep"*. That was
measured on the four `ll_rep_*` arms. It now has the full population behind it.

**`MEASURED` · the same 165 rows, incumbent (`pc6`, banked, 3 seeds) vs repaired (route A, 3 seeds) ·
`R2b_incumbent_vs_repaired_seed_stability`**

| statistic over 165 rows | **incumbent (defective)** | **repaired (route A)** |
|---|---|---|
| ⭐ **rows whose alpha is the SAME on all 3 seeds** | ⭐ **132** | ⭐ **42** |
| rows whose K1 verdict is the same on all 3 seeds | 156 | 142 |
| max K1 seed spread over the 165 | 4.239 | 2.812 |

⇒ ⭐ **The load-bearing number is the alpha row: the defect froze the choice on 80 % of rows, the
repair on 25 %.** That is the frozen sweep, quantified, and it is why the ladder's *"seed spread is
exactly zero on 8 of 11 rungs, so ≥3 seeds supply no uncertainty here"* was an artefact of the
instrument rather than a property of the question.

⚠️ **AND ONE NUMBER IN THAT TABLE POINTS THE OTHER WAY — reported rather than dropped, because a table
where every column agrees is the shape a cherry-picked table has.** The **max K1 seed spread is LARGER
on the incumbent (4.239) than on the repaired solve (2.812)**. `MEASURED`: the incumbent's worst row is
`proxyv0 n_agents_all`, whose incumbent alphas are `1e3 / 1e3 / 1e-2` — the defect did **not** freeze
every row, it froze most of them. ⇒ **The correct claim is "the repair unfroze the alpha choice on the
majority of rows", NOT "the incumbent had no seed variance anywhere".** The second would be the same
over-generalisation the ladder's original "exactly zero on 8 of 11" was.

---

## 5. ⛔ THE TRIVIAL-PROXY CONTROL, ON EVERY ROW (C92)

⭐ **A control that did not previously exist, and its absence was invisible.** C100's re-read carries
**one** C-V0 arm (`proxyv0`), fitted on the **cells** cache (n_eval 2221–3023). Three of its 15 arms
— `tok11250`, `tok11250null`, `cells_tokwin` — are fitted on the **TOKENS** cache (n_eval 1103–1507).
Comparing those against the cells-window C-V0 would be an **unpaired** comparison across different
window sets. ⇒ ⛔ **33 of the 165 rows had no trivial-proxy control at all, and nothing said so.**
This run adds **`proxytok`** — the identical ridge, identical split/seeds/estimator, on the **tokens**
windows, features replaced by the single scalar `v0` — so every row now has a paired control.
⚠️ **It is 11 ADDITIONAL rows, never folded into the 165.**

**`MEASURED` · route A · 3-seed mean K1B · K1B is negative-is-better, so `margin = arm − C-V0` and
POSITIVE MEANS THE SCALAR WINS.**

| | |
|---|---|
| rows with a paired C-V0 on the same window set | **154** (176 total − the 22 C-V0 rows themselves) |
| ⛔ **rows where the SCALAR matches or beats the latent** | ⛔ **120** |
| rows where the latent beats the scalar | 34 |
| ⚠️ **rows whose margin SIGN flips across seeds** | ⚠️ **38** |
| rows with no paired control | **0** |

**By rung, over the 14 non-C-V0 arms:**

| rung | scalar wins/ties | latent wins | margin sign flips across seeds |
|---|---|---|---|
| `ego_v0` | **14 / 14** | 0 | 0 |
| `lead_gap` | **13 / 14** | 1 | 0 |
| `nearest_any` | **13 / 14** | 1 | 2 |
| `n_agents_grid` | **13 / 14** | 1 | 0 |
| `ego_accel` | **13 / 14** | 1 | 1 |
| `ego_curv` | 11 / 14 | 3 | 5 |
| `n_agents_all` | **10 / 14** | 4 | ⚠️ **6** |
| `lead_closing` | 10 / 14 | 4 | 3 |
| `lead_present` | 8 / 14 | 6 | ⚠️ **11** |
| `lead_inv_ttc` | 8 / 14 | 6 | 5 |
| `ego_yawrate` | 7 / 14 | 7 | 5 |

⚠️ **Read the bottom of that table with the scale beside it.** The four rungs where the latent "wins"
about half the time (`lead_present`, `lead_inv_ttc`, `ego_yawrate`, `ego_curv`) are the rungs where
**every** arm's K1B is ~1e-4 or smaller and the margin's **sign flips across seeds on 5–11 of 14 arms**.
Those are coin flips at physically nil magnitude, not latent wins. The rungs that carry any signal —
`ego_v0`, `lead_gap`, `nearest_any`, `n_agents_grid` — go to the **single scalar on 13 or 14 of 14
arms.**

### 5.1 ⭐⭐ THE ONLY 3-SEED-STABLE GUARDED PASSES ON A v6 ARM — and the scalar wins all four

⛔ **The list a positive claim may be taken from, and nothing else.** A row qualifies only if it is
`PASS` on **all three** seeds **and** `guard OK` on **all three**. `MEASURED` · route A ·
`R5_quotable_guarded_PASS_3seed`: **33 rows qualify, of which 29 are controls** (the GT oracle, the
four EGO-ORACLE noise levels, and the two C-V0 arms themselves). **Four are on a v6 latent arm — all
of them `n_agents_all`:**

| arm | v6 K1B, seeds 0/1/2 | v6 mean | C-V0 K1B, seeds 0/1/2 | C-V0 mean | margin/seed | ⛔ **margin, 3-seed mean** |
|---|---|---|---|---|---|---|
| `v6F@9000` | −2.715 / −0.226 / −1.333 | −1.425 | −2.243 / −0.579 / −2.238 | −1.687 | −0.472 / +0.353 / +0.905 | ⛔ **+0.262 — scalar wins** |
| `v6F@9250` | −2.759 / −0.236 / −1.338 | −1.444 | ″ | −1.687 | −0.516 / +0.344 / +0.901 | ⛔ **+0.243 — scalar wins** |
| `v6F@10000` | −2.805 / −0.240 / −1.365 | −1.470 | ″ | −1.687 | −0.561 / +0.340 / +0.874 | ⛔ **+0.217 — scalar wins** |
| ⭐ `v6F@11250` | −2.785 / −0.269 / −1.376 | −1.477 | ″ | −1.687 | −0.541 / +0.310 / +0.863 | ⛔ **+0.211 — scalar wins** |

⇒ ⭐⭐ **C103 MEASURED THIS AT ONE CHECKPOINT; IT HOLDS AT FOUR, AND SEED 0 IS THE OUTLIER ON EVERY
ONE.** On all four rows seed 0 is the **only** seed that favours the latent, and seeds 1 and 2 favour
the scalar by more than seed 0 favours the latent. ⇒ **The 2 048-dimension v6 operative latent does not
out-read one number of ego speed on any rung of this ladder at three seeds.**

⚠️ **Stated with its own instability, because the mean is not the whole story.** The margin's sign is
**not** seed-stable on those four rows — it is negative on seed 0 and positive on seeds 1 and 2. ⇒ The
honest form is: **the margin is smaller than the arm's own seed noise, and the 3-seed mean puts it on
the scalar's side.** That is a reason to stop treating a 0.012 gt_sd margin as evidence at all — which
is exactly what C103 concluded — and **not** a licence to quote "+0.211" as an effect.
⚠️ **`v6F@2000` is the one v6 checkpoint whose `n_agents_all` guard verdict is itself SEED-UNSTABLE**
(and the scalar beats it on all three seeds, +0.677 / +0.326 / +0.742), so it does not reach the
quotable list at all.

---

## 6. ⛔ THE TWO REPAIR ROUTES AT THREE SEEDS — SIDE BY SIDE, NEVER POOLED

*(Filled in from `R4_two_routes_3seed` once route B lands; see §9.)*

---

## 7. THE FOUR METRIC FAMILIES

Per the binding rule, every family is addressed with the reason and the `n` where it does not apply.
⛔ **ADE is not reported and is not applicable:** this is a frozen-latent state readout, not a
trajectory eval. **All numbers route A, 3-seed.**

| family | what this run reports | verdict |
|---|---|---|
| **LONGITUDINAL** | target speed: `ego_v0` — **the scalar wins on 14 of 14 arms**, and the v6 arm's own K1B is seed-unstable (−0.236 / −0.020 / +0.374). Distance keeping: `lead_gap` — scalar wins **13 of 14**; `lead_closing` and `lead_inv_ttc` — every arm's K1B ≤ ~1e-2 with the margin sign flipping across seeds. | ⛔ **Nothing quotable.** The family's own control variable is read better by a single number the model is handed, and its time-gap / TTC half is at the resolution of noise. |
| **LATERAL** | `ego_yawrate` — the **only** rung with 3-seed-stable `survive_both` rows, all at `\|K1B\|/gt_sd` ≤ 0.013 and **one of them on a random-latent null**. `ego_curv` — 11 of 14 arms lose to the scalar, 5 of 14 margins flip sign. | ⛔ **Nothing quotable, and the rung that "survives" is reproduced by `torch.randn`.** ⚠️ Both rungs still lack a positive control (`LATENT_LINEAR_LADDER` §4.4) — an unverified negative. |
| **TACTICAL** | ⚠️ **NOT MEASURED, n = 0.** The ladder is regression-only; the caches bank no manoeuvre label. | ⚠️ Absent by instrument scope. **A work item, not an excuse** — a multinomial ridge on the same features makes this family T0-reportable; already item 6 of `LATENT_LINEAR_LADDER` §14. |
| **STRATEGIC** | ⚠️ **NOT MEASURED, n = 0.** No route or goal label exists in this 130-clip lead-enriched pool, and per `CLAUDE.md` PhysicalAI-AV carries no map, lane graph or route signal at all. | ⚠️ Not computable on this corpus with this cache. |

---

## 8. SUITES

**`MEASURED` by this run.** ⭐ **I modified nothing under `stack/` or `taniteval/`** — every artifact
is a new file under `…/incoming/2026-08-18-ladder-3seed/`, plus an in-place update to
`LATENT_LINEAR_LADDER.md`.

*(Results in §9 — the first `taniteval` run is recorded there together with a measurement worth
keeping.)*

---

## 9. STATUS

⚠️ **This document is banked incrementally.** §1–§5 and §7 are complete and quotable. §6 (route B) and
§8 (suites) are filled in by the same run and are marked here so a reader can see what is outstanding
rather than discovering a gap.
