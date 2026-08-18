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

⭐⭐ **AND THE SHARPEST FORM OF THE INVENTORY IS NOT THE BUCKET TABLE — IT IS THIS, and it is
IDENTICAL ON BOTH ROUTES.** A row can be `SEED-UNSTABLE` because the *mechanism* that kills it varies
while the death does not. Separating those:

| of the 87 banked separated-FAILs | **route A** | **route B** |
|---|---|---|
| ⛔ **DEAD ON ALL THREE SEEDS** (die at the repair or killed by the guard on every seed; on 13 of them the mechanism varies but the outcome does not) | ⛔ **58** | ⛔ **58** |
| ⭐ flip to PASS — the positive controls, **unanimous** | 11 | 11 |
| survive both **on all three seeds** — all `ego_yawrate`, all at `\|K1B\|/gt_sd` ≤ 0.013, one on a random-latent null | 9 | 9 |
| ⚠️ survive on **at least one seed but not all** — includes C100's substantive row | ⚠️ **9** | ⚠️ **9** |
| ⭐ **SUBSTANTIVE** | ⛔ **0** | ⛔ **0** |

⇒ ⛔ **C100's DIRECTION IS CONFIRMED AND ITS ONE EXCEPTION IS GONE.** C100 said *"65 of 87 were
instrument"* at seed 0; at three seeds **58 are dead on every seed, 11 are the instrument validating
itself, 9 are arithmetic on the one rung a `torch.randn` cache reproduces, and 9 are seed-dependent
survivors of which NONE reaches the substantive threshold.**

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

## 6. ⛔ THE TWO REPAIR ROUTES AT THREE SEEDS — AND THE ROUTE-EQUIVALENCE CLAIM WAS *ALSO* A ONE-SEED CLAIM

⭐ **This section exists because the brief's instruction to "stay on route B" and C100's inventory
being on route A are in tension, and the disciplined resolution is to run BOTH and never pool them.**
Both routes were run at 3 seeds over all 16 arms. Every table above is route A because that is C100's
route; route B is reported here and in `R4b`.

⚠️ **AND THE REASON THIS WAS WORTH THE COMPUTE.** C103's route comparison — *"44 paired rows, 2 alpha
choices differ, **0 verdicts differ**"* — was itself **measured at ONE SEED**. That is the same shape
as the *"seed spread is exactly zero"* claim C103 retracted. So it was re-measured at three.

**`MEASURED` · `R4_two_routes_3seed` · 176 paired rows · side by side ONLY, never pooled**

| | **C103 (44 rows, seed 0)** | **here (176 rows, 3 seeds)** |
|---|---|---|
| alpha choices / vectors that differ | 2 | **29** |
| ⚠️ **3-seed verdicts that differ** | *0* | ⚠️ **2** |
| guard verdicts that differ | 0 | ⚠️ **11** |
| **max abs K1 gap** | 0.3957 | ⛔ **0.7212** (`v6F@2000 n_agents_all`: A −3.032 / B −2.311) |
| …and its K1B gap | ×8 on `ego_v0` | **0.7283** on the same row |

⇒ ⚠️ **"The verdicts are robust to the route" does not fully survive three seeds** — but read what the
two differences actually are, because the honest form is narrower than the headline:

| row | route A | route B |
|---|---|---|
| `v6F@10000` `ego_v0` | `not-separated` (all 3 seeds) | **SEED-UNSTABLE** |
| `v6F@2000` `n_agents_all` | `PASS` (all 3 seeds) | **SEED-UNSTABLE** |

⇒ ⭐ **Neither is a PASS↔FAIL flip. Both are "one route reaches a stable verdict where the other does
not."** The 11 guard differences are the same shape and 10 of them sit at `|ΔK1B| ≤ 0.0015`, i.e.
tie-breaking at physically nil magnitude; the exception is `v6F@10000 ego_v0` at `|ΔK1B|` **0.134**.

⭐⭐ **AND THE CONCLUSION IS ROUTE-INVARIANT, WHICH IS THE POINT THAT MATTERS.**

| of the 87 banked separated-FAILs | **route A** | **route B** |
|---|---|---|
| die at the repair | 12 | 11 |
| killed by the guard | 33 | 33 |
| flip to PASS | 11 | 11 |
| survive both | 9 | 9 |
| SEED-UNSTABLE | 22 | 23 |
| ⭐ **SUBSTANTIVE** | ⛔ **0** | ⛔ **0** |

⇒ **Both routes independently return ZERO substantive survivors at three seeds.** ⛔ **The numbers
still must never be pooled** — `v6F@2000 n_agents_all` differs by 0.72 K1 between them — but the
finding does not depend on which repair route is used.

---

## 6a. ⚠️ THE RUNG PROFILE AT THREE SEEDS — the ordering holds as a SET, not as a RANKING

`r²` is the quantity every downstream citation of this ladder quotes, and `LATENT_LINEAR_LADDER` §4.2
argues it is the least-biased one: C92 and C97 act on the fit's **dispersion**, and correlation is
scale-invariant, so **at a fixed alpha neither defect can move `r²`**. But they truncated the alpha
sweep, alpha selection is upstream of the fit, and **the seed moves alpha** — so `r²` moves with the
seed as well.

**`MEASURED` · route A · `v6F-SW-30k@11250` · `R6_rung_profile_r2_3seed`**

| rung | r² seed 0 | r² per seed | **r² 3-seed mean** | own null (3-seed mean) |
|---|---|---|---|---|
| `n_agents_all` | 0.1519 | 0.1519 / 0.1573 / 0.1746 | **0.1613** | 0.0002 |
| `nearest_any` | 0.0964 | 0.0964 / 0.1004 / 0.0964 | **0.0977** | 0.0007 |
| `ego_v0` | 0.1032 | 0.1032 / 0.0913 / 0.0756 | **0.0900** | 0.0007 |
| `n_agents_grid` | 0.0200 | 0.0200 / **0.0880** / 0.0305 | **0.0462** | 0.0003 |
| `ego_accel` | 0.0161 | 0.0161 / 0.0350 / 0.0051 | **0.0187** | 0.0001 |
| `lead_present` | 0.0118 | 0.0118 / 0.0091 / 0.0053 | **0.0088** | 0.0000 |
| `lead_gap` | 0.0053 | 0.0053 / 0.0097 / 0.0057 | **0.0069** | 0.0001 |
| `ego_yawrate` | 0.0009 | 0.0009 / 0.0009 / 0.0015 | **0.0011** | 0.0001 |
| `lead_closing` | 0.0013 | 0.0013 / 0.0000 / 0.0013 | **0.0009** | 0.0000 |
| `lead_inv_ttc` | 0.0008 | 0.0008 / 0.0008 / 0.0008 | **0.0008** | **0.0009** |
| `ego_curv` | 0.0000 | 0.0000 ×3 | **0.0000** | **0.0005** |

⇒ ⭐ **THE SHAPE SURVIVES: a cliff, not a slope.** The top-3 **set** (`n_agents_all`, `ego_v0`,
`nearest_any`) and the bottom-4 **set** (`ego_yawrate`, `lead_closing`, `lead_inv_ttc`, `ego_curv`) are
unchanged, and two of the bottom four remain **below their own nulls**.
⚠️ **But the RANKING is not seed-stable and `LATENT_LINEAR_LADDER` §4.2's "the ordering held" must be
read at set level only:** `nearest_any` and `ego_v0` **swap** 2nd/3rd, and `ego_yawrate` and
`lead_closing` swap 8th/9th, between seed 0 and the 3-seed mean. `n_agents_grid` moves **0.0200 →
0.0880 → 0.0305** across seeds — a 4.4× swing on one rung.
⛔ **⇒ Any citation quoting an individual `r²` from this ladder must quote the 3-seed mean and its seed
spread, not a single seed's value.**

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

## 8. SUITES — and a trap worth keeping

**`MEASURED` by this run.** ⭐ **I modified nothing under `stack/` or `taniteval/`** — every artifact
is a new file under `…/incoming/2026-08-18-ladder-3seed/`, plus an in-place update to
`LATENT_LINEAR_LADDER.md`. `git diff --cached --name-only | grep -E '^(stack|taniteval)/'` from this
agent: **empty**.

| suite | result | briefed baseline | verdict |
|---|---|---|---|
| `taniteval` | **1136 passed, 0 failed**, 200 s (`raw/suite_taniteval.txt`) | 1136 / 0 | ✅ **GREEN — exact match** |
| `stack` | **3868 passed, 0 failed, 7 skipped, 2 xfailed**, 610 s (`raw/suite_stack.txt`) | 3861 / 0 / 7 / 2 | ✅ **GREEN** ⚠️ **+7, and none of it is mine — see below** |

⚠️ **THE `stack` COUNT MOVED BY +7 AND IT IS ATTRIBUTED, not absorbed into "green".** `MEASURED`:
the working tree carries another agent's `stack/tests/test_seam_dump_import_guard.py`, which has
**exactly 7 test definitions** — the whole delta. `git diff --cached --name-only | grep -E
'^(stack|taniteval)/'` from this agent is **empty**. **Zero failures either way, so the gate is
satisfied** — but a future agent should be briefed with **3868**, not 3861.

⛔⛔ **A TRAP MEASURED HERE, AND IT IS THE "EXIT CODES ARE NOT EVIDENCE" FAMILY IN A NEW COSTUME.**
The **first** `taniteval` run — launched deliberately in parallel with the 15-arm ladder chains —
returned ⛔ **22 failed, 1114 passed**, including
`test_render_openloop_video.py::test_cli_help_works_without_a_gpu` and
`test_t1_eval.py::test_cli_help_works_without_a_gpu`. **Nothing under `taniteval/` had been touched.**
`MEASURED`: re-running **those same two files alone**, with the CPU quiet, gives **34 passed in 65 s**;
the full suite alone then gives **1136 / 0**.
⇒ ⭐ **The 22 failures were CPU CONTENTION FROM MY OWN CONCURRENT JOB.** The failing tests spawn
subprocesses with timeouts, and 12–16 busy ridge processes push them past the deadline.
⇒ ⚠️ **RULE: never run the suite as a gate while a multi-process CPU job is live — its FAILs are not
about the code.** *(Same family as `df` on a pod and `free` on Thor: a probe answering a different
question than the one asked. And the pipeline that produced it printed `exit=0`, because the exit code
belonged to `tail`.)*

---

## 9. ⛔ ESCALATIONS — these need a decision and are not filed in a README

1. ⛔⛔ **DECISION-GRADE — TWO DOCUMENTS STILL CITE STALE LADDER NUMBERS, AND ONE OF THEM CAN COST A
   TRAINING RUN.** `MEASURED` by opening both files today:
   * `…/Research/2026-08-18-pooling-bottleneck-R1R2/POOLING_BOTTLENECK_R1R2.md` **§1.5, lines
     111–119** — unchanged since commit `87ff185`, i.e. it predates the entire correction. It quotes
     `n_agents_all` r² **0.076** (now **0.1613** on the 3-seed mean), `ego_curv` **0.0001** (now
     **0.0000**), `lead_closing` **0.0000** (now **0.0009**), and *"partialling `v0` out leaves
     r +0.052"* (now **r −0.107**). Its five **line-number** citations into `LATENT_LINEAR_LADDER.md`
     (`:178-194`, `:158-164`, `:234`, `:264`, `:299-316`) are all invalidated by the in-place
     corrections and must become **section-heading** citations.
   * `…/Research/2026-08-17-O234-DESIGN-RESEARCH.md` **§3.4a (line 427) and its E-PROBE-A row (line
     1015)** — the same three r² values, plus *"K1 −1.562 PASS vs +1.580 FAIL"*. ⚠️ Its own top
     banner (line 13) already flags §3.4 as inverted; **§3.4a's table was not updated with it.**
   ⛔ **I did not edit either file** — both are in another stream's `Research/` directory, and the
   pooling one was touched by `280fb9b` (C104) hours ago, i.e. it is live. **The request:** re-quote
   from `LATENT_LINEAR_LADDER.md` §8.1's **3-seed** column, not §4.2's seed-0 one, and cite by
   section heading.
2. ⚠️ **OPEN — A `C-V0` ARM IS REQUIRED PER WINDOW FAMILY, not per experiment.** `MEASURED`: the
   re-read's single `C-V0` sits on the cells cache, so **33 of its 165 rows had no trivial-proxy
   control at all and nothing in the artifact said so** (§5). A control fitted on different windows is
   not a control. The `proxytok` arm added here closes it for this ladder; the general rule needs
   adopting.
3. ⚠️ **OPEN — THE ALPHA GRID IS STILL BINDING.** The repaired alpha lands on a **grid edge** on
   **78 / 94 / 82** of 176 rows (seeds 0/1/2), mostly at the top (1e7). On those rows "alpha selection"
   is a boundary hit, not an optimum. **A wider grid belongs in the next run**; `alpha_at_grid_edge`
   already makes it checkable.
4. ⚠️⚠️ **OPEN, AND IT HAPPENED AGAIN — "STAGE, NEVER PUSH" DOES NOT PROTECT AN AGENT'S WORK. THIS IS
   THE SIXTH OCCURRENCE.** ⛔ **I never ran `git commit` or `git push`.** `MEASURED`: while this run
   was still producing route B, commit **`14623d7`** — *"C105 — a defect that would have killed S-T at
   a checkpoint boundary…"*, **66 files, naming none of mine** — swept in **5 of my staged files**
   (`LADDER_3SEED.md`, `code/chain_3seed.sh`, `code/reread3_table.py`, `code/run_proxytok.sh`,
   `raw/reread3_table.json`), all in a mid-run state. Prior occurrences: `60265d3`, `3d41bd0`,
   `109406c`, `ec26ca9`. ⇒ **The rule needs a mechanism, not another warning.** *(I note a
   `stack/scripts/scoped_commit.py` has appeared in the shared index from another stream — if that is
   the mechanism, it should be adopted explicitly rather than left as an untracked convention.)*
   **Escalating rather than proposing a policy change unilaterally — this is the PI's to decide.**

5. ⚠️ **A VERIFICATION THAT FALSELY PASSED, RECORDED BECAUSE THE BRIEF WARNS ABOUT EXACTLY THIS.**
   My first staging check was a shell loop `for P in $(git ls-files --cached …)`. Every path in this
   repo contains spaces (`TanitAD Research Hub`, `Architecture & Inference`), so the shell
   **word-split them into 360 fragments**; `git ls-files --stage` and `git hash-object` both returned
   **empty** for each fragment, the two empties compared **equal**, and the check printed
   ⛔ **"files checked: 360, blob mismatches: 0 — ALL STAGED"**. `MEASURED`: redone with `-z` and no
   shell splitting, the true figure is **72 files, 0 missing, 0 mismatches** — the same verdict, but
   the first run had no power to produce any other one. ⇒ **A check that cannot FAIL is not a check.**
   *(Same family as C79's missing positive control, and as the polling monitor that matches its own
   echoed command.)*

6. ⭐ **PROPOSED `RETRACTION_LOG.md` ENTRY — text ready, DELIBERATELY NOT APPENDED BY ME**, because
   the log is serialised and several agents are live. ⚠️ **Number it at write time.**
   > **C1xx — C100's LAST SUBSTANTIVE ROW WAS A SEED ARTEFACT, AND C103's OWN REASSURANCE ABOUT THE
   > TWO REPAIR ROUTES WAS A ONE-SEED CLAIM TOO.** The 165-row re-read was re-run at 3 seeds on both
   > repair routes (`…/incoming/2026-08-18-ladder-3seed/`; reproduction gate **3 465/3 465** fields
   > identical to the banked seed-0 rows). **C100's one substantive survivor — `ll_s09000 lead_gap`,
   > K1B +0.748 [+0.002, +1.624] — separates on ONE seed of three** (K1B +0.748 / −0.022 / +0.145;
   > alpha 1e3 / 1e7 / 1e5). ⇒ ⛔ **The substantive count is ZERO, on BOTH routes.** **22 of the 87
   > rows have no stable verdict at all.** ⭐ The frozen-alpha mechanism is now measured on the full
   > population: **the defective instrument picks the same alpha on all 3 seeds for 132 of 165 rows,
   > the repaired one for 42.** ⚠️ **And the same class caught C103's own route reassurance:** *"44
   > rows, 0 verdicts differ"* was measured at one seed; at three seeds over 176 rows **2 verdicts and
   > 11 guard verdicts differ and max |ΔK1| grows 0.396 → 0.721** — though neither difference is a
   > PASS↔FAIL flip and both routes return the same inventory. ⇒ **ROOT-CAUSE CLASS (C103's, applied
   > recursively): every reassurance about an estimator — stability, route-equivalence, seed-
   > insensitivity — inherits the seed count it was measured at. A one-seed reassurance about a
   > seed-sensitive instrument is not evidence, INCLUDING when it appears inside the retraction that
   > established the sensitivity.** *(Sibling, same run: 33 of the 165 rows carried no trivial-proxy
   > control because the only C-V0 arm sat on a different window set — a missing control whose absence
   > was invisible.)*

---

## 10. DELIVERABLE MANIFEST

**All paths relative to the repo root** `G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\`.
⛔ **Nothing produced by this run lives in only one place** — every artifact is in the repo and staged.

| artifact | path | what it is |
|---|---|---|
| **this report** | `repo:…/incoming/2026-08-18-ladder-3seed/LADDER_3SEED.md` | the findings |
| **in-place update** | `repo:…/incoming/2026-08-17-latent-linear-ladder/LATENT_LINEAR_LADDER.md` | banner, §0.5, §2, §5.3/§5.3a/§5.3b, §8.0/§8.1, §10.3a, §12, §14.2, §17.2b, escalations 3/4/6 |
| **the chain** | `repo:…/incoming/2026-08-18-ladder-3seed/code/chain_3seed.sh` | derived from — **not editing** — `…/2026-08-18-k1-degeneracy-guard/code/chain_reread.sh`; OUT/SEEDS/MODE parameterised, md5 + real-import sync gate |
| **the missing control** | `repo:…/incoming/2026-08-18-ladder-3seed/code/run_proxytok.sh` | `C-V0` on the TOKENS window set, both routes |
| **the table builder** | `repo:…/incoming/2026-08-18-ladder-3seed/code/reread3_table.py` | opens banked JSON and arranges it; computes nothing about the model |
| ⭐ **route A, 3 seeds** | `repo:…/raw/reread_unpen/ll3_*.json` + `log3_*.txt` (16 arms) | C100's route |
| ⭐ **route B, 3 seeds** | `repo:…/raw/reread_centred/ll3_*.json` + `log3_*.txt` (16 arms) | §5.3/§8's route |
| ⭐ **the 3-seed tables** | `repo:…/raw/reread3_table.json` | R0 gate · R1 per-row (176) · R2 inventory · R2b seed stability · R3 trivial proxy · R4/R4b routes · R5 quotable PASSes · R6 rung profile |
| suites | `repo:…/raw/suite_taniteval.txt`, `repo:…/raw/suite_stack.txt` | §8 |

**Inputs read and NOT copied** (large, already banked by the precedent runs): the seven
`…/scratchpad/sp2/cache_*/latents.pt`, `…/scratchpad/pc/cache_orcdir/latents.pt`, the four
`…/scratchpad/ll/cache_egoorc_n*/latents.pt`, `…/scratchpad/sp2/p3_selection.json`,
`…/scratchpad/sp2/lead130_agents.jsonl`, and the 130-clip episode cache
`…/scratchpad/sp2/cache/slotprobe-lead130-w120-256x640cyl/`.
⚠️ **The `cache_egoorc_n*` caches exist only in scratch** (regenerable in ~30 s by `ll2_ego_oracle.py`),
as recorded by the precedent run.

⛔ **STAGED, NEVER PUSHED.** No `git commit` and no `git push` was run by this agent.
