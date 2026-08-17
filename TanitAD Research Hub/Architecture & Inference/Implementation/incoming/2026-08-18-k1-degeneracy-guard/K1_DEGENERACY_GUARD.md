# THE K1 DEGENERACY GUARD · AND THE RE-READ OF THE 165 LADDER ROWS

**Date:** 2026-08-18 · **Branch:** `agent/arch-inf-20260803` · **Agent:** k1-degeneracy-guard
**Eval tier:** ⛔ **T0-DIAGNOSTIC throughout.** A frozen-latent ridge readout is a world-model
diagnostic. Per `EVAL_DOCTRINE.md` only T1 carries a capability claim; nothing here is a driving
number.
**GPU used: ZERO.** Thor's v6F S-W 30k run (PID 25477) was not touched, not read, not queued behind.
Every fit here is CPU arithmetic on already-banked latent caches.
**Estimator:** `taniteval.ci.paired_episode_cluster_bootstrap`, seed 0, n_boot 2000.
⛔ `overlapping_holdout_se` appears nowhere in this path.
**Evidence classes:** `MEASURED (ours + artifact path)` · `DERIVED (algebra on our source, file:line)`
· `PUBLISHED` · `INHERITED (not re-verified)` · `ESTIMATED` · `HYPOTHESIS`.

---

## ⭐ THE ANSWER IN ONE PAGE

**JOB 1 — the guard exists, it decides with NO free parameter, and it is pinned in both directions.**

The guard rests on an **exact decomposition**, not a threshold. Writing `c_own = mean(pred)`:

> **K1 = [MAE(pred) − MAE(c_own)] + [MAE(c_own) − MAE(C-CONST)] = K1B + K1C**

* **K1B** is what the readout's **variation** buys over its own mean level — the
  **latent-attributable** part, and it is **algebraically invariant to C-CONST** (pinned by test).
* **K1C** is a contest between two constants. On a z-scored design the ridge intercept is
  `mean(y_train)`, which carries **zero** latent information, so K1C is never evidence about the
  latent.

⭐ **And K1B is bounded exactly**: by the reverse triangle inequality then Jensen,
`|K1B| ≤ mean|pred − c_own| ≤ pred_sd`. ⇒ **a row whose `|K1_delta|` exceeds its own `pred_sd` has a
PROVEN constant-offset component of at least `|K1_delta| − pred_sd`** — no threshold, no refit, no
bootstrap, and computable from fields already banked in every artifact.

**JOB 1's second question — mean or median for C-CONST? ⇒ THE MEDIAN STAYS, and the question
dissolves rather than being adjudicated.** Full argument in §3; the short form is that MAE's optimal
constant *is* the median, so switching to the mean would **weaken the baseline** — manufacturing
PASSes, which is C97's own failure mode a third time — and K1B does the isolation job better than
"C-CONST = mean" ever could, because it uses the arm's **own** prediction level and is invariant to
the choice.

⭐ **THE GUARD'S OWN CONTROLS, ON REAL BANKED DATA — both reported, per C92's rule.**
The zero-compute screen over all **214** banked verdict rows:

| control | what it must do | MEASURED |
|---|---|---|
| **negative** (`torch.randn` / matched-random latent arms) | be caught | ⭐ **5 of 5** of their PASSes flagged |
| **positive** (GT-oracle arms, `ego_v0` anchors) | **NOT** be caught | ⭐ **0 of 9** of their PASSes flagged |
| **trivial-proxy** (C-V0, the ego-speed scalar) | be measured, not conflated | 8 PASSes, 2 flagged as degenerate, 6 left standing — correctly, because the v0 proxy's skill is **trivial but real**, which is a different axis from degenerate |

⛔ **And the screen's headline number: of the 34 separated verdicts already sitting on a REPAIRED
solve, 21 are screened SUSPECT (62 %).** C97's `n_agents_all` null-PASS was not a one-off.

**JOB 2 — 165 rows re-read under repair + guard. ⛔ OF 87 BANKED SEPARATED-FAILs, **ONE** SURVIVES
AS A SUBSTANTIVE FINDING.**

| of the 87 banked separated-FAILs | n |
|---|---|
| die at the repair (CI now spans zero) | 23 |
| ⛔ survive the repair, **killed by the guard** | ⛔ **42** |
| flip to PASS | 11 |
| survive repair **and** guard | 11 |
| …of which **substantive** (`\|K1B\|/gt_sd ≥ 0.02`) | ⭐ **1** |

**The one survivor is `ll_s09000 lead_gap`, K1 +1.291, K1B +0.748 [+0.002, +1.624].** The other 10
are all `ego_yawrate` at **K1B = +0.0000 with CI [+0.0000, +0.0000]** — and **two of those ten sit on
RANDOM-LATENT NULL arms**, which is the proof that they cannot be findings about any latent: a
`torch.randn` cache produces the identical verdict.

⭐ **THE GUARD'S CLEANEST DEMONSTRATION — 15 arms, one target, one window set.** Under the repair
`n_agents_all` returns **K1 PASS on all 15 arms**. The guard rejects **exactly the two that contain
zero information** (`ll_nullmatched` K1B **−0.003**, `ll_tok11250null` K1B **+0.000**, both
DEGENERATE-CONSTANT) and passes every information-bearing arm. And the ego-oracle SNR sweep is
**monotone in the injected noise** — K1B −2.219 (n0.1) → −2.109 (n1) → −1.950 (n3) → −1.044 (n10):
the guard's statistic behaves as a **dose-response curve**, which is what a working instrument does.

⛔⛔ **AND THE TRIVIAL-PROXY CONTROL KILLS THE ONE NEW PASS ANYWAY — C92's defect, on a new rung.**
The v6 arm's `n_agents_all` PASS (K1B **−2.785**) must be read against **the ego-speed scalar alone**
(`C-V0`, K1B **−2.243**): the margin is **−0.541 on a target of sd 46.46 = 0.012 gt_sd**, i.e.
**~80 % of the v6 latent's scene-density readout is reproducible from `v0`**. And on `n_agents_grid`
⛔ **the single scalar PASSES (K1B −0.540) while the 2 048-dim v6 latent does not even separate
(K1 +0.576).** ⇒ **"the v6 latent reads scene density" is NOT supported**; the mechanism is faster
ego ⇒ open road ⇒ fewer agents.

---

## 1. THE GUARD

### 1.1 `DERIVED` — the decomposition, with the algebra stated so it can be checked

`taniteval/taniteval/degeneracy.py`. For per-window absolute errors on a common window set:

```
K1   = mean|pred − y| − mean|C-CONST − y|
K1B  = mean|pred − y| − mean|c_own  − y|      c_own = mean(pred)
K1C  = mean|c_own − y| − mean|C-CONST − y|
K1   = K1B + K1C                                EXACT — pinned to < 1e-9
```

**The bound.** Per window, `||a| − |b|| ≤ |a − b|` with `a = pred_i − y_i`, `b = c_own − y_i`, so
`a − b = pred_i − c_own`. Averaging, `|K1B| ≤ mean|pred − c_own| = pred_mad`, and `pred_mad ≤ pred_sd`
by Jensen. Both inequalities are pinned by a 12-seed fuzz test, because layer 1's validity over 214
banked rows rests entirely on them.

### 1.2 The three layers — and which one decides

| layer | statistic | status |
|---|---|---|
| **1** | `pred_sd < \|K1\|` | **EXACT (a theorem).** Screens banked JSON at zero compute. |
| **2** | **K1B, paired episode-cluster bootstrap** | ⭐ **THE DECISION.** Same estimator as K1 itself. **No free parameter.** |
| **3** | `sd_ratio = pred_sd / gt_sd` | **DESCRIPTIVE.** The only layer with a threshold (`SD_RATIO_FLAT_FLOOR = 0.05`) — so it never decides alone. |

⚠️ **Layer 3 is the one C97 asked for by name, and it is deliberately the weakest.** A bare
`pred_sd/gt_sd` cut is a *threshold nobody registered* — precisely what `ci._render_bounds` refused
to introduce when it faced the same temptation. It is reported because it is readable and because it
works on banked files; the verdict rests on layer 2.

**Verdicts:** `OK` · `CONSTANT-OFFSET-ONLY` (K1 separates, K1B does not) · `DEGENERATE-CONSTANT`
(the same failure, where the readout is *also* a flat line) · `NO-VERDICT-TO-GUARD` (K1 never
separated). Producers emit `K1_PASSES_GUARDED` beside the raw `K1_PASSES`.

### 1.3 ⚠️ A FALSE POSITIVE MY OWN FIRST DRAFT WOULD HAVE MANUFACTURED — caught, and pinned

The first version of `k1_guard` let `flat_line` **short-circuit** K1B, so anything with
`sd_ratio < 0.05` was rejected outright. **That contradicts this module's own docstring** ("layer 3
never decides alone") and it is wrong on the data we actually have: `sd_ratio` compares against
`gt_sd`, which a handful of extreme windows inflate without making the bulk harder. **`n_agents_all`
is exactly that shape — `gt_sd` 46.459 against a median of 34.0.** A readout tracking the bulk can
look "flat" while genuinely beating its own mean.

⇒ **K1B now overrides the flat-line label**, and `test_a_HEAVY_TAILED_target_does_not_turn_a_real_readout_into_a_false_alarm`
pins it: a readout at `sd_ratio` **0.0007** with a separated negative K1B must return **OK**.

⭐ **This is the brief's own warning landing on me inside the same file that exists to prevent it** —
*a guard that rejects everything is as useless as one that rejects nothing.* I built the
rejects-everything kind first, and only the requirement to test the direction I was not trying to fix
caught it. **The failure mode is not rare; it is the default.**

### 1.4 ⛔ THE GUARD IS NOT A SUBSTITUTE FOR THE C92 REPAIR — they fix different defects

`MEASURED` (`raw/reread/llGATE_nullmatched.json`): on the **incumbent** solve the random-latent null
scores `n_agents_all` **K1 +5.9155**, a separated FAIL — and the guard returns **OK**, i.e. *"this
verdict really is about the prediction's variation."* It is: the C92 penalty **forced** the noise arm
to load features to reach y's level, so it had variation, and that variation genuinely made it worse.

⇒ **C92 biases the FLOOR and needs the repair. C97 biases the CONSTANT and needs the guard. Neither
fix covers the other, and a row is only readable once BOTH are applied.** Every number in §4 carries
both.

---

## 2. `MEASURED` — THE ZERO-COMPUTE SCREEN OVER ALL 214 BANKED ROWS

`code/k1_screen_banked.py` → `raw/k1_screen_banked.json`. ⛔ Every row opened from a JSON on disk,
never from a headline (C91).

| | |
|---|---|
| rows read | **214** (24 files) |
| rows carrying a verdict (separated) | **145** |
| ⛔ **screened SUSPECT** | **44** |
| …on the INCUMBENT solve | 23 / 111 |
| ⛔ …on the **REPAIRED** solve | ⛔ **21 / 34 (62 %)** |
| PASS rows screened suspect | **11 / 32** |
| FAIL-separated rows screened suspect | **33 / 113** |

⭐ **The single most alarming row, and it is not the one C97 named.** `ll_tok11250null.json` — a
**MATCHED-RANDOM feature set**, i.e. pure noise — carries **three separate K1 PASSes**
(`ego_accel` −0.007, `lead_closing` −0.011, `lead_inv_ttc` −0.001), all on the **incumbent** solve,
all at `sd_ratio` **0.0075–0.0099**. C97 found one noise-PASS on the repaired solve; there were
**five noise-PASSes across the corpus**, and two of them predate the repair entirely.

⇒ ⚠️ **The mean-vs-median degeneracy is NOT purely a side-effect of the C92 repair.** It also fires
on the incumbent solve wherever the target's spread is small enough that a nearly-constant readout
lands closer than the train median. The repair made it *worse and more visible*; it did not create it.

---

## 3. ⛔ THE DECISION: C-CONST STAYS THE **MEDIAN**

The mismatch C97 identified is real — a fully-shrunk repaired ridge **is** the train mean, while
C-CONST is the train median. Four reasons the answer is nevertheless *not* to switch:

1. **The loss picks the baseline, and MAE's optimal constant is the MEDIAN.** C-CONST = train median
   is therefore the **strongest honest constant** under the loss K1 actually uses. Replacing it with
   the mean would install a constant that is *worse* on the fitting distribution — i.e. it would
   **mechanically manufacture PASSes**. ⚠️ That is C95's lesson word for word: *loosening a criterion
   is a candidate FAIL-suppressor*, and it is exactly how C97 happened in the first place.
2. **The observed inversion is a train/eval SHIFT, not evidence the mean is a better baseline.** On
   `n_agents_all` the train mean (62.8) beat the train median (34.0) *on the eval episodes* because
   the eval clips' centre sits above the train median. A baseline chosen because it lands better on
   eval is a baseline chosen **on the eval set** — the same hindsight the O2 report was careful to
   label as cheating when it did the alpha sweep.
3. **It would silently rewrite 214 banked verdict rows** whose filenames would not change. That is
   the C92 precedent exactly: make the correct behaviour **available and explicit**, re-read the
   banked numbers, never mutate the code underneath them.
4. ⭐ **K1B strictly dominates the proposed fix.** "Compare against the train mean" is an attempt to
   isolate the latent from the constant. K1B does that *better*: it uses the arm's **own** prediction
   level (which equals `mean(y_train)` only when the fit is fully shrunk, and is the correct
   reference when it is not), and it is **invariant to C-CONST** — pinned by
   `test_K1B_is_INVARIANT_to_the_choice_of_C_CONST`. **Switching C-CONST to the mean would be a
   worse version of the guard.**

**What changes in practice, so the gap is visible rather than invisible:** `k1_guard` reports
`c_mean_value`, `c_mean_err` and `mean_minus_median_const_gap` on every row. **No default flips**, and
every banked artifact keeps its meaning.

---

## 4. JOB 2 — THE RE-READ OF THE 165 INCUMBENT LADDER ROWS

### 4.1 What was run, and the two gates it had to clear first

15 incumbent `ll_*.json` (`fit_mode: pc6`) × 11 targets = **165 rows**, re-fitted with
`--fit-mode unpen` — the repair taken **from the module** (`ridge_fit(..., intercept_col=-1)`) — and
the guard on every rung. Same caches, same `p3_selection.json` split, same targets, same estimator.
`code/chain_reread.sh` → `raw/reread/llR_*.json`; table by `code/reread_table.py` →
`raw/reread_table.json`.

⛔ **The scratch copies were STALE and a launch from them would have silently re-run the defect.**
MEASURED: the scratch `pc6_linear_readout.py` still carried the **pre-C92** `ridge_fit(X, y, alpha)`
with **no `intercept_col` parameter at all**. This is the pod-drift trap in local costume. The chain
now syncs repo→scratch and verifies with a **real `import` + signature check**, not an `ls`.

✅ **REPRODUCTION GATE — the edited producers still reproduce the banked incumbent BIT-EXACTLY.**
`308` field comparisons (2 arms × 11 targets × 14 fields: err, K1 δ/lo/hi, separation, verdict,
alpha, corr, pred_sd, gt_sd, R², n) — **zero divergence** (`raw/reread/llGATE_*.json`). The guard and
the `unpen` branch are additive; this is what turns "should be" into MEASURED.

⚠️ **Seed 0 only, and why that is not a shortcut.** Both the C91 audit and this table read
`per_seed["0"]`, and `fit_one` draws its own `default_rng(seed)` per call — so seed 0's row is
bit-identical whether or not seeds 1–2 also ran. Seed stability *under the repair* is already
MEASURED on the 4 banked `ll_rep_*` arms (3 seeds each).

### 4.2 ⛔ A CLAIM OF MINE, MEASURED AND RETRACTED IN THE SAME PACKAGE

I wrote in `ll1_ladder._solve` that `centred` and `unpen` "can differ by ~1e-12 on the inner split".
**The mechanism was right and the magnitude was invented.** MEASURED
(`raw/equiv_centred_vs_unpen.json`, 4 arms × 11 targets):

* **Full fit: identical to 5e-14** across alpha 1e-2..1e7 — the block-diagonal argument holds.
* ⛔ **Inner split: up to 6.3e-2** on a synthetic subset, and the two routes' **inner-split MAE on
  the real caches differs by up to 0.74**. Not 1e-12 — **eleven orders of magnitude out.**
* ⇒ **that difference flips near-tied alpha choices.** On `ego_v0`@11250 the two candidate alphas sit
  **9.4e-3** apart in inner MAE, `centred` picks 1e4 and `unpen` picks 1e5, and **K1 moves
  +0.4274 → +0.0317**. Across the 44 overlapping rows: **2 alpha choices differ, 0 verdicts differ.**

⇒ **The two repair routes are NOT numerically interchangeable and their numbers must not be pooled.**
All 165 rows here use `unpen`. *(Root-cause class: an ESTIMATE stated in the register of a
MEASUREMENT — operating standard #1, in a docstring rather than a report.)*

### 4.3 `MEASURED` — the 165-row result

| | old (incumbent) | new (repaired) |
|---|---|---|
| PASS | 20 | 39 |
| FAIL-separated | 87 | 94 |
| not-separated | 58 | 32 |
| **verdicts changed by the repair** | — | **93 / 165** |

**Guard verdicts over the 165:** `DEGENERATE-CONSTANT` **62** · `OK` **46** ·
`CONSTANT-OFFSET-ONLY` **25** · `NO-VERDICT-TO-GUARD` **32**.

**Crossed with the repaired verdict:**

| repaired verdict | guard OK | CONSTANT-OFFSET-ONLY | DEGENERATE-CONSTANT |
|---|---|---|---|
| PASS (39) | **34** | 3 | 2 |
| FAIL-separated (94) | 12 | 22 | **60** |

⇒ ⛔ **82 of the 94 repaired separated-FAILs (87 %) are constant-contests, not facts about a latent.**

**Magnitude, applied on top (§1.3's caveat):** of the 46 `OK` rows, only **30** clear
`|K1B|/gt_sd ≥ 0.02` — and ⭐ **0 of those 30 sit on a null arm.** The two null rows that reach `OK`
(`ego_yawrate`) both land in the negligible bucket. **Guard + scale together separate the null arms
from the substantive rows perfectly, at 165 rows.**

**The top of the substantive list is the positive controls, in the right order** — `C-V0` on `ego_v0`
(K1B −4.186, a scalar predicting itself), `EGO-ORACLE-n0.1` on `ego_v0` (−4.153),
`GT-ORACLE-DIRECT` on `lead_gap` (−4.571), `GT-ORACLE` on `n_agents_grid` (−7.075). **The instrument
measures when there is something to measure**, which is what licenses reading a negative row as a
fact about the latent (the D1-withdrawal lesson).

### 4.4 ⭐ WHICH FAILs WERE EVER FINDINGS — the deliverable distinction

| | verdict |
|---|---|
| `ll_s09000` **`lead_gap`** — K1 +1.811 → **+1.291**, K1B **+0.748 [+0.002, +1.624]**, `\|K1B\|/gt_sd` 0.121 | ⭐ **SURVIVES — the only substantive one of 87** |
| `ll_s11250` `lead_gap` — K1 +1.580 → +0.736, K1B **+0.404 [−0.214, +1.101] NOT separated** | ⛔ **DOWNGRADED to CONSTANT-OFFSET-ONLY** |
| 10 × `ego_yawrate` across 10 arms — K1B **+0.0000 [+0.0000, +0.0000]** | ⛔ **arithmetic, not findings** — 2 of the 10 are on NULL arms |
| 23 further banked FAILs | die at the repair; CI now spans zero |
| 42 further banked FAILs | survive the repair, **killed by the guard** |

⚠️ **This SHARPENS the O2 stream's "2 of 3 FAILs survive an unbiased floor".** On the ladder's own
`lead_gap` rows the **@9000 FAIL survives the guard and the @11250 FAIL does not** — the residual
+0.736 is not separable from its constant offset. The O2 conclusion was correct as far as the repair
went; the guard moves one of its two survivors.

### 4.5 ⛔ THE `n_agents_all` STORY — the one place the repair CREATED a new PASS, and why it is not one

Under the repair `n_agents_all` PASSes on **all 15 arms**. Read with both controls:

| arm | K1B | guard | reading |
|---|---|---|---|
| `GT-ORACLE-DIRECT` | **−9.546** | OK | the ceiling |
| `v6F@11250` | −2.785 | OK | |
| `v6F@10000 / @9250 / @9000` | −2.805 / −2.759 / −2.715 | OK | flat across checkpoints |
| ⛔ **`C-V0` (ego speed, ONE scalar)** | ⛔ **−2.243** | OK | **80 % of the arm's value** |
| `EGO-ORACLE` n0.1→n10 | −2.219 → −1.044 | OK | monotone in injected noise |
| `RANDOM-LATENT-NULL` | **−0.003** | ⛔ DEGENERATE | caught |
| `TOKENS MATCHED-RANDOM NULL` | **+0.000** | ⛔ DEGENERATE | caught |

⇒ **The v6 latent beats the single ego-speed scalar by 0.541 agents on a target of sd 46.46 —
0.012 gt_sd.** And on `n_agents_grid` **the scalar PASSES while the latent does not separate at all.**
⛔ **This is C92's finding again on a different rung: a margin over a random null read as capability
when a scalar the model is HANDED explains ~80 % of it.** The negative control is necessary and not
sufficient; the trivial-proxy control is what settles it.

---

## 5. ⛔ ESCALATIONS — decisions or edits by their owners, filed HERE and not as "please merge"

1. ⛔ **C97's blocking requirement is DISCHARGED.** `taniteval.degeneracy.k1_guard` exists, is wired
   into **both** producers, and is pinned in both directions (29 tests). Repaired PASSes are
   quotable **only** with `guard_verdict: OK` **and** a stated `|K1B|/gt_sd`.
2. ⛔ **`LATENT_LINEAR_LADDER.md` must be re-read against `raw/reread_table.json`.** **93 of its 165
   verdict rows change** under the repair, and **82 of the 94 repaired separated-FAILs are
   constant-contests**. I did not edit that document — its owner should, and the table is the input.
3. ⛔ **Any claim of the form "the v6 latent carries scene density" must be withdrawn or restated**
   with the C-V0 margin (§4.5). The `n_agents_all` PASS is ~80 % ego-speed proxy and the
   `n_agents_grid` result is *worse* than the scalar.
4. ⚠️ **`ridge_fit`'s default stays PENALISED and `--repair-intercept` is opt-in** (C92 precedent).
   Anyone reading a `pc6_ridge_*.json` or `ll_*.json` **without** a `k1_guard` block is reading a
   pre-2026-08-18 artifact on a biased floor — the guard block's presence is the version marker.
5. ⚠️ **C-CONST stays the train MEDIAN** (§3). If the PI wants the mean instead, that is a decision
   that rewrites 214 banked rows and weakens the baseline under MAE; my recommendation is no, and
   K1B makes it unnecessary either way.
6. ⚠️ **`centred` and `unpen` results must not be pooled** (§4.2). The 4 banked `ll_rep_*` files are
   `centred`; everything here is `unpen`.
7. ⚠️ **Proposed `RETRACTION_LOG.md` entry — text ready, NOT appended by me** because the log is
   serialised and other agents are live. ⚠️ **Number it at write time, not from this text:** while I
   was running, HEAD moved `87ff185 → 4e7a18d` and **C98 and C99 were both taken by another stream**,
   so the next free id is **C100** *unless* it has moved again — check before pasting.
   > **C100 — A GUARD BUILT TO STOP FALSE PASSES WAS FIRST BUILT AS A FALSE-FAIL MACHINE, AND THE
   > BANKED FAILs WERE 1-IN-87.** The C97 guard's first draft let `sd_ratio` short-circuit K1B,
   > which rejects a genuinely-skilled readout on a heavy-tailed target. Re-reading the 165 ladder
   > rows under repair + guard: of **87** banked separated-FAILs, **23** die at the repair, **42**
   > are killed by the guard, **11** flip to PASS, and of the **11** that survive both, **10 are
   > `ego_yawrate` at K1B +0.0000** — two of them on random-latent nulls. **One substantive finding
   > remains.** ⇒ ROOT-CAUSE CLASS: *a criterion corrected in one direction is a candidate bias in
   > the other, and the correction is not finished until both are tested* (C95, third instance).

---

## 6. SUITES — MEASURED by me

| suite | result |
|---|---|
| `taniteval` | ⭐ **1136 passed, 0 failed**, 229 s (`raw/suite_taniteval.txt`) — the 1107 baseline **+ exactly my 29 new tests** |
| `stack` | ⭐ **3842 passed, 0 failed, 7 skipped, 2 xfailed**, 546 s (`raw/suite_stack.txt`) — **GREEN** |

⚠️ **The `stack` count is +26 on the 3816 baseline I was briefed with, and NONE of it is mine.** I
staged nothing under `stack/` (`git diff --cached --name-only | grep ^stack/` is empty). The delta is
another agent's `stack/tests/test_v6_dump_sw_latents.py` (**24 test defs**), which was untracked at
the start of this session and is tracked now. **Stated rather than quietly absorbed into "green" —
a suite count that moved for someone else's reason is exactly the kind of number this programme has
mis-attributed before.**

⚠️ **Wall-clock note:** the box was shared with another agent's `er10_pool_ladder.py` throughout, so
these timings are not comparable to an idle-box baseline.

⚠️ **Ops fact, MEASURED, worth carrying:** `TaskStop` on a background **wrapper** did **not** stop the
chain — the shell kept spawning the next arm after each child died, and only killing the `bash` PIDs
stopped it. Same family as the `pgrep -f` self-match trap: **kill the SUPERVISING shell, then the
children, then re-verify `ps` is clear.** Verified here that only my own PIDs were killed; another
agent's `er10_pool_ladder.py` and `pytest` were identified by full command line and left running.

---

## 7. ⚠️ WHAT I DID NOT DO

* **No re-read of the 4 banked `ll_rep_*` files.** They are already on a repaired solve; §2's screen
  covers them (21 of their 34 separated verdicts are SUSPECT) but they were not refitted with the
  guard. That is a ~10-minute zero-GPU job and it is **escalation 2's** natural companion.
* **No magnitude GATE.** `|K1B|/gt_sd` is reported, never thresholded — a cut there would be a
  tunable nobody registered.
* **No edit to `LATENT_LINEAR_LADDER.md`, `PROBE_POSITIVE_CONTROL.md`, `MODEL_REGISTRY.md` or
  `RETRACTION_LOG.md`.** Other agents own them; §5 is the escalation.
* **No T1 claim, no GPU, no gradient measurement.** Everything is T0-DIAGNOSTIC.
* **The latent caches remain in scratch and were NOT banked.** They were already present from the
  prior stream, so **nothing was regenerated and no regeneration time was spent**. They are
  multi-hundred-MB `.pt` files regenerable from checkpoints; banking them is a storage decision for
  the owner of the probe-positive-control package, escalated rather than assumed.

---

## 8. DELIVERABLE MANIFEST

⭐ **Everything below is in the REPO and STAGED. Nothing exists only in scratch or only in this
agent's context.** The one disclosed scratch dependency is the input caches (§7), which pre-date this
package.

| artifact | where it lives | what it is |
|---|---|---|
| `taniteval/taniteval/degeneracy.py` | `repo:taniteval/taniteval/` | ⭐ **the guard** — canonical, importable, program-wide |
| `taniteval/tests/test_k1_degeneracy_guard.py` | `repo:taniteval/tests/` | ⭐ **29 tests**, both directions + wiring pins |
| `pc6_linear_readout.py` (modified) | `repo:…/2026-08-17-probe-positive-control/code/` | guard wired in; `--repair-intercept` opt-in |
| `ll1_ladder.py` (modified) | `repo:…/2026-08-17-latent-linear-ladder/code/` | guard on every rung; `--fit-mode unpen`; corrected `_solve` docstring |
| `K1_DEGENERACY_GUARD.md` | `repo:…/incoming/2026-08-18-k1-degeneracy-guard/` | this report |
| `code/k1_screen_banked.py` | ″ `/code/` | layer-1+3 screen over all banked rows |
| `code/chain_reread.sh` | ″ `/code/` | the 15-arm re-read chain, with sync + import verification |
| `code/reread_table.py` | ″ `/code/` | the JOB-2 table builder |
| `code/equiv_centred_vs_unpen.py` | ″ `/code/` | the two-repair-routes measurement |
| `raw/k1_screen_banked.json` | ″ `/raw/` | 214 rows screened, 44 SUSPECT |
| `raw/reread_table.json` | ″ `/raw/` | ⭐ **165 rows: file · arm · target · old K1 · new K1 · K1B · guard · change** |
| `raw/equiv_centred_vs_unpen.json` | ″ `/raw/` | 44 paired rows, 2 alpha flips, 0 verdict flips |
| `raw/reread/llR_*.json` (15) | ″ `/raw/reread/` | the re-read artifacts, one per arm |
| `raw/reread/llGATE_*.json` (2) | ″ `/raw/reread/` | the bit-exact reproduction gate |
| `raw/reread/logR_*.txt`, `logGATE_*.txt` | ″ `/raw/reread/` | run logs |
| `raw/suite_taniteval.txt`, `raw/suite_stack.txt` | ″ `/raw/` | suite output |
