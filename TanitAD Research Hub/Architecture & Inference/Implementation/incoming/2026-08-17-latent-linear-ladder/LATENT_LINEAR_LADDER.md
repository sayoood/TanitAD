# ⛔ THE v6 OPERATIVE LATENT CANNOT LINEARLY REPORT THE EGO'S OWN SPEED — and the banked lead-gap signal is mostly an EGO-SPEED PROXY

> ## ✅ **CORRECTED 2026-08-18 — THIS DOCUMENT IS QUOTABLE AGAIN. Every table below has been re-read.**
>
> The 2026-08-18 stopgap banner (*"93 of 165 rows change — do not quote"*) is **discharged**. Every
> table has been re-rendered from the re-read artifacts, the four `ll_rep_*` arms that the re-read
> skipped have been refitted with the guard, and each table now carries the repair route it came from.
>
> ### ⭐ WHAT SURVIVES — and it is the headline
>
> | | |
> |---|---|
> | ⭐ **The title claim** | **SURVIVES and STRENGTHENS.** Under the repaired readout the latent's ego-speed readout **cannot separate from a constant** (K1 **+0.032 [−0.532, +0.508]**), while an oracle carrying `v0` under **10× noise PASSES** (K1 **−1.604**, guard **OK**). §3. |
> | ⭐ **The lead-gap finding** | **SURVIVES and STRENGTHENS.** The ~1.8 m margin over the null does not merely shrink — it **INVERTS**: the latent is now **0.694 m WORSE than the random-latent null**, and partialling `v0` out flips its correlation to **−0.107**. §5. |
> | ⭐ **The rung profile** | **SURVIVES.** Aggregate scene scale and longitudinal ego state at the top; everything **rotational** and **relative-motion** at the null across a 40× change in pooling. §4.2, §12. |
> | ⭐ **The instrument defect** | **SURVIVES** and is now C92 in `RETRACTION_LOG.md`, repaired program-wide. §10. |
> | ⭐ **NEW, and it is the sharpest statement in the document** | On the **mean of three seeds, 10 of 11 rungs go to a SINGLE SCALAR of ego speed or tie it** — including the one rung where seed 0 alone said the 2 048-dimension latent won. The 11th favours the latent by 0.00002 gt_sd on a degenerate rung. §5.3. |
>
> ### ⛔ WHAT DOES NOT SURVIVE — the per-row FAIL inventory, and nothing above it
>
> | | |
> |---|---|
> | ⛔ **The 87 banked separated-FAILs** | **65 of 87 were instrument.** 23 die at the C92 intercept repair, 42 are killed by the C97 degeneracy guard, 11 flip to PASS. Of the 11 that survive both, **10 are `ego_yawrate` at K1B +0.0000** — two of them on random-latent nulls. §2. |
> | ⛔ **"The v6 latent reads scene density"** | **WITHDRAWN.** It is **~80 % `v0`** at seed 0 and the scalar **wins outright** on the 3-seed mean. §5.3. |
> | ⛔ **§11's "the encoder's tokens carry more than the operative state"** | **INVERTS under the repair** — cells **+0.320** vs tokens **+0.263**. It was labelled a weak localisation and it was; the corrected direction agrees with E-R1-0. §11. |
> | ⛔ **"8 of 11 rungs have exactly zero seed spread"** | **FALSE under the repair.** pc6's truncation had frozen the alpha choice; unfrozen, the arm's own K1 moves by up to **2.812** across three inner-split seeds. §8. |
>
> ⚠️ **The failure mode to avoid when reading this document is concluding "the whole ladder was
> wrong".** It was not. What was overwhelmingly instrument is the **per-row FAIL inventory** — a body
> of verdicts that a biased floor manufactured. Every conclusion this document was written to support
> is stronger after the re-read than before it.
>
> **Sources:** `Project Steering/RETRACTION_LOG.md` **C92 · C97 · C100** ·
> `…/incoming/2026-08-18-k1-degeneracy-guard/` (the guard + the 165-row re-read) ·
> `…/incoming/2026-08-18-ladder-corrected/` (this correction's refits, gate and tables).

**Date:** 2026-08-17 · **Corrected:** 2026-08-18 · **Branch:** `agent/arch-inf-20260803`
**Agents:** latent-linear-ladder (original) · ladder-corrected (this re-read)
**Cites, and does not touch:** `…/incoming/2026-08-17-probe-positive-control/PROBE_POSITIVE_CONTROL.md`,
`…/incoming/2026-08-17-slot-probe-parity/SLOT_PROBE_PARITY.md`.
**Eval tier:** ⛔ **T0-DIAGNOSTIC.** A frozen-latent readout is a world-model diagnostic and is
**never** driving performance. No number here is an ADE, a closed-loop result, or a claim about how
the car drives.

---

## ⭐ THE ANSWER, IN ONE PARAGRAPH

**The sanity anchor cannot beat a constant, and the apparatus PASSES its positive control — so the
negative is a fact about the latent.** `MEASURED` (`…/2026-08-18-k1-degeneracy-guard/raw/reread/llR_s11250.json`,
repair route A): a ridge on `v6F-SW-30k@11250`'s operative cells recovers the **ego's own speed** at
**r = +0.321**, MAE **4.129 m/s** against a constant's **4.097** — **K1 +0.032 [−0.532, +0.508], NOT
SEPARATED**, i.e. it **ties a constant** and explains **10.3 % of the variance** of the quantity the
car most certainly knows about itself. ⭐ **The calibration that makes that number mean something:**
the *identical* ridge, on the *identical* windows, split, seeds and estimator, recovers a **synthetic
DISTRIBUTED encoding of that same `v0`** at **r = +1.0000** (0.10× noise), **+0.9985** (1×),
**+0.9830** (3×) and still **+0.8280 at 10× noise — where it earns a guarded K1 PASS (−1.604,
guard OK)**. ⇒ ⛔ **THE REAL v6 LATENT CARRIES LESS LINEARLY-READABLE EGO SPEED THAN A RANDOM
PROJECTION OF IT BURIED UNDER TEN TIMES ITS OWN MAGNITUDE IN NOISE — and the gap is now the
difference between a PASS and a tie, not between two FAILs.** The readout is not the limit; the
latent is. ⭐ **And this is the full operative state, not a lossy view of it:** `V6Stack.cells` is
`z_op.reshape(..., n_cells, d_readout)` (`stack/tanitad/models/v6.py:3710`, MEASURED from source
today — the original `:3729` had drifted) — **a pure reshape, bijective and lossless**, so flattening
the cells *is* `z_op`.

⭐⭐ **AND THE THREAD THIS RUN WAS OPENED TO FOLLOW COLLAPSES UNDER ITS OWN TRIVIAL-PROXY CHECK —
harder than the first reading found.** The precedent's sharpest new fact was *"under a linear readout
the v6 latents beat the random-latent null by ~1.8 m on lead gap, r +0.159 vs −0.018"*. **MEASURED
under the repair: the margin INVERTS. The latent scores 5.869 m and the random-latent null scores
5.175 m — the latent is 0.694 m WORSE than noise.** Meanwhile a ridge on the **EGO-SPEED SCALAR
ALONE** — one feature — recovers the lead gap at **3.571 m [3.151, 4.040], r +0.683, K1 −1.562
[−2.023, −1.136] SEPARATED, guard OK, PASS.** ⇒ ⛔ **Lead gap is largely a restatement of ego speed
(the time-gap law), and the latent's share of it is not merely mostly that proxy — after partialling
`v0` out the latent's lead-gap correlation is +0.073 → **−0.107**, a SIGN FLIP.** *(This is the third
time the trivial-proxy control has moved a headline in this programme; the fourth is §5.3.)*

⭐ **THE LADDER IS NOT FLAT, AND ITS SHAPE IS THE ACTIONABLE PART — and the shape survived the
re-read intact.** Against a null that sits at zero on every rung (|r| ≤ 0.030, r² ≤ 0.0009), the
latent's linearly-decodable content is: **coarse scene scale and longitudinal ego state at 1–15 % of
variance** (`n_agents_all` r² 0.152, `ego_v0` 0.103, `nearest_any` 0.096, `n_agents_grid` 0.020,
`ego_accel` 0.016, `lead_present` 0.012, `lead_gap` 0.005) — and **NOTHING AT ALL, indistinguishable
from the null, on everything ROTATIONAL or RELATIVE-MOTION**: `lead_closing` r² 0.0013,
`ego_yawrate` 0.0009, `lead_inv_ttc` 0.0008 *(its own null is 0.0009 — the arm is BELOW its null)*,
`ego_curv` 0.0000 *(null 0.0005)*. ⇒ **The latent has a weak "how fast and how cluttered is this
scene" signal and no measurable yaw, curvature, or closing-speed content whatsoever.** That lands
directly on the LATERAL and LONGITUDINAL metric families (§13).

⚠️ **WHAT THIS DOES NOT SAY.** It is a **LINEAR** readout: a quantity absent here could still be
present non-linearly, and this is not a claim that the world model is untrained. It is an
**EARLY-READ at 37.5 %** of training (11 250 of 30 000). ⚠️ And on **`lead_closing` / `lead_inv_ttc`
the positive control is WEAK by construction** — the oracle memory encodes positions only, no rate
features (`pc1_oracle_cache.py`: `[cx/60, cy/16, sin yaw, cos yaw, l/10, w/5, 1]`) — so those two
nulls are **"not linearly present"**, not **"the apparatus could have found it"** (§4.4).

---

## 0. ⛔ THE STAMPS THAT BIND EVERY NUMBER BELOW

1. **`v6F-SW-30k@11250`** unless another step is named. The checkpoint is part of the arm. The
   checkpoint ladder (§9) additionally reads `@2000 / @9000 / @9250 / @10000` from banked caches.
2. ⚠️ **EVERY v6 POINT IS AN EARLY-READ at 11 250 / 30 000 = 37.5 %.** 30 000 remains the primary
   read.
3. ⚠️ **LEAD-ENRICHED, NOT PARITY** — inherited unchanged from the parity run and the positive
   control (`parity: False` in every cache meta). 130 clips, 60 probe-train / **70 eval**, clip-
   disjoint. ⛔ **No episode is selected, added, removed, reordered or re-hashed by this work.**
4. **Estimator: `taniteval.ci.paired_episode_cluster_bootstrap`, n_boot 2000, clustered on the 70
   eval episodes.** ⛔ `overlapping_holdout_se` is never imported.
5. ⛔ **EVERY TABLE NAMES ITS REPAIR ROUTE, AND THE TWO ROUTES ARE NEVER POOLED (§10).**
   **Route A = `--fit-mode unpen`** — the repair taken from the module, `ridge_fit(…, intercept_col=-1)`;
   the 15-arm 165-row re-read. **Route B = `--fit-mode centred`** — the locally re-derived repair;
   the four `ll_rep_*` arms at 3 seeds. **MEASURED here (§10): 44 paired rows, 2 alpha choices
   differ, 0 verdicts differ, and `ego_v0`'s K1 differs by 0.396.**
6. ⭐ **THE READOUT IS pc6's, AND EQUIVALENCE IS PROVED RATHER THAN ASSERTED.** `ll1_ladder.py`
   IMPORTS `ridge_fit` from `pc6_linear_readout` (no second implementation of the solve) and its
   `--gate-pc6` asserts that target `lead_gap` at seed 0 reproduces the banked
   `pc6_ridge_s11250.json`. **MEASURED, and it is exact:**

   | gate check | banked | ours |
   |---|---|---|
   | `ridge_err_m` | 6.7128 | **6.7128** |
   | `c_const_err_m` | 5.1329 | **5.1329** |
   | `K1_delta` | +1.5799 | **+1.5799** |
   | `corr_pred_gt` | +0.1592 | **+0.1592** |
   | `alpha_chosen` | 10.0 | **10.0** |
   | n train / eval windows | 2231 / 2721 | **2231 / 2721** |

   ⭐ **And the re-read carries TWO further reproduction gates, because a corrected producer that
   silently changed its own numbers would make the correction unattachable to the banked rows:**
   (a) the 165-row re-read reproduces the banked incumbent **bit-exactly** on 308 field comparisons
   (`…/2026-08-18-k1-degeneracy-guard/raw/reread/llGATE_*.json`); (b) ⭐ **this correction's guard
   refit reproduces the banked `ll_rep_*` bit-exactly on 4 752 of 4 752 field comparisons**
   (`…/2026-08-18-ladder-corrected/raw/rep_guard_gate.json`, `GATE: PASS`).
7. **GPU: none. This is a CPU ridge.** ⛔ **Thor was never used for compute** — no checkpoint was
   pulled and nothing was run there, in either the original run or this correction.

---

## 1. ⭐ THE POSITIVE CONTROL, RUN FIRST — because a negative without one taught this programme nothing

The D1 withdrawal is the standing lesson: **five negative controls proved the slot probe was not
cheating and not one proved it could measure.** So the anchor's negative is inadmissible until the
readout is shown able to find ego speed when it is there.

`ll2_ego_oracle.py` takes the REAL @11250 cache and replaces **only** `cells`, keeping every window,
target, class, rate, `clip_id`, `episode_uid` and the declared split — so the paired bootstrap's
clusters are literally the same objects. Cell *k* = `P_k @ [v0/10, 1]` with an **independent random
`[128, 2]` projection per cell** (seed 20260817), rescaled to the real cells' global std
(**0.030898** — identical to `pc1`'s banked value) and noised at `noise_rel ×` that std.

⭐ **It is deliberately the DISTRIBUTED control, not the easy one.** The banked `ORC-DIRECT` puts its
answer at a **fixed address** (cell 0), which does not test whether a ridge with 2 049 features and
~2 200 training windows can find a signal **smeared across every dimension** — which is how a
learned latent would carry it. Here no feature is a copy of the label.

**`MEASURED` · route A (`unpen`) · seed 0 · C-CONST err 4.097 m/s · n 3023 / 70 clusters**

| `ego_v0` | MAE (m/s) | **K1 [CI]** | **K1B [CI]** | guard | **r** | r² | r within-ep |
|---|---|---|---|---|---|---|---|
| **EGO-ORACLE 0.10× noise** | **0.033** | **−4.064 [−4.96, −3.26] PASS ✅** | **−4.153 [−4.91, −3.44]** | **OK** | **+1.0000** | 0.9999 | +0.9999 |
| **EGO-ORACLE 1.0× noise** | **0.249** | **−3.848 [−4.74, −3.05] PASS ✅** | **−3.938 [−4.69, −3.23]** | **OK** | **+0.9985** | 0.9970 | +0.9930 |
| **EGO-ORACLE 3.0× noise** | **0.813** | **−3.284 [−4.17, −2.48] PASS ✅** | **−3.375 [−4.13, −2.68]** | **OK** | **+0.9830** | 0.9664 | +0.9272 |
| ⭐ **EGO-ORACLE 10× noise** | **2.493** | ⭐ **−1.604 [−2.41, −0.88] PASS ✅** | **−1.720 [−2.37, −1.12]** | **OK** | **+0.8280** | 0.6857 | +0.5670 |
| **`ORC-DIRECT` (GT oracle memory)** | 3.272 | **−0.825 [−1.42, −0.26] PASS ✅** | −0.898 [−1.39, −0.43] | **OK** | +0.6275 | 0.3937 | +0.3392 |
| ⛔ **`v6F-SW-30k@11250` (the real arm)** | **4.129** | ⛔ **+0.032 [−0.53, +0.51] not separated** | −0.236 [−0.63, +0.08] **ns** | — | **+0.3212** | **0.1031** | +0.2425 |
| RANDOM-LATENT NULL | 4.291 | +0.194 [−0.12, +0.49] not separated | +0.005 [+0.00, +0.01] | — | **−0.0262** | 0.0007 | −0.0111 |
| `C-V0` (the scalar predicting itself) | 0.000 | −4.097 PASS ✅ | −4.186 | **OK** | +1.0000 | 1.0000 | +1.0000 |

⇒ ⛔ **THE ONE-LINE STATEMENT OF THIS RUN, restated on the repaired readout:**

> **On a memory built from a random projection of the ego's own speed with TEN TIMES that signal's
> magnitude added as noise, this ridge recovers ego speed at r = +0.828 and earns a guarded K1 PASS.
> On the trained v6 operative latent it recovers it at r = +0.321 and cannot separate from a
> constant.**

⇒ The pre-registered third outcome — *"nothing recoverable, not even ego speed ⇒ the cache or the
readout is broken, report that and stop"* — is **REFUTED**. The cache is sound (the pose grid binds
exactly, §0/§11), the readout is sound, and three reproduction gates pass. **The anchor's failure is
a property of the latent.**

⚠️ **The repair changed the anchor's WORDING and not its meaning.** Before: the arm *lost* to a
constant (K1 +1.429 FAIL) — but §10 shows a no-signal arm was driven toward predicting **zero**, so
that loss was partly manufactured. After: the arm **ties** a constant. **A tie is the honest form of
the same finding**, and it is the form that cannot be dismissed as an instrument artefact.

⚠️ **A limit stated rather than hidden:** the control establishes the readout can find a *linear*
distributed encoding. It cannot establish that the readout would find an encoding that is present
but **non-linear**. Everything below is a statement about **linear decodability**.

---

## 2. ⛔ THE 87 BANKED FAILs — WHICH WERE EVER FINDINGS

`MEASURED`, and **recomputed for this correction directly from the 15 incumbent `ll_*.json` and the
15 re-read `llR_*.json`, not copied from C100's table** (`code/corrected_tables.py` →
`raw/corrected_tables.json`, table `T1`). **The independent recount reproduces C100 exactly.**

| of the 87 banked separated-FAILs | n | what it means |
|---|---|---|
| **die at the C92 repair** | **23** | the biased floor made them fail; the CI now spans zero |
| ⛔ **killed by the C97 guard** | ⛔ **42** | the verdict was a contest between two constants, not a fact about a latent |
| **flip to PASS** | **11** | see the note below — this set is almost entirely the positive controls |
| **survive both** | **11** | of which **substantive** (`\|K1B\|/gt_sd ≥ 0.02`): ⭐ **1** |

⭐ **THE ONE SUBSTANTIVE SURVIVOR, and it is a NEGATIVE about the latent:**

> **`ll_s09000` `lead_gap` — K1 +1.811 → +1.291, K1B +0.748 [+0.002, +1.624], guard OK,
> `|K1B|/gt_sd` 0.121.** At step 9 000 the latent's lead-gap readout is separably **WORSE than its
> own mean level**: its variation actively hurts.

⚠️ **And it is fragile, which is stated rather than glossed.** Its CI's lower bound is **+0.002** —
it separates by a hair — and **the same row at @11250 does NOT survive**: K1B **+0.404
[−0.214, +1.101], CONSTANT-OFFSET-ONLY**. The previous stream reported the @11250 FAIL as surviving;
that is **DOWNGRADED**. One checkpoint separates and its neighbour does not, so **no trend is claimed
and this survivor should not be built on**.

⛔ **THE OTHER TEN SURVIVORS ARE ARITHMETIC, NOT FINDINGS.** All ten are `ego_yawrate`, at
**K1B +0.0000 with CI [+0.0000, +0.0000]** and `|K1B|/gt_sd` ≤ 0.0020 — and **two of the ten sit on
RANDOM-LATENT NULL arms** (`ll_nullmatched`, `ll_tok11250null`). A `torch.randn` cache produces the
identical verdict, which is the proof that they cannot be findings about any latent.

⭐ **THE 11 FLIPS-TO-PASS ARE THE INSTRUMENT VALIDATING ITSELF, and they are worth reading.**
`MEASURED`: **9 of the 11 belong to the EGO-ORACLE arms** (`n_agents_grid`, `n_agents_all`,
`lead_gap` at noise 1×/3×/10×), all with **guard OK** and K1B monotone in injected noise; **the other
2 are the two NULL arms on `n_agents_all` — and the guard catches BOTH as `DEGENERATE-CONSTANT`.**
⇒ **the repair promotes arms that carry signal and the guard rejects the two that carry none.** That
is the behaviour a working instrument has, and it is why the negatives below are readable.

---

## 3. ⭐ K1B — WHY THE RESULTS TABLES BELOW REPORT IT, AND NOT ONLY K1

**Read this paragraph once and the rest of the document needs no further explanation.** Writing
`c_own = mean(pred)`, K1 decomposes **exactly**:

> **K1 = [MAE(pred) − MAE(c_own)] + [MAE(c_own) − MAE(C-CONST)] = K1B + K1C**

* **K1B** is what the readout's **variation** buys over its own mean level. It is the
  **latent-attributable** part, and it is **algebraically INVARIANT to the choice of C-CONST** —
  pinned by `test_K1B_is_INVARIANT_to_the_choice_of_C_CONST` in `taniteval/tests/test_k1_degeneracy_guard.py`.
* **K1C** is a contest between two constants. On a z-scored design the repaired ridge's intercept is
  `mean(y_train)`, which carries **zero** latent information, so K1C is **never** evidence about a
  latent.

⇒ ⭐ **K1B is the quantity that belongs in a results table and K1 is not**, because K1 mixes a
statement about the latent with a statement about which constant the baseline happens to use. And
because K1B is invariant to C-CONST, the whole *"should C-CONST be the mean or the median?"* question
**dissolves instead of needing a ruling** — the median stays (it is MAE-optimal on train; switching
would weaken the baseline and manufacture PASSes), and K1B does the isolation job regardless.

⭐ **The bound that makes this cheap:** `|K1B| ≤ mean|pred − c_own| ≤ pred_sd` (reverse triangle
inequality, then Jensen). ⇒ **any banked row whose `|K1_delta|` exceeds its own `pred_sd` has a
PROVEN constant-offset component**, computable with no refit and no bootstrap from fields already in
every artifact. **Verdicts:** `OK` · `CONSTANT-OFFSET-ONLY` (K1 separates, K1B does not) ·
`DEGENERATE-CONSTANT` (the same, where the readout is also a flat line) · `NO-VERDICT-TO-GUARD`
(K1 never separated). Module: `taniteval/taniteval/degeneracy.py`.

---

## 4. THE LADDER

### 4.1 Correlation with the truth, every rung with its null and both controls

**`MEASURED` · route A (`unpen`) · seed 0.** `r_wep` = the correlation after **both** prediction and
truth are demeaned by their own eval episode — the **ANTI-EPISODE-IDENTITY** statistic (within-episode
gap SD is only ~4 m, so a readout that merely identifies the episode scores well globally and ~0
here). `r_pv0` = the **partial** correlation with **ego speed partialled out** — the TRIVIAL-PROXY
test.

| rung | target | n win/clust | **v6 r** | **NULL r** | **C-V0 r** | ORACLE r | v6 `r_wep` | v6 `r_pv0` |
|---|---|---|---|---|---|---|---|---|
| EGO ⭐anchor | `ego_v0` | 3023/70 | **+0.321** | −0.026 | +1.000 | +0.628 | +0.243 | — |
| EGO | `ego_accel` | 3023/70 | **+0.127** | −0.011 | −0.112 | −0.044 | +0.090 | **+0.116** |
| EGO | `ego_yawrate` | 3023/70 | **+0.030** | −0.008 | −0.020 | +0.026 | +0.071 | +0.036 |
| EGO | `ego_curv` | 2221/70 | **+0.002** | −0.022 | −0.085 | −0.048 | +0.023 | +0.028 |
| SCENE | `n_agents_grid` | 3023/70 | **+0.142** | −0.017 | +0.322 | +0.876 | +0.042 | +0.094 |
| SCENE | `n_agents_all` | 3023/70 | **+0.390** | +0.014 | +0.335 | +0.681 | +0.086 | +0.323 |
| OBJECT | `lead_present` | 3023/70 | **+0.109** | +0.005 | +0.152 | +0.465 | +0.091 | +0.084 |
| OBJECT | `nearest_any` | 3019/70 | **+0.311** | −0.028 | +0.408 | +0.868 | +0.060 | +0.237 |
| OBJECT | `lead_gap` | 2721/70 | **+0.073** | −0.011 | **+0.684** | +0.993 | +0.259 | ⛔ **−0.107** |
| OBJ-DYN | `lead_closing` | 2721/70 | **−0.036** | +0.005 | +0.017 | +0.432 | +0.056 | −0.034 |
| OBJ-DYN | `lead_inv_ttc` | 2721/70 | **−0.029** | +0.030 | −0.037 | +0.118 | +0.021 | −0.022 |

⭐ **THE NULL IS CLEAN ON EVERY ROW** (|r| ≤ 0.030), which is what makes the small positives readable
at all. ⛔ **`lead_gap`'s `r_pv0` is now NEGATIVE (−0.107)** — the latent's lead-gap correlation is
not merely mostly `v0`, it is **entirely** `v0` plus a residual pointing the wrong way.

### 4.2 ⭐ The same ladder as VARIANCE EXPLAINED — the fairest single number per rung

⚠️ **Why `r²` and not `R²`.** The fit is **over-dispersed**: it emits close to full variance at low
correlation, so its MAE can lose to a constant while `r` is positive and its `R²` is negative.
`r2_ceiling = r²` is the variance a **perfectly rescaled version of the same readout** would explain.
Quoting both stops "K1 fails" from being misread as "r is zero", and stops "r is positive" from being
oversold. ⭐ **`r²` is also the quantity that neither C92 nor C97 can bias** — both act on the fit's
*dispersion*, not on its correlation — which is why the rung profile survives the re-read while the
K1 verdicts do not.

**`MEASURED` · route A (`unpen`) · seed 0. `old` = the incumbent pc6 solve, shown so the movement is
visible rather than silently replaced.**

| rung | target | **v6 r² (repaired)** | *v6 r² (old)* | NULL r² | C-V0 r² | ORACLE r² |
|---|---|---|---|---|---|---|
| SCENE | `n_agents_all` | **0.1519** | *0.0758* | 0.0002 | 0.1124 | 0.4636 |
| EGO ⭐ | `ego_v0` | **0.1031** | *0.0609* | 0.0007 | 1.0000 | 0.3937 |
| OBJECT | `nearest_any` | **0.0964** | *0.0480* | 0.0008 | 0.1667 | 0.7525 |
| SCENE | `n_agents_grid` | **0.0200** | *0.0200* | 0.0003 | 0.1038 | 0.7676 |
| EGO | `ego_accel` | **0.0160** | *0.0350* | 0.0001 | 0.0125 | 0.0020 |
| OBJECT | `lead_present` | **0.0118** | *0.0087* | 0.0000 | 0.0231 | 0.2166 |
| OBJECT | `lead_gap` | **0.0053** | *0.0254* | 0.0001 | **0.4672** | 0.9864 |
| OBJ-DYN | `lead_closing` | **0.0013** | *0.0000* | 0.0000 | 0.0003 | 0.1865 |
| EGO | `ego_yawrate` | **0.0009** | *0.0036* | 0.0001 | 0.0004 | 0.0007 |
| OBJ-DYN | `lead_inv_ttc` | **0.0008** | *0.0001* | **0.0009** | 0.0013 | 0.0139 |
| EGO | `ego_curv` | **0.0000** | *0.0001* | **0.0005** | 0.0071 | 0.0023 |

⇒ ⭐ **THE SHAPE, STATED PLAINLY — and it is the same shape as before the re-read.** There is a
**cliff, not a slope**. Everything the latent carries is **coarse scene scale and longitudinal ego
state**. Everything **rotational** (`ego_yawrate`, `ego_curv`) and everything **relative-motion**
(`lead_closing`, `lead_inv_ttc`) is **at the null — two of the four are BELOW their own nulls.**
The programme's LATERAL family (curvature error, yaw-rate error) and the longitudinal family's
*distance-keeping* half (TTC, closing) are asking the planner to use information **that is not
linearly in the state it is planning from**.

⚠️ **Two rungs moved materially and both are noted rather than buried.** `lead_gap` falls
**0.0254 → 0.0053** (it drops from 5th rung to 7th — the repair takes away most of what looked like
lead content), and `ego_accel` falls **0.0350 → 0.0160**. The three top rungs rise because the
repair lets the fit shrink properly instead of collapsing toward zero.

### 4.3 ⛔ Every rung in error, K1, K1B and the guard — no row without its null AND its trivial proxy

**`MEASURED` · route A (`unpen`) · seed 0 · positive K1/K1B = the arm is worse.** C-CONST is the
**probe-train median**. ⭐ **The last column is the mandatory trivial-proxy margin: what the
2 048-dimension latent buys OVER a single scalar of ego speed, in units of the target's own sd.**
**Positive = the scalar wins.**

| target | unit | C-CONST err | **v6 K1 [CI] / verdict** | **v6 K1B [CI] / guard** | NULL K1B / guard | C-V0 K1B / guard | ORACLE K1B / guard | ⛔ **(v6−C-V0) K1B / gt_sd** |
|---|---|---|---|---|---|---|---|---|
| `ego_v0` | m/s | 4.097 | +0.032 [−0.53, +0.51] **ns** | −0.236 [−0.63, +0.08] ns | +0.005 ns | **−4.186 OK** | −0.898 OK | **+0.708** |
| `ego_accel` | m/s² | 0.5095 | +0.016 [+0.01, +0.02] FAIL | −0.000 ns **DEGEN** | +0.000 DEGEN | −0.001 DEGEN | +0.000 DEGEN | +0.001 |
| `ego_yawrate` | rad/s | 0.0285 | +0.0001 FAIL | **+0.0001 sep OK** | **+0.0000 sep OK** | −0.0000 DEGEN | +0.0000 — | +0.002 |
| `ego_curv` | 1/m | 0.0069 | +0.0003 FAIL | +0.0000 ns **DEGEN** | +0.0000 DEGEN | −0.0000 DEGEN | −0.0000 DEGEN | +0.000 |
| `n_agents_grid` | ct | 9.798 | +0.576 [−0.25, +1.43] **ns** | +0.804 [+0.10, +1.61] sep | +0.000 ns | ⛔ **−0.540 sep OK** | −7.075 OK | **+0.091** |
| `n_agents_all` | ct | 37.936 | ⭐ **−5.003 [−7.74, −2.17] PASS** | ⭐ **−2.785 [−5.06, −0.64] sep OK** | ⛔ −0.003 **DEGEN** | **−2.243 sep OK** | −9.546 OK | ⚠️ **−0.012** |
| `lead_present` | prob | 0.0999 | +0.129 [+0.11, +0.15] FAIL | −0.005 ns **CONST-ONLY** | −0.000 CONST-ONLY | −0.004 CONST-ONLY | −0.038 CONST-ONLY | −0.000 |
| `nearest_any` | m | 5.484 | +0.071 [−0.42, +0.53] **ns** | −0.421 [−0.81, −0.06] sep | +0.000 DEGEN | −0.444 sep | −2.593 OK | **+0.003** |
| `lead_gap` | m | 5.133 | +0.736 [+0.13, +1.44] FAIL | +0.404 [−0.21, +1.10] ns **CONST-ONLY** | +0.000 ns | ⛔ **−1.588 sep OK** | −4.571 OK | **+0.321** |
| `lead_closing` | m/s | 1.112 | +0.094 [+0.06, +0.13] FAIL | +0.000 ns **DEGEN** | −0.000 DEGEN | −0.000 DEGEN | −0.076 sep | +0.000 |
| `lead_inv_ttc` | 1/s | 0.0802 | +0.004 [+0.00, +0.01] FAIL | −0.000 ns **DEGEN** | +0.000 DEGEN | +0.000 DEGEN | −0.004 sep | −0.000 |

⇒ ⛔ **FOUR READINGS, AND THE FOURTH IS THE VERDICT.**

1. **The v6 latent earns exactly ONE guarded K1 PASS in the entire ladder** — `n_agents_all`.
   Every other rung is a tie, a `DEGENERATE-CONSTANT`, or a `CONSTANT-OFFSET-ONLY`.
2. ⛔ **That one PASS is beaten by a single scalar on the 3-seed mean (§5.3).** At seed 0 the latent
   leads the ego-speed scalar by **0.012 gt_sd**; across three seeds the scalar **wins**.
3. ⛔ **On `n_agents_grid` the scalar PASSES (K1B −0.540, guard OK) while the 2 048-dimension latent
   does not separate at all** (K1 +0.576) — and the latent's K1B is **positive** (+0.804), i.e. its
   variation makes it worse than its own mean.
4. ⛔ **THE VERDICT: there is no rung where the v6 latent earns a guarded PASS that a single scalar of
   ego speed does not match or beat** (§5.3 settles the one seed-0 exception). The only unambiguous
   guarded passes in the table belong to the GT oracle and to `C-V0`.

⚠️ **The seven `DEGENERATE-CONSTANT` / `CONSTANT-OFFSET-ONLY` rows are printed rather than filtered.**
Filtering them would be choosing which controls to believe; naming them is what makes the one real
PASS legible.

### 4.4 ⚠️ WHERE THE POSITIVE CONTROL DOES NOT REACH — stated, not glossed

The `ORACLE` column is `ORC-DIRECT`, whose cells encode **positions and extents only**:
`[cx/60, cy/16, sin yaw, cos yaw, l/10, w/5, presence]` (`pc1_oracle_cache.py`). **There are no rate
features in it.** So on `lead_closing` (oracle r +0.432) and `lead_inv_ttc` (oracle r **+0.118**,
r² 0.014) the oracle is a **weak** upper bound — it only reaches those quantities through their
correlation with geometry. ⇒ **For the two OBJECT-DYNAMICS rungs the honest verdict is "not linearly
present in the latent", NOT "an apparatus that could have found it did not".** The ROTATIONAL rungs
are worse served still: `ego_yawrate` and `ego_curv` have **no positive control at all** in this run
(the oracle scores +0.026 / −0.048 on them, i.e. it has no ego content either), so their nulls are
the weakest rows in the ladder. **Building an EGO-ORACLE for yaw-rate is one line of
`ll2_ego_oracle.py` and is the first item in §14.**

---

## 5. ⭐⭐ THE TRIVIAL-PROXY FINDING — the reason the precedent's new fact does not survive

The brief's own warning: *"if a quantity is recoverable, ask whether it is recoverable from something
TRIVIAL — a positive that is really a proxy is worse than a null."* `C-V0` is the identical ridge with
the 2 048-dimension latent replaced by **the single scalar `v0`**.

### 5.1 `lead_gap` — where the margin over the null INVERTS

**`MEASURED` · route A (`unpen`) · seed 0 · n 2721 / 70 clusters · C-CONST err 5.133 m**

| arm | MAE (m) | **K1 [CI]** | **K1B [CI]** | guard | **r** | r² | `r_pv0` |
|---|---|---|---|---|---|---|---|
| `ORC-DIRECT` (GT oracle) | **0.580** [0.562, 0.598] | −4.553 [−5.16, −4.05] **PASS ✅** | −4.571 [−5.15, −4.07] | **OK** | +0.993 | 0.986 | +0.987 |
| ⭐ **`C-V0` — ONE FEATURE, ego speed** | **3.571** [3.151, 4.040] | **−1.562 [−2.02, −1.14] PASS ✅** | **−1.588 [−2.05, −1.15]** | **OK** | **+0.684** | **0.467** | — |
| ⛔ **`v6F-SW-30k@11250` — 2 048 features** | **5.869** [5.083, 6.806] | **+0.736 [+0.13, +1.44] sep FAIL** | +0.404 [−0.21, +1.10] **ns** | **CONST-ONLY** | +0.073 | 0.005 | ⛔ **−0.107** |
| RANDOM-LATENT NULL | **5.175** [4.667, 5.764] | +0.043 [−0.09, +0.18] **ns** | +0.000 ns | — | −0.011 | 0.000 | −0.005 |

⇒ ⛔ **Three things follow, and the third is the one that matters.**

1. ⛔ **The precedent's ~1.8 m gap over the null does not survive — it INVERTS.** On the incumbent
   solve the latent was 1.821 m better than noise; on the repaired solve it is **0.694 m WORSE**
   (5.869 vs 5.175). The mechanism is C97's: the penalised intercept **forced the noise arm to load
   features** to reach y's 15 m level, giving it a large error; repaired, the noise arm collapses to
   a constant (`pred_sd` **0.014** against `gt_sd` 6.200) and simply predicts the mean.
2. ⭐ **A single scalar of ego speed beats the entire latent by 2.30 m and passes K1 with a clean
   guard where the latent fails.** Drivers hold headway roughly proportional to speed; the lead gap
   is therefore **46.7 % predictable from `v0` alone** with no perception whatsoever.
3. ⛔ **The latent's residual lead-gap signal points the WRONG WAY.** Partialling `v0` out takes
   r **+0.073 → −0.107**. ⇒ **"The latent carries some linearly-decodable lead information" is not
   supportable as a claim about AGENTS** — there is no positive residual left to attribute.

⭐ **The same pattern on presence, by AUC** (the natural metric for a binary rung, reported because
MAE on a 90 %-positive label is uninformative): `lead_present` AUC, repaired — **ORACLE 0.823 ·
C-V0 0.673 · v6 0.582 · NULL 0.498.** **Ego speed alone detects the presence of a lead better than
the world model's own state does.**

### 5.2 ⛔ SCENE DENSITY — the claim that the repair appeared to create, and the control that kills it

Under the repair `n_agents_all` PASSes K1 on **all 15 arms**. Read with both controls
(`…/2026-08-18-k1-degeneracy-guard/raw/reread/`, route A, seed 0):

| arm | K1B | guard | reading |
|---|---|---|---|
| `GT-ORACLE-DIRECT` | **−9.546** | OK | the ceiling |
| `v6F@11250` | −2.785 | OK | |
| `v6F@10000 / @9250 / @9000` | −2.805 / −2.759 / −2.715 | OK | flat across checkpoints |
| `v6F@2000` | −1.567 | ⚠️ CONSTANT-OFFSET-ONLY | the one place training moved the latent (§9) |
| ⛔ **`C-V0` (ego speed, ONE scalar)** | ⛔ **−2.243** | OK | **80 % of the arm's value at seed 0** |
| `EGO-ORACLE` n0.1 → n10 | −2.219 → −1.044 | OK | ⭐ monotone in injected noise — a dose-response curve |
| `RANDOM-LATENT-NULL` | **−0.003** | ⛔ **DEGENERATE** | caught |
| `TOKENS MATCHED-RANDOM NULL` | **+0.000** | ⛔ **DEGENERATE** | caught |

⇒ **At seed 0 the latent beats the single ego-speed scalar by 0.541 agents on a target of sd 46.46 —
0.012 gt_sd.** And on `n_agents_grid` **the scalar PASSES while the latent does not separate.**
⛔ **"The v6 latent reads scene density" is NOT SUPPORTED**; the mechanism is faster ego ⇒ open road
⇒ fewer agents.

### 5.3 ⭐⭐ NEW IN THIS CORRECTION — the 0.012 gt_sd margin does not survive SEEDING, and the scalar wins outright

The 165-row re-read ran **seed 0 only** (justified: `fit_one` draws its own RNG per call, so seed 0's
row is bit-identical either way). The four `ll_rep_*` arms refitted here carry **three seeds**, and
that is what exposes the following.

**`MEASURED` · route B (`centred`) · seeds 0/1/2 · `raw/seed_stability_v6_vs_cv0.json`.**
K1B is negative-is-better; **margin = v6 − C-V0, so POSITIVE MEANS THE SCALAR WINS.**

| rung | v6 K1B, seeds 0/1/2 | v6 mean | C-V0 K1B, seeds 0/1/2 | C-V0 mean | margin @seed 0 | ⭐ **margin @3-seed mean** |
|---|---|---|---|---|---|---|
| `ego_v0` | −0.030 / −0.020 / +0.374 | +0.108 | −4.186 ×3 | −4.186 | +4.156 | **+4.294** |
| ⛔ `n_agents_all` | **−2.785 / −0.269 / −1.376** | **−1.477** | −2.243 / −0.579 / −2.243 | **−1.689** | ⚠️ **−0.541** | ⛔ **+0.212 — THE SIGN FLIPS** |
| `n_agents_grid` | +0.804 / −0.063 / −0.165 | +0.192 | −0.540 ×3 | −0.540 | +1.344 | **+0.732** |
| `nearest_any` | −0.421 / −0.042 / −0.421 | −0.294 | −0.444 ×3 | −0.444 | +0.024 | **+0.150** |
| `lead_gap` | +0.404 / −0.018 / +0.072 | +0.153 | −1.588 ×3 | −1.588 | +1.992 | **+1.741** |
| the other 6 rungs | all per-seed `\|K1B\| ≤ 0.021` | — | all per-seed `\|K1B\| ≤ 0.005` | — | ≥ 0 on 5 of 6 | ≥ 0 on 5 of 6 |

⇒ ⛔⛔ **THE STATEMENT THIS DOCUMENT ENDS ON.** On the **mean of three inner-split seeds, 10 of the 11
rungs go to the SINGLE SCALAR OF EGO SPEED or tie it** — the 2 048-dimension v6 operative latent does
not linearly out-read one number. The single rung where seed 0 said otherwise — `n_agents_all`, the
one guarded PASS in §4.3 — **reverses**: the latent's mean K1B is **−1.477** against the scalar's
**−1.689**.

⚠️ **Stated exactly, because "not one rung" would be an overclaim by 6×10⁻⁶.** The eleventh rung,
`lead_inv_ttc`, favours the latent by **margin −0.000006 K1B = −0.00002 gt_sd** — on a rung where
**both** arms are `DEGENERATE-CONSTANT` and both K1B are ~1e-5. It is numerically zero, and it is
printed rather than rounded away.

⚠️ **The mechanism, stated so nobody re-derives it.** `C-V0` has ONE feature, so its inner-split alpha
search is near-degenerate and lands on the same alpha for every seed (`seed_K1_range` **0.0000** on 4
of these 5 rungs). The v6 arm has 2 049 features and a wide repaired alpha grid, so its alpha choice
moves with the seed — `n_agents_all` picks 1e5 / 1e7 / 1e6 across seeds 0/1/2, and its K1B moves
**2.516**. ⇒ **The margin is smaller than the arm's own seed noise, in both directions.** This is not
a reason to prefer one seed; it is a reason to report the mean and to stop treating a 0.012 gt_sd
margin as evidence at all.

---

## 6. EPISODE IDENTITY — the leakage the brief warned a rich latent is MORE prone to

`C-EPMEAN` (leave-one-out mean of the window's own eval episode) is carried on every fit as **K5**,
and `r_wep` is reported on every row. **v6 fails K5 on every rung** — it never beats the
episode-identity oracle. The sharper statistic is what survives episode demeaning
(**`MEASURED` · route A · seed 0**):

| target | v6 r | **v6 `r_wep`** | reading |
|---|---|---|---|
| `n_agents_all` | +0.390 | **+0.086** | ⛔ **~⅘ episode identity** |
| `nearest_any` | +0.311 | **+0.060** | ⛔ **mostly episode identity** |
| `n_agents_grid` | +0.142 | **+0.042** | ⛔ mostly episode identity |
| `ego_v0` | +0.321 | **+0.243** | ✅ genuine within-episode content |
| `ego_accel` | +0.127 | **+0.090** | ✅ genuine within-episode content |
| `lead_gap` | +0.073 | **+0.259** | ⚠️ not episode identity — but §5.1: `r_pv0` is **−0.107** |

⇒ ⭐ **The SCENE-density rungs are largely the readout recognising WHICH CLIP it is in**, which is
exactly the failure mode the brief flagged — and it is the **same rung** that carries §4.3's one
guarded PASS and §5.3's sign flip. **Three independent controls (episode demeaning, the trivial
proxy, and seeding) each remove most or all of `n_agents_all`'s apparent content.** **The EGO rungs
are not** episode identity — their (small) signal is real within-episode variation. **So the honest
ranking of what the latent actually encodes is narrower than §4.2's raw `r²` column suggests:
longitudinal ego state, weakly, and little else.**

---

## 7. WHAT THE LATENT ACTUALLY CARRIES — the one-table summary

**`MEASURED` · route A seed 0, except the last column which is route B 3-seed mean (§5.3).**

| quantity | linearly present? | evidence |
|---|---|---|
| **ego speed** | ⚠️ **weakly — r² 0.103, ties a constant** | genuine within-episode (`r_wep` +0.243), but **12× less readable than a 10×-noise oracle** (r² 0.686) |
| **ego acceleration** | ⚠️ **very weakly — r² 0.016** | K1B `DEGENERATE`, and the scalar matches it |
| **aggregate scene density** | ⛔ **not attributable to the latent** | one guarded PASS at seed 0, **but** ⅘ episode identity, ~80 % `v0`, and the scalar wins on 3 seeds |
| **nearest-object distance** | ⛔ **not attributable** | `r_wep` +0.060; the scalar's K1B is better on both seed 0 and the mean |
| **lead gap** | ⛔ **NO — worse than noise** | 0.694 m worse than the random null; `r_pv0` **−0.107** |
| **yaw rate / curvature** | ⛔ **NO** | r² 0.0009 / 0.0000, at or **below** their nulls ⚠️ *no positive control (§4.4)* |
| **closing speed / inverse TTC** | ⛔ **NO** | r² 0.0013 / 0.0008; `lead_inv_ttc` is **below** its null ⚠️ *weak positive control (§4.4)* |

---

## 8. SEED SPREAD — and the incumbent's "zero spread" was an artefact of the defect

The brief requires ≥3 seeds and a between-condition vs between-seed comparison, because the parity
run measured **3.096 m of K1 spread across three seeds** on one frozen cache. ⚠️ **A ridge is a
CLOSED-FORM SOLVE: there is no optimiser seed.** The only seed-dependent step is the episode-disjoint
inner split that chooses `alpha`.

⛔ **THE INCUMBENT REPORTED "8 of 11 rungs have EXACTLY ZERO seed spread". THAT IS FALSE UNDER THE
REPAIR, AND THE REASON IS THE DEFECT ITSELF.** pc6's penalised intercept made large alphas unusable,
so the alpha sweep was silently truncated and every seed landed on the same small alpha. Repaired,
with the grid extended to 1e7, the choice is genuinely contested.

**`MEASURED` · route B (`centred`) · v6@11250 · `seed_K1_range` over seeds 0/1/2:**

| target | K1 seed range | K1B seed range | alphas chosen (seed 0/1/2) |
|---|---|---|---|
| `n_agents_all` | ⛔ **2.812** | ⛔ **2.516** | 1e5 / 1e7 / 1e6 |
| `n_agents_grid` | **1.048** | 0.968 | 10 / 1e7 / 1e4 |
| `ego_v0` | **0.823** | 0.403 | 1e4 / 1e7 / 1e3 |
| `lead_gap` | **0.711** | 0.422 | 1e4 / 1e7 / 1e5 |
| `nearest_any` | 0.406 | 0.379 | 1e5 / 1e7 / 1e5 |
| the other 6 rungs | ≤ 0.030 | ≤ 0.021 | — |

⇒ ⛔ **On the five rungs that carry any signal, between-seed spread is the SAME ORDER as, or larger
than, every effect this document reports.** The uncertainty that counts remains the **episode-cluster
bootstrap CI**, quoted on every headline number — but the seed spread is the reason §5.3's 3-seed
mean, not seed 0, is the quotable form of the trivial-proxy comparison. ⚠️ **The guard verdict itself
flips across seeds on two rungs** (`n_agents_grid` NO-VERDICT / OK / NO-VERDICT; `nearest_any`
NO-VERDICT / DEGENERATE / NO-VERDICT), which is a further reason no single-seed guard verdict on a
contested rung should be quoted alone.

---

## 9. THE CHECKPOINT TRAJECTORY — 9 250 steps of training changed nothing measurable

Five banked caches, identical windows, split, seed and estimator. **`MEASURED` · route A (`unpen`) ·
seed 0 · `r` per rung.**

| target | @2000 | @9000 | @9250 | @10000 | @11250 | range |
|---|---|---|---|---|---|---|
| `ego_v0` | ⭐ **+0.3618** | +0.3131 | +0.3238 | +0.3276 | +0.3212 | 0.049 |
| `ego_accel` | +0.1225 | +0.1101 | +0.1280 | **+0.1295** | +0.1267 | 0.019 |
| `ego_yawrate` | +0.0041 | +0.0236 | +0.0249 | +0.0261 | **+0.0299** | 0.026 |
| `ego_curv` | −0.0051 | +0.0014 | +0.0016 | **+0.0027** | +0.0022 | 0.008 |
| ⚠️ `n_agents_grid` | +0.2555 | +0.1475 | +0.2934 | **+0.2950** | +0.1415 | ⚠️ **0.154** |
| `n_agents_all` | +0.3490 | +0.3873 | +0.3852 | +0.3878 | **+0.3898** | 0.041 |
| `lead_present` | ⭐ **+0.1137** | +0.1124 | +0.1069 | +0.1053 | +0.1088 | 0.008 |
| `nearest_any` | ⭐ **+0.3143** | +0.3029 | +0.3045 | +0.3038 | +0.3105 | 0.011 |
| `lead_gap` | ⭐ **+0.1232** | +0.0841 | +0.0675 | +0.0643 | +0.0726 | 0.059 |
| `lead_closing` | −0.0161 | −0.0367 | −0.0314 | −0.0286 | −0.0362 | 0.021 |
| `lead_inv_ttc` | −0.0214 | −0.0295 | −0.0297 | −0.0277 | −0.0285 | 0.008 |

⇒ ⭐ **The linear content of the latent is essentially SET BY STEP 2 000 and does not grow — and the
repair makes this CLEANER, not weaker.** ⭐ **Four rungs are at their best at @2000** (`ego_v0`,
`lead_present`, `nearest_any`, `lead_gap`), and `ego_v0` — the anchor — is **best at 2 000 and lower
at every later checkpoint**. Ten of the eleven rungs move by **less than 0.06 in `r` across 9 250
steps**. ⚠️ **The eleventh, `n_agents_grid`, moves 0.154 and is NON-MONOTONE** (0.256 → 0.148 →
0.293 → 0.295 → 0.142), which is the shape of noise rather than of learning; §8 measures a K1 seed
range of **1.048** on that same rung. ⚠️ **No trend is claimed as significant** — these are point `r`
values without per-checkpoint CIs. **What IS claimed is the absence of a large effect**, which is the
relevant fact for whether waiting for 30 000 will change the verdict. ⚠️ It might; this is an early
read.

⚠️ **One rung's K1B does move meaningfully back at @2000, and it is the honest counterweight:**
`n_agents_all` K1B is **−1.567 (`CONSTANT-OFFSET-ONLY`) at @2000** and **−2.715 / −2.759 / −2.805 /
−2.785 (all `OK`) at @9000 / @9250 / @10000 / @11250** — so the scene-density readout *does*
consolidate between 2 000 and 9 000 and is then flat. **That is the one place training visibly
changed the latent's linear content**, and §5.2/§5.3 then show that what it consolidated is
reproducible from `v0`.

---

## 10. ⛔ THE INSTRUMENT DEFECT, ITS REPAIR, AND THE TWO ROUTES THAT MUST NEVER BE POOLED

### 10.1 The defect (C92)

`pc6_linear_readout.ridge_fit` built

```python
A = X.T @ X + alpha * np.eye(d)          # d INCLUDES the appended ones-column
```

and `prep()` appends a bias column to `X`. **The intercept was therefore shrunk like any other
coefficient**, so `pred → 0` as `alpha → ∞`, not `pred → mean(y)`.

**MEASURED**, a synthetic target of mean **7.0** with essentially no signal, MAE by alpha:

| alpha | 1e-2 | 1 | 1e2 | 1e4 | 1e6 | 1e8 |
|---|---|---|---|---|---|---|
| **pc6 `ridge_fit`** | 0.0001 | 0.0140 | 1.1455 | 6.6312 | 6.9962 | **7.0001** |
| **intercept unpenalised** | 0.0001 | 0.0001 | 0.0014 | 0.0073 | 0.0077 | **0.0077** |

⇒ **Two consequences, both material.** (a) **The alpha sweep was silently truncated** — which is also
why §8's "zero seed spread" was an artefact. (b) ⛔ **The readout could not express the null
hypothesis it is scored against**, so `K1 fails` conflated *"no signal"* with *"the instrument cannot
fall back to a constant"*. **That is what made 23 of the 87 banked FAILs disappear.**

⭐ **Status: FIXED PROGRAM-WIDE.** `ridge_fit` now takes `intercept_col`; the module's default stays
**penalised** and the repair is **opt-in**, deliberately, so no banked artifact silently changes
meaning (the C92 precedent). **A `pc6_ridge_*.json` or `ll_*.json` without a `k1_guard` block is a
pre-2026-08-18 artifact on a biased floor — the guard block's presence is the version marker.**

### 10.2 The mirror-image defect (C97) and the guard

A fully-shrunk *repaired* fit **is** "predict the train mean", while `C-CONST` is the train **median**
— so on a skewed target K1 degenerates into mean-vs-median and a pure `torch.randn` null "PASSES".
**C92 made no-signal arms FAIL by construction; its own repair made them PASS by construction.**
The guard (§3) is the fix, and it is what removes **42 of the 87**.

### 10.3 ⛔ THE TWO REPAIR ROUTES ARE NOT INTERCHANGEABLE — MEASURED on these exact rows

`centred` (centre `y`, solve unpenalised on the feature block) and `unpen`
(`ridge_fit(…, intercept_col=-1)`) are algebraically identical **on the full fit** — the z-scored
design makes the normal equations block-diagonal. **They are not identical on the INNER SPLIT**,
whose feature mean is not zero, and that is where alpha is chosen.

**`MEASURED` (`raw/corrected_tables.json` → `T5_two_routes`): the 4 arms × 11 rungs that exist on
BOTH routes, seed 0.**

| | |
|---|---|
| paired rows | **44** |
| alpha choices that differ | **2** |
| verdicts that differ | **0** |
| guard verdicts that differ | **0** |
| ⛔ **max abs K1 gap** | ⛔ **0.3957** — `ego_v0`@11250: route A **+0.0317**, route B **+0.4274** |
| …and its K1B | **−0.236 (A) vs −0.030 (B)** — a factor of **8** |

⚠️ **This is the corrected form of a claim this document previously made and got wrong.** The earlier
text said the two routes agree to *"~1e-12"* on the inner split. **The full fit agrees to 5e-14; the
inner split differs by up to 0.74 MAE — eleven orders out.** ⇒ **The two routes' numbers must never
be pooled.** Every table in this document names its route in the caption; **§1, §4, §5.1, §5.2, §6,
§9 and §11 are route A; §5.3 and §8 are route B.**

---

## 11. WHERE IN THE STACK IS IT LOST? — the localisation INVERTS under the repair

The chain is `pixels → encoder tokens [640, 768] → readout → z_op [2048] → cells [16, 128]`.

⭐ **`cells` is NOT a lossy view.** `V6Stack.cells` (`stack/tanitad/models/v6.py:3710`, re-verified
from source 2026-08-18) is `z_op.reshape(*z_op.shape[:-1], n_cells, d_readout)` — **a pure reshape**.
Flattening the 16×128 cells reproduces `z_op` exactly, so **§4's ladder probes the full operative
state**, and the only learned, potentially-lossy module between the encoder's tokens and it is
`self.readout`.

`cache_tok11250` banks the encoder's patch tokens on a **subset** of the grid. Same ridge, same split,
same estimator; features are the tokens **mean-pooled over the 16×40 grid → 768**, with a
**matched-random null** built from the same features' per-dimension mean/sd.

**`MEASURED` · route A (`unpen`) · seed 0 · MATCHED windows n 1507 / 70**

| `ego_v0` | MAE (m/s) | K1 | K1B | **r** | r² | `r_wep` |
|---|---|---|---|---|---|---|
| operative **CELLS**, same windows | **4.065** | −0.030 **ns** | −0.243 | ⭐ **+0.320** | **0.1026** | +0.232 |
| encoder **TOKENS**, mean-pooled | 4.272 | +0.177 **ns** | −0.007 | **+0.263** | 0.0693 | +0.107 |
| TOKENS **matched-random NULL** | 4.325 | +0.230 **ns** | +0.043 | **−0.038** | 0.0014 | +0.011 |

⇒ ⛔ **THE DIRECTION REVERSES.** On the incumbent solve the mean-pooled tokens read `ego_v0` at
+0.285 against the cells' +0.241, and this document concluded *"the pooled tokens carry ego speed …
only marginally more than the operative state"*. **Repaired, the CELLS beat the TOKENS: +0.320 vs
+0.263.** ⇒ **The learned `readout` is NOT where the ego-speed information is lost — it ADDS
readable structure over a mean-pool of its own input.**

⭐ **AND THAT CORRECTED DIRECTION IS INDEPENDENTLY CONFIRMED.**
`…/incoming/2026-08-18-pooling-ladder-ER10/POOLING_LADDER_ER10.md` §2.2 (`INHERITED`, not re-verified
by me) measures the same contrast on a different harness across 11 rungs: **the learned
`Linear(768→128)` beats a random projection of the same pooled tokens on 9 of 11 rungs**, by up to
1.6×, and the paired Δ on `n_agents_all` is the only contrast in that experiment separated on all
five seeds. ⇒ *"the 49 280-parameter geometry firewall is doing real work."*

⚠️ **This is still a WEAK localisation and must not be over-read in either direction.** Mean-pooling
is **lossy**, so the tokens' +0.263 is a **LOWER bound** on what they hold: this run **cannot**
establish that ego motion is absent from the encoder. What it now does say — with the repair and with
ER10 agreeing — is that **the readout is not the culprit**, which redirects the search upstream to
the encoder and the corpus.

---

## 12. ⛔ THE POOLING THESIS — what this ladder does and does not license, after the re-read

`…/Research/2026-08-18-pooling-bottleneck-R1R2/POOLING_BOTTLENECK_R1R2.md` §1.5 cites **this
document's §2.2** as *"the independent confirmation"* of the 40:1 `AvgPool2d((4,10))` bottleneck,
quoting `n_agents_all` r² **0.076**, `ego_curv` r² **0.0001**, `lead_closing` r² **0.0000**. Because
that citation may cost a full training run through the R1/R2 decision, it is answered here precisely.

**1. ⭐ The rung PROFILE survives the re-read.** `MEASURED` (§4.2): aggregate scene scale and
longitudinal ego state remain the top three rungs; the four rotational / relative-motion rungs remain
at the null, and **two of them are now BELOW their own nulls**. The profile is a statement about `r²`,
and **`r²` is exactly the quantity C92 and C97 cannot bias** — both act on the fit's dispersion. That
is why the profile is the part that held.

**2. ⚠️ But every NUMBER in the citation moves, so §1.5 must be re-quoted, not just re-read.**

| §1.5 quotes | incumbent | **repaired (route A)** | its own null |
|---|---|---|---|
| `n_agents_all` r² | 0.076 | **0.1519** | 0.0002 |
| `ego_curv` r² | 0.0001 | **0.0000** | 0.0005 |
| `lead_closing` r² | 0.0000 | **0.0013** | 0.0000 |

⛔ **"Relative motion is exactly 0.0000" is no longer literally true** (`lead_closing` 0.0013), though
it remains at the null. And ⛔ **the `n_agents_all` half of the citation — "aggregate scene density
survives" — is the half the trivial-proxy control destroys** (§5.2, §5.3): it is ⅘ episode identity,
~80 % `v0` at seed 0, and the ego-speed scalar beats it on the 3-seed mean.

**3. ⛔ THE K1-FAIL SIDE OF THE CONFIRMATION IS GONE ENTIRELY.** All four rungs the pooling
hypothesis was built on carried separated-FAIL verdicts in the incumbent table. After the re-read:
`ego_curv`, `lead_closing` and `lead_inv_ttc` are **`DEGENERATE-CONSTANT`** — constant contests, not
facts about a latent — and `ego_yawrate` survives only at **K1B +0.0001**, a verdict the
**random-latent null earns identically**. ⇒ **Not one of the four rungs contributes a surviving
verdict.**

**4. ⭐⭐ AND THE DECISIVE POINT: THIS LADDER NEVER LOCALISED THE LOSS, AND THE EXPERIMENT THAT DID
HAS DROPPED R1.** A profile showing *"these quantities are not linearly present"* is consistent with
a pooling bottleneck, with an encoder that never encoded them, and with a corpus that never contained
them. It cannot distinguish those. **E-R1-0 ran the discriminating experiment**
(`…/incoming/2026-08-18-pooling-ladder-ER10/`, `INHERITED` — I did not re-run it): the same readout at
**40:1 / 10:1 / 4:1 / 1:1** pooling moves those four rungs by **|Δr²| ≤ 0.0002 with the paired CI
containing zero on all five seeds**, while its `PC-2OBJ` positive control — two tokens inside one
deployed cell with opposite sign — shows a clean **0 → 1 step at exactly the deployed ratio**.
⇒ ⛔ **By its own pre-registration that is the `R1 IS DROPPED` branch: the information is not in the
tokens either.**

⇒ ⭐ **THE ANSWER TO THE QUESTION AS ASKED.** The rungs that carried the confirmation are **not**
among the 65 that died — the *profile* is intact. What died is the claim that the profile was
**evidence for the pool specifically**, and that claim is now refuted from two directions at once:
this document's §11 (the learned readout **adds**, so the loss is upstream of it) and E-R1-0
(removing the pool entirely changes nothing). ⇒ **The pooling thesis does not rest on "the direct
derivation alone" — it rests on a direct derivation that E-R1-0 has since MEASURED to be
quantitatively too strong** (a localised plant survives the deployed 40:1 mean at r² = 1.0000; SNR
falls as `√(k/K)` ≈ 3×, not 40×). **R1/R2 should be decided on E-R1-0, and this ladder should be
cited for the profile only — never as independent evidence about the pool.**

---

## 13. THE FOUR METRIC FAMILIES

Per the binding rule, every family is addressed, with the reason and the `n` where it does not apply.
⛔ **ADE is not reported and is not applicable:** this is a frozen-latent state readout, not a
trajectory eval. **All numbers route A, seed 0, unless §5.3 is cited.**

| family | what this run reports | verdict |
|---|---|---|
| **LONGITUDINAL** | target-speed: `ego_v0` (r **+0.321**, r² 0.103, **K1 ties a constant**) and `ego_accel` (+0.127, 0.016). Distance-keeping: `lead_gap` (r +0.073, **`r_pv0` −0.107**, and **0.694 m WORSE than the random null**), `lead_closing` (**−0.036, AT THE NULL**), inverse-TTC (**−0.029, BELOW its null**). | ⛔ The family's *own control variable* is at 10 % of variance and cannot beat a constant; its **time-gap / TTC half is at or below zero**, and its distance-keeping half is **worse than noise**. |
| **LATERAL** | heading-rate: `ego_yawrate` (r **+0.030**, r² 0.0009). Curvature: `ego_curv` (**+0.002**, 0.0000, **below its null of 0.0005**), n 2221 (windows below the 2.0 m/s floor are dropped and counted). Cross-track is **not applicable** — there is no predicted trajectory in a frozen-latent readout. | ⛔ **Nothing. Both rotational rungs are indistinguishable from the null** — and both lack a positive control (§4.4), so this is the weakest-evidenced row. |
| **TACTICAL** | ⚠️ **NOT MEASURED, n = 0.** This instrument has no manoeuvre decode and no selected-vs-executed comparison; the cache banks no manoeuvre label. `lead_present` (AUC **0.582** vs C-V0's **0.673**) is a *precursor*, not a manoeuvre-decision metric. | ⚠️ Absent by instrument scope, not by choice. **A work item, not an excuse** — a multinomial ridge on the same features would make this family T0-reportable for the first time; §14.6. |
| **STRATEGIC** | ⚠️ **NOT MEASURED, n = 0.** No route or goal label exists in this 130-clip lead-enriched pool, and per `CLAUDE.md` PhysicalAI-AV carries no map, lane graph or route signal at all. | ⚠️ Not computable on this corpus with this cache. |

---

## 14. ⭐ WHAT I WOULD RUN NEXT, in cost order (all CPU or ≤1 GPU-hour, no trunk compute)

1. ⭐ **An EGO-ORACLE for YAW-RATE and CURVATURE** — one line of `ll2_ego_oracle.py` (swap the
   projected scalar). §4.4 is explicit that the two ROTATIONAL rungs have **no positive control**,
   which makes them the weakest rows despite being the LATERAL family's whole basis. **Until this
   runs, "the latent has no yaw content" is an unverified negative.**
2. ⭐ **RE-RUN THE 165-ROW LADDER AT ≥3 SEEDS.** §5.3 and §8 show the seed spread on the five
   signal-bearing rungs is the same order as every effect reported. The re-read ran one seed; the
   correction found the sign flip only because four arms had three. **This is the cheapest change
   that would raise the whole document's evidence grade**, and it is one flag (`LLR_SEEDS="0 1 2"`).
3. **Re-run the ladder at 30 000** when Thor finishes. Every v6 number here is a 37.5 % early read.
4. **An unpooled token probe** (§11) — spatially-structured or patch-subset features rather than a
   mean-pool, so the encoder-vs-`readout` localisation becomes decisive rather than suggestive.
   ⚠️ **Now lower priority than it was**: E-R1-0 has already answered the pooling half (§12).
5. **Report the `C-V0` column on every future latent readout in this programme, at ≥3 seeds.** §5
   shows a single privileged scalar matching or beating a 2 048-dimension latent on **every rung**.
   **Any latent-readout result without it is not interpretable.**
6. **A TACTICAL rung** — multinomial ridge on the executed manoeuvre class from the same features,
   with the same controls. It converts §13's `n = 0` row into a measurement.
7. **A non-linear readout (small MLP) with the same caches, splits, controls and positive control.**
   Everything here is a statement about **linear** decodability only.

---

## 15. THOR — untouched by both runs

⛔ **No compute of any kind was run on Thor by the original run or by this correction**; no checkpoint
was pulled and no snapshot was made. Everything ran on the dev box's CPU from banked local caches.

⚠️ **The health block below is the ORIGINAL RUN's measurement, stamped 2026-08-17 ~23:5x and NOT
re-measured today** (Thor is off-limits: the S-W run holds PID 25477). It is retained as the record
of that run, not as a current status.

| | (2026-08-17, historical) |
|---|---|
| trainer PID | **25477 — ALIVE**, uptime **1-23:05:03** (169 516 s) |
| process state / CPU | `Ssl`, **302 jiffies of CPU in a 6 s sample** — working, not hung |
| run | `v6F-SW-30k`, `train_v6_staged.py --stage S-W --steps 30000` |
| **live `train_log.jsonl`** | **step 12 650**, written **121 s** before the probe |
| `metrics.json` / `ckpt.pt` | step 12 500, both **4 077 s** old |

⚠️ **AND A FALSE STALL WAS NEARLY FILED — the near-miss is worth keeping.** Reading `metrics.json`
alone showed **step 12 500 unchanged across 67 minutes**, which looks exactly like a hung trainer. It
was not: **`metrics.json` and `ckpt.pt` are written on the CHECKPOINT cadence (`--save-every 250`),
not the log cadence.** ⇒ **ROOT-CAUSE CLASS: reading one file's write cadence as the training
cadence** — the same family as judging pod disk with `df` and reading `step_s` as per-step.

## 16. SUITES

**`MEASURED` by this correction** (`…/2026-08-18-ladder-corrected/raw/suite_*.txt`). ⭐ **I modified
nothing under `stack/` or `taniteval/`** — every artifact of this correction is a new file under
`…/incoming/2026-08-18-ladder-corrected/`, plus this document.

| suite | result | briefed baseline | verdict |
|---|---|---|---|
| `taniteval` | see `raw/suite_taniteval.txt` | 1136 / 0 | ✅ **GREEN, matches** |
| `stack` | see `raw/suite_stack.txt` | 3842 / 0 / 7 / 2 | ✅ **GREEN, matches** |

✅ **The `stack` RED reported by the original run on 2026-08-17 (`test_v6_staged.py:1157`,
`sel_gap` stage-gate semantics) is RESOLVED** — that agent's paired test update has landed. **The
escalation is closed**; it is retained in the escalation list below marked CLOSED rather than
deleted, because a resolved escalation that vanishes teaches nobody that it was resolved.

---

## 17. DELIVERABLE MANIFEST

**All paths relative to the repo root** `G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\`.

### 17.1 The original run — `…/incoming/2026-08-17-latent-linear-ladder/`

| artifact | path | what it is |
|---|---|---|
| **this report** | `LATENT_LINEAR_LADDER.md` | the findings, **corrected in place 2026-08-18** |
| rendered tables | `raw/RENDER_TABLES.md` | ⚠️ **INCUMBENT SOLVE — superseded by `…/2026-08-18-ladder-corrected/raw/corrected_tables.json`** |
| assembled summary | `raw/SUMMARY.json` | all 19 arm blobs in one object ⚠️ **incumbent solve** |
| **the ladder** | `code/ll1_ladder.py` | pc6's ridge (imported `ridge_fit`) over a graded target ladder; pose-grid binding proof, pc6 equivalence gate, `--fit-mode centred|unpen`, `--features tokens_mean`, `--randomise-features` null, `k1_guard` on every rung |
| **anchor positive control** | `code/ll2_ego_oracle.py` | builds the EGO-ORACLE caches (distributed random projection of `v0`, 4 noise levels) |
| renderer | `code/ll3_summarise.py` | reads `raw/ll_*.json` → tables; computes nothing |
| chain | `code/chain_ladder.sh` | one invocation per arm: `ctrl / main / ckpt / egoorc / tokens / repair` |
| per-arm results (19) | `raw/ll_*.json` | ⚠️ **incumbent (15) + route-B repaired (4)**; the 15 incumbent are superseded by the route-A re-read |
| per-arm logs (19) | `raw/log_*.txt` | stdout of every run |
| suites (original run) | `raw/suite_*.txt` | including the 2026-08-17 `stack` RED, now resolved (§16) |

### 17.2 ⭐ This correction — `…/incoming/2026-08-18-ladder-corrected/`

| artifact | path | what it is |
|---|---|---|
| **the guard refit chain** | `code/chain_rep_guard.sh` | the 4 `ll_rep_*` arms refitted with the guard on **route B**, with repo→scratch sync + a real-import verification |
| **the reproduction gate** | `code/verify_rep_guard.py` | field-by-field vs the banked `ll_rep_*` |
| **the table builder** | `code/corrected_tables.py` | opens banked JSON and arranges it; computes nothing about the model |
| ⭐ **the guarded rep arms (4)** | `raw/rep_guard/llrepG_*.json` | **the last unre-read rows, now read** — 3 seeds × 11 rungs × 4 arms |
| run logs (4) | `raw/rep_guard/logrepG_*.txt` | stdout |
| ⭐ **the gate result** | `raw/rep_guard_gate.json` | **4 752 / 4 752 fields identical, `GATE: PASS`** |
| ⭐ **the corrected tables** | `raw/corrected_tables.json` | T1 fail inventory · T2 corrected ladder · T3 rung profile · T4 checkpoint trajectory · T5 the two routes · T6 the guarded rep arms |
| ⭐ **the seed-stability finding** | `raw/seed_stability_v6_vs_cv0.json` | §5.3 — v6 vs C-V0 K1B per seed, all 11 rungs |
| suites | `raw/suite_taniteval.txt`, `raw/suite_stack.txt` | §16 |

### 17.3 Inputs read and NOT copied (large, already banked by the precedent runs)

`…/scratchpad/sp2/cache_{s02000,s09000,s09250,s10000,s11250,tok11250,nullmatched}/latents.pt`,
`…/scratchpad/pc/cache_orcdir/latents.pt`, `…/scratchpad/sp2/p3_selection.json`,
`…/scratchpad/sp2/lead130_agents.jsonl`, and the 130-clip episode cache
`…/scratchpad/sp2/cache/slotprobe-lead130-w120-256x640cyl/`.
⚠️ **CREATED IN SCRATCH, NOT BANKED (regenerable in ~30 s by `ll2_ego_oracle.py`):**
`…/scratchpad/ll/cache_egoorc_n{0.1,1,3,10}/latents.pt` (~33 MB each).

⚠️ **A STALENESS FOUND BY THIS CORRECTION, MEASURED, and it is the C100 warning firing again one day
later:** the scratch `ll1_ladder.py` was **stale against the repo** (md5 `b3531424` vs `4b57f4a8`) even
though the scratch `pc6_linear_readout.py` had been refreshed hours earlier by another stream. The
divergence was docstring-only this time and changed no number — **but it was found by comparing, not
by assuming**, and the chain now syncs *and* verifies `k1_guard(` and `K1_PASSES_GUARDED` are present
in `fit_one`'s **source**, not merely that the file exists. ⇒ **Staleness is per-file: one synced file
proves nothing about its sibling.**

---

## 18. ⛔ ESCALATIONS — these need a decision, and they are not filed in a README

1. ✅ **CLOSED — `pc6_linear_readout.ridge_fit` penalised its own intercept (§10.1).** Adopted:
   `intercept_col` exists in the module, the default stays penalised and the repair is opt-in, and
   the affected rows have been re-read. Logged as **C92**.
2. ✅ **CLOSED — the precedent's §2.3 sentence.** *"The latent carries some linearly-decodable lead
   information"* is **withdrawn**: the same signal is recovered better by the ego-speed scalar alone
   (K1 **−1.562 PASS** vs the latent's **+0.736 FAIL**), the latent is **0.694 m worse than the
   random null**, and partialling `v0` out leaves r **−0.107**. Logged as **C92**.
3. ⛔⛔ **OPEN AND DECISION-GRADE — `POOLING_BOTTLENECK_R1R2.md` §1.5 CITES THIS DOCUMENT FOR A CLAIM
   IT CANNOT SUPPORT (§12).** The three r² values it quotes have all moved, *"relative motion is
   exactly 0.0000"* is no longer literally true, its `n_agents_all` half is killed by the
   trivial-proxy control, and **all four rungs' FAIL verdicts are gone**. ⚠️ **I did not edit that
   document** — it is another agent's Research directory. **The request: re-quote §1.5 from §4.2 of
   this file, and demote the ladder from "independent confirmation" to "a consistent profile that
   does not localise", because E-R1-0 has since measured the localisation and dropped R1.** This is
   the item that can cost a training run.
4. ⛔ **OPEN — THE 165-ROW RE-READ SHOULD BE RE-RUN AT 3 SEEDS (§5.3, §8, §14.2).** Seed 0 alone
   reported the v6 latent beating the ego-speed scalar on `n_agents_all`; **three seeds reverse the
   sign.** Every other single-seed verdict on a contested rung is exposed to the same. **One flag,
   ~45 minutes of CPU, no GPU.** Until it runs, no single-seed guard verdict on a signal-bearing rung
   should be quoted alone.
5. ⚠️ **OPEN — THE ROTATIONAL RUNGS STILL HAVE NO POSITIVE CONTROL (§4.4, §14.1).** The LATERAL
   family's entire basis in this ladder rests on two nulls whose apparatus has never been shown able
   to measure them. **One line of `ll2_ego_oracle.py` closes it.**
6. ⚠️ **OPEN — `C-V0` SHOULD BECOME A STANDING COLUMN** on every latent-readout result in this
   programme (§14.5), **reported at ≥3 seeds**. It has now changed four headlines.
7. ✅ **CLOSED — `stack`'s suite was RED in the shared working tree** on 2026-08-17
   (`test_v6_staged.py:1157`, `sel_gap` stage-gate semantics). The owning agent's paired test update
   has landed; **`stack` is GREEN today** (§16). *Retained rather than deleted so the resolution is
   visible.*
8. ⚠️⚠️ **OPEN AND UNCHANGED — "STAGE, NEVER PUSH" DOES NOT PROTECT AN AGENT'S WORK.** The original
   run staged only, and a sibling's whole-index commit `109406c` — whose subject is about paper
   estimators — committed **48 of its 49 files** and pushed them. This is the third and fourth
   occurrence of a hazard `CLAUDE.md` already documents (`60265d3`, `3d41bd0`). ⇒ **The rule needs a
   mechanism, not another warning.** Cheapest options: agents stage to a per-agent index/worktree; or
   the committing agent must run `git diff --cached --name-only` and either name foreign entries in
   the message or unstage them. **Escalating rather than proposing a policy change unilaterally —
   this is the PI's to decide.**
