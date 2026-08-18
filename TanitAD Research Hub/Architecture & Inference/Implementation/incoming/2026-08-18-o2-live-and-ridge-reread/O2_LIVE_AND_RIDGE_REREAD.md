# E-O2-A ON THE LIVE LOG · AND THE K1 RE-READ UNDER AN UNBIASED FLOOR

**Date:** 2026-08-18 · **Branch:** `agent/arch-inf-20260803` · **Agent:** o2-live-and-ridge-reread
**Eval tier:** ⛔ **T0-DIAGNOSTIC throughout.** Nothing here is a driving number. A training-log
decomposition and a ridge readout on a latent are world-model diagnostics; per `EVAL_DOCTRINE.md`
only T1 (action-closed loop) carries a capability claim.
**GPU used: ZERO.** Thor's live v6F S-W 30k run (PID 25477) was read **read-only** over `ssh -n` and
not touched. Every fit here is CPU arithmetic on already-banked caches.

**Evidence classes:** `MEASURED (ours + artifact path)` · `DERIVED (algebra on our source, file:line)`
· `INHERITED (not re-verified)` · `ESTIMATED` · `HYPOTHESIS`.

---

## ⭐ THE ANSWER IN ONE PAGE

**JOB 1 — the banked E-O2-A finding is OVERTURNED, in both of its two claims.**

| | banked (`2026-08-17-O234-DESIGN-RESEARCH.md` §2.1a) | **live run (this report)** |
|---|---|---|
| source | dry-ladder rows, **steps 1–2**, **batch 2** | **v6F-SW-30k, steps 50–12 650** |
| n | 7 | **254** |
| `|Cov|/unweighted` | 0.45–3.33 %, **median 1.81 %** | 4.18–60.10 %, **median 34.23 %** |
| sign of `Cov` | **UNSTABLE — 4 −, 3 +** | ⭐ **254 −, 0 + — PERFECTLY STABLE** |
| trend | (not measurable at 2 steps) | **16.2 % → 46.4 %**, late/early **2.59×** |

⇒ ⛔ **O2's unique content does NOT stay ~1.8 % and the sign does NOT stay unstable.** It is
**~34 % of O2's value at the median, rising through training, and negative on every one of 254
rows.** The pre-registered reading — *"if it stays sign-unstable, say plainly that O2 is not
separately weightable"* — **does not fire.** On the evidence, **O2 carries a large, systematic,
single-signed component that is not O5's step-`j` term**, and the case for folding it into O5 as a
redundant duplicate is **not supported by the live run**.

⚠️ **AND THE HONEST BOUND ON THAT, STATED UP FRONT.** This measures O2's unique content **in VALUE**,
which is what the identity gives for free. It is **not** a gradient measurement, and *large distinct
value ≠ useful gradient*. Two things must travel with the headline:
1. **The direction O2 spends this on is unchanged and still looks wrong.** `Cov < 0` on every row
   means **high-weight (near) cells have LOW error and low-weight (far) cells have HIGH error** — so
   O2 systematically **de-emphasises the high-error far field**. §2.2 of the O234 doc stands: the
   1.70× row is road surface **3.0 m** under the ego's nose, and the lead's measured p10–p90
   (8.15–24.33 m) sits in rows weighted **0.77–1.40**. ⇒ **O2 is now measurably doing MORE — of
   something aimed away from the agents.**
2. ⛔ **The other half of the "O2 is redundant" case is ALSO an initialisation-time artifact, and
   nobody had said so.** The `cos(g_O2, g_O5) = +0.870` everyone quotes comes from a probe whose own
   meta reads verbatim: *"SYNTHETIC CPU build — **NOT** the live v6F S-W checkpoint"*, at
   **`n_params_all` 732 541** against the live model's **336 542 025** — a **459× smaller,
   randomly-initialised fixture** (`…/2026-08-16-o6-ablation/raw/mask_grad_probe.json`). It is a
   legitimate measurement of what it measured; it is **not** a live-run fact and should stop being
   quoted as one.

**JOB 2 — 2 of the 3 separated K1 FAILs survive an unbiased floor. The third was 98.7 % instrument.**

| arm | banked K1 | banked verdict | **repaired K1** | **new verdict** | survives? |
|---|---|---|---|---|---|
| `v6F-SW-30k@11250` | +1.580 [+0.765, +2.455] | FAIL sep | **+0.736 [+0.130, +1.442]** | FAIL sep | ✅ **SURVIVES** |
| `v6F-SW-30k@9000` | +1.811 [+1.044, +2.692] | FAIL sep | **+1.291 [+0.570, +2.142]** | FAIL sep | ✅ **SURVIVES** |
| `RANDOM-LATENT-NULL` | +3.401 [+3.014, +3.766] | FAIL sep | **+0.043 [−0.093, +0.176]** | **NOT separated** | ⛔ **DOES NOT SURVIVE** |
| `GT-ORACLE-CELLS` | +0.429 [−0.074, +0.895] | not sep | +0.103 [−0.353, +0.523] | not sep | — (never was a FAIL) |
| `GT-ORACLE-DIRECT` | −4.116 [−4.726, −3.603] | **PASS** | **−4.553 [−5.155, −4.045]** | **PASS** | ✅ **survives, strengthened** |

⭐ **All five incumbent refits reproduce the banked JSON BIT-EXACTLY**, so the only thing that moved
is the intercept penalty (`raw/pc6_refit_unbiased.json`, `all_incumbent_fits_reproduce_banked: true`).

⛔⛔ **AND THE CASUALTY IS A DERIVED CLAIM NOBODY LISTED AS AT RISK — it does not merely weaken, it
INVERTS, separated in both directions.** `O234 §3.4` reads the banked table as *"the v6 latent beats
a random-vector null by 1.6–1.8 m with positive correlation."* Re-run **paired, on identical
windows, both arms refitted**:

| | incumbent | **repaired** |
|---|---|---|
| `@11250` arm − null | **−1.821 [−2.566, −1.006]** sep ⇒ arm beats null | **+0.691 [+0.102, +1.362]** sep ⇒ ⛔ **arm does NOT beat null** |
| `@9000` arm − null | **−1.590 [−2.341, −0.721]** sep ⇒ arm beats null | **+1.246 [+0.562, +2.096]** sep ⇒ ⛔ **arm does NOT beat null** |

**The mechanism, MEASURED not hypothesised:** with the bias column penalised, the only way for the
ridge to reach y's ~15 m level was to **load the feature columns** — so the *noise* arm was forced to
hallucinate variance. Its incumbent `pred_sd` is **8.468 m against a GT sd of 6.200 m (1.366×)**.
Repaired, it collapses to `pred_sd` **0.131 m (0.021×)** — a constant, which is exactly what a
`torch.randn` latent should be. **The defect did not bias both arms equally; it punished the
no-signal arm hardest, and the arm-vs-null MARGIN is precisely the quantity it most inflated.**

---

## 0. THE STAMPS

1. **Thor was read, never touched.** `ssh -n … 'cat …/train_log.jsonl'`. md5 verified equal on both
   ends: **`370e778b0b7f79917c94302337f142c1`**, 286 145 B, 319 lines. The log is banked at
   `raw/v6F-SW-30k_train_log.jsonl` with the same md5. ⚠️ `ssh -n` was used on every call (a nested
   `ssh` in a pipe eats the rest of the script's stdin).
2. **No file owned by another agent was modified.** `pc6_linear_readout.py` is **imported**, never
   edited — its default stays at the incumbent behaviour and its pinning tests are untouched.
3. **Estimator:** `taniteval.ci.paired_episode_cluster_bootstrap`, seed 0, n_boot 2000.
   ⛔ `overlapping_holdout_se` appears nowhere in this path.
4. **Suites, MEASURED by me, not inherited:** `taniteval` **1107 passed, 0 failed** (154 s).
   ⚠️ **This contradicts the mid-task correction I was sent, which said 1101.** My brief said
   1092 + 15 = 1107; **1107 is what the suite prints.** `stack` result in §5.

---

## 1. JOB 1 — E-O2-A ON THE LIVE LOG

### 1.1 `DERIVED` — the identity, re-established from source rather than inherited

```
o2_near_field_loss                       stack/scripts/train_v6_staged.py:784-818
    err  = (pred_cells - true_cells).abs().mean(dim=-1)     # [B, C]
    loss = (w * err).mean()                                 # -> "o2_loss"
    log            "o2_unweighted": err.mean()              # -> O5's step-j term
time_to_reach_weights                    stack/tanitad/models/v6.py:325-345
    normalize=True  ->  w = w / w.mean(dim=-1, keepdim=True)   # MEAN-1 OVER CELLS
```

Both means run over the **same `B` and `C` axes**, and `w` is mean-1 **per batch element** over the
cell axis. Therefore

> **`o2_loss − o2_unweighted` = `E_{b,c}[(w−1)·err]` = `E_b[ Cov_c(w_b, err_b) ]` — EXACT, no
> residual, nothing approximated.**

✅ Both halves re-read at `file:line` in the working tree. The O234 doc asserted this; it now stands
on a second, independent reading.

### 1.2 ⚠️ THE TWO ROW SCHEMAS — handled by schema, never by position

`train_log.jsonl` carries **three** schemas, and the instrument prints the census so the filtering is
auditable (`code/o2_live_trend.py`):

| n | schema |
|---|---|
| **254** | **TRAIN** — `loss`/`gnorm`/`o2_*`/`o5_*`/`o3_*`/`o6_*` |
| **63** | **SPECTRUM** — `{"step": N, "spectrum": {…}}`, **no `loss` field** |
| **2** | **RUN_START** |

⛔ A parser using `.get("loss", 0)` turns the 63 spectrum rows into zeros and they read as a training
collapse. The filter here is *"both `o2_loss` and `o2_unweighted` present"*, which a spectrum row
cannot satisfy.
⚠️ **Two `run_start` rows = the run was RESTARTED.** Step **6300 appears twice**, from two different
processes (`step_s_note` reads *"over the 6300 steps THIS process ran"* and *"over the 50 steps"*),
with **different** values — a 50-step replay after resuming from a step-6250 checkpoint. Both are
legitimate independent samples and both are kept; the duplicate is disclosed rather than silently
deduped.

### 1.3 ⭐ `MEASURED (ours — raw/o2_live_trend.json)` — the result

**n = 254 training rows, steps 50 → 12 650.**

| statistic | value |
|---|---|
| `|Cov|/unweighted` min / p25 / **median** / p75 / max | 4.18 % / 24.04 % / **34.23 %** / 43.21 % / 60.10 % |
| **sign of `Cov`** | ⭐ **254 negative, 0 positive** |

**Trend (1000-step bins):**

| steps | n | median `|Cov|`/unwt | signs | median `o2_unweighted` |
|---|---|---|---|---|
| 0–999 | 19 | **16.19 %** | 19 − / 0 + | 0.37279 |
| 1000–1999 | 20 | 17.54 % | 20 − / 0 + | 0.11871 |
| 2000–2999 | 20 | 27.22 % | 20 − / 0 + | 0.06051 |
| 3000–3999 | 20 | 10.72 % | 20 − / 0 + | 0.40592 |
| 4000–4999 | 20 | 38.84 % | 20 − / 0 + | 0.09081 |
| 5000–5999 | 20 | 32.63 % | 20 − / 0 + | 0.06991 |
| 6000–6999 | 21 | 39.74 % | 21 − / 0 + | 0.05300 |
| 7000–7999 | 20 | 34.25 % | 20 − / 0 + | 0.05360 |
| 8000–8999 | 20 | 33.65 % | 20 − / 0 + | 0.05205 |
| 9000–9999 | 20 | 42.92 % | 20 − / 0 + | 0.05572 |
| 10000–10999 | 20 | 44.00 % | 20 − / 0 + | 0.04502 |
| 11000–11999 | 20 | 44.58 % | 20 − / 0 + | 0.03903 |
| **12000–12999** | 14 | **46.41 %** | 14 − / 0 + | 0.03231 |

✅⭐ **INDEPENDENTLY CROSS-CHECKED WITH THE BANKED INSTRUMENT ITSELF.** The O234 package's own
`o2_cov_from_logs.py`, run **unmodified** on this log, returns **n 254 · min 4.175 % · median
34.232 % · max 60.097 % · 254 negative, 0 positive** — identical to my instrument to every printed
digit (`raw/o2_cov_from_logs_BANKED_INSTRUMENT_on_live_log.json`). ⇒ **the overturn is not an
artifact of a new parser.** ⚠️ The banked instrument was run from a **COPY in scratch**, because it
writes its output next to itself and would otherwise have overwritten another agent's banked
`raw/2026-08-17-O234/o2_cov_from_logs.json`; that file is verified untouched.

**The admissible growth statement.** OLS of `|Cov|/unwt` on step over **[50, 12 650]** gives
**+2.621 %/1000 steps at R² 0.525, n 254**. ⛔ Per `CLAUDE.md`, **below R² 0.80 there is no quotable
slope** — so the quotable statement is the **matched-step ratio**:

> **early [50, 1000], n 20: median 17.09 % · late [11 650, 12 650], n 21: median 44.30 % ·
> late/early = 2.59×.**

⚠️ **Estimator note, stated so it is not asked for.** No episode-cluster bootstrap is quoted on these
numbers **on purpose**: this is a **serially correlated time series of training-batch statistics**,
not a per-window eval metric over the 40 val episodes. The paired episode-cluster bootstrap is the
decision-grade estimator for *eval* numbers; applying it here would manufacture an interval for a
quantity that has no episode clustering. What is reported instead is the full trajectory, quantiles,
the sign census, and a fit carrying its window/R²/n. ⛔ `overlapping_holdout_se` was not used.

### 1.4 ⛔ WHY THE BANKED NUMBER WAS WRONG — worse than "initialisation rows"

The O234 doc's own caveat said the banked n = 7 came from *"dry-ladder rows at steps 1–2, at or near
initialisation"*. **Opening the raw files shows it is narrower than that** (`MEASURED`,
`…/2026-08-16-v6-stage-chain/raw/dry_ladder_{default,arms}.json`, `dry_ladder_default.log`):

| | dry ladder (banked n = 7) | **live run (n = 254)** |
|---|---|---|
| `o3_n_masked` / `o3_mask_rate` ⇒ **B × C** | 10 / 0.3125 ⇒ **32** ⇒ C 16, **batch 2** | 57 / 0.4453 ⇒ **128** ⇒ C 16, **batch 8** |
| `o5_k` | **12** | **60** |
| `o2_w_min` / `o2_w_max` | **byte-identical across all four files at a given step** | varies every row |
| step | **1–2** | 50–12 650 |

⇒ The identical `w` extrema across all four banked files mean the dry ladder **replayed one fixed
dummy batch**. The banked "n = 7" is **2 distinct batches at 2 steps of a smoke test on 2 windows**,
counted across stages and chain variants. At that point the per-cell error profile is near-uniform
and `Cov_c(w, err) ≈ 0` **by construction** — which is precisely what the doc predicted and flagged.
**The flag was correct; the number was never load-bearing.**

### 1.5 ⭐ `DERIVED` — the mechanism, and a floor on the cross-cell error dispersion

A single-signed negative `Cov` says the error profile across cells is **anti-correlated with `w`**.
From `|E_b Cov_c(w,err)| ≤ max_b std_c(w_b) · E_b[std_c(err_b)]` and
`std_c(w_b) ≤ (w_max − w_min)/2`, the log's own `o2_w_min`/`o2_w_max` give a **lower bound** on the
batch-mean cross-cell coefficient of variation of the per-cell error
(`raw/o2_cell_dispersion_floor.json`):

| steps | n | median CV(err) **floor** |
|---|---|---|
| 0–1 999 | 39 | **16.3 %** |
| 2 000–3 999 | 40 | 14.8 % |
| 4 000–5 999 | 40 | 28.9 % |
| 6 000–7 999 | 41 | 31.4 % |
| 8 000–9 999 | 40 | 33.4 % |
| 10 000–11 999 | 40 | **38.7 %** |
| 12 000–13 999 | 14 | 36.6 % |

⇒ **The per-cell error profile starts nearly flat and becomes strongly non-uniform.** That is the
precondition for O2's weight profile to do *anything*: at initialisation `w` is a no-op because every
cell has the same error; by step 12 000 the far cells are much harder than the near cells and `w`
re-allocates real loss mass away from them.

### 1.6 ⛔ THE ANSWER TO THE PI'S QUESTION, STATED PLAINLY

> *Does O2's unique contribution stay ~1.8 % and sign-unstable, or does it grow into something that
> justifies its own weight?*

**It grows, and the sign stabilises completely.** `n = 254`; median **34.23 %**; **254/254 negative**;
late/early **2.59×**. ⇒ ⛔ **"O2 is not separately weightable" is NOT the finding, and I am not able
to write that sentence.** The banked basis for it (1.81 %, 4 −/3 +) was **2 dummy batches at steps
1–2**, and the live run contradicts both halves.

⚠️ **What this does NOT license, so it is not over-read:**
* It is **not** evidence O2 is *useful*. It is evidence O2 is *distinct*. Those come apart, and §1.3's
  sign says the distinct part points at the **near field, which is the easy field** — O2 spends its
  one degree of freedom down-weighting exactly the cells the model is worst at.
* It is a **value** decomposition, not a **gradient** one. `∇O2 − ∇per_j = E[(w−1)·∇err]` is a
  different object from `Cov_c(w, err)` and this measurement does not bound it.
* ⭐ **The corresponding gradient claim is now the WEAKER of the two, not the stronger.**
  `cos(g_O2, g_O5) = +0.870` is `MEASURED` on a **732 541-parameter synthetic CPU fixture at random
  init** (its own meta says so verbatim), against a live model of **336 542 025** parameters.
  **Both legs of "O2 is redundant with O5" are therefore initialisation-time; the live run has now
  refuted one of them and left the other unmeasured at scale.**
* ⚠️ **EARLY-READ:** 12 650 of 30 000 steps (**42 %**), S-W stage only. The trend is monotone in the
  binned medians from step 4 000 on, but the run is not finished.

---

## 2. JOB 2 — THE K1 RE-READ UNDER AN UNBIASED FLOOR

### 2.1 The defect, and the fact that it is visible in the banked files themselves

`pc6_linear_readout.ridge_fit` put the appended **ones-column inside `alpha * np.eye(d)`**, so the
intercept was shrunk like any coefficient and predictions collapsed toward **ZERO, not the MEAN**.
The readout could not express the constant it was scored against ⇒ **a no-signal arm scored worse
than a constant BY CONSTRUCTION**, which biases the **floor** and therefore taints **K1 FAIL**
verdicts specifically.

⭐ **It is legible in the banked `alpha_inner_mae` tables without refitting anything.** Every
incumbent table **rises** at high alpha — `nullmatched` reaches **16.23 m at alpha 1e5** while
`c_const_err` is **5.13 m** and `mean(y) ≈ 15 m`. An MAE converging to *mean(|y|)* rather than to the
constant's error **is** the collapse-toward-zero signature. Under the repair the same tables **fall**
to **4.7–5.1 m**, i.e. toward the constant.

### 2.2 ⭐ The inventory — ENUMERATED BY OPENING THE ARTIFACTS

⛔ Per C91, a verdict inventory taken from a headline instead of the artifact under-counts. Every row
below was read out of a JSON on disk (`code/ridge_artifact_audit.py`, `raw/ridge_artifact_audit.json`).

**Two producers share the defective solve.** `ll1_ladder.py:86` does
`from pc6_linear_readout import ridge_fit` — there is **no second implementation**.

| | |
|---|---|
| files opened | **24** |
| verdict rows | **214** |
| **still on the INCUMBENT (biased) solve** | ⛔ **170** |
| on a repaired solve | 44 |
| verdicts overall | 32 PASS · 113 FAIL-separated · 69 not-separated |
| ⛔ **separated-FAIL verdicts standing on the biased floor** | ⛔ **90** |

⇒ **The blast radius is not 5 verdicts. It is 5 `pc6_ridge_*.json` (1 target each) plus 19
`ll_*.json` (11 targets each), of which 15 ladder files are `fit_mode: "pc6"` — the incumbent.**

### 2.3 ⭐ The re-read — all five, with a bit-exact reproduction gate

Each arm was fitted **three** ways so the change is attributable, and the incumbent fit **must**
reproduce the banked file before anything else is believed:

| path | intercept | alpha |
|---|---|---|
| **A incumbent** | penalised | re-selected — **must reproduce banked** |
| **B repaired-held** | **un**penalised | held at the banked choice — isolates the penalty |
| **C repaired** | **un**penalised | re-selected — **the honest re-read** |

✅ **`all_incumbent_fits_reproduce_banked: true`** — alpha, error, K1 δ/lo/hi, separation, verdict and
correlation all match to < 1e-6 on all five arms.

⭐ **Path B is the cleanest single number in this report: with alpha held, the repair moves K1 by at
most +0.14.** (`@11250` +1.580 → +1.601; `@9000` +1.811 → +1.831; null +3.401 → +3.405;
`orcdir` −4.116 → −4.116, unchanged.) ⇒ **almost the entire effect of the repair comes through
ALPHA RE-SELECTION**, i.e. through the fact that the un-penalised fit is finally *allowed* to shrink
to the mean. That is the defect's real shape.

**Verdict table (path C; robust to extending the alpha grid to 1e7 —
`raw/pc6_refit_unbiased_extgrid.json`, arms re-select the same alphas):**

| arm | old K1 | old verdict | **new K1** | **new verdict** | change |
|---|---|---|---|---|---|
| `v6F-SW-30k@11250` | +1.580 [+0.765, +2.455] | FAIL sep | **+0.736 [+0.130, +1.442]** | FAIL sep | ✅ survives, halved |
| `v6F-SW-30k@9000` | +1.811 [+1.044, +2.692] | FAIL sep | **+1.291 [+0.570, +2.142]** | FAIL sep | ✅ survives |
| `RANDOM-LATENT-NULL` | +3.401 [+3.014, +3.766] | FAIL sep | **+0.043 [−0.093, +0.176]** | **NOT sep** | ⛔ **dies — 98.7 % artifact** |
| `GT-ORACLE-CELLS` | +0.429 [−0.074, +0.895] | not sep | +0.103 [−0.353, +0.523] | not sep | — |
| `GT-ORACLE-DIRECT` | −4.116 [−4.726, −3.603] | **PASS** | **−4.553 [−5.155, −4.045]** | **PASS** | ✅ strengthened (1.016 → **0.580 m**, r **+0.9932**) |

**The degeneracy diagnostic that makes these readable** — `pred_sd / gt_sd` (gt_sd = 6.200 m):

| arm | incumbent | **repaired** | reading |
|---|---|---|---|
| `RANDOM-LATENT-NULL` | **1.366** | **0.021** | ⭐ incumbent **hallucinated more spread than the truth**; repaired is a **constant** — correct null behaviour |
| `@11250` | 0.932 | 0.654 | a real, weak readout |
| `@9000` | 1.002 | 0.833 | a real, weak readout |
| `GT-ORACLE-CELLS` | 0.696 | 0.460 | real, weak |
| `GT-ORACLE-DIRECT` | 1.006 | 0.987 | real, strong |

### 2.4 ⛔⛔ THE DERIVED CLAIM THAT INVERTS — and its true magnitude

`O234 §3.4` / `§3.4a` read the banked table as *"the v6 latent beats a random-vector null by
1.6–1.8 m with positive correlation"*. That is a **comparison of two fits that were both defective**,
and the defect is **asymmetric**: it punishes the no-signal arm hardest.

Re-run **paired** on windows verified elementwise identical (`pA_null_matched.py` replaces only the
`cells` tensor with `torch.randn`, keeping every window; `code/pc6_arm_vs_null.py` **asserts**
`yev`/`eev` equality and refuses to report an unpaired comparison as paired):

| | incumbent | **repaired** |
|---|---|---|
| `@11250` arm − null | **−1.821 [−2.566, −1.006]** sep | **+0.691 [+0.102, +1.362]** sep |
| `@9000` arm − null | **−1.590 [−2.341, −0.721]** sep | **+1.246 [+0.562, +2.096]** sep |

⇒ ⛔ **A SEPARATED SIGN FLIP on both checkpoints.** The banked "+1.6–1.8 m advantage over the null"
must be **retracted**.

⚠️ **BUT DO NOT REPLACE IT WITH "THE LATENT IS WORSE THAN NOISE" — that would be the same
over-reading in the opposite direction, and I checked.** The alpha sweep
(`raw/pc6_alpha_sensitivity.json`, repaired solve, full grid to 1e7, `c_const_err` **5.133 m**):

| alpha | `@11250` err / pred_sd | `@9000` err / pred_sd | null err / pred_sd |
|---|---|---|---|
| 1e3 | 6.281 / 4.73 | 6.424 / 5.17 | 5.987 / 3.75 |
| 1e4 | 5.869 / 4.05 | 6.022 / 4.63 | 5.253 / 0.99 |
| 1e5 | 5.348 / 2.45 | 5.434 / 2.85 | 5.178 / 0.13 |
| **1e6** | **5.113** / 0.77 | **5.105** / 0.96 | 5.175 / 0.01 |
| 1e7 | 5.158 / 0.13 | 5.155 / 0.14 | 5.175 / 0.00 |

⇒ ⭐ **The defensible statement, and it is the one that should enter the registry:**
**at the eval-optimal alpha — chosen with hindsight ON THE EVAL SET, i.e. cheating in the arm's
favour — the v6 latent reaches 5.113 m against a constant's 5.133 m and a noise latent's 5.175 m.**
That is an edge of **~0.02 m over a constant and ~0.06 m over noise**, achieved at `pred_sd` **0.77 m
against a GT sd of 6.20 m** — i.e. **by being a constant**. **No alpha anywhere on the grid reaches a
K1 PASS.**

> **The v6 latent's linear lead-gap content is real but its magnitude is ~0.02–0.07 m, not 1.6–1.8 m
> — a 25–90× overstatement.** The arm-vs-null *inversion* is an honest consequence of honest
> (inner-split) alpha selection landing at 1e4 for the arm and 1e6 for the null; the *magnitude*
> statement above is the one that does not depend on that choice.

⭐ **And the diagnosis this yields is more useful than either headline:** at a **matched** alpha of
1e4 the arm's inner-split MAE is **4.870 m against the null's 5.079 m** — the arm really does carry
something — while its **eval** error at that same alpha is **5.869 m, worse than a constant (5.133)
and worse than the null (5.253).** ⇒ **the latent carries something the probe-train clips call signal and the
held-out episodes call noise. That is episode-level overfitting, not agent geometry** — and it is
consistent with the ladder's independent finding that the lead-gap correlation is an **ego-speed
proxy** (partialling `v0` out drops r from +0.159 to +0.052).

> ⛔ **CORRECTION 2026-08-18 (citation sweep) — `+0.052` IS STALE AND HAD THE WRONG SIGN.** Under the
> repaired ridge at three inner-split seeds, `lead_gap`'s partial-`v0` correlation is
> **−0.0884 (3-seed mean)**, per seed **−0.1065 / −0.0665 / −0.0922** — ⚠️ that bracket is a **SEED
> SPREAD, NOT a confidence interval**, and the **−0.107** circulating as the replacement is the
> **seed-0** value. ⭐ **This document's conclusion strengthens**: the correlation does not merely
> shrink toward zero once `v0` is removed, **it changes sign** — which is exactly the *"the probe-train
> clips call it signal and the held-out episodes call it noise"* reading above.
> `MEASURED` · `v6F-SW-30k@11250` ⚠️ **early read, 37.5 %** · **T0-DIAGNOSTIC** · 70 eval clips ·
> `intercept_col=-1` + C97 guard · route A (`unpen`); **identical on route B for this rung** ⛔ **but
> the routes are never pooled — `ego_v0`'s K1 differs between them by 0.3957.** Artifact
> `…/incoming/2026-08-18-ladder-3seed/raw/reread_unpen/ll3_s11250.json`; re-derivation
> `…/incoming/2026-08-18-citation-sweep/raw/canonical_requote_table.json`. Authority **C103 · C107**.

### 2.5 ⛔ A SECOND DEFECT FOUND WHILE RE-READING — K1 DEGENERATES INTO MEAN-vs-MEDIAN

The ladder's own 4 repaired arms change **21 of 44** rung-verdicts. ⚠️ **Most of those changes are
not findings, and one is an outright false positive that must be caught before it is quoted.**

⛔ **MEASURED: the pure-`torch.randn` null "PASSES" K1 on `n_agents_all` after the repair**
(−1.884 [−3.517, −0.290], separated) — `ll_rep_nullmatched.json`. A latent containing **zero**
information cannot have skill. Opening the record explains it:

| | incumbent | repaired |
|---|---|---|
| err | 43.851 | **36.051** |
| `c_const_err` (train **MEDIAN** = 34.0) | 37.936 | 37.936 |
| `pred_sd` vs `gt_sd` **46.459** | 20.020 | ⭐ **0.715** |
| `corr` | +0.027 | **+0.014** |

⇒ **`pred_sd` 0.715 against `gt_sd` 46.459 means the readout is a FLAT LINE.** It "beats" C-CONST
only because a ridge's shrinkage target is the train **MEAN** while `C-CONST` is the train
**MEDIAN**, and on this **right-skewed** target (mean 62.8, median 34.0) the mean is the better MAE
constant for these eval episodes.

> ⛔ **ROOT-CAUSE CLASS: under the repair, a fully-shrunk ridge IS "predict the train mean", so K1
> becomes a MEAN-versus-MEDIAN contest on skewed targets — a verdict about which constant, not about
> the latent.** This is the *mirror image* of C92: C92 made no-signal arms **fail** by construction;
> this makes them **pass** by construction. **Loosening a floor is a candidate FAIL-suppressor** —
> the same lesson as C95.

⇒ **The guard is already computable and already banked:** any verdict whose `pred_sd/gt_sd` is
negligible (the null's repaired `lead_gap` is **0.002**, `n_agents_grid` **0.0001**) is a
**DEGENERATE CONSTANT** and its K1 must be reported as *constant-vs-constant*, never as evidence
about the latent. ✅ **On `lead_gap` — the target this report turns on — the repaired null behaves
correctly (K1 +0.043, NOT separated), so §2.3's conclusions are unaffected.**

---

## 3. ⛔ ESCALATIONS — these need a decision or an edit by their owner, and are NOT filed as
"please merge" in a README

1. ⛔ **`2026-08-17-O234-DESIGN-RESEARCH.md` §2.1a must be corrected.** Its headline
   (*"~98 % of O2's value is literally O5's step-`j` term, and the remaining ~2 % does not even have
   a stable sign"*) is refuted on the live run: **median 34.23 %, 254/254 single-signed**. Its own
   caveat predicted exactly this and named E-O2-A as the settling measurement. **E-O2-A has now run.**
2. ⛔ **`O234 §3.4`'s "beats a random-vector null by 1.6–1.8 m" must be RETRACTED** (separated sign
   flip on both checkpoints) and replaced with the magnitude statement in §2.4. **This propagates:
   §3.4a and the one-page summary both lean on it.**
3. ⛔ **`cos(g_O2, g_O5) = +0.870` must stop being quoted as a live-run fact.** It is a **732 541-param
   synthetic CPU fixture at random init**; the meta says so and every re-quote has dropped that. With
   §1.6, **both legs of "O2 is redundant with O5" are initialisation-time.** ⇒ **the cheapest real
   test is a single-batch gradient probe on the LIVE checkpoint** — `ckpt.pt` is on Thor and a CPU
   probe needs no GPU, but it does need the 3.5 GB checkpoint moved, so it is not free and is **not**
   claimed here.
4. ⛔ **170 of 214 banked ridge verdict rows — including 90 separated-FAILs — still stand on the
   biased floor.** Only `lead_gap` (5 arms) and the ladder's 4 repaired arms have been re-read. The
   remaining 15 `ll_*.json` are a **zero-GPU** re-read.
5. ⛔ **NEW DEFECT (§2.5): K1's ridge-vs-C-CONST comparison degenerates into mean-vs-median on skewed
   targets under the repair, and a pure-noise null already "PASSES" one rung because of it.** Needs
   (a) a `pred_sd/gt_sd` degeneracy guard on every K1 verdict, and (b) a decision on whether C-CONST
   should be the train **mean** for an MAE-scored ridge. **Until then, no repaired K1 PASS on a
   skewed target is quotable.**
6. ⚠️ **The banked alpha grid `[1e-2, 1e5]` is too narrow for the repaired solve** — the eval optimum
   sits at **1e6**. The verdicts here are robust to extending it (arms re-select the same alpha), but
   any future repaired fit should carry the wider grid and report `alpha_at_grid_edge`.
7. ⚠️ **Suite count correction:** `taniteval` measures **1107**, not the 1101 I was sent mid-task.

---

## 4. WHAT I DID NOT DO

* **No gradient measurement.** §1 is a **value** decomposition. The gradient question is open and
  escalation 3 is the cheapest route to it.
* **No re-read of the 15 incumbent `ll_*.json` ladder files** (165 rung-verdicts). Inventoried and
  escalated, not refitted — out of scope for two jobs and it needs escalation 5 settled first, or it
  will manufacture false PASSes on the skewed rungs.
* **No edit to `pc6_linear_readout.py`, `MODEL_REGISTRY.md`, or any O234/ladder document.** Other
  agents own them; §3 is the escalation.
* **No T1 claim.** Everything is T0-DIAGNOSTIC.

---

## 5. SUITES

| suite | result | note |
|---|---|---|
| `taniteval` | ⭐ **1107 passed, 0 failed**, 154 s (`raw/suite_taniteval.txt`) | ⚠️ contradicts the **1101** I was sent mid-task; **1107** matches my brief and is what the suite prints |
| `stack` | ⭐ **3816 passed, 0 failed, 7 skipped, 2 xfailed**, 451 s (`raw/suite_stack.txt`) | ✅ exactly the corrected baseline — **GREEN**. The `test_v6_staged.py:1157` failure I was originally briefed about is gone (E4's work landed at `0a0f421`). |

Nothing in this package is imported by either suite — all new files live under
`TanitAD Research Hub/…/incoming/2026-08-18-o2-live-and-ridge-reread/`.

---

## 6. DELIVERABLE MANIFEST

⭐ **Everything below is in the REPO and STAGED. Nothing exists only on a pod, only in scratch, or
only in this agent's context.** The Thor log is the one artifact that also lives elsewhere — it is
**copied into the repo**, md5-verified, so the pod copy is a duplicate and not a dependency.

| artifact | where it lives | what it is |
|---|---|---|
| `O2_LIVE_AND_RIDGE_REREAD.md` | `repo:…/incoming/2026-08-18-o2-live-and-ridge-reread/` | this report |
| `code/o2_live_trend.py` | ″ `/code/` | E-O2-A instrument — schema-filtered live-log decomposition |
| `code/pc6_refit_unbiased.py` | ″ `/code/` | 3-path refit + bit-exact reproduction gate |
| `code/pc6_arm_vs_null.py` | ″ `/code/` | paired arm-vs-null, asserts window identity |
| `code/ridge_artifact_audit.py` | ″ `/code/` | opens every ridge artifact; verdict inventory |
| `raw/v6F-SW-30k_train_log.jsonl` | ″ `/raw/` | **the live log**, md5 `370e778b…` (also `thor:/home/nvidia/experiments/v6F-SW-30k/train_log.jsonl`) |
| `raw/o2_live_trend.json` | ″ `/raw/` | n=254 rows, bins, sign census, fit, matched-step ratio |
| `raw/o2_cov_from_logs_BANKED_INSTRUMENT_on_live_log.json` | ″ `/raw/` | ⭐ the **O234 package's own unmodified instrument** on the same log — identical result |
| `raw/o2_cell_dispersion_floor.json` | ″ `/raw/` | DERIVED cross-cell CV floor per row |
| `raw/pc6_refit_unbiased.json` | ″ `/raw/` | 5 arms × 3 paths, banked alpha grid |
| `raw/pc6_refit_unbiased_extgrid.json` | ″ `/raw/` | same, alpha grid extended to 1e7 |
| `raw/pc6_arm_vs_null.json` | ″ `/raw/` | the separated sign flip |
| `raw/pc6_alpha_sensitivity.json` | ″ `/raw/` | eval error vs alpha — bounds the true magnitude |
| `raw/ridge_artifact_audit.json` | ″ `/raw/` | 24 files, 214 verdict rows, 44 paired re-reads |
| `raw/suite_taniteval.txt`, `raw/suite_stack.txt` | ″ `/raw/` | suite output |

⚠️ **Depends on scratch, and this is disclosed rather than hidden:** the refit scripts read the
latent caches at
`…/scratchpad/{pc/cache_orc010, pc/cache_orcdir, sp2/cache_nullmatched, sp2/cache_s09000,
sp2/cache_s11250}/latents.pt` and the split `…/scratchpad/sp2/p3_selection.json`. **These are
multi-hundred-MB `.pt` files that exist in ONE PLACE (scratch) and are not banked here.** They are
regenerable from the checkpoints, but if the caches are wanted for re-analysis they should be banked
deliberately — that is a decision for the owner of the probe-positive-control package, and it is
**escalated, not silently assumed**.
