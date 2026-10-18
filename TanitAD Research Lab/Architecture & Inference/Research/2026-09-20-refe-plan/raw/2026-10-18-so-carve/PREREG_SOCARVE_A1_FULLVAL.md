# PREREG_SOCARVE addendum A1: the same constraint quantity on the FULL RL-val (3,000 sets / 75 logs), from the banked live score dumps

Written 2026-10-18, AFTER the main read of `PREREG_SOCARVE.md` (sha256 in `PREREG_SOCARVE.sha256`, frozen 04:47:33Z, outcome read ~04:55Z) and BEFORE any number of this addendum was computed. Its sha256 is in `PREREG_SOCARVE_A1_FULLVAL.sha256`. It changes NOTHING in the main prereg: the main label stands as read.

## 0. Why this addendum exists (Rule Zero: the main read ended on an unresolved gap)

The main read labelled CARVE-TRACKS-TRAIN with flags **F1** (the carve is in-sample by construction) and **F2** (the 95 % interval of `G_tv` spans zero by a wide margin, because the harness's val pool is only 62 sets / 13 logs). The cheapest experiment that can still tighten the val side costs 0 compute: the R0 arm dumped its LIVE scorer logits (base, A, B) on the **whole RL-val, 3,000 sets / 75 logs**, at u250 / u500 / u750 / u1000 (`C:/Users/Admin/rls1b_dl/r0/final/scores_update*_{A,B}.npz`, `scores_step0_base.npz`, `space = logit`). With the true PDM labels of RL-val, g can be computed on all 75 logs.

What it does and does not repair: it tightens the **val side** only. The train side stays the harness's 302-set / 30-log cached pool (no train-set dumps exist), and F1 is untouched.

## 1. Disclosure

* I have seen every main-read number (`socarve.json`): at R0 u1000, P2 TTC, `L_tr = +1.012`, `L_va(pool) = +1.356`, `L_c = +1.043`, `G_tv = +0.344 [-3.30, +4.50]`.
* I have seen, as INHERITED context in `PREREG_SAFEOBJ.md`, that R0's RL-val TTC at u1000 vs the BASE is -1.00 [-1.73, -0.30] (pick, 3,000 sets). I have NOT read `ab_vs_base.json` (which holds R0's B - A on RL-val) nor computed anything from the dumps.
* So the sign of the full-val pick loss is roughly known (about 1 point against the base); P2 and the B - A version are not.

## 2. Quantities (all as in the main prereg s.2, now on RL-val's 3,000 sets)

* Labels: `attrib_rls1b.load_side(pdm_val, terms_val)` (`C:/Users/Admin/rls1b_dl/data/x/{pdm_val,terms_val}/...`), the same RL-val label join as `ab_vs_base.py` and the R0d reads; `attrib_rls1b.comps` for the TTC / NC / EP columns. Logits: `attrib_rls1b.load_scores_aligned` (logit space, key-aligned, stale-checked).
* Policy: `pi = softmax(log(sel(sigmoid(sx)) + 1e-6) / 0.1)` over the 64 proposals, `sel = NC * DAC * (5 EP + 5 TTC + 2 C) / 12 * lane` (`attrib_rls1b.sel_v1_lane`, `T_POL = 0.1`, `EPS = 1e-6`; the trainer's `epo_policy`, `sel_score`). **P2**: `d_s = 100 * (sum_i pi^B_i y_i - sum_i pi^A_i y_i)`; **P1**: `d_s = 100 * (y[pick_B] - y[pick_A])`, picks by `attrib_rls1b.picks`. `L = -mean_s d_s` (sets-weighted), as before.
* Evidence class: **MEASURED** for the val side (live bf16 scorer logits dumped by the arm, true PDM labels; RL-val selection tier, not an endpoint, not T1). The train side remains **ESTIMATED** (harness, cached contexts).
* Estimator: paired log-cluster bootstrap, 10,000, seed 20260927. Val side: `VI75 = default_rng(20260927).integers(0, 75, size=(10000, 75))`. Train side: the main read's `TI` (the same resamples). The two sides are independent. "Another draw of LOGS", training state fixed, no replicate.

## 3. Controls (A1 is INVALID if the first fails)

1. **Reproduction of a banked number**: my P1 (pick) `B - A` on the 3,000 sets, for TTC, NC, EP and PDMS (x100), at R0 u250 / u500 / u1000, must equal `PK/raw/2026-10-15-rls1-adam-dose/ab_vs_base.json`, `rows.r0_u{250,500,1000}.B_minus_A.<comp>` (mean and both CI endpoints) to 1e-6. I read that file for this comparison only AFTER writing it into the code path. My P1 reuses `attrib_rls1b.picks` and `boot_delta`, so a mismatch means a join or alignment error.
2. **Zero control**: `B := A` gives `L = 0.0` exactly, P1 and P2, on all 3,000 sets and in every bootstrap resample.
3. **Fidelity of the cached-context harness (reported, NOT a gate)**: the harness's 62 val-pool sets are among the 3,000. On those 62 sets, compare the dump-based per-set `d_s` (live, bf16) with the harness's (cached context, fp32): report the max and mean |difference| of `d_s` (P2 TTC, P1 TTC) and the pooled `L` on the 62. Tag **FAITHFUL** iff |L_pool(dump) - L_pool(harness)| <= 0.25 points (P2 TTC, u1000), else **FIDELITY-GAP**. This bounds how far the harness's TRAIN-side `L_tr` (which has no dump counterpart) can be trusted.
4. The main read's label function and its planted controls are reused unchanged (`so_carve_analysis.label_fn`).

## 4. Outcomes of A1 (committed; they qualify the main read, they do not replace it)

For each R0 state (u1000 decides the reading; u250 / u500 descriptive) and for P2 TTC (primary) and P1 TTC (secondary):

* `L_va_full`, its interval (75 logs); `G_tv_full = L_va_full - L_tr` and its interval (train and val resampled independently).
* **A1-GAP**: RESOLVED-POSITIVE iff the `G_tv_full` interval lies above 0; RESOLVED-NEGATIVE iff below 0; else UNRESOLVED. Additionally **BAR-SIZED-GAP-EXCLUDED** iff the interval lies inside (-0.5, +0.5).
* **A1-LABEL**: the main read's three-outcome rule with the full-val figures, `label_fn(L_tr, L_va_full, L_c, CI(L_va_full))`, where `L_c` is the main read's in-sample carve (m = 13 of the 30 train-pool logs, R = 10,000 draws, seed 20260927). F1 is raised on it again.
* **A1-REP**: the pool's 62-set `L_va` is inside / outside the interval of `L_va_full`.

**Reading rules, committed before any number:**

* **R-A**: A1-GAP RESOLVED-POSITIVE at u1000 (P2 TTC) AND A1-LABEL CARVE-TRACKS-TRAIN => a real train-to-val gap exists at the R0 state and an in-sample carve is blind to it (as F1 predicts). What stays open is whether a HELD-OUT carve closes it: only R0e can say.
* **R-B**: BAR-SIZED-GAP-EXCLUDED at u1000 (P2 TTC) => the unconstrained R0 state shows no train-to-val gap of the bar's size; the O4 breach is then not a property of the data split at R0's state. Candidate causes that the registered readings can still separate: the gap opens when the dual drives the in-sample quantity to about 0 (an optimisation effect, R0 cannot show it), and the pick-vs-expectation wedge (RZ-1). The carve lever's expected value drops accordingly.
* **R-C**: A1-GAP UNRESOLVED and not BAR-SIZED-GAP-EXCLUDED => still unresolved; the next step is the trainer-side one (a held-out train carve, PI decision).
* **R-D**: A1-GAP RESOLVED-NEGATIVE (val loses LESS than train) => the sign of the O4 story is reversed at R0's state: the in-sample reading is the worse of the two. Reported as such.
* FIDELITY-GAP (control 3) downgrades any statement that leans on `L_tr` to "the harness may be off by more than 0.25 points on the train side".

## 5. Limits stated in advance

* One run (R0, seed 0), no replicate; the val side is 75 logs, the train side 30 logs. The result is a statement about R0 at these states, not about R0d, whose dumps are not on this dev box.
* The unconstrained arm's gap is not the constrained arm's gap (main prereg s.12).
* Four metric families: as the main prereg s.10 (LONGITUDINAL partial; LATERAL, TACTICAL not computable; STRATEGIC PAUSED).
