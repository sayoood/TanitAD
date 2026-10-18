# PREREG SOCARVE: would a carve-fed dual have SEEN the R0d breach? The train-vs-carve-vs-val gap of the SAFEOBJ constraint quantity, on banked R0 states (A2 harness)

Written 2026-10-18 04:47Z by the Research Lab agent (Master Mind brief 2026-10-18), BEFORE any per-set array, per-pool TTC value or carve statistic of this package was computed or read. Its sha256 is in `PREREG_SOCARVE.sha256` (hash of these exact bytes).
It executes the 0-A40 first step committed by `PREREG_SAFEOBJ.md` (sha256 `b404c2fb7b2f724775a1dda5a9de06c9778505a3374e6467060ae7c7a9e04bcb`) outcome O4, row "dual fed from a log-disjoint carve of RL-train (never RL-val). First, 0 A40: measure the train-vs-carve gap on banked R0 states with the A2 harness".
PK = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`.

## 0. Disclosure: what I had seen before this freeze (so the freeze is not oversold as blind)

| seen | where | effect on this prereg |
|---|---|---|
| the R0d outcome (O4, in-sample g -0.00034, RL-val TTC -0.77 [-1.49, -0.06]) | `PK/raw/2026-10-18-r0d-outcome/RESULT.md` | motivation only |
| pooled R0 FULL P2 TTC B-A -1.53 / -1.25 / -1.07 and pooled P1 TTC -2.20 / -2.47 / -2.47 (x100, u250 / u500 / u1000, 364 sets) | `PK/raw/2026-10-16-safety-head/RESULT.md` s.1.1 | the harness control compares to these |
| **val minus train P2 TTC on R0's states: -0.21 (u500) and -0.35 (u1000)** | `PREREG_SAFEOBJ.md` line 116 | **the SIGN and approximate size of G_tv at u500 and u1000 were known before this freeze.** The ratio's denominator is not blind. The numerator G_tc is not known to me: no per-pool value, no per-log value, no carve draw has been seen. |
| pool composition: train pool 302 sets / 30 logs, val pool 62 sets / 13 logs, RL-train 16,000 sets / 166 logs (sets per log 2 / median 52 / max 1,080) | counts only, from the pickled pools and the extracted RL-train sets | design (carve size) |
| the SELTTC result (CONTROLS_FAILED; diagnostic S1 -1.44 / -1.28 / -1.13) | `PK/raw/2026-10-16-safety-head/RESULT.md` s.10 | Rule Zero section 9 |

Nothing else of this package's outcome exists yet. A dry import of the harness (80 s setup, 2.0 GB RSS, pools 302/30 and 62/13) read no TTC value.

## 1. Carve provenance, established FROM SOURCE before any outcome number

The question "is there a free carve (RL-train logs the policy never trained on)?" is answered by the trainer, not by the pool.

| fact | evidence (MEASURED, by reading the source) |
|---|---|
| R0's trainer blob | commit `a120418a870f94f7f769b24a593ea0c91d131c3d`, `refe/scorer_finetune.py` blob `fec6174de90d7f428579d9dc2ac85889170096b0` (sha256 `b777d7292fb9973b...`), copy `src/scorer_finetune_a120418.py` |
| the training index | `tr_idx = [i for i, r in enumerate(ds.rows) if key(r) in op.by and r.get("log_name") not in ho_logs and (not a.require_pdm or key(r) in op.pdm_keys)]` (:496-497), then restricted to sets with an RL terms record (:509). So RL-train = EVERY on-policy set of the RL-train bank whose log is not an RL-val log. No other filter, no sub-sampling, no hold-out of train logs. R0's `sft.jsonl` data event: `train_sets` 16,000 = `train_sets_in_bank` 16,000; `train_rows_in_heldout_logs` 0 (RL-train and RL-val are log-disjoint, gate G7) |
| the sampler | `g = torch.Generator().manual_seed(a.seed)` (:531); `DataLoader(make_train_set(...), batch_size=a.batch, shuffle=True, generator=g, ..., drop_last=True, ...)` (:532-534). R0 argv (PASS token `PASS_a120418a870f.json`): `--seed 0`, `--batch` default 8, `--accum` default 8, `--epochs 4`, `--max-updates 1000` |
| the loop | `total = ceil(len(tr_idx) / a.batch * a.epochs / a.accum)` (:927), capped by `--max-updates` (:928-929); `while not done: for batch in dl:` (:937-938): each pass over `dl` is ONE fresh uniform permutation WITHOUT replacement of all 16,000 sets (2,000 micro-batches; 16,000 % 8 = 0 so `drop_last` drops nothing); an optimiser update = `a.accum` = 8 micro-batches (:946-947) |
| the arithmetic | 1,000 updates x 8 micro-batches x 8 sets = 64,000 draws = **exactly 4 epochs**. u250 = 2,000 micro-batches = **exactly 1 epoch** |
| R0d | the tip trainer (`ac07b1a`, blob `e063928807b7e0df79b1c1ddfe192596ca1ff601`, sha256 `f5b46136910cd015...`) carries the same lines unchanged (:908, :931, :965-967, :1638, :1652-1653); R0d's argv is R0's plus the five `--epo-con*` flags, seed 0 (SPEC_SAFEOBJ s.3). So R0d consumed the same data stream. Inferred from identical code and seed; the batch order was not logged |

**Conclusion (committed before the numbers): R0 consumed EVERY RL-train set, hence every RL-train log, once by u250 and four times by u1000. There is no free carve. A "log-disjoint carve of RL-train" cut out of R0's data is NOT held out from the policy: R0's scorer B (and A) were trained on it.**

Flag **F1 IN-SAMPLE-CARVE** is therefore raised on every outcome of this package, by construction.

### 1.1 The narrower question this measurement answers (named)

The brief's question is "would a carve-fed dual have SEEN the breach?". For a carve that is held out from the policy, the R0 states cannot answer it, because no R0 state has a held-out train log. What the R0 states CAN answer, and what this package answers, is:

> **N1.** At the R0 u1000 state, how large is the train-vs-val gap in the constraint quantity g (resolved against its estimator), and does a log-disjoint subset of the logs the policy trained on (an in-sample carve) read like the rest of the train logs or like val? I.e. how much of the gap can *log composition alone* produce?
> **N2 (secondary).** Is the gap a property of *exposure* (it grows with passes over the data) or of *composition* (already present after one pass)? (S-DOSE)
> **N3 (secondary).** Do the train-pool and val-pool populations differ on quantities the RL training cannot have touched? That is the necessary condition for a HELD-OUT train carve to be val-like. (S-EXCH)

Under exchangeable logs, E[G_tc] = 0 for an in-sample random carve, so CARVE-TRACKS-TRAIN is the **expected** label of N1 by construction. Section 7 commits how that label may and may not be read.

## 2. The constraint quantity (what the dual sees), with file:line

SPEC_SAFEOBJ s.1.2 (`PK/raw/2026-10-17-safety-objective/SPEC_SAFEOBJ.md`, sha256 `54151735997b06e4...`): `c^k_X = (1/b) sum_s sum_i pi^X_{s,i} y^k_{s,i}`, `g^k = c^k_A - eps - c^k_B` (> 0 = violated), `eps = 0.0`, k in {TTC, NC}. Trainer: `epo_con_term` at `scorer_finetune.py:256-287` (tip blob): `piB = epo_policy(sxB)`, `piA = epo_policy(sxA_ref)` (:268-269), `cA_s, cB_s = (piA * yk).sum(1), (piB * yk).sum(1)` (:274), `g = cA_s.mean() - eps - cB_s.mean()` (:275); the dual step reads the mean of g over the update's micro-batches (`EpoDual.step`, :308-317, the update at :316). `pi = softmax(log(sel_score(sigmoid(sx)) + 1e-6) / 0.1)` over the 64 proposals (`epo_policy`, :193).

The harness equivalent is `Evaluator.evaluate`'s `pi_comp` (`phead_common.py`, `out["pi_comp"].append((pi[..., None] * tg6).sum(1))`) = `sum_i pi_i y_i` per set, column TTC = 3, NC = 0 (`sh_summarize.REPORT`), and `eval_views`' identical arithmetic (`sh_common.py`). So **the quantity read here is g itself**, not the pick delta:

* per-set `d_s = 100 * (pi_comp_B[s, 3] - pi_comp_A[s, 3])` (= the harness's `P2 TTC`, x100);
* for a set of logs S: `D(S) = mean_{s in S} d_s` (sets-weighted, as the dual's batch mean is);
* **`L(S) = -D(S) = 100 * g^TTC(S)` (eps = 0), in points. L > 0 = a violation in the dual's sense; the RL-val bar S is L > 0.5.**
* The dual's constraint is on this soft expectation, E_pi[TTC]. The RL-val bar reads the argmax PICK's TTC (`rl_val_eval`; trainer logs it as `p1gap_ttc`, :279-284). **P1** (`pick_comp[:, 3]`, same sign and scale) is read as a registered secondary quantity (section 8).

## 3. Data, estimator, tier

* **States** (banked, dev box): `C:/Users/Admin/rls1_adam_dl/rls1-r0/scorer_states/scorer_states_u{250,500,1000}.pt` (R0, A and B; sha256 in `rls1_adam_dl/inputs_sha256.json`) and `rls1-c0/.../scorer_states_u1000.pt` (C0, the floor). R0d scorer states are **not on this dev box** (searched `rls1_adam_dl`, `rls1b_dl`, `r0d_out`) and are not used. R0b is a different lever (`--epo-lam 0.25`), not read here.
* **Pools**: the A2 harness's cached-context pools, `phead/pool_train.pkl` (RL-train: **302 sets / 30 logs**, 1.9 % of the 16,000 sets, 18 % of the 166 logs) and `phead/pool_val.pkl` (RL-val: **62 sets / 13 logs**, 2.1 % of 3,000 sets, 17 % of 75 logs). Coverage is set by the visual-context cache; whether that coverage is random with respect to log properties is UNVERIFIED.
* **Harness**: A2's `sh_common.eval_views` + `sh_summarize.prox`, unchanged, imported from the dev-box archive of the safety-head package. File sha256s (identical to the blobs at `ac07b1a`): `sh_common.py` `9b3c8f0a92f13082...`, `sh_summarize.py` `3b33873e8bf5b98b...`, `sh_realstates.py` `8bdfb9d49846d219...`, `phead_common.py` `eea9ac0d210f6d3f...`, `adam_common.py` `6523ce50ca42e061...`, `adam_sim.py` `adcbfdc2270b5e1a...`. CPU fp32, 4 threads (A2's default), cached visual contexts.
* **Evidence class: ESTIMATED.** **Tier: offline real-state selection read on cached contexts, NOT T1, not a driving claim** (the same tier as A2 and SELTTC). This is not the trainer's live forward (bf16, live trunk): A2 reproduced the live arms' sign and size (phead control (d)), not their bytes.
* **Estimator** (named): paired log-cluster bootstrap, 10,000 resamples, 95 % percentile interval, seed 20260927; the cluster unit is the LOG; the training state is FIXED. It answers "another draw of LOGS". It does NOT answer "another training run" (one run per arm, no replicate) and there is no inference randomness (softmax / argmax are deterministic). The seed-floor of the arm is therefore not measured here.

## 4. Harness control (committed; INVALID if it fails)

On the **val pool** (and the train pool, both are in the banked block) my stage 1 must reproduce the banked A2 `realstates.json` (`PK/raw/2026-10-16-safety-head/realstates.json`, sha256 `84b2f1df72840e17...`), `states.r0_u{250,500,1000}.FULL`:

* for each of `P1_{TTC,NC,DAC,EP}`, `P2_{TTC,NC,DAC,EP}`: `pooled`, `ci[0]`, `ci[1]`, `train`, `val`; and `P1_PDMS`: `pooled`, `ci[0]`, `ci[1]`;
* tolerance: |mine - banked| <= 1e-4 (points, x100) on every mean (`pooled`, `train`, `val`) and <= 1e-3 on every CI endpoint;
* additionally stage 2's recomputed per-pool means from the SAVED per-set arrays must equal the banked `train` / `val` of `P2_TTC` and `P1_TTC` at each R0 state, same tolerance (`L(Tr) = -banked.train`, `L(Va) = -banked.val`);
* **C0 floor**: at C0 u1000 the FULL `P2_TTC` pooled must satisfy |.| < 0.5 (A2's own floor rule).

**If any of these fails: STOP. The read is INVALID, no outcome is drawn, and the report says INVALID with the failing key.**

## 5. Estimands and the carve (all committed)

Notation: Tr = the 30 train-pool logs (302 sets), Va = the 13 val-pool logs (62 sets). Unit of every statistic: points (x100). State: **R0 u1000** for the decision; u250 / u500 descriptive.

* `L_tr = L(Tr)` : "the logs the trainer actually consumed" (all of them, section 1).
* `L_va = L(Va)`.
* **Carve** `C_r` = a uniformly random subset of `m` of the 30 train-pool logs, drawn without replacement; the remaining `30 - m` logs are the "rest". **Primary m = 13** (the val pool's log count, so carve and val have equal log-count sampling noise). Sensitivity m in {6, 20}, descriptive only. Logs are sorted by name; `rng = np.random.default_rng(20260927)`; the r-th draw is `rng.choice(30, size=m, replace=False)`; **R = 10,000 draws** (a fresh `default_rng(20260927)` per m).
* `L_c = mean_r L(C_r)` (the carve's point reading).
* **`G_tv = L_va - L_tr`** (the train-to-val gap the carve would have to close); **`G_tc = L_c - L_tr`** (how far the carve moves away from the in-sample reading). Sign convention: a positive gap = val (or the carve) loses MORE TTC than the in-sample reading. `rho = G_tc / G_tv`.
* **Intervals** (bootstrap, 10,000, seed 20260927): `rng = np.random.default_rng(20260927)`; for each resample b: `ti = rng.integers(0, 30, 30)` (train logs with replacement), `vi = rng.integers(0, 13, 13)` (val logs with replacement), `pos = rng.permutation(30)[:13]` (the carve = the resampled train positions `ti[pos]`); compute `L_tr_b, L_va_b, L_c_b`, hence `G_tv_b`, `G_tc_b`. Report the 2.5 / 97.5 percentiles of `L_tr, L_va, L_c, G_tv, G_tc`. "Within val's CI" means `L_c` lies in the percentile interval of `L_va_b`.
* Sensitivity (descriptive): the "in-sample" reading taken as the REST of the carve, `L(rest_r)`, instead of `L_tr`.

## 6. Why the controls below exist

The stage-2 code (carve draws, bootstrap, labelling) is new. A same-breath zero control and a pair of label controls, run on the real arrays and on planted arrays, prove it can read 0, can read a gap, and labels both directions correctly. See section 9.

## 7. Outcomes, committed in advance (quantity: P2 TTC at R0 u1000 = g; points)

* **CARVE-TRACKS-VAL**: `G_tv > 0` AND `rho >= 0.5` AND `L_c` within val's CI. => The carve reads like val: it would be a valid dual feed, and the R0e arm (dual fed from the carve) is worth proposing to the PI.
* **CARVE-TRACKS-TRAIN**: `G_tv > 0` AND `rho < 0.25`. => A carve of RL-train reads like the in-sample logs: it cannot see the breach on these states; the O4 lever is predicted inert; name the next lever (it must never feed RL-val into training; candidates: a tightened in-sample threshold derived from training data only, or the SELTTC deployed-selection lever).
* **INTERMEDIATE**: anything else, with its sub-reason stated: `NO-GAP` (`G_tv <= 0`: val is not worse than train, the question is moot), `MID` (`0.25 <= rho < 0.5`), `OUT-OF-CI` (`rho >= 0.5` but `L_c` outside val's CI).

**Flags, committed with the outcome (they qualify the READING, they do not change the label):**

* **F1 IN-SAMPLE-CARVE** (raised always, section 1). With F1 raised, CARVE-TRACKS-TRAIN is the label expected by construction under exchangeable logs. **It does NOT license "a HELD-OUT carve cannot see the breach" and does NOT license "the O4 lever is inert".** It licenses exactly: "at the R0 u1000 state, log composition within RL-train does not by itself produce val-like loss at this n". A CARVE-TRACKS-VAL with F1 raised would be a finding about log heterogeneity (the carve's logs differ systematically from the rest), not about hold-out.
* **F2 UNRESOLVED-GAP**: the 95 % interval of `G_tv` contains 0. Then `rho` is a ratio to a gap that is not distinguishable from zero; the label is reported as `<LABEL> [F2]` and the next-lever statement is "undetermined on these states", not "predicted inert".

## 8. Secondary registered readings (descriptive; none of them decides the outcome)

* **S-DOSE (N2)**: `G_tv(u)` with its interval at u250 (1 epoch), u500 (2), u1000 (4). Tag from `r_d = G_tv(u250) / G_tv(u1000)` when both are positive: `r_d < 0.25` EXPOSURE-GROWN; `0.25 <= r_d < 0.75` MIXED; `r_d >= 0.75` PRESENT-AT-ONE-PASS; undefined otherwise. Note u250 sits just after the 200-update lambda_pg ramp.
* **S-EXCH (N3)**: per set, from the frozen BASE scorer's logits (`sx0`) and the labels: `c0_TTC`, `c0_NC`, `c0_EP` (`pi0 . y`), the label-only means `mean_i y^TTC`, `mean_i y^NC`, and the base pick's PDMS. Differences train-pool minus val-pool with independent log-cluster bootstrap intervals (10,000, seed 20260927). Tag EXCH-NOT-CONTRADICTED iff every interval contains 0, else EXCH-CONTRADICTED with the covariates named. Limitation stated now: the base scorer's own SFT exposure to RL-train versus RL-val logs is UNVERIFIED, so only the label-only means are exposure-free in the strict sense.
* **S-POWER**: over the R carve draws (m = 13 and m = 6): the standard deviation of `L(C_r)`, the share with `L(C_r) > 0` (the dual would see a violation), and the share with `L(C_r) >= 0.5` (the RL-val bar magnitude).
* **Quantity variants, same rules, reported not deciding**: P1 TTC (the pick, the bar's quantity), P2 NC, P1 NC. Longitudinal context: P2 and P1 EP (the progress gain that the trade buys).
* **Per-log table** (descriptive): the 30 train logs and 13 val logs with `n_sets` and `L`.
* **RZ-1, R0d's own in-sample log (MEASURED from the arm's `sft.jsonl`, `C:/Users/Admin/r0d_out/raw/sft.jsonl`)**: over the SAFEOBJ addendum window, updates 651..750 (the four logged rows at 675 / 700 / 725 / 750): the mean of `con_g_ttc` (x100), of `con_p1_gap_ttc` (x100, the B pick minus A pick TTC, negative = B picks worse) and of `con_cA_ttc`, `con_cB_ttc`. Tag **P1-IN-SAMPLE-HELD** iff the window mean of `con_p1_gap_ttc` x100 is > -0.5, else **P1-IN-SAMPLE-BREACHED**. This is the cheapest test of an alternative to the train-to-val story: that the dual constrained E_pi[TTC] while the bar reads the PICK. It is read after this freeze.

## 9. Controls on the stage-2 code (all must pass before any outcome is read)

1. **Zero control**: feed A's arrays as both arms (B := A) through the full stage-2 pipeline (the point draws and the bootstrap). Every `L`, `G` and interval endpoint must be exactly 0.0 (`np.array_equal`).
2. **Label controls on planted arrays** (independent of the data): train deltas planted at -0.2, val deltas at -1.0 (so `G_tv = +0.8`): carve := val logs => CARVE-TRACKS-VAL; carve := train logs => CARVE-TRACKS-TRAIN; a carve planted halfway (a 50/50 mix) => INTERMEDIATE; `G_tv` planted at 0 => INTERMEDIATE(NO-GAP).
3. **Real-array label controls**: with the real R0 u1000 arrays, replace the carve by the val logs (the oracle carve): `L_c = L_va`, `rho = 1`; replace it by the whole train pool: `rho = 0`. The labelling function must return the matching labels (when `G_tv > 0`).
4. **Sign control**: swapping B and A flips the sign of every `L` and `G` exactly.
5. Harness control of section 4 and the C0 floor.

## 10. The four metric families, as far as these states allow

* **LONGITUDINAL**: partial. TTC (P2 = g, P1 = pick) and EP (the progress gain) from the scorer's sub-score logits against PDM labels. Target-speed accuracy and distance-keeping (headway, time-gap, TTC to the lead agent): NOT COMPUTABLE here (a scorer-only state on fixed proposals; no plan-vs-reference trace).
* **LATERAL**: NOT COMPUTABLE (n = 0): heading, curvature, yaw-rate and cross-track need executed trajectories; the scorer selects among fixed proposals. NC and DAC are collision / drivable-area sub-scores, not lateral-family metrics.
* **TACTICAL**: NOT COMPUTABLE (n = 0): no manoeuvre-decision instrument at this tier.
* **STRATEGIC**: **PAUSED** by the PI ruling of 2026-09-19 (`PI_DECISION_QUEUE.md` item 25). Reported as paused: not a pass, not omitted.
* Per family, never pooled. No composite is reported.

## 11. Rule Zero: the next lever, committed before the numbers

* If the label is CARVE-TRACKS-TRAIN or INTERMEDIATE (the expected case), the turn is not finished at the label. The cheapest zero-training lever the brief names, the SELTTC-style TTC-heavier DEPLOYED selection, **has already been run on these exact R0 states** (`PK/raw/2026-10-16-safety-head/selttc.json`, RESULT s.10): reach control failed (S3 changed 0 picks, CONTROLS_FAILED), and the diagnostic S1 / S2 moves the real TTC loss by at most +0.09 (about 6 %). I will report it as already measured, ESTIMATED, not re-run.
* The new cheap discriminating reading is **RZ-1** (section 8), which costs 0 A40 and 0 compute: whether the dual's quantity (E_pi[TTC]) and the bar's quantity (the pick) differ in-sample, using the arm's own log and the harness's P1 / P2 on the same states. If P1-IN-SAMPLE-BREACHED, the next lever is NOT a carve but a pick-aligned constraint (for example the constraint evaluated at a lower policy temperature, or a hardmax surrogate), and that is named as the O4 successor. If P1-IN-SAMPLE-HELD, the train-to-val story stands, and the next lever is the held-out carve arm itself (R0e), which needs a PI decision (A40 spend) and whose reach cannot be tested on R0 states.
* Nothing here is a lever-effect claim (single seed per arm, no replicate; H-ESTIM-SEED-1).

## 12. What this package cannot show (stated in advance)

* Whether a HELD-OUT carve tracks val: no R0 state has a held-out train log (section 1). Only an R0e run can test it.
* The generalisation gap of a CONSTRAINED policy: R0 is the unconstrained arm, so its train-pool L is not pushed to 0. R0d's in-sample g was driven to about 0 by the dual; its val-versus-train gap is a different quantity from R0's.
* Anything about the arm's seed floor, or about the live bf16 forward.
* Population claims beyond 30 train and 13 val logs (the cache-covered subset).

## 13. Execution plan (committed)

Stage 1 `so_eval_states.py` (heavy; detached if it exceeds ~8 min): R0 u250 / u500 / u1000 and C0 u1000, views own (B), A_own (A), base; saves per-set arrays to `out/persets_<run>_u<u>.npz` and the section-4 control to `harness_control.json`. Stage 2 `so_carve_analysis.py` (light, numpy): sections 5, 7, 8, 9 on the saved arrays -> `socarve.json`. Everything stays under `C:/Users/Admin/socarve/`; no commit, no push, CPU only, the pod is not touched.
