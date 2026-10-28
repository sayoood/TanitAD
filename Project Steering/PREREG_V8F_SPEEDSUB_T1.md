# PREREG_V8F_SPEEDSUB_T1: does a causal ego read-out in O's five speed cells, at every replan tick, move DAG-OACC at T1-kin?

**Status: PRE-REGISTERED, NOT RUN (2026-10-28).** Promoted by the Master Mind (draft sha256 237240e3…) before any T1-kin substitution rollout and before `mlp_ego_c` is fitted. The instrument (the fit, the `forward_infer` hook, the reader) is NOT BUILT; it is built against this text, and any amendment is appended before the first rollout. No T1-kin substitution number exists. The read-out this PREREG uses, `mlp_ego_c`,
is not yet fitted. Its recipe is fixed here (section 2), and it is used as fitted. Claim row `H-V8F-SPEEDSUB-T1-1` opens on promotion.
Commissioned by `PREREG_V8F_SPEEDSUB` section 6 (outcome PASS, read 2026-10-28 08:05). Base commit read: `ea5e3d3b`.
**Tier of every decisive row: T1-kin (self-action open loop: m = 10, causal bucket, dead-reckoned nav, logged scene, periodic, executor
plan). Never "closed loop", never T2.**

## 0. Why this experiment, and what was seen before it was written

| step | finding | class |
|---|---|---|
| SPEEDSUB | `mlp_ego` in the five speed cells cut M2b-mixo's own-code ADE6 by −0.1290 [−0.2107, −0.0548] (s0) and −0.0686 [−0.1151, −0.0232] (s1), with the bar at −0.06035; the deliberate-regression arm was separated-worse at both seeds; outcome PASS | MEASURED, T1-class open loop, `F:/TanitAD-scratch/speedsub/subread/SUBREAD.json` (RESULT_V8F_SPEEDSUB, staged) |
| DAG-ORC floor | the replicate pair set every crawl bar; REP s0 − CTRL s0 at T1-kin read ADE6 +0.4025, yaw-rate MAE +15.86, curvature MAE +0.1324 (separated-worse); REP s1 ≈ 0 | MEASURED, `RESULT_V8F_DAG_ORC.md` section 4 (c), landed `ea5e3d3b` |
| REP s0's draw | REP s0's T predicts all five speed cells 0.73–1.17 m/s slow against CTRL s0, separated, on the same 2,399 windows; REP s1 and A ≈ 0; REP s0's final along-track bias −3.08 m against −0.75 | MEASURED, T0 diagnostic, `dagland/rep_s0_diag/REP_S0_SPEED_BIAS.md`, `rep_s0_speedcells.json` |
| attribution | on TRAIN fold 0 the read-out's skill survives without `nav` (0.2475 → 0.2631 SD units) and not without `limit` (→ 0.2803); `p10f` + `frame_ok` alone reads 0.3085 and beats `T_own` in 1 of 5 cells | MEASURED, `dagland/speedsub_t1_prep/FINDING_1_ATTRIBUTION.md`, `refit_all.json` sha256 `0f830665…` |
| the lane | the logged `limit` bucket is the bucket of the ego's own realised future max speed (`D-SPEEDBAND-EGO-ECHO-1`; a declared input by PI ruling). The T1-kin lane feeds `--limit causal`: `speed_max_bin(max measured ego speed over rows [j−20, j])` (`stack/scripts/v8f_t1_eval.py:261-272`) | source-read, `FINDING_1` section 3, `FINDING_3_T1KIN_PATH.md` |
| pair-compat | DAG-OACC's checkpoints match the read-out's fit on `kin_mode` yaw, `cons_mu` / `cons_sd` (bitwise), speed-cell code indices [46, 56, 57, 58, 59], grid `pool4x10` and corpus parity | MEASURED, `FINDING_2_PAIR_COMPAT.md` |

So the banked `mlp_ego` cannot be carried into the T1-kin lane as it is: its strongest input changes meaning there. This PREREG refits the
same design on the causal bucket and reads it in the lane where the adopted base is scored. ⚠️ **What was seen before writing:** the SPEEDSUB
outcome (T0-class, a different base), REP s0's T0 speed cells, and TRAIN-fold attribution numbers. No T1-kin substitution has run, and no
eval-141 number of any causal read-out exists.

## 1. Hypotheses

* **H-V8F-SPEEDSUB-T1-1 (the lever).** At T1-kin on the DAG-OACC base, replacing at every replan tick only the five speed cells of the code
  O reads by `mlp_ego_c` (section 2) cuts ADE6 against the same run's `identity` rollout by at least **F/2 = 0.06035 m**, separated, at **each
  of four runs**: CTRL s0, CTRL s1 (DAG-OACC, `67265ec1…` / `90b01474…`) and REP s0, REP s1 (DAG-OACC's argv at trainer seed 10 / 11,
  `b03fd6ef…` / `77853e56…`, banked by DAG-ORC).
* **H-V8F-SPEEDSUB-T1-2 (the floor).** The read-out does not depend on T's training draw, so it shrinks the replicate spread. Read beside the
  table, with its own committed rule (section 6.2).
* **The programme claim it can reach.** If the lever holds, Read A (`PREREG_V8F_DAG_REREAD` section 5: D1, D2, D3 and the crawl rows,
  against P10) is read on the substituted CTRL at both seeds. DOMINANT there is the dominance claim (section 6.1).
* **Why F/2 and four runs.** The contrast `egoc − identity` is on one trained model, so no training-run term sits inside it. The
  training-run question is answered by requiring the effect at four training runs (two seeds × two replicates; `H-ESTIM-SEED-1`), instead of
  by DAG-ORC's floor, which measures the level difference between two trained models. The bar is SPEEDSUB's, unchanged. ESTIMATED prior:
  PASS is not the expected outcome. The causal fit loses the oracle part of `limit`, and the pure-dynamics fit beats `T_own` in only one cell.

## 2. The read-out `mlp_ego_c` (recipe fixed here; fitted once, used as fitted)

* **Design.** The 22-column ego block of `mlp_ego`, in the same order (`p10f` 0:8, `nav` 8:13, `limit` 13:17, `limit_ok` 17, `frame_ok` 18:22;
  `FINDING_1` section 4). Only `limit` and `limit_ok` change: at the tick frame of each TRAIN window they are recomputed with the T1 tool's
  **own** `causal_bucket` (imported from the pinned tool, not re-implemented), from the measured ego speed of that clip's rows [j − 20, j].
  Every other column is read from `F:/TanitAD-scratch/speedread/train/train_arrays.npz` unchanged.
* **Fit.** SPEEDREAD A1's own `standardiser`, `fold_of`, `mlp_fit`, with `mlp_ego`'s `mlp_cfg`, folds, holdout fold, seed and early stopping,
  all read from `fit/fit.json`. There is one fit, no selection among read-outs, and no tuning. Its TRAIN holdout RMSE is reported beside
  and decides nothing.
* **Controls before any rollout** (any failure ⇒ REFUSED):
  * **C-FIT:** the same code path, run on the logged columns, reproduces `mlp_ego`. Holdout predictions are within 1e-5 m/s of the banked
    `readout.pt` (`FINDING_1` measured 9.41e-6 on this path).
  * **C-CAUSAL:** on the 2,399 eval-141 windows at tick 0, the bucket the fit code computes equals, window for window, the `limit` /
    `limit_ok` that `v8f_t1_eval.py build_item` builds under `--limit causal`.
  * **C-COVER:** every TRAIN window has a speed history for the bucket, or is set `limit_ok = 0` exactly as the tool does with no finite
    past. The count is reported.
  * **C-DISJ / C-PARITY:** eval ∩ fit clips = ∅; TRAIN parity 4,512 clips / 306,566 windows, as in `fit.json`.
* **Admissibility.** Every column is a non-visual input T already reads at inference under `--limit causal` (`INFERENCE_INPUT_KEYS`,
  `stack/tanitad/models/v8f_mini.py:137`; nav is an input at training and inference, PI). The causal bucket is a function of measured past ego
  speed (PI ruling D-PI-YAWRATE-1 gives T measured ego motion). The cells are speed cells, not a goal, and no situation-classifier output is
  read. At ticks 1 and 2 the read-out reads the model's own executed motion, outside the fit's distribution by construction (disclosed;
  that is what a deployed sub-head would read).

## 3. Arms (dev-box CPU; `v8f_t1_eval.py run` at the pinned tool, sha256 `61e140ed…`, with DAG-OACC's banked argv; OMP / MKL 8, one rollout at a time)

Per run X ∈ {CTRL s0, CTRL s1, REP s0, REP s1}:

| mode | the five cells O reads at every tick | role |
|---|---|---|
| `identity` | X's own, through the hook | the paired base of every contrast; control K-ID |
| **`egoc`** | `mlp_ego_c(ego block of the tick item)`, standardised by X's `cons_mu` / `cons_sd`, at code indices [46, 56, 57, 58, 59] | **the arm** |
| `egoshufc` | `egoc`'s recorded cells for the same tick, permuted across the 2,399 windows by `numpy.random.default_rng(20261029).permutation(2399)` | deliberate-regression arm (a control with a committed reading) |

Beside (CTRL only, lowest priority, decides nothing): `identity_l` and `egol` in the `--limit logged` lane with the banked `mlp_ego`
(`a205609a…`). It bridges back to SPEEDSUB under a **declared open channel** (`D-SPEEDBAND-EGO-ECHO-1`) and is never a capability row.

* **Hook.** A wrapper patches `V8FMini.forward_infer` at class level and calls the pinned tool's `main()` (`FINDING_3` section 4, seam B). It
  computes T's own code, substitutes the five cells, and runs O on the new code as `forward_diag` does. Every other code dimension O reads
  stays T's own. The P10 floor arm never calls the model and cannot be touched.
* **Rows and estimator.** The reader's rows are DAG-ORC's 19 Read-A rows (`RESULT_V8F_DAG_ORC.md` section 2.1) plus the four families
  (section 7). Each contrast `mode − identity` and `mode − P10` uses `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000, seed 0, over
  141 episodes / 2,399 windows, per run. That bootstrap answers the EPISODE question. The four runs answer the TRAINING-RUN question. The
  planner is deterministic, so there is no inference-seed term.

## 4. Controls (each must read its known value before any row is read; any failure ⇒ REFUSED)

| id | must read | catches |
|---|---|---|
| K-ID | `identity` per-window ADE6 equals X's banked own rollout (CTRL: `v8f-dagoacc-t1/`; REP: `F:/TanitAD-scratch/dagorc_t1_out/`) within 1e-4 m, max reported | hook drift, a thread-count or tool-tree change |
| K-ZERO | the reader's `identity − identity` reads exactly 0.0 on every row | a reader that manufactures differences |
| K-NONSPEED | at every (window, tick), the 55 non-speed code dimensions O reads equal T's own, max \|Δ\| = 0.0 | a wider substitution than declared |
| K-REC | the cells O reads in `egoc` equal `mlp_ego_c` on the recorded ego block, standardised, ≤ 1e-5 | a substitution that is not the read-out |
| K-REACH | `egoc` moves the cells away from T's own on ≥ 99 % of (window, tick); `egoshufc` differs from `egoc` on ≥ 95 % | a substitution that never reaches O |
| K-SHUF | the permutation sha is the registered one; fixed points ≤ max(3, 1 %) | an unregistered shuffle |
| K-FLOOR | the P10 floor arm of every run equals the banked floor arm bit for bit | a patch reaching the floor |
| K-SHA | the checkpoints, `mlp_ego_c` (sha pinned at fit time, before any rollout), `readout.pt`, the tool and the wrapper equal their pins | a swapped file |
| K-REG | `egoshufc − identity` ADE6 is separated-worse at all four runs | an instrument that cannot see a harm |

A real-window smoke (`--max-windows 200`, CTRL s0, `identity` and `egoc`) runs first and must pass K-NONSPEED, K-REC and K-REACH before any
full rollout. The full chain stops after CTRL s0 if K-ID, K-REC or K-REACH fail.

## 5. Guards

* **G-WORSE:** no Read-A row (ADE6 included) has `egoc − identity` separated-worse at ≥ 3 of the 4 runs. TAC-LAT and TAC-LON are among the
  rows.
* **G-NEW:** no row r for which `identity − P10` is not separated-worse at a CTRL seed becomes separated-worse for `egoc − P10` at both CTRL
  seeds.
* **G-ADE:** `egoc − identity` ADE6 is not separated-worse at any run.

## 6. The decision (both outcomes committed now; REFUSED first, then HARMS, then PASS)

`d_X` = ADE6(`egoc`) − ADE6(`identity`) at run X. A **hit** at X means separated-better (CI upper end < 0) **and** `d_X ≤ −0.06035`.

| outcome | condition | committed next |
|---|---|---|
| REFUSED | a control fails, or a run is missing | fix the control; no row is read |
| **HARMS** | a guard fires (G-ADE, G-WORSE or G-NEW) | not adopted. Next: section 8 (b), the per-cell attribution at T1-kin, which finds the cell that harms |
| **PASS** | no guard; a hit at **all four** runs | adopt the read-out as the DAG-OACC base's speed path. Build it as a `forward_infer` speed sub-head (buffers absent by default, every earlier checkpoint bit-identical; its own launch-gate checks) and read section 6.1 |
| **PARTIAL** | no guard; hits at both CTRL seeds, not at all four runs | not adopted, and not called a lever effect (`H-ESTIM-SEED-1`). Next: section 8 (a) on the runs that missed |
| **NULL** | anything else | next: section 8 (a), the history-aware read-out |

PASS and HARMS cannot both hold. The labels NULL-ONE (hits at 1–3 runs, not both CTRL seeds) and NULL-FLAT (no hit) describe and decide
nothing.

### 6.1 On PASS: Read A on the substituted CTRL (the dominance read; the same rollouts, no new run)

Read A's own table (`PREREG_V8F_DAG_REREAD` section 5), applied to `egoc − P10` at both CTRL seeds:
* D1: ADE6 separated-better, ≤ −2F.
* D2: no row separated-worse.
* D3: every family has a separated-better row.
* The crawl rows (`RESULT_V8F_DAG_OACC.md` section 1, Amendment 1).

* **DOMINANT at both seeds ⇒** the dominance claim at T1-kin. Next: re-render the evidence videos on it, report to the PI, and ask for
  compute to scale (the PI's standing instruction of 2026-10-25).
* **Otherwise ⇒** the substituted base is the base, and the open rows' lever is next. On crawl rows that is LAUNCH-INTENT, re-based on the
  substituted base by its own amendment.

### 6.2 H-V8F-SPEEDSUB-T1-2, the floor rule (beside; it decides which floors later arms use)

Let `g_id(r)` = point of REP s0^identity − CTRL s0^identity on row r, and `g_c(r)` = point of REP s0^egoc − CTRL s0^egoc, both paired on the
same windows. Rows: ADE6, yaw-rate MAE, curvature MAE.
* **COLLAPSE** when `g_id(r) > 0` and `g_c(r) ≤ 0.5 · g_id(r)` on all three rows. Later arms on the substituted base then take RF from the
  `egoc` replicate pair.
* **NOT COLLAPSE** otherwise. The floors stay as DAG-ORC's.
* The s1 pair is reported the same way. It cannot collapse what is not there (REP s1 − CTRL s1 ≈ 0).

## 7. The four families (per run, per family, never pooled; STRATEGIC is reported, not paused, for v8F)

| family | rows (T1-kin; `egoc − identity` and `egoc − P10`) | not computable |
|---|---|---|
| **LONGITUDINAL** | speed MAE, along-track MAE, accel MAE; DK-VIOL / DK-HW / DK-TG on LEAD windows | none (the DAG-ORC reader computes DK at T1-kin); the anti-echo block is reported if the reader carries it, else NOT BUILT with the reason |
| **LATERAL** | heading, yaw-rate, curvature, cross-track, L1 lateral-accel, L2 (GT ≥ 5.2 m/s) | none |
| **TACTICAL** | TAC-LAT, TAC-LON (T head at ticks ≥ 1) | goal-head F1 at T1-kin (the paired tool reads eval dumps, not tick dumps), with its n |
| **STRATEGIC** | S-POS@30 / 50 / 80 against CTRV, per run | paired S-POS across arms is not computed |

## 8. HARMS, PARTIAL or NULL: the next lever, named now (Rule Zero)

* **(a) The history-aware causal read-out.** The tick-frame design discards the four-frame history that `frame_ok` only masks. The same MLP is
  fitted on the four history frames of `p10f` plus the causal bucket and nav (an extract over `train_arrays`' window index; ESTIMATED under
  1 CPU-h), under its own amendment with the same table. Its reason: on the pure-dynamics fit the per-tick information is what loses to `T_own`.
* **(b) The per-cell attribution at T1-kin** (SPEEDCELL's design in this lane). Each of the five cells is substituted alone, at CTRL s0 / s1:
  10 rollouts, ESTIMATED about 1 h.
* Neither needs box 2 or a GPU.

## 9. Run (dev-box CPU; `CUDA_VISIBLE_DEVICES=-1`; every stage longer than 4 min DETACHED and handed off by file)

The order:
1. Fit `mlp_ego_c` and its controls (C-FIT, C-CAUSAL, C-COVER, C-DISJ, C-PARITY). ESTIMATED under 15 min.
2. Pin its sha.
3. The smoke.
4. CTRL s0 (`identity`, `egoc`, `egoshufc`), then the preflight stop.
5. CTRL s1, REP s0, REP s1.
6. The reader.
7. The beside logged lane.

Twelve decisive rollouts at about 305 s (MEASURED for DAG-OACC, `FINDING_3` section 2) ≈ 1.0 h at 8 threads, plus 4 beside. The outputs go to
`F:/TanitAD-scratch/speedsub_t1/`, because E: has about 2 GB free.

## 10. What this does not license, and disclosed limits

* ⛔ No bar, permutation, tolerance, row set or table entry moves after any rollout. An amendment after promotion is appended with its date
  and reason, and the body stays as promoted.
* ⛔ The read-out is never fitted, tuned or selected on eval-141. Its design is fixed above, before it is fitted.
* ⛔ T1-kin is a self-action open loop. The lead follows its logged future and the cameras are logged, so this is not T2 and not
  closed-loop driving.
* **Limits.**
  1. There is one read-out fit. "Would another fit say this?" is not answered.
  2. The ticks 1–2 inputs are out of distribution.
  3. `egoshufc` is a global shuffle, as in SPEEDSUB.
  4. REP s0 is a slow-T draw (section 0). A hit there partly measures repair of that draw. The decision therefore needs both CTRL seeds as
     well.
  5. The four runs are two seeds × two replicates of one recipe, not four recipes.
* Amendments: none.
