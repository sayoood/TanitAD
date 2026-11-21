# PREREG P5-R: regularised tracker inversion as the label lever behind P5-C's TB-6 (written 2026-11-12, BEFORE any P5-R code or number exists)

**This is the "own §9 entry" the P5 card commits for P5-R** (`PK/eval/PREREG_P5_TRACKER_AWARE_TARGETS.md` §9, Amendment
P5-S, "Both FAIL" branch, read at `c335b2fe`): *"P5-R, a regularised inversion: solve for r20 minimising the 8-pose
tracking error plus λ·‖Δ²r20‖², rather than interpolating the error; it gets its own §9 entry before any code."*
It lives in the P5-C hold (`SG/p5r/`) because its trigger is P5-C's table gate TB-6 (`SG/p5c/PREREG_P5C.md` §4.2,
FROZEN sha256 `49338ab8…`). It is frozen by `PREREG_P5R.sha256`; after the freeze it is never edited, and any change
goes into a separate, dated `PREREG_P5R_ADDENDUM_<n>.md` written **before** the data it affects is read.

Abbreviations: `PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`;
`SG` = `E:/Projects/TanitAD-artifacts/scorer_gym_20261107`; `ND` = `PK/refe/navsim_dac.py`;
`P` = `PK/eval/p5_target_labels.py` (blob `294644df`, imported unchanged, never modified).

**Status at writing:** no P5-R module, test, run, timing or number exists. Nothing below was chosen after seeing P5-R
output. The inputs in §0 were read before writing.

---

## 0 · Inputs (read before writing; inputs, not bars)

| fact | value | class |
|---|---|---|
| TB-6 on the 63,495 dev-box-labelled rows (Tc20s vs `traj`, ratio of medians) | rank 0: acc **1.681**, heading rate **1.939** (PASS, thin); rank 1: 1.651 / 1.729 | MEASURED, `F:/tanitad_scratch/p5c_tb6/out/gates.json` |
| rank-0 raw median max heading rate, labelled rows | 4.937 °/s (label 9.576 °/s) | MEASURED, same file |
| rank-0 raw median max heading rate, unlabelled rows / full bank | 3.56 / 4.05 °/s | INHERITED (MM, 2026-11-12), re-measured by this run (§5.3) |
| full-bank rank-0 TB-6 heading-rate ratio, projected | ≈ 2.09 ⇒ likely FAIL | ESTIMATED (MM reweighting), INHERITED |
| Tc20s tracking repair, P5-S held-out (1,018 rows) | median 8-knot max-xy 0.414 m (T) → 0.057 m (Tc20s); Δ max-xy −0.400 m [−0.459, −0.339]; Δ sim8 −7.07 pp [−13.26, −1.83] | MEASURED, `PK/raw/2026-10-11-p5s-smoothed/p5s_cubic.json`, `PK/eval/RESULT_P5S.md` |
| Tc20s interp8 cost, P5-S held-out | interp8 fail 13.06 % vs T 0.59 % | MEASURED, `RESULT_P5S.md` §3 |
| Tc20s sim8 repair on dev-box bank rows | rank 0: 3.67 % → 0.67 %, +3.00 pp [1.06, 5.28]; rank 1: 5.33 % → 0.67 %, +4.67 pp [2.33, 7.21] (n 300 each; seed 20261111) | MEASURED, `SG/p5c/labels/RANK1_REPAIR.md` |

The failure the MM projects is **denominator-driven**: on near-straight rows Tc20s adds a heading-rate pre-distortion
out of proportion to the tiny raw value. P5-R attacks the numerator: it penalises the roughness of the pre-distortion
itself, while still minimising the tracker's error.

## 1 · The question

Can a label that is the **minimiser** of (NAVSIM tracking error at the 8 NAVSIM knots) + λ·(roughness of the
pre-distortion) keep Tc20s's tracker repair (sim8, 8-knot error) **within a committed non-inferiority margin** while
bringing TB-6 to **≤ 2× on rank 0 with headroom that survives the shift to the full bank's straighter rows**?

## 2 · The solver (committed)

### 2.1 Objective

Notation (all ego frame, rear axle, heading in rad):
- `traj` ∈ R^{20×3}: the bank row's raw teacher trajectory at t = 0.2 … 4.0 s.
- `T = ND.to_navsim(traj)` ∈ R^{8×3}: the intended 8 NAVSIM poses. This is exactly `iterate20`'s target.
- `d` ∈ R^{20×3}: the correction; the label is `r20 = traj + d`. The heading is **not** wrapped (as in `iterate20`).
- `P(d) = ND.to_navsim(traj + d)` and `s(d) = p4_tracker_comp.tracked8(ND, P(d)[None], ego)[0]`. This is the same
  `navsim_dac.simulated_states` → LQR + kinematic bicycle path `iterate20` calls; nothing is re-implemented.
- **Tracking residual** `e(d)` ∈ R^{24}: per knot k = 1..8, `[T_x − s_x, T_y − s_y, L · wrap(T_h − s_h)]`.
  - `L = ND.HALF_LENGTH = (4.049 + 1.127)/2 = 2.588 m/rad`.
  - Units: metres. A heading error δ moves the footprint's front/rear corners by ≈ L·δ, and the footprint is what
    sim8 (DAC) judges. So L converts heading into corner metres.
- **Roughness residual** `g(d)` ∈ R^{60}: per channel, `D d_c` with `D` ∈ R^{20×20} the second difference of the
  padded sequence `z = [z₋₁ = 0, z₀ = 0, d₁, …, d₂₀]`, so `(D d)_k = d_k − 2 d_{k−1} + d_{k−2}` with
  `d₀ = d₋₁ = 0`. The channels are x, y and `L · h` (corner metres).
- **Objective** `J(d; λ) = ‖e(d)‖² + λ ‖g(d)‖²` (m²; λ is dimensionless).

**Reading of the P5 card's "λ·‖Δ²r20‖²", committed here.** The penalty acts on the **correction** `r20 − traj`, not on
`r20` itself.
- A penalty on `r20` itself is minimised by straightening the teacher's own path: λ → ∞ would return a straight line,
  not `traj`. That moves the intent, which is not the lever. It would also make control C-R1 (§8) impossible.
- TB-6 is a ratio **against `traj`**, so the quantity to control is the label's excess over `traj`. By the triangle
  inequality, `max |Δ²(traj + d)| ≤ max |Δ²traj| + max |Δ²d|`.

**The clamped start** (`z₋₁ = z₀ = 0`) does two jobs.
- The tracker starts from the true ego state, so a correction that changes the reference's initial velocity cannot be
  executed at t = 0. The k = 1 term `d₁` penalises exactly that.
- It makes `D` lower-triangular with a unit diagonal, hence invertible, so `‖g‖²` is positive definite. That is what
  makes λ → ∞ return `d = 0`, i.e. `traj`, exactly (control C-R1).

**Δ² acts on yaw too, scaled by L.** Three reasons:
- TB-6's thin bar is the **heading-rate** one (1.939). A yaw channel left unregularised would leave the binding defect
  untouched.
- Without a yaw penalty, λ → ∞ cannot reproduce `traj`: the yaw correction would be unconstrained in the 8 null
  indices that `to_navsim` never reads.
- On the 8 label indices `to_navsim` never reads (t = 0.2, 0.8, 1.2, 1.8, 2.2, 2.8, 3.2, 3.8 s), and on the
  difference within each half-second pair, the tracker term is blind. **Only** the regulariser sets the label there.
  Heading is not penalised by Δ¹: the P5 card's form is Δ², and with the clamped start a Δ² penalty bounds the
  correction's own heading rate through its cumulative sum.

### 2.2 Optimiser: Levenberg–Marquardt with one finite-difference Jacobian and Broyden updates

- **Warm start:** `d⁰ = Tc20s − traj`, with Tc20s from the P5-C label table (`SG/p5c/labels/out/labels_s{0,1}of2.jsonl`),
  matched on (log, token, rank, `traj_sha256`). A row without a Tc20s label starts at `d⁰ = 0` and is counted.
- **Jacobian of e:** `J_e = −W · J_s · A`, with `W = diag(1, 1, L)` per knot.
  - `A = ∂P/∂r20` (24 × 60) by forward differences of `ND.to_navsim` itself, step 1e-3. It is exact for the linear
    xy part, and a local linearisation of the wrapped yaw interpolation. It needs no tracker call.
  - `J_s = ∂[s_x, s_y, wrap(s_h − s_h⁰)]/∂P` (24 × 24) by forward differences at `P(d⁰)`, step `h = 1e-3` (m or rad).
  - It costs **one batched `simulated_states` call of 25 proposals** (base + 24), computed once per row and shared by
    every λ solved on that row.
- **Jacobian of g:** exact and constant, `J_g = blockdiag(D, D, L·D)`.
- **Broyden (good) rank-1 update of J_s** after each iteration, using that iteration's best trial:
  `J_s ← J_s + ((Δs − J_s ΔP) ΔPᵀ) / (ΔPᵀ ΔP)`, skipped if ‖ΔP‖ < 1e-12.
  - Δs uses the wrapped yaw difference.
  - There is no Jacobian refresh. A stale Jacobian shows up as failed iterations, which the stall rule ends.
- **Step:** `δ(μ) = −(J_eᵀJ_e + λ J_gᵀJ_g + μ I)⁻¹ (J_eᵀ e + λ J_gᵀ g)`, solved with `numpy.linalg.solve`.
- **Damping:** `μ₀ = 1e-3 · max diag(H)` with `H = J_eᵀJ_e + λ J_gᵀJ_g`. μ never goes below
  `μ_min = 1e-12 · max diag(H)` (or 1e-12 if that is 0).
  - Each iteration simulates the three candidates `μ ∈ {μ_k/10, μ_k, 10 μ_k}` in **one batched call**. On TUNING, the
    call holds the candidates of every still-active λ.
  - The candidate with the lowest J is accepted **iff** its J < J_k, and then `μ_{k+1}` = its μ.
  - Otherwise the iteration has failed, and `μ_{k+1} = 100 μ_k`.
- **Stop** at the first of:
  - `K_max = 10` iterations;
  - 3 consecutive failed iterations;
  - an accepted step with `‖δ‖∞ < 1e-9` (m or rad).
  - There is no relative-decrease stop. At large λ it would fire while `‖d‖` is still ~1e-7 (pre-freeze self-review).
- **Returned:** the lowest-J evaluated iterate. Acceptance is descent-only, so this is the last accepted iterate, or
  `d⁰` if none was accepted.
- **λ = ∞** is a code path that returns `traj` bit for bit. It is the identity control, as `ident` is for `iterate20`.
- **Fallback:** an exception on a row gives the identity label (`source: raw_fallback`), and the row is counted (§5.4).

### 2.3 Tracker calls and the cost formula (written before any timing exists)

**Per row, one λ:**
- `n_prop = 25 + 3·K` proposals in `n_call = 1 + K` batched calls, with K ≤ K_max = 10. So at most 55 proposals in 11
  calls.
- Scoring adds 1 call holding raw, Tc20s and P5-R (3 proposals), plus `drivable_polygons`, sim8 DAC and interp8.

**On TUNING, G λ values jointly:** `n_prop = 25 + 3·Σ_λ K_λ` (≤ 25 + 3·9·10 = 295) in `1 + max_λ K_λ` calls. Scoring
adds 1 call of 2 + G proposals.

**Model:** a call of n proposals costs `t_sim(n) = a + b·n`.
```
t_row(G) = t_bld + t_sim(25) + Σ_{k=1..K̄} t_sim(3·G_k) + t_scr
```
- `t_bld` = scenario build and ego state per row; `t_scr` = scoring; `G_k` = λ values still active at iteration k.

**Bank scale (one λ\*, N = 150,603 rows):**
```
CPU-h  = N · t_row(1) / 3600
pod wall-h = N · t_row(1) / (3600 · 128 · η · ρ)
```
- η = parallel efficiency, 0.8 (ESTIMATED).
- ρ = pod-core speed relative to a dev-box core, 1.0 (ESTIMATED, until measured on the pod).

`a`, `b`, `t_bld`, `t_scr` and `K̄` are **MEASURED after the freeze**, in the controls stage (§8), and written into
RESULT_P5R with this formula filled in. P5-C's label build on the pod runs `iterate20` first (the warm start) and adds
`t_row(1)` on top.

## 3 · Data, split, samples

- **Bank:** `F:/tanitad_scratch/p5c_tb6/bank/targets_rank{0,1}.jsonl` (P7's bank, 150,603 rows; MM-owned, read-only).
  `sim_hz = 10` is asserted per row.
- **DBs:** `navtrain_scenarios.index_dbs(db_root="E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits")`.
- **Maps and sim:** `NUPLAN_MAPS_ROOT=E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0`, `REFE_SIM_HZ=10`,
  `CUDA_VISIBLE_DEVICES=-1`, one BLAS thread per process.
- **Eligible logs:** logs with a local DB **and** at least one Tc20s-labelled row (source `tc20s`) in
  `labels_s{0,1}of2.jsonl`. These are the 482 dev-box-labelled logs; the count is re-measured and reported.
- **Split, by log:** `h = int(sha256(("p5r-split|" + log_name).encode("utf-8")).hexdigest()[:8], 16)`.
  **TUNING iff `h % 4 == 0`**, else **EVAL**.
- **Rows within (split, rank):** eligible rows with a Tc20s label, sorted ascending by
  `sha256(f"p5r-row|{log}|{token}|{rank}")`. The first N are taken.
  - **TUNING:** N_t = **250 per rank**.
  - **EVAL:** N_e = **1,200 per rank**.
  - **CONTROL:** the first **50 per rank** of the TUNING sample.
- **Budget rule** (mechanical; the whole run stays ≤ 3.0 CPU-h): after the controls stage, the run projects
  `controls + 2·N_t·t_row(9) + 2·N_e·t_row(1) + 0.1 h` from the measured terms.
  - If the projection is > 3.0 CPU-h, N_e drops to the largest multiple of 100 (≥ 600) that fits.
  - If it still does not fit, N_t drops to the largest multiple of 50 (≥ 150) that fits.
  - If it still does not fit: STOP, and an addendum is written before any TUNING row is solved.
- **Concurrency:** at most 2 solver processes.
- ⛔ **P5-R never touches navtest.** No GPU, no pod, no box 2.

## 4 · λ selection (TUNING only; mechanical; EVAL is never read for it)

- **Grid:** `Λ = {0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100}`.
- On each TUNING row, every λ ∈ Λ is solved from the same warm start, with the shared FD Jacobian.
- **For each λ and each rank k**, on the TUNING rows of rank k:
  - `R_acc(λ,k)` and `R_hr(λ,k)`: TB-6 exactly as P5-C reads it (§5.1);
  - `R_acc^proj(λ,k)` and `R_hr^proj(λ,k)`: the full-bank projection (§5.3).
- **λ is admissible** iff, for **rank 0 AND rank 1**, all four ratios are **≤ 1.80**.
  - 1.80 leaves 10 % headroom under the 2.0 bar, for TUNING → EVAL sampling noise.
  - The full-bank shift is handled separately, by including the projection.
- **λ\* = the smallest admissible λ in Λ.**
  - Tracking error does not decrease with λ, so the smallest admissible λ keeps the most repair. No repair metric is
    needed in the rule.
- **If no λ is admissible:** λ\* = 100, recorded as `NO_ADMISSIBLE_LAMBDA`. EVAL still runs and is read as written.
- **Recorded:** the full TUNING table (every λ, both ranks: the four ratios, sim8 rate, mean 8-knot max-xy,
  J_track). It is descriptive and decides nothing beyond λ\*.
- λ\* and the table are written to `lambda_star.json` **before** any EVAL row is solved.

## 5 · Metrics

### 5.1 TB-6, as P5-C reads it

Per rank, on a set of rows:
```
R_m = median_i(m(label_i)) / median_i(m(traj_i))
```
- m ∈ {max |acc|, max |heading rate|} from `P.learnability`: the origin is prepended, dt = 0.2 s, and the result is
  in m/s² and °/s.
- This is a **ratio of medians** (P5-S's own rule, `P.py:388`), not a median of ratios.
- Point estimate. A log-cluster bootstrap CI of the ratio is reported as descriptive only.

### 5.2 Tracking

All of these come from the **real** tracker, on the label as stored.
- **sim8:** NAVSIM DAC on the executed footprint (`state_array_to_coords_array`, CENTER excluded,
  `ND.violations(ND.drivable_polygons(...))`). This is exactly `p5c_labels.py repair`.
- **E_maxxy:** `max_k ‖s_xy − T_xy‖` (m) over the 8 knots, `iterate20`'s own selection metric.
- **J_track:** `‖e‖²` (m²), the objective's tracking term.
- **Descriptive:** interp8 (DAC on `ND.at_10hz` of the label's 8 poses, no tracker), comfort
  (`ND.comfortable_from_states`), and `P.exec_vs_intended` against T.

### 5.3 Full-bank projection of TB-6 (ESTIMATED; it answers *"what will the pod's full-table TB-6 read?"*)

For rank k and metric m:
- `x_j = m(traj_j)` over **all** bank rows of rank k (92,896 / 57,707 rows; the bank holds `traj` for every row, so
  this is exact).
- **Denominator** = `median_j x_j`. It is EXACT, and it is what the pod's TB-6 will divide by.
- **Bins:** the 9 inner deciles of x over all bank rows of rank k. Each sample row i gets `bin_i`, and the weight
  `w_i = 0.1 / (n_bin_i / n_sample)`. A bin with no sample row is dropped, and its mass is reported.
- **Numerator** = the w-weighted median of `m(label_i)` over the sample rows: the first sorted value whose cumulative
  weight is ≥ half the total.
- `R_m^proj` = numerator / denominator.
- **Assumption, stated:** within a decile of the raw metric, the label/raw relation is the same on DB-less logs as on
  dev-box logs. That is why the class is ESTIMATED.
- **Descriptive check of the method:** the same projection applied to **Tc20s** on the same rows, set against the MM's
  ≈ 2.09.

### 5.4 Sanity, as TB-2 and TB-5 (a gate: a FAIL means GATE_FAIL and no verdict)

| id | check | bar |
|---|---|---|
| S1 | label finite | 100 % |
| S2 | `max |label_h − traj_h| < π` (the yaw-branch check) | 100 % |
| S3 | knot fit / re-simulation identity: `tracked8` of the **stored** label, simulated **alone**, equals the solver's recorded final `s` | ≤ 1e-6 (m, rad) on 100 % of rows, the bar of P5's own batch-independence control C7 (`P.py:323`); the maximum is recorded. This also proves batch independence |
| S4 | solver fallbacks (exception ⇒ identity label) | ≤ 1.0 % of rows per rank (TB-4's threshold) |
| S5 | λ reaches the objective: every row with λ > 0 and `d ≠ 0` records `λ‖g‖² > 0`, and the recorded J equals `‖e‖² + λ‖g‖²` | 100 %, to 1e-9 relative |

## 6 · Bars on EVAL, per rank (committed)

Arms on the same rows: **raw** (`traj`), **Tc20s**, **P5-R(λ\*)**, all paired.

| id | bar | reading |
|---|---|---|
| **B1** | TB-6 MEASURED on the EVAL rows: `R_acc ≤ 2.0` AND `R_hr ≤ 2.0` | P5-C's TB-6, unchanged |
| **B1p** | TB-6 PROJECTED to the full bank (§5.3): `R_acc^proj ≤ 2.0` AND `R_hr^proj ≤ 2.0` | the reason P5-R exists; ESTIMATED |
| **B2** | sim8 non-inferiority vs Tc20s: upper 95 % CI of `fail(P5-R) − fail(Tc20s)` ≤ **+1.0 pp** | the margin is about a third of Tc20s's dev-box rank-0 repair (+3.00 pp) |
| **B3** | still a repair: upper 95 % CI of `fail(P5-R) − fail(raw)` **< 0** | |
| **B4** | 8-knot non-inferiority vs Tc20s: upper 95 % CI of `mean E_maxxy(P5-R) − mean E_maxxy(Tc20s)` ≤ **+0.10 m** | Tc20s's repair is ≈ 0.40 m (P5-S), so this keeps ≥ ~75 % of it |
| B5 | S1–S5 (§5.4) | a gate, not a verdict bar |

- **Per-rank verdict:** PASS iff B1 ∧ B1p ∧ B2 ∧ B3 ∧ B4. If B5 fails, the result is GATE_FAIL and no verdict is
  read.
- **P5-R PASSES iff rank 0 PASSES.** Rank 1's verdict is the P5-R analogue of P5-C §4.3's RK4/RK5, and feeds the
  rank-1 policy.
- The failing bars are always named.

## 7 · Estimator, evidence class, tier

- **Estimator:** P5 §3's paired log-cluster bootstrap, using the **imported, unchanged** `P.boot_weights` and
  `P.paired_ci`: 10,000 draws, seed **20260927**, 95 % percentile CI.
  - Rates use the ratio estimator. Continuous metrics use the difference of cluster-weighted means.
  - Draws are per rank, over that rank's EVAL logs, and every arm is paired on the same rows.
  - TB-6 ratios are point estimates (as in P5-C), with a descriptive bootstrap CI from the same `W`.
- **What the interval answers:** only *"another draw of logs"*.
  - The construction is deterministic: no training, no sampling planner, a deterministic tracker, a deterministic LM.
  - S3 checks batch independence.
  - So the training-variance and inference-variance questions do not arise here. ⚠ They do arise for P5-C's student,
    and are P5-C's to answer.
- **Evidence class:** MEASURED (own, CPU) for B1–B4 and S1–S5; **ESTIMATED** for B1p, because of the projection
  assumption.
- **Tier stamp:** *label-construction diagnostic, tracker-in-the-loop (NAVSIM v1.1 PDM components, vendored
  `navsim_pdm`), navtrain rows of P7's bank with a local nuPlan DB, CPU, dev box*.
  - ⛔ It is **not** a driving-performance number, **not** navtest, and **not** a PDMS number.
  - "Held-out" here means only that EVAL logs are disjoint from TUNING logs. These rows are P7 **training-bank** rows.
- **Four families** (descriptive, per family, never pooled; executed vs intended T, `P.exec_vs_intended`):
  - LONGITUDINAL: mean |along|, along at 4 s.
  - LATERAL: mean |cross|, heading MAE, E_maxxy.
  - TACTICAL: **not computable**. A label construction has no manoeuvre decision (n = 0, reason stated).
  - STRATEGIC: **PAUSED** (PI ruling 2026-09-19).

## 8 · Controls and deliberate-regression arms (CONTROL rows; any FAIL means GATE_FAIL, and no TUNING or EVAL is read)

| id | arm | must read |
|---|---|---|
| **C-R1** | λ = 1e12 | `max |r20 − traj| ≤ 1e-6` (m, rad) on 100 % of rows; and the λ = ∞ path gives `label == traj` **bit for bit** on 100 %. Why 1e-6 and not 0: the exact minimiser at finite λ is `d* ≈ (λ DᵀD)⁻¹ J_eᵀe(0)`, of order `‖∇J_track‖ / (λ · σ_min(D)²)` ≈ 1 / (1e12 · ~4e-5) ≈ 1e-8 m. So 1e-6 is 100× above the true minimiser and far below any label effect (the corrections are ~0.1–1 m). "Exactly" belongs to the λ = ∞ path. |
| **C-R2** | λ = 0 (only the LM damping) | `J_track(P5-R) ≤ J_track(Tc20s)` on 100 % of rows (descent from the warm start), **and** median `J_track(P5-R)` **<** median `J_track(Tc20s)` (a solver that cannot beat 3 fixed-point steps at zero regularisation is broken) |
| **C-R3** | broken tracker A: inside the solver `s := P` (the identity, "no tracker"); scoring uses the real tracker; λ = 1 | known value: `max |label − traj| ≤ 1e-6` on 100 % (the minimiser of `‖W A d‖² + λ‖g‖²` is `d = 0`), **and** B4 on these rows (pooled over ranks) reads **FAIL** |
| **C-R4** | broken tracker B: the solver's tracker samples the simulation 0.1 s early (`sim[:, 4::5]`, an off-by-one knot bug); scoring uses the real tracker; λ = 1 | B4 on these rows (pooled) reads **FAIL** |
| C-R5 | S1–S5 on every control row | as §5.4 |

- **Unit-test regression** (`test_p5r_labels.py`, synthetic tracker, no nuPlan, no scipy): with the λ term mutated
  out of the objective (`_mutate="drop_lambda"`), the C-R1 check must go **RED**. Without the mutation it must be
  **GREEN**.
- **Zero control:** at λ = 0 the recorded `λ‖g‖²` term must read **exactly 0.0**.
- The controls stage also records the timings of §2.3 (`a`, `b`, `t_bld`, `t_scr`, `K̄`), and applies the budget rule
  of §3.

## 9 · Comparators

- **Tc20s:** the existing P5-C labels, matched on `traj_sha256`. It is the reference for B2 and B4, and the warm start.
- **raw** (`traj`): the reference for B3 and the TB-6 denominator.
- **Tsel-s:** Tc20s where raw fails sim8, else `traj`. It is derived with no extra simulation, and is SECONDARY.
  - It **decides nothing**. Its median-ratio TB-6 is ≈ 1 by construction, because most rows are untouched, so it
    cannot be read on the bar that triggers P5-R.
  - It is reported with its sim8 rate, its TB-6, and its learnability on the touched rows only, as the input to the
    next lever (§10).

## 10 · Outcomes and consequences, committed both ways (Rule Zero: no branch ends at a verdict)

**Trigger.** P5-R replaces Tc20s only if P5-C's own trigger fires: the pod's full-table TB-6 reads **rank-0 FAIL** on
Tc20s, or RK0 stops P5-C.
- If the pod's TB-6 passes on Tc20s, P5-C stays on Tc20s, whatever P5-R reads. P5-R is then banked as a contingency.
- Switching a frozen card's label without its trigger would be a moved goalpost.

**P5-R PASS (rank 0) and the trigger fires** ⇒ P5-R(λ\*) becomes C1's label. `PREREG_P5C.md` then needs a dated
amendment, before the launch gate, at these points:
1. **Title, §1, §7 wording:** the label name ("tracker-aware regularised labels, λ\*"). The PI ruling of 2026-11-11
   (tracker-aware labels admissible, tracked AND transparent) must be **confirmed to cover P5-R by the MM**. It is the
   same class of label: a pre-distortion of the same teacher row for the same tracker.
2. **§4.1 builder:** `p5r_labels.py` blob, λ\*, and the §2 constants. The pod runs `iterate20` (unchanged, the warm
   start), then P5-R.
3. **§4.2:**
   - **TB-1** adds a reproduction of this run's EVAL labels.
   - **TB-2** (spline knot exactness) is replaced by S3 (re-simulation identity).
   - **TB-0** needs a tolerance decision. The LM's `numpy.linalg.solve`, like the tracker's own `pinv`, goes through
     LAPACK, so pod-vs-dev-box bit identity is **not guaranteed**. The proposal is: max abs diff ≤ 1e-6 m and
     identical sim8 verdicts, or bit identity demonstrated.
   - **TB-6 is unchanged and stays binding** on the full P5-R table. P5-R's B1p is a projection, never a substitute.
4. **§4.3:** r_k is recomputed with P5-R labels. RK0 is unchanged.
5. **§10:** the launch-gate pins (`P5C_LABEL_TABLE_SHA256`, code blobs).
6. **§14:** pod CPU cost from §2.3's filled formula.

**P5-R FAIL (rank 0)** ⇒ the next lever is **P5-T, the selective tracker-aware label**:
- Tc20s, unchanged, on rows whose raw `traj` fails sim8 at label time; raw `traj` elsewhere. This is the P5 card's
  Tsel-s construction, promoted from an input to an arm.
- It needs its own §9 entry. That entry must add a learnability guard **on the touched rows**: a median-ratio TB-6 is
  ≈ 1 there by construction, so it cannot police them. P5-S measured 1.49× / 1.33× on its 82 touched rows.
- This holds whichever bar failed:
  - **learnability failed** (B1 or B1p, including `NO_ADMISSIBLE_LAMBDA`): the regulariser cannot reach the bar inside
    the repair-preserving range;
  - **repair failed** (B2, B3 or B4): the regulariser costs the repair, and P5-T keeps Tc20s's full repair on exactly
    the rows that need it.
- The P5-T draft is written in the same turn as a FAIL result.

**GATE_FAIL** (any control or S-check): no verdict. The defect is fixed under an addendum, and the affected stage is
rerun.

**Rank 1 FAIL with rank 0 PASS** ⇒ P5-C's §4.3 applies as written: rank-0 only.

## 11 · What this does NOT license

- Not a student-learnability claim. TB-6's 2× bar is a proxy, as RESULT_P5S says.
- Not a PDMS, navtest or driving-performance claim.
- A PASS does not waive the pod's full-table TB-6.
- B1p is ESTIMATED, and is never quoted as a measurement of the full bank.
- Nothing here changes `p5_target_labels.py` or the frozen PREREG_P5C.

## 12 · Outputs

- **Hold** `SG/p5r/` (LF line endings):
  - `PREREG_P5R.md` + `.sha256`; `p5r_labels.py` (solver, no nuPlan import at module level); `test_p5r_labels.py`;
    `p5r_run.py` (driver);
  - `raw/controls.json`, `raw/budget.json`, `raw/tuning.json`, `raw/lambda_star.json`, `raw/eval.json`,
    `raw/verdict.json` (+ logs);
  - `RESULT_P5R.md`, or `HANDOFF.md` if the run outlives this agent.
- **Scratch:** `F:/tanitad_scratch/p5r_agent/` (per-row jsonl; deleted at the end once the hold copies are verified).
