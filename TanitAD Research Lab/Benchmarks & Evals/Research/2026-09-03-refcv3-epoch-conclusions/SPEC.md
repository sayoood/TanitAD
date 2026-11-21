# SPEC — refcv3 epoch conclusions from its own in-training metrics (PRE-REGISTERED)

**Owner:** Benchmarks & Evals FlyWheel · **written:** 2026-09-03, BEFORE any number in
`RESULT.md` was computed · **tier of everything below: T0** (an in-training WM/loss diagnostic;
⛔ never "driving performance", EVAL_DOCTRINE rule 2) · **compute:** 0 GPU, dev box, stdlib only.

> ⛔ **The pod `tanitad-refcv3` is NOT contacted.** refcv3 is training there (step ~25,650 of
> 40,284 at the time of the pull). Every source below is the LOCAL pull at
> `C:\Users\Admin\refcv3_diag\`, banked into `raw/` of this package so it stops living on one
> disk. Thor is not contacted either.

---

## 0. Why a SPEC exists before the numbers

Two of this programme's most expensive failures were *reads*, not runs: a trend quoted across a
boundary that changed the measurement (`C-REFCV3-EVAL-PRIOR-LEAK`, `D-REFCV3-NAV-SWITCHED`), and a
paired comparison that was VOID rather than negative because nobody printed the arms' shape first
(`D-REFAV1-PAIRED-READ-VOID`). This file commits, in advance, to **what would make each reading
wrong**. A reading whose refutation condition fires is reported as refuted, not quietly dropped.

---

## 1. Primary sources (the only quotable inputs)

| file | sha256 | size | rows | banked at |
|---|---|---|---|---|
| `metrics.jsonl` | `8207d81d94921a9834b004456d9581031fba97a2e325b9d46fa6b536a4d8d3c6` | 298,548 B | 614 | `raw/metrics.jsonl` |
| `supervisor.log` | `71ad68b06bd9f7714a07f73aee352944442a12f05c83a24cd8558b5322f2c5db` | 982 B | 12 | `raw/supervisor.log` |
| `config.json` | `668e4520351fe67e99f7992b7da7e558dae9e6bcc117faa33ae6096fa8cc3874` | 4,115 B | — | `raw/config.json` |

Secondary (code, read for SEMANTICS only, never for numbers):
`stack/scripts/refc_v3_train.py` (the eval block `:1195–1262`, the log block `:1146–1165`),
`taniteval/taniteval/ci.py`, `taniteval/tools/t1_eval.py`, `taniteval/tools/refav1_arm.py`.
Register rows read as CONTEXT (evidence class INHERITED unless re-measured here):
`C-REFCV3-EVAL-DEATH-DIAGNOSED`, `D-REFCV3-NAV-SWITCHED`, `C-REFCV3-EVAL-PRIOR-LEAK`,
`D-REFCV3-SAVE-BEFORE-EVAL`, `D-REFCV3-U8-SHIPPED`, `D-REFCV3-19K`, `D-REFCV3-21K`,
`D-V7-READINESS-2026-09-02`, `D-REFAV1-HA0-ARM`, `D-REFAV1-PAIRED-READ-VOID`,
`C-REFCV3-ARM-SAME-DEFECT`, `C-STEER-CURVATURE-INTERFACE`.

**Instrument written for this read:** `taniteval/tools/refcv3_metrics_read.py` (new, owned by this
package). It emits everything in `raw/`; no number in `RESULT.md` is typed by hand.

---

## 2. What the file actually is (semantics, read from source before reading values)

* A **train row** is written every `--log-every 50` steps and holds the **single-batch** loss at
  that step (`refc_v3_train.py:1152–1161`; `batch 20` ⇒ 20 windows). It is NOT an average over the
  interval.
* `elapsed_s` is `time.time() - t0` with `t0` at **launch**, i.e. **cumulative since the current
  launch** — so the per-step rate is `Δelapsed_s / Δstep`. ⚠️ It is **not** the `step_s` field the
  CLAUDE.md divisor rule is about; that rule does not apply here and must not be applied.
* An **eval row** (`eval_*`) is the mean over `--eval-batches 8` × `batch 20` = **160 windows**,
  drawn ONCE by `torch.randperm(len(e_ds), generator=Generator().manual_seed(12345))[:160]` from the
  whole held-out corpus and iterated in a fixed order (`refc_v3_train.py:1037–1041`) — a **fixed**
  set, by construction, at every step and across every relaunch.
* The eval set is the **141 B1 clips** that carry pixels out of the v7.2 EVAL release's 147 records
  (the other 6 are the deployed-val40 clips the parity gate drops, `D-REFCV3-EVAL-LEAK`).
* `lat_tac` / `lon_tac` are cross-entropies of **8-class** heads (`tac_vocab_version "v7.0"`,
  `config.json`), so the no-information value is **ln 8 = 2.0794**.
* Deaths and switches: the trainer resumes from `ckpt.pt` (500 steps before the death), so the file
  contains **replayed steps**. Row order is therefore NOT a time series.

---

## 3. The reads, and what would refute each

Every read is stated as a claim, an instrument, and the condition under which the claim is
**withdrawn**. `n` and the window travel with every fitted number (CLAUDE.md exponent rule).

### R1 — Reconstruct the run's true shape from the file itself
**Claim.** The 614 rows decompose into **N launches**; a launch boundary is an `elapsed_s` decrease
on a train row; the canonical series is **last-writer-wins per (step, kind)**; the replayed rows are
the recomputation cost of the deaths and switches.
**Refuted if** the launch count or the resume steps recovered from `elapsed_s` disagree with
`supervisor.log`'s own relaunch lines, or if any resume step is not (death step − 500).

### R2 — The held-out window set is FIXED across the whole run (the enabling condition)
**Claim.** All 55 eval rows are computed on the same 160 windows, so the eval series is
step-to-step comparable and the only confounds are the ones in §R3.
**Instrument.** `eval_windows`, `eval_batches`, `eval_slot_valid_frac`, `eval_tac_label_rows`,
`eval_tac_label_v7`, `eval_nav_injected` must each take **exactly one** distinct value over all 55
rows.
**Refuted if** any of them varies. ⚠️ If refuted, every cross-step eval comparison in this package
collapses and must be withdrawn — this is the load-bearing check, not a formality.

### R3 — Segment the series; never pool across a boundary that changed the measurement
**Claim.** Three eras, and no trend is quoted across a boundary:

| era | steps | nav source | eval prior leak | note |
|---|---|---|---|---|
| **A** | eval ≤ 16,500 | refb-derived (`follow` on 94.6 % of windows) | **PRESENT** | contaminated: partly self-referential |
| **B** | 17,000 – 17,500 | v7.2 token | **PRESENT** | 2 points; transition only, never a trend |
| **C** | ≥ 18,000 | v7.2 token | **GATED** | the only clean era |

The gate shipped **at** 17,500 and the run resumed from the 17,500 checkpoint, so the 17,500 eval
was produced by the OLD (leaking) trainer and the **first clean eval is 18,000**.
**Refuted if** the launch that produced the 17,000/17,500/18,000 eval rows is not the one the
register's switch times imply, or if `supervisor.log` places the file swap after the 17,500 eval.

### R4 — Direction within an era, with the exponent discipline
**Claim.** Per era and per column: OLS slope per 1,000 steps, with **R²**, **n** and the **exact
fit window** printed beside it.
**Not quotable if** `R² < 0.80` (CLAUDE.md: below 0.80 there is no quotable exponent) — in that case
the reading falls back to the **matched-step / half-split ratio** and is reported as *direction, not
rate*. ⛔ No extrapolation beyond the fitted range; no comparison of slopes fit over different
windows.

### R5 — The clean-era plateau (the read that decides whether more steps buy anything)
**Claim.** In era C the headline columns (`eval_loss`, `eval_traj`, `eval_lat_tac`, `eval_lon_tac`,
`eval_goal2s_err_m`, `eval_anchor_acc`) move between the era's two halves by **less than 2×** the
series' own step-to-step scatter, i.e. they are flat at this instrument's resolution.
**Instrument.** `sd_step = pstdev(first differences)/√2` (the per-point scatter implied by
consecutive evals), compared to `|mean(H2) − mean(H1)|`.
**Refuted if** any of those columns moves ≥ 2 × `sd_step`.
⚠️ **This ratio is a descriptive signal-to-scatter number, NOT a t-statistic and NOT a CI.** The 55
points are autocorrelated and share one 160-window sample; nothing here licenses an interval.

### R6 — The strategic goal path is still opening
**Claim.** `eval_goal_gate` (the zero-init strategic gate) rises with a fit that clears the R² bar in
BOTH era A and era C, and `eval_goal_score_absmean` falls.
**Refuted if** either fit is below R² 0.80, or the direction differs between eras.

### R7 — The train–eval gap and its sign
**Claim.** `eval_X − mean(train X over (N−500, N])` is **negative** on `loss` in era A and **positive**
in era C — the generalisation gap opening.
**Robustness (pre-registered).** Recompute with the **first two rows after every relaunch dropped**;
if the sign change survives, it is not an artifact of the post-resume spikes.
**Refuted if** the sign change does not survive that robustness pass.
⚠️ **Level caveat stated in advance:** train rows are 20-window samples of the CURRENT batches and
the eval is a fixed 160-window set — different windows. Only the **trend** of the gap is
interpretable, never its absolute level.

### R8 — Post-resume spikes are transient
**Claim.** The first train row after each resume is 1.0–1.4× the pre-death level and is back at or
below it by the **next** row (≤ 50 steps).
**Refuted if** any launch needs more than one log interval to return, which would mean a resume
perturbs training rather than merely re-warming it.

### R9 — Pace, measured twice through two different mechanisms
**Claim.** ~4.69–4.78 s/step before the uint8 switch and materially faster after it, from
`Δelapsed_s/Δstep`.
**Second probe (different mechanism, per the "two probes" rule):** wall-clock deltas between
`supervisor.log` relaunch timestamps, divided by the steps between the corresponding death steps.
**Refuted if** the two probes disagree by more than 5 %. *(Repeating the same probe is one probe;
this is the reason the second one is a different clock.)*

### R10 — What this file CANNOT support (pre-registered as a NEGATIVE)
**Claim.** No confidence interval on any `eval_*` column can be computed from `metrics.jsonl`,
by any estimator, because the file carries only the **pooled mean over the 160 windows** — no
per-window values and no `eid`. `taniteval.ci.episode_cluster_bootstrap` requires per-window values
plus an episode index (`ci.py:225`, `:275`); neither exists here.
**Refuted if** any per-window array or episode identifier is found in the file.
⇒ Consequence committed in advance: every era-level statement in `RESULT.md` is a **direction**, and
any sentence that would need an interval is written as *"cannot be had from this file"* with the
job that could produce one named.

---

## 4. Controls and invariants that must read a known value

| control | must read | why |
|---|---|---|
| `eval_tac_label_v7` | exactly `1.0` on all 55 rows | the v7.2 tactical labels really reached the eval |
| `eval_nav_injected` | exactly `1.0` on all 55 rows | nav reached the model at eval time in every era |
| `eval_windows` | exactly `160` on all 55 rows | the denominator never changed |
| `lat_tac` / `lon_tac` chance line | `ln 8 = 2.0794` | any value near it means an untrained head, not a result |
| replayed-step accounting | `Σ replayed rows × 50` = the recomputation cost | if it does not reconcile with `supervisor.log`, R1 is refuted |

---

## 5. Out of scope, stated so absence is not read as a finding

* ⛔ No T1 number, no driving claim, no four-family row is produced here — `metrics.jsonl` is a T0
  loss monitor on the training loss surface (`refc_v3_train.py:1200–1207` says so itself).
* ⛔ No cross-arm comparison to refav1 or the flagship is computed. §Task-3 of `RESULT.md` states
  what such a comparison would *require*; it does not perform one.
* ⛔ No edit to `taniteval/tools/refcv3_arm.py` or to the rescued draft (BACKLOG R20, not this
  package's file). Both are read only.
* The eval's **episode** composition (how many of the 141 clips the 160 windows touch) is not
  recoverable from this file and is listed as a work item, not estimated.

---

## 6. Deliverables this SPEC commits to

1. `taniteval/tools/refcv3_metrics_read.py` — the instrument (new; mine).
2. `raw/refcv3_epoch_read.json` + `raw/refcv3_eval_series.csv` + `raw/refcv3_train_series.csv` —
   every derived number, machine-readable.
3. `raw/metrics.jsonl`, `raw/supervisor.log`, `raw/config.json` — the primaries, banked.
4. `RESULT.md` — the honest read (Task 1), the comparability analysis (Task 3), the proposed
   register rows.
5. `T1_CHECKLIST.md` — the executable end-of-epoch T1 runbook (Task 2).
