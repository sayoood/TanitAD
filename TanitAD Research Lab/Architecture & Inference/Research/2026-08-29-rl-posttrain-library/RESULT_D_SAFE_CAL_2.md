# RESULT — D-SAFE-CAL-2: the seed-1 arms RAN on 2026-08-30 and were never read. Read now: ⛔ VOID (V1).

**Date of the arms:** 2026-08-30 08:51–09:12 (five arms, `tanitad-data/rl-pilot/d2-*`, logs beside them) ·
**Date of this readout:** 2026-09-05 (DeployFlyWheel, during the REF-C RL-readiness audit
`…/Deployment & Optimization/Research/2026-09-05-refc-rl-readiness/`) · **Tier: T0 training-side** ·
**Pre-registration:** `PREREG_D_SAFE_CAL_2.md` (committed before any arm; seed 1 chosen so the exits are read
against numbers the author had not seen) · **Evidence class:** MEASURED (ours; the banked seed-1 readouts) ·
⚠️ NON-PARITY pilot corpus (REF-C v2.1, 54 train / 15 val joined episodes) — no number here enters a cross-arm
table. **Estimator:** paired episode-cluster bootstrap, 4000 reps, 15 val clusters, eval-mode deterministic readout.

---

## 0. Why this document exists

The experiment ran to completion — 5/5 arms, checkpoints, before/after readouts, `dsafe2_prox_decomp.json` for the
`R4_full` proximity term — and then nothing: no `RESULT_D_SAFE_CAL_2.md`, no register row, no commit after
`7be8aed` (2026-08-30, "the D-SAFE-CAL package that returned VOID"). The 2026-09-05 audit found the arms by
listing `tanitad-data/rl-pilot/` (operating-standard rule 3: an artifact on one disk is not done). The readout
below is the committed table applied **mechanically** (`code/dsafe2_analyze.py`, first match in the committed
order wins, VOID conditions override everything) — the same discipline `RESULT_D_SAFE_CAL.md` used.

## 1. The arms (all five ran clean; per-episode values stored; `eval_mode: true`)

| arm | `w_anchor` | `d_safe` / `w_prox` | ΔR1 composed reward | ΔR2 fan collision (pp) | ΔR3 sel-ADE (m) |
|---|---|---|---|---|---|
| **A** | 1.0 | 2.0 / 0.5 | **−0.1718** [−0.2029, −0.1422] SEP | −0.215 [−0.501, +0.046] | **+0.1096** [+0.0357, +0.1723] SEP |
| **B** | 10.0 | 2.0 / 0.5 | **−0.0286** [−0.0375, −0.0199] SEP | −0.052 [−0.169, +0.046] | +0.0361 [−0.0234, +0.0831] |
| **C** | 1.0 | 5.0 / 0.5 | +0.0025 [−0.0175, +0.0210] | **−0.260** [−0.475, −0.091] SEP | **+0.0522** [+0.0020, +0.1062] SEP |
| **D** (repro) | 1.0 | 2.0 / **0.0** | **−0.0327** [−0.0479, −0.0171] SEP | −0.026 [−0.202, +0.150] | **+0.1112** [+0.0767, +0.1481] SEP |
| **REG** | 1.0 | hackable | **−0.0846** [−0.0918, −0.0779] SEP | −0.098 [−0.449, +0.169] | **+0.0553** [+0.0320, +0.0807] SEP |

`R4_full` proximity Δ (post hoc from `ckpt_after.pt`, zero GPU): A **+0.0017**, B +0.0003, C +0.0005, D +0.0004,
REG −0.0011. R5 on the frozen 5.0 m ruler: A +0.0250, B/C/D +0.0083, REG +0.0000 (none separated except REG's
exact zero). Raw: `raw/dsafe_cal_2/dsafe2_result.json`.

## 2. The committed table, in its committed order

| # | rule (PREREG_D_SAFE_CAL_2 §3) | reading | fires? |
|---|---|---|---|
| **V1** | arm D must MATCH the banked no-proximity arm (`s2-w1`, seed 0) within the paired CI; a DIVERGENCE is instrument failure | banked ΔR1 **−0.0660** lies OUTSIDE D's CI [−0.0479, −0.0171]; banked ΔR3 +0.1115 lies INSIDE D's CI [+0.0767, +0.1481] | ⛔ **YES — on R1** |
| V2 | REG must degrade with paired separation on R1 or R3 | R1 −0.0846 SEP and R3 +0.0553 SEP | no (licence holds) |
| 1 | ΔR2 falls with separation in A and C reproduces the banked null | A's R2 not separated | no |
| 2 | A ≈ C, neither separates on R2, proximity moves < 0.01 in A | A −0.215 pp n.s., C **−0.260 pp SEP**, proximity +0.0017 | no (C separates) |
| 3 | ΔR2 falls in A but R5 rises on the frozen ruler | — | no |
| 4 | ΔR3 degrades with separation in A while ΔR2 flat | A R3 +0.1096 SEP, R2 flat | would match |
| 5 | none | — | — |

⇒ **VOID (V1).** The first rule in the committed order fires, and V1 overrides everything below it. ⛔ Nothing in
§1 may be read as a result about `d_safe`.

## 3. What the VOID says — and what it does not

**It says the reproduction test as WRITTEN cannot pass across seeds.** Arm D is bit-identical in objective to the
banked `s2-w1` arm except for `seed` (1 vs 0). Its deployed-path delta reproduces (ΔR3 +0.1112 vs +0.1115 —
three decimals); its composed-reward delta does not (ΔR1 −0.0327 vs −0.0660, both separated from zero, both the
same sign). The prereg chose seed 1 deliberately (§0, so the exits are read on unseen numbers) and then committed
V1 to a *same-value* match against a seed-0 reference, with no seed-variance allowance. Under that rule the
seed-to-seed spread of ΔR1 reads as a harness failure. **The rule names the wrong object** — it asks the
harness to reproduce a number that a different seed is not expected to reproduce exactly — which is the
TRAIN-C6 / TRAIN-C12 class (*a committed rule that does not match the object it names*), now in its third
appearance in this line. Logged as **TRAIN-C13** in `RETRACTION_LOG.md` (drafted by this readout; the Master
Mind owns the log's numbering).

**It does not say the harness is broken.** V2 holds (REG degrades on both axes with separation), D reproduces
R3, and every arm ran with checkpoints and a done-marker. A same-seed reproduction (seed-1 default arm vs
seed-1 D, or seed-0 D vs the seed-0 banked arm) is the corrected V1 and costs ~8 min on the 4060.

**⛔ DESCRIPTIVE ONLY — may NOT select a branch (the same discipline as `RESULT_D_SAFE_CAL.md` §4):** with V1 set
aside, exit 4 would match (A's ΔR3 degrades with separation while its ΔR2 is flat), i.e. the same reading the
whole P-RC21 line has produced — the barrier fights the trust region and the deployed path drifts. The one row
that separates on R2 is C (`d_safe` = 5.0, −0.260 pp), the incumbent threshold, not the recalibrated one — and a
15-cluster interval that just clears zero is not a finding either.

## 4. Consequence for the refcv3 RL-readiness work

`proximity` stays OUT of the minimal refcv3 experiment's reward (`DEFAULT_WEIGHTS` never contained it), and
`proximity_safe_m = 5.0` stays in code (`THRESHOLD_CALIBRATION` marks it MISCALIBRATED). The D-SAFE-CAL line is
closed as **VOID twice**, both times on the pre-registration rather than on the science; a third attempt needs
the same-seed V1 above and nothing else changed.

## 5. Deliverables

| artifact | where |
|---|---|
| this readout | `…/2026-08-29-rl-posttrain-library/RESULT_D_SAFE_CAL_2.md` |
| the mechanical analyzer | `…/2026-08-29-rl-posttrain-library/code/dsafe2_analyze.py` |
| raw exit record | `…/2026-08-29-rl-posttrain-library/raw/dsafe_cal_2/dsafe2_result.json` |
| the five arm readouts + checkpoints (⚠️ single-disk) | `C:\Users\Admin\tanitad-data\rl-pilot\d2-*` (readout_before/after.json, config.json, summary.json, pilot_summary.json, ckpt_after.pt) — the readouts are copied into `raw/dsafe_cal_2/` here; the checkpoints (5 × ~420 MB) are not banked |
