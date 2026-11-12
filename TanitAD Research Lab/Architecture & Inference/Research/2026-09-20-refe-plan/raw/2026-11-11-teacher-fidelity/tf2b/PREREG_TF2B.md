<title>TF-2b pre-registration</title>

# PREREG TF-2b: the DAC-certified teacher label (CERT) on fresh held-out navtrain logs

Written 2026-11-11 (UTC) by the TF-2b agent, **before any TF-2b rollout row, label row or score row exists**. Its sha256 is
recorded in `PREREG_TF2B.sha256` and in `RESULT_TF2B.md`. Nothing here may change after the first fresh row is written; a
later change is an amendment with its own time and reason. (Pre-row work that touched only TF-2-population rows: a 3-row
smoke of the roll script in control mode, max |dxy| 0.0 m against the bank, 3.3 s/row.)

## 1. Question

TF-2 (`../tf2/RESULT_TF2.md`, `../tf2/PREREG_TF2.md` sha256 `65189b98...`) found that CERT, the first of
[Tc20s, plain teacher T, margin-shaped M] whose LQR-tracked footprint passes NAVSIM's drivable-area rule (map + t0 state only),
scored PDMS 95.18 [94.02, 96.20] with DAC 100.0 on the 1,018 held-out navtrain targets of 11 logs. Its gain over the
baseline Tc20s, +0.82 [0.00, +1.53], was **NOT SHOWN**: the CI lower bound is exactly 0 and 9 of the 11 TF-1 zero rows sit in
2 logs. TF-2b re-runs CERT on **fresh** logs, none of which entered TF-2, so that the gain is tested on a draw of logs that
was not used to build it.

The CERT rule, the arms, the margin-shaping constants, the tracker and the scorer are **exactly TF-2's** (`tf2_score.py`
sha256 `3ccd5694...`, unchanged). Only the population and the controls' scale change.

Admissibility is as in TF-2: CERT reads the static map, the t0 ego state, NAVSIM's public LQR tracker and the `dac` field of
the tracked arms. It reads no future agents and no NC, TTC or EP verdicts. It is privileged map geometry at t0, admissible for
teacher LABELS (PI, 2026-08-03). It is a label-quality statement, not evidence about checkpoint identity.

## 2. Population, tier, estimator

* **Population (frozen).** The 687 rank-0, step-0 held-out navtrain samples that are in
  `teacher_check_heldout.jsonl` (the `HELD` set of `eval/p5_target_labels.py`, 1,705 samples) and **not** in
  `p5s_cubic.npz` (1,018 samples, 11 logs). They sit in **13 logs, none of them in the 11 of TF-2**. List:
  `tf2b_samples.txt`, one `log:token` per line, sorted, sha256
  `d2dcfae97cb9430de286c6b5a9d8f7b283610a6c5ee7fe391f5c3b3924c38d56`. Navtrain held-out, **not navtest**. The K ledger is
  unchanged.
* **Cluster sizes (known before any row).** One log holds 236 of 687 rows (34 %), then 96, 89, 73, 43, 35, 30, 21, 18, 12, 12,
  11, 11. The bootstrap is therefore dominated by one cluster, and the leave-one-log-out point estimates are reported beside
  every CI (descriptive).
* **Tier.** NAVSIM v1.1 PDMS, non-reactive pseudo-simulation, vendored navsim @ 3e8291b, one PDMScorer call per
  (log, token) with all arms in one call, as in TF-2.
* **Rollout.** The registered teacher protocol: `eval/run_p4.py --hz 10 --script tf2b_roll.py`, which calls
  `teacher_navtest_seam.roll` (rank 0, step 0, per-token scenario, 10 Hz, CPU, `CUDA_VISIBLE_DEVICES=-1`). `tf2b_roll.py`
  sha256 `7591105b59395942cf67719abf8a7acee33be54ca1f3b96a1fde94b331d88666`.
* **Labels.** t20 = the rolled 20 poses at 0.2 s. Tc20s = `iterate20` cubic of t20 (`tf2b_mknpz.py`, sha256
  `34fcfa494d4ed935c43e1aace9f06f704c0a30e5858c9dd7005f27c5dcabd38d`), written in the p5s npz format.
* **Estimator.** Log-cluster bootstrap over the new logs, 10,000 draws, seed 20260927, 95 % percentile interval, ratio mean
  (`tf2b_analyze.py`, sha256 `54ade5f19bdf213f21f588b3f6bd7b86d3190eb3aa5213b59435ff0760802bc5`). It answers *another draw of
  LOGS*. Labels, tracker and scorer are deterministic, so it does not speak to training or inference variance. The verdict
  reads the **unrounded** lower bound.

## 3. Arms (TF-2's, unchanged)

T_tracked, Tc20s_tracked (**baseline**), Tc20s_re_tracked, M_tracked, Mw_tracked (deliberate regression), CERT_tracked
(**the arm under test**), H_tracked, T_sub20_tracked, T_transparent, M_transparent. One scorer call per row with all arms in it.

## 4. Availability rule (added because the DBs are a known risk)

The rollout needs each log's nuPlan `.db`. At writing time the local index finds **2 of the 13 logs** (325 of 687 rows;
`2021.07.09.20.59.12_veh-38_03292_04331`, `2021.06.14.18.13.35_veh-26_04547_04710`). The other 11 logs (362 rows) are not on
this box. Committed in advance:

* The verdict is read **only if rows exist for ≥ 680 of the 687 samples (C6)**. Otherwise the outcome is **NO VERDICT** and
  nothing below is read as WORKS or NOT SHOWN.
* A run on fewer logs (for example the 2 present now) is **DESCRIPTIVE ONLY**: with 2 clusters the bootstrap is not an
  estimator of anything, and no CI from it may be quoted as a result.
* The sample list is **not** shrunk to fit what is on disk. Any other population is a new pre-registration.

## 5. Controls (all must pass, or no verdict is read)

* **C0, rollout reproduces the bank.** `tf2b_roll.py --control` re-rolls 30 banked samples (every 34th of the sorted 1,018) and
  compares with the banked t20. PREREG_P4 C3 bar: 0 errors, median max-|dxy| ≤ 0.05 m, ≥ 90 % of samples ≤ 0.25 m. And
  `iterate20` cubic on the banked t20 of those 30 reproduces the banked Tc20 label within 1e-9 m.
* **C1, scoring identity.** The same 30 control rows scored with `tf2_score.py` equal TF-2's banked rows bit for bit on every
  sub-score of every arm (max abs diff 0.0).
* **C2, derivation identity.** On the fresh rows, Tc20s_re equals the Tc20s label within 1e-9 m and the tracked scores are
  identical.
* **C3, unshifted rows.** Where M's shift is zero, the M label equals Tc20s_re bit for bit.
* **C4, deliberate regression.** Mw has ≥ 7 more DAC-zero rows than Tc20s_tracked. (TF-2's bar was ≥ 10 of 1,018; scaled by
  687/1,018 and rounded up; TF-2 read 154 against 10.)
* **C5, liveness.** M's label differs from Tc20s_re on ≥ 68 rows (TF-2's ≥ 100 of 1,018, scaled and rounded up; TF-2 read 459).
* **C6, coverage.** Rows exist for ≥ 680 of 687 samples (section 4). Any missing sample is listed with its error.

## 6. Outcome, committed in advance

**CERT − Tc20s, tracked, paired, on the fresh rows.**

* **WORKS** if both hold: CERT has **≤ 2 DAC-zero rows** (0.4 % of 687 rounded down; TF-2's bar was ≤ 4 of 1,018), and the
  ΔPDMS 95 % CI lower bound is **> 0** (unrounded).
* Otherwise **NOT SHOWN**. A DAC-zero count above 2, or a lower bound ≤ 0, both give NOT SHOWN. A CERT point estimate < 0 is
  reported as such, with the same label.
* If the controls fail or C6 fails: **NO VERDICT**.

**Consequence for N1** (`../PREREG_TF_N1_DRAFT.md` §4), committed now:

* WORKS ⇒ N1's candidate teacher label becomes **CERT**. Re-resolve §4 before N1 is frozen.
* NOT SHOWN or NO VERDICT ⇒ N1's candidate **stays Tc20s**, and CERT is reported as a descriptive arm only.

## 7. Descriptive, not decisive

Reported next to the outcome and never used to decide it: the per-arm table (PDMS, DAC, NC, EP, TTC, C, PDMS/DAC/NC-zero
counts); CERT − H, CERT − T_transparent, M − Tc20s, Mw − Tc20s; the EP on rows alive in both CERT and Tc20s; which logs
carry the CERT − Tc20s difference; the leave-one-log-out point estimates; and a **pooled** CERT − Tc20s over TF-2's 1,018 plus
these 687 (24 logs, 1,705 rows). The pooled interval reuses TF-2 rows that already informed the hypothesis, so it is
exploratory and cannot upgrade a NOT SHOWN.

## 8. Cost

CPU only, ≤ 2 processes of my own, `CUDA_VISIBLE_DEVICES=-1`. MEASURED rollout 3.3 s/row (smoke) against the brief's 1.4, so
687 rows ≈ 38 CPU-min, plus scoring about 2.3 s/row (TF-2: 0.65 CPU-h for 1,018 rows).
