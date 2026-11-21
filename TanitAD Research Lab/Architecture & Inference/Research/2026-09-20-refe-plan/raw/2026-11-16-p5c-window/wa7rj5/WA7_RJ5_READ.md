# W-A7-RJ5 read (PREREG_P5C_A2 section 6 definition/estimator/bar, unchanged; PREREG_P5C_A3 section 5). REFe agent, 2026-11-17

## Formal outcome row: INVALID (control (i') could not run: one input missing). Nothing was fixed.
Every other control passed. The Delta, read against the committed bar, is a W-A7-PASS-shaped result, but it is NOT a W-A7-PASS until (i') is run.

## Result (MEASURED, `run/w_a7_result.json`; units PDMS x 100 = pp; n = 1,018 rows, 11 logs)
Estimator: paired log-cluster bootstrap over the 11 logs, 10,000 draws, seed 20260927, ratio mean, 95 % percentile (TF-1/E2 code verbatim). It answers "would another draw of logs say this?" only.
Bar (A2 section 6): REBUILD iff CI upper bound < 0 AND point <= -0.10 pp.

| label | PDMS no A7 | PDMS with A7 | Delta [95 % CI] pp | rows Delta != 0 (neg / pos) |
|---|---|---|---|---|
| **P5-R (decides)** | 95.8372 | 95.8239 | **-0.01337 [-0.09283, +0.12245]** | 513 (297 / 216) |
| Tc20s (reported only) | 95.0799 | 94.3527 | -0.72715 [-1.88232, -0.05672] | 515 (365 / 150) |

- P5-R: the CI upper bound is +0.122 (not < 0) and the point is -0.013 (not <= -0.10): both REBUILD conditions fail by a wide margin. Had (i') passed, the row would be **W-A7-PASS**.
- Context only, not a decision (Tc20s decides nothing): Tc20s Delta is separated and large (-0.73 pp). The bank-traj W-A7 read had Tc20s -0.313 [-1.191, +0.184] and P5-R -0.030 [-0.098, +0.055] (context).
- Raw-fallback label rows in this run: P5-R 0, Tc20s 0.

## Membership claim (A3 section 5) VERIFIED, [M `run/w_a7.log`, `run/prep.json`]
- 1,018 / 1,018 rank-0 keys over 11 logs have a join row with status `rj5`; every join row's `bank_traj_sha256` and `bank_key` equal the row's own bank traj sha (the driver refuses otherwise); every `rj5_traj` matches its recorded sha.
- Sources: local 490, B 393, A 135 (equals A3 section 5). Fed trajs moved vs bank: 842; identical: 176.
- Join used: `F:/tanitad_scratch/rj5nt/join_c/rj5_bank_join.jsonl.gz`, sha256 `4670f88a...d739` (the C1 build's join, LABEL_SELECTION.md). The label table sha256 `3b198314...37a2`, sidecar `cc6e9bf0...`, both verified on disk.
- **A3.3 fallback row, token `d06ff3cbd9025da7`: NOT among the 1,018** (0 matches in the population file). So the W-A7 population has no P5-R-to-Tc20s label swap.

## Controls
| control | result |
|---|---|
| coverage | PASS: 1,018 / 1,018 scored, 0 error rows |
| (i') C1 table reproduction (pinned gates' `_p5r_vs_control`, R-TOL-1/2/3) | **NOT RUN, counts as FAIL under the driver's rule**: needs `--c1-records` = the pod's `labels_rekeyed.jsonl` (150,603 records; out_sha256 `90318b56...5c5e`), which exists only on the pod (`/workspace/data/p5c_label_a33/`); it is not on HF, F:, E: or the mirror. The brief forbids the pod. |
| (i'') instrument validity, bank-traj run's control (i) VALID | **PASS when checked on its own** (`bank_run_validity_check.json`: outcome W-A7-PASS, control (i) max abs diff 0.0 over n = 1,018, tol 1e-6). The driver printed it as failed only because it short-circuits both controls when `--c1-records` is absent (`run/W_A7_RESULT_asrun_INVALID.md`). |
| (ii) RED arm (+0.5 rad on index 19) | PASS: PDMS changed on 755 rows (need >= 1) |
| (iii) identity (A7 on h19 := h18 label) | PASS: plans and scores bit-equal on 1,018 / 1,018 |
| (iv) determinism (rebuilt parts, rescored) | PASS: 102 rerun rows, 0 not bit-identical |

### Supplement to (i'), NOT the pinned control (`ctl_i_prime_partial.json`)
- Label values: this run's no-A7 P5-R labels vs the C1 table labels on the 1,018 keys: all 1,018 keys present; per-row max abs diff median 9.0e-6, p99 1.53e-3, max 0.058; 10 bit-equal. The R-TOL-1 thresholds (median <= 1e-3, p99 <= 5e-3) are met.
- sim8: 490 / 490 equal on the rows that the dev-box local RJ5 records cover (context). The other 528 rows, and the TB-6 ratios, need the pod records.
- This does not substitute for (i'); it only says the labels are reproduced.

## Next step (one, cheap)
Copy `labels_rekeyed.jsonl` from the pod to the dev box (PI/MM decision, since the brief forbids the pod; verify sha256 `90318b56...`), then re-run only the analysis (a few seconds, scores are banked in `run/scores.jsonl`):
`python w_a7_read.py --run run --stage analyze --traj-source <join> --expect-join-sha256 4670f88a... --bank-run-result <bank w_a7_result.json> --c1-table <table> --c1-records <labels_rekeyed.jsonl>`
(the driver gate-checks the gates module blob `5c2f49cf`, present at `.../p5c/a1/p5c_table_gates.py`). The Delta will not change; only the row can.

## Instrument pins and run facts
- Driver: A3.1 pin blob `3b02634e` (`w_a7_read.py`, verified equal to the mirror tip `c937b4c6` copy); transp_score `3cbdd118`; all driver pin checks passed. Gates module for (i'): A3.1 pin `5c2f49cf` (A3.3 changed R-A only; R-B and R-TOL are unchanged).
- Dev box, CPU only (`CUDA_VISIBLE_DEVICES=-1`), 1 thread, serial, TMP on F: scratch, driver's free-virtual-memory gate (5 GB; it read 10.5 GB). Labeller 663 s, scoring 1,060 s.
- No commit, no push, nothing staged (the E: tree has git "dubious ownership"; the hold is outside the repo).

## MM addendum (2026-11-17 ~00:45 local): control (i') run, the formal row is W-A7-PASS
- **The missing input, now supplied:** `labels_rekeyed.jsonl`, sha256 `90318b56cb88e55ad15dccbebf3b0ec843fdd17773f1811437650d88184b5c5e`, 150,603 records.
  - Pulled by the MM from the pod's `p5c_label_a33/rekeyed/` as gzip (sha256 `42727203...`); the decompressed copy was verified against the sha above.
  - Until today it lived only on the pod. It is now also banked on HF `Sayood/tanitad-refe-scorer1p6` `evidence/2026-11-17-p5c-c1-records/` (private, LFS sha verified).
- **What was re-run:** the analyze stage only, with the command in `HANDOFF.md` (`run/w_a7_analyze_with_c1records.log`). The as-run INVALID result is kept as `run/w_a7_result_asrun_noC1records.json`.
- **Controls:** all pass, with coverage 1,018 / 1,018, 0 rows without a record, and 0 table labels differing from their record.
  - coverage;
  - (i') C1-table reproduction;
  - (i) bank-run instrument validity;
  - (ii) RED arm, 755 rows changed;
  - (iii) identity, 1,018 / 1,018 bit-equal;
  - (iv) determinism, 102 rerun rows.
- **Delta (unchanged, as it must be):** P5-R -0.01337 pp [-0.09283, +0.12245] (decides); Tc20s -0.72715 [-1.88232, -0.05672] (reported).
- **Outcome row: W-A7-PASS**, against the A2 section 6 bar as written. The REBUILD side needs an upper bound < 0 AND a point <= -0.10, and neither holds. MEASURED; the estimator is the paired log-cluster bootstrap, 10,000 draws, seed 20260927 (another draw of logs only); n = 1,018 rows, 11 logs.
