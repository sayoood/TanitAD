# DSCALE2X_READ_DEFS: the measures and controls of read_dscale2x.py as implemented (card PREREG_V8F_PROOF_ADD123)

Tier T0. The banked reader of card ADD45 (`dscale_lead_read.py`, wrapper `dscale_lead_read_n5.py`) is imported and never edited. B 2,000 and the banked rng seeds come with its functions
(`D.boot_diff`, `D.two_level`: rng 0; episode draw C10 of the window context). Kit folder: `DSCALE_LEAD_KIT_DIR` (default two folders above the reader).

## Probes (function `probe_specs`)
Banked 11 (`D.ARM_MAP`): P16C_sK, P16U_sK (19,164 steps), P16U2_s0 (38,328), P16CSH_s0, P16USH_s0. New 12: `P16Q_sK@STEP`, STEP in 19164 (folder `runQ_s19164`), 28746 (`runQ_s28746`), 38328 (`runQ`);
`P16Q_sK` is the 38,328 probe. Per-probe read: `tokprobe_read.read_arm` (through `read_probe`, default `D.TKR.read_arm`, controls=False). A probe that cannot be read fails the controls that need it.

## Measures (mode names only)
* M2 / gap error per probe: `D.summ_one` (gap R2 with its episode interval; mean absolute gap error). M4: the AUROC row of `read_arm`.
* S: `D.two_level_pair(ctx, outs, measure, a_kind="Q", b_kind="C")` for measure M2 and MAE: per seed K the paired difference `boot_diff(P16Q_sK, P16C_sK)` (Q minus C), then `D.two_level` (seeds, then episodes).
* L: per seed `boot_diff(P16Q_sK@38328, P16Q_sK@28746)` on M2, `D.two_level`.
* F: `|r2(P16Q_s0) - r2(P16U2_s0)|` (r2 from `D.summ_one`); beside it the signed difference and `RP.ci` of the `boot_diff` vector of the same pair.
* Beside: gain 19,164 to 28,746 (same two-level form); P16Q minus banked P16U (M2 and MAE); fall values (first and last lead term of `D.p5_u`) per probe; M4 per probe.

## Names (function `decide`; the order of the tests is the card's)
* above = `D._above(S_M2)`: mean > 0, lower end > 0, at least 3 of 4 seeds above 0. mae_ok: mean over seeds of gap error of P16Q <= 0.90 x that of P16C. mean S (M2) > F.
* dose = upper end of the L interval <= +0.020.
* SCALES-GAP = above and mae_ok and mean S > F (dose not asked). Else, if dose fails: SHORT-OF-TRAINING. Else WORSE = `D._below(S_M2)` (mean < 0, upper end < 0) and mean S < -F. Else FLAT.

## Controls (function `run_controls`; each a dict with `pass`)
* q0: for each of the 11 banked probes, the `ckpt <folder>/<arm> <sha>` line of the banked done file equals `ckpt_sha256` of the state record `<arm>_eval.npz.json`; and in `<new-run>/precheck/` the records
  PRECHECK, CLOCK, EVALSET with their PLANTED records pass through `D.precheck_controls`, `D.clock_controls`, `D.evalset_controls` (all entries must pass; an absent record fails).
* q1: `runQ/r1_batches.json` equals the banked `runU2/r1_batches.json` (whole-file equality; differing top-level keys listed); the first record after the header (step 1) of `runQ/TK16_s0` and banked
  `runU2/TK16_s0` agree within 1e-4 on every term of `lead_sup_bce, lead_sup_gap, lead_sup_bce_mean, lead_sup_gap_mean, lead_sup_clos_mean` present in either (the first two must be present). The equality of the two
  full logs is printed (`printed_not_judged_full_logs_equal`) and judged by nothing.
* q2: `D.d3_corpus_live` on P16Q_sK (38,328) and P16C_sK, K 0 to 3: interval of gap R2 above 0, AUROC >= 0.85. Pass flags only are written.
* q3: `D.d5_permuted_dead` on P16CSH_s0 and P16USH_s0: interval includes 0 or lies below, AUROC >= 0.80. Pass flags only.
* q4: per new seed: the header checks of `D.p5_u` (total 38,328, w_clos 0, no shuffle, evaluation guard with 0 shared, no clip list) all hold; the last record is step 38,328; the minimum of n_gap over the
  full-batch records (n_pres >= 0.5 x median, `N5.gap_clause_n5`) is above 0; for each of the three states `D.check_records_u` passes (step, arm, probe form) and the record's `ckpt_sha256` equals the done file
  (`ckpt_step runQ/<arm> <step> <sha>`; for 38,328 also `ckpt runQ/<arm> <sha>`); the first line of the new done file starts with `DSCALE_2X_DONE`. Under `beside`: first and last lead term, the frozen fall
  clause verdict and the share of short records, judged by nothing.
* q5: `D.control_d7` passes; the pytest output file (`--test-out`) has a last summary with N passed (N > 0), no failed, error, skipped, xfailed, xpassed, deselected, Traceback, and its modification time
  is earlier than `--test-before` (the driver sets it to the start of the read when the argument file has none).
* q6: for all 23 probes the array `clips` of the evaluation state file equals that of the first probe (same ids, same order), `R["M2"]["head"]["n"]` is equal across probes, and the context's scored count
  equals the card's 429.

## Modes and files
`controls` writes `DSCALE2X_CONTROLS.json` (key `controls`, `all_pass`, `failed`, `input_hashes`) and prints `ctl <name> PASS|FAIL` and one `ZZ` line. `names` needs that file with all_pass true and the same
input hashes (sha256 of both done files, the batch logs, CK_RECORD and TRAIN_DONE, the precheck records, every state record and the four new train logs), then writes `DSCALE2X_RESULT.json` and
`DSCALE2X_VERDICT.md`. A file name that exists is never overwritten (`_2`, `_3`). Exit 0 / 2 / 3.
