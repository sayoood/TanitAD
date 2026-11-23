# DSCALE_EQP_READ_DEFS: what the reader of card PREREG_V8F_PROOF_ADD125 reads

Tier T0: a probe on cached tokens of logged frames. The reader `read_dscale_eqp.py` imports the reader of card ADD123 (`read_dscale2x.py`, sha256 prefix 0deb4744579f651c, asserted at import) unchanged, as a second instance with its step constants set to 61,568. Three runs: banked (card ADD45: `P16C_sK` runC, `P16U_sK` runU, `P16CSH_s0`, `P16USH_s0`), run of card ADD123 (`P16Q_sK` = `states/runQ`, 38,328 steps), new run (`P16E_sK` = `states/runQ` of the new output folder, 61,568 steps). Mode `controls` writes `DSCALE_EQP_CONTROLS.json` and prints booleans only; mode `names` refuses unless that file says `all_pass` for the same input hashes.

## Controls

**e0 inputs.** Reads the three done files (`parse_done`: first line is a marker; lines `ckpt <folder>/<arm>[/ckpt_last] <sha256>`), and for each of the 18 probes the record file `<arm>_eval.npz.json` key `ckpt_sha256`, which has to equal the done-file hash. Reads `<new>/precheck/PRECHECK.json` (and the planted, clock and evaluation-set records) through the imported precheck controls, and `checks.counts` for `union_manifest_clips` 14,512 and `corpus_ids` 4,512. Reads the new done file line `tok_probe_u_ck.py sha256: <64 hex>` and `runQ/CK_RECORD.json` key `tool_sha256`, both with prefix 5888527b5c37003b. The run-script hash (prefix a01c5ca8cb43a2c5) is read from a done-file line `pod_dscale_lead_2x.sh sha256:` if one exists; the done file of the run script writes none, so otherwise the sha256 of the landed script file (`EQP_RUN_SCRIPT`, default `../../add123/run/pod_dscale_lead_2x.sh`) is used. Reads `runQ/TRAIN_DONE.json`: `steps` 61,568, `arms` the four arms, `p5.<arm>.last_step` 61,568 on each.

**e1 same start.** Reads the second line (step 1, `first` true) of `<run>/runQ/TK16_sK/train_log.jsonl` of the new run and of the run of card ADD123; the terms `lead_sup_bce`, `lead_sup_gap` (required) and the three `*_mean` terms (if present) agree within 1e-4 on each of the four arms. Beside, never judged: the entries of key `hashes` in `runQ/r1_batches.json` of both runs that the two files share, equal entry for entry.

**e2 live probes.** The eight probes of E (`P16E_sK`, `P16C_sK`): the interval of gap R2 (`M2.head.r2_ci95`) has its lower end above 0 and the presence AUROC (`M4.auroc`) is at least 0.85 (the banked function `d3_corpus_live`).

**e3 dead probes.** `P16CSH_s0` and `P16USH_s0`: the interval of gap R2 includes 0 or lies below it and the presence AUROC is at least 0.80 (`d5_permuted_dead`).

**e4 complete runs.** The imported form of control q4 with the last step set to 61,568, on the four new arms: done marker line `DSCALE_2X_DONE`; train log header total 61,568, `w_clos` 0, evaluation guard with 0 shared ids, no clip list; the lead term of the last 1,000 steps below that of the first 1,000; gap-labelled frames above 0 at every logged step and in every full batch record; last logged step 61,568; the state record has grid, token count, width 16, `input_norm` ln, `step` 61,568, the arm name, and its `ckpt_sha256` equals the `ckpt_last` line of the done file. The states at 19,164 and 28,746 steps are not read.

**e5 tests.** The planted-difference control of the two-level reader on the labels of the evaluation windows (a gain at every seed reads above 0; a gain at one of four does not; four identical pairs read exactly 0; the swapped order reads the mirror), and the pytest output file named by `--test-out`: a `N passed` line, no failure text, modification time older than `--test-before`.

**e6 one evaluation set.** Every probe's state file lists the same clip ids in the same order (`clips` in `<arm>_eval.npz`), every probe is scored on the same window count (`M2.head.n`), and the window context counts the 429 scored windows of the card.

## Measures (names mode only)

**E.** Per seed K, gap R2 of `P16E_sK` minus that of `P16C_sK` on the same windows, and the same for the gap error in metres; the interval is the two-level bootstrap of the imported reader (seed indices, then episodes, `--B` draws, rng 0). Its upper end is printed as the gain the read leaves open.

**T.** The same with `P16E_sK` minus `P16Q_sK` (4.0 against 2.5 passes).

**Beside.** `P16Q_sK` minus `P16U_sK` (2.5 against 1.25 passes) next to the banked value of row 192; and S of card ADD123 (`P16Q_sK` minus `P16C_sK`) recomputed. Nothing beside names anything.

## Names

`LEAD-READ-EQP-SCALES-GAP`: interval of E on gap R2 above 0, mean above 0, at least three of four seeds above 0, and the mean gap error of `P16E` at most 0.90 times that of `P16C`. `LEAD-READ-EQP-R2-ONLY`: the same R2 clauses without the error bar. `LEAD-READ-EQP-WORSE`: the interval below 0, mean below 0 and at least three of four seeds below 0. `LEAD-READ-EQP-FLAT`: none of the three.

## Outputs

In `--out-dir`: `DSCALE_EQP_CONTROLS.json` (`controls.e0` to `e6` with `pass`, `all_pass`, `failed`, `input_hashes`), `DSCALE_EQP_RESULT.json` (`names`, `measures` with `E`, `T`, `beside`, `probes`, `input_hashes`), `DSCALE_EQP_VERDICT.md`. A second call writes `..._2`. Exit 0 all pass, 2 a control failed, 3 refused.
