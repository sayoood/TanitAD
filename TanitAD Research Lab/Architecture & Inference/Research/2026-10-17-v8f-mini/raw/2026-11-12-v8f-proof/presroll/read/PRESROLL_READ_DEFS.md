# PRESROLL read definitions (card PREREG_V8F_PROOF_ADD120, kit `presroll/read`)

`make_presroll_reader.py` writes `read_presroll10k.py` and `podread_presroll10k.sh` from `gapmargin/read/read_gapmargin10k.py` (LF text, sha256 prefix 0aab17c67eb60756) and
`gapmargin/read/podread_gapmargin10k.sh` (b44d8181d3894ae9); any other text is refused, an existing output is refused. Tests: `test_presroll_read.py`. Pull: `pull_presroll.sh`.
The engine calls, the estimator (paired two-level bootstrap, 2,000 draws, `--n-boot`), the argument rule, the levels control, the tests control, the engine control and the print path are those of the source.
The generated reader refuses `--selftest`. Settings in the order CONV, K0; arms PO, POA, GEc, A90, PL, RLc, RR; six pairs of the card. The controls print as `e0` to `e8` (= q0 to q8); the names stage refuses unless all nine are present and pass.

## Controls (exact definitions)
- **e0 wrapper (q0).** Args `--wrapper-p-file`, `--wrapper-p-record`, `--wrapper-p-check` and the source's `--wrapper-file`, `--wrapper-check`, `--wrapper-hashes`. Passes when: the study wrapper file's sha256 begins `b2a1d5f7242c3700`; the record has
  `pass` true, `variant_sha256` begins `b2a1d5f7242c3700` and equals the file's full sha256, `source_sha256` begins `28bc31f21ecb4ccb`; the last line of the check output is `ZZ CHECK-WRAPPER-P PASS`; as in the source, the unchanged wrapper file has the
  hash `28bc31f21ecb4ccb`, the banked check files exist, the pod check ends `ZZ CHECK-WRAPPER-V PASS` and its hashes name `28bc31f21ecb4ccb`; the run text of each of the eight new runs and of IDP names (`wrapper sha256 <64 hex>`, found by the model reader's
  `run_file_sha`) a hash that begins `b2a1d5f7242c3700`. Keys read in the record: `pass`, `variant_sha256`, `source_sha256`.
- **e1 identity run (q1).** Run folder `I10RQ50Hb20gGEcIDP_s0` against `I10RQ50Hb20gGEc_s0`. Dump folder `dump_I10<run>_s0_m10_causal_heads`, files `ep*.npz`: the two sets of names are equal and hold 141 (`--expect-dumps`); for every file the key sets are equal and every
  array has equal dtype, shape and values (NaN equal to NaN; `np.load(allow_pickle=False)`). Rule logs (CONV: the first log, plus the mid log if it lies on disk): same number of logs, same key names, every array equal in the same sense, **except text and meta keys: an array
  of string or object dtype (either side) and any key named `meta`** (a key that is text on one side and numbers on the other is a failure). The side file(s) of IDP (`<log minus .npz>_rd.npz`) must exist and have no column `p_used`.
- **e2 presence files (q2).** Arg `--pres-dir` (default pod `$V/preshead2/dump`): `D_H5_s<k>/pres.npz` and `D_H5_s<k>/pres_meta.json`. The sha256 of each file equals `dump_sha256` of its meta, and the meta key `seed` equals k. For each new run of model seed k: the first
  line names ` vis pres sha256 <hash>` (the 64-hex token; matched by prefix of at least 16 characters) equal to the hash of the file of head seed k, and ` vis-pres-mode or`; in addition the rule log meta (`vis_pres_file_sha256`, `vis_pres_mode`) names the same hash and mode.
  This covers the last clause of q3 (the new line names the file's hash and the mode).
- **e3 arguments (q3).** The source's argument rule on the first line of each new run against that of the banked run with the same gap file at its setting (GEc for PO, A90 for POA): every field both carry is equal (fields: form, replan, every, readout, kappa,
  preroll, exec, nav, grid, model seed, reader seed, ckpt; defaults for a reference without preroll, grid or nav as in note N1), and the rule log shows a share of rule steps with a lead above 0.
- **e4 identity of levels (q4).** Source control: banked levels (C, U, F) within 1e-4 for GEc, A90, PL, RLc, RR at CONV and at K0 (table `EXPECTED`, the card's numbers), by the same code path as the new cells.
- **e5 tests (q5), e6 engine (q6).** As the source's tests and engine controls (the two lead test outputs are `lead_tests/lead_pytest_queue.out` and `lead_pytest_read.out` in the card's pod root; engine: grid read and strategic grid of both settings).
- **e7 counts (q7).** Every new run and IDP has `T1_DONE` and 141 `ep*.npz` dump files; the queue done marker reads `ZZ PRESROLL10K DONE <time> all steps done`.
- **e8 rule logs (q8).** Per new run, over all rows of its rule log(s) and the side files, row for row (CONV: one log; K0: the log and `rule_log_mid_...`, both with side files). Presence file = the one of the run's model seed; row `off[ci] + j`, usable when `has_out` and p finite
  (frame inside 0..n-1 of the clip, clip index inside the file). Keys read: log `ci j present src_bad p_src g_src c_src`, side `p p_used ok`, meta `vis_gap_file_sha256 vis_closing vis_pres_counts vis_pres_file_sha256 vis_pres_mode`.
  (a) `p_used` = `max(p, pf)` (`np.fmax`) within 1e-6 where usable, else `p` (NaN equal to NaN); (b) `present == (p_used >= 0.5) & ok & ~src_bad`; (c) `p_src == p_used` within 1e-6; (d) the model reader's `w2_one` against the run's own gap file (PO: `--gap-file`, the base
  VGAPLE; POA: `--margin-dir`/VGAPLE_A90) with the meta naming that file's hash and `vis_closing` 1; (e) `vis_pres_counts` (JSON text in the meta: rows, with_file, no_file, raised, lowered) equals the recomputation over these rows (rows = number of side rows, with_file = usable rows,
  no_file = rows - with_file, raised = `p < 0.5 <= p_used`, lowered = `p >= 0.5 > p_used`) and lowered is 0 (recomputed and recorded). Printed with no name, per new run: share of rule steps with a lead, raised and rows (`rollout <key>: {...}` lines).

## Open decisions (the brief left them open)
1. **Raised rows in (d).** The source's `w2_one` checks a lead row with no gap-file output against the rows of the reference run RR (same setting, same model seed). A row the presence input raised (reader p < 0.5) has no RR lead row, so with no gap-file output it has nothing to
   be compared with and would fail by construction. Decision: such rows (raised, no output in the run's gap file, no RR row) are left out of (d) and counted (`n_raised_rows_without_reference`, printed in the rollout line). Raised rows that have a gap-file output stay in (d).
2. **Run-text wrapper hash.** e0 requires the run text of the new runs and of IDP to name the study wrapper's hash (`b2a1...`); the queue has to write the hash of the wrapper it runs in the first line, as the banked runs wrote theirs.
3. **Presence fields in the first line.** The wrapper only prints them; the queue must append ` vis-pres-file <path> vis pres sha256 <hash> vis-pres-mode or` (the text of `pres_fields`) to the run line; e2 reads them there.
4. **Where the runs lie.** The driver links the new runs, IDP and the banked runs from `$V/integ10k/t1/`, as the source does for its runs. The gap files come from `$V/gapmargin/gapfiles` (A90) and `$V/cropscale/gapfiles/VGAPLE.npz` (base); the presence files from `$V/preshead2/dump`.
5. **Driver selftest.** As the source driver, it runs the selftest of the reader of card ADD114 from `$V/presattr/kit_read` before the controls stage; the generated reader has no selftest of its own.
6. **Time stamp of the names.** `lead_tests/lead_pytest_*.out` times, the start marker and the done marker are used as in the source.
7. **Old control of the scaled gap-file record** (card ADD117's first control) is not part of this card's nine; the A90 file is tied to the runs through the rule log meta hash in (d). The base file's own hash is only recorded in `inputs`.
8. **Names.** Keys `<S>_PO_<M>`, `<S>_POA_<M>`, `<S>_POA_LABEL`, `<S>_OVERALL`; 16 names. The bound is written the same way in both routines: `U(POA) <= U(RLc) + 0.0045`.
