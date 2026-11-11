# TF-2 HANDOFF (updated 2026-11-12; TF-2 is COMPLETE)

**State.** TF-2 ran and is analyzed; all controls PASS. See `RESULT_TF2.md` and `tf2_result.json`.
* PRIMARY M (ungated margin shaping): **C, REFUTED IN THIS FORM**.
* SECONDARY CERT (DAC-certified fallback): **NOT SHOWN**. DAC-zero went from 10 to 0 and PDMS reached 95.18, but the CI
  lower bound is 0.000 against a bar of > 0.
* Probe P-S20: **S20-NO**.

The rows are banked in `rows/` (sha256 in RESULT_TF2.md), in ONE place only, so the MM lands them.

**Deletion log (CLAUDE.md rule 4).**
* Deleted: `F:/tanitad_scratch/tf_design/tf2w`, 11,552,352 B. It held:
  * mirror-tip exports: refe/, eval/, p5s/;
  * `dz.txt`, text extracted from the Library PDF;
  * `out/`: the TF-2 rows, banked and verified by sha256 before the delete;
  * smoke files and the finisher, which is banked as `finish_s2.sh`.
* All of it is re-creatable. `F:/tanitad_scratch/tf_design/w` is TF-1's scratch, owned by another session, and was not
  touched.

**Next arm: TF-2b, CERT on fresh held-out logs, about 1 CPU-h.**
1. List the rank-0 held-out navtrain samples **without** a banked 10 Hz target. That is 1,705 − 1,018 = 687 samples,
   in logs other than D1's 11. Source: `PK/raw/2026-10-04-lane-discipline/teacher_check_heldout.jsonl` (the `HELD`
   constant of `eval/p5_target_labels.py`), minus the keys in `p5s_cubic.npz`.
2. Roll them with the registered teacher protocol: `PK/eval/teacher_navtest_seam.py --heldout` through `eval/run_p4.py`,
   10 Hz, CPU, about 1.4 s per row. These are torch processes, so check that FreeVirtualMemory is > 10 GB first, and run
   ≤ 3 of them.
3. Write a `p5s`-format npz: `logs`, `tokens`, and `labels20_t20_tc20_tw` with column 0 = t20 and column 1 = Tc20s
   from `iterate20` cubic, so that C2 stays meaningful.
4. Pre-register before any row. Use the same CERT bar (DAC-zero ≤ 0.4 % of rows, ΔPDMS CI lower bound > 0) and the same
   estimator. The new logs are the clusters.
5. Score with `tf2_score.py --w <dir holding refe/ eval/ p5s/>`. Export refe and eval from the mirror tip with
   `git -c core.autocrlf=false archive`.

If TF-2b reads WORKS, re-resolve `../PREREG_TF_N1_DRAFT.md` §4 (the candidate becomes CERT) before freezing.

⚠ The three WMI launches (`launch_tf2.cmd`) died silently in this session. The cmd logs showed `^C`, there was no
traceback and no DONE line, and the cause is UNVERIFIED. The run succeeded as session background tasks.
