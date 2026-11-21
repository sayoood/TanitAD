# BRIEF: finish HNFP3 (HNSCREEN-FP-COST (b′)) to landable (Master Mind, 2026-10-20 ~04:50)

FIRST read and obey, as binding: `C:/Users/Admin/par18/preamble.txt` and `C:/Users/Admin/par18/common.md`.
- Then read `C:/Users/Admin/par18/hnfp3/HANDOFF.md` and `RESULT.md`. The work dir is `C:/Users/Admin/par18/hnfp3/`.
- CPU only (`CUDA_VISIBLE_DEVICES=-1`).
- C: has about 0.4 GB free. Do NOT re-run `test_launch_gate.py`, `test_gate_suite_prerun.py` or `test_aux_s1_wiring.py`: they are the disk-heavy ones, and the base run skipped them for lack of disk.
- Small files only. Set TEMP / TMP to a NEW empty dir under your work dir, and remove only what YOUR runs create there.
- **Do NOT delete `hnfp3/tmp/` or anything you did not create.** That decision is the PI's.
- If C: drops below 60 MB, stop and write the HANDOFF.
- Never commit, push or `git add`.

## State (MEASURED by the MM, ~04:45)
- `run_rest.cmd` finished (`logs/run_rest.ALLDONE` at 01:58).
- `suite_mine.json`: 30 rows, 2 with rc 1:
  - (i) a mutation self-check: "PROBLEM mutation 'gate_token._content_listing: a symlinked directory is silently skipped' stayed GREEN: no arm can see it";
  - (ii) a `refe/test_rls1*` file. It also fails in `suite_base.json`, so it FAILS ON BOTH.
- `suite_base.json`: 25 rows; 3 files `SKIPPED_NO_DISK` (named above), no `.DONE`.

## Do
1. **Failure (ii):** read its tail in both suites. If it is a missing `raw/` or `taniteval/` path, export it into BOTH trees with `scripts/export_extra.py` and re-run that ONE file in both. If it is anything else, diagnose it to `file:line` and say whether it pre-exists on the tip, testing against `base/`.
2. **Failure (i):** is this mutant NEW in HNFP3 or pre-existing? Was `gate_token._content_listing` touched by HNFP3? A surviving mutant in code HNFP3 changed is a gap HNFP3 must close: add the arm that sees it (on exFAT, D:, symlinks cannot exist, so state what the arm can and cannot test on this box). Then re-run that mutation file only. If the mutant pre-exists, say so with the evidence; it does not block the landing, but it gets named.
3. **The 3 base files never ran:** their MINE runs are in `suite_mine.json`. Report each one's MINE result, and state plainly that NO base comparison exists for them, with the reason (disk). Do not run them.
4. Run `scripts/finish_pkg.py`, then fill the `RESULT.md` blocks it leaves, per HANDOFF steps 1-3. The verdict line must name what is UNVERIFIED (the 3 base files; anything from 1-2).
5. Rebuild `out/` and `out/paths.json`.

## Hand-back (250 words or fewer)
- What each failure is, and whether it is new or pre-existing.
- What you changed.
- The final counts.
- What stays UNVERIFIED.
- The manifest path and its count.
