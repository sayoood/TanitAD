# FINISH_NOTES -- HNFP3 finish pass (2026-10-20 ~05:00): the two rc-1 files, the chain mutant, the three base-less files

CPU only, no commit / push / `git add` / register edit. `hnfp3/tmp/` (6.7 GB, not mine) was NOT touched. My scratch was `hnfp3/tmp_fin/` (about 125 KB; removed at the end; nothing else of mine on C:). C: free stayed 324-330 MB all pass.

## 1. Failure (ii): `refe/test_rls1_queue.py` -- PRE-EXISTING on the tip, a race in the test itself, nothing to do with HNFP3
* Same rc 1, same line, in BOTH trees: `test_rls1_queue.py:184` `assert time.time() - t_start >= 3.5, "returned before S1 was over"` (base 3.0 s, mine 3.4 s). `test_rls1_queue.py` and `rls1_pod.sh` are BYTE-IDENTICAL base vs mine (`cmp`); HNFP3 edits no rls1 file.
* Not a missing `raw/` or `taniteval/` path (nothing to export). Mechanism (MEASURED, `scripts/rls1_queue_race_probe.py`, 3 runs): `Queue.other()` (:111-117) starts the fake S1 chain with `exec 8>lock; flock -n 8 || exit 9` and NO retry, while `Queue.observe()` (:122-126) probes the SAME lock every 0.25 s with a
  perl `flock` (an exclusive try-lock). On this Windows box a perl start takes long (the machine is loaded), so the two collide: run 1 S1 got the lock and the test body passed (4.6 s), run 2 S1 exited rc 9 at once (`other rc 9`), the queue saw nothing ahead and returned after 1.6 s -> the :184 assertion. The chain script
  itself (`rls1_pod.sh`) retries this collision 6 times ("a collision with a probe is not a second RL-S1"); the TEST's fake S1 does not. Flaky on this box, not a defect of the shipped chain. Verdict: FAILS ON BOTH, pre-existing, does not block the landing.

## 2. Failure (i): `refe/test_hn_content_fp.py` -- NEW in HNFP3 as a suite failure (base rc 0, 81 PASS), a mutant-TABLE defect, now closed
* Not one surviving mutant: **six** mutants of hnfp2's table (12 PROBLEM lines in `suite_mine/refe__test_hn_content_fp.py.log`; the tail the brief quotes shows only the last). The harness requires each mutant snippet to occur EXACTLY once; HNFP3's code made six snippets stop being unique, and the harness reports that as `stayed GREEN: no arm can see it`:
  * 4 launch_gate snippets occurred 0 times: the `verify_token(...)` call and the 3 `data_manifest(...)` calls gained `ctx_anchor=...` (HNFP3's hand-through of the anchor).
  * 2 gate_token snippets occurred 2 times: `_content_listing`'s symlink-problem statement is reproduced verbatim by the NEW `_stat_listing`, and `fp_key`'s `n_problems` guard line by the NEW `dir-anchor` branch.
* Was `_content_listing` touched by HNFP3? **No** (function text identical to the tip, MEASURED by comparing the two function bodies; `content_check`, `content_fingerprint` identical too; `fp_key` is changed: it gained the `dir-anchor` branch). The symlink ARM (`U_a_symlinked_directory_is_a_problem_not_a_silent_skip`) was green all along (62 arms green); only the mutant could not be applied.
* The code HNFP3 added is covered by its own mutants: `_stat_listing` symlink -> `R4_a_symlinked_directory_is_a_problem_not_a_silent_skip` RED; the `dir-anchor-INVALID` guard -> the two-layer mutant, RED (`test_hn_ctx_anchor_matrix_run3.log`).
* Fix (test only, no production code): `refe/test_hn_content_fp.py` MUTATIONS re-anchored (`scripts/patch_fp_table.py`): the 4 launch_gate snippets carry the new argument; the symlink snippet is extended by the 4 lines that follow it in `_content_listing` (unique); the `fp_key` snippet names the `dir-content-INVALID` return line (unique). All 19 snippets verified to apply exactly once; re-run
  `logs/test_hn_content_fp_run2.log`: **ZZTEST_PASS 81 (62 arms + 19 RED mutants)** = the base's 81.
* What the symlink arm can and cannot test on D: / exFAT (no symlinks): it cannot create one; it makes the walk's own predicate say yes (`os.path.islink` patched for `units`), which tests the code path, not the filesystem's behaviour. A real symlinked directory is UNVERIFIED on this box (NTFS here needs a privilege; not attempted).

## 3. Not in the brief, found while checking HANDOFF step 2: the chain matrix and the gate matrix were NOT green
* `test_hnscreen_ctx_chain_matrix_run1.log` ended `ZZTEST_FAIL 1`: mutant "verify_pass_token: the start verify is not run with the anchor flags" was RED (by the cost arm) but the table also demanded `C_R2_...`, which cannot tell the modes apart (a content verify refuses a size change too). Over-specified expectation, not an undetected mutant; the R2 name was removed from that mutant's list with the reason in a comment
  (`test_hnscreen_ctx_chain.py`, table only; the 9 arms are untouched). Re-run `logs/test_hnscreen_ctx_chain_matrix_run2.log`: **ZZTEST_PASS 19 (9 arms + 10 RED mutants)**.
* `test_hn_ctx_anchor_matrix_run2.log` DIED at mutant 8 with `[WinError 112]` (disk full) and still carried a `.DONE`; its 7 completed mutants were RED. `launch_gate.py` was last edited at 22:59, AFTER matrix run1 (22:53), so run1 did not cover the final file. Re-run the whole matrix on the final tree: `logs/test_hn_ctx_anchor_matrix_run3.log`: **ZZTEST_PASS 72 (38 arms + 34 RED mutants)**.

## 4. The three base-less files (MINE results, `suite_mine.json`; NO base comparison)
The base run's runner printed `SKIPPED_NO_DISK` for them (`logs/runner_base.out`): C: had too little free space for their temp trees, and I was told not to run them. So these three have a MINE result and NO base result; a regression in them cannot be told apart from a pre-existing state.
| file | MINE rc | MINE count | secs |
|---|---|---|---|
| `refe/test_aux_s1_wiring.py` | 0 | ZZTEST_PASS 20 | 186 |
| `refe/test_gate_suite_prerun.py` | 0 | ZZTEST_PASS 144 | 285 |
| `refe/test_launch_gate.py` (full form) | 0 | ZZLG_FULL_PASS 1213 | 459 |
(Context: the static form of `test_launch_gate.py` IS compared: base 1167, mine 1168 = +1 for the new `HNSCREEN_SUITE` entry, confirmed.)
