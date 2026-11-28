# RESULT_PGT_BUILD: the instruments for PREREG_V8F_PGT (arm M2b-pgt), built 2026-10-22

**INSTRUMENTS READY, NOT LANDED. INTEGRATION NEEDED:** nothing below is in any repo. It is one edited library file (`stack/tanitad/models/v8f_mini.py`, +1 line), one edited queue (`v8f_mini_box2_queue.cmd`, +80/-1) and 8 new files. Base = tip `0a827845a05795000afbe17ea52c1ef1d475dc08` (read-only export); the lander must rebase onto the live tip. The chain REFUSES to start if the tip's `built_levers` lacks the one-line record (`stop "the tip's built_levers does not record p_gt_start"`), so `v8f_mini.py` must reach the remote tip before the chain is armed. Training was not run, box 2 and every GPU were not touched, the PREREG and every register were not edited, the lander was not used.

Evidence classes: MEASURED = ours, artifact path given (all paths repo-relative under PKGDIR). No model number is produced by this build; the three numeric blocks below are properties of the instruments.

## 1. What was built (priority order, each banked before the next)

| # | deliverable | what it is |
|---|---|---|
| 1 | `stack/tests/test_v8f_pgt.py` (NEW, 29 test items) | reachability argv -> cfg -> `p_gt_schedule` -> `model.loss(b, p_gt=...)` -> the masks in `v8f_mini._forward`, then the reader, the queue stages and the chain. 6 reach functions (7 items), 13 reader, 5 queue, 4 chain. |
| 1b | `stack/tanitad/models/v8f_mini.py` (+1 line in `built_levers()`) | records `p_gt_start` / `p_gt_end` in `selfcheck.json` and `summary.json` levers. Before this the G-DVB entries compared cfg to cfg and nothing recorded what was built. |
| 2 | `.../code/v8f_pgt_read.py` (NEW) | PRIMARY, M-LEAK, M-DK, the four families beside, stage logic (`s0` / `final`), every PREREG refusal, `--check-inputs`. Imports `v8f_dk_read`, `v8f_wc_read`, `v8f_tac_groups_read`; copies none. |
| 3 | `.../code/v8f_mini_box2_queue.cmd` (stages `gatepgt`, `trainpgt SEED`, `evalpgt SEED`; dose `PGTS=0.5` / `PGTE=0.0` defined ONCE) | one command string per command; the dry run echoes it, the real run executes it. |
| 3 | `.../code/box2_ops/chain_v8f_pgt.sh` (NEW) | seed 0 -> `v8f_pgt_read.py --stage s0` -> CONTINUE gate -> seed 1 -> `--stage final`. `--dry-run` prints the queue's own lines. Re-asserts the BUILT `p_gt_start` / `p_gt_end` = the queue's dose from the pulled `selfcheck.json` BEFORE training. ARMED ONLY ON THE PI's GO. |
| 4 | `.../raw/2026-10-22-pgt/` : `mutate_pgt.py`, `mutate_pgt.out`, `real_proofs.py`, `real_proofs.out`, `check_inputs_identity.json`, `SYNTHETIC_pgt_read_regression.json`, `SYNTHETIC_pgt_read_positive.json` | the proofs of section 2. The two SYNTHETIC files are NOT a result of any trained model. |

## 2. Proofs (all MEASURED, dev box, CUDA_VISIBLE_DEVICES=-1)

* **Tests.** `stack/tests/test_v8f_pgt.py`: 29 passed (about 55 s). In a tree with no research package: 7 passed, 21 skipped, no import error. The neighbouring v8F family (pgt, mini, mini_full, mini_full_report, wc, dk, cz, tac, t1, beta, beta_box2, smooth_read, launch_gate_v8f, yaw_rate_ruling): **265 passed, 5 skipped** (the 5 skips are in `test_v8f_beta_box2.py` x2 and `test_yaw_rate_ruling.py` x3, none in mine). Output: `family_final.out` (PKGDIR root).
* **Mutation proof** (`raw/2026-10-22-pgt/mutate_pgt.out`, bytes mutated = the delivered bytes, sha256 in its header): **194 / 194 mutants RED**, unmutated tree GREEN before and after in each of 8 parallel parts. By set: reach 21 (the masks, floors, `rc_ok`, `has`, the generator, the rate, `forward`/`loss` dropping `p_gt`, `built_levers`, the schedule, the trainer's `make_config` / loop / gate / default floor), reader 88, queue stages 49, chain 36. A reader/queue/chain mutant is tried on its own tests first and on the whole file if it survives them.
* **What the mutation run caught in MY first draft** (each fixed, then re-proved): 13 reader survivors (the PRIMARY bar could be F or 3F; "separated" was not needed at the PRIMARY; the mechanism could be ANY seed; each seed's bars could read seed 0's reference; M2b's baseline LAMBDA could read all windows; the per-term known-value checks and the identity control's bounds were shadowed by the DK reader's own identical-looking refusal), 6 queue survivors (a failing trainer / eval, or a trainer that exits 0 without `summary.json`, returned success; the grid and the checkpoint were not pinned), and 6 chain survivors seen before that run was stopped (string-presence tests only: the chain tests were rebuilt, the gate block is now RUN against 25 corrupted or edge-case `selfcheck.json` files and 30 load-bearing lines are pinned). 2 equivalent mutants were removed and are not counted.
* **Reader on the banked M2b evals** (`check_inputs_identity.json`, `real_proofs.out`): F = 0.1207 m; LAMBDA(M2b) = +0.2678 / +0.1667 m; S5 excess over P10 = +0.0442 / +0.0250; the PREREG's premises reproduce within 6e-5 (excess, F) / 6e-4 (LAMBDA). The paired instrument reads **exactly +0 [0, 0]** on all 13 contrasts x 2 seeds (PRIMARY, M-LEAK, M-DK, the S5 companion, present-slot MSE, oracle ADE6, 7 family cells).
* **Deliberate regression** (`SYNTHETIC_pgt_read_regression.json`; M2b's plans plus a forward ramp at 2 s: +2.0 m on the standard plan, +0.5 m on the all-latcons plan): ADE6 +0.557 / +0.779 m; M-LEAK +0.5000 / +0.5000 m; M-DK +11.79 / +10.94 pp, all separated and WORSE; VERDICT FAIL-NO-MECHANISM.
* **Positive control** (`SYNTHETIC_pgt_read_positive.json`; the standard plan moves 60 % toward the GT, the leak cut to 40 %): VERDICT SUCCESS (ADE6 -1.305 / -1.233; M-LEAK -0.1607 / -0.1000; M-DK -4.99 / -3.34 pp). The reader is not structurally unable to pass.
* **Printed = run.** Static (the trainer / eval appear once, inside the one `set`, and that string is executed once), dry (the printed lines carry the lever and are M2b's own gate / train lines plus the two lever flags) and REAL (`gatepgt`, `trainpgt`, `evalpgt` executed against fake trainer / eval scripts that record their argv: token-for-token equal to the printed lines; failures return rc 1, an unfinished arm rc 2, and each of the 4 evals is skipped alone when its npz exists).

## 3. Findings and spec notes (REPORTED; the PREREG was not edited)

1. **Built `p_gt` was not recorded.** Fixed by the 1-line `built_levers()` addition (above); the reader refuses a run whose `summary.json` OR `selfcheck.json` lacks it or differs.
2. **The `p_gt` draw governs three reads, not two:** S->T and S->O share one draw (`ck_for_t`, `ck_for_o`), T->O a second (`use_t`). The attribution of a PASS is still "the stack", not one pathway.
3. **A non-positive M2b symptom is unspecified.** The reader marks M-LEAK / M-DK NOT-APPLICABLE and NOT met, says so, and never triggered it on banked data.
4. **`:tacgtwo` cannot score M2b-pgt** (it requires TAC `tacpred` / `constraints` evals). `evalpgt` carries its own all-latcons run (E3, same flags as `:tacgtwo`'s `all-<g>` runs).
5. **M-DK's population.** S5 pairs the LEAD windows where the lead is inside BOTH arms' corridors for >= 2 of the 10 steps (min-TTC finite); an arm whose path never has the lead in its corridor leaves the pair, scored neither at risk nor safe. MEASURED on the banked evals (`SYNTHETIC_pgt_read_positive.json` rows.*.base.excess_ci): min-TTC is undefined on 38 / 21 of 743 LEAD windows for M2b (seeds 0 / 1) and 20 for P10; the M2b-vs-P10 pair drops 42 / 24 windows; the same excess over EVERY LEAD window reads +0.0417 / +0.0242 against the deciding +0.0442 / +0.0250. Convention kept (it reproduces the PREREG's premises); the reader reports `n_dropped_no_lead_in_path` and `alt_all_lead` beside it for the M2b-pgt contrast. A swerve-clear arm is not credited, which is conservative against a PASS.
6. **Exit-code defect in a cmd pattern, MEASURED** (`cmd /c script.cmd`): an `exit /b 1` inside a parenthesised multi-line block makes the script return rc 0 although the FAIL line printed; a single-line top-level `if errorlevel 1 (... & exit /b 1)` returns 1. The tip's inherited `:eval` stage has this shape (an `else (` block around `exit /b 1`). `mk_v8f_job.py` takes `RC=!ERRORLEVEL!` after `%RUNSAFE% cmd /c queue.cmd`, so a failed `:eval` may read as `rc=0` on box 2 (UNVERIFIED there; not fixed, outside my stages). My `:evalpgt` is flat for this reason and a test pins it.
7. **Guards added beyond the PREREG's refusal list, all in the refusing direction:** every other argv flag of an M2b-pgt run equals M2b's (one lever); the loop's own `train_log.jsonl` `p_gt` equals the declared schedule at every logged step (the measured proof that the flags reached the masks); an oracle eval is an oracle run (json mode); the all-latcons oracle carries the GT outside latcons; the banked premises reproduce; the identity control. Cosmetic: the `strategic_family` string now reads "NOT PAUSED".

## 4. UNVERIFIED (not run, or not runnable here)

* The chain has never run against box 2, the mirror, ssh or `scp`: only `bash -n`, `--dry-run`, the static / line-pin tests and the gate block executed on synthetic `selfcheck.json` files.
* The real `evalpgt` against the real `v8f_mini_eval.py` (fake scripts only). E3's flags are copied from `:tacgtwo`'s all-`<g>` runs.
* The box-2 wrapper's rc path for finding 6.
* A full reader run on a real M2b-pgt run: only synthetic arms derived from banked arrays. The families beside (LON / LAT cells, `_famfn`) are exercised by the real-data proofs only, not by pytest.
* Whether `L:/v8f_mini/runs/M2b_s*/train_log.jsonl` exists on the dev box: the chain falls back to NOT-CHECKED (M2b is 1.0 -> 0.2 by construction) and says so.
* The banked M2b all-latcons `.json` mode check: the file path is long on this tree; the check is skipped when the json is absent.
* The tip may have moved since `0a827845`.

## 5. Integration (what the lander must do)

1. Land together: `stack/tanitad/models/v8f_mini.py` (diff: `diffs/v8f_mini.py.diff`), `stack/tests/test_v8f_pgt.py`, `.../code/v8f_pgt_read.py`, `.../code/v8f_mini_box2_queue.cmd` (diff: `diffs/v8f_mini_box2_queue.cmd.diff`; CRLF is kept, tip blob is CRLF), `.../code/box2_ops/chain_v8f_pgt.sh` (LF), `.../raw/2026-10-22-pgt/*`.
2. Run `pytest -q stack/tests/test_v8f_pgt.py` on the landed tree before anything else; then the family above.
3. Do NOT arm `chain_v8f_pgt.sh` before the PI's go (it waits for `chain_v8f_wc.out`'s `ZZV8FMINI-DONE` by default).

## 6. Manifest (single-location = only in PKGDIR, in no repo and no pod)

PKGDIR = `C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/pgt/`. R = `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini`.

| artifact | location | note |
|---|---|---|
| `stack/tests/test_v8f_pgt.py` | PKGDIR:`stack/tests/test_v8f_pgt.py` | NEW, LF, single-location, sha256 2b31fa0af536 |
| `stack/tanitad/models/v8f_mini.py` | PKGDIR:`stack/tanitad/models/v8f_mini.py` | EDITED (+1 line), LF, single-location, f470191ffae1 |
| `R/code/v8f_pgt_read.py` | PKGDIR:`R/code/v8f_pgt_read.py` | NEW, LF, single-location, dfafcb4f94d1 |
| `R/code/v8f_mini_box2_queue.cmd` | PKGDIR:`R/code/v8f_mini_box2_queue.cmd` | EDITED (+80 / -1), CRLF, single-location, ece0a85fa119 |
| `R/code/box2_ops/chain_v8f_pgt.sh` | PKGDIR:`R/code/box2_ops/chain_v8f_pgt.sh` | NEW, LF, single-location, 2e7923c02a59 |
| `R/raw/2026-10-22-pgt/mutate_pgt.py`, `mutate_pgt.out` | PKGDIR:`R/raw/2026-10-22-pgt/` | NEW, single-location; .py sha 0b89cad02fa6, .out a8e09fe16fb6 |
| `R/raw/2026-10-22-pgt/real_proofs.py`, `real_proofs.out`, `check_inputs_identity.json`, `SYNTHETIC_pgt_read_regression.json`, `SYNTHETIC_pgt_read_positive.json` | PKGDIR:`R/raw/2026-10-22-pgt/` | NEW, single-location |
| `diffs/v8f_mini.py.diff`, `diffs/v8f_mini_box2_queue.cmd.diff` | PKGDIR:`diffs/` | against tip `0a827845`; line endings compared as LF |
| `RESULT_PGT_BUILD.md`, `family_final.out` | PKGDIR root | this file; the family run |
| scratch, NOT deliverables | PKGDIR:`export_a`, `export_b` (the working tree), `export_c*`, `export_d`, `export_q*`, `export_r`, `export_s`, `export_t1..8`, `real_proofs_run*`, `mutate_*.out`, helper scripts | never deleted, per the rules; `export_b` is the source of every delivered byte |
