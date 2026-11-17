# BUILD: DSCALE10K block-2 deciding read (built, not launched)

Folder `F:/TanitAD-scratch/datascale/read2/`. The live `read/` was not touched (read only: nothing written there). Nothing trained, launched, committed or staged. Evidence class of test results: MEASURED (this session); of times and disk: ESTIMATED.

## Headline for the MM
1. **The zero-nav route pass was a block-1 defect and is ported here.** `ds_run.py:20` mapped `zero` to `--nav-withheld`; the pinned cdbff971 evaluator has no such flag (rc 2, block-1 log 21:38). STRAT1's evaluator has it. S1 and S2 need only real and donor (card: skill = real - donor), so **in read2 the interim port DROPS the zero pass** (2 route passes per arm, no `S_CKPT_zero` cell). This is NOT the fix the other agent is landing in `read/` (zero on 008f48b7 plus an identity gate). **I ported, not copied** (that fix had not landed). After it lands, the MM should either copy `ds_run.py`, `READ_PINS.json` and the tests from `read/` into `read2/` (then re-run the read2 tests), or keep this port: the reader is robust to both, because K-RE (B1) ignores banked `*zero*` cells (`ds_read_b2.py` strategic K-RE line) and `route_arrays` reads real and donor only. If the MM wants zero cells at seed 1, read2 must run them on the same evaluator as block 1 (else the seed-0 and seed-1 strategic blocks differ).
2. **Launch needs `read/` to reach `ZZ DSCALE10K B1 READ DONE ZZ`.** The launcher waits for it, takes the sentinel after B1's latest `start` line only (the log still holds an older FAIL), stops on a later FAIL, and needs `read/out/RESULT_DSCALE10K_B1.json`.
3. The label stays NOT-YET (never "no gain") if any family is unread at either seed or the R_s0 reach is unread.

## What was built (changes against the block-1 files, `file:line` in read2)
| change | where |
|---|---|
| seed-1 constants: `B1_ROOT`, `ARMS_B2`, `B2_ITEMS`, `NOT_YET_B2`, `BLOCK`, `DEFAULT_ITEMS` (env `DS_BLOCK=2` selects the 4 seed-1 items; default 1 keeps block-1 behaviour and its tests) | `ds_lib.py:38-46` |
| default item set | `ds_pull.py:109`, `ds_run.py:151` (`L.DEFAULT_ITEMS`) |
| zero pass dropped (see headline 1) | `ds_run.py:20`, `ds_read.py:161,170,283`, `ds_read_b2.py` miss0, `test_ds_read.py:249`, `smoke_ds_read.py:16` |
| NEW reader: both seeds, labels, K-RE (B1), K-RE mechanism, cross-venue seed 1 | `ds_read_b2.py` (imports `ds_read.py` unedited for logic, loaders, `family_status`, `outcome_label`, `kre_compare`); `outcome_b2:36`, `seed_families:52`, `kre_b1:66`, `kre_mech:78`, `main:105` |
| NEW tests, mutation harness, smoke | `test_ds_read_b2.py`, `mutate_check_b2.py`, `smoke_ds_read_b2.py` |
| launcher | `launch_dscale10k_b2.sh`, `job_dscale10k_b2.cmd`, `launch_wmi_b2.ps1` |

Reading choices where the card is silent (each in the `ds_read_b2.py` header): (1) the label uses stage B at both seeds, cell rule, headline cells and family rule unchanged; (2) the floor F is the seed-0 replicate pair's |B_rep - B_s0|, applied at both seeds (there is one replicate pair); (3) seed-0 rows are reused from `RESULT_DSCALE10K_B1.json`, the pipeline recomputes them and **K-RE (B1)** demands every field equal exactly (NaN equals NaN; a banked cell not recomputed fails; zero rows compared fails), else the problem list is non-empty, the label is marked not quotable and the exit code is 1; (4) "at both seeds" = the same family at both seeds (deviation 4 of block 1, accepted); (5) mechanism (stage A, T0) at seed 1 is read beside with the seed-0 stage-A T floor; reach stays R_s0 vs U_s0.

Reported: the four families with STRATEGIC in full (S1, S2 and the six RC-A cells at 30 / 50 / 80 m, both seeds, the not-in-card members with reasons), PRIMARY beside, per cell Delta, interval, separated, F, gain / cost, n windows / episodes, estimator (paired episode-cluster bootstrap B 2,000 seed 0), tier (T1-kin; STRATEGIC and mechanism T0), evidence class MEASURED. Cross-venue beside at seed 1: `abs(B_s1.B - OBD-CTL_s1)` and the paired rows for B_s1.B and U_s1.B (seed 0 reused from block 1). DK-VIOL and the tactical F1 / provenance split print NOT COMPUTABLE with reason and n (MM ruling: they wait for the W-FAM-STRAT1-TTC-TACCONF instrument). Block-3 arms print NOT-YET.

## Tests (MEASURED)
* `test_ds_read_b2.py` (24 tests) plus the block-1 `test_ds_read.py` (33) against the copy, run AFTER the zero-pass port and the last reader edit: **57 passed, 2 skipped** (the two slow K-RE tests need `DS_SLOW=1`). Planted arms (cells built by the card's own cell rule) reach DATA-HELPS, -PARTIAL, -NEUTRAL, -HURTS, MIXED and SEEDS-DISAGREE (x3 variants); NO-REACH, BLOCKED, NOT-YET precedences; an unread family gives NOT-YET where the block-1 rule alone would have given DATA-HELPS; K-RE (B1) exactness, tamper, missing cell, zero rows, NaN; seed-1 argvs; item defaults per `DS_BLOCK`; the launcher's wait on the B1 sentinel (earlier FAIL ignored, still running never proceeds, later FAIL stops) and its 25 GB bar.
* **Mutations:** first run `14 of 15 RED`. The survivor (NO-REACH guard removed) was an equivalent mutant: `outcome_b2` passed `reach_ok` on to the block-1 label function, which has the same guard. Fixed by deciding reach once in `outcome_b2` (`ds_read_b2.py`, final return passes True). The re-run was still going at hand-off (`mutate_check_b2.py`, waiting on dev-box commit headroom; result `out/MUTATIONS_B2.json`, stdout `out/mut_b2_stdout.txt`). Do not quote the count until that file reads `all_red: true`.
* **NOT RUN at hand-off:** `smoke_ds_read_b2.py` (stand-in end-to-end code check of `ds_read_b2.main`; HANDOFF.md has the command). `ds_read_b2.main` itself (loading, floors, strategic, mechanism, md output) is therefore covered by unit tests of its pure parts only, not yet executed end to end.

## MM launch step (after B1 is re-run and DONE)
```
powershell -File F:\TanitAD-scratch\datascale\read2\launch_wmi_b2.ps1
```
Sentinel `ZZ DSCALE10K B2 READ DONE ZZ` in `read2/out/dscale10k_b2.log`; result `read2/out/RESULT_DSCALE10K_B2.json` + `.md`. Pieces run singly with `DS_BLOCK=2`: `ds_pull.py`, `ds_run.py dumps|mech|route`. Seed-1 pulls come from HF via the banker records (`hf_summary_path` supplies the step count).

## Estimates (ESTIMATED)
Pull 4 x 87 MB; 4 T1 dumps x ~5.3 min; 8 T0 mechanism evals (stage A) x ~0.8-7 min (block-1 log shows 42-53 s for the tac evals); 4 route passes x ~0.5 min; reader ~15 min (11 dumps, 4 pairs, strategic). **About 1.5-2 h.** Disk under 2 GB (checkpoints 0.35 GB, dumps 25 MB each). The launcher keeps F: >= 25 GB (F: read 29 GB at build) and commit headroom >= 5.5 GB before every pass.

## Deliverable manifest
| artifact | where | single copy? |
|---|---|---|
| `ds_read_b2.py`, `ds_lib.py`, `ds_pull.py`, `ds_run.py`, `ds_read.py` (zero port), `ds_t0.py`, `test_ds_read_b2.py`, `test_ds_read.py`, `mutate_check_b2.py`, `smoke_ds_read_b2.py` | `F:/TanitAD-scratch/datascale/read2/` | yes, not staged |
| `launch_dscale10k_b2.sh`, `job_dscale10k_b2.cmd`, `launch_wmi_b2.ps1`, `READ_PINS.json`, this file, `HANDOFF.md` | same | yes |
| `theory/`, `ds_code/`, `ref/`, `kre/` | same | copies of block-1 copies |

Integration for the MM: land read2 next to `read_b1/` under `raw/2026-11-15-v8f-datascale/`, after settling headline 1.

## Deletions (own temporary data, by literal path)
`F:/TanitAD-scratch/datascale/read2/mut/` (mutation copies; the script removes each mutant and the folder at its end), `.../read2/bt/` (pytest basetemp), `.../read2/__pycache__/`, `.../read2/smoke/` (stand-in fixture). Items still present at hand-off are listed in HANDOFF.md; the log of what was actually removed is appended there.
