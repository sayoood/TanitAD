# RJ5 navtrain labels: plan (2026-11-13). Status: BLOCKED on box-2 disk; dev-box path READY. Nothing staged, committed or pushed.

## 1. Provenance of the bank `traj`
- **Rank 0** = `refe/build_teacher_rollouts.py --source navtrain --rank 0`, one teacher rollout per navtrain frame: ego seeded at the logged state, background actors replay the log, 4 s, 20 poses at 5 Hz (`build_teacher_rollouts.py:1-40,209-290,338-399`). Run on the pod by `code/pod_dataprep.sh` stage `rank0` (l.140), `REFE_SIM_HZ=10` (l.61), `TEACHER_DEVICE=cpu` (l.69), registered jerk (no overrides). Inputs = per-token scenarios built from the nuPlan DB (`navtrain_scenarios.build_scenarios_for_log`) plus nuPlan maps v1.0. **No exported inputs file exists**; the DBs are the input. [MEASURED reading]
- **Bank copy on the dev box:** `E:/Projects/TanitAD-artifacts/_rlfan_work/bank_full/targets_rank0.jsonl`, 103,037 rows / 1,191 logs. Dropping the 16 reserve sessions of `P7-PINS raw/reserve_filter_report.json` leaves **92,896 rows / 1,075 logs, exactly the P7 rank-0 count** [MEASURED]. Its sha256 (`5471b25a…`) is not the pod file's `sha256_in` (`daa23b96…`), so byte identity is **UNVERIFIED**; row identity is supported by the control below. 3,412 P7 rows carry `teacher_device: None` (older rows).
- **Rank 1 (57,707 of P7)** are NOT a second sample of the same teacher: they are augmentation twins from the "3-C" bank. `augment_search.py:1-60` (ladder: lane ranks, then goal horizons 8/16/20/6 s, kind written into `aug`) and `2026-10-17-twin-bank-3c/code/reroll_bank_3c.py:1-50` (lane-rank twins re-rolled with the route held at step 0, goal horizon 6 s, a 10 s rollout, kept only if f(10 s) >= 0.75, row = first 4 s; other rows pass through byte-identical). **RJ5 for rank 1 needs the twin machinery (route patch, twin parts, commit filter, which can change under RJ5) and the rank-1 rows are not on the dev box** (192 MB `refe_train_grow_cmd_3c_p7.tar.zst` on private HF; not downloaded). `PREREG_P5C.md` l.189: in C1 rank-1 rows keep the identity label. **Recommendation: RJ5 rank 0 only for the first fine-tune; disclose that rank-1 rows stay registered-jerk.**

## 2. Inputs off the pod
- Dev box `E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/trainval`: DBs for **198 of 1,075 logs = 16,133 of 92,896 rows (17.4 %)** [MEASURED]. Maps: `E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0` (1.36 GB).
- Box 2 has no nuPlan DB, no maps, no devkit.
- The other **877 logs** (~97 GiB transit, public nuPlan S3, range-fetched and CRC-checked per DB) come from `code/fetch_navtrain_dbs.py --logs-file` (INHERITED: 17.3 MB/s single stream, 8 jobs scale). Box 2 reaches pypi and HF (200). The chain fetches a batch (~12 logs, ~1.3 GB), rolls it, and deletes it.

## 3. Box-2 environment
The roll needs: Python 3.11, torch 2.7.1 (CPU is enough), the 113 pins in `box2/req_rj5.txt` (from the dev box `driverl-eval`), DriveRL `src` + `nuplan-devkit` + `release/` (checkpoint_2400, 29 MB tar), `refe/`, `code/`, the TF-N2 patched `build_teacher_rollouts.py`, maps. Box 2 has Python 3.11 (WindowsApps) and 3.13.
**I built it (`box2/setup_rj5_env.cmd`, worked, ZZSETUP_DONE) and then DELETED it (coordinator disk rule).** Sizes [MEASURED]: venv 2,375 MB, maps 1,360 MB, dz 27, navsim 13, w 20; total ~3.8 GB.
**Disk numbers:** D: free before 5.50 GB; peak low 1.60 GB (with the tars deleted mid-way, ~0.2 GB lower with them); after cleanup 5.46 GB. All my files under `D:\tanitad\rj5`, `tmp\rj5_*`, `jobs\rj5_*`, `logs\rj5_setup.log` removed by literal path (tars: maps 1,426,595,840 B, dz 29,204,480, navsim 14,499,840, w 22,056,960).
**BLOCKER:** env 3.8 GB + one DB batch ~1.3 GB + outputs does not fit in "keep >= 5 GB free" on a drive that has 5.5 GB. Unblocking options: (a) PI/MM points me at another drive on box 2 (G:/H: show ~18.8 GB free; C: 19.8; outside `D:\tanitad`, so a PI call); (b) run on the dev box only (below). A lighter venv could drop ray/bokeh/tensorboard/selenium/boto (maybe -0.5 GB), UNVERIFIED that the roll never imports them.

## 4. Smoke
- **Box 2: not run** (disk rule). Box-2 s/scene and best N are **UNMEASURED**.
- **Dev box (i9-12900F, 1 process, CPU, jerk arms R0 and RJ5, 6 bank rows of log `2021.08.31.14.40.58_veh-40_00679_00892`)**, `raw/smoke_dev/`:
  - **Control: R0 reproduces the bank `traj` with max |diff| = 0.0 on 6/6 rows** (all 60 numbers, bit-exact; the pod bank is CPU too). Tolerance to use for the full run: 0.0 on the control rows; any nonzero is reported, the bar is <= 1e-6 m (TF-N2 C-R0 bar).
  - **Reach:** knobs read back from the built planner = 5.0 on every RJ5 row; RJ5 moved 4 of 6 rows (dxy 0.004, 0.009, 1.149, 1.324 m), 2 rows 0.000.
  - **Cost:** R0 3.1-3.7 s/row warm (first row 22.6 s incl. map load), RJ5 1.4-2.9 s/row; whole job 12 rolls in 62.7 s.
  - `rj5_roll.py` calls the bank's own `make_row` (stub cameras), so it is the bank's construction.
- Offline tests: `test_rj5.py`, 10/10 pass (knob mismatch mutation, shard partition, resume, sha, merge coverage/fallback/conflict refusal, determinism).

## 5. Chain and wall-time
Code (all in this hold): `rj5_roll.py` (roller), `rj5_chain.py` (per box: fetch batch, N single-thread rollers, delete the DBs it fetched, per-batch `.done` with sha256 of each rows file; resumable; disk guard), `rj5_merge.py` (bank-keyed table, coverage = every bank key, fallbacks counted, conflicting duplicates REFUSE, R0 control read, output sha256), `rj5_label.py` (prep / merge around the **unchanged** `p5c_dual_labels.py`, key = log, token, rank, sha256(traj)).
Order: shard logs (`logs file` per box, disjoint: dev-box-local logs go to the dev box) -> chain per box -> merge -> label (`P5R_BLAS_THREADS=1`, all BLAS vars 1) -> label merge. Controls: R0 on every 20th log of a batch (identity vs bank); coverage 92,896/92,896 or REFUSE; label fallback <= 1 % (S4).
Launch box 2 via `launch.ps1`. **Box-wide claim:** before any box-2 job, on the dev box `set -o noclobber; echo "rj5 <what> <utc>" > C:/Users/Admin/box2_ship/BOX2_BUSY.txt`; if it exists, wait; release only if the content is mine. The full roll loads all 16 threads, so it counts.

**Wall-time (ESTIMATED; rank 0 only, 92,896 rows).** Cost per row 3.5 s (dev smoke, warm, R0-like) to 9.4 s (TF-2b, INHERITED) => 90 to 243 CPU-h. Labels 0.60 s/row (INHERITED, `a1/HANDOFF.md`) => 15.5 CPU-h.
| config | rolls | + labels |
|---|---|---|
| box 2 alone, ~9 effective procs (8C/16T, assumed; unmeasured) | 10 to 27 h | +1.7 h |
| box 2 + dev box 4 procs from 06:00Z (~13 effective) | 7 to 19 h | +1.2 h |
| **dev box alone, 2 procs now / 4 from 06:00Z** | 11-30 h (4 procs) | +4 h (4 procs) |
Dev box: only 198 logs need no fetch (16,133 rows = 3.9 h at 4 procs, 3.5 s/row); the rest streams through F: (32 GB free, ~1.3 GB per batch).

**Honest cost cuts:**
1. **Skip rows RJ5 cannot plausibly change.** Not provable from the rows (the bank holds 20 poses, not the executed jerk). Screening on the bank traj [MEASURED, pure python]: 201 rows are stationary (4 s path < 0.5 m); **42,628 more (45.9 %)** have max finite-difference positive jerk < 0.5 m/s^3 on the 0.2 s grid. Skipping both leaves ~50,100 rows (-46 %). Fidelity cost: those rows keep registered-jerk trajectories; the claim becomes "rows where RJ5 was screened to be inert", ESTIMATED. Needs an **audit arm**: roll ~300 random rows of the skip set at both jerks; the cut stands only if every |dxy| < 1e-6. Dev smoke: 2 of 6 rows had dxy 0.000 and 2 of 6 only 0.004-0.009, so inertness is real but not exact.
2. **Smaller dose:** first fine-tune on the 16,133 dev-box-local rows (17 %, 3.9 h) or a log-stratified half; cost = fewer scenes and a city/vehicle skew (dev-box logs are not a random sample; stratify by `scenario_type` first).
3. Rank 1 skipped (above).

## 6. Escalate
- **MM/PI:** which drive on box 2 may RJ5 use (needs ~5.5 GB env + ~2 GB scratch), or approve dev-box-only. The dev-box path is ready now.
- **MM:** confirm rank-1 policy (stay registered), and the skip-inert cut with its audit.
- **Pod:** a list of sha256(traj) for P7 rank-0 rows (or the file) is needed to prove the keys equal the pod's before labels are keyed to them (bank sha mismatch above).

## 7. Deletions
`F:/tanitad_scratch/rj5nt/` (1.5 GB: code export 22 MB, ship tars, slim bank 55.7 MB gz, dev-smoke output) deleted by literal path at the end; re-creatable from tip `13e30630`, `bank_full` and the reserve list. Box 2: see section 3.
