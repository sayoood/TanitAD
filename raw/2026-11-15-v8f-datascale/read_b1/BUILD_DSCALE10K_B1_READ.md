# BUILD: DSCALE10K block-1 read (built and tested, not run)

Card `PREREG_V8F_DSCALE10K` (frozen, sha256 `48a74901...`), MM rulings `build/MM_RULINGS_DSCALE10K.md` and `gate/MM_RULING_R.md`. All files are in `F:/TanitAD-scratch/datascale/read/`. Nothing was trained, launched against the pod GPU, committed or staged. No checkpoint was pulled (F: has 14.8 GB free).
Evidence class of everything below: MEASURED for test results (this build, this session), ESTIMATED for wall time and disk.

## Headline for the MM
1. **F: is below the card's bar.** The launcher and `ds_pull.py` / `ds_run.py` refuse under 25 GB free; F: read 14.8 GB. Free space on F: before the start. The read's own footprint is under 3 GB (ESTIMATED: 7 checkpoints 0.6 GB, 7 dumps about 6 MB each, evals a few hundred MB).
2. **The banker's record format is not known** (`gate/banker/` does not exist yet). `ds_pull.py` reads `<bank-dir>/<arm>.<stage>.json` = `{sha256, hf_repo, hf_path[, hf_rev, hf_summary_path, hf_type]}` (default dir `gate/banker`), else the pod's `bank_ok/<arm>.<stage>.sha256` (which must equal `bank_ready/<arm>.<stage>.txt`). If the banker writes another shape, adapt `expected_sha` (ds_pull.py:24); the tests plant both shapes.
3. **Seed 0 only: no outcome label is possible in block 1.** The card's table needs both seeds, so the reader prints the seed-0 family read, the reach precondition and the floor, plus a `provisional_if_seed_1_agrees` field that is never a label.

## Pieces
| piece | file:line | what it does |
|---|---|---|
| pull + verify | `ds_pull.py:24` expected_sha, `:41` verify_and_record, `:84` pull_item | expected sha256 from the banker (HF record preferred, else pod bank_ok), pull, hash, remove a mismatching copy, write `ckpts/<arm>.<stage>/BANK_RECORD.json` (arm, stage, sha256, source, commit, steps) |
| gates | `ds_lib.py:82` disk_problem, `:88` bank_problem, `:111` tree_problem | F: >= 25 GB; checkpoint sha256 = record, record names arm / stage / commit `cdbff971` / 19,164 or 3,000 steps; key tree files equal the commit's blobs (`READ_PINS.json`, `make_pins.py`) |
| dumps and T0 passes | `ds_run.py:33` t1_argv, `:51` mech_argvs, `:60` route_argv, `:96` cmd_dumps | evaluators `v8f_t1_eval.py` / `v8f_mini_eval.py` UNEDITED from the cdbff971 export; 2 threads, CPU, one environment; commit headroom >= 5.5 GB waited for; done files; skip when present |
| mechanism read (T0) | `ds_t0.py:79` mech_pair, `:49` decide_t, `:59` decide_reach, `:69` load_z | wraps the DSCALE instrument unedited (`ds_code/code/`, exported from commit `213250ce`): load_all, check_gt_codes, present_error, mlat_row are called, nothing re-implemented |
| the reader | `ds_read.py:52` family_status, `:67` outcome_label, `:95` seed0_reading, `:158` route_arrays, `:185` kre_t1, `:199` kre_t0, `:220` main | 24 cells x 2 stages, per-stage floors, headline cells, the card's section-6 table, STRATEGIC, the init control, cross-venue beside, `RESULT_DSCALE10K_B1.json` + `.md` |
| cell rule | `theory/obd/read/obd_read.py` cell_delta (the OBD / STRAT1 reader, unedited; `read_lib.py` line 17 now reads the cdbff971 export, as the OBD read did) | gain / cost = separated AND |Delta| > F |
| launcher | `launch_wmi.ps1` -> `job_dscale10k_b1.cmd` -> `launch_dscale10k_b1.sh`, waiter `wait_dscale10k_b1.sh` | WMI CreateFlags 0x08000200, `cmd /c X.cmd < NUL`, `Git\usr\bin\bash.exe`; last line `ZZ DSCALE10K B1 READ DONE ZZ` or `... FAIL ZZ <why>` |

What the read computes, in card terms:
* **Dumps (T1-kin, deciding tier for LONGITUDINAL / LATERAL / TACTICAL):** 7 checkpoints: B_s0, U_s0, B_rep at stages A and B, R_s0 at stage A. Tag `DS-<arm>-<stage>_m10_causal_heads`.
* **Cells:** 24 per pair (11 metric rows x operative 0-2 s and tactical 2-6 s, plus TAC-LAT and TAC-LON), `U - B_s0` at stage B (deciding) and stage A (beside), `R - U` at stage A (beside), `B_rep - B_s0` (the floor pair, no floor applied to itself). Floor `F_cell = |B_rep - B_s0|` per cell and stage.
* **Headline cells and family labels:** L1 / L2, A1 / A2, T1 / T2, S1 / S2 exactly as card section 4; family gain = a headline gain and no headline cost; family cost = a headline cost.
* **STRATEGIC (T0, stage B, 9 passes):** S1 = RC-A checkpoint position error at 80 m (the six RC-A cells at 30 / 50 / 80 m are printed), S2 = S_CKPT(real) - S_CKPT(donor) at 80 m per window (the card's "skill = real - donor", paired), ground truth and S_CKPT from the unedited STRAT1 helpers. Members not in this card print with the card's reasons (detected information n/a, derived goals not in this arm, intent 6-30 s not computable).
* **Mechanism (T0, stage A, 16 evals):** r = RMSE_latcons(U)/RMSE_latcons(B) - 1 with the paired MSE delta and the DSCALE Amendment 1 sign rule; T-DATA-LIMITED = r <= -5 %, separated, |Delta| above the stage-A T floor. Oracle ADE6 and per-group RMSE beside.
* **Precondition:** R_s0 vs U_s0 at stage A, r >= +5 %, separated, sign rule -> REACHED, else NO-REACH (precedes every label).
* **Committed outcomes:** NO-REACH, DATA-HELPS, DATA-HELPS-PARTIAL, DATA-NEUTRAL, DATA-HURTS, MIXED, SEEDS-DISAGREE, BLOCKED, each with the card's next step text (`ds_read.py:39`). Block-2 / block-3 arms (B_s1, U_s1, UE_s0, BL_s0, X_s0, seeds 3 and 4) print NOT-YET, as does the `H-DATA-SCALE-1` sub-read.
* **Beside:** C-INIT read from the gate's `init.json` (`B_s0_vs_U_s0_params_equal`), cross-venue `abs(B_s0.B - box-2 OBD-CTL_s0)` from the banked box-2 dump (`otband_read/t1`, same tree and thread count), the replicate pair `B_rep - B_s0`.
* **Controls in the result:** K1 self-pair (reads exactly 0 [0, 0]); T0 identity control (an arm against itself r = 0 exactly); **K-RE (T1)** recomputes STRAT1's `S1_s0 - C0_s0` 24-cell row from the STRAT1 dumps through this pipeline and compares every field to `RESULT_STRAT1.json`; **K-RE (T0)** recomputes the DSCALE read's `M2b-df50 - M2b` D-T row from the banked eval files. Any control failure makes the process exit 1.

## Tests (MEASURED, this session)
* `test_ds_read.py`: **33 passed, 2 slow** (`DS_SLOW=1`: the two K-RE tests, **both pass**: T1 24 cells x 9 fields exact against `RESULT_STRAT1.json`; T0 `r` 0.06253327145340393, slot RMSEs and MSE delta point 0.02482379449882957 exactly the banked values).
* Every committed block-1 outcome is reached by a synthetic input (parametrised rows for DATA-HELPS, -PARTIAL, -NEUTRAL, -HURTS, MIXED, SEEDS-DISAGREE x4 variants, NO-REACH, BLOCKED, and NOT-YET on one seed); the cell rule (gain, cost, floor, no floor, higher-is-better), the reach rule and the T-label rule with each of its four conditions, the bank / tree / disk refusals (every branch), the pulled-copy removal on a sha256 mismatch, the argv of every dump and eval, and the NOT-YET result on missing dumps.
* `mutate_check.py`: **17 of 17 mutations RED** (unmutated copy green): NO-REACH precedence, MIXED, DATA-HELPS count, one-seed cost, the two-seed requirement, cost-over-gain, the seed-0 label, K-RE field count, the floor comparison, the reach bar, the sign rule, the T floor, checkpoint sha256, step count, F: bar, pulled-copy hash, the all-latcons eval flags. Result in `out/MUTATIONS.json`.
* An end-to-end run of `ds_read.py` on STAND-IN inputs (STRAT1 dumps and T0 passes, the banked DSCALE evals copied under the block-1 names; R_s0 = a copy of U_s0) produced a JSON with both K-RE controls passing, reach NO-REACH (as the stand-in must), and BLOCKED because no checkpoint was present: a code check, never a result. It found one defect (the DSCALE instrument takes its GT block from the arm keyed `M2b`; `ds_t0.load_z` aliases the GT reference arm to that key, and `present_error` itself refuses an arm whose GT block differs).

## Estimated wall time and disk (ESTIMATED)
Pull 7 x 87 MB (minutes); T1 dumps 7 x ~5.5 min (STRAT1 measured); T0 passes 25 x ~7-8 min (the `--project --families` eval may be slower); reader ~3 min per 2,000-draw pass set plus K-RE. **About 4 h.** Disk: under 3 GB. Commit headroom >= 5.5 GB is waited for before every pass.

## The exact MM start command
```
powershell -File F:\TanitAD-scratch\datascale\read\launch_wmi.ps1
```
Prerequisites: F: >= 25 GB free; the banker's records (or pod `bank_ok` markers) for the 7 items; `READ_PINS.json` passes (it does today). Watch `out/dscale10k_b1.log` for the sentinel, or run `bash wait_dscale10k_b1.sh`. Result: `out/RESULT_DSCALE10K_B1.json` and `.md`. Pieces can be run singly: `python ds_pull.py --items U_s0.A`, `python ds_run.py dumps B_s0.A`, `python ds_run.py mech`, `python ds_run.py route`.
The pulled checkpoints stay in `ckpts/` after the read (re-pullable from the bank; delete when the MM says the read is final).

## Deviations for the MM (where the card is silent; each quotes the card)
1. **Code tree.** The card: "reads on the dev-box CPU, dumps in one environment and one thread count". Used: the cdbff971 export (the code of record every arm trained from, MM ruling 4), `otband_read/tree`, whose seven key files equal the commit's blobs (`READ_PINS.json`). The STRAT1 read used the e81bde01 tree; the evaluator `v8f_t1_eval.py` differs between them (cdbff971 `61e140ed`, e81bde01 `180044eb`). The cdbff971 evaluators also produced the box-2 OBD-CTL dumps read beside, so the cross-venue row compares like with like. `theory/otband/read/read_lib.py` line 17 is the only edited line of the copied readers (the tree path).
2. **Which stage decides.** Card 2.1: "Stage B is where the deployable model comes from, so it carries the deciding read. Stage A ... carries the mechanism read." Used: the family labels and the seed-0 read are taken at stage B; stage A cells are printed beside; mechanism and reach are at stage A.
3. **Floors per stage.** Card 4: "F_cell = abs(B_rep - B_s0) per cell, per stage". Used as written; the STRATEGIC cells S1 / S2 take their floor from the same pair at stage B.
4. **"Family cost ... at both seeds" is read as the same family at both seeds.** Card 6: "family cost in >= 1 family at both seeds". Used: DATA-HURTS needs a cost in one family at both seeds; costs in different families at the two seeds, or at one seed, give SEEDS-DISAGREE. MIXED needs a gain and a cost, both at both seeds, and is checked before DATA-HURTS. DATA-HELPS needs no cost at either seed ("no family cost at either seed"); a cost at one seed therefore gives SEEDS-DISAGREE ("a family gain or cost at one seed only, not covered above"). Only block 2 can exercise these.
5. **"The stage-A T floor".** Card 4: "abs(Delta) > the stage-A T floor". The card does not define it. Used: `|mean per-window latcons MSE(B_rep) - mean(B_s0)|` over the windows present in both (the Delta the card names is the paired MSE delta); `abs_r` of the pair is printed beside. One replicate pair, a weak floor.
6. **Deciding definition of r.** The slot-pooled latcons RMSE (the DSCALE instrument's deciding form); the window-weighted `r_window` is printed beside, as in DSCALE.
7. **Reach "separated".** Card 6: "R_s0 not worse than U_s0 on T latcons by r >= +5 %, separated". Used with DSCALE Amendment 1's sign rule: the MSE delta (R - U) must be positive and separated, and r >= +5 %.
8. **STRATEGIC passes at stage B only (9 T0 passes).** Card 4: "STRATEGIC route cells: T0 at the window's first tick". Stage B carries the deciding read (deviation 2), so stage A has no route passes and prints NOT-YET for that family. S2 is the per-window difference S_CKPT(real) - S_CKPT(donor) at 80 m; the card's best constant 0.794 is printed as the reference, not decided on.
9. **Not built (named, each with its reason).** DK-VIOL counts (a beside row of the OBD reader, outside its 24 cells); T-LAT-F1, T-LON-F1, goal macro-F1 split by provenance (VLM vs geometry bits) and the nav-echo split (STRAT1 4.3): no instrument exists in the STRAT1 / OBD readers, the heads are in the `--dump-heads` dumps; the per-country / platform eval breakdown (card 8). The goal-bit F1 split is the card's own diagnosis for DATA-HURTS, so it is the first item to build if that label appears. The 24-cell set plus S1 / S2 carry every family's headline cells.
10. **Cross-venue beside.** Card 2.3: "read beside B_s0 / B_s1 as cross-venue replicates". Block 1 reads OBD-CTL_s0 against B_s0.B and U_s0.B; the seed-1 pair is NOT-YET.
11. **C-INIT.** Card 2.3 asks whether B_s0 and U_s0 share the step-0 state. The gate's `init.json` records parameters equal (the gate report: sha256 `0fdf9534...`) and state / buffers not equal; the reader prints `params_equal` and reads the lever as data alone on that basis. The MM may prefer the stricter state reading.
12. **Mechanism evals.** The four evals follow the queue's `:evaldscale` flags exactly (no `--ages`); route passes follow STRAT1 (`--ages fresh`). The default of `--ages` was not checked here.
13. **Pod pull.** If the HF bank is not ready the pod path (`pssh` / `pscp`, files read only) is used when `bank_ok` equals `bank_ready`. No pod address, port or id is written to any file.

## Deletions (own temporary data, by literal path)
`F:/TanitAD-scratch/datascale/read/mut/` (the mutation harness copy, removed by the script on every run), `.../read/bt/` (pytest basetemp, 5 times), `.../read/smoke/` (the stand-in fixture, 90 MB), `.../read/tmp/` (empty), `.../read/__pycache__/`. All re-creatable from the sources here; none was refused. No pod data, git object, register or result was touched.

## Deliverable manifest
| artifact | where | single copy? |
|---|---|---|
| `ds_lib.py`, `ds_pull.py`, `ds_run.py`, `ds_t0.py`, `ds_read.py`, `make_pins.py`, `READ_PINS.json` | `F:/TanitAD-scratch/datascale/read/` | yes, not staged, not in git |
| `test_ds_read.py`, `mutate_check.py`, `smoke_ds_read.py`, `out/MUTATIONS.json` | same | yes |
| `launch_wmi.ps1`, `job_dscale10k_b1.cmd`, `launch_dscale10k_b1.sh`, `wait_dscale10k_b1.sh` | same | yes |
| `theory/` (OBD and OTBAND readers, one line edited), `ds_code/code/` (DSCALE instrument and its helpers, commit `213250ce`, unedited), `kre/` (banked B and Q evals for K-RE, 17 MB), `ref/` (banked `dscale_read.json`, the queue, the instrument source) | same | copies of repo files |
| this file | `F:/TanitAD-scratch/datascale/read/BUILD_DSCALE10K_B1_READ.md` | yes |

Integration needed (MM): land the folder under the study path with the card, once the banker's record shape is confirmed (headline item 2).
