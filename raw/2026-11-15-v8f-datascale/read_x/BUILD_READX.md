# BUILD: DSCALE10K X_s0 read (the 10,000 clips alone), built, tested, smoked, dry-run, NOT launched

Folder `F:/TanitAD-scratch/datascale/readx/`. Nothing committed, staged or pushed. `read/`, `read2/`, `read3/`, `diag/`, `blockx/` were only read (code copied, never `ckpts/`, `t1/`, `out/`). Pod: read-only `pssh cat/grep` of the banker ledger and the chain log (no GPU, no process touched, no address written). Evidence class of every test, mutation and smoke result: MEASURED (this session, CPU, `CUDA_VISIBLE_DEVICES=-1`).

## 1. Headline for the MM
1. **Rules are frozen and checked.** `blockx/MM_RULINGS_X_READ.md` sha256 `e6e3f24cc79fb65098af3e6b166ebb9375d2c65ca939e5c4e937e989a58aecca` verified (matches `MM_RULINGS_X_READ.sha256`); the reader and the launcher re-check it on every run (`ds_read_x.py:rulings_problem`) and refuse on a mismatch. Nothing in it was edited.
2. **X_s0.A is already banked and verified from the dev box** (ledger BANKED 2026-11-18 09:00:55Z, private repo `Sayood/tanitad-v8f-dscale10k-runs`, rev `f1df0e70`, sha256 `1dd437f1...3ef1`, hub LFS sha256 equal, `gate/checks.json` sha256 equal to the ledger's). `banker/X_s0.A.json` and `evidence/X_s0.A.checks.json` are written; I pulled the stage-A checkpoint once through `ds_pull.py` to test the path (sha256 verified), then deleted it (section 8). **X_s0.B is not banked yet and the chain log has no DONE line** (`make_banker_x.py` refused at 10:04Z, as designed).
3. **The read is wired end to end on stand-ins.** Smoke OK at B 2,000 (X = B_rep stand-in): K-RE reproduces every banked block-1 / block-3 field exactly, the 0.0767 stage-A floor included; K1, K-EVALX, K-VLM-ZERO, C-JOIN-EQ pass; the exact boundary `Delta == -F` was reached on real data (inclusive X-CONTENT). With unmasked ticks K-VLM-ZERO fails and the outcome reads CONTROL FAILED.
4. **MM must decide / do:**
   * Launch only after the chain's `ZZ DSCALE10K BLOCKX DONE ZZ` (the launcher stops itself otherwise; re-runnable).
   * F: has 28 GB free, the card bar is 25 GB: ok now, but the pulled checkpoints + dumps need under 1 GB.
   * C-JOIN-EQ is quoted from the chain's gate record (`X_s0.<stage>/gate/checks.json`, banked with the arm, sha256-checked against the ledger) because `blockx.log` carries no C-JOIN-EQ line (grep on the pod: none). Say if you want the chain log lines as well.
   * The ruling's "block-1 floor" for K-VLM-ZERO is read as `|VLM macro-F1(B_rep) - VLM macro-F1(B_s0)|` at stage A from the same tool (0.0274, `diag/RESULT_DIAG.json`), so the bar is max(B_s0 0.1257 - 0.0274 = 0.0984, 0.05) = 0.0984, passing at the lower of the two bars. Say if "block-1 floor" meant something else.
   * No stage-A / stage-B T0 mechanism (ADE6, latcons) evals are run: the ruling asks for none.

## 2. What the read does (R1-R4, hops)
| rule | where |
|---|---|
| R1 deciding cell TAC-LAT, pairs X-B_s0 (deciding), X-U_s0 (beside), stages A and B, B 2,000 seed 0, F_cell = abs(B_rep - B_s0) per stage | `ds_read_x.py` `PAIRS`, `CELL`, main (`F = {st: OR.floors_from(...)}`, `pairs[...]`) |
| R2 labels, stage A decides, STAGE-SENSITIVE | `label_x`, `stage_labels`, `label_text` |
| R3 four families + STRATEGIC per pair per stage, never pooled, fog unscored | `pair_family_status`, `fam_rows_x`, main `rep["families"]`; STRATEGIC from route passes: X in `out/t0route`, references from `read3/out/t0route` (stage A) and `read/out/t0route` (stage B) |
| R4 K-VLM-ZERO | `k_vlm_zero`, `f1_part` (calls `diag/diag_read.py` unedited: `counts`, `f1_from`, `group_macro`, `boot_f1`, `load_ticks0` pointed at this folder for X) |
| R4 K1 | synthetic `OR.identity_control` + real X - X cells at both stages `self_pair_problems` |
| R4 K-RE | `B2.kre_b1` / `kre_strat_b1`: U-B_s0 and B_rep-B_s0 T1 cells at both stages and floors vs block 1, the 0.0767 floor and U-B_s0.A vs block 3, STRATEGIC rows (stage B vs block 1, stage A vs block 3), goal-F1 B_s0 / B_rep macros vs `RESULT_DIAG.json` (`kre_f1`) |
| R4 K-EVALX | `ds_run.py cmd_kevalx` now covers stages A and B (`ROUTE_STAGES`); `kevalx_problem_x`, `route_problem_x` (also demands the references' banked K-EVALX records) |
| R4 C-JOIN-EQ | `cjoin_problem` on `evidence/X_s0.<stage>.checks.json` |
| precedence BLOCKED > NOT-YET > label > CONTROL FAILED | `outcome_x` |
| bank / pull | `make_banker_x.py` (ledger + chain sentinel + HF), `ds_pull.py` (unchanged except the Git bash path), `ds_lib.py` block 4 (`X_ITEMS`, `REF_ARMS_X`, `ck_root`) |

Outputs: `out/RESULT_DSCALE10K_X.json` + `.md`. Outcome text always carries "(one seed)"; the R5 caveat (1.8 vs 4.0 passes per window, ESTIMATED) is printed with the label.

## 3. Tests, mutations, smoke (MEASURED)
* `test_ds_read_x.py`: **55 passed** (`out/` log in the launcher run). Whole file, never `-k`. Covers label boundaries both signs and exact +-F, floor / NaN guards, mutual exclusivity over a grid, STAGE-SENSITIVE (both directions, per-stage floors, stage-B NOT-YET), precedence, K-VLM-ZERO pass / fail / boundary / NaN, K1, K-EVALX both stages, route completeness, C-JOIN-EQ, goal-F1 K-RE, ruling sha256, block-4 constants and roots, X bank steps (19,164 / 3,000), route jobs (10), `cmd_kevalx` at both stages, ledger / hub / sentinel logic, launcher order and sentinels.
* `test_ds_read.py` (block-1 file, run against this copy): 41 passed, 2 skipped.
* Mutations `out/MUTATIONS_X.json` (see section 3a for the final count) and smoke-level `out/MUTATIONS_X_SMOKE.json`.
* Smoke `smoke_ds_read_x.py`: SMOKE OK (`out/smoke_x_run.txt`).
* **Launcher dry check (no real read):** `bash -n` OK; steps 1 (ruling sha256) executed OK; `make_banker_x.py` on the live pod refused (no DONE line). The launcher itself was not run on real X data.

### 3a. Mutation result
Unit mutations (one per committed rule, wiring, bank / hub / sentinel logic and launcher bar): **65 of 65 RED**, unmutated copy GREEN. `out/MUTATIONS_X.full.json` is the full run (64 of 65 RED; the survivor was a sentinel-regex mutation that was an equivalent mutant, replaced by a non-equivalent one that goes RED: `out/MUTATIONS_X.partial.json`). Smoke-level (reader wiring in `main`): **4 of 4 RED** (`out/MUTATIONS_X_SMOKE.json`). Two earlier survivors (launcher `|| fail` guards, X K-EVALX stage coverage) were fixed by stricter tests; an earlier dropped mutation (route default) was an equivalent mutant.

## 4. Launch command (the MM, after the chain's DONE line)
```
powershell -File F:\TanitAD-scratch\datascale\readx\launch_wmi_x.ps1
```
(WMI `CreateFlags 0x08000200`, stdin `< NUL`, `job_dscale10k_x.cmd` calls `Git\usr\bin\bash.exe`). Sentinel `ZZ DSCALE10K X READ DONE ZZ` / `... FAIL ZZ <why>` in `readx/out/dscale10k_x.log`; result `readx/out/RESULT_DSCALE10K_X.json` + `.md`. Steps: ruling sha256, prerequisite results and K-EVALX records, F: >= 25 GB, `wait_commit.py 5.5 30`, tests, `make_banker_x.py`, `ds_pull.py`, reference check-only, 2 T1 dumps, 10 route passes + K-EVALX, reader. ESTIMATED 25-40 min (block-3 MEASURED: dumps about 5 min each, route passes 30-100 s each, reader about 2 min at B 2,000; the smoke at B 2,000 took about 100 s). Re-runnable.

## 5. UNVERIFIED
* The real X dumps, route passes and reader have not run (X_s0.B not banked).
* `ds_run.py` argv for X stage B dumps / route passes is the same builder as the references; not executed on the real X checkpoints.
* `make_banker_x.py`'s hub path was exercised on X_s0.A only; stage B follows once banked.
* K-VLM-ZERO's "block-1 floor" reading (section 1).

## 6. Deletions log (own temporary data, literal paths)
`readx/smoke` (stand-in fixtures, re-creatable by `smoke_ds_read_x.py`), `readx/bt` (pytest basetemp), `readx/mut` (mutation copies), `readx/ckpts` (the X_s0.A checkpoint pulled for the pull test, 83 MB, re-pullable from HF, sha256 verified before deletion). Kept: `banker/X_s0.A.json`, `evidence/X_s0.A.checks.json`.

## 7. Deliverable manifest
| artifact | where | single-copy? |
|---|---|---|
| reader, launcher (`.sh`, `.cmd`, `.ps1`), `make_banker_x.py`, tests, mutation / smoke scripts, `ds_lib.py` `ds_run.py` `ds_pull.py` (modified copies), unedited copies `ds_read.py` `ds_read_b2.py` `ds_t0.py` `READ_PINS.json` `theory/` | `F:/TanitAD-scratch/datascale/readx/` | yes (F: scratch, not staged; the MM lands) |
| `banker/X_s0.A.json`, `evidence/X_s0.A.checks.json` | `readx/` | re-creatable from the ledger and HF |
| results `out/MUTATIONS_X*.json`, `out/smoke_x_run.txt` | `readx/out/` | yes |
| X_s0.A checkpoint | pod, private HF (rev f1df0e70) | 2 copies, none here |
