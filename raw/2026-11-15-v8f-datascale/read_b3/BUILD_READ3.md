# BUILD: DSCALE10K block-3 bank + read (H-DATA-SCALE-1 sub-read), built, dry-run, NOT launched

Folder `F:/TanitAD-scratch/datascale/read3/`. Nothing committed, staged or pushed; `read/`, `read2/`, `card/` and the pod were only read (pod: read-only commands, no GPU, no process touched). Evidence class of every test / mutation / smoke result: MEASURED (this session, CPU, `CUDA_VISIBLE_DEVICES=-1`). Times: ESTIMATED from the block-2 log (MEASURED there).

## 1. Headline for the MM
1. **Block 3 is BANKED and verified (deliverable 1, MEASURED).** The pod's banker had already banked both arms (`banker_ledger.jsonl` BANKED 2026-11-18 01:53:54Z UE_s0.A, 05:28:47Z BL_s0.A; `bank_ok` + `bank_ready` markers present). I re-checked from the dev box: private repo `Sayood/tanitad-v8f-dscale10k-runs` (private True), at the banked revisions the checkpoint's hub LFS sha256 equals the marker and `summary.json` is present (`hf_verify_b3.py`); the pod's `sha256sum` of `runs/<arm>/A/ckpt_last.pt` equals the marker; I pulled both checkpoints from HF into `read3/ckpts/<arm>.A/` and their sha256 equals the banked one (`BANK_RECORD.json`, steps 61,568 from each `summary.json`). Three copies (pod, HF private, dev box); none is single-copy. UE_s0.A `2fd4b747...df914` rev `3ba330c6`; BL_s0.A `0b374ded...e94b` rev `bb35e025`. (The `block3.sh` line "banking pending" predates the banker's run.) I deleted nothing on the pod.
2. **The block-3 read is built and wired**: `launch_dscale10k_b3.sh` (+ `.cmd`, `launch_wmi_b3.ps1`), reader `ds_read_b3.py`. Tests 73 passed 2 skipped (32 new + the block-1 file against this copy), mutations 31/31 + 3/3 smoke-level RED, smoke OK on stand-ins.
3. **MM must decide / do:**
   * **Launch bar: F: has 22 GB free, the card bar is 25 GB. The launcher will stop on it. Free at least 3 GB first.** (Not moved.)
   * **Confirm the reading of "U - BL" (below, item A).** The card's line 197 is ambiguous; I run both readings and print READING-SENSITIVE (not one label) if they disagree.
   * **Stage-A STRATEGIC costs 25 extra route passes (item C).** Block 1/2 made route passes for stage B only; UE_s0 / BL_s0 have no stage B. It is the bulk of the run time (ESTIMATED 15-40 min of about 1 h). Say if you want it dropped (STRATEGIC would then read NOT-YET with that reason).

## 2. What the read does (card lines 195-197, 276 quoted in `ds_read_b3.py` docstring)
Stage A, seed 0 only. Arms: UE_s0 (U set, 4 union epochs, 61,568 steps, `--steps-mult 4`), BL_s0 (B set, the same 61,568 steps; MM_RULING_BLOCK3_STEPS), against the block-1 stage-A arms B_s0, U_s0, B_rep (their checkpoints, dumps and T0 evals are read from `read/`, K-RE reproduces them).
* **Label** (`verdict`, `ds_read_b3.py:54`; `data_scale_label`, `:66`): ADE6 (the T0 E0 eval's per-window `ade6`, the DSCALE instrument's own, card line 27 / 196), F_A = abs(mean ADE6(B_rep) - mean ADE6(B_s0)); DATA > EPOCHS iff the first arm is better (lower ADE6) by MORE than 2F_A (strict), EPOCHS > DATA the reverse, else NEITHER. A non-positive or non-finite floor, or a NaN delta, is NOT-YET.
* **Quantities** (`PAIRS`, `:38`): U-B_s0 (Delta_data), BL-B_s0 (Delta_ep), UE-BL, U-BL (the literal "U - BL"), and beside UE-B_s0, UE-U_s0. Paired episode-cluster bootstrap, B 2,000, seed 0 ("another draw of EPISODES"); F = one replicate pair (B_rep), a weak floor; ONE seed, reported as one seed (card 276).
* **Four families, every pair, never pooled** (`pair_family_status`, `fam_rows_all`): T1-kin cells (L1/L2, A1/A2, T1/T2 headlines + the rest of the 24 cells, ADE/FDE beside) with the card's cell rule and F_cell = abs(B_rep - B_s0) at stage A; STRATEGIC S1 / S2 / RC-A / S_CKPT zero at T0 from stage-A route passes, K-EVALX required. Per-member "not in this card" reasons as block 2. T's latcons (T0, beside) per pair with the card's T-DATA-LIMITED rule and the block-2 UNIT-SENSITIVE rule (`mech_ref` frames, `:249`).
* **Controls** (`controls`): K1 self-pair, T0 identity (BL vs itself), **K-RE (B1)** of the 216 T1 cells of U-B_s0.A, the ADE6 means, and the 15 mechanism rows (U vs B, R vs U, floor); **K-RE (B2)** of the block-2 copy of the same T1 rows; K-EVALX at stage A for five arms. No banked stage-A STRATEGIC row exists (K-RE does not apply there, stated in the result). Block-2 seed-1 rows are not inputs.
* Outcome precedence (`outcome_b3`, `:80`): BLOCKED (a bank item refused) > NOT-YET (ADE6 unread) > the label; a failed control makes it "(CONTROL FAILED ...; not quotable)".

### Reading choices (the card is silent; for the MM)
* **A. "U - BL" (card line 197) vs "UE_s0 - BL_s0" (line 195, third quantity).** Literal: U_s0 - BL_s0 (= Delta_data - Delta_ep; compares 3.2x unique clips with 3.2x steps, each against B). Also run: UE_s0 - BL_s0 (data at the larger compute; section 6 uses it for DATA-NEUTRAL). `label_literal` keeps the literal one; if the two differ the label is READING-SENSITIVE with both printed (the UNIT-SENSITIVE precedent). Say which the register row takes.
* **B. Bar is magnitude only**, as written ("better by more than 2F_A"); the paired interval is printed beside, not added as a condition. No bar moved.
* **C. STRATEGIC at stage A**: 5 arms x 5 variants (real, donor on the read tree; zero, realx, donorx on the STRAT1 tree). One stage-A pass (B_s0 real) ran at build time: rc 0, 99 s, ADE6 2.1361 (a code check, fixture deleted).
* **D. Block-1 pre-read (not a block-3 result):** F_A = 0.0828 m (so 2F_A = 0.1656 m) is printed in `out/smoke_b3_run.txt` (computed from the real block-1 B_rep / B_s0 evals); the stand-in report also showed Delta_data U_s0 - B_s0 ADE6 = -0.1280 m (its md was deleted with the fixture; recomputed in the real read).

## 3. File:line hops of the change
| change | where |
|---|---|
| block-3 constants (`ARMS_B3`, `B3_ITEMS`, `STEPS_ARM` 61,568, `STAGE_A_REF`, `ROUTE_ITEMS_B3`, `DS_BLOCK=3` default items) | `ds_lib.py:47-56` |
| `steps_of` (per-arm step count), `ck_root` (block-1 arms read their verified checkpoints from `read/ckpts`) | `ds_lib.py:59-67`, used in `bank_problem` (`:121`, `:134`) |
| stage-A route passes in block 3 (`ROUTE_STAGE`, stage-aware K-EVALX, route default items) | `ds_run.py:23,104,231,244,255` |
| `ck_path`, dump bank-record check use `ck_root` | `ds_run.py:27,148` |
| pull: `--min-free` (default the card bar 25; I used 5 for the banking step only, logged), bank dir default `read3/banker` | `ds_pull.py:108-113` |
| block-1 mechanism evals copied into one folder, sha256-verified | `sync_b1.py` |
| reader | `ds_read_b3.py` |
| unchanged copies of block-1 / block-2 code | `ds_read.py`, `ds_read_b2.py` (helpers only), `ds_t0.py`, `READ_PINS.json`, `theory/`, `ds_code/`, `make_pins.py`, `test_ds_read.py` |

## 4. Tests, mutations, smoke (MEASURED)
* `out/tests_all.txt`: `env -u DS_BLOCK ... pytest test_ds_read_b3.py test_ds_read.py` = **73 passed, 2 skipped** (the two DS_SLOW K-RE tests). New (32): strict 2F_A bar both signs and the boundary, floor guard, NaN guard, both readings agreeing / READING-SENSITIVE / NOT-YET, precedences, K-RE of ADE6 means, block-3 constants, `ck_root` roots, `bank_problem` wants 61,568 steps for the block-3 arms and 19,164 for the others (and refuses the swap), 25 stage-A route jobs on the right trees, K-EVALX / route completeness at stage A, `parse_items`, sync_b1 (idempotent, never overwrites), family rule, launcher wait and bars.
* **Mutations: `out/MUTATIONS_B3.json` (unmutated GREEN; 30 of 31 RED), the survivor m29 (the launcher skips the re-check of the reference checkpoints) fixed by a stricter test and re-run RED: `out/MUTATIONS_B3.partial.json`; smoke-level (reader wiring in `main`): `out/MUTATIONS_B3_SMOKE.json` 3 of 3 RED. So 34 of 34.** The first run of the unit list also failed its baseline (a test pinned the folder name `read3`; fixed to `HERE`).
* **Smoke `smoke_ds_read_b3.py`: SMOKE OK** (`out/smoke_b3_run.txt`; 92 s reader at B 200). Stand-ins: UE_s0 = R_s0's dumps/evals, BL_s0 = B_rep's, route passes from block-1 stage-B passes renamed; asserts K-RE passes (216 + 216 T1 rows at B1 and B2, 2 ADE6 means, 15 mechanism rows), STRATEGIC read, BL-B_s0 == B_rep-B_s0 and UE-U_s0 == the block-1 R_s0-U_s0 row cell by cell, the label's deltas == the independently read ADE6 mean differences, F_A == |B_rep - B_s0|. Its outcome (READING-SENSITIVE) is a stand-in, not a result.

## 5. Cost (ESTIMATED from block-2 MEASURED: dumps 318-378 s, mechanism evals 47-66 s, route passes 29-41 s; stage-A route pass 99 s measured once)
Pull skipped (already verified) ~0; pytest ~1 min; 2 dumps ~11 min; 8 mechanism evals ~8 min; 25 route passes 15-40 min; reader (6 + 6 pairs at B 2,000) ~5-6 min. **About 45-70 min.** Disk under 1 GB (checkpoints 0.17 GB already held). Pod: nothing runs (no GPU, no pod contact in the launcher).

## 6. UNVERIFIED
* The real run itself (dumps / mechanism / route on the UE and BL checkpoints) did not run: only wiring on stand-ins and one real stage-A route pass of B_s0.
* Stage-A route passes for UE_s0 / BL_s0 (their 61,568-step checkpoints) were not run; the evaluator accepts both trees' argv for stage A (`ds_run.preflight(..., route=True)` returned no problem with the F: bar stubbed).
* Whether the real UE_s0 / BL_s0 mechanism evals carry `ade6` is assumed from the same eval code (it does for the four block-1 stage-A arms).
* F: bar: not met now (22 GB).

## 7. MM launch step
```
powershell -File F:\TanitAD-scratch\datascale\read3\launch_wmi_b3.ps1
```
(after F: >= 25 GB). Sentinel `ZZ DSCALE10K B3 READ DONE ZZ` in `read3/out/dscale10k_b3.log`; result `read3/out/RESULT_DSCALE10K_B3.json` + `.md`. Re-runnable (finished items skipped). The block-2 read has finished (chain_done), so the launcher's wait returns at once.

## 8. Deletions log (own temporary data, literal paths)
`F:/TanitAD-scratch/datascale/read3/smoke` 106,924 KB (stand-in fixtures + stage-A route smoke, re-creatable by `smoke_ds_read_b3.py`); `.../read3/bt` 308 KB (pytest basetemp); `.../read3/ckpts/UE_s0.A/_hf` 8 KB and `.../read3/ckpts/BL_s0.A/_hf` 8 KB (hub download cache stubs); `.../read3/smoke_route_one.py` 4 KB (one-off script); `.../read3/mut` (mutation copies, removed by `mutate_check_b3.py`; two aborted attempts' leftovers removed by hand). Kept: `ckpts/` 166 MB (the verified dev-box copies; re-pullable from HF, delete after the read lands if F: is short).

## 9. Deliverable manifest
| artifact | where | single-copy? |
|---|---|---|
| UE_s0.A, BL_s0.A checkpoints + summary + BANK_RECORD | `read3/ckpts/` (dev box); pod `/root/dscale10k/runs/<arm>/A/`; HF private `Sayood/tanitad-v8f-dscale10k-runs` | no (3 copies) |
| banker records `UE_s0.A.json`, `BL_s0.A.json`, `hf_verify_b3.py` | `read3/banker/`, `read3/` | yes (dev box; re-creatable from the pod markers) |
| reader, launcher, tests, mutation / smoke scripts, `sync_b1.py`, modified `ds_lib.py` `ds_run.py` `ds_pull.py` | `read3/` | yes: F: scratch, not staged (the MM lands) |
| results `out/tests_all.txt`, `MUTATIONS_B3*.json`, `smoke_b3_run.txt`, `mutate_b3*_run.txt` | `read3/out/` | yes |
| this file, `HANDOFF.md` | `read3/` | yes |
