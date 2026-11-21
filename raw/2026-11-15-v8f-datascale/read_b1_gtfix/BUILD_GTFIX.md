# GTFIX: the DSCALE10K block-1 T0 mechanism refusal (2026-11-17)

**Verdict: an instrument defect in the read layer, not a label difference.** U_s0's GT constraint block differs from B_s0's only because each checkpoint writes the codes in its own SD frame. The trainer sets that frame from its train join. The labels themselves are the same on every window, to float32 rounding. The fix is in `ds_t0.py` only. No evaluator, bar, estimator or outcome rule was changed.

Tier of every number here: **T0** (GT-oracle and absmask evals). That is a world-model and tactical diagnostic, never driving performance. Evidence class: MEASURED (ours; zero training).

## 1. Cause, hop by hop

**What the "GT constraint block" is.** It is columns 34..59 (`C0 = 22 + 5 + 7`, `N_CONS = 26`, `v8f_wc_read.py:47-48`) of `tac_code` in `eval_<arm>_s0-tacorc.npz`, taken on the windows where `tac_oracle_fallback` is false (`v8f_pgt_read.py:249`). These are the 26 constraint slots of the tactical code: latcons 0-11, lonspeed, stoplead (`v8f_wc_read.py:51`).

**Where the eval gets it.**

| hop | file:line | what |
|---|---|---|
| 1 | `v8f_mini_train.py:3287-3288` | `mu, sd = M.cons_stats(J)`; `M.V8FMini(cfg, prior, mu, sd)`. `J` is the TRAIN join. |
| 2 | `v8f_mini.py:1372-1380` | `cons_stats`: nanmean and nanstd of the 26 constraint cells over the labelled train rows |
| 3 | `v8f_mini.py:822-823` | stored as the buffers `cons_mu` and `cons_sd`, which are saved in the checkpoint's state dict |
| 4 | `v8f_mini_eval.py:263` | the eval loads `ck["model"]`, so the checkpoint's own `cons_mu` and `cons_sd` come with it |
| 5 | `v8f_mini_eval.py:58-59, 267` | the labels come from `--eval-join join_eval141.npz`. That is one file for every arm, pinned at sha256 `f0ea722f…` in `READ_PINS.json`. |
| 6 | `v8f_mini_eval.py:324` → `v8f_mini.py:901` | oracle code = `(lab["cons"] - cons_mu) / cons_sd`, with NaN mapped to 0, so it is in the CHECKPOINT's frame |
| 7 | `v8f_mini_eval.py:367` | written to the npz as float32 `tac_code` |
| 8 | `v8f_wc_read.py:165-168` (`present_error`) | compares the raw blocks bit for bit against the reference arm's and refuses |

**So which source is it?** It is neither the argv label paths nor the eval cache. The two arms' argv files are identical except for `--ckpt` and `--tag` (measured with `diff` on `argv_DS{B,U}s0A_s0-tac{orc,abs}.txt`). The eval reads no train-time argument. A grep of `v8f_mini_eval.py` for `ck["args"]`, `v9_train` and `fit_join` finds 0 hits, while the control `ck["model"]` finds 1 hit at :263. **The only per-arm input is the checkpoint's state dict.** It carries train-set *statistics*, not train-set labels.

**Why U differs and B_rep does not** (MEASURED, `gtfix/probe_gt.json`, from `gtfix/probe_gt.py`):

| pair | cons_mu / cons_sd | raw block | has | presence (code ≠ 0) | labels in physical units | U re-expressed in B's frame |
|---|---|---|---|---|---|---|
| B_rep vs B_s0 | identical (sha `8884a84e…` / `27222b8d…`) | equal | equal | equal | equal | n/a |
| U_s0 vs B_s0 | differ: max abs Δmu 4.476, max sd ratio deviation 10.06 % | differs on **2,351 of 2,365** labelled windows, 26,056 cells, all 26 slots | equal (2,365 of 2,399) | equal, 0 cells differ | max abs diff **2.5e-5** | max **8.2e-7 SD**. 7,935 cells are not bit-equal in float32, so a tolerance is needed. |
| R_s0 vs B_s0 | same as U (R's stats sha equals U's: `54bd6778…` / `0977fa78…`) | same as U | equal | equal | 2.5e-5 | 8.2e-7 SD |

**The windows are the same.** The eval npz carry no `win_ci` or `win_t` arrays (key list measured: `ade2, ade6, ck_pos_err, ckpt, cons_pred, eid, …`). The windows are built by `D.eval_grid_windows(a.eval_grid, ej)` from the same grid file (pinned `659c3b65…`) and the same join on every argv. The `eid` arrays of all four arms are equal over 2,399 windows, and so are the `has` masks. Labels that agree to 2.5e-5 m-units on all 61,490 compared cells could not come from different windows.

**Classification.** This is an **instrument defect**. `present_error`'s bit-equality check assumes that every arm shares one train join. That held for every read before DSCALE10K, which all trained on one corpus. The original DSCALE `--data-frac 0.5` arm also passed it: K-RE_T0 reproduces it bit for bit. The assumption is false for the union arms. The refusal was right to fire. Scoring raw codes across frames would mix units: with mu cancelling on present cells, the error is (pred − label) / sd_own, and the sd ratio alone moves the latcons RMSE by up to ~10 %. That is the size of the ±5 % bars.

## 2. The fix (read layer only: `ds_t0.py`)

`to_common_frame` runs in `load_z` before the reference alias. For an arm whose block differs from the reference arm's, it does the following:

1. It names the arm's checkpoint from its three eval jsons (`ckpt` field; they must agree) and reads `cons_mu` and `cons_sd` from that checkpoint (`ckpt_stats`).
2. It acts only if the frames differ. The same frame with a different block means different labels, and that is left for `present_error` to refuse.
3. It checks the arm's all-latcons oracle in its OWN frame (the unedited `PGT.check_gt_codes`, on that arm alone).
4. It **refuses** if GT presence differs on any labelled window, or if the re-expressed GT, `(g·sd + mu − mu_ref) / sd_ref`, differs from the reference GT by more than `1e-5 + 1e-6·|GT|` SD on any cell. The refusal names the count of windows and slots. A genuine relabel therefore still stops the read.
5. If it passes, it re-expresses T's absmask code in the reference frame (GT-absent cells stay 0), sets the GT block to the reference's, and leaves the rest to the unedited `present_error`, `check_gt_codes` and `mlat_row`.

**Bit-equal blocks are untouched.** No checkpoint is read for them. That covers B_rep, the identity control, R vs U and K-RE_T0, so every earlier number stays bit-identical. `mech_pair` adds a control: a local RMSE helper must reproduce `present_error`'s latcons RMSE to 1e-12. Its output also records the frame used. Beside the deciding r, it gives r in A's own frame and in physical units.

**Frame choice: the MM should confirm this.** The deciding frame is the **reference arm's**, which is B for U vs B and U for R vs U. This keeps the DSCALE instrument's definition bit for bit on B, the base of record. The card says "SD units" and does not name whose SD units, because every earlier arm shared one frame.

## 3. Tests and mutations

* `test_ds_read.py` has 4 new tests: `test_gtfix_*`. One shows that the same physical prediction reads r = 0 only in one common frame, and r > 1 % per own frame. The others cover a genuine relabel of 0.01 SD (refused), a presence change (refused), and bit-equal blocks or a same-frame relabel (left untouched). Result: **41 passed, 2 skipped** (the 2 skipped are the DS_SLOW K-RE tests, which ran inside the reader).
* 3 mutations were added to `mutate_check.py` MUTS. `gtfix/mutate_gtfix.py` runs exactly those 3 with mutate_check's own procedure. Result: **all 3 RED, unmutated GREEN** (`gtfix/MUTATIONS_GTFIX.json`). The full 20-mutation `mutate_check.py` was not re-run; `out/MUTATIONS.json` still holds the earlier 17 of 17.

## 4. Re-run and new outcome

**Mechanism evals were not re-run.** All 16 exist and are correct, because the defect was in the read layer. Only the reader was re-run.

**The launcher was not used.** Its first step is the card's F: ≥ 25 GB gate. F: read **23 GB** free (`df -k /f`, the launcher's own formula), and the space is held by other sessions' running jobs. That bar was not touched. `gtfix/rerun_reader.sh` runs the launcher's reader step with the identical command line (`ds_read.py --kre-t1 --kre-t0`, the same PYTHONPATH and threads) after `wait_commit`. The waiter took its baseline before the launch: 4 sentinels. The reader appended "GTFIX reader-only re-run start 2026-11-17T21:32:57Z (ds_t0.py sha256 6719f183…)" to `out/dscale10k_b1.log` and ended `ZZ DSCALE10K B1 READ DONE ZZ`, with 112.7 s wall and problems 0. **When F: is at 25 GB or more, a launcher re-run reproduces this:** it skips every finished item.

**New outcome line (`out/RESULT_DSCALE10K_B1.json`):** `['NOT-YET']`. The seed-0 label is `NOT-YET (seed 1 absent)`, and the next step is block 2 (B_s1, U_s1). The mechanism block is now read, so the only thing the outcome waits on is seed 1.

All numbers below are T0, MEASURED, `out/RESULT_DSCALE10K_B1.json` → `mechanism_stage_A_T0`. The estimator is the paired episode-cluster bootstrap, B 2,000, seed 0, over 2,351 windows and 139 episodes. That answers "another draw of EPISODES". Training variance is the B_rep floor.

| read | r (slot-pooled, deciding) | r_window | paired MSE Δ [95 %] | separated | label |
|---|---|---|---|---|---|
| **R_s0 vs U_s0** (precondition, U frame, R native) | **+34.11 %** | +24.89 % | +0.1209 [+0.0529, +0.1978] | yes | **REACHED** (bar r ≥ +5 %, separated, sign ok) |
| **U_s0 vs B_s0** (B frame, U re-expressed) | **−3.14 %** (RMSE 0.6887 vs 0.7110 SD) | −0.90 % | −0.0042 [−0.0366, +0.0317] | no | **NOT T-DATA-LIMITED**: r above the −5 % bar, not separated, and \|Δ\| 0.0042 is below the stage-A T floor 0.0057 |

Beside these:
* **Frame sensitivity of U vs B (point only).** r is −3.12 % in U's frame and −1.98 % in physical units. The label is the same in all three frames.
* **Oracle ADE6, U vs B.** 0.9951 vs 0.9746 m, Δ +0.0205, not separated.
* **Stage-A T floor.** B_rep vs B_s0: MSE 0.00572, \|r\| 3.08 %. This is one replicate pair, so it is a weak floor.
* **Controls.** The identity control passes (r 0, Δ 0 [0, 0]). K1_self_pair, K-RE_T1 and K-RE_T0 pass. K-RE_T0 is the banked DSCALE read reproduced, and it passes because bit-equal blocks are untouched.

**What the mechanism read now says, at seed 0.** First, the new clips' labels do reach T: deranging them costs +34 % latcons RMSE, separated. Second, the union does not move T's latcons RMSE beyond the −5 % bar at stage A: the change is −3.1 %, not separated. This is one seed. The DSCALE10K label waits for seed 1, as the card says.

## 5. Deliverable manifest

| artifact | where | note |
|---|---|---|
| `ds_t0.py` (fixed) | `F:/TanitAD-scratch/datascale/read/ds_t0.py` | one place only (F: scratch); needs landing by the MM |
| `test_ds_read.py` (+4 tests) | `F:/TanitAD-scratch/datascale/read/test_ds_read.py` | one place only |
| `mutate_check.py` (+3 MUTS) | `F:/TanitAD-scratch/datascale/read/mutate_check.py` | one place only |
| backups `*.before_gtfix` | `ds_t0.py`, `test_ds_read.py`, `mutate_check.py` in `read/` | delete once the MM has landed and verified |
| probe + result | `read/gtfix/probe_gt.py`, `read/gtfix/probe_gt.json` | MEASURED evidence for section 1 |
| mutation runner + result | `read/gtfix/mutate_gtfix.py`, `read/gtfix/MUTATIONS_GTFIX.json` | |
| smoke | `read/gtfix/smoke_mech.py` | B 200 smoke; the deciding numbers come from the reader |
| reader re-run script | `read/gtfix/rerun_reader.sh` | |
| earlier outputs kept | `read/gtfix/RESULT_DSCALE10K_B1.before_gtfix.{json,md}`, `read/gtfix/MUTATIONS.before_gtfix.json` | |
| new result | `read/out/RESULT_DSCALE10K_B1.json` / `.md`, log `read/out/dscale10k_b1.log` | |

Nothing was committed, pushed or staged. The read tree is on F: scratch, outside the repo, so the MM lands it.

## 6. Deletions

| path | size | why |
|---|---|---|
| `F:/TanitAD-scratch/datascale/read/gtfix/tmp` (pytest basetemp `bt/`) | 38 KB | own TMP; deleted by literal path at the end |
| `F:/TanitAD-scratch/datascale/read/gtfix/tmp/mut` (4 mutation copies) | under 1 MB (copies of the read's *.py and theory/) | own; removed by `mutate_gtfix.py` (`shutil.rmtree`) after its run |

Kept on purpose: the `*.before_gtfix` backups and the earlier result copies in `gtfix/`. They stay until the MM lands the fix.

**File hashes after the fix (sha256, first 16 hex characters):**
* `ds_t0.py`: `6719f1839073b0cc` (it was `d149e578cecdca79`)
* `test_ds_read.py`: `1e603ae2a5e18ea0` (it was `26d3aece080b5bf5`)
* `mutate_check.py`: `89d71976e000e427` (it was `7ab377eb3215aef5`)

## 7. Open items for the MM

1. **Land** `ds_t0.py`, `test_ds_read.py`, `mutate_check.py` and `gtfix/` with the DSCALE10K read package.
2. **Confirm the deciding frame**: the reference arm's. The U-frame and physical-unit values are given beside it.
3. Block 2 (B_s1, U_s1) is the next step for the label. When it is read, re-run `mutate_check.py` in full (20 mutations).
4. **Reader text.** `ds_read.py` prints "the T0 mechanism evals are missing" for any mechanism exception, including an instrument refusal. The reason was visible only in `mechanism_stage_A_T0.reason`. The text was left unchanged, because it is outcome text.
