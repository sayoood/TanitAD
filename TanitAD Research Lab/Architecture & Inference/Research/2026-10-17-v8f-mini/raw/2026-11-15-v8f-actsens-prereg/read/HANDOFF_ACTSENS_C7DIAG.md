# HANDOFF ACTSENS C7 diagnosis (2026-11-15): HWM plans differ because of the torch thread count; driver patched; C7b rerun running detached

Tier T0 (oracle goal, world-model diagnostic). No training. Box 2 not written (4 TRAIN clips read by sftp for one probe, as the driver itself does).

## Integration to escalate
* `F:/TanitAD-scratch/actsens/read/read_actsens.sh` is **patched in place** (original kept beside it as `read_actsens.sh.orig_c7diag_20261115`, sha256 `7b379fa8...`). The read scripts live only on F: scratch, not in git: the Master Mind decides whether to bank them.
* No ACTSENS arm was read with the old driver (only `out/C7_R35_s0` and `out/C8` exist), so no arm output needs to be redone.

## Root cause (MEASURED)
**Phase 1 planned at torch intra-op threads 1. C7 planned at 2.** Torch takes its default thread count from `MKL_NUM_THREADS` when it is set, and only otherwise from `OMP_NUM_THREADS`:
* phase 1: `F:/TanitAD-scratch/hwm_plan/common.sh:1` sets `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. `phase1_R35.sh:6` then raises only OMP to 2, so torch ran at **1** thread.
* read driver: `F:/TanitAD-scratch/actsens/read/env.sh:4` sets `MKL_NUM_THREADS="${OMP_NUM_THREADS:-2}"`, which is 2. `read_actsens.sh` (original line 70) sets only OMP=2, so torch ran at **2** threads.
* Measured with `torch.get_num_threads()` in the venv (torch 2.11.0): OMP 2 + MKL 1 gives 1. OMP 2 + MKL 2 gives 2.
* Two inherited premises were wrong. "Both used OMP 2" hides the MKL setting. Phase 1 also ran with **`--workers 1`**, not 2: `F:/TanitAD-scratch/hwm_plan/phase1_R35/WAITER_STATUS.txt` reads `workers=1`, because the waiter picks 1 when free RAM is below 8 GB.

**Mechanism.** CPU `nn.Linear` rounds differently at 1 and 2 threads on large batches. The first module that diverges is `act_embed.0` (Linear 2→32, `tanitad/models/hjepa.py:524`): its input is bit-equal and 1 output element differs (`stage_probe_result.json`). This drift reaches the HWM path at two places:
1. **The prior fit.** `hwm_data.py:34` runs `st.encode` on the TRAIN latents. The macro actions `act1` differ in 114 of 293,280 elements and `act2` in 286 of 58,656. As a result the level-3 prior mean (`hwm_data.py:37`) differs in 6 of 32 dimensions, by at most 3.7e-9. The level-1 and level-2 priors stay bit-equal.
2. **The planning itself.** The HWM planner's own operations also depend on the thread count.

The HWM planner amplifies these 1e-9 differences into plan differences of metres. The flat path is bit-equal at both thread counts.

## Discriminating experiments (one unit: hwm_oracle, inference seed 0, batch 0, 64 windows; `c7diag/one_unit.sh`; evidence in `c7diag/evidence/E*.npz.json`)
| run | tree | torch threads | priors fitted at | vs banked phase 1 | vs C7 rows |
|---|---|---|---|---|---|
| E1 | ladder_tree | 1 | 1 | **bit-equal, all 6 arrays** | differs (plan_raw max 1.88) |
| E3 | ACTSENS tree | 1 | 1 | **bit-equal** | differs |
| E2 (control) | ACTSENS tree | 2 | 2 | differs (plan_raw 1.88) | **bit-equal** |
| E5 | ACTSENS tree | 2 | 1 (injected) | differs (2.10) | differs (1.37) |
| E6 | ACTSENS tree | 1 | 2 (injected) | differs (1.85) | differs (1.88) |

⇒ **The code tree is not the cause** (E1 = E3). **The worker count and pool mode are not the cause**: E2 ran in the main process and reproduces C7's spawn-pool rows. Either half of the thread effect alone (priors or planning) is enough to move the plans (E5, E6). **Self-repeatability at a fixed thread count holds**: E2 matches C7 although the two ran 2 h apart in different process modes, and E1 matches E3.

**Real-path check of the fix** (`c7diag/realpath_check.sh`, `realpath_result.json`). The patched step-4 command line, with env.sh, MKL 1 and `--workers 2` (spawn pool), ran on the first 128 SCORE windows with hwm_oracle seed 1. It is **bit-equal to banked on every array**. This covers batch 1, where C7's `iters1` was 82 and phase 1's was 81: the patched run gives 81.

**Collector probe** (`collect_probe_result.json`). The driver's `collect_latents_arm.py` at OMP 4 and at OMP 1, run on the first 4 of the 141 TRAIN clips (68 windows), reproduces the banked `latents_train_R35_s0.pt` rows bit-for-bit. Step 3 needs no change. This is MEASURED on 4 clips only.

## The fix (record of every change to `read_actsens.sh`)
1. Header: 3 comment lines explaining the change.
2. Step 2, windows: prefixed with `OMP_NUM_THREADS=2 MKL_NUM_THREADS=1` (phase 1's windows env).
3. Before step 4: a fail-closed probe, `fail` unless `torch.get_num_threads() == 1` in the plan env. Its result is logged.
4. Step 4, plan: `OMP_NUM_THREADS=2` became `OMP_NUM_THREADS=2 MKL_NUM_THREADS=1`.

Not changed: env.sh, the IA read (it matched exactly), decode_check (step3_nn matched), the collector, and the default `HWM_WORKERS` 3.

## Running (detached, WMI pid 41192, started 04:52)
**C7b** is the full driver run with R35_s0 as the arm, the patched driver and `HWM_WORKERS=2`. Its outputs go to `F:/TanitAD-scratch/actsens/read/c7diag/out/C7b_R35_s0/`. This folder is not the verdict's input folder, so `arm_*.json` does not mix with real arms there. ETA is about 2-2.5 h (ESTIMATED from the per-unit times). Done when `ARM_DONE_C7b_R35_s0.txt` or `ARM_FAILED_*.txt` exists, and `c7diag/out/c7b_run.exit` has a line.

**Next step:** read `c7diag/out/C7b_R35_s0/c7_result.json`. Committed outcomes:
* `C7_PASS` true, with all 4 plan files bit-equal, the check reproducing exactly, and arm-minus-R35 at 0.0 [0, 0] on 39+39 cells ⇒ the fix is verified on the full chassis-0 run. The arms can be read with the patched driver.
* any HWM file not bit-equal ⇒ a second cause exists. Locate the first differing batch, and rerun it with `one_unit.sh <label> <tree> 1 0 hwm_oracle <seed> <batch> <c7b win dir>`.

Chassis 1 (R35_s1 as the arm) has not been controlled yet. A C7 on s1 costs the same time.

## Measured caution and proposals for the Master Mind (not decided here)
* **A pure rounding perturbation produced "separated" cells.** In the original C7 (`out/C7_R35_s0/contrast_hwm.json`), the same checkpoint at 2 threads against 1 thread gives a separated CI on **2 of 39** cells: LATERAL head_err 2-6 s and full6s, −0.0022 [−0.0034, −0.0009]. The 0-2 s cells all stay below 0.0003. Same lesson as `H-ESTIM-SEED-1`: the episode bootstrap cannot see this variance.
* **Proposal A** (preferred): keep arms and the R35 reference in one numeric environment, using the patched driver at 1 thread. Exact reproduction is possible, so regenerating the R35 reference plans is not needed.
* **Proposal B** (fallback, only if a later environment change breaks bit-equality): regenerate R35_s0 and R35_s1 reference plans in the read environment with the same driver, and quote the 2-of-39 thread floor beside every HWM contrast. The tolerance and the card's bar are the Master Mind's call.

## Deliverable manifest (all on the dev box; single copies unless noted)
* `F:/TanitAD-scratch/actsens/read/read_actsens.sh` (patched) and `read_actsens.sh.orig_c7diag_20261115` (original).
* `F:/TanitAD-scratch/actsens/read/c7diag/`, containing:
  * scripts: `one_unit.py/.sh`, `prior_probe.py`, `encode_probe.py`, `realpath_check.sh`, `collect_probe.sh`, `c7b_run.sh/.cmd`;
  * results: `realpath_result.json`, `stage_probe_result.json`, `collect_probe_result.json`;
  * `evidence/` (E1-E6 json and logs, probe logs, `deleted_temp_manifest.tsv`);
  * `out/C7b_R35_s0/` (running).
* This handoff.

## Deletions (own temp data, re-creatable from the scripts above, plus box-2 clips whose source was just read)
`F:/TanitAD-scratch/actsens/read/c7diag/tmp`, 298,307,632 B in total, 40 files listed with their sizes in `evidence/deleted_temp_manifest.tsv`. The largest items:
* `tmp/enc_mkl1.npz` and `tmp/enc_mkl2.npz`, 124,808,284 B each;
* `tmp/col1` and `tmp/col4`, 21,477,911 B each (including each one's 20.9 MB `_current.pt` clip copy);
* `tmp/rp`, 2,212,096 B;
* `tmp/prior_mkl{1,2}.npz`, 1,636,618 B each;
* `tmp/E*.npz`, 46,802 B each.

F: has 8.6 GB free.
