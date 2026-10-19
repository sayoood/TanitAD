# build_candidate_bank: how to make S0's bank for any REFe checkpoint (P6-READ-PREP, Research Lab agent, 2026-10-19)

Script: `code/build_candidate_bank.py` (a parameterised wrapper). Tests: `code/test_build_candidate_bank.py` (12, OK). Nothing here was run on a GPU, nothing touched the pod, and `srttc_s0/` was only read. Checked at the end (MEASURED): `find srttc_s0 -newermt "2026-10-19 02:00"` returns nothing (674 files) (S0 finished 00:39), and the unit test `test_zz_s0_directory_is_untouched` (path, size and mtime of every S0 file before and after the builder tests) passes.

Evidence classes: MEASURED = ours, artifact named. UNVERIFIED = not run here.

## 1. What an S0 bank is

A directory of units, one per dump batch (23 units in S0): `<unit>.sets.npz`, `<unit>.fan.npz`, `<unit>.report.json`, plus `MANIFEST.sha256.json` (sha256 and bytes of every unit file). S0's bank is `C:/Users/Admin/srttc_s0/work/bank/` (408 files, 0.56 GiB, MEASURED from its manifest).

`<unit>.sets.npz` keys (MEASURED by reading `L7c301cd189.sets.npz`, 57 sets):

| key | dtype, shape | meaning |
|---|---|---|
| `log`, `token`, `step`, `rank` | str, str, int32, int32 | the set's key; `rank` mod 5 is the fold |
| `ckpt_step` | int64 | the checkpoint step that produced the fit set |
| `feat` | float16 [S,64,256] | the input of `score_head`, per proposal |
| `logits` | float32 [S,64,6] | the candidate's 6 score-head logits per proposal, order `label_order` = nc, dac, ep, ttc, c, ddc |
| `top8` | int16 [S,8] | the frozen contender membership (rule `navsim_v1`, aggregate `p_nc * p_dac * (5 p_ep + 5 p_ttc + 2 p_c) / 12`, stable argsort, first 8) |
| `labels6` | float32 [S,64,6] | P2's PDM training labels for the same proposals (the label lookup) |
| `ctrl_traj_maxabs`, `ctrl_logit_maxabs`, `ctrl_fanpath_logit_maxabs` | float64 [S] | identity controls (section 3) |
| `rule`, `ckpt`, `what` | str | provenance; `ckpt` is the checkpoint file the dump ran |

`ep_admission.py` reads `logits` (the candidate's rows at every proposal), `labels6`, `top8`, the key and the fold. It never reads `feat`.

## 2. How S0's bank was made from P2 (the recipe this wrapper repeats)

All scripts are the LANDED SR-TTC S0 code in `C:/Users/Admin/srttc_s0/code/` (repo: `raw/2026-10-18-sel-rank/code/`), identical on the dev box and at the tip 807a29c (compared 2026-10-19). `build_candidate_bank.py` runs them BY PATH after a sha256 check (`S0_PINS`, 9 files) and re-implements none of them.

| step | script | what it does |
|---|---|---|
| plan | `srttc_s0_plan.py` | the registered sample: `master_plan.json` (117 logs, 8,922 tokens, 15,808 sets), `ranked_fit_logs.tsv` (sha256 `a61275037a3d2b06903dc7e1ad0f6f99fb740f668a6ec7488fdef6fe901ce9b1`), `selection.json` |
| sets | `srttc_s0_podsets.py` | per-log set stores `work/sets/<log>.npz` (traj3 [S,64,20,3], labels6, ckpt_step) built from the pod's stored on-policy 64-proposal fans of EARLIER checkpoints (`/workspace/data/refe_navtrain/onpolicy/sets`) and P2's PDM labels |
| stage | `srttc_s0_stage.py` | stages the nuPlan inputs (S0 staged 15.48 GiB over 19 batches, summed from `STATUS.json` `bytes_staged`) |
| dump | `srttc_s0_dump.py` | the forward (below) |
| drive | `srttc_s0_driver.py` | resumable per-unit loop: lock, resume, gates per unit (`gate_unit`), `S0_DONE`, `S0_SUMMARY.json`, bank `MANIFEST.sha256.json` |

**Which forward produced `feat` and the logits** (`srttc_s0_dump.py`):

1. ONE image forward per token through the planner (`planner._image_for`, `planner.infer`, line 123), using the checkpoint under test. The scoring decoder's visual context `sctx` is captured by a forward-pre-hook on the first scoring block `score_dec[0]` (line 101): the same tensor `score_trajectories` is given.
2. For every stored fit set of that token, the candidate scorer is applied to the STORED trajectories: `planner.model.score_trajectories(cxy, sctx)` (line 165) gives the set's logits [1,64,6]. This is the very call the trainer made (`REFe.forward(score_extra=...)` -> `score_trajectories`).
3. `feat` is hooked at the input of `score_head` in that same call (the hook must fire exactly once, line 167; the hooked output must equal the returned logits, line 171).
4. `top8` is the frozen rule on the logits (line 174). Folds are `rank mod 5`.
5. The `.fan.npz` of a unit holds the model's OWN 64-proposal fan for the token (its logits, its pick, a fingerprint); it exists to run the identity controls and is what the CPU smoke compares.

So the bank is the candidate's scorer applied to the SAME stored sets, with the SAME labels, as P2's. A candidate trained end to end (P6) is read on the sets it did not generate.

**The four constants this wrapper overrides in the landed driver** (and nothing else): `SRC` (the run root), `CKPT`, `CKPT_SHA256` and the GPU-lock `OWNER`. Two pitfalls found while wrapping it, both guarded:

- The landed driver DELETES the `S0_DONE` marker in its run root on start, so the run root must never be `srttc_s0/` (`load_s0_driver` refuses it; tested).
- DG-DUMP (`srttc_s0_dgdump.py`) has its own P2 checkpoint constant; it is not part of a candidate build and is not run.

## 3. Gates and controls built into a candidate build

- `preflight` (seconds, CPU): the checkpoint has `score_head` 256 -> 6, the same tensor names and shapes as P2, and compares `meta` (backbone, cameras, frozen_sha256, ...). On P2 itself: PASS, 643 tensors, format `refe-full-v1`, no difference from P2 (MEASURED, `raw/logs/preflight_p2.json`). If the head is not 256 -> 6 the landed `gate_unit` refuses every unit (`feat_dim_ok`), so a differently shaped P6 must be told apart BEFORE GPU hours.
- `verify-sets`: every rebuilt set store equals S0's `labels6` and `ckpt_step` on the same (log, token, step, rank) before any GPU is used. A changed pod store would change what the pick label means.
- per unit, the landed `gate_unit`: finite values, labels in [0,1], unique keys, keys in plan, `ckpt_step` equals the plan, positive identity control (P2's own fan through `score_trajectories` reproduces `forward`'s logits to 1e-4), negative control (a stored set differs from the fan).
- `manifest`: refuses without `S0_DONE` and exact counts (8,922 tokens, 15,808 sets); prints the bank manifest sha256, which the registration must NAME.
- `verify`: the finished bank is on P2's sets with P2's labels (bit-equal), identity controls hold, the `ckpt` stored in the bank equals the candidate's file, and NO set reproduces P2's logits bit for bit (that would mean the dump ran on the wrong checkpoint). Measured on a planted P2-copy bank: FAIL as it must (`same_logits_as_p2_on_any_set` 15,808).

## 4. Cost, from S0's own measured rate (RTX 4060 dev box)

| quantity | value | evidence |
|---|---|---|
| dump GPU time | 14,173.1 s = 3.94 h over 23 units | MEASURED, sum of `seconds` in the 23 `*.report.json` of `srttc_s0/work/bank/` |
| per token | 1.589 s/token (8,922 tokens) | MEASURED, same |
| S0 wall clock | 13 h 02 min (2026-10-18 11:37 to 2026-10-19 00:39) | MEASURED, `driver_main.log`; it includes P6-guard waits, a relaunch at 23:10 and the per-batch staging |
| bank on disk | 0.56 GiB (sets 0.53 GiB) | MEASURED, manifest |
| set stores | about 1 GiB | S0 `work/sets_pod` 997 MB, `work/sets` 257 MB (MEASURED with `du`) |
| staging | 15.48 GiB total, 19 batches | MEASURED, `STATUS.json` |
| free space asked | R on D: with at least 12 GiB free (`prepare` refuses less); the S0 driver itself stops below 5 GiB free on the work drive | code |

Plan for about 4 h of GPU on the RTX 4060 (ESTIMATED from the S0 rate, assuming the same forward cost for P6's architecture) and a wall clock up to the 13 h S0 took, depending on the dev-box lock and on staging.

**A40 alternative: UNVERIFIED.** No A40 speed-up was measured for this dump, and the pod environment for it (the nuPlan devkit, the maps, the planner imports of `srttc_s0_dump.py`, `P2HO_TREE`, the dev-box GPU lock protocol) is dev-box specific and was not checked on the pod. The pod is off limits to this step. If the Master Mind wants the A40, the first action is a 3-token `smoke` on the pod (section 5) before any batch; until then the 4060 path is the only one with a measurement behind it. P6 itself trains on the A40, so the A40 is busy until it finishes.

## 5. CPU smoke on P2 (the only forward run in this task)

Command (driverl-eval interpreter; `CUDA_VISIBLE_DEVICES=-1` is set by the wrapper; RAM gate 4 GB):

```
C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I build_candidate_bank.py smoke --ckpt D:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt --n-tokens 3 --workdir C:/Users/Admin/par18/p6prep/run/smoke
```

Result (MEASURED, `raw/logs/smoke_report.json`; first run `smoke_report_run1.json`): PASS, run twice, the second time with the final script.

| | |
|---|---|
| tokens | 3 of log `2021.10.05.06.24.06_veh-50_00717_01300`, unit `L1b95fda46f` |
| time, memory | 141.5 s total (an earlier run of the same command: 131.6 s, identical logit differences), peak RSS 3.06 GB (CPU forward is feasible) |
| max abs logit difference vs S0's banked FAN logits | 1.28e-3 (per token 0.75e-3, 0.81e-3, 1.28e-3) |
| tolerance | 5e-3 (the landed DG-DUMP CPU-vs-GPU tolerance; its own first smoke measured 1.74e-3) |
| picks, fingerprints | equal on all three |
| negative control (another token's banked logits) | differs by 3.88 to 4.86, so the comparison can fail |

**NOT reproduced: the SET path** (`score_trajectories` on the stored fit sets' `traj3`). The stored trajectories live only on the pod and S0's `work/sets` was deleted after banking; the pod is off limits here. The fan path above is the same forward, the same weights and the same hooks; the set path differs only in the trajectories it is fed. The set path's own positive identity control (the fan through `score_trajectories` equals `forward`, bound 1e-4) is read per unit by the landed gate on the day.

## 6. Commands (the day-of order is in `READY_P6_READ.md`)

Interpreter for every `build_candidate_bank.py` command: `C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I <path>/build_candidate_bank.py <sub> ...`

```
preflight   --ckpt C [--ckpt-sha256 X] [--out F]
prepare     --root R --ckpt C [--ckpt-sha256 X] [--candidate-id p6]
sets        --root R [--skip-fetch]
verify-sets --root R [--p2-bank B]
dump        --root R [--device cuda|cpu] [--no-stage] [--dry-run]
launch      --root R                (prints the WMI-detached PowerShell body)
status      --root R
manifest    --root R
verify      --root R [--p2-bank B]
smoke       --ckpt C [--n-tokens 3] --workdir W
```

`dump --dry-run` imports the landed driver with the overrides and exits without taking the lock (tested). `dump` itself takes the shared dev-box GPU lock through the landed driver, never from this wrapper; the GPU launch needs the coordinator's approval.

## 7. What was and was not verified

| claim | status |
|---|---|
| the 9 landed S0 files are pinned and the wrapper refuses a changed one | MEASURED (test `test_a_changed_landed_file_is_refused`) |
| the four-constant override routes the landed driver's preflight to the new root and hashes the candidate checkpoint | MEASURED (test `test_the_driver_override_routes_everything_to_the_new_root`) |
| a bank written in S0's format passes `manifest`, `verify` and the admission CLI at S0's full geometry (15,808 sets, 117 logs); a P2-copy bank is refused | MEASURED (`raw/logs/integration_disk_replication.json`, 93 s; D_M equals the landed EP-RELIAB value to 1e-9) |
| the planner forward on CPU reproduces S0's banked fan logits within 5e-3 | MEASURED, 3 tokens, 1.28e-3 |
| the SET path reproduces S0's banked set logits | NOT REPRODUCED (traj3 only on the pod) |
| a real GPU `dump` on a candidate | NOT RUN (by the brief) |
| the A40 can run the dump | UNVERIFIED |
| P6's checkpoint has P2's architecture | UNKNOWN until `preflight` runs on it; the checkpoint does not exist yet |
