# PLAN -- the FINAL-checkpoint chain (refcv8 step 48,852)

Written 2026-10-19 (EvalFlyWheel engineer). PKG = `C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery` (unlanded qland hold; nothing git-added, nothing on G:).
Evidence classes: MEASURED (ours, artifact named) / ESTIMATED / UNVERIFIED. Thor was touched read-only (ssh -n stat/grep/md5sum/test, one scp of `config.json`).

## 1. What the trainer writes at the final step (MEASURED from source)

Source: the launch-tree `stack/scripts/refc_v3_train.py` on Thor, md5 `ecfb366e48015f7deffe25cda99aa2aa`, **byte-identical** to the battery's pinned export `C:/lgs/Lfb47a58376/stack/scripts/refc_v3_train.py` (md5 compared 2026-10-19 08:05).
Run: `--steps 48852 --save-every 500 --eval-every 500 --log-every 50` (`config.json['argv']`), EMA decay 0.9998 (`config.json['ema']`).

| fact | file:line |
|---|---|
| `MILESTONES = (5000, 15000, 20000, 30000)` -> **no `ckpt_48852.pt` and no `ckpt_ema_48852.pt` are ever written** (same reason 10,500 / 40,000 needed a snapshot) | `refc_v3_train.py:232`, `:10834-10839` |
| the step-48852 metrics row is written **before** the checkpoint (`step % log_every == 0 or step == args.steps`) | `:10723` |
| `ckpt.pt` (full: `model`, `opt`, `step`, `data_pos`, `ema` state, `b1_guard`) is rewritten **in place** at `step % save_every == 0 or step == args.steps`; the line `[v3:hier] ckpt step N -> ckpt.pt` is printed **after** the write | `:10822-10832`; payload `:5167-5191` |
| `model_ema.pt` = EMA weights, model-only, `_save_ema_model` layout `{model, step, ema_decay, num_updates, kind:'ema_model_only'}` -- written **inside the loop only**, at `step == args.steps` (a resume AT step==steps skips it) | `:10840-10842`, `:5203-5207` |
| final in-training eval at `step == args.steps` (so `metrics.jsonl` gets an eval row at 48852; the battery's `pick_metrics` needs exactly one) | `:10849-10852` |
| `summary.json` `{"done": true, "step", "arm", "seed", "wallclock_s", "ema_decay", "ema_num_updates", "b1_guard"}` written **after** the final eval; then `DONE at N` | `:11039-11046` |
| the **supervisor overwrites** `summary.json` with `{"done": true, "arm", "final_step", "steps_requested", "out_dir", "gate_token", "written_utc", "note"}` (**`final_step`, not `step`**) whenever `metrics.jsonl` reaches STEPS-1 (up to POLL_S=300 s after the trainer exits) | `stack/ops/sup_refcv7.sh:177-215`, `:245-249`, `:312-317` |

### Consequence 1 -- summary.json is necessary, never sufficient
The metrics row (`:10723`) precedes the save (`:10822`). A kill in that window (the 2026-10-19 06:18 global OOM is that class) leaves `metrics.jsonl` at 48852, so `run_is_complete` is true, the supervisor **writes a done-marker with `final_step: 48852` and does not relaunch** -- while `ckpt.pt` is still step 48,500 and no `model_ema.pt` exists.
The proof of completeness is therefore a conjunction, each part on an artifact:
1. `summary.json` done@48852 (either schema) -- *necessary*;
2. `train.log` carries `ckpt step 48852 -> ckpt.pt` (printed after the write) -- else the chain stops with `ZZPFNHOLEZZ`;
3. `stat` (size, mtime) of `ckpt.pt`/`summary.json`/`model_ema.pt` identical across a 180 s settle and across the copy;
4. size inside the band 1.50-1.62 GB (**not** equality, see Finding 1), local size == remote size, **md5 on both ends equal**, both 32 hex (an empty string is INCONCLUSIVE, never a match);
5. the chain re-reads the md5 four ways (local, `.md5` sidecar, `.provenance.json`, the DONE marker);
6. the step **inside** the file: `extract_ema_r8.py extract --expect-step 48852` (refuses before any model build) **and** the chain re-asserts `step==48852, ok==true, src_md5==this pull` on the provenance artifact; EMA `num_updates == 48852` is asserted by the extractor;
7. EMA gate (`ema_check_r8.py`: strict load clean, EMA differs from raw, md5-pinned).
8. **Independent route for the EMA weights** (new): `xcheck_model_ema_r8.py` compares the trainer-written `model_ema.pt` with the EMA derived from `ckpt.pt['ema']` tensor by tensor (N/A, and said so, if the trainer wrote none).

### Artifacts to pull (all from `/home/nvidia/refcv8_run/runs/refcv8-r101-s0/`, read-only)
`ckpt.pt` (~1.56 GB) -> `E:/refcv8_eval_kit/ckpt/ckpt_full_48852.pt` (+ `.md5`, `.provenance.json`); `model_ema.pt` (~403 MB, optional) -> `ckpt/model_ema_48852_thor.pt` (+ `.md5`); `summary.json` -> `ckpt/summary_thor_48852.json`; `metrics.jsonl` (the existing `thor_pull_r8.sh` metrics copy, `thor_reads/`). `ckpt.pt` is **not** overwritten after the last step, so unlike 40,000 there is no race window and no snapshot poller is needed.

## 2. Stages and dependency order (`code/chain_final_r8.sh`, markers `ZZCFN...ZZ` in `E:/refcv8_eval_kit/chain/chain_final.log`)

```
a PULL (pull_final_r8.sh: wait -> settle -> scp -> md5 both ends)  -> ZZCFNPULLSEEN, MD5OK, SUMMARYOK            [abort a on any failure]
b EXTRACT (--expect-step 48852) -> EMA GATE -> XCHECK                -> EXTRACTOK, EMAGATEOK, XCHECKOK|NA       [abort b on any failure]
 |
 +-- GPU, serial, each stage takes the shared O_EXCL lock itself ----------------------------------------------------------------------------+
 |   c EMA battery (step48852ema_unc, seeds 0,1, --no-g0, PROVISIONAL, pair_read vs refcv7-50,400, bank)   -> EMADONE | EMAFAIL
 |   k K-draw rolls (kdraw_queue_r8.sh KD_STEP=48852: ctrl + seeds 2..7, ONE lock hold, RELEASE_R7 not rewritten)  needs c  -> KDDONE | KDSKIP | KDFAIL
 |   (K-draw read: run_kdraw_analysis_r8.sh step48852ema_unc, CPU, right after k)                              -> KDANALYSIS overall=...
 |   f RAW: metrics copy -> chain_g0_r8.sh 48852 -> wrapper probe -> chain_battery_r8.sh 48852 (pass 1)   -> RAWDONE | RAWINCOMPLETE
 |
 +-- CPU LANE (background, memory-gated >= 5 GB commit, one job at a time; EMA tree first, then RAW) ---------------------------------------------+
     d clamp_posthoc read -> compliant_reread read (BOTH, this order)  [capped plans on ceiling-compliant windows; E1/E3 vs ha0_ext_pose, ha0_ext beside]
     e a0k0_headroom / _lon / _retime `forward --tree T` (frozen 30k files UNCHANGED)
     g heading_drift (earlier list = only trees that exist)
     then: retime_trainfit `forward --tree T` when FROZEN_GAINS_RETIME_train.json(+.sha256) exists (waits <= CFN_TF_WAIT_S=4 h for the train-fit chain)
     then: render x3, summarize_40k_r8.py --step 48852 -> raw/step48852/READ_FINAL.{json,md}
 tail: combo (run_combo_ens_retime_r8.sh step48852ema_unc train; needs K-draw rolls + EMA tree + train gains, names the missing one otherwise) -> final_index_r8.py -> raw/step48852/FINAL_INDEX.{json,md}
```
Stages are independent where the data allow: a failed EMA battery skips the K-draw and the EMA lane but not the raw stage; a failed K-draw blocks only the combo. Resumable: re-run the same command (a stage whose artifact says done is skipped).
Terminal markers: `ZZCFNENDZZ` (line lists `ema= kd= raw= combo=`) or `ZZCFNABORTZZ` (stage a or b).

## 3. Inputs and their status

| input | status |
|---|---|
| `ckpt.pt` / `model_ema.pt` / `summary.json` at 48,852 | **does not exist yet** (Thor at step ~36,100 at 08:05 on 10-19; ETA below) -- produced by the trainer |
| `E:/refcv8_eval_kit/ckpt/config.json` | present. Diffed against Thor's `config.json` rewritten by launch 2 (06:28): only run-metadata differs (`resumed_from`, `b1_guard` decision `keep`, `ema.init/num_updates_at_start 35500`, `init_from.applied`) -- **no architecture field**. MEASURED (scp from Thor, key-by-key diff). |
| `ckpt_full_48852.pt`, `ckpt_48852.pt`, `ckpt_ema_48852.pt`, `.md5`, provenance | produced earlier in the chain (stages a, b) |
| `thor_reads/metrics_*.jsonl` with exactly one eval row at 48852 | produced (thor_pull_r8.sh, 'have' mode because `ckpt_48852.md5` exists); the final eval row exists by `:10849` |
| `battery/code/{gpu_lock,wait_commit,chain_marker,gpu_watch}.py`, `extract_ema_r8.py`, `ema_check_r8.py`, `run_battery_r8.py`, `pair_read_r8.py`, `bank_battery_r8.py`, `chain_g0_r8.sh`, `chain_battery_r8.sh`, `thor_pull_r8.sh`, `wrapper_clause_r8fix_gpu.py` | present (same ones chain_40k uses) |
| `clamp_posthoc_r8.py`, `compliant_reread_r8.py` (D-R8-E1-REF built in), `a0k0_headroom{,_lon,_retime}_r8.py` | present; `read/forward --tree <any>` is generic (`compliant_reread_r8.py` `tree_complete`); A0K0 known-value control is "NOT APPLICABLE" off the 30k trees by design. **Not yet executed on a non-30k tree by anyone** (the 40k chain has not reached it) -- UNVERIFIED on a real 40k/48,852 tree until 40k runs. |
| frozen 30k files for the three forwards | present, unchanged |
| `raw/retime_trainfit/FROZEN_GAINS_RETIME_train.json` + `.sha256` | **absent now** (train-fit chain queued behind the GPU queue). Produced by `chain_retime_trainfit_r8.sh`. Chain waits <= 4 h then `ZZCFNTFSKIPZZ` + `ZZCFNCOMBOSKIPZZ` (re-run later). |
| `battery_summary.json`, `windows_s0.json`, `dump_s0` under `refcv8_eval_archive/battery/step48852ema_unc/` | names **VERIFIED** against the banked `step30000ema_unc` tree (the K-draw handoff marked them UNVERIFIED); produced by stage c. `kdraw_roll_r8.py` also needs `battery_summary.json['ckpt_md5'] == md5(ckpt_ema_48852.pt)` -- asserted by the chain before it starts k. |
| `kdraw_queue_r8.sh` side-conditions | its yields (float64 probe, force_s2) read 30k-era logs that are no longer alive -> no yield; `KD_NO_RELEASE=1` stops it from rewriting `/c/lgs/RELEASE_R7`; it is single-instance via `chain/kdraw_queue.pid` (the chain waits for a live one). Peak: `KD_NEED_GB=10` free commit twice 60 s apart. |
| mask sha `4ebc6b86...` for the K-draw/combo read | to be checked in the output (`KDRAW_E1.json`, combo): VOID if different. UNVERIFIED until it runs. |
| `summarize_40k_r8.py` | present; its output file names are `READ_40K.*` (a misnomer); the chain copies them to `READ_FINAL.*` |
| disk | `E:` 107 GB free; `C:` (battery archive) **7.7 GB free** -- each banked tree is ~89 MB (MEASURED), so the 2 final trees fit; C: at 100 % is the dev box's commit-limit neighbour. |

## 4. GPU priority (ESTIMATED from the scripts and logs; measured facts marked)

* **Mechanism: none.** `gpu_lock.py` = O_EXCL file + `nvidia-smi` rule; no queue, no priority file, a foreign holder is never broken (MEASURED, `battery/code/gpu_lock.py:1-14`). `gpu_yield.py` is an evening-release rule, not a priority. The final chain takes the lock exactly like everyone else (60 s polls) and **cannot preempt**.
* Live now (ps, 08:05): `chain_r7_matched_phase.sh`, `chain_r7on_50400.sh`, `queue_k2k4_r8.sh 30000ema`, `chain_retime_trainfit_r8.sh`, `chain_40k_r8.sh` (+ `snap40k_r8.sh`). Lock holder `refcv7-ceilon-step50400` since 06:37 (MEASURED, `k2k4_step30000ema.log`).
* GPU-hours still ahead (ESTIMATED; one EMA unc pass took 3 h 12 m at 30k -- `battery_timings_step30000.txt`; one K-draw seed 33 min -- `kdraw_queue.log`): ceilon-50400 ~2 h left, r7m matched phase 3 x ~3.5 h, K2/K4 ~3 h, train-fit rolls ~3 h, 40k EMA battery 3.2 h + 40k raw ~4 h (the 40k snapshot is due ~20:00 on 10-19). **~28 GPU-h from 08:30 on 10-19 -> drains ~13:00 on 10-20 if nothing is added**, i.e. ~6-8 h before the final (ETA below). Margin is positive but not a guarantee: anything launched meanwhile (REFe/NavSim, MM) or a slip in the r7m estimate puts one 3-4 h job in front.
* Yield: the running refcv7 chains and K2/K4 source `yield40k_r8.sh` (yield to the **40k** EMA battery only) -- none yields to the final, and they are running so they cannot be edited. `code/yield48852_r8.sh` (new, tested) is the sibling: `acquire_y48` is a drop-in for `acquire_y40`, pending from "`ckpt_full_48852.pt` banked" until the EMA battery **and** the K-draw have a terminal marker (cap 6 h). **Wire it into any chain that starts after now** (e.g. the MM's next relaunch of a refcv7/K2K4 job, or a 40k K-draw).
* The final chain never starves past the end of a finite queue (its acquire waits up to 172,800 s), and it does not yield to the REFe evening seam (`gpu_yield.py`) -- a policy call, stated: the brief says the final outranks everything.
* ETA of the final step (ESTIMATED): `ckpt.pt` step 36,000 at 07:43 on 10-19 and 30,000 at 14:46 on 10-18 -> 354 steps/h (includes the 06:18 kill and the 500-step redo); supervisor ticks 36,051 @08:03 and 36,251 @08:35 -> 375 steps/h; (48,852-36,251)/327..375 = 34-39 h -> **10-20 ~18:00 to 23:00 Berlin** (the PI's ~19:00 is inside it). The chain waits for the artifacts (up to 4 days), so the ETA only matters for the GPU queue.

## 5. Findings escalated (details in the hand-back)

1. **`snap40k_r8.sh` (live, pid 1027887) will abort at 40,000.** It compares the remote `ckpt.pt` size for **equality** with `SNAP_EXPECT_SIZE=1560403737` (the size at step 10,500). The live file is **1,560,412,761 B at step 36,000** (`stat` on Thor 08:27). `ckpt.pt` is not constant-size -> `ZZS40BADSIZEZZ` -> `chain_40k` prints `SNAPFAIL` and the 40,000 window is lost when 40,500 overwrites it. Fix: `code/snap40k_fix_r8.sh` (size band, everything else identical); swap = kill 1027887 by pid, start the fix. Tested here.
2. The supervisor done-marker can be written over a **stale** `ckpt.pt` (section 1). Handled by the conjunction above.
3. `retime_trainfit_r8.py forward` hard-codes `res["role"]` to the 30k wording ("BESIDE (not in the verdict)" for every tree except `step30000ema_unc`). For `step48852ema_unc` that string is **wrong** -- the FORWARD_TRAINFIT file for the final tree IS the verdict tree. Not editable (a chain uses it); the chain notes it in the marker and `FINAL_INDEX` does not quote `role`.
4. `kdraw_queue_r8.sh` rewrites `/c/lgs/RELEASE_R7` on exit unless `KD_NO_RELEASE=1` (set by the chain).

## 6. Known limits
Single training seed; T1 open loop on logged eval139 frames; every EMA number PROVISIONAL (no G0 for EMA weights); estimator answers "another draw of episodes" only (seed floor: `H-ESTIM-SEED-1`; the K-draw sets A/B are the inference replicate). The chain was dry-run only against fakes; the heavy tools (extract, battery, compliant, forwards, K-draw, combo) have never been run on a 48,852 tree.
