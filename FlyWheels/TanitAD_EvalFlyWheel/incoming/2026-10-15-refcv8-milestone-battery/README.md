# refcv8 milestone battery kit -- G0 at step 5,000 (EvalFlyWheel, 2026-10-15)

**Status at hand-off: the kit is built and verified; the G0 run is QUEUED behind the shared dev-box GPU, no G0 number exists yet.**
Everything below is tier **T1 self-action OPEN LOOP on logged eval139 frames** (G0 itself is a REPRODUCTION gate on the run's own in-run eval
row, not a driving metric) and an **EARLY READ at step 5,000 of 48,852 -- never a bar verdict**. NOT staged into git, NOT committed, NOT pushed.

## Headline (integration escalation)

1. **`stack/tanitad/eval/refcv7_loader.py::build_eval_dataset` never calls `enable_pose_sync`, but `refc_v3_train.train()` does on the eval dataset
   (`refc_v3_train.py:9765-9766`, "refcv8 X10: the eval GT rides the SAME clock").** MEASURED by reading both at the launch commit `fb47a583763f`
   and in the working tree. The run's own `config.json` stamps `pose_sync.eval = {n_clips 139, n_covered 139, sign 1.0}`, i.e. the in-run eval row was
   scored on the SHIFTED clock. The launch gate cannot see this: its G-EVAL (`launch_gate.py:4989-5030`) hands the trainer's model and the loader's
   model the SAME loader-built batch. Unpatched, every consumer of the loader (G0, `r7_roll`, `perc_pass_r7`) would score refcv8 on a different eval set
   than the run reported. **Interim fix in this package (no repo file touched): `code/r8_repoint.py` wraps `L.build_eval_dataset`.**
   **Durable fix for the Master Mind to land** (3 lines, `refcv7_loader.py` after `e_ds.u8_frames = ...`, mirroring train()):
   `if getattr(args, "pose_sync_sidecar", None): rec["pose_sync"] = e_ds.enable_pose_sync(args.pose_sync_sidecar)`
   (UNVERIFIED: `run_battery_r7.py` cites the loader by blob `a8eb3e7e`, so any pin on that blob must be revisited). Not put under `mods-rw/` because it would change
   a landed, pinned file that the launch gate already passed; it is the PI/MM's call.
2. **The launch gate's remap file `stack/ops/refcv8_eval_remap_a2_thor.json` (4 flags) is NOT enough on the dev box.** Three more Thor paths are read
   while building the model / eval set and exist only on Thor: `--r8-v9-labels-eval`, `--pose-sync-sidecar`, `--join-defect-masks` (found because the CPU smoke
   died on the last one at model build). `code/refcv8_eval_remap_devbox.json` maps all seven.
3. **Eval-side inputs: 294 of 294 compared files are md5-identical to Thor** (`raw/eval_inputs_md5_compare.json`), including the v9 eval release against the
   argv's `--r8-v9-eval-md5`. Checkpoint md5s of the three pulled files equal Thor's. CPU smoke: strict load of `ckpt_5000.pt` (step 5000, 0 missing / 0 unexpected,
   `param_breakdown` equal), 139 episodes / 23,772 windows, pose-sync 139/139 clips, the 128-window in-run subset has the same `perm_sha256`
   (`99adbc7e...6bf6975`) as refcv7's G0.

## What was built (all new files; `mods-rw/` is empty -- no existing repo file changes)

| file (under `code/`) | purpose | evidence |
|---|---|---|
| `r8_repoint.py` | env (KIT, remap overrides, launch tree) + the one behavioural patch (pose-sync on the eval dataset); imported first | `test_r8_repoint.py` 10 passed |
| `refcv8_eval_remap_devbox.json` | dev-box twin of the launch gate's remap + the 3 extra flags | remap resolves all 15 eval/build-side path flags of the real refcv8 argv to existing files (test) |
| `g0_refcv8.py` | thin wrapper: re-point, then `g0_refcv7.main()` UNCHANGED (A7 judge, M1/M2/M4/M5, 24 seeds); writes `<out>.r8_repoint.json` | `--help` + import under CPU; the run itself is QUEUED |
| `run_battery_r8.py` | thin wrapper for the later stages; **REFUSES without a G0 artifact**; rolls/panels on refcv8 **UNVERIFIED** | `--help` + refusal tested only |
| `chain_g0_r8.sh`, `pick_metrics.py`, `probe_check.py` | precondition asserts on content -> wait_commit -> shared-GPU lock -> 1-batch PROBE (asserts strict load, pose-sync attached, tanitad from the launch tree) -> 24-seed G0 -> summarise + sanitised bank | `sh -n` + the precondition stage ran (markers in `E:/refcv8_eval_kit/chain/chain_step5000.log`) |
| `summarize_g0_r8.py`, `bank_g0_r8.py` | slim bankable summary (the 10 MB `g0.json` stays on E:); sanitiser reuses the battery's UUID scan | self-tested on the banked refcv7 step-50,400 `g0.json` |
| `smoke_cpu_r8.py` | CPU-only check of model build + dataset + pose-sync item shift | ran: `E:/refcv8_eval_kit/smoke/smoke_cpu_r8.json` |
| `test_r8_repoint.py` | pytest, each guard with a `test_REGRESSION_*` arm (the loader WITHOUT the patch -> `assert_pose_sync_on` goes RED; a remap WITHOUT the 3 new flags -> 3 unresolved paths reported) | 10 passed |

Existing G0 / A7 suites, unmodified, in the launch tree: `test_g0_a7.py`, `test_g0_a7_reports.py`, `test_g0_main_a7_flow.py` -> **63 passed** (17.7 s).

## Numbers carried (evidence class per number)

| number | value | class |
|---|---|---|
| ckpt md5 (config / ckpt_5000 / ckpt_ema_5000) | 4ce2c772... / a32e3b70... / 598b695b... (= Thor, recomputed locally) | MEASURED (`kit_checks/ckpt_md5_recomputed.txt`) |
| in-run eval row at step 5000 | `eval_loss` 53.86168, `eval_traj` 0.58254, 128 windows in 8 batches, 1,136 `eval_*` terms | MEASURED (Thor `metrics.jsonl` copy `E:/refcv8_eval_kit/thor_reads/metrics_20261015T213148.jsonl`, exactly one row at step 5000) |
| eval-input md5 compare | 294/294 identical, 0 differ | MEASURED (`raw/eval_inputs_md5_compare.json`) |
| pose-sync shift on one window (CPU smoke) | `pose_last` moved 0.0316, `v0` 9.4936 vs 9.4963 m/s, `future_poses` 0.0313 (max abs) | MEASURED (`smoke_cpu_r8.json`, window 20658) -- shows the omission is small per window but real |
| G0 verdict at step 5,000 | **NONE YET** | -- |

## NOT done (read before quoting anything)

* **No G0 result.** The chain is waiting for the shared RTX 4060 (another session's `aux_navtest_forward` holds `devbox_gpu.lock`; the gate also needs
  `used < 4300 MiB`, no foreign python compute, free RAM >= 8 GB). A held lock is never broken.
* No S2 / S6 roll, panel, four-family row, tactical or perception pass was run on refcv8. `run_battery_r8.py` is a re-pointing stub for them and is
  UNVERIFIED past G0 (`r7_roll --arm refcv7`, the refcv8 allocation emitters, `perc_pass_r7` must be smoke-tested first).
* No claim register / registry row is updated here; the Master Mind owns `GOALS_AND_CLAIMS.md`, `MODEL_REGISTRY.md`.
* Seeds 0/1 of the battery and the paired four-family read vs the refcv7-50,400 baseline (`incoming/2026-10-11-r7base-fourfamily/`) are NEXT.
* The A7 / A6 "registered before this G0" test is the file mtime of the launch-tree archive (the commit time of `fb47a58`), not the 2026-10-04
  registration instant; A7 text is the registered one, but "registered before" is not re-derived here.
* The step-5,000 `ckpt_ema_5000.pt` (decay 0.9998) was pulled but is NOT evaluated; G0 compares the live weights to the in-run row, as in refcv7.
* Single training seed; a 5,000-step checkpoint. refcv7's own G0 read FAIL as coded at step 5,000 and at step 50,400 (PASS at 50,400 only under the A7 text ruling; `battery/raw/step*/g0.json`, SPEC_REFCV8_DRAFT section 5), so a coded FAIL here is not by itself proof of a loader defect: read the reasons per term class before concluding anything.

## Deliverable manifest

| artifact | where it lives | note |
|---|---|---|
| this package (README, HANDOFF, code/, raw/) | `C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/` | **single copy, not in git** |
| commit message draft | `C:/Users/Admin/qland/hold_r8batt/msg.txt` | |
| `ckpt_5000.pt`, `ckpt_ema_5000.pt`, `config.json` + md5 files | `E:/refcv8_eval_kit/ckpt/` (403 MB x2) | **single copy on E:** (Thor holds the originals) |
| new eval-side inputs (extrinsics, clip-clock ts33, pose-sync, nav tau, vis1 sidecar 189 MB, v9 eval npz) | `E:/refcv8_eval_kit/data/` | copies of Thor; md5-verified; matched refcv6/7 files are reused from `E:/refcv6_eval_kit/data` (NOT copied) |
| md5 tables (Thor, local, compare) | `E:/refcv8_eval_kit/kit_checks/` (+ `raw/eval_inputs_md5_compare.json` here) | |
| metrics copy | `E:/refcv8_eval_kit/thor_reads/metrics_20261015T213148.jsonl` | |
| CPU smoke | `E:/refcv8_eval_kit/smoke/` | |
| chain logs / G0 work dir (will hold `g0.json` ~10 MB, `g0.log`) | `E:/refcv8_eval_kit/chain/`, `E:/refcv8_eval_kit/battery/step5000/` | work dir may hold raw clip ids -- never land it; `bank_g0_r8.py` writes the sanitised slim copy to `raw/step5000/` here |
| `KIT.log` | `E:/refcv8_eval_kit/KIT.log` | step-by-step progress |

---

# STAGE 2 -- the post-G0 four-family battery, armed unattended (EvalFlyWheel, 2026-10-15 23:31)

**Status: ARMED (`code/chain_battery_r8.sh 5000`, MSYS pid 45849), waiting for the G0 chain, which itself still waits for the shared card (NavSim's lock). No GPU work
has run; every statement below about the rolls is from a 2-clip x 3-window CPU smoke (a code-path check, not a result).** Tier T1 self-action OPEN LOOP on logged eval139
frames; EARLY READ at step 5,000 of 48,852; single training seed; never a bar verdict. Run/inspect instructions: `HANDOFF.md` (top section).

## Escalations for the Master Mind (integration, not buried)
1. **Land the Stage-2 code** (`code/r8_battery_patch.py, roll_seed_r8.py, run_battery_r8.py, pair_read_r8.py, perc_pass_r8.py, make_windows_r8.py, chain_helpers_r8.py,
   bank_battery_r8.py, chain_battery_r8.sh, collect_smoke_r8.py` + tests) -- no repo file was changed; the refcv7 battery runs unmodified from the launch tree.
2. **`r7_panel.BASELINES` (launch tree `battery/code/r7_panel.py:43-47`) names `C:/Users/Admin/refcv5cmp/out/...`, which the 2026-10-15 C: cleanup MOVED to
   `D:/Projects/TanitAD-artifacts/refcv5cmp/out/` (the C: path does not exist).** `b_refcv4b` is also the S2 WINDOW SOURCE, so any refcv7-style battery dies at the window
   list. Re-pointed in-process by `r8_battery_patch.fix_baselines` (by content); a durable fix is a path-resolution env in `r7_panel.py`.
3. **SPEC_REFCV8 section 5's control "model-free arms and GT bit-identical between the refcv8 and refcv7 dumps" holds ONLY on the standalone (uncorrected) pose clock** (MEASURED,
   smoke: `g_max_abs_diff` 0.0 standalone; corrected-vs-standalone GT differs by 0.0458 m, `v0` by 0.0337 m/s, `ha0_ext` by 0.0673 m). The SPEC's 2.3.6 "report on both clocks"
   therefore means: paired refcv7 comparison on the standalone clock (pass 1), corrected-clock absolute rows + E1/E3 (pass 2; `build_s6_dump` raises on a GT mismatch, so
   pass 2 carries no banked baseline). Say so in the SPEC/registry text.
4. **Policy decision taken (reversible): G0 FAIL-as-coded + M1 detected still runs the battery** (the brief's skip rule is only "no g0.json / M1 not detected"), with every number
   stamped PROVISIONAL. `R8_REQUIRE_G0_PASS=1` at launch turns it into a skip.

## Defects found in the refcv7 battery when re-pointed at refcv8 (each fixed in the wrapper, each pinned by a test)
| # | defect | where (launch tree `battery/code/`) | fix | evidence |
|---|---|---|---|---|
| 1 | baseline dumps moved off `C:` (see escalation 2) | `r7_panel.py:43-47` | `fix_baselines` | smoke ran with the re-pointed paths; `test_fix_baselines_*` |
| 2 | `KeyError: 'b_refcv7_50400_s0'` when any extra baseline is registered: `build_s6_dump` does `TIERS[arm]` for every baseline | `r7_panel.py:627` | `add_refcv7_baselines` sets `P.TIERS[lab]` | **found by the CPU smoke** (first attempt died there); `test_add_refcv7_baselines_*` |
| 3 | the roll CHILD (`roll_seed_r7.py`) imports the loader directly: it carries neither the dev-box remap nor the pose-sync policy (monkey-patches are per process) | `run_battery_r7.py:430` | `roll_seed_r8.py` + `SubprocessShim.rewrite_child_cmd` | smoke log: `pose-sync attached ... 139/139` in the corrected child, `OFF` in the standalone one; `test_rewrite_child_cmd`, `test_REGRESSION_unrewritten_child_*` |
| 4 | hard `SystemExit` on a G0 that is not PASS | `run_battery_r7.py:396-398` | exact-anchor source edit (must match exactly once, else refuse), explicit `--g0-nonblocking`, stamped | `test_g0_stop_behaviour_*` drives the REAL `_run` to the next call; `test_g0_stage_reads_a_real_g0_artifact_*` on the banked refcv7-50,400 `g0.json` |
| 5 | pairing refcv8 (corrected clock) against banked baselines would compare two eval sets, and S6 raises after hours | `r7_panel.py:604` | `run_battery_r8.prepare_argv` refuses corrected-clock-with-baselines up front; pass 2 uses `--no-baselines` | `test_prepare_argv_refusals_and_defaults` |

## What the CPU smoke ran (all `E:/refcv8_eval_kit/smoke_battery/`, summary `raw/smoke_battery_cpu_r8.json`, 2 clips x 3 windows, `--smoke-cpu`, no lock)
`run_battery_r8.py` end to end on `ckpt_5000.pt`: strict model build (0 missing/unexpected), refcv8 dataset with the label speed input, T1 rolls for inference seeds 0 (all arms,
99.7 s) and 1 (os-only, 31.1 s), S2 panels with banked refcv4b / refcv5v2 / **refcv7-50,400** baselines, S6 build, `analyze` (four families incl. distance keeping), void gates 3-4
(pass), cross-paired, tactical, acceptance, criteria_check, seed replicate; `pair_read_r8.py` on both passes; the corrected-clock variant (`--no-baselines --os-only-all`);
`perc_pass_r8.py` on CPU, 4 windows (LAW-frame control PASS; perc.json written; see `smoke_battery_cpu_r8.json`, key `perception_pass`). The refcv8 `os` there is far worse than refcv7 and the echo -- 6 windows, a CPU fp32 trunk, step 5,000: **no number is
a result**; the smoke shows the plumbing and the clock/GT facts only.

## New files (all under `code/`)
| file | purpose |
|---|---|
| `r8_battery_patch.py` | baselines re-point, refcv7-50,400 as a paired baseline, child rewrite, the exact-anchor G0-stop edit (does NOT touch `r8_repoint.py`, which the queued G0 chain imports) |
| `roll_seed_r8.py`, `perc_pass_r8.py` | child-process entry points that apply the re-pointing first |
| `run_battery_r8.py` | the battery wrapper (refusals, `--smoke-cpu`, `--g0-nonblocking`, `--no-baselines`, `--os-only-all`, r8 stamp in `battery_summary.json["r8"]`) |
| `make_windows_r8.py` | S2/S6 window files + the set-equality proof vs refcv7-50,400 |
| `pair_read_r8.py` | paired four-family read, BAR-R8-E1/E3/NR per SPEC section 5, controls, STRATEGIC NOT APPLICABLE stamp |
| `chain_battery_r8.sh`, `chain_helpers_r8.py`, `bank_battery_r8.py` | the unattended chain, its gate/waiter helpers, the sanitised bank + `BATTERY_SUMMARY` |
| `collect_smoke_r8.py` | distils the smoke runs |
| `test_r8_battery.py` (24), `test_chain_battery_r8.py` (9) | known-value tests (closed-form circle, identical inputs => delta 0.0 CI [0,0]), each guard with a `test_REGRESSION_*` arm; the chain control flow in a sandbox with stub scripts |

## Not covered by the paired read (own instruments): distance keeping / LEAD-VIOL (X7), ceiling compliance (X3), the dense-v9 tactical block (R8-4 i-iii), S-ROUTE, NavSim, perception scoring.

---

# STAGE 3 -- the unattended QUEUE behind the step-5,000 chain (EvalFlyWheel, 2026-10-16 ~11:00)

Tier T1 self-action OPEN LOOP on logged eval139 frames; EARLY READS; single training seed; nothing here is a bar verdict. Run/inspect instructions: `HANDOFF.md` (top section).

## Escalations for the Master Mind
1. **There is NO step-10,000 checkpoint pair, and none will ever exist.** MEASURED 2026-10-16 10:17 on Thor (`ssh -n`, read-only): the run dir holds `ckpt_5000.pt`, `ckpt_ema_5000.pt` and the rolling `ckpt.pt` only;
   the trainer writes `ckpt_<N>.pt` / `ckpt_ema_<N>.pt` ONLY at `MILESTONES = (5000, 15000, 20000, 30000)` (`stack/scripts/refc_v3_train.py:232`, `:10834-10839`; the run's argv has `--save-every 500`, `--steps 48852`).
   The next real pair is **step 15,000** (training was at ~10,750 at 10:14, ~6 steps/min => roughly tonight). `ckpt.pt` is overwritten every 500 steps (it was step 10,500, mtime 09:30; the step-11,000 save overwrites it).
2. **What the queue does instead (a deviation from the brief, reversible):** the milestone is **step 10,500**, DERIVED on the dev box from a md5-verified pull of the rolling `ckpt.pt` (`thor_pull_r8.sh` -> `slim_rolling_ckpt_r8.py`):
   `ckpt_10500.pt` = `{model, step}` and `ckpt_ema_10500.pt` = the `ema_model_only` layout, byte-for-tensor what the trainer would have written at a milestone (derivation pinned against the REAL `WeightEMA.model_state_dict`
   in `test_queue_after_5000.py`); sidecar `E:/refcv8_eval_kit/ckpt/ckpt_10500.provenance.json` says DERIVED, not trainer-written. The queue is generic: `Q_MILESTONE=15000 sh code/queue_after_5000.sh` uses the trainer's own pair.
3. **Land** (no repo file was changed): `code/queue_after_5000.sh, queue_helpers_r8.py, thor_pull_r8.sh, slim_rolling_ckpt_r8.py, ema_check_r8.py, wrapper_clause_r8fix_gpu.py, wrapper_probe_r8fix.py` (copy of the W-R8-1 probe, md5-equal) + `test_queue_after_5000.py`.

## New files (all under `code/`)
| file | purpose |
|---|---|
| `queue_after_5000.sh` | the orchestrator: wait for the newest segment of `battery_step5000.log` to end -> wrapper-fix GPU re-run -> EMA battery pass 1 -> milestone (pull -> G0 -> wrapper-fix probe -> battery pass 1); each stage logged, a failure never stops the next one |
| `queue_helpers_r8.py` | `battstate` (newest-segment state of the battery chain log), `metrics_clean` (drops the live file's torn tail, asserts exactly one eval row at the step), `freegb`, `md5` |
| `thor_pull_r8.sh` | READ-ONLY Thor pull with md5 on both sides, rolling-ckpt fallback, metrics copy |
| `slim_rolling_ckpt_r8.py` | rolling `ckpt.pt` -> `ckpt_<S>.pt` + `ckpt_ema_<S>.pt` (+ provenance), refuses a wrong step / an overwrite |
| `ema_check_r8.py` | CPU gate: EMA file loads STRICT through the battery's builder and really differs from raw |
| `wrapper_clause_r8fix_gpu.py` | the W-R8-1 clause on the GPU: unfixed-probe control (must reproduce G0's terms), fixed probe, amended reading (`r8_spd_crc` dropped) |
| `test_queue_after_5000.py` | 9 tests, each guard with a regression arm; the queue control flow runs in a sandbox with stub scripts; 4 mutations of the orchestrator went RED |
