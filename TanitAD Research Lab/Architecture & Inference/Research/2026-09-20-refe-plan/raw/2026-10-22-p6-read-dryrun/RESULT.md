# P6-READ-DRYRUN: can P6's pre-registered navtest read run on P6's checkpoint format? (dev box, 2026-10-22)

**Scope kept.** No navtest token was loaded, planned or scored by any P6 checkpoint. No PDMS or sub-score was computed on any P6 output. The pod was never touched (HF download only). P6's performance was not read.
Code was read from mirror tip `1664143` (`git archive` of `eval/` and `refe/` to `p6dry/pkx/`, NOT a deliverable). The E: working tree equals that tip for the 11 files compared (sha256, MEASURED): `stage2_sft.py`, `refe_navtest_seam.py`, `eval_checkpoint.py`, `proxy_eval.py`, `planner.py`, `ckpt_io.py`, `model.py`, `nav_command.py`, `navtrain_scenarios.py`, `load_dinov3.py`, `calib_table.py`.

## Headline

**The checkpoint side is proven; the read can run on P6's format. The pre-registered read is NOT yet fully runnable as the instruments stand: three small instrument gaps (G1-G3 below) need a Master Mind decision and landing before the day. One finding outside this brief blocks READY_P6_READ Part B (G4).**

| # | verdict | evidence class |
|---|---|---|
| Q1 loader | **PASS.** The planner's own loader builds P6's architecture from epoch-6 `ckpt_last.pt` (PARTIAL format): 0 unexpected keys, missing keys = exactly the 366 frozen trunk tensors, frozen-trunk fingerprint equals the recorded one, 277 trained tensors / 18,998,850 elements land bit-equal. No adapter is needed. The FULL format that `model_final.pt` will use was rehearsed with the trainer's own writer call and loads strictly. | MEASURED, `raw/load_audit.json` (12/12 gates) |
| Q2 inputs | **No mismatch found** (cameras, resolution, ego vector, command mode). Details below. | MEASURED + code read |
| Q3 determinism | **Bit-identical.** 48 navtrain tokens: in-process repeat, a second process start, and a third process start of the archived final code all give max abs delta exactly 0.0 on the raw outputs. | MEASURED, `raw/identity_*.json` |
| Q4 throughput | **1.42 s/token mean (0.70 tok/s)** steady, RTX 4060 (range 1.26-1.51 over six passes). Full 12,146-token read: **about 5.3 h** (4.8 h seam on this rate, 4.9 h on u400's measured seam rate, plus 0.4 h scoring). | MEASURED (rate), ESTIMATED (total) |
| Q5 runbook | Below, with disk and the three gaps. | derived from tip code; real file UNVERIFIED |

## Q1. Loader (`refe/planner.py:352,361`, `refe/ckpt_io.py:116-160,68-90`)

Calls under test are the planner's own: `config_for_checkpoint` then `REFe(cfg)` then `load_for_inference(..., backbone="vitl16")`. Run on CPU with an independent key audit and negative controls (`code/p6_load_audit.py`, `raw/load_audit.json`).

- `ckpt_last.pt` is **PARTIAL** (`format refe-partial-v1`, 228,324,493 B incl. optimiser; sha256 `eda651447f31...`, saved 2026-10-20T12:52:18Z). It is NOT the multi-GB file the brief expected: the 1.3 GB FULL file is `model_final.pt`.
- Config: `config_for_checkpoint` changes exactly one field from the ViT-L default, `nav_mode goal -> command`, read from the file's `meta` (`ckpt_io.py:132`). `img 512x960`, 4 cameras, 64 proposals, 6 score outputs, `ego_dim 7`, `undistort True`.
- Params 322,078,274 total / 18,998,850 trainable = the run's `config.json` (equal). `source_sha256` in `config.json` for `model.py`, `ckpt_io.py`, `load_dinov3.py`, `train.py` equal the tip blobs: the architecture at inference is the architecture that trained.
- Keys: partial = exactly trainable params plus persistent buffers (277); `strict=False` audit: unexpected `[]`, missing = frozen set (366) exactly. (`backbone.img_mean/img_std` are non-persistent buffers and never appear; my first auditor draft expected them and was wrong, fixed.)
- Frozen-trunk fingerprint `b0fc28856aa5...` equals `meta.frozen_sha256` and the run's identity block; trunk loaded from the dev-box DINOv3 weights (`D:/Projects/TanitAD/data/backbones/dinov3-vitl16`, sha pinned on the pod as `45172f20...`).
- **Negative controls all raise:** a trained key dropped; an extra key; a wrong fingerprint; the command partial into a goal-mode model (size mismatch on `ego_enc.0.weight` 71 vs 43).
- **FULL format rehearsal.** Written with the trainer's exact call (`train.py:1902`: `{"format": FULL, "model": full_state(model), "meta": dict(meta, step, epoch, finished_at)}`), 1,288,516,956 B, loaded back through the planner: `refe-full-v1`, `nav_mode` read from meta, **bit-equal** to the partial-built model. The same FULL file into a goal-mode config fails strictly (`Missing goal_freqs, Unexpected cmd_embed.weight`), so the meta hook is load-bearing and present. This is a format rehearsal on epoch-6 weights, not P6's final weights; the rehearsal file was deleted.
- Setting changes to any registered eval setting: **none.** `--rule navsim_v1` (P6's registered rule, 6 outputs; `navsim_v1_lane` needs 7 and is refused), A7 repair ON (`planner.py` `REPAIR_LAST_HEADING = True`), `sanitize_goal` OFF.

## Q2. Does the harness feed P6 what it trained on?

| input | training | inference (harness) | result |
|---|---|---|---|
| cameras | `meta.cameras = CAM_F0, CAM_L0, CAM_R0, CAM_B0`, n=4 | `planner.CAMERAS` (`planner.py:54`), resolved per camera, worst residual decides | same order. MEASURED 48/48 tokens resolved 4 frames |
| resolution / decode | `cv2.imdecode`, `cv2.resize(img_w, img_h)`, BGR to RGB, /255 (`train.py:832`) | `cv2.resize(self.cfg.img_w, self.cfg.img_h)[:, :, ::-1]` (`planner.py:615`) | same operations, `512x960` |
| per-sample rig | `per_sample_calib True` | read from `meta`, rig read from each log's DB (`planner.py:366,472`) | honoured (planner reported True) |
| ego | 7 scalars at t0, no history input exists (`model.py:185` `ego_dim 7`: vx, vy, ax, ay, yaw rate, tire steer, speed) | `planner._ego_vec` same order (`planner.py:440`; `build_targets.py:223-228`) | same. P6/REFe takes a single-time ego vector: there is no history to mismatch |
| nav | `--nav-mode command`; one-hot of NAVSIM `driving_command` on rank-0 rows (`nav_command.command_of_row`); twins use the goal rule | `set_driving_command` (`refe_navtest_seam.py:78-105`, used `:249`) then `_command_for` (`planner.py:468`) | same encoding. Navtest commands: LEFT 2,501, STRAIGHT 8,070, RIGHT 1,575, **UNKNOWN 0** (export), P6 trained on LEFT 25,904 / STRAIGHT 65,338 / RIGHT 11,795 / UNKNOWN 0: no unseen command at test |
| route fix | none in command mode | `infer` takes the command branch and never calls `_goal_for` (`planner.py:467-470`) | N/A, as PREREG says |

**Command really reaches the model (control, MEASURED, `raw/command_swap_control.json`):** feeding LEFT and RIGHT swapped on the 48 tokens moved the raw output of all 16 LEFT and all 16 RIGHT tokens (median executed-pose change 3.6 m and 4.0 m), and left the 16 STRAIGHT tokens **exactly 0.0 changed**. Read as a wiring statement only.

Cosmetic mismatch to know about: the seam report will record `goal_fix: pdm_route` (the default) although it is unused in command mode. **Do not pass `--goal-fix off`** for P6: `stage3_full.py:395` gates `seam_settings_equal_to_base` on `goal_fix`, and u400's report says `pdm_route`.

## Q3. Determinism (`code/p6_navtrain_plan.py`, `code/p6_compare.py`)

Driver = the seam's per-token loop (`refe_navtest_seam.py:226-262`) on **navtrain** tokens, reusing the seam's own `to_navsim` and `set_driving_command`, the planner class and the scenario-builder call unchanged. Inputs: 48 tokens, 16 per command class, 11 trainval logs (`raw/tokens_sel.json`), frames from `refe_proxy/pixels`, command from P6's own training table (`HF logs/commands.json`, 103,037 scenes). Guard in code: refuses any token in `navtest_all_tokens.json` (overlap measured 0 of 12,146) and any DB under `splits/test`.

| comparison (all GPU, RTX 4060, fp32 raw outputs) | tokens | arrays compared | max abs delta | verdict |
|---|---|---|---|---|
| same process, pass 0 vs pass 1 | 48 | poses, raw traj [64,20,3], raw score [64,6], pick, ego vec, command seen | **0.0** all | IDENTICAL |
| second process start | 48 | same | **0.0** all | IDENTICAL |
| third process start (archived final driver) | 48 | same | **0.0** all | IDENTICAL |
| FULL-format file vs PARTIAL file (same weights) | 48 | same | **0.0** all | IDENTICAL |

Controls in every comparison: zero control (A vs A reads 0.0, asserted), and a **deliberate 1-ulp regression** on one raw value that the comparison must flag (caught in all four). Raw bytes compared with `tobytes()`, not only `allclose`. Scope: the planner leg only. TF32 matmul flag False, cudnn TF32 True, no deterministic-algorithms flag set (the harness sets none), so this is default-settings determinism on this GPU and driver.
**Not covered:** the scoring harness leg. Its determinism given identical poses is covered by `stage2_sft.py`'s own "same-pose tokens must score identically" gate, last exercised on P2-s1 (2026-10-16, INHERITED). PREREG's "second full eval bit-identical in per-token PDMS" costs another ~5 h of GPU on the day; this dry run lowers the risk it fails, it does not replace it.

## Q4. Throughput (MEASURED, `raw/runs/*.report.json`)

Steady per-token loop (image decode 0.09-0.24 s, forward 1.17-1.26 s, post 0.001 s), warm-up excluded: **1.39, 1.26, 1.47, 1.44, 1.48, 1.51 s/token across six passes, mean 1.42** (0.66-0.79 tok/s). Peak CUDA allocation 1,646 MiB (partial) / 2,459 MiB (full-file path). Model load 6-42 s (cold disk).
Cross-checks: u400's full read was **1.441 s/token in the real seam** (17,496.8 s / 12,146, `RESULT_SFT4.md`); the S0 dump measured 1.589 s/token. **Full read ESTIMATE:** seam 4.8 h (this rate, 12,146 x 1.42 s) to 4.9 h (u400 rate), harness scoring 0.37 h (1,335.7 s MEASURED for u400), families about 2 min: **about 5.3 h, matching PREREG's 5.3 h.** Contention on the shared GPU was 2.6x on one day, so the lock matters.

## Q5. Read-day runbook (model_final.pt on HF to the PDMS JSON)

`PKG="E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"`, `PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe`, `PYD=C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe`, `OUT=D:/Projects/TanitAD-artifacts/p6_final`, `DATA=E:/Projects/TanitAD/data/refe_navtest` (D: and E: are the same volume).

**Disk.** Need about **1.5 GB**: `model_final.pt` 1.29 GB (size from the rehearsal, u400's was 1,288,489,838 B), seam npz about 4 MB, score dir about 26 MB, proposal dump (G2) about 95 MB, seam and score logs. Frames, DBs, C1 and C2 tables already on disk. D: free was 9.7 GB, C: 9.6 GB (nothing needed on C:; keep the HF cache off C: by using `local_dir`). Do not create loose scratch trees on D: (1 MiB clusters).

0. **Preflight (1 min).** `ls C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock C:/Users/Admin/qland/work/refcv7/devbox_gpu.chain_active` must both be absent; `nvidia-smi --query-gpu=memory.used --format=csv` at most 3,500 MiB; free RAM above 4 GB (harness guard waits otherwise). Harness dependencies live in a **prior session's temp directory** (G3): `SP=C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/a085b71c-4fb3-409b-ae6b-65f605ed575e/scratchpad`; `ls $SP/score_e.py $SP/w3e/code $SP/ev6e/families6.py` and `md5sum $SP/ev6e/families6.py` must read `e275637c93e1f18ec80a347719680e02` (all present and matching on 2026-10-22).
1. **Fetch (about 2-4 min at 6-12 MB/s; run detached if slower).** `export HF_HUB_DISABLE_XET=1 HF_TOKEN=$(grep -oE 'hf_[A-Za-z0-9]+' E:/Projects/TanitAD/Keys.txt | head -1)`, then `$PY -I -c "from huggingface_hub import hf_hub_download as d; [d('Sayood/tanitad-refe-p6-paper-faithful', f, local_dir='D:/Projects/TanitAD-artifacts/p6_final') for f in ('run/model_final.pt','run/summary.json')]"` (the `run/` prefix is UNVERIFIED for the final file: today's repo holds `run/ckpt_last.pt`, `run/snap_epoch001-006.pt`, `run/config.json`). `summary.json` must say `"done": true`; `sha256sum $OUT/run/model_final.pt` must equal its `model_final_sha256` (`$SHA`). A result exists only when it matches.
2. **No-navtest smoke of the real file (about 3 min GPU, takes the lock itself):** `cd C:/Users/Admin/par18/p6dry`, then `$PYD -I code/p6_navtrain_plan.py plan --sel raw/tokens_sel.json --ckpt $OUT/run/model_final.pt --device cuda --limit 12 --repeat 2 --out $OUT/smoke.npz` then `$PY -I code/p6_compare.py --a $OUT/smoke.pass0.npz --b $OUT/smoke.pass1.npz --mutate`. Expect `ZZPLAN_OK 12`, `format refe-full-v1`, `nav_mode command`, `IDENTICAL true`. (Needs `commands.json` at `D:/Projects/TanitAD-artifacts/p6dry/hf/logs/`, present.) This replaces the P2 runbook's 4-navtest-token smoke, which would be a navtest look.
3. **The read (detached, about 5.3 h).** Same two-call pattern as `P2_READOUT_RUNBOOK.md` section 3 (attempt 1 ends `INCOMPLETE (seam failed)` at the 36-token frame-control gate, expected; attempt 2 reuses the seam and scores without `--tokens`, avoiding `WinError 206`):
   `ARGS="--ckpt $OUT/run/model_final.pt --name p6_full --rule navsim_v1 --tokens $DATA/navtest_full_tokens.json"`; `nohup bash -c "$PY $PKG/eval/stage2_sft.py $ARGS; test -f $DATA/seams/stage2_p6_full.npz && $PY $PKG/eval/stage2_sft.py $ARGS && $PY $PKG/eval/stage3_full.py --name p6_full --ckpt $OUT/run/model_final.pt --expect-ckpt-sha256 $SHA --base u400 --expect-rule navsim_v1" > $OUT/p6_full.out 2>&1 &`. **Do not pass `--exploratory`, `--goal-fix off` or `--sanitize-goal`.** Wait with one background Monitor on `$OUT/p6_full.out` (a second `INCOMPLETE`, a `Traceback`, or `ZZSTAGE3` ends it); do not hold a live agent.
4. **Outputs.** PDMS JSON: `$PKG/raw/2026-10-05-sft-stage2/p6_full/result.json` (P6 vs DEPLOYED, **not** the P6 verdict), `.../score_status.json` (harness PASS 12,146/12,146 and `summary_x100_4dp`), per-token table `$DATA/score/refe_stage2_p6_full/refe_stage2_p6_full.csv`, seam `$DATA/seams/stage2_p6_full.npz` + `.report.json` (must show `nav_mode command`, `driving_command_set 12146`, `rule navsim_v1`, `rows 12146`, `misses 0`). The **P6 vs C1** block: `$PKG/raw/p2_readout/p6_full/r2_result.json` (`stage3_full.py` OUT_ROOT; its GO/NO-GO words are P2's, read the numbers: paired delta, CI, adverse sub-scores, families). Map to PREREG words: ADOPT iff lower CI > 0 and no longitudinal or lateral family component adverse.
5. **After:** record the CSV and seam md5 and the `model_final.pt` sha256 in the registry row; delete `$OUT/smoke*`; keep `model_final.pt` until the registry row lands.

## Gaps and blockers, named (decision owner: Master Mind)

- **G1. No C2 comparator in the instrument.** `stage3_full.py:86` `BASELINES` holds `u400` and `deployed` only, and `--base/--also` take choices from it, so "P6 vs C2 (P2, 88.82), separately" cannot be produced by the unmodified script. Smallest fix: add a `p2` entry. Pins MEASURED today: CSV md5 `bed5687fb75e884d45cf991c2ff391c9` (`score/refe_stage2_p2_full/...csv`), seam md5 `49a1f6ecf28c5ebdaac4c68c2997c5f0` (`seams/stage2_p2_full.npz`), families md5 `a6a381b4cebe3172f8380d9d3a77b85c` (`raw/2026-10-05-sft-stage2/p2_full/p2_full_families.json`), PDMS 88.8184; then run with `--also p2`. **Known-value control passed (MEASURED, `raw/pair_known_value_C2_vs_C1.json`):** the registered estimator on the C2 and C1 tables reads C1 88.2469, C2 88.8184, C2 minus C1 **+0.5715 [+0.0179, +1.1755]**, the registered +0.57 [+0.02, +1.18]; the C1 CSV md5 equals its pin `b17c4fba...`. Fallback without any edit: pair the P6 CSV against both CSVs with `code/p6_pair_known_value.py` (read-only, same `proxy_eval.boot`).
- **G2. PREREG's TACTICAL rows "selection quality random / actual / best-of-64" and "command adherence" are not produced by the registered path.** `families6.py`'s tactical block holds lateral/longitudinal/5-way decision and goal setting only (keys read from `sft4_B_u400_navtest_full_families.json`), and `stage2_sft.py` never passes the seam's `--dump-proposals` (0 matches). I found no "command adherence" instrument by grep in `eval/*.py`. The proposals and raw logits come free from the same forward pass if `--dump-proposals <npz>` (about 95 MB) is added to the seam command in `stage2_sft.py` (a one-line change; the md5 of `stage2_sft.py` is recorded, not gated, by `stage3_full.py:65`); the adherence metric needs an owner. Otherwise a second full pass of about 4.7 h is the only way to get them.
- **G3. Fragile paths.** `stage2_sft.py:37-38,153` takes `score_e.py`, `w3e/` and `ev6e/families6.py` from a **previous session's `AppData/Local/Temp` scratchpad** (the same class of directory whose cleanup already deleted the eval venv once, `eval_checkpoint.py` comment). If it vanishes, `families()` returns None and the verdict silently becomes NOT PROVEN (P2 runbook warning). Copy those three to a permanent path before the day, or at least run the step-0 `ls`.
- **G4. Outside this brief, blocks READY_P6_READ Part B.** `raw/2026-10-18-sel-rank/code/srttc_s0_dump.py:75-78` **refuses** any command-mode checkpoint ("navtrain tokens do not carry NAVSIM's driving_command here"), and `build_candidate_bank.py preflight/smoke` is only exercised on P2. The EP-ADMISSION bank for P6 therefore cannot be built until a command source is wired: P6's own `commands.json` (103,037 scenes, MEASURED) is exactly that source and my driver shows the 10-line hop (`p6_navtrain_plan.py` `one_pass`).
- **UNVERIFIED:** the real `model_final.pt` and its HF path; the seam's main loop (frame control, bookkeeping) and the scoring harness on a P6 file, which cannot be exercised without a navtest look (the hop and the loader are covered by `eval/test_p6_planner_command.py`, `ZZP6PLAN_PASS 172`, on random-init ViT-S, INHERITED, not re-run here); FP32-vs-bf16 numerics of the final weights.

## Deliverable manifest (all under `C:/Users/Admin/par18/p6dry/`; NOT staged, NOT committed, no git tree touched)

| artifact | where | single copy? |
|---|---|---|
| `RESULT.md` (this file) | `p6dry/RESULT.md` | **dev box only** |
| `code/p6dry_common.py`, `p6_load_audit.py`, `p6_navtrain_plan.py`, `p6_compare.py`, `p6_pair_known_value.py` | `p6dry/code/` | **dev box only** |
| `raw/load_audit.json`, `identity_{inprocess,second_process,final_code,full_vs_partial}.json`, `command_swap_control.json`, `pair_known_value_C2_vs_C1.json`, `tokens_sel.json`, `p6_run_config.json`, `p6_token_verify.json`, `ckpt_last_epoch6.sha256` | `p6dry/raw/` | **dev box only** |
| `raw/runs/` (5 runs: npz, `.report.json`, `.log`; about 5 MB) | `p6dry/raw/runs/` | **dev box only** |
| `MANIFEST.sha256` (sha256 of every file above) | `p6dry/MANIFEST.sha256` | dev box only |
| epoch-6 partial checkpoint, run config, `commands.json` (7.5 MB), gate evidence | `D:/Projects/TanitAD-artifacts/p6dry/hf/` (also on HF `Sayood/tanitad-refe-p6-paper-faithful`) | copy on HF |
| `p6dry/pkx/` (tip `1664143` `eval/` + `refe/` export, 8.9 MB) | dev box | scratch, not a deliverable, `git archive` reproduces it |

The Master Mind lands `code/` and `raw/` plus this file; the GPU lock was taken and released by each run (verified absent after the last).
