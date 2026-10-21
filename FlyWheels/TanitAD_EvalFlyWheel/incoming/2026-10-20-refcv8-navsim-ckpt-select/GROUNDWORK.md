# GROUNDWORK: refcv8 checkpoint selection on NavSim navtrain (D-R8-CKPT-RULE)

EvalFlyWheel, 2026-10-20, 22:19 to about 23:20 Europe/Berlin. Rule read from git `FETCH_HEAD beba0241` (`C:/Users/Admin/tanitad-push`), not from the G: tree.

**Status in one line:** the Thor runtime works on navtest (VALIDATION ONLY). The navtrain sample **cannot be fixed yet**, because no reachable machine holds a single navtrain token the bridge can read. Every acquisition path is a download that needs PI approval (section 2). **No PDMS, sub-score or other score was produced on any token in this task, navtrain or navtest.**

Evidence classes used: MEASURED (this task, artifact path given) · INHERITED (another record, not re-measured) · ESTIMATED. Every Thor number is tagged **VALIDATION ONLY**. They are bridge plans and timings on 5 navtest tokens; refcv8 has no NavSim score here. Tier: NavSim open-loop (T1-family), with no scoring done.

---

## 1. Inventory

**What the bridge reads** (MEASURED from source):
* the per-token NavSim `AgentInput` export (`--inputs`: ego statuses, driving command, timestamps);
* a 416×1024 cylindrical frame bank stitched from **CAM_L0 + CAM_F0 + CAM_R0**. The bank stores the **4 history frames** per token (`BankNavtest416`, keep 4). The W3 builder **refuses a token unless all 4 × 3 jpgs exist** (KB3, `build_navtest_frames.py`). The ST arms (R8_LEGAL / _s1, and R8_RT derived from them) then read only the t0 frame (`slot_sources6` with ST: every slot comes from t0);
* the OpenScene log pickle of each token, for the camera rig (`stage1_rigs`);
* `road_plane_navhard_warmup_logs.json`;
* the checkpoint plus `config.json`;
* for R8_RT, the stage-loadable deploy table.

| asset | dev box | Thor | A40 pod | size | readable from Thor |
|---|---|---|---|---|---|
| navtrain scene filter (v1.1 `3e8291b`) | `D:/Archive/devbox-C/navsim/navsim-3e8291b…/…/scene_filter/navtrain.yaml` (sha256 `c37fea56…`; 1,192 logs, 103,288 tokens; sorted-token sha256 `68f30132…`) | `/home/nvidia/navsim-thor/src/navsim-v11/…/navtrain.yaml`, **identical sha256** | n/a | 4 MB | **yes** |
| OpenScene **trainval** metadata (`navsim_logs/trainval`) | **absent** (only `navsim_logs/test`, 147 logs, 1.1 GB) | 19 of 1,192 navtrain logs: `/home/nvidia/tanit-astra-research/data/navsim_pilot_20261003/metadata{,_probe}/`, sorted ranks 0–3, 5–8, 10–13, 16, 18–20, 258, 437, 1124 | was under `/workspace/data`; **unreachable** | full: 7,052,210,642 B (HF) | 19 logs only |
| navtrain **current** frames (all 8 cams) | `E:/Projects/TanitAD/data/openscene_probe/navtrain_current_25.tgz`, **complete**: 4,496,159,137 B, sha256 = HF LFS oid `41d6f5fb…` (MEASURED). 38 navtrain logs, sorted ranks 387–514, 1,517 frames per camera | CAM_F0 **only**, t0 of 1,309 tokens in the 16 pilot logs (277 MB). L0/R0: **0 of 1,309** (MEASURED) | `navtrain_pixels`, 413,152 frames × 4 cams; **unreachable** | 32 shards, 304,548,267,046 B (HF) | F0-only, so **no** |
| navtrain **history** frames | absent | absent | unreachable | 32 shards, 144,459,143,360 B | **no** |
| REFe pixel sample | `E:/Projects/TanitAD/data/refe_m5/pixels`: 1,900 **random** tokens over 206 logs, B0/F0/L0/R0, current frame only, no metadata | — | — | 1.57 GB | never a whole log, so **no** |
| navtest export (`navtest_inputs.json.gz`) | `D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/` (8,188,529 B) | `/home/nvidia/r8ns/navtest/inputs/` (md5-verified) | — | 8 MB | **yes** |
| navtest 416 frame bank | `D:/Archive/devbox-C/navsim/exp/refcv6_navtest416/frame_bank` (66 files, 22,842,641,472 B) | 4 shards + index (`s01, s02, s28, s31`; 2.88 GB) in `/home/nvidia/r8ns/navtest/frame_bank/`, enough for smoke5 | — | 22.8 GB | partial (smoke only) |
| navtest log pickles | `D:/…/navsim_logs/test` (1.1 GB) | the 5 smoke logs (31 MB) | — | | partial |
| navtest metric cache (v1.1) | `D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest` (12 GB) | the Thor CPU backend's `/dev/shm/navsim` copy is **gone** | — | 12 GB | **no** (re-ship) |
| navtrain metric cache | none anywhere | none | REFe-internal only | — | **no** (build) |
| nuPlan maps | `D:/Archive/devbox-C/navsim/data/maps` + `nuplan-maps-v1.1.zip` | `/home/nvidia/tanit-astra-research/data/navsim_maps/nuplan-maps-v1.0`: **Las Vegas only** | — | ~1 GB zip | partial (re-ship for scoring) |
| refcv8 EMA-40k / EMA-48,852 + config | `E:/refcv8_eval_kit/ckpt/ckpt_ema_{40000,48852}.pt` (md5 `57807456…` / `b4364029…`), `config.json` (`4ce2c772…`) | `/home/nvidia/r8ns/ckpt/`, **md5-verified** | — | 2 × 403 MB | **yes** |
| RETIME per-frame table (registered) | `C:/Users/Admin/qland/hold_r8pf/…/raw/retime_trainfit_pf/FROZEN_GAINS_RETIME_train_pf.json`, sha256 **`2aacc9ff…`** (MEASURED) | `/home/nvidia/r8ns/code/retime/` | — | 9 KB | **yes** |
| RETIME deploy table (what the stage loads) | `…/pfpkg/raw/retime_deploy/refcv8_retime_deploy_gains_pf.json`, sha256 `3319b003…` = its sidecar; `source_sha256` = `2aacc9ff…` | `/home/nvidia/r8ns/code/retime/` (+ sidecar) | — | 1 KB | **yes** |

**A40 pod** `growing_orange_anaconda` (69.30.85.13:22175): ssh returned `Connection refused` at 22:22. Whether the pod is Stopped or Terminated is unknown (`…/2026-09-20-refe-plan/STOP_CONSEQUENCES_2026-10-11.md`). No RunPod API key exists on the dev box.

**Thor** (`thor6`, up 66 days) was idle at every check: GPU 0 %, no compute apps, no python of user `nvidia`, MemAvailable 79.5 GB, 251 GB free disk. The flock `/home/nvidia/refcv7_post/thor_gpu.lock` had no holder. The only cron (`refav1_drift_check_v3.py`) is a CPU log reader. Nothing was killed and nothing was paused.

## 2. Sample: NOT FIXED; acquisition needs PI approval

**The intersection is empty** (MEASURED, `raw_thor_pilot_meta_count.json`, `NAVTRAIN_SAMPLE_TOKENS.json`). No navtrain log anywhere reachable has both its trainval metadata **and** CAM_L0/F0/R0 frames:
* metadata ∩ shard-25 logs = 0;
* metadata ∩ REFe-m5 logs = 0;
* the 16 metadata logs that do have frames have CAM_F0 only.

So `NAVTRAIN_SAMPLE_TOKENS.json` reads `NOT_FIXED…`, with count 0 and sha256 of the empty list.

⚠️ **This is not yet a fixed data-availability statement.** No navtrain score exists, so availability is still open. If the PI declines every option and the pod is gone, the rule's own clause applies: fewer than 1,000 tokens means **UNDERPOWERED**, and the EMA-48,852 default holds.

**Cheapest admissible sources.** All are public on HF `OpenDriveLab/OpenScene`, revision `a76f840b…` (= `main`, unchanged since 2025-04-28), cc-by-nc-sa-4.0, not gated. Exact URLs and sha256 values are in the JSON.

| option | files | bytes | ETA to Thor at 31.2 MB/s (INHERITED) | ETA to dev box at 1.6–3.8 MB/s (INHERITED) | sample yield |
|---|---|---|---|---|---|
| **S1 (recommended)** | `openscene-v1.1/openscene_metadata_trainval.tgz` (7,052,210,642) + `navsim/navtrain_current_1.tgz` (9,469,285,958) + `navsim/navtrain_history_1.tgz` (3,831,962,664) | **20,353,459,264** | **10.9 min** | 1.5–3.5 h | ESTIMATED ≥ 2,000. Shard 1 opens with sorted ranks 0..20 (16 logs, 1,309 tokens, MEASURED); the average shard holds 3,228 tokens |
| S25 (cheaper, smaller sample) | metadata tgz + `navsim/navtrain_history_25.tgz` (2,938,296,090). `current_25` is already on the dev box: 4.5 GB over the LAN, ~2.2 min | 9,990,506,732 | 5.3 min | 0.7–1.7 h | ≤ 1,517 tokens / 38 logs: ≥ 1,000, so admissible, but below the 2,000 target |
| A40 restart | PI checks the RunPod console. Only works if the pod is *Stopped*; cost not priced | — | — | — | the full navtrain set, if the disk survived |

Rates and sources:
* Thor HF rate: 6,730,137,600 B in 216.0 s, from the TanitAstra pilot record (`/home/nvidia/tanit-astra-research/data/navsim_pilot_20261003/acquisition.json`, 2026-10-03). INHERITED, not re-measured.
* Dev-box rates: from `…/2026-09-20-refe-plan/raw/navtrain_size_measured.txt`.
* LAN dev box → Thor: **34 MB/s MEASURED today** (2.93 GB in 86 s, md5-verified). This is lower than the memory note's 63 MB/s, likely because the source was the exFAT D:.

## 3. Thor runtime: navtest smoke, 5 tokens per checkpoint, VALIDATION ONLY

* **Setup** (`/home/nvidia/r8ns/`, all md5- or sha256-verified):
  * code bundle: the launch tree `fb47a583763f` (stack, taniteval minus `results/`, and the 4 NavSim code dirs), the refcv8 package from FETCH_HEAD, and the FETCH_HEAD battery code. That is 1,907 files, 0 sha256 failures.
  * Thor's own `/home/nvidia/refcv8_run/launch_tree` is the same commit and was **not touched**.
  * Remap: `stack/ops/refcv8_eval_remap_a2_thor.json`, with `R8_EVAL_KIT=/home/nvidia`. Every other argv path is native on Thor (verified present).
* **Venv departure** (blocker B3):
  * `tanitad-edge`, the inference venv under the two-venv rule, **has no `timm`**. Attempt 1 refused at model build: `ImportError: refcv6 §2 needs timm` (`raw_thor_smoke/attempt1_edge_venv_no_timm/`).
  * The smoke ran in **`tanitad-train`** instead. This follows the precedent of refcv7 WP-RL on Thor (`wprl_env.sh`). That venv has timm 1.0.29 and torch 2.13.0+cu130.
  * It also used a **read-only overlay** that exposes the edge venv's scipy 1.18.0 (same Python 3.12.3, same numpy 2.5.1), appended **last** on PYTHONPATH. scipy is needed only for `knots_to_navsim`'s CubicSpline. Analytic check: a straight line maps to exact poses, max error 0.0.
  * **Nothing was installed into either venv.**
* **The package's own controls on CUDA** (the test module run unchanged under a decorator stub, since there is no pytest in either venv): `premises`, **K0** (same seed gives a bit-identical plan) and **KD** (exact dedup within 1 mm) **PASS for both checkpoints**. Precision `as_trained` (bf16).
* **Bridge** (`run_bridge8.py` unchanged, smoke5 tokens: TURN_L, TURN_R ×2, FOLLOW ×2):
  * R8_LEGAL and R8_LEGAL_s1: 5/5 rows, 0 stand-ins, `partial false`, KPR PASS (max 1.9e-06 m);
  * `r8_nav` all-zero, `v_max_valid` 0;
  * PRIOR_ha0p and R8_RT derived; R8_RT_s1 via `derive_rt_seed.py` (5/5).
* **R8_RT uses the registered table: PASS on all 5 content checks** (`raw_thor_smoke/RT_ASSERT.json`):
  1. the frozen table hashes to `2aacc9ff…` (literal);
  2. the deploy table hashes to its sidecar = `3319b003…` (literal), and its `source_sha256` = `2aacc9ff…`;
  3. **re-derived independently**: the deploy table's 8 knots and gains equal the frozen table's S2 (0.5–2 s) and S6 (3–6 s) RETIME values exactly (max |Δ| 0.0);
  4. all 4 R8_RT seam manifests record that table;
  5. the stage acted: knots moved 2.17–2.28 m against the source rows.
* **CUDA rate** (MEASURED, `raw_thor_smoke/SMOKE_SUMMARY_VALIDATION_ONLY.json`):

  | quantity | value |
  |---|---|
  | warm forward per scene | **0.177–0.180 s** (n = 9 per checkpoint), about 5.6 forwards/s |
  | per token including bank I/O | 0.18–0.20 s (an arm of 5 tokens took 0.9–1.0 s) |
  | first forward (warm-up) | 1.0 s |
  | model load | 4.2–4.3 s |
  | `torch.cuda.max_memory_allocated` | **1.16–1.18 GB** |
  | bridge wall per checkpoint (10 forwards) | 9.8 s |

* ⚠️ **Finding: CPU and CUDA reads are different inference draws.**
  * Setup: the same EMA-40k, the same 5 tokens, identical frames (sha16) and identical scene seeds. The dev-box CPU fp32 rows were compared with the Thor CUDA bf16 rows.
  * Result: the anchor pick differs on 4 of 10 rows, and knots differ by up to 19.3 m.
  * That is the **same size as Thor's own seed-0 vs seed-1 difference** (up to 19.6 m on the same tokens). n = 5, so this is descriptive only (`raw_thor_smoke/CPU_vs_THOR_CUDA_plans_ema40000_VALIDATION_ONLY.json`).
  * ⇒ A backend switch acts like a fresh inference draw. The selection, the later navtest read, and every floor compared with them must run on **one backend**, Thor CUDA (as RULING_BACKEND_POLICY §2 requires). The navtest floors banked on the dev box (STOP, CV, HUMAN, refcv7-50,400) would be MIXED-backend for N4.

## 4. Wall-time estimates

All use 0.20 s/forward (the measured upper bound, including I/O). ESTIMATED unless marked.

| job | forwards | GPU | CPU / other |
|---|---|---|---|
| **navtrain selection**, N ≈ 2,000–2,300 tokens × 2 checkpoints × 2 seeds | 8,000–9,200 | **≈ 27–31 min** (+ ~1 min of loads and controls) | download S1 ~11 min. Extract, inputs export and bank stitch: unmeasured, ESTIMATED 30–60 min. Navtrain metric cache: unmeasured, ESTIMATED 15–40 min. Scoring 7 seams × N at the Thor backend rate of 44 scenes/s (INHERITED): ~6 min. **Total ≈ 1.5–2.5 h wall.** |
| **single navtest read** of the chosen EMA (A-R8-NS1): navtest R8_LEGAL + _s1 + R8_PRIV; navhard R8_LEGAL + _s1; warmup R8_LEGAL + _s1 + R8_BLIND | 36,438 + 11,824 + 660 = 48,922 | **≈ 2.7 h** (RAW weights alongside: +18,278 forwards ≈ 1.0 h) | ship the navtest bank (22.8 GB ≈ 11 min at 34 MB/s MEASURED) and the navhard/warmup banks (size not measured). Scoring on the Thor backend (INHERITED): navtest 8.4 min/arm × ~6, navhard 21.3 min/arm × ~4, about 2.2 h, overlappable with the GPU |

The handoff's 4060 estimate (0.35–0.67 s/scene, 6.5–12.5 GPU-h) is superseded on Thor by the measured 0.18 s warm.

## 5. Launch command for the scoring step (NOT executed)

`code/r8ns_navtrain_select_LAUNCH.sh` (Thor: `/home/nvidia/r8ns/code/thor/`; it must be copied there first):
```
ssh tanitad-thor 'cd /home/nvidia/r8ns && nohup flock -n /home/nvidia/refcv7_post/thor_gpu.lock \
   bash /home/nvidia/r8ns/code/thor/r8ns_navtrain_select_LAUNCH.sh > /home/nvidia/r8ns/navtrain_sel/select.out 2>&1 < /dev/null &'
```
Read completion from the `ZZROLLSDONE` / `ZZREFUSE_<why>` markers in `select.out`, never from an exit code. The script refuses at these gates:
* G0: no `AMENDMENT_ACK.txt`;
* G1: the sample is not FIXED, has fewer than 1,000 tokens, or its sha256 does not re-derive;
* G2: an input or bank is missing, or a checkpoint md5 is wrong;
* G3: the GPU or the box is busy;
* K0 fails on CUDA;
* G4 (after the run): any seam partial, any stand-in, KPR not PASS, or anything other than 4 R8_RT seams;
* the RT table assertion fails.

The scorer and decision steps after it are **not built** (B5).

## 6. Blockers, and what unblocks each

| # | blocker | unblocks it |
|---|---|---|
| **B1** | No readable navtrain token anywhere: the A40 pod refuses ssh, and nothing on the dev box or Thor has metadata together with L0/F0/R0 frames | **PI:** approve download **S1** (20.35 GB to Thor, ~11 min) or S25 (9.99 GB, ≤ 1,517 tokens). Or check the RunPod console for the A40 pod (Stopped vs Terminated) |
| **B2** | The rule's camera criterion says "**CAM_F0 history**". The bridge needs **CAM_L0 + CAM_F0 + CAM_R0**, and its bank builder needs 4 history frames each (KB3). Read literally, the 16 Thor pilot logs (1,309 tokens, F0 only) would count as "readable" although the bridge cannot run them | **MM:** file an amendment **before any navtrain score** (draft text below) |
| **B3** | `tanitad-edge` has no `timm`, so refcv8 cannot load in the two-venv rule's inference venv. The smoke ran in `tanitad-train` with a read-only scipy overlay | **PI/MM:** accept `tanitad-train` + overlay for this read (the refcv7 Thor precedent). Or approve installing `timm==1.0.29` into `tanitad-edge` with `--no-deps`, after a dependency check: a package download, and torch must not move (CLAUDE.md uv trap) |
| **B4** | `run_bridge8 --retime` builds R8_RT from **seed-0 rows only** (`run_bridge8.py:281-297`), but the rule needs R8_RT at seeds 0 and 1 | **MM:** accept `code/derive_rt_seed.py`: new code, using the bridge's own functions, smoke-validated (5/5, table PASS) |
| **B5** | Not built (0 GPU, EvalFlyWheel, ~1 agent-day): (a) a navtrain port of W3 `export_navtest_inputs.py` (trainval logs, sample tokens, `stage` = 1); (b) a navtrain port of `build_navtest416.py` / W3 `build_navtest_frames.py` (member prefix `navtrain_{current,history}_N/<log>/<CAM>/`); (c) the navtrain metric cache (W3 `run_v1.py cache`, `train_test_split=navtrain` + token filter, maps re-shipped to Thor); (d) a navtrain scorer port (`score_navtest_thor.py` / `score_navtest8.py`, `cache_name`); (e) the decision script (paired 40k − 48,852 PDMS per seed, log-cluster bootstrap B 2000, floor \|Δ_s0 − Δ_s1\|, controls: token-shuffled pairing widens the interval, 48,852 − 48,852 = 0 exactly, STOP and CV rows beside) | EvalFlyWheel, once B1 is decided |
| **B6** | Backend: CPU and CUDA plans differ like separate inference draws (section 3) | Run the selection and the navtest read on Thor CUDA, and re-score their floors on Thor (state MIXED-backend wherever the dev-box floors are quoted). Re-ship the navtest metric cache, maps and banks to Thor |
| **B7** | `run_bridge8` hard-refuses missing frames only when split is `navtest`. On `navtrain` a missing frame would silently become a declared CV stand-in | Built into the launch script's G4 gate: zero stand-ins required |

**Draft amendment text for B2 + B4.** The MM files this in GOALS_AND_CLAIMS before the first navtrain score:

> *D-R8-CKPT-RULE Amendment 1 (before any navtrain score): "readable" means a token whose OpenScene trainval metadata row exists AND whose 4 history frames × {CAM_L0, CAM_F0, CAM_R0} are present (the navtest bank builder's KB3 criterion; the ST arms read the t0 frame). Seed-1 R8_RT is R8_LEGAL_s1's rows re-timed by the same stage and table (`derive_rt_seed.py`). The rolls run on Thor CUDA in `tanitad-train` with the edge scipy overlay; every compared row is scored on the Thor backend.*

## 7. Deliverable manifest

| artifact | location |
|---|---|
| this report, `GROUNDWORK.json` | dev box `C:/Users/Admin/refcv8_pf/navsim_ckpt/` |
| `NAVTRAIN_SAMPLE_TOKENS.json` (NOT FIXED, with acquisition options) | dev box, same dir |
| `navtrain_filter_v11.json` (sorted logs and tokens of the v1.1 filter) | dev box, same dir |
| `raw_thor_pilot_meta_count.json` (per-log navtrain tokens and camera presence, 19 logs) | dev box, same dir |
| `raw_hf_tree_navsim.json`, `raw_hf_tree_openscene_v11.json`, `raw_hf_dataset_info.json` (HF metadata API, nothing downloaded), `raw_shard25_sha256.txt` | dev box, same dir |
| `raw_thor_smoke/`: smoke summary, RT assertion, CPU-vs-CUDA plans, per-checkpoint bridge rows, seams, manifests, logs, the attempt-1 refusal | dev box, same dir (copy of Thor `/home/nvidia/r8ns/out/`, md5-verified tar) |
| `code/`: `r8ns_thor_smoke.sh`, `r8ns_cuda_controls.py`, `r8ns_bridge_wrap.py`, `derive_rt_seed.py` (NEW), `rt_assert.py`, `smoke_summary.py`, `overlay_check.py`, `parse_filter.py`, `navtrain_logs.py`, `thor_meta_count.py`, `r8ns_navtrain_select_LAUNCH.sh` (NOT executed) | dev box `…/navsim_ckpt/code/`; Thor `/home/nvidia/r8ns/code/thor/` (the smoke tools + `derive_rt_seed.py`, `rt_assert.py`, `smoke_summary.py`, `overlay_check.py`; the LAUNCH script is NOT on Thor) |
| `thor_bundle/` (`BUNDLE_SHA256.txt` + the staged tree; the shipped tar md5 `41774a75…` is on Thor as `/home/nvidia/r8ns/r8ns_code.tar`) | dev box `…/navsim_ckpt/thor_bundle/`; Thor `/home/nvidia/r8ns/code/` |
| Thor runtime: checkpoints, navtest smoke data, scipy overlay, smoke outputs | Thor `/home/nvidia/r8ns/{ckpt,navtest,overlay_scipy,out}` |
| `pkg/`, `batt_fh/` (FETCH_HEAD extracts, read-only inputs) | dev box `…/navsim_ckpt/` |

Nothing was committed or staged, nothing was written to G:, and nothing was downloaded beyond HF metadata JSON.
