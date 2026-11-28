# SCOPE: the "more label logs" lever for the S1c drivable-area head (2026-10-16, dev box, CPU only)

Author: TanitAD S1d-scope agent (Sonnet). Nothing committed, pushed, shipped or launched; no pod touched. Tier: T0-like (head perception on teacher-forced frames), never driving performance.
Evidence classes: **MEASURED** (ours, artifact named) / **PUBLISHED** / **INHERITED** (another doc, not re-verified) / **ESTIMATED** / **HYPOTHESIS**.
`PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan` (tree at commit c935a6d). `W` = `C:/Users/Admin/s1d_scope/raw/2026-10-16-s1d-scope/` (this package).

## 0. HEADLINE (read this first)

**The premise of lever (a) is wrong for the S1/S1c head, so there is no navtrain label build to scope.** "Believed to generalise from only ~100 label logs" is the regime of the **A2 dev-box probe**
(123 navtest logs in 5 folds, each fold unit trains on ~98 logs / ~846 tokens: `PK/raw/2026-10-16-scene-tap-a2/folds.json`, MEASURED), not of S1c.

| who trained | distinct frames | distinct logs | where it comes from |
|---|---|---|---|
| A2 probe unit (the "~100 logs") | ~846 | ~98 | `folds.json` (123 logs, 1,058 tokens, 5 folds), MEASURED |
| S1 head, 25,600 draws, seed 0 | 23,936 | 1,133 | `PK/raw/2026-10-13-aux-s1-data-audit/evidence/sim_draws.json`, reproduced here |
| **S1c head, 80,000 draws, seed 0** | **64,146** | **1,161 of the 1,167 train logs** | `W/sim_s1c_draws.json` (MEASURED, same torch 2.11.0+cu128 draw) |
| the whole train pool of S1/S1c | 99,495 | 1,167 | `W/sim_s1c_draws.json` `train_pool` (174,912 rows; 24 further logs are held out) |

* S1's draw is a literal **prefix** of S1c's (`S1_draw_is_prefix_of_S1c: true`); S1c adds 40,210 frames never seen at S1. **100 % of drawn frames carry a label** (`frames_without_label: 0`).
* The label store already covers **all 1,192 navtrain logs, 103,139 keys** (`D:/Projects/TanitAD-artifacts/aux_drivable/label_store/manifest.json`; sha256 `4a17db67...`). **Navtrain label logs not yet used by S1c: 6 logs.** Frames in the pool not yet drawn: 35,349 (35.5 %), which is the *budget* lever, not a label lever.
* The probe's own prereg already said so, before S1: *"The data lever is more head updates / frames seen, not more logs"* (`PK/raw/2026-10-15-scene-tap-probe/PREREG_SCENE_TAP_PROBE.md:53`). The A2 row U2 then wrote "S1c with more updates AND more label logs" relative to the PROBE's 98 logs; S1c took its logs from navtrain, i.e. it already is the "many logs" arm.
* **The head already generalises to unseen LOGS at the K2 bar:** held-out navtrain P1 [0,40) = **0.8210** on 24 logs it never trained on (`PK/raw/2026-10-16-s1c-read/out/RESULT.md` section 2, MEASURED). The navtest shortfall (0.7230 C3, 0.7638 natural) is a **composition** effect per `SHIFT_LOCATE.md` (city mix + failure enrichment account for the point estimate of the gap; residual +0.013, interval +-0.07). On the natural navtest population the head reads **LV 0.860, BOS 0.713, PIT 0.703, SG 0.632** (RESULT.md), while LV is 65.9 % of the train pool and BOS/PIT/SG are 12.4 / 10.4 / 11.3 % (`W/sim_s1c_draws.json`). The missing 0.036 (natural) is carried by the three minority cities.

**What "more label logs" CAN honestly mean** (and what each costs; section 4 ranks them):

| reading | available? | what it needs |
|---|---|---|
| more NAVTRAIN logs | **no** (6 logs / 0.5 %) | nothing to build |
| **x2 = add the 1,167 nuPlan-val logs outside navtrain** | **labels yes, pixels NO** | CPU label build (cheap, scripted and tested here); camera pixels on no disk we hold; a head-only data path in the trainer that does not exist; an A40 run |
| x3 (also the 1,213 non-navtest logs of the `test/` DB dir) | labels yes, pixels no | same, plus **place-leak risk against navtest** (same split family); NOT proposed |
| x4 | **does not exist** on any disk we hold | - |
| dose-response by REMOVING logs (x1/2, x1/4) | yes, with the unchanged trainer | a gate arm-class registration + a `--heldout-logs` file (generated and pinned here); 4.3 A40-h per arm; it is the cheapest way to learn whether adding logs could pay |

## 1. Q1: the label build, measured from source and artifacts

**Script and inputs.** `PK/refe/aux_label_store.py` (`cmd_logmap` :472, `cmd_build` :519, `process_log` :313, `load_yaml_tokens` :434, `index_dbs` :454, `db_map_name` :462). A label is a pure function of (map, rear-axle pose): the origin comes from the bank's `origin_world` (103,037 rows) or, for a frame with no bank row, from the nuPlan DB `ego_pose` (`_db_origins` :301, `process_log` :327); the map polygons from `nuplan-maps-v1.0` (all four cities on D:); the token universe from a navtrain-style yaml; `--navtest` makes the builder refuse every navtest token and log (`process_log` :339).

| item | value | evidence |
|---|---|---|
| production build | 103,139 keys / 1,192 logs in **798.9 s, 4 workers** = 7.75 ms wall / **31 ms worker-time per key**; 0 failed; 351,383,740 B (3.41 KB/key) | `label_store/manifest.json` `runs[0].seconds`, MEASURED |
| origin cross-check, bank vs DB `ego_pose` | max 0.0 m, 8.9e-16 rad over 37,932 rows | same manifest, MEASURED |
| navtrain logs with a DB on the dev box | 463 of 1,192 (the other 729 logs' labels came from bank origins and an inferred map, 462/462 agreement with DB maps on the logs that have one) | `W/inventory_db.json`; `PK/raw/2026-10-12-aux-label-store/README.md` |
| fresh sample, this session (DB-origin path, 4 non-navtrain logs, one per city, stride 60) | 240 keys incl. a 41-key control, 58 s build / 60.3 s wall at 1 worker; 792 keys (stride 15, same 4 logs) 75.0 s wall => **marginal ~27 ms/key, fixed ~54 s per process** (map load + DB glob) | `W/sample/`, ESTIMATED from two runs (n=2) |
| per-key store size, new logs | 3.32 KB/key (2,630,345 B / 792) | `W/sample/store15` (pruned; number kept here), MEASURED |
| **control** | the DB-origin path reproduces the production label of a navtrain log whose labels came from bank origins: **41/41 blobs byte-identical**, max pose diff 2.2e-16; deliberate-regression arm (shifted pairing) **0/40** identical | `W/control_check.json`, MEASURED |

**Where the inputs are.**

| input | dev box | A40 pod | note |
|---|---|---|---|
| navtrain label store (352 MB) | `D:/Projects/TanitAD-artifacts/aux_drivable/label_store` (1,192 shards) | shipped for S1 and S1c (S1c trained on it) | INHERITED for the pod; S1c's RESULT.md consistent |
| navtrain pixels, 4 cameras | **not there** (6,044 of 412,148 images, 1.47 %: `S1_DATA_AUDIT.md`) | `$PIX`, 32 shards (INHERITED) | the head cannot train on the dev box |
| nuPlan DBs, navtrain | 463 / 1,192 | not needed for labels (bank origins) | |
| **nuPlan DBs, trainval logs OUTSIDE navtrain** | **1,167** in `splits/trainval/`, 146,439,364,608 B, 80.6 h, 5,804,017 lidar_pc rows | not probed (never touch the pod) | `W/inventory_db.json`, `W/count_frames.json`, MEASURED |
| nuPlan DBs, `test/` outside navtest | 1,213 of 1,349 (136 are navtest) | - | `W/inventory_db.json`; place-leak risk |
| maps (4 cities) | `D:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0` | - | |
| **camera pixels for the 1,167 extra logs** | **absent**: `data/nuplan-camera` covers 64 mini logs (1 of them is in the extra set); `refe_navtest/frames` covers the 136 navtest logs (0 overlap); `openscene_probe` is navtrain shard 25 | not on the pod's navtrain-only `$PIX` (INHERITED) | `W/cam_overlap.json`, MEASURED (two locations probed) |

**Size of the x2 label build** (`W/count_frames.json`, `W/extra_logs_by_city.json`). Navtrain tokens are 0.01646 of lidar_pc rows on the 462 navtrain logs that have a DB (37,932 tokens / 2,304,657 rows, 32.0 h), i.e. one frame per ~61 rows (~0.33 Hz). The registered token policy S1D-TOK is **stride 60 from row 30**, so the 1,167 extra logs give **~96.7k frames** (5,804,017 / 60; the navtrain-ratio estimate is 95.5k). NOT NAVSIM's scene filter: UNVERIFIED that every such frame is acceptable to a trunk/scorer; a head-only consumer needs only image + calibration.

| city | extra logs | extra frames (ESTIMATED) | train-pool frames (MEASURED) | pool share now | pool share after x2 (all extras) | after minority-only extras |
|---|---|---|---|---|---|---|
| Las Vegas | 704 | 64,715 | 65,608 | 65.9 % | 66.8 % | 50.4 % |
| Boston | 125 | 6,705 | 12,351 | 12.4 % | 9.8 % | 14.6 % |
| Pittsburgh | 134 | 9,401 | 10,334 | 10.4 % | 10.1 % | 15.1 % |
| Singapore | 204 | 14,707 | 11,202 | 11.3 % | 13.3 % | 19.9 % |
| navtest natural margins (S1c RESULT.md) | | | | BOS 30.7, LV 33.5, PIT 20.1, SG 15.8 | | |

**Adding all extras does not change the city imbalance** (LV 66.8 %, BOS falls to 9.8 %); only a **minority-city-only** extension (463 logs, ~30.8k frames) moves it (LV 50.4 %). Boston is scarce everywhere we hold (125 extra, 124 in the test dir).

**CPU cost of the full x2 build.** ESTIMATED 96.7k keys x 27 ms = **~44 min at 1 worker (+~1 min fixed)**, ~13 min at 4 workers (production ratio), ~0.83 CPU-h, ~321 MB output. **RAM is the binding constraint today:** the builder reported 3.6 GB free at the start of the 5-log run and 5.9 GB at its end (it refuses `--workers > 1` below 6 GB), so run 1-2 workers. D: has **6.8 GB free** (df, MEASURED now): the output fits, nothing else should.
Script: `W/build_extra_labels.sh` (tested end to end on 3 logs: `ZZAUXSTORE_DONE 241 0 new=241`, manifest leak 0/0, missing 0). **Recommendation: do NOT run it now** (labels without pixels train nothing; run it only if the dose arm reads row L1, section 4). Exact launch command is in section 6.

**Pixels for x2 (the real cost; UNVERIFIED, ranges not a plan).** 96.7k frames x 4 cameras x 210,312 B (the S1 audit's pod-side mean over 464 navtrain JPEGs, MEASURED, applied to a different corpus = ESTIMATED) = **~81 GB**; minority-only (30.8k frames) **~26 GB**. That assumes per-frame ranged fetches. `code/fetch_front_camera.py` fetches contiguous per-log runs, and its docstring (INHERITED) puts whole-val 4-camera at ~400 GB (3.97 x 99 GB front-camera). Transfer at the audit's measured rates (dev downlink 9.6-13.2 MB/s; pod egress 4.68 MB/s, one file, INHERITED) is 2-9 h for 26-81 GB per leg, relay-bound; on D: (6.8 GB free) it does not fit at all. Whether nuPlan's camera archives are ZIPs with a central directory or TARs is contradicted inside the repo (`fetch_front_camera.py` says ZIP64, `fetch_navtrain_dbs.py` says "TARs with no central directory (R14)"): a `--list` pass on one archive is the first step and is a PI/Master Mind call (a download).

## 2. Q2: can the head arm avoid the A40?

**Short answer: not at S1c scale; a probe-sized version already runs on the dev box.**

| quantity | value | evidence |
|---|---|---|
| head input | `visual_ctx` [B, 7680, 256]; the head (2,440,450 params) reads nothing else (tap "visual") | `PK/refe/aux_drivable_head.py` docstring lines 7-9 and `AuxDrivableConfig.tap` :77; `scorer_finetune.py:42-43,731` |
| cached size per frame | **7,864,448 B fp32** (the `ctx_*.npy` files); **~4.2 MB fp16** (a 64-frame Measure-5 shard is 263,168 KiB) | `D:/Projects/TanitAD-artifacts/aux_navtest_forward/c3_full_ctx/`; `D:/Projects/TanitAD/data/refe_trunk_cache/d7c59f4f/navtrain_train/shard_00000`, MEASURED |
| S1c's 64,146 drawn frames | **504.5 GB fp32 / ~270 GB fp16** | arithmetic on the two rows above |
| the whole 99,495-frame pool | 782 GB fp32 / ~419 GB fp16 | same |
| disk | D: **6.8 GB free**, C: 177 GB free (`df`, MEASURED now) | |
| producing the cache on the dev box | trunk `infer` **1.85 s/token warm** (3.2 s in the cold 32-token run; 12.5 s/token all-in cold), batch 1, fp32, 8 GB card | `PK/raw/2026-10-13-aux-navtest-forward/RESULT.md` lines 32-33, MEASURED |
| => 64,146 frames | **>= 33 h** of pure trunk time (62 h at 3.5 s, 223 h at 12.5 s) | arithmetic |
| navtrain pixels on the dev box | 1.47 % | `S1_DATA_AUDIT.md`, INHERITED |
| head-only training speed on cached ctx | **0.117 s/update** at batch 8 (A2: 5,858 s for 5 units x 10,000 updates) vs **1.275 s/update** for S1c on the A40 (RESULT.md operational check) = 10.9 x faster | MEASURED |

**Does S1c's trainer support a cached-feature path? No.** `grep` of `refe/scorer_finetune.py` for `head_only`/`head-only`/`ctx_cache`/`trunk_cache` finds nothing; S1C_DESIGN section 3 records the same (MEASURED). The head reads `sctx`, the live scoring-decoder context captured by a forward pre-hook in the same frozen-trunk forward (:731), and both scorer arms are always built and stepped (S1C_DESIGN section 3). A cached head-only path needs (i) a `--ctx-cache` data path and a store format (7.86 MB/frame), (ii) a head-only loop that skips the scorer pair, (iii) a new launch-gate flag + G-DVB entry + preflight shape + run-check argument. The only cached head-only trainer that exists is the **probe's** (`PK/raw/2026-10-16-scene-tap-a2/scene_tap_probe.py` `train_unit` :366), on 1,058 navtest tokens.
*Not implemented here, by instruction.* The feasible dev-box use of that rig is a **small, in-domain dose-response** (train on 25 / 50 / 100 % of a fold's logs, read the held-out fold; ~19.5 min per unit): it measures the slope of IoU against log count at ~25-100 logs, a regime where the head reads 0.675 not 0.72 (HYPOTHESIS that the slope transfers to the 1,100-log regime: it may not). It needs a log-subset flag in the probe (ESTIMATED ~10 lines) and ~5 h of shared dev-box GPU; it is the only 0-A40 experiment on this lever that can run before P6 ends.

## 3. What the data say the limiting lever is (ranking by measured gap, Rule Zero)

Remaining gap to K2 after S1c: **C3 0.0770, natural 0.0362** (RESULT.md section 4, MEASURED). Candidate levers, each with its cost and what blocks it:

| # | lever | evidence for | cost | blocker |
|---|---|---|---|---|
| 1 | **Dose arm S1d-q50** (train on half of the train logs, same budget): measures whether log count matters at all and bounds what x2 could add | no arm has varied log count at this budget | **4.3 A40-h** + a gate arm-class patch | A40 is on P6 (~7 days); gate patch (one build-agent session, HYPOTHESIS) |
| 2 | **City-balanced draws** (zero new data): the gap lives in BOS/PIT/SG (0.63-0.71 vs LV 0.860) and S1c's draws are 66.6 % LV | `SHIFT_LOCATE.md` section 6 names it as the lever if the head is "not budget-limited"; per-city table above | 4.3 A40-h + a sampler flag (a TRAINER change, so a larger gate job than #1) | A40; trainer change |
| 3 | S1c replicate (`H1u10k-s1`) | H-ESTIM-SEED-1: S1c's Delta is "single seed, not yet a lever claim" | 4.3 A40-h, a one-line gate change (PREREG_S1C section 5) | PI spend |
| 4 | longer budget (20k updates; within 2x of the fitted top) | held-out P1 rose 0.7953 -> 0.8210 over the last 5,000 updates, but under a cosine schedule that is not a clean learning curve | ~8 A40-h | PI spend |
| 5 | x2 minority-city extension (463 logs) | would move BOS/PIT/SG to 14.6/15.1/19.9 % | labels ~15 min CPU (ready); pixels ~26 GB (UNVERIFIED); head-only trainer path; A40 | pixels (PI), trainer path, A40; **gated on #1 reading L1** |
| 6 | trunk lever (head on P6's trained trunk) | S1c RESULT: head fits in-sample on the probe (0.894), A3R = R-BUDGET | - | P6's checkpoint, PI |

**Rule Zero accounting.** Every row of the S1c O2 "NEXT lever (a)" is either executed or has a named blocker: the label scope and the build script are done (0 A40, this package); the dose arm is *drafted* (`PREREG_S1D_DRAFT.md`, with generated and pinned log subsets in `W/dose/`); what blocks it is the A40 (P6) and a gate registration. **The cheapest experiment that could still make lever (a) work is the dose arm, and it costs 4.3 A40-h, which cannot run for ~7 days.** The dev-box probe dose-response (section 2) is the only thing that can run today, and it is a different regime.

## 4. The dose-response design that makes the lever testable with the UNCHANGED trainer

Mechanism (MEASURED from source): `scorer_finetune.py:663-666`, `tr_idx` drops every bank row whose log is in `--heldout-logs`, while `ho_idx` (the evaluation set) comes from the `--heldout` SETS directory, independent of that file. So passing **(the 24 real held-out logs) U (the train logs to DROP)** shrinks the training pool and leaves the held-out evaluation byte-identical. G7 (`scorer_finetune.py:1179`) checks `train_rows_in_heldout_logs == 0` and `len(ho_idx) > 0`.
`W/make_dose_logs.py` generated the subsets (seed 20261016; per city, logs ordered by sha256(seed|log); nested; checks run):

| arm | kept logs (LV/BOS/PIT/SG) | kept rows / frames | draws per row at 80,000 | `--epochs` to PIN | updates reached | `--heldout-logs` sha256 |
|---|---|---|---|---|---|---|
| S1c (x1, reference) | 1,167 (731/225/116/95) | 174,912 / 99,495 | 0.457 | default 1.0 | 10,000 | pod file (24 logs) `b814998d...` |
| **q50** | 584 (366/112/58/48) | 92,714 / 52,798 | 0.863 | 1.0 (0.87 needed) | 10,000 | `c804ff32...` |
| q25 | 292 (183/56/29/24) | 47,426 / 27,056 | 1.687 | **1.69** | 10,000 | `3874c65d...` |

**TL-1 hazard, measured:** at the default `--epochs 1.0`, q25 would stop silently at **5,929 updates** (`total = min(ceil(rows/batch*epochs/accum), max_updates)`, S1C_DESIGN section 2); `--epochs` must be NAMED and gate-registered for q25 (S1c's `scorer_schedule_rows` already re-derives the reach). All parity checks read **0**: kept U real held-out = {}, kept U navtest logs = {}, file contains all 24 held-out logs, nested q25 in q50 = true.

## 5. What I did NOT establish

* Pod state (never touched): whether the pod still holds S1c's `$DATA` tree; the pod's actual S1c draw (the pod's torch may differ; the audit's caveat stands, so the 64,146 / 1,161 are MEASURED for the dev-box torch and UNVERIFIED against the pod).
* The S1c head's **in-sample** IoU is not banked (A2's 0.894 is the probe's). Held-out navtrain P1 0.8210 is the only unseen-log read besides navtest.
* Whether nuPlan-val logs behave like navtrain logs for this head (domain/place coverage): the place overlap of the extra frames with navtest is NOT measured (pre-launch audit registered in the draft prereg).
* Path A (NuPlanScenario) equivalence for non-bank frames beyond the 41/41 control and the 102 DB-only navtrain tokens already gated in S0.
* Pixel source, format and size for the extra logs (section 1); trunk acceptance of non-NAVSIM-filtered frames.
* No full build, no A40, no trainer or gate edit was done.

## 6. Launch commands for the Master Mind (nothing is launched)

* **Extra-label build (CPU, dev box; ONLY after dose arm row L1; resumable; marker = last line of `build.log`):**
  `bash "C:/Users/Admin/s1d_scope/raw/2026-10-16-s1d-scope/build_extra_labels.sh"` with `WORKERS=1` (2-4 only if >= 6 GB RAM is free) and, to survive the session, launched through WMI `Win32_Process.Create` (memory note: detached jobs die with the session otherwise), e.g.
  `powershell -c "([wmiclass]'Win32_Process').Create('\"C:\Program Files\Git\bin\bash.exe\" -c \"WORKERS=1 PK=D:/Projects/TanitAD/TanitAD\ Research\ Lab/Architecture\ \&\ Inference/Research/2026-09-20-refe-plan bash /c/Users/Admin/s1d_scope/raw/2026-10-16-s1d-scope/build_extra_labels.sh\"')"`
  (copy the package under the repo first so the path is stable). Output: `D:/Projects/TanitAD-artifacts/aux_drivable_extra/` (`label_store/`, `build.log`). Expected last line `ZZAUXSTORE_DONE ~96734 0 new=~96734 skipped=0`. Verify: `manifest.json` `counts.stored_keys` == yaml token count, `leakage` 0/0, `failures_all.jsonl` empty, `missing_tokens.txt` empty, per-map counts vs `W/extra_logs_by_city.json`.
* **Dose arm (A40, after P6 or on a PI reorder):** register arm classes `H1u10k-q50` (and optionally `-q25`) in `launch_gate.py` exactly as S1c did for `H1u10k` (arm class, seed table, the `--heldout-logs` file pinned by sha256 per class, `--epochs` named for q25), then `S1AUX_ARMS="H1u10k-q50-s0"` with the pod copy of `W/dose/heldout_logs_S1D_q50.txt` substituted for `--heldout-logs` and nothing else changed.

## 7. Deliverable manifest

All under `W` = `C:/Users/Admin/s1d_scope/raw/2026-10-16-s1d-scope/` (dev box, working tree of this scratch dir, **unstaged and outside the repo: ONE place only**; the Master Mind lands them). The extracted `PK` tree beside it is a scratch `git archive` of c935a6d (not a deliverable).

| file | what |
|---|---|
| `SCOPE.md` | this report |
| `PREREG_S1D_DRAFT.md` | the draft prereg (dose arms + conditional x2 minority extension) |
| `inventory_db.py/.json` | which local DBs are navtrain / navtest / neither (1,167 / 1,213) |
| `count_frames.py/.json` | lidar_pc rows, hours, bytes, navtrain token ratio |
| `extra_logs_by_city.py/.json`, `test_dir_cities.py/.json` | city mix of extras and of the test-dir non-navtest logs; the minority-extension arithmetic |
| `sim_s1c_draws.py/.json` | what S1 and S1c's head saw (frames, logs, per-city, label coverage, prefix property) |
| `cam_overlap.py/.json` | pixel availability for the extra logs (absent) |
| `make_dose_logs.py`, `dose/` (`heldout_logs_S1D_q50.txt`, `..._q25.txt`, `kept_logs_S1D_q50/q25.txt`, `dose_logs.json`) | the registered log subsets and their checks |
| `make_extra_yaml.py`, `build_extra_labels.sh` | token-universe writer and the tested build wrapper |
| `sample/` (+ `control_check.py/.json`) | the 5-log sample runs, the 3-log smoke test and the byte-identity control |
