# FEASIBILITY — PREREG_L1C (Architecture & Data agent, 2026-10-25)

Written after the measurements below and BEFORE any fresh clip was selected, built or rolled (Amendment 1 sealed 2026-10-25T13:29:38+0100;
its header says "~14:30 Berlin" — that is a narrative typo, the file mtime / seal time is 13:29:38+0100; the sealed file is not edited).

**Outcome: FEASIBLE on the dev box. Branch: PRIMARY (`v_lim` is produced by the corpus's own pipeline; the fallback branch is NOT taken).
No input needs Thor, a VLM run or a missing builder. No blocker.**

## 1. Input-by-input (what the eval139 roll reads, and what produces it for a fresh clip) — evidence class MEASURED unless stated

The roll builds its eval dataset with `with_perception_targets=False` (`battery/code/r7_roll.py:402`), so the agent join, SAM3 map GT, join3d and VIS-1 targets are not read.

| input | builder (launch tree fb47a58) | fresh-clip status |
|---|---|---|
| episode cache (`*.v2ep.pt`, 416x1024 cylindrical PNG) | `stack/scripts/v2_compressed.py build ... --hfov 120 --height 416 --width 1024 --projection-mode cylindrical --codec png --corpus-role eval` (CLI unchanged; role/ingest gate at `:638/:648`) | runs on the dev box from a fetched root; ~37 s/clip/process |
| v8 label record (nav token, SPEED_BAND, tactical ids) | `s2_geom_emit_v7.py:863 emit_one` then `build_v8_speed_max_labels.py` | runs from egomotion only; Alpamayo augmentation absent (stamped opt-out `alpamayo_records.py:111`) |
| **`v_lim`** (fed speed ceiling) | `build_refcv6_speed_max_window.py:71 build`; ceiling = `SPEED_BAND.v_hi_ms` snapped up (`build_v8_speed_max_labels.py:59,228`) | geometry/ego-future only: **reproduced on eval139 with Alpamayo absent: SPEED_BAND and g_str differ in 0 of 139 clips** => PRIMARY branch |
| clip clock sidecar | `build_clip_clock_sidecar.py` (K1/K3 refusal rules `:102-108`) | needs only the cache + the 100 Hz egomotion; refuses the same clips the shipped build refuses (2 of the 24 platform-control clips refused here and absent from the shipped sidecar too) |
| v9 per-frame nav release | `build_v9_labels.py build` (`:1287`), loader md5 check `refcv7_loader.py:815` (`args.r8_v9_eval_md5`) | built without the lead/lane-change passes (both feed TARGETS only) |
| per-clip extrinsics table | shipped 4,711-clip table + the fresh clips' `sensor_extrinsics` rows (same format) | rows read from the dataset's own calibration parquet |
| anchors / class weights / tau / join-defect masks / vis1 name | clip-independent model-build files (E: kit, read-only) | no per-clip build |
| intrinsics | dataset's own `camera_intrinsics` parquet | 300/300 from the parquet |

**Roll path:** `l1_ztac_roll.py --mode eval` (L1) -> `roll_seed_r8_pf.py` (per-frame nav feed) -> `roll_seed_r7.main` -> `run_battery_r7.roll_one(os-only)`. Every eval path derives from
`L.KIT = env R8_EVAL_KIT`, so swapping the kit root swaps cache, labels, speed sidecar, anchors and class weights; three remap entries (extrinsics, clock sidecar, v9) and
`args.r8_v9_eval_md5` are re-pointed. Implemented as `code/l1c_ztac_roll.py` (diff vs the L1 file: `code/l1c_ztac_roll.DIFF.txt`).

## 2. Measured controls behind that verdict

* **The parent's premise "the SAME r0 filters the corpus used" is false as stated** (Amendment 1, A1.1). All 4,719 corpus clips scored with `physicalai_r0._urban_score_from_egomotion`
  (`physicalai_r0.py:78`): only **71.7 %** pass the r0 hard gates, 26.3 % have mean speed above 14 m/s, the corpus spans train/val/test (r0 keeps `split == train`, `:53`), and the corpus
  intersects r0 / r1 selection files in 38 / 144 clips. It was selected by availability of the Alpamayo augmentation. Resolution sealed before selection: eligibility E1-E5; r0 status recorded
  per clip, r0-passing subset scored BESIDE (S-r0). Artifacts: `F:/l1c/work/r0_gate_flat.py`, `F:/l1c/work/corpus_r0_scores_flat.parquet`.
* **Label emission with Alpamayo absent reproduces eval139:** 139/139 clips reproduce SPEED_BAND and g_str; nav token / tactical actions differ in 2/139 (1.44 %), exactly the two whose shipped
  record carries a CoT-driven `turn_suppression`. Stated limitation of fresh labels.
* **Dev-box episode build vs shipped (Thor) build:** 2 eval139 clips rebuilt with the unchanged builder: poses and actions bit-identical, frames within 3/255 (`F:/l1c/work/repro_cmp.py`).
  Becomes the D-platform control (24 first-by-sha256 eval139 clips rebuilt and rolled).
* **Ranged zip reads** (camera and egomotion chunk zips are STORED) avoid the multi-GB chunk archives: ~14 MB/clip fetched, 330 candidates fetched in 950 s, 0 failures, mp4 sha256 and sizes re-verified
  and 12 spot-decoded frames non-zero (HEVC 1920x1080).

## 3. Universe, selection, size (sha12 only; raw ids exist only under F:/l1c) - UPDATED after the build

* Funnel + ordered candidate list: `select/candidates_sha12.json` (code `code/l1c_select.py`, sha256 3b200f31...). 420 rig-eligible candidates in sha256(clip_id) order; 330 fetched (0 failures).
* MEASURED build outcome (A2.1, applied literally): the first 300 built (3 shards, 3,831 s, 0 failures); labels 300/300; clock-sidecar rows 286/300 - the 14 refusals are ALL "fewer than 20 moving rows"
  (stationary clips; the shipped eval139 sidecar refuses 2 of 139 for the same reason and keeps them). Removed by sha12 (`select/chain_failures_sha12.json`) and replaced by candidates 301-315 (one of
  them refused too): final split 300 clips, 300 episode files, labels 300, clock 300, speed sidecar 300 (v_lim valid 300/300), v9 59,650 per-frame rows, **10,456 windows** (`t % 5 == 0`), rig B 220 / rig A 80
  (eval139: B 74 / A 65), r0-passing 195/300. Content assertions: every episode file codec=png, 416x1024 cylindrical, first/middle/last frame decoded non-flat (pixel std >= 10.9), poses/actions finite.
* Estimates (and measured): download ~14 MB/clip (measured); v2 build ~37 s/clip/process (measured, 3 shards); roll 0.49-0.77 s/window MEASURED on the L1 eval rolls (`refcv9_l1/chain.log`), so 10,456 fresh
  windows ~85-135 min on the RTX 4060 plus the wait for the shared GPU lock; fit/score CPU-only (the dry-run regression took ~7 min).
* MEASURED toolchain checks BEFORE the lock was won: (a) CPU smoke of the whole roll on the platform kit (2 clips x 4 windows, `F:/l1c/smoke2`): kit swap, remap, v9 md5 patch, per-frame nav feed,
  `lat_head_tac` hook (d_tac 512), bank save all work; rebuilt-vs-shipped z_tac relative difference 0.9-2.3 % per window (vs 9-114 % between different windows); a CPU roll costs ~90 s/window, so the roll
  needs the GPU. (b) REGRESSION of the analysis chain on L1's own data (`F:/l1c/dry`): l1c_fit_score / l1c_roc / l1c_verdict / l1c_next reproduce L1's H2/H2s/H0/H2m/OS/CF2 cells BIT-EXACTLY on both
  surfaces and seeds (K1, K2, K3, D-shift 0.5472 all equal), and the K4 control reads RED when it should (37 m).
* Disk: F:/l1c only (556 GB free at start); C: carries KB-scale text.

## 4. Residual risks that are NOT blockers (named so they are read)

1. Build platform (x86 PyAV vs Thor): D-platform measures it; rule in A1.5 says D-fresh is "platform-confounded" if D-platform AUC > 0.70.
2. Fresh labels lack the Alpamayo CoT: 1.44 % of eval139 clips differ in nav/tactical; SPEED_BAND (the `v_lim` source) differs in 0.
3. One training seed; the paired bootstrap answers "another draw of episodes" only (H-ESTIM-SEED-1). Inference-seed floor (-0.0021) is the P1 bar's second leg.
4. RAM: ~2-5 GB free during builds (3 v2 processes ~1.6 GB each); the builder is resumable.
