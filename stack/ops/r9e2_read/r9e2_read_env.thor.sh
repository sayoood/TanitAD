#!/usr/bin/env bash
# r9e2_read_env.thor.sh -- concrete Thor env pins for r9e2_roll.sh.  usage:  source r9e2_read_env.thor.sh <K2 arm name, e.g. K2 | K2M1N20 | K2M3N20>
# NEW file (E2-S6C-WIRE, 2026-11-08); completed 2026-11-09 (E2-READ-ENV): each pin was probed over ssh and every shipped file sha256-verified.  Markers: [OK] exists (probed), [SHIPPED] copied and verified,
#   [MISS] not there yet (the chain writes it).  Sourcing this file does NOT make a REAL run startable until the run dirs exist; R9E2_DRY=1 bash r9e2_roll.sh passes with them absent.
K2ARM="${1:?usage: source r9e2_read_env.thor.sh <K2 arm name>}"
R=/home/nvidia/r8ns/r9e2
export R9E2_K0_RUN="$R/runs/r9e2-K0"                              # [OK] the chain writes runs/r9e2-<ARM> (HYPHEN; MEASURED 2026-11-11 `ls runs/` = r9e2-K0 r9e2-K2M1N20). Was r9e2_K0 (underscore), a guess written before any run dir existed: a REAL read would not have found the checkpoint
export R9E2_K2_RUN="$R/runs/r9e2-$K2ARM"                          # [OK] same naming (r9e2-K2M1N20 MEASURED 2026-11-11, training)
export R9E2_OUT="$R/runs/E2_READ_$K2ARM"                          # output root under the chain's runs dir (created by the roll)
export R9E2_TREE="${R9E2_TREE:-$R/read_tree_cf653ca4bb47_lfix_e77c39d4}"       # [OK] FROZEN READ TREE (2026-11-11) = K2M1N20's launch_tree cf653ca4bb47 + EXACTLY e77c39d4's refcv7_loader.py + r9_e1_dump.py (the T1v-history loader fix; blob sha1 verified on Thor; provenance in <tree>.provenance) + FlyWheels packages, built by r9e2_make_read_tree.sh; dry run 72 commands rc 0. Was read_tree_6d0e629f37a2.  ⚠️ RE-RUN that script on the NEW launch tree after the e2 chain restarts at a later tip, then export R9E2_TREE=<new read tree> before sourcing
export R9E2_SPLIT=/home/nvidia/r8ns/r9dump/DEV_HOLD_SPLIT.json   # [OK] 12K (the roll checks sha256 ca5af020...)
export R9E2_S2=/home/nvidia/r8ns/perc8/inputs/s2_windows.json     # [OK] 28K
export R9E2_LEAD_BLOCK="$R/inputs/b1_eval_lead_block_k4_dt0p5_psync.npz"  # [SHIPPED 2026-11-12] FIXED: the block on the runs' POSE-SYNC clock
#   (build_lead_block_b1.py --k 4 --dt 0.5 --pose-sync-sidecar refcv8_pose_sync_sidecar.jsonl --pose-sync-sign 1.0 --pose-sync-raw-offset 2; sha256 in
#   <block>.sha256 beside it). It WAS b1_eval_lead_block_k4_dt0p5.npz (pose clock): the first real E2 read (15:10Z) then REFUSED distance keeping on all
#   70 DEV episodes (max |dump v0 - block speed| up to 0.060 m/s) because K0 / K2M1N20 read their ego state at the image instant (pose_sync, 0..35 ms later).
export R9E2_PF_CODE=/home/nvidia/r8ns/fanceil/code/pf/code        # [OK] holds roll_seed_r8_pf.py (roll_seed_r8_pf.py is blob-identical to the tip: 05276aa0 -- MEASURED 2026-11-09)
export PF_R8_CODE=/home/nvidia/r8ns/code/batt                     # [OK] holds r8_repoint.py (r8_repoint.py is blob-identical to the tip: ccd6c53c -- MEASURED 2026-11-09)
export L1_PF_CODE="$R9E2_PF_CODE" L1_R8_CODE="$PF_R8_CODE"        # == R9E2_PF_CODE / PF_R8_CODE by the roll's own contract
export R9E2_GATES=/home/nvidia/r8ns/perc8/out/T_nosync/refcv8_det_presence_gates_train.json   # [OK] 4.0K (optional pin; the roll checks its sha256)
export R9E2_REMAP="$R9E2_TREE/stack/ops/refcv8_eval_remap_a2_thor.json"   # [OK] FIXED 2026-11-12: the LANDED Thor remap (blob 720cca5a; the E1 dump chain's
#   REFCV6_REMAP_OVERRIDES; == r8_repoint's own Thor default refcv8_eval_remap_devbox.json, 4 flags). It WAS an EMPTY {} json: the first real roll (10:27Z) then
#   REFUSED in refcv7_loader on exactly these 4 flags (--agent-rig-extrinsics, --vis1-sidecar, --nav-compliance-tau-file, --clip-clock-sidecar: the run's
#   argv names the refcv8 train4572/eval139 files, the kit's defaults are refcv6's). The dry run never loads the model, so nothing reached it.
export PF_NF_HOLD="$R9E2_TREE/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-nav-feed-fix"   # [OK] PREREG_NAV_FEED.md + code/overlay inside the frozen read tree (git export of the package)
export R8_EVAL_KIT=/home/nvidia                                   # [OK] Thor's data/ IS the kit layout (refcv7_loader.py:276; fanceil run_fanroll_thor.sh uses the same pin): data/refcv6-b1-416x1024-eval139 + data/v8labels/labels/s2_labels_v8_eval.jsonl.gz probed present
export R9E2_LIFT_VALID=/home/nvidia/r8ns/r9e1/read_v3/LIFT_VALID_M1.npz   # [OK] 19,662 B, the E1 floor's lift-valid sidecar (read-only reference; E1 dir untouched)
export R9E2_MAP_GT_ROOT="$R/inputs/sam3_gt_v3_eval"              # [SHIPPED] 141 files / 637 MB from dev box E:/refcv6_eval_kit/data/sam3_gt_v3_eval, sha256 of every file verified on Thor (inputs/gt.sha)
# R9E2_BASE_CWD HAZARD (r9e2_roll.sh header): the battery parent needs ALL THREE refcv5cmp baseline dumps (refcv4b_devbox_dump, refcv5-v2_dump, refcv5-v2-seed1_dump) at a RELATIVE path
#   under that cwd.  SHIPPED 2026-11-09 from dev-box E:/Projects/TanitAD-artifacts/refcv5cmp/out (ep*.npz, decisions/) + manifest.json from C:/Users/Admin/c_cleanup_archive_2026-10-15_b (E: lacks it); sha256 verified (inputs/b_*.sha).
export R9E2_BASE_CWD="$R/inputs/basecwd"                           # [SHIPPED] holds D:/Projects/TanitAD-artifacts/refcv5cmp/out/{refcv4b_devbox_dump,refcv5-v2_dump,refcv5-v2-seed1_dump} (ARTIFACT_ROOTS entry 1, relative)
# FlyWheels packages: the roll's `need` lines read them under $R9E2_TREE/FlyWheels/TanitAD_EvalFlyWheel/incoming.  They are exported from git (autocrlf off, blob hashes verified) into $R/inputs/kit_pkgs and
#   copied into the frozen read tree by r9e2_make_read_tree.sh <launch_tree> $R/inputs/kit_pkgs  (also adds 2026-09-28-refcv7-standard-tests/battery/code: r8_repoint.BATTERY_CODE).
