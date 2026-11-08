#!/usr/bin/env bash
# r9e2_read_env.thor.sh -- concrete Thor env pins for r9e2_roll.sh.  usage:  source r9e2_read_env.thor.sh <K2 arm name, e.g. K2 | K2M1N20 | K2M3N20>
# NEW file (E2-S6C-WIRE, 2026-11-08).  Each line is marked by a read-only `ls` / `stat` / `du` over ssh on 2026-11-08:  [OK] exists, [MISS] not found at that path (a candidate
# only), [UNVERIFIED] not probed.  Sourcing this file does NOT make the run startable: every [MISS] pin is a shipment the roll REFUSES without (exit 4 / NO-INPUT abort).
K2ARM="${1:?usage: source r9e2_read_env.thor.sh <K2 arm name>}"
R=/home/nvidia/r8ns/r9e2
export R9E2_K0_RUN="$R/runs/r9e2_K0"                              # [MISS] $R/runs does not exist yet: the chain writes it (model_ema.pt + config.json); name = the arm id of the chain
export R9E2_K2_RUN="$R/runs/r9e2_$K2ARM"                          # [MISS] same
export R9E2_OUT="$R/runs/E2_READ_$K2ARM"                          # output root under the chain's runs dir (created by the roll)
export R9E2_TREE="$R/launch_tree"                                 # [OK] stack/ + taniteval/ only: NO FlyWheels/ below it (see MISSING list)
export R9E2_SPLIT=/home/nvidia/r8ns/r9dump/DEV_HOLD_SPLIT.json   # [OK] 12K (the roll checks sha256 ca5af020...)
export R9E2_S2=/home/nvidia/r8ns/perc8/inputs/s2_windows.json     # [OK] 28K
export R9E2_LEAD_BLOCK="$R/inputs/b1_eval_lead_block_k4_dt0p5.npz"  # [OK] 1.7M
export R9E2_PF_CODE=/home/nvidia/r8ns/fanceil/code/pf/code        # [OK] holds roll_seed_r8_pf.py (candidate for 2026-10-20-refcv8-perframe-final/code; identity UNVERIFIED)
export PF_R8_CODE=/home/nvidia/r8ns/code/batt                     # [OK] holds r8_repoint.py (candidate for 2026-10-15-refcv8-milestone-battery/code; identity UNVERIFIED)
export L1_PF_CODE="$R9E2_PF_CODE" L1_R8_CODE="$PF_R8_CODE"        # == R9E2_PF_CODE / PF_R8_CODE by the roll's own contract
export R9E2_GATES=/home/nvidia/r8ns/perc8/out/T_nosync/refcv8_det_presence_gates_train.json   # [OK] 4.0K (optional pin; the roll checks its sha256)
export R9E2_REMAP="$R/inputs/empty_remap.json"                    # [MISS] an EMPTY {} json: create it on Thor (2 bytes) -- not a shipment
export PF_NF_HOLD=/UNSET_MISSING/2026-10-20-refcv8-nav-feed-fix   # [MISS] no such dir found on Thor (find -maxdepth 4 over r8ns + data)
export R8_EVAL_KIT=/UNSET_MISSING/eval139_kit                     # [MISS] eval139 cache + s2_labels_v8_eval.jsonl.gz: the labels file IS at /home/nvidia/data/v8labels/labels/s2_labels_v8_eval.jsonl.gz [OK]; /home/nvidia/data/refcv6-b1-416x1024-eval139 [OK]; the kit dir the pin means is UNVERIFIED
export R9E2_LIFT_VALID=/UNSET_MISSING/lift_valid.npz              # [MISS] the arm-independent lift-valid mask: no file found (dev-box source: the E1 floor's sidecar; size UNVERIFIED)
export R9E2_MAP_GT_ROOT=/UNSET_MISSING/sam3_gt_v3_eval            # [MISS] refcv6 eval GT-map root; dev-box E:/refcv6_eval_kit/data/sam3_gt_v3_eval (size UNVERIFIED: du on E: not run)
# R9E2_BASE_CWD HAZARD (r9e2_roll.sh header): the battery parent needs ALL THREE refcv5cmp baseline dumps (refcv4b_devbox_dump, refcv5-v2_dump, refcv5-v2-seed1_dump) at a RELATIVE path
#   under that cwd.  [MISS] none found on Thor.  Dev-box source: D:/Projects/TanitAD-artifacts/refcv5cmp/out (size UNVERIFIED).
export R9E2_BASE_CWD=/UNSET_MISSING/basecwd
# FlyWheels incoming packages the roll's `need` lines read under $R9E2_TREE/FlyWheels/TanitAD_EvalFlyWheel/incoming: [MISS] under $R9E2_TREE (no FlyWheels/ dir).  /home/nvidia/r8ns/code/tree
#   HAS a FlyWheels/ dir (contents UNVERIFIED) -- the cheaper fix may be to point R9E2_TREE's FlyWheels at it, or to ship only these packages:
#   2026-11-03-refcv9-camblind/code, 2026-10-27-refcv9-anticheat (PREREG_ANTICHEAT_REFCV8.md/.sha256 + code/anticheat_common.py), 2026-10-25-refcv9-mapsel (code/ + the six PREREG_MAPSEL*.md/.sha256 in its PARENT).
