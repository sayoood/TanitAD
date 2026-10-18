#!/usr/bin/env bash
# =============================================================================
# cleanup_thor_A.sh  --  TanitAD Thor disk cleanup, CATEGORY A (re-creatable / git-identical / caches)
# Generated 2026-10-04 by the READ-ONLY Thor inventory (see INVENTORY_THOR.md).
# RUN ON THOR as user nvidia (host thor6). Review every line first.
#
#   bash cleanup_thor_A.sh            -> DRY RUN (default): prints what would go, deletes NOTHING
#   DRYRUN=0 bash cleanup_thor_A.sh   -> really deletes
#
# Safety nets built in:
#   * KEEP guard: refuses any path that is inside a KEEP entry OR that CONTAINS a KEEP entry,
#     anything outside /home/nvidia and /tmp, and any path with glob characters or '..'.
#   * Refuses to run for real while a GPU eval holds /home/nvidia/refcv7_post/thor_gpu.lock (flock) or a python job of user nvidia is alive
#     (override only with FORCE=1).
#   * Each rm line is preceded by a comment: <size> <category> <reason> | surviving copy.
#   * Deleting uses nice/ionice so a running box is not disturbed.
# =============================================================================
set -u
DRYRUN="${DRYRUN:-1}"
FORCE="${FORCE:-0}"
GPU_LOCK="/home/nvidia/refcv7_post/thor_gpu.lock"
SALVAGE_DIR="${SALVAGE_DIR:-/home/nvidia/_cleanup_salvage_2026-10-04}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOTAL=0; NDEL=0; REFUSED=0
LOWPRIO=(nice -n 19); command -v ionice >/dev/null 2>&1 && LOWPRIO+=(ionice -c3)

KEEP=(
  '/home/nvidia/refcv7_run/runs'
  '/home/nvidia/refcv7_run/runs.d'
  '/home/nvidia/refcv7_run/fec3a0dccf'
  '/home/nvidia/refcv7_run/sup_refcv7-r101-s0.out'
  '/home/nvidia/refcv7_post'
  '/home/nvidia/venvs'
  '/home/nvidia/refcv7_gate_fec3a0dccf'
  '/home/nvidia/refcv7_bind/run_map'
  '/home/nvidia/refcv7_bind/run_box'
  '/home/nvidia/refcv7_dry_thor/q4c_grid_vs_egolog_ALLTRAIN.json'
  '/home/nvidia/bx_0252/calib'
  '/home/nvidia/.cache/huggingface/hub/models--timm--resnet101.a1_in1k'
  '/home/nvidia/.ssh'
  '/home/nvidia/refcv7_run_thor_final.sh'
  '/home/nvidia/refcv7_run_thor_finish.sh'
  '/home/nvidia/data/physicalai-b1-w120-416x1024cyl'
  '/home/nvidia/data/refcv6-b1-416x1024-train'
  '/home/nvidia/data/refcv6-b1-416x1024-eval139'
  '/home/nvidia/data/sam3_gt_v3'
  '/home/nvidia/data/v8labels'
  '/home/nvidia/data/v72'
  '/home/nvidia/data/joins'
  '/home/nvidia/data/join3d'
  '/home/nvidia/data/anchors'
  '/home/nvidia/data/refcv7'
  '/home/nvidia/data/physicalai-b1-EVAL6-w120-256x640cyl'
  '/home/nvidia/data/refcv6_clip_clock_sidecar.jsonl'
  '/home/nvidia/data/refcv6_speed_max_v8_eval.jsonl'
  '/home/nvidia/data/refcv6_speed_max_v8_eval.jsonl.meta.json'
  '/home/nvidia/data/refcv6_speed_max_v8_train.jsonl'
  '/home/nvidia/data/refcv6_speed_max_v8_train.jsonl.meta.json'
  '/home/nvidia/data/refcv6_train_eval139_extrinsics.json'
  '/home/nvidia/data/refcv6_eval139_extrinsics.json'
  '/home/nvidia/data/refcv6_train_extrinsics.json'
  '/home/nvidia/data/refcv6_train_clips.txt'
  '/home/nvidia/data/eval139_ids.txt'
  '/home/nvidia/data/pai_extrinsics_table.py'
  '/home/nvidia/data/physicalai-b1/calibration'
  '/home/nvidia/data/physicalai-b1/r0'
  '/home/nvidia/data/physicalai-b1/clip_index.parquet'
)

human() { numfmt --to=iec --suffix=B "$1" 2>/dev/null || echo "$1 B"; }

guard() {   # 0 = may delete
  local p="$1" rp k
  case "$p" in /home/nvidia/*|/tmp/*) ;; *) echo "REFUSE (outside /home/nvidia and /tmp): $p" >&2; return 1;; esac
  if printf '%s' "$p" | grep -q '[][*?{}]' || [[ "$p" == *..* ]]; then echo "REFUSE (glob char or '..'): $p" >&2; return 1; fi
  rp="$(realpath -m -- "$p")" || return 1
  case "$rp" in ""|"/"|"/home"|"/home/nvidia"|"/tmp") echo "REFUSE (root-like): $p" >&2; return 1;; esac
  for k in "${KEEP[@]}"; do
    if [ "$rp" = "$k" ] || [[ "$rp" == "$k"/* ]]; then echo "REFUSE (inside KEEP $k): $p" >&2; return 1; fi
    if [[ "$k" == "$rp"/* ]]; then echo "REFUSE (contains KEEP $k): $p" >&2; return 1; fi
  done
  return 0
}

preflight() {
  [ "$(whoami)" = "nvidia" ] || { echo "must run as user nvidia"; exit 2; }
  [ "$(hostname)" = "thor6" ] || { echo "this script is for host thor6 (got $(hostname))"; exit 2; }
  if [ -e "$GPU_LOCK" ]; then
    exec 9<"$GPU_LOCK"
    if ! flock -n 9; then echo "GPU lock $GPU_LOCK is HELD -> an eval is running; refusing."; exit 3; fi
    [ "$DRYRUN" = 1 ] && flock -u 9      # real run keeps the lock until the script ends
  fi
  local py; py="$(ps -u nvidia -o comm= | grep -E '^(python|python3|torchrun)' | wc -l)"
  if [ "$py" -gt 0 ] && [ "$DRYRUN" != 1 ] && [ "$FORCE" != 1 ]; then
    echo "$py python process(es) of user nvidia are alive; refusing a real run (FORCE=1 to override)."; exit 3
  fi
  [ "$DRYRUN" = 1 ] && echo "### DRY RUN -- nothing will be deleted ###" || echo "### REAL RUN -- deleting ###"
  START_AVAIL="$(df -B1 --output=avail / | tail -1 | tr -d ' ')"
  df -h / | tail -1
}

rm_rf() {   # rm_rf PATH   (the guarded, size-reporting form of: rm -rf -- PATH)
  local p="$1" sz
  guard "$p" || { REFUSED=$((REFUSED+1)); return 0; }
  if [ ! -e "$p" ] && [ ! -L "$p" ]; then echo "ABSENT      $p"; return 0; fi
  sz="$("${LOWPRIO[@]}" du -sB1 -- "$p" 2>/dev/null | cut -f1)"; sz="${sz:-0}"
  TOTAL=$((TOTAL+sz)); NDEL=$((NDEL+1))
  printf '%-9s %9s  %s\n' "$([ "$DRYRUN" = 1 ] && echo WOULD-RM || echo RM)" "$(human "$sz")" "$p"
  if [ "$DRYRUN" != 1 ]; then
    "${LOWPRIO[@]}" rm -rf -- "$p"
    if [ -e "$p" ] || [ -L "$p" ]; then echo "FAILED-TO-REMOVE $p" >&2; REFUSED=$((REFUSED+1)); fi
  fi
}

finish() {
  echo "-----------------------------------------------------------------"
  printf 'items: %d   sum of per-line sizes: %s   refused-by-guard: %d\n' "$NDEL" "$(human "$TOTAL")" "$REFUSED"
  echo "(the sum can exceed the space really freed: hardlinked twin trees such as nb2a19_thor/code <-> nb2r5_2ac0/code are counted once per line)"
  df -h / | tail -1
  if [ "$DRYRUN" != 1 ]; then
    local now; now="$(df -B1 --output=avail / | tail -1 | tr -d ' ')"
    echo "ACTUALLY FREED (df avail before/after): $(human "$((now-START_AVAIL))")"
  else
    echo "DRY RUN only. Re-run with DRYRUN=0 to delete."
  fi
}


salvage() {   # salvage ROOT TGZ rel1 rel2 ...   (keeps the few WIP files that are NOT in git, before the tree is removed)
  local root="$1" tgz="$2"; shift 2
  local n=$#
  if [ "$DRYRUN" = 1 ]; then echo "WOULD-SALVAGE $n file(s) from $root -> $SALVAGE_DIR/$tgz"; return 0; fi
  mkdir -p -- "$SALVAGE_DIR" || return 1
  if [ -s "$SALVAGE_DIR/$tgz" ]; then echo "SALVAGE-EXISTS $SALVAGE_DIR/$tgz"; return 0; fi
  tar -C "$root" -czf "$SALVAGE_DIR/$tgz.part" -- "$@" && [ "$(tar -tzf "$SALVAGE_DIR/$tgz.part" | wc -l)" -eq "$n" ] \
    && mv -- "$SALVAGE_DIR/$tgz.part" "$SALVAGE_DIR/$tgz" && echo "SALVAGED $n file(s) -> $SALVAGE_DIR/$tgz" \
    || { echo "SALVAGE FAILED for $root (nothing will be deleted for it)"; rm -f -- "$SALVAGE_DIR/$tgz.part"; }
}

rm_rf_salv() {   # rm_rf_salv PATH TGZ [TGZ...] : like rm_rf, but only if every salvage tgz exists (real run)
  local p="$1"; shift
  if [ "$DRYRUN" != 1 ]; then
    local t; for t in "$@"; do [ -s "$SALVAGE_DIR/$t" ] || { echo "SKIP (no salvage tgz $t): $p"; REFUSED=$((REFUSED+1)); return 0; }; done
  fi
  rm_rf "$p"
}

preflight


# ---------------------------------------------------------------------------
# 0. SALVAGE: WIP files that exist ONLY inside a tree that is about to be deleted (all small)
# ---------------------------------------------------------------------------
# 2 file(s) not in git in bx_0252/tree
salvage '/home/nvidia/bx_0252/tree' 'bx_0252__tree.tgz' 'stack/tanitad/data/vis1.py' 'stack/tanitad/data/vis1_zbuffer.py'
# 13 file(s) not in git in bx_0412/tree
salvage '/home/nvidia/bx_0412/tree' 'bx_0412__tree.tgz' 'stack/tests/test_occ_knob_is_stamped.py' 'stack/tests/test_refcv7_vis1.py' 'stack/tests/test_refc_v3_agent_provenance.py' 'stack/tests/test_declared_vs_built.py' 'stack/tests/test_refcv7_box_head_wiring.py' 'stack/tests/test_g_box_overfit.py' 'stack/tanitad/data/vis1_zbuffer.py' 'stack/tanitad/models/slot_presence.py' 'stack/tanitad/models/refcv6_perception_branch.py' 'stack/tanitad/train/declared_vs_built.py' 'stack/scripts/refc_v3_train.py' 'stack/scripts/g_box_overfit.py' 'taniteval/tools/refcv3_arm.py'
# 14 file(s) not in git in bx_0412/tree_r6
salvage '/home/nvidia/bx_0412/tree_r6' 'bx_0412__tree_r6.tgz' 'stack/tests/test_occ_knob_is_stamped.py' 'stack/tests/test_refcv7_vis1.py' 'stack/tests/test_refc_v3_agent_provenance.py' 'stack/tests/test_declared_vs_built.py' 'stack/tests/test_refcv7_box_head_wiring.py' 'stack/tests/test_refcv7_r6_denoise.py' 'stack/tests/test_g_box_overfit.py' 'stack/tanitad/data/vis1_zbuffer.py' 'stack/tanitad/models/slot_presence.py' 'stack/tanitad/models/refcv6_perception_branch.py' 'stack/tanitad/train/declared_vs_built.py' 'stack/scripts/refc_v3_train.py' 'stack/scripts/g_box_overfit.py' 'taniteval/tools/refcv3_arm.py'
# 10 file(s) not in git in bx_0524/tree_main
salvage '/home/nvidia/bx_0524/tree_main' 'bx_0524__tree_main.tgz' 'stack/tests/test_occ_knob_is_stamped.py' 'stack/tests/test_refcv7_vis1.py' 'stack/tests/test_declared_vs_built.py' 'stack/tests/test_g_box_overfit.py' 'stack/tanitad/data/vis1_zbuffer.py' 'stack/tanitad/models/refcv6_perception_branch.py' 'stack/tanitad/train/declared_vs_built.py' 'stack/scripts/refc_v3_train.py' 'stack/scripts/g_box_overfit.py' 'taniteval/tools/refcv3_arm.py'
# 10 file(s) not in git in bx_0524/tree_r6
salvage '/home/nvidia/bx_0524/tree_r6' 'bx_0524__tree_r6.tgz' 'stack/tests/test_occ_knob_is_stamped.py' 'stack/tests/test_refcv7_vis1.py' 'stack/tests/test_declared_vs_built.py' 'stack/tests/test_g_box_overfit.py' 'stack/tanitad/data/vis1_zbuffer.py' 'stack/tanitad/models/refcv6_perception_branch.py' 'stack/tanitad/train/declared_vs_built.py' 'stack/scripts/refc_v3_train.py' 'stack/scripts/g_box_overfit.py' 'taniteval/tools/refcv3_arm.py'
# 3 file(s) not in git in bx_anch_1116/tree
salvage '/home/nvidia/bx_anch_1116/tree' 'bx_anch_1116__tree.tgz' 'stack/tests/test_declared_vs_built.py' 'stack/scripts/refc_v3_train.py' 'taniteval/tools/refcv3_arm.py'
# 8 file(s) not in git in bx_cmain_1014/tree
salvage '/home/nvidia/bx_cmain_1014/tree' 'bx_cmain_1014__tree.tgz' 'stack/tests/test_occ_knob_is_stamped.py' 'stack/tests/test_declared_vs_built.py' 'stack/tests/test_g_box_overfit.py' 'stack/tanitad/models/refcv6_perception_branch.py' 'stack/tanitad/train/declared_vs_built.py' 'stack/scripts/refc_v3_train.py' 'stack/scripts/g_box_overfit.py' 'taniteval/tools/refcv3_arm.py'
# 3 file(s) not in git in bx_gate_1136/tree_anch
salvage '/home/nvidia/bx_gate_1136/tree_anch' 'bx_gate_1136__tree_anch.tgz' 'stack/tests/test_declared_vs_built.py' 'stack/scripts/refc_v3_train.py' 'taniteval/tools/refcv3_arm.py'
# 8 file(s) not in git in bx_tests_0934/tree_main
salvage '/home/nvidia/bx_tests_0934/tree_main' 'bx_tests_0934__tree_main.tgz' 'stack/tests/test_occ_knob_is_stamped.py' 'stack/tests/test_declared_vs_built.py' 'stack/tests/test_g_box_overfit.py' 'stack/tanitad/models/refcv6_perception_branch.py' 'stack/tanitad/train/declared_vs_built.py' 'stack/scripts/refc_v3_train.py' 'stack/scripts/g_box_overfit.py' 'taniteval/tools/refcv3_arm.py'
# 9 file(s) not in git in bx_tests_0934/tree_r6
salvage '/home/nvidia/bx_tests_0934/tree_r6' 'bx_tests_0934__tree_r6.tgz' 'stack/tests/test_cls_weight_stamp.py' 'stack/tests/test_occ_knob_is_stamped.py' 'stack/tests/test_declared_vs_built.py' 'stack/tests/test_g_box_overfit.py' 'stack/tanitad/models/refcv6_perception_branch.py' 'stack/tanitad/train/declared_vs_built.py' 'stack/scripts/refc_v3_train.py' 'stack/scripts/g_box_overfit.py' 'taniteval/tools/refcv3_arm.py'
# 1 file(s) not in git in gmo_bind_pf/tree
salvage '/home/nvidia/gmo_bind_pf/tree' 'gmo_bind_pf__tree.tgz' 'stack/ops/runs.d/refcv7-r101-s0.argv.json'
# 9 file(s) not in git in gmo_early_0327/code
salvage '/home/nvidia/gmo_early_0327/code' 'gmo_early_0327__code.tgz' 'stack/tests/test_map_head_hires.py' 'stack/tests/test_semantic_map_gt_fine.py' 'stack/tests/test_declared_vs_built.py' 'stack/tanitad/refs/refc_v3.py' 'stack/tanitad/refs/refc.py' 'stack/tanitad/train/declared_vs_built.py' 'stack/scripts/refc_v3_train.py' 'taniteval/tests/test_map_hires_metrics.py' 'taniteval/taniteval/map_hires_metrics.py'
# 6 file(s) not in git in gmo_early_0327/code_t1
salvage '/home/nvidia/gmo_early_0327/code_t1' 'gmo_early_0327__code_t1.tgz' 'stack/tests/test_semantic_map_gt_fine.py' 'stack/tests/test_declared_vs_built.py' 'stack/tanitad/refs/refc_v3.py' 'stack/tanitad/refs/refc.py' 'stack/tanitad/train/declared_vs_built.py' 'stack/scripts/refc_v3_train.py'
# 2 file(s) not in git in nb2r4_1b81/code
salvage '/home/nvidia/nb2r4_1b81/code' 'nb2r4_1b81__code.tgz' 'stack/tests/test_map_hires_overfit.py' 'stack/scripts/map_hires_overfit.py'
# 2 file(s) not in git in refcv6_run/35be81285e
salvage '/home/nvidia/refcv6_run/35be81285e' 'refcv6_run__35be81285e.tgz' 'ops/smoke_refcv6_ckpt.sh' 'ops/run_refcv6_ckpt.sh'
# 2 file(s) not in git in refcv6_run/4ca6168822
salvage '/home/nvidia/refcv6_run/4ca6168822' 'refcv6_run__4ca6168822.tgz' 'ops/smoke_refcv6_ckpt.sh' 'ops/run_refcv6_ckpt.sh'
# 2 file(s) not in git in refcv6_run/pre_C
salvage '/home/nvidia/refcv6_run/pre_C' 'refcv6_run__pre_C.tgz' 'ops/smoke_refcv6_ckpt.sh' 'ops/run_refcv6_ckpt.sh'
# 2 file(s) not in git in refcv6_run/pre_D
salvage '/home/nvidia/refcv6_run/pre_D' 'refcv6_run__pre_D.tgz' 'ops/smoke_refcv6_ckpt.sh' 'ops/run_refcv6_ckpt.sh'
# 3 file(s) not in git in refcv6_run/pre_E
salvage '/home/nvidia/refcv6_run/pre_E' 'refcv6_run__pre_E.tgz' 'stack/scripts/refc_v3_train.py' 'ops/smoke_refcv6_ckpt.sh' 'ops/run_refcv6_ckpt.sh'
# 6 file(s) not in git in refcv6_run/pre_F
salvage '/home/nvidia/refcv6_run/pre_F' 'refcv6_run__pre_F.tgz' 'stack/tanitad/models/timm_trunk.py' 'stack/tanitad/refs/refc.py' 'stack/scripts/refc_v3_train.py' 'ops/smoke_refcv6_ckpt.sh' 'ops/run_refcv6.sh' 'ops/run_refcv6_ckpt.sh'
# 2 file(s) not in git in refcv6_run/pre_G
salvage '/home/nvidia/refcv6_run/pre_G' 'refcv6_run__pre_G.tgz' 'ops/smoke_refcv6_ckpt.sh' 'ops/run_refcv6_ckpt.sh'
# 1 file(s) not in git in refcv7_bind/tree
salvage '/home/nvidia/refcv7_bind/tree' 'refcv7_bind__tree.tgz' 'stack/ops/runs.d/refcv7-r101-s0.argv.json'
# 1 file(s) not in git in refcv7_bind/tree_map
salvage '/home/nvidia/refcv7_bind/tree_map' 'refcv7_bind__tree_map.tgz' 'stack/ops/runs.d/refcv7-r101-s0.argv.json'

# ---------------------------------------------------------------------------
# 1. CACHES AND TEMP (all re-created on demand)
# ---------------------------------------------------------------------------
# 371 MB  A  pip http cache  | surviving: PyPI
rm_rf '/home/nvidia/.cache/pip'
# 1.82 GB  A  uv package cache; MEASURED no inode in it is hardlinked from outside (venvs keep their own copies); re-downloaded on demand  | surviving: PyPI
rm_rf '/home/nvidia/.cache/uv'
# 272 MB  A  NVIDIA JIT ComputeCache; kernels recompile on next use (minutes)  | surviving: regenerated
rm_rf '/home/nvidia/.nv'
# 447 MB  A  2026-08-30 measurement scratch: 12 loose 17-42 MB *.v2ep.pt sample episodes, no consumer (the real caches are physicalai-b1-w120-*)  | surviving: same episodes live in physicalai-b1-w120-256x640cyl
rm_rf '/home/nvidia/data/_measure_tmp'
# 1.94 GB  A  SAM3 corpus production per-launch scratch (100 dirs of intermediate *.npz); run finished 2026-09-22 (DONE), outputs are in corpus/out and on HF  | surviving: sam3map/corpus/out + HF dataset
rm_rf '/home/nvidia/sam3map/corpus/scratch'
# 387 MB  A  torch inductor compile cache; recompiled on next --trunk-compile (a few minutes)  | surviving: regenerated
rm_rf '/tmp/torchinductor_nvidia'
# 29 MB  A  /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27  | surviving: regenerated by the v4 smoke tests
rm_rf '/tmp/v4smoke_0wvs52sm'
# 29 MB  A  /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27  | surviving: regenerated by the v4 smoke tests
rm_rf '/tmp/v4smoke_105wfszz'
# 29 MB  A  /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27  | surviving: regenerated by the v4 smoke tests
rm_rf '/tmp/v4smoke_16qah13r'
# 29 MB  A  /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27  | surviving: regenerated by the v4 smoke tests
rm_rf '/tmp/v4smoke_1b02_69g'
# 29 MB  A  /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27  | surviving: regenerated by the v4 smoke tests
rm_rf '/tmp/v4smoke_30ez15_2'
# 29 MB  A  /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27  | surviving: regenerated by the v4 smoke tests
rm_rf '/tmp/v4smoke_4zhvalf9'
# 29 MB  A  /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27  | surviving: regenerated by the v4 smoke tests
rm_rf '/tmp/v4smoke_54bvyz_g'
# 29 MB  A  /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27  | surviving: regenerated by the v4 smoke tests
rm_rf '/tmp/v4smoke_5kg8xyfo'
# 29 MB  A  /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27  | surviving: regenerated by the v4 smoke tests
rm_rf '/tmp/v4smoke_6pv5uq8h'
# 29 MB  A  /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27  | surviving: regenerated by the v4 smoke tests
rm_rf '/tmp/v4smoke_6wdgg5s0'
# 29 MB  A  /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27  | surviving: regenerated by the v4 smoke tests
rm_rf '/tmp/v4smoke_6wkqebvm'
# 29 MB  A  /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27  | surviving: regenerated by the v4 smoke tests
rm_rf '/tmp/v4smoke_8b_gozb4'
# 29 MB  A  /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27  | surviving: regenerated by the v4 smoke tests
rm_rf '/tmp/v4smoke_8kabur3m'
# 29 MB  A  /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27  | surviving: regenerated by the v4 smoke tests
rm_rf '/tmp/v4smoke_9exjidvo'
# 29 MB  A  /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27  | surviving: regenerated by the v4 smoke tests
rm_rf '/tmp/v4smoke_9oeb5jl1'
# 29 MB  A  /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27  | surviving: regenerated by the v4 smoke tests
rm_rf '/tmp/v4smoke_ax56op1d'
# 29 MB  A  /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27  | surviving: regenerated by the v4 smoke tests
rm_rf '/tmp/v4smoke_dvkviwk9'
# 29 MB  A  /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27  | surviving: regenerated by the v4 smoke tests
rm_rf '/tmp/v4smoke_g00f5tlo'
# 29 MB  A  /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27  | surviving: regenerated by the v4 smoke tests
rm_rf '/tmp/v4smoke_h5k_ym4n'
# 29 MB  A  /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27  | surviving: regenerated by the v4 smoke tests
rm_rf '/tmp/v4smoke_huu6p_zv'
# 29 MB  A  /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27  | surviving: regenerated by the v4 smoke tests
rm_rf '/tmp/v4smoke_jcewkq7y'
# 29 MB  A  /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27  | surviving: regenerated by the v4 smoke tests
rm_rf '/tmp/v4smoke_kepq63z1'
# 29 MB  A  /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27  | surviving: regenerated by the v4 smoke tests
rm_rf '/tmp/v4smoke_kmmri5li'
# 29 MB  A  /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27  | surviving: regenerated by the v4 smoke tests
rm_rf '/tmp/v4smoke_o3pknx_t'
# 29 MB  A  /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27  | surviving: regenerated by the v4 smoke tests
rm_rf '/tmp/v4smoke_p0fyakbc'
# 29 MB  A  /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27  | surviving: regenerated by the v4 smoke tests
rm_rf '/tmp/v4smoke_pzoaobxt'
# 29 MB  A  /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27  | surviving: regenerated by the v4 smoke tests
rm_rf '/tmp/v4smoke_s2_7bugz'
# 29 MB  A  /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27  | surviving: regenerated by the v4 smoke tests
rm_rf '/tmp/v4smoke_v_cvopes'
# 29 MB  A  /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27  | surviving: regenerated by the v4 smoke tests
rm_rf '/tmp/v4smoke_vc4owzr0'
# 29 MB  A  /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27  | surviving: regenerated by the v4 smoke tests
rm_rf '/tmp/v4smoke_w9kx4j3g'
# 29 MB  A  /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27  | surviving: regenerated by the v4 smoke tests
rm_rf '/tmp/v4smoke_wxh1czp1'
# 29 MB  A  /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27  | surviving: regenerated by the v4 smoke tests
rm_rf '/tmp/v4smoke_y457bocr'
# 29 MB  A  /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27  | surviving: regenerated by the v4 smoke tests
rm_rf '/tmp/v4smoke_z0hfvgny'
# 29 MB  A  /tmp smoke-test run dir (ckpt.pt + ckpt_best.pt + metrics), 2026-09-26/27  | surviving: regenerated by the v4 smoke tests
rm_rf '/tmp/v4smoke_z_hc3iae'

# ---------------------------------------------------------------------------
# 2. SMOKE / PROFILE / PROBE / GATE-SCRATCH CHECKPOINTS (config, metrics and logs beside each one are KEPT)
# ---------------------------------------------------------------------------
# 1.30 GB  A  refcv6-V0 smoke checkpoint (1.3 GB); config/logs beside it are kept  | surviving: re-run of the smoke
rm_rf '/home/nvidia/refcv6/raw/smoke-V0/ckpt.pt'
# 1.17 GB  A  refcv6 smoke/profile run end-state checkpoint; config/metrics/logs beside it are kept  | surviving: re-run of the smoke (trainer is in git)
rm_rf '/home/nvidia/refcv6_run/prof/S8/ckpt.pt'
# 1.17 GB  A  refcv6 smoke/profile run end-state checkpoint; config/metrics/logs beside it are kept  | surviving: re-run of the smoke (trainer is in git)
rm_rf '/home/nvidia/refcv6_run/smoke/S10_b16_fold_dedup/ckpt.pt'
# 1.17 GB  A  refcv6 smoke/profile run end-state checkpoint; config/metrics/logs beside it are kept  | surviving: re-run of the smoke (trainer is in git)
rm_rf '/home/nvidia/refcv6_run/smoke/S11b_b16_fold_dedup_compile/ckpt.pt'
# 1.17 GB  A  refcv6 smoke/profile run end-state checkpoint; config/metrics/logs beside it are kept  | surviving: re-run of the smoke (trainer is in git)
rm_rf '/home/nvidia/refcv6_run/smoke/S3_b8_ce10_ck8/ckpt.pt'
# 1.17 GB  A  refcv6 smoke/profile run end-state checkpoint; config/metrics/logs beside it are kept  | surviving: re-run of the smoke (trainer is in git)
rm_rf '/home/nvidia/refcv6_run/smoke/S4_b8_ce10_ck8_bf16_cl/ckpt.pt'
# 1.17 GB  A  refcv6 smoke/profile run end-state checkpoint; config/metrics/logs beside it are kept  | surviving: re-run of the smoke (trainer is in git)
rm_rf '/home/nvidia/refcv6_run/smoke/S5_ck24/ckpt.pt'
# 1.17 GB  A  refcv6 smoke/profile run end-state checkpoint; config/metrics/logs beside it are kept  | surviving: re-run of the smoke (trainer is in git)
rm_rf '/home/nvidia/refcv6_run/smoke/S6_ck8_bench/ckpt.pt'
# 1.17 GB  A  refcv6 smoke/profile run end-state checkpoint; config/metrics/logs beside it are kept  | surviving: re-run of the smoke (trainer is in git)
rm_rf '/home/nvidia/refcv6_run/smoke/S7_b4_nockpt/ckpt.pt'
# 1.17 GB  A  refcv6 smoke/profile run end-state checkpoint; config/metrics/logs beside it are kept  | surviving: re-run of the smoke (trainer is in git)
rm_rf '/home/nvidia/refcv6_run/smoke/S8_ck8_bf16_cl_fold/ckpt.pt'
# 1.17 GB  A  refcv6 smoke/profile run end-state checkpoint; config/metrics/logs beside it are kept  | surviving: re-run of the smoke (trainer is in git)
rm_rf '/home/nvidia/refcv6_run/smoke/S9_ck8_bf16_cl_fold_dedup/ckpt.pt'
# 1.17 GB  A  refcv6 smoke/profile run end-state checkpoint; config/metrics/logs beside it are kept  | surviving: re-run of the smoke (trainer is in git)
rm_rf '/home/nvidia/refcv6_run/smoke/ck_b4_nocd/ckpt.pt'
# 1.17 GB  A  refcv6 smoke/profile run end-state checkpoint; config/metrics/logs beside it are kept  | surviving: re-run of the smoke (trainer is in git)
rm_rf '/home/nvidia/refcv6_run/smoke/prof_S3/ckpt.pt'
# 403 MB  A  gate scratch (G-EVAL trainer-state / diag mt.pt) of a superseded gate attempt; the gate rebuilds it  | surviving: re-run of the gate
rm_rf '/home/nvidia/refcv7_gate_02ffcebfc3/scratch/g_eval_trainer_state.pt'
# 1.17 GB  A  launch-gate G-LIVE smoke checkpoint of a SUPERSEDED/FAILED gate attempt; metrics/summary beside it are kept  | surviving: re-run of the gate
rm_rf '/home/nvidia/refcv7_gate_02ffcebfc3/smoke/20260927T194710/ckpt.pt'
# 403 MB  A  gate scratch (G-EVAL trainer-state / diag mt.pt) of a superseded gate attempt; the gate rebuilds it  | surviving: re-run of the gate
rm_rf '/home/nvidia/refcv7_gate_14907a0a5a/diag_scratch/mt.pt'
# 1.17 GB  A  launch-gate G-LIVE smoke checkpoint of a SUPERSEDED/FAILED gate attempt; metrics/summary beside it are kept  | surviving: re-run of the gate
rm_rf '/home/nvidia/refcv7_gate_14907a0a5a/smoke/20260927T203203/ckpt.pt'
# 403 MB  A  gate scratch (G-EVAL trainer-state / diag mt.pt) of a superseded gate attempt; the gate rebuilds it  | surviving: re-run of the gate
rm_rf '/home/nvidia/refcv7_gate_53ecf9b806/scratch/g_eval_trainer_state.pt'
# 1.17 GB  A  launch-gate G-LIVE smoke checkpoint of a SUPERSEDED/FAILED gate attempt; metrics/summary beside it are kept  | surviving: re-run of the gate
rm_rf '/home/nvidia/refcv7_gate_53ecf9b806/smoke/20260927T190203/ckpt.pt'
# 403 MB  A  gate scratch (G-EVAL trainer-state / diag mt.pt) of a superseded gate attempt; the gate rebuilds it  | surviving: re-run of the gate
rm_rf '/home/nvidia/refcv7_gate_6fa5e8bba4/scratch/g_eval_trainer_state.pt'
# 1.17 GB  A  launch-gate G-LIVE smoke checkpoint of a SUPERSEDED/FAILED gate attempt; metrics/summary beside it are kept  | surviving: re-run of the gate
rm_rf '/home/nvidia/refcv7_gate_6fa5e8bba4/smoke/20260927T180523/ckpt.pt'
# 1.17 GB  A  refcv7 argv-probe smoke checkpoint; train.log/probe_result.json are kept  | surviving: re-run of the probe
rm_rf '/home/nvidia/refcv7_probe/run/ckpt.pt'

# ---------------------------------------------------------------------------
# 3. LAUNCH-GATE DEV-ITERATION EVAL OUTPUT
# ---------------------------------------------------------------------------
# 405 MB  A  launch-gate dev-iteration eval output (387 MB), regenerated by launch_gate.py; the PASS evidence is in refcv7_gate_fec3a0dccf (kept)  | surviving: re-run of launch_gate.py
rm_rf '/home/nvidia/gate_lg_0415/ev/arms_r6_eval'
# 405 MB  A  launch-gate dev-iteration eval output (387 MB), regenerated by launch_gate.py; the PASS evidence is in refcv7_gate_fec3a0dccf (kept)  | surviving: re-run of launch_gate.py
rm_rf '/home/nvidia/gate_lg_2317/out/arms_r6_eval'

# ---------------------------------------------------------------------------
# 4. CODE TARS (git-archive output; commit proven on origin, or stream-hashed against git)
# ---------------------------------------------------------------------------
# 393 MB  A  git-archive tar; pax header commit a756e81522ba is an ancestor of origin/agent/arch-inf-20260803 (it is the only content of bha_2325/code)  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/bha_2325/code/bha_tree_a756e81.tar'
# 113 MB  A  git-archive tar; pax header commit is on origin/agent/arch-inf-20260803  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/bx_0252/tip_68e8ec6_stack.tar'
# 393 MB  A  git-archive tar; pax header commit is on origin/agent/arch-inf-20260803  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/bx_0412/tip.tar'
# 393 MB  A  git-archive tar; pax header commit is on origin/agent/arch-inf-20260803  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/bx_0524/tip_cef9709.tar'
# 394 MB  A  git-archive tar; pax header commit is on origin/agent/arch-inf-20260803  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/bx_1600/tip_2ac0bfb.tar'
# 394 MB  A  git-archive tar; pax header commit is on origin/agent/arch-inf-20260803  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/bx_gate_1136/tip_2374cd2.tar'
# 394 MB  A  git-archive tar; pax header commit is on origin/agent/arch-inf-20260803  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/bx_gate_1251/tip_1b8170e.tar'
# 393 MB  A  git-archive tar; pax header commit is on origin/agent/arch-inf-20260803  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_b2_0201/tree_tip.tar'
# 393 MB  A  git-archive tar; pax header commit is on origin/agent/arch-inf-20260803  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_b3_0258/tree_tip.tar'
# 395 MB  A  git-archive tar; pax header commit is on origin/agent/arch-inf-20260803  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_bxa17_1626/tree_tip.tar'
# 395 MB  A  git-archive tar; pax header commit is on origin/agent/arch-inf-20260803  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_bxa2_1252/tree_tip.tar'
# 395 MB  A  git-archive tar; pax header commit is on origin/agent/arch-inf-20260803  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_bxa_1156/tree_tip.tar'
# 395 MB  A  git-archive tar; pax header commit is on origin/agent/arch-inf-20260803  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_c3_2200/tree_tip.tar'
# 395 MB  A  git-archive tar; pax header commit is on origin/agent/arch-inf-20260803  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_c4_2245/tree_tip.tar'
# 395 MB  A  git-archive tar; pax header commit is on origin/agent/arch-inf-20260803  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_c5_2355/tree_tip.tar'
# 395 MB  A  git-archive tar; pax header commit is on origin/agent/arch-inf-20260803  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_comb_1735/tree_tip.tar'
# 395 MB  A  git-archive tar; pax header commit is on origin/agent/arch-inf-20260803  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_final_1855/tree_tip.tar'
# 393 MB  A  git-archive tar; pax header commit is on origin/agent/arch-inf-20260803  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_fix_2224/tree_2c510fb.tar'
# 395 MB  A  git-archive tar; pax header commit is on origin/agent/arch-inf-20260803  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_glive_2110/tree_tip.tar'
# 394 MB  A  git-archive tar; pax header commit is on origin/agent/arch-inf-20260803  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_lg_0353/t49f10b4.tar'
# 394 MB  A  git-archive tar; pax header commit is on origin/agent/arch-inf-20260803  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_lg_0415/t3d6ff9a.tar'
# 394 MB  A  git-archive tar; pax header commit is on origin/agent/arch-inf-20260803  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_lg_0504/tc178cb1.tar'
# 394 MB  A  git-archive tar; pax header commit is on origin/agent/arch-inf-20260803  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_lg_0508/t3cca805.tar'
# 394 MB  A  git-archive tar; pax header commit is on origin/agent/arch-inf-20260803  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_lg_0518/t3cca805.tar'
# 394 MB  A  git-archive tar; pax header commit is on origin/agent/arch-inf-20260803  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_lg_0928/t3cca805.tar'
# 394 MB  A  git-archive tar; pax header commit is on origin/agent/arch-inf-20260803  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_lg_0944/t3cca805.tar'
# 394 MB  A  git-archive tar; pax header commit is on origin/agent/arch-inf-20260803  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_lg_0951/tree_tip.tar'
# 393 MB  A  git-archive tar; pax header commit is on origin/agent/arch-inf-20260803  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_lg_2317/ab436ee.tar'
# 394 MB  A  git-archive tar; pax header commit is on origin/agent/arch-inf-20260803  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_lgt_1033/tree_tip.tar'
# 394 MB  A  git-archive tar; pax header commit is on origin/agent/arch-inf-20260803  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_nb2_0413/tree_tip.tar'
# 395 MB  A  git-archive tar; pax header commit is on origin/agent/arch-inf-20260803  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_nb2r2_1037/tree_tip.tar'
# 395 MB  A  git-archive tar; pax header commit is on origin/agent/arch-inf-20260803  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_nb2r3_1149/tree_tip.tar'
# 394 MB  A  git-archive tar; pax header commit is on origin/agent/arch-inf-20260803  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_nb2t_0502/tree_tip.tar'
# 393 MB  A  git-archive tar; pax header commit is on origin/agent/arch-inf-20260803  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_new1_2259/tree_tip.tar'
# 393 MB  A  git-archive tar; pax header commit is on origin/agent/arch-inf-20260803  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_new1r2_0048/tree_tip.tar'
# 397 MB  A  wrapper tar = exactly {tree_tip.tar, overlay.tar, plan_thor.json}; all three sit beside it with identical sizes  | surviving: gate_r4_0928/{tree_tip.tar,overlay.tar,plan_thor.json}
rm_rf '/home/nvidia/gate_r4_0928/gate_r4.tar'
# 395 MB  A  git-archive tar; pax header commit is on origin/agent/arch-inf-20260803  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_r4_0928/tree_tip.tar'
# 394 MB  A  candidate tar; 9 WIP members NOT in git; same 9 files salvaged from gmo_early_0327/code  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803 + salvage tgz
rm_rf_salv '/home/nvidia/gmo_early_0327/cand_b4a5.tar' 'gmo_early_0327__code.tgz' 'gmo_early_0327__code_t1.tgz'
# 393 MB  A  candidate tar; stream-hashed: all 2,809 members are git blobs reachable from origin (0 non-git members)  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/nb2_gate_4797ffb/cand_4797.tar'
# 396 MB  A  candidate tar; stream-hashed: all members are git blobs reachable from origin (0 non-git members)  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/nb2r2_cef9/cand_r2.tar'
# 397 MB  A  candidate tar; stream-hashed: all members are git blobs reachable from origin (0 non-git members)  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/nb2r3_2374/cand_r3.tar'
# 397 MB  A  candidate tar; 2 members (map_hires_overfit.py + its test) are NOT in git; identical files are in nb2r4_1b81/code and are salvaged  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803 + salvage tgz
rm_rf_salv '/home/nvidia/nb2r4_1b81/cand_r4.tar' 'nb2r4_1b81__code.tgz'
# 399 MB  A  candidate tar; stream-hashed: all members are git blobs reachable from origin (0 non-git members)  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/nb2r5_2ac0/cand_r5.tar'
# 395 MB  A  git-archive tar; pax commit on origin; extracted dir twin compared byte-exact (tar --compare: 0 content diffs)  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/refcv7_run/02ffcebfc3.tar'
# 395 MB  A  git-archive tar; pax commit on origin; extracted dir twin compared byte-exact (tar --compare: 0 content diffs)  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/refcv7_run/0fcb1c1cbb.tar'
# 395 MB  A  git-archive tar; pax commit on origin; extracted dir twin compared byte-exact (tar --compare: 0 content diffs)  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/refcv7_run/14907a0a5a.tar'
# 395 MB  A  git-archive tar; pax commit on origin; extracted dir twin compared byte-exact (tar --compare: 0 content diffs)  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/refcv7_run/53ecf9b806.tar'
# 395 MB  A  git-archive tar; pax commit on origin; extracted dir twin compared byte-exact (tar --compare: 0 content diffs)  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/refcv7_run/60f4c068a8.tar'
# 395 MB  A  git-archive tar; pax commit on origin; extracted dir twin compared byte-exact (tar --compare: 0 content diffs)  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/refcv7_run/6fa5e8bba4.tar'
# 395 MB  A  git-archive tar of the KEPT launch tree; pax header commit fec3a0dccfda (on origin); its extracted dir is kept  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/refcv7_run/fec3a0dccf.tar'

# ---------------------------------------------------------------------------
# 5. EXTRACTED CODE TREES (every file proven identical to a git blob on origin; WIP residue salvaged in step 0)
# ---------------------------------------------------------------------------
# 398 MB  A  code tree (2835 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/bha_2325/tree'
# 119 MB  A  code tree (1681 files); 2 WIP file(s) (0.03 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block)  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803 + salvage tgz
rm_rf_salv '/home/nvidia/bx_0252/tree' 'bx_0252__tree.tgz'
# 402 MB  A  code tree (2815 files); 13 WIP file(s) (1.18 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block), compared after CRLF->LF normalisation  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803 + salvage tgz
rm_rf_salv '/home/nvidia/bx_0412/tree' 'bx_0412__tree.tgz'
# 402 MB  A  code tree (2817 files); 14 WIP file(s) (1.19 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block)  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803 + salvage tgz
rm_rf_salv '/home/nvidia/bx_0412/tree_r6' 'bx_0412__tree_r6.tgz'
# 416 MB  A  code tree (2825 files); 10 WIP file(s) (1.17 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block)  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803 + salvage tgz
rm_rf_salv '/home/nvidia/bx_0524/tree_main' 'bx_0524__tree_main.tgz'
# 416 MB  A  code tree (2827 files); 10 WIP file(s) (1.18 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block)  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803 + salvage tgz
rm_rf_salv '/home/nvidia/bx_0524/tree_r6' 'bx_0524__tree_r6.tgz'
# 415 MB  A  code tree (2811 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/bx_0524/tree_tip'
# 418 MB  A  code tree (2835 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/bx_1600/tree_a17'
# 418 MB  A  code tree (2835 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/bx_1600/tree_tip'
# 418 MB  A  code tree (2835 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/bx_1600/tree_toy'
# 403 MB  A  code tree (2829 files); 3 WIP file(s) (0.96 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block)  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803 + salvage tgz
rm_rf_salv '/home/nvidia/bx_anch_1116/tree' 'bx_anch_1116__tree.tgz'
# 403 MB  A  code tree (2827 files); 8 WIP file(s) (1.16 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block)  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803 + salvage tgz
rm_rf_salv '/home/nvidia/bx_cmain_1014/tree' 'bx_cmain_1014__tree.tgz'
# 418 MB  A  code tree (2835 files); 3 WIP file(s) (0.96 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block)  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803 + salvage tgz
rm_rf_salv '/home/nvidia/bx_gate_1136/tree_anch' 'bx_gate_1136__tree_anch.tgz'
# 417 MB  A  code tree (2819 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/bx_gate_1136/tree_tip'
# 418 MB  A  code tree (2835 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/bx_gate_1251/tree_anch'
# 417 MB  A  code tree (2819 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/bx_gate_1251/tree_tip'
# 416 MB  A  code tree (2825 files); 8 WIP file(s) (1.14 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block)  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803 + salvage tgz
rm_rf_salv '/home/nvidia/bx_tests_0934/tree_main' 'bx_tests_0934__tree_main.tgz'
# 416 MB  A  code tree (2827 files); 9 WIP file(s) (1.17 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block)  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803 + salvage tgz
rm_rf_salv '/home/nvidia/bx_tests_0934/tree_r6' 'bx_tests_0934__tree_r6.tgz'
# 442 MB  A  code tree (2839 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_b2_0201/CAND'
# 441 MB  A  code tree (2837 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_b2_0201/TIP'
# 442 MB  A  code tree (2840 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_b3_0258/CAND'
# 442 MB  A  code tree (2839 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_b3_0258/TIP'
# 446 MB  A  code tree (2884 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_bxa17_1626/CAND'
# 446 MB  A  code tree (2884 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_bxa17_1626/TIP'
# 446 MB  A  code tree (2884 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_bxa2_1252/CAND'
# 445 MB  A  code tree (2868 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_bxa2_1252/TIP'
# 446 MB  A  code tree (2884 files); all but 3 overlay file(s) are git blobs; those 3 are kept in gate_bxa_1156/overlay  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803 + kept overlay/
rm_rf '/home/nvidia/gate_bxa_1156/CAND'
# 445 MB  A  code tree (2868 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_bxa_1156/TIP'
# 447 MB  A  code tree (2891 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_c3_2200/CAND'
# 447 MB  A  code tree (2891 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_c3_2200/TIP'
# 447 MB  A  code tree (2891 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_c4_2245/CAND'
# 447 MB  A  code tree (2891 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_c4_2245/TIP'
# 447 MB  A  code tree (2891 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_c5_2355/CAND'
# 447 MB  A  code tree (2891 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_c5_2355/TIP'
# 447 MB  A  code tree (2887 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_comb_1735/CAND'
# 446 MB  A  code tree (2884 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_comb_1735/TIP'
# 447 MB  A  code tree (2890 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_final_1855/CAND'
# 447 MB  A  code tree (2887 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_final_1855/TIP'
# 415 MB  A  code tree (2835 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_fix_2224/FIX'
# 415 MB  A  code tree (2827 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_fix_2224/TIP'
# 403 MB  A  code tree (2835 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_fix_2224/TIPRED'
# 446 MB  A  code tree (2891 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_glive_2110/CAND'
# 446 MB  A  code tree (2891 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_glive_2110/TIP'
# 400 MB  A  code tree (2852 files); all but 6 overlay file(s) are git blobs; those 6 are kept in gate_lg_0353/overlay  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803 + kept overlay/
rm_rf '/home/nvidia/gate_lg_0353/cab'
# 399 MB  A  code tree (2845 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_lg_0353/tip'
# 399 MB  A  code tree (2854 files); all but 5 overlay file(s) are git blobs; those 5 are kept in gate_lg_0415/overlay  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803 + kept overlay/
rm_rf '/home/nvidia/gate_lg_0415/cab'
# 399 MB  A  code tree (2847 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_lg_0415/tip'
# 399 MB  A  code tree (2855 files); all but 5 overlay file(s) are git blobs; those 5 are kept in gate_lg_0504/overlay  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803 + kept overlay/
rm_rf '/home/nvidia/gate_lg_0504/cab'
# 399 MB  A  code tree (2847 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_lg_0504/tip'
# 400 MB  A  code tree (2867 files); all but 4 overlay file(s) are git blobs; those 4 are kept in gate_lg_0508/overlay  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803 + kept overlay/
rm_rf '/home/nvidia/gate_lg_0508/cab'
# 399 MB  A  code tree (2859 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_lg_0508/tip'
# 400 MB  A  code tree (2867 files); all but 3 overlay file(s) are git blobs; those 3 are kept in gate_lg_0518/overlay  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803 + kept overlay/
rm_rf '/home/nvidia/gate_lg_0518/cab'
# 399 MB  A  code tree (2859 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_lg_0518/tip'
# 400 MB  A  code tree (2867 files); all but 2 overlay file(s) are git blobs; those 2 are kept in gate_lg_0928/overlay  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803 + kept overlay/
rm_rf '/home/nvidia/gate_lg_0928/cab'
# 399 MB  A  code tree (2859 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_lg_0928/tip'
# 400 MB  A  code tree (2867 files); all but 1 overlay file(s) are git blobs; those 1 are kept in gate_lg_0944/overlay  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803 + kept overlay/
rm_rf '/home/nvidia/gate_lg_0944/cab'
# 399 MB  A  code tree (2859 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_lg_0944/tip'
# 445 MB  A  code tree (2868 files); all but 1 overlay file(s) are git blobs; those 1 are kept in gate_lg_0951/overlay  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803 + kept overlay/
rm_rf '/home/nvidia/gate_lg_0951/CAND'
# 443 MB  A  code tree (2859 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_lg_0951/TIP'
# 398 MB  A  code tree (2841 files); all but 4 overlay file(s) are git blobs; those 4 are kept in gate_lg_2317/overlay  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803 + kept overlay/
rm_rf '/home/nvidia/gate_lg_2317/cab'
# 398 MB  A  code tree (2836 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_lg_2317/tip'
# 401 MB  A  code tree (2869 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_lgt_1033/CAND'
# 399 MB  A  code tree (2860 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_lgt_1033/TIP'
# 443 MB  A  code tree (2857 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_nb2_0413/CAND'
# 442 MB  A  code tree (2845 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_nb2_0413/TIP'
# 445 MB  A  code tree (2868 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_nb2r2_1037/CAND'
# 445 MB  A  code tree (2868 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_nb2r2_1037/TIP'
# 445 MB  A  code tree (2868 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_nb2r3_1149/CAND'
# 445 MB  A  code tree (2868 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_nb2r3_1149/TIP'
# 404 MB  A  code tree (2859 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_nb2t_0502/CAND'
# 402 MB  A  code tree (2847 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_nb2t_0502/TIP'
# 441 MB  A  code tree (2837 files); all but 3 overlay file(s) are git blobs; those 3 are kept in gate_new1_2259/overlay  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803 + kept overlay/
rm_rf '/home/nvidia/gate_new1_2259/NEW1'
# 441 MB  A  code tree (2835 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_new1_2259/TIP'
# 441 MB  A  code tree (2837 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_new1r2_0048/CAND'
# 441 MB  A  code tree (2835 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/gate_new1r2_0048/TIP'
# 407 MB  A  code tree (2955 files); 1 WIP file(s) (0.01 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block)  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803 + salvage tgz
rm_rf_salv '/home/nvidia/gmo_bind_pf/tree' 'gmo_bind_pf__tree.tgz'
# 402 MB  A  code tree (2828 files); 9 WIP file(s) (1.35 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block), compared after CRLF->LF normalisation  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803 + salvage tgz
rm_rf_salv '/home/nvidia/gmo_early_0327/code' 'gmo_early_0327__code.tgz'
# 404 MB  A  code tree (2828 files); 6 WIP file(s) (1.27 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block)  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803 + salvage tgz
rm_rf_salv '/home/nvidia/gmo_early_0327/code_t1' 'gmo_early_0327__code_t1.tgz'
# 408 MB  A  code tree (2841 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/nb2_gate_4797ffb/code'
# 409 MB  A  code tree (2956 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/nb2a19_thor/code'
# 409 MB  A  code tree (2955 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/nb2ml2_thor/code'
# 409 MB  A  code tree (2844 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/nb2r2_cef9/code'
# 409 MB  A  code tree (2853 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/nb2r3_2374/code'
# 409 MB  A  code tree (2854 files); 2 WIP file(s) (0.07 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block)  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803 + salvage tgz
rm_rf_salv '/home/nvidia/nb2r4_1b81/code' 'nb2r4_1b81__code.tgz'
# 407 MB  A  code tree (2954 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/nb2r5_2ac0/code'
# 186 MB  A  code tree (2200 files) = commit 284393ccff3e; every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/refcv6_run/284393ccff'
# 185 MB  A  code tree (2201 files) = commit 287d72ece83b; every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/refcv6_run/287d72ece8'
# 187 MB  A  code tree (2196 files) = commit 35be81285ec8; 2 WIP file(s) (0.01 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block), compared after CRLF->LF normalisation  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803 + salvage tgz
rm_rf_salv '/home/nvidia/refcv6_run/35be81285e' 'refcv6_run__35be81285e.tgz'
# 187 MB  A  code tree (2195 files) = commit 4ca616882283; 2 WIP file(s) (0.01 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block), compared after CRLF->LF normalisation  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803 + salvage tgz
rm_rf_salv '/home/nvidia/refcv6_run/4ca6168822' 'refcv6_run__4ca6168822.tgz'
# 325 MB  A  code tree (2670 files) = commit 82c2331a2fcb; every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/refcv6_run/82c2331a2f'
# 187 MB  A  code tree (2196 files); 2 WIP file(s) (0.01 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block), compared after CRLF->LF normalisation  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803 + salvage tgz
rm_rf_salv '/home/nvidia/refcv6_run/pre_C' 'refcv6_run__pre_C.tgz'
# 187 MB  A  code tree (2196 files); 2 WIP file(s) (0.01 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block), compared after CRLF->LF normalisation  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803 + salvage tgz
rm_rf_salv '/home/nvidia/refcv6_run/pre_D' 'refcv6_run__pre_D.tgz'
# 187 MB  A  code tree (2196 files); 3 WIP file(s) (0.61 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block), compared after CRLF->LF normalisation  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803 + salvage tgz
rm_rf_salv '/home/nvidia/refcv6_run/pre_E' 'refcv6_run__pre_E.tgz'
# 187 MB  A  code tree (2196 files); 6 WIP file(s) (0.96 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block), compared after CRLF->LF normalisation  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803 + salvage tgz
rm_rf_salv '/home/nvidia/refcv6_run/pre_F' 'refcv6_run__pre_F.tgz'
# 187 MB  A  code tree (2196 files); 2 WIP file(s) (0.01 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block), compared after CRLF->LF normalisation  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803 + salvage tgz
rm_rf_salv '/home/nvidia/refcv6_run/pre_G' 'refcv6_run__pre_G.tgz'
# 405 MB  A  code tree (2884 files); 1 WIP file(s) (0.01 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block)  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803 + salvage tgz
rm_rf_salv '/home/nvidia/refcv7_bind/tree' 'refcv7_bind__tree.tgz'
# 404 MB  A  code tree (2887 files); 1 WIP file(s) (0.01 MB) are NOT in git -> salvaged to a tgz first (SALVAGE block)  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803 + salvage tgz
rm_rf_salv '/home/nvidia/refcv7_bind/tree_map' 'refcv7_bind__tree_map.tgz'
# 401 MB  A  code tree (2899 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/refcv7_bind_r4/tree'
# 401 MB  A  code tree (2884 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/refcv7_dry_thor/tree'
# 404 MB  A  code tree (2884 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/refcv7_probe/tree'
# 401 MB  A  code tree (2891 files) = commit 02ffcebfc351; every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/refcv7_run/02ffcebfc3'
# 401 MB  A  code tree (2896 files) = commit 0fcb1c1cbb88; every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/refcv7_run/0fcb1c1cbb'
# 405 MB  A  code tree (2891 files) = commit 14907a0a5add; every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/refcv7_run/14907a0a5a'
# 401 MB  A  code tree (2891 files) = commit 53ecf9b80634; every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/refcv7_run/53ecf9b806'
# 401 MB  A  code tree (2890 files) = commit 60f4c068a8b6; every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/refcv7_run/60f4c068a8'
# 401 MB  A  code tree (2891 files) = commit 6fa5e8bba478; every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/refcv7_run/6fa5e8bba4'
# 399 MB  A  code tree (2847 files); every file byte-identical to a git blob reachable from origin  | surviving: git @ github.com/sayoood/TanitAD, branch agent/arch-inf-20260803
rm_rf '/home/nvidia/wgate_0505'

finish
