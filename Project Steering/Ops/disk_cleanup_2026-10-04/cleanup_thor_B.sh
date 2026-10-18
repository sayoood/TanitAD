#!/usr/bin/env bash
# =============================================================================
# cleanup_thor_B.sh  --  TanitAD Thor disk cleanup, CATEGORY B (redundant copies; every deletion is preceded by a hash check)
# Generated 2026-10-04 by the READ-ONLY Thor inventory (see INVENTORY_THOR.md).
# RUN ON THOR as user nvidia (host thor6). Review every line first.
#
#   bash cleanup_thor_B.sh            -> DRY RUN (default): prints what would go, deletes NOTHING
#   DRYRUN=0 bash cleanup_thor_B.sh   -> really deletes
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


# B-specific: every file is re-hashed ON THOR right before it is removed; a mismatch means NO deletion.
# VERIFY=1 (default, also in DRYRUN) reads the data (~36 GB in total, nice/ionice); VERIFY=0 skips the hashing (not recommended).
VERIFY="${VERIFY:-1}"

rm_if_hash() {   # rm_if_hash ALGO EXPECTED_HASH PATH
  local algo="$1" want="$2" p="$3" got=""
  guard "$p" || { REFUSED=$((REFUSED+1)); return 0; }
  [ -f "$p" ] || { echo "ABSENT      $p"; return 0; }
  if [ "$VERIFY" = 1 ]; then
    case "$algo" in
      md5)    got="$("${LOWPRIO[@]}" md5sum -- "$p" | cut -d' ' -f1)";;
      sha256) got="$("${LOWPRIO[@]}" sha256sum -- "$p" | cut -d' ' -f1)";;
    esac
    if [ "$got" != "$want" ]; then echo "HASH-MISMATCH (not deleted): $p  got=$got want=$want"; REFUSED=$((REFUSED+1)); return 0; fi
    echo "verified $algo $want  $p"
  fi
  rm_rf "$p"
}

rm_dir_if_same_file() {   # rm_dir_if_same_file DIR FILE_INSIDE_DIR OTHER_FILE : delete DIR only if FILE and OTHER_FILE have equal sha256
  local d="$1" f="$2" o="$3" a b
  guard "$d" || { REFUSED=$((REFUSED+1)); return 0; }
  [ -d "$d" ] || { echo "ABSENT      $d"; return 0; }
  [ -f "$o" ] || { echo "SURVIVING COPY MISSING ($o) -> not deleting $d"; REFUSED=$((REFUSED+1)); return 0; }
  f="$(ls -- $f 2>/dev/null | head -1)"; [ -f "$f" ] || { echo "no file to compare in $d"; REFUSED=$((REFUSED+1)); return 0; }
  a="$("${LOWPRIO[@]}" sha256sum -- "$f" | cut -d' ' -f1)"; b="$("${LOWPRIO[@]}" sha256sum -- "$o" | cut -d' ' -f1)"
  if [ -z "$a" ] || [ "$a" != "$b" ]; then echo "SHA256 DIFFERS (not deleted): $d"; REFUSED=$((REFUSED+1)); return 0; fi
  echo "verified sha256 $a  $f == $o"
  rm_rf "$d"
}

rm_manifest() {   # rm_manifest MANIFEST : `sha256sum -c` over the whole list, delete the listed files only if ALL verify
  local m="$1" out rc n bytes shape_bad left
  [ -f "$m" ] || { echo "manifest $m not found next to the script"; REFUSED=$((REFUSED+1)); return 0; }
  n="$(wc -l < "$m")"
  # the two directories the manifests may touch must pass the guard, and every line must have the exact expected shape
  guard "/home/nvidia/sam3map/corpus/out/x.npz" >/dev/null 2>&1 && guard "/home/nvidia/data/sam3_corpus/semantic_maps/gt/x.npz" >/dev/null 2>&1 \
    || { echo "guard refuses the manifest directories"; REFUSED=$((REFUSED+1)); return 0; }
  shape_bad="$(grep -v -c -E '^[0-9a-f]{64}  /home/nvidia/(sam3map/corpus/out/[0-9a-f]{12}\.(sam3mapgt|worldmap)\.npz|data/sam3_corpus/semantic_maps/gt/[0-9a-f-]{36}\.sam3mapgt\.npz)$' "$m")"
  if [ "$shape_bad" != "0" ]; then echo "manifest $m has $shape_bad line(s) of unexpected shape; refusing"; REFUSED=$((REFUSED+1)); return 0; fi
  if [ "$VERIFY" = 1 ]; then
    out="$("${LOWPRIO[@]}" sha256sum -c --quiet "$m" 2>&1)"; rc=$?
    if [ "$rc" -ne 0 ] || [ -n "$out" ]; then echo "MANIFEST VERIFY FAILED (rc=$rc) -> nothing deleted for $m"; echo "$out" | head -5; REFUSED=$((REFUSED+1)); return 0; fi
    echo "verified sha256 of $n files in $m"
  fi
  bytes="$(sed -E 's/^[0-9a-f]{64}  //' "$m" | tr '\n' '\0' | du -cb --files0-from=- 2>/dev/null | tail -1 | cut -f1)"
  TOTAL=$((TOTAL+${bytes:-0})); NDEL=$((NDEL+n))
  printf '%-9s %9s  %s files listed in %s\n' "$([ "$DRYRUN" = 1 ] && echo WOULD-RM || echo RM)" "$(human "${bytes:-0}")" "$n" "$m"
  if [ "$DRYRUN" != 1 ]; then
    sed -E 's/^[0-9a-f]{64}  //' "$m" | tr '\n' '\0' | "${LOWPRIO[@]}" xargs -0 rm -f --
    left="$(sed -E 's/^[0-9a-f]{64}  //' "$m" | tr '\n' '\0' | xargs -0 ls -d 2>/dev/null | wc -l)"
    [ "$left" = 0 ] || { echo "FAILED-TO-REMOVE $left file(s) of $m" >&2; REFUSED=$((REFUSED+1)); }
  fi
}

preflight


# ---------------------------------------------------------------------------
# B items. Surviving copies named in each comment were verified from the dev box on 2026-10-04.
# D: is 100 percent full: after this script the D: copy is the only local one for the refcv6 ckpts (HF holds ckpt_step38000).
# ---------------------------------------------------------------------------
# 404 MB  B  refcv6 milestone ckpt; Thor md5 == D: md5 (both computed 2026-10-04)  | surviving: D:/refcv6_eval_kit/ckpt/ckpt_5000.pt
rm_if_hash md5 8a1e4da0dc6561353b28e3947146751e '/home/nvidia/refcv6_run/runs/refcv6-r101-s0/ckpt_5000.pt'
# 404 MB  B  refcv6 milestone ckpt; Thor md5 == D: md5 (both computed 2026-10-04)  | surviving: D:/refcv6_eval_kit/ckpt/ckpt_15000.pt
rm_if_hash md5 46eb21857234a9dccf350c6cfd0ddee8 '/home/nvidia/refcv6_run/runs/refcv6-r101-s0/ckpt_15000.pt'
# 404 MB  B  refcv6 milestone ckpt; Thor md5 == D: md5 (both computed 2026-10-04)  | surviving: D:/refcv6_eval_kit/ckpt/ckpt_20000.pt
rm_if_hash md5 f4320c8971dcc88b895a35bcc9f35c76 '/home/nvidia/refcv6_run/runs/refcv6-r101-s0/ckpt_20000.pt'
# 404 MB  B  refcv6 milestone ckpt; Thor md5 == D: md5 (both computed 2026-10-04)  | surviving: D:/refcv6_eval_kit/ckpt/ckpt_30000.pt
rm_if_hash md5 0c5c67b3ecf28495d54348e3436371bf '/home/nvidia/refcv6_run/runs/refcv6-r101-s0/ckpt_30000.pt'
# 1.18 GB  B  refcv6 rolling ckpt at the PI stop (step 38000); Thor md5 == D: md5 == STOPPED_BY_PI.json; Thor sha256 de34332d.. == HF LFS sha256  | surviving: D:/refcv6_eval_kit/ckpt/ckpt_step38000_stopped.pt + D:/refcv6_eval_kit/ckpt_final/ckpt_step38000.pt + HF Sayood/tanitad-refcv6-r101-s0/ckpt_step38000.pt
rm_if_hash md5 5a2e7222a9f5f8c7aa7bf38ef4698d8a '/home/nvidia/refcv6_run/runs/refcv6-r101-s0/ckpt.pt'
# 1.18 GB  B  refcv6 step-35000 rolling snapshot; Thor md5 == D: md5 == snapshot MD5SUMS  | surviving: D:/refcv6_eval_kit/ckpt/ckpt_35000.pt
rm_if_hash md5 68a4ef3bbb0a3ec707b04f0fb2b6b86f '/home/nvidia/refcv6_run/snapshots/refcv6-r101-s0/ckpt_step35000.pt'
# 1.29 GB  B  refcv4b FINAL ckpt; Thor sha256 == HF LFS sha256 (private repo)  | surviving: HF Sayood/tanitad-refc-v4b/ckpt_40284_FINAL.pt
rm_if_hash sha256 090ee8d2584acc81e3738ab28f8f5bf2ed6bfb0d0cc6fba887de2a0cbddcbaa4 '/home/nvidia/refcv4b/ckpt_40284_FINAL.pt'
# 1.25 GB  B  refc-base 30k ckpt; Thor sha256 == HF LFS sha256 (public repo)  | surviving: HF Sayood/tanitad-refc-base/ckpt.pt
rm_if_hash sha256 6b7002c42fab31eb187818a0e26aa5ea469613416fccb1877bee735151f67f32 '/home/nvidia/tmp/refc-base-30k/ckpt.pt'
# 429 MB  B  refcv3 ckpt; Thor sha256 == HF LFS sha256  | surviving: HF Sayood/tanitad-refc-v3/ckpt_40284.pt
rm_if_hash sha256 5a163eaf57674e52ff4846ff4e460563884d596af9639d92005ca3c3e3d04203 '/home/nvidia/models/refcv3-b1-v72-40284/ckpt.pt'
# 1.97 GB  B  B1 egomotion bundle tar; Thor sha256 == HF LFS sha256 (private corpus repo)  | surviving: HF Sayood/tanitad-v7-training-corpus:egomotion/egomotion_alpamayo.tar
rm_if_hash sha256 d38592bed40cec2a562ee3228d7b632d787f6f84557248e4898d186cf760bd8f '/home/nvidia/data/b1-bundle/egomotion/egomotion_alpamayo.tar'
# 3.45 GB  B  HF-cache copy of facebook/sam3 (3.45 GB = 3.29 GiB); its sam3.pt sha256 9999e234.. is IDENTICAL to tanit-astra-research/models/sam3/sam3.pt  | surviving: /home/nvidia/tanit-astra-research/models/sam3/sam3.pt (same sha256) and the gated HF repo facebook/sam3
rm_dir_if_same_file '/home/nvidia/.cache/huggingface/hub/models--facebook--sam3' '/home/nvidia/.cache/huggingface/hub/models--facebook--sam3/snapshots/*/sam3.pt' '/home/nvidia/tanit-astra-research/models/sam3/sam3.pt'
# 14.37 GB  B  SAM3 map products of 4,677 clips (gt 4,569 + gt_flagged 108 + worldmap 4,677 = 9,354 files, 14.37 GB): every file's sha256 equals the publisher ledger entry (MEASURED on Thor 2026-10-04: 9,354 ok / 0 bad / 0 missing); the ledger records 'sha256 verified on the Hub'; live Hub sample 240/240 sha256+bytes match (2 had been re-tiered gt_flagged -> gt)  | surviving: HF dataset Sayood/tanitad-v7-training-corpus (private): semantic_maps/{gt,gt_flagged,worldmap}/
rm_manifest "$HERE/B_sha256_sam3map_corpus_out.txt"
# 9.98 GB  B  verbatim HF pull (_pull_record.json: revision 355f4adf, 4,652 files); all 4,652 sha256 match the publisher ledger (MEASURED on Thor 2026-10-04). It is the input of the sam3_gt_v3 export, which is KEPT  | surviving: HF dataset Sayood/tanitad-v7-training-corpus @355f4adf (private): semantic_maps/gt/
rm_manifest "$HERE/B_sha256_sam3_corpus_gt.txt"

finish
