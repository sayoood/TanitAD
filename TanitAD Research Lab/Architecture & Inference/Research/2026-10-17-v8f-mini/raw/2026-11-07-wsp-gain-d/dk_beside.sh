#!/usr/bin/env bash
# dk_beside (PREREG_WSP_GAIN_D.md section 3, item 2 "distance keeping beside"): the eval tool's plans `pred` [2399,30,2] for C0 and E-GAIN-D at both seeds (T1-class open loop, the eval-141 grid),
# then the pinned dk141.py read: headway, time gap, min-TTC, TTC-violation rate, tight-gap rate, LEAD-VIOL, paired deltas with the episode-cluster bootstrap (B 2,000 seed 0).
# Lowest priority; run by chain_d.sh after READ 2 is written, or alone. Resumable (an existing eval_<tag>.npz is skipped). Outputs: raw/DK141_READ_D.json, raw/DK141_TABLE_D.md, raw/dk141_perwindow_D.npz, logs/dk_*.out.
set -u
export PATH="/usr/bin:/bin:$PATH"
PK=/e/Projects/TanitAD-artifacts/dagland/motion_c/theory/wsp_edit_d
PKW=E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/wsp_edit_d
ART=E:/Projects/TanitAD-artifacts
DATA="$ART/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
SW=F:/TanitAD-scratch/motion_c/wsp_edit_d/dk; S=/f/TanitAD-scratch/motion_c/wsp_edit_d/dk
DK=$ART/dagland/motion_c/dk141/dk141.py
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=${DK_TH:-8} MKL_NUM_THREADS=${DK_TH:-8}
mkdir -p "$S" "$PK/logs"
log() { echo "$(date +%FT%T) $*" | tee -a "$PK/logs/chain_d.out"; }
run_eval() {   # tag ckpt
  local tag=$1 ck=$2
  [ -f "$S/eval_$tag.npz" ] && { log "ZZWSPD-DK-SKIP $tag"; return 0; }
  log "ZZWSPD-DK-EVAL $tag"
  "$PY" -I "$PKW/dk_eval_wrap.py" run --ckpt "$ck" --tag "$tag" --tracks "$ART/sfresh_inputs/inputs/native_tracks.pt" --eval-cache "$ART/sfresh_inputs/cache" \
     --eval-grid "$DATA/eval_grid_windows.json" --eval-kin "$DATA/eval141_kin.npz" --eval-join "$DATA/join_eval141.npz" --out-dir "$SW" --device cpu --batch 64 < /dev/null > "$PK/logs/dk_eval_$tag.out" 2>&1
  local rc=$?; [ $rc -eq 0 ] && [ -f "$S/eval_$tag.npz" ] || { log "ZZWSPD-DK-EVAL-FAIL $tag rc=$rc"; return 1; }
}
nf=0
run_eval C0_s0 "$ART/v8f-recover/M2b_s0_recft_C0/ckpt_last.pt" || nf=$((nf+1))
run_eval C0_s1 "$ART/v8f-recover/M2b_s1_recft_C0/ckpt_last.pt" || nf=$((nf+1))
run_eval D_s0 "$ART/v8f-wspedit-d/M2b_s0_wspedit_d/ckpt_last.pt" || nf=$((nf+1))
run_eval D_s1 "$ART/v8f-wspedit-d/M2b_s1_wspedit_d/ckpt_last.pt" || nf=$((nf+1))
[ $nf -eq 0 ] || { log "ZZWSPD-DK-FAIL $nf evals failed"; exit 3; }
cd "$PK"
"$PY" -I "$DK" read --arm C0_s0="$SW/eval_C0_s0.npz" --arm C0_s1="$SW/eval_C0_s1.npz" --arm D_s0="$SW/eval_D_s0.npz" --arm D_s1="$SW/eval_D_s1.npz" \
   --pair D_s0:C0_s0 --pair D_s1:C0_s1 --pair D_s0:P10 --pair D_s1:P10 --pair C0_s0:P10 --pair C0_s1:P10 \
   --out "$PKW/raw/DK141_READ_D.json" --table-md "$PKW/raw/DK141_TABLE_D.md" --perwindow-out "$PKW/raw/dk141_perwindow_D.npz" > "$PK/logs/dk141_read_d.out" 2>&1
rc=$?; log "ZZWSPD-DK141-READ rc=$rc"; exit $rc
