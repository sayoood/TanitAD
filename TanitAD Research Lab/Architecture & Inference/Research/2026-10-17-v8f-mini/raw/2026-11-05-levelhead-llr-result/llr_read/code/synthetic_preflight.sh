#!/usr/bin/env bash
# synthetic_preflight.sh -- the chain's gate-4 pre-flight (6 windows, the chain's own tool call) on the SYNTHETIC LLR-shaped checkpoint: proves an LLR-shaped checkpoint (token wiring, lead head on) loads and rolls in the LLR read tree
# under the three scene families.  No deletion: each run writes a fresh dir named by the clock.  usage: bash code/synthetic_preflight.sh
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
ART=E:/Projects/TanitAD-artifacts
PK=/e/Projects/TanitAD-artifacts/dagland/motion_c/llr_read
DATA="$ART/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
TOOL=F:/TanitAD-scratch/motion_c/llr_read/tree/stack/scripts/v8f_t1_eval_ri.py
CK=F:/TanitAD-scratch/motion_c/llr_read/pt/synth/M2b_s0_recft_LLR/ckpt_last.pt
CARD_LINE="PREREG_LEVELHEAD_A2.md sha256 $(sha256sum $PK/../a2_prereg/PREREG_LEVELHEAD_A2.md | cut -c1-64)"
T=$(date +%H%M%S)
go() {  # name threads flags...
  local n=$1 th=$2; shift 2; local OD=F:/TanitAD-scratch/motion_c/llr_read/smoke/synth_${n}_$T
  mkdir -p "/f/TanitAD-scratch/motion_c/llr_read/smoke/synth_${n}_$T"
  OMP_NUM_THREADS=$th MKL_NUM_THREADS=$th CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 "$PY" -I "$TOOL" run --ckpt "$CK" --tag SYN_$n --tracks "$ART/sfresh_inputs/inputs/native_tracks.pt" --eval-cache "$ART/sfresh_inputs/cache" \
    --eval-grid "$DATA/eval_grid_windows.json" --eval-join "$DATA/join_eval141.npz" --out-dir "$OD" --replan-steps 10 --nav deadreckon --schedule periodic --limit causal --dump-ticks "$OD/ticks" --dump-heads --device cpu \
    --max-windows 6 --smooth sg72 "$@" < /dev/null > "$PK/logs/synthetic_preflight_$n.out" 2>&1
  echo "$n rc=$? ticks=$(ls /f/TanitAD-scratch/motion_c/llr_read/smoke/synth_${n}_$T/ticks 2>/dev/null | tr '\n' ' ')"
}
go ri_kbox 12 --scene reindexed --feasible kbox
go hold 16 --scene hold_now --freeze-card-ref "$CARD_LINE"
go logged 16 --scene logged
