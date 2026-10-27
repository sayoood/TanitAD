#!/bin/bash
# rollout_lane.sh <seed>: PREREG_V8F_DAG_HACC section 2 rollouts for one seed, SEQUENTIAL.
#   (1) DECISIVE  DAGHACC_s<S>_m10_causal_heads        --scene logged --limit causal   (DAG's decisive settings; the only 2 rollouts the PREREG names, one per seed)
#   (2) beside    DAGHACCSTRICT_s<S>_m10_frozen_causal_heads  --scene frozen --limit causal   (STRICT, PREREG "beside")
#   (3) beside    DAGHACC_s<S>_m10_logged_heads        --scene logged --limit logged   (DECLARATIONS D6: so the unchanged t1kf reader, which consumes three dump configs, can run)
# Exactly DAG's decisive argv (E:/Projects/TanitAD-artifacts/dagt1/logs/chain.out "start DAG_s<S>_m10_causal_heads") and the same tool tree (digest 6a4b47c8...):
#   only --ckpt and --tag change; the two OUTPUT-PATH flags (--out-dir, --dump-ticks) point into this package. Environment copied from dag_t1_chain.sh (OMP/MKL 8 threads, 2 lanes = 2 processes).
S=$1
PKG=/e/Projects/TanitAD-artifacts/daghacc
ART=E:/Projects/TanitAD-artifacts
TREE=$ART/dagt1_tree_b43c7a9a
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
export TANITEVAL_STACK_OVERRIDE=$(cygpath -w "$TREE/stack")
export TMP=$PKG/pt_tmp TEMP=$PKG/pt_tmp
GRID="$TREE/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/eval_grid_windows.json"
JOIN="$TREE/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz"
OUT=$ART/daghacc/t1
mkdir -p $PKG/t1/ticks $PKG/argv $PKG/logs
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $PKG/logs/lane$S.log; }
CK=$ART/daghacc/ckpt/M2b_s${S}_dag_hacc.pt
EXP=$(grep -o '"out_sha256": "[0-9a-f]*"' $PKG/ckpt/BAKE_REPORT_seed$S.json | head -1 | cut -d'"' -f4)
runone() {  # runone <tag> <scene> <limit>
  local tag=$1 scene=$2 limit=$3 t0 rc b a
  b=$(sha256sum < "$(cygpath -u "$CK")" | cut -c1-64)
  if [ "$b" != "$EXP" ]; then say "STOP $tag: ckpt sha256 before = $b, expected $EXP"; return 1; fi
  local argv=("$PY" "$TREE/stack/scripts/v8f_t1_eval.py" run --ckpt "$CK" --tag "$tag" --tracks "$ART/sfresh_inputs/inputs/native_tracks.pt"
    --eval-cache "$ART/sfresh_inputs/cache" --eval-grid "$GRID" --eval-join "$JOIN" --out-dir "$OUT"
    --replan-steps 10 --nav deadreckon --scene "$scene" --schedule periodic --limit "$limit" --dump-ticks "$OUT/ticks" --dump-heads --device cpu)
  printf '%s\n' "${argv[@]}" > $PKG/argv/$tag.argv
  say "start $tag (scene $scene limit $limit) ckpt sha ${b:0:12}"
  t0=$(date +%s)
  "${argv[@]}" > $PKG/logs/$tag.out 2>&1 < /dev/null; rc=$?
  a=$(sha256sum < "$(cygpath -u "$CK")" | cut -c1-64)
  printf '{"tag": "%s", "scene": "%s", "limit": "%s", "rc": %s, "ckpt": "%s", "ckpt_sha256_before": "%s", "ckpt_sha256_after": "%s", "wall_s": %s, "n_dumps": %s}\n' \
    "$tag" "$scene" "$limit" "$rc" "$CK" "$b" "$a" "$(( $(date +%s) - t0 ))" "$(ls $PKG/t1/dump_$tag 2>/dev/null | grep -c '^ep.*\.npz$')" > $PKG/t1/done_$tag.json
  say "end $tag rc=$rc wall=$(( $(date +%s) - t0 ))s sha_after_equal=$([ "$a" = "$b" ] && echo yes || echo NO)"
}
runone DAGHACC_s${S}_m10_causal_heads logged causal
runone DAGHACCSTRICT_s${S}_m10_frozen_causal_heads frozen causal
runone DAGHACC_s${S}_m10_logged_heads logged logged
say "ZZLANE-DONE seed $S"
