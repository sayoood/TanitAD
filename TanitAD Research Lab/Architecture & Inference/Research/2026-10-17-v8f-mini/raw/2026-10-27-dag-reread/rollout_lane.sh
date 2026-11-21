#!/bin/bash
# rollout_lane.sh <seed>: PREREG_V8F_DAG_REREAD B-2 + beside STRICT rollouts for one seed, SEQUENTIAL (FLAT T1-kin, then DAG STRICT).
# Exactly DAG's decisive argv (E:/Projects/TanitAD-artifacts/dagt1/logs/chain.out "start DAG_s<S>_m10_causal_heads"):
#   only --ckpt and --tag change; the two OUTPUT-PATH flags (--out-dir, --dump-ticks) point into the package; DAG STRICT adds --scene frozen.
# Environment copied from dag_t1_chain.sh:118. Two lanes (seed 0, seed 1) = at most 2 rollout processes.
S=$1
PKG=/e/Projects/TanitAD-artifacts/dagreread
ART=E:/Projects/TanitAD-artifacts
TREE=$ART/dagt1_tree_b43c7a9a
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
export TANITEVAL_STACK_OVERRIDE=$(cygpath -w "$TREE/stack")
GRID="$TREE/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/eval_grid_windows.json"
JOIN="$TREE/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz"
OUT=$ART/dagreread/t1
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $PKG/logs/lane$S.log; }
runone() {  # runone <ckpt (windows-style)> <expected sha256> <tag> <scene>
  local ck=$1 exp=$2 tag=$3 scene=$4 t0 rc b a
  b=$(sha256sum < "$(cygpath -u "$ck")" | cut -c1-64)
  if [ "$b" != "$exp" ]; then say "STOP $tag: ckpt sha256 before = $b, expected $exp"; return 1; fi
  local argv=("$PY" "$TREE/stack/scripts/v8f_t1_eval.py" run --ckpt "$ck" --tag "$tag" --tracks "$ART/sfresh_inputs/inputs/native_tracks.pt"
    --eval-cache "$ART/sfresh_inputs/cache" --eval-grid "$GRID" --eval-join "$JOIN" --out-dir "$OUT"
    --replan-steps 10 --nav deadreckon --scene "$scene" --schedule periodic --limit causal --dump-ticks "$OUT/ticks" --dump-heads --device cpu)
  printf '%s\n' "${argv[@]}" > $PKG/argv/$tag.argv
  say "start $tag (scene $scene) ckpt sha ${b:0:12}"
  t0=$(date +%s)
  "${argv[@]}" > $PKG/logs/$tag.out 2>&1 < /dev/null; rc=$?
  a=$(sha256sum < "$(cygpath -u "$ck")" | cut -c1-64)
  printf '{"tag": "%s", "scene": "%s", "rc": %s, "ckpt": "%s", "ckpt_sha256_before": "%s", "ckpt_sha256_after": "%s", "wall_s": %s, "n_dumps": %s}\n' \
    "$tag" "$scene" "$rc" "$ck" "$b" "$a" "$(( $(date +%s) - t0 ))" "$(ls $PKG/t1/dump_$tag 2>/dev/null | grep -c '^ep.*\.npz$')" > $PKG/t1/done_$tag.json
  say "end $tag rc=$rc wall=$(( $(date +%s) - t0 ))s sha_after_equal=$([ "$a" = "$b" ] && echo yes || echo NO)"
}
if [ "$S" = 0 ]; then FLAT_SHA=8496934a2ce262c128a3e975da156de2f40e422925893c9979169a4c114493cc; DAG_SHA=d186fd0bcfa7ee9fbfe5c9de543fe85edda7c3e6b9c56c898958f027ae2b8cb1
else FLAT_SHA=6fdfc9b02ce87248c5315dd7604d3642a3b39f7ae0f593ec6b6ef5ef8df64c95; DAG_SHA=; fi
if [ "$S" = 1 ]; then DAG_SHA=$(grep -o '"ckpt_sha256": "[0-9a-f]*"' /e/Projects/TanitAD-artifacts/v8f-dagger-t1/done_DAG_s1_m10_causal_heads.txt | cut -d'"' -f4); fi
runone $ART/v8f-mini-flat/M2b_s${S}_flat.pt $FLAT_SHA FLAT_s${S}_m10_causal_heads logged
runone $ART/v8f-dagger/M2b_s${S}_dag/ckpt_last.pt $DAG_SHA DAGSTRICT_s${S}_m10_frozen_causal_heads frozen
say "ZZLANE-DONE seed $S"
