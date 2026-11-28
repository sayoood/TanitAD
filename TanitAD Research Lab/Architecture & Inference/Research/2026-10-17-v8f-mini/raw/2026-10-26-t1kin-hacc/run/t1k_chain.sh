#!/bin/bash
# PREREG_V8F_T1KIN_HACC (landed 323624f2 BEFORE any data): the 12 T1-kin runs on the dev-box CPU, two lanes (s0, s1), deciding runs first.
# Code = an export of the landed tip; inputs = the repo's eval grid / join (md5-asserted) and the dev-box pooled eval cache (md5-verified 141/141).
set -u
export PATH="/usr/bin:/mingw64/bin:$PATH"
TIP=${1:?tip}
W=/c/Users/Admin/t1k; OUT=/c/Users/Admin/t1k_run; LOG=$OUT/logs
SC=/c/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad
SF=$SC/sfresh; FX=$SC/viz/work
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
ART=E:/Projects/TanitAD-artifacts/v8f-mini-hacc
mkdir -p "$LOG"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$LOG/chain.out"; }
stop() { say "STOP: $*"; echo "ZZT1K-STOP $*" >> "$LOG/chain.out"; exit 1; }

say "== export $TIP"
[ -e "$W" ] && stop "$W exists"
mkdir -p "$W"
git -C /c/Users/Admin/tanitad-push -c safe.directory=* archive "$TIP" stack taniteval "$PK/raw/eval_grid_windows.json" "$PK/raw/join_eval141.npz" | tar -x -C "$W" || stop "export failed"
GRID="$W/$PK/raw/eval_grid_windows.json"; JOIN="$W/$PK/raw/join_eval141.npz"
[ "$(md5sum < "$GRID" | cut -c1-32)" = 34ed8ac56365eeff72e42581bde2f5c4 ] || stop "eval grid md5"
[ "$(md5sum < "$JOIN" | cut -c1-32)" = 532fcd644c0e0835e847b2f1cee011ad ] || stop "eval join md5"
declare -A SHA=([M2b_s0_hacc]=71de3c6195ce651ee656d26579bd68a0cc5d7370090044140ed634fe10b627c0 [M2b_s1_hacc]=32e4809dc4204e70544bceb5c3c57d54310c387decd3b95ae0a830177616523a
                [M2b_s0]=601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8 [M2b_s1]=6104fb15dc21d524b6ed53be632b8d53ec5413526ac587734c22684c7e50c2e6)
declare -A CK=([M2b_s0_hacc]=$ART/M2b_s0_hacc.pt [M2b_s1_hacc]=$ART/M2b_s1_hacc.pt [M2b_s0]=$FX/ckpt/M2b_s0.pt [M2b_s1]=$FX/ckpt/M2b_s1.pt)
for k in "${!CK[@]}"; do [ "$(sha256sum < "${CK[$k]}" | cut -c1-64)" = "${SHA[$k]}" ] || stop "checkpoint sha256 $k"; done
say "inputs OK (grid, join, 4 checkpoints)"
m() { cygpath -m "$1"; }                      # native python gets C:/... paths (Git bash would hand it /c/... in env vars)
Wm=$(m "$W"); SFm=$(m "$SF"); OUTm=$(m "$OUT"); GRIDm=$(m "$GRID"); JOINm=$(m "$JOIN")

say "== C-4: the instrument's suite on the export"
mkdir -p /c/Users/Admin/pt_t1k
( cd "$W/stack" && CUDA_VISIBLE_DEVICES=-1 PYTHONPATH="$Wm/stack;$Wm/taniteval;$Wm/stack/scripts" "$PY" -m pytest -q -p no:cacheprovider --basetemp C:/Users/Admin/pt_t1k/c4 tests/test_v8f_t1.py ) > "$LOG/c4_test_v8f_t1.out" 2>&1
tail -1 "$LOG/c4_test_v8f_t1.out" | grep -q " passed" && ! tail -1 "$LOG/c4_test_v8f_t1.out" | grep -q "failed\|error" || stop "C-4 suite not green"
say "C-4 $(tail -1 "$LOG/c4_test_v8f_t1.out")"

run() {   # run <ckpt key> <tag> <m> [extra args]
  local key=$1 tag=$2 m=$3; shift 3
  [ -f "$OUT/t1v8f_$tag.json" ] && { say "skip $tag (exists)"; return 0; }
  say "start $tag"
  CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONIOENCODING=utf-8 PYTHONPATH="$Wm/stack;$Wm/taniteval;$Wm/stack/scripts" \
    "$PY" "$Wm/stack/scripts/v8f_t1_eval.py" run --ckpt "$(m "${CK[$key]}")" --tag "$tag" --tracks "$SFm/inputs/native_tracks.pt" \
    --eval-cache "$SFm/cache" --eval-grid "$GRIDm" --eval-join "$JOINm" --out-dir "$OUTm" --replan-steps "$m" --device cpu "$@" \
    > "$LOG/$tag.out" 2>&1 < /dev/null
  local rc=$?
  say "end $tag rc=$rc"
}
lane() {  # lane <seed>
  local s=$1
  run M2b_s${s}_hacc HACC_s${s}_m10 10
  run M2b_s${s}_hacc HACC_s${s}_m30 30
  run M2b_s${s}      B_s${s}_m10    10
  run M2b_s${s}_hacc HACC_s${s}_m10_frozen 10 --scene frozen
  run M2b_s${s}_hacc HACC_s${s}_m10_limoff 10 --limit off
  run M2b_s${s}_hacc HACC_s${s}_m5  5
  say "lane s$s done"
}
lane 0 & P0=$!
lane 1 & P1=$!
wait $P0 $P1
n=$(ls "$OUT"/t1v8f_*.json 2>/dev/null | wc -l)
say "runs with a json: $n / 12"
echo "ZZT1K-DONE $n" >> "$LOG/chain.out"
