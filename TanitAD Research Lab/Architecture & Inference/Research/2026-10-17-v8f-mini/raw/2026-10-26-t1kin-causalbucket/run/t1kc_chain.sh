#!/bin/bash
# PREREG_V8F_T1KIN_CAUSALBUCKET (landed d2ed136d BEFORE any data): C-L1 smokes, the 2 decisive runs (--limit causal), the 2 beside runs
# (--limit causal_all), on the dev-box CPU. Patched code = /c/Users/Admin/t1kc (export of d2ed136d + the causal mode); the UNPATCHED reference
# for C-L1 = /c/Users/Admin/t1k (export of 323624f2; v8f_t1_eval.py, v8f_mini.py, v8f_mini_data.py, floors_kin.py identical to d2ed136d).
set -u
export PATH="/usr/bin:/mingw64/bin:$PATH"
WP=/c/Users/Admin/t1kc; WU=/c/Users/Admin/t1k; OUT=/c/Users/Admin/t1kc_run; LOG=$OUT/logs
SC=/c/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad
SF=$SC/sfresh
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
ART=E:/Projects/TanitAD-artifacts/v8f-mini-hacc
mkdir -p "$LOG"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$LOG/chain.out"; }
stop() { say "STOP: $*"; echo "ZZT1KC-STOP $*" >> "$LOG/chain.out"; exit 1; }

GRID="$WU/$PK/raw/eval_grid_windows.json"; JOIN="$WU/$PK/raw/join_eval141.npz"
[ "$(md5sum < "$GRID" | cut -c1-32)" = 34ed8ac56365eeff72e42581bde2f5c4 ] || stop "eval grid md5"
[ "$(md5sum < "$JOIN" | cut -c1-32)" = 532fcd644c0e0835e847b2f1cee011ad ] || stop "eval join md5"
declare -A SHA=([s0]=71de3c6195ce651ee656d26579bd68a0cc5d7370090044140ed634fe10b627c0 [s1]=32e4809dc4204e70544bceb5c3c57d54310c387decd3b95ae0a830177616523a)
for s in s0 s1; do [ "$(sha256sum < "$ART/M2b_${s}_hacc.pt" | cut -c1-64)" = "${SHA[$s]}" ] || stop "checkpoint sha256 $s"; done
for f in tanitad/models/v8f_mini.py tanitad/data/v8f_mini_data.py; do cmp -s "$WP/stack/$f" "$WU/stack/$f" || stop "$f differs between the trees"; done
cmp -s "$WP/taniteval/taniteval/floors_kin.py" "$WU/taniteval/taniteval/floors_kin.py" || stop "floors_kin differs"
say "inputs OK (grid, join, 2 checkpoints, model/data/floor code identical across the trees)"
TOOL_P=d0ac8ce685ea9083083d380d4764510ce43677439eaee27cf3ab8760df0303ce      # the patched v8f_t1_eval.py (C-L2-tested, mutants restored)
BLOB_U=$(git -C /c/Users/Admin/tanitad-push -c safe.directory=* show d2ed136d:stack/scripts/v8f_t1_eval.py | sha256sum | cut -c1-64)
[ "$(tr -d '\r' < "$WU/stack/scripts/v8f_t1_eval.py" | sha256sum | cut -c1-64)" = "$BLOB_U" ] || stop "the unpatched tool is not the tip's"   # git archive wrote CRLF
TOOL_U=$(sha256sum < "$WU/stack/scripts/v8f_t1_eval.py" | cut -c1-64)
say "tools pinned: patched ${TOOL_P:0:12}, unpatched ${TOOL_U:0:12} (= d2ed136d)"
m() { cygpath -m "$1"; }
SFm=$(m "$SF"); GRIDm=$(m "$GRID"); JOINm=$(m "$JOIN")

run() {   # run <tree> <seed> <tag> <outdir> [extra]
  local tree=$1 s=$2 tag=$3 od=$4; shift 4
  local Wm; Wm=$(m "$tree")
  [ -f "$od/t1v8f_$tag.json" ] && { say "skip $tag (exists)"; return 0; }
  mkdir -p "$od"
  # the run reads exactly the tool whose sha256 is pinned (the 01:19 launch was killed: the C-L2 mutation proof was editing this file)
  local want; [ "$tree" = "$WP" ] && want=$TOOL_P || want=$TOOL_U
  [ "$(sha256sum < "$tree/stack/scripts/v8f_t1_eval.py" | cut -c1-64)" = "$want" ] || stop "tool sha256 before $tag"
  say "start $tag ($(basename "$tree")) tool $(echo $want | cut -c1-12)"
  CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONIOENCODING=utf-8 PYTHONPATH="$Wm/stack;$Wm/taniteval;$Wm/stack/scripts" \
    "$PY" "$Wm/stack/scripts/v8f_t1_eval.py" run --ckpt "$ART/M2b_${s}_hacc.pt" --tag "$tag" --tracks "$SFm/inputs/native_tracks.pt" \
    --eval-cache "$SFm/cache" --eval-grid "$GRIDm" --eval-join "$JOINm" --out-dir "$(m "$od")" --replan-steps 10 --device cpu "$@" \
    > "$LOG/$tag.$(basename "$tree").out" 2>&1 < /dev/null
  local rc=$?
  [ "$(sha256sum < "$tree/stack/scripts/v8f_t1_eval.py" | cut -c1-64)" = "$want" ] || stop "tool sha256 after $tag"
  say "end $tag ($(basename "$tree")) rc=$rc"
  return $rc
}

say "== C-L1: logged and off, unpatched vs patched, --max-windows 24"
for lim in logged off; do
  run "$WU" s0 cl1_$lim "$OUT/cl1_unpatched" --limit $lim --max-windows 24 || stop "C-L1 unpatched $lim"
  run "$WP" s0 cl1_$lim "$OUT/cl1_patched" --limit $lim --max-windows 24 || stop "C-L1 patched $lim"
  cmp -s "$OUT/cl1_unpatched/t1v8f_cl1_$lim.json" "$OUT/cl1_patched/t1v8f_cl1_$lim.json" || stop "C-L1 json differs ($lim)"
  for f in "$OUT/cl1_unpatched/dump_cl1_$lim"/*.npz; do cmp -s "$f" "$OUT/cl1_patched/dump_cl1_$lim/$(basename "$f")" || stop "C-L1 dump differs ($lim, $(basename "$f"))"; done
  say "C-L1 $lim: json and $(ls "$OUT/cl1_patched/dump_cl1_$lim" | wc -l) dumps byte-identical"
done

say "== decisive: --limit causal, both seeds"
( run "$WP" s0 HACC_s0_m10_causal "$OUT" --limit causal ) & P0=$!
( run "$WP" s1 HACC_s1_m10_causal "$OUT" --limit causal ) & P1=$!
wait $P0 $P1
say "== beside: --limit causal_all, both seeds"
( run "$WP" s0 HACC_s0_m10_causalall "$OUT" --limit causal_all ) & P0=$!
( run "$WP" s1 HACC_s1_m10_causalall "$OUT" --limit causal_all ) & P1=$!
wait $P0 $P1
n=$(ls "$OUT"/t1v8f_HACC_*.json 2>/dev/null | wc -l)
say "runs with a json: $n / 4"
echo "ZZT1KC-DONE $n" >> "$LOG/chain.out"
