#!/bin/bash
# PREREG_V8F_T1KIN_STRICT (landed 963906e8 BEFORE any data): the 2 strict runs (--scene frozen --limit causal), queued behind the HACC-LAT
# chain (dev-box memory: 2.3 GB free physical with HLAT's two lanes and another session's jobs running). Code = /c/Users/Admin/t1kc, whose
# v8f_t1_eval.py is the tool landed at 76c7d98e (CRLF export; raw sha256 pinned below and asserted before and after each run).
set -u
export PATH="/usr/bin:/mingw64/bin:$PATH"
WP=/c/Users/Admin/t1kc; WU=/c/Users/Admin/t1k; OUT=/c/Users/Admin/t1ks_run; LOG=$OUT/logs
SC=/c/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad
SF=$SC/sfresh; PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
ART=E:/Projects/TanitAD-artifacts/v8f-mini-hacc
HL=/c/Users/Admin/hlat/logs/chain.out
mkdir -p "$LOG"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$LOG/chain.out"; }
stop() { say "STOP: $*"; echo "ZZT1KS-STOP $*" >> "$LOG/chain.out"; exit 1; }

say "== waiting for the HACC-LAT chain (ZZHLAT-DONE / ZZHLAT-STOP in $HL), up to 4 h"
for i in $(seq 1 480); do grep -aq "ZZHLAT-DONE\|ZZHLAT-STOP" "$HL" 2>/dev/null && break; sleep 30; done
grep -aq "ZZHLAT-DONE\|ZZHLAT-STOP" "$HL" || stop "HACC-LAT chain did not finish within 4 h"
say "HACC-LAT chain finished: $(grep -a 'ZZHLAT-' "$HL" | tail -1)"
GRID="$WU/$PK/raw/eval_grid_windows.json"; JOIN="$WU/$PK/raw/join_eval141.npz"
[ "$(md5sum < "$GRID" | cut -c1-32)" = 34ed8ac56365eeff72e42581bde2f5c4 ] || stop "eval grid md5"
[ "$(md5sum < "$JOIN" | cut -c1-32)" = 532fcd644c0e0835e847b2f1cee011ad ] || stop "eval join md5"
declare -A SHA=([s0]=71de3c6195ce651ee656d26579bd68a0cc5d7370090044140ed634fe10b627c0 [s1]=32e4809dc4204e70544bceb5c3c57d54310c387decd3b95ae0a830177616523a)
for s in s0 s1; do [ "$(sha256sum < "$ART/M2b_${s}_hacc.pt" | cut -c1-64)" = "${SHA[$s]}" ] || stop "checkpoint sha256 $s"; done
TOOL_P=d0ac8ce685ea9083083d380d4764510ce43677439eaee27cf3ab8760df0303ce
BLOB=$(git -C /c/Users/Admin/tanitad-push -c safe.directory=* show 963906e8:stack/scripts/v8f_t1_eval.py | sha256sum | cut -c1-64)
[ "$(tr -d '\r' < "$WP/stack/scripts/v8f_t1_eval.py" | sha256sum | cut -c1-64)" = "$BLOB" ] || stop "the tool is not the one landed at 963906e8"
say "inputs OK; tool ${TOOL_P:0:12} (= 963906e8's v8f_t1_eval.py, CR-stripped)"
m() { cygpath -m "$1"; }
Wm=$(m "$WP"); SFm=$(m "$SF"); GRIDm=$(m "$GRID"); JOINm=$(m "$JOIN"); OUTm=$(m "$OUT")
run() {   # run <seed>
  local s=$1 tag=HACC_${1}_m10_strict
  [ -f "$OUT/t1v8f_$tag.json" ] && { say "skip $tag (exists)"; return 0; }
  [ "$(sha256sum < "$WP/stack/scripts/v8f_t1_eval.py" | cut -c1-64)" = "$TOOL_P" ] || stop "tool sha256 before $tag"
  say "start $tag tool ${TOOL_P:0:12}"
  CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONIOENCODING=utf-8 PYTHONPATH="$Wm/stack;$Wm/taniteval;$Wm/stack/scripts" \
    "$PY" "$Wm/stack/scripts/v8f_t1_eval.py" run --ckpt "$ART/M2b_${s}_hacc.pt" --tag "$tag" --tracks "$SFm/inputs/native_tracks.pt" \
    --eval-cache "$SFm/cache" --eval-grid "$GRIDm" --eval-join "$JOINm" --out-dir "$OUTm" --replan-steps 10 --device cpu \
    --nav deadreckon --scene frozen --limit causal --schedule periodic > "$LOG/$tag.out" 2>&1 < /dev/null
  local rc=$?
  [ "$(sha256sum < "$WP/stack/scripts/v8f_t1_eval.py" | cut -c1-64)" = "$TOOL_P" ] || stop "tool sha256 after $tag"
  say "end $tag rc=$rc"
}
( run s0 ) & P0=$!
( run s1 ) & P1=$!
wait $P0 $P1
n=$(ls "$OUT"/t1v8f_HACC_*_strict.json 2>/dev/null | wc -l)
say "runs with a json: $n / 2"
echo "ZZT1KS-DONE $n" >> "$LOG/chain.out"
