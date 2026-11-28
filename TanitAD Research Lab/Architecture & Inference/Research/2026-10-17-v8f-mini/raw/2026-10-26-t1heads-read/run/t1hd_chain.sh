#!/usr/bin/env bash
# PREREG_V8F_T1_HEADS (landed ae659d49 BEFORE any head output exists): the 4 runs of its section 3, on the dev-box CPU.
# Tool = a pinned export of 81aa3e36 (stack + taniteval + the grid/join files) on E:; sha256-pinned before and after every run.
# Inputs, flags and checkpoints = the banked causal run's (t1kc_chain.sh), plus --dump-ticks --dump-heads.
set -u
SC=/c/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad
SF=$SC/sfresh
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
ART=E:/Projects/TanitAD-artifacts/v8f-mini-hacc
W=/e/Projects/TanitAD-artifacts/v8f-t1heads-run/tool_81aa3e36
OUT=/c/Users/Admin/t1hd_run
LOG=$OUT/logs
mkdir -p "$LOG"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$LOG/chain.out"; }
stop() { say "STOP: $*"; echo "ZZT1HD-STOP $*" >> "$LOG/chain.out"; exit 1; }
m() { cygpath -m "$1"; }
export CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONPATH="$(m "$W/stack");$(m "$W/taniteval");$(m "$W/stack/scripts")"

if [ ! -f "$W/EXPORT_OK" ]; then
  mkdir -p "$W" || stop "mkdir export"
  git -C /c/Users/Admin/tanitad-push -c safe.directory=* archive 81aa3e36 stack taniteval "$PK/raw/eval_grid_windows.json" "$PK/raw/join_eval141.npz" \
    | tar -x -C "$W" || stop "export"
  touch "$W/EXPORT_OK"
fi
BLOB=$(git -C /c/Users/Admin/tanitad-push -c safe.directory=* show 81aa3e36:stack/scripts/v8f_t1_eval.py | sha256sum | cut -c1-64)
[ "$(tr -d '\r' < "$W/stack/scripts/v8f_t1_eval.py" | sha256sum | cut -c1-64)" = "$BLOB" ] || stop "the exported tool is not 81aa3e36's"
TOOL=$(sha256sum < "$W/stack/scripts/v8f_t1_eval.py" | cut -c1-64)
grep -q -- '"--dump-heads"' "$W/stack/scripts/v8f_t1_eval.py" || stop "no --dump-heads in the tool"
grep -q -- '"causal"' "$W/stack/scripts/v8f_t1_eval.py" || stop "no causal limit in the tool"
grep -q -- '"--replan-steps"' "$W/stack/scripts/v8f_t1_eval.py" || stop "control grep found no --replan-steps: the probe is broken"
GRID="$W/$PK/raw/eval_grid_windows.json"; JOIN="$W/$PK/raw/join_eval141.npz"
[ "$(md5sum < "$GRID" | cut -c1-32)" = 34ed8ac56365eeff72e42581bde2f5c4 ] || stop "eval grid md5"
[ "$(md5sum < "$JOIN" | cut -c1-32)" = 532fcd644c0e0835e847b2f1cee011ad ] || stop "eval join md5"
declare -A SHA=([s0]=71de3c6195ce651ee656d26579bd68a0cc5d7370090044140ed634fe10b627c0 [s1]=32e4809dc4204e70544bceb5c3c57d54310c387decd3b95ae0a830177616523a)
for s in s0 s1; do [ "$(sha256sum < "$ART/M2b_${s}_hacc.pt" | cut -c1-64)" = "${SHA[$s]}" ] || stop "checkpoint sha256 $s"; done
say "inputs OK (tool $TOOL = 81aa3e36, grid, join, 2 checkpoints)"
SFm=$(m "$SF"); GRIDm=$(m "$GRID"); JOINm=$(m "$JOIN")

cfree_mb() { df -m /c | awk 'NR==2{print $4}'; }
run() {   # run <seed> <tag> <limit>
  local s=$1 tag=$2 lim=$3
  local w=0; while [ "$(cfree_mb)" -lt 2000 ]; do [ $w -ge 120 ] && { echo "C: under 2 GB free for 2 h" >> "$LOG/$tag.out"; return 8; }; [ $w -eq 0 ] && say "$tag: waiting for C: >= 2 GB free (now $(cfree_mb) MB; the pagefile is on C:)"; w=$((w+1)); sleep 60; done
  [ "$(sha256sum < "$W/stack/scripts/v8f_t1_eval.py" | cut -c1-64)" = "$TOOL" ] || { echo "tool sha256 before $tag" >> "$LOG/$tag.out"; return 9; }
  [ -s "$OUT/t1v8f_$tag.json" ] && { echo "exists, skipped" >> "$LOG/$tag.out"; return 0; }
  "$PY" "$(m "$W/stack/scripts/v8f_t1_eval.py")" run --ckpt "$ART/M2b_${s}_hacc.pt" --tag "$tag" --tracks "$SFm/inputs/native_tracks.pt" \
    --eval-cache "$SFm/cache" --eval-grid "$GRIDm" --eval-join "$JOINm" --out-dir "$(m "$OUT")" --replan-steps 10 --device cpu \
    --limit "$lim" --dump-ticks "$(m "$OUT/ticks")" --dump-heads > "$LOG/$tag.out" 2>&1 < /dev/null
  local rc=$?
  [ "$(sha256sum < "$W/stack/scripts/v8f_t1_eval.py" | cut -c1-64)" = "$TOOL" ] || { echo "tool sha256 after $tag" >> "$LOG/$tag.out"; return 9; }
  return $rc
}
# SEQUENTIAL since 04:43: the first launch ran the two seeds in parallel and causal s0 SEGFAULTED after one batch (no traceback; s1 ran on).
# One retry per run on a native crash (rc 139), logged; the outputs of a crashed attempt are never read (the JSON is written only at the end).
runr() { run "$@"; local rc=$?; if [ $rc -eq 139 ]; then say "$2: segfault (rc 139), one retry"; mv "$LOG/$2.out" "$LOG/$2.segv.out"; run "$@"; rc=$?; fi; return $rc; }
say "== decisive: --limit causal, both seeds (sequential)"
runr s0 HACC_s0_m10_causal_heads causal || stop "causal s0 (see logs)"
runr s1 HACC_s1_m10_causal_heads causal || stop "causal s1 (see logs)"
say "== K4 and beside: --limit logged, both seeds (sequential)"
runr s0 HACC_s0_m10_logged_heads logged || stop "logged s0 (see logs)"
runr s1 HACC_s1_m10_logged_heads logged || stop "logged s1 (see logs)"
say "ZZT1HD-DONE 4"
