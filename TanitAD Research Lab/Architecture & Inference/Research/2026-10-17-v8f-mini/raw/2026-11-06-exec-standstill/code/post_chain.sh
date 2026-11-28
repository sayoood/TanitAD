#!/usr/bin/env bash
# post_chain.sh: waits (a script, not a model turn) for raw/CHAIN.done, then: extract the 8 per-window array sets (3 single-thread workers), gate-full (G-ID on the full grid),
# the read (refuses unless every gate passed), and the result tables. Markers: raw/POST.running, raw/POST.done, raw/POST.fail (the failing step inside).
set -u
EXP=/e/Projects/TanitAD-artifacts/dagland/motion_c/exec_standstill
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
CODE=E:/Projects/TanitAD-artifacts/dagland/motion_c/exec_standstill/code
export CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$EXP/logs" "$EXP/raw"
log() { echo "$(date +%FT%T) $*" | tee -a "$EXP/logs/post.out"; }
fail() { log "ZZPOST-FAIL $1"; echo "$(date +%FT%T) $1" > "$EXP/raw/POST.fail"; exit "${2:-3}"; }
rm -f "$EXP/raw/POST.done" "$EXP/raw/POST.fail"
echo $$ > "$EXP/raw/POST.running"
trap 'echo "exited $(date +%FT%T)" > "$EXP/raw/POST.running"' EXIT
log "ZZPOST-WAIT for raw/CHAIN.done"
while [ ! -f "$EXP/raw/CHAIN.done" ]; do
  [ -f "$EXP/raw/CHAIN.fail" ] && fail "the rollout chain failed: $(cat "$EXP/raw/CHAIN.fail")" 4
  sleep 20
done
log "ZZPOST-EXTRACT"
"$PY" -I "$CODE/exec_read.py" extract --workers 3 >> "$EXP/logs/post.out" 2>&1 || fail "extract rc=$?" 5
log "ZZPOST-GATE-FULL"
"$PY" -I "$CODE/exec_read.py" gate-full >> "$EXP/logs/post.out" 2>&1 || fail "gate-full rc=$? (G-ID on the full grid; see raw/GATE_ID_FULL.json; NO contrast was read)" 6
log "ZZPOST-READ"
"$PY" -I "$CODE/exec_read.py" read >> "$EXP/logs/post.out" 2>&1 || fail "read rc=$?" 7
"$PY" -I "$CODE/make_tables.py" >> "$EXP/logs/post.out" 2>&1 || fail "make_tables rc=$?" 8
echo "$(date +%FT%T) tables written" > "$EXP/raw/POST.done"
log "ZZPOST-DONE"
