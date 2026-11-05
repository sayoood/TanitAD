#!/usr/bin/env bash
# CL READ chain (PREREG_LEVELHEAD_CL.md section 3): the two CL checkpoints (RECOVER arm CL: C0's argv + --level-head on --w-level W, trained on box 2) over ALL 2,399 eval windows, then the reader.
# Dev-box CPU, ZERO training, no ssh, ONE rollout at a time.  NOT launched by the package that wrote it: the Master Mind starts it AFTER the CL checkpoints are pulled into E:/Projects/TanitAD-artifacts/v8f-recover/
#   usage:  cl_read_chain.sh --dry-run     lists the 10 rollouts, the gates and what is present; runs nothing, writes no marker
#           cl_read_chain.sh               (normal; start through cl_read_chain.cmd = WMI Win32_Process.Create so it survives the caller)
# Rollouts, in PRIORITY order (a killed chain still yields the deciding contrast first), all with the reader's flags (recover_next/read/rnext_read_chain.sh run_tool: --replan-steps 10 --nav deadreckon
#   --schedule periodic --limit causal --dump-ticks <out>/ticks --dump-heads --device cpu) plus --smooth sg72:
#   RICL_s0 RICL_s1 (--scene reindexed, DECIDING)  CL_s0 CL_s1 (--scene logged)  RICLC_s0 RICLC_s1 (--scene reindexed --feasible kbox, D)  CLC_s0 CLC_s1 (--scene logged --feasible kbox)  FZCL_s0 FZCL_s1 (--scene hold_now)
#   Threads: logged / hold_now arms CL_TH_LOGGED (default 16 = the banked C0 / C0C / FZC0 / C0REP rollouts' count), reindexed arms CL_TH_RI (default 12 = RIC0 / RIC0C's).  MEASURED (raw/KA_NEW.json): the same tool at
#   the base's own thread count and batches reproduces the banked rollout bit for bit; at another thread count a rounding difference of 1e-5 m at tick 0 grows to 0.6 m on one window at tick 2.  Override with the env vars
#   when the CPU is shared (the read then carries that noise on a few windows, as the RI read already does).
# Gates (REFUSE = raw/CLRO.fail with the reason, exit code):
#   0  PINS.txt: the reader, pre-check, KA scripts, tests, this script and the tree's tool / model / trainer match their sha256; the unit-test log is green; raw/KA_NEW.json, raw/SMOKE_LEVEL_ON.json and both reader
#      known answers say PASS                                                                                                          (exit 2)
#   1  code/cl_precheck.py: both CL checkpoints present (absent = REFUSED), sha256 == SHA256SUMS.txt == the box-2 marker cl_markers/RUN_CL_{s}.done evidence.ckpt_sha256, argv = C0's + the two level tokens with the
#      FROZEN W, the checkpoint's own cfg says level_head on and w_level W                                                                 (exit 5 absent, 8 failed)
#   2  the reused arrays' sha256 (cl_read.py REUSED_SHA), the card's rules sha, the reader's own pins                                      (exit 6)
#   3  no other tool rollout / extractor runs (CPU shared; waits at most 3 h)
#   4  a 6-window PRE-FLIGHT per scene family: s0 --scene reindexed --feasible kbox, s1 --scene hold_now: each must exit 0 and write its ticks file   (exit 7)
# Each rollout writes done_<tag>.txt only on rc 0; a finished run is skipped on re-launch (RESUME = launch cl_read_chain.cmd again); an incomplete out dir is MOVED to _incomplete_*, never deleted.
# Markers: ZZCL-* lines in logs/chain.out; raw/CLRO.done (all ten rollouts, exit 0) / raw/CLRO.fail (reason inside); then the reader (code/cl_read.py --mode production): raw/CLREAD.done (raw/CL_READ.json + RESULTS.md written)
#   / raw/CLREAD.fail.  A REFUSED control of the reader is a result to report, not to patch around.
# WRITES: this package (logs, raw, RESULTS.md) and F:/TanitAD-scratch/motion_c/cl_read/{out,smoke,work}.  Nothing else.
set -u
DRY=0; for t in "$@"; do [ "$t" = "--dry-run" ] && DRY=1; done
PK=/e/Projects/TanitAD-artifacts/dagland/motion_c/cl_read
PKW=E:/Projects/TanitAD-artifacts/dagland/motion_c/cl_read
S=/f/TanitAD-scratch/motion_c/cl_read
SW=F:/TanitAD-scratch/motion_c/cl_read
TOOLW=$SW/tree/stack/scripts/v8f_t1_eval_ri.py
TOOL=$S/tree/stack/scripts/v8f_t1_eval_ri.py
OUT=$S/out
OUTW=$SW/out
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
ART=E:/Projects/TanitAD-artifacts
RDB=/e/Projects/TanitAD-artifacts/v8f-recover
RDBW=$ART/v8f-recover
DATA="$ART/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
CARD=/e/Projects/TanitAD-artifacts/dagland/motion_c/cl_prereg/PREREG_LEVELHEAD_CL.md
TH_LOGGED=${CL_TH_LOGGED:-16}
TH_RI=${CL_TH_RI:-12}
mkdir -p "$OUT" "$PK/logs" "$PK/raw"
log() { echo "$(date +%FT%T) $*" | tee -a "$PK/logs/chain.out"; }
fail() { log "ZZCL-FAIL $1"; echo "$(date +%FT%T) $1" > "$PK/raw/CLRO.fail"; exit "${2:-3}"; }
table() {   # tag seed scene feasible   (PRIORITY order; tab-separated)
  local r
  for r in "RICL_s0 0 reindexed off" "RICL_s1 1 reindexed off" "CL_s0 0 logged off" "CL_s1 1 logged off" "RICLC_s0 0 reindexed kbox" "RICLC_s1 1 reindexed kbox" \
           "CLC_s0 0 logged kbox" "CLC_s1 1 logged kbox" "FZCL_s0 0 hold_now off" "FZCL_s1 1 hold_now off"; do
    set -- $r
    printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4"
  done
}
pin() { sed -n "s|^$1 sha256 \([0-9a-f]*\)\$|\1|p" "$PK/PINS.txt" | tr -d '\r'; }

if [ $DRY -eq 1 ]; then
  echo "CL read chain, DRY RUN (nothing runs, no marker is written)"
  echo "rollouts in priority order (tag seed scene feasible; threads: reindexed $TH_RI, others $TH_LOGGED):"; table
  for s in 0 1; do
    r=M2b_s${s}_recft_CL
    echo "checkpoint $RDB/$r/ckpt_last.pt: $([ -f "$RDB/$r/ckpt_last.pt" ] && echo PRESENT || echo ABSENT)   marker $RDB/cl_markers/RUN_CL_$s.done: $([ -f "$RDB/cl_markers/RUN_CL_$s.done" ] && echo PRESENT || echo ABSENT)"
  done
  echo "FROZEN_W.txt: $([ -f "$(dirname "$CARD")/FROZEN_W.txt" ] && echo PRESENT || echo ABSENT)"
  echo "tree tool: $([ -f "$TOOL" ] && sha256sum "$TOOL" | cut -c1-16 || echo ABSENT)   KA_NEW: $(grep -c '"PASS": true' "$PK/raw/KA_NEW.json" 2>/dev/null || echo 0) PASS line(s)   PINS.txt: $([ -f "$PK/PINS.txt" ] && echo PRESENT || echo ABSENT)"
  exit 0
fi

if [ -f "$PK/raw/CLRO.running" ]; then
  op=$(cat "$PK/raw/CLRO.running" | tr -d '\r\n ')
  if [ -n "$op" ] && kill -0 "$op" 2>/dev/null; then log "ZZCL-ALREADY-RUNNING pid $op: not starting a second chain"; exit 9; fi
  mv "$PK/raw/CLRO.running" "$PK/raw/_stale_CLRO.running_$(date +%H%M%S)"
fi
echo $$ > "$PK/raw/CLRO.running"
trap 'echo "exited $(date +%FT%T)" > "$PK/raw/CLRO.running"' EXIT
for m in CLRO.done CLRO.fail CLREAD.done CLREAD.fail; do [ -f "$PK/raw/$m" ] && mv "$PK/raw/$m" "$PK/raw/_old_${m}_$(date +%H%M%S)"; done

# ---- gate 0: pins, tests, known answers
[ -f "$PK/PINS.txt" ] || fail "PINS.txt is absent (run: python -I code/make_pins.py)" 2
for f in code/cl_read.py code/cl_precheck.py code/ka_new.py code/ka_new_summary.py code/run_tool_subset.py code/smoke_level_on.py code/build_read_tree.py code/make_pins.py tests/test_cl_read.py tests/test_cl_chain_gates.py cl_read_chain.sh; do
  want=$(pin "$f"); got=$(sha256sum "$PK/$f" | cut -c1-64)
  [ -n "$want" ] && [ "$want" = "$got" ] || fail "$f sha $got != PINS.txt ($want)" 2
done
for kv in "tree/stack/scripts/v8f_t1_eval_ri.py:tool" "tree/stack/tanitad/models/v8f_mini.py:model" "tree/stack/scripts/v8f_mini_train.py:trainer"; do
  f=${kv%%:*}; k=${kv##*:}; want=$(pin "$k"); got=$(sha256sum "$S/$f" | cut -c1-64)
  [ -n "$want" ] && [ "$want" = "$got" ] || fail "scratch tree $k ($f) sha $got != PINS.txt ($want)" 2
done
for lg in pytest_cl_read pytest_cl_chain_gates; do
  last=$(grep -v '^[[:space:]]*$' "$PK/logs/$lg.log" | tail -n 1)
  case "$last" in *" passed"*) case "$last" in *failed*|*error*) fail "unit tests not green ($lg): $last" 2;; esac;; *) fail "unit tests not green ($lg): $last" 2;; esac
done
grep -q "^MUTATIONS ALL RED" "$PK/logs/mutations_cl.log" || fail "the mutation log does not carry MUTATIONS ALL RED" 2
"$PY" -I -c "
import json,sys
pk=sys.argv[1]
for f,k in (('KA_NEW','PASS'),('SMOKE_LEVEL_ON','PASS'),('KA_READER_ka-null','KA_PASS'),('KA_READER_ka-ri','KA_PASS')):
    d=json.load(open(pk+'/raw/'+f+'.json',encoding='utf-8'))
    if d.get(k) is not True: print('NOT PASS:',f); sys.exit(1)
" "$PKW" || fail "a known answer is not PASS (raw/KA_NEW.json, raw/SMOKE_LEVEL_ON.json, raw/KA_READER_ka-null.json, raw/KA_READER_ka-ri.json)" 2
log "ZZCL-GATE0 pins, tests, known answers ok (tool $(pin tool | cut -c1-8), model $(pin model | cut -c1-8), trainer $(pin trainer | cut -c1-8))"

# ---- gate 1: the pulled CL checkpoints (REFUSED when absent)
"$PY" -I "$PKW/code/cl_precheck.py" > "$PK/logs/precheck.out" 2>&1; rc=$?
cat "$PK/logs/precheck.out" | tee -a "$PK/logs/chain.out"
[ $rc -eq 0 ] || fail "pulled-file controls refuse the read (rc $rc): see logs/precheck.out and raw/CL_PRECHECK.json" "$rc"
log "ZZCL-GATE1 CL checkpoints present, sha256 == SHA256SUMS.txt == box-2 marker, argv = C0's + the two level tokens, cfg level_head on"

# ---- gate 2: reused arrays and the card
"$PY" -I -c "
import sys
sys.path.insert(0, sys.argv[1])
import importlib.util
spec = importlib.util.spec_from_file_location('cl_read', sys.argv[1] + '/cl_read.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.init()
bad = []
for s in m.SEEDS:
    _, c = m.reused_arrays(s)
    bad += [k for k, v in c.items() if not v['PASS']]
if m.card_rules_sha() != m.CARD_RULES_SHA: bad.append('card rules sha')
print('REUSED/CARD', 'BAD ' + str(bad) if bad else 'ok')
sys.exit(1 if bad else 0)" "$PKW/code" > "$PK/logs/gate2.out" 2>&1 || fail "gate 2 (reused arrays / card rules / reader pins): $(tail -n 2 "$PK/logs/gate2.out" | tr '\n' ' ')" 6
log "ZZCL-GATE2 reused arrays (12) and the card's rules match their pins"

# ---- gate 3: CPU
busy() { powershell -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'v8f_t1_eval_[a-z_0-9]+[.]py\S*\s+run|extract_smooth[.]py|run_tool_subset[.]py' -and \$_.ProcessId -ne \$PID }).Count" | tr -d '\r '; }
w=0; while [ "$(busy)" != "0" ] && [ -n "$(busy)" ]; do sleep 30; w=$((w+30)); [ $w -ge 10800 ] && { log "ZZCL-BUSY-TIMEOUT another rollout or extractor still runs after 3 h; proceeding"; break; }; done
[ $w -gt 0 ] && log "ZZCL-CPU-WAIT ${w}s"

CARD_LINE="PREREG_LEVELHEAD_CL.md sha256 $(sha256sum "$CARD" | cut -c1-64)"
run_tool() {   # tag ckpt outdir threads extra-flags... (stdin closed so a tool never eats the table)
  local tag=$1 ckpt=$2 od=$3 th=$4; shift 4
  OMP_NUM_THREADS=$th MKL_NUM_THREADS=$th OPENBLAS_NUM_THREADS=$th CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
  "$PY" -I "$TOOLW" run --ckpt "$ckpt" --tag "$tag" --tracks "$ART/sfresh_inputs/inputs/native_tracks.pt" --eval-cache "$ART/sfresh_inputs/cache" \
     --eval-grid "$DATA/eval_grid_windows.json" --eval-join "$DATA/join_eval141.npz" --out-dir "$od" \
     --replan-steps 10 --nav deadreckon --schedule periodic --limit causal --dump-ticks "$od/ticks" --dump-heads --device cpu "$@" < /dev/null
}
[ -f "$ART/sfresh_inputs/inputs/native_tracks.pt" ] || fail "input native_tracks.pt absent" 5
[ -d "$ART/sfresh_inputs/cache/eval/clips" ] || fail "eval cache absent" 5
for f in eval_grid_windows.json join_eval141.npz; do [ -f "$DATA/$f" ] || fail "input $f absent from the data dir" 5; done

# ---- gate 4: pre-flight (6 windows)
pre() {   # tag seed threads flags...
  local tag=$1 s=$2 th=$3; shift 3; rm -rf "$S/smoke/$tag"; mkdir -p "$S/smoke/$tag"
  run_tool "$tag" "$RDBW/M2b_s${s}_recft_CL/ckpt_last.pt" "$SW/smoke/$tag" "$th" --max-windows 6 "$@" > "$PK/logs/smoke_$tag.out" 2>&1; local rc=$?
  [ $rc -eq 0 ] && [ -f "$S/smoke/$tag/ticks/ticks_$tag.npz" ] || fail "pre-flight $tag rc=$rc (a CL checkpoint does not load in the read tree, or a flag is refused): see logs/smoke_$tag.out" 7
  log "ZZCL-PREFLIGHT-OK $tag"
}
pre PRE_CL_s0_ri_kbox 0 "$TH_RI" --smooth sg72 --scene reindexed --feasible kbox
pre PRE_CL_s1_hold 1 "$TH_LOGGED" --smooth sg72 --scene hold_now --freeze-card-ref "$CARD_LINE"
log "ZZCL-START tool $(pin tool | cut -c1-8) threads reindexed=$TH_RI logged=$TH_LOGGED"

run_one() {   # tag seed scene feas
  local tag=$1 s=$2 scene=$3 feas=$4 ck="$RDBW/M2b_s${2}_recft_CL/ckpt_last.pt"
  [ -f "$OUT/done_$tag.txt" ] && { log "ZZCL-SKIP $tag"; return 0; }
  [ -d "$OUT/$tag" ] && mv "$OUT/$tag" "$OUT/_incomplete_${tag}_$(date +%H%M%S)"
  mkdir -p "$OUT/$tag"
  local flags="--smooth sg72 --scene $scene"; local extra=()
  [ "$feas" != "off" ] && flags="$flags --feasible $feas"
  [ "$scene" = "hold_now" ] && extra=(--freeze-card-ref "$CARD_LINE")
  local th=$TH_LOGGED; [ "$scene" = "reindexed" ] && th=$TH_RI
  local t0=$(date +%s)
  log "ZZCL-RUN $tag flags=$flags threads=$th"
  # shellcheck disable=SC2086
  run_tool "$tag" "$ck" "$OUTW/$tag" "$th" $flags "${extra[@]}" > "$PK/logs/run_$tag.out" 2>&1; local rc=$?
  if [ $rc -eq 0 ]; then
    printf '{"tag": "%s", "flags": "%s", "ckpt": "%s", "ckpt_sha256": "%s", "tool": "%s", "tool_sha256": "%s", "threads": %s, "wall_s": %s}\n' "$tag" "$flags" "$ck" \
       "$(sha256sum "$RDB/M2b_s${s}_recft_CL/ckpt_last.pt" | cut -c1-64)" "$TOOLW" "$(pin tool)" "$th" "$(( $(date +%s) - t0 ))" > "$OUT/done_$tag.txt"
    log "ZZCL-RUN-DONE $tag wall=$(( $(date +%s) - t0 ))s"
  else
    log "ZZCL-RUN-FAIL $tag rc=$rc"
  fi
  return $rc
}
nfail=0
while IFS=$'\t' read -r tag seed scene feas <&3; do
  run_one "$tag" "$seed" "$scene" "$feas" || nfail=$((nfail+1))
done 3< <(table)
log "ZZCL-ROLLOUTS-DONE nfail=$nfail"
missing=""; while IFS=$'\t' read -r tag rest; do [ -f "$OUT/done_$tag.txt" ] || missing="$missing $tag"; done < <(table)
[ -z "$missing" ] || fail "missing done markers:$missing (see logs/run_<tag>.out)" 3
echo "$(date +%FT%T) ten CL rollouts done" > "$PK/raw/CLRO.done"
log "ZZCL-CLRO-DONE"

# ---- the reader (controls first; a refused control is a result to report)
log "ZZCL-READ start"
"$PY" -I "$PKW/code/cl_read.py" --mode production --out-root "$OUTW" --work-dir "$SW/work" --raw "$PKW/raw" --workers 6 > "$PK/logs/read.out" 2>&1; rc=$?
log "ZZCL-READ rc=$rc"
tail -n 3 "$PK/logs/read.out" | tee -a "$PK/logs/chain.out"
if [ $rc -eq 0 ] && [ -f "$PK/raw/CL_READ.json" ] && [ -f "$PK/RESULTS.md" ]; then
  echo "$(date +%FT%T) CL_READ.json and RESULTS.md written" > "$PK/raw/CLREAD.done"
  log "ZZCL-CHAIN-DONE"
else
  echo "$(date +%FT%T) reader rc=$rc (see logs/read.out; a REFUSED control names itself in raw/CL_READ.json 'refused')" > "$PK/raw/CLREAD.fail"
  log "ZZCL-READ-FAIL rc=$rc"
  exit "$rc"
fi
