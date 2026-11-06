#!/usr/bin/env bash
# LFO chain (PREREG_LEVELFLOOR_A4.md, sha256 8bfbc386...): arm LFO075 = planner C0_s{s} + the lead cap from the LLR_s{s} heads + the level floor F at kappa 0.75 with the level input v_hat := the LOGGED route speed (--floor-vhat-source logged),
# a CEILING diagnostic (privileged input, inadmissible at inference). Four rollouts on `reindexed` over ALL 2,399 eval windows, then the reader. Dev-box CPU, ZERO training, no ssh, ONE rollout at a time. Cloned from lf_read/lf_read_chain.sh.
# Launched by the Master Mind / the build agent (lfo_chain.cmd = WMI Win32_Process.Create so it survives the caller).
#   usage:  lfo_chain.sh --dry-run     lists the 4 rollouts, the gates and what is present; runs nothing, writes no marker
#           lfo_chain.sh               (normal; RESUME = launch the .cmd again, finished rollouts are skipped)
# READ TREE: F:/TanitAD-scratch/motion_c/lfo_read/tree: the LF read tree (the LLR read tree's 83c9ee0 model / data / trainer) + the LFO tool v8f_t1_eval_ri.py (flags --level-floor on --floor-kappa K --floor-ckpt LLR ckpt --floor-vhat-source logged)
#   + level_floor.py (the LF operator, unchanged). The tool's sha256 is `patched_tool` in motion_c/lfo_build/PINS.txt; gate 0 refuses when that file or key is absent or the tree's tool differs.
# Rollouts in PRIORITY order (a killed chain yields the bare arm first): RILFO075_s0 RILFO075_s1 (bare: --smooth sg72) then RILFO075C_s0 RILFO075C_s1 (composite: --smooth sg72 --feasible kbox), all `--scene reindexed`, 12 threads (LFO_TH).
#   Flags = the banked RIC0 / RIC0C flags + `--level-floor on --floor-kappa 0.75 --floor-ckpt .../M2b_s{s}_recft_LLR/ckpt_last.pt --floor-vhat-source logged` (the source is NAMED explicitly: heads is the tool's default). --ckpt = .../M2b_s{s}_recft_C0/ckpt_last.pt.
#   `logged` and `hold_now` are not read (A4 card section 3). NO base rollout: the banked C0, C0C, C0REP, RIC0, RIC0C, RILF075, RILF075C arrays (sha256-pinned in the reader).
# Gates (REFUSE = raw/LFO.fail with the reason, exit code):
#   0  PINS.txt: reader, pre-check, tests, this script match their sha256; the shared files (A4 card, lf_read.py, lf_precheck.py, llr_read.py, LF_READ.json) match theirs; the tree's model / data / trainer (= the LLR read's pins), the LFO tool
#      (= lfo_build/PINS.txt patched_tool) and level_floor.py (= its `level_floor` key) match; the unit-test logs are green; the mutation logs say ALL RED; the known answers say PASS: KA_LFO.json (KA-O1, KA-O2, run with exactly
#      this tool), SMOKE_LFO.json, and the reader's ka-lf and ka-null                                                                                           (exit 2)
#   1  code/lfo_precheck.py: both seeds' C0 and LLR checkpoints present (absent = 5), sha256 == the banked ones == SHA256SUMS.txt                                   (exit 5 absent, 8 failed)
#   2  the reused arrays' sha256 (10 base + 4 LF075), the CL card's rules sha, the A3 and A4 cards, the pinned readers                                              (exit 6)
#   3  no other tool rollout / extractor runs (CPU shared; waits at most 3 h)
#   4  a 6-window PRE-FLIGHT with the oracle ON: s0 bare and s1 composite: each must exit 0 and write a ticks file carrying the lf_* fields and lf_vhat_heads    (exit 7)
# After every finished rollout lfo_precheck.py --rollout TAG checks the output JSON (level_floor on, kappa 0.75, floor_ckpt sha256 == the seed's LLR sha, floor_vhat_source logged, label CEILING) and the planner sha (exit 8, chain stops).
# Markers (raw/): LFO.running (pid; rewritten "exited ..." on exit), LFO_ROLLOUTS.done, then LFO.done (raw/LFO_READ.json and RESULTS.md written) or LFO.fail (the failing step). A chain that dies for any reason that a shell can trap
# (error, TERM, INT, HUP) leaves LFO.fail naming the step it was in. A REFUSED control of the reader is a result.
# WRITES: this package (logs, raw, RESULTS.md) and F:/TanitAD-scratch/motion_c/lfo_read/{out,smoke,work}. The LFO tree is READ ONLY here.
set -u
DRY=0; for t in "$@"; do [ "$t" = "--dry-run" ] && DRY=1; done
PK=${LFO_PK:-/e/Projects/TanitAD-artifacts/dagland/motion_c/lfo_read}
PKW=${LFO_PKW:-E:/Projects/TanitAD-artifacts/dagland/motion_c/lfo_read}
S=/f/TanitAD-scratch/motion_c/lfo_read
SW=F:/TanitAD-scratch/motion_c/lfo_read
TREE=${LFO_TREE:-$S/tree}
TOOLW=$SW/tree/stack/scripts/v8f_t1_eval_ri.py
TOOL=$TREE/stack/scripts/v8f_t1_eval_ri.py
OPR=$TREE/stack/scripts/level_floor.py
OUT=$S/out
OUTW=$SW/out
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
ART=E:/Projects/TanitAD-artifacts
MC=/e/Projects/TanitAD-artifacts/dagland/motion_c
MCW=$ART/dagland/motion_c
RDB=/e/Projects/TanitAD-artifacts/v8f-recover
RDBW=$ART/v8f-recover
DATA="$ART/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
CARD=$MC/a4_prereg/PREREG_LEVELFLOOR_A4.md
CARD_SHA=8bfbc386083ce9b78f4938f3c44cb8772cbb17f3e6beb2ee96a018fcb7066721
BUILD_PINS=${LFO_BUILD_PINS_FILE:-$MC/lfo_build/PINS.txt}
BUILD_DIR=${LFO_BUILD_DIR:-$MC/lfo_build}
BUILD_DIRW=${LFO_BUILD_DIRW:-$MCW/lfo_build}
LLR_PINS=$MC/llr_read/PINS.txt
KAPPA=0.75
TH=${LFO_TH:-12}
mkdir -p "$OUT" "$PK/logs" "$PK/raw" "$S/smoke" "$S/work"
log() { echo "$(date +%FT%T) $*" | tee -a "$PK/logs/chain.out"; }
STEP=start
fail() { log "ZZLFO-FAIL $1"; echo "$(date +%FT%T) step '$STEP': $1" > "$PK/raw/LFO.fail"; exit "${2:-3}"; }
table() {   # tag seed scene feasible   (PRIORITY order; tab-separated)
  local r
  for r in "RILFO075_s0 0 reindexed off" "RILFO075_s1 1 reindexed off" "RILFO075C_s0 0 reindexed kbox" "RILFO075C_s1 1 reindexed kbox"; do
    set -- $r
    printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4"
  done
}
pin() { sed -n "s|^$1 sha256 \([0-9a-f]*\)\$|\1|p" "$PK/PINS.txt" | tr -d '\r'; }
bpin() { grep -E "^$1([[:space:]:=]|\$)" "$BUILD_PINS" 2>/dev/null | grep -oE '[0-9a-f]{64}' | head -n 1; }
lpin() { sed -n "s|^$1 sha256 \([0-9a-f]*\)\$|\1|p" "$LLR_PINS" | tr -d '\r'; }

if [ $DRY -eq 1 ]; then
  echo "LFO chain, DRY RUN (nothing runs, no marker is written)"
  echo "rollouts in priority order (tag seed scene feasible; threads $TH; kappa $KAPPA; --floor-vhat-source logged; CEILING):"; table
  for s in 0 1; do
    echo "seed $s: C0 checkpoint $RDB/M2b_s${s}_recft_C0/ckpt_last.pt: $([ -f "$RDB/M2b_s${s}_recft_C0/ckpt_last.pt" ] && echo PRESENT || echo ABSENT)   LLR checkpoint (presence / gap heads) $RDB/M2b_s${s}_recft_LLR/ckpt_last.pt: $([ -f "$RDB/M2b_s${s}_recft_LLR/ckpt_last.pt" ] && echo PRESENT || echo ABSENT)   banked base arrays (C0 C0C C0REP RIC0 RIC0C RILF075 RILF075C): $([ -f "$MC/../recover_next/read/raw/arrays/C0_s$s.npz" ] && [ -f "$MC/../recover_next/read/raw/arrays/C0C_s$s.npz" ] && [ -f "$MC/../recover_next/read/raw/arrays/C0REP_s$s.npz" ] && [ -f "$MC/ri_read/raw/arrays/RIC0_s$s.npz" ] && [ -f "$MC/ri_read/raw/arrays/RIC0C_s$s.npz" ] && [ -f "$MC/lf_read/raw/arrays/RILF075_s$s.npz" ] && [ -f "$MC/lf_read/raw/arrays/RILF075C_s$s.npz" ] && echo PRESENT || echo ABSENT)"
  done
  echo "A4 card + FROZEN_SHA: $([ -f "$CARD" ] && [ -f "$MC/a4_prereg/FROZEN_SHA.txt" ] && echo PRESENT || echo ABSENT)   card sha256 $(sha256sum "$CARD" 2>/dev/null | cut -c1-8) (want ${CARD_SHA:0:8})"
  echo "lfo_build/PINS.txt: $([ -f "$BUILD_PINS" ] && echo PRESENT || echo ABSENT)   patched_tool pin: $([ -n "$(bpin patched_tool)" ] && echo PRESENT || echo ABSENT)   tree tool: $([ -f "$TOOL" ] && sha256sum "$TOOL" | cut -c1-16 || echo ABSENT)   tree model: $([ -f "$TREE/stack/tanitad/models/v8f_mini.py" ] && sha256sum "$TREE/stack/tanitad/models/v8f_mini.py" | cut -c1-16 || echo ABSENT)   PINS.txt: $([ -f "$PK/PINS.txt" ] && echo PRESENT || echo ABSENT)"
  exit 0
fi

if [ -f "$PK/raw/LFO.running" ]; then
  op=$(cat "$PK/raw/LFO.running" | tr -d '\r\n ')
  if [ -n "$op" ] && kill -0 "$op" 2>/dev/null; then log "ZZLFO-ALREADY-RUNNING pid $op: not starting a second chain"; exit 9; fi
  mv "$PK/raw/LFO.running" "$PK/raw/_stale_LFO.running_$(date +%H%M%S)"
fi
for m in LFO.done LFO.fail LFO_ROLLOUTS.done; do [ -f "$PK/raw/$m" ] && mv "$PK/raw/$m" "$PK/raw/_old_${m}_$(date +%H%M%S)"; done
echo $$ > "$PK/raw/LFO.running"
on_exit() {
  local rc=$?
  if [ ! -f "$PK/raw/LFO.done" ] && [ ! -f "$PK/raw/LFO.fail" ]; then echo "$(date +%FT%T) the chain exited without a done marker at step '$STEP' (rc=$rc)" > "$PK/raw/LFO.fail"; fi
  echo "exited $(date +%FT%T) rc=$rc" > "$PK/raw/LFO.running"
}
trap on_exit EXIT
trap 'exit 143' TERM INT HUP

# ---- gate 0: pins, tests, known answers
STEP=gate0
[ -f "$PK/PINS.txt" ] || fail "PINS.txt is absent (run: python -I code/make_pins.py)" 2
for f in $(sed -n 's#^\(code/[^ ]*\|tests/[^ ]*\|lfo_chain[.]sh\|lfo_chain[.]cmd\) sha256 .*#\1#p' "$PK/PINS.txt" | tr -d '\r'); do
  want=$(pin "$f"); got=$(sha256sum "$PK/$f" | cut -c1-64)
  [ -n "$want" ] && [ "$want" = "$got" ] || fail "$f sha $got != PINS.txt ($want)" 2
done
for need in code/lfo_read.py code/lfo_precheck.py tests/test_lfo_read.py tests/test_lfo_chain_gates.py lfo_chain.sh; do
  [ -n "$(pin "$need")" ] || fail "PINS.txt carries no pin for $need" 2
done
check_shared() {   # key file
  local want got; want=$(pin "$1"); [ -n "$want" ] || fail "PINS.txt carries no pin for $1" 2
  [ -f "$2" ] || fail "shared file for $1 is absent: $2" 2
  got=$(sha256sum "$2" | cut -c1-64); [ "$want" = "$got" ] || fail "$1 ($2) sha $got != PINS.txt ($want)" 2
}
check_shared a4_card "$CARD"
check_shared lf_read_py "$MC/lf_read/code/lf_read.py"
check_shared lf_precheck_py "$MC/lf_read/code/lf_precheck.py"
check_shared llr_read_py "$MC/llr_read/code/llr_read.py"
check_shared lf_read_json "$MC/lf_read/raw/LF_READ.json"
[ "$(sha256sum "$CARD" | cut -c1-64)" = "$CARD_SHA" ] || fail "A4 card sha != $CARD_SHA" 2
grep -q "^PREREG_LEVELFLOOR_A4.md $CARD_SHA" "$MC/a4_prereg/FROZEN_SHA.txt" || fail "FROZEN_SHA.txt does not carry the A4 card sha" 2
[ -f "$BUILD_PINS" ] || fail "lfo_build/PINS.txt is absent: the LFO tool's pin cannot be checked" 2
want=$(bpin patched_tool); [ -n "$want" ] || fail "lfo_build/PINS.txt carries no patched_tool key" 2
[ -f "$TOOL" ] || fail "the tree's tool is absent: $TOOL" 2
got=$(sha256sum "$TOOL" | cut -c1-64); [ "$want" = "$got" ] || fail "tree tool sha $got != lfo_build/PINS.txt patched_tool ($want)" 2
wantop=$(bpin level_floor); [ -n "$wantop" ] || fail "lfo_build/PINS.txt carries no level_floor key" 2
[ -f "$OPR" ] || fail "the tree's level_floor.py is absent: $OPR" 2
gotop=$(sha256sum "$OPR" | cut -c1-64); [ "$wantop" = "$gotop" ] || fail "tree level_floor.py sha $gotop != lfo_build/PINS.txt level_floor ($wantop)" 2
for kv in "stack/tanitad/models/v8f_mini.py:model" "stack/scripts/v8f_mini_train.py:trainer" "stack/tanitad/data/v8f_mini_data.py:data"; do
  f=${kv%%:*}; k=${kv##*:}; want=$(lpin "$k"); [ -f "$TREE/$f" ] || fail "LFO tree $k ($f) is absent" 2
  got=$(sha256sum "$TREE/$f" | cut -c1-64)
  [ -n "$want" ] && [ "$want" = "$got" ] || fail "LFO tree $k ($f) sha $got != the LLR read's pin ($want)" 2
done
green() {   # logfile label
  local last; last=$(grep -v '^[[:space:]]*$' "$1" 2>/dev/null | tail -n 1)
  case "$last" in *" passed"*) case "$last" in *failed*|*error*) fail "unit tests not green ($2): $last" 2;; esac;; *) fail "unit tests not green ($2): $last" 2;; esac
}
green "$BUILD_DIR/logs/pytest_lfo.log" lfo_build/test_lfo
green "$BUILD_DIR/logs/pytest_lfsuite_on_lfo.log" lf_build_suite_on_the_lfo_tool
green "$PK/logs/pytest_lfo_read.log" lfo_read/test_lfo_read
green "$PK/logs/pytest_lfo_chain_gates.log" lfo_read/test_lfo_chain_gates
grep -q "^MUTATIONS ALL RED" "$BUILD_DIR/logs/MUTATIONS_LFO.txt" || fail "the build mutation log does not carry MUTATIONS ALL RED" 2
grep -q "^CARD MUTATIONS (m1, m2) FAIL KA-O3" "$BUILD_DIR/logs/MUTATIONS_LFO.txt" || fail "the build mutation log does not say the card mutations fail KA-O3" 2
grep -q "^MUTATIONS ALL RED" "$PK/logs/mutations_lfo.log" || fail "the reader mutation log does not carry MUTATIONS ALL RED" 2
"$PY" -I -c "
import json,sys
pk,bd,tool=sys.argv[1],sys.argv[2],sys.argv[3]
d=json.load(open(bd+'/raw/KA_LFO.json',encoding='utf-8'))
if d.get('PASS') is not True or set(d.get('cases',{}))!={'O1','O2'} or not all(v.get('PASS') is True for v in d['cases'].values()): print('KA-O1/KA-O2 NOT PASS'); sys.exit(1)
if d.get('tool_sha256')!=tool: print('KA_LFO.json was run with another tool:',d.get('tool_sha256')); sys.exit(1)
s=json.load(open(bd+'/raw/SMOKE_LFO.json',encoding='utf-8'))
if s.get('PASS') is not True: print('SMOKE_LFO NOT PASS'); sys.exit(1)
for m in ('ka-lf','ka-null'):
    r=json.load(open(pk+'/raw/KA_READER_'+m+'.json',encoding='utf-8'))
    if r.get('KA_PASS') is not True: print('reader',m,'NOT PASS'); sys.exit(1)
" "$PKW" "$BUILD_DIRW" "$(bpin patched_tool)" || fail "a known answer is not PASS or was run with another tool (KA-O1, KA-O2, smoke, reader ka-lf, ka-null)" 2
log "ZZLFO-GATE0 pins, tests, mutations, known answers ok (LFO tool $(bpin patched_tool | cut -c1-8), operator $(bpin level_floor | cut -c1-8), model $(lpin model | cut -c1-8))"

# ---- gate 1: checkpoints (REFUSED when absent)
STEP=gate1
"$PY" -I "$PKW/code/lfo_precheck.py" > "$PK/logs/precheck.out" 2>&1; rc=$?
cat "$PK/logs/precheck.out" | tee -a "$PK/logs/chain.out"
[ $rc -eq 0 ] || fail "checkpoint controls refuse the read (rc $rc): see logs/precheck.out and raw/LFO_PRECHECK.json" "$rc"
log "ZZLFO-GATE1 C0 and LLR checkpoints present, sha256 == the banked ones == SHA256SUMS.txt"

# ---- gate 2: reused arrays and the cards
STEP=gate2
"$PY" -I -c "
import sys
import importlib.util
spec = importlib.util.spec_from_file_location('lfo_read', sys.argv[1] + '/lfo_read.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.init()
bad = []
for s in m.SEEDS:
    _, c = m.reused_arrays(s)
    bad += [k for k, v in c.items() if not v['PASS']]
if m.LR.card_rules_sha() != m.LR.CARD_RULES_SHA: bad.append('CL card rules sha')
if m.sha256_file(m.CARD) != m.CARD_SHA or m.frozen_card_sha() != m.CARD_SHA: bad.append('A4 card sha')
if m.sha256_file(m.A3_CARD) != m.A3_CARD_SHA: bad.append('A3 card sha')
if m.sha256_file(m.LF_READ_JSON) != m.LF_READ_JSON_SHA: bad.append('LF_READ.json sha')
print('REUSED/CARDS', 'BAD ' + str(bad) if bad else 'ok')
sys.exit(1 if bad else 0)" "$PKW/code" > "$PK/logs/gate2.out" 2>&1 || fail "gate 2 (reused arrays / cards / reader pins): $(tail -n 2 "$PK/logs/gate2.out" | tr '\n' ' ')" 6
log "ZZLFO-GATE2 reused arrays (10 base + 4 LF075), the CL card's rules, the A3 and A4 cards (8bfbc386) and the pinned readers match their pins"

# ---- gate 3: CPU
STEP=gate3
busy() { powershell -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'v8f_t1_eval_[a-z_0-9]+[.]py\S*\s+run|extract_smooth[.]py|run_tool_subset[.]py|cl_readout[.]py|llr_readout[.]py|ka_lfo[.]py' -and \$_.ProcessId -ne \$PID }).Count" | tr -d '\r '; }
w=0; while [ "$(busy)" != "0" ] && [ -n "$(busy)" ]; do sleep 30; w=$((w+30)); [ $w -ge 10800 ] && { log "ZZLFO-BUSY-TIMEOUT another rollout or extractor still runs after 3 h; proceeding"; break; }; done
[ $w -gt 0 ] && log "ZZLFO-CPU-WAIT ${w}s"

run_tool() {   # tag seed outdir threads extra-flags... (stdin closed so a tool never eats the table)
  local tag=$1 s=$2 od=$3 th=$4; shift 4
  OMP_NUM_THREADS=$th MKL_NUM_THREADS=$th OPENBLAS_NUM_THREADS=$th CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
  "$PY" -I "$TOOLW" run --ckpt "$RDBW/M2b_s${s}_recft_C0/ckpt_last.pt" --tag "$tag" --tracks "$ART/sfresh_inputs/inputs/native_tracks.pt" --eval-cache "$ART/sfresh_inputs/cache" \
     --eval-grid "$DATA/eval_grid_windows.json" --eval-join "$DATA/join_eval141.npz" --out-dir "$od" \
     --replan-steps 10 --nav deadreckon --schedule periodic --limit causal --dump-ticks "$od/ticks" --dump-heads --device cpu \
     --level-floor on --floor-kappa "$KAPPA" --floor-ckpt "$RDBW/M2b_s${s}_recft_LLR/ckpt_last.pt" --floor-vhat-source logged "$@" < /dev/null
}
STEP=inputs
[ -f "$ART/sfresh_inputs/inputs/native_tracks.pt" ] || fail "input native_tracks.pt absent" 5
[ -d "$ART/sfresh_inputs/cache/eval/clips" ] || fail "eval cache absent" 5
for f in eval_grid_windows.json join_eval141.npz; do [ -f "$DATA/$f" ] || fail "input $f absent from the data dir" 5; done

# ---- gate 4: pre-flight (6 windows, oracle ON)
pre() {   # tag seed threads flags...
  local tag=$1 s=$2 th=$3; shift 3; STEP="preflight $tag"; rm -rf "$S/smoke/$tag"; mkdir -p "$S/smoke/$tag"
  run_tool "$tag" "$s" "$SW/smoke/$tag" "$th" --max-windows 6 "$@" > "$PK/logs/smoke_$tag.out" 2>&1; local rc=$?
  [ $rc -eq 0 ] && [ -f "$S/smoke/$tag/ticks/ticks_$tag.npz" ] || fail "pre-flight $tag rc=$rc (the LFO tool refuses a flag, or the C0 / LLR checkpoints do not load): see logs/smoke_$tag.out" 7
  "$PY" -I -c "
import sys
import numpy as np
z = np.load(sys.argv[1], allow_pickle=True)
need = ('lf_bind', 'lf_cap', 'lf_vf', 'lf_vhat', 'lf_pi', 'lf_gap', 'lf_lift', 'lf_vhat_heads')
ok = all(k in z.files for k in need) and len(z['lf_bind']) > 0 and str(z['floor_vhat_source']) == 'logged' and str(z['label']) == 'CEILING'
sys.exit(0 if ok else 1)" "$SW/smoke/$tag/ticks/ticks_$tag.npz" || fail "pre-flight $tag: the ticks file lacks the lf_* fields, lf_vhat_heads, or the logged / CEILING labels" 7
  log "ZZLFO-PREFLIGHT-OK $tag"
}
pre PRE_LFO_s0_ri 0 "$TH" --smooth sg72 --scene reindexed
pre PRE_LFO_s1_ri_kbox 1 "$TH" --smooth sg72 --scene reindexed --feasible kbox
STEP=rollouts
log "ZZLFO-START LFO tool $(bpin patched_tool | cut -c1-8) threads=$TH kappa=$KAPPA --floor-vhat-source logged (CEILING)"

run_one() {   # tag seed scene feas
  local tag=$1 s=$2 scene=$3 feas=$4 ck="$RDBW/M2b_s${2}_recft_C0/ckpt_last.pt"
  [ -f "$OUT/done_$tag.txt" ] && { log "ZZLFO-SKIP $tag"; return 0; }
  [ -d "$OUT/$tag" ] && mv "$OUT/$tag" "$OUT/_incomplete_${tag}_$(date +%H%M%S)"
  mkdir -p "$OUT/$tag"
  STEP="rollout $tag"
  local flags="--smooth sg72 --scene $scene"
  [ "$feas" != "off" ] && flags="$flags --feasible $feas"
  local ffl="--level-floor on --floor-kappa $KAPPA --floor-ckpt $RDBW/M2b_s${s}_recft_LLR/ckpt_last.pt --floor-vhat-source logged"
  local t0=$(date +%s)
  log "ZZLFO-RUN $tag flags=$flags $ffl threads=$TH"
  # shellcheck disable=SC2086
  run_tool "$tag" "$s" "$OUTW/$tag" "$TH" $flags > "$PK/logs/run_$tag.out" 2>&1; local rc=$?
  if [ $rc -eq 0 ]; then
    printf '{"tag": "%s", "flags": "%s %s", "ckpt": "%s", "ckpt_sha256": "%s", "tool": "%s", "tool_sha256": "%s", "threads": %s, "wall_s": %s}\n' "$tag" "$flags" "$ffl" "$ck" \
       "$(sha256sum "$RDB/M2b_s${s}_recft_C0/ckpt_last.pt" | cut -c1-64)" "$TOOLW" "$(bpin patched_tool)" "$TH" "$(( $(date +%s) - t0 ))" > "$OUT/done_$tag.txt"
    log "ZZLFO-RUN-DONE $tag wall=$(( $(date +%s) - t0 ))s"
    "$PY" -I "$PKW/code/lfo_precheck.py" --rollout "$tag" --out-root "$OUTW" > "$PK/logs/rollout_check_$tag.out" 2>&1; local crc=$?
    cat "$PK/logs/rollout_check_$tag.out" | tee -a "$PK/logs/chain.out"
    if [ $crc -ne 0 ]; then mv "$OUT/done_$tag.txt" "$OUT/done_$tag.txt.REFUSED"; fail "rollout $tag: the output JSON's floor keys / CEILING labels / planner checkpoint refuse (rc $crc): see logs/rollout_check_$tag.out" 8; fi
  else
    log "ZZLFO-RUN-FAIL $tag rc=$rc"
  fi
  return $rc
}
nfail=0
while IFS=$'\t' read -r tag seed scene feas <&3; do
  run_one "$tag" "$seed" "$scene" "$feas" || nfail=$((nfail+1))
done 3< <(table)
log "ZZLFO-ROLLOUTS-DONE nfail=$nfail"
missing=""; while IFS=$'\t' read -r tag rest; do [ -f "$OUT/done_$tag.txt" ] || missing="$missing $tag"; done < <(table)
[ -z "$missing" ] || fail "missing done markers:$missing (see logs/run_<tag>.out)" 3
echo "$(date +%FT%T) four LFO rollouts done" > "$PK/raw/LFO_ROLLOUTS.done"
log "ZZLFO-ROLLOUTS-OK"

# ---- the reader (controls first; a refused control is a result to report)
STEP=reader
log "ZZLFO-READ start"
"$PY" -I "$PKW/code/lfo_read.py" --mode production --out-root "$OUTW" --work-dir "$SW/work" --raw "$PKW/raw" --workers 4 > "$PK/logs/read.out" 2>&1; rc=$?
log "ZZLFO-READ rc=$rc"
tail -n 3 "$PK/logs/read.out" | tee -a "$PK/logs/chain.out"
if [ $rc -eq 0 ] && [ -f "$PK/raw/LFO_READ.json" ] && [ -f "$PK/RESULTS.md" ]; then
  echo "$(date +%FT%T) LFO_READ.json and RESULTS.md written" > "$PK/raw/LFO.done"
  log "ZZLFO-READ-DONE"
  log "ZZLFO-CHAIN-DONE"
else
  echo "$(date +%FT%T) step 'reader': reader rc=$rc (see logs/read.out; a REFUSED control names itself in raw/LFO_READ.json 'refused')" > "$PK/raw/LFO.fail"
  log "ZZLFO-READ-FAIL rc=$rc"
  exit "$rc"
fi
