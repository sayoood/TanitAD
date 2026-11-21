#!/usr/bin/env bash
# LF5 chain (PREREG_LEVELFLOOR_A5.md, sha256 2aac1dbe..., frozen 04:39:36 Europe/Berlin): the level floor F at kappa 1.0 -- LFO100 (logged speed, CEILING), LFN100x10 / LFN100x20 (logged speed under a priori noise sigma 0.10 / 0.20, generator seed 20261106, CEILING),
# LF100 (the trained LLR heads, the ADMISSIBLE arm) -- 20 rollouts on ALL 2,399 eval windows, then the reader. Dev-box CPU, ZERO training, no ssh, ONE rollout at a time. Cloned from lfo_read/lfo_chain.sh (gate / marker / EXIT-trap discipline).
# Launched through WMI Win32_Process.Create (lf5_chain.cmd) so it survives the caller.
#   usage:  lf5_chain.sh --dry-run     lists the 20 rollouts, the gates and what is present; runs nothing, writes no marker
#           lf5_chain.sh               (normal; RESUME = launch the .cmd again, finished rollouts are skipped)
# READ TREE: F:/TanitAD-scratch/motion_c/lf5_read/tree = the LFO read tree + the LF5 tool v8f_t1_eval_ri.py (its sha256 is `patched_tool` in motion_c/lf5_build/PINS.txt; gate 0 refuses when it differs) + level_floor.py (unchanged).
# Rollouts in PRIORITY order (a killed chain yields value first): RILF100_s0 RILF100_s1 RILF100C_s0 RILF100C_s1 (heads, the admissible arm, reindexed), RILFO100 x4, RILFN100x10 x4, RILFN100x20 x4 (logged source; bare = --smooth sg72, composite = + --feasible kbox; reindexed),
#   then LF100 x4 on `logged` (heads; the LEAK-BORNE rule), then the reader. Flags = the banked RIC0 / RIC0C flags + `--level-floor on --floor-kappa 1.0 --floor-ckpt LLR --floor-vhat-source SRC` (+ `--floor-vhat-noise SIGMA --floor-vhat-noise-seed 20261106`).
# Gates (REFUSE = raw/LF5.fail with the reason, exit code):
#   0  PINS.txt: reader, pre-check, tests, this script; the shared files (A5 card, lfo_read.py, lfo_precheck.py, lf_read.py, lf_precheck.py, llr_read.py, LFO_READ.json) match their sha256; the tree's model / data / trainer (= the LLR read's pins), the LF5 tool
#      (= lf5_build/PINS.txt patched_tool) and level_floor.py match; unit-test logs green; mutation logs ALL RED; KA_LF5.json PASS with exactly this tool; the reader's ka-a4 PASS                                         (exit 2)
#   1  lf5_precheck.py: both seeds' C0 and LLR checkpoints present and equal to the banked sha256                                                                                                                 (exit 5 absent, 8 failed)
#   2  the reused arrays' sha256 (10 base + 4 RILFO075), the cards and the pinned readers                                                                                                                         (exit 6)
#   3  no other tool rollout / extractor runs (CPU shared; waits at most 3 h)
#   4  a 6-window PRE-FLIGHT, heads and noise paths                                                                                                                                                               (exit 7)
# After every finished rollout lf5_precheck.py --rollout TAG checks the output JSON (kappa 1.0, floor_ckpt sha256 == the seed's LLR sha, source, noise level and seed, label) and the planner sha (exit 8, chain stops).
# Markers (raw/): LF5.running (pid), LF5_ROLLOUTS.done, then LF5.done (raw/LF5_READ.json and RESULTS.md written) or LF5.fail (the failing step). A chain that dies for any reason a shell can trap (error, TERM, INT, HUP) leaves LF5.fail naming the step.
# WRITES: this package (logs, raw, RESULTS.md) and F:/TanitAD-scratch/motion_c/lf5_read/{out,smoke,work}. The tree is READ ONLY here.
set -u
DRY=0; for t in "$@"; do [ "$t" = "--dry-run" ] && DRY=1; done
PK=${LF5_PK:-/e/Projects/TanitAD-artifacts/dagland/motion_c/lf5_read}
PKW=${LF5_PKW:-E:/Projects/TanitAD-artifacts/dagland/motion_c/lf5_read}
S=/f/TanitAD-scratch/motion_c/lf5_read
SW=F:/TanitAD-scratch/motion_c/lf5_read
TREE=${LF5_TREE:-$S/tree}
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
CARD=$MC/a5_prereg/PREREG_LEVELFLOOR_A5.md
CARD_SHA=2aac1dbea651ac82f599d874fce48f26d2cd83d19a8ec62f3353a28389333310
BUILD_PINS=${LF5_BUILD_PINS_FILE:-$MC/lf5_build/PINS.txt}
BUILD_DIR=${LF5_BUILD_DIR:-$MC/lf5_build}
BUILD_DIRW=${LF5_BUILD_DIRW:-$MCW/lf5_build}
LLR_PINS=$MC/llr_read/PINS.txt
KAPPA=1.0
NOISE_SEED=20261106
TH=${LF5_TH:-12}
mkdir -p "$OUT" "$PK/logs" "$PK/raw" "$S/smoke" "$S/work"
log() { echo "$(date +%FT%T) $*" | tee -a "$PK/logs/chain.out"; }
STEP=start
fail() { log "ZZLF5-FAIL $1"; echo "$(date +%FT%T) step '$STEP': $1" > "$PK/raw/LF5.fail"; exit "${2:-3}"; }
table() {   # tag seed scene feasible source sigma   (PRIORITY order; tab-separated; sigma "-" = none)
  local b c s sc fe src sg
  for spec in "RILF100 reindexed heads -" "RILFO100 reindexed logged -" "RILFN100x10 reindexed logged 0.10" "RILFN100x20 reindexed logged 0.20" "LF100 logged heads -"; do
    set -- $spec; b=$1; sc=$2; src=$3; sg=$4
    for c in "" C; do
      for s in 0 1; do
        fe=off; [ -n "$c" ] && fe=kbox
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${b}${c}_s${s}" "$s" "$sc" "$fe" "$src" "$sg"
      done
    done
  done
}
pin() { sed -n "s#^$1 sha256 \([0-9a-f]*\)\$#\1#p" "$PK/PINS.txt" | tr -d '\r'; }
bpin() { grep -E "^$1([[:space:]:=]|\$)" "$BUILD_PINS" 2>/dev/null | grep -oE '[0-9a-f]{64}' | head -n 1; }
lpin() { sed -n "s#^$1 sha256 \([0-9a-f]*\)\$#\1#p" "$LLR_PINS" | tr -d '\r'; }

if [ $DRY -eq 1 ]; then
  echo "LF5 chain, DRY RUN (nothing runs, no marker is written)"
  echo "rollouts in priority order (tag seed scene feasible source sigma; threads $TH; kappa $KAPPA):"; table
  for s in 0 1; do
    echo "seed $s: C0 $([ -f "$RDB/M2b_s${s}_recft_C0/ckpt_last.pt" ] && echo PRESENT || echo ABSENT)   LLR $([ -f "$RDB/M2b_s${s}_recft_LLR/ckpt_last.pt" ] && echo PRESENT || echo ABSENT)   banked arrays (C0 C0C C0REP RIC0 RIC0C RILF075 RILFO075 + C): $([ -f "$MC/../recover_next/read/raw/arrays/C0_s$s.npz" ] && [ -f "$MC/ri_read/raw/arrays/RIC0_s$s.npz" ] && [ -f "$MC/lf_read/raw/arrays/RILF075C_s$s.npz" ] && [ -f "$MC/lfo_read/raw/arrays/RILFO075C_s$s.npz" ] && echo PRESENT || echo ABSENT)"
  done
  echo "A5 card + FROZEN_SHA: $([ -f "$CARD" ] && [ -f "$MC/a5_prereg/FROZEN_SHA.txt" ] && echo PRESENT || echo ABSENT)   card sha256 $(sha256sum "$CARD" 2>/dev/null | cut -c1-8) (want ${CARD_SHA:0:8})"
  echo "lf5_build/PINS.txt: $([ -f "$BUILD_PINS" ] && echo PRESENT || echo ABSENT)   patched_tool pin: $([ -n "$(bpin patched_tool)" ] && echo PRESENT || echo ABSENT)   tree tool: $([ -f "$TOOL" ] && sha256sum "$TOOL" | cut -c1-16 || echo ABSENT)   PINS.txt: $([ -f "$PK/PINS.txt" ] && echo PRESENT || echo ABSENT)"
  exit 0
fi

if [ -f "$PK/raw/LF5.running" ]; then
  op=$(cat "$PK/raw/LF5.running" | tr -d '\r\n ')
  if [ -n "$op" ] && kill -0 "$op" 2>/dev/null; then log "ZZLF5-ALREADY-RUNNING pid $op: not starting a second chain"; exit 9; fi
  mv "$PK/raw/LF5.running" "$PK/raw/_stale_LF5.running_$(date +%H%M%S)"
fi
for m in LF5.done LF5.fail LF5_ROLLOUTS.done; do [ -f "$PK/raw/$m" ] && mv "$PK/raw/$m" "$PK/raw/_old_${m}_$(date +%H%M%S)"; done
echo $$ > "$PK/raw/LF5.running"
on_exit() {
  local rc=$?
  if [ ! -f "$PK/raw/LF5.done" ] && [ ! -f "$PK/raw/LF5.fail" ]; then echo "$(date +%FT%T) the chain exited without a done marker at step '$STEP' (rc=$rc)" > "$PK/raw/LF5.fail"; fi
  echo "exited $(date +%FT%T) rc=$rc" > "$PK/raw/LF5.running"
}
trap on_exit EXIT
trap 'exit 143' TERM INT HUP

# ---- gate 0: pins, tests, known answers
STEP=gate0
[ -f "$PK/PINS.txt" ] || fail "PINS.txt is absent (run: python -I code/make_pins.py)" 2
for f in $(sed -n 's#^\(code/[^ ]*\|tests/[^ ]*\|lf5_chain[.]sh\|lf5_chain[.]cmd\) sha256 .*#\1#p' "$PK/PINS.txt" | tr -d '\r'); do
  want=$(pin "$f"); got=$(sha256sum "$PK/$f" | cut -c1-64)
  [ -n "$want" ] && [ "$want" = "$got" ] || fail "$f sha $got != PINS.txt ($want)" 2
done
for need in code/lf5_read.py code/lf5_precheck.py tests/test_lf5_read.py lf5_chain.sh; do
  [ -n "$(pin "$need")" ] || fail "PINS.txt carries no pin for $need" 2
done
check_shared() {   # key file
  local want got; want=$(pin "$1"); [ -n "$want" ] || fail "PINS.txt carries no pin for $1" 2
  [ -f "$2" ] || fail "shared file for $1 is absent: $2" 2
  got=$(sha256sum "$2" | cut -c1-64); [ "$want" = "$got" ] || fail "$1 ($2) sha $got != PINS.txt ($want)" 2
}
check_shared a5_card "$CARD"
check_shared lfo_read_py "$MC/lfo_read/code/lfo_read.py"
check_shared lfo_precheck_py "$MC/lfo_read/code/lfo_precheck.py"
check_shared lf_read_py "$MC/lf_read/code/lf_read.py"
check_shared lf_precheck_py "$MC/lf_read/code/lf_precheck.py"
check_shared llr_read_py "$MC/llr_read/code/llr_read.py"
check_shared lfo_read_json "$MC/lfo_read/raw/LFO_READ.json"
[ "$(sha256sum "$CARD" | cut -c1-64)" = "$CARD_SHA" ] || fail "A5 card sha != $CARD_SHA" 2
grep -q "^PREREG_LEVELFLOOR_A5.md $CARD_SHA" "$MC/a5_prereg/FROZEN_SHA.txt" || fail "FROZEN_SHA.txt does not carry the A5 card sha" 2
[ -f "$BUILD_PINS" ] || fail "lf5_build/PINS.txt is absent: the LF5 tool's pin cannot be checked" 2
want=$(bpin patched_tool); [ -n "$want" ] || fail "lf5_build/PINS.txt carries no patched_tool key" 2
[ -f "$TOOL" ] || fail "the tree's tool is absent: $TOOL" 2
got=$(sha256sum "$TOOL" | cut -c1-64); [ "$want" = "$got" ] || fail "tree tool sha $got != lf5_build/PINS.txt patched_tool ($want)" 2
wantop=$(bpin level_floor); [ -n "$wantop" ] || fail "lf5_build/PINS.txt carries no level_floor key" 2
[ -f "$OPR" ] || fail "the tree's level_floor.py is absent: $OPR" 2
gotop=$(sha256sum "$OPR" | cut -c1-64); [ "$wantop" = "$gotop" ] || fail "tree level_floor.py sha $gotop != lf5_build/PINS.txt level_floor ($wantop)" 2
for kv in "stack/tanitad/models/v8f_mini.py:model" "stack/scripts/v8f_mini_train.py:trainer" "stack/tanitad/data/v8f_mini_data.py:data"; do
  f=${kv%%:*}; k=${kv##*:}; want=$(lpin "$k"); [ -f "$TREE/$f" ] || fail "LF5 tree $k ($f) is absent" 2
  got=$(sha256sum "$TREE/$f" | cut -c1-64)
  [ -n "$want" ] && [ "$want" = "$got" ] || fail "LF5 tree $k ($f) sha $got != the LLR read's pin ($want)" 2
done
green() {   # logfile label
  local last; last=$(grep -v '^[[:space:]]*$' "$1" 2>/dev/null | tail -n 1)
  case "$last" in *" passed"*) case "$last" in *failed*|*error*) fail "unit tests not green ($2): $last" 2;; esac;; *) fail "unit tests not green ($2): $last" 2;; esac
}
green "$BUILD_DIR/logs/pytest_lf5.log" lf5_build/test_lf5
green "$PK/logs/pytest_lf5_read.log" lf5_read/test_lf5_read
grep -q "^MUTATIONS ALL RED" "$BUILD_DIR/logs/MUTATIONS_LF5.txt" || fail "the build mutation log does not carry MUTATIONS ALL RED" 2
grep -q "^MUTATIONS ALL RED" "$PK/logs/mutations_lf5_read.log" || fail "the reader mutation log does not carry MUTATIONS ALL RED" 2
"$PY" -I -c "
import json,sys
pk,bd,tool=sys.argv[1],sys.argv[2],sys.argv[3]
d=json.load(open(bd+'/raw/KA_LF5.json',encoding='utf-8'))
if d.get('PASS') is not True or set(d.get('cases',{}))!={'N1','N2','N3'} or not all(v.get('PASS') is True for v in d['cases'].values()): print('KA-N1/N2/N3 NOT PASS'); sys.exit(1)
if d.get('tool_sha256')!=tool: print('KA_LF5.json was run with another tool:',d.get('tool_sha256')); sys.exit(1)
r=json.load(open(pk+'/raw/KA_READER_ka-a4.json',encoding='utf-8'))
if r.get('KA_PASS') is not True: print('reader ka-a4 NOT PASS'); sys.exit(1)
" "$PKW" "$BUILD_DIRW" "$(bpin patched_tool)" || fail "a known answer is not PASS or was run with another tool (KA-N1, KA-N2, KA-N3, reader ka-a4)" 2
log "ZZLF5-GATE0 pins, tests, mutations, known answers ok (LF5 tool $(bpin patched_tool | cut -c1-8), operator $(bpin level_floor | cut -c1-8), model $(lpin model | cut -c1-8))"

# ---- gate 1: checkpoints (REFUSED when absent)
STEP=gate1
"$PY" -I "$PKW/code/lf5_precheck.py" > "$PK/logs/precheck.out" 2>&1; rc=$?
cat "$PK/logs/precheck.out" | tee -a "$PK/logs/chain.out"
[ $rc -eq 0 ] || fail "checkpoint controls refuse the read (rc $rc): see logs/precheck.out and raw/LF5_PRECHECK.json" "$rc"
log "ZZLF5-GATE1 C0 and LLR checkpoints present, sha256 == the banked ones == SHA256SUMS.txt"

# ---- gate 2: reused arrays and the cards
STEP=gate2
"$PY" -I -c "
import sys
import importlib.util
spec = importlib.util.spec_from_file_location('lf5_read', sys.argv[1] + '/lf5_read.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.init()
bad = []
for s in m.SEEDS:
    _, c = m.reused_arrays(s)
    bad += [k for k, v in c.items() if not v['PASS']]
if m.LR.card_rules_sha() != m.LR.CARD_RULES_SHA: bad.append('CL card rules sha')
if m.sha256_file(m.CARD) != m.CARD_SHA or m.frozen_card_sha() != m.CARD_SHA: bad.append('A5 card sha')
if m.sha256_file(m.LFO_READ_JSON) != m.LFO_READ_JSON_SHA: bad.append('LFO_READ.json sha')
print('REUSED/CARDS', 'BAD ' + str(bad) if bad else 'ok')
sys.exit(1 if bad else 0)" "$PKW/code" > "$PK/logs/gate2.out" 2>&1 || fail "gate 2 (reused arrays / cards / reader pins): $(tail -n 2 "$PK/logs/gate2.out" | tr '\n' ' ')" 6
log "ZZLF5-GATE2 reused arrays (10 base + 4 RILFO075), the CL card's rules, the A5 card (2aac1dbe) and the pinned readers match their pins"

# ---- gate 3: CPU
STEP=gate3
busy() { powershell -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'v8f_t1_eval_[a-z_0-9]+[.]py\S*\s+run|extract_smooth[.]py|run_tool_subset[.]py|cl_readout[.]py|llr_readout[.]py|ka_lf5[.]py|ka_lfo[.]py' -and \$_.ProcessId -ne \$PID }).Count" | tr -d '\r '; }
w=0; while [ "$(busy)" != "0" ] && [ -n "$(busy)" ]; do sleep 30; w=$((w+30)); [ $w -ge 10800 ] && { log "ZZLF5-BUSY-TIMEOUT another rollout or extractor still runs after 3 h; proceeding"; break; }; done
[ $w -gt 0 ] && log "ZZLF5-CPU-WAIT ${w}s"

run_tool() {   # tag seed outdir threads source sigma extra-flags... (stdin closed so a tool never eats the table)
  local tag=$1 s=$2 od=$3 th=$4 src=$5 sg=$6; shift 6
  local nz=""; [ "$sg" != "-" ] && nz="--floor-vhat-noise $sg --floor-vhat-noise-seed $NOISE_SEED"
  # shellcheck disable=SC2086
  OMP_NUM_THREADS=$th MKL_NUM_THREADS=$th OPENBLAS_NUM_THREADS=$th CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
  "$PY" -I "$TOOLW" run --ckpt "$RDBW/M2b_s${s}_recft_C0/ckpt_last.pt" --tag "$tag" --tracks "$ART/sfresh_inputs/inputs/native_tracks.pt" --eval-cache "$ART/sfresh_inputs/cache" \
     --eval-grid "$DATA/eval_grid_windows.json" --eval-join "$DATA/join_eval141.npz" --out-dir "$od" \
     --replan-steps 10 --nav deadreckon --schedule periodic --limit causal --dump-ticks "$od/ticks" --dump-heads --device cpu \
     --level-floor on --floor-kappa "$KAPPA" --floor-ckpt "$RDBW/M2b_s${s}_recft_LLR/ckpt_last.pt" --floor-vhat-source "$src" $nz "$@" < /dev/null
}
STEP=inputs
[ -f "$ART/sfresh_inputs/inputs/native_tracks.pt" ] || fail "input native_tracks.pt absent" 5
[ -d "$ART/sfresh_inputs/cache/eval/clips" ] || fail "eval cache absent" 5
for f in eval_grid_windows.json join_eval141.npz; do [ -f "$DATA/$f" ] || fail "input $f absent from the data dir" 5; done

# ---- gate 4: pre-flight (6 windows): the heads path and the noise path
pre() {   # tag seed threads source sigma flags...
  local tag=$1 s=$2 th=$3 src=$4 sg=$5; shift 5; STEP="preflight $tag"; rm -rf "$S/smoke/$tag"; mkdir -p "$S/smoke/$tag"
  run_tool "$tag" "$s" "$SW/smoke/$tag" "$th" "$src" "$sg" --max-windows 6 "$@" > "$PK/logs/smoke_$tag.out" 2>&1; local rc=$?
  [ $rc -eq 0 ] && [ -f "$S/smoke/$tag/ticks/ticks_$tag.npz" ] || fail "pre-flight $tag rc=$rc (the LF5 tool refuses a flag, or the C0 / LLR checkpoints do not load): see logs/smoke_$tag.out" 7
  "$PY" -I -c "
import sys
import numpy as np
z = np.load(sys.argv[1], allow_pickle=True)
src, sg = sys.argv[2], sys.argv[3]
need = ['lf_bind', 'lf_cap', 'lf_vf', 'lf_vhat', 'lf_pi', 'lf_gap', 'lf_lift'] + (['lf_vhat_heads'] if src == 'logged' else []) + (['lf_vlog', 'lf_eps'] if sg != '-' else [])
ok = all(k in z.files for k in need) and len(z['lf_bind']) > 0 and str(z['floor_vhat_source']) == src and (('label' in z.files and str(z['label']) == 'CEILING') if src == 'logged' else ('label' not in z.files))
sys.exit(0 if ok else 1)" "$SW/smoke/$tag/ticks/ticks_$tag.npz" "$src" "$sg" || fail "pre-flight $tag: the ticks file lacks the lf_* fields (lf_vhat_heads / lf_vlog / lf_eps) or the source / CEILING labels" 7
  log "ZZLF5-PREFLIGHT-OK $tag"
}
pre PRE_LF5_heads_s0 0 "$TH" heads - --smooth sg72 --scene reindexed
pre PRE_LF5_noise_s1_kbox 1 "$TH" logged 0.20 --smooth sg72 --scene reindexed --feasible kbox
STEP=rollouts
log "ZZLF5-START LF5 tool $(bpin patched_tool | cut -c1-8) threads=$TH kappa=$KAPPA noise seed $NOISE_SEED"

run_one() {   # tag seed scene feas source sigma
  local tag=$1 s=$2 scene=$3 feas=$4 src=$5 sg=$6 ck="$RDBW/M2b_s${2}_recft_C0/ckpt_last.pt"
  [ -f "$OUT/done_$tag.txt" ] && { log "ZZLF5-SKIP $tag"; return 0; }
  [ -d "$OUT/$tag" ] && mv "$OUT/$tag" "$OUT/_incomplete_${tag}_$(date +%H%M%S)"
  mkdir -p "$OUT/$tag"
  STEP="rollout $tag"
  local flags="--smooth sg72 --scene $scene"
  [ "$feas" != "off" ] && flags="$flags --feasible $feas"
  local ffl="--level-floor on --floor-kappa $KAPPA --floor-ckpt $RDBW/M2b_s${s}_recft_LLR/ckpt_last.pt --floor-vhat-source $src"
  [ "$sg" != "-" ] && ffl="$ffl --floor-vhat-noise $sg --floor-vhat-noise-seed $NOISE_SEED"
  local t0=$(date +%s)
  log "ZZLF5-RUN $tag flags=$flags $ffl threads=$TH"
  # shellcheck disable=SC2086
  run_tool "$tag" "$s" "$OUTW/$tag" "$TH" "$src" "$sg" $flags > "$PK/logs/run_$tag.out" 2>&1; local rc=$?
  if [ $rc -eq 0 ]; then
    printf '{"tag": "%s", "flags": "%s %s", "ckpt": "%s", "ckpt_sha256": "%s", "tool": "%s", "tool_sha256": "%s", "threads": %s, "wall_s": %s}\n' "$tag" "$flags" "$ffl" "$ck" \
       "$(sha256sum "$RDB/M2b_s${s}_recft_C0/ckpt_last.pt" | cut -c1-64)" "$TOOLW" "$(bpin patched_tool)" "$TH" "$(( $(date +%s) - t0 ))" > "$OUT/done_$tag.txt"
    log "ZZLF5-RUN-DONE $tag wall=$(( $(date +%s) - t0 ))s"
    "$PY" -I "$PKW/code/lf5_precheck.py" --rollout "$tag" --out-root "$OUTW" > "$PK/logs/rollout_check_$tag.out" 2>&1; local crc=$?
    cat "$PK/logs/rollout_check_$tag.out" | tee -a "$PK/logs/chain.out"
    if [ $crc -ne 0 ]; then mv "$OUT/done_$tag.txt" "$OUT/done_$tag.txt.REFUSED"; fail "rollout $tag: the output JSON's floor keys / labels / noise keys / planner checkpoint refuse (rc $crc): see logs/rollout_check_$tag.out" 8; fi
  else
    log "ZZLF5-RUN-FAIL $tag rc=$rc"
  fi
  return $rc
}
nfail=0
while IFS=$'\t' read -r tag seed scene feas src sg <&3; do
  run_one "$tag" "$seed" "$scene" "$feas" "$src" "$sg" || nfail=$((nfail+1))
done 3< <(table)
log "ZZLF5-ROLLOUTS-DONE nfail=$nfail"
missing=""; while IFS=$'\t' read -r tag rest; do [ -f "$OUT/done_$tag.txt" ] || missing="$missing $tag"; done < <(table)
[ -z "$missing" ] || fail "missing done markers:$missing (see logs/run_<tag>.out)" 3
echo "$(date +%FT%T) twenty LF5 rollouts done" > "$PK/raw/LF5_ROLLOUTS.done"
log "ZZLF5-ROLLOUTS-OK"

# ---- the reader (controls first; a refused control is a result to report)
STEP=reader
log "ZZLF5-READ start"
"$PY" -I "$PKW/code/lf5_read.py" --mode production --out-root "$OUTW" --work-dir "$SW/work" --raw "$PKW/raw" --workers 4 > "$PK/logs/read.out" 2>&1; rc=$?
log "ZZLF5-READ rc=$rc"
tail -n 3 "$PK/logs/read.out" | tee -a "$PK/logs/chain.out"
if [ $rc -eq 0 ] && [ -f "$PK/raw/LF5_READ.json" ] && [ -f "$PK/RESULTS.md" ]; then
  echo "$(date +%FT%T) LF5_READ.json and RESULTS.md written" > "$PK/raw/LF5.done"
  log "ZZLF5-READ-DONE"
  log "ZZLF5-CHAIN-DONE"
else
  echo "$(date +%FT%T) step 'reader': reader rc=$rc (see logs/read.out; a REFUSED control names itself in raw/LF5_READ.json 'refused'; the 20 rollouts are banked either way)" > "$PK/raw/LF5.fail"
  log "ZZLF5-READ-FAIL rc=$rc"
  exit "$rc"
fi
