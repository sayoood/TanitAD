#!/usr/bin/env bash
# LF6 chain (PREREG_LEVELFLOOR_A6.md, sha256 fbf5de02..., frozen 2026-11-06 06:37:47 Europe/Berlin): the bounded-lift floor F_B at kappa 1.0, v_f(tau) = min(kappa * v_hat [lead cap], v0 + 1.0 * tau, v_O(tau) + DELTA), DELTA in {0.5, 1.0} --
# LFB05 / LFB10 (the trained LLR heads, ADMISSIBLE; reindexed deciding, `logged` for the LEAK-BORNE rule) and LFOB05 / LFOB10 (the logged speed, CEILING) -- 24 rollouts on ALL 2,399 eval windows, then the reader.
# Dev-box CPU, ZERO training, no ssh, ONE rollout at a time. Cloned from lf5_read/lf5_chain.sh (gate / marker / EXIT-trap discipline). Launched through WMI Win32_Process.Create (lf6_chain.cmd) so it survives the caller.
#   usage:  lf6_chain.sh --dry-run     lists the 24 rollouts, the gates and what is present; runs nothing, writes no marker
#           lf6_chain.sh               (normal; RESUME = launch the .cmd again, finished rollouts are skipped)
# READ TREE: F:/TanitAD-scratch/motion_c/lf6_read/tree = the LF5 read tree + the LF6 tool v8f_t1_eval_ri.py (its sha256 is `patched_tool` in motion_c/lf6_build/PINS.txt; gate 0 refuses when it differs) + level_floor.py (unchanged, sha `level_floor`).
# Rollouts in PRIORITY order (a killed chain yields value first): RILFB10 (s0, s1, C s0, C s1), RILFB05 x4 (heads, the admissible arms, reindexed), RILFOB10 x4, RILFOB05 x4 (logged source, CEILING, reindexed), then LFB10 x4, LFB05 x4 on `logged` (heads; the LEAK-BORNE rule),
#   then the reader. Flags = the banked RIC0 / RIC0C flags (bare = --smooth sg72, composite = + --feasible kbox) + `--level-floor on --floor-kappa 1.0 --floor-ckpt LLR --floor-vhat-source SRC --floor-lift-cap DELTA`.
# Gates (REFUSE = raw/LF6.fail with the reason, exit code):
#   0  PINS.txt: reader, pre-check, tests, this script; the shared files (A6 card, lfo_read.py, lfo_precheck.py, lf_read.py, lf_precheck.py, llr_read.py, LFO_READ.json, LF5_READ.json) match their sha256; the tree's model / data / trainer (= the LLR read's pins), the LF6 tool
#      (= lf6_build/PINS.txt patched_tool) and level_floor.py match; unit-test logs green; mutation logs ALL RED; KA_LF6.json PASS (K1, K2, K3) with exactly this tool; the reader's ka-a5 PASS                       (exit 2)
#   1  lf6_precheck.py: both seeds' C0 and LLR checkpoints present and equal to the banked sha256                                                                                                                 (exit 5 absent, 8 failed)
#   2  the reused arrays' sha256 (10 base + 8 banked A5), the cards and the pinned readers                                                                                                                        (exit 6)
#   3  no other tool rollout / extractor runs (CPU shared; waits at most 3 h)
#   4  a 6-window PRE-FLIGHT, heads and logged paths, the ticks file loaded with allow_pickle=False (the defect of A5)                                                                                            (exit 7)
# After every finished rollout lf6_precheck.py --rollout TAG checks the output JSON (kappa 1.0, floor_ckpt sha256 == the seed's LLR sha, source, lift cap, no noise keys, label) and the planner sha (exit 8, chain stops).
# Markers (raw/): LF6.running (pid), LF6_ROLLOUTS.done, then LF6.done (raw/LF6_READ.json and RESULTS.md written) or LF6.fail (the failing step). A chain that dies for any reason a shell can trap (error, TERM, INT, HUP) leaves LF6.fail naming the step.
# WRITES: this package (logs, raw, RESULTS.md) and F:/TanitAD-scratch/motion_c/lf6_read/{out,smoke,work}. The tree is READ ONLY here.
set -u
DRY=0; for t in "$@"; do [ "$t" = "--dry-run" ] && DRY=1; done
PK=${LF6_PK:-/e/Projects/TanitAD-artifacts/dagland/motion_c/lf6_read}
PKW=${LF6_PKW:-E:/Projects/TanitAD-artifacts/dagland/motion_c/lf6_read}
S=/f/TanitAD-scratch/motion_c/lf6_read
SW=F:/TanitAD-scratch/motion_c/lf6_read
TREE=${LF6_TREE:-$S/tree}
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
CARD=$MC/a6_prereg/PREREG_LEVELFLOOR_A6.md
CARD_SHA=fbf5de02f1444e3c4b283a61c32c4be051dd146dbe4f52847df856160de92eb1
BUILD_PINS=${LF6_BUILD_PINS_FILE:-$MC/lf6_build/PINS.txt}
BUILD_DIR=${LF6_BUILD_DIR:-$MC/lf6_build}
BUILD_DIRW=${LF6_BUILD_DIRW:-$MCW/lf6_build}
LLR_PINS=$MC/llr_read/PINS.txt
KAPPA=1.0
TH=${LF6_TH:-12}
mkdir -p "$OUT" "$PK/logs" "$PK/raw" "$S/smoke" "$S/work"
log() { echo "$(date +%FT%T) $*" | tee -a "$PK/logs/chain.out"; }
STEP=start
fail() { log "ZZLF6-FAIL $1"; echo "$(date +%FT%T) step '$STEP': $1" > "$PK/raw/LF6.fail"; exit "${2:-3}"; }
table() {   # tag seed scene feasible source delta   (PRIORITY order; tab-separated)
  local b c s sc fe src dl
  for spec in "RILFB10 reindexed heads 1.0" "RILFB05 reindexed heads 0.5" "RILFOB10 reindexed logged 1.0" "RILFOB05 reindexed logged 0.5" "LFB10 logged heads 1.0" "LFB05 logged heads 0.5"; do
    set -- $spec; b=$1; sc=$2; src=$3; dl=$4
    for c in "" C; do
      for s in 0 1; do
        fe=off; [ -n "$c" ] && fe=kbox
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${b}${c}_s${s}" "$s" "$sc" "$fe" "$src" "$dl"
      done
    done
  done
}
pin() { sed -n "s#^$1 sha256 \([0-9a-f]*\)\$#\1#p" "$PK/PINS.txt" | tr -d '\r'; }
bpin() { grep -E "^$1([[:space:]:=]|\$)" "$BUILD_PINS" 2>/dev/null | grep -oE '[0-9a-f]{64}' | head -n 1; }
lpin() { sed -n "s#^$1 sha256 \([0-9a-f]*\)\$#\1#p" "$LLR_PINS" | tr -d '\r'; }

if [ $DRY -eq 1 ]; then
  echo "LF6 chain, DRY RUN (nothing runs, no marker is written)"
  echo "rollouts in priority order (tag seed scene feasible source delta; threads $TH; kappa $KAPPA):"; table
  for s in 0 1; do
    echo "seed $s: C0 $([ -f "$RDB/M2b_s${s}_recft_C0/ckpt_last.pt" ] && echo PRESENT || echo ABSENT)   LLR $([ -f "$RDB/M2b_s${s}_recft_LLR/ckpt_last.pt" ] && echo PRESENT || echo ABSENT)   banked A5 arrays (RILF100 RILF100C RILFO100 RILFO100C): $([ -f "$MC/lf5_read/raw/arrays/RILF100_s$s.npz" ] && [ -f "$MC/lf5_read/raw/arrays/RILFO100C_s$s.npz" ] && echo PRESENT || echo ABSENT)"
  done
  echo "A6 card + FROZEN_SHA: $([ -f "$CARD" ] && [ -f "$MC/a6_prereg/FROZEN_SHA.txt" ] && echo PRESENT || echo ABSENT)   card sha256 $(sha256sum "$CARD" 2>/dev/null | cut -c1-8) (want ${CARD_SHA:0:8})"
  echo "lf6_build/PINS.txt: $([ -f "$BUILD_PINS" ] && echo PRESENT || echo ABSENT)   patched_tool pin: $([ -n "$(bpin patched_tool)" ] && echo PRESENT || echo ABSENT)   tree tool: $([ -f "$TOOL" ] && sha256sum "$TOOL" | cut -c1-16 || echo ABSENT)   PINS.txt: $([ -f "$PK/PINS.txt" ] && echo PRESENT || echo ABSENT)"
  exit 0
fi

if [ -f "$PK/raw/LF6.running" ]; then
  op=$(cat "$PK/raw/LF6.running" | tr -d '\r\n ')
  if [ -n "$op" ] && kill -0 "$op" 2>/dev/null; then log "ZZLF6-ALREADY-RUNNING pid $op: not starting a second chain"; exit 9; fi
  mv "$PK/raw/LF6.running" "$PK/raw/_stale_LF6.running_$(date +%H%M%S)"
fi
for m in LF6.done LF6.fail LF6_ROLLOUTS.done; do [ -f "$PK/raw/$m" ] && mv "$PK/raw/$m" "$PK/raw/_old_${m}_$(date +%H%M%S)"; done
echo $$ > "$PK/raw/LF6.running"
on_exit() {
  local rc=$?
  if [ ! -f "$PK/raw/LF6.done" ] && [ ! -f "$PK/raw/LF6.fail" ]; then echo "$(date +%FT%T) the chain exited without a done marker at step '$STEP' (rc=$rc)" > "$PK/raw/LF6.fail"; fi
  echo "exited $(date +%FT%T) rc=$rc" > "$PK/raw/LF6.running"
}
trap on_exit EXIT
trap 'exit 143' TERM INT HUP

# ---- gate 0: pins, tests, known answers
STEP=gate0
[ -f "$PK/PINS.txt" ] || fail "PINS.txt is absent (run: python -I code/make_pins.py)" 2
for f in $(sed -n 's#^\(code/[^ ]*\|tests/[^ ]*\|lf6_chain[.]sh\|lf6_chain[.]cmd\) sha256 .*#\1#p' "$PK/PINS.txt" | tr -d '\r'); do
  want=$(pin "$f"); got=$(sha256sum "$PK/$f" | cut -c1-64)
  [ -n "$want" ] && [ "$want" = "$got" ] || fail "$f sha $got != PINS.txt ($want)" 2
done
for need in code/lf6_read.py code/lf6_precheck.py tests/test_lf6_read.py lf6_chain.sh; do
  [ -n "$(pin "$need")" ] || fail "PINS.txt carries no pin for $need" 2
done
check_shared() {   # key file
  local want got; want=$(pin "$1"); [ -n "$want" ] || fail "PINS.txt carries no pin for $1" 2
  [ -f "$2" ] || fail "shared file for $1 is absent: $2" 2
  got=$(sha256sum "$2" | cut -c1-64); [ "$want" = "$got" ] || fail "$1 ($2) sha $got != PINS.txt ($want)" 2
}
check_shared a6_card "$CARD"
check_shared lfo_read_py "$MC/lfo_read/code/lfo_read.py"
check_shared lfo_precheck_py "$MC/lfo_read/code/lfo_precheck.py"
check_shared lf_read_py "$MC/lf_read/code/lf_read.py"
check_shared lf_precheck_py "$MC/lf_read/code/lf_precheck.py"
check_shared llr_read_py "$MC/llr_read/code/llr_read.py"
check_shared lfo_read_json "$MC/lfo_read/raw/LFO_READ.json"
check_shared lf5_read_json "$MC/lf5_read/raw/LF5_READ.json"
[ "$(sha256sum "$CARD" | cut -c1-64)" = "$CARD_SHA" ] || fail "A6 card sha != $CARD_SHA" 2
grep -q "^PREREG_LEVELFLOOR_A6.md $CARD_SHA" "$MC/a6_prereg/FROZEN_SHA.txt" || fail "FROZEN_SHA.txt does not carry the A6 card sha" 2
[ -f "$BUILD_PINS" ] || fail "lf6_build/PINS.txt is absent: the LF6 tool's pin cannot be checked" 2
want=$(bpin patched_tool); [ -n "$want" ] || fail "lf6_build/PINS.txt carries no patched_tool key" 2
[ -f "$TOOL" ] || fail "the tree's tool is absent: $TOOL" 2
got=$(sha256sum "$TOOL" | cut -c1-64); [ "$want" = "$got" ] || fail "tree tool sha $got != lf6_build/PINS.txt patched_tool ($want)" 2
wantop=$(bpin level_floor); [ -n "$wantop" ] || fail "lf6_build/PINS.txt carries no level_floor key" 2
[ -f "$OPR" ] || fail "the tree's level_floor.py is absent: $OPR" 2
gotop=$(sha256sum "$OPR" | cut -c1-64); [ "$wantop" = "$gotop" ] || fail "tree level_floor.py sha $gotop != lf6_build/PINS.txt level_floor ($wantop)" 2
for kv in "stack/tanitad/models/v8f_mini.py:model" "stack/scripts/v8f_mini_train.py:trainer" "stack/tanitad/data/v8f_mini_data.py:data"; do
  f=${kv%%:*}; k=${kv##*:}; want=$(lpin "$k"); [ -f "$TREE/$f" ] || fail "LF6 tree $k ($f) is absent" 2
  got=$(sha256sum "$TREE/$f" | cut -c1-64)
  [ -n "$want" ] && [ "$want" = "$got" ] || fail "LF6 tree $k ($f) sha $got != the LLR read's pin ($want)" 2
done
green() {   # logfile label
  local last; last=$(grep -v '^[[:space:]]*$' "$1" 2>/dev/null | tail -n 1)
  case "$last" in *" passed"*) case "$last" in *failed*|*error*) fail "unit tests not green ($2): $last" 2;; esac;; *) fail "unit tests not green ($2): $last" 2;; esac
}
green "$BUILD_DIR/logs/pytest_lf6.log" lf6_build/test_lf6
green "$PK/logs/pytest_lf6_read.log" lf6_read/test_lf6_read
grep -q "^MUTATIONS ALL RED" "$BUILD_DIR/logs/MUTATIONS_LF6.txt" || fail "the build mutation log does not carry MUTATIONS ALL RED" 2
grep -q "^MUTATIONS ALL RED" "$PK/logs/mutations_lf6_read.log" || fail "the reader mutation log does not carry MUTATIONS ALL RED" 2
"$PY" -I -c "
import json,sys
pk,bd,tool=sys.argv[1],sys.argv[2],sys.argv[3]
d=json.load(open(bd+'/raw/KA_LF6.json',encoding='utf-8'))
if d.get('PASS') is not True or set(d.get('cases',{}))!={'K1','K2','K3'} or not all(v.get('PASS') is True for v in d['cases'].values()): print('KA-1/2/3 NOT PASS'); sys.exit(1)
if d.get('tool_sha256')!=tool: print('KA_LF6.json was run with another tool:',d.get('tool_sha256')); sys.exit(1)
r=json.load(open(pk+'/raw/KA_READER_ka-a5.json',encoding='utf-8'))
if r.get('KA_PASS') is not True: print('reader ka-a5 NOT PASS'); sys.exit(1)
" "$PKW" "$BUILD_DIRW" "$(bpin patched_tool)" || fail "a known answer is not PASS or was run with another tool (KA-1, KA-2, KA-3, reader ka-a5)" 2
log "ZZLF6-GATE0 pins, tests, mutations, known answers ok (LF6 tool $(bpin patched_tool | cut -c1-8), operator $(bpin level_floor | cut -c1-8), model $(lpin model | cut -c1-8))"

# ---- gate 1: checkpoints (REFUSED when absent)
STEP=gate1
"$PY" -I "$PKW/code/lf6_precheck.py" > "$PK/logs/precheck.out" 2>&1; rc=$?
cat "$PK/logs/precheck.out" | tee -a "$PK/logs/chain.out"
[ $rc -eq 0 ] || fail "checkpoint controls refuse the read (rc $rc): see logs/precheck.out and raw/LF6_PRECHECK.json" "$rc"
log "ZZLF6-GATE1 C0 and LLR checkpoints present, sha256 == the banked ones == SHA256SUMS.txt"

# ---- gate 2: reused arrays and the cards
STEP=gate2
"$PY" -I -c "
import sys
import importlib.util
spec = importlib.util.spec_from_file_location('lf6_read', sys.argv[1] + '/lf6_read.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.init()
bad = []
for s in m.SEEDS:
    _, c = m.reused_arrays(s)
    bad += [k for k, v in c.items() if not v['PASS']]
if m.LR.card_rules_sha() != m.LR.CARD_RULES_SHA: bad.append('CL card rules sha')
if m.sha256_file(m.CARD) != m.CARD_SHA or m.frozen_card_sha() != m.CARD_SHA: bad.append('A6 card sha')
if m.sha256_file(m.LFO_READ_JSON) != m.LFO_READ_JSON_SHA: bad.append('LFO_READ.json sha')
if m.sha256_file(m.LF5_READ_JSON) != m.LF5_READ_JSON_SHA: bad.append('LF5_READ.json sha')
print('REUSED/CARDS', 'BAD ' + str(bad) if bad else 'ok')
sys.exit(1 if bad else 0)" "$PKW/code" > "$PK/logs/gate2.out" 2>&1 || fail "gate 2 (reused arrays / cards / reader pins): $(tail -n 2 "$PK/logs/gate2.out" | tr '\n' ' ')" 6
log "ZZLF6-GATE2 reused arrays (10 base + 8 banked A5), the CL card's rules, the A6 card (fbf5de02) and the pinned readers match their pins"

# ---- gate 3: CPU
STEP=gate3
busy() { powershell -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'v8f_t1_eval_[a-z_0-9]+[.]py\S*\s+run|extract_smooth[.]py|run_tool_subset[.]py|cl_readout[.]py|llr_readout[.]py|ka_lf[0-9a-z]*[.]py|ka_lfo[.]py' -and \$_.ProcessId -ne \$PID }).Count" | tr -d '\r '; }
w=0; while [ "$(busy)" != "0" ] && [ -n "$(busy)" ]; do sleep 30; w=$((w+30)); [ $w -ge 10800 ] && { log "ZZLF6-BUSY-TIMEOUT another rollout or extractor still runs after 3 h; proceeding"; break; }; done
[ $w -gt 0 ] && log "ZZLF6-CPU-WAIT ${w}s"

run_tool() {   # tag seed outdir threads source delta extra-flags... (stdin closed so a tool never eats the table)
  local tag=$1 s=$2 od=$3 th=$4 src=$5 dl=$6; shift 6
  # shellcheck disable=SC2086
  OMP_NUM_THREADS=$th MKL_NUM_THREADS=$th OPENBLAS_NUM_THREADS=$th CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
  "$PY" -I "$TOOLW" run --ckpt "$RDBW/M2b_s${s}_recft_C0/ckpt_last.pt" --tag "$tag" --tracks "$ART/sfresh_inputs/inputs/native_tracks.pt" --eval-cache "$ART/sfresh_inputs/cache" \
     --eval-grid "$DATA/eval_grid_windows.json" --eval-join "$DATA/join_eval141.npz" --out-dir "$od" \
     --replan-steps 10 --nav deadreckon --schedule periodic --limit causal --dump-ticks "$od/ticks" --dump-heads --device cpu \
     --level-floor on --floor-kappa "$KAPPA" --floor-ckpt "$RDBW/M2b_s${s}_recft_LLR/ckpt_last.pt" --floor-vhat-source "$src" --floor-lift-cap "$dl" "$@" < /dev/null
}
STEP=inputs
[ -f "$ART/sfresh_inputs/inputs/native_tracks.pt" ] || fail "input native_tracks.pt absent" 5
[ -d "$ART/sfresh_inputs/cache/eval/clips" ] || fail "eval cache absent" 5
for f in eval_grid_windows.json join_eval141.npz; do [ -f "$DATA/$f" ] || fail "input $f absent from the data dir" 5; done

# ---- gate 4: pre-flight (6 windows): the heads path and the logged path (the ticks file must load with allow_pickle=False)
pre() {   # tag seed threads source delta flags...
  local tag=$1 s=$2 th=$3 src=$4 dl=$5; shift 5; STEP="preflight $tag"; rm -rf "$S/smoke/$tag"; mkdir -p "$S/smoke/$tag"
  run_tool "$tag" "$s" "$SW/smoke/$tag" "$th" "$src" "$dl" --max-windows 6 "$@" > "$PK/logs/smoke_$tag.out" 2>&1; local rc=$?
  [ $rc -eq 0 ] && [ -f "$S/smoke/$tag/ticks/ticks_$tag.npz" ] || fail "pre-flight $tag rc=$rc (the LF6 tool refuses a flag, or the C0 / LLR checkpoints do not load): see logs/smoke_$tag.out" 7
  "$PY" -I -c "
import sys
import numpy as np
z = np.load(sys.argv[1], allow_pickle=False)
src, dl = sys.argv[2], float(sys.argv[3])
need = ['lf_bind', 'lf_cap', 'lf_vf', 'lf_vhat', 'lf_pi', 'lf_gap', 'lf_lift', 'lf_dterm', 'lf_dterm_knots', 'lf_s', 'lf_sp'] + (['lf_vhat_heads'] if src == 'logged' else [])
ok = all(k in z.files for k in need) and len(z['lf_bind']) > 0 and str(z['floor_vhat_source']) == src and float(z['floor_lift_cap']) == dl and (('label' in z.files and str(z['label']) == 'CEILING') if src == 'logged' else ('label' not in z.files))
ok = ok and bool((z['lf_sp'] - z['lf_s'] - dl * 0.2 * np.arange(1, 31)[None, :]).max() <= 1e-6)
sys.exit(0 if ok else 1)" "$SW/smoke/$tag/ticks/ticks_$tag.npz" "$src" "$dl" || fail "pre-flight $tag: the ticks file does not load with allow_pickle=False, or lacks the lf_* / lf_dterm / lf_s / lf_sp fields, the lift cap, the source / CEILING labels, or breaks s' - s <= DELTA t" 7
  log "ZZLF6-PREFLIGHT-OK $tag"
}
pre PRE_LF6_heads_s0 0 "$TH" heads 0.5 --smooth sg72 --scene reindexed
pre PRE_LF6_logged_s1_kbox 1 "$TH" logged 1.0 --smooth sg72 --scene reindexed --feasible kbox
STEP=rollouts
log "ZZLF6-START LF6 tool $(bpin patched_tool | cut -c1-8) threads=$TH kappa=$KAPPA"

run_one() {   # tag seed scene feas source delta
  local tag=$1 s=$2 scene=$3 feas=$4 src=$5 dl=$6 ck="$RDBW/M2b_s${2}_recft_C0/ckpt_last.pt"
  [ -f "$OUT/done_$tag.txt" ] && { log "ZZLF6-SKIP $tag"; return 0; }
  [ -d "$OUT/$tag" ] && mv "$OUT/$tag" "$OUT/_incomplete_${tag}_$(date +%H%M%S)"
  mkdir -p "$OUT/$tag"
  STEP="rollout $tag"
  local flags="--smooth sg72 --scene $scene"
  [ "$feas" != "off" ] && flags="$flags --feasible $feas"
  local ffl="--level-floor on --floor-kappa $KAPPA --floor-ckpt $RDBW/M2b_s${s}_recft_LLR/ckpt_last.pt --floor-vhat-source $src --floor-lift-cap $dl"
  local t0=$(date +%s)
  log "ZZLF6-RUN $tag flags=$flags $ffl threads=$TH"
  # shellcheck disable=SC2086
  run_tool "$tag" "$s" "$OUTW/$tag" "$TH" "$src" "$dl" $flags > "$PK/logs/run_$tag.out" 2>&1; local rc=$?
  if [ $rc -eq 0 ]; then
    printf '{"tag": "%s", "flags": "%s %s", "ckpt": "%s", "ckpt_sha256": "%s", "tool": "%s", "tool_sha256": "%s", "threads": %s, "wall_s": %s}\n' "$tag" "$flags" "$ffl" "$ck" \
       "$(sha256sum "$RDB/M2b_s${s}_recft_C0/ckpt_last.pt" | cut -c1-64)" "$TOOLW" "$(bpin patched_tool)" "$TH" "$(( $(date +%s) - t0 ))" > "$OUT/done_$tag.txt"
    log "ZZLF6-RUN-DONE $tag wall=$(( $(date +%s) - t0 ))s"
    "$PY" -I "$PKW/code/lf6_precheck.py" --rollout "$tag" --out-root "$OUTW" > "$PK/logs/rollout_check_$tag.out" 2>&1; local crc=$?
    cat "$PK/logs/rollout_check_$tag.out" | tee -a "$PK/logs/chain.out"
    if [ $crc -ne 0 ]; then mv "$OUT/done_$tag.txt" "$OUT/done_$tag.txt.REFUSED"; fail "rollout $tag: the output JSON's floor keys / labels / lift cap / planner checkpoint refuse (rc $crc): see logs/rollout_check_$tag.out" 8; fi
  else
    log "ZZLF6-RUN-FAIL $tag rc=$rc"
  fi
  return $rc
}
nfail=0
while IFS=$'\t' read -r tag seed scene feas src dl <&3; do
  run_one "$tag" "$seed" "$scene" "$feas" "$src" "$dl" || nfail=$((nfail+1))
done 3< <(table)
log "ZZLF6-ROLLOUTS-DONE nfail=$nfail"
missing=""; while IFS=$'\t' read -r tag rest; do [ -f "$OUT/done_$tag.txt" ] || missing="$missing $tag"; done < <(table)
[ -z "$missing" ] || fail "missing done markers:$missing (see logs/run_<tag>.out)" 3
echo "$(date +%FT%T) twenty-four LF6 rollouts done" > "$PK/raw/LF6_ROLLOUTS.done"
log "ZZLF6-ROLLOUTS-OK"

# ---- the reader (controls first; a refused control is a result to report)
STEP=reader
log "ZZLF6-READ start"
"$PY" -I "$PKW/code/lf6_read.py" --mode production --out-root "$OUTW" --work-dir "$SW/work" --raw "$PKW/raw" --workers 4 > "$PK/logs/read.out" 2>&1; rc=$?
log "ZZLF6-READ rc=$rc"
tail -n 3 "$PK/logs/read.out" | tee -a "$PK/logs/chain.out"
if [ $rc -eq 0 ] && [ -f "$PK/raw/LF6_READ.json" ] && [ -f "$PK/RESULTS.md" ]; then
  echo "$(date +%FT%T) LF6_READ.json and RESULTS.md written" > "$PK/raw/LF6.done"
  log "ZZLF6-READ-DONE"
  log "ZZLF6-CHAIN-DONE"
else
  echo "$(date +%FT%T) step 'reader': reader rc=$rc (see logs/read.out; a REFUSED control names itself in raw/LF6_READ.json 'refused'; the 24 rollouts are banked either way)" > "$PK/raw/LF6.fail"
  log "ZZLF6-READ-FAIL rc=$rc"
  exit "$rc"
fi
