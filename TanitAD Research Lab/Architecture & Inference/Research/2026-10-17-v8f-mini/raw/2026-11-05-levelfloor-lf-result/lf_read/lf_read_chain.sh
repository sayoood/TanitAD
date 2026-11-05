#!/usr/bin/env bash
# LF READ chain (PREREG_LEVELFLOOR_A3.md, sha256 977a785e...): arms LF050 and LF075 = planner C0_s{s} + the LLR_s{s} heads read as a speed floor (the floor operator F, patched tool), 16 rollouts over ALL 2,399 eval windows,
# then the reader.  Dev-box CPU, ZERO training, no ssh, ONE rollout at a time.  Cloned from llr_read/llr_read_chain.sh.
# NOT launched by the package that wrote it: the Master Mind starts it (lf_read_chain.cmd = WMI Win32_Process.Create so it survives the caller).
#   usage:  lf_read_chain.sh --dry-run     lists the 16 rollouts, the gates and what is present; runs nothing, writes no marker
#           lf_read_chain.sh               (normal)
# READ TREE: F:/TanitAD-scratch/motion_c/lf_read/tree (owned by the tool agent): the LLR read tree's 83c9ee0 model / data / trainer + the PATCHED tool v8f_t1_eval_ri.py (flags --level-floor on --floor-kappa K --floor-ckpt LLR ckpt).
#   The patched tool's sha256 is `patched_tool` in motion_c/lf_build/PINS.txt; gate 0 refuses when that file or key is absent or the tree's tool differs.
# Rollouts in PRIORITY order (a killed chain yields the deciding scene first), reader flags = the banked C0 / RIC0 rollouts' (--replan-steps 10 --nav deadreckon --schedule periodic --limit causal --dump-ticks <out>/ticks --dump-heads
#   --device cpu --smooth sg72 --scene S [--feasible kbox]) + `--level-floor on --floor-kappa K --floor-ckpt .../M2b_s{s}_recft_LLR/ckpt_last.pt`; --ckpt = .../M2b_s{s}_recft_C0/ckpt_last.pt:
#   RILF050 RILF050C RILF075 RILF075C (reindexed, DECIDING; 12 threads, LF_TH_RI)  then LF050 LF050C LF075 LF075C (logged, LEAK-BORNE only; 16 threads, LF_TH_LOGGED), seeds 0 and 1 each.
#   hold_now is not read (A3 card section 3).  NO base rollout: the banked C0, C0C, C0REP, RIC0, RIC0C arrays (sha256-pinned in the reader).
# Gates (REFUSE = raw/LFRO.fail with the reason, exit code):
#   0  PINS.txt: reader, pre-check, tests, this script, the tree's model / data / trainer (= the LLR read's pins) and tool (= lf_build/PINS.txt patched_tool) match their sha256; the unit-test logs are green; the mutation log says
#      ALL RED; the known answers ka-null and ka-llr say PASS                                                                                                        (exit 2)
#   1  code/lf_precheck.py: both seeds' C0 and LLR checkpoints present (absent = 5), sha256 == the banked ones == SHA256SUMS.txt                                             (exit 5 absent, 8 failed)
#   2  the reused arrays' sha256 (10), the CL card's rules sha, the A3 card sha = FROZEN_SHA.txt = 977a785e..., llr_read.py's sha                                               (exit 6)
#   3  no other tool rollout / extractor runs (CPU shared; waits at most 3 h)
#   4  a 6-window PRE-FLIGHT per scene family with the floor ON: s0 RILF050C (reindexed + kbox), s1 RILF075 (reindexed), s0 LF050 (logged): each must exit 0 and write a ticks file carrying the seven lf_* fields      (exit 7)
# After every finished rollout lf_precheck.py --rollout TAG checks the output JSON (level_floor on, floor_kappa == the dose, floor_ckpt sha256 == the seed's LLR sha) and the planner checkpoint sha (exit 8, chain stops).
# Each rollout writes done_<tag>.txt only on rc 0; a finished run is skipped on re-launch (RESUME = launch lf_read_chain.cmd again); an incomplete out dir is MOVED to _incomplete_*, never deleted.
# Markers: ZZLF-* lines in logs/chain.out; raw/LFRO.done (all 16 rollouts) / raw/LFRO.fail; then the reader: raw/LFREAD.done (raw/LF_READ.json + RESULTS.md) / raw/LFREAD.fail.  A REFUSED control of the reader is a result.
# WRITES: this package (logs, raw, RESULTS.md) and F:/TanitAD-scratch/motion_c/lf_read/{out,smoke,work}.  The LF tree is READ ONLY here.
set -u
DRY=0; for t in "$@"; do [ "$t" = "--dry-run" ] && DRY=1; done
PK=${LF_PK:-/e/Projects/TanitAD-artifacts/dagland/motion_c/lf_read}
PKW=${LF_PKW:-E:/Projects/TanitAD-artifacts/dagland/motion_c/lf_read}
S=/f/TanitAD-scratch/motion_c/lf_read
SW=F:/TanitAD-scratch/motion_c/lf_read
TREE=$S/tree
TOOLW=$SW/tree/stack/scripts/v8f_t1_eval_ri.py
TOOL=$TREE/stack/scripts/v8f_t1_eval_ri.py
OUT=$S/out
OUTW=$SW/out
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
ART=E:/Projects/TanitAD-artifacts
MC=/e/Projects/TanitAD-artifacts/dagland/motion_c
RDB=/e/Projects/TanitAD-artifacts/v8f-recover
RDBW=$ART/v8f-recover
DATA="$ART/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
CARD=$MC/a3_prereg/PREREG_LEVELFLOOR_A3.md
BUILD_PINS=${LF_BUILD_PINS_FILE:-$MC/lf_build/PINS.txt}
LLR_PINS=$MC/llr_read/PINS.txt
CARD_SHA=977a785e54df0b7e13bb8f81639d88eb52bb64092e5e1657e70b6e09fde2d405
TH_LOGGED=${LF_TH_LOGGED:-16}
TH_RI=${LF_TH_RI:-12}
mkdir -p "$OUT" "$PK/logs" "$PK/raw" "$S/smoke" "$S/work"
log() { echo "$(date +%FT%T) $*" | tee -a "$PK/logs/chain.out"; }
fail() { log "ZZLF-FAIL $1"; echo "$(date +%FT%T) $1" > "$PK/raw/LFRO.fail"; exit "${2:-3}"; }
table() {   # tag seed scene feasible dose kappa   (PRIORITY order; tab-separated)
  local r
  for r in "RILF050_s0 0 reindexed off 050 0.5" "RILF050_s1 1 reindexed off 050 0.5" "RILF050C_s0 0 reindexed kbox 050 0.5" "RILF050C_s1 1 reindexed kbox 050 0.5" \
           "RILF075_s0 0 reindexed off 075 0.75" "RILF075_s1 1 reindexed off 075 0.75" "RILF075C_s0 0 reindexed kbox 075 0.75" "RILF075C_s1 1 reindexed kbox 075 0.75" \
           "LF050_s0 0 logged off 050 0.5" "LF050_s1 1 logged off 050 0.5" "LF050C_s0 0 logged kbox 050 0.5" "LF050C_s1 1 logged kbox 050 0.5" \
           "LF075_s0 0 logged off 075 0.75" "LF075_s1 1 logged off 075 0.75" "LF075C_s0 0 logged kbox 075 0.75" "LF075C_s1 1 logged kbox 075 0.75"; do
    set -- $r
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" "$6"
  done
}
pin() { sed -n "s|^$1 sha256 \([0-9a-f]*\)\$|\1|p" "$PK/PINS.txt" | tr -d '\r'; }
bpin() { grep -E "^$1([[:space:]:=]|\$)" "$BUILD_PINS" 2>/dev/null | grep -oE '[0-9a-f]{64}' | head -n 1; }
lpin() { sed -n "s|^$1 sha256 \([0-9a-f]*\)\$|\1|p" "$LLR_PINS" | tr -d '\r'; }

if [ $DRY -eq 1 ]; then
  echo "LF read chain, DRY RUN (nothing runs, no marker is written)"
  echo "rollouts in priority order (tag seed scene feasible dose kappa; threads: reindexed $TH_RI, others $TH_LOGGED):"; table
  for s in 0 1; do
    echo "seed $s: C0 checkpoint $RDB/M2b_s${s}_recft_C0/ckpt_last.pt: $([ -f "$RDB/M2b_s${s}_recft_C0/ckpt_last.pt" ] && echo PRESENT || echo ABSENT)   LLR checkpoint (floor heads) $RDB/M2b_s${s}_recft_LLR/ckpt_last.pt: $([ -f "$RDB/M2b_s${s}_recft_LLR/ckpt_last.pt" ] && echo PRESENT || echo ABSENT)   banked base arrays (C0 C0C C0REP RIC0 RIC0C): $([ -f "$MC/../recover_next/read/raw/arrays/C0_s$s.npz" ] && [ -f "$MC/../recover_next/read/raw/arrays/C0C_s$s.npz" ] && [ -f "$MC/../recover_next/read/raw/arrays/C0REP_s$s.npz" ] && [ -f "$MC/ri_read/raw/arrays/RIC0_s$s.npz" ] && [ -f "$MC/ri_read/raw/arrays/RIC0C_s$s.npz" ] && echo PRESENT || echo ABSENT)"
  done
  echo "A3 card + FROZEN_SHA: $([ -f "$CARD" ] && [ -f "$MC/a3_prereg/FROZEN_SHA.txt" ] && echo PRESENT || echo ABSENT)   card sha256 $(sha256sum "$CARD" 2>/dev/null | cut -c1-8) (want ${CARD_SHA:0:8})"
  echo "lf_build/PINS.txt: $([ -f "$BUILD_PINS" ] && echo PRESENT || echo ABSENT)   patched_tool pin: $([ -n "$(bpin patched_tool)" ] && echo PRESENT || echo ABSENT)   tree tool: $([ -f "$TOOL" ] && sha256sum "$TOOL" | cut -c1-16 || echo ABSENT)   tree model: $([ -f "$TREE/stack/tanitad/models/v8f_mini.py" ] && sha256sum "$TREE/stack/tanitad/models/v8f_mini.py" | cut -c1-16 || echo ABSENT)   PINS.txt: $([ -f "$PK/PINS.txt" ] && echo PRESENT || echo ABSENT)"
  exit 0
fi

if [ -f "$PK/raw/LFRO.running" ]; then
  op=$(cat "$PK/raw/LFRO.running" | tr -d '\r\n ')
  if [ -n "$op" ] && kill -0 "$op" 2>/dev/null; then log "ZZLF-ALREADY-RUNNING pid $op: not starting a second chain"; exit 9; fi
  mv "$PK/raw/LFRO.running" "$PK/raw/_stale_LFRO.running_$(date +%H%M%S)"
fi
echo $$ > "$PK/raw/LFRO.running"
trap 'echo "exited $(date +%FT%T)" > "$PK/raw/LFRO.running"' EXIT
for m in LFRO.done LFRO.fail LFREAD.done LFREAD.fail; do [ -f "$PK/raw/$m" ] && mv "$PK/raw/$m" "$PK/raw/_old_${m}_$(date +%H%M%S)"; done

# ---- gate 0: pins, tests, known answers
[ -f "$PK/PINS.txt" ] || fail "PINS.txt is absent (run: python -I code/make_pins.py)" 2
for f in $(sed -n 's|^\(code/[^ ]*\|tests/[^ ]*\|lf_read_chain[.]sh\|lf_read_chain[.]cmd\) sha256 .*|\1|p' "$PK/PINS.txt" | tr -d '\r'); do
  want=$(pin "$f"); got=$(sha256sum "$PK/$f" | cut -c1-64)
  [ -n "$want" ] && [ "$want" = "$got" ] || fail "$f sha $got != PINS.txt ($want)" 2
done
for need in code/lf_read.py code/lf_precheck.py tests/test_lf_read.py tests/test_lf_chain_gates.py lf_read_chain.sh; do
  [ -n "$(pin "$need")" ] || fail "PINS.txt carries no pin for $need" 2
done
[ -f "$BUILD_PINS" ] || fail "lf_build/PINS.txt is absent: the patched tool's pin cannot be checked" 2
want=$(bpin patched_tool); [ -n "$want" ] || fail "lf_build/PINS.txt carries no patched_tool key" 2
[ -f "$TOOL" ] || fail "the tree's tool is absent: $TOOL" 2
got=$(sha256sum "$TOOL" | cut -c1-64); [ "$want" = "$got" ] || fail "tree tool sha $got != lf_build/PINS.txt patched_tool ($want)" 2
for kv in "stack/tanitad/models/v8f_mini.py:model" "stack/scripts/v8f_mini_train.py:trainer" "stack/tanitad/data/v8f_mini_data.py:data"; do
  f=${kv%%:*}; k=${kv##*:}; want=$(lpin "$k"); [ -f "$TREE/$f" ] || fail "LF tree $k ($f) is absent" 2
  got=$(sha256sum "$TREE/$f" | cut -c1-64)
  [ -n "$want" ] && [ "$want" = "$got" ] || fail "LF tree $k ($f) sha $got != the LLR read's pin ($want)" 2
done
for lg in pytest_lf_read pytest_lf_chain_gates; do
  last=$(grep -v '^[[:space:]]*$' "$PK/logs/$lg.log" | tail -n 1)
  case "$last" in *" passed"*) case "$last" in *failed*|*error*) fail "unit tests not green ($lg): $last" 2;; esac;; *) fail "unit tests not green ($lg): $last" 2;; esac
done
grep -q "^MUTATIONS ALL RED" "$PK/logs/mutations_lf.log" || fail "the mutation log does not carry MUTATIONS ALL RED" 2
"$PY" -I -c "
import json,sys
pk=sys.argv[1]
for p,k in ((pk+'/raw/KA_READER_ka-null.json','KA_PASS'),(pk+'/raw/KA_READER_ka-llr.json','KA_PASS')):
    d=json.load(open(p,encoding='utf-8'))
    if d.get(k) is not True: print('NOT PASS:',p); sys.exit(1)
" "$PKW" || fail "a known answer is not PASS (KA_READER ka-null / ka-llr)" 2
log "ZZLF-GATE0 pins, tests, known answers ok (patched tool $(bpin patched_tool | cut -c1-8), model $(lpin model | cut -c1-8))"

# ---- gate 1: checkpoints (REFUSED when absent)
"$PY" -I "$PKW/code/lf_precheck.py" > "$PK/logs/precheck.out" 2>&1; rc=$?
cat "$PK/logs/precheck.out" | tee -a "$PK/logs/chain.out"
[ $rc -eq 0 ] || fail "checkpoint controls refuse the read (rc $rc): see logs/precheck.out and raw/LF_PRECHECK.json" "$rc"
log "ZZLF-GATE1 C0 and LLR checkpoints present, sha256 == the banked ones == SHA256SUMS.txt"

# ---- gate 2: reused arrays and the cards
"$PY" -I -c "
import sys
import importlib.util
spec = importlib.util.spec_from_file_location('lf_read', sys.argv[1] + '/lf_read.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.init()
bad = []
for s in m.SEEDS:
    _, c = m.reused_arrays(s)
    bad += [k for k, v in c.items() if not v['PASS']]
if m.LR.card_rules_sha() != m.LR.CARD_RULES_SHA: bad.append('CL card rules sha')
if m.sha256_file(m.CARD) != m.CARD_SHA or m.frozen_card_sha() != m.CARD_SHA: bad.append('A3 card sha')
if m.sha256_file(m.LLR_READ_JSON) != m.LLR_READ_JSON_SHA: bad.append('LLR_READ.json sha')
print('REUSED/CARDS', 'BAD ' + str(bad) if bad else 'ok')
sys.exit(1 if bad else 0)" "$PKW/code" > "$PK/logs/gate2.out" 2>&1 || fail "gate 2 (reused arrays / cards / reader pins): $(tail -n 2 "$PK/logs/gate2.out" | tr '\n' ' ')" 6
log "ZZLF-GATE2 reused arrays (10), the CL card's rules, the A3 card (977a785e) and llr_read.py match their pins"

# ---- gate 3: CPU
busy() { powershell -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'v8f_t1_eval_[a-z_0-9]+[.]py\S*\s+run|extract_smooth[.]py|run_tool_subset[.]py|cl_readout[.]py|llr_readout[.]py' -and \$_.ProcessId -ne \$PID }).Count" | tr -d '\r '; }
w=0; while [ "$(busy)" != "0" ] && [ -n "$(busy)" ]; do sleep 30; w=$((w+30)); [ $w -ge 10800 ] && { log "ZZLF-BUSY-TIMEOUT another rollout or extractor still runs after 3 h; proceeding"; break; }; done
[ $w -gt 0 ] && log "ZZLF-CPU-WAIT ${w}s"

run_tool() {   # tag seed outdir threads kappa extra-flags... (stdin closed so a tool never eats the table)
  local tag=$1 s=$2 od=$3 th=$4 kap=$5; shift 5
  OMP_NUM_THREADS=$th MKL_NUM_THREADS=$th OPENBLAS_NUM_THREADS=$th CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
  "$PY" -I "$TOOLW" run --ckpt "$RDBW/M2b_s${s}_recft_C0/ckpt_last.pt" --tag "$tag" --tracks "$ART/sfresh_inputs/inputs/native_tracks.pt" --eval-cache "$ART/sfresh_inputs/cache" \
     --eval-grid "$DATA/eval_grid_windows.json" --eval-join "$DATA/join_eval141.npz" --out-dir "$od" \
     --replan-steps 10 --nav deadreckon --schedule periodic --limit causal --dump-ticks "$od/ticks" --dump-heads --device cpu \
     --level-floor on --floor-kappa "$kap" --floor-ckpt "$RDBW/M2b_s${s}_recft_LLR/ckpt_last.pt" "$@" < /dev/null
}
[ -f "$ART/sfresh_inputs/inputs/native_tracks.pt" ] || fail "input native_tracks.pt absent" 5
[ -d "$ART/sfresh_inputs/cache/eval/clips" ] || fail "eval cache absent" 5
for f in eval_grid_windows.json join_eval141.npz; do [ -f "$DATA/$f" ] || fail "input $f absent from the data dir" 5; done

# ---- gate 4: pre-flight (6 windows, floor ON)
pre() {   # tag seed threads kappa flags...
  local tag=$1 s=$2 th=$3 kap=$4; shift 4; rm -rf "$S/smoke/$tag"; mkdir -p "$S/smoke/$tag"
  run_tool "$tag" "$s" "$SW/smoke/$tag" "$th" "$kap" --max-windows 6 "$@" > "$PK/logs/smoke_$tag.out" 2>&1; local rc=$?
  [ $rc -eq 0 ] && [ -f "$S/smoke/$tag/ticks/ticks_$tag.npz" ] || fail "pre-flight $tag rc=$rc (the patched tool refuses a flag, or the C0 / LLR checkpoints do not load): see logs/smoke_$tag.out" 7
  "$PY" -I -c "
import sys
import numpy as np
z = np.load(sys.argv[1], allow_pickle=True)
need = ('lf_bind', 'lf_cap', 'lf_vf', 'lf_vhat', 'lf_pi', 'lf_gap', 'lf_lift')
sys.exit(0 if all(k in z.files for k in need) and len(z['lf_bind']) > 0 else 1)" "$SW/smoke/$tag/ticks/ticks_$tag.npz" || fail "pre-flight $tag: the ticks file lacks the lf_* fields" 7
  log "ZZLF-PREFLIGHT-OK $tag"
}
pre PRE_LF050C_s0_ri_kbox 0 "$TH_RI" 0.5 --smooth sg72 --scene reindexed --feasible kbox
pre PRE_LF075_s1_ri 1 "$TH_RI" 0.75 --smooth sg72 --scene reindexed
pre PRE_LF050_s0_logged 0 "$TH_LOGGED" 0.5 --smooth sg72 --scene logged
log "ZZLF-START patched tool $(bpin patched_tool | cut -c1-8) threads reindexed=$TH_RI logged=$TH_LOGGED"

run_one() {   # tag seed scene feas dose kappa
  local tag=$1 s=$2 scene=$3 feas=$4 kap=$6 ck="$RDBW/M2b_s${2}_recft_C0/ckpt_last.pt"
  [ -f "$OUT/done_$tag.txt" ] && { log "ZZLF-SKIP $tag"; return 0; }
  [ -d "$OUT/$tag" ] && mv "$OUT/$tag" "$OUT/_incomplete_${tag}_$(date +%H%M%S)"
  mkdir -p "$OUT/$tag"
  local flags="--smooth sg72 --scene $scene"
  [ "$feas" != "off" ] && flags="$flags --feasible $feas"
  local ffl="--level-floor on --floor-kappa $kap --floor-ckpt $RDBW/M2b_s${s}_recft_LLR/ckpt_last.pt"
  local th=$TH_LOGGED; [ "$scene" = "reindexed" ] && th=$TH_RI
  local t0=$(date +%s)
  log "ZZLF-RUN $tag flags=$flags $ffl threads=$th"
  # shellcheck disable=SC2086
  run_tool "$tag" "$s" "$OUTW/$tag" "$th" "$kap" $flags > "$PK/logs/run_$tag.out" 2>&1; local rc=$?
  if [ $rc -eq 0 ]; then
    printf '{"tag": "%s", "flags": "%s %s", "ckpt": "%s", "ckpt_sha256": "%s", "tool": "%s", "tool_sha256": "%s", "threads": %s, "wall_s": %s}\n' "$tag" "$flags" "$ffl" "$ck" \
       "$(sha256sum "$RDB/M2b_s${s}_recft_C0/ckpt_last.pt" | cut -c1-64)" "$TOOLW" "$(bpin patched_tool)" "$th" "$(( $(date +%s) - t0 ))" > "$OUT/done_$tag.txt"
    log "ZZLF-RUN-DONE $tag wall=$(( $(date +%s) - t0 ))s"
    "$PY" -I "$PKW/code/lf_precheck.py" --rollout "$tag" --out-root "$OUTW" > "$PK/logs/rollout_check_$tag.out" 2>&1; local crc=$?
    cat "$PK/logs/rollout_check_$tag.out" | tee -a "$PK/logs/chain.out"
    if [ $crc -ne 0 ]; then mv "$OUT/done_$tag.txt" "$OUT/done_$tag.txt.REFUSED"; fail "rollout $tag: the output JSON's floor keys / planner checkpoint refuse (rc $crc): see logs/rollout_check_$tag.out" 8; fi
  else
    log "ZZLF-RUN-FAIL $tag rc=$rc"
  fi
  return $rc
}
nfail=0
while IFS=$'\t' read -r tag seed scene feas dose kap <&3; do
  run_one "$tag" "$seed" "$scene" "$feas" "$dose" "$kap" || nfail=$((nfail+1))
done 3< <(table)
log "ZZLF-ROLLOUTS-DONE nfail=$nfail"
missing=""; while IFS=$'\t' read -r tag rest; do [ -f "$OUT/done_$tag.txt" ] || missing="$missing $tag"; done < <(table)
[ -z "$missing" ] || fail "missing done markers:$missing (see logs/run_<tag>.out)" 3
echo "$(date +%FT%T) sixteen LF-read rollouts done" > "$PK/raw/LFRO.done"
log "ZZLF-LFRO-DONE"

# ---- the reader (controls first; a refused control is a result to report)
log "ZZLF-READ start"
"$PY" -I "$PKW/code/lf_read.py" --mode production --out-root "$OUTW" --work-dir "$SW/work" --raw "$PKW/raw" --workers 6 > "$PK/logs/read.out" 2>&1; rc=$?
log "ZZLF-READ rc=$rc"
tail -n 3 "$PK/logs/read.out" | tee -a "$PK/logs/chain.out"
if [ $rc -eq 0 ] && [ -f "$PK/raw/LF_READ.json" ] && [ -f "$PK/RESULTS.md" ]; then
  echo "$(date +%FT%T) LF_READ.json and RESULTS.md written" > "$PK/raw/LFREAD.done"
  log "ZZLF-READ-DONE"
  log "ZZLF-CHAIN-DONE"
else
  echo "$(date +%FT%T) reader rc=$rc (see logs/read.out; a REFUSED control names itself in raw/LF_READ.json 'refused')" > "$PK/raw/LFREAD.fail"
  log "ZZLF-READ-FAIL rc=$rc"
  exit "$rc"
fi
