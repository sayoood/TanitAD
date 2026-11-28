#!/usr/bin/env bash
# CLR READ chain (PREREG_LEVELHEAD_A2.md, Reads and Outcomes): the two CLR checkpoints (RAPR-P01's recovery recipe + the level head, trained on box 2) AND the attribution base (RAPR-P01's own checkpoint under `reindexed`)
# over ALL 2,399 eval windows, then the reader, then A1's O-sensitivity read-out on the CLR checkpoints.  Dev-box CPU, ZERO training, no ssh, ONE rollout at a time.  Cloned from cl_read/cl_read_chain.sh.
# NOT launched by the package that wrote it: the Master Mind starts it AFTER the CLR checkpoints are pulled into E:/Projects/TanitAD-artifacts/v8f-recover/ (M2b_s{0,1}_recft_CLR/, clr_markers/).
#   usage:  clr_read_chain.sh --dry-run     lists the 12 rollouts, the gates and what is present; runs nothing, writes no marker
#           clr_read_chain.sh               (normal; start through clr_read_chain.cmd = WMI Win32_Process.Create so it survives the caller)
# Rollouts in PRIORITY order (a killed chain still yields the deciding contrast first), reader flags (--replan-steps 10 --nav deadreckon --schedule periodic --limit causal --dump-ticks <out>/ticks --dump-heads --device cpu --smooth sg72):
#   RICLR_s0 RICLR_s1 (--scene reindexed, DECIDING)  RICLRC_s0 RICLRC_s1 (reindexed + --feasible kbox, D)  RIRAPRP01_s0 RIRAPRP01_s1 (RAPR-P01's checkpoint, reindexed: the attribution base)
#   CLR_s0 CLR_s1 (logged)  CLRC_s0 CLRC_s1 (logged + kbox)  FZCLR_s0 FZCLR_s1 (hold_now)
#   Threads: logged / hold_now arms CLR_TH_LOGGED (default 16 = the banked C0 / C0C / FZC0 / C0REP rollouts' count), reindexed arms (RICLR, RICLRC, RIRAPRP01) CLR_TH_RI (default 12 = RIC0 / RIC0C's).  MEASURED in the CL package
#   (cl_read/raw/KA_NEW.json): the same tool at the base's own thread count reproduces the banked rollout bit for bit; at another count a 1e-5 m rounding grows to 0.6 m on one window.
# Gates (REFUSE = raw/CLRRO.fail with the reason, exit code):
#   0  PINS.txt: the reader, pre-check, readout clones, wiring, tests, this script, the shared tree's tool / model / trainer and the CL package's evidence (KA_NEW, SMOKE_LEVEL_ON, CL_READ) match their sha256;
#      the unit-test logs are green; the mutation log says ALL RED; both reader known answers (ka-null, ka-ri, ka-cl) say KA_PASS                                                   (exit 2)
#   1  code/clr_precheck.py: both CLR checkpoints present (absent = REFUSED), sha256 == SHA256SUMS.txt == the box-2 marker clr_markers/RUN_CLR_{s}.done evidence.ckpt_sha256, argv = RAPR-P01's + the two level
#      tokens with the FROZEN W, the recovery lever = RAPR-P01's, marker recover_n > 0, cfg level_head on, RAPR-P01's own checkpoint present and consistent                       (exit 5 absent, 8 failed)
#   2  the reused arrays' sha256 (14), the CL card's rules sha, the A2 card sha = FROZEN_SHA.txt                                                                                  (exit 6)
#   3  no other tool rollout / extractor runs (CPU shared; waits at most 3 h)
#   4  a 6-window PRE-FLIGHT per checkpoint family: s0 CLR --scene reindexed --feasible kbox, s1 CLR --scene hold_now, s1 RAPR-P01 --scene reindexed: each must exit 0 and write its ticks file  (exit 7)
# Each rollout writes done_<tag>.txt only on rc 0; a finished run is skipped on re-launch (RESUME = launch clr_read_chain.cmd again); an incomplete out dir is MOVED to _incomplete_*, never deleted.
# Markers: ZZCLR-* lines in logs/chain.out; raw/CLRRO.done (all twelve rollouts) / raw/CLRRO.fail (reason inside); then the reader (code/clr_read.py --mode production): raw/CLREAD.done (raw/CLR_READ.json + RESULTS.md) /
#   raw/CLREAD.fail; then the read-out of A1 on the CLR checkpoints: raw/CLRREADOUT.done (readout/CLR_READOUT.json, raw/CLR_WIRING.json, WIRING.md) / raw/CLRREADOUT.fail.  A REFUSED control of the reader is a result.
# WRITES: this package (logs, raw, readout, RESULTS.md, WIRING.md) and F:/TanitAD-scratch/motion_c/clr_read/{out,smoke,work,readout}.  The tree F:/TanitAD-scratch/motion_c/cl_read/tree is READ ONLY (shared, sha-pinned).
set -u
DRY=0; for t in "$@"; do [ "$t" = "--dry-run" ] && DRY=1; done
PK=${CLR_PK:-/e/Projects/TanitAD-artifacts/dagland/motion_c/clr_read}
PKW=${CLR_PKW:-E:/Projects/TanitAD-artifacts/dagland/motion_c/clr_read}
S=/f/TanitAD-scratch/motion_c/clr_read
SW=F:/TanitAD-scratch/motion_c/clr_read
TREE=/f/TanitAD-scratch/motion_c/cl_read/tree
TOOLW=F:/TanitAD-scratch/motion_c/cl_read/tree/stack/scripts/v8f_t1_eval_ri.py
TOOL=$TREE/stack/scripts/v8f_t1_eval_ri.py
OUT=$S/out
OUTW=$SW/out
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
ART=E:/Projects/TanitAD-artifacts
RDB=/e/Projects/TanitAD-artifacts/v8f-recover
RDBW=$ART/v8f-recover
DATA="$ART/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
CARD=/e/Projects/TanitAD-artifacts/dagland/motion_c/a2_prereg/PREREG_LEVELHEAD_A2.md
TH_LOGGED=${CLR_TH_LOGGED:-16}
TH_RI=${CLR_TH_RI:-12}
mkdir -p "$OUT" "$PK/logs" "$PK/raw" "$S/smoke" "$S/work" "$S/readout"
log() { echo "$(date +%FT%T) $*" | tee -a "$PK/logs/chain.out"; }
fail() { log "ZZCLR-FAIL $1"; echo "$(date +%FT%T) $1" > "$PK/raw/CLRRO.fail"; exit "${2:-3}"; }
table() {   # tag seed scene feasible run   (PRIORITY order; tab-separated)
  local r
  for r in "RICLR_s0 0 reindexed off CLR" "RICLR_s1 1 reindexed off CLR" "RICLRC_s0 0 reindexed kbox CLR" "RICLRC_s1 1 reindexed kbox CLR" "RIRAPRP01_s0 0 reindexed off RAPRP01" "RIRAPRP01_s1 1 reindexed off RAPRP01" \
           "CLR_s0 0 logged off CLR" "CLR_s1 1 logged off CLR" "CLRC_s0 0 logged kbox CLR" "CLRC_s1 1 logged kbox CLR" "FZCLR_s0 0 hold_now off CLR" "FZCLR_s1 1 hold_now off CLR"; do
    set -- $r
    printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5"
  done
}
pin() { sed -n "s|^$1 sha256 \([0-9a-f]*\)\$|\1|p" "$PK/PINS.txt" | tr -d '\r'; }

if [ $DRY -eq 1 ]; then
  echo "CLR read chain, DRY RUN (nothing runs, no marker is written)"
  echo "rollouts in priority order (tag seed scene feasible run; threads: reindexed $TH_RI, others $TH_LOGGED):"; table
  for s in 0 1; do
    r=M2b_s${s}_recft_CLR; b=M2b_s${s}_recft_RAPRP01
    echo "checkpoint $RDB/$r/ckpt_last.pt: $([ -f "$RDB/$r/ckpt_last.pt" ] && echo PRESENT || echo ABSENT)   marker $RDB/clr_markers/RUN_CLR_$s.done: $([ -f "$RDB/clr_markers/RUN_CLR_$s.done" ] && echo PRESENT || echo ABSENT)   attribution base $RDB/$b/ckpt_last.pt: $([ -f "$RDB/$b/ckpt_last.pt" ] && echo PRESENT || echo ABSENT)"
  done
  echo "FROZEN_W.txt: $([ -f "$(dirname "$CARD")/../cl_prereg/FROZEN_W.txt" ] && echo PRESENT || echo ABSENT)   A2 card + FROZEN_SHA: $([ -f "$CARD" ] && [ -f "$(dirname "$CARD")/FROZEN_SHA.txt" ] && echo PRESENT || echo ABSENT)"
  echo "tree tool: $([ -f "$TOOL" ] && sha256sum "$TOOL" | cut -c1-16 || echo ABSENT)   PINS.txt: $([ -f "$PK/PINS.txt" ] && echo PRESENT || echo ABSENT)"
  exit 0
fi

if [ -f "$PK/raw/CLRRO.running" ]; then
  op=$(cat "$PK/raw/CLRRO.running" | tr -d '\r\n ')
  if [ -n "$op" ] && kill -0 "$op" 2>/dev/null; then log "ZZCLR-ALREADY-RUNNING pid $op: not starting a second chain"; exit 9; fi
  mv "$PK/raw/CLRRO.running" "$PK/raw/_stale_CLRRO.running_$(date +%H%M%S)"
fi
echo $$ > "$PK/raw/CLRRO.running"
trap 'echo "exited $(date +%FT%T)" > "$PK/raw/CLRRO.running"' EXIT
for m in CLRRO.done CLRRO.fail CLREAD.done CLREAD.fail CLRREADOUT.done CLRREADOUT.fail; do [ -f "$PK/raw/$m" ] && mv "$PK/raw/$m" "$PK/raw/_old_${m}_$(date +%H%M%S)"; done

# ---- gate 0: pins, tests, known answers
[ -f "$PK/PINS.txt" ] || fail "PINS.txt is absent (run: python -I code/make_pins.py)" 2
for f in $(sed -n 's|^\(code/[^ ]*\|tests/[^ ]*\|clr_read_chain[.]sh\|clr_read_chain[.]cmd\) sha256 .*|\1|p' "$PK/PINS.txt" | tr -d '\r'); do
  want=$(pin "$f"); got=$(sha256sum "$PK/$f" | cut -c1-64)
  [ -n "$want" ] && [ "$want" = "$got" ] || fail "$f sha $got != PINS.txt ($want)" 2
done
for need in code/clr_read.py code/clr_precheck.py code/clr_readout.py code/clr_merge_readout.py code/clr_wiring.py tests/test_clr_read.py tests/test_clr_chain_gates.py clr_read_chain.sh; do
  [ -n "$(pin "$need")" ] || fail "PINS.txt carries no pin for $need" 2
done
for kv in "stack/scripts/v8f_t1_eval_ri.py:tool" "stack/tanitad/models/v8f_mini.py:model" "stack/scripts/v8f_mini_train.py:trainer"; do
  f=${kv%%:*}; k=${kv##*:}; want=$(pin "$k"); got=$(sha256sum "$TREE/$f" | cut -c1-64)
  [ -n "$want" ] && [ "$want" = "$got" ] || fail "shared tree $k ($f) sha $got != PINS.txt ($want)" 2
done
for ev in "dagland/motion_c/cl_read/raw/KA_NEW.json:cl_ka_new" "dagland/motion_c/cl_read/raw/SMOKE_LEVEL_ON.json:cl_smoke_level_on" "dagland/motion_c/cl_read/raw/CL_READ.json:cl_read_json"; do
  f=${ev%%:*}; k=${ev##*:}; want=$(pin "$k"); got=$(sha256sum "/e/Projects/TanitAD-artifacts/$f" | cut -c1-64)
  [ -n "$want" ] && [ "$want" = "$got" ] || fail "CL package evidence $k ($f) sha $got != PINS.txt ($want)" 2
done
for lg in pytest_clr_read pytest_clr_chain_gates; do
  last=$(grep -v '^[[:space:]]*$' "$PK/logs/$lg.log" | tail -n 1)
  case "$last" in *" passed"*) case "$last" in *failed*|*error*) fail "unit tests not green ($lg): $last" 2;; esac;; *) fail "unit tests not green ($lg): $last" 2;; esac
done
grep -q "^MUTATIONS ALL RED" "$PK/logs/mutations_clr.log" || fail "the mutation log does not carry MUTATIONS ALL RED" 2
"$PY" -I -c "
import json,sys
pk=sys.argv[1]; cl=sys.argv[2]
for p,k in ((cl+'/raw/KA_NEW.json','PASS'),(cl+'/raw/SMOKE_LEVEL_ON.json','PASS'),(pk+'/raw/KA_READER_ka-null.json','KA_PASS'),(pk+'/raw/KA_READER_ka-ri.json','KA_PASS'),(pk+'/raw/KA_READER_ka-cl.json','KA_PASS')):
    d=json.load(open(p,encoding='utf-8'))
    if d.get(k) is not True: print('NOT PASS:',p); sys.exit(1)
" "$PKW" "E:/Projects/TanitAD-artifacts/dagland/motion_c/cl_read" || fail "a known answer is not PASS (KA_NEW, SMOKE_LEVEL_ON, KA_READER ka-null / ka-ri / ka-cl)" 2
log "ZZCLR-GATE0 pins, tests, known answers ok (tool $(pin tool | cut -c1-8), model $(pin model | cut -c1-8), trainer $(pin trainer | cut -c1-8))"

# ---- gate 1: the pulled CLR checkpoints (REFUSED when absent)
"$PY" -I "$PKW/code/clr_precheck.py" > "$PK/logs/precheck.out" 2>&1; rc=$?
cat "$PK/logs/precheck.out" | tee -a "$PK/logs/chain.out"
[ $rc -eq 0 ] || fail "pulled-file controls refuse the read (rc $rc): see logs/precheck.out and raw/CLR_PRECHECK.json" "$rc"
log "ZZCLR-GATE1 CLR checkpoints present, sha256 == SHA256SUMS.txt == box-2 marker, argv = RAPR-P01's + the two level tokens, recovery lever = RAPR-P01's, cfg level_head on"

# ---- gate 2: reused arrays and the cards
"$PY" -I -c "
import sys
import importlib.util
spec = importlib.util.spec_from_file_location('clr_read', sys.argv[1] + '/clr_read.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.init()
bad = []
for s in m.SEEDS:
    _, c = m.reused_arrays(s)
    bad += [k for k, v in c.items() if not v['PASS']]
if m.card_rules_sha() != m.CARD_RULES_SHA: bad.append('CL card rules sha')
if m.sha256_file(m.CARD) != m.A2_CARD_SHA or m.sha256_file(m.CARD) != open(m.A2_FROZEN, encoding='utf-8').readline().split()[0]: bad.append('A2 card sha')
print('REUSED/CARDS', 'BAD ' + str(bad) if bad else 'ok')
sys.exit(1 if bad else 0)" "$PKW/code" > "$PK/logs/gate2.out" 2>&1 || fail "gate 2 (reused arrays / cards / reader pins): $(tail -n 2 "$PK/logs/gate2.out" | tr '\n' ' ')" 6
log "ZZCLR-GATE2 reused arrays (14), the CL card's rules and the A2 card match their pins"

# ---- gate 3: CPU
busy() { powershell -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'v8f_t1_eval_[a-z_0-9]+[.]py\S*\s+run|extract_smooth[.]py|run_tool_subset[.]py|cl_readout[.]py|clr_readout[.]py' -and \$_.ProcessId -ne \$PID }).Count" | tr -d '\r '; }
w=0; while [ "$(busy)" != "0" ] && [ -n "$(busy)" ]; do sleep 30; w=$((w+30)); [ $w -ge 10800 ] && { log "ZZCLR-BUSY-TIMEOUT another rollout or extractor still runs after 3 h; proceeding"; break; }; done
[ $w -gt 0 ] && log "ZZCLR-CPU-WAIT ${w}s"

CARD_LINE="PREREG_LEVELHEAD_A2.md sha256 $(sha256sum "$CARD" | cut -c1-64)"
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
pre() {   # tag seed run threads flags...
  local tag=$1 s=$2 run=$3 th=$4; shift 4; rm -rf "$S/smoke/$tag"; mkdir -p "$S/smoke/$tag"
  run_tool "$tag" "$RDBW/M2b_s${s}_recft_${run}/ckpt_last.pt" "$SW/smoke/$tag" "$th" --max-windows 6 "$@" > "$PK/logs/smoke_$tag.out" 2>&1; local rc=$?
  [ $rc -eq 0 ] && [ -f "$S/smoke/$tag/ticks/ticks_$tag.npz" ] || fail "pre-flight $tag rc=$rc (a ${run} checkpoint does not load in the read tree, or a flag is refused): see logs/smoke_$tag.out" 7
  log "ZZCLR-PREFLIGHT-OK $tag"
}
pre PRE_CLR_s0_ri_kbox 0 CLR "$TH_RI" --smooth sg72 --scene reindexed --feasible kbox
pre PRE_CLR_s1_hold 1 CLR "$TH_LOGGED" --smooth sg72 --scene hold_now --freeze-card-ref "$CARD_LINE"
pre PRE_RAPRP01_s1_ri 1 RAPRP01 "$TH_RI" --smooth sg72 --scene reindexed
log "ZZCLR-START tool $(pin tool | cut -c1-8) threads reindexed=$TH_RI logged=$TH_LOGGED"

run_one() {   # tag seed scene feas run
  local tag=$1 s=$2 scene=$3 feas=$4 run=$5 ck="$RDBW/M2b_s${2}_recft_${5}/ckpt_last.pt"
  [ -f "$OUT/done_$tag.txt" ] && { log "ZZCLR-SKIP $tag"; return 0; }
  [ -d "$OUT/$tag" ] && mv "$OUT/$tag" "$OUT/_incomplete_${tag}_$(date +%H%M%S)"
  mkdir -p "$OUT/$tag"
  local flags="--smooth sg72 --scene $scene"; local extra=()
  [ "$feas" != "off" ] && flags="$flags --feasible $feas"
  [ "$scene" = "hold_now" ] && extra=(--freeze-card-ref "$CARD_LINE")
  local th=$TH_LOGGED; [ "$scene" = "reindexed" ] && th=$TH_RI
  local t0=$(date +%s)
  log "ZZCLR-RUN $tag flags=$flags threads=$th ckpt=M2b_s${s}_recft_${run}"
  # shellcheck disable=SC2086
  run_tool "$tag" "$ck" "$OUTW/$tag" "$th" $flags "${extra[@]}" > "$PK/logs/run_$tag.out" 2>&1; local rc=$?
  if [ $rc -eq 0 ]; then
    printf '{"tag": "%s", "flags": "%s", "ckpt": "%s", "ckpt_sha256": "%s", "tool": "%s", "tool_sha256": "%s", "threads": %s, "wall_s": %s}\n' "$tag" "$flags" "$ck" \
       "$(sha256sum "$RDB/M2b_s${s}_recft_${run}/ckpt_last.pt" | cut -c1-64)" "$TOOLW" "$(pin tool)" "$th" "$(( $(date +%s) - t0 ))" > "$OUT/done_$tag.txt"
    log "ZZCLR-RUN-DONE $tag wall=$(( $(date +%s) - t0 ))s"
  else
    log "ZZCLR-RUN-FAIL $tag rc=$rc"
  fi
  return $rc
}
nfail=0
while IFS=$'\t' read -r tag seed scene feas run <&3; do
  run_one "$tag" "$seed" "$scene" "$feas" "$run" || nfail=$((nfail+1))
done 3< <(table)
log "ZZCLR-ROLLOUTS-DONE nfail=$nfail"
missing=""; while IFS=$'\t' read -r tag rest; do [ -f "$OUT/done_$tag.txt" ] || missing="$missing $tag"; done < <(table)
[ -z "$missing" ] || fail "missing done markers:$missing (see logs/run_<tag>.out)" 3
echo "$(date +%FT%T) twelve CLR-read rollouts done" > "$PK/raw/CLRRO.done"
log "ZZCLR-CLRRO-DONE"

# ---- the reader (controls first; a refused control is a result to report)
log "ZZCLR-READ start"
"$PY" -I "$PKW/code/clr_read.py" --mode production --out-root "$OUTW" --work-dir "$SW/work" --raw "$PKW/raw" --workers 6 > "$PK/logs/read.out" 2>&1; rc=$?
log "ZZCLR-READ rc=$rc"
tail -n 3 "$PK/logs/read.out" | tee -a "$PK/logs/chain.out"
if [ $rc -eq 0 ] && [ -f "$PK/raw/CLR_READ.json" ] && [ -f "$PK/RESULTS.md" ]; then
  echo "$(date +%FT%T) CLR_READ.json and RESULTS.md written" > "$PK/raw/CLREAD.done"
  log "ZZCLR-READ-DONE"
else
  echo "$(date +%FT%T) reader rc=$rc (see logs/read.out; a REFUSED control names itself in raw/CLR_READ.json 'refused')" > "$PK/raw/CLREAD.fail"
  log "ZZCLR-READ-FAIL rc=$rc"
  exit "$rc"
fi

# ---- A1's O-sensitivity read-out on the CLR checkpoints (decides LLR's wiring); a failure here does not touch CLREAD.done
mkdir -p "$PK/readout"
ro_rc=0
for s in 0 1; do
  OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
    "$PY" -I "$PKW/code/clr_readout.py" --seed $s --threads 12 --batch 32 > "$PK/logs/readout_s$s.out" 2>&1 < /dev/null || ro_rc=$?
  log "ZZCLR-READOUT s$s rc=$ro_rc"
done
if [ $ro_rc -eq 0 ] && "$PY" -I "$PKW/code/clr_merge_readout.py" > "$PK/logs/readout_merge.out" 2>&1 && "$PY" -I "$PKW/code/clr_wiring.py" > "$PK/logs/wiring.out" 2>&1; then
  echo "$(date +%FT%T) readout/CLR_READOUT.json, raw/CLR_WIRING.json and WIRING.md written" > "$PK/raw/CLRREADOUT.done"
  tail -n 1 "$PK/logs/wiring.out" | tee -a "$PK/logs/chain.out"
  log "ZZCLR-CHAIN-DONE"
else
  echo "$(date +%FT%T) read-out rc=$ro_rc (see logs/readout_s*.out, logs/readout_merge.out, logs/wiring.out)" > "$PK/raw/CLRREADOUT.fail"
  log "ZZCLR-READOUT-FAIL"
  exit 4
fi
