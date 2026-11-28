#!/usr/bin/env bash
# LLR READ chain (PREREG_LEVELHEAD_A2.md, Reads and Outcomes + A2_RULING_1.md + FROZEN_LLR.txt): the two LLR checkpoints (CLR's recipe + the lead head, token wiring, trained on box 2) over ALL 2,399 eval windows,
# then the reader, then the O-sensitivity read-out adapted to the token wiring.  Dev-box CPU, ZERO training, no ssh, ONE rollout at a time.  Cloned from clr_read/clr_read_chain.sh.
# NOT launched by the package that wrote it: the Master Mind starts it AFTER the LLR checkpoints are pulled into E:/Projects/TanitAD-artifacts/v8f-recover/ (M2b_s{0,1}_recft_LLR/, llr_markers/).
#   usage:  llr_read_chain.sh --dry-run     lists the 10 rollouts, the gates and what is present; runs nothing, writes no marker
#           llr_read_chain.sh               (normal; start through llr_read_chain.cmd = WMI Win32_Process.Create so it survives the caller)
# READ TREE: F:/TanitAD-scratch/motion_c/llr_read/tree = the CL read tree with the 83c9ee0 lead-head model / data / trainer (build_llr_read_tree.py); the CL tree's model cannot build an LLR checkpoint.  KA-NEW2 (raw/KA_NEW2.json)
# proves the tree reproduces the banked C0 / RIC0 / RICLR rollouts bit for bit before any LLR read.
# Rollouts in PRIORITY order (a killed chain still yields the deciding contrast first), reader flags (--replan-steps 10 --nav deadreckon --schedule periodic --limit causal --dump-ticks <out>/ticks --dump-heads --device cpu --smooth sg72):
#   RILLR_s0 RILLR_s1 (--scene reindexed, DECIDING)  RILLRC_s0 RILLRC_s1 (reindexed + --feasible kbox, D)  LLR_s0 LLR_s1 (logged)  LLRC_s0 LLRC_s1 (logged + kbox)  FZLLR_s0 FZLLR_s1 (hold_now)
#   NO attribution-base rollout: the card's attribution LLR - CLR reads CLR's banked arrays (clr_read/raw/arrays: CLR logged, RICLR reindexed; sha256-pinned in the reader).
#   Threads: logged / hold_now arms LLR_TH_LOGGED (default 16 = the banked C0 / C0C / FZC0 / C0REP / CLR rollouts' count), reindexed arms (RILLR, RILLRC) LLR_TH_RI (default 12 = RIC0 / RIC0C / RICLR's).  MEASURED (KA-NEW2): the same tool at
#   the base's own thread count reproduces the banked rollout bit for bit; at another count a 1e-5 m rounding grows to 0.6 m on one window.
# Gates (REFUSE = raw/LLRRO.fail with the reason, exit code):
#   0  PINS.txt: the reader, pre-check, read-out, tests, this script, the LLR tree's tool / model / trainer / data and the CLR package's evidence match their sha256; the unit-test logs are green; the mutation log says ALL RED;
#      KA-NEW2 and the reader known answers (ka-null, ka-ri, ka-clr) say PASS                                                                                                     (exit 2)
#   1  code/llr_precheck.py: both LLR checkpoints present (absent = REFUSED), sha256 == SHA256SUMS.txt == the box-2 marker llr_markers/RUN_LLR_{s}.done evidence, argv = CLR's + `--lead-head on --w-lead W_LEAD --aux-cond token`,
#      cfg lead_head on / w_lead / aux_cond, e_aux keys, gate token, CLR's own checkpoint = the one the banked CLR arrays came from                                                (exit 5 absent, 8 failed)
#   2  the reused arrays' sha256 (16: C0 family, RIC0 family, CLR, RICLR), the CL card's rules sha, the A2 card sha = FROZEN_SHA.txt                                                (exit 6)
#   3  no other tool rollout / extractor runs (CPU shared; waits at most 3 h)
#   4  a 6-window PRE-FLIGHT per scene family: s0 LLR --scene reindexed --feasible kbox, s1 LLR --scene hold_now, s0 LLR --scene logged: each must exit 0 and write its ticks file                (exit 7)
# Each rollout writes done_<tag>.txt only on rc 0; a finished run is skipped on re-launch (RESUME = launch llr_read_chain.cmd again); an incomplete out dir is MOVED to _incomplete_*, never deleted.
# Markers: ZZLLR-* lines in logs/chain.out; raw/LLRRO.done (all ten rollouts) / raw/LLRRO.fail (reason inside); then the reader (code/llr_read.py --mode production): raw/LLREAD.done (raw/LLR_READ.json + RESULTS.md) /
#   raw/LLREAD.fail; then the token-wiring read-out: raw/LLRREADOUT.done (readout/LLR_READOUT.json, SENSITIVITY.md) / raw/LLRREADOUT.fail.  A REFUSED control of the reader is a result.
# WRITES: this package (logs, raw, readout, RESULTS.md, SENSITIVITY.md) and F:/TanitAD-scratch/motion_c/llr_read/{out,smoke,work,readout}.  The shared CL tree is not touched; the LLR tree is READ ONLY here.
set -u
DRY=0; for t in "$@"; do [ "$t" = "--dry-run" ] && DRY=1; done
PK=${LLR_PK:-/e/Projects/TanitAD-artifacts/dagland/motion_c/llr_read}
PKW=${LLR_PKW:-E:/Projects/TanitAD-artifacts/dagland/motion_c/llr_read}
S=/f/TanitAD-scratch/motion_c/llr_read
SW=F:/TanitAD-scratch/motion_c/llr_read
TREE=/f/TanitAD-scratch/motion_c/llr_read/tree
TOOLW=F:/TanitAD-scratch/motion_c/llr_read/tree/stack/scripts/v8f_t1_eval_ri.py
TOOL=$TREE/stack/scripts/v8f_t1_eval_ri.py
OUT=$S/out
OUTW=$SW/out
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
ART=E:/Projects/TanitAD-artifacts
RDB=/e/Projects/TanitAD-artifacts/v8f-recover
RDBW=$ART/v8f-recover
DATA="$ART/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
CARD=/e/Projects/TanitAD-artifacts/dagland/motion_c/a2_prereg/PREREG_LEVELHEAD_A2.md
TH_LOGGED=${LLR_TH_LOGGED:-16}
TH_RI=${LLR_TH_RI:-12}
mkdir -p "$OUT" "$PK/logs" "$PK/raw" "$S/smoke" "$S/work" "$S/readout"
log() { echo "$(date +%FT%T) $*" | tee -a "$PK/logs/chain.out"; }
fail() { log "ZZLLR-FAIL $1"; echo "$(date +%FT%T) $1" > "$PK/raw/LLRRO.fail"; exit "${2:-3}"; }
table() {   # tag seed scene feasible run   (PRIORITY order; tab-separated)
  local r
  for r in "RILLR_s0 0 reindexed off LLR" "RILLR_s1 1 reindexed off LLR" "RILLRC_s0 0 reindexed kbox LLR" "RILLRC_s1 1 reindexed kbox LLR" \
           "LLR_s0 0 logged off LLR" "LLR_s1 1 logged off LLR" "LLRC_s0 0 logged kbox LLR" "LLRC_s1 1 logged kbox LLR" "FZLLR_s0 0 hold_now off LLR" "FZLLR_s1 1 hold_now off LLR"; do
    set -- $r
    printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5"
  done
}
pin() { sed -n "s|^$1 sha256 \([0-9a-f]*\)\$|\1|p" "$PK/PINS.txt" | tr -d '\r'; }

if [ $DRY -eq 1 ]; then
  echo "LLR read chain, DRY RUN (nothing runs, no marker is written)"
  echo "rollouts in priority order (tag seed scene feasible run; threads: reindexed $TH_RI, others $TH_LOGGED):"; table
  for s in 0 1; do
    r=M2b_s${s}_recft_LLR; b=M2b_s${s}_recft_CLR
    echo "checkpoint $RDB/$r/ckpt_last.pt: $([ -f "$RDB/$r/ckpt_last.pt" ] && echo PRESENT || echo ABSENT)   marker $RDB/llr_markers/RUN_LLR_$s.done: $([ -f "$RDB/llr_markers/RUN_LLR_$s.done" ] && echo PRESENT || echo ABSENT)   gate token: $([ -f "$RDB/llr_markers/gate_token_LLR_$s.json" ] && echo PRESENT || echo ABSENT)   base (CLR ckpt) $RDB/$b/ckpt_last.pt: $([ -f "$RDB/$b/ckpt_last.pt" ] && echo PRESENT || echo ABSENT)   banked CLR arrays: $([ -f "$ART/dagland/motion_c/clr_read/raw/arrays/CLR_s$s.npz" ] && [ -f "$ART/dagland/motion_c/clr_read/raw/arrays/RICLR_s$s.npz" ] && echo PRESENT || echo ABSENT)"
  done
  echo "FROZEN_LLR.txt + A2 card + FROZEN_SHA: $([ -f "$(dirname "$CARD")/FROZEN_LLR.txt" ] && [ -f "$CARD" ] && [ -f "$(dirname "$CARD")/FROZEN_SHA.txt" ] && echo PRESENT || echo ABSENT)   landed_tip.txt: $([ -f "$(dirname "$CARD")/../llr_launch/landed_tip.txt" ] && echo PRESENT || echo ABSENT)"
  echo "tree tool: $([ -f "$TOOL" ] && sha256sum "$TOOL" | cut -c1-16 || echo ABSENT)   tree model: $([ -f "$TREE/stack/tanitad/models/v8f_mini.py" ] && sha256sum "$TREE/stack/tanitad/models/v8f_mini.py" | cut -c1-16 || echo ABSENT)   PINS.txt: $([ -f "$PK/PINS.txt" ] && echo PRESENT || echo ABSENT)   KA_NEW2.json: $([ -f "$PK/raw/KA_NEW2.json" ] && echo PRESENT || echo ABSENT)"
  exit 0
fi

if [ -f "$PK/raw/LLRRO.running" ]; then
  op=$(cat "$PK/raw/LLRRO.running" | tr -d '\r\n ')
  if [ -n "$op" ] && kill -0 "$op" 2>/dev/null; then log "ZZLLR-ALREADY-RUNNING pid $op: not starting a second chain"; exit 9; fi
  mv "$PK/raw/LLRRO.running" "$PK/raw/_stale_LLRRO.running_$(date +%H%M%S)"
fi
echo $$ > "$PK/raw/LLRRO.running"
trap 'echo "exited $(date +%FT%T)" > "$PK/raw/LLRRO.running"' EXIT
for m in LLRRO.done LLRRO.fail LLREAD.done LLREAD.fail LLRREADOUT.done LLRREADOUT.fail; do [ -f "$PK/raw/$m" ] && mv "$PK/raw/$m" "$PK/raw/_old_${m}_$(date +%H%M%S)"; done

# ---- gate 0: pins, tests, known answers
[ -f "$PK/PINS.txt" ] || fail "PINS.txt is absent (run: python -I code/make_pins.py)" 2
for f in $(sed -n 's|^\(code/[^ ]*\|tests/[^ ]*\|llr_read_chain[.]sh\|llr_read_chain[.]cmd\) sha256 .*|\1|p' "$PK/PINS.txt" | tr -d '\r'); do
  want=$(pin "$f"); got=$(sha256sum "$PK/$f" | cut -c1-64)
  [ -n "$want" ] && [ "$want" = "$got" ] || fail "$f sha $got != PINS.txt ($want)" 2
done
for need in code/llr_read.py code/llr_precheck.py code/llr_readout.py code/llr_edits.py code/ka_new2.py tests/test_llr_read.py tests/test_llr_chain_gates.py tests/test_llr_readout.py llr_read_chain.sh; do
  [ -n "$(pin "$need")" ] || fail "PINS.txt carries no pin for $need" 2
done
for kv in "stack/scripts/v8f_t1_eval_ri.py:tool" "stack/tanitad/models/v8f_mini.py:model" "stack/scripts/v8f_mini_train.py:trainer" "stack/tanitad/data/v8f_mini_data.py:data"; do
  f=${kv%%:*}; k=${kv##*:}; want=$(pin "$k"); got=$(sha256sum "$TREE/$f" | cut -c1-64)
  [ -n "$want" ] && [ "$want" = "$got" ] || fail "LLR tree $k ($f) sha $got != PINS.txt ($want)" 2
done
for ev in "dagland/motion_c/clr_read/raw/CLR_READ.json:clr_read_json" "dagland/motion_c/clr_read/raw/CLR_PRECHECK.json:clr_precheck_json"; do
  f=${ev%%:*}; k=${ev##*:}; want=$(pin "$k"); got=$(sha256sum "/e/Projects/TanitAD-artifacts/$f" | cut -c1-64)
  [ -n "$want" ] && [ "$want" = "$got" ] || fail "CLR package evidence $k ($f) sha $got != PINS.txt ($want)" 2
done
for lg in pytest_llr_read pytest_llr_chain_gates pytest_llr_readout; do
  last=$(grep -v '^[[:space:]]*$' "$PK/logs/$lg.log" | tail -n 1)
  case "$last" in *" passed"*) case "$last" in *failed*|*error*) fail "unit tests not green ($lg): $last" 2;; esac;; *) fail "unit tests not green ($lg): $last" 2;; esac
done
grep -q "^MUTATIONS ALL RED" "$PK/logs/mutations_llr.log" || fail "the mutation log does not carry MUTATIONS ALL RED" 2
"$PY" -I -c "
import json,sys
pk=sys.argv[1]
for p,k in ((pk+'/raw/KA_NEW2.json','PASS'),(pk+'/raw/KA_READER_ka-null.json','KA_PASS'),(pk+'/raw/KA_READER_ka-ri.json','KA_PASS'),(pk+'/raw/KA_READER_ka-clr.json','KA_PASS')):
    d=json.load(open(p,encoding='utf-8'))
    if d.get(k) is not True: print('NOT PASS:',p); sys.exit(1)
" "$PKW" || fail "a known answer is not PASS (KA_NEW2, KA_READER ka-null / ka-ri / ka-clr)" 2
log "ZZLLR-GATE0 pins, tests, known answers ok (tool $(pin tool | cut -c1-8), model $(pin model | cut -c1-8), trainer $(pin trainer | cut -c1-8))"

# ---- gate 1: the pulled LLR checkpoints (REFUSED when absent)
"$PY" -I "$PKW/code/llr_precheck.py" > "$PK/logs/precheck.out" 2>&1; rc=$?
cat "$PK/logs/precheck.out" | tee -a "$PK/logs/chain.out"
[ $rc -eq 0 ] || fail "pulled-file controls refuse the read (rc $rc): see logs/precheck.out and raw/LLR_PRECHECK.json" "$rc"
log "ZZLLR-GATE1 LLR checkpoints present, sha256 == SHA256SUMS.txt == box-2 marker, argv = CLR's + the three lead tokens (W_LEAD, token wiring), cfg lead_head on, e_aux keys, gate token, CLR base checkpoint = the banked arrays' checkpoint"

# ---- gate 2: reused arrays and the cards
"$PY" -I -c "
import sys
import importlib.util
spec = importlib.util.spec_from_file_location('llr_read', sys.argv[1] + '/llr_read.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.init()
bad = []
for s in m.SEEDS:
    _, c = m.reused_arrays(s)
    bad += [k for k, v in c.items() if not v['PASS']]
if m.card_rules_sha() != m.CARD_RULES_SHA: bad.append('CL card rules sha')
if m.sha256_file(m.CARD) != m.A2_CARD_SHA or m.sha256_file(m.CARD) != open(m.A2_FROZEN, encoding='utf-8').readline().split()[0]: bad.append('A2 card sha')
print('REUSED/CARDS', 'BAD ' + str(bad) if bad else 'ok')
sys.exit(1 if bad else 0)" "$PKW/code" > "$PK/logs/gate2.out" 2>&1 || fail "gate 2 (reused arrays / cards / reader pins): $(tail -n 2 "$PK/logs/gate2.out" | tr '\n' ' ')" 6
log "ZZLLR-GATE2 reused arrays (16, the banked CLR base included), the CL card's rules and the A2 card match their pins"

# ---- gate 3: CPU
busy() { powershell -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'v8f_t1_eval_[a-z_0-9]+[.]py\S*\s+run|extract_smooth[.]py|run_tool_subset[.]py|cl_readout[.]py|llr_readout[.]py' -and \$_.ProcessId -ne \$PID }).Count" | tr -d '\r '; }
w=0; while [ "$(busy)" != "0" ] && [ -n "$(busy)" ]; do sleep 30; w=$((w+30)); [ $w -ge 10800 ] && { log "ZZLLR-BUSY-TIMEOUT another rollout or extractor still runs after 3 h; proceeding"; break; }; done
[ $w -gt 0 ] && log "ZZLLR-CPU-WAIT ${w}s"

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
  log "ZZLLR-PREFLIGHT-OK $tag"
}
pre PRE_LLR_s0_ri_kbox 0 LLR "$TH_RI" --smooth sg72 --scene reindexed --feasible kbox
pre PRE_LLR_s1_hold 1 LLR "$TH_LOGGED" --smooth sg72 --scene hold_now --freeze-card-ref "$CARD_LINE"
pre PRE_LLR_s0_logged 0 LLR "$TH_LOGGED" --smooth sg72 --scene logged
log "ZZLLR-START tool $(pin tool | cut -c1-8) threads reindexed=$TH_RI logged=$TH_LOGGED"

run_one() {   # tag seed scene feas run
  local tag=$1 s=$2 scene=$3 feas=$4 run=$5 ck="$RDBW/M2b_s${2}_recft_${5}/ckpt_last.pt"
  [ -f "$OUT/done_$tag.txt" ] && { log "ZZLLR-SKIP $tag"; return 0; }
  [ -d "$OUT/$tag" ] && mv "$OUT/$tag" "$OUT/_incomplete_${tag}_$(date +%H%M%S)"
  mkdir -p "$OUT/$tag"
  local flags="--smooth sg72 --scene $scene"; local extra=()
  [ "$feas" != "off" ] && flags="$flags --feasible $feas"
  [ "$scene" = "hold_now" ] && extra=(--freeze-card-ref "$CARD_LINE")
  local th=$TH_LOGGED; [ "$scene" = "reindexed" ] && th=$TH_RI
  local t0=$(date +%s)
  log "ZZLLR-RUN $tag flags=$flags threads=$th ckpt=M2b_s${s}_recft_${run}"
  # shellcheck disable=SC2086
  run_tool "$tag" "$ck" "$OUTW/$tag" "$th" $flags "${extra[@]}" > "$PK/logs/run_$tag.out" 2>&1; local rc=$?
  if [ $rc -eq 0 ]; then
    printf '{"tag": "%s", "flags": "%s", "ckpt": "%s", "ckpt_sha256": "%s", "tool": "%s", "tool_sha256": "%s", "threads": %s, "wall_s": %s}\n' "$tag" "$flags" "$ck" \
       "$(sha256sum "$RDB/M2b_s${s}_recft_${run}/ckpt_last.pt" | cut -c1-64)" "$TOOLW" "$(pin tool)" "$th" "$(( $(date +%s) - t0 ))" > "$OUT/done_$tag.txt"
    log "ZZLLR-RUN-DONE $tag wall=$(( $(date +%s) - t0 ))s"
  else
    log "ZZLLR-RUN-FAIL $tag rc=$rc"
  fi
  return $rc
}
nfail=0
while IFS=$'\t' read -r tag seed scene feas run <&3; do
  run_one "$tag" "$seed" "$scene" "$feas" "$run" || nfail=$((nfail+1))
done 3< <(table)
log "ZZLLR-ROLLOUTS-DONE nfail=$nfail"
missing=""; while IFS=$'\t' read -r tag rest; do [ -f "$OUT/done_$tag.txt" ] || missing="$missing $tag"; done < <(table)
[ -z "$missing" ] || fail "missing done markers:$missing (see logs/run_<tag>.out)" 3
echo "$(date +%FT%T) ten LLR-read rollouts done" > "$PK/raw/LLRRO.done"
log "ZZLLR-LLRRO-DONE"

# ---- the reader (controls first; a refused control is a result to report)
log "ZZLLR-READ start"
"$PY" -I "$PKW/code/llr_read.py" --mode production --out-root "$OUTW" --work-dir "$SW/work" --raw "$PKW/raw" --workers 6 > "$PK/logs/read.out" 2>&1; rc=$?
log "ZZLLR-READ rc=$rc"
tail -n 3 "$PK/logs/read.out" | tee -a "$PK/logs/chain.out"
if [ $rc -eq 0 ] && [ -f "$PK/raw/LLR_READ.json" ] && [ -f "$PK/RESULTS.md" ]; then
  echo "$(date +%FT%T) LLR_READ.json and RESULTS.md written" > "$PK/raw/LLREAD.done"
  log "ZZLLR-READ-DONE"
else
  echo "$(date +%FT%T) reader rc=$rc (see logs/read.out; a REFUSED control names itself in raw/LLR_READ.json 'refused')" > "$PK/raw/LLREAD.fail"
  log "ZZLLR-READ-FAIL rc=$rc"
  exit "$rc"
fi

# ---- the O-sensitivity read-out adapted to the token wiring (the token's level input +-3 m/s, the lead presence input 0 <-> 1); a failure here does not touch LLREAD.done
mkdir -p "$PK/readout"
ro_rc=0
for s in 0 1; do
  OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
    "$PY" -I "$PKW/code/llr_readout.py" --seed $s --threads 12 --batch 32 > "$PK/logs/readout_s$s.out" 2>&1 < /dev/null || ro_rc=$?
  log "ZZLLR-READOUT s$s rc=$ro_rc"
done
if [ $ro_rc -eq 0 ] && "$PY" -I "$PKW/code/llr_readout.py" --merge > "$PK/logs/readout_merge.out" 2>&1; then
  echo "$(date +%FT%T) readout/LLR_READOUT.json and SENSITIVITY.md written" > "$PK/raw/LLRREADOUT.done"
  tail -n 1 "$PK/logs/readout_merge.out" | tee -a "$PK/logs/chain.out"
  log "ZZLLR-CHAIN-DONE"
else
  echo "$(date +%FT%T) read-out rc=$ro_rc (see logs/readout_s*.out, logs/readout_merge.out)" > "$PK/raw/LLRREADOUT.fail"
  log "ZZLLR-READOUT-FAIL"
  exit 4
fi
