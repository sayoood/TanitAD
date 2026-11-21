#!/usr/bin/env bash
# E-GAIN-D chain (PREREG_WSP_GAIN_D.md section 3), dev-box CPU, zero training. Resumable: a finished stage is skipped on relaunch.
#   A  READ 1 control (128 TRAIN windows: the instrument reproduces the banked fit-time C0 values; the edit's plan-speed shift equals its analytic value)
#   B  READ 1 on the 20,000 TRAIN fit windows, 3 shards in parallel (3 workers; the banked train measure used 3 x 5 threads) -> raw/READ1D_TRAIN_shard{0..2}.npz
#   C  READ 1 verdict (MECH-OK / MECH-FAIL)                                               -> raw/READ1_D.json.  MECH-FAIL STOPS THE CHAIN: nothing else is read (card section 3, item 1).
#   D  READ 1 beside: the eval-141 band gains (586 windows)                              -> raw/READ1_D_EVAL.json
#   E  READ 2: two T1-kin rollouts of the edited C0 on eval-141, scene `reindexed` (the deciding scene of the LF / A-series), ONE at a time
#   F  READ 2 reader (wspd_read.py)                                                      -> raw/READ2_D.json, raw/RESULTS_READ2_D.md
# Markers: raw/CHAIN_STAGE_<X>.done ; raw/CHAIN_D.done (all stages) | raw/CHAIN_D.mechfail (READ 1 failed, stopped by design) | raw/CHAIN_D.fail (a stage failed). Log: logs/chain_d.out (ZZWSPD-* lines).
# Launch (survives the caller):  powershell -NoProfile -Command "Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd.exe /c E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\wsp_edit_d\chain_d.cmd'}"
set -u
export PATH="/usr/bin:/bin:$PATH"
PK=/e/Projects/TanitAD-artifacts/dagland/motion_c/theory/wsp_edit_d
PKW=E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/wsp_edit_d
CARD=/e/Projects/TanitAD-artifacts/dagland/motion_c/theory/PREREG_WSP_GAIN_D.md
SW=F:/TanitAD-scratch/motion_c/wsp_edit_d; S=/f/TanitAD-scratch/motion_c/wsp_edit_d
TOOLW=F:/TanitAD-scratch/motion_c/ri/tree_ri/stack/scripts/v8f_t1_eval_ri.py
TOOL=/f/TanitAD-scratch/motion_c/ri/tree_ri/stack/scripts/v8f_t1_eval_ri.py
TOOL_SHA=0417c838355a51f6f7084c17b5acada96c5b6cbd9eef54cc2602d377bf3c61e7
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
ART=E:/Projects/TanitAD-artifacts
DATA="$ART/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
EDIT=$ART/v8f-wspedit-d
NT=${WSPD_NT:-5}       # threads per READ 1 shard (3 shards => 15 threads, the banked train-measure configuration)
TH=${WSPD_TH:-12}      # threads of a rollout
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1
mkdir -p "$S/out" "$S/work" "$PK/logs" "$PK/raw"
log() { echo "$(date +%FT%T) $*" | tee -a "$PK/logs/chain_d.out"; }
fail() { log "ZZWSPD-FAIL $1"; echo "$(date +%FT%T) $1" > "$PK/raw/CHAIN_D.fail"; exit "${2:-3}"; }
rm -f "$PK/raw/CHAIN_D.fail"
# ---- gates
grep -q "PIN " <(sed -n '/## 6. Log/,$p' "$CARD") || fail "no PIN entry in the D card's section 6" 2
head -c 5211 "$CARD" | sha256sum | grep -q '^d6e691a3065fbd7c8fa6014931ba41a447477ceb8109bfff460228c5e503e2eb' || fail "the D card's frozen prefix changed" 2
[ "$(sha256sum "$TOOL" | cut -c1-64)" = "$TOOL_SHA" ] || fail "tool sha != the RIC0 tool's" 2
"$PY" -I -c "
import json,hashlib,sys
f=json.load(open(sys.argv[1]))
for s in ('0','1'):
    a=f['seeds'][s]; h=hashlib.sha256(open(a['ckpt'],'rb').read()).hexdigest()
    assert h==a['ckpt_sha256'], s
" "$PKW/raw/FIT_RESULT_D.json" || fail "an edited checkpoint differs from FIT_RESULT_D.json" 2
stage_done() { [ -f "$PK/raw/CHAIN_STAGE_$1.done" ]; }
stage_mark() { echo "$(date +%FT%T)" > "$PK/raw/CHAIN_STAGE_$1.done"; log "ZZWSPD-STAGE-$1-DONE"; }
# ---- A control
if ! stage_done A; then
  log "ZZWSPD-A control"
  "$PY" -I "$PKW/read1_d.py" control --nt 8 > "$PK/logs/read1d_control.out" 2>&1 || fail "READ 1 control failed (see logs/read1d_control.out)" 4
  stage_mark A
fi
# ---- B READ 1 on the train fit windows (4 shards)
if ! stage_done B; then
  log "ZZWSPD-B read1 train shards x3 (nt $NT)"
  pids=""
  for i in 0 1 2; do
    [ -f "$PK/raw/READ1D_TRAIN_shard$i.npz" ] && continue
    "$PY" -I "$PKW/read1_d.py" train --shard $i --nshards 3 --nt $NT < /dev/null > "$PK/logs/read1d_train_shard$i.out" 2>&1 &
    pids="$pids $!"
  done
  nf=0; for p in $pids; do wait $p || nf=$((nf+1)); done
  [ $nf -eq 0 ] || fail "$nf READ 1 train shards failed (see logs/read1d_train_shard*.out)" 3
  ls "$PK"/raw/READ1D_TRAIN_shard[0-2].npz > /dev/null 2>&1 || fail "READ 1 train shards missing" 3
  stage_mark B
fi
# ---- C verdict
if ! stage_done C; then
  "$PY" -I "$PKW/read1_d.py" combine > "$PK/logs/read1d_combine.out" 2>&1 || fail "READ 1 combine failed (see logs/read1d_combine.out)" 3
  log "$(tail -n 3 "$PK/logs/read1d_combine.out" | tr '\n' ' ')"
  stage_mark C
fi
if ! grep -q '"verdict": "MECH-OK"' "$PK/raw/READ1_D.json"; then
  echo "$(date +%FT%T) READ 1 on the TRAIN fit windows is MECH-FAIL: nothing else is read (card section 3, item 1)" > "$PK/raw/CHAIN_D.mechfail"; log "ZZWSPD-MECH-FAIL"; exit 0
fi
# ---- D eval-141 gains beside
if ! stage_done D; then
  "$PY" -I "$PKW/read1_d.py" eval --nt 8 > "$PK/logs/read1d_eval.out" 2>&1 || fail "READ 1 eval-141 beside failed (see logs/read1d_eval.out)" 3
  stage_mark D
fi
# ---- E rollouts (one at a time; wait while another v8f_t1_eval run or extract_smooth is alive)
busy() { powershell -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'v8f_t1_eval_[a-z_0-9]+[.]py\S*\s+run|extract_smooth[.]py' -and \$_.ProcessId -ne \$PID }).Count" | tr -d '\r '; }
w=0; while [ "$(busy)" != "0" ] && [ -n "$(busy)" ]; do sleep 30; w=$((w+30)); [ $w -ge 7200 ] && break; done
run_one() {   # tag seed
  local tag=$1 s=$2 ck="$EDIT/M2b_s${2}_wspedit_d/ckpt_last.pt"
  [ -f "$S/out/done_$tag.txt" ] && { log "ZZWSPD-SKIP $tag"; return 0; }
  [ -d "$S/out/$tag" ] && mv "$S/out/$tag" "$S/out/_incomplete_${tag}_$(date +%H%M%S)"
  mkdir -p "$S/out/$tag"
  local t0=$(date +%s)
  log "ZZWSPD-RUN $tag ckpt=$ck"
  OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH \
  "$PY" -I "$TOOLW" run --ckpt "$ck" --tag "$tag" --tracks "$ART/sfresh_inputs/inputs/native_tracks.pt" --eval-cache "$ART/sfresh_inputs/cache" \
     --eval-grid "$DATA/eval_grid_windows.json" --eval-join "$DATA/join_eval141.npz" --out-dir "$SW/out/$tag" --replan-steps 10 --nav deadreckon --schedule periodic --limit causal \
     --dump-ticks "$SW/out/$tag/ticks" --dump-heads --device cpu --smooth sg72 --scene reindexed < /dev/null > "$PK/logs/run_$tag.out" 2>&1; local rc=$?
  if [ $rc -eq 0 ]; then
    printf '{"tag": "%s", "flags": "--smooth sg72 --scene reindexed", "ckpt": "%s", "ckpt_sha256": "%s", "tool": "%s", "tool_sha256": "%s", "threads": %s, "wall_s": %s}\n' "$tag" "$ck" "$(sha256sum "$ck" | cut -c1-64)" "$TOOLW" "$TOOL_SHA" "$TH" "$(( $(date +%s) - t0 ))" > "$S/out/done_$tag.txt"
    log "ZZWSPD-RUN-DONE $tag wall=$(( $(date +%s) - t0 ))s"
  else log "ZZWSPD-RUN-FAIL $tag rc=$rc"; fi
  return $rc
}
if ! stage_done E; then
  nfail=0
  for r in "RIWSPD_s0 0" "RIWSPD_s1 1"; do set -- $r; run_one "$1" "$2" || nfail=$((nfail+1)); done
  [ $nfail -eq 0 ] || fail "$nfail rollouts failed (see logs/run_RIWSPD_*.out)" 3
  stage_mark E
fi
# ---- F reader
if ! stage_done F; then
  "$PY" -I "$PKW/wspd_read.py" --out-root "$SW/out" --work-dir "$SW/work" --raw "$PKW/raw" --workers 4 > "$PK/logs/read2_d.out" 2>&1; rc=$?
  log "ZZWSPD-READ rc=$rc"; tail -n 4 "$PK/logs/read2_d.out" | tee -a "$PK/logs/chain_d.out"
  [ $rc -eq 0 ] || fail "reader rc=$rc (see logs/read2_d.out)" 3
  stage_mark F
fi
echo "$(date +%FT%T) E-GAIN-D chain done (READ 1 MECH-OK, READ 2 written)" > "$PK/raw/CHAIN_D.done"; log "ZZWSPD-CHAIN-DONE"
# ---- G distance keeping BESIDE (lowest priority; a failure here never changes the READ 1 / READ 2 results): dk_beside.sh, if present
[ -f "$PK/dk_beside.sh" ] && { bash "$PK/dk_beside.sh" > "$PK/logs/dk_beside.out" 2>&1; rc=$?; log "ZZWSPD-DK-BESIDE rc=$rc"; echo "$(date +%FT%T) rc=$rc" > "$PK/raw/DK_BESIDE.finished"; }
exit 0
