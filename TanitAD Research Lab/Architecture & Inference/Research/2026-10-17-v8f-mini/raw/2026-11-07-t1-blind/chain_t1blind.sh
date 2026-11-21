#!/usr/bin/env bash
# T1-BLIND chain (PREREG_T1_BLIND.md), dev-box CPU, zero training. Resumable: a finished stage is skipped on relaunch.
#   P  preflight: the card's frozen prefix (4,574 bytes, sha256 9619414a...), the tool sha, the pinned mean, the gate tests' record (logs/pytest_gates.out: all passed)
#   A  ID_s0 (mode off: the replacement switched off, the RIC0 path) -> G-ID seed 0 (bitwise vs the banked RIC0_s0)           FAIL STOPS THE CHAIN (CHAIN_T1BLIND.gidfail)
#   B  ID_s1 -> G-ID seed 1
#   C  BLIND1_s0 (mode blind)         D  BLIND1_s1
#   E  gates all (G-ID x2, G-MEAN x2 on every received frame, G-EGO x2)  -> raw/GATES.json   FAIL STOPS THE CHAIN (CHAIN_T1BLIND.gatefail)
#   F  reader t1blind_read.py -> raw/READ_T1BLIND.json     G  result writer t1blind_result.py -> RESULT_T1BLIND.md / .json
# Markers in raw/: CHAIN_STAGE_<X>.done ; CHAIN_T1BLIND.done | .fail | .gidfail | .gatefail.  Log: logs/chain_t1blind.out (ZZT1B-* lines).
# Launch (survives the caller):  powershell -NoProfile -Command "Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd.exe /c E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\t1_blind\chain_t1blind.cmd'}"
set -u
export PATH="/usr/bin:/bin:$PATH"
PK=/e/Projects/TanitAD-artifacts/dagland/motion_c/theory/t1_blind
PKW=E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/t1_blind
CARD=/e/Projects/TanitAD-artifacts/dagland/motion_c/theory/PREREG_T1_BLIND.md
SW=F:/TanitAD-scratch/motion_c/t1_blind; S=/f/TanitAD-scratch/motion_c/t1_blind
TOOL=/f/TanitAD-scratch/motion_c/ri/tree_ri/stack/scripts/v8f_t1_eval_ri.py
TOOL_SHA=0417c838355a51f6f7084c17b5acada96c5b6cbd9eef54cc2602d377bf3c61e7
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
ART=E:/Projects/TanitAD-artifacts
DATA="$ART/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
TH=${T1B_TH:-12}      # threads of a rollout = RIC0's (done_RIC0_s*.txt: threads 12); bitwise G-ID depends on it
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1
mkdir -p "$S/out/receipts" "$S/work" "$PK/logs" "$PK/raw"
log() { echo "$(date +%FT%T) $*" | tee -a "$PK/logs/chain_t1blind.out"; }
fail() { log "ZZT1B-FAIL $1"; echo "$(date +%FT%T) $1" > "$PK/raw/CHAIN_T1BLIND.fail"; exit "${2:-3}"; }
rm -f "$PK/raw/CHAIN_T1BLIND.fail"
stage_done() { [ -f "$PK/raw/CHAIN_STAGE_$1.done" ]; }
stage_mark() { echo "$(date +%FT%T)" > "$PK/raw/CHAIN_STAGE_$1.done"; log "ZZT1B-STAGE-$1-DONE"; }
# ---- P preflight
head -c 4574 "$CARD" | sha256sum | grep -q '^9619414ac48ca90b28cb6fecdb044227aff43ffce14d2d365c599a602b678ecd' || fail "the card's frozen prefix changed" 2
[ "$(sha256sum "$TOOL" | cut -c1-64)" = "$TOOL_SHA" ] || fail "tool sha != the RIC0 tool's" 2
grep -qE '^[0-9]+ passed' "$PK/logs/pytest_gates.out" && ! grep -qE 'failed|error' "$PK/logs/pytest_gates.out" || fail "the gate tests' record is absent or not all-passed" 2
busy() { powershell -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'v8f_t1_eval_[a-z_0-9]+[.]py\S*\s+run|extract_smooth[.]py' -and \$_.ProcessId -ne \$PID }).Count" | tr -d '\r '; }
run_one() {   # tag mode seed
  local tag=$1 mode=$2 s=$3 ck="$ART/v8f-recover/M2b_s${3}_recft_C0/ckpt_last.pt"
  [ -f "$S/out/done_$tag.txt" ] && { log "ZZT1B-SKIP $tag"; return 0; }
  local w=0; while [ "$(busy)" != "0" ] && [ -n "$(busy)" ]; do sleep 30; w=$((w+30)); [ $w -ge 7200 ] && break; done
  [ -d "$S/out/$tag" ] && mv "$S/out/$tag" "$S/out/_incomplete_${tag}_$(date +%H%M%S)"
  mkdir -p "$S/out/$tag"
  local t0=$(date +%s)
  log "ZZT1B-RUN $tag mode=$mode ckpt=$ck"
  OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH \
  "$PY" -I "$PKW/t1blind_run.py" --blind-mode "$mode" --receipt-dir "$SW/out/receipts" \
     --ckpt "$ck" --tag "$tag" --tracks "$ART/sfresh_inputs/inputs/native_tracks.pt" --eval-cache "$ART/sfresh_inputs/cache" \
     --eval-grid "$DATA/eval_grid_windows.json" --eval-join "$DATA/join_eval141.npz" --out-dir "$SW/out/$tag" --replan-steps 10 --nav deadreckon --schedule periodic --limit causal \
     --dump-ticks "$SW/out/$tag/ticks" --dump-heads --device cpu --smooth sg72 --scene reindexed < /dev/null > "$PK/logs/run_$tag.out" 2>&1; local rc=$?
  if [ $rc -eq 0 ] && [ -f "$S/out/receipts/receipt_$tag.json" ]; then
    printf '{"tag": "%s", "flags": "--smooth sg72 --scene reindexed", "blind_mode": "%s", "ckpt": "%s", "ckpt_sha256": "%s", "tool": "%s", "tool_sha256": "%s", "wrapper_sha256": "%s", "mean_tensor_sha256": "cd6a7ab5e3c5450ec3528889b04e06e5605fea30ea9ff02a0e36d9ba47a2fcb9", "threads": %s, "wall_s": %s}\n' \
       "$tag" "$mode" "$ck" "$(sha256sum "$ck" | cut -c1-64)" "F:/TanitAD-scratch/motion_c/ri/tree_ri/stack/scripts/v8f_t1_eval_ri.py" "$TOOL_SHA" "$(sha256sum "$PK/t1blind_run.py" | cut -c1-64)" "$TH" "$(( $(date +%s) - t0 ))" > "$S/out/done_$tag.txt"
    log "ZZT1B-RUN-DONE $tag wall=$(( $(date +%s) - t0 ))s"
  else log "ZZT1B-RUN-FAIL $tag rc=$rc"; fi
  return $rc
}
gate_id() {   # seed
  log "ZZT1B-GID s$1"
  "$PY" -I "$PKW/t1blind_gates.py" id --seed "$1" > "$PK/logs/gate_id_s$1.out" 2>&1; local rc=$?
  log "ZZT1B-GID-RESULT s$1 rc=$rc: $(tail -n 1 "$PK/logs/gate_id_s$1.out")"
  if [ $rc -ne 0 ]; then echo "$(date +%FT%T) G-ID seed $1 rc=$rc (see raw/GATE_ID_s$1.json)" > "$PK/raw/CHAIN_T1BLIND.gidfail"; exit 0; fi
}
# ---- A, B: the identity rollouts and G-ID
if ! stage_done A; then run_one ID_s0 off 0 || fail "ID_s0 rollout failed (logs/run_ID_s0.out)" 3; gate_id 0; stage_mark A; fi
if ! stage_done B; then run_one ID_s1 off 1 || fail "ID_s1 rollout failed (logs/run_ID_s1.out)" 3; gate_id 1; stage_mark B; fi
# ---- C, D: BLIND1
if ! stage_done C; then run_one BLIND1_s0 blind 0 || fail "BLIND1_s0 rollout failed (logs/run_BLIND1_s0.out)" 3; stage_mark C; fi
if ! stage_done D; then run_one BLIND1_s1 blind 1 || fail "BLIND1_s1 rollout failed (logs/run_BLIND1_s1.out)" 3; stage_mark D; fi
# ---- E: all gates
if ! stage_done E; then
  "$PY" -I "$PKW/t1blind_gates.py" all --out-root "$SW/out" > "$PK/logs/gates_all.out" 2>&1; rc=$?
  log "ZZT1B-GATES rc=$rc: $(tail -n 1 "$PK/logs/gates_all.out")"
  if [ $rc -ne 0 ]; then echo "$(date +%FT%T) gates rc=$rc (see raw/GATES.json)" > "$PK/raw/CHAIN_T1BLIND.gatefail"; exit 0; fi
  stage_mark E
fi
# ---- F: reader
if ! stage_done F; then
  "$PY" -I "$PKW/t1blind_read.py" --out-root "$SW/out" --work-dir "$SW/work" --raw "$PKW/raw" --out-dir "$PKW" --workers 2 > "$PK/logs/read_t1blind.out" 2>&1; rc=$?
  log "ZZT1B-READ rc=$rc"; tail -n 4 "$PK/logs/read_t1blind.out" | tee -a "$PK/logs/chain_t1blind.out"
  [ $rc -eq 0 ] || fail "reader rc=$rc (see logs/read_t1blind.out)" 3
  stage_mark F
fi
# ---- G: result writer, if present
if ! stage_done G && [ -f "$PK/t1blind_result.py" ]; then
  "$PY" -I "$PKW/t1blind_result.py" > "$PK/logs/result_t1blind.out" 2>&1; rc=$?
  log "ZZT1B-RESULT rc=$rc"; [ $rc -eq 0 ] || fail "result writer rc=$rc" 3
  stage_mark G
fi
echo "$(date +%FT%T) T1-BLIND chain done (gates PASS, READ written)" > "$PK/raw/CHAIN_T1BLIND.done"; log "ZZT1B-CHAIN-DONE"
exit 0
