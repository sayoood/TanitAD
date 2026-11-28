#!/usr/bin/env bash
# TSWAP chain (PREREG_V8F_TSWAP.md), dev-box CPU, 4 threads, zero training. Resumable: a finished rollout (done_<tag>.txt) and a finished stage (raw/CHAIN_STAGE_<X>.done) are skipped on relaunch.
#   A rollouts per seed: OWN_s (arm own) | TSWAP_s | MUT_s (arm mutant)       B gates per seed (G-ID vs TOWN's BARE_s, G-TICK0, G-MUT, G-LIVE) -> raw/GATES_s{0,1}.json      C tswap_result.py -> RESULT_TSWAP.md/.json (INVALID if a gate failed)
# End marker: theory/tswap/CHAIN_TSWAP.done containing `rc=<n>` (0 = every stage ran; a failed gate is data and gives INVALID in the result, rc stays 0). Log: logs/chain_tswap.out (ZZTSWAP-* lines).
# Launch: Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd /c "E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\tswap\chain_tswap.cmd" < NUL'}
set -u
export PATH="/usr/bin:/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TSWAP_THREADS=4
PK=/e/Projects/TanitAD-artifacts/dagland/motion_c/theory/tswap; PKW=E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/tswap
SW=F:/TanitAD-scratch/motion_c/tswap; S=/f/TanitAD-scratch/motion_c/tswap; BARE=F:/TanitAD-scratch/motion_c/town/out; BARES=/f/TanitAD-scratch/motion_c/town/out
ART=E:/Projects/TanitAD-artifacts; PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
DATA="$ART/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
CARD=/e/Projects/TanitAD-artifacts/dagland/motion_c/theory/PREREG_V8F_TSWAP.md
TOOL_SHA=0417c838355a51f6f7084c17b5acada96c5b6cbd9eef54cc2602d377bf3c61e7
mkdir -p "$S/out/receipts" "$S/work" "$PK/logs" "$PK/raw"
log() { echo "$(date +%FT%T) $*" | tee -a "$PK/logs/chain_tswap.out"; }
finish() { echo "rc=$1 $(date +%FT%T) $2" > "$PK/CHAIN_TSWAP.done"; log "ZZTSWAP-CHAIN-END rc=$1 $2"; exit "$1"; }
stage_done() { [ -f "$PK/raw/CHAIN_STAGE_$1.done" ]; }
stage_mark() { echo "$(date +%FT%T)" > "$PK/raw/CHAIN_STAGE_$1.done"; log "ZZTSWAP-STAGE-$1-DONE"; }
rm -f "$PK/CHAIN_TSWAP.done"
head -c 3642 "$CARD" | sha256sum | grep -q '^fc775ca1d3ebaadc792d546e639d7dcc01506cf4f4bb939fe5526f808fd53cf0' || finish 2 "the card's frozen prefix changed"
(cd "$PK" && sha256sum -c IMPL_CHOICES_TSWAP.sha256 > /dev/null 2>&1) || finish 2 "IMPL_CHOICES_TSWAP.md changed after the freeze"
for s in 0 1; do   # the bare-tool references: TOWN's BARE_s* (pinned tool, 4 threads, the same flags)
  d="$BARES/done_BARE_s$s.txt"; [ -f "$d" ] && grep -q "$TOOL_SHA" "$d" && grep -q '"threads": 4' "$d" && grep -q "\"seed\": $s" "$d" || finish 2 "bare reference done_BARE_s$s.txt missing or not the pinned tool at 4 threads"
done
busy() { powershell -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'python[.]exe\S*\s+-I\s+\S*(tswap_run|town_run|v8f_t1_eval_[a-z_0-9]+)[.]py' -and \$_.ProcessId -ne \$PID }).Count" | tr -d '\r '; }
run_one() {   # tag arm seed
  local tag=$1 arm=$2 s=$3 ck="$ART/v8f-recover/M2b_s${3}_recft_C0/ckpt_last.pt"
  [ -f "$S/out/done_$tag.txt" ] && { log "ZZTSWAP-SKIP $tag"; return 0; }
  local w=0; while [ "$(busy)" != "0" ] && [ -n "$(busy)" ]; do sleep 30; w=$((w+30)); [ $w -ge 7200 ] && break; done
  [ -d "$S/out/$tag" ] && mv "$S/out/$tag" "$S/out/_incomplete_${tag}_$(date +%H%M%S)"
  mkdir -p "$S/out/$tag"; local t0=$(date +%s)
  log "ZZTSWAP-RUN $tag arm=$arm ckpt=$ck"
  "$PY" -I "$PKW/tswap_run.py" --arm "$arm" --receipt-dir "$SW/out/receipts" -- --ckpt "$ck" --tag "$tag" --tracks "$ART/sfresh_inputs/inputs/native_tracks.pt" --eval-cache "$ART/sfresh_inputs/cache" \
     --eval-grid "$DATA/eval_grid_windows.json" --eval-join "$DATA/join_eval141.npz" --out-dir "$SW/out/$tag" --replan-steps 10 --nav deadreckon --schedule periodic --limit causal \
     --dump-ticks "$SW/out/$tag/ticks" --dump-heads --device cpu --smooth sg72 --scene reindexed < /dev/null > "$PK/logs/run_$tag.out" 2>&1; local rc=$?
  if [ $rc -eq 0 ] && [ -f "$S/out/receipts/receipt_$tag.json" ]; then
    printf '{"tag": "%s", "arm": "%s", "seed": %s, "ckpt": "%s", "ckpt_sha256": "%s", "tool_sha256": "%s", "wrapper_sha256": "%s", "threads": 4, "wall_s": %s}\n' \
       "$tag" "$arm" "$s" "$ck" "$(sha256sum "$ck" | cut -c1-64)" "$TOOL_SHA" "$(sha256sum "$PK/tswap_run.py" | cut -c1-64)" "$(( $(date +%s) - t0 ))" > "$S/out/done_$tag.txt"
    log "ZZTSWAP-RUN-DONE $tag wall=$(( $(date +%s) - t0 ))s"
  else log "ZZTSWAP-RUN-FAIL $tag rc=$rc"; fi
  return $rc
}
if ! stage_done A; then
  for s in 0 1; do run_one OWN_s$s own $s || finish 3 "OWN_s$s failed"; run_one TSWAP_s$s tswap $s || finish 3 "TSWAP_s$s failed"; run_one MUT_s$s mutant $s || finish 3 "MUT_s$s failed"; done
  stage_mark A
fi
if ! stage_done B; then
  for s in 0 1; do
    "$PY" -I "$PKW/tswap_gates.py" --own "$SW/out/OWN_s$s" --own-tag OWN_s$s --bare "$BARE/BARE_s$s" --bare-tag BARE_s$s --tswap "$SW/out/TSWAP_s$s" --tswap-tag TSWAP_s$s --mutant "$SW/out/MUT_s$s" --mutant-tag MUT_s$s \
      --rec-tswap "$SW/out/receipts/receipt_TSWAP_s$s.json" --rec-mutant "$SW/out/receipts/receipt_MUT_s$s.json" --out "$PKW/raw/GATES_s$s.json" > "$PK/logs/gates_s$s.out" 2>&1; rc=$?
    log "ZZTSWAP-GATES s$s rc=$rc"
  done
  stage_mark B
fi
if ! stage_done C; then
  "$PY" -I "$PKW/tswap_result.py" --out-root "$SW/out" --work-dir "$SW/work" --raw "$PKW/raw" --out-dir "$PKW" --gates-dir "$PKW/raw" > "$PK/logs/result_tswap.out" 2>&1; rc=$?
  log "ZZTSWAP-RESULT rc=$rc: $(tail -n 1 "$PK/logs/result_tswap.out")"
  [ $rc -eq 0 ] || finish 3 "tswap_result.py failed (logs/result_tswap.out); rerun: bash chain_tswap.sh (rollouts and gates are skipped)"
  stage_mark C
fi
finish 0 "stages A B C done (result: RESULT_TSWAP.md)"
