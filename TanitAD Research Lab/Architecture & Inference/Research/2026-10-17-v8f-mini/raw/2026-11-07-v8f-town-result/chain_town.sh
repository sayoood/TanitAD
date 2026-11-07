#!/usr/bin/env bash
# TOWN chain (PREREG_V8F_TOWN.md), dev-box CPU, 4 threads, zero training. Resumable: a finished stage (raw/CHAIN_STAGE_<X>.done) is skipped on relaunch.
#   A BARE_s0 (the tool of record alone, 4 threads = the identity reference) + C0_s0 (real twin) + G-ID vs BARE_s0 (decides) + vs the banked 12-thread RIC0_s0 (informational) + twin gates    B C0_s1 (same, vs RIC0_s1)    C REG_s0 (--zero-t-ego) + twin/reg gates    D REG_s1    E reads (town_read.py) -> raw/READ_<tag>.json
# Markers: raw/CHAIN_TOWN.done | .fail | .gidfail (identity failed: NOTHING is read, report INVALID) | .gatefail.  Log: logs/chain_town.out (ZZTOWN-* lines).
# Launch (survives the caller): powershell -NoProfile -Command "Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd.exe /c E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\town\chain_town.cmd'}"
set -u
export PATH="/usr/bin:/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOWN_THREADS=4
PK=/e/Projects/TanitAD-artifacts/dagland/motion_c/theory/town; PKW=E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/town
SW=F:/TanitAD-scratch/motion_c/town; S=/f/TanitAD-scratch/motion_c/town; BANK=F:/TanitAD-scratch/motion_c/ri/out
ART=E:/Projects/TanitAD-artifacts; PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
DATA="$ART/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
CARD=/e/Projects/TanitAD-artifacts/dagland/motion_c/theory/PREREG_V8F_TOWN.md
mkdir -p "$S/out/receipts" "$PK/logs" "$PK/raw"
log() { echo "$(date +%FT%T) $*" | tee -a "$PK/logs/chain_town.out"; }
fail() { log "ZZTOWN-FAIL $1"; echo "$(date +%FT%T) $1" > "$PK/raw/CHAIN_TOWN.fail"; exit "${2:-3}"; }
stage_done() { [ -f "$PK/raw/CHAIN_STAGE_$1.done" ]; }
stage_mark() { echo "$(date +%FT%T)" > "$PK/raw/CHAIN_STAGE_$1.done"; log "ZZTOWN-STAGE-$1-DONE"; }
head -c 4676 "$CARD" | sha256sum | grep -q '^707e29f388bac0cd3d0b077b5be07a93be29d82fdd7b06859070ba169bda56b2' || fail "the card's frozen prefix changed" 2
(cd "$PK" && sha256sum -c IMPL_CHOICES.sha256 > /dev/null 2>&1) || fail "IMPL_CHOICES.md changed after the freeze" 2
busy() { powershell -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'v8f_t1_eval_[a-z_0-9]+[.]py\S*\s+run|town_run[.]py' -and \$_.ProcessId -ne \$PID }).Count" | tr -d '\r '; }
run_one() {   # tag seed zero(0|1)
  local tag=$1 s=$2 z=$3 ck="$ART/v8f-recover/M2b_s${2}_recft_C0/ckpt_last.pt" zf=""
  [ "$z" = "1" ] && zf="--zero-t-ego"
  [ -f "$S/out/done_$tag.txt" ] && { log "ZZTOWN-SKIP $tag"; return 0; }
  local w=0; while [ "$(busy)" != "0" ] && [ -n "$(busy)" ]; do sleep 30; w=$((w+30)); [ $w -ge 7200 ] && break; done
  [ -d "$S/out/$tag" ] && mv "$S/out/$tag" "$S/out/_incomplete_${tag}_$(date +%H%M%S)"
  mkdir -p "$S/out/$tag"; local t0=$(date +%s)
  log "ZZTOWN-RUN $tag zero_t_ego=$z ckpt=$ck"
  local runner=("$PY" -I "$PKW/town_run.py" $zf --receipt-dir "$SW/out/receipts" --); case "$tag" in BARE_*) runner=("$PY" -I "F:/TanitAD-scratch/motion_c/ri/tree_ri/stack/scripts/v8f_t1_eval_ri.py" run);; esac
  "${runner[@]}" --ckpt "$ck" --tag "$tag" --tracks "$ART/sfresh_inputs/inputs/native_tracks.pt" --eval-cache "$ART/sfresh_inputs/cache" \
     --eval-grid "$DATA/eval_grid_windows.json" --eval-join "$DATA/join_eval141.npz" --out-dir "$SW/out/$tag" --replan-steps 10 --nav deadreckon --schedule periodic --limit causal \
     --dump-ticks "$SW/out/$tag/ticks" --dump-heads --device cpu --smooth sg72 --scene reindexed < /dev/null > "$PK/logs/run_$tag.out" 2>&1; local rc=$?
  if [ $rc -eq 0 ] && { [ -f "$S/out/receipts/receipt_$tag.json" ] || case "$tag" in BARE_*) true;; *) false;; esac; }; then
    printf '{"tag": "%s", "seed": %s, "zero_t_ego": %s, "ckpt": "%s", "ckpt_sha256": "%s", "tool_sha256": "0417c838355a51f6f7084c17b5acada96c5b6cbd9eef54cc2602d377bf3c61e7", "wrapper_sha256": "%s", "threads": 4, "wall_s": %s}\n' \
       "$tag" "$s" "$z" "$ck" "$(sha256sum "$ck" | cut -c1-64)" "$(sha256sum "$PK/town_run.py" | cut -c1-64)" "$(( $(date +%s) - t0 ))" > "$S/out/done_$tag.txt"
    log "ZZTOWN-RUN-DONE $tag wall=$(( $(date +%s) - t0 ))s"
  else log "ZZTOWN-RUN-FAIL $tag rc=$rc"; fi
  return $rc
}
twin_gate() { "$PY" -I "$PKW/town_gates.py" twin --run "$SW/out/$1" --tag "$1" --receipt "$SW/out/receipts/receipt_$1.json" > "$PK/logs/gate_twin_$1.out" 2>&1; local rc=$?
  cp "$PK/logs/gate_twin_$1.out" "$PK/raw/GATE_TWIN_$1.json"; log "ZZTOWN-GATE-TWIN $1 rc=$rc"; [ $rc -eq 0 ] || { echo "$(date +%FT%T) twin gate $1 rc=$rc" > "$PK/raw/CHAIN_TOWN.gatefail"; exit 0; }; }
id_gate_bank() { "$PY" -I "$PKW/town_gates.py" id --run "$SW/out/$1" --tag "$1" --bank "$BANK/RIC0_s$2" --bank-tag "RIC0_s$2" > "$PK/raw/GATE_ID_INFO_${1}_vs_banked_RIC0_s$2.json" 2>&1; log "ZZTOWN-ID-INFO $1 vs banked RIC0_s$2 (12 threads; informational) rc=$?"; }
id_gate() { "$PY" -I "$PKW/town_gates.py" id --run "$SW/out/$1" --tag "$1" --bank "$SW/out/BARE_s$2" --bank-tag "BARE_s$2" > "$PK/logs/gate_id_$1.out" 2>&1; local rc=$?
  cp "$PK/logs/gate_id_$1.out" "$PK/raw/GATE_ID_$1.json"; log "ZZTOWN-GATE-ID $1 rc=$rc"; [ $rc -eq 0 ] || { echo "$(date +%FT%T) G-ID $1 rc=$rc" > "$PK/raw/CHAIN_TOWN.gidfail"; exit 0; }; }
if ! stage_done A; then run_one BARE_s0 0 0 || fail "BARE_s0 failed" 3; run_one C0_s0 0 0 || fail "C0_s0 rollout failed" 3; id_gate C0_s0 0; id_gate_bank C0_s0 0; twin_gate C0_s0; stage_mark A; fi
if ! stage_done B; then run_one BARE_s1 1 0 || fail "BARE_s1 failed" 3; run_one C0_s1 1 0 || fail "C0_s1 rollout failed" 3; id_gate C0_s1 1; id_gate_bank C0_s1 1; twin_gate C0_s1; stage_mark B; fi
if ! stage_done C; then run_one REG_s0 0 1 || fail "REG_s0 rollout failed" 3; twin_gate REG_s0; stage_mark C; fi
if ! stage_done D; then run_one REG_s1 1 1 || fail "REG_s1 rollout failed" 3; twin_gate REG_s1; stage_mark D; fi
if ! stage_done E; then
  for t in C0_s0 C0_s1 REG_s0 REG_s1; do "$PY" -I "$PKW/town_read.py" --twin "$SW/out/$t/ticks/twin_$t.npz" --out "$PK/raw/READ_$t.json" >> "$PK/logs/read_town.out" 2>&1 || fail "read $t failed (logs/read_town.out)" 3; done
  stage_mark E
fi
echo "$(date +%FT%T) TOWN chain done (gates PASS, reads written)" > "$PK/raw/CHAIN_TOWN.done"; log "ZZTOWN-CHAIN-DONE"
exit 0
