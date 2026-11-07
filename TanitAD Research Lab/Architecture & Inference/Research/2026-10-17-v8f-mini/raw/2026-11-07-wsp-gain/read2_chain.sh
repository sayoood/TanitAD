#!/usr/bin/env bash
# READ 2 chain (PREREG_WSP_GAIN.md section 3): 4 T1-kin rollouts of the EDITED C0 on eval-141, scene `reindexed` (the deciding scene of the LF / A-series), then the reader. Dev-box CPU, zero training, ONE rollout at a time.
# Launch (survives the caller):  powershell -NoProfile -Command "Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd.exe /c <this dir>\read2_chain.cmd'}"
# Gates: card section 6 has a PIN entry; raw/READ1.json says MECH-OK; edited ckpt shas == raw/FIT_RESULT.json; tool sha == the RIC0 tool's (0417c838...); a finished rollout is skipped on re-launch.
set -u
export PATH="/usr/bin:/bin:$PATH"
PK=/e/Projects/TanitAD-artifacts/dagland/motion_c/theory/wsp_edit
PKW=E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/wsp_edit
SW=F:/TanitAD-scratch/motion_c/wsp_edit; S=/f/TanitAD-scratch/motion_c/wsp_edit
TOOLW=F:/TanitAD-scratch/motion_c/ri/tree_ri/stack/scripts/v8f_t1_eval_ri.py
TOOL=/f/TanitAD-scratch/motion_c/ri/tree_ri/stack/scripts/v8f_t1_eval_ri.py
TOOL_SHA=0417c838355a51f6f7084c17b5acada96c5b6cbd9eef54cc2602d377bf3c61e7
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
ART=E:/Projects/TanitAD-artifacts
DATA="$ART/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
EDIT=$ART/v8f-wspedit
TH=${WSP_TH:-12}
mkdir -p "$S/out" "$PK/logs"
log() { echo "$(date +%FT%T) $*" | tee -a "$PK/logs/read2_chain.out"; }
fail() { log "ZZWSP-FAIL $1"; echo "$(date +%FT%T) $1" > "$PK/raw/READ2_RO.fail"; exit "${2:-3}"; }
grep -q "PIN " <(sed -n '/## 6. Log/,$p' "$ART/dagland/motion_c/theory/PREREG_WSP_GAIN.md") || fail "no PIN entry in the card's section 6" 2
grep -q '"verdict": "MECH-OK"' "$PK/raw/READ1.json" || fail "READ1.json is absent or not MECH-OK" 2
[ "$(sha256sum "$TOOL" | cut -c1-64)" = "$TOOL_SHA" ] || fail "tool sha != the RIC0 tool's" 2
"$PY" -I -c "
import json,hashlib,sys
f=json.load(open(sys.argv[1]))
for s in ('0','1'):
    for t in ('05','10'):
        a=f['seeds'][s]['arms'][t]; h=hashlib.sha256(open(a['ckpt'],'rb').read()).hexdigest()
        assert h==a['ckpt_sha256'], (s,t)
" "$PKW/raw/FIT_RESULT.json" || fail "an edited checkpoint differs from FIT_RESULT.json" 2
busy() { powershell -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'v8f_t1_eval_[a-z_0-9]+[.]py\S*\s+run|extract_smooth[.]py' -and \$_.ProcessId -ne \$PID }).Count" | tr -d '\r '; }
w=0; while [ "$(busy)" != "0" ] && [ -n "$(busy)" ]; do sleep 30; w=$((w+30)); [ $w -ge 7200 ] && break; done
run_one() {   # tag seed tagdose
  local tag=$1 s=$2 d=$3 ck="$EDIT/M2b_s${2}_wspedit_a${3}/ckpt_last.pt"
  [ -f "$S/out/done_$tag.txt" ] && { log "ZZWSP-SKIP $tag"; return 0; }
  [ -d "$S/out/$tag" ] && mv "$S/out/$tag" "$S/out/_incomplete_${tag}_$(date +%H%M%S)"
  mkdir -p "$S/out/$tag"
  local t0=$(date +%s)
  log "ZZWSP-RUN $tag ckpt=$ck"
  OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
  "$PY" -I "$TOOLW" run --ckpt "$ART/v8f-wspedit/M2b_s${s}_wspedit_a${d}/ckpt_last.pt" --tag "$tag" --tracks "$ART/sfresh_inputs/inputs/native_tracks.pt" --eval-cache "$ART/sfresh_inputs/cache" \
     --eval-grid "$DATA/eval_grid_windows.json" --eval-join "$DATA/join_eval141.npz" --out-dir "$SW/out/$tag" --replan-steps 10 --nav deadreckon --schedule periodic --limit causal \
     --dump-ticks "$SW/out/$tag/ticks" --dump-heads --device cpu --smooth sg72 --scene reindexed < /dev/null > "$PK/logs/run_$tag.out" 2>&1; local rc=$?
  if [ $rc -eq 0 ]; then
    printf '{"tag": "%s", "flags": "--smooth sg72 --scene reindexed", "ckpt": "%s", "ckpt_sha256": "%s", "tool": "%s", "tool_sha256": "%s", "threads": %s, "wall_s": %s}\n' "$tag" "$ck" "$(sha256sum "$ck" | cut -c1-64)" "$TOOLW" "$TOOL_SHA" "$TH" "$(( $(date +%s) - t0 ))" > "$S/out/done_$tag.txt"
    log "ZZWSP-RUN-DONE $tag wall=$(( $(date +%s) - t0 ))s"
  else log "ZZWSP-RUN-FAIL $tag rc=$rc"; fi
  return $rc
}
nfail=0
for r in "RIWSP10_s0 0 10" "RIWSP10_s1 1 10" "RIWSP05_s0 0 05" "RIWSP05_s1 1 05"; do set -- $r; run_one "$1" "$2" "$3" || nfail=$((nfail+1)); done
[ $nfail -eq 0 ] || fail "$nfail rollouts failed (see logs/run_*.out)" 3
echo "$(date +%FT%T) four WSP-edit rollouts done" > "$PK/raw/READ2_RO.done"; log "ZZWSP-ROLLOUTS-DONE"
"$PY" -I "$PKW/wspedit_read.py" --out-root "$SW/out" --work-dir "$SW/work" --raw "$PKW/raw" --workers 4 > "$PK/logs/read2.out" 2>&1; rc=$?
log "ZZWSP-READ rc=$rc"; tail -n 4 "$PK/logs/read2.out" | tee -a "$PK/logs/read2_chain.out"
[ $rc -eq 0 ] && echo "$(date +%FT%T) READ2.json written" > "$PK/raw/READ2.done" || echo "$(date +%FT%T) reader rc=$rc" > "$PK/raw/READ2.fail"
