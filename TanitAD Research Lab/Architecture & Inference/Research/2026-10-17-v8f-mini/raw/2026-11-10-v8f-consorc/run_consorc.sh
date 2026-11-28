#!/usr/bin/env bash
# PREREAD_V8F_CONSORC runner: waits for the pair read (topk_pair_read.py) to exit, checks the recorded sha256s, runs a 64-window smoke
# (controls only), then the full read on six arms. Each step writes its own log under out/; the full read only starts if the smoke controls pass.
set -u
D=/e/Projects/TanitAD-artifacts/dagland/motion_c/theory/consorc
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
mkdir -p "$D/out"
for i in $(seq 1 1440); do
  n=$(powershell -NoProfile -Command "@(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -like '*topk_pair_read.py*' -and \$_.Name -eq 'python.exe' }).Count" | tr -d '\r')
  [ "$n" = "0" ] && break
  sleep 60
done
[ "$n" = "0" ] || { echo "CONSORC-NOT-STARTED: the pair read still runs after 24 h"; exit 3; }
echo "[$(date '+%F %T')] pair read ended"
cd "$D" || exit 2
"$PY" -I -c "
import hashlib,sys
for line in open('PREREAD_V8F_CONSORC.sha256'):
    h,f,_=line.split()
    got=hashlib.sha256(open(f,'rb').read()).hexdigest()
    if got!=h: sys.exit(f'sha256 of {f} changed: {got} != {h}')
print('sha256 ok')" || exit 4
OMP_NUM_THREADS=2 "$PY" -I cons_orc_read.py --tags PARITY_DAGOACC_s0 --smoke 64 --out "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/consorc/out/SMOKE.json" > "$D/out/smoke.log" 2>&1
rc=$?; echo "smoke rc=$rc"; [ "$rc" -eq 0 ] || { tail -n 5 "$D/out/smoke.log"; exit 5; }
"$PY" -I -c "
import json,sys
c=json.load(open('out/SMOKE.json'))['controls']['PARITY_DAGOACC_s0']
ok=c['ident_vs_top1_max_abs_m']<=1e-5 and c['own_all_vs_ident_max_abs_m']<=1e-6
print('smoke controls', c['ident_vs_top1_max_abs_m'], c['own_all_vs_ident_max_abs_m'], 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)" || exit 6
OMP_NUM_THREADS=2 "$PY" -I cons_orc_read.py --tags PARITY_DAGOACC_s0 PARITY_DAGOACC_s1 INIT_M2b_s0 INIT_M2b_s1 PARITY_DAGOACCREP_s0 PARITY_DAGOACCREP_s1 \
  --out "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/consorc/out/CONSORC.json" > "$D/out/run.log" 2>&1
rc=$?; echo "full rc=$rc at $(date '+%T')"; grep -E '^\{"tag"' "$D/out/run.log" | cut -c1-700
exit $rc
