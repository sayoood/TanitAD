#!/usr/bin/env bash
# AMENDMENT 1 runner (AMENDMENT_1_MEMORY.md): pair read PARITY_DAGOACC_s1 + OB-CTL_s0, then CONSORC (sha check, 64-window smoke + controls, full read),
# each frozen script run unchanged through mem_amend.py (memory guards 4 GB per arm, 3 GB per chunk). Sequential: the two reads never overlap.
set -u
T=/e/Projects/TanitAD-artifacts/dagland/motion_c/theory
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
W=E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/consorc/mem_amend.py
shacheck() {   # shacheck <sha256 file>, run in its folder: every listed file must still hash to its line
  "$PY" -I -c "
import hashlib,sys
for line in open(sys.argv[1]):
    t=line.split()
    if not t: continue
    h,f=t[0],t[1].lstrip('*')
    got=hashlib.sha256(open(f,'rb').read()).hexdigest()
    if got!=h: sys.exit(f'sha256 of {f} changed: {got} != {h}')
print('sha256 ok', sys.argv[1])" "$1"
}
cd "$T/topk" || exit 2
shacheck PREREAD_V8F_TOPK_PAIR.sha256 || exit 4
echo "[$(date '+%F %T')] pair read start"
OMP_NUM_THREADS=2 "$PY" -I "$W" topk_pair_read.py --tags PARITY_DAGOACC_s1 OB-CTL_s0 --out E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/topk/out/TOPK_PAIR_s1ctl.json > "$T/topk/out/run_s1ctl.log" 2>&1
rc=$?; echo "[$(date '+%F %T')] pair read rc=$rc"; grep -E '^\{|Error|assert' "$T/topk/out/run_s1ctl.log" | cut -c1-700
cd "$T/consorc" || exit 2
shacheck PREREAD_V8F_CONSORC.sha256 || exit 4
mkdir -p out
OMP_NUM_THREADS=2 "$PY" -I "$W" cons_orc_read.py --tags PARITY_DAGOACC_s0 --smoke 64 --out E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/consorc/out/SMOKE.json > "$T/consorc/out/smoke.log" 2>&1
rc=$?; echo "[$(date '+%F %T')] smoke rc=$rc"; [ "$rc" -eq 0 ] || { tail -n 8 "$T/consorc/out/smoke.log"; exit 5; }
"$PY" -I -c "
import json,sys
c=json.load(open('out/SMOKE.json'))['controls']['PARITY_DAGOACC_s0']
ok=c['ident_vs_top1_max_abs_m']<=1e-5 and c['own_all_vs_ident_max_abs_m']<=1e-6
print('smoke controls', c['ident_vs_top1_max_abs_m'], c['own_all_vs_ident_max_abs_m'], 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)" || exit 6
OMP_NUM_THREADS=2 "$PY" -I "$W" cons_orc_read.py --tags PARITY_DAGOACC_s0 PARITY_DAGOACC_s1 INIT_M2b_s0 INIT_M2b_s1 PARITY_DAGOACCREP_s0 PARITY_DAGOACCREP_s1 \
  --out E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/consorc/out/CONSORC.json > "$T/consorc/out/run.log" 2>&1
rc=$?; echo "[$(date '+%F %T')] consorc full rc=$rc"; grep -E '^\{"tag"' "$T/consorc/out/run.log" | cut -c1-700
exit $rc
