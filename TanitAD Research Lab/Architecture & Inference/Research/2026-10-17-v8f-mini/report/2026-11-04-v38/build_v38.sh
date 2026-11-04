#!/bin/bash
# build_v38.sh <tip> : patch -> numbers audit -> checks -> offline render (CPU only, git show reads only); writes out/ and v38_checks.txt
set -e
export CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
SHA=${1:?tip}
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
HERE=E:/Projects/TanitAD-artifacts/dagland/report_v38
cd "$HERE"
$PY -I patch_v38.py $SHA $HERE/out
$PY -I audit_v38.py $SHA $HERE/out $HERE/v38_NUMBERS_AUDIT.md | head -2
$PY -I checks_v38.py $SHA $HERE/out $HERE/v38_checks.txt | tail -1
( node render_check_v38.js out/v8f_report.html; node render_check_v38.js out/V8F_SCIENTIFIC_REPORT.md ) > v38_render_check.txt
{ echo; echo "== numbers audit (audit_v38.py): $(sed -n 5p v38_NUMBERS_AUDIT.md)"; echo "== offline render (the template's own pre-processing + local marked; v37 baseline: see report_v37/v37_render_check.txt):"; sed 's/^/   /' v38_render_check.txt; } >> v38_checks.txt
