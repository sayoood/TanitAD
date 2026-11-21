#!/bin/bash
# build_v37.sh <tip> : patch -> numbers audit -> checks -> offline render (CPU only, git show reads only); writes out/ and v37_checks.txt
set -e
export CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
SHA=${1:?tip}
HERE=E:/Projects/TanitAD-artifacts/dagland/report_v37
cd "$HERE"
python -I patch_v37.py $SHA $HERE/out
python -I audit_v37.py $SHA $HERE/out $HERE/v37_NUMBERS_AUDIT.md | head -2
python -I checks_v37.py $SHA $HERE/out $HERE/v37_checks.txt | tail -1
( node render_check_v37.js out/v8f_report.html; node render_check_v37.js out/V8F_SCIENTIFIC_REPORT.md ) > v37_render_check.txt
{ echo; echo "== numbers audit (audit_v37.py): $(sed -n 5p v37_NUMBERS_AUDIT.md)"; echo "== offline render (the template's own pre-processing + local marked; v36 baseline: see report_v36/v36_render_check.txt):"; sed 's/^/   /' v37_render_check.txt; } >> v37_checks.txt
