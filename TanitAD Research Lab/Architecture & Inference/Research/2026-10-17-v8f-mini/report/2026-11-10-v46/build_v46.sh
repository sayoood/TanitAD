#!/bin/bash
# build_v46.sh <tip> : patch -> numbers audit -> checks (incl. check_numbers_v46.py and its mutation arm) -> offline render (CPU only, git reads only)
# writes out/, v46_NUMBERS_AUDIT.md, v46_checks.txt, v46_render_check.txt, NUMBERS_V46.md, MANIFEST.txt. <tip> is the full 40-hex sha of the repository tip.
# Exit status: 0 only when the patch built, the audit has 0 unmatched tokens and every check passes. (No PDF: dropped by the Master Mind.)
export CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
SHA=${1:?tip}
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
HERE=E:/Projects/TanitAD-artifacts/dagland/report_v46
cd "$HERE" || exit 2
$PY -I patch_v46.py $SHA $HERE/out || { echo "PATCH FAILED"; exit 2; }
$PY -I audit_v46.py $SHA $HERE/out $HERE/v46_NUMBERS_AUDIT.md | head -4; ARC=${PIPESTATUS[0]}
$PY -I checks_v46.py $SHA $HERE/out $HERE/v46_checks.txt | tail -1; CRC=${PIPESTATUS[0]}
( node render_check_v46.js out/v8f_report.html; node render_check_v46.js out/V8F_SCIENTIFIC_REPORT.md ) > v46_render_check.txt
{ echo; echo "== numbers audit (audit_v46.py): $(sed -n 5p v46_NUMBERS_AUDIT.md | cut -c1-200)"; echo "== offline render (the template pre-processing + local marked; v45 baseline: report_v45/v45_render_check.txt):"; sed 's/^/   /' v46_render_check.txt; } >> v46_checks.txt
$PY -I manifest_v46.py
echo "audit rc $ARC, checks rc $CRC"
[ "$ARC" = "0" ] && [ "$CRC" = "0" ]
