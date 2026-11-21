#!/bin/bash
# build_v45.sh <tip> : patch -> numbers audit -> checks (incl. check_numbers_v45.py and its mutation arm) -> offline render (CPU only, git show reads only)
# writes out/, v45_NUMBERS_AUDIT.md, v45_checks.txt, v45_render_check.txt, NUMBERS_V45.md, MANIFEST.txt. <tip> is the full 40-hex sha of the repository tip.
# Exit status: 0 only when the patch built, the audit has 0 unmatched tokens and 0 unresolved scope misses, and every check passes.
export CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
SHA=${1:?tip}
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
HERE=E:/Projects/TanitAD-artifacts/dagland/report_v45
cd "$HERE" || exit 2
$PY -I patch_v45.py $SHA $HERE/out || { echo "PATCH FAILED"; exit 2; }
$PY -I audit_v45.py $SHA $HERE/out $HERE/v45_NUMBERS_AUDIT.md | head -4; ARC=${PIPESTATUS[0]}
$PY -I checks_v45.py $SHA $HERE/out $HERE/v45_checks.txt | tail -1; CRC=${PIPESTATUS[0]}
( node render_check_v45.js out/v8f_report.html; node render_check_v45.js out/V8F_SCIENTIFIC_REPORT.md ) > v45_render_check.txt
{ echo; echo "== numbers audit (audit_v45.py): $(sed -n 5p v45_NUMBERS_AUDIT.md | cut -c1-200)"; echo "== offline render (the template's own pre-processing + local marked; v44 baseline: report_v44/v44_render_check.txt, 296 tables, 0 ragged):"; sed 's/^/   /' v45_render_check.txt; } >> v45_checks.txt
$PY -I - <<'PYEOF'
import hashlib, os
H = "E:/Projects/TanitAD-artifacts/dagland/report_v45"
names = ["out/V8F_SCIENTIFIC_REPORT.md", "out/v8f_report.html", "out/DIRECTIVES.txt", "patch_v45.py", "audit_v45.py", "checks_v45.py", "check_numbers_v45.py", "build_v45.sh", "render_check_v45.js", "edits_v45.md",
         "sec_749_imagination.md", "sec_750_horizon_bands.md", "NUMBERS_V45.md", "HANDOFF_V45_BUILD.md", "v45_NUMBERS_AUDIT.md", "v45_checks.txt", "v45_render_check.txt"]
lines = ["# MANIFEST report v45 (built by the build agent for the Master Mind to land). name | bytes | sha256", ""]
for n in names:
    p = os.path.join(H, n)
    if os.path.exists(p):
        b = open(p, "rb").read()
        lines.append("%s | %d | %s" % (n, len(b), hashlib.sha256(b).hexdigest()))
    else:
        lines.append("%s | MISSING" % n)
open(os.path.join(H, "MANIFEST.txt"), "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
PYEOF
echo "audit rc $ARC, checks rc $CRC"
[ "$ARC" = "0" ] && [ "$CRC" = "0" ]
