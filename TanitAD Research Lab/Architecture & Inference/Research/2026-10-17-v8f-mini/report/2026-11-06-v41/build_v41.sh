#!/bin/bash
# build_v41.sh <tip> : patch -> numbers audit -> checks -> offline render (CPU only, git show reads only); writes out/, v41_NUMBERS_AUDIT.md, v41_checks.txt, v41_render_check.txt, MANIFEST.txt
# Exit status: 0 only when the patch built, the audit has 0 unmatched tokens and every check passes (the placeholder check fails while any {{ remains).
export CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
SHA=${1:?tip}
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
HERE=E:/Projects/TanitAD-artifacts/dagland/report_v41
cd "$HERE" || exit 2
$PY -I patch_v41.py $SHA $HERE/out || { echo "PATCH FAILED"; exit 2; }
$PY -I audit_v41.py $SHA $HERE/out $HERE/v41_NUMBERS_AUDIT.md | head -4; ARC=${PIPESTATUS[0]}
$PY -I checks_v41.py $SHA $HERE/out $HERE/v41_checks.txt | tail -1; CRC=${PIPESTATUS[0]}
( node render_check_v41.js out/v8f_report.html; node render_check_v41.js out/V8F_SCIENTIFIC_REPORT.md ) > v41_render_check.txt
{ echo; echo "== numbers audit (audit_v41.py): $(sed -n 5p v41_NUMBERS_AUDIT.md)"; echo "== offline render (the template's own pre-processing + local marked; v40 baseline: see report_v40/v40_render_check.txt):"; sed 's/^/   /' v41_render_check.txt; } >> v41_checks.txt
$PY -I - <<'PYEOF'
import hashlib, os
H = "E:/Projects/TanitAD-artifacts/dagland/report_v41"
names = ["out/V8F_SCIENTIFIC_REPORT.md", "out/v8f_report.html", "out/DIRECTIVES.txt", "out/FAMILY_COUNTS.txt", "patch_v41.py", "audit_v41.py", "checks_v41.py", "build_v41.sh", "render_check_v41.js", "edits_v41.md",
         "sec_741_levelfloor.md", "HANDOFF_V41.md", "msg_v41.txt", "v41_NUMBERS_AUDIT.md", "v41_checks.txt", "v41_render_check.txt"]
lines = ["# MANIFEST report v41 (built by the build agent for the Master Mind to land). name | bytes | sha256", ""]
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
