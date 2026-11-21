#!/bin/bash
# build_v40.sh <tip> : patch -> numbers audit -> checks -> offline render (CPU only, git show reads only); writes out/, v40_NUMBERS_AUDIT.md, v40_checks.txt, v40_render_check.txt, MANIFEST.txt
# Exit status: 0 only when the patch built, the audit has 0 unmatched tokens and every check passes (the placeholder check fails while any {{ remains).
export CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
SHA=${1:?tip}
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
HERE=E:/Projects/TanitAD-artifacts/dagland/report_v40
cd "$HERE" || exit 2
$PY -I patch_v40.py $SHA $HERE/out || { echo "PATCH FAILED"; exit 2; }
$PY -I audit_v40.py $SHA $HERE/out $HERE/v40_NUMBERS_AUDIT.md | head -4; ARC=${PIPESTATUS[0]}
$PY -I checks_v40.py $SHA $HERE/out $HERE/v40_checks.txt | tail -1; CRC=${PIPESTATUS[0]}
( node render_check_v40.js out/v8f_report.html; node render_check_v40.js out/V8F_SCIENTIFIC_REPORT.md ) > v40_render_check.txt
{ echo; echo "== numbers audit (audit_v40.py): $(sed -n 5p v40_NUMBERS_AUDIT.md)"; echo "== offline render (the template's own pre-processing + local marked; v39 baseline: see report_v39/v39_render_check.txt):"; sed 's/^/   /' v40_render_check.txt; } >> v40_checks.txt
$PY -I - <<'PYEOF'
import hashlib, os
H = "E:/Projects/TanitAD-artifacts/dagland/report_v40"
names = ["out/V8F_SCIENTIFIC_REPORT.md", "out/v8f_report.html", "out/DIRECTIVES.txt", "patch_v40.py", "audit_v40.py", "checks_v40.py", "build_v40.sh", "render_check_v40.js", "edits_v40.md",
         "sec_740_levelfloor.md", "HANDOFF_V40.md", "msg_v40.txt", "v40_NUMBERS_AUDIT.md", "v40_checks.txt", "v40_render_check.txt"]
lines = ["# MANIFEST report v40 (built and landed by the Master Mind). name | bytes | sha256", ""]
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
