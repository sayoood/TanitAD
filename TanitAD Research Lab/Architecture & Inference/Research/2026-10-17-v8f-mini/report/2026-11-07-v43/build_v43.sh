#!/bin/bash
# build_v43.sh <tip> : patch -> numbers audit -> checks -> offline render (CPU only, git show reads only); writes out/, v43_NUMBERS_AUDIT.md, v43_checks.txt, v43_render_check.txt, MANIFEST.txt
# Exit status: 0 only when the patch built, the audit has 0 unmatched tokens and every check passes (the placeholder check fails while any {{ remains).
export CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
SHA=${1:?tip}
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
HERE=E:/Projects/TanitAD-artifacts/dagland/report_v43
cd "$HERE" || exit 2
$PY -I patch_v43.py $SHA $HERE/out || { echo "PATCH FAILED"; exit 2; }
$PY -I audit_v43.py $SHA $HERE/out $HERE/v43_NUMBERS_AUDIT.md | head -4; ARC=${PIPESTATUS[0]}
$PY -I checks_v43.py $SHA $HERE/out $HERE/v43_checks.txt | tail -1; CRC=${PIPESTATUS[0]}
( node render_check_v43.js out/v8f_report.html; node render_check_v43.js out/V8F_SCIENTIFIC_REPORT.md ) > v43_render_check.txt
{ echo; echo "== numbers audit (audit_v43.py): $(sed -n 5p v43_NUMBERS_AUDIT.md)"; echo "== offline render (the template's own pre-processing + local marked; v42 baseline: see report_v42/v42_render_check.txt):"; sed 's/^/   /' v43_render_check.txt; } >> v43_checks.txt
$PY -I - <<'PYEOF'
import hashlib, os
H = "E:/Projects/TanitAD-artifacts/dagland/report_v43"
names = ["out/V8F_SCIENTIFIC_REPORT.md", "out/v8f_report.html", "out/DIRECTIVES.txt", "patch_v43.py", "audit_v43.py", "checks_v43.py", "build_v43.sh", "render_check_v43.js", "edits_v43.md",
         "sec_745_t2attrib.md", "sec_746_theory.md", "HANDOFF_V43.md", "BUILD_BRIEF_V43.md", "v43_NUMBERS_AUDIT.md", "v43_checks.txt", "v43_render_check.txt"]
lines = ["# MANIFEST report v43 (built by the build agent for the Master Mind to land). name | bytes | sha256", ""]
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
