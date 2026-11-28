#!/bin/bash
# build_v42.sh <tip> : patch -> numbers audit -> checks -> offline render (CPU only, git show reads only); writes out/, v42_NUMBERS_AUDIT.md, v42_checks.txt, v42_render_check.txt, MANIFEST.txt
# Exit status: 0 only when the patch built, the audit has 0 unmatched tokens and every check passes (the placeholder check fails while any {{ remains).
export CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
SHA=${1:?tip}
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
HERE=E:/Projects/TanitAD-artifacts/dagland/report_v42
cd "$HERE" || exit 2
$PY -I patch_v42.py $SHA $HERE/out || { echo "PATCH FAILED"; exit 2; }
$PY -I audit_v42.py $SHA $HERE/out $HERE/v42_NUMBERS_AUDIT.md | head -4; ARC=${PIPESTATUS[0]}
$PY -I checks_v42.py $SHA $HERE/out $HERE/v42_checks.txt | tail -1; CRC=${PIPESTATUS[0]}
( node render_check_v42.js out/v8f_report.html; node render_check_v42.js out/V8F_SCIENTIFIC_REPORT.md ) > v42_render_check.txt
{ echo; echo "== numbers audit (audit_v42.py): $(sed -n 5p v42_NUMBERS_AUDIT.md)"; echo "== offline render (the template's own pre-processing + local marked; v41 baseline: see report_v41/v41_render_check.txt):"; sed 's/^/   /' v42_render_check.txt; } >> v42_checks.txt
$PY -I - <<'PYEOF'
import hashlib, os
H = "E:/Projects/TanitAD-artifacts/dagland/report_v42"
names = ["out/V8F_SCIENTIFIC_REPORT.md", "out/v8f_report.html", "out/DIRECTIVES.txt", "patch_v42.py", "audit_v42.py", "checks_v42.py", "build_v42.sh", "render_check_v42.js", "edits_v42.md",
         "sec_742_t2.md", "sec_743_lorapod.md", "sec_744_theory.md", "HANDOFF_V42.md", "msg_v42.txt", "v42_NUMBERS_AUDIT.md", "v42_checks.txt", "v42_render_check.txt"]
lines = ["# MANIFEST report v42 (built by the build agent for the Master Mind to land). name | bytes | sha256", ""]
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
