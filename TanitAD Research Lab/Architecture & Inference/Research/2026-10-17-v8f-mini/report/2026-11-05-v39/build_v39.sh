#!/bin/bash
# build_v39.sh <tip> : patch -> numbers audit -> checks -> offline render (CPU only, git show reads only); writes out/, v39_NUMBERS_AUDIT.md, v39_checks.txt, v39_render_check.txt, MANIFEST.txt
# Exit status: 0 only when the patch built, the audit has 0 unmatched tokens and every check passes (the placeholder check fails while a {{LLR_...}} placeholder remains).
export CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
SHA=${1:?tip}
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
HERE=E:/Projects/TanitAD-artifacts/dagland/report_v39
cd "$HERE" || exit 2
$PY -I patch_v39.py $SHA $HERE/out || { echo "PATCH FAILED"; exit 2; }
$PY -I audit_v39.py $SHA $HERE/out $HERE/v39_NUMBERS_AUDIT.md | head -4; ARC=${PIPESTATUS[0]}
$PY -I checks_v39.py $SHA $HERE/out $HERE/v39_checks.txt | tail -1; CRC=${PIPESTATUS[0]}
( node render_check_v39.js out/v8f_report.html; node render_check_v39.js out/V8F_SCIENTIFIC_REPORT.md ) > v39_render_check.txt
{ echo; echo "== numbers audit (audit_v39.py): $(sed -n 5p v39_NUMBERS_AUDIT.md)"; echo "== offline render (the template's own pre-processing + local marked; v38 baseline: see report_v38/v38_render_check.txt):"; sed 's/^/   /' v39_render_check.txt; } >> v39_checks.txt
$PY -I - <<'PYEOF'
import hashlib, os
H = "E:/Projects/TanitAD-artifacts/dagland/report_v39"
names = ["out/V8F_SCIENTIFIC_REPORT.md", "out/v8f_report.html", "out/DIRECTIVES.txt", "patch_v39.py", "audit_v39.py", "checks_v39.py", "build_v39.sh", "render_check_v39.js", "edits_v39.md",
         "sec_737_rpt.md", "sec_738_motion_c.md", "sec_739_levelhead.md", "sec_739_llr_result.md", "sec_739_tail.md", "PLACEHOLDERS.md", "HANDOFF_V39.md", "msg_v39.txt", "v39_NUMBERS_AUDIT.md", "v39_checks.txt", "v39_render_check.txt"]
lines = ["# MANIFEST report v39 (staged only; the Master Mind lands). name | bytes | sha256", ""]
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
