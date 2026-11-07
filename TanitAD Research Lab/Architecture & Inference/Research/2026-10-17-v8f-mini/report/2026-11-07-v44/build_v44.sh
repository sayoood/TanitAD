#!/bin/bash
# build_v44.sh <tip> : patch -> numbers audit -> checks (incl. check_numbers_v44.py and its mutation arm) -> offline render (CPU only, git show reads only)
# writes out/, v44_NUMBERS_AUDIT.md, v44_checks.txt, v44_render_check.txt, NUMBERS_V44.md, MANIFEST.txt. <tip> is the full 40-hex sha of the repository tip.
# Exit status: 0 only when the patch built, the audit has 0 unmatched tokens and 0 unresolved scope misses, and every check passes.
export CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
SHA=${1:?tip}
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
HERE=E:/Projects/TanitAD-artifacts/dagland/report_v44
cd "$HERE" || exit 2
$PY -I patch_v44.py $SHA $HERE/out || { echo "PATCH FAILED"; exit 2; }
$PY -I audit_v44.py $SHA $HERE/out $HERE/v44_NUMBERS_AUDIT.md | head -4; ARC=${PIPESTATUS[0]}
$PY -I checks_v44.py $SHA $HERE/out $HERE/v44_checks.txt | tail -1; CRC=${PIPESTATUS[0]}
( node render_check_v44.js out/v8f_report.html; node render_check_v44.js out/V8F_SCIENTIFIC_REPORT.md ) > v44_render_check.txt
{ echo; echo "== numbers audit (audit_v44.py): $(sed -n 5p v44_NUMBERS_AUDIT.md | cut -c1-200)"; echo "== offline render (the template's own pre-processing + local marked; v43 baseline: report_v43/v43_render_check.txt, 281 tables, 0 ragged):"; sed 's/^/   /' v44_render_check.txt; } >> v44_checks.txt
$PY -I - <<'PYEOF'
import hashlib, os
H = "E:/Projects/TanitAD-artifacts/dagland/report_v44"
names = ["out/V8F_SCIENTIFIC_REPORT.md", "out/v8f_report.html", "out/DIRECTIVES.txt", "patch_v44.py", "audit_v44.py", "checks_v44.py", "check_numbers_v44.py", "build_v44.sh", "render_check_v44.js", "edits_v44.md",
         "sec_747_world_model.md", "sec_748_renc_town.md", "NUMBERS_V44.md", "HANDOFF_V44.md", "HANDOFF_V44_BUILD.md", "v44_NUMBERS_AUDIT.md", "v44_checks.txt", "v44_render_check.txt"]
lines = ["# MANIFEST report v44 (built by the build agent for the Master Mind to land). name | bytes | sha256", ""]
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
