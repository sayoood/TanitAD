"""Writes MANIFEST.txt of the v46 package (name, bytes, sha256). Called by build_v46.sh."""
import hashlib, os
H = "E:/Projects/TanitAD-artifacts/dagland/report_v46"
names = ["out/V8F_SCIENTIFIC_REPORT.md", "out/v8f_report.html", "out/DIRECTIVES.txt", "patch_v46.py", "audit_v46.py", "checks_v46.py", "check_numbers_v46.py", "build_v46.sh", "manifest_v46.py", "render_check_v46.js",
         "edits_v46.md", "sec_751_planner_gates_ladder.md", "sec_752_otband.md", "orig_sections/sec_751_planner_gates_ladder.md", "orig_sections/sec_752_otband.md", "NUMBERS_V46.md", "HANDOFF_V46_BUILD.md",
         "v46_NUMBERS_AUDIT.md", "v46_checks.txt", "v46_render_check.txt"]
lines = ["# MANIFEST report v46 (built by the build agent for the Master Mind to land). name | bytes | sha256", ""]
for n in names:
    p = os.path.join(H, n)
    if os.path.exists(p):
        b = open(p, "rb").read()
        lines.append("%s | %d | %s" % (n, len(b), hashlib.sha256(b).hexdigest()))
    else:
        lines.append("%s | MISSING" % n)
open(os.path.join(H, "MANIFEST.txt"), "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
