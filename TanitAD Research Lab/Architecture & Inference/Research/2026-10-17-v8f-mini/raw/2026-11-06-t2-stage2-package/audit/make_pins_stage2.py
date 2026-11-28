import hashlib, os, sys
T = "E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f"
files = ["code/t2v8f_exec_hold.py", "code/t2v8f_stage2_cell.py", "code/t2v8f_stage2.py", "code/make_stage2.py", "code/t2v8f_stage2_score.py", "code/run_mutations_stage2.py",
         "stage2_src/jobs/stage2.cmd", "tests/test_t2v8f_stage2.py", "stage2/t2v8f_stage2_overlay.tar", "stage2/t2v8f_stage2_overlay.tar.sha256", "stage2/BUILD_RESULT.json",
         "stage2/SCENE_SOURCES.json", "stage2/scenes_put.sftp", "STAGE2_LAUNCH.md", "HANDOFF_I.md", "logs/mutations_stage2.log", "logs/pytest_stage2_full.log"]
out = ["# PINS_G (AGENT I, stage-2 overlay): sha256 of every file written or edited by the stage-2 work, 2026-11-06. LF endings; nothing committed or staged.",
       "# The overlay tar contains code/t2v8f_stage2.py, code/t2v8f_stage2_cell.py, code/t2v8f_exec_hold.py, jobs/stage2.cmd, MANIFEST2.json (NEW files only; the 7 stage-1 code files are untouched)."]
for f in files:
    p = f"{T}/{f}"
    if os.path.exists(p):
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        crlf = b"\r" in open(p, "rb").read() and not f.endswith((".tar", ".log"))
        out.append(f"{h}  {os.path.getsize(p):>10}  {f}{'  <-- CR FOUND' if crlf else ''}")
    else:
        out.append(f"{'MISSING':64}  {'':>10}  {f}")
open(f"{T}/PINS_G.txt", "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
print("\n".join(out))
