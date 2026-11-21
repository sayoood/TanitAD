import hashlib
import os

T = "E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f"
files = ["DEFINITIONS_K.md", "STAGE3_LAUNCH.md", "HANDOFF_K.md",
         "code/t2v8f_stage3.py", "code/make_stage3.py", "code/make_starts3.py", "code/download_refs3.py", "code/extract_ref_frames3.py", "code/t2v8f_stage3_score.py", "code/run_mutations_stage3.py",
         "stage3_src/jobs/stage3.cmd", "tests/test_t2v8f_stage3.py", "audit/make_pins_stage3.py",
         "stage3/t2v8f_stage3_overlay.tar", "stage3/t2v8f_stage3_overlay.tar.sha256", "stage3/BUILD_RESULT3.json", "stage3/SCENE_SOURCES3.json", "stage3/scenes_put3.sftp", "stage3/stage3_plan.json",
         "stage3/GUARD_STARTS.json", "stage3/ref_frames/DOWNLOAD_RESULT3.json", "stage3/ref_frames/EXTRACT_RESULT3.json", "stage3/ref_frames/BITCHECK_7c72937c.json",
         "stage3/ref_frames/REF_FRAMES_SOURCES3.json",
         "logs/download_refs3.log", "logs/extract_ref_frames3.log", "logs/make_stage3.log", "logs/pytest_stage3.log", "logs/mutations_stage3.log", "logs/pytest_stage3_full.log"]
out = ["# PINS_K (AGENT K, stage 3: the 20 guard scenes): sha256 of every file written, 2026-11-06. LF endings; nothing committed, staged or deleted; stage-2 files NOT edited.",
       "# Outside the repo tree (sha256 in stage3/ref_frames/DOWNLOAD_RESULT3.json and EXTRACT_RESULT3.json): 20 mp4 + 20 ref_frames.npz in C:/Users/Admin/TanitAD-scratch/t2_scenes/<id>/.",
       f"# stage-2 driver (must stay 33b65b19...): {hashlib.sha256(open(T + '/code/t2v8f_stage2.py', 'rb').read()).hexdigest()}"]
for f in files:
    p = f"{T}/{f}"
    if os.path.exists(p):
        b = open(p, "rb").read()
        crlf = b"\r" in b and not f.endswith((".tar", ".log"))
        out.append(f"{hashlib.sha256(b).hexdigest()}  {len(b):>10}  {f}{'  <-- CR FOUND' if crlf else ''}")
    else:
        out.append(f"{'MISSING':64}  {'':>10}  {f}")
with open(f"{T}/PINS_K.txt", "w", encoding="utf-8", newline="\n") as fh:
    fh.write("\n".join(out) + "\n")
print("\n".join(out))
