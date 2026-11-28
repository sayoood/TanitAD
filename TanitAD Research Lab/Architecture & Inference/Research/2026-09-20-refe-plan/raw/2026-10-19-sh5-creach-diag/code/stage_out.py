"""Stage the diagnosis package into out/ (repo layout) and write out/paths.json = [[repo path, abs source, "raw"], ...].  Copies only small text files (code, raw json, logs, grids, md)."""
import json, os, shutil, glob
ROOT = "C:/Users/Admin/par18/sh5diag"
REPO = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-sh5-creach-diag"
OUT = f"{ROOT}/out"
srcs = [f"{ROOT}/PREREG_SH5_DAC_PROX_AMENDMENT4.draft.md", f"{ROOT}/HANDOFF.md"] + sorted(glob.glob(f"{ROOT}/PREREG_GRID*.md")) + sorted(glob.glob(f"{ROOT}/PREREG_GRID*.md.sha256"))
for pat in ("code/*.py", "raw/*.json", "work/*.json", "work/*.log", "work/*.ps1", "work/diag*/*.json", "work/confirm*/*.json", "work/confirm*/*.log", "code/amend4/*.py"):
    srcs += sorted(glob.glob(f"{ROOT}/{pat}"))
rows = []
if os.path.isdir(f"{OUT}/TanitAD Research Lab"):
    shutil.rmtree(f"{OUT}/TanitAD Research Lab")
for s in srcs:
    s = s.replace("\\", "/")
    rel = os.path.relpath(s, ROOT).replace("\\", "/")
    if os.path.getsize(s) > 5_000_000:
        print("SKIP large", rel); continue
    dst = f"{OUT}/{REPO}/{rel}"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(s, dst)
    rows.append([f"{REPO}/{rel}", dst, "raw"])
json.dump(rows, open(f"{OUT}/paths.json", "w"), indent=1)
print(f"staged {len(rows)} files")
