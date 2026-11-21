"""Stage the EP-ADMISSION-R2-CONFIRM deliverables under C:/Users/Admin/par18/epadm2/out/ and write paths.json [[repo path, abs source, "raw"], ...].  Idempotent."""
import hashlib
import json
import os
import shutil

REPO_BASE = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw"
W = "D:/epadm2/w/raw"
RUN = "D:/epadm2/run"
OUT = "C:/Users/Admin/par18/epadm2/out"
R1, R2 = "2026-10-19-ep-admission-r1", "2026-10-19-ep-admission-r2-confirm"

items = []                                                                          # (repo path relative to REPO_BASE, source)
for f in ("ep_admission_sim.py", "test_ep_admission_sim.py", "r2_confirm.py", "a2_power_g25_g05.py"):
    items.append((f"{R1}/code/{f}", f"{W}/{R1}/code/{f}"))
items.append((f"{R2}/RESULT.md", "D:/epadm2/RESULT.md"))
for f in ("chain.cmd", "cmp_stats.py", "patch_sim.py", "patch_test.py", "stage.py"):
    items.append((f"{R2}/code/{f}", f"D:/epadm2/{f}"))
for f in ("sim_results.json", "precompute.json", "r2_confirm.json", "result_tables.md", "repro.json", "repro_offset0_vs_banked.json", "repro_offset1e6_vs_banked_RED.json",
          "stats_vs_banked.json", "a2_power_g25_g05.json", "replicates.npz", "stats.npz", "a2_power_g25_g05_replicates.npz"):
    items.append((f"{R2}/raw/results/{f}", f"{RUN}/{f}"))
for f in ("chain.log", "precompute.log", "stats_vs_banked.log", "repro_offset0_vs_banked.log", "run.log", "repro.log", "repro_offset1e6_vs_banked_RED.log", "a2_power_g25_g05.log",
          "r2_confirm.log", "r2_confirm_selftest.log", "test_sim_BASE.log", "test_sim_MINE.log"):
    items.append((f"{R2}/raw/logs/{f}", f"{RUN}/{f}"))

paths, rows = [], []
for rel, src in items:
    if not os.path.exists(src):
        print("MISSING", src)
        continue
    repo_path = f"{REPO_BASE}/{rel}"
    dst = f"{OUT}/{repo_path}"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    h = hashlib.sha256(open(dst, "rb").read()).hexdigest()
    paths.append([repo_path, dst, "raw"])
    rows.append((rel, os.path.getsize(dst), h))
os.makedirs(OUT, exist_ok=True)
json.dump(paths, open(f"{OUT}/paths.json", "w", encoding="utf-8", newline="\n"), indent=1)
for rel, size, h in sorted(rows):
    print(f"{size:>10} {h}  {rel}")
print(len(paths), "files staged ->", OUT)
