"""Stage the EPADM-SEARCHBANK-S0 deliverables under out/<repo path>/ and write out/paths.json ([repo path, absolute source, "raw"]).  Does not git add / commit / push.   python stage.py"""
import json
import os
import shutil

BASE = "C:/Users/Admin/par18/s0amend"
REPO = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-20-epadm-s0bank"
FILES = {
    "AMENDMENT_S0BANK_DRAFT.md": "AMENDMENT_S0BANK_DRAFT.md", "HANDOFF.md": "HANDOFF.md", "SIM_DESIGN_S0.md": "SIM_DESIGN_S0.md", "SIM_DESIGN_S0.sha256": "SIM_DESIGN_S0.sha256",
    "SIM_DESIGN_S0.hashed_at.txt": "SIM_DESIGN_S0.hashed_at.txt", "geometry_s0.json": "geometry_s0.json", "TIP_SHA.txt": "TIP_SHA.txt",
    "code/epadm_sim_s0.py": "code/epadm_sim_s0.py", "code/test_sim_s0.py": "code/test_sim_s0.py", "code/make_tables.py": "code/make_tables.py",
    "code/mutation_check_s0.py": "code/mutation_check_s0.py", "code/stage.py": "code/stage.py",
    "work/sim/sim_results.json": "raw/results/sim_results.json", "work/sim/cost.json": "raw/results/cost.json", "work/sim/pc3.json": "raw/results/pc3.json", "work/sim/pc8.json": "raw/results/pc8.json",
    "work/tables.md": "raw/results/tables.md", "work/mutation.json": "raw/results/mutation.json", "work/logs/pytest_sim_s0.log": "raw/logs/pytest_sim_s0.log",
    "work/logs/chain1.log": "raw/logs/chain1.log", "work/logs/chain2.log": "raw/logs/chain2.log", "work/logs/finish.log": "raw/logs/finish.log",
}
for j in ("C-GS41-A", "C-GS41-T", "C-GS117-A", "C-GS117-T", "X-GS41-A", "X-GS117-A"):
    FILES[f"work/sim/run_{j}.json"] = f"raw/results/run_{j}.json"
    FILES[f"work/sim/replicates_{j}.npz"] = f"raw/results/replicates_{j}.npz"
    if j.endswith("-A"):
        FILES[f"work/sim/repro_{j}.json"] = f"raw/results/repro_{j}.json"
FILES["work/token_shas.json"] = "raw/results/token_shas.json"
FILES["work/stats_r2confirm.npz"] = "raw/inputs/stats_r2confirm_copy_of_tip_blob.npz"


def main():
    out = os.path.join(BASE, "out")
    paths = []
    for src, dst in FILES.items():
        s = os.path.join(BASE, src)
        if not os.path.exists(s):
            print("MISSING (not staged):", src)
            continue
        d = os.path.join(out, REPO, dst)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copyfile(s, d)
        paths.append([f"{REPO}/{dst}", s, "raw"])
    rs = os.path.join(BASE, "RESULT.md")
    if os.path.exists(rs):
        shutil.copyfile(rs, os.path.join(out, "RESULT.md"))
    json.dump(paths, open(os.path.join(out, "paths.json"), "w"), indent=1)
    print(len(paths), "files staged")


if __name__ == "__main__":
    main()
