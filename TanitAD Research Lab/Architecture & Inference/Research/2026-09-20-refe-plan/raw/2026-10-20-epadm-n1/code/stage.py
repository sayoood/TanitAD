"""Stage the EPADM-N1 deliverables under out/<repo path>/ and write out/paths.json ([repo path, absolute source, "raw"]) + out/RESULT.md.  Does not git add / commit / push.   python stage.py"""
import json
import os
import shutil

BASE = "C:/Users/Admin/par18/n1sim"
REPO = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-20-epadm-n1"
FILES = {
    "SIM_DESIGN_N1.md": "SIM_DESIGN_N1.md", "SIM_DESIGN_N1.sha256": "SIM_DESIGN_N1.sha256", "SIM_DESIGN_N1.hashed_at.txt": "SIM_DESIGN_N1.hashed_at.txt", "HANDOFF.md": "HANDOFF.md",
    "TIP_SHA_N1.txt": "TIP_SHA_N1.txt", "geometry_s0.json": "geometry_s0.json",
    "code/epadm_n1.py": "code/epadm_n1.py", "code/epadm_sim_n1.py": "code/epadm_sim_n1.py", "code/epadm_sim_s0.py": "code/epadm_sim_s0.py", "code/test_n1.py": "code/test_n1.py", "code/test_sim_s0.py": "code/test_sim_s0.py",
    "code/mutation_check_n1.py": "code/mutation_check_n1.py", "code/make_tables_n1.py": "code/make_tables_n1.py", "code/make_result_n1.py": "code/make_result_n1.py", "code/stage.py": "code/stage.py",
    "work/args.sh": "code/run/args.sh", "work/env.sh": "code/run/env.sh", "work/chain1.sh": "code/run/chain1.sh", "work/chain2.sh": "code/run/chain2.sh", "work/finish.sh": "code/run/finish.sh",
    "work/sim/sim_results.json": "raw/results/sim_results.json", "work/sim/cost.json": "raw/results/cost.json", "work/sim/pc3.json": "raw/results/pc3.json", "work/sim/pc8.json": "raw/results/pc8.json",
    "work/sim/basex_n1.json": "raw/results/basex_n1.json", "work/sim/basex_n1.npz": "raw/inputs/basex_n1.npz", "work/stats_r2confirm.npz": "raw/inputs/stats_r2confirm_copy_of_tip_blob.npz",
    "work/tables.md": "raw/results/tables.md", "work/mutation.json": "raw/results/mutation.json", "work/logs/pytest_n1.log": "raw/logs/pytest_n1.log", "work/logs/pytest_sim_s0_copy.log": "raw/logs/pytest_sim_s0_copy.log",
    "work/logs/mutation_n1.log": "raw/logs/mutation_n1.log", "work/logs/chain1.log": "raw/logs/chain1.log", "work/logs/chain2.log": "raw/logs/chain2.log",
}
for j in ("C-GS41-A", "C-GS41-P", "C-GS117-A", "C-GS117-P", "X-GS41-A", "X-GS117-A"):
    FILES[f"work/sim/run_{j}.json"] = f"raw/results/run_{j}.json"
    FILES[f"work/sim/replicates_{j}.npz"] = f"raw/results/replicates_{j}.npz"
    FILES[f"work/sim/repro_{j}.json"] = f"raw/results/repro_{j}.json"


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
        paths.append([f"{REPO}/{dst}", s.replace("\\", "/"), "raw"])
    rs = os.path.join(BASE, "RESULT.md")
    if os.path.exists(rs):
        shutil.copyfile(rs, os.path.join(out, "RESULT.md"))
        paths.append([f"{REPO}/RESULT.md", rs.replace("\\", "/"), "raw"])
        shutil.copyfile(rs, os.path.join(out, REPO, "RESULT.md"))
    json.dump(paths, open(os.path.join(out, "paths.json"), "w"), indent=1)
    print(len(paths), "files staged")


if __name__ == "__main__":
    main()
