"""IMAG1B cleanup (CLAUDE.md auto-delete rule): deletes THIS run's own new scratch by literal absolute path, only after RESULT_IMAG1B.json and RESULT_IMAG1B.md exist, and logs path + size in
DELETIONS_IMAG1B.log. A dataset that did not run (NOT RUN) keeps its offset index and real-stage pickle so it can resume. IMAG1's caches (imag1_ctx_*.pkl, imag1_real_*.pkl, latents_rand.pt) are NOT touched:
they are left for the Master Mind. The draws (draws_imag1b_<ds>.jsonl, in this folder) are kept."""
import json, os, sys, time
sys.dont_write_bytecode = True
os.chdir(os.path.dirname(os.path.abspath(__file__)))
assert os.path.exists("RESULT_IMAG1B.json") and os.path.exists("RESULT_IMAG1B.md"), "nothing is deleted before the result and the report exist"
R = json.load(open("RESULT_IMAG1B.json", encoding="utf-8"))
ran = {ds for ds, D in R["datasets"].items() if D.get("status") == "RUN"}
CAND = {"eval141": ["F:/TanitAD-scratch/imag1b_off_eval141.pkl", "F:/TanitAD-scratch/imag1b_real_eval141.pkl"],
        "train": ["F:/TanitAD-scratch/imag1b_off_train.pkl", "F:/TanitAD-scratch/imag1b_real_train.pkl"]}
with open("DELETIONS_IMAG1B.log", "a", encoding="utf-8") as lg:
    for ds, paths in CAND.items():
        if ds not in ran:
            lg.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} KEPT (dataset {ds} did not run): {paths}\n")
            continue
        for p in paths:
            if os.path.exists(p):
                n = os.path.getsize(p)
                os.remove(p)
                lg.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} deleted {p} {n} bytes (re-creatable: build_off_imag1b.py / rerun of the real stage)\n")
print(open("DELETIONS_IMAG1B.log", encoding="utf-8").read())
