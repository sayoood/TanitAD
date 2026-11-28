"""Writes CODE_PIN.json: sha256 of every code file the REWARD-2 chain runs (scorer_gym/*.py, scorer_gym/tests/*.py, tests/*.py, launch_chain_v6.py), the frozen prereg, and every data file
the tests, R1-R4 and the judges read (stored M-GYM-4/5 records, the reference table raw/resim, centrelines, the REWARD-1 state list). chain_v6 refuses to run on any mismatch or on an unpinned code file.
Run by the implementer after the last edit; NOT run by the chain. Not pinned (written later by the chain): PREREG_REWARD_2_EXCLUDE.json, PREREG_REWARD_2_STATES.json (bound by DRAW_GATE / their own .sha256)."""
import glob, hashlib, json, os, sys
P6 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(P6, "scorer_gym"))
import chain_v6 as chain
PY = r"C:\Users\Admin\venvs\driverl-eval\Scripts\python.exe"
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
rels = chain.pinned_code_files(P6) + ["launch_chain_v6.py", "PREREG_REWARD_2.md", "PREREG_REWARD_2.sha256", "CURVE_SPEED_DEFECT.md", "reward2_counterfactuals.json"]
rels += sorted(os.path.relpath(f, P6).replace("\\", "/") for pat in ("raw/resim/*.json", "raw/cl/*.npz", "raw/cl_fast/*.npz") for f in glob.glob(os.path.join(P6, pat)))
rels += ["../pilot_v5/" + x for x in ("PREREG_REWARD_1_STATES.json", "PREREG_REWARD_1_STATES.sha256", "raw/results_v5.json", "raw/grepro_v5.json", "raw/run_main/states.jsonl.gz", "raw/ladder_20_160/states.jsonl.gz")]
rels += ["../pilot_v4/" + x for x in ("PREREG_M_GYM_4_STATES.json", "PREREG_M_GYM_4_STATES.sha256", "raw/results_v4.json")]
rels += ["../pilot_v3_diag/raw/q3_stall9.json", "../pilot_v2/run_main/states.jsonl.gz", "../pilot_v2/run_stall/states.jsonl.gz"]
files = {}
for r in rels:
    f = os.path.normpath(os.path.join(P6, r))
    assert os.path.exists(f), f
    files[r] = sha(f)
json.dump(dict(files=files, python=PY, pinned_before="the exclusion freeze, R1-R4, G-REPRO, the draw of seed 20261111 and every fresh rollout"), open(os.path.join(P6, "CODE_PIN.json"), "w", encoding="utf-8"), indent=1)
print(len(files), "files pinned; sha256(CODE_PIN.json) =", sha(os.path.join(P6, "CODE_PIN.json")))
