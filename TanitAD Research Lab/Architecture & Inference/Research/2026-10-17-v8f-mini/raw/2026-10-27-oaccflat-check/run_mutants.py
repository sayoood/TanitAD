#!/usr/bin/env python3
"""Run each in-memory source mutant (cg_lib.MUTANTS) against the arm script(s) that must catch it; a mutant that leaves its arm GREEN is a hole in the check, reported as such.
    python run_mutants.py            # writes out/MUTANTS.json (+ one out/arm*_s0__MUT_<name>.json each)"""
import json, os, subprocess, sys, time
from pathlib import Path
ROOT = Path("E:/Projects/TanitAD-artifacts/oaccflat_check")
TREE = "E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a"
PLAN = [("acc_line", "arm1_dag_oacc.py", "arm1_s0"), ("swap_apply", "arm1_dag_oacc.py", "arm1_s0"), ("swap_apply", "arm2_flat_dag.py", "arm2_s0"),
        ("cut_ck", "arm2_flat_dag.py", "arm2_s0"), ("cut_code", "arm2_flat_dag.py", "arm2_s0"), ("loop_dag_apply", "arm1_dag_oacc.py", "arm1_s0")]
out = []
for name, script, tag in PLAN:
    env = {**os.environ, "CG_MUTANT": name, "CG_SEED": "0", "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": f"{TREE}/stack;{TREE}/taniteval;{TREE}/stack/scripts"}
    t = time.time()
    p = subprocess.run([sys.executable, str(ROOT / script)], env=env, capture_output=True, text=True, timeout=600, cwd=str(ROOT))
    (ROOT / "logs" / f"{tag}__MUT_{name}.log").write_text(p.stdout + "\n--stderr--\n" + p.stderr[-3000:], encoding="utf-8")
    j = ROOT / "out" / f"{tag}__MUT_{name}.json"
    r = json.loads(j.read_text(encoding="utf-8")) if j.exists() else None
    failed = [c for c in (r or {}).get("cases", []) if not c["ok"]]
    rec = {"mutant": name, "arm": tag, "script": script, "rc": p.returncode, "went_RED": p.returncode != 0 or bool(failed), "n_failed_cases": len(failed),
           "first_failed": [(c["name"][:110], c["detail"][:260]) for c in failed[:3]], "applied": (r or {}).get("mutant"), "seconds": round(time.time() - t)}
    out.append(rec)
    print(json.dumps(rec)[:600], flush=True)
(ROOT / "out" / "MUTANTS.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print("MUTANTS done:", {(r["mutant"], r["arm"]): r["went_RED"] for r in out})
