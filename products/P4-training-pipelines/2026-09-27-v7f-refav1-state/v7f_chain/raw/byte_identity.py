"""Default-profile BYTE-IDENTITY harness for scripts/v6_chain.py.

Runs a representative set of CLI invocations (NO --profile flag = the v6 default) through two
copies of the chain that sit side by side in the SAME scripts/ directory (so both import the same
train_v6_staged.py), and compares stdout, stderr, return code and every file an invocation writes,
byte for byte.

usage: python byte_identity.py <orig_chain.py> <new_chain.py> <workdir> <out_json>
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PY = sys.executable
orig, new, work, out_json = sys.argv[1:5]
work = Path(work)
if work.exists():
    shutil.rmtree(work)
work.mkdir(parents=True)
stack = Path(new).resolve().parents[1]
env = dict(os.environ)
env.update(PYTHONPATH=f"{stack};{stack.parent}", CUDA_VISIBLE_DEVICES="-1",
           OMP_NUM_THREADS="4", PYTHONUTF8="1")

# a banked-shaped geometry source, derived through the ORIGINAL chain's own parser
geo = work / "geom" / "config.json"
geo.parent.mkdir(parents=True)
r = subprocess.run([PY, "-c", "import sys, json; sys.path.insert(0, r'%s'); import importlib.util as u; "
                    "s = u.spec_from_file_location('c', r'%s'); m = u.module_from_spec(s); "
                    "sys.modules['c'] = m; s.loader.exec_module(m); print(json.dumps({'args': m.parse_argv_geometry([])}))"
                    % (stack / "scripts", orig)], capture_output=True, text=True, env=env, cwd=str(stack))
assert r.returncode == 0, r.stderr
geo.write_text(r.stdout, encoding="utf-8")
G = str(geo).replace("\\", "/")


def root(tag):
    return str(work / "root" / tag).replace("\\", "/")


CASES = {
    "A_plan": ["plan", "--root", root("A")],
    "B_plan_arms": ["plan", "--root", root("B"), "--st-arms", "goal", "mlp", "--st-winner", "goal"],
    "C_plan_dry": ["plan", "--root", root("C"), "--dry"],
    "D_plan_a40_notgc": ["plan", "--root", root("D"), "--a40", "--no-tac-goal-cond",
                         "--st-steps", "12000"],
    "E_commands_all": ["commands", "--root", root("E"), "--geometry-from", G],
    "F_commands_ST_override": ["commands", "--root", root("F"), "--step", "S-T", "--geometry-from", G,
                               "--allow-inconclusive-gate", "--gate-off-reason", "PI directive: x y"],
    "G_commands_arms": ["commands", "--root", root("G"), "--st-arms", "goal", "mlp",
                        "--st-winner", "mlp", "--geometry-from", G],
    "H_commands_SS_noseam": ["commands", "--root", root("H"), "--no-seam-dump", "--a40",
                             "--geometry-from", G, "--step", "S-S"],
    "I_admission": ["admission", "--root", root("I")],
    "J_status": ["status", "--root", root("J")],
    "K_next": ["next", "--root", root("K")],
    "L_manifests": ["manifests", "--root", root("L"), "--geometry-from", G,
                    "--dest", str(work / "manifests").replace("\\", "/")],
    "M_verify": ["verify", "--root", root("M"), "--step", "S-T"],
    "N_commands_dry_refuses": ["commands", "--root", root("N"), "--dry"],
    "O_unknown_step": ["commands", "--root", root("O"), "--step", "S-X"],
    "P_plan_tiny": ["plan", "--root", root("P"), "--tiny"],
    "Q_tilde": ["commands", "--root", "~/x"],
    "R_admission_arms": ["admission", "--root", root("R"), "--st-arms", "goal"],
    "S_commands_SW_nogeom": ["commands", "--root", root("S"), "--step", "S-W"],
    "T_plan_outjson": ["plan", "--root", root("T"),
                       "--out-json", str(work / "plan_T.json").replace("\\", "/")],
    "U_plan_default_root": ["plan"],
    "V_commands_ST_geom_nodefault_python": ["commands", "--step", "S-T", "--geometry-from", G,
                                            "--python", "/venv/bin/python", "--workdir",
                                            "/w/TanitAD/stack", "--n-candidates", "4"],
}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def run(chain, case, argv):
    # side files an invocation writes are cleared before each run so both see the same state
    for p in (work / "manifests", work / "plan_T.json"):
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
    rr = subprocess.run([PY, chain] + argv, capture_output=True, env=env, cwd=str(stack))
    files = {}
    for p in sorted((work / "manifests").glob("*")) if (work / "manifests").exists() else []:
        files[p.name] = sha(p.read_bytes())
    if (work / "plan_T.json").exists():
        files["plan_T.json"] = sha((work / "plan_T.json").read_bytes())
    return {"rc": rr.returncode, "stdout_sha": sha(rr.stdout), "stderr_sha": sha(rr.stderr),
            "stdout_len": len(rr.stdout), "files": files,
            "stderr_tail": rr.stderr.decode("utf-8", "replace").strip().splitlines()[-2:]}


res = {}
for case, argv in CASES.items():
    a = run(orig, case, argv)
    b = run(new, case, argv)
    same = (a["rc"], a["stdout_sha"], a["stderr_sha"], a["files"]) == \
           (b["rc"], b["stdout_sha"], b["stderr_sha"], b["files"])
    res[case] = {"argv": argv, "identical": same, "orig": a, "new": b}
    print(f"{'IDENTICAL' if same else 'DIFFERS  '} rc={a['rc']}/{b['rc']} "
          f"stdout={a['stdout_len']}B files={len(a['files'])} {case}")
summary = {"n_cases": len(res), "n_identical": sum(v["identical"] for v in res.values()),
           "orig_sha256": sha(Path(orig).read_bytes()), "new_sha256": sha(Path(new).read_bytes()),
           "cases": res}
Path(out_json).write_text(json.dumps(summary, indent=1), encoding="utf-8")
print(f"{summary['n_identical']}/{summary['n_cases']} identical")
sys.exit(0 if summary["n_identical"] == summary["n_cases"] else 1)
