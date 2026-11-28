"""A CPU dry run of SPEC_REFCV7 A11 on Thor, BEFORE any binding GPU run: wrap the REAL map harness
(`--help`: it imports everything it imports at module level, then exits 0) in closure_run.py, and judge
the record with the gate's own `judge_closure` against the same tree and Thor's launch venv.

    python closure_probe_thor.py <gate dir>      -> <gate dir>/out/closure_probe/probe.json

Expected: exactly two refusals, both by construction of a dry run -- NOT binding (no --binding) and
no PASS JSON (--help writes none). Anything else is a defect of the closure path on this host
(shadowing, an environment mismatch, an unrecorded module-level import, a looked-for module that
resolves in the tree)."""
import json
import os
import subprocess
import sys
from pathlib import Path

G = Path(sys.argv[1])
tree = G / "cab"
out = G / "out" / "closure_probe"
out.mkdir(parents=True, exist_ok=True)
harness = "stack/scripts/map_hires_overfit.py"
env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1",
           PYTHONPATH=f"{tree / 'stack'}:{tree / 'taniteval'}")
clo = out / "help.closure.json"
p = subprocess.run([sys.executable, str(tree / "stack/scripts/closure_run.py"), "--out", str(clo),
                    "--tree", str(tree), "--", str(tree / harness), "--help"],
                   cwd=str(tree), env=env, capture_output=True, text=True, timeout=600)
res = {"rc": p.returncode, "stderr_tail": p.stderr[-600:], "closure_written": clo.is_file()}
if clo.is_file():
    sys.path.insert(0, str(tree / "stack" / "scripts"))
    sys.path.insert(0, str(tree / "stack"))
    import launch_gate as LG
    c = json.loads(clo.read_text(encoding="utf-8"))
    host = LG._host_env()
    reasons, det = LG.judge_closure("G-MAP-OVERFIT", str(clo), None, tree, harness, host_env=host)
    res.update(n_modules=len(c["modules"]), n_probed=len(c["probed"]), n_absent=len(c["absent"]),
               outside_tree=c["outside_tree"], children=c["children"], env_record=c["env"],
               env_host=host, exit_status=c["exit_status"],
               tanitad_modules=sum(1 for r, _ in c["modules"] if r.startswith("stack/tanitad/")),
               static_eager_files=det.get("static_eager_files"), reasons=reasons)
    expected = [r for r in reasons if "NOT binding" in r or "PASS JSON" in r]
    res["verdict"] = ("AS EXPECTED (only the two dry-run refusals)"
                      if len(expected) == 2 and len(reasons) == 2 else "UNEXPECTED REFUSALS")
(out / "probe.json").write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8")
print("ZZPROBE-" + str(res.get("verdict", "NO-CLOSURE")).split(" ")[0] + "ZZ")
