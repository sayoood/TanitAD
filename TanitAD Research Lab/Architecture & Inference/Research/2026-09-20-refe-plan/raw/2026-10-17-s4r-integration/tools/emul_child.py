"""emul_child.py <refe dir> <script> [--shim] [--utf8] : run <script> EXACTLY as gate_checks.check_suite spawns a suite entry (argv [sys.executable, refe/script], cwd=refe, env + PYTHONIOENCODING=utf-8 +
REFE_GATE_NESTED=<json [entry id]>, capture_output text utf-8) -- in particular WITHOUT `-X utf8` unless --utf8 -- and print rc / token / the output tail."""
import json, os, subprocess, sys
refe = os.path.abspath(sys.argv[1]); script = sys.argv[2]; flags = set(sys.argv[3:])
sys.path.insert(0, refe)
import gate_checks as C
e = {"script": script, "token": "ZZTEST_PASS"}
outer = ["0000000000000000:outer_not_a_suite_entry"] if "--outer" in flags else []     # --outer = the readiness harness's gate form on a job that is NOT a suite entry (the e2e suite-prerun)
env = dict(os.environ, PYTHONIOENCODING="utf-8", **{C.SUITE_NEST_ENV: json.dumps(outer + [C.suite_entry_id(e)])})
env.pop("PYTHONPATH", None)
if "--shim" in flags:
    env["PYTHONPATH"] = "C:/so/shim"
argv = [sys.executable] + (["-X", "utf8"] if "--utf8" in flags else []) + [os.path.join(refe, script)]
r = subprocess.run(argv, cwd=refe, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
print("RC", r.returncode, "| sentinel", env[C.SUITE_NEST_ENV])
print("---- stdout tail"); print("\n".join(r.stdout.splitlines()[-8:]))
print("---- stderr tail"); print("\n".join(r.stderr.splitlines()[-25:]))
