"""PREREG_V8F_OACC Amendment 2: the one-lever control on the REAL pulled records, reader as it ran (the 98713259 blob) vs fixed. Reads the
records only (argv / summary / selfcheck of B_s0 and X_s0); computes no row. python -I real_records_check.py <fixed tree> <tip reader copy>"""
import importlib.util, json, sys
R = "C:/Users/Admin/box2_ship/logs/v8f_oacc/runs/"
sys.path.insert(0, sys.argv[1] + "/stack"); sys.path.insert(0, sys.argv[1] + "/taniteval")
def mod(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
j = lambda f: json.load(open(R + f))
sm, sc, argv, rsm, rargv = j("M2b_s0_oacc/summary.json"), j("M2b_s0_oacc/selfcheck.json"), j("M2b_s0_oacc/argv.json"), j("M2b_s0/summary.json"), j("M2b_s0/argv.json")
argv, rargv = [a if isinstance(a, list) else a["argv"] for a in (argv, rargv)]
w = float(open("C:/Users/Admin/box2_ship/logs/v8f_oacc/w_acc.txt").read().strip())
fixed = sys.argv[1] + "/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/code/v8f_oacc_read.py"
for name, p in (("as run (98713259)", sys.argv[2]), ("fixed (Amendment 2)", fixed)):
    print(name, "->", json.dumps(mod(p, name.split()[0]).check_x_run(0, sm, argv, sc, rsm, rargv, w, holdout=None)))
