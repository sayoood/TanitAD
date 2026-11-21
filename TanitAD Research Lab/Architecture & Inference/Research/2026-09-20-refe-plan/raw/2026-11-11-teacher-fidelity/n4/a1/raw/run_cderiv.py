import sys, subprocess, os, json
RUN = "F:/tanitad_scratch/n4nt/run"
sys.path.insert(0, RUN + "/n4")
import n4_nt_chain as C
mode = sys.argv[1]          # real | mut
n = sys.argv[2] if len(sys.argv) > 2 else "200"
out = f"F:/tanitad_scratch/n4a1/{mode}.jsonl"
if os.path.exists(out): os.remove(out)
argv = [C.PY, "F:/tanitad_scratch/n4a1/new/n4_nt_build.py", "derive", "--route", "cache", "--n2r-plans", RUN + "/inputs/n2r/n2_plans_N2R.npz",
        "--rolls", RUN + "/inputs/n2r/rolls_RJ5/rows_s*of2.jsonl", "--metric-cache-root", C.CACHE_ROOT, "--out", out, "--shard", "0/1", "--cderiv-only"]
if mode == "mut": argv += ["--mutate", "label_ego_cache", "--smoke-allow-mutation"]
r = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", stdin=subprocess.DEVNULL, env=C.env(RUN), cwd="F:/tanitad_scratch/n4a1/new")
print(r.returncode, r.stdout[-500:], r.stderr[-1500:])
