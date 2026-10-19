"""run a list of test scripts ONE AT A TIME against a tree (base or new); write a JSON summary + per-test logs.  usage: run_batch.py <tree: base|new> <out_dir> <timeout_s> test[:args...] ...
A test token is `script.py` or `script.py|--arg|--arg` (args separated by |).  Runs on CPU only; never touches the GPU lock; no pkill."""
import json, os, subprocess, sys, time
tree, out, tmo = sys.argv[1], sys.argv[2], int(sys.argv[3])
ROOT = {"base": "C:/Users/Admin/hn34/base", "new": "C:/Users/Admin/hn34/tree"}[tree]
PK = ROOT + "/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
os.makedirs(out, exist_ok=True)
res = {}
for tok in sys.argv[4:]:
    parts = tok.split("|")
    script, args = parts[0], parts[1:]
    d = PK + ("/eval" if script.startswith("eval/") else "/refe")
    name = script.split("/")[-1]
    key = name + ("::" + " ".join(args) if args else "")
    log = os.path.join(out, key.replace("::", "__").replace(" ", "_").replace("--", "") + ".log")
    t0 = time.time()
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    try:
        with open(log, "wb") as fh:
            p = subprocess.run([PY, "-u", name] + args, cwd=d, stdout=fh, stderr=subprocess.STDOUT, timeout=tmo, env=env)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        rc = "TIMEOUT"
    txt = open(log, "rb").read().decode("utf-8", errors="replace")
    toks = [ln.strip() for ln in txt.splitlines() if ln.startswith(("ZZ", "ZZTEST", "ZZLG")) or "PASS" in ln[:40] or "FAIL" in ln[:40]]
    res[key] = {"rc": rc, "secs": round(time.time() - t0, 1), "token_lines": toks[-4:], "tail": txt.strip().splitlines()[-3:], "log": log}
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(key, rc, res[key]["secs"], flush=True)
print("BATCH_DONE", flush=True)
