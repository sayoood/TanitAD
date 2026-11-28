"""Run each listed test file in FULL (no -k), one pytest process per file, sequentially.
usage: run_list.py <tree> <tag> <listfile>   -> logs/<tag>.log (tails) and logs/<tag>.tsv (counts)"""
import os, re, subprocess, sys, time
tree, tag, lst = sys.argv[1], sys.argv[2], sys.argv[3]
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
LOGD = "C:/Users/Admin/hjunits2/logs"
env = dict(os.environ, PYTHONUTF8="1", CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8")
env.pop("PYTHONPATH", None)
files = [l.strip().replace("\\", "/") for l in open(lst, encoding="utf-8") if l.strip()]
log = open(f"{LOGD}/{tag}.log", "w", encoding="utf-8")
tsv = open(f"{LOGD}/{tag}.tsv", "w", encoding="utf-8")
t00 = time.time()
for f in files:
    top = f.split("/")[0]
    cwd = os.path.join(tree, top if top in ("stack", "taniteval") else ".")
    rel = f[len(top) + 1:] if top in ("stack", "taniteval") else f
    e = dict(env)
    if top == "tools":
        e["PYTHONPATH"] = os.pathsep.join([os.path.join(tree, "stack"), os.path.join(tree, "stack", "scripts"), os.path.join(tree, "taniteval"), os.path.join(tree, "tools")])
    t0 = time.time()
    try:
        p = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", rel], cwd=cwd, env=e,
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3000)
        out = (p.stdout or "") + (p.stderr or "")
        rc = p.returncode
    except subprocess.TimeoutExpired as ex:
        out, rc = "TIMEOUT", -9
    dt = time.time() - t0
    last = [l for l in out.strip().splitlines() if re.search(r"\b(passed|failed|error|errors|skipped|deselected|no tests ran)\b", l)]
    summ = last[-1] if last else out.strip().splitlines()[-1:] and out.strip().splitlines()[-1] or ""
    def cnt(k):
        m = re.search(r"(\d+) " + k, summ)
        return int(m.group(1)) if m else 0
    tsv.write(f"{f}\t{cnt('passed')}\t{cnt('failed')}\t{cnt('error')}\t{cnt('skipped')}\t{rc}\t{dt:.0f}s\n"); tsv.flush()
    log.write(f"=== {f}  rc={rc}  {dt:.0f}s\n")
    log.write("\n".join(out.strip().splitlines()[-25:]) + "\n"); log.flush()
log.write(f"=== DONE total {time.time()-t00:.0f}s\n"); log.close(); tsv.close()
