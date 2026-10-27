"""Re-run ONLY chosen mutants of mutate_dagger.py (no 21 baselines, no final baselines) on a COPY of the tree.   python survivor_check.py <copy root> <python> "<name substring>" ...
It takes the MUT list verbatim from mutate_dagger.py (same anchors, same -k selectors), applies one mutant at a time to the copy, runs the selector, restores, and verifies the copy byte-identical (sha256).
It is a quick pre-check, NOT the proof: the proof is the full mutate_dagger.py run (57 / 57). R77: never point it at the deliverable tree."""
import hashlib, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT, PY, ONLY = Path(sys.argv[1]).resolve(), sys.argv[2], sys.argv[3:]
src = (HERE.parent / "mutate_dagger.py").read_text(encoding="utf-8")
seg = src[src.index("KB4 = "):src.index("if LIST:")]
LIB = ROOT / "stack" / "tanitad" / "data" / "v8f_dagger.py"
BANK = ROOT / "stack" / "scripts" / "v8f_dagger_bank.py"
ns = {"LIB": LIB, "BANK": BANK}
exec(seg, ns)
MUT = ns["MUT"]
sha = lambda b: hashlib.sha256(b).hexdigest()
orig = {p: p.read_bytes() for p in (LIB, BANK)}
env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
base = Path(os.environ.get("DAGGER_MUT_BASETEMP") or "E:/Projects/TanitAD-artifacts/dg/pt") / "pt_survivors"
bad = 0
for path, name, edits, sel in MUT:
    if not any(o in name for o in ONLY):
        continue
    txt = orig[path].decode("utf-8")
    for a, b in edits:
        assert txt.count(a) == 1, f"anchor not found exactly once: {a!r}"
        txt = txt.replace(a, b, 1)
    compile(txt, str(path), "exec")
    for _ in range(360):
        r = subprocess.run([PY, "C:/Users/Admin/orcread/pytest_runs.py"], capture_output=True, text=True)
        if int(r.stdout.strip().splitlines()[-1]) < 2:
            break
        time.sleep(10)
    path.write_text(txt, encoding="utf-8", newline="")
    try:
        r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", str(base), "stack/tests/test_v8f_dagger.py", "-k", sel], cwd=ROOT, capture_output=True, text=True, env=env, timeout=1200)
    finally:
        path.write_bytes(orig[path])
    out = r.stdout.strip().splitlines()
    failed = [x for x in out if x.startswith("FAILED")]
    print(("RED      " if (r.returncode != 0 and failed) else "SURVIVED ") + f"[{path.name}] {name}  -k '{sel}'  ->  {len(failed)} failed: " + " | ".join(x[:150] for x in failed[:4]) + "  || " + (out[-1] if out else ""), flush=True)
    bad += int(not (r.returncode != 0 and failed))
ok = all(sha(p.read_bytes()) == sha(b) for p, b in orig.items())
print("RESTORED byte-identical:", ok, flush=True)
sys.exit(0 if (bad == 0 and ok) else 1)
