import hashlib, subprocess, sys, pathlib, os
SC = pathlib.Path(r"C:\Users\Admin\msi_scratch")
TGT = SC / "stack" / "scripts" / "refc_v3_train.py"
PY  = r"C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
orig = TGT.read_bytes()
md5o = hashlib.md5(orig).hexdigest()
print("ORIGINAL md5 =", md5o, "bytes =", len(orig))

# The refusal under test: enable_max_speed's "not one window receives a value" guard.
NEEDLE = b"        if n_win and n_win_valid == 0:\r\n" if b"\r\n" in orig[:4000] else b"        if n_win and n_win_valid == 0:\n"
if NEEDLE not in orig:
    NEEDLE = b"if n_win and n_win_valid == 0:"
print("needle found:", orig.count(NEEDLE), "occurrence(s)")
assert orig.count(NEEDLE) == 1, "needle must be unique"
MUT = NEEDLE.replace(b"if n_win and n_win_valid == 0:", b"if False and n_win and n_win_valid == 0:")

def run(tag):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONPATH=str(SC/"stack"))
    r = subprocess.run([PY,"-m","pytest","stack/tests/test_max_speed_wiring.py","-q","--no-header",
                        "-k","REFUSED or refus or v8"], cwd=str(SC), env=env,
                       capture_output=True, text=True, errors="replace")
    tail = "\n".join([l for l in r.stdout.strip().splitlines() if l.strip()][-6:])
    print(f"\n### {tag}  exit={r.returncode}\n{tail}")
    return r.returncode, r.stdout

try:
    print("\n=== STEP 1: BASELINE (refusal intact) ===")
    rc1,_ = run("BASELINE")
    print("\n=== STEP 2: MUTATION (refusal disabled) ===")
    TGT.write_bytes(orig.replace(NEEDLE, MUT))
    rc2,out2 = run("MUTATED")
    reds = [l for l in out2.splitlines() if l.startswith("FAILED")]
    print("tests that went RED:")
    for l in reds: print("   ", l.strip()[:150])
finally:
    TGT.write_bytes(orig)
    md5r = hashlib.md5(TGT.read_bytes()).hexdigest()
    print("\n=== STEP 3: RESTORED ===")
    print("restored md5 =", md5r, " bytes restored EXACTLY:", md5r == md5o)
    rc3,_ = run("RESTORED")

print("\n" + "="*64)
verdict = (rc1==0 and rc2!=0 and rc3==0 and md5r==md5o)
print("VERDICT: the refusal HAS TEETH ->", verdict)
print("  baseline GREEN:", rc1==0, "| mutation RED:", rc2!=0, "| restored GREEN:", rc3==0, "| bytes exact:", md5r==md5o)
