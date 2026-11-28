"""Copy repo -> mirror, VERIFIED BY SHA256. Exit non-zero on any mismatch."""
import hashlib, io, os, shutil, sys, time

SRC = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
DST = r"C:\Users\Admin\tanitad-wt"
FILES = sys.argv[1:] or [
    r"stack\tanitad\refs\refc.py",
    r"stack\tanitad\refs\refc_sampler.py",
    r"stack\tanitad\refs\refc_v3.py",
    r"stack\scripts\refc_v3_train.py",
    r"stack\tests\test_refc_sampler.py",
    r"stack\tests\test_refc_v3_refcv5_wiring.py",
]

def sha(p):
    for _ in range(12):
        try:
            return hashlib.sha256(io.open(p, "rb").read()).hexdigest()
        except OSError as e:
            print("  hash retry:", e, file=sys.stderr); time.sleep(1)
    return None

bad = 0
for rel in FILES:
    s, d = os.path.join(SRC, rel), os.path.join(DST, rel)
    a = sha(s)
    if a is None:
        print("UNREADABLE-SRC", rel); bad += 1; continue
    for _ in range(12):
        try:
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.copyfile(s, d); break
        except OSError as e:
            print("  copy retry:", e, file=sys.stderr); time.sleep(1)
    b = sha(d)
    if a == b and len(a) == 64:
        print("VERIFIED", a[:12], rel)
    else:
        print("MISMATCH", rel, a, b); bad += 1
# stale bytecode is invisible to a source-only sync
for root, dirs, _ in os.walk(os.path.join(DST, "stack")):
    for x in list(dirs):
        if x == "__pycache__":
            shutil.rmtree(os.path.join(root, x), ignore_errors=True)
print("BAD =", bad)
sys.exit(1 if bad else 0)
