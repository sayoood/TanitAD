import sys, tempfile
from pathlib import Path
tree = sys.argv[1]
pk = Path(tree) / "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
sys.path.insert(0, str(pk / "refe")); sys.path.insert(0, str(pk / "eval"))
import test_rls1_queue as T
orig = T.armloop_run
def wrapped(*a, **k):
    r = orig(*a, **k)
    print("ARMS TREE:", sorted(str(p.relative_to(r["R"])).replace("\\", "/") for p in (r["R"] / "arms").rglob("*")))
    print("CALLS naming R1d:", [l[:80] for l in r["calls"].splitlines() if "R1d" in l])
    return r
T.armloop_run = wrapped
text = T.CHAIN.read_text(encoding="utf-8")
with tempfile.TemporaryDirectory() as tmp:
    try: T.rls1d_checks(text, tmp, {"r0d_kill"})
    except AssertionError: print("assert failed as before")
