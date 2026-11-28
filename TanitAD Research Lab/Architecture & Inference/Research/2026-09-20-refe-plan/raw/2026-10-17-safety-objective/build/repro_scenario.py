# usage: python run_only.py <tree-root> <fn> <key[,key...]>   : runs one function of test_rls1_queue with only= keys on that tree (never edits it)
import sys, tempfile, time
from pathlib import Path
tree, fn, keys = sys.argv[1], sys.argv[2], set(sys.argv[3].split(","))
pk = Path(tree) / "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
sys.path.insert(0, str(pk / "refe")); sys.path.insert(0, str(pk / "eval"))
import test_rls1_queue as T
text = T.CHAIN.read_text(encoding="utf-8")
t0 = time.time()
with tempfile.TemporaryDirectory() as tmp:
    try:
        n = getattr(T, fn)(text, tmp, keys)
        print("RESULT PASS", n, round(time.time() - t0, 1), "s")
    except Exception as e:
        print("RESULT FAIL", type(e).__name__, str(e)[:900])
