"""Run test_launch_gate.hnscreen_static() with a NON-ABORTING `ok`, so every failing assertion is listed (the real `ok` stops at the first).
usage: mut_harness.py <refe dir>   -> prints one line per failing assertion: <test line> <message head>; then a summary line."""
import importlib.util
import os
import sys

refe = os.path.abspath(sys.argv[1])
os.chdir(refe)
sys.path.insert(0, refe)
spec = importlib.util.spec_from_file_location("tlg", os.path.join(refe, "test_launch_gate.py"))
T = importlib.util.module_from_spec(spec)
sys.argv = [sys.argv[0], "--static"]          # test_launch_gate reads sys.argv for --no-banked at import
spec.loader.exec_module(T)
fails, total = [], [0]


def rec(cond, msg=""):
    total[0] += 1
    if not cond:
        fr = sys._getframe(1)
        while fr is not None and fr.f_code.co_name != "hnscreen_static":      # red() / clean() / pre() are helpers: report the line of the arm that called them
            fr = fr.f_back
        fails.append((fr.f_lineno if fr is not None else -1, str(msg).replace("\n", " ")[:170]))


T.ok = rec
err = None
try:
    T.hnscreen_static()
except Exception as e:                        # a non-ok line that depends on a failed assertion
    err = f"{type(e).__name__}: {str(e)[:170]}"
for ln, m in fails:
    print(f"RED line {ln}: {m}")
print(f"SUMMARY assertions={total[0]} failing={len(fails)} aborted_with={err}")
