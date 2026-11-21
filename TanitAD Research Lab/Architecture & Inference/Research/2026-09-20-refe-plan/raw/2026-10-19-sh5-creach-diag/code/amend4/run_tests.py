"""Run the SH5 CPU test files and write tests_summary.json (the evidence behind the `red_table` gate).  CPU only; at most 2 threads.

    python -B run_tests.py --out tests_summary.json [--files test_sh5_lib.py test_sh5_branch.py test_sh5_ctx.py test_sh5_guard.py test_sh5_pipeline.py]
"""
import argparse
import io
import json
import os
import sys
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--files", nargs="*", default=["test_sh5_lib.py", "test_sh5_branch.py", "test_sh5_ctx.py", "test_sh5_guard.py", "test_sh5_pipeline.py"])
    a = ap.parse_args()
    res, all_ok, log = {}, True, []
    for f in a.files:
        t0 = time.time()
        buf = io.StringIO()
        suite = unittest.defaultTestLoader.loadTestsFromName(f[:-3])
        r = unittest.TextTestRunner(stream=buf, verbosity=1).run(suite)
        ok = r.wasSuccessful()
        all_ok = all_ok and ok
        res[f] = {"ran": r.testsRun, "failures": len(r.failures), "errors": len(r.errors), "ok": ok, "seconds": round(time.time() - t0, 1)}
        log.append(f"=== {f}\n{buf.getvalue()}")
    res["all_passed"] = all_ok
    res["log"] = "\n".join(log)
    json.dump(res, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "log"}, indent=1))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
