#!/usr/bin/env python
"""run_tests_plain.py -- lead's plain runner for a test file where pytest is not installed (the fork pod's venv has none and nothing is installed there).
Imports the test file by path, calls every function whose name starts with test_ (functions with no arguments only; one that takes arguments is counted as failed),
prints one line per test and a last line in the form of a pytest result line: "N passed in X.XXs" or "M failed, N passed in X.XXs". Exit 0 only with no failed test.

  python run_tests_plain.py <test file>
"""
import importlib.util
import inspect
import os
import sys
import time
import traceback


def main(argv):
    path = os.path.abspath(argv[1])
    sys.path.insert(0, os.path.dirname(path))
    spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(path))[0], path)
    mod = importlib.util.module_from_spec(spec)
    t0 = time.time()
    spec.loader.exec_module(mod)
    names = [n for n, f in sorted(vars(mod).items()) if n.startswith("test_") and inspect.isfunction(f)]
    ok, bad = 0, 0
    print("plain runner (no pytest): %s, %d test functions" % (os.path.basename(path), len(names)))
    for n in names:
        f = getattr(mod, n)
        try:
            if inspect.signature(f).parameters:
                raise RuntimeError("the plain runner cannot call a test that takes arguments")
            f()
            ok += 1
            print("PASSED " + n)
        except Exception:
            bad += 1
            print("FAILED " + n)
            print(traceback.format_exc()[-1500:])
    dt = time.time() - t0
    if not names:
        print("no tests ran in %.2fs" % dt)
        return 4
    print(("%d failed, %d passed in %.2fs" % (bad, ok, dt)) if bad else ("%d passed in %.2fs" % (ok, dt)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
