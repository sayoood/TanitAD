#!/usr/bin/env python3
"""Run the A2 tests with real pytest when importable, else with the shim (driverl-eval has no pytest; the shim has NO fixtures, so run test_behaviour_gate.py under real pytest in the tanitad venv).   usage: python tests/run_tests.py [file.py ...] [-k substr]"""
import importlib.util
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
args = [a for a in sys.argv[1:]]
kexpr = None
if "-k" in args:
    i = args.index("-k"); kexpr = args[i + 1]; del args[i:i + 2]
files = args or sorted(f for f in os.listdir(HERE) if f.startswith("test_") and f.endswith(".py"))
files = [f if os.path.isabs(f) else os.path.join(HERE, f) for f in files]
try:
    import pytest  # noqa: F401
    sys.exit(pytest.main(files + (["-k", kexpr] if kexpr else []) + ["-q", "-p", "no:cacheprovider"]))
except ImportError:
    sys.path.insert(0, os.path.join(HERE, "_pytest_shim"))
    import pytest
    ok = bad = skip = 0
    for f in files:
        spec = importlib.util.spec_from_file_location(os.path.basename(f)[:-3], f)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        try:
            spec.loader.exec_module(mod)
        except pytest.Skipped as e:
            print(f"SKIP {os.path.basename(f)}: {e}"); skip += 1; continue
        for name in sorted(n for n in dir(mod) if n.startswith("test_")):
            if kexpr and kexpr not in name:
                continue
            t0 = time.time()
            try:
                getattr(mod, name)()
                ok += 1; print(f"PASS {os.path.basename(f)}::{name} ({time.time() - t0:.1f}s)")
            except pytest.Skipped as e:
                skip += 1; print(f"SKIP {name}: {e}")
            except Exception:
                bad += 1; print(f"FAIL {os.path.basename(f)}::{name}"); traceback.print_exc()
    print(f"\n{ok} passed, {bad} failed, {skip} skipped")
    sys.exit(1 if bad else 0)
