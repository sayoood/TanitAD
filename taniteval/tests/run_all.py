"""Run the whole TanitEval test suite without pytest (which is not installed on
the eval pod). Discovers every test_*.py beside this file, runs its test_*
functions, and exits non-zero if any fail.

    PYTHONPATH=/root/taniteval:/root/TanitAD/stack python tests/run_all.py
"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The repo layout FIRST, then the pod layout — so the suite runs from a clone
# as well as from /root/taniteval (it previously died with
# `ModuleNotFoundError: taniteval` anywhere except the eval pod).
sys.path.insert(0, str(HERE.parent))                       # taniteval/
sys.path.insert(0, str(HERE.parents[1] / "stack"))
sys.path.insert(0, str(HERE.parents[1] / "stack" / "scripts"))
sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")


def _load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    total = passed = 0
    for f in sorted(HERE.glob("test_*.py")):
        print(f"\n=== {f.name} ===")
        mod = _load(f)
        fns = [v for k, v in sorted(vars(mod).items())
               if k.startswith("test_") and callable(v)]
        for fn in fns:
            total += 1
            try:
                fn()
                passed += 1
                print(f"PASS {fn.__name__}")
            except Exception as e:  # noqa: BLE001
                import traceback
                print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
                traceback.print_exc()
    print(f"\n==== {passed}/{total} passed ====")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
