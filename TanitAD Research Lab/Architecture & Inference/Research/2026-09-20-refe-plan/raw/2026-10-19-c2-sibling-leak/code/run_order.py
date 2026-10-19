"""Run ONE test file in ONE runner and ONE explicit order, and print one machine-readable line.  CPU only.

    python -B -I run_order.py --list  <tree> <unittest|pytest> <test_file.py>                    -> prints the default-order test ids as JSON
    python -B -I run_order.py --run   <tree> <unittest|pytest> <test_file.py> <order> <ids.json>   -> runs them in `order`, prints   RESULT {json}

`order` (all derive from the default order `ids.json` holds):
    default   the runner's own order (unittest: class name, then method name; pytest: definition order)
    first     the POISONERS first, then the rest        (poisoners = the C2-GATE-SIBLING-LEAK test class when the file has it + the real-run test below)
    last      the rest, then the POISONERS
    reversed  the whole file reversed
The real-run test is the one the earlier (not landed) attempt broke with FileNotFoundError in `srttc_c2.run_c2`.
`tree` is a flat code dir (the files of 2026-10-18-sel-rank/code); the test file imports its siblings from its own dir.  pytest is APPENDED to sys.path from $PYTEST_SITE (it is not installed in the
gates' interpreter), so torch / numpy still resolve from the venv.
"""
import json
import os
import sys
import time

REAL_RUN = "test_a_valid_token_writes_the_c2_rows_result_and_attestation_bound_to_the_r1_run"
NEW_CLASS = "TestEachLoadUsesExactlyTheVerifiedFiles"


def poisoners(ids):
    return [i for i in ids if NEW_CLASS in i] + [i for i in ids if REAL_RUN in i]


def reorder(ids, order):
    if order == "default":
        return list(ids)
    if order == "reversed":
        return list(reversed(ids))
    grp = poisoners(ids)
    rest = [i for i in ids if i not in set(grp)]
    return grp + rest if order == "first" else rest + grp


def main():
    mode, tree, runner, fname = sys.argv[1:5]
    tree = os.path.abspath(tree)
    ids_file_abs = os.path.abspath(sys.argv[6]) if len(sys.argv) > 6 else None
    os.chdir(tree)
    sys.path.insert(0, tree)
    mod = fname[:-3]
    if runner == "pytest":
        sys.path.append(os.environ.get("PYTEST_SITE", "C:/Users/Admin/par18/c2leak/pytest_site"))
    if mode == "--list":
        if runner == "unittest":
            import importlib
            import unittest
            ids = []

            def walk(s):
                for t in s:
                    walk(t) if isinstance(t, unittest.TestSuite) else ids.append(t.id())
            walk(unittest.TestLoader().loadTestsFromModule(importlib.import_module(mod)))
        else:
            import pytest
            ids = []

            class Grab:
                def pytest_collection_modifyitems(self, items):
                    ids.extend(i.nodeid for i in items)
            pytest.main(["--collect-only", "-q", "-p", "no:cacheprovider", fname], plugins=[Grab()])
        print("IDS " + json.dumps(ids))
        return 0
    order = sys.argv[5]
    ids = reorder(json.load(open(ids_file_abs)), order)
    t0 = time.time()
    outcomes = {}
    if runner == "unittest":
        import importlib
        import unittest
        importlib.import_module(mod)
        loader = unittest.TestLoader()

        class Res(unittest.TextTestResult):
            def addSuccess(self, test):
                outcomes[test.id()] = "ok"
                super().addSuccess(test)

            def addFailure(self, test, err):
                outcomes[test.id()] = "FAIL"
                super().addFailure(test, err)

            def addError(self, test, err):
                outcomes[test.id()] = "ERROR"
                super().addError(test, err)

            def addSkip(self, test, reason):
                outcomes[test.id()] = "skipped"
                super().addSkip(test, reason)
        suite = unittest.TestSuite(loader.loadTestsFromName(i) for i in ids)
        res = unittest.TextTestRunner(stream=sys.stdout, verbosity=2, resultclass=Res).run(suite)
        ran, passed = res.testsRun, sum(1 for v in outcomes.values() if v == "ok")
        failed, errors, skipped = len(res.failures), len(res.errors), len(res.skipped)
    else:
        import pytest

        class Count:
            def pytest_runtest_logreport(self, report):
                if report.when == "call":
                    outcomes[report.nodeid] = {"passed": "ok", "failed": "FAIL", "skipped": "skipped"}[report.outcome]
                elif report.outcome == "failed":
                    outcomes[report.nodeid] = "ERROR"                                  # a setup / teardown failure
        rc = pytest.main(["-q", "-p", "no:cacheprovider", "--tb=short", "-rfE", *ids], plugins=[Count()])
        ran = len(outcomes)
        passed = sum(1 for v in outcomes.values() if v == "ok")
        failed = sum(1 for v in outcomes.values() if v == "FAIL")
        errors = sum(1 for v in outcomes.values() if v == "ERROR")
        skipped = sum(1 for v in outcomes.values() if v == "skipped")
    named = {k: v for k, v in outcomes.items() if REAL_RUN in k}
    new = {k.split(NEW_CLASS)[-1].lstrip(".:"): v for k, v in outcomes.items() if NEW_CLASS in k}
    bad = sorted(k for k, v in outcomes.items() if v != "ok")
    print("RESULT " + json.dumps({"runner": runner, "file": fname, "order": order, "listed": len(ids), "ran": ran, "passed": passed, "failed": failed, "errors": errors, "skipped": skipped,
                                  "real_run_test": next(iter(named.values()), "absent"), "new_class": new, "not_ok": bad, "seconds": round(time.time() - t0, 1)}))
    return 0 if (failed == 0 and errors == 0 and ran == len(ids)) else 1


if __name__ == "__main__":
    sys.exit(main())
