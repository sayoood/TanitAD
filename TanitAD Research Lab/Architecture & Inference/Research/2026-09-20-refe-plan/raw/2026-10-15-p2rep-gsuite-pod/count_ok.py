"""Run test_p2rep_chain.main() with ok() instrumented: record the caller's line number for every check, so two hosts' check sets can be diffed.
Usage (cwd = the refe/ dir holding test_p2rep_chain.py):  python count_ok.py <out.json>"""
import collections
import inspect
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import test_p2rep_chain as T  # noqa: E402

calls = collections.Counter()
_orig = T.ok


def ok(cond, *a, **k):
    calls[inspect.stack()[1].lineno] += 1
    return _orig(cond, *a, **k)


T.ok = ok
rc = 1
try:
    rc = T.main()
finally:
    json.dump({"rc": rc, "n": sum(calls.values()), "by_line": {str(k): v for k, v in sorted(calls.items())}}, open(sys.argv[1], "w"), indent=0)
    print("ZZCOUNT", sum(calls.values()))
