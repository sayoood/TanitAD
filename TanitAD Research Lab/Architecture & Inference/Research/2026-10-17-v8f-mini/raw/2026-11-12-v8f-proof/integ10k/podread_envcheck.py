"""podread_envcheck.py -- control e1 of the pod-side read (note of 2026-11-19 in TIMES_NOTE.md): the pod reproduces a banked dev-box read cell for cell.
The same reader command that produced the banked result files of card ADD46 (row 101) is run on the pod; this tool compares, per result file, every leaf under
the named subtrees: numbers within TOL (NaN equals NaN), strings, booleans and nulls exactly, and the key sets. Numpy is not needed.
Usage: python podread_envcheck.py --bank <dir with hw/RESULT_HEADWAY.json and q3/RESULT_Q3.json> --got <dir with the same layout> --out ENV_E1.json
       python podread_envcheck.py --selftest
Exit 0 only if both files PASS."""
import argparse
import io
import json
import math
import sys

TOL = 1e-9
MIN_LEAVES = 100
FILES = {"hw/RESULT_HEADWAY.json": ("pairs", "levels"), "q3/RESULT_Q3.json": ("pairs", "arm_levels", "window_counts")}


def leaves(node, path, out):
    if isinstance(node, dict):
        for k in sorted(node):
            leaves(node[k], path + (str(k),), out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            leaves(v, path + (str(i),), out)
    else:
        out["/".join(path)] = node


def same(a, b):
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None or isinstance(a, str) or isinstance(b, str):
        return (type(a) is type(b) and a == b), 0.0
    fa, fb = float(a), float(b)
    if math.isnan(fa) or math.isnan(fb):
        return (math.isnan(fa) and math.isnan(fb)), 0.0
    d = abs(fa - fb)
    return d <= TOL, d


def compare(bank, got, subtrees):
    A, B = {}, {}
    for s in subtrees:
        leaves(bank.get(s, {}), (s,), A)
        leaves(got.get(s, {}), (s,), B)
    only_a, only_b = sorted(set(A) - set(B)), sorted(set(B) - set(A))
    diff, mx = [], 0.0
    for k in sorted(set(A) & set(B)):
        ok, d = same(A[k], B[k])
        mx = max(mx, d)
        if not ok:
            diff.append(k)
    r = {"n_leaves_bank": len(A), "n_leaves_got": len(B), "only_in_bank": only_a[:8], "only_in_got": only_b[:8], "n_only_in_bank": len(only_a), "n_only_in_got": len(only_b),
         "n_differing": len(diff), "first_differing": diff[:8], "max_abs_diff": mx, "tol": TOL}
    r["PASS"] = bool(len(A) >= MIN_LEAVES and not only_a and not only_b and not diff)
    return r


def selftest():
    base = {"pairs": {"a_minus_b": {"cells": {"U": {"D": -0.0623, "A_lo": -0.101, "A_hi": -0.0271, "d_s": [-0.0623, -0.0623], "verdict": "below"}}}},
            "levels": {"a": {"cells": {"U": {"per_seed": [0.1 + 0.001 * i for i in range(120)], "nan": float("nan"), "none": None, "flag": True}}}}}
    cp = lambda: json.loads(json.dumps(base).replace("NaN", "null"))
    def mk():
        d = cp()
        d["levels"]["a"]["cells"]["U"]["nan"] = float("nan")
        return d
    cases = []
    cases.append(("equal", mk(), mk(), True))
    d = mk(); d["pairs"]["a_minus_b"]["cells"]["U"]["D"] += 1e-6
    cases.append(("one number moved by 1e-6", mk(), d, False))
    d = mk(); d["pairs"]["a_minus_b"]["cells"]["U"]["D"] += 1e-12
    cases.append(("one number moved by 1e-12", mk(), d, True))
    d = mk(); d["pairs"]["a_minus_b"]["cells"]["U"]["verdict"] = "above"
    cases.append(("one word changed", mk(), d, False))
    d = mk(); del d["pairs"]["a_minus_b"]["cells"]["U"]["A_hi"]
    cases.append(("one key missing", mk(), d, False))
    d = mk(); d["levels"]["a"]["cells"]["U"]["nan"] = 0.0
    cases.append(("NaN against a number", mk(), d, False))
    d = mk(); d["levels"]["a"]["cells"]["U"]["flag"] = 1
    cases.append(("boolean against a number", mk(), d, False))
    small = {"pairs": {"x": {"D": 1.0}}, "levels": {}}
    cases.append(("too few leaves", small, json.loads(json.dumps(small)), False))
    bad = 0
    for name, a, b, want in cases:
        got = compare(a, b, ("pairs", "levels"))["PASS"]
        bad += 0 if got is want else 1
        print("selftest %-28s want %-5s got %-5s" % (name, want, got))
    print("ZZ SELFTEST %s" % ("PASS" if bad == 0 else "FAIL"))
    return 0 if bad == 0 else 1


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--got", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"control": "e1", "what": "the pod read reproduces the banked dev-box read of card ADD46 (row 101), same reader command", "files": {}}
    for rel, subtrees in FILES.items():
        bank = json.load(io.open(a.bank + "/" + rel, encoding="utf-8"))
        got = json.load(io.open(a.got + "/" + rel, encoding="utf-8"))
        res["files"][rel] = compare(bank, got, subtrees)
    res["PASS"] = bool(all(v["PASS"] for v in res["files"].values()))
    json.dump(res, io.open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    for rel, v in res["files"].items():
        print("e1 %-26s %s leaves %d / %d differing %d only-bank %d only-got %d max abs diff %.3g" % (rel, "PASS" if v["PASS"] else "FAIL", v["n_leaves_bank"], v["n_leaves_got"],
                                                                                                 v["n_differing"], v["n_only_in_bank"], v["n_only_in_got"], v["max_abs_diff"]))
        if v["first_differing"]:
            print("   first differing: %s" % v["first_differing"][:4])
    print("ZZ ENV-E1 %s" % ("PASS" if res["PASS"] else "FAIL"))
    return 0 if res["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
