"""Diff each r9e1 arm's argv against C0, and K2M1N20 against K0, at the tip (flags only; values for changed flags)."""
import json, subprocess, sys
GD, TIP = "C:/Users/Admin/tanitad-push/.git", sys.argv[1]
def argv(n):
    b = subprocess.run(["git", f"--git-dir={GD}", "show", f"{TIP}:stack/ops/runs.d/{n}.argv.json"], capture_output=True, check=True).stdout
    j = json.loads(b.decode("utf-8")); a = j["argv"] if isinstance(j, dict) and "argv" in j else j
    d, k = {}, None
    for t in a:
        if isinstance(t, str) and t.startswith("--"): k = t; d[k] = True
        elif k: d[k] = t if d[k] is True else f"{d[k]} {t}"
    return d, (j.get("steps") if isinstance(j, dict) else None)
def diff(a, b):
    A, _ = argv(a); B, _ = argv(b)
    add = {k: A[k] for k in A if k not in B}; rem = [k for k in B if k not in A]; chg = {k: (B[k], A[k]) for k in A if k in B and A[k] != B[k] and k not in ("--out", "--run-name", "--seed")}
    print(f"{a} vs {b}: +{add} -{rem} ~{chg}")
diff("r9e2_K2M1N20", "r9e2_K0")
for arm in ("M1N20", "T1v", "T1h", "T1", "B2", "B3", "M1"):
    diff(f"r9e1_{arm}", "r9e1_C0")
diff("r9e2_K0", "r9e1_C0")
