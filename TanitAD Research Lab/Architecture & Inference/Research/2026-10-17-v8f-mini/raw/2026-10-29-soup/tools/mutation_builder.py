#!/usr/bin/env python3
"""Mutation pass on soup_build.py: each mutant is one textual change (it must match exactly once), applied to a COPY in a scratch dir; the builder's
own test file runs against the copy (SOUP_BUILD_DIR) and must FAIL.  A green unmutated baseline is required first.  One pytest at a time."""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.dirname(HERE)
PY = sys.executable
SRC = os.path.join(W, "soup_build.py")
TESTS = os.path.join(W, "tests", "test_soup_build.py")
M = [
    ("f32-accumulate", "acc += sd[k].to(torch.float64)", "acc = (acc + sd[k]).to(t0.dtype).to(torch.float64)"),
    ("wrong-divisor", "out[k] = (acc / n).to(t0.dtype)", "out[k] = (acc / max(n - 1, 1)).to(t0.dtype)"),
    ("key-order-as-set", "if list(sd.keys()) != keys:", "if set(sd.keys()) != set(keys):"),
    ("no-shape-check", "if tuple(t.shape) != tuple(t0.shape):", "if False:"),
    ("no-dtype-check", "if t.dtype != t0.dtype:", "if False:"),
    ("nonfloat-unchecked", "if not torch.equal(sd[k], t0):", "if False:"),
    ("nonfinite-pattern-unchecked", "if not torch.equal(fa, fb):", "if False:"),
    ("nonfinite-values-unchecked", "if not bool(fa.all()) and not torch.equal(ta[~fa], tb[~fb]):", "if False:"),
    ("cfg-allowlist-widened", 'CFG_ALLOW = frozenset({"launch_intent_w"})', 'CFG_ALLOW = frozenset({"launch_intent_w", "hist"})'),
    ("must-equal-drops-cons_mu", '"p10_path", "cons_mu", "cons_sd")', '"p10_path", "cons_sd")'),
    ("B1-disabled", "B1_FACTOR = 2.0", "B1_FACTOR = 1e-9"),
    ("B2-disabled", "B2_FACTOR = 3.0", "B2_FACTOR = 1e9"),
    ("R_ft-max-not-median", "r_ft = float(np.median(list(r_ref.values())))", "r_ft = float(np.max(list(r_ref.values())))"),
    ("clauses-ignored", 'ok = (b1_ok or "B1" not in clauses) and (b2_ok or "B2" not in clauses)', "ok = b1_ok and b2_ok"),
    ("distance-includes-nonfinite", "s += float(((ta.to(torch.float64)[fin] - sb[k].to(torch.float64)[fin]) ** 2).sum())",
     "s += float(((ta.to(torch.float64) - sb[k].to(torch.float64)) ** 2).sum())"),
    ("opt-kept", 'out["opt"] = None', 'out["opt"] = out.get("opt")'),
    ("donor-last-member", 'donor = order.index(s["donor"])', "donor = len(order) - 1"),
    ("record-drops-member-sha", '"member_sha256": {n: shas[n] for n in order}', '"member_sha256": {}'),
    ("record-weights-wrong", '"weights": [1.0 / len(order)] * len(order)', '"weights": [0.5] * len(order)'),
    ("no-post-load-hash", "if h1 != h0:", "if False:"),
    ("no-pin-check", "if want_sha is not None and h0 != want_sha:", "if False:"),
    ("kmid-no-numpy-mean", "worst = max(worst, float(dev.max()))", "worst = worst"),
    ("kmid-nonfinite-unchecked", "if not np.array_equal(np.isfinite(got), fin) or not all(np.array_equal(got[~fin], sd[k].numpy()[~fin]) for sd in sds):", "if False:"),
    ("verify-opt-unchecked", 'if ck.get("opt") is not None:', "if False:"),
    ("verify-entries-unchecked", "if _canon(ck.get(k)) != _canon(donor.get(k)):", "if False:"),
    ("verify-output-hash-unchecked", 'if sha256_file(ck_path) != rec.get("output_sha256"):', "if False:"),
    ("build-ignores-basin", 'if not kb["PASS"]:', "if False:"),
    ("build-overwrites", "if os.path.exists(ck_path) or os.path.exists(rec_path):", "if False:"),
    ("build-writes-xinit", 'raise SoupRefusal(f"{name} is a deliberate-regression soup', 'print(f"{name} is a deliberate-regression soup'),
    ("cli-regression-always-ok", 'r["regression_as_required"] = bool(not r["full"]["PASS"] and not r["B2_only"]["PASS"] and not r["B1_only"]["PASS"])',
     'r["regression_as_required"] = True'),
    ("kzero-ignores-dtype", "same = all(torch.equal(s[k], sd[k]) and s[k].dtype == sd[k].dtype for k in sd)", "same = True"),
]
NOT_MUTATED = {
    "K-MID: the member-range clause": "implied by the exact-mean clause (a correctly rounded mean of values in [min, max] lies in [min, max]); run 1 showed it as an equivalent mutant",
    "K-MID: the N = 2 distance identity": "implied by the exact-mean clause; run 1 showed it as an equivalent mutant. Kept as an independent cross-check on real weights",
    'verify: `if _canon(ck.get("cfg")) != _canon(donor.get("cfg"))`': "redundant by design: the donor-entries loop below it compares cfg too (defence in depth)",
    "sqdist: `if not np.isfinite(s)`": "unreachable once non-finite elements are excluded (float32 squares cannot overflow float64); it guards a future dtype",
}


def run_tests(build_dir, base):
    env = dict(os.environ, SOUP_BUILD_DIR=build_dir, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", TESTS, f"--basetemp={base}"], env=env, capture_output=True, text=True, timeout=600)
    return r.returncode, (r.stdout.strip().splitlines() or [""])[-1]


def main():
    src = open(SRC, encoding="utf-8").read()
    scratch = tempfile.mkdtemp(prefix="soup_mut_", dir=os.environ.get("SOUP_MUT_SCRATCH"))
    res = {"baseline": None, "mutants": {}, "not_mutated": NOT_MUTATED}
    d0 = os.path.join(scratch, "base")
    os.makedirs(d0)
    shutil.copy(SRC, d0)
    rc, tail = run_tests(d0, os.path.join(scratch, "bt_base"))
    res["baseline"] = {"rc": rc, "tail": tail}
    if rc != 0:
        print("BASELINE NOT GREEN", tail)
        return 2
    killed = survived = invalid = 0
    for i, (name, old, new) in enumerate(M):
        n = src.count(old)
        if n != 1:
            res["mutants"][name] = {"status": f"INVALID (matches {n})"}
            invalid += 1
            continue
        d = os.path.join(scratch, f"m{i:02d}")
        os.makedirs(d)
        with open(os.path.join(d, "soup_build.py"), "w", encoding="utf-8", newline="\n") as f:
            f.write(src.replace(old, new))
        rc, tail = run_tests(d, os.path.join(scratch, f"bt{i:02d}"))
        st = "KILLED" if rc != 0 else "SURVIVED"
        killed += st == "KILLED"
        survived += st == "SURVIVED"
        res["mutants"][name] = {"status": st, "rc": rc, "tail": tail}
        print(f"{name:34s} {st}  ({tail})")
    res["summary"] = {"mutants": len(M), "killed": killed, "survived": survived, "invalid": invalid, "builder_sha256": __import__("hashlib").sha256(src.encode()).hexdigest()}
    with open(os.path.join(W, "evidence", "mutation_builder.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    print("MUTATION", res["summary"])
    return 0 if survived == 0 and invalid == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
